"""Tag-free camera calibration: the ARM is the calibration rig.

Sweep the orange gripper tip through a known 3D grid of base-frame points, detect
the tip pixel at each, and solve the FULL 3x4 camera projection matrix P by DLT.
Decompose P -> intrinsics K + camera pose (R, C) in the B601 base frame. Unlike
the old flat z=0 homography, this is a true 6-DOF calibration from points at
MULTIPLE heights, so it is valid through the whole workspace VOLUME. No AprilTag,
no checkerboard.

CRITICAL CONSTRAINT (measured): the Reachy head is NOT repeatable -- commanding
the same pose after moving away lands the camera ~50-120px off, though it is rock
stable (<1px) while UNCOMMANDED. So this tool aims the head ONCE, then never
touches it, and the resulting P is valid only until the head is next commanded.
The whole localize->grasp cycle must run at this same locked head pose.

Run (arm + reachy bridge must be up):
    <lerobot python> -m arm_agent.cam_calib_dlt --run
    <lerobot python> -m arm_agent.cam_calib_dlt --dry     # preview grid, no motion
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

import cv2
import httpx
import numpy as np

from .tip_detect import annotate, detect_tip

ARM = os.getenv("REBOT_ARM_SERVICE_URL", "http://127.0.0.1:8001")
BRIDGE = os.getenv("REACHY_BRIDGE_URL", "http://127.0.0.1:8712")
OUT_DIR = Path(os.getenv("REBOT_CALIB_DIR",
              "/home/manav/rebot_lerobot/rebot_arm_service/calib"))
DEBUG_DIR = Path("/tmp/claude-1000/-home-manav-rebot-lerobot/"
                 "f773b4c4-d21a-48aa-acb4-6ca472d858e9/scratchpad/dlt_frames")
PITCH = 1.30          # wrist pitched down so the tips face the oblique camera
FIXED_YAW = 0.0       # keep the wrist from rotating -> tip appearance is constant
BORDER = 40           # reject tips within this many px of the frame edge (clipped)


# ---- arm + camera IO -------------------------------------------------------
def move(x, y, z, dur=2.0, yaw=FIXED_YAW):
    r = httpx.post(f"{ARM}/move/pose", json={"x": x, "y": y, "z": z, "roll": 0.0,
                   "pitch": PITCH, "yaw": yaw, "mode": "smooth", "duration": dur},
                   timeout=dur + 40)
    return r.status_code == 200 and r.json().get("accepted", True)


def ee_pos():
    s = httpx.get(f"{ARM}/state", timeout=5).json()
    s = s.get("state", s)
    return list(s["end_effector"]["position_m"])


def gripper(action):
    httpx.post(f"{ARM}/gripper", json={"action": action}, timeout=20)


def head_set(pose, dur=1.2):
    httpx.post(f"{BRIDGE}/head", json={**pose, "duration": dur}, timeout=15)


def head_look(u, v, dur=0.8):
    httpx.post(f"{BRIDGE}/head", json={"look_pixel": [u, v], "duration": dur}, timeout=15)


def head_pose():
    return httpx.get(f"{BRIDGE}/health", timeout=5).json().get("head_pose")


def frame_bgr(warmup=4):
    r = httpx.get(f"{BRIDGE}/frame.jpg", params={"warmup": warmup}, timeout=15)
    r.raise_for_status()
    return cv2.imdecode(np.frombuffer(r.content, np.uint8), cv2.IMREAD_COLOR)


def in_frame(u, v, w, h):
    return u is not None and BORDER < u < w - BORDER and BORDER < v < h - BORDER


# ---- DLT camera solve ------------------------------------------------------
def solve_dlt(world, img):
    """Full 3x4 projection matrix from >=6 non-coplanar 3D<->2D correspondences,
    with Hartley normalization for conditioning."""
    W = np.asarray(world, float)
    I = np.asarray(img, float)
    cW = W.mean(0)
    sW = math.sqrt(3) / (np.linalg.norm(W - cW, axis=1).mean() + 1e-12)
    U = np.array([[sW, 0, 0, -sW * cW[0]], [0, sW, 0, -sW * cW[1]],
                  [0, 0, sW, -sW * cW[2]], [0, 0, 0, 1]])
    cI = I.mean(0)
    sI = math.sqrt(2) / (np.linalg.norm(I - cI, axis=1).mean() + 1e-12)
    T = np.array([[sI, 0, -sI * cI[0]], [0, sI, -sI * cI[1]], [0, 0, 1]])
    Wn = (U @ np.c_[W, np.ones(len(W))].T).T[:, :3]
    In = (T @ np.c_[I, np.ones(len(I))].T).T[:, :2]
    A = []
    for (X, Y, Z), (u, v) in zip(Wn, In):
        A.append([0, 0, 0, 0, -X, -Y, -Z, -1, v * X, v * Y, v * Z, v])
        A.append([X, Y, Z, 1, 0, 0, 0, 0, -u * X, -u * Y, -u * Z, -u])
    _, _, Vt = np.linalg.svd(np.asarray(A))
    Pn = Vt[-1].reshape(3, 4)
    P = np.linalg.inv(T) @ Pn @ U
    return P / np.linalg.norm(P[2, :3])


def reproj_px(P, world, img):
    W = np.c_[np.asarray(world, float), np.ones(len(world))]
    x = (P @ W.T).T
    uv = x[:, :2] / x[:, 2:3]
    return np.linalg.norm(uv - np.asarray(img, float), axis=1)


def solve_dlt_ransac(world, img, iters=800, thresh=6.0, seed=0):
    """Robust DLT: random 6-point fits, keep the model with the most inliers,
    then refit on all inliers. Immunizes the solve against a few bad detections."""
    rng = np.random.default_rng(seed)
    W = np.asarray(world, float)
    I = np.asarray(img, float)
    n = len(W)
    best = None
    for _ in range(iters):
        s = rng.choice(n, 6, replace=False)
        try:
            P = solve_dlt(W[s], I[s])
        except Exception:
            continue
        err = reproj_px(P, W, I)
        inl = np.where(err < thresh)[0]
        if best is None or len(inl) > len(best[1]):
            best = (P, inl)
    P, inl = best
    if len(inl) >= 6:
        P = solve_dlt(W[inl], I[inl])
    return P, inl


def decompose(P):
    K, R, t = cv2.decomposeProjectionMatrix(P)[:3]
    K = K / K[2, 2]
    C = (t[:3] / t[3]).ravel()
    return K, R, C


def backproject(K, R, C, u, v, z=0.0):
    d = R.T @ (np.linalg.inv(K) @ np.array([u, v, 1.0]))
    lam = (z - C[2]) / d[2]
    p = C + lam * d
    return float(p[0]), float(p[1]), float(p[2])


# ---- sweep -----------------------------------------------------------------
def grid_points():
    pts = []
    for z in (0.17, 0.11, 0.05):
        ys = (-0.11, 0.0, 0.11)
        for xi, x in enumerate((0.25, 0.30, 0.35)):
            row = list(ys) if xi % 2 == 0 else list(reversed(ys))
            for y in row:
                pts.append((x, y, z))
    return pts


def aim_head_once():
    """Center the grid on the image, then NEVER command the head again."""
    move(0.30, 0.0, 0.11, 2.5)
    time.sleep(1.0)
    head_set(dict(yaw_deg=0.0, pitch_deg=18.0, roll_deg=0.0, body_yaw_deg=-45.0), dur=1.2)
    time.sleep(1.5)
    fr = frame_bgr()
    h, w = fr.shape[:2]
    for _ in range(4):
        u, v, info = detect_tip(fr)
        if u is None:
            break
        if abs(u - w / 2) < 90 and abs(v - h / 2) < 90:
            break
        head_look(u, v, dur=0.7)   # turn head to bring the tip toward center
        time.sleep(1.2)
        fr = frame_bgr()
    print("head aimed; LOCKING at", head_pose())
    return w, h


def run(dry=False):
    DEBUG_DIR.mkdir(parents=True, exist_ok=True)
    pts = grid_points()
    print(f"grid: {len(pts)} points  yaw locked at {FIXED_YAW}")
    if dry:
        for p in pts:
            print("  ", p)
        return

    gripper("close")
    time.sleep(0.6)
    w, h = aim_head_once()        # <-- last head command; head is fixed from here
    hp = head_pose()

    world, img, rows = [], [], []
    for i, (x, y, z) in enumerate(pts):
        if not move(x, y, z, 2.0):
            print(f"[{i}] move rejected {x,y,z}")
            continue
        time.sleep(1.3)
        try:
            p = ee_pos()
            fr = frame_bgr(warmup=4)
        except Exception as e:
            print(f"[{i}] io error {e}")
            continue
        u, v, info = detect_tip(fr)
        cv2.imwrite(str(DEBUG_DIR / f"pt{i:02d}.jpg"), annotate(fr, u, v, info))
        if in_frame(u, v, w, h) and info.get("area", 0) > 150:
            world.append(p)
            img.append((u, v))
            rows.append({"i": i, "cmd": [x, y, z], "ee": p, "px": [u, v], "area": info["area"]})
            print(f"[{i}] ee=({p[0]:.3f},{p[1]:.3f},{p[2]:.3f}) px=({u:.0f},{v:.0f}) area={info['area']:.0f}")
        else:
            why = "off-frame/clipped" if u is not None else "no tip"
            print(f"[{i}] SKIP ({why}) px={None if u is None else (round(u),round(v))}")

    move(0.27, 0.0, 0.18, 2.5)   # park (still no head command)

    n = len(world)
    print(f"\ncollected {n}/{len(pts)} usable points")
    if n < 10:
        print("too few points for a stable DLT; aborting save")
        return

    world = np.asarray(world)
    img = np.asarray(img)
    test_idx = list(range(0, n, 5))
    train_idx = [i for i in range(n) if i not in test_idx]
    P, inl = solve_dlt_ransac(world[train_idx], img[train_idx])
    K, R, C = decompose(P)

    tr = reproj_px(P, world[train_idx], img[train_idx])
    te = reproj_px(P, world[test_idx], img[test_idx])
    mm = []
    for j in test_idx:
        bx, by, _ = backproject(K, R, C, img[j][0], img[j][1], z=world[j][2])
        mm.append(1000 * math.hypot(bx - world[j][0], by - world[j][1]))
    mm = np.asarray(mm)

    print(f"\nRANSAC inliers: {len(inl)}/{len(train_idx)} train points")
    print(f"intrinsics fx={K[0,0]:.0f} fy={K[1,1]:.0f} cx={K[0,2]:.0f} cy={K[1,2]:.0f}")
    print(f"camera center in base frame C = ({C[0]:.3f}, {C[1]:.3f}, {C[2]:.3f}) m")
    print(f"reproj RMS  train={np.sqrt((tr**2).mean()):.1f}px  test={np.sqrt((te**2).mean()):.1f}px")
    print(f"HELD-OUT metric error: mean={mm.mean():.1f}mm  max={mm.max():.1f}mm  (n={len(mm)})")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / "cam_calib_dlt.json"
    json.dump({
        "P": P.tolist(), "K": K.tolist(), "R": R.tolist(), "C": C.tolist(),
        "head_pose": hp, "image_size": [int(w), int(h)],
        "n_points": n, "reproj_test_px": float(np.sqrt((te**2).mean())),
        "heldout_mm_mean": float(mm.mean()), "heldout_mm_max": float(mm.max()),
        "points": rows,
    }, open(out, "w"), indent=2)
    print(f"saved {out}\nNOTE: valid ONLY until the head is next commanded.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--dry", action="store_true")
    args = ap.parse_args()
    run(dry=not args.run)
