"""Fixed RGB-D pick: Orbbec Gemini 336 -> metric 3D -> arm base frame -> grasp.

Pipeline (once the camera<->base transform is calibrated):
    detect object (RGB) -> median depth over it -> deproject with intrinsics
    -> object 3D in CAMERA frame -> apply rigid transform -> 3D in ARM BASE frame
    -> move above -> torque-gated descend -> close -> lift.

Calibration is a one-time 3D<->3D rigid fit (no parallax/scale ambiguity, since
both sides are real millimetres): the arm moves its orange tip through a grid of
known base poses; at each the Orbbec measures the tip's 3D; Umeyama solves R,t.
Re-run it whenever the camera (on its own stand) is moved.

All arm motion goes through arm_safe (workspace AABB + /sim/ik verify + small step
cap + smooth IK move + torque-gated descend). Gripper only opens to the 68% cap.

    # no arm needed -- prove localization:
    <lerobot py> -m arm_agent.orbbec_pick --localize "red marker"
    # with arm connected:
    <lerobot py> -m arm_agent.orbbec_pick --calibrate
    <lerobot py> -m arm_agent.orbbec_pick --pick "red marker" --run
"""
from __future__ import annotations

import argparse
import io
import json
import sys
import time
from pathlib import Path

import cv2
import httpx
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "rebot_arm_service" / "src"))
sys.path.insert(0, str(ROOT))

from arm_agent import arm_safe as A

ORBBEC = "http://127.0.0.1:8720"
CALIB = ROOT / "rebot_arm_service" / "calib" / "orbbec_cam2base.json"
DEBUG = Path("/tmp/claude-1000/-home-manav-rebot-lerobot/"
             "f773b4c4-d21a-48aa-acb4-6ca472d858e9/scratchpad/orbbecpick")


# ---- RGB-D source ----------------------------------------------------------
class Orbbec:
    def frame(self):
        r = httpx.get(f"{ORBBEC}/frame.npz", timeout=15)
        r.raise_for_status()
        z = np.load(io.BytesIO(r.content))
        return z["color"], z["depth"], (float(z["fx"]), float(z["fy"]), float(z["cx"]), float(z["cy"]))


def deproject(u, v, Z, intr):
    fx, fy, cx, cy = intr
    return np.array([(u - cx) * Z / fx, (v - cy) * Z / fy, Z], float)


def median_depth(depth, u, v, r=5):
    win = depth[max(0, int(v) - r):int(v) + r, max(0, int(u) - r):int(u) + r]
    win = win[win > 0]
    return float(np.median(win)) if len(win) >= 8 else 0.0


# ---- wiggle tip detection (color-independent; works on the black gripper) ---
def wiggle_tip_3d(orb, debug_tag=None, settle=0.7):
    """Localize the gripper fingers in 3D WITHOUT color/markers: freeze the arm,
    snapshot with the gripper CLOSED, open it, snapshot again. Only the fingers
    move, so the frame difference isolates them; among the changed pixels, those
    NEARER the camera in the closed frame are the fingers at their closed (grasp)
    position. Take the bottom slice (fingertip ends) and deproject its median.
    Leaves the gripper CLOSED. Returns (p_cam_mm | None, info)."""
    A.gripper("close"); time.sleep(settle)
    color_a, depth_a, intr = orb.frame()
    A.gripper("open"); time.sleep(settle)
    color_b, depth_b, _ = orb.frame()
    A.gripper("close"); time.sleep(settle)

    gray = cv2.absdiff(cv2.cvtColor(color_a, cv2.COLOR_BGR2GRAY),
                       cv2.cvtColor(color_b, cv2.COLOR_BGR2GRAY))
    moved = (gray > 25).astype(np.uint8) * 255
    moved = cv2.morphologyEx(moved, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    cnts, _ = cv2.findContours(moved, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = [c for c in cnts if cv2.contourArea(c) > 150]
    info = {"changed_px": int(moved.sum() // 255), "n_blobs": len(cnts)}
    if not cnts:
        return None, info
    blob = np.zeros(moved.shape, np.uint8)
    cv2.drawContours(blob, sorted(cnts, key=cv2.contourArea, reverse=True)[:4], -1, 255, -1)
    # fingers in the CLOSED frame = changed pixels distinctly nearer in depth_a
    ys, xs = np.where((blob > 0) & (depth_a > 0) & (depth_b > 0)
                      & (depth_a < depth_b - 15.0))
    if len(ys) < 40:   # fallback: any changed pixel with valid closed-frame depth
        ys, xs = np.where((blob > 0) & (depth_a > 0))
        info["fallback"] = True
    if len(ys) < 40:
        return None, info
    # fingertip ends = bottom slice of the finger pixels
    y_lo = ys.max() - max(3, int(0.2 * (ys.max() - ys.min())))
    sel = ys >= y_lo
    zs = depth_a[ys[sel], xs[sel]]
    Z = float(np.median(zs))
    u, v = float(xs[sel].mean()), float(ys[sel].mean())
    info.update(finger_px=int(len(ys)), tip_uv=(round(u), round(v)), tip_Z=round(Z))
    if debug_tag:
        DEBUG.mkdir(parents=True, exist_ok=True)
        vis = color_a.copy()
        vis[blob > 0] = (0, 140, 255)
        cv2.drawMarker(vis, (int(u), int(v)), (0, 255, 0), cv2.MARKER_CROSS, 40, 3)
        cv2.imwrite(str(DEBUG / f"{debug_tag}.jpg"), vis)
    return deproject(u, v, Z, intr), info


# ---- object localization (camera frame) ------------------------------------
class Detector:
    def __init__(self):
        from rebot_arm_service.scene import LocateAnythingLocator
        self.la = LocateAnythingLocator()

    def locate_3d(self, color, depth, intr, name):
        """Detect `name`, take robust depth over its box, deproject -> 3D (camera mm)."""
        r = self.la.locate(color, name)
        if not r.get("found"):
            return None, None
        b = [int(v) for v in r["box"]]
        u, v = (b[0] + b[2]) // 2, (b[1] + b[3]) // 2
        box = depth[b[1]:b[3], b[0]:b[2]]
        valid = box[box > 0]
        if len(valid) < 10:
            return None, b
        Z = float(np.median(valid))
        return deproject(u, v, Z, intr), b


# ---- rigid transform (Umeyama, no scaling) ---------------------------------
def rigid_fit(cam, base):
    """Solve R,t minimizing ||R*cam_i + t - base_i||. cam/base: Nx3 (same units, m)."""
    cam = np.asarray(cam, float)
    base = np.asarray(base, float)
    cc, cb = cam.mean(0), base.mean(0)
    H = (cam - cc).T @ (base - cb)
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1, 1, d]) @ U.T
    t = cb - R @ cc
    resid = np.linalg.norm((cam @ R.T + t) - base, axis=1)
    return R, t, resid


def load_calib():
    if not CALIB.exists():
        return None
    d = json.loads(CALIB.read_text())
    return np.array(d["R"]), np.array(d["t"])


def cam_to_base(p_cam_mm, R, t):
    """p_cam in mm -> base frame in metres."""
    return R @ (np.asarray(p_cam_mm, float) / 1000.0) + t


# ---- calibration: wiggle the gripper through a grid, Orbbec measures it ----
def calibrate(verify_object: str | None = None):
    """At each grid pose the ARM FREEZES and only the gripper opens/closes; the
    frame difference isolates the fingers (works on the black gripper, no
    markers). Pairs (finger 3D in camera, TCP in base) -> Umeyama R,t. Then, if
    `verify_object` is given, one automatic verification hover measures the
    residual constant offset (finger-point vs true TCP) and folds it into t."""
    DEBUG.mkdir(parents=True, exist_ok=True)
    if not A.preflight().ok:
        print("arm not ready"); return
    orb = Orbbec()
    grid = [(x, y, z) for z in (0.12, 0.07) for x in (0.26, 0.30, 0.34)
            for y in (-0.08, 0.0, 0.08)]
    cam_pts, base_pts = [], []
    for i, (x, y, z) in enumerate(grid):
        if not A.check_ik(x, y, z).ok:
            print(f"[{i}] {x,y,z} unreachable, skip"); continue
        r = A.safe_goto(x, y, z)
        if not r.ok:
            print(f"[{i}] move refused: {r.reason}"); continue
        time.sleep(0.5)                              # let the arm settle: frozen from here
        base = np.array(A.ee_pos(), float)           # TCP in BASE frame (m)
        cam, info = wiggle_tip_3d(orb, debug_tag=f"wiggle{i:02d}")
        if cam is None:
            print(f"[{i}] fingers not isolated ({info})"); continue
        cam_pts.append(cam / 1000.0); base_pts.append(base)
        print(f"[{i}] base=({base[0]:.3f},{base[1]:.3f},{base[2]:.3f}) "
              f"cam=({cam[0]:.0f},{cam[1]:.0f},{cam[2]:.0f})mm  ({info['finger_px']}px)")
    A.safe_goto(0.27, 0.0, 0.16)
    if len(cam_pts) < 4:
        print(f"only {len(cam_pts)} points; need >=4"); return
    R, t, resid = rigid_fit(cam_pts, base_pts)
    print(f"\ncalibrated on {len(cam_pts)} points  RMS residual={1000*np.sqrt((resid**2).mean()):.1f}mm  "
          f"max={1000*resid.max():.1f}mm")
    CALIB.parent.mkdir(parents=True, exist_ok=True)
    CALIB.write_text(json.dumps({"R": R.tolist(), "t": t.tolist(),
                                 "rms_mm": float(1000*np.sqrt((resid**2).mean())),
                                 "n": len(cam_pts), "method": "wiggle"}, indent=2))
    print(f"saved {CALIB}")
    if verify_object:
        verify_offset(verify_object, orb)


def verify_offset(name: str, orb=None):
    """Automatic offset correction: hover the TCP above the detected object (per
    the current calibration), wiggle once, and measure finger-vs-object directly
    in the CAMERA frame. Their XY gap in base coords is the constant finger-vs-TCP
    offset the wiggle fit absorbed; fold it into t so picks land on the TCP."""
    cal = load_calib()
    if cal is None:
        print("no calibration to verify"); return
    R, t = cal
    orb = orb or Orbbec()
    det = Detector()
    color, depth, intr = orb.frame()
    p_cam, _ = det.locate_3d(color, depth, intr, name)
    if p_cam is None:
        print(f"verify: '{name}' not found; skipping offset correction"); return
    pb = cam_to_base(p_cam, R, t)
    hover = (float(pb[0]), float(pb[1]), max(A.Z_MIN + 0.04, float(pb[2]) + 0.07))
    if not A.check_ik(*hover).ok:
        print(f"verify: hover {hover} unreachable; skipping"); return
    r = A.safe_goto(*hover)
    if not r.ok:
        print(f"verify: hover move refused ({r.reason})"); return
    time.sleep(0.5)
    tip_cam, info = wiggle_tip_3d(orb, debug_tag="verify_wiggle")
    A.safe_goto(0.27, 0.0, 0.16)
    if tip_cam is None:
        print(f"verify: tip not seen ({info}); offset not corrected"); return
    # object hasn't moved -- use its PRE-hover measurement (hovering gripper may
    # occlude it), and compare XY only (the tip hovers above it by design)
    delta = R @ ((np.asarray(p_cam) - np.asarray(tip_cam)) / 1000.0)
    delta[2] = 0.0                                   # Z is handled by the torque gate
    print(f"verify: measured XY offset = ({1000*delta[0]:+.1f}, {1000*delta[1]:+.1f}) mm")
    if np.linalg.norm(delta[:2]) > 0.05:
        print("verify: offset >5cm is implausible; NOT applying (check debug frames)"); return
    d = json.loads(CALIB.read_text())
    t2 = np.array(d["t"]) + delta
    d["t"] = t2.tolist(); d["verified_offset_mm"] = [float(1000*delta[0]), float(1000*delta[1])]
    CALIB.write_text(json.dumps(d, indent=2))
    print(f"verify: correction folded into t; saved {CALIB}")


# ---- localize (no arm) -----------------------------------------------------
def localize(name):
    orb = Orbbec()
    det = Detector()
    color, depth, intr = orb.frame()
    p_cam, box = det.locate_3d(color, depth, intr, name)
    if p_cam is None:
        print(f"'{name}' not found (or no depth)"); return
    print(f"'{name}' 3D in CAMERA frame = ({p_cam[0]:.0f}, {p_cam[1]:.0f}, {p_cam[2]:.0f}) mm")
    cal = load_calib()
    if cal is None:
        print("(no camera<->base calibration yet -- run --calibrate with the arm to get base-frame coords)")
    else:
        R, t = cal
        pb = cam_to_base(p_cam, R, t)
        print(f"'{name}' in ARM BASE frame = ({pb[0]:.3f}, {pb[1]:.3f}, {pb[2]:.3f}) m")


# ---- pick ------------------------------------------------------------------
def pick(name, dry):
    cal = load_calib()
    if cal is None:
        print("no calibration; run --calibrate first"); return
    R, t = cal
    if not A.preflight().ok:
        print("arm not ready"); return
    orb = Orbbec(); det = Detector()
    color, depth, intr = orb.frame()
    p_cam, box = det.locate_3d(color, depth, intr, name)
    if p_cam is None:
        print(f"'{name}' not found"); return
    pb = cam_to_base(p_cam, R, t)
    print(f"'{name}' base-frame target = ({pb[0]:.3f}, {pb[1]:.3f}, {pb[2]:.3f}) m")
    approach = (float(pb[0]), float(pb[1]), max(A.Z_MIN + 0.03, float(pb[2]) + 0.06))
    for nm, p in [("approach", approach),
                  ("over", (approach[0], approach[1], max(A.Z_MIN, float(pb[2]) + 0.02)))]:
        ik = A.check_ik(*p)
        print(f"  IK[{nm}] {tuple(round(v,3) for v in p)}: {ik.ok} {ik.reason}")
        if not ik.ok:
            print("  target unreachable; abort"); return
    if dry:
        print("DRY: object localized + approach reachable. Safe to --run."); return
    A.gripper("open")
    r = A.safe_goto(*approach)
    print(f"approach: {r.ok} {r.reason}")
    if not r.ok:
        return
    cur = tuple(A.ee_pos())
    res = A.safe_descend(cur[0], cur[1], max(A.Z_MIN, float(pb[2]) - 0.005))
    print(f"descend: {res.reason} (z={None if not res.reached else round(res.reached[2],4)})")
    if not res.ok:
        print("descent failed; not closing the gripper")
        return
    cur = res.reached or cur
    A.gripper("close"); time.sleep(0.8)
    r = A.safe_move(cur[0], cur[1], min(A.Z_MAX, cur[2] + A.MAX_Z_STEP), cur=cur)
    print(f"lift: {r.ok} {r.reason}\ndone.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--localize", metavar="OBJECT")
    ap.add_argument("--calibrate", action="store_true")
    ap.add_argument("--verify", metavar="OBJECT",
                    help="after --calibrate (or standalone): hover + wiggle over OBJECT to auto-correct the residual offset")
    ap.add_argument("--pick", metavar="OBJECT")
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--run", action="store_true")
    args = ap.parse_args()
    if args.localize:
        localize(args.localize)
    elif args.calibrate:
        calibrate(verify_object=args.verify)
    elif args.verify:
        verify_offset(args.verify)
    elif args.pick:
        pick(args.pick, dry=not args.run)
    else:
        print("use --localize OBJ | --calibrate [--verify OBJ] | --pick OBJ [--run]")
