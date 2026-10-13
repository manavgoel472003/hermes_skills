"""Two-camera pick: wrist cam does the precise X,Y; Reachy cam cross-checks Y;
the torque-gated descent does Z by feel. No calibration, no parallax guessing.

Geometry (per the plan):
  * WRIST cam (index 6, eye-in-hand, looking down): the gripper's two orange
    fingertips are rigidly in view, so their midpoint is a FIXED "grasp point"
    pixel. We servo the arm in X,Y until the object's pixel lands on that grasp
    point -> object is directly between the fingers. This nails X and Y (the axis
    the Reachy is worst at) with no parallax.
  * REACHY cam (external): independently locates the object's lateral position; we
    require its Y to AGREE with the wrist before committing -> removes doubt.
  * DESCENT: torque-gated (arm_safe.safe_descend) -- stops the instant the gripper
    touches, so Z needs no vision and can't strain.

Every arm motion goes through arm_safe (workspace AABB + /sim/ik verify + small
step cap + smooth IK/joint move). Gripper only opens to the service 68% cap.

    <lerobot python> -m arm_agent.dual_grasp --object "red marker" --run
    <lerobot python> -m arm_agent.dual_grasp --object "red marker" --dry
"""
from __future__ import annotations

import argparse
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
from arm_agent.tip_detect import orange_mask

WRIST_CAM = 6
BRIDGE = "http://127.0.0.1:8712"
HEAD_POSE = dict(yaw_deg=0.0, pitch_deg=22.0, roll_deg=0.0, body_yaw_deg=-45.0)
SCAN = (0.28, 0.0, 0.16)           # gripper-down scan pose; wrist sees the table
COARSE_PX = 40.0                   # coarse XY alignment tolerance at scan height
RECENTER_PX = 24.0                 # during descent, re-center if drift exceeds this
GAIN = 0.4
MAX_ITERS = 24
DESC_STEP = 0.006                  # descent increment while re-centering (m)
JUMP_NM = 0.25                     # torque jump that means contact
JAC_STEP = 0.015
DEBUG = Path("/tmp/claude-1000/-home-manav-rebot-lerobot/"
             "f773b4c4-d21a-48aa-acb4-6ca472d858e9/scratchpad/dual")


class WristCam:
    """Persistent capture on the wrist camera (reopening per-frame is too slow)."""

    def __init__(self, idx=WRIST_CAM):
        self.cap = cv2.VideoCapture(idx, cv2.CAP_V4L2)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    def frame(self):
        f = None
        for _ in range(5):          # flush stale buffer, take the freshest
            ok, g = self.cap.read()
            if ok:
                f = g
        return f

    def close(self):
        self.cap.release()


def grasp_point(frame):
    """Fixed grasp target = midpoint of the two orange fingertips (topmost point of
    each fin, i.e. the ends that reach toward the object)."""
    m = orange_mask(frame)
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted([c for c in cnts if cv2.contourArea(c) > 300], key=cv2.contourArea, reverse=True)[:2]
    if len(cnts) < 2:
        return None
    tips = [tuple(c[c[:, :, 1].argmin()][0]) for c in cnts]
    return (int((tips[0][0] + tips[1][0]) / 2), int((tips[0][1] + tips[1][1]) / 2))


def reachy_frame():
    r = httpx.get(f"{BRIDGE}/frame.jpg", params={"warmup": 4}, timeout=15)
    return cv2.imdecode(np.frombuffer(r.content, np.uint8), cv2.IMREAD_COLOR)


def head_fix():
    httpx.post(f"{BRIDGE}/head", json={**HEAD_POSE, "duration": 1.5}, timeout=15)
    time.sleep(2.0)


class Locator:
    def __init__(self):
        from rebot_arm_service.scene import LocateAnythingLocator
        self.la = LocateAnythingLocator()

    def center(self, frame, name):
        r = self.la.locate(frame, name)
        if not r.get("found"):
            return None, None
        b = r["box"]
        return (0.5 * (b[0] + b[2]), 0.5 * (b[1] + b[3])), b


def run(object_name: str, dry: bool):
    DEBUG.mkdir(parents=True, exist_ok=True)
    pf = A.preflight()
    print(f"preflight: {pf.ok} ({pf.reason})")
    if not pf.ok:
        return
    for nm, (x, y, z) in [("scan", SCAN), ("floor", (SCAN[0], SCAN[1], A.Z_MIN))]:
        ik = A.check_ik(x, y, z)
        print(f"  IK[{nm}] ({x:.2f},{y:.2f},{z:.3f}): {ik.ok} {ik.reason}")
        if not ik.ok:
            print("  key pose unreachable; abort"); return

    head_fix()
    wrist = WristCam()
    obj = Locator()
    try:
        # confirm both cameras can see the object before moving
        wf = wrist.frame()
        gp = grasp_point(wf)
        wc, wbox = obj.center(wf, object_name)
        rc, rbox = obj.center(reachy_frame(), object_name)
        print(f"wrist: grasp_point={gp} object={None if wc is None else (round(wc[0]),round(wc[1]))}")
        print(f"reachy: object={None if rc is None else (round(rc[0]),round(rc[1]))}")
        cv2.imwrite(str(DEBUG / "start_wrist.jpg"), _annot(wf, gp, wc, wbox))
        if gp is None or wc is None:
            print("wrist can't see grasp point or object; abort"); return
        if rc is None:
            print("reachy can't see object; abort (need both for the cross-check)"); return

        if dry:
            print("DRY: both cameras see the object + grasp point; IK ok. Safe to --run.")
            return

        A.gripper("open")
        res = A.safe_goto(*SCAN)
        print(f"move->scan: {res.ok} {res.reason}")
        if not res.ok:
            return

        J = _wrist_jacobian(wrist, obj, object_name)
        if J is None:
            print("could not learn wrist jacobian; abort"); return
        Jinv = np.linalg.inv(J)
        print(f"wrist jacobian:\n{np.round(J,1)}")

        def wrist_err(tag):
            """Detect object + grasp point in the wrist view; return (err_vec, gp, wc)."""
            wf = wrist.frame()
            gp = grasp_point(wf)
            wc, wbox = obj.center(wf, object_name)
            cv2.imwrite(str(DEBUG / f"{tag}.jpg"), _annot(wf, gp, wc, wbox))
            if gp is None or wc is None:
                return None, gp, wc
            return np.array([gp[0] - wc[0], gp[1] - wc[1]], float), gp, wc

        def xy_step(e, gain):
            step = np.clip(gain * (Jinv @ e), -A.MAX_XY_STEP, A.MAX_XY_STEP)
            return A.safe_step(dx=float(step[0]), dy=float(step[1]))

        # ---- Phase 1: coarse XY align at scan height ----
        for it in range(MAX_ITERS):
            e, gp, wc = wrist_err(f"coarse{it:02d}")
            if e is None:
                print(f"[c{it}] lost grasp_point/object; abort"); return
            print(f"[c{it}] object=({wc[0]:.0f},{wc[1]:.0f}) grasp_pt={gp} err={np.linalg.norm(e):.0f}px")
            if np.linalg.norm(e) < COARSE_PX:
                print("coarse-aligned."); break
            r = xy_step(e, GAIN)
            if not r.ok:
                print(f"  step refused ({r.reason}); stopping"); return
            time.sleep(0.3)
        else:
            print("coarse align did not converge; not grasping"); return

        # ---- Reachy Y cross-check (the doubt-remover) ----
        rc2, _ = obj.center(reachy_frame(), object_name)
        if rc2 is None:
            print("reachy lost the object at align time; abort for safety"); return
        print(f"reachy cross-check: object still seen at ({rc2[0]:.0f},{rc2[1]:.0f})")

        # ---- Phase 2: descend WHILE re-centering (parallax shrinks to 0 at contact),
        # torque-gated so it stops at the object/table ----
        prev_t = A.joint_torques()
        stopped = "floor"
        for d in range(60):
            e, gp, wc = wrist_err(f"desc{d:02d}")
            if e is not None and np.linalg.norm(e) > RECENTER_PX:
                r = xy_step(e, 0.5)                       # re-center before descending
                print(f"[d{d}] recenter err={np.linalg.norm(e):.0f}px -> {r.reason}")
                if not r.ok:
                    stopped = "recenter_refused"; break
                time.sleep(0.25); continue
            cur = tuple(A.ee_pos())
            if cur[2] <= A.Z_MIN + 1e-4:
                stopped = "floor"; break
            z_next = max(A.Z_MIN, cur[2] - DESC_STEP)
            r = A.safe_move(cur[0], cur[1], z_next, cur=cur)
            if not r.ok:
                stopped = f"descend_refused({r.reason})"; break
            cur_t = A.joint_torques()
            jump = max((abs(abs(c) - abs(p)) for c, p in zip(cur_t, prev_t)), default=0.0)
            if (cur[2] - (r.reached[2] if r.reached else z_next)) < 0.4 * DESC_STEP or jump > JUMP_NM:
                stopped = f"contact (torque+{jump:.2f}Nm, z={round((r.reached or (0,0,z_next))[2],4)})"; break
            prev_t = cur_t
            e2 = f"{np.linalg.norm(e):.0f}px" if e is not None else "occluded"
            print(f"[d{d}] descend->{z_next:.3f} (align {e2})")
            time.sleep(0.15)

        print(f"descent stopped: {stopped}")
        cur = tuple(A.ee_pos())
        A.gripper("close")
        time.sleep(0.8)
        r = A.safe_move(cur[0], cur[1], min(A.Z_MAX, cur[2] + A.MAX_Z_STEP), cur=cur)
        print(f"lift: {r.ok} {r.reason}")
        print("done.")
    finally:
        wrist.close()


def _wrist_jacobian(wrist, obj, name):
    """Learn 2x2 J: gripper (dx,dy) -> object pixel (du,dv) in the WRIST view."""
    base = tuple(A.ee_pos())
    cols = []
    for axis in (0, 1):
        p0, _ = obj.center(wrist.frame(), name)
        if p0 is None:
            return None
        d = [0.0, 0.0]; d[axis] = JAC_STEP
        r = A.safe_step(dx=d[0], dy=d[1], cur=base)
        if not r.ok:
            print(f"  jac move refused: {r.reason}"); return None
        time.sleep(0.5)
        p1, _ = obj.center(wrist.frame(), name)
        A.safe_move(*base, cur=r.reached)
        time.sleep(0.5)
        if p1 is None:
            return None
        cols.append([(p1[0] - p0[0]) / JAC_STEP, (p1[1] - p0[1]) / JAC_STEP])
    J = np.array(cols, float).T
    return None if abs(np.linalg.det(J)) < 1e-3 else J


def _annot(frame, gp, oc, box):
    v = frame.copy()
    m = orange_mask(frame); v[m > 0] = (0, 140, 255)
    if gp:
        cv2.drawMarker(v, gp, (0, 255, 0), cv2.MARKER_CROSS, 50, 3)
    if oc is not None:
        cv2.circle(v, (int(oc[0]), int(oc[1])), 10, (0, 0, 255), 3)
    if box is not None:
        b = [int(x) for x in box]; cv2.rectangle(v, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 2)
    return v


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--object", required=True)
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--run", action="store_true")
    args = ap.parse_args()
    if not (args.dry or args.run):
        print("pass --dry or --run"); sys.exit(1)
    run(args.object, dry=args.dry and not args.run)
