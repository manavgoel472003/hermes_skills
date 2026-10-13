"""Closed-loop visual servo grasp -- tag-free, calibration-free, drift-immune.

The idea (see the design discussion): a low, oblique Reachy camera can't measure
range (base X) well, and its head pose isn't repeatable. So we DON'T estimate the
object's 3D pose open-loop. Instead we drive the gripper's grasp-center pixel onto
the object's table-contact pixel in the SAME image -- a relative alignment, so
camera drift and the grazing angle cancel (both tip and object move together).

Servo target = the object box's BOTTOM-CENTER (where it meets the table). We hold
the tip at just-above-table height so tip and target are ~coplanar; then pixel
alignment == real XY alignment. A small pixel->base Jacobian is learned online
with two tiny test moves, so no metric camera calibration is needed.

SAFETY: every arm motion goes through arm_safe (hard workspace AABB + /sim/ik
dry-run verify + small step cap + smooth IK/joint-interp move). The gripper only
opens to the service's 68% hard cap. The head is fixed once and never moved.

    <lerobot python> -m arm_agent.visual_servo --object "green sponge" --table-z 0.0 --dry
    <lerobot python> -m arm_agent.visual_servo --object "green sponge" --table-z 0.0 --run
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
from arm_agent.tip_detect import annotate, detect_grasp_center

BRIDGE = "http://127.0.0.1:8712"
HEAD_POSE = dict(yaw_deg=0.0, pitch_deg=22.0, roll_deg=0.0, body_yaw_deg=-45.0)
START = (0.30, 0.0, 0.12)          # safe in-view start (open gripper)
ALIGN_PX = 14.0                    # aligned when tip-target pixel error below this
GAIN = 0.5                         # servo gain (under 1 = damped, stable)
MAX_ITERS = 18
JAC_STEP = 0.015                   # tiny base move to learn the image Jacobian (m)
HOVER_Z = 0.05                     # safe, comfortable height to run the XY servo at
GRASP_FLOOR = 0.015                # hard floor for the torque-gated grasp descent
DEBUG = Path("/tmp/claude-1000/-home-manav-rebot-lerobot/"
             "f773b4c4-d21a-48aa-acb4-6ca472d858e9/scratchpad/servo")


def head_fix():
    """Set the head ONCE to a stable pose, then never touch it again."""
    httpx.post(f"{BRIDGE}/head", json={**HEAD_POSE, "duration": 1.5}, timeout=15)
    time.sleep(2.0)


def frame():
    r = httpx.get(f"{BRIDGE}/frame.jpg", params={"warmup": 4}, timeout=15)
    return cv2.imdecode(np.frombuffer(r.content, np.uint8), cv2.IMREAD_COLOR)


class ObjLocator:
    def __init__(self):
        from rebot_arm_service.scene import LocateAnythingLocator
        self.la = LocateAnythingLocator()

    def target(self, fr, name):
        """Object box -> grasp target pixel = box CENTER. The tip hovers above the
        table, so by parallax it aligns with the object's center (not its base)."""
        r = self.la.locate(fr, name)
        if not r.get("found"):
            return None, None
        b = r["box"]
        return (0.5 * (b[0] + b[2]), 0.5 * (b[1] + b[3])), b   # box center, box


def _grasp_px():
    u, v, _ = detect_grasp_center(frame())
    return None if u is None else np.array([u, v], float)


def estimate_jacobian():
    """Learn 2x2 J: base (dx,dy) -> pixel (du,dv), via two tiny IK-checked moves.
    Re-measures the baseline pixel before each axis so slow camera drift can't bias
    the columns. Always returns the arm to the base pose."""
    base = tuple(A.ee_pos())
    cols = []
    for axis in (0, 1):
        p0 = _grasp_px()                        # fresh baseline at `base`
        if p0 is None:
            return None
        d = [0.0, 0.0]; d[axis] = JAC_STEP
        res = A.safe_step(dx=d[0], dy=d[1], cur=base)
        if not res.ok:
            print(f"  jac move refused: {res.reason}"); return None
        time.sleep(0.5)
        p1 = _grasp_px()
        A.safe_move(*base, cur=res.reached)     # return to base
        time.sleep(0.5)
        if p1 is None:
            return None
        cols.append((p1 - p0) / JAC_STEP)
    J = np.array(cols, float).T                 # columns = per-axis pixel deltas
    if abs(np.linalg.det(J)) < 1e-3:
        print("  jacobian near-singular; aborting"); return None
    return J


def run(object_name: str, dry: bool):
    DEBUG.mkdir(parents=True, exist_ok=True)

    pf = A.preflight()
    print(f"preflight: {pf.ok} ({pf.reason}) ee={pf.reached}")
    if not pf.ok:
        return
    # IK-reachability of the key poses BEFORE anything moves
    for name, (x, y, z) in [("start", START), ("hover", (START[0], START[1], HOVER_Z)),
                            ("floor", (START[0], START[1], GRASP_FLOOR))]:
        ik = A.check_ik(x, y, z)
        print(f"  IK[{name}] ({x:.2f},{y:.2f},{z:.3f}): {ik.ok} {ik.reason}")
        if not ik.ok:
            print("  -> key pose unreachable; abort"); return

    head_fix()
    fr = frame()
    obj = ObjLocator()
    tgt, box = obj.target(fr, object_name)
    if tgt is None:
        print(f"object '{object_name}' not found in view; abort")
        cv2.imwrite(str(DEBUG / "noobj.jpg"), fr)
        return
    print(f"object target pixel (table contact): ({tgt[0]:.0f},{tgt[1]:.0f}) box={[round(v) for v in box]}")

    # open gripper (to the hard-clamped 68% cap) and go to the safe start
    if not dry:
        A.gripper("open")
    res = A.safe_goto(*START, dry=dry)
    print(f"move->start: {res.ok} {res.reason}")
    if not res.ok:
        return

    # descend to the (safe, comfortable) hover height, in capped sub-steps
    cur = tuple(A.ee_pos()) if not dry else START
    res = A.safe_goto(cur[0], cur[1], HOVER_Z, cur=cur, dry=dry)
    print(f"descend->hover {HOVER_Z:.3f}: {res.ok} {res.reason}")
    if not res.ok:
        return

    if dry:
        print("DRY: detection + IK all validated; skipping motion servo. Safe to --run.")
        return

    J = estimate_jacobian()
    if J is None:
        print("could not learn image jacobian; abort (no risky motion taken)")
        return
    Jinv = np.linalg.inv(J)
    print(f"image jacobian learned:\n{np.round(J,1)}")

    # ---- servo loop -------------------------------------------------------
    for it in range(MAX_ITERS):
        f = frame()
        if it % 5 == 0:                       # refresh object target (drift-safe)
            t2, _ = obj.target(f, object_name)
            if t2 is not None:
                tgt = t2
        tu, tv, info = detect_grasp_center(f)
        vis = annotate(f, tu, tv, info)
        cv2.circle(vis, (int(tgt[0]), int(tgt[1])), 10, (255, 0, 255), 2)
        cv2.imwrite(str(DEBUG / f"servo{it:02d}.jpg"), vis)
        if tu is None:
            print(f"[{it}] tip not visible; abort"); return
        e = np.array([tgt[0] - tu, tgt[1] - tv], float)
        print(f"[{it}] tip=({tu:.0f},{tv:.0f}) target=({tgt[0]:.0f},{tgt[1]:.0f}) err={np.linalg.norm(e):.0f}px")
        if np.linalg.norm(e) < ALIGN_PX:
            print("aligned."); break
        step = GAIN * (Jinv @ e)
        step = np.clip(step, -A.MAX_XY_STEP, A.MAX_XY_STEP)
        res = A.safe_step(dx=float(step[0]), dy=float(step[1]))
        if not res.ok:
            print(f"  servo step refused ({res.reason}); stopping safely"); return
        time.sleep(0.3)
    else:
        print("did not converge within iteration cap; not grasping"); return

    # ---- torque-gated descent & grasp -------------------------------------
    # Straight down until the gripper touches the marker/table (torque jump or
    # stall) or the hard floor -- never pushing, so no strain like the probe.
    cur = tuple(A.ee_pos())
    res = A.safe_descend(cur[0], cur[1], GRASP_FLOOR)
    zr = None if not res.reached else round(res.reached[2], 4)
    print(f"grasp-descend: {res.reason}  (stopped at z={zr})")
    if not res.ok:
        print("descent failed; not closing the gripper")
        return
    cur = res.reached or cur
    A.gripper("close")
    time.sleep(0.8)
    # lift straight up to confirm the grasp
    res = A.safe_move(cur[0], cur[1], min(A.Z_MAX, cur[2] + A.MAX_Z_STEP), cur=cur)
    print(f"lift: {res.ok} {res.reason}")
    print("done.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--object", required=True)
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--run", action="store_true")
    args = ap.parse_args()
    if not (args.dry or args.run):
        print("pass --dry (validate, no motion) or --run"); sys.exit(1)
    run(args.object, dry=args.dry and not args.run)
