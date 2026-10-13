"""Headless core of the 'hand me the <object>' pipeline, for the MCP tools.

The interactive demo (``reachy_handme.py``) owns the Tk window + threads; this
module reuses its *pure* helpers (grasp geometry, hand/gripper detection) but runs
each phase synchronously with NO window and NO background grabber, so the phases
can be exposed as MCP tools and orchestrated by Hermes:

    reachy_pick(object)  ->  reachy_find_palm()  ->  reachy_present()
                          ->  (Hermes/user confirms)  ->  reachy_release()

Frames are grabbed on demand from the Reachy bridge; the head parks at the pick
view for the whole pick and only moves during palm-finding (same as the demo).
"""
from __future__ import annotations

import json, math, os, time
from typing import Any, Callable

import httpx, numpy as np

from .reachy_client import ReachyBridgeClient
from .reachy_handme import (
    ARM, CALIB, X_MIN, X_MAX, Y_MIN, Y_MAX, PRESENT, PRESENT_PITCH,
    REACH_Y_PER_DEG, REACH_Y_MAX,
    vlm_boxes, grasp_from_frame, object_on_table,
    _hand_detector, _detect_hand,
)
from rebot_arm_service.scene.calibration import TableCalibration

Log = Callable[[str], None]


# ------------------------------------------------------------- arm + frames
def _move(x, y, z, dur=2.5, yaw=None, pitch=1.30):
    yaw = math.atan2(y, x) if yaw is None else yaw
    return httpx.post(f"{ARM}/move/pose", json={"x": x, "y": y, "z": z, "roll": 0.0,
                      "pitch": pitch, "yaw": yaw, "mode": "smooth", "duration": dur}, timeout=45).status_code == 200


def _gripper(a):
    httpx.post(f"{ARM}/gripper", json={"action": a}, timeout=20)


def _home():
    httpx.post(f"{ARM}/home", json={}, timeout=50)


def load_calib_gv():
    cal = TableCalibration.load(f"{CALIB}/reachy_cam.json")
    gv = json.load(open(f"{CALIB}/reachy_calibration_meta.json"))["grasp_view_head_pose"]
    return cal, gv


def park_pick_view(reachy: ReachyBridgeClient, gv, log: Log = print):
    """Park the head at the pick view (grasp_view); it stays here for the pick."""
    log("parking head at pick view")
    reachy.head(pose=gv, duration=1.2); time.sleep(0.4)


# --------------------------------------------------------------------- pick
def pick(reachy: ReachyBridgeClient, cal, gv, obj: str, color: str | None = None,
         grasp_z: float = 0.02, home_first: bool = True, log: Log = print) -> dict[str, Any]:
    """VLM-detect at the (fixed, calibrated) pick view, grasp oriented across the
    object's short axis, verify it left the table, retry once. Head does not move."""
    park_pick_view(reachy, gv, log)
    if home_first:
        log("homing arm to clear the view"); _home(); time.sleep(0.4)
    log(f"detecting '{obj}' with the VLM")
    boxes = vlm_boxes(obj)
    if not boxes:
        return {"ok": False, "reason": "not_found", "object": obj}
    frame = reachy.frame_bgr(warmup=4)
    g = grasp_from_frame(cal, frame, boxes[0], color)
    if g is None:
        return {"ok": False, "reason": "localize_failed", "object": obj}
    x, y, yaw, (u, vv) = g
    if not (X_MIN <= x <= X_MAX and Y_MIN <= y <= Y_MAX):
        return {"ok": False, "reason": "outside_workspace", "grasp_m": [round(x, 3), round(y, 3)]}
    log(f"grasp target base=({x:+.3f},{y:+.3f}) yaw={math.degrees(yaw):+.0f}deg")

    for attempt in range(2):
        log(f"grasping (try {attempt + 1})")
        _gripper("open"); time.sleep(0.3)
        _move(x, y, 0.12, yaw=yaw)
        _move(x, y, grasp_z - 0.008 * attempt, yaw=yaw); time.sleep(0.5)
        _gripper("close"); time.sleep(1.0)
        _move(x, y, 0.18, yaw=yaw)
        _move(0.22, -0.02, 0.20, yaw=0.0); time.sleep(0.3)   # swing aside; head still at pick view
        chk = reachy.frame_bgr(warmup=2)
        if chk is None or not object_on_table(cal, chk, color, (u, vv)):
            log("pick verified (object left the table)")
            return {"ok": True, "object": obj, "grasp_m": [round(x, 3), round(y, 3)],
                    "yaw_deg": round(math.degrees(yaw), 1)}
        log("object still on the table -> retrying")
        _gripper("open"); time.sleep(0.3); _move(x, y, 0.12, yaw=yaw)
    return {"ok": False, "reason": "grasp_failed_after_retry", "object": obj}


# --------------------------------------------------------------- find palm
_SWEEP = [dict(yaw_deg=0, pitch_deg=-4, roll_deg=0, body_yaw_deg=b) for b in (0, -25, 25, -12, 12)]


def find_palm(reachy: ReachyBridgeClient, timeout: float = 30.0, log: Log = print) -> dict[str, Any]:
    """Sweep the head to find an OPEN palm held up; centre on it. Returns the head
    bearing (deg, + = palm to Reachy's right) so the arm can reach toward it."""
    detector, mp = _hand_detector()
    try:
        log("looking around for an open palm")
        t0 = time.time(); i = 0
        while time.time() - t0 < timeout:
            pose = _SWEEP[i % len(_SWEEP)]; i += 1
            reachy.head(pose=pose, duration=0.6); time.sleep(0.3)
            frame = reachy.frame_bgr(warmup=1)
            if frame is None:
                continue
            hand, _lms, is_open = _detect_hand(detector, mp, frame)
            if hand is not None and is_open:
                log("palm found -> centering")
                for _ in range(3):
                    f2 = reachy.frame_bgr(warmup=1)
                    if f2 is None:
                        break
                    h2, _l2, _o2 = _detect_hand(detector, mp, f2)
                    if h2 is None:
                        break
                    reachy.head(look_pixel=(h2[0], h2[1]), duration=0.25); time.sleep(0.2)
                p = reachy.health().get("head_pose", {})
                bearing = float(p.get("body_yaw_deg", 0.0)) + float(p.get("yaw_deg", 0.0))
                return {"ok": True, "found": True, "bearing_deg": round(bearing, 1)}
        return {"ok": True, "found": False, "reason": "no_palm_seen"}
    finally:
        detector.close()


# ----------------------------------------------------------------- present
def present(reachy: ReachyBridgeClient, gv, bearing_deg: float | None = None, log: Log = print) -> dict[str, Any]:
    """Reach the held object toward the palm's bearing (approximate; no depth)."""
    y = 0.0 if bearing_deg is None else float(np.clip(-REACH_Y_PER_DEG * bearing_deg, -REACH_Y_MAX, REACH_Y_MAX))
    px, _, pz = PRESENT
    log(f"reaching toward the palm (y={y:+.2f})")
    ok = _move(px, y, pz, dur=3.0, yaw=0.0, pitch=PRESENT_PITCH)
    return {"ok": bool(ok), "present_m": [px, round(y, 3), pz], "awaiting": "reachy_release"}


def release(log: Log = print) -> dict[str, Any]:
    log("releasing (opening gripper)")
    _gripper("open"); time.sleep(0.6)
    return {"ok": True, "released": True}
