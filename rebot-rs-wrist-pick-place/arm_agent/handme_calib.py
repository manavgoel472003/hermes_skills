"""Headless Reachy->B601 calibration, callable from the MCP tool (so Hermes can
run it) or a script. Location-independent: works from any Reachy position as long
as the arm is visible.

  1. AUTO-AIM: move the gripper to a central pose, sweep Reachy's head, pick the
     head pose where the orange tips are largest+centered -> the pick view.
  2. CALIBRATE: at that pose, step the arm through a grid (mode='smooth', tracks
     accurately), motion-gate the orange tips (robust to clutter), pair each pixel
     with the arm's ACTUAL end-effector (x,y).
  3. Fit the pixel->base homography with RANSAC; save reachy_cam.json + meta
     (grasp_view_head_pose), which the pick pipeline re-aims to before detecting.

Re-run whenever Reachy moves OR the arm re-zeros (USB re-enumeration) -> the old
homography points at the wrong arm frame ("way off" picks).
"""
from __future__ import annotations

import math, os, time
from typing import Any, Callable

import httpx, numpy as np, cv2

from .reachy_client import ReachyBridgeClient
from .reachy_handme import ARM, CALIB
from rebot_arm_service.scene.calibration import TableCalibration

Log = Callable[[str], None]
PITCH = 1.30


def _move(x, y, z, dur=2.5):
    yaw = math.atan2(y, x)
    r = httpx.post(f"{ARM}/move/pose", json={"x": x, "y": y, "z": z, "roll": 0.0,
                   "pitch": PITCH, "yaw": yaw, "mode": "smooth", "duration": dur}, timeout=45)
    return r.status_code == 200


def _gripper(a):
    httpx.post(f"{ARM}/gripper", json={"action": a}, timeout=20)


def _ee():
    s = httpx.get(f"{ARM}/state", timeout=5).json()
    s = s.get("state", s)
    return s["end_effector"]["position_m"]


def _orange_mask(img):
    return cv2.inRange(cv2.cvtColor(img, cv2.COLOR_BGR2HSV), (6, 140, 140), (26, 255, 255))


def _largest_orange(img):
    m = cv2.morphologyEx(_orange_mask(img), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    cn, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cn = [c for c in cn if cv2.contourArea(c) > 120]
    if not cn:
        return None, 0
    c = max(cn, key=cv2.contourArea); M = cv2.moments(c)
    return (M["m10"] / M["m00"], M["m01"] / M["m00"]), cv2.contourArea(c)


def _tips_midpoint(frame, bg):
    d = cv2.cvtColor(cv2.absdiff(frame, bg), cv2.COLOR_BGR2GRAY); _, mv = cv2.threshold(d, 22, 255, 0)
    mv = cv2.dilate(mv, np.ones((9, 9), np.uint8))
    mask = cv2.morphologyEx(cv2.bitwise_and(_orange_mask(frame), mv), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    cn, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cn = sorted([c for c in cn if cv2.contourArea(c) > 80], key=cv2.contourArea, reverse=True)[:2]
    if not cn:
        return None
    cs = [(cv2.moments(c)["m10"] / cv2.moments(c)["m00"], cv2.moments(c)["m01"] / cv2.moments(c)["m00"]) for c in cn]
    return (float(np.mean([c[0] for c in cs])), float(np.mean([c[1] for c in cs])))


def calibrate(reachy: ReachyBridgeClient, log: Log = print, save: bool = True) -> dict[str, Any]:
    """Run the full auto-aim + grid calibration. Moves the arm (~3-4 min).
    Returns {ok, reproj_error_mm, points_used, inliers, grasp_view}."""
    if not reachy.available():
        return {"ok": False, "reason": "reachy_bridge_unreachable"}

    def head(pose):
        reachy.head(pose=pose, duration=1.0)

    def frame():
        return reachy.frame_bgr(warmup=6)

    # Close the gripper so the two orange pads read as ONE compact blob -- a far
    # more stable tip target than two spread pads (which the midpoint mis-locates
    # if one pad is momentarily occluded).
    _gripper("close"); time.sleep(0.6)

    # 1. AUTO-AIM -----------------------------------------------------------
    log("auto-aim: locating the arm in Reachy's view")
    _move(0.28, 0.0, 0.14, 3.0); time.sleep(0.5)
    best = None
    for byaw in (-45, -30, -15, 0, 15, 30):
        for pitch in (16, 24):
            pose = dict(yaw_deg=0.0, pitch_deg=float(pitch), roll_deg=0.0, body_yaw_deg=float(byaw))
            head(pose); time.sleep(0.3)
            img = frame(); c, area = _largest_orange(img)
            if c is None:
                continue
            H, W = img.shape[:2]
            centered = 1.0 - (abs(c[0] - W / 2) / (W / 2) * 0.5 + abs(c[1] - H / 2) / (H / 2) * 0.5)
            score = area * max(0.2, centered)
            if best is None or score > best[0]:
                best = (score, pose)
    if best is None:
        return {"ok": False, "reason": "autoaim_failed_arm_not_visible"}
    grasp_view = best[1]
    log(f"pick view chosen: body_yaw={grasp_view['body_yaw_deg']:+.0f} pitch={grasp_view['pitch_deg']:.0f}")
    head(grasp_view); time.sleep(0.5)

    # 2. CALIBRATE ----------------------------------------------------------
    log("calibrating: stepping the arm through a grid")
    for z in (0.24, 0.20, 0.16):
        if _move(0.22, 0.0, z):
            break
    time.sleep(0.4); head(grasp_view); time.sleep(0.4)
    bg = frame()
    pix: list[tuple[float, float]] = []
    wact: list[tuple[float, float]] = []
    for x in (0.22, 0.28, 0.34):
        for y in (-0.12, 0.0, 0.12):
            if not _move(x, y, 0.12):
                break
            if not _move(x, y, 0.03):
                break
            time.sleep(0.7)
            p = _ee(); c = _tips_midpoint(frame(), bg)
            if c:
                pix.append(c); wact.append((p[0], p[1]))
            _move(x, y, 0.12)
    for z in (0.24, 0.20, 0.16):
        if _move(0.22, 0.0, z):
            break

    Wd, Hd = bg.shape[1], bg.shape[0]
    if len(pix) < 6:
        return {"ok": False, "reason": "too_few_detections", "points_used": len(pix)}

    # 3. FIT -- RANSAC first, fall back to least-squares if it can't lock on --
    src = np.asarray(pix, float); dst = np.asarray(wact, float)
    method = "ransac"
    Hm, mask = cv2.findHomography(src, dst, method=cv2.RANSAC, ransacReprojThreshold=0.02)
    if Hm is None or mask is None or int(np.asarray(mask).sum()) < 5:
        Hm2, _ = cv2.findHomography(src, dst, method=0)   # least-squares over all points
        if Hm2 is not None:
            Hm, mask, method = Hm2, np.ones((len(src), 1), np.uint8), "least_squares"
    if Hm is None:
        return {"ok": False, "reason": "homography_fit_failed", "points_used": len(pix),
                "pixels": [list(map(float, p)) for p in pix],
                "world": [list(map(float, w)) for w in wact]}
    inl = [i for i in range(len(mask)) if mask[i]]
    cal = TableCalibration(H=Hm, table_z=0.0, image_size=(Wd, Hd),
                           notes="reachy auto-aim smooth %s (%d/%d inliers)" % (method, len(inl), len(pix)))
    err = cal.reprojection_error([tuple(src[i]) for i in inl], [tuple(dst[i]) for i in inl])
    result: dict[str, Any] = {
        "ok": True, "reproj_error_mm": round(err * 1000, 1), "method": method,
        "points_used": len(pix), "inliers": len(inl), "grasp_view": grasp_view,
        "warning": ("error > 20mm; consider re-running (lighting/occlusion)" if err > 0.020 else None),
    }
    if save:
        os.makedirs(CALIB, exist_ok=True)
        cal.save(f"{CALIB}/reachy_cam.json")
        import json as _json
        _json.dump({"grasp_view_head_pose": grasp_view, "image_size": [Wd, Hd],
                    "reproj_error_mm": err * 1000, "inliers": inl, "method": method,
                    "pixels": [list(p) for p in pix], "world": [list(w) for w in wact]},
                   open(f"{CALIB}/reachy_calibration_meta.json", "w"), indent=2)
        log(f"saved calibration ({err*1000:.1f}mm, {len(inl)}/{len(pix)} inliers)")
    return result
