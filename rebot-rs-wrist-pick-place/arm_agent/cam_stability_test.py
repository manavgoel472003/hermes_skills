"""Separate CAMERA motion from ARM repeatability.

Fixed explicit head pose (no look_pixel). Return the arm to the SAME pose several
times, interleaved with far excursions. For each return frame measure:
  * tip pixel (arm repeatability + camera together)
  * background shift of a static top strip via phase correlation (camera ALONE)
If the background strip shifts between identical-arm frames, the camera is moving
(vibration/drift). If only the tip moves, it's the arm.
"""
from __future__ import annotations
import time
import cv2, httpx, numpy as np
from .tip_detect import detect_tip

ARM = "http://127.0.0.1:8001"
BRIDGE = "http://127.0.0.1:8712"
HEAD = dict(yaw_deg=0.0, pitch_deg=22.0, roll_deg=0.0, body_yaw_deg=-45.0)
P0 = (0.28, 0.0, 0.10)
FAR = [(0.34, 0.11, 0.16), (0.25, -0.11, 0.05), (0.35, -0.10, 0.17), (0.24, 0.11, 0.06)]


def move(x, y, z, dur=2.2):
    httpx.post(f"{ARM}/move/pose", json={"x": x, "y": y, "z": z, "roll": 0.0,
               "pitch": 1.30, "yaw": 0.0, "mode": "smooth", "duration": dur}, timeout=45)


def frame():
    r = httpx.get(f"{BRIDGE}/frame.jpg", params={"warmup": 5}, timeout=15)
    return cv2.imdecode(np.frombuffer(r.content, np.uint8), cv2.IMREAD_COLOR)


def bg_strip(img):
    # top strip = static office background, away from the arm/table
    g = cv2.cvtColor(img[0:250, :], cv2.COLOR_BGR2GRAY)
    return np.float32(g)


def main():
    httpx.post(f"{ARM}/gripper", json={"action": "close"}, timeout=20)
    httpx.post(f"{BRIDGE}/head", json={**HEAD, "duration": 1.5}, timeout=15)
    time.sleep(2.0)
    move(*P0, dur=3.0)
    time.sleep(1.5)

    ref = frame()
    ref_bg = bg_strip(ref)
    u0, v0, _ = detect_tip(ref)
    print(f"reference: tip=({u0},{v0})")

    tips, bgshifts = [(u0, v0)], [(0.0, 0.0)]
    for i, far in enumerate(FAR):
        move(*far, dur=2.2); time.sleep(1.4)   # excursion
        move(*P0, dur=2.2); time.sleep(1.6)     # return to identical pose
        f = frame()
        u, v, _ = detect_tip(f)
        (dx, dy), resp = cv2.phaseCorrelate(ref_bg, bg_strip(f))
        tips.append((u, v)); bgshifts.append((dx, dy))
        print(f"return {i}: tip=({u},{v})  bg_shift=({dx:+.1f},{dy:+.1f})px resp={resp:.2f}")

    t = np.array([[a, b] for a, b in tips if a is not None], float)
    b = np.array(bgshifts, float)
    print(f"\nTIP pixel spread over identical arm pose: "
          f"dx={t[:,0].max()-t[:,0].min():.0f} dy={t[:,1].max()-t[:,1].min():.0f} px")
    print(f"BACKGROUND shift magnitude (camera motion): "
          f"max={np.linalg.norm(b,axis=1).max():.1f} px")
    print("  -> big BG shift = camera is moving; small BG but big TIP = arm not repeatable")


if __name__ == "__main__":
    main()
