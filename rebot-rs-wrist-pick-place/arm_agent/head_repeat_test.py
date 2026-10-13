"""Diagnose whether the Reachy head pose is STABLE + REPEATABLE.

Arm held dead still; we only move the HEAD. If the tip pixel wanders across
repeats of the SAME commanded head pose, the head isn't repeatable and a fixed
camera-matrix calibration is impossible (P assumes a bolted-in camera).
"""
from __future__ import annotations
import time
import cv2, httpx, numpy as np
from .tip_detect import detect_tip, annotate

ARM = "http://127.0.0.1:8001"
BRIDGE = "http://127.0.0.1:8712"
OUT = ("/tmp/claude-1000/-home-manav-rebot-lerobot/"
       "f773b4c4-d21a-48aa-acb4-6ca472d858e9/scratchpad/headtest")

POSE_A = dict(yaw_deg=0.0, pitch_deg=16.0, roll_deg=0.0, body_yaw_deg=-45.0)
POSE_B = dict(yaw_deg=0.0, pitch_deg=0.0, roll_deg=0.0, body_yaw_deg=0.0)  # look away


def head(pose, dur=1.2):
    httpx.post(f"{BRIDGE}/head", json={**pose, "duration": dur}, timeout=15)


def frame():
    r = httpx.get(f"{BRIDGE}/frame.jpg", params={"warmup": 5}, timeout=15)
    return cv2.imdecode(np.frombuffer(r.content, np.uint8), cv2.IMREAD_COLOR)


def pose_now():
    return httpx.get(f"{BRIDGE}/health", timeout=5).json().get("head_pose")


def shot(label):
    fr = frame()
    u, v, info = detect_tip(fr)
    import os
    os.makedirs(OUT, exist_ok=True)
    cv2.imwrite(f"{OUT}/{label}.jpg", annotate(fr, u, v, info))
    print(f"  {label}: tip=({u},{v}) commanded_vs_reported={pose_now()}")
    return (u, v)


def main():
    # gripper closed, arm dead still at one in-frame pose
    httpx.post(f"{ARM}/gripper", json={"action": "close"}, timeout=20)
    httpx.post(f"{ARM}/move/pose", json={"x": 0.28, "y": 0.0, "z": 0.10, "roll": 0.0,
               "pitch": 1.30, "yaw": 0.0, "mode": "smooth", "duration": 3.0}, timeout=45)
    time.sleep(1.0)

    print("== head at POSE_A ==")
    head(POSE_A); time.sleep(1.5)
    a1 = shot("A1")
    print("== 3s later, no commands (drift/settle test) ==")
    time.sleep(3.0)
    a2 = shot("A2")
    print("== move head AWAY to POSE_B, then back to POSE_A (repeatability) ==")
    head(POSE_B); time.sleep(1.5)
    head(POSE_A); time.sleep(1.5)
    a3 = shot("A3")
    print("== away and back again ==")
    head(POSE_B); time.sleep(1.5)
    head(POSE_A); time.sleep(1.5)
    a4 = shot("A4")

    pts = [p for p in (a1, a2, a3, a4) if p[0] is not None]
    if len(pts) >= 2:
        arr = np.array(pts, float)
        spread = arr.max(0) - arr.min(0)
        print(f"\nPIXEL SPREAD across repeats: dx={spread[0]:.0f}px dy={spread[1]:.0f}px")
        print("  <~5px = rock solid; 5-20px = usable w/ care; >20px = head not repeatable")


if __name__ == "__main__":
    main()
