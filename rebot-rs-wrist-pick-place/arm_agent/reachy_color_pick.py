"""GPU-free 'hand me the <color> one' demo: Reachy sees a colored object, the
B601 picks it and hands it over. Uses the Reachy bridge + the Reachy->base
calibration + smooth arm motion. This is the full pipeline without the VLM
(named-object detection needs the GPU, which is currently down).

Run (arm service :8001 + reachy bridge :8712 both up):
    python -m arm_agent.reachy_color_pick red
    python -m arm_agent.reachy_color_pick blue --no-handover
"""
from __future__ import annotations

import argparse, json, math, os, sys, time
import httpx, numpy as np, cv2

sys.path.insert(0, "/home/manav/rebot_lerobot/rebot_arm_service/src")
from arm_agent.reachy_client import ReachyBridgeClient
from rebot_arm_service.scene.calibration import TableCalibration

ARM = os.getenv("REBOT_ARM_SERVICE_URL", "http://127.0.0.1:8001")
CALIB_DIR = "/home/manav/rebot_lerobot/rebot_arm_service/calib"
PITCH = 1.30
# generous HSV ranges for common object colors
COLORS = {
    "red":    [((0, 110, 80), (10, 255, 255)), ((170, 110, 80), (180, 255, 255))],
    "orange": [((8, 130, 130), (22, 255, 255))],
    "yellow": [((22, 110, 110), (34, 255, 255))],
    "green":  [((38, 70, 50), (85, 255, 255))],
    "blue":   [((95, 100, 50), (130, 255, 255))],
    "purple": [((130, 60, 50), (160, 255, 255))],
}


def move(x, y, z, dur=2.5, yaw=None):
    yaw = math.atan2(y, x) if yaw is None else yaw
    r = httpx.post(f"{ARM}/move/pose", json={"x": x, "y": y, "z": z, "roll": 0.0,
                   "pitch": PITCH, "yaw": yaw, "mode": "smooth", "duration": dur}, timeout=45)
    return r.status_code == 200


def gripper(action):
    httpx.post(f"{ARM}/gripper", json={"action": action}, timeout=15)


def home():
    httpx.post(f"{ARM}/home", json={}, timeout=25)


def workspace_roi(cal, shape, xr=(0.19, 0.37), yr=(-0.16, 0.16)):
    """Mask of the reachable table region in the image (project base corners)."""
    Hinv = np.linalg.inv(cal.H)
    pts = []
    for x in xr:
        for y in yr:
            q = Hinv @ np.array([x, y, 1.0]); pts.append([q[0]/q[2], q[1]/q[2]])
    pts = np.array([pts[0], pts[1], pts[3], pts[2]], np.int32)  # order into a quad
    m = np.zeros(shape[:2], np.uint8); cv2.fillPoly(m, [pts], 255)
    return m


def detect(cal, gv, reachy, color, out):
    reachy.head(pose=gv, duration=1.2); time.sleep(0.4)
    img = reachy.frame_bgr(warmup=6)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = np.zeros(hsv.shape[:2], np.uint8)
    for lo, hi in COLORS[color]:
        mask = cv2.bitwise_or(mask, cv2.inRange(hsv, np.array(lo), np.array(hi)))
    mask = cv2.bitwise_and(mask, workspace_roi(cal, img.shape))  # only reachable table
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    cn, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cn = [c for c in cn if cv2.contourArea(c) > 200]
    if not cn:
        cv2.imwrite(f"{out}/pick_view.jpg", img)
        return None, img
    c = max(cn, key=cv2.contourArea)
    M = cv2.moments(c); u, v = M["m10"]/M["m00"], M["m01"]/M["m00"]
    x, y, z = cal.pixel_to_pose(u, v)
    ov = img.copy(); cv2.drawContours(ov, [c], -1, (0, 255, 0), 2)
    cv2.drawMarker(ov, (int(u), int(v)), (0, 0, 255), cv2.MARKER_CROSS, 40, 3)
    cv2.imwrite(f"{out}/pick_view.jpg", ov)
    return (float(u), float(v), float(x), float(y), float(z)), img


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("color", choices=list(COLORS))
    ap.add_argument("--grasp-z", type=float, default=0.02)
    ap.add_argument("--no-handover", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="detect + print pose, no motion")
    args = ap.parse_args()
    out = "/tmp/claude-1000/-home-manav-rebot-lerobot/b78a7a4f-2dd2-43aa-be1c-fd91a7c4b649/scratchpad"
    os.environ.pop("REACHY_MOCK", None)
    reachy = ReachyBridgeClient()
    cal = TableCalibration.load(f"{CALIB_DIR}/reachy_cam.json")
    gv = json.load(open(f"{CALIB_DIR}/reachy_calibration_meta.json"))["grasp_view_head_pose"]

    print("homing to clear Reachy's view...", flush=True); home(); time.sleep(1.0)
    det, _ = detect(cal, gv, reachy, args.color, out)
    if det is None:
        print(f"No {args.color} object found in the reachable workspace (see pick_view.jpg).")
        return 1
    u, v, x, y, z = det
    print(f"{args.color} object: pixel=({u:.0f},{v:.0f}) -> base=({x:+.3f},{y:+.3f})", flush=True)
    if args.dry_run:
        return 0
    print("grasping...", flush=True)
    gripper("open"); time.sleep(0.5)
    move(x, y, 0.12); move(x, y, args.grasp_z); time.sleep(0.6)
    gripper("close"); time.sleep(1.0)
    move(x, y, 0.16)
    if not args.no_handover:
        print("presenting to hand...", flush=True)
        move(0.30, 0.0, 0.22, yaw=0.0); time.sleep(0.5)
        print("releasing in 2s", flush=True); time.sleep(2.0)
        gripper("open")
    home()
    print("done.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
