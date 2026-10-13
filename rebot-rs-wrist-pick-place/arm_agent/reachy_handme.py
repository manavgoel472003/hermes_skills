"""Full 'hand me the <object>' demo with a live view window.

Reachy names the object (LocateAnything VLM, GPU), the B601 picks it precisely,
then Reachy LOOKS AROUND to find your open palm and the arm reaches toward it and
places the object -- but only AFTER you confirm (press ENTER in the terminal).

A Tkinter window shows what Reachy sees the whole time, with overlays: the grasp
target, the gripper tip, your hand landmarks, and the gripper<->hand distance.

    python -m arm_agent.reachy_handme "green sponge" --color green
    python -m arm_agent.reachy_handme "sponge"

Needs: arm service :8001 (REBOT_DETECTOR=locate_anything), reachy bridge :8712,
and rebot_arm_service/calib/reachy_cam.json.
"""
from __future__ import annotations

import argparse, json, math, os, sys, threading, time
import httpx, numpy as np, cv2

sys.path.insert(0, "/home/manav/rebot_lerobot/rebot_arm_service/src")
from arm_agent.reachy_client import ReachyBridgeClient
from arm_agent.reachy_color_pick import workspace_roi, COLORS
from rebot_arm_service.scene.calibration import TableCalibration

ARM = os.getenv("REBOT_ARM_SERVICE_URL", "http://127.0.0.1:8001")
CALIB = "/home/manav/rebot_lerobot/rebot_arm_service/calib"
HAND_MODEL = "/home/manav/rebot_lerobot/models/hand_landmarker.task"

# reachable table workspace (base frame, metres)
X_MIN, X_MAX, Y_MIN, Y_MAX = 0.19, 0.37, -0.17, 0.17
# where the arm holds the object up for the handover (reachable, gripper visible)
PRESENT = (0.30, 0.0, 0.19)
PRESENT_PITCH = 1.10
# small sideways "reach" toward the hand (base y per deg of head bearing), clamped
REACH_Y_PER_DEG = 0.0025
REACH_Y_MAX = 0.10


# ---------------------------------------------------------------- arm helpers
def move(x, y, z, dur=2.5, yaw=None, pitch=1.30):
    yaw = math.atan2(y, x) if yaw is None else yaw
    return httpx.post(f"{ARM}/move/pose", json={"x": x, "y": y, "z": z, "roll": 0.0,
                      "pitch": pitch, "yaw": yaw, "mode": "smooth", "duration": dur}, timeout=45).status_code == 200


def gripper(a):
    httpx.post(f"{ARM}/gripper", json={"action": a}, timeout=20)


def home():
    httpx.post(f"{ARM}/home", json={}, timeout=50)


def vlm_boxes(object):
    """Ask the arm service (LocateAnything on the reachy camera) for the object;
    return its VLM box(es), even when the service rejects the grasp geometry."""
    r = httpx.post(f"{ARM}/skill/detect_object", params={"dry_run": "false"},
                   json={"object": object, "camera": "reachy"}, timeout=120).json()
    det = r.get("detail") or r
    return [o["box"] for o in (det.get("vlm_detections") or []) if o.get("box")]


# ------------------------------------------------------------- shared UI state
class Shared:
    def __init__(self):
        self.lock = threading.Lock()
        self.raw = None            # latest BGR frame from the grab thread
        self.status = "starting..."
        self.prompt = ""           # big bottom banner (e.g. PRESS ENTER)
        self.hand = None           # (u,v) palm centre
        self.landmarks = None      # [(u,v), ...] hand landmarks
        self.grip = None           # (u,v) gripper tip
        self.boxes = []            # [(x1,y1,x2,y2,label), ...]
        self.target = None         # (u,v) grasp target
        self.done = False
        self.paused = False        # pause grabbing (while the arm service reads the cam)


S = Shared()


def set_status(t):
    with S.lock:
        S.status = t
    print(t, flush=True)


def set_prompt(t):
    with S.lock:
        S.prompt = t


def latest():
    with S.lock:
        return None if S.raw is None else S.raw.copy()


def _grab_loop(reachy):
    while not S.done:
        if S.paused:
            time.sleep(0.05); continue
        try:
            f = reachy.frame_bgr(warmup=1)
        except Exception:
            time.sleep(0.15); continue
        if f is not None:
            with S.lock:
                S.raw = f
        time.sleep(0.02)


def _draw(frame):
    ov = frame.copy(); H, W = ov.shape[:2]
    with S.lock:
        hand, lms, grip = S.hand, S.landmarks, S.grip
        boxes, target, status, prompt = list(S.boxes), S.target, S.status, S.prompt
    for (x1, y1, x2, y2, label) in boxes:
        cv2.rectangle(ov, (int(x1), int(y1)), (int(x2), int(y2)), (0, 200, 255), 2)
        cv2.putText(ov, label, (int(x1), int(y1) - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)
    if target:
        cv2.drawMarker(ov, (int(target[0]), int(target[1])), (0, 0, 255), cv2.MARKER_CROSS, 44, 3)
    if lms:
        for (u, v) in lms:
            cv2.circle(ov, (int(u), int(v)), 3, (0, 255, 0), -1)
    if hand:
        cv2.circle(ov, (int(hand[0]), int(hand[1])), 15, (0, 255, 0), 2)
        cv2.putText(ov, "palm", (int(hand[0]) + 16, int(hand[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    if grip:
        cv2.circle(ov, (int(grip[0]), int(grip[1])), 12, (0, 140, 255), 2)
        cv2.putText(ov, "grip", (int(grip[0]) + 14, int(grip[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 140, 255), 2)
    if grip and hand:
        cv2.line(ov, (int(grip[0]), int(grip[1])), (int(hand[0]), int(hand[1])), (255, 255, 0), 2)
        d = math.hypot(hand[0] - grip[0], hand[1] - grip[1])
        cv2.putText(ov, f"{d:.0f}px", (int((grip[0] + hand[0]) / 2), int((grip[1] + hand[1]) / 2) - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    cv2.rectangle(ov, (0, 0), (W, 44), (0, 0, 0), -1)
    cv2.putText(ov, status, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    if prompt:
        cv2.rectangle(ov, (0, H - 58), (W, H), (0, 0, 0), -1)
        cv2.putText(ov, prompt, (12, H - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.95, (0, 255, 255), 2)
    return ov


# ----------------------------------------------------------------- detectors
def _hand_detector():
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision
    opts = vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=HAND_MODEL),
        running_mode=vision.RunningMode.IMAGE, num_hands=1)
    return vision.HandLandmarker.create_from_options(opts), mp


def _detect_hand(detector, mp, frame):
    """Return (palm_center_uv, landmarks_uv, is_open) or (None, None, False)."""
    res = detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB,
                                   data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
    if not res.hand_landmarks:
        return None, None, False
    H, W = frame.shape[:2]
    lm = res.hand_landmarks[0]
    pts = [(p.x * W, p.y * H) for p in lm]
    palm_idx = [0, 5, 9, 13, 17]
    cx = float(np.mean([pts[i][0] for i in palm_idx]))
    cy = float(np.mean([pts[i][1] for i in palm_idx]))
    # openness: mean fingertip distance from wrist, normalised by palm width
    wrist = np.array(pts[0]); palm_w = np.linalg.norm(np.array(pts[5]) - np.array(pts[17])) + 1e-6
    tips = [4, 8, 12, 16, 20]
    spread = float(np.mean([np.linalg.norm(np.array(pts[t]) - wrist) for t in tips])) / palm_w
    return (cx, cy), pts, spread > 1.6


def _gripper_pixel(frame):
    """Locate the gripper (bright orange fingertips) in the frame."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    m = cv2.inRange(hsv, np.array((6, 140, 140)), np.array((26, 255, 255)))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    cn, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cn = [c for c in cn if cv2.contourArea(c) > 150]
    if not cn:
        return None
    c = max(cn, key=cv2.contourArea); M = cv2.moments(c)
    return (M["m10"] / M["m00"], M["m01"] / M["m00"])


def _color_blob(frame, cal, box, color):
    """Largest colour blob inside the VLM box AND the reachable workspace ROI.
    Returns the contour, or None."""
    H, W = frame.shape[:2]
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    m = np.zeros((H, W), np.uint8)
    for lo, hi in COLORS[color]:
        m = cv2.bitwise_or(m, cv2.inRange(hsv, np.array(lo), np.array(hi)))
    if box is not None:
        x1, y1, x2, y2 = [int(v) for v in box]
        bm = np.zeros((H, W), np.uint8); bm[y1:y2, x1:x2] = 255
        m = cv2.bitwise_and(m, bm)
    m = cv2.bitwise_and(m, workspace_roi(cal, frame.shape))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    cn, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cn = [c for c in cn if cv2.contourArea(c) > 200]
    return max(cn, key=cv2.contourArea) if cn else None


def grasp_from_frame(cal, frame, box, color):
    """Grasp (x, y, yaw, (u,v)) from the coloured blob. Uses the blob centroid for
    position and its long axis (mapped to base frame) for a short-axis grasp yaw.
    Falls back to the box centre / radial yaw when there's no usable blob."""
    H, W = frame.shape[:2]
    u = v = None; yaw = None
    if color and color in COLORS:
        c = _color_blob(frame, cal, box, color)
        if c is not None:
            M = cv2.moments(c); u, v = M["m10"] / M["m00"], M["m01"] / M["m00"]
            (_, _), (rw, rh), ang = cv2.minAreaRect(c)
            if max(rw, rh) / (min(rw, rh) + 1e-6) > 1.4:      # only orient if elongated
                a = math.radians(ang if rw >= rh else ang + 90.0)  # long-axis dir (px)
                du, dv = math.cos(a), math.sin(a)
                p1 = cal.pixel_to_pose(u - 20 * du, v - 20 * dv)
                p2 = cal.pixel_to_pose(u + 20 * du, v + 20 * dv)
                yaw = math.atan2(p2[1] - p1[1], p2[0] - p1[0])
    if u is None:
        if box is None:
            return None
        x1, y1, x2, y2 = [float(t) for t in box]
        u, v = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    x, y, _ = cal.pixel_to_pose(u, v)
    if yaw is None:
        yaw = math.atan2(y, x)
    return float(x), float(y), float(yaw), (float(u), float(v))


def object_on_table(cal, frame, color, near_uv, tol_px=90.0):
    """True if a colour blob of `color` remains in the workspace near `near_uv`
    (i.e. the object is still on the table -> grasp failed)."""
    c = _color_blob(frame, cal, None, color)
    if c is None:
        return False
    M = cv2.moments(c)
    if M["m00"] == 0:
        return False
    cu, cv_ = M["m10"] / M["m00"], M["m01"] / M["m00"]
    return math.hypot(cu - near_uv[0], cv_ - near_uv[1]) < tol_px


# ------------------------------------------------------------------- phases
def pick(reachy, cal, gv, obj, color, grasp_z):
    """VLM-detect, grasp (oriented), verify the object left the table, retry once.
    Returns True on a verified pick."""
    set_status(f"looking for '{obj}' with the VLM...")
    home(); time.sleep(0.6)
    S.paused = True                       # give the arm service the camera
    boxes = vlm_boxes(obj)
    S.paused = False
    if not boxes:
        set_status(f"VLM did not find '{obj}'."); return False
    # head is already parked at the pick view (grasp_view) and stays there for the
    # whole pick -- just grab a fresh frame at that fixed, calibrated pose.
    time.sleep(0.3)
    frame = _wait_frame()
    if frame is None:
        set_status("no camera frame."); return False
    g = grasp_from_frame(cal, frame, boxes[0], color)
    if g is None:
        set_status("could not localise the object."); return False
    x, y, yaw, (u, vv) = g
    with S.lock:
        S.boxes = [(*boxes[0], obj)]; S.target = (u, vv)
    set_status(f"target base=({x:+.3f},{y:+.3f}) yaw={math.degrees(yaw):+.0f}deg")
    if not (X_MIN <= x <= X_MAX and Y_MIN <= y <= Y_MAX):
        set_status(f"grasp ({x:.3f},{y:.3f}) outside reach; aborting."); return False

    for attempt in range(2):
        set_status(f"grasping (try {attempt + 1})...")
        gripper("open"); time.sleep(0.3)
        move(x, y, 0.12, yaw=yaw)
        move(x, y, grasp_z - 0.008 * attempt, yaw=yaw); time.sleep(0.5)
        gripper("close"); time.sleep(1.0)
        move(x, y, 0.18, yaw=yaw)
        # verify: swing aside so the table is unobstructed, then look for the object
        # (head never moved -- still at the calibrated pick view)
        move(0.22, -0.02, 0.20, yaw=0.0); time.sleep(0.3)
        chk = _wait_frame()
        with S.lock:
            S.boxes = []
        if chk is None or not object_on_table(cal, chk, color, (u, vv)):
            set_status("pick verified (object left the table).")
            with S.lock:
                S.target = None
            return True
        set_status("object still on the table -> retrying.")
        gripper("open"); time.sleep(0.3)
        move(x, y, 0.12, yaw=yaw)
    set_status("pick failed after retry.")
    return False


def find_palm(reachy, detector, mp, timeout=30.0):
    """Sweep Reachy's head to find an OPEN palm held up; then keep it centred.
    Returns the head bearing (deg, + = palm to Reachy's right) or None."""
    set_status("looking around for your open palm...")
    sweep = [dict(yaw_deg=0, pitch_deg=-4, roll_deg=0, body_yaw_deg=b) for b in (0, -25, 25, -12, 12)]
    t0 = time.time(); i = 0
    while time.time() - t0 < timeout:
        pose = sweep[i % len(sweep)]; i += 1
        reachy.head(pose=pose, duration=0.6); time.sleep(0.25)
        # give the palm a moment to be seen at this view
        for _ in range(6):
            frame = latest()
            if frame is None:
                time.sleep(0.05); continue
            hand, lms, is_open = _detect_hand(detector, mp, frame)
            with S.lock:
                S.hand, S.landmarks = hand, lms
            if hand is not None and is_open:
                set_status("palm found -> centering.")
                # centre the head on the palm so the window frames it
                for _ in range(3):
                    f2 = latest()
                    if f2 is None:
                        break
                    h2, l2, _ = _detect_hand(detector, mp, f2)
                    with S.lock:
                        S.hand, S.landmarks = h2, l2
                    if h2 is None:
                        break
                    reachy.head(look_pixel=(h2[0], h2[1]), duration=0.25); time.sleep(0.2)
                pose_now = reachy.health().get("head_pose", {})
                return float(pose_now.get("body_yaw_deg", 0.0)) + float(pose_now.get("yaw_deg", 0.0))
            time.sleep(0.05)
    set_status("couldn't find an open palm (timeout).")
    return None


def present_and_release(reachy, detector, mp, bearing_deg):
    """Reach the object toward the palm's bearing, keep tracking it in the window,
    then release ONLY after the user confirms with ENTER."""
    y = 0.0 if bearing_deg is None else float(np.clip(-REACH_Y_PER_DEG * bearing_deg, -REACH_Y_MAX, REACH_Y_MAX))
    px, _, pz = PRESENT
    set_status(f"reaching toward your palm (y={y:+.2f})...")
    move(px, y, pz, dur=3.0, yaw=0.0, pitch=PRESENT_PITCH)

    # keep tracking the palm + gripper for the live view while we wait
    stop = threading.Event()

    def _track():
        while not stop.is_set():
            frame = latest()
            if frame is None:
                time.sleep(0.05); continue
            hand, lms, _ = _detect_hand(detector, mp, frame)
            grip = _gripper_pixel(frame)
            with S.lock:
                S.hand, S.landmarks, S.grip = hand, lms, grip
            if hand is not None:
                reachy.head(look_pixel=(hand[0], hand[1]), duration=0.2)
            time.sleep(0.15)

    tracker = threading.Thread(target=_track, daemon=True); tracker.start()
    set_prompt("PALM READY  ->  press ENTER in the terminal to release")
    set_status("confirm the handover in the terminal (ENTER).")
    try:
        input("\n>>> Palm in place? Press ENTER to release the object (Ctrl-C to abort): ")
    except (EOFError, KeyboardInterrupt):
        pass
    stop.set(); time.sleep(0.2)
    set_prompt(""); set_status("releasing.")
    gripper("open"); time.sleep(0.8)


def _wait_frame(tries=25):
    for _ in range(tries):
        f = latest()
        if f is not None:
            return f
        time.sleep(0.05)
    return None


def worker(reachy, cal, gv, obj, color, grasp_z, skip_pick):
    try:
        # Park the head at the pick view (grasp_view) -- this is the head's HOME.
        # It stays here for the whole pick and only moves during palm-finding.
        set_status("parking head at pick view...")
        reachy.head(pose=gv, duration=1.2); time.sleep(0.4)
        if not skip_pick:
            if not pick(reachy, cal, gv, obj, color, grasp_z):
                return
        else:
            set_status("skip-pick: assuming the object is already held.")
        detector, mp = _hand_detector()
        bearing = find_palm(reachy, detector, mp)          # the ONLY head motion
        present_and_release(reachy, detector, mp, bearing)
        detector.close()
        set_status("done -> homing."); reachy.head(pose=gv, duration=0.8); home()
        set_status("done.")
    except Exception as exc:
        set_status(f"error: {exc}")
        import traceback; traceback.print_exc()
    finally:
        time.sleep(1.0)
        S.done = True


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("object")
    ap.add_argument("--color", default=None)
    ap.add_argument("--grasp-z", type=float, default=0.02)
    ap.add_argument("--skip-pick", action="store_true", help="skip the pick (test look-around/handover)")
    ap.add_argument("--no-window", action="store_true", help="run headless (no live view)")
    args = ap.parse_args()
    os.environ.pop("REACHY_MOCK", None)

    reachy = ReachyBridgeClient()
    cal = TableCalibration.load(f"{CALIB}/reachy_cam.json")
    gv = json.load(open(f"{CALIB}/reachy_calibration_meta.json"))["grasp_view_head_pose"]

    threading.Thread(target=_grab_loop, args=(reachy,), daemon=True).start()
    wt = threading.Thread(target=worker, args=(reachy, cal, gv, args.object, args.color,
                                               args.grasp_z, args.skip_pick), daemon=True)
    wt.start()

    if args.no_window:
        while not S.done:
            time.sleep(0.2)
        return 0

    import tkinter as tk
    from PIL import Image, ImageTk
    root = tk.Tk(); root.title("Reachy view")
    label = tk.Label(root); label.pack()

    def update():
        frame = latest()
        if frame is not None:
            ov = cv2.resize(_draw(frame), (960, 540))
            img = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(ov, cv2.COLOR_BGR2RGB)))
            label.img = img; label.config(image=img)
        if S.done:
            root.after(600, root.destroy); return
        root.after(45, update)

    root.after(200, update)
    try:
        root.mainloop()
    finally:
        S.done = True
        time.sleep(0.2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
