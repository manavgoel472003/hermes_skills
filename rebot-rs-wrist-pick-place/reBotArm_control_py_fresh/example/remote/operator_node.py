#!/usr/bin/env python3
"""Operator node for remote teleop — runs wherever the human driver is.

Tracks the operator's hand with a plain webcam (MediaPipe Hands, no depth camera
required) and streams smooth, workspace-clamped XYZ targets to the relay, which
forwards them to whichever arm node is in the same room. So you can drive a robot
on the other side of the world; to *see* it, open the relay's /view URL.

Mapping (relative, re-zeroed each time you engage):
  * hand left/right  -> arm Y
  * hand up/down     -> arm Z
  * hand size (near/far) -> arm X reach   (only with --reach; mono depth is noisy)
Signs are easy to flip (--mirror/--flip-*); reuse of arm_teleop_constraints keeps
the smoothing, slew-limit and workspace clamp identical to the local demos.

Controls (preview window): e=engage/disengage  r=re-zero  o=open grip  c=close grip
                           q/Esc=quit

Run:
    pip install websocket-client            # once, into this venv (mediapipe already present)
    python operator_node.py --relay YOUR_VM:8765 --room b601
Test the link with no webcam (auto-sweeps a target):
    python operator_node.py --relay YOUR_VM:8765 --room b601 --fake
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent))          # protocol, relay_link (this dir)
sys.path.insert(0, str(_HERE.parents[1])) 	   # example/ : arm_teleop_constraints

import protocol as P                            # noqa: E402
import arm_teleop_constraints as C              # noqa: E402
from relay_link import RelayLink                # noqa: E402

# MediaPipe hand model (Tasks API — same model the depth demo uses)
HAND_MODEL = _HERE.parents[2] / "models" / "hand_landmarker.task"


def _operator_axis_map(reach: bool, mirror: bool,
                       flip_y: bool, flip_z: bool, flip_x: bool) -> C.AxisMap:
    """AxisMap for a 2D webcam whose hand coords are already pre-scaled into arm
    metres (so scale stays 1.0). image-x -> arm Y, image-y -> arm Z,
    hand-size -> arm X reach."""
    y_sign = -1.0 if mirror else 1.0
    if flip_y:
        y_sign = -y_sign
    return C.AxisMap(
        arm_x_from="z", arm_x_sign=(-1.0 if flip_x else 1.0) if reach else 0.0,
        arm_y_from="x", arm_y_sign=y_sign,
        arm_z_from="y", arm_z_sign=(1.0 if flip_z else -1.0),
        scale=1.0,
    )


def _hand_from_landmarks(lm, w: int, h: int, y_span: float, z_span: float,
                         reach_span: float, reach: bool) -> np.ndarray:
    """MediaPipe hand landmarks -> pre-scaled pseudo-3D hand position (arm metres).
    Anchor on landmark 9 (middle-finger MCP); size from wrist(0)->MCP(9) span."""
    u, v = lm[9].x, lm[9].y                    # normalized [0,1]
    dx = (lm[0].x - lm[9].x) * w
    dy = (lm[0].y - lm[9].y) * h
    size = float(np.hypot(dx, dy)) / float(np.hypot(w, h))   # scale-invariant
    hx = (u - 0.5) * y_span
    hy = (v - 0.5) * z_span
    hz = (size * reach_span) if reach else 0.0
    return np.array([hx, hy, hz], dtype=np.float64)


def run_camera(args, link: RelayLink) -> None:
    import mediapipe as mp
    if not HAND_MODEL.exists():
        raise FileNotFoundError(f"hand model missing: {HAND_MODEL}")
    landmarker = mp.tasks.vision.HandLandmarker.create_from_options(
        mp.tasks.vision.HandLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(HAND_MODEL)),
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_hands=1,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        ))

    cap = cv2.VideoCapture(int(args.camera) if str(args.camera).isdigit() else args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    if not cap.isOpened():
        raise RuntimeError(f"could not open webcam {args.camera!r}")

    axis = _operator_axis_map(args.reach, args.mirror, args.flip_y, args.flip_z, args.flip_x)
    mapper = C.HandFollowMapper(axis_map=axis, anchor=C.ARM_HOME)
    engaged = bool(args.auto_engage)
    grip = C.GRIPPER_MIN_OPEN_FRACTION
    last = time.monotonic()
    last_ts_ms = -1
    view_url = f"http://{args.relay.split('://')[-1]}/view?room={args.room}"
    print(f"[operator] driving room={args.room!r}; watch the arm at: {view_url}")

    while True:
        ok, frame = cap.read()
        if not ok:
            time.sleep(0.02)
            continue
        if args.mirror:
            frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        now = time.monotonic()
        dt = now - last
        last = now
        ts_ms = max(int(now * 1000), last_ts_ms + 1)   # must strictly increase
        last_ts_ms = ts_ms
        result = landmarker.detect_for_video(mp_image, ts_ms)

        hand_xyz = None
        if result.hand_landmarks:
            lm = result.hand_landmarks[0]
            hand_xyz = _hand_from_landmarks(lm, w, h, args.y_span, args.z_span,
                                            args.reach_span, args.reach)
            cx, cy = int(lm[9].x * w), int(lm[9].y * h)
            cv2.circle(frame, (cx, cy), 8, (0, 220, 60) if engaged else (0, 180, 240), -1)

        target = mapper.update(hand_xyz if engaged else None, dt)
        link.send_text(P.target_msg(target[0], target[1], target[2], grip, engaged, time.time()))

        if not args.headless:
            hud = [f"target {np.round(target, 3)}   grip {grip:.2f}",
                   "e=engage  r=rezero  o/c=grip  q=quit"]
            P.draw_hud(frame, hud, engaged)
            cv2.putText(frame, view_url, (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (200, 200, 200), 1, cv2.LINE_AA)
            cv2.imshow("operator (remote teleop)", frame)
            k = cv2.waitKey(1) & 0xFF
            if k in (ord("q"), 27):
                break
            elif k == ord("e"):
                engaged = not engaged
                if engaged:
                    mapper.reset(anchor=C.ARM_HOME)   # re-zero neutral on engage
                print(f"[operator] {'ENGAGED' if engaged else 'disengaged'}")
            elif k == ord("r"):
                mapper.reset(anchor=C.ARM_HOME)
                print("[operator] re-zeroed neutral")
            elif k == ord("o"):
                grip = C.GRIPPER_MAX_OPEN_FRACTION
            elif k == ord("c"):
                grip = C.GRIPPER_MIN_OPEN_FRACTION
        else:
            time.sleep(max(0.0, 1 / 30 - (time.monotonic() - now)))

    cap.release()
    cv2.destroyAllWindows()


def run_fake(args, link: RelayLink) -> None:
    """No webcam: sweep a target around HOME so you can verify the whole chain."""
    print("[operator] FAKE: auto-engaged, sweeping a target (no camera).")
    lo, hi = C.workspace_bounds()
    ctr = 0.5 * (lo + hi)
    amp = 0.35 * (hi - lo)
    t0 = time.time()
    while True:
        t = time.time() - t0
        tgt = ctr + amp * np.array([0.0, np.sin(t), np.cos(t * 0.7)])
        tgt = C.clamp_workspace(tgt)
        link.send_text(P.target_msg(tgt[0], tgt[1], tgt[2],
                                    C.GRIPPER_MIN_OPEN_FRACTION, True, time.time()))
        time.sleep(0.05)


def main() -> None:
    ap = argparse.ArgumentParser(description="Operator (hand-tracking) node for remote teleop")
    ap.add_argument("--relay", required=True, help="relay host:port (or ws://host:port)")
    ap.add_argument("--room", default=P.DEFAULT_ROOM)
    ap.add_argument("--camera", default="0", help="webcam index")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--reach", action="store_true",
                    help="also control arm X (reach) from hand size (noisy on mono webcams)")
    ap.add_argument("--y-span", type=float, default=0.30, help="arm metres across the image width")
    ap.add_argument("--z-span", type=float, default=0.18, help="arm metres across the image height")
    ap.add_argument("--reach-span", type=float, default=0.6, help="reach gain from hand size")
    ap.add_argument("--mirror", action="store_true", help="mirror the view (selfie) + flip Y sign")
    ap.add_argument("--flip-x", action="store_true")
    ap.add_argument("--flip-y", action="store_true")
    ap.add_argument("--flip-z", action="store_true")
    ap.add_argument("--headless", action="store_true", help="no preview window")
    ap.add_argument("--auto-engage", action="store_true", help="engage at startup (headless)")
    ap.add_argument("--fake", action="store_true", help="no webcam; sweep a target (link test)")
    args = ap.parse_args()

    url = P.build_ws_url(args.relay, args.room, "operator")
    link = RelayLink(url, name="operator")
    link.start()
    # give the socket a moment to connect before the first send
    time.sleep(0.5)

    try:
        if args.fake:
            run_fake(args, link)
        else:
            run_camera(args, link)
    except KeyboardInterrupt:
        pass
    finally:
        link.stop()


if __name__ == "__main__":
    main()
