#!/usr/bin/env python3
"""Follower node driven by the VERIFIED reBotArm_control_py `RebotArm` stack
(RobStride), instead of the lerobot Seeed plugin.

Why: the Seeed RS plugin passes leader degrees straight to the motors (no
RobStride position gains written) and the arm mirrors the leader (whole-arm
reversed). This follower uses the tested RebotArm RS control (arm group in
POS_VEL with the yaml's pos gains, gripper in MIT) and applies a per-joint SIGN
(default -1 on all 6 arm joints -> un-mirror) + OFFSET so the follower tracks the
leader in the correct direction.

It still receives the leader's `{<motor>.pos: deg}` action from the relay (so the
leader keeps running as `leader_node.py` in the conda env), maps it onto the
RebotArm joints, and drives them.

Runs in the FRESH .venv (py3.10 — has RebotArm + motorbridge 0.4.9):
    cd ~/rebot_lerobot/reBotArm_control_py_fresh
    sudo ip link set can0 up type can bitrate 1000000
    .venv/bin/python example/remote/follower_rebotarm.py --relay 127.0.0.1:8765 --room b601

SAFETY:
  * software SLEW limit: the command starts at the arm's CURRENT measured pose and
    eases toward the leader target at <= --max-joint-speed rad/s, so it never
    lurches even if the leader/follower zeros differ.
  * leader degrees are clamped to the B601 joint limits before mapping.
  * gripper clamped to a safe travel band; watchdog holds on link loss.
  * conservative POS_VEL velocity limit; Ctrl+C disables torque.
Per-joint signs/offsets are CLI-tunable — flip any joint that still goes the wrong
way without touching code.
"""
from __future__ import annotations

import argparse
import math
import signal
import sys
import threading
import time
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent))          # protocol, relay_link, camera
sys.path.insert(0, str(_HERE.parents[2]))      # reBotArm_control_py package

import protocol as P                            # noqa: E402
from relay_link import RelayLink                # noqa: E402
from camera import Camera                       # noqa: E402

# Leader (lerobot B601) joint -> RebotArm arm-group joint order (joint1..joint6).
ARM_ORDER = ["shoulder_pan", "shoulder_lift", "elbow_flex",
             "wrist_flex", "wrist_yaw", "wrist_roll"]
GRIP_NAME = "gripper"

# B601 joint limits (degrees, leader/B601 convention) — safety clamp on the leader
# value before it is mapped to the arm.
LEADER_LIMITS_DEG = {
    "shoulder_pan": (-145.0, 145.0), "shoulder_lift": (-170.0, 1.0),
    "elbow_flex": (-200.0, 1.0), "wrist_flex": (-80.0, 90.0),
    "wrist_yaw": (-90.0, 90.0), "wrist_roll": (-90.0, 90.0),
    "gripper": (-270.0, 0.0),
}


class Shared:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.action: dict | None = None
        self.last_rx = 0.0
        self.measured = np.zeros(7)
        self.cmd = np.zeros(7)
        self.following = False

    def set_action(self, a: dict) -> None:
        with self.lock:
            self.action = a
            self.last_rx = time.monotonic()

    def snap_action(self):
        with self.lock:
            return (dict(self.action) if self.action else None, self.last_rx)

    def publish(self, measured, cmd, following) -> None:
        with self.lock:
            self.measured = np.asarray(measured).copy()
            self.cmd = np.asarray(cmd).copy()
            self.following = bool(following)

    def snap_status(self):
        with self.lock:
            return self.measured.copy(), self.cmd.copy(), self.following


def _clamp_deg(name: str, deg: float) -> float:
    lo, hi = LEADER_LIMITS_DEG[name]
    return max(lo, min(hi, deg))


def _map_action_to_targets(action: dict, signs: np.ndarray, offsets: np.ndarray,
                           grip_sign: float, grip_offset: float,
                           grip_band: tuple[float, float]) -> np.ndarray:
    """Leader {<motor>.pos: deg} -> 7-vector of RebotArm joint targets (rad)."""
    out = np.zeros(7)
    for i, name in enumerate(ARM_ORDER):
        deg = _clamp_deg(name, float(action.get(f"{name}.pos", 0.0)))
        out[i] = signs[i] * math.radians(deg) + offsets[i]
    gdeg = _clamp_deg(GRIP_NAME, float(action.get(f"{GRIP_NAME}.pos", 0.0)))
    grip = grip_sign * math.radians(gdeg) + grip_offset
    out[6] = float(np.clip(grip, grip_band[0], grip_band[1]))
    return out


def _control_loop(arm, state: Shared, signs, offsets, grip_sign, grip_offset,
                  grip_band, vlim: float, max_speed: float, watchdog: float,
                  stop: threading.Event) -> None:
    n_arm = arm.arm.num_joints           # 6
    # start the command at the current measured pose so nothing jumps
    meas0, _, _ = arm.get_state()
    cmd = np.asarray(meas0, dtype=np.float64).copy()
    if cmd.shape[0] < 7:
        cmd = np.zeros(7)
    print(f"[follower-rb] holding at measured pose {np.round(cmd, 2)}; waiting for leader…")

    vlim_arm = np.full(n_arm, vlim, dtype=np.float64)
    last_pub = 0.0
    dt = 0.01
    while not stop.is_set():
        t0 = time.perf_counter()
        action, last_rx = state.snap_action()
        fresh = action is not None and (time.monotonic() - last_rx) < watchdog
        if fresh:
            desired = _map_action_to_targets(action, signs, offsets,
                                             grip_sign, grip_offset, grip_band)
        else:
            desired = cmd                      # hold
        # software slew limit
        max_step = max_speed * dt
        cmd = cmd + np.clip(desired - cmd, -max_step, max_step)
        try:
            arm.arm.send_pos_vel(cmd[:n_arm], vlim_arm)
            if arm.has_gripper:
                arm.gripper.send_mit(np.array([cmd[6]]))
        except Exception as e:   # noqa: BLE001
            print(f"[follower-rb] send error: {e}")

        now = time.time()
        if now - last_pub > 0.2:
            last_pub = now
            meas, _, _ = arm.get_state()
            state.publish(meas, cmd, fresh)

        elapsed = time.perf_counter() - t0
        if elapsed < dt:
            time.sleep(dt - elapsed)


def _camera_loop(cam, link, state, quality, fps, stop) -> None:
    period = 1.0 / max(fps, 1.0)
    while not stop.is_set():
        t0 = time.time()
        frame = cam.read()
        if frame is None:
            time.sleep(0.05)
            continue
        meas, cmd, following = state.snap_status()
        hud = [f"RS follower (RebotArm)  {'live'}",
               "meas [" + " ".join(f"{math.degrees(v):.0f}" for v in meas) + "] deg"]
        P.draw_hud(frame, hud, following, active_label="FOLLOWING", idle_label="HOLD")
        jpeg = P.encode_jpeg(frame, quality)
        if jpeg:
            link.send_frame(jpeg)
        dt = time.time() - t0
        if dt < period:
            time.sleep(period - dt)


def _parse_vec(s: str | None, default: np.ndarray) -> np.ndarray:
    if not s:
        return default
    vals = [float(x) for x in s.replace(" ", "").split(",")]
    return np.array(vals, dtype=np.float64)


def main() -> None:
    ap = argparse.ArgumentParser(description="RebotArm-driven follower for remote leader-follower teleop")
    ap.add_argument("--relay", required=True)
    ap.add_argument("--room", default=P.DEFAULT_ROOM)
    ap.add_argument("--hw-yaml", default="rebotarm_rs.yaml")
    ap.add_argument("--signs", default=None,
                    help="6 comma-sep arm-joint signs (default -1,-1,-1,-1,-1,-1 = un-mirror)")
    ap.add_argument("--offsets", default=None, help="6 comma-sep arm-joint offsets in rad (default 0)")
    ap.add_argument("--grip-sign", type=float, default=1.0)
    ap.add_argument("--grip-offset", type=float, default=0.0)
    ap.add_argument("--grip-min", type=float, default=-4.5, help="gripper travel floor (rad, open)")
    ap.add_argument("--grip-max", type=float, default=0.0, help="gripper travel ceil (rad, closed)")
    ap.add_argument("--vlim", type=float, default=4.0, help="POS_VEL velocity limit (rad/s)")
    ap.add_argument("--max-joint-speed", type=float, default=1.2, help="software slew cap (rad/s)")
    ap.add_argument("--watchdog", type=float, default=0.4)
    ap.add_argument("--camera", default="none", help="cv2 index, 'test', or 'none' (no video)")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--fps", type=float, default=15.0)
    ap.add_argument("--jpeg-quality", type=int, default=P.JPEG_QUALITY)
    args = ap.parse_args()

    signs = _parse_vec(args.signs, np.full(6, -1.0))
    offsets = _parse_vec(args.offsets, np.zeros(6))

    from reBotArm_control_py.actuator import RebotArm    # noqa: E402

    state = Shared()

    def _on_text(msg: str) -> None:
        m = P.parse_msg(msg)
        if not m or m.get("type") != P.TYPE_ACTION:
            return
        a = m.get("a") or {}
        state.set_action({(k if k.endswith(".pos") else f"{k}.pos"): float(v) for k, v in a.items()})

    link = RelayLink(P.build_ws_url(args.relay, args.room, "arm"), on_text=_on_text, name="follower-rb")
    stop = threading.Event()

    print(f"[follower-rb] connecting RebotArm ({args.hw_yaml}) on the CAN bus…")
    arm = RebotArm(args.hw_yaml)
    arm.connect()
    arm.enable_all()
    time.sleep(0.2)
    arm.arm.mode_pos_vel()          # writes RobStride pos/vel gains + POS_VEL mode
    if arm.has_gripper:
        arm.gripper.mode_mit()
    print("[follower-rb] enabled: arm=POS_VEL, gripper=MIT")

    link.start()
    threading.Thread(target=_control_loop, name="ctrl", daemon=True,
                     args=(arm, state, signs, offsets, args.grip_sign, args.grip_offset,
                           (args.grip_min, args.grip_max), args.vlim, args.max_joint_speed,
                           args.watchdog, stop)).start()
    if args.camera != "none":
        cam = Camera(args.camera, args.width, args.height)
        threading.Thread(target=_camera_loop, name="cam", daemon=True,
                         args=(cam, link, state, args.jpeg_quality, args.fps, stop)).start()
    else:
        cam = None

    def _shutdown(*_a):
        stop.set()
    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    host = args.relay.split("://")[-1]
    print(f"[follower-rb] room={args.room!r}; signs={signs.tolist()}; watch: http://{host}/view?room={args.room}")
    try:
        while not stop.is_set():
            time.sleep(0.2)
    finally:
        stop.set()
        time.sleep(0.3)
        link.stop()
        if cam is not None:
            cam.release()
        print("[follower-rb] disabling torque + disconnecting")
        try:
            arm.disconnect()
        except Exception:   # noqa: BLE001
            pass


if __name__ == "__main__":
    main()
