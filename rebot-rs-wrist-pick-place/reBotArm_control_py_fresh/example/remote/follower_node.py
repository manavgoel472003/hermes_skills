#!/usr/bin/env python3
"""Follower node for remote LEADER-FOLLOWER teleop — runs where the arm is.

Connects the B601 follower (RS *or* DM) through the lerobot plugin
(`seeed_b601_rs_follower` / `seeed_b601_dm_follower`), opens at least one camera,
and links to the self-hosted relay:

  * RECEIVES the leader's joint-position action dict and applies it with the
    official plugin's `send_action`, including its RS mapping and joint limits.
  * SENDS its camera as JPEG frames (status HUD overlaid) so browser viewers
    anywhere can watch, plus periodic joint state.

Pairs with leader_node.py. Bidirectional: either machine can be leader or
follower, so you can drive their arm or they can drive yours.

SAFETY AND SMOOTHNESS:
  * every action is clipped to the follower's configured joint_limits (plugin)
  * control runs at a fixed rate (60 Hz max), independent of network/camera jitter
  * only the newest sequenced packet is applied; stale packets are rejected
  * feedback is polled at a lower rate so it does not flood the seven-motor CAN bus
  * watchdog: if no action arrives within --watchdog s (link drop/pause), we stop
    sending and the RS MIT controller holds its last commanded pose
  * Ctrl+C disconnects (plugin disables torque on disconnect by default)

Run in the verified `rebot_rs` env from the Seeed B601-RS guide:
    conda activate rebot_rs
    sudo ip link set can0 down 2>/dev/null
    sudo ip link set can0 type can bitrate 1000000 restart-ms 100
    sudo ip link set can0 up
    python follower_node.py --relay YOUR_VM:8765 --room b601 \
        --arm rs --port can0 --can-adapter socketcan --id follower1 --camera 0
    # Damiao serial bridge instead:
    python follower_node.py --relay YOUR_VM:8765 --room b601 \
        --arm dm --port /dev/ttyACM0 --can-adapter damiao --id follower1 --camera 0
No-hardware link/video test:
    python follower_node.py --relay YOUR_VM:8765 --room b601 --fake --camera test
"""
from __future__ import annotations

import argparse
import importlib.metadata
import math
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent))          # protocol, relay_link, camera (this dir)

import protocol as P                            # noqa: E402
from relay_link import RelayLink                # noqa: E402  (websocket-client)
from camera import Camera                       # noqa: E402

JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex",
          "wrist_yaw", "wrist_roll", "gripper"]
ACTION_KEYS = {f"{joint}.pos" for joint in JOINTS}


def _version_tuple(value: str) -> tuple[int, ...]:
    parts = []
    for piece in value.split("."):
        digits = "".join(ch for ch in piece if ch.isdigit())
        if not digits:
            break
        parts.append(int(digits))
    return tuple(parts)


def preflight(arm: str, port: str, can_adapter: str) -> None:
    required = {
        "lerobot": "0.4.4",
        "lerobot-robot-seeed-b601": "1.0.0",
        "motorbridge": "0.5.0",
        "websocket-client": "1.0.0",
    }
    for package, minimum in required.items():
        try:
            actual = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError as exc:
            raise RuntimeError(f"missing package {package!r}; activate conda env rebot_rs") from exc
        if _version_tuple(actual) < _version_tuple(minimum):
            raise RuntimeError(f"{package} {actual} is too old; need >= {minimum}")
    if arm == "rs" and can_adapter != "socketcan":
        raise RuntimeError("B601-RS requires --can-adapter socketcan")
    if arm == "rs":
        check = subprocess.run(
            ["ip", "-details", "link", "show", port], text=True,
            capture_output=True, check=False,
        )
        if check.returncode:
            raise RuntimeError(f"SocketCAN interface {port!r} does not exist")
        details = check.stdout
        if "UP" not in details.splitlines()[0] or "bitrate 1000000" not in details:
            raise RuntimeError(
                f"{port} must be UP at 1,000,000 bit/s; follow the Seeed CAN setup commands"
            )
        if "restart-ms 100" not in details:
            print(f"[follower] WARNING: {port} is not configured with restart-ms 100")
    print("[follower] preflight OK: official B601 plugin and MotorBridge stack")


def build_follower(arm: str, port: str, can_adapter: str, robot_id: str,
                   max_rel: float | None):
    """Instantiate + return a connected lerobot B601 follower (no cameras — we
    stream our own webcam). Importing the plugin registers its config subclass."""
    if arm == "rs":
        from lerobot_robot_seeed_b601.seeed_b601_rs_follower import SeeedB601RSFollower as Cls
        from lerobot_robot_seeed_b601.config_seeed_b601_rs_follower import (
            SeeedB601RSFollowerConfig as Cfg)
    else:
        from lerobot_robot_seeed_b601.seeed_b601_dm_follower import SeeedB601DMFollower as Cls
        from lerobot_robot_seeed_b601.config_seeed_b601_dm_follower import (
            SeeedB601DMFollowerConfig as Cfg)
    cfg = Cfg(port=port, can_adapter=can_adapter, id=robot_id,
              cameras={}, max_relative_target=max_rel)
    return Cls(cfg)


class FollowerState:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.action: dict | None = None
        self.action_number = 0
        self.last_rx = 0.0
        self.session: str | None = None
        self.last_seq = -1
        self.measured: dict[str, float] = {}
        self.following = False
        self.connected = False

    def set_action(self, a: dict, session: str, seq: int) -> bool:
        with self.lock:
            if session != self.session:
                self.session = session
                self.last_seq = -1
            if seq <= self.last_seq:
                return False
            self.last_seq = seq
            self.action = a
            self.action_number += 1
            self.last_rx = time.monotonic()
            return True

    def snapshot_action(self):
        with self.lock:
            return (dict(self.action) if self.action else None, self.last_rx,
                    self.action_number)

    def publish(self, measured: dict, following: bool, connected: bool) -> None:
        with self.lock:
            self.measured = dict(measured)
            self.following = following
            self.connected = connected

    def snapshot_status(self):
        with self.lock:
            return dict(self.measured), self.following, self.connected


def _control_loop(robot, state: FollowerState, watchdog: float, control_hz: float,
                  feedback_hz: float, fake: bool, stop: threading.Event) -> None:
    period = 1.0 / control_hz
    feedback_period = 1.0 / feedback_hz
    last_applied = -1
    last_feedback = 0.0
    measured: dict[str, float] = {}
    control_count = 0
    count_t0 = time.monotonic()
    while not stop.is_set():
        tick = time.monotonic()
        action, last_rx, action_number = state.snapshot_action()
        fresh = action is not None and (tick - last_rx) < watchdog
        if fake:
            if fresh:
                measured = {k.removesuffix(".pos"): v for k, v in action.items()}
            state.publish(measured, fresh, False)
            stop.wait(max(0.0, period - (time.monotonic() - tick)))
            continue
        try:
            # Apply each newest network sample once. Dropping superseded samples is
            # intentional: it prevents a delayed queue from replaying stale motion.
            if fresh and action_number != last_applied:
                robot.send_action(action)
                last_applied = action_number
                control_count += 1
            if tick - last_feedback >= feedback_period:
                obs = robot.get_observation()
                measured = {m: float(obs.get(f"{m}.pos", 0.0)) for m in JOINTS}
                last_feedback = tick
            state.publish(measured, fresh, True)
        except Exception as e:   # noqa: BLE001 - keep streaming even if a poll hiccups
            print(f"[follower] control error: {e}")
            state.publish(measured, fresh, True)
        now = time.monotonic()
        if now - count_t0 >= 5.0:
            print(f"[follower] applied action rate={control_count / (now - count_t0):.1f} Hz")
            control_count = 0
            count_t0 = now
        stop.wait(max(0.0, period - (time.monotonic() - tick)))


def _camera_loop(cam: Camera, link: RelayLink, state: FollowerState, arm: str,
                 quality: int, fps: float, stop: threading.Event) -> None:
    period = 1.0 / max(fps, 1.0)
    last_state = 0.0
    while not stop.is_set():
        t0 = time.time()
        frame = cam.read()
        if frame is None:
            time.sleep(0.05)
            continue
        measured, following, connected = state.snapshot_status()
        if measured:
            vals = np.array([measured.get(m, 0.0) for m in JOINTS])
            meas_str = "[" + " ".join(f"{v:.0f}" for v in vals) + "] deg"
        else:
            meas_str = "arm sim" if not connected else "…"
        hud = [f"{arm.upper()} follower   {'live' if connected else 'sim'}",
               f"joints {meas_str}"]
        P.draw_hud(frame, hud, following, active_label="FOLLOWING", idle_label="HOLD")
        jpeg = P.encode_jpeg(frame, quality)
        if jpeg:
            link.send_frame(jpeg)
        now = time.time()
        if now - last_state > 0.25:
            last_state = now
            ee = [measured.get(m, 0.0) for m in JOINTS]
            link.send_text(P.state_msg(connected, following, ee, ee, 0.0, now))
        dt = time.time() - t0
        if dt < period:
            time.sleep(period - dt)


def main() -> None:
    ap = argparse.ArgumentParser(description="Follower node for remote leader-follower teleop")
    ap.add_argument("--relay", required=True, help="relay host:port (or ws://host:port)")
    ap.add_argument("--room", default=P.DEFAULT_ROOM)
    ap.add_argument("--arm", choices=("rs", "dm"), default="rs")
    ap.add_argument("--port", default="can0", help="CAN port (can0 socketcan, or /dev/ttyACMx damiao)")
    ap.add_argument("--can-adapter", default="socketcan", choices=("socketcan", "damiao", "robstride"))
    ap.add_argument("--id", default="follower1", help="lerobot robot id (calibration file)")
    ap.add_argument("--no-calibrate", action="store_true", help="skip the connect-time calibration prompt")
    ap.add_argument("--max-relative-target", type=float, default=0.0,
                    help="optional plugin safety cap; 0 matches Seeed's verified default")
    ap.add_argument("--camera", default="0", help="cv2 camera index, or 'test'")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--fps", type=float, default=20.0)
    ap.add_argument("--jpeg-quality", type=int, default=P.JPEG_QUALITY)
    ap.add_argument("--watchdog", type=float, default=0.4)
    ap.add_argument("--control-hz", type=float, default=60.0,
                    help="maximum action application rate (official LeRobot default: 60)")
    ap.add_argument("--feedback-hz", type=float, default=10.0,
                    help="CAN feedback/HUD polling rate")
    ap.add_argument("--relay-timeout", type=float, default=10.0)
    ap.add_argument("--fake", action="store_true", help="don't connect hardware (link/video test)")
    args = ap.parse_args()
    if not 1.0 <= args.control_hz <= 60.0:
        ap.error("--control-hz must be between 1 and 60")
    if not 1.0 <= args.feedback_hz <= args.control_hz:
        ap.error("--feedback-hz must be between 1 and --control-hz")
    if args.watchdog <= 0:
        ap.error("--watchdog must be positive")

    state = FollowerState()

    def _on_text(message: str) -> None:
        m = P.parse_msg(message)
        if not m or m.get("type") != P.TYPE_ACTION:
            return
        a = m.get("a") or {}
        # normalise to '<motor>.pos' keys the plugin expects
        try:
            action = {(k if k.endswith(".pos") else f"{k}.pos"): float(v)
                      for k, v in a.items()}
            seq = int(m.get("seq", 0))
            session = str(m.get("session", "legacy"))
        except (TypeError, ValueError):
            return
        if set(action) != ACTION_KEYS or not all(math.isfinite(v) for v in action.values()):
            print("[follower] rejected malformed action")
            return
        state.set_action(action, session, seq)

    url = P.build_ws_url(args.relay, args.room, "arm")   # follower is the 'arm' role on the relay
    link = RelayLink(url, on_text=_on_text, name="follower")
    stop = threading.Event()
    cam = Camera(args.camera, args.width, args.height)

    robot = None
    if not args.fake:
        preflight(args.arm, args.port, args.can_adapter)
        max_rel = args.max_relative_target if args.max_relative_target and args.max_relative_target > 0 else None
        robot = build_follower(args.arm, args.port, args.can_adapter, args.id, max_rel)
        if args.no_calibrate and not robot.is_calibrated:
            raise RuntimeError(
                f"no calibration for robot id {args.id!r}; run lerobot-calibrate first"
            )
        print(f"[follower] connecting {args.arm} on {args.port} ({args.can_adapter}) …")
        robot.connect(calibrate=not args.no_calibrate)
        print("[follower] connected. Waiting for the leader…")

    link.start()
    if not link.wait_connected(args.relay_timeout):
        if robot is not None:
            robot.disconnect()
        cam.release()
        link.stop()
        raise RuntimeError(f"relay did not connect within {args.relay_timeout:.1f}s: {url}")
    threading.Thread(target=_control_loop, name="ctrl",
                     args=(robot, state, args.watchdog, args.control_hz,
                           args.feedback_hz, args.fake, stop), daemon=True).start()
    threading.Thread(target=_camera_loop, name="cam",
                     args=(cam, link, state, args.arm, args.jpeg_quality, args.fps, stop),
                     daemon=True).start()

    def _shutdown(*_a):
        stop.set()
    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    host = args.relay.split("://")[-1]
    print(f"[follower] room={args.room!r}; watch at: http://{host}/view?room={args.room}")
    try:
        while not stop.is_set():
            time.sleep(0.2)
    finally:
        stop.set()
        time.sleep(0.3)
        link.stop()
        cam.release()
        if robot is not None:
            try:
                robot.disconnect()
            except Exception:   # noqa: BLE001
                pass


if __name__ == "__main__":
    main()
