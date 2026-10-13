#!/usr/bin/env python3
"""Leader node for remote LEADER-FOLLOWER teleop — runs where the human driver is.

Reads the reBot Arm 102 leader's joint positions through the lerobot
teleoperator plugin (`rebot_arm_102_leader`) and streams them as an action dict
to the relay, which forwards them to whichever follower node is in the same room.
So you can drive a B601 on the other side of the world; to *see* it, open the
relay's /view URL (the follower node streams the arm's camera there).

Pairs with follower_node.py. Bidirectional: either machine can be leader or
follower.

Run in the verified conda environment from the Seeed B601-RS guide:
    conda activate rebot_rs
    python leader_node.py --relay YOUR_VM:8765 --room b601 \
        --port /dev/ttyUSB0 --id rebot_arm_102_leader
No-hardware link test (sweeps a gentle action):
    python leader_node.py --relay YOUR_VM:8765 --room b601 --fake
"""
from __future__ import annotations

import argparse
import importlib.metadata
import math
import signal
import sys
import time
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent))          # protocol, relay_link (this dir)

import protocol as P                            # noqa: E402
from relay_link import RelayLink                # noqa: E402

# gentle amplitudes (deg) for --fake, per B601 joint (safe, near zero)
FAKE_JOINTS = {
    "shoulder_pan": 20.0, "shoulder_lift": 15.0, "elbow_flex": -20.0,
    "wrist_flex": 15.0, "wrist_yaw": 20.0, "wrist_roll": 20.0, "gripper": 25.0,
}
EXPECTED_JOINTS = {f"{name}.pos" for name in FAKE_JOINTS}


def _version_tuple(value: str) -> tuple[int, ...]:
    parts = []
    for piece in value.split("."):
        digits = "".join(ch for ch in piece if ch.isdigit())
        if not digits:
            break
        parts.append(int(digits))
    return tuple(parts)


def preflight(port: str) -> None:
    required = {
        "lerobot": "0.4.4",
        "lerobot-teleoperator-rebot-arm-102": "1.0.0",
        "motorbridge-smart-servo": "0.0.4",
        "websocket-client": "1.0.0",
    }
    for package, minimum in required.items():
        try:
            actual = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError as exc:
            raise RuntimeError(f"missing package {package!r}; activate conda env rebot_rs") from exc
        if _version_tuple(actual) < _version_tuple(minimum):
            raise RuntimeError(f"{package} {actual} is too old; need >= {minimum}")
    device = Path(port)
    if not device.exists():
        raise RuntimeError(f"leader serial port does not exist: {port}")
    print("[leader] preflight OK: official reBot 102 plugin and smart-servo stack")


def build_leader(port: str, baudrate: int, leader_id: str):
    from lerobot_teleoperator_rebot_arm_102.rebot_arm_102_leader import RebotArm102Leader
    from lerobot_teleoperator_rebot_arm_102.config_rebot_arm_102_leader import (
        RebotArm102LeaderConfig)
    return RebotArm102Leader(RebotArm102LeaderConfig(port=port, baudrate=baudrate, id=leader_id))


def run_leader(args, link: RelayLink, stop) -> None:
    preflight(args.port)
    leader = build_leader(args.port, args.baudrate, args.id)
    print(f"[leader] connecting on {args.port} …")
    leader.connect(calibrate=not args.no_calibrate)
    print("[leader] connected. Streaming actions — move the leader arm.")
    view = f"http://{args.relay.split('://')[-1]}/view?room={args.room}"
    print(f"[leader] watch the follower at: {view}")
    period = 1.0 / args.fps
    last_log = 0.0
    sent = 0
    seq = 0
    session = P.new_session_id()
    hz_t0 = time.monotonic()
    try:
        while not stop[0]:
            t0 = time.monotonic()
            action = leader.get_action()          # {'<motor>.pos': deg}
            if set(action) != EXPECTED_JOINTS or not all(
                    math.isfinite(float(value)) for value in action.values()):
                raise RuntimeError(f"invalid leader action keys/values: {action}")
            if link.send_text(P.action_msg(action, seq=seq, session=session)):
                sent += 1
                seq += 1
            now = time.monotonic()
            if now - last_log > 1.0:
                elapsed = max(now - hz_t0, 1e-6)
                actual_hz = sent / elapsed
                last_log = now
                sent = 0
                hz_t0 = now
                g = action.get("gripper.pos", 0.0)
                status = "relay live" if link.connected.is_set() else "relay DOWN"
                print(f"[leader] {status}; tx={actual_hz:.1f} Hz; gripper={g:.1f}°   ", end="\r")
            dt = time.monotonic() - t0
            if dt < period:
                time.sleep(period - dt)
    finally:
        try:
            leader.disconnect()
        except Exception:   # noqa: BLE001
            pass


def run_fake(args, link: RelayLink, stop) -> None:
    print("[leader] FAKE: sweeping a gentle action (no leader arm).")
    period = 1.0 / max(args.fps, 1.0)
    t0 = time.time()
    seq = 0
    session = P.new_session_id()
    while not stop[0]:
        t = time.time() - t0
        action = {f"{m}.pos": amp * np.sin(0.5 * t) for m, amp in FAKE_JOINTS.items()}
        if link.send_text(P.action_msg(action, seq=seq, session=session)):
            seq += 1
        time.sleep(period)


def main() -> None:
    ap = argparse.ArgumentParser(description="Leader node for remote leader-follower teleop")
    ap.add_argument("--relay", required=True, help="relay host:port (or ws://host:port)")
    ap.add_argument("--room", default=P.DEFAULT_ROOM)
    ap.add_argument("--port", default="/dev/ttyUSB0", help="leader UART port")
    ap.add_argument("--baudrate", type=int, default=1_000_000)
    ap.add_argument("--id", default="rebot_arm_102_leader",
                    help="lerobot teleop id (must match the calibration file)")
    ap.add_argument("--no-calibrate", action="store_true", help="skip connect-time calibration prompt")
    ap.add_argument("--fps", type=float, default=60.0,
                    help="maximum leader sampling rate; official LeRobot default is 60")
    ap.add_argument("--relay-timeout", type=float, default=10.0)
    ap.add_argument("--fake", action="store_true", help="no leader arm; sweep an action (link test)")
    args = ap.parse_args()
    if not 1.0 <= args.fps <= 60.0:
        ap.error("--fps must be between 1 and 60")

    url = P.build_ws_url(args.relay, args.room, "operator")   # leader is the 'operator' role
    link = RelayLink(url, name="leader")
    link.start()
    if not link.wait_connected(args.relay_timeout):
        link.stop()
        raise RuntimeError(f"relay did not connect within {args.relay_timeout:.1f}s: {url}")

    stop = [False]
    signal.signal(signal.SIGINT, lambda *_a: stop.__setitem__(0, True))
    signal.signal(signal.SIGTERM, lambda *_a: stop.__setitem__(0, True))

    try:
        if args.fake:
            run_fake(args, link, stop)
        else:
            run_leader(args, link, stop)
    finally:
        link.stop()


if __name__ == "__main__":
    main()
