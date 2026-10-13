#!/usr/bin/env python3
"""Arm (follower) node for remote teleop — runs wherever the physical arm is.

Connects the B601 arm (RS *or* DM, chosen by --hw-yaml / --arm), opens at least
one camera, and links to the self-hosted relay over a single WebSocket:

  * RECEIVES operator target messages (XYZ + grip + engage) and drives the arm
    with the SAME tested, safety-clamped SmoothedEndPose pipeline used by the
    localhost UDP path (arm_target_server.py) — see the SAFETY notes below.
  * SENDS its camera as JPEG frames (with a status HUD overlaid) so browser
    viewers anywhere can watch the teleop, plus periodic arm-state JSON.

Because control is just a message stream, either machine can be the arm node or
the operator node — you can drive their arm or they can drive yours.

SAFETY (identical policy to arm_target_server.py):
  * every received target is re-clamped to the shared workspace box
  * the arm follows ONLY while the operator says engaged=True
  * homes + CONFIRMS home before accepting any target
  * watchdog: no target within --watchdog s -> hold last pose
  * gripper hard-limited to the 5%..70%-open safe band; Ctrl+C safe-homes+disables

Run (real arm, RobStride):
    pip install websocket-client                 # once, into this venv
    python arm_node.py --relay YOUR_VM:8765 --room b601 --arm rs --camera 0
Run (Damiao):
    python arm_node.py --relay YOUR_VM:8765 --room b601 --arm dm --camera 0
Test the link with no hardware (synthetic camera, no arm):
    python arm_node.py --relay YOUR_VM:8765 --room b601 --fake-arm --camera test
"""
from __future__ import annotations

import argparse
import signal
import sys
import threading
import time
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent))          # protocol.py (this dir)
sys.path.insert(0, str(_HERE.parents[1]))      # example/ : arm_teleop_constraints, core demo
sys.path.insert(0, str(_HERE.parents[2]))      # repo root : reBotArm_control_py package

import protocol as P                            # noqa: E402
import arm_teleop_constraints as C              # noqa: E402
from relay_link import RelayLink                # noqa: E402  (needs websocket-client)
from camera import Camera                       # noqa: E402


# ── config yaml for RS vs DM ──────────────────────────────────────────────────
def resolve_hw_yaml(arm: str | None, hw_yaml: str | None) -> str | None:
    if hw_yaml:
        return hw_yaml
    if arm == "rs":
        return "config/rebotarm_rs.yaml"
    if arm == "dm":
        return "config/rebotarm_dm.yaml"
    return None   # -> RebotArm default (config/rebotarm.yaml)


class SharedState:
    """Thread-safe hand-off between the WS reader, the camera sender and the arm
    control loop."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.target: np.ndarray | None = None       # last commanded XYZ (clamped)
        self.grip_frac: float = C.GRIPPER_MIN_OPEN_FRACTION
        self.op_engaged: bool = False               # operator's engage flag
        self.last_rx: float = 0.0
        # published by the control loop for HUD + state msg:
        self.ee = np.array(C.ARM_HOME, dtype=np.float64)
        self.cmd_target = np.array(C.ARM_HOME, dtype=np.float64)
        self.err_mm = 0.0
        self.arm_connected = False
        self.holding = False

    def set_operator_target(self, x, y, z, grip, engaged, t) -> None:
        tgt = C.clamp_workspace(np.array([x, y, z], dtype=np.float64))
        with self.lock:
            self.target = tgt
            self.grip_frac = float(np.clip(grip, C.GRIPPER_MIN_OPEN_FRACTION,
                                           C.GRIPPER_MAX_OPEN_FRACTION))
            self.op_engaged = bool(engaged)
            self.last_rx = time.monotonic()

    def snapshot_cmd(self):
        with self.lock:
            return (self.target.copy() if self.target is not None else None,
                    self.grip_frac, self.op_engaged, self.last_rx)

    def publish_status(self, ee, cmd_target, err_mm, connected, holding, engaged) -> None:
        with self.lock:
            self.ee = np.asarray(ee, dtype=np.float64).copy()
            self.cmd_target = np.asarray(cmd_target, dtype=np.float64).copy()
            self.err_mm = float(err_mm)
            self.arm_connected = bool(connected)
            self.holding = bool(holding)
            self.op_engaged = bool(engaged)

    def snapshot_status(self):
        with self.lock:
            return (self.ee.copy(), self.cmd_target.copy(), self.err_mm,
                    self.arm_connected, self.holding, self.op_engaged)


def _camera_loop(cam: Camera, link: RelayLink, state: SharedState,
                 quality: int, fps: float, stop: threading.Event) -> None:
    period = 1.0 / max(fps, 1.0)
    last_state_sent = 0.0
    while not stop.is_set():
        t0 = time.time()
        frame = cam.read()
        if frame is None:
            time.sleep(0.05)
            continue
        ee, cmd_target, err_mm, connected, holding, engaged = state.snapshot_status()
        hud = [
            f"target {np.round(cmd_target, 3)}   ee {np.round(ee, 3)}",
            f"err {err_mm:.0f}mm   arm {'OK' if connected else 'sim' } "
            f"{'HOLD(watchdog)' if holding else ''}",
        ]
        P.draw_hud(frame, hud, engaged)
        jpeg = P.encode_jpeg(frame, quality)
        if jpeg:
            link.send_frame(jpeg)
        now = time.time()
        if now - last_state_sent > 0.25:
            last_state_sent = now
            link.send_text(P.state_msg(connected, engaged, ee, cmd_target, err_mm, now))
        dt = time.time() - t0
        if dt < period:
            time.sleep(period - dt)


# ── Arm control loop ──────────────────────────────────────────────────────────
def _run_fake_arm(state: SharedState, stop: threading.Event) -> None:
    """No hardware: echo targets as if the arm tracked perfectly (for testing the
    link + video end-to-end)."""
    print("[arm-node] FAKE arm: not connecting hardware; echoing targets.")
    ee = np.array(C.ARM_HOME, dtype=np.float64)
    while not stop.is_set():
        tgt, grip, engaged, last_rx = state.snapshot_cmd()
        holding = (time.monotonic() - last_rx) > 0.5
        if tgt is not None and engaged and not holding:
            ee += 0.4 * (tgt - ee)      # simple 1st-order tracking
            cmd = tgt
        else:
            cmd = ee.copy()
        err = float(np.linalg.norm(ee - cmd)) * 1000.0
        state.publish_status(ee, cmd, err, False, holding, engaged)
        time.sleep(0.02)


def _run_real_arm(state: SharedState, hw_yaml: str | None,
                  watchdog: float, stop: threading.Event) -> None:
    from reBotArm_control_py.actuator import RebotArm         # noqa: E402
    from reBotArm_control_py.kinematics import joint_to_pose  # noqa: E402
    import bimanual_hand_control_6dof_demo as core            # noqa: E402

    arm = RebotArm(hw_yaml)
    endpos = core.SmoothedEndPose(arm, arm_control_mode="posvel")
    endpos.start()

    grip_hold = core.GRIPPER_CLOSE_POS + C.GRIPPER_MIN_OPEN_FRACTION * (
        core.GRIPPER_OPEN_POS - core.GRIPPER_CLOSE_POS)
    endpos.set_gripper_target(grip_hold)

    home = C.ARM_HOME
    print(f"[arm-node] homing to {np.round(home, 3)} …")
    endpos.move_to_traj(x=float(home[0]), y=float(home[1]), z=float(home[2]),
                        roll=0.0, pitch=0.0, yaw=0.0, duration=2.5)

    n_arm = endpos._n
    deadline = time.monotonic() + 25.0
    confirmed = False
    while time.monotonic() < deadline and not stop.is_set():
        q, vel, _ = arm.get_state()
        pos, _ = joint_to_pose(q[:n_arm])
        if (float(np.linalg.norm(np.asarray(pos) - home)) < 0.02
                and float(np.max(np.abs(vel[:n_arm]))) < 0.08
                and not endpos._moving):
            confirmed = True
            break
        time.sleep(0.1)
    if not confirmed:
        print("[arm-node] WARNING: home not confirmed; refusing targets for safety")
        stop.set()
        try:
            endpos.end()
        except Exception:   # noqa: BLE001
            pass
        return
    print("[arm-node] HOME confirmed. Waiting for the operator to engage…")

    cur_target = home.copy()
    while not stop.is_set():
        tgt, grip_frac, engaged, last_rx = state.snapshot_cmd()
        now = time.monotonic()
        holding = (now - last_rx) > watchdog
        # follow ONLY when the operator is engaged and targets are fresh
        if tgt is not None and engaged and not holding:
            cur_target = tgt
            endpos.move_to_ik(x=float(cur_target[0]), y=float(cur_target[1]),
                              z=float(cur_target[2]), roll=0.0, pitch=0.0, yaw=0.0)
            grip_pos = core.GRIPPER_CLOSE_POS + grip_frac * (
                core.GRIPPER_OPEN_POS - core.GRIPPER_CLOSE_POS)
            endpos.set_gripper_target(grip_pos)
        q, _, _ = arm.get_state()
        ee, _ = joint_to_pose(q[:n_arm])
        err = float(np.linalg.norm(np.asarray(ee) - cur_target)) * 1000.0
        state.publish_status(ee, cur_target, err, True, holding, engaged)
        time.sleep(0.01)

    print("[arm-node] stopping: safe-home + disable")
    try:
        endpos.end()
    except Exception:   # noqa: BLE001
        pass


def main() -> None:
    ap = argparse.ArgumentParser(description="Arm/follower node for remote teleop")
    ap.add_argument("--relay", required=True, help="relay host:port (or ws://host:port)")
    ap.add_argument("--room", default=P.DEFAULT_ROOM)
    ap.add_argument("--arm", choices=("rs", "dm"), default=None,
                    help="sugar for --hw-yaml (rs->rebotarm_rs.yaml, dm->rebotarm_dm.yaml)")
    ap.add_argument("--hw-yaml", default=None, help="explicit arm hardware yaml")
    ap.add_argument("--camera", default="0", help="cv2 camera index, or 'test' for a synthetic pattern")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--fps", type=float, default=20.0)
    ap.add_argument("--jpeg-quality", type=int, default=P.JPEG_QUALITY)
    ap.add_argument("--watchdog", type=float, default=0.5)
    ap.add_argument("--fake-arm", action="store_true", help="don't connect hardware (link/video test)")
    args = ap.parse_args()

    state = SharedState()

    def _on_text(message: str) -> None:
        m = P.parse_msg(message)
        if not m or m.get("type") != P.TYPE_TARGET:
            return
        state.set_operator_target(m.get("x", 0.0), m.get("y", 0.0), m.get("z", 0.0),
                                  m.get("grip", C.GRIPPER_MIN_OPEN_FRACTION),
                                  m.get("engaged", False), m.get("t", 0.0))

    url = P.build_ws_url(args.relay, args.room, "arm")
    link = RelayLink(url, on_text=_on_text, name="arm-node")
    stop = threading.Event()

    cam = Camera(args.camera, args.width, args.height)

    link.start()
    cam_thread = threading.Thread(
        target=_camera_loop, name="cam",
        args=(cam, link, state, args.jpeg_quality, args.fps, stop), daemon=True)
    cam_thread.start()

    def _shutdown(*_a):
        stop.set()
    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    print(f"[arm-node] room={args.room!r} relay={url}")
    print(f"[arm-node] watch at: http://{args.relay.split('://')[-1]}/view?room={args.room}")
    try:
        hw_yaml = resolve_hw_yaml(args.arm, args.hw_yaml)
        if args.fake_arm:
            _run_fake_arm(state, stop)
        else:
            _run_real_arm(state, hw_yaml, args.watchdog, stop)
    finally:
        stop.set()
        link.stop()
        cam.release()
        time.sleep(0.2)


if __name__ == "__main__":
    main()
