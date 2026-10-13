#!/usr/bin/env python3
"""Arm-side receiver for the depth hand-follow teleop.

Runs in the ARM venv (fresh repo .venv, Py3.10 — has the CAN stack). Holds the
B601-RS via the tested SmoothedEndPose controller, homes it, then listens on a
localhost UDP socket for XYZ end-effector targets (metres) and drives the arm to
them with IK. This is the counterpart to depth_hand_follow_demo.py --arm, which
runs in the camera venv and streams targets here.

SAFETY (belt-and-braces; the camera client already clamps, we clamp again):
  * every received target is re-clamped to the shared workspace box
    (arm_teleop_constraints.clamp_workspace) — identical limits to the gesture demo
  * SmoothedEndPose slew-limits + low-pass filters the joint command (no lurching)
  * the arm homes and CONFIRMS home before it accepts any target
  * watchdog: if no target arrives for --watchdog seconds (client died / paused),
    the arm holds its last commanded pose instead of drifting
  * the gripper is held at the 5%-open safe position (not controlled by depth mode)
  * Ctrl+C safe-homes and disables

Usage:
    # sanity-check the link WITHOUT moving the arm (prints received targets):
    .venv/bin/python example/arm_target_server.py --dry

    # real run:
    .venv/bin/python example/arm_target_server.py
"""
from __future__ import annotations

import argparse
import signal
import socket
import struct
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import arm_teleop_constraints as C

TARGET_STRUCT = "<3f"        # 3 little-endian float32: x, y, z (metres)
TARGET_NBYTES = struct.calcsize(TARGET_STRUCT)


def _recv_latest_target(sock: socket.socket) -> np.ndarray | None:
    """Drain the UDP buffer and return the most recent valid target (latest-wins),
    or None if nothing was waiting."""
    latest = None
    while True:
        try:
            data, _ = sock.recvfrom(256)
        except BlockingIOError:
            break
        except OSError:
            break
        if len(data) == TARGET_NBYTES:
            latest = np.array(struct.unpack(TARGET_STRUCT, data), dtype=np.float64)
    return latest


def _run_dry(sock: socket.socket) -> None:
    print("[arm-server] DRY: printing received targets, arm NOT connected. Ctrl+C to stop.")
    running = [True]
    signal.signal(signal.SIGINT, lambda *a: running.__setitem__(0, False))
    n = 0
    while running[0]:
        tgt = _recv_latest_target(sock)
        if tgt is not None:
            clamped = C.clamp_workspace(tgt)
            hit = ~np.isclose(clamped, tgt, atol=1e-6)
            flags = "".join(a for a, c in zip("XYZ", hit) if c) or "-"
            if n % 15 == 0:
                print(f"[dry] rx target {np.round(tgt,3)} -> clamped {np.round(clamped,3)} "
                      f"clamp={flags}")
            n += 1
        else:
            time.sleep(0.005)


def main() -> None:
    ap = argparse.ArgumentParser(description="Arm-side UDP target receiver for depth hand-follow")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8721)
    ap.add_argument("--hw-yaml", default=None, help="arm hardware yaml (default: config/rebotarm.yaml)")
    ap.add_argument("--watchdog", type=float, default=0.5,
                    help="hold pose if no target arrives within this many seconds")
    ap.add_argument("--dry", action="store_true",
                    help="do NOT connect the arm; just print received targets")
    args = ap.parse_args()

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind((args.host, args.port))
    sock.setblocking(False)
    print(f"[arm-server] listening on udp://{args.host}:{args.port}")

    if args.dry:
        _run_dry(sock)
        return

    # ── connect + home the arm (import here so --dry needs no arm stack) ──
    from reBotArm_control_py.actuator import RebotArm
    from reBotArm_control_py.kinematics import joint_to_pose
    import bimanual_hand_control_6dof_demo as core   # tested SmoothedEndPose + gripper consts

    arm = RebotArm(args.hw_yaml)
    endpos = core.SmoothedEndPose(arm, arm_control_mode="posvel")
    endpos.start()

    # hold the gripper at the 5%-open safe position (depth mode doesn't drive it)
    grip_hold = core.GRIPPER_CLOSE_POS + core.GRIPPER_MIN_OPEN_FRACTION * (
        core.GRIPPER_OPEN_POS - core.GRIPPER_CLOSE_POS)
    endpos.set_gripper_target(grip_hold)

    home = C.ARM_HOME
    print(f"[arm-server] homing to {np.round(home,3)} ...")
    endpos.move_to_traj(x=float(home[0]), y=float(home[1]), z=float(home[2]),
                        roll=0.0, pitch=0.0, yaw=0.0, duration=2.5)

    def _shutdown(*_a):
        print("\n[arm-server] stopping: safe-home + disable")
        try:
            endpos.end()
        except Exception:
            pass
        sys.exit(0)

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    # confirm HOME before accepting targets
    n_arm = endpos._n
    deadline = time.monotonic() + 25.0
    confirmed = False
    while time.monotonic() < deadline:
        q, vel, _ = arm.get_state()
        pos, _ = joint_to_pose(q[:n_arm])
        if (float(np.linalg.norm(np.asarray(pos) - home)) < 0.02
                and float(np.max(np.abs(vel[:n_arm]))) < 0.08
                and not endpos._moving):
            confirmed = True
            break
        time.sleep(0.1)
    if not confirmed:
        print("[arm-server] WARNING: home not confirmed; refusing targets for safety")
        _shutdown()
    print("[arm-server] HOME confirmed. Streaming targets — set neutral on the camera "
          "client, then move your hand.")

    last_rx = time.monotonic()
    last_status = 0.0
    holding = False
    cur_target = home.copy()
    while True:
        tgt = _recv_latest_target(sock)
        now = time.monotonic()
        if tgt is not None:
            cur_target = C.clamp_workspace(tgt)
            endpos.move_to_ik(x=float(cur_target[0]), y=float(cur_target[1]), z=float(cur_target[2]),
                              roll=0.0, pitch=0.0, yaw=0.0)
            last_rx = now
            holding = False
        else:
            if now - last_rx > args.watchdog and not holding:
                holding = True
                print("[arm-server] no targets — holding pose (watchdog)")
            time.sleep(0.005)

        if now - last_status > 0.5:
            last_status = now
            q, _, _ = arm.get_state()
            ee, _ = joint_to_pose(q[:n_arm])
            err = float(np.linalg.norm(np.asarray(ee) - cur_target))
            print(f"[arm-server] target={np.round(cur_target,3)} ee={np.round(ee,3)} "
                  f"err={err*1000:.0f}mm {'(holding)' if holding else ''}")


if __name__ == "__main__":
    main()
