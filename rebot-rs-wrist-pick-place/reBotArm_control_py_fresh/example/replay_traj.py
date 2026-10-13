#!/usr/bin/env python3
"""Replay a trajectory recorded with example/record_traj.py.

Safety:
  - validates the file (finite values, sane per-step motion) before moving
  - ramps smoothly (min-jerk) from the current pose to the first frame
  - Ctrl+C at any point stops and disables the motors

Usage:
    python example/replay_traj.py --file recordings/trajectory.npz
    python example/replay_traj.py --file recordings/my.npz --speed 0.5
    python example/replay_traj.py --file recordings/my.npz --hw-yaml rebotarm_rs.yaml
"""
import argparse
import signal
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from reBotArm_control_py.actuator import RebotArm

MAX_STEP_RAD = 0.15     # max joint change between consecutive frames (at 1x)
RAMP_SPEED   = 0.5      # rad/s used to size the ramp-to-start duration
VLIM_ARM     = 2.5      # rad/s limit sent with every POS_VEL command
VLIM_GRIP    = 4.0


def parse_args():
    p = argparse.ArgumentParser(description="Replay recorded arm + gripper trajectory")
    p.add_argument("--file", default="recordings/trajectory.npz")
    p.add_argument("--speed", type=float, default=1.0,
                   help="playback speed factor (0.5 = half speed)")
    p.add_argument("--smooth", type=int, default=5,
                   help="moving-average window over frames (1 = off)")
    p.add_argument("--hw-yaml", default=None,
                   help="hardware config yaml (default: from config/rebotarm.yaml)")
    return p.parse_args()


def smooth_traj(pos: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return pos
    kernel = np.ones(window) / window
    out = pos.copy()
    for j in range(pos.shape[1]):
        out[:, j] = np.convolve(np.pad(pos[:, j], (window // 2, window - 1 - window // 2),
                                       mode="edge"), kernel, mode="valid")
    return out


def _min_jerk(t: float) -> float:
    t = max(0.0, min(1.0, t))
    return 10 * t**3 - 15 * t**4 + 6 * t**5


def main():
    args = parse_args()
    if args.speed <= 0 or args.speed > 2.0:
        sys.exit("--speed must be in (0, 2.0]")

    data = np.load(args.file)
    positions = np.asarray(data["positions"], dtype=np.float64)
    hz = float(data["hz"]) if "hz" in data else 50.0

    # ── validate ─────────────────────────────────────────────────────────
    if positions.ndim != 2 or len(positions) < 2:
        sys.exit(f"Bad file: positions shape {positions.shape}")
    if not np.all(np.isfinite(positions)):
        sys.exit("Bad file: non-finite values in positions")
    positions = smooth_traj(positions, args.smooth)
    step = np.abs(np.diff(positions, axis=0)).max()
    if step * args.speed > MAX_STEP_RAD:
        sys.exit(f"Trajectory too fast: max step {step:.3f} rad x speed "
                 f"{args.speed} exceeds {MAX_STEP_RAD} rad — lower --speed")

    rng = np.degrees(positions.max(0) - positions.min(0))
    print(f"Loaded {len(positions)} frames ({len(positions)/hz:.1f}s at {hz:.0f} Hz)")
    print(f"Joint ranges (deg): {rng.round(1).tolist()}")
    if not np.any(rng > 1.0):
        print("WARNING: trajectory is flat — the arm will not visibly move.")

    # ── connect ──────────────────────────────────────────────────────────
    arm = RebotArm(args.hw_yaml)
    print(f"Using config: {arm.hardware_yaml}")

    def cleanup(sig=None, frame=None):
        print("\n--- Stopping: disabling motors ---")
        try:
            arm.disconnect()
        except Exception:
            pass
        sys.exit(0)

    signal.signal(signal.SIGINT, cleanup)
    signal.signal(signal.SIGTERM, cleanup)

    arm.connect()
    print("--- Connected ---")

    n_arm = arm.arm.num_joints
    n_grip = arm.gripper.num_joints
    n_total = n_arm + n_grip
    if positions.shape[1] != n_total:
        sys.exit(f"File has {positions.shape[1]} joints, robot has {n_total}")

    # RobStride: set run mode BEFORE enabling — switching modes while enabled
    # is silently ignored by some firmwares (vendor examples use this order).
    arm.disable_all()
    time.sleep(0.3)
    arm.arm.mode_pos_vel(vlim=np.full(n_arm, VLIM_ARM))
    if n_grip:
        arm.gripper.mode_pos_vel(vlim=np.full(n_grip, VLIM_GRIP))
    print("--- POS_VEL mode ---")

    arm.enable_all()
    time.sleep(0.5)

    def send(frame: np.ndarray) -> None:
        arm.arm.send_pos_vel(frame[:n_arm])
        if n_grip:
            arm.gripper.send_pos_vel(frame[n_arm:])

    # ── ramp to first frame ──────────────────────────────────────────────
    # Warm up feedback and require every motor to reply before ANY motion —
    # ramping from a bogus all-zero pose would jerk the arm.
    for _ in range(10):
        arm.get_state()
        time.sleep(0.05)
    missing = [n for n in arm.joint_names
               if arm._motor_map[n].get_state() is None]
    if missing:
        print(f"ERROR: no feedback from {missing} — aborting before motion.")
        cleanup()
    current = arm.get_positions()
    first = positions[0]
    dist = np.abs(first - current).max()
    ramp_t = max(2.0, dist / RAMP_SPEED)
    print(f"Ramping to start over {ramp_t:.1f}s (max joint dist "
          f"{np.degrees(dist):.1f} deg)...")
    steps = int(ramp_t * 50)
    for i in range(steps + 1):
        s = _min_jerk(i / steps)
        send(current + s * (first - current))
        time.sleep(1.0 / 50)

    # ── play ─────────────────────────────────────────────────────────────
    dt = 1.0 / (hz * args.speed)
    print(f"Playing at {args.speed:.2f}x ({1.0/dt:.0f} Hz)... Ctrl+C to stop")
    t0 = time.perf_counter()
    for k, frame in enumerate(positions):
        send(frame)
        target_t = (k + 1) * dt
        sleep = target_t - (time.perf_counter() - t0)
        if sleep > 0:
            time.sleep(sleep)

    print("--- Done, holding final pose for 1s ---")
    time.sleep(1.0)
    cleanup()


if __name__ == "__main__":
    main()
