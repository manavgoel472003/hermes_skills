#!/usr/bin/env python3
"""Record a joint-space trajectory (arm + gripper) at a fixed rate.

The arm is put in FREE-DRIVE (MIT mode, zero stiffness): motors stay enabled
so position feedback keeps flowing (RobStride motors go silent when disabled),
but exert no torque — move the arm and gripper freely by hand.

Recording runs until you press Enter, then saves to a .npz file with keys:
    positions  (N, num_joints)  radians
    velocities (N, num_joints)  rad/s
    timestamps (N,)             seconds from start
    hz         ()               sample rate

Usage:
    python example/record_traj.py
    python example/record_traj.py --out recordings/my_traj.npz --hz 50
    python example/record_traj.py --hw-yaml rebotarm_rs.yaml
"""
import argparse
import signal
import sys
import threading
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from reBotArm_control_py.actuator import RebotArm


def parse_args():
    p = argparse.ArgumentParser(description="Record arm + gripper trajectory to .npz")
    p.add_argument("--out", default="recordings/trajectory.npz", help="Output file path")
    p.add_argument("--hz", type=float, default=50.0, help="Sample rate in Hz (default 50)")
    p.add_argument("--hw-yaml", default=None,
                   help="hardware config yaml (default: from config/rebotarm.yaml)")
    return p.parse_args()


def main():
    args = parse_args()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    arm = RebotArm(args.hw_yaml)
    print(f"Using config: {arm.hardware_yaml}")

    stop_event = threading.Event()
    signal.signal(signal.SIGINT, lambda *a: stop_event.set())
    signal.signal(signal.SIGTERM, lambda *a: stop_event.set())

    arm.connect()
    print("--- Connected ---")

    n_arm = arm.arm.num_joints
    n_grip = arm.gripper.num_joints          # 0 when no gripper configured

    # Free-drive: MIT mode with zero gains, motors enabled but torque-free.
    arm.arm.mode_mit(kp=np.zeros(n_arm), kd=np.zeros(n_arm))
    if n_grip:
        arm.gripper.mode_mit(kp=np.zeros(n_grip), kd=np.zeros(n_grip))
    arm.enable_all()
    time.sleep(0.5)

    zeros_arm = np.zeros(n_arm)
    zeros_grip = np.zeros(n_grip)

    def freedrive(r: RebotArm, dt: float) -> None:
        r.arm.send_mit(pos=zeros_arm, kp=zeros_arm, kd=zeros_arm)
        if n_grip:
            r.gripper.send_mit(pos=zeros_grip, kp=zeros_grip, kd=zeros_grip)

    arm.start_control_loop(freedrive, rate=100.0)

    start = arm.get_positions()
    print(f"Start pose (deg): {np.degrees(start).round(2).tolist()}")
    print(f"\nRecording at {args.hz:.0f} Hz — motors are torque-free, move the arm by hand.")
    print("Press Enter to stop recording...\n")

    frames, vels, stamps = [], [], []
    dt = 1.0 / args.hz
    t0 = time.perf_counter()

    def sample_loop():
        while not stop_event.is_set():
            t_iter = time.perf_counter()
            pos, vel, _ = arm.get_state()
            stamps.append(time.perf_counter() - t0)
            frames.append(pos.copy())
            vels.append(vel.copy())
            remaining = dt - (time.perf_counter() - t_iter)
            if remaining > 0:
                time.sleep(remaining)

    sampler = threading.Thread(target=sample_loop, daemon=True)
    sampler.start()

    try:
        input()
    except (KeyboardInterrupt, EOFError):
        pass
    stop_event.set()
    sampler.join(timeout=2.0)

    arm.disconnect()  # stops loop, disables, closes

    if len(frames) < 2:
        print("Nothing recorded.")
        sys.exit(1)

    positions = np.asarray(frames)
    np.savez(out_path,
             positions=positions,
             velocities=np.asarray(vels),
             timestamps=np.asarray(stamps),
             hz=args.hz)

    rng = np.degrees(positions.max(0) - positions.min(0))
    print(f"\nSaved {len(frames)} frames ({stamps[-1]:.1f}s) -> {out_path}")
    print(f"Joint ranges (deg): {rng.round(1).tolist()}")
    if not np.any(rng > 1.0):
        print("WARNING: trajectory is flat (arm did not move / feedback frozen)")


if __name__ == "__main__":
    main()
