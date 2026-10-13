#!/usr/bin/env python3
"""Move the arm to a home pose with a smooth min-jerk ramp, then hold.

Press Enter to release (motors stay enabled), Ctrl+C to stop + disable.

To redefine home: run example/2_zero_and_read.py, move the arm to the pose
you want, read the angles, and update HOME_DEG below.

Usage:
    python example/go_home.py                     # config from config/rebotarm.yaml
    python example/go_home.py rebotarm_rs.yaml    # force a hardware config
"""
import signal
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from reBotArm_control_py.actuator import RebotArm

# ── Home pose (degrees), arm joints only ─────────────────────────────────────
HOME_DEG = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
DURATION = 5.0          # seconds to reach home
RAMP_HZ  = 50.0
VLIM     = 2.0          # rad/s speed limit sent with each POS_VEL command
# ─────────────────────────────────────────────────────────────────────────────


def _min_jerk(t: float) -> float:
    t = max(0.0, min(1.0, t))
    return 10 * t**3 - 15 * t**4 + 6 * t**5


def main() -> None:
    hw_yaml = sys.argv[1] if len(sys.argv) > 1 else None
    arm = RebotArm(hw_yaml)
    print(f"Using config: {arm.hardware_yaml}")

    def cleanup(sig=None, frame=None):
        print("\nStopping...")
        try:
            arm.disconnect()   # stops loop, disables, closes
        except Exception:
            pass
        sys.exit(0)

    signal.signal(signal.SIGINT, cleanup)
    signal.signal(signal.SIGTERM, cleanup)

    arm.connect()

    n = arm.arm.num_joints
    home = np.radians(HOME_DEG[:n])

    # RobStride: set run mode BEFORE enabling — switching modes while enabled
    # is silently ignored by some firmwares.
    if not arm.arm.mode_pos_vel(vlim=np.full(n, VLIM)):
        print("WARNING: not all joints switched to POS_VEL mode")
    arm.enable_all()
    time.sleep(0.5)

    q_start = arm.arm.get_positions()
    print("Current position (deg):", np.degrees(q_start).round(2).tolist())

    print(f"Moving to home {HOME_DEG[:n]} over {DURATION}s ... (Ctrl+C to abort)")
    dt = 1.0 / RAMP_HZ
    steps = max(2, int(DURATION * RAMP_HZ))
    for i in range(steps + 1):
        s = _min_jerk(i / steps)
        arm.arm.send_pos_vel(q_start + s * (home - q_start))
        time.sleep(dt)

    time.sleep(0.5)
    q_final = arm.arm.get_positions()
    print("Reached (deg):", np.degrees(q_final).round(2).tolist())
    err = np.degrees(q_final - home)
    print("Error   (deg):", err.round(2).tolist())

    print("\nArm is holding home position.")
    try:
        input("Press Enter to exit (motors STAY enabled), Ctrl+C to disable...")
    except (KeyboardInterrupt, EOFError):
        cleanup()

    # Exit leaving motors enabled: close bus without disabling.
    for ctrl in arm._ctrl_map.values():
        ctrl.close()


if __name__ == "__main__":
    main()
