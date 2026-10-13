#!/usr/bin/env python3
"""Enable all motors and print their state, leaving them enabled on exit.

Run this before any motion script to confirm all motors are ready.
Note: RobStride motors only stream feedback AFTER enable, so a "no reply"
before running this is normal.

Usage:
    python example/enable_all.py                  # config from config/rebotarm.yaml
    python example/enable_all.py rebotarm_rs.yaml # force a hardware config
"""
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from reBotArm_control_py.actuator import RebotArm


def main() -> None:
    hw_yaml = sys.argv[1] if len(sys.argv) > 1 else None
    arm = RebotArm(hw_yaml)
    print(f"Using config: {arm.hardware_yaml}")
    arm.connect()
    print("--- Connected ---")

    arm.enable_all()
    time.sleep(0.5)

    # Poll feedback a few times so every motor gets a chance to reply.
    for _ in range(10):
        arm.get_state()
        time.sleep(0.05)

    pos, _, _ = arm.get_state()
    print(f"\n  {'Motor':<10} {'Pos(deg)':>10} {'Status'}")
    print(f"  {'-' * 34}")
    ok = True
    for i, name in enumerate(arm.joint_names):
        st = arm._motor_map[name].get_state()
        if st is None:
            status = "NO REPLY"
            ok = False
        elif st.status_code != 0:
            status = f"fault={st.status_code}"
            ok = False
        else:
            status = "ok"
        print(f"  {name:<10} {np.degrees(pos[i]):>+10.2f}   {status}")

    if ok:
        print("\n--- All motors enabled (left enabled on exit) ---")
    else:
        print("\n--- WARNING: some motors did not reply or report faults ---")

    # Close the bus WITHOUT disabling so motors stay enabled.
    # (arm.disconnect() would disable everything.)
    for ctrl in arm._ctrl_map.values():
        ctrl.close()

    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
