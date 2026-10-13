#!/usr/bin/env python3
"""Disable all motors and report which ones acknowledged.

Run this after a crash or whenever motors are stuck enabled (stiff/making noise).

Note: RobStride motors only send feedback in response to enable/control frames,
so this script cannot report joint positions (that would require enabling).
A motor listed as "acknowledged" replied to the disable command; "no reply"
usually means it was already disabled or is unpowered.

Usage:
    python example/disable_all.py                  # config from config/rebotarm.yaml
    python example/disable_all.py rebotarm_rs.yaml # force a hardware config
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from reBotArm_control_py.actuator import RebotArm


def main() -> None:
    hw_yaml = sys.argv[1] if len(sys.argv) > 1 else None
    arm = RebotArm(hw_yaml)
    print(f"Using config: {arm.hardware_yaml}")
    arm.connect()
    print("--- Connected ---")

    arm.stop_control_loop()
    arm.disable_all()
    time.sleep(0.3)
    arm.disable_all()   # second pass for good measure
    time.sleep(0.3)

    print(f"\n  {'Motor':<10} {'Result'}")
    print(f"  {'-' * 30}")
    for name in arm.joint_names:
        st = arm._motor_map[name].get_state()
        if st is None:
            result = "no reply (already off / unpowered?)"
        elif st.status_code != 0:
            result = f"acknowledged, fault={st.status_code}"
        else:
            result = "acknowledged"
        print(f"  {name:<10} {result}")

    print("\n--- All motors disabled ---")
    arm.disconnect()


if __name__ == "__main__":
    main()
