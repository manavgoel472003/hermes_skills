from __future__ import annotations

import math
import threading
from unittest import TestCase
from unittest.mock import patch

import numpy as np

from arm_agent.local_arm import LocalArm


class _Group:
    def __init__(self):
        self._mit_kp = np.array([8.0])
        self._mit_kd = np.array([1.0])


class _Arm:
    def __init__(self, start_deg: float):
        self.position = math.radians(start_deg)

    def get_state(self):
        q = np.zeros(7)
        q[6] = self.position
        return q, np.zeros(7), np.zeros(7)


class _Controller:
    def __init__(self, arm: _Arm):
        self.arm = arm
        self._gripper_group = _Group()
        self.commands: list[float] = []

    def set_gripper_target(self, target: float):
        self.commands.append(float(target))
        self.arm.position = float(target)


def _local(arm: _Arm, controller: _Controller) -> LocalArm:
    local = object.__new__(LocalArm)
    local._lock = threading.RLock()
    local._require = lambda: (arm, controller)
    return local


class PositionOnlyGripperTest(TestCase):
    def test_explicit_sized_close_reaches_only_that_position(self):
        arm = _Arm(start_deg=150.0)
        controller = _Controller(arm)
        local = _local(arm, controller)

        with patch("arm_agent.local_arm.time.sleep", lambda seconds: None):
            result = local.gripper(
                "close",
                target_rad=math.radians(60.0),
                limit_rads=(0.0, math.radians(338.63)),
                stop_on_contact=False,
            )

        self.assertTrue(result["ok"])
        self.assertFalse(result["contact_aware"])
        self.assertAlmostEqual(result["position_deg"], 60.0, places=1)
        self.assertAlmostEqual(math.degrees(result["target_pos_rad"]), 60.0, places=1)
        self.assertAlmostEqual(math.degrees(min(controller.commands)), 60.0, places=6)

    def test_contact_inference_request_is_refused_without_motion(self):
        arm = _Arm(start_deg=150.0)
        controller = _Controller(arm)
        local = _local(arm, controller)

        result = local.gripper(
            "close",
            target_rad=math.radians(60.0),
            limit_rads=(0.0, math.radians(338.63)),
            stop_on_contact=True,
        )

        self.assertFalse(result["ok"])
        self.assertIn("disabled", result["reason"])
        self.assertEqual(controller.commands, [])

    def test_open_does_not_use_contact_inference(self):
        arm = _Arm(start_deg=15.0)
        controller = _Controller(arm)
        local = _local(arm, controller)

        with patch("arm_agent.local_arm.time.sleep", lambda seconds: None):
            result = local.gripper(
                "open",
                limit_rads=(0.0, math.radians(338.63)),
            )

        self.assertTrue(result["ok"])
        self.assertFalse(result["contact_aware"])
        self.assertAlmostEqual(result["position_deg"], 150.0, places=1)
