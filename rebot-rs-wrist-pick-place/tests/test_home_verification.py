from __future__ import annotations

import threading
from unittest import TestCase
from unittest.mock import patch

import numpy as np

from arm_agent.local_arm import LocalArm


class _Arm:
    def get_state(self):
        return np.array([2.0, 0.5, 0.0, 0.0, 0.0, 0.0, 1.0]), np.zeros(7), np.zeros(7)


class _Controller:
    def __init__(self):
        self.timeout = None

    def safe_home(self, *, max_vel, timeout):
        self.timeout = timeout
        return True


class HomeVerificationTest(TestCase):
    def test_home_timeout_scales_with_trajectory_and_requires_final_state(self):
        arm = _Arm()
        controller = _Controller()
        local = object.__new__(LocalArm)
        local._lock = threading.RLock()
        local._require = lambda: (arm, controller)
        local.state = lambda: {
            "joints_rad": [0.01, -0.01, 0.0, 0.0, 0.0, 0.0],
            "joint_vel_rad_s": [0.01] * 6,
            "end_effector": {"position_m": [0.3, 0.0, 0.22]},
        }

        with patch("arm_agent.local_arm._joint_vlim", lambda: 0.25):
            result = local.home()

        self.assertGreater(controller.timeout, 15.0)
        self.assertTrue(result["ok"])
        self.assertLess(result["max_joint_error_rad"], 0.03)

    def test_home_does_not_claim_success_when_controller_cannot_confirm(self):
        arm = _Arm()
        controller = _Controller()
        controller.safe_home = lambda **kwargs: False
        local = object.__new__(LocalArm)
        local._lock = threading.RLock()
        local._require = lambda: (arm, controller)
        local.state = lambda: {
            "joints_rad": [0.4, 0.0, 0.0, 0.0, 0.0, 0.0],
            "joint_vel_rad_s": [0.0] * 6,
            "end_effector": {"position_m": [0.32, 0.0, 0.30]},
        }

        result = local.home()

        self.assertFalse(result["ok"])
        self.assertIn("home_not_confirmed", result["reason"])

    def test_home_does_not_confirm_while_joint_is_still_moving(self):
        arm = _Arm()
        controller = _Controller()
        local = object.__new__(LocalArm)
        local._lock = threading.RLock()
        local._require = lambda: (arm, controller)
        local.state = lambda: {
            "joints_rad": [0.005] * 6,
            "joint_vel_rad_s": [0.12, 0.0, 0.0, 0.0, 0.0, 0.0],
            "end_effector": {"position_m": [0.3, 0.0, 0.22]},
        }

        with (
            patch("arm_agent.local_arm.HOME_SETTLE_TIMEOUT_S", 0.0),
            patch("arm_agent.local_arm._joint_vlim", lambda: 0.25),
        ):
            result = local.home()

        self.assertFalse(result["ok"])
        self.assertIn("max_joint_velocity", result["reason"])
