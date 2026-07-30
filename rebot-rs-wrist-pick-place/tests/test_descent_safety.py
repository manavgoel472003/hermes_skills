from __future__ import annotations

from unittest import TestCase
from unittest.mock import patch

from arm_agent import arm_safe


class SafeDescendTest(TestCase):
    def test_refuses_all_zero_torque_without_moving(self):
        moves = []
        with (
            patch.object(arm_safe, "ee_pos", lambda: [0.3, 0.0, 0.1]),
            patch.object(arm_safe, "joint_torques", lambda: [0.0] * 6),
            patch.object(arm_safe, "safe_move", lambda *a, **k: moves.append((a, k))),
        ):
            result = arm_safe.safe_descend(0.3, 0.0, 0.02)

        self.assertFalse(result.ok)
        self.assertEqual(result.status, "feedback_unavailable")
        self.assertEqual(moves, [])

    def test_move_refusal_is_failure_not_contact(self):
        refused = arm_safe.MoveResult(False, "ik_failed", (0.3, 0.0, 0.1))
        with (
            patch.object(arm_safe, "ee_pos", lambda: [0.3, 0.0, 0.1]),
            patch.object(arm_safe, "joint_torques", lambda: [1.0] * 6),
            patch.object(arm_safe, "safe_move", lambda *a, **k: refused),
        ):
            result = arm_safe.safe_descend(0.3, 0.0, 0.02)

        self.assertFalse(result.ok)
        self.assertEqual(result.status, "move_refused")
        self.assertIn("ik_failed", result.reason)

    def test_position_stall_is_successful_contact(self):
        stalled = arm_safe.MoveResult(True, "arrived", (0.3, 0.0, 0.099))
        with (
            patch.object(arm_safe, "ee_pos", lambda: [0.3, 0.0, 0.1]),
            patch.object(arm_safe, "joint_torques", lambda: [1.0] * 6),
            patch.object(arm_safe, "safe_move", lambda *a, **k: stalled),
        ):
            result = arm_safe.safe_descend(0.3, 0.0, 0.02, step=0.004)

        self.assertTrue(result.ok)
        self.assertEqual(result.status, "contact_stall")
