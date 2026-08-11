import unittest
from unittest.mock import patch

import numpy as np

from arm_agent import wrist_servo as ws


class GlobalRouteIntegrationTest(unittest.TestCase):
    def setUp(self):
        self.plan = {
            "ok": True,
            "kind": "certified_joint_plan",
            "joint_path": [[0.0] * 6, [0.1] * 6],
            "scene_version": 4,
            "planner": {"nodes": 18, "elapsed_s": 0.2, "reason": "found"},
        }

    def test_carry_uses_global_search_after_corridors_fail(self):
        with patch.object(ws.A, "ee_pos", return_value=[0.30, 0.0, 0.22]), \
                patch.object(ws.A, "check_pose_clearance",
                             return_value=ws.A.MoveResult(True, "clear")), \
                patch.object(ws, "_plan_fixed_approach_trajectory",
                             return_value=(None, "blocked")), \
                patch.object(ws.A, "plan_joint_motion",
                             return_value=self.plan) as planner:
            route, reason = ws._high_release_waypoints(
                np.array([0.40, 0.10, 0.20]), (0.0, 0.0, 0.0)
            )
        self.assertIs(route, self.plan)
        self.assertIn("global joint route", reason)
        planner.assert_called_once()

    def test_grasp_initial_global_search_stops_at_visible_checkpoint(self):
        with patch.object(ws.A, "ee_pos", return_value=[0.30, 0.0, 0.22]), \
                patch.object(ws, "_plan_fixed_approach_trajectory",
                             return_value=(None, "blocked")), \
                patch.object(ws.A, "plan_joint_motion",
                             return_value=self.plan):
            route, _, detail = ws._plan_clearance_axial_trajectory(
                np.array([0.40, 0.0, 0.12]), -1000.0 * np.eye(3),
                (0.0, 0.0, 0.0), phase="initial",
            )
        self.assertIs(route, self.plan)
        self.assertTrue(detail["global_checkpoint_only"])
        self.assertGreater(detail["checkpoint_remaining_m"], 0.06)

    def test_executor_delegates_certified_joint_plan(self):
        result = ws.A.MoveResult(True, "done")
        messages = []
        with patch.object(ws.A, "execute_joint_plan", return_value=result) as execute:
            ok, reason = ws._execute_fixed_approach_trajectory(
                self.plan, messages.append
            )
        self.assertTrue(ok)
        self.assertEqual(reason, "done")
        execute.assert_called_once_with(self.plan)
        self.assertTrue(any("revalidating" in message for message in messages))


if __name__ == "__main__":
    unittest.main()
