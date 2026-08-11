from __future__ import annotations

import unittest

import numpy as np

from arm_agent.joint_planner import PlannerSettings, plan_rrt_connect


class JointPlannerTest(unittest.TestCase):
    def test_direct_path_is_densified_and_bounded(self):
        result = plan_rrt_connect(
            np.array([0.0, 0.0]), np.array([0.2, -0.1]),
            np.array([-1.0, -1.0]), np.array([1.0, 1.0]),
            lambda _q: True,
            settings=PlannerSettings(edge_step_rad=0.03),
        )

        self.assertTrue(result.ok, result.reason)
        path = np.asarray(result.path)
        self.assertTrue(np.allclose(path[0], [0.0, 0.0]))
        self.assertTrue(np.allclose(path[-1], [0.2, -0.1]))
        self.assertLessEqual(np.max(np.abs(np.diff(path, axis=0))), 0.030001)

    def test_rrt_routes_around_blocked_joint_space(self):
        def valid(q):
            # A vertical wall blocks the direct path but leaves room above it.
            return not (-0.12 <= q[0] <= 0.12 and q[1] < 0.45)

        result = plan_rrt_connect(
            np.array([-0.8, 0.0]), np.array([0.8, 0.0]),
            np.array([-1.0, -1.0]), np.array([1.0, 1.0]), valid,
            settings=PlannerSettings(
                timeout_s=1.0, max_nodes=2500, seed=7,
                step_rad=0.12, edge_step_rad=0.025,
            ),
        )

        self.assertTrue(result.ok, result.reason)
        self.assertTrue(all(valid(np.asarray(q)) for q in result.path))
        self.assertGreater(np.max(np.asarray(result.path)[:, 1]), 0.45)

    def test_impossible_goal_fails_without_relaxing_validity(self):
        result = plan_rrt_connect(
            np.array([-0.8, 0.0]), np.array([0.0, 0.0]),
            np.array([-1.0, -1.0]), np.array([1.0, 1.0]),
            lambda q: not (-0.2 <= q[0] <= 0.2),
            settings=PlannerSettings(timeout_s=0.05, max_nodes=50),
        )

        self.assertFalse(result.ok)
        self.assertIn("goal", result.reason)


if __name__ == "__main__":
    unittest.main()
