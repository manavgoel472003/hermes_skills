import unittest

import numpy as np

from arm_agent.local_arm import LocalArm


class JointPlanSafetyTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.arm = LocalArm()

    def test_configuration_gate_rejects_joint_hard_limit(self):
        q = np.zeros(6)
        q[0] = self.arm._joint_hi[0] + 0.1
        result = self.arm.configuration_valid(q)
        self.assertFalse(result["ok"])
        self.assertIn("hard limit", result["reason"])

    def test_path_gate_rejects_stale_scene_version(self):
        version = self.arm.collision_scene_version
        self.arm.clear_collision_scene()
        result = self.arm.validate_joint_path(
            [np.zeros(6)], scene_version=version
        )
        self.assertFalse(result["ok"])
        self.assertIn("stale scene", result["reason"])

    def test_plan_certificate_detects_path_mutation(self):
        start = np.zeros(6)
        path = [np.zeros(6), np.full(6, 0.01)]
        certificate = self.arm._plan_certificate(3, start, path)
        path[-1][2] += 0.001
        self.assertNotEqual(
            certificate, self.arm._plan_certificate(3, start, path)
        )


if __name__ == "__main__":
    unittest.main()
