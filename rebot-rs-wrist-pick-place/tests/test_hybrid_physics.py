"""Physics/evaluator tests, NOT demonstrations of policy task success.

Run with the lerobot environment (MuJoCo installed). Fixture qpos assignments
below are initial conditions only; production actions only command actuators.
"""
import importlib.util
from pathlib import Path
import sys
import unittest

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
_VENDOR = _ROOT / ".vendor" / "hybrid_physics_deps"
if _VENDOR.is_dir():
    sys.path.insert(0, str(_VENDOR))
HAS_MUJOCO = importlib.util.find_spec("mujoco") is not None


@unittest.skipUnless(HAS_MUJOCO, "MuJoCo not installed in this interpreter")
class PhysicsTests(unittest.TestCase):
    def setUp(self):
        from arm_agent.hybrid_physics import PhysicsWorld
        self.world = PhysicsWorld("marker", seed=8)

    def tearDown(self):
        self.world.close()

    def test_scene_has_free_objects_no_weld_or_mocap(self):
        import mujoco
        model = self.world.model
        self.assertEqual(model.neq, 0)
        self.assertEqual(model.nmocap, 0)
        for name in ("target_free", "cup_free"):
            self.assertEqual(model.joint(name).type, mujoco.mjtJoint.mjJNT_FREE)
        self.assertEqual(model.nu, 8)  # Robot joints only, no object actuators.

    def test_actuator_command_does_not_teleport_robot_or_object(self):
        world = self.world
        before = world.data.qpos.copy()
        world.command_arm([.15, .1, .1, 0, 0, 0])
        np.testing.assert_array_equal(world.data.qpos, before)
        world.step(1000)
        self.assertGreater(abs(world.data.qpos[0]-before[0]), .05)
        # The marker is not parented or attached to the arm as it moves.
        initial = np.asarray(world.meta["object_initial_xyz"])
        np.testing.assert_allclose(world.data.body("target").xpos[:2], initial[:2], atol=.002)

    def test_zero_action_cannot_pass_the_task(self):
        result = self.world.verdict()
        self.assertFalse(result["success"])
        self.assertFalse(result["lift_observed"])
        self.assertFalse(result["displaced_over_40mm"])

    def test_marker_remains_upright_under_gravity(self):
        self.world.step(3000)
        axis = self.world.data.body("target").xmat.reshape(3, 3)[:, 2]
        self.assertGreater(float(axis[2]), .98)

    def test_actual_contact_gravity_and_hollow_cup(self):
        # Evaluator/physics fixture: release a ball above the cup at reset.
        # This is explicitly NOT a policy run or evidence of grasp success.
        from arm_agent.hybrid_physics import PhysicsWorld, evaluate_sample
        world = PhysicsWorld("ball", seed=3)
        try:
            address = world.model.jnt_qposadr[world.model.joint("target_free").id]
            cup_xyz = world.data.body("cup").xpos.copy()
            world.data.qpos[address:address+3] = cup_xyz + [0, 0, .13]
            world.step(1500)
            sample = evaluate_sample(world.model, world.data, world.meta)
            self.assertTrue(sample["inside"])
            self.assertTrue(sample["released"])
            self.assertTrue(sample["settled"])
            self.assertFalse(sample["left_of_cup"])
        finally:
            world.close()

    def test_camera_frame_has_only_rgb_depth_and_intrinsics(self):
        rgb, depth, intr = self.world.frame()
        self.assertEqual(rgb.shape, (480, 640, 3))
        self.assertEqual(depth.shape, (480, 640))
        self.assertEqual(len(intr), 4)
        self.assertGreater(np.count_nonzero(depth), 100)
        self.assertTrue(np.isfinite(depth).all())

    def test_reset_seed_is_reproducible_and_changes_layout(self):
        from arm_agent.hybrid_physics import build_scene
        a, b, meta, _ = build_scene("marker", seed=8)
        np.testing.assert_allclose(meta["object_initial_xyz"], self.world.meta["object_initial_xyz"])
        _, _, other, _ = build_scene("marker", seed=9)
        self.assertNotEqual(meta["object_initial_xyz"], other["object_initial_xyz"])


if __name__ == "__main__":
    unittest.main()
