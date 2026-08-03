import unittest

import numpy as np

from arm_agent.scene_collision import build_scene_map, payload_radius_m


class SceneCollisionMapTest(unittest.TestCase):
    def test_depth_points_become_inflated_base_frame_columns(self):
        depth = np.zeros((12, 12), dtype=np.float32)
        depth[4:8, 4:8] = 200.0
        scene = build_scene_map(
            depth, (100.0, 100.0, 6.0, 6.0),
            ee_xyz_m=(0.30, 0.0, 0.20),
            camera_mapping_mm_per_m=-1000.0 * np.eye(3),
            jaw_target_cam_mm=(0.0, 0.0, 100.0),
            table_z_m=0.0,
            workspace_lo_m=(0.0, -0.5, 0.0),
            workspace_hi_m=(0.6, 0.5, 0.6),
            near_mm=100.0, far_mm=300.0, stride_px=1,
            voxel_m=0.02, inflation_m=0.025,
            minimum_points_per_column=1,
        )
        self.assertGreater(len(scene.boxes), 0)
        self.assertTrue(all(box.size_m[0] >= 0.07 for box in scene.boxes))
        self.assertEqual(scene.valid_depth_points, 16)
        self.assertGreater(scene.mapped_obstacle_points, 0)

    def test_target_pixels_are_not_obstacles(self):
        depth = np.full((10, 10), 200.0, dtype=np.float32)
        scene = build_scene_map(
            depth, (100.0, 100.0, 5.0, 5.0),
            ee_xyz_m=(0.30, 0.0, 0.20),
            camera_mapping_mm_per_m=-1000.0 * np.eye(3),
            jaw_target_cam_mm=(0.0, 0.0, 100.0),
            table_z_m=0.0,
            workspace_lo_m=(0.0, -0.5, 0.0),
            workspace_hi_m=(0.6, 0.5, 0.6),
            exclude_pixel_boxes=[(0, 0, 10, 10)], target_pad_px=0,
            near_mm=100.0, far_mm=300.0, stride_px=1,
        )
        self.assertEqual(scene.excluded_target_points, 100)
        self.assertEqual(scene.boxes, ())

    def test_payload_envelope_is_bounded(self):
        context = {
            "grasp_geometry": {
                "envelope_detection": {"max_width_mm": 1000.0},
            }
        }
        self.assertEqual(payload_radius_m(context), 0.08)

    def test_rs_urdf_rejects_obstacle_at_home_gripper(self):
        from arm_agent.local_arm import LocalArm

        arm = LocalArm()
        arm.configure_collision_scene([{
            "center_m": [0.3015, 0.0, 0.218],
            "size_m": [0.06, 0.06, 0.06],
        }], table_z_m=0.01196)
        detail = arm._configuration_collision(np.zeros(6))
        self.assertTrue(detail["collision"])
        self.assertEqual(detail["kind"], "environment")

    def test_rs_urdf_home_is_clear_with_empty_table_scene(self):
        from arm_agent.local_arm import LocalArm

        arm = LocalArm()
        arm.configure_collision_scene([], table_z_m=0.01196)
        detail = arm._configuration_collision(np.zeros(6))
        self.assertFalse(detail["collision"])


if __name__ == "__main__":
    unittest.main()
