import unittest
from tempfile import TemporaryDirectory

import numpy as np

from arm_agent.scene_collision import (
    build_dense_view_cloud, build_scene_map, payload_radius_m,
    save_scene_artifacts,
)


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

    def test_scene_map_exports_ply_and_interactive_recording(self):
        depth = np.zeros((12, 12), dtype=np.float32)
        depth[4:8, 4:8] = 200.0
        scene = build_scene_map(
            depth, (100.0, 100.0, 6.0, 6.0),
            ee_xyz_m=(0.30, 0.0, 0.20),
            camera_mapping_mm_per_m=-1000.0 * np.eye(3),
            jaw_target_cam_mm=(0.0, 0.0, 100.0), table_z_m=0.0,
            workspace_lo_m=(0.0, -0.5, 0.0),
            workspace_hi_m=(0.6, 0.5, 0.6),
            near_mm=100.0, far_mm=300.0, stride_px=1,
            minimum_points_per_column=1,
        )
        with TemporaryDirectory() as directory:
            artifacts = save_scene_artifacts(scene, directory)
            with open(artifacts["ply"], encoding="utf-8") as exported:
                text = exported.read()
            self.assertIn("element vertex", text)
            self.assertGreater(artifacts["ply_vertices"], len(scene.points_m))
            if "rerun" in artifacts:
                with open(artifacts["rerun"], "rb") as recording:
                    self.assertGreater(len(recording.read()), 0)

    def test_dense_view_cloud_preserves_camera_rgb(self):
        depth = np.full((12, 12), 200.0, dtype=np.float32)
        color = np.zeros((12, 12, 3), dtype=np.uint8)
        color[:] = [10, 20, 230]  # BGR input -> RGB output
        points, colors = build_dense_view_cloud(
            color, depth, (100.0, 100.0, 6.0, 6.0),
            ee_xyz_m=(0.30, 0.0, 0.20),
            camera_mapping_mm_per_m=-1000.0 * np.eye(3),
            jaw_target_cam_mm=(0.0, 0.0, 100.0),
            workspace_lo_m=(0.0, -0.5, 0.0),
            workspace_hi_m=(0.6, 0.5, 0.6),
            near_mm=100.0, far_mm=300.0, voxel_m=0.0015,
        )
        self.assertGreater(len(points), 20)
        self.assertTrue(np.all(colors == np.array([230, 20, 10])))

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
