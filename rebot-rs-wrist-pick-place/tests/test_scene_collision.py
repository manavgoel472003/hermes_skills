import unittest
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np

from arm_agent.scene_collision import (
    ObstacleBox, SceneMap, build_dense_view_cloud, build_scene_map,
    consolidate_obstacle_boxes, consolidate_scene_map, crop_dense_view_clouds,
    fuse_dense_view_clouds, fuse_scene_maps, register_dense_view_clouds,
    median_depth_image, _column_owned_by_target_obb,
    payload_radius_m,
    save_scene_artifacts,
)


class SceneCollisionMapTest(unittest.TestCase):
    def test_consolidation_exactly_merges_overlapping_equal_height_columns(self):
        boxes = (
            ObstacleBox((0.00, 0.00, 0.10), (0.035, 0.035, 0.10), 4),
            ObstacleBox((0.02, 0.00, 0.10), (0.035, 0.035, 0.10), 6),
        )

        compact = consolidate_obstacle_boxes(boxes)

        self.assertEqual(len(compact), 1)
        self.assertTrue(np.allclose(compact[0].center_m, (0.01, 0.0, 0.10)))
        self.assertTrue(np.allclose(compact[0].size_m, (0.055, 0.035, 0.10)))
        self.assertEqual(compact[0].point_count, 10)

    def test_consolidation_never_weakens_source_box_coverage(self):
        boxes = (
            ObstacleBox((0.00, 0.00, 0.10), (0.035, 0.035, 0.10), 4),
            # Same conservative table bottom and only 5 mm additional height.
            ObstacleBox((0.02, 0.00, 0.1025),
                        (0.035, 0.035, 0.105), 6),
        )

        compact = consolidate_obstacle_boxes(boxes)

        self.assertEqual(len(compact), 1)
        merged_lo = np.asarray(compact[0].center_m) \
            - np.asarray(compact[0].size_m) / 2.0
        merged_hi = np.asarray(compact[0].center_m) \
            + np.asarray(compact[0].size_m) / 2.0
        for source in boxes:
            source_lo = np.asarray(source.center_m) \
                - np.asarray(source.size_m) / 2.0
            source_hi = np.asarray(source.center_m) \
                + np.asarray(source.size_m) / 2.0
            self.assertTrue(np.all(merged_lo <= source_lo + 1e-12))
            self.assertTrue(np.all(merged_hi >= source_hi - 1e-12))

    def test_consolidation_does_not_fill_large_diagonal_concavity(self):
        boxes = (
            ObstacleBox((0.00, 0.00, 0.10), (0.035, 0.035, 0.10), 4),
            ObstacleBox((0.02, 0.02, 0.10), (0.035, 0.035, 0.10), 6),
        )

        compact = consolidate_obstacle_boxes(boxes)

        self.assertEqual(compact, boxes)

    def test_consolidated_scene_reports_before_and_after_counts(self):
        scene = SceneMap(
            boxes=(
                ObstacleBox((0.00, 0.00, 0.10),
                            (0.035, 0.035, 0.10), 4),
                ObstacleBox((0.02, 0.00, 0.10),
                            (0.035, 0.035, 0.10), 6),
            ),
            table_z_m=0.0, voxel_m=0.02, inflation_m=0.0075,
            valid_depth_points=10, mapped_obstacle_points=10,
            excluded_target_points=0, closest_obstacle_m=None,
        )

        compact = consolidate_scene_map(scene)
        detail = compact.as_dict()

        self.assertEqual(detail["unconsolidated_obstacle_columns"], 2)
        self.assertEqual(detail["consolidated_obstacle_columns"], 1)
        self.assertEqual(detail["consolidated_obstacle_reduction"], 1)
        self.assertEqual(detail["obstacle_columns"], 1)

    def test_base_frame_multiview_fusion_unions_distinct_columns(self):
        first = SceneMap(
            (), 0.0, 0.02, 0.0075, 10, 2, 0, None,
            ((0.101, 0.001, 0.05), (0.102, 0.002, 0.06)),
        )
        second = SceneMap(
            (), 0.0, 0.02, 0.0075, 10, 2, 0, None,
            ((0.141, 0.001, 0.07), (0.142, 0.002, 0.08)),
        )
        fused = fuse_scene_maps([first, second], ee_xyz_m=(0.3, 0.0, 0.2))
        self.assertEqual(len(fused.boxes), 2)
        self.assertEqual(fused.mapped_obstacle_points, 4)
        self.assertEqual(fused.valid_depth_points, 20)

    def test_multiview_consensus_rejects_one_frame_noise(self):
        first = SceneMap(
            (), 0.0, 0.02, 0.0075, 10, 4, 0, None,
            ((0.101, 0.001, 0.05), (0.102, 0.002, 0.06),
             (0.181, 0.001, 0.09), (0.182, 0.002, 0.10)),
        )
        second = SceneMap(
            (), 0.0, 0.02, 0.0075, 10, 2, 0, None,
            ((0.101, 0.001, 0.051), (0.102, 0.002, 0.061)),
        )
        fused = fuse_scene_maps(
            [first, second], ee_xyz_m=(0.3, 0.0, 0.2),
            minimum_views_per_column=2,
        )
        self.assertEqual(len(fused.boxes), 1)
        self.assertAlmostEqual(fused.boxes[0].center_m[0], 0.11)

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

    def test_semantic_target_mask_is_excluded_without_box_expansion(self):
        depth = np.full((10, 10), 200.0, dtype=np.float32)
        mask = np.zeros((10, 10), dtype=np.uint8)
        mask[:, :5] = 255
        scene = build_scene_map(
            depth, (100.0, 100.0, 5.0, 5.0),
            ee_xyz_m=(0.30, 0.0, 0.20),
            camera_mapping_mm_per_m=-1000.0 * np.eye(3),
            jaw_target_cam_mm=(0.0, 0.0, 100.0),
            table_z_m=0.0,
            workspace_lo_m=(0.0, -0.5, 0.0),
            workspace_hi_m=(0.6, 0.5, 0.6),
            exclude_pixel_masks=[mask],
            near_mm=100.0, far_mm=300.0, stride_px=1,
            minimum_points_per_column=1,
        )
        self.assertEqual(scene.excluded_target_points, 50)
        self.assertGreater(scene.mapped_obstacle_points, 0)
        self.assertGreater(len(scene.boxes), 0)

    def test_target_padding_removes_neighboring_depth_pixels(self):
        depth = np.full((20, 20), 200.0, dtype=np.float32)
        scene = build_scene_map(
            depth, (100.0, 100.0, 10.0, 10.0),
            ee_xyz_m=(0.30, 0.0, 0.20),
            camera_mapping_mm_per_m=-1000.0 * np.eye(3),
            jaw_target_cam_mm=(0.0, 0.0, 100.0),
            table_z_m=0.0,
            workspace_lo_m=(0.0, -0.5, 0.0),
            workspace_hi_m=(0.6, 0.5, 0.6),
            exclude_pixel_boxes=[(8, 8, 12, 12)], target_pad_px=8,
            near_mm=100.0, far_mm=300.0, stride_px=1,
            minimum_points_per_column=1,
        )
        self.assertEqual(scene.excluded_target_points, 400)
        self.assertEqual(scene.boxes, ())

    def test_target_capsule_removes_only_measured_3d_volume(self):
        depth = np.full((20, 20), 200.0, dtype=np.float32)
        common = dict(
            ee_xyz_m=(0.30, 0.0, 0.20),
            camera_mapping_mm_per_m=-1000.0 * np.eye(3),
            jaw_target_cam_mm=(0.0, 0.0, 100.0), table_z_m=0.0,
            workspace_lo_m=(0.0, -0.5, 0.0),
            workspace_hi_m=(0.6, 0.5, 0.6),
            near_mm=100.0, far_mm=300.0, stride_px=1,
            minimum_points_per_column=1,
        )
        unfiltered = build_scene_map(
            depth, (100.0, 100.0, 10.0, 10.0), **common
        )
        filtered = build_scene_map(
            depth, (100.0, 100.0, 10.0, 10.0),
            exclude_base_capsules=[(
                (0.30, 0.0, 0.30), (1.0, 0.0, 0.0), 0.005, 0.005
            )], **common
        )
        self.assertLess(filtered.mapped_obstacle_points,
                        unfiltered.mapped_obstacle_points)
        self.assertGreater(filtered.mapped_obstacle_points, 0)

    def test_target_obb_removes_target_returns_but_preserves_close_neighbor(self):
        depth = np.full((31, 31), 200.0, dtype=np.float32)
        common = dict(
            ee_xyz_m=(0.30, 0.0, 0.20),
            camera_mapping_mm_per_m=-1000.0 * np.eye(3),
            jaw_target_cam_mm=(0.0, 0.0, 100.0), table_z_m=0.0,
            workspace_lo_m=(0.0, -0.5, 0.0),
            workspace_hi_m=(0.6, 0.5, 0.6),
            near_mm=100.0, far_mm=300.0, stride_px=1,
            voxel_m=0.01, inflation_m=0.0075,
            minimum_points_per_column=1,
        )
        unfiltered = build_scene_map(
            depth, (100.0, 100.0, 15.0, 15.0), **common
        )
        # Camera/base mapping places the image centre at [0.30, 0.00, 0.30].
        # Exclude a rotated 40x20x20 mm target while retaining returns only
        # 30 mm away along Y, outside its measured oriented footprint.
        angle = np.deg2rad(25.0)
        axes = np.array([
            [np.cos(angle), np.sin(angle), 0.0],
            [-np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ])
        filtered = build_scene_map(
            depth, (100.0, 100.0, 15.0, 15.0),
            exclude_base_obbs=[(
                (0.30, 0.0, 0.30), axes, (0.020, 0.010, 0.010)
            )], **common
        )

        self.assertLess(filtered.mapped_obstacle_points,
                        unfiltered.mapped_obstacle_points)
        remaining = np.asarray(filtered.points_m, dtype=float)
        self.assertGreater(len(remaining), 0)
        local = (remaining - np.array([0.30, 0.0, 0.30])) @ axes.T
        self.assertFalse(np.any(np.all(
            np.abs(local) <= np.array([0.020, 0.010, 0.010]), axis=1
        )))
        self.assertTrue(np.any(np.abs(remaining[:, 1]) >= 0.025))

    def test_target_obb_owns_boundary_voxel_but_not_taller_neighbor(self):
        yaw = np.deg2rad(-87.0)
        axes = np.array([
            [np.cos(yaw), np.sin(yaw), 0.0],
            [-np.sin(yaw), np.cos(yaw), 0.0],
            [0.0, 0.0, 1.0],
        ])
        target = (
            np.array([0.40858, 0.10495, 0.1398]), axes,
            np.array([0.013335, 0.03779, 0.03111]),
        )
        # This is the real failing boundary cell: its centre narrowly misses
        # the OBB, while its raw 20 mm footprint intersects it. Its extra height
        # is within one voxel of the measured target and is quantisation spill.
        self.assertTrue(_column_owned_by_target_obb(
            (0.370, 0.110), 0.18866, 0.020, target
        ))
        # A few millimetres of bounded RealSense/fusion height spread must not
        # recreate the selected object as a collision column.
        self.assertTrue(_column_owned_by_target_obb(
            (0.370, 0.110), 0.194, 0.020, target
        ))
        # A materially taller return in the same footprint remains an obstacle.
        self.assertFalse(_column_owned_by_target_obb(
            (0.370, 0.110), 0.205, 0.020, target
        ))
        self.assertFalse(_column_owned_by_target_obb(
            (0.330, 0.110), 0.160, 0.020, target
        ))

    def test_fusion_cannot_resurrect_target_owned_boundary_column(self):
        depth = np.zeros((20, 20), dtype=np.float32)
        # These returns sit just outside the exact target OBB, but inside a raw
        # 20 mm cell whose footprint and bounded height belong to the target.
        depth[10, 16:19] = 200.0
        common = dict(
            ee_xyz_m=(0.30, 0.0, 0.20),
            camera_mapping_mm_per_m=-1000.0 * np.eye(3),
            jaw_target_cam_mm=(0.0, 0.0, 100.0), table_z_m=0.0,
            workspace_lo_m=(0.0, -0.5, 0.0),
            workspace_hi_m=(0.6, 0.5, 0.6),
            exclude_base_obbs=[(
                (0.30, 0.0, 0.28), np.eye(3), (0.010, 0.010, 0.005)
            )],
            near_mm=100.0, far_mm=300.0, stride_px=1,
            voxel_m=0.02, inflation_m=0.0075,
            minimum_points_per_column=1,
        )
        first = build_scene_map(
            depth, (100.0, 100.0, 10.0, 10.0), **common
        )
        second = build_scene_map(
            depth, (100.0, 100.0, 10.0, 10.0), **common
        )
        fused = fuse_scene_maps([first, second], ee_xyz_m=(0.30, 0.0, 0.20))

        self.assertEqual(first.excluded_target_columns, 1)
        self.assertEqual(first.points_m, ())
        self.assertEqual(fused.boxes, ())
        self.assertEqual(fused.excluded_target_columns, 2)

    def test_target_footprint_prevents_edge_return_column_regeneration(self):
        depth = np.zeros((30, 30), dtype=np.float32)
        depth[14:17, 14:17] = 200.0
        scene = build_scene_map(
            depth, (100.0, 100.0, 15.0, 15.0),
            ee_xyz_m=(0.30, 0.0, 0.20),
            camera_mapping_mm_per_m=-1000.0 * np.eye(3),
            jaw_target_cam_mm=(0.0, 0.0, 100.0), table_z_m=0.0,
            workspace_lo_m=(0.0, -0.5, 0.0),
            workspace_hi_m=(0.6, 0.5, 0.6),
            exclude_base_capsules=[(
                (0.30, 0.0, 0.30), (1.0, 0.0, 0.0), 0.04, 0.02
            )],
            near_mm=100.0, far_mm=300.0, stride_px=1,
            minimum_points_per_column=1,
        )
        self.assertEqual(scene.boxes, ())

    def test_requested_seven_point_five_mm_inflation_is_honored(self):
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
            voxel_m=0.02, inflation_m=0.0075,
            minimum_points_per_column=1,
        )
        self.assertEqual(scene.inflation_m, 0.0075)
        self.assertTrue(all(np.isclose(box.size_m[0], 0.035)
                            for box in scene.boxes))

    def test_requested_three_mm_inflation_is_honored(self):
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
            voxel_m=0.02, inflation_m=0.003,
            minimum_points_per_column=1,
        )
        self.assertEqual(scene.inflation_m, 0.003)
        self.assertTrue(all(np.isclose(box.size_m[0], 0.026)
                            for box in scene.boxes))

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
            object_box = {
                "center_m": [0.30, 0.02, 0.04],
                "conservative_size_m": [0.10, 0.04, 0.03],
                "axes_base": [[0.0, 1.0, 0.0],
                              [-1.0, 0.0, 0.0],
                              [0.0, 0.0, 1.0]],
            }
            artifacts = save_scene_artifacts(
                scene, directory,
                named_view_clouds={
                    "ready pose": (
                        np.asarray(scene.points_m, dtype=float),
                        np.tile(
                            np.array([[10, 20, 30]], dtype=np.uint8),
                            (len(scene.points_m), 1),
                        ),
                    ),
                },
                object_boxes_3d={"pick_target": object_box},
            )
            with open(artifacts["ply"], encoding="utf-8") as exported:
                text = exported.read()
            self.assertIn("element vertex", text)
            self.assertGreater(artifacts["ply_vertices"], len(scene.points_m))
            self.assertEqual(artifacts["object_boxes"], ["pick_target"])
            self.assertIn("40 255 80", text)
            if "rerun" in artifacts:
                self.assertEqual(artifacts["named_views"], ["ready pose"])
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

    def test_dense_view_cloud_never_snaps_object_top_to_table(self):
        depth = np.full((12, 12), 250.0, dtype=np.float32)
        color = np.zeros((12, 12, 3), dtype=np.uint8)

        points, _ = build_dense_view_cloud(
            color, depth, (100.0, 100.0, 6.0, 6.0),
            ee_xyz_m=(0.30, 0.0, 0.20),
            camera_mapping_mm_per_m=1000.0 * np.eye(3),
            jaw_target_cam_mm=(0.0, 0.0, 100.0),
            workspace_lo_m=(0.0, -0.5, 0.0),
            workspace_hi_m=(0.6, 0.5, 0.6),
            table_z_m=0.012,
            near_mm=100.0, far_mm=300.0, voxel_m=0.0015,
        )

        # This broad 50 mm-high surface used to win the per-view table
        # histogram and get shifted down to ~12 mm. Preserve measured geometry.
        self.assertGreater(len(points), 20)
        self.assertAlmostEqual(float(np.median(points[:, 2])), 0.05, places=6)

    def test_dense_view_cloud_can_retain_points_outside_workspace(self):
        depth = np.full((8, 8), 200.0, dtype=np.float32)
        color = np.full((8, 8, 3), 120, dtype=np.uint8)
        common = dict(
            ee_xyz_m=(2.0, 2.0, 2.0),
            camera_mapping_mm_per_m=-1000.0 * np.eye(3),
            jaw_target_cam_mm=(0.0, 0.0, 100.0),
            workspace_lo_m=(0.0, -0.5, 0.0),
            workspace_hi_m=(0.6, 0.5, 0.6),
        )
        cropped, _ = build_dense_view_cloud(
            color, depth, (100.0, 100.0, 4.0, 4.0),
            crop_to_workspace=True, **common,
        )
        diagnostic, _ = build_dense_view_cloud(
            color, depth, (100.0, 100.0, 4.0, 4.0),
            crop_to_workspace=False, crop_below_table=False,
            minimum_neighbor_support=1, **common,
        )
        self.assertEqual(len(cropped), 0)
        self.assertGreater(len(diagnostic), 0)

    def test_stationary_depth_median_rejects_one_frame_layer(self):
        first = np.full((3, 4), 200.0, dtype=np.float32)
        second = np.full((3, 4), 200.0, dtype=np.float32)
        third = np.full((3, 4), 200.0, dtype=np.float32)
        fourth = np.full((3, 4), 200.0, dtype=np.float32)
        outlier = np.full((3, 4), 250.0, dtype=np.float32)
        first[0, 0] = 0.0
        second[0, 1] = np.nan

        median = median_depth_image([first, second, third, fourth, outlier])

        self.assertTrue(np.allclose(median, 200.0))
        self.assertTrue(np.allclose(
            median_depth_image([third, outlier]), 200.0
        ))

    def test_dense_cloud_crop_does_not_translate_view_geometry(self):
        colors = np.zeros((3, 3), dtype=np.uint8)
        anchor = np.array([
            [0.30, 0.00, 0.01],
            [0.31, 0.00, 0.04],
            [0.32, 0.00, -0.03],
        ])
        displaced = anchor + np.array([0.0, 0.0, 0.053])

        cropped = crop_dense_view_clouds(
            [(anchor, colors), (displaced, colors)],
            table_z_m=0.012,
            workspace_lo_m=(0.15, -0.25, 0.0),
            workspace_hi_m=(0.50, 0.25, 0.35),
        )

        # Cropping may remove below-table returns, but it must never snap one
        # view to another or alter their 53 mm relative transform.
        self.assertTrue(np.allclose(cropped[0][0], anchor[:2]))
        self.assertTrue(np.allclose(cropped[1][0], displaced))
        self.assertAlmostEqual(
            cropped[1][0][0, 2] - cropped[0][0][0, 2], 0.053
        )

    def test_dense_cloud_fusion_unions_complementary_returns(self):
        first = (np.array([[0.0, 0.0, 0.1]]),
                 np.array([[255, 0, 0]], dtype=np.uint8))
        second = (np.array([[0.02, 0.0, 0.1]]),
                  np.array([[0, 255, 0]], dtype=np.uint8))
        points, colors = fuse_dense_view_clouds(
            [first, second], voxel_m=0.001
        )
        self.assertEqual(len(points), 2)
        self.assertEqual(len(colors), 2)

    def test_dense_multiview_registration_removes_bounded_duplicate_layer(self):
        rng = np.random.default_rng(7)
        keys = np.unique(np.column_stack((
            rng.integers(50, 80, 900),
            rng.integers(-20, 20, 900),
            rng.integers(5, 28, 900),
        )), axis=0)[:500]
        anchor = keys.astype(float) * 0.004 + 0.0001
        transform_error = np.array([0.008, -0.004, 0.008])
        ghosted = anchor + transform_error
        colors = np.tile(np.array([[20, 180, 60]], dtype=np.uint8),
                         (len(anchor), 1))

        registered, reports = register_dense_view_clouds(
            [(anchor, colors), (ghosted, colors)],
            table_z_m=0.0, voxel_m=0.004, max_translation_m=0.012,
        )

        self.assertTrue(reports[1]["accepted"], reports[1])
        self.assertTrue(np.allclose(
            reports[1]["translation_mm"], [-8.0, 4.0, -8.0]
        ))
        self.assertTrue(np.allclose(registered[1][0], anchor, atol=1e-9))

    def test_registration_preserves_real_layers_inside_each_view(self):
        xy = np.array([[0.20 + 0.004 * index, 0.0]
                       for index in range(60)])
        lower = np.column_stack((xy, np.full(len(xy), 0.025)))
        upper = np.column_stack((xy, np.full(len(xy), 0.045)))
        anchor = np.vstack((lower, upper))
        ghosted = anchor + np.array([0.008, 0.0, 0.008])
        colors = np.zeros((len(anchor), 3), dtype=np.uint8)

        registered, reports = register_dense_view_clouds(
            [(anchor, colors), (ghosted, colors)],
            table_z_m=0.0, voxel_m=0.004, max_translation_m=0.012,
            minimum_overlap=0.05,
        )

        self.assertTrue(reports[1]["accepted"], reports[1])
        corrected_z = np.unique(np.round(registered[1][0][:, 2], 3))
        self.assertEqual(corrected_z.tolist(), [0.025, 0.045])

    def test_registration_keeps_already_aligned_view_for_fusion(self):
        rng = np.random.default_rng(11)
        keys = np.unique(np.column_stack((
            rng.integers(50, 80, 900),
            rng.integers(-20, 20, 900),
            rng.integers(5, 28, 900),
        )), axis=0)[:500]
        anchor = keys.astype(float) * 0.004 + 0.0001
        colors = np.zeros((len(anchor), 3), dtype=np.uint8)

        _registered, reports = register_dense_view_clouds(
            [(anchor, colors), (anchor.copy(), colors.copy())],
            table_z_m=0.0,
        )

        self.assertFalse(reports[1]["accepted"], reports[1])
        self.assertTrue(reports[1]["valid_for_fusion"], reports[1])

    def test_registration_adaptively_recovers_strong_boundary_match(self):
        keys = np.array([
            [x, y, z]
            for x in range(50, 70)
            for y in range(10)
            for z in range(5, 10)
        ], dtype=float)
        anchor = keys * 0.004 + 0.0001
        ghosted = anchor + np.array([0.016, 0.0, 0.0])
        colors = np.zeros((len(anchor), 3), dtype=np.uint8)

        registered, reports = register_dense_view_clouds(
            [(anchor, colors), (ghosted, colors.copy())],
            table_z_m=0.0, voxel_m=0.004, max_translation_m=0.012,
            adaptive_max_translation_m=0.020,
        )

        self.assertTrue(reports[1]["adaptive_search"], reports[1])
        self.assertTrue(reports[1]["accepted"], reports[1])
        self.assertFalse(reports[1]["pinned_to_bound"], reports[1])
        self.assertEqual(reports[1]["search_bound_mm"], 20.0)
        self.assertTrue(np.allclose(
            reports[1]["translation_mm"], [-16.0, 0.0, 0.0]
        ))
        self.assertTrue(np.allclose(registered[1][0], anchor, atol=1e-9))

    def test_registration_rejects_match_still_pinned_at_adaptive_bound(self):
        keys = np.array([
            [x, y, z]
            for x in range(50, 58)
            for y in range(10)
            for z in range(5, 10)
        ], dtype=float)
        anchor = keys * 0.004 + 0.0001
        ghosted = anchor + np.array([0.028, 0.0, 0.0])
        colors = np.zeros((len(anchor), 3), dtype=np.uint8)

        registered, reports = register_dense_view_clouds(
            [(anchor, colors), (ghosted, colors.copy())],
            table_z_m=0.0, voxel_m=0.004, max_translation_m=0.012,
            adaptive_max_translation_m=0.020,
        )

        self.assertTrue(reports[1]["adaptive_search"], reports[1])
        self.assertFalse(reports[1]["accepted"], reports[1])
        self.assertFalse(reports[1]["valid_for_fusion"], reports[1])
        self.assertTrue(reports[1]["pinned_to_bound"], reports[1])
        self.assertTrue(np.allclose(registered[1][0], ghosted))

    def test_registration_excludes_unrepairable_rotated_view_from_fusion(self):
        # A distinctive L-shaped scene has essentially no voxel agreement after
        # a wrist-angle error; bounded translation cannot repair that rotation.
        x = np.linspace(0.20, 0.46, 200)
        horizontal = np.column_stack((
            x, np.full_like(x, -0.10),
            0.04 + 0.02 * np.sin(np.arange(len(x)) * 0.3),
        ))
        y = np.linspace(-0.10, 0.16, 200)
        vertical = np.column_stack((
            np.full_like(y, 0.20), y,
            0.07 + 0.02 * np.cos(np.arange(len(y)) * 0.3),
        ))
        anchor = np.vstack((horizontal, vertical))
        angle = np.deg2rad(10.0)
        rotation = np.array([
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ])
        centre = np.mean(anchor, axis=0)
        rotated = (anchor - centre) @ rotation.T + centre
        colors = np.zeros((len(anchor), 3), dtype=np.uint8)

        registered, reports = register_dense_view_clouds(
            [(anchor, colors), (rotated, colors.copy())],
            table_z_m=0.0, voxel_m=0.004, max_translation_m=0.012,
            minimum_overlap=0.15,
        )

        self.assertFalse(reports[1]["accepted"], reports[1])
        self.assertFalse(reports[1]["valid_for_fusion"], reports[1])
        # The raw view is retained only so opt-in per-view diagnostics can show
        # why it was rejected. Callers must not include it in the fused cloud.
        self.assertTrue(np.allclose(registered[1][0], rotated))

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

    def test_installed_scene_rejects_collision_with_measured_capture_pose(self):
        from arm_agent.local_arm import LocalArm

        arm = LocalArm()
        arm.configure_collision_scene([{
            "center_m": [0.3015, 0.0, 0.218],
            "size_m": [0.06, 0.06, 0.06],
        }], table_z_m=0.01196)
        with patch.object(arm, "state", return_value={
            "joints_rad": [0.0] * 6,
        }):
            detail = arm.validate_current_collision_scene()

        self.assertFalse(detail["ok"])
        self.assertIn("measured capture pose", detail["reason"])
        self.assertEqual(detail["collision"]["kind"], "environment")


if __name__ == "__main__":
    unittest.main()
