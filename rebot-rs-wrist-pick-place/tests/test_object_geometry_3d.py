from __future__ import annotations

import unittest

import numpy as np

from arm_agent import object_geometry_3d as og
from arm_agent import wrist_servo as ws


class ObjectGeometry3DTest(unittest.TestCase):
    def test_masked_points_cannot_escape_cleaned_detection_box(self):
        depth = np.full((30, 40), 200.0, dtype=np.float32)
        mask = np.full((30, 40), 255, dtype=np.uint8)
        detection = type("Detection", (), {
            "box": (10, 8, 30, 23),
            "info": {"mask_u8": mask},
        })()
        points = og.masked_points_base(
            (np.zeros((30, 40, 3), dtype=np.uint8), depth,
             (200.0, 200.0, 20.0, 15.0)),
            detection,
            ee_xyz_m=(0.3, 0.0, 0.2),
            camera_mapping_mm_per_m=np.eye(3) * 1000.0,
            tool_camera_mm=(0.0, 0.0, 70.0),
        )

        # One-pixel erosion leaves only the 18 x 13 interior of the box.
        self.assertEqual(len(points), 18 * 13)

    def test_legacy_box_fallback_keeps_only_seeded_depth_component(self):
        depth = np.full((40, 60), 500.0, dtype=np.float32)
        # A narrow target and a disconnected background patch both lie inside
        # the same legacy bounding box.
        depth[8:32, 18:25] = 210.0
        depth[8:32, 35:50] = 280.0
        detection = type("Detection", (), {
            "box": (15, 5, 52, 35),
            "uv": (21.0, 20.0),
            "info": {},
        })()

        points = og.masked_points_base(
            (np.zeros((40, 60, 3), dtype=np.uint8), depth,
             (200.0, 200.0, 30.0, 20.0)),
            detection,
            ee_xyz_m=(0.3, 0.0, 0.2),
            camera_mapping_mm_per_m=np.eye(3) * 1000.0,
            tool_camera_mm=(0.0, 0.0, 70.0),
        )

        # One-pixel erosion of the 7 x 24 target leaves 5 x 22 pixels. The
        # larger 280 mm patch must not leak into the object's metric box.
        self.assertEqual(len(points), 5 * 22)

    def test_extent_consensus_rejects_one_oversized_mask_view(self):
        rng = np.random.default_rng(55)
        good_a = rng.uniform(
            [0.30, -0.01, 0.02], [0.40, 0.01, 0.04], size=(1200, 3)
        )
        good_b = rng.uniform(
            [0.3005, -0.0102, 0.0201],
            [0.4002, 0.0101, 0.0402], size=(1200, 3),
        )
        contaminated = rng.uniform(
            [0.25, -0.06, 0.01], [0.48, 0.08, 0.12], size=(1800, 3)
        )

        indices, detail = og.consistent_view_extent_indices(
            [good_a, contaminated, good_b], table_z_m=0.012
        )

        self.assertTrue(detail["ok"], detail)
        self.assertEqual(indices, [0, 2])

    def test_tabletop_box_uses_support_plane_for_hidden_height(self):
        # Two noisy RGB-D frames see the top and sides of a 100 x 40 x 30 mm
        # object standing on the calibrated z=12 mm table.
        xs = np.linspace(0.30, 0.40, 31)
        ys = np.linspace(-0.02, 0.02, 17)
        xx, yy = np.meshgrid(xs, ys)
        top = np.column_stack((xx.ravel(), yy.ravel(),
                               np.full(xx.size, 0.042)))
        side = np.column_stack((
            np.full(80, 0.30),
            np.linspace(-0.02, 0.02, 80),
            np.linspace(0.012, 0.042, 80),
        ))
        points = np.vstack((top, side))

        box = og.fit_tabletop_box(
            [points, points + np.array([0.0002, -0.0001, 0.0001])],
            table_z_m=0.012,
        )

        dimensions = sorted(box.observed_size_m)
        self.assertAlmostEqual(dimensions[0], 0.030, delta=0.004)
        self.assertAlmostEqual(dimensions[1], 0.04, delta=0.006)
        self.assertAlmostEqual(dimensions[2], 0.10, delta=0.008)
        self.assertTrue(np.all(
            np.asarray(box.conservative_size_m)
            >= np.asarray(box.observed_size_m)
        ))
        self.assertEqual(box.views, 2)
        self.assertFalse(box.support_completed)

    def test_near_vertical_measured_axis_snaps_to_gravity(self):
        rng = np.random.default_rng(31)
        local = rng.uniform(
            [-0.04, -0.015, -0.06], [0.04, 0.015, 0.06], size=(3000, 3)
        )
        angle = np.radians(14.0)
        rotation_y = np.array([
            [np.cos(angle), 0.0, np.sin(angle)],
            [0.0, 1.0, 0.0],
            [-np.sin(angle), 0.0, np.cos(angle)],
        ])
        points = local @ rotation_y.T + np.array([0.36, 0.02, 0.12])

        box = og.fit_tabletop_box([points], table_z_m=0.012)

        axes = np.asarray(box.axes_base)
        self.assertTrue(np.any(np.isclose(np.abs(axes[:, 2]), 1.0)))
        self.assertTrue(box.orientation_mode.startswith("gravity_snapped"))

    def test_axis_outside_gravity_tolerance_remains_measured(self):
        rng = np.random.default_rng(32)
        local = rng.uniform(
            [-0.04, -0.015, -0.06], [0.04, 0.015, 0.06], size=(3000, 3)
        )
        angle = np.radians(29.0)
        rotation_y = np.array([
            [np.cos(angle), 0.0, np.sin(angle)],
            [0.0, 1.0, 0.0],
            [-np.sin(angle), 0.0, np.cos(angle)],
        ])
        points = local @ rotation_y.T + np.array([0.36, 0.02, 0.15])

        box = og.fit_tabletop_box([points], table_z_m=0.012)

        self.assertEqual(box.orientation_mode, "measured_3d_pca")

    def test_trusted_axes_refit_extents_without_pca_shear(self):
        rng = np.random.default_rng(33)
        first = rng.uniform(
            [-0.04, -0.02, -0.01], [0.04, 0.02, 0.01], size=(1800, 3)
        ) + np.array([0.40, 0.03, 0.04])
        # Registration error shears the aggregate enough to tempt a fresh PCA
        # axis, but the trusted frame must remain gravity aligned.
        second = first + np.array([0.008, -0.004, 0.008])
        axes = np.eye(3)

        box = og.fit_tabletop_box_with_axes(
            [first, second], table_z_m=0.012, axes_base=axes,
            orientation_mode="test_consensus_lock",
        )

        np.testing.assert_allclose(np.asarray(box.axes_base).T, axes)
        self.assertEqual(box.orientation_mode, "test_consensus_lock")
        self.assertGreater(box.observed_size_m[0], 0.07)
        self.assertLess(box.observed_size_m[2], 0.04)

    def test_measured_box_does_not_extend_an_elevated_object_to_table(self):
        rng = np.random.default_rng(12)
        points = rng.uniform(
            [0.30, -0.02, 0.140], [0.38, 0.02, 0.170], size=(2000, 3)
        )

        box = og.fit_tabletop_box([points], table_z_m=0.012)

        self.assertAlmostEqual(box.center_m[2], 0.155, delta=0.004)
        self.assertAlmostEqual(
            sorted(box.observed_size_m)[0], 0.030, delta=0.005
        )
        self.assertLess(max(box.observed_size_m), 0.09)
        self.assertEqual(box.source, "multiframe_rgbd_measured_obb")

    def test_shape_prior_can_expand_but_never_shrink_metric_extent(self):
        observed = np.array([0.10, 0.04, 0.03])
        conservative = observed + 0.006
        rng = np.random.default_rng(4)
        vertices = rng.uniform(
            [-0.5, -0.30, -0.20], [0.5, 0.30, 0.20], size=(500, 3)
        )
        expanded, detail = og.expand_with_shape_prior(
            observed, conservative, vertices
        )
        self.assertTrue(detail["used"])
        self.assertTrue(np.all(expanded >= conservative))

    def test_support_release_places_rotated_box_above_plane(self):
        context = {
            "payload_box": {
                "center_offset_ee_m": [0.0, 0.0, -0.020],
                "full_axes_ee": np.eye(3).tolist(),
                "full_size_m": [0.10, 0.04, 0.03],
            }
        }
        endpoint, detail = ws._support_aware_release_pose(
            [0.35, -0.10], context, (0.0, 0.0, 0.0),
            support_z_m=0.012, release_gap_m=0.010,
        )
        self.assertIsNotNone(endpoint)
        # Object centre is at 12 + 15 + 10 = 37 mm. Its centre is 20 mm
        # below the EE, so the required EE height is 57 mm.
        self.assertTrue(np.allclose(endpoint, [0.35, -0.10, 0.057]))
        self.assertEqual(detail["desired_object_center_m"],
                         [0.35, -0.1, 0.037])


if __name__ == "__main__":
    unittest.main()
