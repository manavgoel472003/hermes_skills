from __future__ import annotations

from unittest import TestCase

import numpy as np

from arm_agent.scene_collision import ObstacleBox, SceneMap
from arm_agent.voxel_navigation import plan_topdown_route
from arm_agent import wrist_servo as ws


def _scene(*boxes: ObstacleBox) -> SceneMap:
    return SceneMap(
        boxes=tuple(boxes), table_z_m=0.02, voxel_m=0.01,
        inflation_m=0.003, valid_depth_points=100,
        mapped_obstacle_points=50, excluded_target_points=10,
        closest_obstacle_m=0.1,
    )


def _plan(scene: SceneMap, start=(0.25, 0.0, 0.15), goal=(0.45, 0.0)):
    return plan_topdown_route(
        scene, np.asarray(start, dtype=float), np.asarray(goal, dtype=float),
        workspace_lo_m=(0.15, -0.20, 0.0),
        workspace_hi_m=(0.50, 0.20, 0.30),
        workspace_margin_m=0.01, ee_radius_m=0.02,
        vertical_clearance_m=0.02, drop_clearance_m=0.03,
        resolution_m=0.01,
    )


class VoxelNavigationTest(TestCase):
    def test_payload_scale_sets_bounded_minimum_carry_height(self):
        floor, detail = ws._payload_minimum_travel_height_m(
            {"size_m": [0.082, 0.045, 0.021]}, 0.20, 0.20,
        )
        self.assertIsNotNone(floor, detail)
        self.assertAlmostEqual(floor, 0.241, places=4)
        self.assertAlmostEqual(detail["extra_lift_m"], 0.041, places=4)

    def test_measured_payload_extends_required_clearance_below_tool(self):
        context = {
            "payload_box": {
                "full_size_m": [0.021, 0.017, 0.101],
                "full_axes_ee": np.eye(3).T.tolist(),
                "center_offset_ee_m": [-0.007, -0.006, -0.041],
            }
        }
        envelope = ws._voxel_payload_envelope(context, (0.0, 0.0, 0.0))
        self.assertTrue(envelope["ok"], envelope)
        self.assertAlmostEqual(envelope["downward_extent_m"], 0.0915, places=4)
        self.assertGreater(envelope["radial_extent_m"], 0.02)

    def test_missing_payload_geometry_fails_closed(self):
        envelope = ws._voxel_payload_envelope({}, (0.0, 0.0, 0.0))
        self.assertFalse(envelope["ok"])
        self.assertIn("unavailable", envelope["reason"])

    def test_routes_around_column_at_current_height(self):
        wall = ObstacleBox((0.35, 0.0, 0.08), (0.03, 0.12, 0.12))
        route, detail = _plan(_scene(wall))
        self.assertIsNotNone(route, detail)
        assert route is not None
        self.assertTrue(any(abs(point[1]) > 0.05 for point in route.anchors_m))
        self.assertAlmostEqual(route.detail["travel_z_m"], 0.15, places=3)

    def test_uses_direct_high_layer_over_low_column(self):
        low = ObstacleBox((0.35, 0.0, 0.05), (0.03, 0.12, 0.06))
        route, detail = _plan(_scene(low), start=(0.25, 0.0, 0.20))
        self.assertIsNotNone(route, detail)
        assert route is not None
        self.assertLessEqual(route.detail["path_turns"], 1)
        self.assertAlmostEqual(route.detail["travel_z_m"], 0.20, places=3)

    def test_release_height_tracks_destination_column_without_support_claim(self):
        support = ObstacleBox((0.45, 0.0, 0.07), (0.04, 0.04, 0.10))
        route, detail = _plan(_scene(support), start=(0.25, 0.0, 0.20))
        self.assertIsNotNone(route, detail)
        assert route is not None
        # Box top is 0.12 m; release is the configured 30 mm above it.
        self.assertAlmostEqual(route.release_endpoint_m[2], 0.15, places=3)
        self.assertAlmostEqual(
            route.detail["destination_column_top_m"], 0.12, places=3
        )

    def test_measured_stack_top_floor_beats_a_coarse_lower_collision_box(self):
        # Collision geometry is intentionally coarse.  A compact RGB-D mask
        # can see a stack reference's top 10 mm higher, and the payload must
        # not graze that surface merely because the collision box is shorter.
        coarse_reference = ObstacleBox(
            (0.45, 0.0, 0.075), (0.04, 0.04, 0.10)
        )
        route, detail = plan_topdown_route(
            _scene(coarse_reference), np.asarray((0.25, 0.0, 0.20)),
            np.asarray((0.45, 0.0)),
            workspace_lo_m=(0.15, -0.20, 0.0),
            workspace_hi_m=(0.50, 0.20, 0.30),
            workspace_margin_m=0.01, ee_radius_m=0.02,
            vertical_clearance_m=0.02, drop_clearance_m=0.03,
            minimum_destination_top_m=0.135,
            resolution_m=0.01,
        )
        self.assertIsNotNone(route, detail)
        assert route is not None
        self.assertAlmostEqual(route.detail["destination_column_top_m"], 0.135)
        self.assertAlmostEqual(
            route.detail["measured_destination_top_floor_m"], 0.135
        )
        self.assertAlmostEqual(route.release_endpoint_m[2], 0.165)

    def test_stack_samples_height_at_payload_center_not_ee_xy(self):
        reference = ObstacleBox((0.45, 0.0, 0.07), (0.02, 0.02, 0.10))
        route, detail = plan_topdown_route(
            _scene(reference), np.asarray((0.25, 0.0, 0.20)),
            np.asarray((0.40, 0.0)),
            workspace_lo_m=(0.15, -0.20, 0.0),
            workspace_hi_m=(0.50, 0.20, 0.30),
            workspace_margin_m=0.01, ee_radius_m=0.005,
            destination_radius_m=0.005,
            destination_height_xy_m=np.asarray((0.45, 0.0)),
            vertical_clearance_m=0.02, drop_clearance_m=0.03,
            resolution_m=0.01,
        )

        self.assertIsNotNone(route, detail)
        assert route is not None
        self.assertAlmostEqual(route.release_endpoint_m[0], 0.40, places=3)
        self.assertAlmostEqual(route.release_endpoint_m[2], 0.15, places=3)
        self.assertEqual(
            route.detail["destination_height_xy_m"], [0.45, 0.0]
        )

    def test_long_payload_clearance_raises_release_above_destination(self):
        destination = ObstacleBox(
            (0.45, 0.0, 0.111), (0.04, 0.04, 0.100)
        )
        route, detail = plan_topdown_route(
            _scene(destination), np.asarray((0.25, 0.0, 0.202)),
            np.asarray((0.45, 0.0)),
            workspace_lo_m=(0.15, -0.20, 0.0),
            workspace_hi_m=(0.50, 0.20, 0.30),
            workspace_margin_m=0.01, ee_radius_m=0.025,
            vertical_clearance_m=0.0965,
            drop_clearance_m=0.0995, resolution_m=0.02,
        )
        self.assertIsNotNone(route, detail)
        assert route is not None
        self.assertAlmostEqual(
            route.detail["destination_column_top_m"], 0.161, places=3
        )
        self.assertAlmostEqual(route.release_endpoint_m[2], 0.2605, places=3)

    def test_high_destination_rises_before_xy_for_payload_clearance(self):
        corridor_obstacle = ObstacleBox(
            (0.35, 0.0, 0.10), (0.03, 0.12, 0.16)
        )
        destination = ObstacleBox(
            (0.45, 0.0, 0.11), (0.04, 0.04, 0.18)
        )
        route, detail = _plan(
            _scene(corridor_obstacle, destination),
            start=(0.25, 0.0, 0.15), goal=(0.45, 0.0),
        )
        self.assertIsNotNone(route, detail)
        assert route is not None
        self.assertAlmostEqual(route.detail["travel_z_m"], 0.23, places=3)
        self.assertEqual(
            route.detail["travel_strategy"],
            "rise_before_xy_then_voxel_astar",
        )
        self.assertTrue(route.detail["rose_before_xy"])
        # The first commanded anchor is a vertical lift at the current XY.
        self.assertEqual(route.anchors_m[0][:2], (0.25, 0.0))
        self.assertAlmostEqual(route.anchors_m[0][2], 0.23, places=3)
        self.assertGreater(route.release_endpoint_m[2], 0.20)

    def test_payload_travel_floor_forces_lift_before_any_xy_motion(self):
        route, detail = plan_topdown_route(
            _scene(), np.asarray((0.25, 0.0, 0.15)),
            np.asarray((0.45, 0.0)),
            workspace_lo_m=(0.15, -0.20, 0.0),
            workspace_hi_m=(0.50, 0.20, 0.30),
            workspace_margin_m=0.01, ee_radius_m=0.02,
            minimum_travel_z_m=0.23,
            vertical_clearance_m=0.02, drop_clearance_m=0.03,
            resolution_m=0.01,
        )
        self.assertIsNotNone(route, detail)
        assert route is not None
        self.assertEqual(route.anchors_m[0][:2], (0.25, 0.0))
        self.assertAlmostEqual(route.anchors_m[0][2], 0.23, places=3)
        self.assertTrue(route.detail["rose_before_xy"])
        self.assertTrue(all(
            attempt["travel_z_m"] >= 0.23
            for attempt in route.detail["attempted_layers"]
        ))

    def test_payload_travel_floor_above_workspace_is_refused(self):
        route, detail = plan_topdown_route(
            _scene(), np.asarray((0.25, 0.0, 0.15)),
            np.asarray((0.45, 0.0)),
            workspace_lo_m=(0.15, -0.20, 0.0),
            workspace_hi_m=(0.50, 0.20, 0.30),
            workspace_margin_m=0.01, ee_radius_m=0.02,
            minimum_travel_z_m=0.31,
            vertical_clearance_m=0.02, drop_clearance_m=0.03,
            resolution_m=0.01,
        )
        self.assertIsNone(route)
        self.assertIn("exceeds", detail["reason"])

    def test_tall_column_still_causes_xy_detour_at_preferred_height(self):
        tall = ObstacleBox(
            (0.35, 0.0, 0.16), (0.03, 0.12, 0.28)
        )
        route, detail = _plan(
            _scene(tall), start=(0.25, 0.0, 0.15), goal=(0.45, 0.0),
        )
        self.assertIsNotNone(route, detail)
        assert route is not None
        self.assertTrue(any(
            abs(point[1]) > 0.05 for point in route.anchors_m
        ))

    def test_route_footprint_does_not_inflate_destination_height(self):
        nearby_tall = ObstacleBox(
            (0.45, 0.10, 0.15), (0.03, 0.03, 0.26)
        )
        route, detail = plan_topdown_route(
            _scene(nearby_tall), np.asarray((0.25, 0.0, 0.20)),
            np.asarray((0.45, 0.0)),
            workspace_lo_m=(0.15, -0.20, 0.0),
            workspace_hi_m=(0.50, 0.20, 0.35),
            workspace_margin_m=0.01,
            ee_radius_m=0.085,
            destination_radius_m=0.025,
            vertical_clearance_m=0.02,
            drop_clearance_m=0.03,
            resolution_m=0.02,
        )
        self.assertIsNotNone(route, detail)
        assert route is not None
        self.assertAlmostEqual(
            route.detail["destination_column_top_m"], 0.02, places=3
        )
        self.assertAlmostEqual(route.release_endpoint_m[2], 0.05, places=3)

    def test_refuses_destination_outside_workspace(self):
        route, detail = _plan(_scene(), goal=(0.60, 0.0))
        self.assertIsNone(route)
        self.assertIn("outside", detail["reason"])
