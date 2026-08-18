"""Bounded full-orientation jaw alignment for arbitrarily posed objects.

Objects arrive in whatever pose they land in.  Yaw only rotates the jaw closing
direction inside the horizontal plane, so an object resting on an edge or
leaning presents its narrow axis where no yaw can reach it.  These tests pin
the behaviour that roll and pitch are searched as well, and that widening the
search never widens the aperture.
"""
from __future__ import annotations

import os
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import patch

import numpy as np

from arm_agent import wrist_servo as ws


def rotating_mapping(current_rpy, *, maximum_orientation_delta_deg=None):
    """Stand-in for the bolted-camera calibration.

    Mirrors the shipped analytic form M(R) = M_cal @ R_cal @ R^T with an
    identity calibration, so the mapping turns with the wrist exactly as the
    real one does.  A fixed mapping would make every candidate orientation look
    identical and hide the behaviour under test.
    """
    rotation = ws._rpy_rotation_matrix(
        tuple(float(value) for value in current_rpy)
    )
    return 1000.0 * rotation.T, {"reason": "unit test mapping"}


def box_for(axis: np.ndarray, *, narrow_mm: float = 17.8,
            other_mm: tuple[float, float] = (41.2, 78.7)) -> dict:
    """A measured OBB whose narrow axis points along ``axis``."""
    narrow = axis / np.linalg.norm(axis)
    helper = np.array([0.0, 0.0, 1.0])
    if abs(float(np.dot(helper, narrow))) > 0.9:
        helper = np.array([1.0, 0.0, 0.0])
    second = np.cross(narrow, helper)
    second /= np.linalg.norm(second)
    third = np.cross(narrow, second)
    observed = [narrow_mm, other_mm[0], other_mm[1]]
    return {
        # Ordered so the narrow measured extent belongs to ``narrow``.
        "axes_base": [narrow.tolist(), second.tolist(), third.tolist()],
        "observed_size_m": [value / 1000.0 for value in observed],
        "conservative_size_m": [(value + 6.0) / 1000.0 for value in observed],
    }


class OrientationRefinementTest(TestCase):
    def setUp(self) -> None:
        # Jaws closing along the tool Y axis, as the bench calibration has it,
        # so roll about the tool X axis genuinely re-aims them.
        self.detail_patch = patch.object(
            ws, "load_grasp_target_detail",
            return_value={"jaw_long_axis_camera_deg": 0.0},
        )
        self.mapping_patch = patch.object(
            ws, "load_saved_camera_jacobian", side_effect=rotating_mapping,
        )
        self.detail_patch.start()
        self.mapping_patch.start()
        self.addCleanup(self.detail_patch.stop)
        self.addCleanup(self.mapping_patch.stop)
        self.env = patch.dict(os.environ, {
            "REBOT_WRIST_JAW_LONG_AXIS_CAMERA_DEG": "0.0",
        })
        self.env.start()
        self.addCleanup(self.env.stop)

    def tilted_axis(self, roll_deg: float, pitch_deg: float,
                    yaw_deg: float) -> np.ndarray:
        rotation = ws._rpy_rotation_matrix(
            tuple(np.radians([roll_deg, pitch_deg, yaw_deg]))
        )
        return rotation @ np.array([0.0, 1.0, 0.0])

    def test_roll_is_disabled_by_default(self):
        """Rolling before the scene map is captured destroys that map.

        The discovery clouds are all captured with a level wrist. A large roll
        leaves the collision map's anchor view sharing almost no occupancy with
        them, so every discovery view is dropped and the map is rebuilt from a
        single viewpoint — removing the protection the rotation most needs.
        """
        axis = self.tilted_axis(35.0, -10.0, 8.0)
        result = ws._find_3d_axis_orientation_refinement(
            box_for(axis), (0.0, 0.0, 0.0), 50.0
        )
        self.assertFalse(result["ok"], result)
        self.assertIn("0deg roll", result["reason"])

    def test_stable_visual_subsection_can_replace_only_unsnapped_obb_attitude(self):
        body = SimpleNamespace(min_width_mm=16.8)
        geometry = SimpleNamespace(grasp_detection=body)
        consensus = {
            "valid_full_geometry_frames": 2,
            "width_spread_mm": 0.2,
        }
        jaw_plan = {"max_aperture_mm": 50.0}
        box = {"orientation_mode": "measured_3d_pca"}
        projection = {"observed_width_mm": 50.4}

        result = ws._stable_visual_subsection_fallback(
            geometry, consensus, jaw_plan, box, projection, {"ok": True},
            independent_metric_crosscheck=True,
        )

        self.assertTrue(result["ok"], result)
        self.assertEqual(result["width_mm"], 16.8)

    def test_visual_subsection_never_overrides_gravity_validated_obb(self):
        geometry = SimpleNamespace(
            grasp_detection=SimpleNamespace(min_width_mm=16.8)
        )
        result = ws._stable_visual_subsection_fallback(
            geometry,
            {"valid_full_geometry_frames": 2, "width_spread_mm": 0.2},
            {"max_aperture_mm": 50.0},
            {"orientation_mode": "gravity_snapped_tabletop"},
            {"observed_width_mm": 50.4},
            {"ok": True},
            independent_metric_crosscheck=True,
        )

        self.assertFalse(result["ok"], result)

    def test_visual_subsection_never_relaxes_aperture(self):
        geometry = SimpleNamespace(
            grasp_detection=SimpleNamespace(min_width_mm=48.0)
        )
        result = ws._stable_visual_subsection_fallback(
            geometry,
            {"valid_full_geometry_frames": 2, "width_spread_mm": 0.2},
            {"max_aperture_mm": 50.0},
            {"orientation_mode": "measured_3d_pca"},
            {"observed_width_mm": 80.0},
            {"ok": True},
            independent_metric_crosscheck=True,
        )

        self.assertFalse(result["ok"], result)
        self.assertIn("jaw limit", result["reason"])

    def test_edge_resting_object_is_reachable_by_roll_and_pitch(self):
        """The narrow axis of an on-edge object is out of the yaw plane.

        Roll is opt-in; this pins the capability itself, not the default.
        """
        axis = self.tilted_axis(35.0, -10.0, 8.0)
        self.assertGreater(abs(float(axis[2])), 0.5,
                           "test axis should be well out of the horizontal plane")
        box = box_for(axis)

        yaw_only = ws._find_3d_axis_yaw_refinement(box, (0.0, 0.0, 0.0), 50.0)
        self.assertFalse(yaw_only["ok"], yaw_only)

        with patch.dict(os.environ, {
            "REBOT_3D_AXIS_REFINE_MAX_ROLL_DEG": "45",
        }):
            result = ws._find_3d_axis_orientation_refinement(
                box, (0.0, 0.0, 0.0), 50.0
            )
        self.assertTrue(result["ok"], result)
        self.assertEqual(result["refinement_mode"], "bounded_full_orientation")
        self.assertLessEqual(result["axis_angle_deg"], 15.0)
        # The jaws really do end up pointing along the measured narrow axis.
        closing = ws._rpy_rotation_matrix(result["target_rpy"]) @ np.array(
            [0.0, 1.0, 0.0]
        )
        angle = np.degrees(np.arccos(
            np.clip(abs(float(np.dot(closing, axis))), 0.0, 1.0)
        ))
        self.assertLessEqual(float(angle), 15.0)

    def test_yaw_only_solution_is_preferred_when_one_exists(self):
        """Reaching for roll/pitch must not cost extra motion needlessly."""
        axis = self.tilted_axis(0.0, 0.0, 6.0)
        result = ws._find_3d_axis_orientation_refinement(
            box_for(axis), (0.0, 0.0, 0.0), 50.0
        )
        self.assertTrue(result["ok"], result)
        self.assertEqual(result["refinement_mode"], "yaw_only")

    def test_orientation_search_never_relaxes_the_aperture(self):
        """Every measured axis is wider than the jaws: no pose can help."""
        axis = self.tilted_axis(35.0, -10.0, 8.0)
        box = box_for(axis, narrow_mm=64.0, other_mm=(70.0, 78.7))
        result = ws._find_3d_axis_orientation_refinement(
            box, (0.0, 0.0, 0.0), 50.0
        )
        self.assertFalse(result["ok"], result)
        self.assertIn("aperture", result["reason"])

    def test_bounds_are_respected(self):
        """A pose outside the configured roll budget is not proposed."""
        axis = self.tilted_axis(35.0, -10.0, 8.0)
        with patch.dict(os.environ, {
            "REBOT_3D_AXIS_REFINE_MAX_ROLL_DEG": "5",
            "REBOT_3D_AXIS_REFINE_MAX_PITCH_DEG": "5",
        }):
            result = ws._find_3d_axis_orientation_refinement(
                box_for(axis), (0.0, 0.0, 0.0), 50.0
            )
        self.assertFalse(result["ok"], result)


class OrientationTrajectoryBudgetTest(TestCase):
    """An aligning pose is worthless if the executor refuses to reach it."""

    def setUp(self) -> None:
        self.ik_calls: list[tuple] = []
        self.clearance_calls: list[tuple] = []

        def check_ik(*args):
            self.ik_calls.append(args)
            return SimpleNamespace(ok=True, reason="ik_ok")

        def check_pose_clearance(*args):
            self.clearance_calls.append(args)
            return SimpleNamespace(ok=True, reason="clear")

        for name, value in (
            ("ee_pos", lambda: (0.26, 0.0, 0.25)),
            ("ee_rpy", lambda: (0.0, 0.0, 0.0)),
            ("check_ik", check_ik),
            ("check_pose_clearance", check_pose_clearance),
        ):
            patcher = patch.object(ws.A, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        clamp = patch.object(ws, "GEOMETRY_MAX_YAW_DELTA_DEG", 20.0)
        clamp.start()
        self.addCleanup(clamp.stop)

    def target(self, roll_deg: float) -> tuple[float, float, float]:
        return (float(np.radians(roll_deg)), 0.0, 0.0)

    def test_default_budget_still_refuses_a_large_rotation(self):
        path, reason = ws._plan_rpy_trajectory(self.target(28.0))
        self.assertIsNone(path)
        self.assertIn("20.0deg clamp", reason)
        self.assertEqual(self.clearance_calls, [])

    def test_extended_budget_admits_it_and_checks_collisions(self):
        path, reason = ws._plan_rpy_trajectory(
            self.target(28.0), max_delta_deg=45.0
        )
        self.assertIsNotNone(path, reason)
        self.assertIn("IK/collision", reason)
        # The wider envelope is validated against the collision map, not IK.
        self.assertTrue(self.clearance_calls)
        self.assertEqual(self.ik_calls, [])

    def test_extended_budget_is_still_a_bound(self):
        path, reason = ws._plan_rpy_trajectory(
            self.target(40.0), max_delta_deg=30.0
        )
        self.assertIsNone(path)
        self.assertIn("30.0deg clamp", reason)

    def test_extended_budget_fails_closed_on_collision(self):
        with patch.object(ws.A, "check_pose_clearance", lambda *args:
                          SimpleNamespace(ok=False, reason="table_collision")):
            path, reason = ws._plan_rpy_trajectory(
                self.target(28.0), max_delta_deg=45.0
            )
        self.assertIsNone(path)
        self.assertIn("table_collision", reason)


class ActiveCameraEnvelopeTest(TestCase):
    """A validated large wrist pose must be interpretable, but only deliberately."""

    def setUp(self) -> None:
        self.addCleanup(ws._set_active_camera_envelope, None)
        # ~30.8deg from the calibrated tool orientation: past the default
        # envelope, inside the explicitly requestable one.
        self.far_rpy = tuple(np.radians([28.0, 8.0, 12.0]))

    def test_default_envelope_refuses_a_far_pose(self):
        ws._set_active_camera_envelope(None)
        mapping, detail = ws.load_saved_camera_jacobian(self.far_rpy)
        self.assertIsNone(mapping)
        self.assertIn("maximum", detail["reason"])

    def test_active_envelope_admits_it(self):
        ws._set_active_camera_envelope(35.0)
        mapping, detail = ws.load_saved_camera_jacobian(self.far_rpy)
        self.assertIsNotNone(mapping, detail)

    def test_clearing_restores_the_default(self):
        ws._set_active_camera_envelope(35.0)
        ws._set_active_camera_envelope(None)
        mapping, _ = ws.load_saved_camera_jacobian(self.far_rpy)
        self.assertIsNone(mapping)

    def test_explicit_argument_still_wins(self):
        ws._set_active_camera_envelope(45.0)
        mapping, detail = ws.load_saved_camera_jacobian(
            self.far_rpy, maximum_orientation_delta_deg=10.0
        )
        self.assertIsNone(mapping, detail)

    def test_result_clears_the_envelope(self):
        ws._set_active_camera_envelope(40.0)
        ws._result(False, "any exit", [])
        self.assertIsNone(ws._ACTIVE_CAMERA_ENVELOPE_DEG)
