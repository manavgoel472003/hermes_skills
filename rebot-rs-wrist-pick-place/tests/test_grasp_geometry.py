from __future__ import annotations

from unittest import TestCase
from unittest.mock import patch

import numpy as np

from arm_agent import grasp_geometry as gg
from arm_agent.wrist_detect import Detection


def detection(*, source: str, width: float, length: float, area: float,
              z: float = 220.0, axis: float = 0.0,
              box=(100, 100, 200, 200), uv=(150.0, 150.0)) -> Detection:
    return Detection(
        box=box, uv=uv, z_mm=z, p_cam_mm=(0.0, 0.0, z),
        source=source, area_px=area, min_width_mm=width,
        max_width_mm=length, info={"long_axis_deg": axis},
    )


class GeometryGraspPlannerTest(TestCase):
    def setUp(self):
        self.color = np.zeros((300, 300, 3), dtype=np.uint8)
        self.depth = np.full((300, 300), 300.0, dtype=np.float32)
        self.intr = (400.0, 400.0, 150.0, 150.0)

    def test_small_colored_feature_uses_graspable_full_envelope(self):
        cap = detection(source="color:green", width=18, length=20, area=500,
                        axis=2)
        bottle = detection(source="depth:object", width=42, length=70,
                           area=1800, axis=7)
        with patch.object(gg, "segment_target_envelope",
                          return_value=bottle):
            plan = gg.plan_grasp_geometry(
                self.color, self.depth, self.intr, cap,
                current_rpy=(0.0, 1.0, 0.0), max_aperture_mm=50.0,
                jaw_long_axis_camera_deg=5.0,
            )

        self.assertTrue(plan.ok, plan.reason)
        self.assertTrue(plan.used_full_envelope)
        self.assertIs(plan.grasp_detection, bottle)
        self.assertAlmostEqual(plan.yaw_correction_deg, 2.0)

    def test_full_envelope_uses_local_body_width_not_wide_cap(self):
        cap = detection(source="color:blue", width=28, length=32, area=500,
                        axis=0)
        marker = detection(source="depth:target_envelope", width=29, length=70,
                           area=1800, axis=0)
        marker.info["local_grasp_width_mm"] = 16.0
        with patch.object(gg, "segment_target_envelope", return_value=marker):
            plan = gg.plan_grasp_geometry(
                self.color, self.depth, self.intr, cap,
                current_rpy=(0.0, 0.0, 0.0), max_aperture_mm=50.0,
                jaw_long_axis_camera_deg=0.0,
            )

        self.assertTrue(plan.ok, plan.reason)
        self.assertIsNotNone(plan.grasp_detection)
        self.assertAlmostEqual(plan.grasp_detection.min_width_mm, 16.0)
        self.assertIn("local body cross-section", plan.reason)

    def test_oversized_body_refuses_non_prominent_cap(self):
        cap = detection(source="color:green", width=18, length=20, area=400,
                        z=220)
        bottle = detection(source="depth:object", width=65, length=90,
                           area=2000, z=224)
        with patch.object(gg, "segment_target_envelope",
                          return_value=bottle):
            plan = gg.plan_grasp_geometry(
                self.color, self.depth, self.intr, cap,
                current_rpy=(0.0, 1.0, 0.0), max_aperture_mm=50.0,
                jaw_long_axis_camera_deg=0.0,
            )

        self.assertFalse(plan.ok)
        self.assertIn("does not fit", plan.reason)

    def test_large_orientation_change_is_refused(self):
        obj = detection(source="depth:object", width=18, length=70,
                        area=1500, axis=35)
        with patch.object(gg, "segment_target_envelope",
                          return_value=obj):
            plan = gg.plan_grasp_geometry(
                self.color, self.depth, self.intr, obj,
                current_rpy=(0.0, 1.0, 0.0), max_aperture_mm=50.0,
                jaw_long_axis_camera_deg=0.0, max_yaw_delta_deg=10.0,
            )

        self.assertFalse(plan.ok)
        self.assertIn("beyond the clamped", plan.reason)

    def test_boundary_vision_noise_is_clamped_to_physical_yaw_limit(self):
        obj = detection(source="depth:object", width=28, length=40,
                        area=1500, axis=79.9)
        with patch.object(gg, "segment_target_envelope",
                          return_value=obj):
            plan = gg.plan_grasp_geometry(
                self.color, self.depth, self.intr, obj,
                current_rpy=(0.0, 0.0, 0.0), max_aperture_mm=50.0,
                jaw_long_axis_camera_deg=-90.0,
                max_yaw_delta_deg=10.0, yaw_measurement_slop_deg=0.5,
            )

        self.assertTrue(plan.ok, plan.reason)
        self.assertAlmostEqual(plan.yaw_correction_deg, -10.0)
        self.assertIn("physical clamp", plan.reason)

    def test_yaw_beyond_measurement_slop_is_refused(self):
        obj = detection(source="depth:object", width=28, length=40,
                        area=1500, axis=79.4)
        with patch.object(gg, "segment_target_envelope",
                          return_value=obj):
            plan = gg.plan_grasp_geometry(
                self.color, self.depth, self.intr, obj,
                current_rpy=(0.0, 0.0, 0.0), max_aperture_mm=50.0,
                jaw_long_axis_camera_deg=-90.0,
                max_yaw_delta_deg=10.0, yaw_measurement_slop_deg=0.5,
            )

        self.assertFalse(plan.ok)
        self.assertIn("vision uncertainty", plan.reason)

    def test_prominent_feature_can_be_grasped_above_oversized_body(self):
        handle = detection(source="color:green", width=18, length=20,
                           area=400, z=190)
        body = detection(source="depth:object", width=65, length=90,
                         area=2000, z=225)
        with patch.object(gg, "segment_target_envelope",
                          return_value=body):
            plan = gg.plan_grasp_geometry(
                self.color, self.depth, self.intr, handle,
                current_rpy=(0.0, 1.0, 0.0), max_aperture_mm=50.0,
                jaw_long_axis_camera_deg=0.0,
                min_feature_prominence_mm=25.0,
            )

        self.assertTrue(plan.ok, plan.reason)
        self.assertFalse(plan.used_full_envelope)
        self.assertIs(plan.grasp_detection, handle)

    def test_feature_only_detection_without_depth_envelope_is_refused(self):
        cap = detection(source="color:green", width=18, length=20, area=400)
        with (
            patch.object(gg, "segment_target_envelope", return_value=None),
            patch.object(gg, "segment_objects", return_value=[]),
        ):
            plan = gg.plan_grasp_geometry(
                self.color, self.depth, self.intr, cap,
                current_rpy=(0.0, 1.0, 0.0), max_aperture_mm=50.0,
                jaw_long_axis_camera_deg=None,
            )

        self.assertFalse(plan.ok)
        self.assertIn("complete depth envelope", plan.reason)

    def test_round_feature_does_not_require_jaw_axis_calibration(self):
        cap = detection(source="color:blue", width=20, length=24,
                        area=600, axis=-90)
        with patch.object(gg, "segment_target_envelope",
                          return_value=cap):
            plan = gg.plan_grasp_geometry(
                self.color, self.depth, self.intr, cap,
                current_rpy=(0.0, 0.0, 0.0), max_aperture_mm=50.0,
                jaw_long_axis_camera_deg=None,
            )

        self.assertTrue(plan.ok, plan.reason)
        self.assertIn("rotation-invariant", plan.reason)

    def test_elongated_feature_without_jaw_axis_is_refused(self):
        marker = detection(source="color:blue", width=15, length=60,
                           area=1200, axis=-90)
        with patch.object(gg, "segment_target_envelope",
                          return_value=marker):
            plan = gg.plan_grasp_geometry(
                self.color, self.depth, self.intr, marker,
                current_rpy=(0.0, 0.0, 0.0), max_aperture_mm=50.0,
                jaw_long_axis_camera_deg=None,
            )

        self.assertFalse(plan.ok)
        self.assertIn("camera-to-jaw axis", plan.reason)

    def test_perspective_stretched_round_cap_uses_circularity(self):
        cap = detection(source="color:blue", width=17.3, length=28.9,
                        area=1100, axis=-90)
        cap.info["circularity"] = 0.656
        with patch.object(gg, "segment_target_envelope",
                          return_value=cap):
            plan = gg.plan_grasp_geometry(
                self.color, self.depth, self.intr, cap,
                current_rpy=(0.0, 0.0, 0.0), max_aperture_mm=50.0,
                jaw_long_axis_camera_deg=None,
            )

        self.assertTrue(plan.ok, plan.reason)
        self.assertIn("circularity 0.66", plan.reason)

    def test_elongated_region_is_allowed_after_bounded_prealignment(self):
        marker = detection(source="depth:target_envelope", width=26,
                           length=66, area=1600, axis=-84)
        with patch.object(gg, "segment_target_envelope",
                          return_value=marker):
            plan = gg.plan_grasp_geometry(
                self.color, self.depth, self.intr, marker,
                current_rpy=(0.0, 0.0, 0.10), max_aperture_mm=50.0,
                jaw_long_axis_camera_deg=None,
                require_axis_for_elongated=False,
            )

        self.assertTrue(plan.ok, plan.reason)
        self.assertIn("previously aligned", plan.reason)
