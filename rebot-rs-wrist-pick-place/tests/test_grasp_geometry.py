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
    def test_valid_semantic_envelope_skips_table_plane_fallback(self):
        box = detection(source="sam3", width=20, length=70, area=1800,
                        axis=0)
        envelope = detection(source="rgbd:semantic_envelope", width=18,
                             length=68, area=1700, axis=0)
        with patch.object(gg, "segment_target_envelope") as seeded, \
                patch.object(gg, "segment_objects") as plane:
            plan = gg.plan_grasp_geometry(
                self.color, self.depth, self.intr, box,
                current_rpy=(0.0, 0.0, 0.0), max_aperture_mm=50.0,
                jaw_long_axis_camera_deg=0.0,
                semantic_envelope=envelope,
            )

        self.assertTrue(plan.ok, plan.reason)
        self.assertIs(plan.grasp_detection, envelope)
        seeded.assert_not_called()
        plane.assert_not_called()

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

    def test_semantic_box_uses_refined_envelope_orientation(self):
        # Coarse semantic boxes have no reliable physical axis. Their RGB-D
        # refinement must drive both width and jaw orientation, even when the
        # refined mask area is smaller than the original box.
        for source in ("locate_anything", "grounding_dino_sam2", "sam3"):
            with self.subTest(source=source):
                box = detection(source=source, width=33.3, length=61.2,
                                area=5000, axis=None)
                envelope = detection(source="rgbd:semantic_envelope", width=35.5,
                                     length=62.4, area=4200, axis=85.56)
                with patch.object(gg, "segment_target_envelope",
                                  return_value=envelope):
                    plan = gg.plan_grasp_geometry(
                        self.color, self.depth, self.intr, box,
                        current_rpy=(0.0, 0.0, 0.0), max_aperture_mm=50.0,
                        jaw_long_axis_camera_deg=-90.0,
                        max_yaw_delta_deg=10.0,
                    )

                self.assertTrue(plan.ok, plan.reason)
                self.assertTrue(plan.used_full_envelope)
                self.assertIs(plan.grasp_detection, envelope)
                self.assertAlmostEqual(abs(plan.yaw_correction_deg), 4.44,
                                       places=2)

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

    def test_nineteen_degree_alignment_fits_twenty_degree_scene_envelope(self):
        obj = detection(source="depth:object", width=44.2, length=76.9,
                        area=9700, axis=70.9)
        with patch.object(gg, "segment_target_envelope",
                          return_value=obj):
            plan = gg.plan_grasp_geometry(
                self.color, self.depth, self.intr, obj,
                current_rpy=(0.0, 0.0, 0.0), max_aperture_mm=50.0,
                jaw_long_axis_camera_deg=-90.0, max_yaw_delta_deg=20.0,
                yaw_measurement_slop_deg=0.5,
            )

        self.assertTrue(plan.ok, plan.reason)
        self.assertAlmostEqual(plan.yaw_correction_deg, -19.1, places=1)

    def test_asymmetric_camera_budget_clamps_within_vision_slop(self):
        obj = detection(source="depth:object", width=47.2, length=75.8,
                        area=13200, axis=70.02)
        with patch.object(gg, "segment_target_envelope", return_value=obj):
            plan = gg.plan_grasp_geometry(
                self.color, self.depth, self.intr, obj,
                current_rpy=(0.0, 0.1257, -0.0263),
                max_aperture_mm=50.0,
                jaw_long_axis_camera_deg=-90.0,
                max_yaw_delta_deg=20.0,
                yaw_delta_bounds_deg=(-19.55, 20.0),
                yaw_measurement_slop_deg=0.5,
            )

        self.assertTrue(plan.ok, plan.reason)
        self.assertAlmostEqual(plan.yaw_correction_deg, -19.55, places=2)
        self.assertIn("physical clamp", plan.reason)

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

    def test_small_residual_yaw_uses_conservative_projected_width(self):
        obj = detection(source="depth:object", width=42, length=75,
                        area=3000, axis=69.0)
        with patch.object(gg, "segment_target_envelope", return_value=obj):
            plan = gg.plan_grasp_geometry(
                self.color, self.depth, self.intr, obj,
                current_rpy=(0.0, 0.0, 0.0), max_aperture_mm=50.0,
                jaw_long_axis_camera_deg=-90.0,
                max_yaw_delta_deg=20.0,
                yaw_delta_bounds_deg=(-20.0, 20.0),
                yaw_measurement_slop_deg=1.5,
            )

        self.assertTrue(plan.ok, plan.reason)
        self.assertAlmostEqual(plan.yaw_correction_deg, -20.0)
        self.assertIsNotNone(plan.grasp_detection)
        self.assertGreater(plan.grasp_detection.min_width_mm, 42.0)
        self.assertAlmostEqual(
            plan.grasp_detection.info["yaw_residual_deg"], 1.0
        )
        self.assertIn("projected jaw span", plan.reason)

    def test_residual_yaw_is_refused_when_projected_width_does_not_fit(self):
        obj = detection(source="depth:object", width=49, length=90,
                        area=3000, axis=69.0)
        with patch.object(gg, "segment_target_envelope", return_value=obj):
            plan = gg.plan_grasp_geometry(
                self.color, self.depth, self.intr, obj,
                current_rpy=(0.0, 0.0, 0.0), max_aperture_mm=50.0,
                jaw_long_axis_camera_deg=-90.0,
                max_yaw_delta_deg=20.0,
                yaw_delta_bounds_deg=(-20.0, 20.0),
                yaw_measurement_slop_deg=1.5,
            )

        self.assertFalse(plan.ok)
        self.assertIn("projected jaw span", plan.reason)
        self.assertIn("beyond the 50.0mm jaw limit", plan.reason)

    def test_bounded_residual_yaw_accepts_projected_width_that_fits(self):
        obj = detection(source="depth:object", width=25.7, length=73.5,
                        area=7000, axis=65.5)
        with patch.object(gg, "segment_target_envelope", return_value=obj):
            plan = gg.plan_grasp_geometry(
                self.color, self.depth, self.intr, obj,
                current_rpy=(0.0, 0.0, 0.0), max_aperture_mm=50.0,
                jaw_long_axis_camera_deg=-90.0,
                max_yaw_delta_deg=20.0,
                yaw_delta_bounds_deg=(-20.0, 19.5),
                yaw_measurement_slop_deg=1.5,
                max_residual_yaw_deg=6.0,
            )

        self.assertTrue(plan.ok, plan.reason)
        self.assertAlmostEqual(plan.yaw_correction_deg, -20.0)
        self.assertAlmostEqual(
            plan.grasp_detection.info["yaw_residual_deg"], 4.5
        )
        self.assertLess(plan.grasp_detection.min_width_mm, 50.0)

    def test_residual_larger_than_bounded_allowance_is_refused(self):
        obj = detection(source="depth:object", width=20, length=70,
                        area=7000, axis=62.0)
        with patch.object(gg, "segment_target_envelope", return_value=obj):
            plan = gg.plan_grasp_geometry(
                self.color, self.depth, self.intr, obj,
                current_rpy=(0.0, 0.0, 0.0), max_aperture_mm=50.0,
                jaw_long_axis_camera_deg=-90.0,
                max_yaw_delta_deg=20.0,
                yaw_delta_bounds_deg=(-20.0, 19.5),
                yaw_measurement_slop_deg=1.5,
                max_residual_yaw_deg=6.0,
            )

        self.assertFalse(plan.ok)
        self.assertIn("bounded 6.0deg residual allowance", plan.reason)

    def test_aperture_derived_residual_accepts_complete_object_that_fits(self):
        obj = detection(source="depth:object", width=22, length=75,
                        area=7000, axis=62.0)
        with patch.object(gg, "segment_target_envelope", return_value=obj):
            plan = gg.plan_grasp_geometry(
                self.color, self.depth, self.intr, obj,
                current_rpy=(0.0, 0.0, 0.0), max_aperture_mm=50.0,
                jaw_long_axis_camera_deg=-90.0,
                max_yaw_delta_deg=20.0,
                yaw_delta_bounds_deg=(-20.0, 20.0),
                yaw_measurement_slop_deg=1.5,
                max_residual_yaw_deg=6.0,
                allow_aperture_derived_residual=True,
            )

        self.assertTrue(plan.ok, plan.reason)
        self.assertAlmostEqual(plan.yaw_correction_deg, -20.0)
        self.assertLess(plan.grasp_detection.min_width_mm, 50.0)
        self.assertGreaterEqual(
            plan.grasp_detection.max_width_mm,
            plan.grasp_detection.min_width_mm,
        )
        self.assertIn("aperture-derived residual", plan.reason)

    def test_aperture_derived_residual_still_refuses_projected_span(self):
        obj = detection(source="depth:object", width=43, length=90,
                        area=7000, axis=62.0)
        with patch.object(gg, "segment_target_envelope", return_value=obj):
            plan = gg.plan_grasp_geometry(
                self.color, self.depth, self.intr, obj,
                current_rpy=(0.0, 0.0, 0.0), max_aperture_mm=50.0,
                jaw_long_axis_camera_deg=-90.0,
                max_yaw_delta_deg=20.0,
                yaw_delta_bounds_deg=(-20.0, 20.0),
                yaw_measurement_slop_deg=1.5,
                max_residual_yaw_deg=6.0,
                allow_aperture_derived_residual=True,
            )

        self.assertFalse(plan.ok)
        self.assertIn("projected jaw span", plan.reason)
        self.assertIn("beyond the 50.0mm jaw limit", plan.reason)

    def test_multiview_metric_axes_replace_tilted_silhouette_width(self):
        obj = detection(source="depth:object", width=43, length=90,
                        area=7000, axis=62.0)
        with patch.object(gg, "segment_target_envelope", return_value=obj):
            plan = gg.plan_grasp_geometry(
                self.color, self.depth, self.intr, obj,
                current_rpy=(0.0, 0.0, 0.0), max_aperture_mm=50.0,
                jaw_long_axis_camera_deg=-90.0,
                max_yaw_delta_deg=20.0,
                yaw_delta_bounds_deg=(-20.0, 20.0),
                yaw_measurement_slop_deg=1.5,
                max_residual_yaw_deg=6.0,
                allow_aperture_derived_residual=True,
                metric_jaw_width_mm=22.0,
                metric_major_mm=74.0,
            )

        self.assertTrue(plan.ok, plan.reason)
        self.assertLess(plan.grasp_detection.min_width_mm, 50.0)
        self.assertEqual(
            plan.grasp_detection.info["jaw_sizing_basis"],
            "multiview metric 3D horizontal axes",
        )

    def test_metric_width_keeps_the_ranked_subsection_station(self):
        """A metric OBB fixes the jaw width; it must not move the grasp station.

        The metric box says how wide the object is, not where along it to grasp.
        Discarding the ranked cross-section here left the grasp point on the
        whole-mask centroid, which on a long object is a different place.
        """
        candidates = [
            {"uv": [120.0, 150.0], "width_mm": 44.0, "axis_fraction": -0.32,
             "end_clearance_fraction": 0.68, "score": 1.2},
            {"uv": [150.0, 150.0], "width_mm": 41.0, "axis_fraction": 0.0,
             "end_clearance_fraction": 1.0, "score": 2.0},
        ]
        obj = detection(source="sam3", width=43, length=90,
                        area=4000, axis=62.0, uv=(178.0, 150.0))
        envelope = detection(source="depth:object", width=43, length=90,
                             area=7000, axis=62.0, uv=(178.0, 150.0))
        envelope.info.update({
            "cross_section_candidates": candidates,
            "local_grasp_width_mm": 41.0,
        })
        plan = gg.plan_grasp_geometry(
            self.color, self.depth, self.intr, obj,
            current_rpy=(0.0, 0.0, 0.0), max_aperture_mm=50.0,
            jaw_long_axis_camera_deg=-90.0,
            max_yaw_delta_deg=20.0,
            yaw_delta_bounds_deg=(-20.0, 20.0),
            yaw_measurement_slop_deg=1.5,
            max_residual_yaw_deg=6.0,
            allow_aperture_derived_residual=True,
            semantic_envelope=envelope,
            metric_jaw_width_mm=22.0,
            metric_major_mm=74.0,
        )

        self.assertTrue(plan.ok, plan.reason)
        # Position comes from the highest-scoring station, not the centroid.
        self.assertEqual(
            plan.grasp_detection.info["selected_candidate"]["axis_fraction"],
            0.0,
        )
        self.assertAlmostEqual(plan.grasp_detection.uv[0], 150.0, places=3)
        # Width still derives from the 22mm metric box rather than the 41mm
        # perspective-stretched 2D cross-section, after residual-yaw projection.
        self.assertLess(float(plan.grasp_detection.min_width_mm), 41.0)
        self.assertIn(
            "metric", plan.grasp_detection.info["jaw_sizing_basis"]
        )

    def test_metric_axes_apply_before_oversized_feature_rejection(self):
        target = detection(source="sam3", width=43, length=90,
                           area=4000, axis=62.0, z=220)
        envelope = detection(source="depth:object", width=60, length=100,
                             area=7000, axis=62.0, z=230)
        plan = gg.plan_grasp_geometry(
            self.color, self.depth, self.intr, target,
            current_rpy=(0.0, 0.0, 0.0), max_aperture_mm=50.0,
            jaw_long_axis_camera_deg=-90.0,
            max_yaw_delta_deg=20.0,
            yaw_delta_bounds_deg=(-20.0, 20.0),
            max_residual_yaw_deg=6.0,
            allow_aperture_derived_residual=True,
            semantic_envelope=envelope,
            metric_jaw_width_mm=22.7,
            metric_major_mm=74.9,
        )

        self.assertTrue(plan.ok, plan.reason)
        self.assertNotIn("does not fit", plan.reason)
        self.assertLess(plan.grasp_detection.min_width_mm, 50.0)

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
