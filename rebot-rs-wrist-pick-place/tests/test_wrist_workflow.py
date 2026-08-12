from __future__ import annotations

import os
import json
import math
import sys
import time
from unittest import TestCase
from unittest.mock import patch

import numpy as np
from types import SimpleNamespace

from arm_agent import wrist_servo as ws
from arm_agent import wrist_sim


class WristWorkflowTest(TestCase):
    def test_hybrid_identity_reanchors_partial_centroid_to_metric_box(self):
        gate = ws._HybridTargetIdentity(
            description="yellow sponge",
            tool_camera_mm=np.zeros(3), log=lambda _line: None,
            anchor_base_m=np.array([0.40, 0.10, 0.11]),
            semantic_points_base_m=[np.array([0.40, 0.10, 0.11])],
            anchor_kind="semantic",
        )
        box = {
            "center_m": [0.38, 0.098, 0.097],
            "observed_size_m": [0.079, 0.028, 0.053],
            "views": 3,
        }

        accepted, detail = gate.reanchor_from_metric_box(box)

        self.assertTrue(accepted, detail)
        self.assertEqual(gate.anchor_kind, "verified_metric_obb_center")
        self.assertTrue(np.allclose(gate.anchor_base_m, box["center_m"]))
        self.assertEqual(len(gate.semantic_points_base_m), 1)

    def test_hybrid_identity_refuses_metric_box_outside_original_cluster(self):
        gate = ws._HybridTargetIdentity(
            description="yellow sponge",
            tool_camera_mm=np.zeros(3), log=lambda _line: None,
            anchor_base_m=np.array([0.40, 0.10, 0.11]),
            semantic_points_base_m=[np.array([0.40, 0.10, 0.11])],
            anchor_kind="semantic",
        )
        box = {
            "center_m": [0.20, -0.10, 0.05],
            "observed_size_m": [0.079, 0.028, 0.053],
            "views": 3,
        }

        accepted, detail = gate.reanchor_from_metric_box(box)

        self.assertFalse(accepted)
        self.assertIn("beyond", detail["reason"])
        self.assertTrue(np.allclose(
            gate.anchor_base_m, [0.40, 0.10, 0.11]
        ))

    def test_environment_route_collision_signature_ignores_waypoint_density(self):
        first = (
            "Cartesian path collision at segment 19, sample 1/1: "
            "{'collision': True, 'kind': 'environment', "
            "'robot_link': 'gripper_end_0', 'obstacle': 'scene_box_6'}"
        )
        second = first.replace("segment 19", "segment 34")
        self.assertEqual(
            ws._environment_route_collision_signature(first),
            "gripper_end_0->scene_box_6",
        )
        self.assertEqual(
            ws._environment_route_collision_signature(first),
            ws._environment_route_collision_signature(second),
        )

    def test_hybrid_shape_authority_rejects_rgbd_tracking_mask(self):
        mask = np.ones((8, 8), dtype=np.uint8) * 255
        envelope = SimpleNamespace(
            source="rgbd:semantic_envelope",
            info={
                "mask_u8": mask,
                "segmentation_source": "sam3:native_mask",
                "identifier_source": "rgbd:colored_object",
            },
        )
        geometry = SimpleNamespace(
            ok=True, envelope_detection=envelope,
            grasp_detection=envelope,
        )

        accepted, reason = ws._exact_semantic_geometry_mask(geometry)

        self.assertFalse(accepted)
        self.assertIn("non-semantic", reason)

    def test_hybrid_shape_authority_accepts_native_sam3_mask(self):
        mask = np.ones((8, 8), dtype=np.uint8) * 255
        envelope = SimpleNamespace(
            source="rgbd:semantic_envelope",
            info={
                "mask_u8": mask,
                "segmentation_source": "sam3:native_mask",
                "identifier_source": "sam3",
            },
        )
        geometry = SimpleNamespace(
            ok=True, envelope_detection=envelope,
            grasp_detection=envelope,
        )

        accepted, reason = ws._exact_semantic_geometry_mask(geometry)

        self.assertTrue(accepted)
        self.assertIn("native semantic", reason)

    def test_hybrid_shape_authority_survives_preliminary_2d_grasp_failure(self):
        """A trusted mask may build the metric box that fixes bad 2D width."""
        mask = np.ones((8, 8), dtype=np.uint8) * 255
        envelope = SimpleNamespace(
            source="rgbd:semantic_envelope",
            info={
                "mask_u8": mask,
                "segmentation_source": "sam3:native_mask",
                "identifier_source": "sam3",
            },
        )
        geometry = SimpleNamespace(
            ok=False, envelope_detection=envelope,
            grasp_detection=envelope,
        )

        accepted, reason = ws._exact_semantic_geometry_mask(geometry)

        self.assertTrue(accepted)
        self.assertIn("native semantic", reason)

    def test_hybrid_shape_authority_accepts_primary_ranged_sam3_mask(self):
        detection = SimpleNamespace(
            source="sam3",
            info={
                "mask_u8": np.ones((8, 8), dtype=np.uint8) * 255,
                "depth_basis": "native_semantic_mask",
            },
        )

        accepted, reason = ws._exact_semantic_geometry_mask(None, detection)

        self.assertTrue(accepted)
        self.assertIn("detection mask", reason)

    def test_hybrid_shape_authority_accepts_validated_rgbd_completion(self):
        detection = SimpleNamespace(
            source="sam3",
            info={
                "mask_u8": np.ones((8, 8), dtype=np.uint8) * 255,
                "depth_basis": "semantic_seeded_rgbd_object",
                "object_completion_validated": True,
            },
        )

        accepted, reason = ws._exact_semantic_geometry_mask(None, detection)

        self.assertTrue(accepted)
        self.assertIn("complete RGB-D object", reason)

    def test_hybrid_shape_authority_rejects_unvalidated_rgbd_completion(self):
        detection = SimpleNamespace(
            source="sam3",
            info={
                "mask_u8": np.ones((8, 8), dtype=np.uint8) * 255,
                "depth_basis": "semantic_seeded_rgbd_object",
                "object_completion_validated": False,
            },
        )

        accepted, reason = ws._exact_semantic_geometry_mask(None, detection)

        self.assertFalse(accepted)
        self.assertIn("not identity-validated", reason)

    def test_scene_reference_consensus_rejects_one_depth_outlier(self):
        points = [
            np.array([164.9, 37.6, 344.6]),
            np.array([148.4, 36.0, 315.3]),
            np.array([151.3, 36.1, 316.2]),
        ]
        median, detail = ws._scene_reference_point_consensus(
            points, required=2, max_spread_mm=15.0
        )

        self.assertIsNotNone(median)
        self.assertEqual(detail["rejected_frame_indices"], [0])
        self.assertLess(float(detail["spread_mm"]), 2.0)

    def test_scene_reference_consensus_does_not_relax_spread_limit(self):
        points = [
            np.array([0.0, 0.0, 200.0]),
            np.array([25.0, 0.0, 200.0]),
            np.array([0.0, 25.0, 200.0]),
        ]
        median, detail = ws._scene_reference_point_consensus(
            points, required=2, max_spread_mm=10.0
        )

        self.assertIsNone(median)
        self.assertIn("no 2-frame", detail["reason"])

    def test_checkpoint_position_consensus_retains_known_target_shape(self):
        anchor = np.array([0.430, 0.120, 0.105])
        samples = [
            {"point_base_m": [0.438, 0.121, 0.106], "name": "a"},
            {"point_base_m": [0.439, 0.120, 0.106], "name": "b"},
            {"point_base_m": [0.520, 0.200, 0.150], "name": "spill"},
        ]
        median, detail, selected = ws._checkpoint_target_position_consensus(
            samples, anchor
        )

        self.assertIsNotNone(median)
        self.assertEqual([sample["name"] for sample in selected], ["a", "b"])
        self.assertIn("shape/width retained", detail["reason"])

    def test_checkpoint_position_consensus_rejects_unstable_candidates(self):
        anchor = np.array([0.430, 0.120, 0.105])
        samples = [
            {"point_base_m": [0.445, 0.120, 0.105]},
            {"point_base_m": [0.410, 0.120, 0.105]},
        ]
        with patch.dict(os.environ, {
            "REBOT_HYBRID_CHECKPOINT_POSITION_SPREAD_M": "0.010"
        }):
            median, detail, selected = \
                ws._checkpoint_target_position_consensus(samples, anchor)

        self.assertIsNone(median)
        self.assertEqual(selected, [])
        self.assertIn("no 2-frame", detail["reason"])

    def test_checkpoint_reuses_verified_metric_width_without_prior_override(self):
        width = {
            "ok": True,
            "observed_width_mm": 27.6,
            "conservative_width_mm": 33.6,
            # The initial hybrid silhouette already agreed with the metric
            # box, so no earlier replacement was necessary.
            "override_applied": False,
        }

        accepted, reason = ws._checkpoint_metric_width_is_authoritative(
            width, {"observed_size_m": [0.079, 0.028, 0.052]}, 50.0,
            np.array([0.397, 0.095, 0.096]),
        )

        self.assertTrue(accepted)
        self.assertIn("27.6mm", reason)

    def test_checkpoint_rejects_metric_width_for_unconfirmed_target(self):
        width = {
            "ok": True,
            "observed_width_mm": 27.6,
            "conservative_width_mm": 33.6,
        }

        accepted, reason = ws._checkpoint_metric_width_is_authoritative(
            width, {"observed_size_m": [0.079, 0.028, 0.052]}, 50.0, None,
        )

        self.assertFalse(accepted)
        self.assertIn("registered target", reason)

    def test_locked_body_propagation_requires_and_reports_slot_improvement(self):
        body = ws.WD.Detection(
            box=(2, 2, 18, 18), uv=(10.0, 10.0), z_mm=100.0,
            p_cam_mm=(0.0, 0.0, 100.0), source="depth:ranked",
            min_width_mm=20.0, max_width_mm=60.0,
        )
        geometry = ws.GG.GraspGeometry(
            True, "stable geometry", body, body,
            (0.0, 0.0, 0.0), candidates=[],
        )
        jaw_plan = {
            "ok": True, "min_width_mm": 20.0,
            "max_width_mm": 60.0, "max_aperture_mm": 50.0,
            "preopen_mm": 50.0, "grasp_mm": 16.0,
            "grasp_rad": 0.8,
        }
        with patch.object(ws, "load_grasp_target_detail", return_value={
            "jaw_long_axis_camera_deg": -90.0
        }):
            result = ws._propagate_locked_body_after_alignment(
                [20.0, 0.0, 100.0], np.eye(3) * 1000.0,
                [0.30, 0.00, 0.20], [0.29, 0.00, 0.20],
                [0.29, 0.00, 0.20], [0.0, 0.0, 100.0],
                jaw_plan, geometry, 8.0, identity_frames=5,
            )

        self.assertTrue(result["ok"], result)
        self.assertTrue(np.allclose(result["body_cam_mm"], [10, 0, 100]))
        self.assertIn("violation", result["reason"])

    def test_locked_body_propagation_rejects_unconfirmed_identity(self):
        result = ws._propagate_locked_body_after_alignment(
            [20.0, 0.0, 100.0], np.eye(3) * 1000.0,
            [0.30, 0.00, 0.20], [0.29, 0.00, 0.20],
            [0.29, 0.00, 0.20], [0.0, 0.0, 100.0],
            {}, SimpleNamespace(), 8.0, identity_frames=2,
        )

        self.assertFalse(result["ok"])
        self.assertIn("identity", result["reason"])

    def test_hybrid_target_identity_requires_semantic_anchor_and_rejects_far_color(self):
        messages = []
        gate = ws._HybridTargetIdentity(
            description="red marker",
            tool_camera_mm=np.zeros(3),
            log=messages.append,
        )
        color_near = SimpleNamespace(
            source="color:red", p_cam_mm=(10.0, 0.0, 205.0)
        )
        semantic = SimpleNamespace(
            source="sam3", p_cam_mm=(0.0, 0.0, 200.0)
        )
        color_far = SimpleNamespace(
            source="color:red", p_cam_mm=(0.0, 0.0, 520.0)
        )
        frame = (np.zeros((2, 2, 3), dtype=np.uint8),
                 np.ones((2, 2), dtype=np.float32),
                 (100.0, 100.0, 1.0, 1.0))
        with (
            patch.dict(os.environ, {
                "REBOT_HYBRID_TARGET_CLUSTER_RADIUS_M": "0.075"
            }),
            patch.object(ws.A, "ee_rpy", return_value=(0.0, 0.0, 0.0)),
            patch.object(ws.A, "ee_pos", return_value=(0.0, 0.0, 0.0)),
            patch.object(
                ws, "load_saved_camera_jacobian",
                return_value=(np.eye(3) * 1000.0, {"reason": "calibrated"}),
            ),
        ):
            accepted, detail = gate(color_near, frame)
            self.assertFalse(accepted)
            self.assertIn("not been established", detail["reason"])

            accepted, detail = gate(semantic, frame)
            self.assertTrue(accepted)
            self.assertIn("established", detail["reason"])

            accepted, detail = gate(color_near, frame)
            self.assertTrue(accepted)
            self.assertLess(detail["distance_from_anchor_mm"], 75.0)

            accepted, detail = gate(color_far, frame)
            self.assertFalse(accepted)
            self.assertGreater(detail["distance_from_anchor_mm"], 75.0)

        self.assertEqual(gate.accepted, 2)
        self.assertEqual(gate.rejected, 2)

    def test_hybrid_target_identity_can_anchor_stable_rgbd_colored_object(self):
        gate = ws._HybridTargetIdentity(
            description="red marker",
            tool_camera_mm=np.zeros(3),
            log=lambda _message: None,
        )
        frame = (np.zeros((2, 2, 3), dtype=np.uint8),
                 np.ones((2, 2), dtype=np.float32),
                 (100.0, 100.0, 1.0, 1.0))
        observations = [
            SimpleNamespace(source="rgbd:colored_object",
                            p_cam_mm=(0.0, 0.0, depth))
            for depth in (230.0, 231.0, 229.5)
        ]
        with (
            patch.object(ws.A, "ee_rpy", return_value=(0.0, 0.0, 0.0)),
            patch.object(ws.A, "ee_pos", return_value=(0.0, 0.0, 0.0)),
            patch.object(
                ws, "load_saved_camera_jacobian",
                return_value=(np.eye(3) * 1000.0, {"reason": "calibrated"}),
            ),
        ):
            first, _ = gate(observations[0], frame)
            second, _ = gate(observations[1], frame)
            third, detail = gate(observations[2], frame)

        self.assertFalse(first)
        self.assertFalse(second)
        self.assertTrue(third)
        self.assertEqual(gate.anchor_kind, "stable_rgbd_colored_object")
        self.assertLessEqual(detail["spread_mm"], 1.0)

    def test_object_geometry_mode_keeps_legacy_default_and_explicit_hybrid(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(ws._object_geometry_mode(), "legacy")
        with patch.dict(
            os.environ, {"REBOT_OBJECT_GEOMETRY_MODE": "hybrid"}, clear=True
        ):
            self.assertEqual(ws._object_geometry_mode(), "hybrid")
        with patch.dict(
            os.environ, {"REBOT_OBJECT_GEOMETRY_MODE": "unknown"}, clear=True
        ):
            self.assertIsNone(ws._object_geometry_mode())

    def test_hybrid_box_mismatch_uses_fresh_depth_only_for_pick_only(self):
        self.assertEqual(
            ws._hybrid_box_selection_policy(
                False, place_after_pick=False,
                scene_collision_enabled=False,
            ),
            "fresh_stationary_pick_only",
        )
        self.assertEqual(
            ws._hybrid_box_selection_policy(
                False, place_after_pick=True,
                scene_collision_enabled=False,
            ),
            "reject",
        )
        self.assertEqual(
            ws._hybrid_box_selection_policy(
                False, place_after_pick=False,
                scene_collision_enabled=True,
            ),
            "reject",
        )
        self.assertEqual(
            ws._hybrid_box_selection_policy(
                True, place_after_pick=True,
                scene_collision_enabled=True,
            ),
            "registered_multiview",
        )

    def test_hybrid_box_accepts_stationary_long_axis_truncation(self):
        hybrid = {
            "observed_size_m": [0.0105, 0.0176, 0.1115],
            "center_m": [0.415, 0.120, 0.118],
            "views": 2,
        }
        stationary = {
            "observed_size_m": [0.0116, 0.0215, 0.0658],
            "center_m": [0.417, 0.119, 0.094],
            "views": 2,
        }
        accepted, detail = ws._stationary_box_is_long_axis_truncation(
            hybrid, stationary
        )
        self.assertTrue(accepted, detail)
        self.assertTrue(detail["cross_section_consistent"])
        self.assertGreater(detail["long_axis_loss_mm"], 40.0)
        self.assertEqual(detail["longer_source"], "registered_multiview")

    def test_hybrid_box_accepts_registered_long_axis_truncation(self):
        hybrid = {
            "observed_size_m": [0.0108, 0.0167, 0.0597],
            "center_m": [0.432, 0.040, 0.130],
            "views": 2,
        }
        stationary = {
            "observed_size_m": [0.0114, 0.0152, 0.1004],
            "center_m": [0.434, 0.041, 0.111],
            "views": 2,
        }
        accepted, detail = ws._stationary_box_is_long_axis_truncation(
            hybrid, stationary
        )
        self.assertTrue(accepted, detail)
        self.assertTrue(detail["cross_section_consistent"])
        self.assertGreater(detail["long_axis_loss_mm"], 40.0)
        self.assertEqual(detail["longer_source"], "fresh_stationary")

    def test_hybrid_box_does_not_hide_cross_section_disagreement(self):
        hybrid = {
            "observed_size_m": [0.0105, 0.0176, 0.1115],
            "center_m": [0.415, 0.120, 0.118],
            "views": 2,
        }
        wrong = {
            "observed_size_m": [0.030, 0.045, 0.0658],
            "center_m": [0.417, 0.119, 0.094],
            "views": 2,
        }
        accepted, detail = ws._stationary_box_is_long_axis_truncation(
            hybrid, wrong
        )
        self.assertFalse(accepted, detail)
        self.assertFalse(detail["cross_section_consistent"])

    def test_hybrid_box_does_not_hide_wrong_instance_center(self):
        hybrid = {
            "observed_size_m": [0.0105, 0.0176, 0.1115],
            "center_m": [0.415, 0.120, 0.118],
            "views": 2,
        }
        wrong = {
            "observed_size_m": [0.0116, 0.0215, 0.0658],
            "center_m": [0.470, 0.170, 0.094],
            "views": 2,
        }
        accepted, detail = ws._stationary_box_is_long_axis_truncation(
            hybrid, wrong
        )
        self.assertFalse(accepted, detail)
        self.assertGreater(detail["center_xy_error_mm"], 25.0)

    def test_aligned_3d_thickness_is_selected_for_jaw_closure(self):
        box = {
            "axes_base": np.eye(3).tolist(),
            "observed_size_m": [0.012, 0.050, 0.080],
            "conservative_size_m": [0.018, 0.056, 0.086],
        }
        with patch.dict(
            "os.environ",
            {"REBOT_WRIST_JAW_LONG_AXIS_CAMERA_DEG": "-90"},
        ):
            detail = ws._aligned_3d_jaw_width(box, np.eye(3) * 1000.0)

        self.assertTrue(detail["ok"])
        self.assertEqual(detail["axis_index"], 0)
        self.assertAlmostEqual(detail["observed_width_mm"], 12.0)
        self.assertAlmostEqual(detail["conservative_width_mm"], 18.0)

    def test_locked_identifier_recovers_body_mask_jitter_without_relaxing_slot(self):
        geometry = SimpleNamespace(grasp_detection=SimpleNamespace(
            p_cam_mm=np.array([3.9, 31.7, 65.0]),
            min_width_mm=14.0,
            max_width_mm=100.0,
        ))
        plan = {
            "grasp_mm": 10.0, "preopen_mm": 28.0,
            "min_width_mm": 14.0,
        }
        with patch.object(
            ws, "_jaw_slot_fit",
            return_value=(True, {"reason": "axis-aware slot check passed"}),
        ):
            result = ws._locked_identifier_body_slot_fallback(
                np.array([5.2, 26.2, 64.3]),
                {"stable_frames": 3, "spread_mm": 1.0},
                np.array([1.3, -5.5, -0.7]),
                np.array([3.9, 31.7, 65.0]),
                plan,
                geometry,
            )
        self.assertTrue(result["ok"], result)
        np.testing.assert_allclose(
            result["body_cam"], [3.9, 31.7, 65.0]
        )

    def test_locked_identifier_fallback_requires_multiframe_identity(self):
        geometry = SimpleNamespace(grasp_detection=SimpleNamespace(
            p_cam_mm=np.array([3.9, 31.7, 65.0]),
            min_width_mm=14.0,
            max_width_mm=100.0,
        ))
        result = ws._locked_identifier_body_slot_fallback(
            np.array([5.2, 26.2, 64.3]),
            {"stable_frames": 1},
            np.array([1.3, -5.5, -0.7]),
            np.array([3.9, 31.7, 65.0]),
            {
                "grasp_mm": 10.0, "preopen_mm": 28.0,
                "min_width_mm": 14.0,
            },
            geometry,
        )
        self.assertFalse(result["ok"])

    def test_optical_thickness_cannot_authorize_jaw_entry(self):
        box = {
            "axes_base": [[0.0, 0.0, 1.0],
                          [0.0, 1.0, 0.0],
                          [1.0, 0.0, 0.0]],
            "observed_size_m": [0.012, 0.050, 0.080],
            "conservative_size_m": [0.018, 0.056, 0.086],
        }
        with patch.dict(
            "os.environ",
            {"REBOT_WRIST_JAW_LONG_AXIS_CAMERA_DEG": "-90"},
        ):
            detail = ws._aligned_3d_jaw_width(box, np.eye(3) * 1000.0)

        # The 12 mm axis points into the camera. The only image-horizontal
        # axis is 80 mm, so hidden thickness cannot masquerade as jaw width.
        self.assertTrue(detail["ok"])
        self.assertEqual(detail["axis_index"], 2)
        self.assertAlmostEqual(detail["observed_width_mm"], 80.0)

    def test_moderately_tilted_visible_3d_thickness_is_not_rejected(self):
        angle = np.deg2rad(16.5)
        box = {
            "axes_base": [
                [float(np.cos(angle)), 0.0, float(np.sin(angle))],
                [0.0, 1.0, 0.0],
                [-float(np.sin(angle)), 0.0, float(np.cos(angle))],
            ],
            "observed_size_m": [0.0206, 0.0723, 0.0562],
            "conservative_size_m": [0.0266, 0.0783, 0.0622],
        }
        with patch.dict(
            "os.environ",
            {"REBOT_WRIST_JAW_LONG_AXIS_CAMERA_DEG": "-90"},
        ):
            detail = ws._aligned_3d_jaw_width(box, np.eye(3) * 1000.0)

        self.assertTrue(detail["ok"], detail)
        self.assertEqual(detail["axis_index"], 0)
        self.assertAlmostEqual(detail["optical_tilt_deg"], 16.5, places=1)
        self.assertGreater(detail["projected_resolved_mm"], 19.0)

    def test_metric_object_box_projects_to_twelve_image_edges(self):
        box = {
            "center_m": [0.0, 0.0, 1.0],
            "observed_size_m": [0.2, 0.1, 0.2],
            "axes_base": np.eye(3).tolist(),
        }
        edges = ws._project_object_box_edges(
            box, ee_xyz_m=[0.0, 0.0, 0.0],
            camera_mapping=-1000.0 * np.eye(3),
            tool_camera_mm=[0.0, 0.0, 0.0],
            intr=(100.0, 100.0, 320.0, 240.0),
        )

        self.assertEqual(len(edges), 12)
        flattened = np.asarray(edges, dtype=float).reshape(-1, 2)
        self.assertTrue(np.all(np.isfinite(flattened)))
        self.assertGreater(np.ptp(flattened[:, 0]), 15.0)
        self.assertGreater(np.ptp(flattened[:, 1]), 8.0)

    def test_terminal_eof_and_q_both_request_stop(self):
        self.assertTrue(ws._terminal_stop_key(""))
        self.assertTrue(ws._terminal_stop_key("q"))
        self.assertTrue(ws._terminal_stop_key("Q"))
        self.assertFalse(ws._terminal_stop_key("x"))

    def test_payload_box_uses_measured_object_center_and_bounded_padding(self):
        detection = SimpleNamespace(
            min_width_mm=20.0,
            max_width_mm=100.0,
            p_cam_mm=np.array([13.0, 52.0, 100.0]),
            info={"long_axis_deg": 0.0},
        )

        size, axis_ee, center_ee = ws._payload_collision_box_plan(
            detection,
            np.eye(3) * 1000.0,
            np.array([3.0, 32.0, 70.0]),
            (0.0, 0.0, 0.0),
            padding_m=0.0075,
        )

        self.assertTrue(np.allclose(size, [0.035, 0.035, 0.115]))
        self.assertTrue(np.allclose(axis_ee, [-1.0, 0.0, 0.0]))
        self.assertTrue(np.allclose(center_ee, [-0.010, -0.020, -0.030]))

    def test_scene_geometry_reuses_primary_sam3_mask_before_requery(self):
        frame = (np.zeros((20, 30, 3), dtype=np.uint8),
                 np.full((20, 30), 210.0, dtype=np.float32),
                 (100.0, 100.0, 15.0, 10.0))
        detection = SimpleNamespace(source="sam3")
        envelope = SimpleNamespace(source="rgbd:semantic_envelope")
        planned = SimpleNamespace(ok=True, reason="ok")
        with patch.dict(os.environ, {"REBOT_ENABLE_LOCATE_ANYTHING": "1"}), \
                patch("arm_agent.gripper_model.max_aperture_mm",
                      return_value=50.0), \
                patch.object(ws.WD, "segment_semantic_envelope",
                             return_value=envelope) as segment, \
                patch.object(ws.WD, "detect_vlm") as requery, \
                patch.object(ws.GG, "plan_grasp_geometry",
                             return_value=planned) as planner:
            result = ws._plan_scene_grasp_geometry(
                frame, detection, current_rpy=(0.0, 0.0, 0.0),
                object_name="red marker",
            )

        self.assertIs(result, planned)
        segment.assert_called_once()
        requery.assert_not_called()
        self.assertIs(planner.call_args.kwargs["semantic_envelope"], envelope)

    def test_preplan_geometry_consensus_retries_one_table_merge(self):
        feature = SimpleNamespace(source="sam3", z_mm=210.0)
        frame = (np.zeros((8, 8, 3), dtype=np.uint8),
                 np.full((8, 8), 210.0, dtype=np.float32),
                 (100.0, 100.0, 4.0, 4.0))
        failed = SimpleNamespace(
            ok=False, reason="depth envelope merged with table",
            grasp_detection=None,
        )
        body_a = SimpleNamespace(
            p_cam_mm=np.array([2.0, 30.0, 208.0]), min_width_mm=16.0
        )
        body_b = SimpleNamespace(
            p_cam_mm=np.array([2.5, 30.4, 208.5]), min_width_mm=16.5
        )
        valid_a = SimpleNamespace(ok=True, reason="ok", grasp_detection=body_a)
        valid_b = SimpleNamespace(ok=True, reason="ok", grasp_detection=body_b)
        messages = []
        with patch.object(
            ws, "_observe", side_effect=[
                (feature, frame, {}), (feature, frame, {}), (feature, frame, {})
            ]
        ), patch.object(
            ws, "_plan_scene_grasp_geometry",
            side_effect=[failed, valid_a, valid_b],
        ) as planner:
            found, found_frame, geometry, detail = \
                ws._observe_full_geometry_consensus(
                    SimpleNamespace(), "red marker", (0.0, 0.0, 0.0),
                    messages.append,
                    metric_jaw_width_mm=15.5,
                    metric_major_mm=79.0,
                )

        self.assertIs(found, feature)
        self.assertIs(found_frame, frame)
        self.assertIn(geometry, (valid_a, valid_b))
        self.assertEqual(geometry.grasp_detection.min_width_mm, 16.5)
        self.assertEqual(detail["width_max_mm"], 16.5)
        self.assertEqual(detail["attempted_frames"], 3)
        self.assertEqual(detail["valid_full_geometry_frames"], 2)
        self.assertTrue(any("ok=False" in message for message in messages))
        self.assertTrue(all(
            call.kwargs["metric_jaw_width_mm"] == 15.5
            and call.kwargs["metric_major_mm"] == 79.0
            for call in planner.call_args_list
        ))

    def test_hybrid_registered_box_fallback_needs_stable_semantic_position(self):
        frame = (np.zeros((20, 20, 3), dtype=np.uint8),
                 np.full((20, 20), 220.0, dtype=np.float32),
                 (100.0, 100.0, 10.0, 10.0))
        features = [
            ws.WD.Detection(
                box=(4, 4, 12, 16), uv=(8.0, 10.0), z_mm=220.0,
                p_cam_mm=(1.0 + delta, 2.0, 220.0), source="sam3",
            )
            for delta in (0.0, 0.5)
        ]
        body = ws.WD.Detection(
            box=(4, 4, 12, 16), uv=(8.0, 10.0), z_mm=220.0,
            p_cam_mm=(0.0, 2.0, 220.0), source="depth:ranked_cross_section",
            min_width_mm=15.0, max_width_mm=80.0,
        )
        valid = ws.GG.GraspGeometry(
            True, "valid full mask", body, body, (0.0, 0.0, 0.0)
        )
        invalid = ws.GG.GraspGeometry(
            False, "native mask too small", None, None, None
        )
        with patch.object(
            ws, "_observe", side_effect=[
                (features[0], frame, {}), (features[1], frame, {})
            ]
        ), patch.object(
            ws, "_plan_scene_grasp_geometry", side_effect=[valid, invalid]
        ):
            found, found_frame, geometry, detail = \
                ws._observe_full_geometry_consensus(
                    SimpleNamespace(), "red marker", (0.0, 0.0, 0.0),
                    lambda _message: None, max_frames=2,
                    metric_jaw_width_mm=15.0, metric_major_mm=80.0,
                    allow_registered_box_fallback=True,
                )

        self.assertEqual(tuple(found.p_cam_mm), (1.25, 2.0, 220.0))
        self.assertEqual(
            found.info["position_basis"],
            "stable_semantic_identifier_consensus",
        )
        self.assertIs(found_frame, frame)
        self.assertIsNotNone(geometry)
        self.assertTrue(detail["registered_box_fallback"])
        self.assertIsNone(geometry.object_box_3d)

    def test_observe_retries_one_transient_camera_timeout(self):
        detection = SimpleNamespace(source="sam3", z_mm=220.0)
        valid_frame = (
            np.zeros((20, 30, 3), dtype=np.uint8),
            np.full((20, 30), 220.0, dtype=np.float32),
            (100.0, 100.0, 15.0, 10.0),
        )

        class FlakyEye:
            def __init__(self):
                self.calls = 0

            def frame(self):
                self.calls += 1
                if self.calls == 1:
                    raise RuntimeError("Frame didn't arrive within 5000")
                return valid_frame

            def object(self, *_args):
                return detection, {"used": "sam3"}

        eye = FlakyEye()
        with patch.object(ws.time, "sleep"):
            found, frame, why = ws._observe(eye, "red marker", retries=3)

        self.assertIs(found, detection)
        self.assertIsNotNone(frame)
        assert frame is not None
        self.assertIs(frame[0], valid_frame[0])
        self.assertIs(frame[1], valid_frame[1])
        self.assertEqual(frame[2], valid_frame[2])
        self.assertEqual(why["used"], "sam3")
        self.assertEqual(eye.calls, 2)

    def test_observe_aborts_after_consecutive_camera_timeouts(self):
        class MissingEye:
            def __init__(self):
                self.calls = 0

            def frame(self):
                self.calls += 1
                raise RuntimeError("Frame didn't arrive within 5000")

        eye = MissingEye()
        with (
            patch.object(ws.time, "sleep"),
            self.assertRaises(ws.CameraFrameUnavailable) as raised,
        ):
            ws._observe(eye, "red marker", retries=3)

        self.assertEqual(eye.calls, 3)
        self.assertIn("after 3 attempts", str(raised.exception))

    def test_table_object_near_bottom_is_not_mistaken_for_partial(self):
        detection = SimpleNamespace(
            box=(164, 287, 198, 478), z_mm=218.0, source="sam3"
        )
        frame = (np.zeros((480, 640, 3), dtype=np.uint8), None, None)

        self.assertIsNone(ws._detection_partial(detection, frame))

    def test_detection_clamped_to_bottom_is_still_partial(self):
        detection = SimpleNamespace(
            box=(164, 287, 198, 479), z_mm=218.0, source="sam3"
        )
        frame = (np.zeros((480, 640, 3), dtype=np.uint8), None, None)

        reason = ws._detection_partial(detection, frame)

        self.assertIsNotNone(reason)
        self.assertIn("clipped", reason)
        self.assertIn("bottom", reason)

    def test_detection_near_side_remains_partial(self):
        detection = SimpleNamespace(
            box=(8, 100, 198, 300), z_mm=218.0, source="sam3"
        )
        frame = (np.zeros((480, 640, 3), dtype=np.uint8), None, None)

        reason = ws._detection_partial(detection, frame)

        self.assertIsNotNone(reason)
        self.assertIn("left", reason)

    def test_saved_camera_rotation_requires_matching_wrist_orientation(self):
        payload = json.dumps({
            "base_to_camera_rotation": [
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [-1.0, 0.0, 0.0],
            ],
            "calibrated_tool_rpy_rad": [0.0, 0.0, 0.0],
        })
        fake_path = SimpleNamespace(read_text=lambda: payload)
        with (
            patch.object(ws, "CAMERA_ROTATION_FILE", fake_path),
            patch.dict(os.environ, {
                "REBOT_WRIST_SAVED_ROTATION_MAX_RPY_DELTA_DEG": "2.0"
            }),
        ):
            accepted, detail = ws.load_saved_camera_jacobian(
                (0.0, 0.0, np.deg2rad(1.5))
            )
            refused, refused_detail = ws.load_saved_camera_jacobian(
                (0.0, 0.0, np.deg2rad(2.5))
            )

        self.assertIsNotNone(accepted, detail)
        assert accepted is not None
        self.assertTrue(np.allclose(accepted.T @ accepted,
                                    np.eye(3) * 1_000_000.0))
        self.assertIsNone(refused)
        self.assertIn("wrist orientation", refused_detail["reason"])

    def test_saved_camera_rotation_tracks_bounded_wrist_yaw(self):
        saved = np.array([
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [-1.0, 0.0, 0.0],
        ])
        payload = json.dumps({
            "base_to_camera_rotation": saved.tolist(),
            "calibrated_tool_rpy_rad": [0.0, 0.0, 0.0],
        })
        fake_path = SimpleNamespace(read_text=lambda: payload)
        current_rpy = (0.0, 0.0, np.deg2rad(6.7))
        with (
            patch.object(ws, "CAMERA_ROTATION_FILE", fake_path),
            patch.dict(os.environ, {
                "REBOT_WRIST_SAVED_ROTATION_MAX_RPY_DELTA_DEG": "10.0"
            }),
        ):
            adjusted, detail = ws.load_saved_camera_jacobian(current_rpy)

        self.assertIsNotNone(adjusted, detail)
        assert adjusted is not None
        expected = 1000.0 * saved @ ws._rpy_rotation_matrix(current_rpy).T
        self.assertTrue(np.allclose(adjusted, expected, atol=1e-6))
        self.assertTrue(detail["orientation_adjusted"])
        self.assertAlmostEqual(detail["orientation_delta_deg"], 6.7, places=2)

    def test_saved_camera_rotation_honors_sixteen_degree_envelope(self):
        payload = json.dumps({
            "base_to_camera_rotation": [
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [-1.0, 0.0, 0.0],
            ],
            "calibrated_tool_rpy_rad": [0.0, 0.0, 0.0],
        })
        fake_path = SimpleNamespace(read_text=lambda: payload)
        with (
            patch.object(ws, "CAMERA_ROTATION_FILE", fake_path),
            patch.dict(os.environ, {
                "REBOT_WRIST_SAVED_ROTATION_MAX_RPY_DELTA_DEG": "16.0"
            }),
        ):
            accepted, detail = ws.load_saved_camera_jacobian(
                (0.0, 0.0, np.deg2rad(11.7))
            )
            refused, refused_detail = ws.load_saved_camera_jacobian(
                (0.0, 0.0, np.deg2rad(16.5))
            )

        self.assertIsNotNone(accepted, detail)
        self.assertIsNone(refused)
        self.assertIn("maximum 16.00deg", refused_detail["reason"])

    def test_adaptive_pitch_uses_only_remaining_camera_orientation_budget(self):
        payload = json.dumps({
            "base_to_camera_rotation": [
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [-1.0, 0.0, 0.0],
            ],
            "calibrated_tool_rpy_rad": [0.0, 0.0, 0.0],
        })
        fake_path = SimpleNamespace(read_text=lambda: payload)
        base_rpy = (0.0, np.deg2rad(8.0), np.deg2rad(15.0))
        with (
            patch.object(ws, "CAMERA_ROTATION_FILE", fake_path),
            patch.dict(os.environ, {
                "REBOT_WRIST_SAVED_ROTATION_MAX_RPY_DELTA_DEG": "20.0"
            }),
        ):
            bounded, applied_deg, detail = ws._camera_bounded_pitch_bias(
                base_rpy, 10.0
            )
            mapping, final_detail = ws.load_saved_camera_jacobian(bounded)

        self.assertGreater(applied_deg, 0.0)
        self.assertLess(applied_deg, 10.0)
        self.assertIsNotNone(mapping, final_detail)
        self.assertLessEqual(detail["orientation_delta_deg"], 19.25)
        self.assertAlmostEqual(
            np.rad2deg(bounded[1] - base_rpy[1]), applied_deg, places=5
        )

    def test_lookaround_pitch_reduces_negative_yaw_to_shared_camera_budget(self):
        payload = json.dumps({
            "calibrated_tool_rpy_rad": [0.0, 0.0, 0.0],
        })
        fake_path = SimpleNamespace(read_text=lambda: payload)
        with (
            patch.object(ws, "CAMERA_ROTATION_FILE", fake_path),
            patch.dict(os.environ, {
                "REBOT_WRIST_SAVED_ROTATION_MAX_RPY_DELTA_DEG": "22.5"
            }),
        ):
            bounds, detail = ws._camera_safe_yaw_bounds(
                (0.00038, 0.12572, -0.02634), 20.0
            )

        self.assertIsNotNone(bounds, detail)
        self.assertAlmostEqual(bounds[0], -19.55, places=2)
        self.assertAlmostEqual(bounds[1], 20.0, places=2)
        self.assertEqual(detail["reserved_margin_deg"], 0.25)

    def test_dynamic_pick_refuses_before_preflight_without_any_jaw_span(self):
        from arm_agent import gripper_model as gm

        with (
            patch.object(ws, "DYNAMIC_GRASP_ENABLED", True),
            patch.dict(os.environ, {"REBOT_ENABLE_LOCATE_ANYTHING": "1"}),
            patch.object(gm, "describe", return_value={
                "calibrated_mm": False,
            }),
            patch.object(ws.A, "preflight",
                         side_effect=AssertionError("must refuse before arm access")),
        ):
            result = ws.fixed_pick_and_place("object", dry=True)

        self.assertFalse(result["ok"])
        self.assertIn("known 150deg jaw span", result["reason"])

    def test_startup_close_targets_calibrated_software_limit(self):
        calls = []
        with (
            patch.object(ws, "_wrist_limits", return_value=(0.25, 2.60)),
            patch.object(
                ws, "_gripper",
                side_effect=lambda action, target: calls.append(
                    (action, target)
                ) or {"ok": True, "position_deg": 15.0},
            ),
        ):
            result = ws._startup_close_gripper()

        self.assertTrue(result["ok"])
        self.assertEqual(calls, [("close", 0.25)])

    def test_wrist_eye_continuously_streams_from_one_locked_camera(self):
        class Camera:
            def __init__(self):
                self.frames = 0
                self.closed = False

            def frame(self):
                self.frames += 1
                return (
                    np.zeros((10, 10, 3), dtype=np.uint8),
                    np.ones((10, 10), dtype=np.float32) * 200.0,
                    (100.0, 100.0, 5.0, 5.0),
                )

            def close(self):
                self.closed = True

        camera = Camera()
        viz = SimpleNamespace(
            _stream=object(),
            show=lambda *_args, **_kwargs: None,
        )
        eye = ws.WristEye(_camera=camera)
        with patch.object(ws, "_VIZ", viz):
            self.assertTrue(eye.start_live_feed())
            time.sleep(0.08)
            # A synchronous detector read safely shares the same frame lock.
            eye.frame()
            eye.close()

        self.assertGreater(camera.frames, 2)
        self.assertTrue(camera.closed)
        self.assertIsNone(eye._live_thread)

    def test_live_pick_refuses_missing_camera_before_arm_or_gripper_motion(self):
        class MissingCameraEye:
            camera_kind = "realsense"

            def preflight(self):
                raise RuntimeError("no RealSense camera is connected")

        with (
            patch.object(ws, "DYNAMIC_GRASP_ENABLED", False),
            patch.object(ws, "load_tcp", return_value=(0.0, 0.0, 70.0)),
            patch.object(ws, "load_grasp_target", return_value=(3.0, 32.0, 69.0)),
            patch.object(
                ws.A, "preflight",
                side_effect=AssertionError("must not connect to the arm"),
            ),
            patch.object(
                ws, "_startup_close_gripper",
                side_effect=AssertionError("must not command the gripper"),
            ),
        ):
            result = ws.fixed_pick_and_place(
                "object", dry=False, eye=MissingCameraEye(),
                place_after_pick=False,
            )

        self.assertFalse(result["ok"])
        self.assertIn("camera preflight failed", result["reason"])
        self.assertIn("no arm or gripper motion", result["reason"])
        camera_step = next(
            step for step in result["steps"]
            if step["step"] == "camera_preflight"
        )
        self.assertFalse(camera_step["ok"])

    def test_dry_geometry_preview_does_not_read_arm_rpy(self):
        det = SimpleNamespace(
            source="depth:object", uv=(100.0, 100.0), z_mm=250.0,
            min_width_mm=18.0, max_width_mm=40.0, area_px=500.0,
            p_cam_mm=(0.0, 0.0, 250.0), box=(80, 80, 120, 120),
            info={"long_axis_deg": 0.0},
        )
        geometry = SimpleNamespace(
            ok=True, reason="safe", grasp_detection=det,
            summary=lambda: {"ok": True},
        )
        eye = SimpleNamespace(
            frame=lambda: (
                np.zeros((200, 200, 3), dtype=np.uint8),
                np.ones((200, 200), dtype=np.float32) * 250.0,
                (100.0, 100.0, 100.0, 100.0),
            ),
            object=lambda *_args, **_kwargs: (det, {}),
            close=lambda: None,
        )
        with (
            patch.object(ws, "load_tcp", return_value=(0.0, 0.0, 70.0)),
            patch.object(ws, "load_drop_zone", return_value=(0.30, 0.0, 0.10)),
            patch.object(ws.A, "preflight",
                         return_value=ws.A.MoveResult(True, "ik_ready")),
            patch.object(ws.A, "ee_rpy",
                         side_effect=AssertionError("dry mode read arm RPY")),
            patch.object(ws.A, "check_ik",
                         return_value=ws.A.MoveResult(True, "ik_ok")),
            patch.object(ws, "_plan_scene_grasp_geometry",
                         return_value=geometry) as plan_geometry,
            patch.object(ws, "_plan_aperture",
                         return_value={
                             "ok": True, "reason": "fits",
                             "min_width_mm": 18.0,
                             "max_width_mm": 40.0,
                             "max_aperture_mm": 50.0,
                             "preopen_mm": 50.0,
                         }),
        ):
            result = ws.fixed_pick_and_place(
                "object", dry=True, eye=eye
            )

        self.assertTrue(result["ok"], result["reason"])
        self.assertEqual(
            plan_geometry.call_args.kwargs["current_rpy"],
            (0.0, 0.0, 0.0),
        )

    def test_dry_preview_retries_a_transient_semantic_miss_without_motion(self):
        det = SimpleNamespace(
            source="sam3", uv=(100.0, 100.0), z_mm=230.0,
            min_width_mm=16.0, max_width_mm=70.0, area_px=500.0,
            p_cam_mm=(0.0, 0.0, 230.0), box=(80, 80, 120, 120),
            info={"long_axis_deg": 0.0},
        )
        geometry = SimpleNamespace(
            ok=True, reason="safe", grasp_detection=det,
            summary=lambda: {"ok": True},
        )

        class FlakySemanticEye:
            camera_kind = "realsense"

            def __init__(self):
                self.object_calls = 0

            def frame(self):
                return (
                    np.zeros((200, 200, 3), dtype=np.uint8),
                    np.ones((200, 200), dtype=np.float32) * 230.0,
                    (100.0, 100.0, 100.0, 100.0),
                )

            def object(self, *_args, **_kwargs):
                self.object_calls += 1
                if self.object_calls == 1:
                    return None, {"semantic_result": "not found"}
                return det, {}

            def close(self):
                return None

        eye = FlakySemanticEye()
        with (
            patch.dict(os.environ, {
                "REBOT_OBJECT_GEOMETRY_MODE": "hybrid"
            }),
            patch.object(ws, "DYNAMIC_GRASP_ENABLED", False),
            patch.object(ws, "load_tcp", return_value=(0.0, 0.0, 70.0)),
            patch.object(ws, "load_drop_zone", return_value=(0.30, 0.0, 0.10)),
            patch.object(ws, "load_grasp_target", return_value=(3.0, 32.0, 69.0)),
            patch.object(ws.A, "preflight",
                         return_value=ws.A.MoveResult(True, "ik_ready")),
            patch.object(ws.A, "check_ik",
                         return_value=ws.A.MoveResult(True, "ik_ok")),
            patch.object(ws, "_plan_scene_grasp_geometry",
                         return_value=geometry),
            patch.object(ws, "_plan_aperture", return_value={
                "ok": True, "reason": "fits", "min_width_mm": 16.0,
                "max_width_mm": 70.0, "max_aperture_mm": 50.0,
                "preopen_mm": 50.0,
            }),
            patch.object(ws, "_validate_object_gripper_fit",
                         return_value=(True, {"reason": "fits"})),
        ):
            result = ws.fixed_pick_and_place(
                "red marker", dry=True, eye=eye
            )

        self.assertTrue(result["ok"], result["reason"])
        self.assertEqual(eye.object_calls, 2)
        retry_step = next(
            step for step in result["steps"]
            if step["step"] == "dry_stationary_target_retry"
        )
        self.assertTrue(retry_step["ok"])

    def test_aperture_closes_to_detected_width_without_extra_margin(self):
        det = SimpleNamespace(min_width_mm=18.5, max_width_mm=28.0)

        with (
            patch.object(ws, "SIZE_SAFETY_MARGIN_MM", 0.0),
            patch.object(ws, "SIZE_COMPRESSION_MM", 0.0),
        ):
            plan = ws._plan_aperture(det)

        self.assertTrue(plan["ok"], plan["reason"])
        self.assertAlmostEqual(plan["grasp_mm"], 18.5, delta=ws.SIZE_TARGET_TOL_MM)
        self.assertIn("exact detected width", plan["reason"])

    def test_aperture_supports_one_mm_bounded_position_compression(self):
        det = SimpleNamespace(min_width_mm=18.5, max_width_mm=28.0)

        with (
            patch.object(ws, "SIZE_SAFETY_MARGIN_MM", 0.0),
            patch.object(ws, "SIZE_COMPRESSION_MM", 1.0),
        ):
            plan = ws._plan_aperture(det)

        self.assertTrue(plan["ok"], plan["reason"])
        self.assertAlmostEqual(plan["grasp_mm"], 17.5, delta=ws.SIZE_TARGET_TOL_MM)
        self.assertEqual(plan["planned_compression_mm"], 1.0)
        self.assertIn("1.0mm planned position compression", plan["reason"])

    def test_aperture_supports_two_mm_position_compression(self):
        det = SimpleNamespace(min_width_mm=18.5, max_width_mm=28.0)

        with (
            patch.object(ws, "SIZE_SAFETY_MARGIN_MM", 0.0),
            patch.object(ws, "SIZE_COMPRESSION_MM", 2.0),
        ):
            plan = ws._plan_aperture(det)

        self.assertTrue(plan["ok"], plan["reason"])
        self.assertAlmostEqual(plan["grasp_mm"], 16.5,
                               delta=ws.SIZE_TARGET_TOL_MM)
        self.assertEqual(plan["planned_compression_mm"], 2.0)
        self.assertIn("2.0mm planned position compression", plan["reason"])

    def test_aperture_caps_position_compression_at_four_mm(self):
        det = SimpleNamespace(min_width_mm=18.5, max_width_mm=28.0)

        with (
            patch.object(ws, "SIZE_SAFETY_MARGIN_MM", 0.0),
            patch.object(ws, "SIZE_COMPRESSION_MM", 20.0),
        ):
            plan = ws._plan_aperture(det)

        self.assertTrue(plan["ok"], plan["reason"])
        self.assertAlmostEqual(plan["grasp_mm"], 14.5,
                               delta=ws.SIZE_TARGET_TOL_MM)
        self.assertEqual(plan["planned_compression_mm"], 4.0)
        self.assertIn("4.0mm planned position compression", plan["reason"])

    def test_aperture_narrow_object_uses_calibrated_safe_close_floor(self):
        det = SimpleNamespace(min_width_mm=6.2, max_width_mm=17.0)

        with (
            patch.object(ws, "SIZE_SAFETY_MARGIN_MM", 0.0),
            patch.object(ws, "SIZE_COMPRESSION_MM", 4.0),
        ):
            plan = ws._plan_aperture(det)

        self.assertTrue(plan["ok"], plan["reason"])
        floor_mm = ws._minimum_commandable_aperture_mm()
        self.assertIsNotNone(floor_mm)
        self.assertAlmostEqual(plan["grasp_mm"], floor_mm, places=4)
        self.assertAlmostEqual(
            plan["planned_compression_mm"], 6.2 - floor_mm, places=4
        )
        self.assertLess(plan["planned_compression_mm"], 4.0)

    def test_aperture_rejects_object_below_safe_close_floor(self):
        floor_mm = ws._minimum_commandable_aperture_mm()
        self.assertIsNotNone(floor_mm)
        det = SimpleNamespace(
            min_width_mm=floor_mm - 0.1,
            max_width_mm=floor_mm + 10.0,
        )

        plan = ws._plan_aperture(det)

        self.assertFalse(plan["ok"])
        self.assertIn("smallest safe commanded jaw opening", plan["reason"])

    @staticmethod
    def _elongated_fit_fixture(width_mm=27.2, major_mm=90.5,
                               end_fraction=1.0):
        grasp = SimpleNamespace(
            min_width_mm=width_mm,
            max_width_mm=major_mm,
            info={"selected_candidate": {
                "end_clearance_fraction": end_fraction,
            }},
        )
        envelope = SimpleNamespace(
            min_width_mm=width_mm,
            max_width_mm=major_mm,
            info={},
        )
        geometry = SimpleNamespace(
            grasp_detection=grasp,
            envelope_detection=envelope,
        )
        jaw_plan = {
            "min_width_mm": width_mm,
            "max_width_mm": major_mm,
            "max_aperture_mm": 50.0,
            "preopen_mm": 50.0,
        }
        return geometry, jaw_plan

    def test_long_object_fit_uses_local_cross_section_and_end_clearance(self):
        geometry, jaw_plan = self._elongated_fit_fixture()
        with (
            patch.object(ws, "GRASP_MIN_OPEN_CLEARANCE_MM", 8.0),
            patch.object(ws, "GRASP_MIN_END_CLEARANCE_MM", 12.0),
            patch.object(ws, "GRASP_MAX_MAJOR_MM", 150.0),
        ):
            ok, detail = ws._validate_object_gripper_fit(geometry, jaw_plan)

        self.assertTrue(ok, detail)
        self.assertAlmostEqual(detail["end_clearance_mm"], 45.25, places=2)
        self.assertIn("local cross-section fits", detail["reason"])

    def test_object_fit_rejects_width_without_opening_clearance(self):
        geometry, jaw_plan = self._elongated_fit_fixture(width_mm=44.0)
        with patch.object(ws, "GRASP_MIN_OPEN_CLEARANCE_MM", 8.0):
            ok, detail = ws._validate_object_gripper_fit(geometry, jaw_plan)

        self.assertFalse(ok)
        self.assertIn("including clearance", detail["reason"])

    def test_object_fit_rejects_grasp_too_near_long_object_end(self):
        geometry, jaw_plan = self._elongated_fit_fixture(end_fraction=0.20)
        with patch.object(ws, "GRASP_MIN_END_CLEARANCE_MM", 12.0):
            ok, detail = ws._validate_object_gripper_fit(geometry, jaw_plan)

        self.assertFalse(ok)
        self.assertIn("from an object end", detail["reason"])

    def test_metric_box_prevents_truncated_view_from_hiding_payload_length(self):
        geometry, jaw_plan = self._elongated_fit_fixture(major_mm=90.0)
        metric_box = {"conservative_size_m": [0.032, 0.040, 0.180]}
        with patch.object(ws, "GRASP_MAX_MAJOR_MM", 150.0):
            ok, detail = ws._validate_object_gripper_fit(
                geometry, jaw_plan, metric_box
            )

        self.assertFalse(ok)
        self.assertIn("180.0mm long", detail["reason"])
        self.assertEqual(detail["box_dimensions_mm"], [32.0, 40.0, 180.0])

    def test_registered_shape_ignores_rejected_close_depth_spill(self):
        geometry, jaw_plan = self._elongated_fit_fixture(major_mm=265.6)
        geometry.grasp_detection.info[
            "position_basis"
        ] = "stable_semantic_identifier_registered_box"
        metric_box = {
            "conservative_size_m": [0.0169, 0.0220, 0.1092],
            "source": "multiframe_rgbd_measured_obb",
            "views": 2,
        }
        with patch.object(ws, "GRASP_MAX_MAJOR_MM", 150.0):
            ok, detail = ws._validate_object_gripper_fit(
                geometry, jaw_plan, metric_box
            )

        self.assertTrue(ok, detail)
        self.assertAlmostEqual(detail["major_mm"], 109.2)
        self.assertEqual(
            detail["major_basis"], "registered_multiview_metric_box"
        )

    def test_axis_aware_slot_accepts_longitudinal_protrusion_not_bad_width(self):
        geometry, jaw_plan = self._elongated_fit_fixture()
        expected = np.array([0.0, 0.0, 0.0])
        with (
            patch.object(ws, "load_grasp_target_detail", return_value={
                "jaw_long_axis_camera_deg": -90.0,
            }),
            patch.object(ws, "DYNAMIC_SLOT_TOL_MM", 6.0),
        ):
            fits, detail = ws._jaw_slot_fit(
                np.array([-5.66, 14.01, 4.7]), expected,
                jaw_plan, geometry,
            )
            misses, miss_detail = ws._jaw_slot_fit(
                np.array([-12.0, 14.01, 4.7]), expected,
                jaw_plan, geometry,
            )

        self.assertTrue(fits, detail)
        self.assertFalse(misses, miss_detail)
        self.assertGreater(detail["along_error_mm"], 8.0)
        self.assertLess(detail["across_error_mm"], detail["across_limit_mm"])

    def test_slot_depth_limit_remains_strict_without_object_specific_slop(self):
        geometry, jaw_plan = self._elongated_fit_fixture(
            width_mm=15.6, major_mm=81.2
        )
        with (
            patch.object(ws, "load_grasp_target_detail", return_value={
                "jaw_long_axis_camera_deg": -90.0,
            }),
            patch.object(ws, "DYNAMIC_SLOT_TOL_MM", 6.0),
            patch.object(ws, "DYNAMIC_SLOT_DEPTH_SLACK_MM", 0.0),
            patch.object(ws, "DYNAMIC_SLOT_DEPTH_BOUNDARY_SLOP_MM", 0.0),
        ):
            inside, detail = ws._jaw_slot_fit(
                np.array([0.8, 4.6, 5.9]), np.zeros(3),
                jaw_plan, geometry,
            )
            outside, outside_detail = ws._jaw_slot_fit(
                np.array([0.8, 4.6, 6.2]), np.zeros(3),
                jaw_plan, geometry,
            )

        self.assertTrue(inside, detail)
        self.assertFalse(outside, outside_detail)
        self.assertEqual(detail["depth_limit_mm"], 6.0)

    def test_slot_depth_deadband_avoids_submillimetre_overcorrection(self):
        geometry, jaw_plan = self._elongated_fit_fixture(
            width_mm=21.9, major_mm=90.6
        )
        with (
            patch.object(ws, "load_grasp_target_detail", return_value={
                "jaw_long_axis_camera_deg": -90.0,
            }),
            patch.object(ws, "DYNAMIC_SLOT_TOL_MM", 6.0),
            patch.object(ws, "DYNAMIC_SLOT_DEPTH_SLACK_MM", 3.0),
            patch.object(ws, "DYNAMIC_SLOT_DEPTH_BOUNDARY_SLOP_MM", 0.0),
        ):
            inside, detail = ws._jaw_slot_fit(
                np.array([1.5, 5.3, 8.9]), np.zeros(3),
                jaw_plan, geometry,
            )
            outside, outside_detail = ws._jaw_slot_fit(
                np.array([1.5, 5.3, 9.2]), np.zeros(3),
                jaw_plan, geometry,
            )

        self.assertTrue(inside, detail)
        self.assertFalse(outside, outside_detail)
        self.assertEqual(detail["depth_base_limit_mm"], 6.0)
        self.assertEqual(detail["depth_slack_mm"], 3.0)
        self.assertEqual(detail["depth_limit_mm"], 9.0)

    def test_slot_depth_boundary_accepts_logged_point_two_mm_residual_only(self):
        geometry, jaw_plan = self._elongated_fit_fixture(
            width_mm=21.9, major_mm=90.6
        )
        with (
            patch.object(ws, "load_grasp_target_detail", return_value={
                "jaw_long_axis_camera_deg": -90.0,
            }),
            patch.object(ws, "DYNAMIC_SLOT_TOL_MM", 6.0),
            patch.object(ws, "DYNAMIC_SLOT_DEPTH_SLACK_MM", 3.0),
            patch.object(ws, "DYNAMIC_SLOT_DEPTH_BOUNDARY_SLOP_MM", 0.5),
        ):
            logged_inside, detail = ws._jaw_slot_fit(
                np.array([1.6, 5.4, 9.2]), np.zeros(3),
                jaw_plan, geometry,
            )
            real_miss, miss_detail = ws._jaw_slot_fit(
                np.array([1.6, 5.4, 11.1]), np.zeros(3),
                jaw_plan, geometry,
            )

        self.assertTrue(logged_inside, detail)
        self.assertFalse(real_miss, miss_detail)
        self.assertEqual(detail["depth_boundary_slop_mm"], 0.5)
        self.assertEqual(detail["depth_limit_mm"], 9.5)

    def test_slot_across_deadband_keeps_object_inside_physical_opening(self):
        geometry, jaw_plan = self._elongated_fit_fixture(
            width_mm=10.0, major_mm=41.2
        )
        jaw_plan["preopen_mm"] = 32.0
        with (
            patch.object(ws, "load_grasp_target_detail", return_value={
                "jaw_long_axis_camera_deg": -90.0,
            }),
            patch.object(ws, "DYNAMIC_SLOT_BOUNDARY_SLOP_MM", 0.5),
        ):
            inside, detail = ws._jaw_slot_fit(
                np.array([-10.2, 0.4, 2.4]), np.zeros(3),
                jaw_plan, geometry,
            )
            outside, outside_detail = ws._jaw_slot_fit(
                np.array([-10.7, 0.4, 2.4]), np.zeros(3),
                jaw_plan, geometry,
            )

        self.assertTrue(inside, detail)
        self.assertFalse(outside, outside_detail)
        self.assertAlmostEqual(detail["across_limit_mm"], 10.5)
        self.assertAlmostEqual(detail["physical_across_limit_mm"], 11.0)
        self.assertAlmostEqual(detail["capture_reserve_mm"], 0.5)

    def test_slot_reuses_registered_box_when_close_view_is_cropped(self):
        geometry, jaw_plan = self._elongated_fit_fixture(
            width_mm=10.0, major_mm=21.0, end_fraction=0.6
        )
        jaw_plan["preopen_mm"] = 32.0
        registered_box = {
            "conservative_size_m": [0.0162, 0.0327, 0.0472],
        }
        with (
            patch.object(ws, "load_grasp_target_detail", return_value={
                "jaw_long_axis_camera_deg": -90.0,
            }),
            patch.object(ws, "DYNAMIC_SLOT_BOUNDARY_SLOP_MM", 0.5),
        ):
            cropped_ok, cropped_detail = ws._jaw_slot_fit(
                np.array([-10.2, 0.4, 2.4]), np.zeros(3),
                jaw_plan, geometry,
            )
            registered_ok, registered_detail = ws._jaw_slot_fit(
                np.array([-10.2, 0.4, 2.4]), np.zeros(3),
                jaw_plan, geometry, registered_box,
            )

        self.assertFalse(cropped_ok, cropped_detail)
        self.assertFalse(cropped_detail["fit_ok"])
        self.assertTrue(registered_ok, registered_detail)
        self.assertTrue(registered_detail["fit_ok"])
        self.assertIn("47.2mm", registered_detail["fit_reason"])

    def test_slot_violation_measures_only_distance_outside_capture_box(self):
        detail = {
            "fit_ok": True,
            "across_error_mm": 7.3,
            "across_limit_mm": 12.0,
            "along_error_mm": 2.8,
            "along_limit_mm": 10.5,
            "depth_error_mm": 9.8,
            "depth_limit_mm": 6.0,
        }

        self.assertAlmostEqual(ws._jaw_slot_violation_mm(detail), 3.8)
        detail["depth_error_mm"] = 5.9
        self.assertEqual(ws._jaw_slot_violation_mm(detail), 0.0)

    def test_slot_violation_refuses_geometry_that_does_not_fit_gripper(self):
        detail = {
            "fit_ok": False,
            "across_error_mm": 0.0,
            "across_limit_mm": 12.0,
            "along_error_mm": 0.0,
            "along_limit_mm": 20.0,
            "depth_error_mm": 0.0,
            "depth_limit_mm": 6.0,
        }

        self.assertTrue(np.isinf(ws._jaw_slot_violation_mm(detail)))

    def test_endpoint_jaw_slot_depth_fallback_accepts_local_foreground(self):
        geometry, jaw_plan = self._elongated_fit_fixture(
            width_mm=10.9, major_mm=100.0
        )
        jaw_plan["preopen_mm"] = 28.7
        color = np.zeros((100, 100, 3), dtype=np.uint8)
        depth = np.full((100, 100), 120.0, dtype=np.float32)
        depth[35:66, 43:58] = 65.0
        frame = (color, depth, (100.0, 100.0, 50.0, 50.0))

        with patch.object(ws, "load_grasp_target_detail", return_value={
            "jaw_long_axis_camera_deg": -90.0,
        }):
            detection, reason = ws._jaw_slot_depth_occupancy_detection(
                frame, np.array([0.0, 0.0, 65.0]), jaw_plan, geometry
            )

        self.assertIsNotNone(detection, reason)
        self.assertEqual(detection.source, "depth:object")
        self.assertEqual(
            detection.info["endpoint_width_basis"],
            "registered_shape_plus_slot_occupancy",
        )
        self.assertAlmostEqual(detection.min_width_mm, 10.9)
        self.assertGreater(
            detection.info["slot_occupancy_prominence_mm"], 40.0
        )

    def test_endpoint_jaw_slot_depth_fallback_rejects_empty_slot(self):
        geometry, jaw_plan = self._elongated_fit_fixture(
            width_mm=10.9, major_mm=100.0
        )
        jaw_plan["preopen_mm"] = 28.7
        color = np.zeros((100, 100, 3), dtype=np.uint8)
        depth = np.full((100, 100), 120.0, dtype=np.float32)
        frame = (color, depth, (100.0, 100.0, 50.0, 50.0))

        with patch.object(ws, "load_grasp_target_detail", return_value={
            "jaw_long_axis_camera_deg": -90.0,
        }):
            detection, reason = ws._jaw_slot_depth_occupancy_detection(
                frame, np.array([0.0, 0.0, 65.0]), jaw_plan, geometry
            )

        self.assertIsNone(detection)
        self.assertIn("no stable-size depth foreground", reason)

    def test_lift_hold_hypothesis_separates_held_static_and_seating(self):
        jacobian = np.eye(3) * 1000.0
        arm_delta = np.array([0.0, 0.0, 0.015])

        held = ws._lift_hold_hypothesis(
            np.zeros(3), np.array([0.5, 0.0, 2.0]), arm_delta,
            jacobian, tolerance_mm=6.0,
        )
        static = ws._lift_hold_hypothesis(
            np.zeros(3), np.array([0.0, 0.0, 14.5]), arm_delta,
            jacobian, tolerance_mm=6.0,
        )
        seating = ws._lift_hold_hypothesis(
            np.zeros(3), np.array([0.0, 0.0, 8.0]), arm_delta,
            jacobian, tolerance_mm=6.0,
        )

        self.assertEqual(held["classification"], "held")
        self.assertEqual(static["classification"], "static")
        self.assertEqual(seating["classification"], "ambiguous")

    def test_hold_observation_prefers_locked_section(self):
        before, after, basis = ws._hold_observation_pair(
            [1.0, 2.0, 3.0], [1.1, 2.1, 3.1],
            [10.0, 20.0, 30.0], [11.0, 21.0, 31.0],
        )

        self.assertTrue(np.allclose(before, [1.0, 2.0, 3.0]))
        self.assertTrue(np.allclose(after, [1.1, 2.1, 3.1]))
        self.assertEqual(basis, "locked_grasp_section")

    def test_hold_observation_uses_semantic_fallback_when_jaws_occlude_section(self):
        before, after, basis = ws._hold_observation_pair(
            [1.0, 2.0, 3.0], None,
            [10.0, 20.0, 30.0], [10.2, 20.1, 30.2],
        )

        self.assertTrue(np.allclose(before, [10.0, 20.0, 30.0]))
        self.assertTrue(np.allclose(after, [10.2, 20.1, 30.2]))
        self.assertEqual(basis, "semantic_identifier_fallback")

    def test_post_close_baseline_uses_identity_feature_when_jaws_hide_body(self):
        baseline = ws._post_close_hold_baseline(
            None, {"feature_cam_mm": [4.2, 51.0, 68.0]}
        )

        self.assertTrue(baseline["ok"], baseline)
        self.assertIsNone(baseline["body_cam_mm"])
        self.assertTrue(np.allclose(
            baseline["feature_cam_mm"], [4.2, 51.0, 68.0]
        ))
        self.assertEqual(
            baseline["basis"], "post_close_semantic_identifier"
        )
        self.assertIn("micro-lift motion witness", baseline["reason"])

    def test_post_close_baseline_still_prefers_physical_body(self):
        baseline = ws._post_close_hold_baseline(
            [8.4, 21.8, 66.1],
            {"feature_cam_mm": [4.2, 51.0, 68.0]},
        )

        self.assertTrue(baseline["ok"], baseline)
        self.assertTrue(np.allclose(
            baseline["body_cam_mm"], [8.4, 21.8, 66.1]
        ))
        self.assertEqual(baseline["basis"], "post_close_physical_body")

    def test_post_close_baseline_rejects_when_all_identity_is_occluded(self):
        baseline = ws._post_close_hold_baseline(
            None, {"feature_cam_mm": None}
        )

        self.assertFalse(baseline["ok"])
        self.assertEqual(baseline["basis"], "unobserved")

    def test_target_point_filter_removes_semantic_spill_outside_identity(self):
        anchor = np.array([0.40, 0.02, 0.10])
        near = anchor + np.column_stack((
            np.linspace(-0.025, 0.025, 30),
            np.zeros(30),
            np.zeros(30),
        ))
        spill = anchor + np.column_stack((
            np.linspace(0.12, 0.20, 25),
            np.zeros(25),
            np.zeros(25),
        ))

        filtered, detail = ws._filter_target_points_to_identity_cluster(
            np.vstack((near, spill)), anchor, 0.075
        )

        self.assertTrue(detail["ok"], detail)
        self.assertEqual(len(filtered), 30)
        self.assertEqual(detail["removed"], 25)
        self.assertAlmostEqual(detail["radius_mm"], 75.0)

    def test_target_point_filter_is_passthrough_without_identity_anchor(self):
        points = np.array([[0.1, 0.2, 0.3], [0.2, 0.3, 0.4]])

        filtered, detail = ws._filter_target_points_to_identity_cluster(
            points, None, None
        )

        self.assertTrue(detail["ok"], detail)
        self.assertTrue(np.array_equal(filtered, points))

    def test_locked_body_is_recovered_from_stable_semantic_feature(self):
        body, detail = ws._locked_body_from_feature(
            [4.0, 18.9, 68.3], [-2.7, 1.1, 11.2]
        )

        self.assertTrue(np.allclose(body, [6.7, 17.8, 57.1]))
        self.assertIn("no new width", detail["reason"])

    def test_locked_body_fallback_rejects_nonfinite_geometry(self):
        body, detail = ws._locked_body_from_feature(
            [4.0, float("nan"), 68.3], [-2.7, 1.1, 11.2]
        )

        self.assertIsNone(body)
        self.assertIn("non-finite", detail["reason"])

    def test_metric_box_selects_horizontal_thickness_not_tall_silhouette(self):
        width, major, detail = ws._metric_horizontal_box_dimensions({
            "axes_base": [
                [0.9985, -0.0546, 0.0],
                [-0.0546, -0.9985, 0.0],
                [0.0, 0.0, -1.0],
            ],
            "observed_size_m": [0.0739, 0.0221, 0.0564],
        })

        self.assertAlmostEqual(width, 22.1)
        self.assertAlmostEqual(major, 73.9)
        self.assertIn("22.1x73.9mm", detail["reason"])

    def test_tilted_metric_box_uses_measured_principal_extents(self):
        width, major, detail = ws._metric_horizontal_box_dimensions({
            "axes_base": [
                [-0.99836, 0.05637, 0.009996],
                [-0.028666, -0.643372, 0.765017],
                [0.049555, 0.763476, 0.643932],
            ],
            "observed_size_m": [0.07209, 0.04460, 0.02296],
            "orientation_mode": "measured_3d_pca",
        })

        self.assertAlmostEqual(width, 22.96)
        self.assertAlmostEqual(major, 72.09)
        self.assertTrue(detail["requires_jaw_axis_projection"])
        self.assertEqual(detail["width_axis_index"], 2)
        self.assertEqual(detail["major_axis_index"], 0)
        self.assertIn("measured 3D principal extents", detail["reason"])

    def test_visible_checkpoint_stops_before_occluding_final_approach(self):
        endpoint = np.array([0.40, 0.0, 0.20])
        waypoints = [
            (0.31, 0.0, 0.20),
            (0.34, 0.0, 0.20),
            (0.37, 0.0, 0.20),
            (0.40, 0.0, 0.20),
        ]

        prefix, checkpoint, remaining = ws._visible_checkpoint_prefix(
            waypoints, endpoint, (0.28, 0.0, 0.20), checkpoint_m=0.070
        )

        self.assertEqual(prefix, waypoints[:2])
        self.assertTrue(np.allclose(checkpoint, [0.34, 0.0, 0.20]))
        self.assertAlmostEqual(remaining, 0.06)

    def test_dynamic_visible_checkpoint_moves_farther_for_large_object(self):
        small = SimpleNamespace(
            envelope_detection=SimpleNamespace(
                min_width_mm=10.0, max_width_mm=20.0,
            ),
            grasp_detection=None,
            object_box_3d=None,
        )
        large = SimpleNamespace(
            envelope_detection=SimpleNamespace(
                min_width_mm=35.0, max_width_mm=100.0,
            ),
            grasp_detection=None,
            object_box_3d=None,
        )
        frame = (
            np.zeros((480, 640, 3), dtype=np.uint8),
            np.zeros((480, 640), dtype=np.uint16),
            (600.0, 600.0, 320.0, 240.0),
        )

        small_m, small_detail = ws._dynamic_visible_checkpoint_m(
            small, frame, [0.0, 0.0, 65.0]
        )
        large_m, large_detail = ws._dynamic_visible_checkpoint_m(
            large, frame, [0.0, 0.0, 65.0]
        )

        self.assertAlmostEqual(small_m, 0.070)
        self.assertGreater(large_m, small_m)
        self.assertLessEqual(large_m, 0.120)
        self.assertEqual(large_detail["major_mm"], 100.0)
        self.assertEqual(small_detail["usable_image_fraction"], 0.65)

    def test_checkpoint_reacquire_reverses_executed_validated_path(self):
        checked = []

        def validate(points, _rpy):
            checked.extend(points)
            return True, "collision checked"

        prefix = [
            (0.31, 0.0, 0.22),
            (0.32, 0.0, 0.22),
            (0.33, 0.0, 0.22),
        ]
        with (
            patch.dict(os.environ, {
                "REBOT_VISIBLE_CHECKPOINT_REACQUIRE_BACKOFF_M": "0.015",
            }),
            patch.object(ws.A, "ee_pos", return_value=[0.33, 0.0, 0.22]),
            patch.object(ws, "_validate_fixed_waypoints",
                         side_effect=validate),
        ):
            retreat, reason = ws._plan_checkpoint_reacquire_retreat(
                prefix, [0.30, 0.0, 0.22], np.eye(3) * 1000.0,
                (0.0, 0.0, 0.0),
            )

        # Runtime safety bounds the requested 15mm to a 20mm minimum.
        self.assertEqual(retreat, [prefix[1], prefix[0]])
        self.assertEqual(checked, retreat)
        self.assertIn("20.0mm", reason)

    def test_fresh_body_observation_recomputes_endpoint_from_measured_pose(self):
        endpoint = ws._endpoint_from_camera_body(
            [0.30, 0.0, 0.22],
            [10.0, 20.0, 155.0],
            [4.0, 32.0, 65.0],
            np.diag([-1000.0, -1000.0, -1000.0]),
        )

        self.assertIsNotNone(endpoint)
        assert endpoint is not None
        self.assertTrue(np.allclose(endpoint, [0.306, -0.012, 0.31]))

    def test_endpoint_body_pose_consensus_uses_grasp_section_not_identifier(self):
        samples = [
            {"body_cam_mm": point}
            for point in (
                [3.8, 31.5, 65.2],
                [4.0, 31.8, 65.0],
                [3.9, 31.6, 65.1],
                [3.7, 31.7, 65.3],
                [3.9, 31.9, 65.0],
            )
        ]
        body, detail = ws._endpoint_body_pose_consensus(samples)

        self.assertIsNotNone(body, detail)
        self.assertTrue(np.allclose(body, [3.9, 31.7, 65.1], atol=0.11))
        self.assertLess(detail["spread_mm"], 0.5)

    def test_endpoint_body_pose_consensus_ignores_one_depth_outlier(self):
        samples = [
            {"body_cam_mm": point}
            for point in (
                [-43.6, 43.5, 152.9],
                [-41.4, 41.3, 145.2],
                [-45.4, 45.3, 159.1],
            )
        ]

        body, detail = ws._endpoint_body_pose_consensus(
            samples, min_frames=2, max_spread_mm=4.0
        )

        self.assertIsNotNone(body, detail)
        self.assertEqual(detail["stable_frames"], 2)
        self.assertEqual(detail["valid_frames"], 3)
        self.assertEqual(len(detail["rejected_frame_indices"]), 1)
        self.assertLessEqual(detail["spread_mm"], 4.0)

    def test_endpoint_body_pose_consensus_rejects_noisy_frames_without_cluster(self):
        samples = [
            {"body_cam_mm": [0.0, 0.0, z]}
            for z in (120.0, 140.0, 160.0)
        ]

        body, detail = ws._endpoint_body_pose_consensus(
            samples, min_frames=2, max_spread_mm=4.0
        )

        self.assertIsNone(body)
        self.assertIn("no 2-frame", detail["reason"])

    def test_endpoint_width_consensus_uses_ready_cross_section_range(self):
        samples = [
            {"width_mm": width, "token": index}
            for index, width in enumerate((18.5, 18.7, 18.8, 18.9, 19.0))
        ]
        selected, detail = ws._endpoint_width_consensus(
            samples, 21.6, 28.6,
            min_frames=3, max_spread_mm=4.0, ready_tolerance_mm=6.0,
        )

        self.assertIsNotNone(selected, detail)
        self.assertAlmostEqual(detail["median_mm"], 18.8)
        self.assertTrue(detail["ok"])

    def test_endpoint_width_consensus_rejects_unstable_frames(self):
        samples = [
            {"width_mm": width}
            for width in (16.0, 20.5, 25.0, 30.0, 35.0)
        ]
        selected, detail = ws._endpoint_width_consensus(
            samples, 20.0, 30.0,
            min_frames=3, max_spread_mm=3.0, ready_tolerance_mm=6.0,
        )

        self.assertIsNone(selected)
        self.assertIn("no 3-frame width cluster", detail["reason"])

    def test_endpoint_width_consensus_rejects_stable_wrong_section(self):
        samples = [
            {"width_mm": width} for width in (9.8, 10.0, 10.1, 10.2, 10.3)
        ]
        selected, detail = ws._endpoint_width_consensus(
            samples, 21.6, 28.6,
            min_frames=3, max_spread_mm=3.0, ready_tolerance_mm=6.0,
        )

        self.assertIsNone(selected)
        self.assertFalse(detail["ok"])
        self.assertIn("ready cross-section range", detail["reason"])

    def test_five_tight_endpoint_frames_override_coarse_ready_width(self):
        samples = [
            {"width_mm": width, "token": index}
            for index, width in enumerate((19.21, 19.36, 19.18, 19.19, 19.16))
        ]
        selected, detail = ws._endpoint_width_consensus(
            samples, 27.17, 28.40,
            min_frames=3, max_spread_mm=4.0, ready_tolerance_mm=6.0,
        )

        self.assertIsNotNone(selected, detail)
        self.assertTrue(detail["authoritative_endpoint_override"])
        self.assertEqual(detail["acceptance"],
                         "authoritative close-range override")
        self.assertAlmostEqual(detail["median_mm"], 19.19, places=2)

    def test_three_frames_cannot_override_incompatible_ready_width(self):
        samples = [
            {"width_mm": width} for width in (19.16, 19.19, 19.21)
        ]
        selected, detail = ws._endpoint_width_consensus(
            samples, 27.17, 28.40,
            min_frames=3, max_spread_mm=4.0, ready_tolerance_mm=6.0,
        )

        self.assertIsNone(selected)
        self.assertFalse(detail["authoritative_endpoint_override"])

    def test_stable_endpoint_body_allows_four_frame_narrow_section(self):
        samples = [
            {"width_mm": width, "token": index}
            for index, width in enumerate((13.5, 14.0, 14.0, 14.6, 18.5))
        ]
        selected, detail = ws._endpoint_width_consensus(
            samples, 29.9, 29.9,
            min_frames=3, max_spread_mm=4.0, ready_tolerance_mm=6.0,
            allow_stable_narrow_override=True,
        )

        self.assertIsNotNone(selected, detail)
        self.assertTrue(detail["stable_body_endpoint_override"])
        self.assertAlmostEqual(detail["median_mm"], 14.0)

    def test_narrow_section_without_stable_body_authority_still_fails(self):
        samples = [
            {"width_mm": width}
            for width in (13.5, 14.0, 14.0, 14.6, 18.5)
        ]
        selected, detail = ws._endpoint_width_consensus(
            samples, 29.9, 29.9,
            min_frames=3, max_spread_mm=4.0, ready_tolerance_mm=6.0,
        )

        self.assertIsNone(selected)
        self.assertFalse(detail["authoritative_endpoint_override"])

    def test_rigid_camera_jacobian_removes_finite_difference_shear(self):
        measured = np.array([
            [30.0, 1000.0, -20.0],
            [-40.0, 20.0, 980.0],
            [-1010.0, 20.0, 40.0],
        ])

        rigid, detail = ws._rigid_camera_jacobian(measured)

        self.assertIsNotNone(rigid, detail)
        metric = rigid.T @ rigid
        self.assertTrue(np.allclose(metric, np.eye(3) * 1_000_000.0,
                                    atol=1e-6))
        self.assertLess(detail["rigid_fit_relative_error"], 0.15)

    def test_rigid_camera_jacobian_refuses_unstable_detector_boxes(self):
        # This matches the scale of the failed whole-object-box run: one axis
        # measured only ~446mm/m and produced a 0.334 rigid-fit error.
        measured = np.diag([1151.0, 931.0, 445.9])

        rigid, detail = ws._rigid_camera_jacobian(measured)

        self.assertIsNone(rigid)
        self.assertIn("do not fit a rigid camera", detail["reason"])

    def test_feature_endpoint_preserves_offset_from_grasp_region(self):
        feature = SimpleNamespace(p_cam_mm=(30.0, 10.0, 200.0))
        grasp = SimpleNamespace(p_cam_mm=(5.0, 15.0, 210.0))
        expected, offset = ws._feature_target_for_grasp_target(
            np.array([4.0, 32.0, 65.0]), feature, grasp
        )

        self.assertTrue(np.allclose(offset, [25.0, -5.0, -10.0]))
        self.assertTrue(np.allclose(expected, [29.0, 27.0, 55.0]))

    def test_positive_pick_target_y_offset_raises_jaws_and_is_bounded(self):
        with patch.object(ws, "FIXED_PICK_TARGET_Y_OFFSET_MM", 8.0):
            adjusted, applied = ws._apply_pick_target_y_offset(
                np.array([3.9, 31.7, 69.0])
            )
        self.assertEqual(applied, 8.0)
        self.assertTrue(np.allclose(adjusted, [3.9, 39.7, 69.0]))

        with patch.object(ws, "FIXED_PICK_TARGET_Y_OFFSET_MM", 100.0):
            adjusted, applied = ws._apply_pick_target_y_offset(
                np.array([3.9, 31.7, 69.0])
            )
        self.assertEqual(applied, 15.0)
        self.assertTrue(np.allclose(adjusted, [3.9, 46.7, 69.0]))

    def test_fixed_pick_place_sizes_grasp_releases_and_homes_in_sim(self):
        world = wrist_sim.SimWorld(
            object_bgr=(200, 80, 20), object_radius_m=0.008,
            object_height_m=0.045,
        )
        arm = wrist_sim.SimArm(
            world, kinematic=True, grasp_tol_m=0.030,
        )
        original = (ws.A, ws._CAMERA_FACTORY, ws.SETTLE_S)
        tcp = ",".join(str(float(v)) for v in world.to_camera(world.ee))
        try:
            with patch.dict(os.environ, {
                "REBOT_WRIST_TCP": tcp,
                "REBOT_WRIST_DROP": "0.275,-0.187,0.059",
            }):
                ws.install_backend(
                    arm=arm,
                    camera_factory=lambda _kind: wrist_sim.SimCamera(world),
                    settle_s=0.0,
                )
                result = ws.fixed_pick_and_place("blue marker")
        finally:
            ws.install_backend(
                arm=original[0], camera_factory=original[1],
                settle_s=original[2],
            )

        self.assertTrue(result["ok"], result["reason"])
        self.assertTrue(result["homed"])
        self.assertFalse(world.held)
        self.assertTrue(np.allclose(world.object_xy, [0.275, -0.187]))
        self.assertEqual(arm.refused, [])
        self.assertFalse(result["jaw_plan"]["contact_inference"])

    def test_combined_pick_hands_verified_object_to_scene_place(self):
        world = wrist_sim.SimWorld(
            object_bgr=(200, 80, 20), object_radius_m=0.008,
            object_height_m=0.045,
        )
        arm = wrist_sim.SimArm(world, kinematic=True, grasp_tol_m=0.030)
        original = (ws.A, ws._CAMERA_FACTORY, ws.SETTLE_S)
        tcp = ",".join(str(float(v)) for v in world.to_camera(world.ee))
        placed_result = {
            "ok": True, "reason": "scene placed", "verified": True,
            "steps": [{"step": "scene_relative_release_plan", "ok": True,
                       "reason": "prevalidated"}],
            "homed": True, "object_held": False,
        }
        try:
            with (
                patch.dict(os.environ, {"REBOT_WRIST_TCP": tcp}),
                patch.object(
                    ws, "_plan_scene_relative_release",
                    return_value=(
                        np.array([0.32, -0.10, 0.20]),
                        [(0.32, -0.10, 0.20)],
                        {"reason": "reference frozen at ready pose"},
                    ),
                ) as scene_plan,
                patch.object(ws, "fixed_place", return_value=placed_result) as place,
            ):
                ws.install_backend(
                    arm=arm,
                    camera_factory=lambda _kind: wrist_sim.SimCamera(world),
                    settle_s=0.0,
                )
                result = ws.fixed_pick_and_place(
                    "blue marker", place_reference="green marker",
                    place_relation="right", place_distance_m=0.10,
                )
        finally:
            ws._OBJECT_HELD = False
            ws._FIXED_HELD_CONTEXT = None
            ws.install_backend(
                arm=original[0], camera_factory=original[1],
                settle_s=original[2],
            )

        self.assertTrue(result["ok"], result["reason"])
        self.assertTrue(result["pick"]["verified"])
        self.assertTrue(world.held)
        place.assert_called_once()
        self.assertEqual(place.call_args.kwargs["reference"], "green marker")
        self.assertEqual(place.call_args.kwargs["relation"], "right")
        self.assertEqual(place.call_args.kwargs["distance_m"], 0.10)
        self.assertIsNotNone(place.call_args.kwargs["eye"])
        self.assertTrue(np.allclose(
            place.call_args.kwargs["planned_release_endpoint"],
            [0.32, -0.10, 0.20],
        ))
        self.assertEqual(
            place.call_args.kwargs["planned_placement"]["reason"],
            "reference frozen at ready pose",
        )
        scene_plan.assert_called_once()

    def test_fixed_pick_and_place_can_run_as_two_same_process_skills(self):
        world = wrist_sim.SimWorld(
            object_bgr=(200, 80, 20), object_radius_m=0.008,
            object_height_m=0.045,
        )
        arm = wrist_sim.SimArm(
            world, kinematic=True, grasp_tol_m=0.030,
        )
        original = (ws.A, ws._CAMERA_FACTORY, ws.SETTLE_S)
        tcp = ",".join(str(float(v)) for v in world.to_camera(world.ee))
        try:
            with patch.dict(os.environ, {
                "REBOT_WRIST_TCP": tcp,
                "REBOT_WRIST_DROP": "0.275,-0.187,0.059",
            }):
                ws.install_backend(
                    arm=arm, camera_factory=lambda _kind: wrist_sim.SimCamera(world),
                    settle_s=0.0,
                )
                picked = ws.fixed_pick("blue marker")

                self.assertTrue(picked["ok"], picked["reason"])
                self.assertTrue(picked["object_held"])
                self.assertTrue(picked["holding"])
                self.assertFalse(picked["homed"])
                self.assertTrue(world.held)

                placed = ws.fixed_place()
        finally:
            ws._OBJECT_HELD = False
            ws._FIXED_HELD_CONTEXT = None
            ws.install_backend(
                arm=original[0], camera_factory=original[1],
                settle_s=original[2],
            )

        self.assertTrue(placed["ok"], placed["reason"])
        self.assertTrue(placed["homed"])
        self.assertFalse(placed["object_held"])
        self.assertFalse(world.held)
        self.assertTrue(np.allclose(world.object_xy, [0.275, -0.187]))
        self.assertEqual(arm.refused, [])

    def test_fixed_place_refuses_without_same_process_pick(self):
        with (
            patch.object(ws, "load_drop_zone", return_value=(0.275, -0.187, 0.059)),
            patch.object(ws.A, "preflight",
                         side_effect=AssertionError("must not connect to arm")),
            patch.object(ws, "_plan_fixed_approach_trajectory",
                         side_effect=AssertionError("must not plan motion")),
        ):
            ws._OBJECT_HELD = False
            ws._FIXED_HELD_CONTEXT = None
            result = ws.fixed_place()

        self.assertFalse(result["ok"])
        self.assertIn("this live controller session", result["reason"])

    def test_scene_relative_left_uses_robot_positive_y(self):
        det = SimpleNamespace(
            min_width_mm=20.0, max_width_mm=40.0, source="sam3"
        )
        context = {"grasp_geometry": {}}
        eye = SimpleNamespace()
        with (
            patch.object(ws, "_scene_reference_consensus", return_value=(
                det, np.array([3.0, 32.0, 70.0]), {"reason": "stable"}
            )),
            patch.object(ws, "load_grasp_target", return_value=(3.0, 32.0, 70.0)),
            patch.object(ws, "load_saved_camera_jacobian", return_value=(
                np.eye(3) * 1000.0, {"reason": "rigid"}
            )),
            patch.object(ws.A, "ee_pos", return_value=(0.30, 0.0, 0.22)),
            patch.object(ws.A, "ee_rpy", return_value=(0.0, 0.0, 0.0)),
            patch.object(ws, "_high_release_waypoints",
                         return_value=([(0.30, 0.10, 0.20)], "ik ok")),
        ):
            endpoint, waypoints, detail = ws._plan_scene_relative_release(
                "green marker", "left", 0.10, context, eye, lambda _msg: None
            )

        self.assertIsNotNone(endpoint, detail)
        assert endpoint is not None
        self.assertTrue(np.allclose(endpoint, [0.30, 0.10, 0.20]))
        self.assertEqual(waypoints, [(0.30, 0.10, 0.20)])
        self.assertEqual(detail["selected_relation"], "left")
        self.assertFalse(detail["collision_model"])

    def test_place_reference_keeps_accepted_look_frame_and_resamples(self):
        frame = (
            np.zeros((10, 10, 3), dtype=np.uint8),
            np.full((10, 10), 200.0, dtype=np.float32),
            (100.0, 100.0, 5.0, 5.0),
        )

        def detection(x, *, partial):
            return SimpleNamespace(
                p_cam_mm=(x, 20.0, 200.0), source="sam3",
                partial=partial,
            )

        initial = [detection(80.0 + index, partial=True)
                   for index in range(3)]
        accepted = detection(100.0, partial=False)
        after_look = [
            detection(90.0, partial=True),
            detection(91.0, partial=True),
            detection(101.0, partial=False),
            detection(102.0, partial=False),
        ]
        observations = [
            (item, frame, "found") for item in [*initial, *after_look]
        ]
        logs = []
        with (
            patch.object(ws, "SCENE_PLACE_CONSENSUS_FRAMES", 3),
            patch.object(ws, "_observe", side_effect=observations),
            patch.object(ws, "_detection_partial",
                         side_effect=lambda det, _frame: (
                             "clipped" if det.partial else None
                         )),
            patch.object(ws, "_look_around_object", return_value=(
                accepted, frame, None,
                {"ok": True, "pose": "yaw_left", "moved": True},
            )),
            patch.object(ws, "load_grasp_target",
                         return_value=(3.0, 32.0, 69.0)),
            patch.object(ws, "_show"),
            patch.object(ws.time, "sleep"),
        ):
            det, reference_cam, detail = ws._scene_reference_consensus(
                SimpleNamespace(), "blue marker", logs.append
            )

        self.assertIsNotNone(det, detail)
        self.assertTrue(np.allclose(reference_cam, [100.5, 20.0, 200.0]))
        self.assertIn("2/3 stable frames", detail["reason"])
        self.assertTrue(detail["look_around_moved"])
        self.assertTrue(any("attempt=accepted_pose" in line for line in logs))

    def test_place_reference_one_look_frame_still_fails_closed(self):
        frame = (
            np.zeros((10, 10, 3), dtype=np.uint8),
            np.full((10, 10), 200.0, dtype=np.float32),
            (100.0, 100.0, 5.0, 5.0),
        )

        def detection(x, *, partial):
            return SimpleNamespace(
                p_cam_mm=(x, 20.0, 200.0), source="sam3",
                partial=partial,
            )

        # One complete ready-pose sample must be discarded after look-around,
        # because the camera frame changed with the wrist pose.
        partials = [
            detection(80.0, partial=False),
            *[detection(81.0 + index, partial=True) for index in range(5)],
        ]
        accepted = detection(100.0, partial=False)
        with (
            patch.object(ws, "SCENE_PLACE_CONSENSUS_FRAMES", 3),
            patch.object(ws, "_observe", side_effect=[
                (item, frame, "found") for item in partials
            ]),
            patch.object(ws, "_detection_partial",
                         side_effect=lambda det, _frame: (
                             "clipped" if det.partial else None
                         )),
            patch.object(ws, "_look_around_object", return_value=(
                accepted, frame, None,
                {"ok": True, "pose": "yaw_left", "moved": True},
            )),
            patch.object(ws.time, "sleep"),
        ):
            det, reference_cam, detail = ws._scene_reference_consensus(
                SimpleNamespace(), "blue marker", lambda _line: None
            )

        self.assertIsNone(det)
        self.assertIsNone(reference_cam)
        self.assertIn("only 1/3 frames", detail["reason"])

    def test_scene_relative_retries_farther_on_same_side(self):
        det = SimpleNamespace(
            min_width_mm=20.0, max_width_mm=40.0, source="sam3"
        )
        attempted_y = []

        def route(endpoint, _rpy, log=None):
            attempted_y.append(float(endpoint[1]))
            if endpoint[1] > -0.13:
                return None, "inflated reference obstacle"
            return [tuple(float(v) for v in endpoint)], "ik ok"

        with (
            patch.object(ws, "_scene_reference_consensus", return_value=(
                det, np.array([3.0, 32.0, 70.0]), {"reason": "stable"}
            )),
            patch.object(ws, "load_grasp_target", return_value=(3.0, 32.0, 70.0)),
            patch.object(ws, "load_saved_camera_jacobian", return_value=(
                np.eye(3) * 1000.0, {"reason": "rigid"}
            )),
            patch.object(ws.A, "ee_pos", return_value=(0.30, 0.0, 0.22)),
            patch.object(ws.A, "ee_rpy", return_value=(0.0, 0.0, 0.0)),
            patch.object(ws, "_high_release_waypoints", side_effect=route),
        ):
            endpoint, waypoints, detail = ws._plan_scene_relative_release(
                "green marker", "right", 0.10, {}, SimpleNamespace(),
                lambda _msg: None,
            )

        self.assertIsNotNone(endpoint, detail)
        self.assertIsNotNone(waypoints)
        self.assertEqual(detail["selected_relation"], "right")
        self.assertEqual(detail["requested_offset_m"], 0.10)
        self.assertEqual(detail["offset_m"], 0.14)
        self.assertTrue(detail["clearance_expanded"])
        self.assertTrue(np.allclose(attempted_y[:3], [-0.10, -0.12, -0.14]))

    def test_scene_reference_outside_workspace_forces_safe_look_retry(self):
        det = SimpleNamespace(
            min_width_mm=20.0, max_width_mm=40.0, source="sam3"
        )
        consensus_calls = []

        def consensus(_eye, _reference, _log, *, force_look=False):
            consensus_calls.append(force_look)
            if not force_look:
                # Maps to x=0.79m from an x=0.30m ready pose.
                return det, np.array([-487.0, 32.0, 70.0]), {
                    "reason": "stable but wrong semantic instance"
                }
            return det, np.array([3.0, 32.0, 70.0]), {
                "reason": "safe look recovered reachable reference",
                "look_around_moved": True,
            }

        with (
            patch.object(ws, "_scene_reference_consensus",
                         side_effect=consensus),
            patch.object(ws, "load_grasp_target",
                         return_value=(3.0, 32.0, 70.0)),
            patch.object(ws, "load_saved_camera_jacobian", return_value=(
                np.eye(3) * 1000.0, {"reason": "rigid"}
            )),
            patch.object(ws.A, "ee_pos", return_value=(0.30, 0.0, 0.22)),
            patch.object(ws.A, "ee_rpy", return_value=(0.0, 0.0, 0.0)),
            patch.object(ws, "_high_release_waypoints",
                         return_value=([(0.30, -0.10, 0.20)], "ik ok")),
        ):
            endpoint, waypoints, detail = ws._plan_scene_relative_release(
                "blue marker", "right", 0.10, {}, SimpleNamespace(),
                lambda _msg: None,
            )

        self.assertEqual(consensus_calls, [False, True])
        self.assertIsNotNone(endpoint, detail)
        self.assertIsNotNone(waypoints)
        self.assertTrue(np.allclose(endpoint, [0.30, -0.10, 0.20]))

    def test_scene_relative_relation_requires_reference_before_arm_access(self):
        with patch.object(
            ws.A, "preflight", side_effect=AssertionError("must not access arm")
        ):
            result = ws.fixed_place(relation="right")

        self.assertFalse(result["ok"])
        self.assertIn("reference object", result["reason"])

    def test_cached_scene_endpoint_is_revalidated_without_carry_redetection(self):
        ws._OBJECT_HELD = True
        ws._FIXED_HELD_CONTEXT = {
            "object_name": "red marker", "jaw_plan": {},
            "grasp_geometry": {},
        }
        recovery = {"homed": True, "reason": "home_confirmed"}
        try:
            with (
                patch.object(ws.A, "preflight",
                             return_value=ws.A.MoveResult(True, "ready")),
                patch.object(ws.A, "ee_rpy", return_value=(0.0, 0.0, 0.0)),
                patch.object(ws, "WristEye",
                             side_effect=AssertionError("must not reopen camera")),
                patch.object(ws, "_high_release_waypoints",
                             return_value=([(0.32, -0.10, 0.20)], "ik ok")) as path,
                patch.object(ws, "_execute_fixed_approach_trajectory",
                             return_value=(True, "arrived")),
                patch.object(ws, "_gripper", return_value={"ok": True}),
                patch.object(ws, "_verify_release", return_value=True),
                patch.object(ws, "_home_after_interrupt", return_value=recovery),
                patch.object(ws.time, "sleep"),
            ):
                result = ws.fixed_place(
                    reference="green marker", relation="right",
                    planned_release_endpoint=[0.32, -0.10, 0.20],
                    planned_placement={"reason": "ready-pose reference frozen"},
                )
        finally:
            ws._OBJECT_HELD = False
            ws._FIXED_HELD_CONTEXT = None

        self.assertTrue(result["ok"], result["reason"])
        self.assertFalse(result["placement"]["reference_redetection_at_carry"])
        self.assertEqual(result["placement"]["reference_snapshot_stage"],
                         "pre_pick_ready_pose")
        path.assert_called_once()

    def test_cached_scene_endpoint_expands_same_side_for_real_payload(self):
        ws._OBJECT_HELD = True
        ws._FIXED_HELD_CONTEXT = {
            "object_name": "red marker", "jaw_plan": {},
            "grasp_geometry": {},
        }
        recovery = {"homed": True, "reason": "home_confirmed"}
        attempts = []

        def route(endpoint, _rpy, log=None):
            attempts.append(np.asarray(endpoint, dtype=float))
            if len(attempts) == 1:
                return None, "attached payload overlaps obstacle"
            return [tuple(float(value) for value in endpoint)], "ik ok"

        try:
            with (
                patch.object(ws.A, "preflight",
                             return_value=ws.A.MoveResult(True, "ready")),
                patch.object(ws.A, "ee_rpy", return_value=(0.0, 0.0, 0.0)),
                patch.object(ws, "_high_release_waypoints", side_effect=route),
                patch.object(ws, "_execute_fixed_approach_trajectory",
                             return_value=(True, "arrived")),
                patch.object(ws, "_gripper", return_value={"ok": True}),
                patch.object(ws, "_verify_release", return_value=True),
                patch.object(ws, "_home_after_interrupt", return_value=recovery),
                patch.object(ws.time, "sleep"),
            ):
                result = ws.fixed_place(
                    reference="blue marker", relation="left",
                    planned_release_endpoint=[0.30, 0.10, 0.20],
                    planned_placement={
                        "reason": "ready-pose reference frozen",
                        "selected_relation": "left",
                        "reference_aligned_xy_m": [0.30, 0.0],
                        "offset_m": 0.10,
                    },
                )
        finally:
            ws._OBJECT_HELD = False
            ws._FIXED_HELD_CONTEXT = None

        self.assertTrue(result["ok"], result["reason"])
        self.assertEqual(len(attempts), 2)
        self.assertTrue(np.allclose(attempts[1], [0.30, 0.12, 0.20]))
        self.assertTrue(result["placement"]["clearance_expanded_after_pick"])
        self.assertAlmostEqual(result["placement"]["offset_m"], 0.12)

    def test_cached_scene_endpoint_searches_tangentially_when_ray_is_blocked(self):
        ws._OBJECT_HELD = True
        ws._FIXED_HELD_CONTEXT = {
            "object_name": "test object", "jaw_plan": {},
            "grasp_geometry": {},
        }
        recovery = {"homed": True, "reason": "home_confirmed"}
        attempts = []

        def route(endpoint, _rpy, log=None):
            point = np.asarray(endpoint, dtype=float)
            attempts.append(point)
            # Model a tall obstacle strip covering the entire requested +Y ray.
            if abs(float(point[0]) - 0.30) < 1e-6:
                return None, "same relation ray remains occupied"
            return [tuple(float(value) for value in point)], "ik ok"

        try:
            with (
                patch.dict(os.environ, {
                    "REBOT_PLACE_SEARCH_RADIAL_STEP_M": "0.020",
                    "REBOT_PLACE_SEARCH_TANGENT_STEP_M": "0.035",
                    "REBOT_PLACE_SEARCH_MAX_RADIUS_M": "0.180",
                    "REBOT_PLACE_SEARCH_TANGENT_LEVELS": "2",
                }),
                patch.object(ws.A, "preflight",
                             return_value=ws.A.MoveResult(True, "ready")),
                patch.object(ws.A, "ee_rpy", return_value=(0.0, 0.0, 0.0)),
                patch.object(ws, "_high_release_waypoints", side_effect=route),
                patch.object(ws, "_execute_fixed_approach_trajectory",
                             return_value=(True, "arrived")),
                patch.object(ws, "_gripper", return_value={"ok": True}),
                patch.object(ws, "_verify_release", return_value=True),
                patch.object(ws, "_home_after_interrupt", return_value=recovery),
                patch.object(ws.time, "sleep"),
            ):
                result = ws.fixed_place(
                    reference="blue marker", relation="left",
                    planned_release_endpoint=[0.30, 0.10, 0.20],
                    planned_placement={
                        "reason": "ready-pose reference frozen",
                        "selected_relation": "left",
                        "reference_aligned_xy_m": [0.30, 0.0],
                        "offset_m": 0.10,
                    },
                )
        finally:
            ws._OBJECT_HELD = False
            ws._FIXED_HELD_CONTEXT = None

        self.assertTrue(result["ok"], result["reason"])
        # Best-first ordering reaches the closest tangential cell without first
        # exhausting every more distant point on the blocked relation ray.
        self.assertGreaterEqual(len(attempts), 3)
        accepted_xy = np.asarray(
            result["placement"]["desired_object_center_xy_m"], dtype=float
        )
        self.assertGreater(accepted_xy[1], 0.0)  # still left of reference
        self.assertFalse(np.isclose(accepted_xy[0], 0.30))
        self.assertLessEqual(float(np.linalg.norm(
            accepted_xy - np.array([0.30, 0.0])
        )), 0.1801)
        self.assertNotEqual(
            result["placement"]["tangential_offset_m"], 0.0
        )
        self.assertIn("same-side 2D", result["placement"]["payload_replan_reason"])

    def test_cached_scene_endpoint_can_rotate_measured_payload_to_fit(self):
        ws._OBJECT_HELD = True
        ws._FIXED_HELD_CONTEXT = {
            "object_name": "long test object", "jaw_plan": {},
            "grasp_geometry": {},
            "payload_box": {
                "full_size_m": [0.03, 0.08, 0.06],
                "full_axes_ee": np.eye(3).tolist(),
                "center_offset_ee_m": [0.0, 0.0, -0.03],
            },
        }
        recovery = {"homed": True, "reason": "home_confirmed"}
        attempted_yaw = []
        executed_rpy = []

        def route(endpoint, rpy, log=None):
            attempted_yaw.append(float(rpy[2]))
            if abs(float(rpy[2])) < math.radians(10.0):
                return None, "long payload overlaps obstacle at this yaw"
            return [tuple(float(value) for value in endpoint)], "ik ok"

        def execute(_waypoints, _log, fixed_rpy=None):
            executed_rpy.append(tuple(fixed_rpy))
            return True, "arrived"

        try:
            with (
                patch.dict(os.environ, {
                    "REBOT_PLACE_SEARCH_YAW_STEP_DEG": "15",
                    "REBOT_PLACE_SEARCH_MAX_YAW_DELTA_DEG": "30",
                }),
                patch.object(ws.A, "preflight",
                             return_value=ws.A.MoveResult(True, "ready")),
                patch.object(ws.A, "ee_rpy", return_value=(0.0, 0.0, 0.0)),
                patch.object(ws, "_high_release_waypoints", side_effect=route),
                patch.object(ws, "_execute_fixed_approach_trajectory",
                             side_effect=execute),
                patch.object(ws, "_gripper", return_value={"ok": True}),
                patch.object(ws, "_verify_release", return_value=True),
                patch.object(ws, "_home_after_interrupt", return_value=recovery),
                patch.object(ws.time, "sleep"),
            ):
                result = ws.fixed_place(
                    reference="blue marker", relation="left",
                    planned_release_endpoint=[0.30, 0.10, 0.20],
                    planned_placement={
                        "reason": "ready-pose reference frozen",
                        "selected_relation": "left",
                        "reference_aligned_xy_m": [0.30, 0.0],
                        "offset_m": 0.10,
                    },
                )
        finally:
            ws._OBJECT_HELD = False
            ws._FIXED_HELD_CONTEXT = None

        self.assertTrue(result["ok"], result["reason"])
        self.assertEqual(len(attempted_yaw), 2)
        self.assertAlmostEqual(abs(attempted_yaw[1]), math.radians(15.0))
        self.assertEqual(executed_rpy, [(0.0, 0.0, attempted_yaw[1])])
        self.assertAlmostEqual(
            abs(result["placement"]["placement_yaw_delta_deg"]), 15.0
        )

    def test_cached_scene_endpoint_searches_beyond_old_radius(self):
        ws._OBJECT_HELD = True
        ws._FIXED_HELD_CONTEXT = {
            "object_name": "test object", "jaw_plan": {},
            "grasp_geometry": {},
            "payload_box": {
                "full_size_m": [0.03, 0.08, 0.06],
                "full_axes_ee": np.eye(3).tolist(),
                "center_offset_ee_m": [0.0, 0.0, -0.03],
            },
        }
        recovery = {"homed": True, "reason": "home_confirmed"}
        attempts = []

        def route(endpoint, _rpy, log=None):
            point = np.asarray(endpoint, dtype=float)
            attempts.append(point)
            distance = float(np.linalg.norm(point[:2] - [0.30, 0.0]))
            if distance < 0.20 - 1e-6:
                return None, "local clutter occupies the near placement area"
            return [tuple(float(value) for value in point)], "ik ok"

        try:
            with (
                patch.dict(os.environ, {
                    "REBOT_PLACE_SEARCH_MAX_RADIUS_M": "0.280",
                    "REBOT_PLACE_SEARCH_MAX_CANDIDATES": "64",
                }),
                patch.object(ws.A, "preflight",
                             return_value=ws.A.MoveResult(True, "ready")),
                patch.object(ws.A, "ee_rpy", return_value=(0.0, 0.0, 0.0)),
                patch.object(ws, "_high_release_waypoints", side_effect=route),
                patch.object(ws, "_execute_fixed_approach_trajectory",
                             return_value=(True, "arrived")),
                patch.object(ws, "_gripper", return_value={"ok": True}),
                patch.object(ws, "_verify_release", return_value=True),
                patch.object(ws, "_home_after_interrupt", return_value=recovery),
                patch.object(ws.time, "sleep"),
            ):
                result = ws.fixed_place(
                    reference="blue marker", relation="left",
                    planned_release_endpoint=[0.30, 0.10, 0.20],
                    planned_placement={
                        "reason": "ready-pose reference frozen",
                        "selected_relation": "left",
                        "reference_aligned_xy_m": [0.30, 0.0],
                        "offset_m": 0.10,
                    },
                )
        finally:
            ws._OBJECT_HELD = False
            ws._FIXED_HELD_CONTEXT = None

        self.assertTrue(result["ok"], result["reason"])
        accepted_xy = np.asarray(
            result["placement"]["desired_object_center_xy_m"], dtype=float
        )
        self.assertGreaterEqual(
            float(np.linalg.norm(accepted_xy - [0.30, 0.0])), 0.20 - 1e-6
        )
        self.assertGreater(len(attempts), 5)

    def test_scene_relation_language_aliases_are_bounded(self):
        self.assertEqual(ws._normalize_scene_relation("left of"), "left")
        self.assertEqual(ws._normalize_scene_relation("in front of"), "front")
        self.assertEqual(ws._normalize_scene_relation("next to"), "next_to")
        self.assertEqual(ws._normalize_scene_relation("somewhere on table"),
                         "free_space")
        self.assertIsNone(ws._normalize_scene_relation("on top of"))

    def test_fixed_pick_hold_q_runs_controlled_recovery(self):
        fake_stdin = SimpleNamespace(
            fileno=lambda: 7,
            isatty=lambda: True,
            read=lambda _count: "q",
        )
        stopped = {"ok": True, "homed": True, "gripper_opened": True}
        ws._OBJECT_HELD = True
        ws._FIXED_HELD_CONTEXT = {"object_name": "test object"}
        try:
            with (
                patch.object(sys, "stdin", fake_stdin),
                patch("select.select", return_value=([fake_stdin], [], [])),
                patch("termios.tcgetattr", return_value=[0]),
                patch("termios.tcsetattr") as restore_tty,
                patch("tty.setcbreak"),
                patch.object(ws, "abort_fixed_hold", return_value=stopped) as abort,
            ):
                result = ws.hold_fixed_pick_until_quit()
        finally:
            ws._OBJECT_HELD = False
            ws._FIXED_HELD_CONTEXT = None

        self.assertIs(result, stopped)
        abort.assert_called_once()
        self.assertIn("Q pressed", abort.call_args.args[0])
        restore_tty.assert_called_once()

    def test_pick_place_cli_installs_q_listener_for_entire_operation(self):
        fake_stdin = SimpleNamespace(isatty=lambda: False)
        completed = {"ok": True, "homed": True}
        with (
            patch.object(ws.sys, "stdin", fake_stdin),
            patch.object(ws, "fixed_pick_and_place",
                         return_value=completed) as operation,
        ):
            result = ws.run_fixed_pick_place_cli(
                "yellow sponge", place_reference="blue marker",
                place_relation="left", place_distance_m=0.10,
            )

        self.assertIs(result, completed)
        operation.assert_called_once_with(
            "yellow sponge", dry=False, verbose=False,
            place_reference="blue marker", place_relation="left",
            place_distance_m=0.10,
        )

    def test_fixed_executor_bisects_a_locally_refused_segment(self):
        current = np.array([0.30, 0.0, 0.22], dtype=float)
        attempted = []
        orientations = []

        def safe_move(x, y, z, *, cur=None, **kwargs):
            target = np.array([x, y, z], dtype=float)
            attempted.append(target.copy())
            orientations.append((kwargs.get("roll"), kwargs.get("pitch"),
                                 kwargs.get("yaw")))
            if float(np.max(np.abs(target - current))) > 0.0051:
                return ws.A.MoveResult(False, "local_ik_error")
            current[:] = target
            return ws.A.MoveResult(True, "arrived", tuple(target))

        with (
            patch.object(ws.A, "ee_pos", lambda: current.tolist()),
            patch.object(ws.A, "safe_move", safe_move),
            patch.object(ws.A, "BACKEND", "service"),
            patch.object(ws, "APPROACH_SEGMENT_BISECT_DEPTH", 4),
        ):
            ok, reason = ws._execute_fixed_approach_trajectory(
                [(0.308, 0.0, 0.216)], lambda _message: None,
                fixed_rpy=(0.01, -0.02, 0.03),
            )

        self.assertTrue(ok, reason)
        self.assertTrue(np.allclose(current, [0.308, 0.0, 0.216]))
        self.assertGreaterEqual(len(attempted), 3)
        self.assertTrue(all(rpy == (0.01, -0.02, 0.03)
                            for rpy in orientations))

    def test_fixed_executor_prefers_continuous_minimum_jerk_on_local_rs(self):
        calls = []
        waypoints = [(0.304, 0.0, 0.218), (0.308, 0.0, 0.216)]

        def continuous(path, **kwargs):
            calls.append((list(path), kwargs))
            return ws.A.MoveResult(
                True, "continuous_minimum_jerk_arrived", path[-1]
            )

        with (
            patch.object(ws.A, "BACKEND", "local"),
            patch.object(ws.A, "safe_move_path", continuous),
            patch.object(ws.A, "safe_move",
                         side_effect=AssertionError("must not stop at waypoints")),
        ):
            ok, reason = ws._execute_fixed_approach_trajectory(
                waypoints, lambda _message: None,
                fixed_rpy=(0.01, -0.02, 0.03),
            )

        self.assertTrue(ok, reason)
        self.assertIn("continuous minimum-jerk", reason)
        self.assertEqual(calls[0][0], waypoints)
        self.assertEqual(calls[0][1], {
            "roll": 0.01, "pitch": -0.02, "yaw": 0.03,
        })

    def test_endpoint_alignment_retry_is_bounded_and_prevalidated(self):
        planned_endpoints = []

        def plan(endpoint, **_kwargs):
            endpoint = np.asarray(endpoint, dtype=float)
            planned_endpoints.append(endpoint.copy())
            return [tuple(endpoint)], "IK path prevalidated"

        # This requests a 30 mm camera correction. With a rigid -1000 mm/m
        # eye-in-hand mapping and 0.85 gain it would move 25.5 mm, so the helper
        # must clamp it to the independent 12 mm endpoint-retry ceiling.
        with (
            patch.object(ws.A, "ee_pos", lambda: [0.30, 0.0, 0.22]),
            patch.object(ws, "_plan_fixed_approach_trajectory", plan),
            patch.object(ws, "FIXED_PICK_MAX_CORRECTION_M", 0.050),
        ):
            waypoints, detail = ws._plan_endpoint_alignment_correction(
                observed_cam_mm=[30.0, 0.0, 65.0],
                expected_cam_mm=[0.0, 0.0, 65.0],
                rigid_jacobian=-1000.0 * np.eye(3),
                fixed_rpy=(0.0, 0.0, 0.0),
            )

        self.assertIsNotNone(waypoints, detail)
        self.assertAlmostEqual(detail["correction_norm_mm"], 12.0)
        self.assertEqual(len(planned_endpoints), 1)
        self.assertAlmostEqual(
            float(np.linalg.norm(planned_endpoints[0] - [0.30, 0.0, 0.22])),
            0.012,
        )

    def test_endpoint_alignment_uses_global_route_when_straight_path_blocked(self):
        certified = {
            "ok": True, "kind": "certified_joint_plan",
            "joint_path": [[0.0] * 6, [0.1] * 6],
        }
        with (
            patch.object(ws, "SCENE_COLLISION_ENABLED", True),
            patch.object(ws.A, "ee_pos", return_value=[0.30, 0.0, 0.22]),
            patch.object(
                ws, "_plan_fixed_approach_trajectory",
                return_value=(None, "straight path collision"),
            ),
            patch.object(
                ws.A, "check_pose_clearance",
                return_value=SimpleNamespace(ok=True, reason="clear"),
            ),
            patch.object(ws.A, "plan_joint_motion", return_value=certified),
        ):
            route, detail = ws._plan_endpoint_alignment_correction(
                [0.0, 0.0, 75.0], [0.0, 0.0, 65.0],
                -1000.0 * np.eye(3), (0.0, 0.0, 0.0),
            )

        self.assertIs(route, certified)
        self.assertEqual(detail["route_kind"], "certified_global_joint")

    def test_partial_alignment_keeps_original_slot_gate(self):
        start = np.array([0.30, 0.0, 0.22])

        def bounded_path(endpoint, **_kwargs):
            displacement = float(np.linalg.norm(np.asarray(endpoint) - start))
            if displacement > 0.006:
                return None, "nearby obstacle blocks larger correction"
            return [tuple(endpoint)], "partial path clear"

        def depth_slot(observed, expected, _plan, _geometry, _box=None):
            depth_error = abs(float(np.asarray(observed)[2] - np.asarray(expected)[2]))
            return depth_error <= 6.0, {
                "reason": f"depth {depth_error:.1f}/6.0mm",
                "depth_error_mm": depth_error,
                "depth_limit_mm": 6.0,
                "fit_ok": True,
            }

        with (
            patch.object(ws.A, "ee_pos", return_value=start.tolist()),
            patch.object(ws, "_plan_fixed_approach_trajectory",
                         side_effect=bounded_path),
            patch.object(ws, "_jaw_slot_fit", side_effect=depth_slot),
        ):
            route, detail = ws._plan_partial_slot_alignment_correction(
                [0.0, 0.0, 75.8], [0.0, 0.0, 65.0],
                -1000.0 * np.eye(3), (0.0, 0.0, 0.0),
                {}, SimpleNamespace(),
            )

        self.assertIsNotNone(route, detail)
        self.assertEqual(detail["partial_fraction"], 0.55)
        self.assertLessEqual(
            detail["predicted_slot"]["depth_error_mm"], 6.0
        )

    def test_alignment_feedback_matches_measured_local_response(self):
        geometric = -1000.0 * np.eye(3)
        tool_delta = np.array([0.006, 0.0, 0.0])

        updated, detail = ws._update_alignment_jacobian_from_feedback(
            geometric,
            tool_delta,
            [5.0, 25.0, 77.0],
            [4.0, 25.0, 77.0],
        )

        self.assertIsNotNone(updated, detail)
        self.assertTrue(detail["ok"])
        np.testing.assert_allclose(
            updated @ tool_delta, [-1.0, 0.0, 0.0], atol=1e-9
        )
        # The saved prior passed by the caller is never mutated.
        np.testing.assert_allclose(geometric, -1000.0 * np.eye(3))

    def test_alignment_feedback_rejects_opposite_visual_motion(self):
        updated, detail = ws._update_alignment_jacobian_from_feedback(
            -1000.0 * np.eye(3),
            [0.006, 0.0, 0.0],
            [5.0, 25.0, 77.0],
            [6.0, 25.0, 77.0],
        )

        self.assertIsNone(updated)
        self.assertFalse(detail["ok"])
        self.assertIn("contradicts", detail["reason"])

    def test_visible_endpoint_can_enter_two_step_bounded_retry_envelope(self):
        with (
            patch.object(ws, "DYNAMIC_GRASP_ENABLED", True),
            patch.object(ws, "FIXED_PICK_TARGET_TOL_MM", 20.0),
            patch.object(ws, "FIXED_PICK_MAX_CORRECTIONS", 2),
            patch.object(ws, "FIXED_PICK_MAX_CORRECTION_M", 0.012),
        ):
            envelope = ws._endpoint_retry_envelope_mm(object_visible=True)

        self.assertEqual(envelope, 24.0)
        self.assertLess(20.01, envelope)

    def test_three_step_retry_envelope_covers_logged_partial_endpoint(self):
        with (
            patch.object(ws, "DYNAMIC_GRASP_ENABLED", True),
            patch.object(ws, "FIXED_PICK_TARGET_TOL_MM", 20.0),
            patch.object(ws, "FIXED_PICK_MAX_CORRECTIONS", 3),
            patch.object(ws, "FIXED_PICK_MAX_CORRECTION_M", 0.012),
        ):
            envelope = ws._endpoint_retry_envelope_mm(object_visible=True)

        self.assertEqual(envelope, 36.0)
        self.assertLess(26.1, envelope)

    def test_retry_gate_subtracts_unchanged_endpoint_tolerance(self):
        with (
            patch.object(ws, "DYNAMIC_GRASP_ENABLED", True),
            patch.object(ws, "DYNAMIC_SLOT_TOL_MM", 8.0),
            patch.object(ws, "FIXED_PICK_TARGET_TOL_MM", 20.0),
            patch.object(ws, "FIXED_PICK_MAX_CORRECTIONS", 3),
            patch.object(ws, "FIXED_PICK_MAX_CORRECTION_M", 0.012),
        ):
            accepted = ws._endpoint_retry_eligibility(
                37.9, object_visible=True
            )
            rejected = ws._endpoint_retry_eligibility(
                45.0, object_visible=True
            )

        self.assertTrue(accepted["eligible"])
        self.assertAlmostEqual(accepted["required_correction_mm"], 29.9)
        self.assertEqual(accepted["correction_capacity_mm"], 36.0)
        self.assertFalse(rejected["eligible"])

    def test_occluded_endpoint_cannot_use_visual_retry_envelope(self):
        with (
            patch.object(ws, "DYNAMIC_GRASP_ENABLED", True),
            patch.object(ws, "FIXED_PICK_TARGET_TOL_MM", 20.0),
            patch.object(ws, "FIXED_PICK_MAX_CORRECTIONS", 2),
            patch.object(ws, "FIXED_PICK_MAX_CORRECTION_M", 0.012),
        ):
            envelope = ws._endpoint_retry_envelope_mm(object_visible=False)

        self.assertEqual(envelope, 20.0)

    def test_fixed_approach_trajectory_prevalidates_small_waypoints(self):
        checked = []

        def check_ik(x, y, z, *rpy):
            checked.append((x, y, z, *rpy))
            return ws.A.MoveResult(True, "ik_ok", (x, y, z))

        with (
            patch.object(ws.A, "ee_pos", lambda: [0.30, 0.0, 0.22]),
            patch.object(ws.A, "ee_rpy", lambda: [0.0, 0.0, 0.0]),
            patch.object(ws.A, "check_ik", check_ik),
            patch.object(ws.A, "check_ik_path", None),
            patch.object(ws.A, "X_MIN", 0.15),
            patch.object(ws.A, "X_MAX", 0.44),
            patch.object(ws.A, "Y_MIN", -0.24),
            patch.object(ws.A, "Y_MAX", 0.24),
            patch.object(ws.A, "Z_MIN", 0.01),
            patch.object(ws.A, "Z_MAX", 0.30),
            patch.object(ws.A, "MAX_XY_STEP", 0.009),
            patch.object(ws.A, "MAX_Z_STEP", 0.009),
            patch.object(ws, "APPROACH_WORKSPACE_MARGIN_M", 0.015),
            patch.object(ws, "APPROACH_TRAJECTORY_CAP_FRACTION", 0.70),
        ):
            waypoints, reason = ws._plan_fixed_approach_trajectory([0.35, 0.02, 0.20])

        self.assertIsNotNone(waypoints, reason)
        self.assertEqual(len(checked), len(waypoints))
        self.assertEqual(waypoints[-1], (0.35, 0.02, 0.20))
        previous = np.array([0.30, 0.0, 0.22])
        for waypoint in waypoints:
            delta = np.abs(np.asarray(waypoint) - previous)
            self.assertLessEqual(float(np.max(delta)), 0.009 * 0.70 + 1e-9)
            previous = np.asarray(waypoint)

    def test_bounded_rpy_plan_uses_two_degree_ik_checked_steps(self):
        checked = []

        def check_ik(x, y, z, roll, pitch, yaw):
            checked.append((x, y, z, roll, pitch, yaw))
            return ws.A.MoveResult(True, "ik_ok", (x, y, z))

        with (
            patch.object(ws.A, "ee_pos", lambda: [0.30, 0.0, 0.25]),
            patch.object(ws.A, "ee_rpy", lambda: [0.0, 0.0, 0.0]),
            patch.object(ws.A, "check_ik", check_ik),
            patch.object(ws, "GEOMETRY_MAX_YAW_DELTA_DEG", 10.0),
            patch.object(ws, "GEOMETRY_RPY_STEP_DEG", 2.0),
        ):
            plan, reason = ws._plan_rpy_trajectory(
                (0.0, 0.0, np.deg2rad(8.0))
            )

        self.assertIsNotNone(plan, reason)
        self.assertEqual(len(plan), 4)
        self.assertEqual(len(checked), 4)
        self.assertAlmostEqual(np.rad2deg(plan[-1][2]), 8.0)

    def test_bounded_rpy_plan_refuses_orientation_beyond_clamp(self):
        with (
            patch.object(ws.A, "ee_pos", lambda: [0.30, 0.0, 0.25]),
            patch.object(ws.A, "ee_rpy", lambda: [0.0, 0.0, 0.0]),
            patch.object(ws.A, "check_ik",
                         side_effect=AssertionError("IK should not be called")),
            patch.object(ws, "GEOMETRY_MAX_YAW_DELTA_DEG", 10.0),
        ):
            plan, reason = ws._plan_rpy_trajectory(
                (0.0, 0.0, np.deg2rad(12.0))
            )

        self.assertIsNone(plan)
        self.assertIn("exceeds", reason)

    def test_bounded_rpy_plan_snaps_tracking_error_to_unchanged_clamp(self):
        checked = []

        def check_ik(x, y, z, roll, pitch, yaw):
            checked.append((roll, pitch, yaw))
            return ws.A.MoveResult(True, "ik_ok", (x, y, z))

        with (
            patch.dict(os.environ, {
                "REBOT_WRIST_RPY_BOUNDARY_SNAP_DEG": "0.10",
            }),
            patch.object(ws.A, "ee_pos", lambda: [0.30, 0.0, 0.25]),
            patch.object(ws.A, "ee_rpy", lambda: [0.0, 0.0, 0.0]),
            patch.object(ws.A, "check_ik", check_ik),
            patch.object(ws, "GEOMETRY_MAX_YAW_DELTA_DEG", 20.0),
            patch.object(ws, "GEOMETRY_RPY_STEP_DEG", 2.0),
        ):
            plan, reason = ws._plan_rpy_trajectory(
                (0.0, 0.0, np.deg2rad(20.04))
            )

        self.assertIsNotNone(plan, reason)
        self.assertAlmostEqual(np.rad2deg(plan[-1][2]), 20.0)
        self.assertLessEqual(max(abs(np.rad2deg(v[2])) for v in checked), 20.0)
        self.assertIn("snapped to unchanged 20.0deg", reason)

    def test_bounded_rpy_plan_rejects_more_than_tracking_snap(self):
        with (
            patch.dict(os.environ, {
                "REBOT_WRIST_RPY_BOUNDARY_SNAP_DEG": "0.10",
            }),
            patch.object(ws.A, "ee_pos", lambda: [0.30, 0.0, 0.25]),
            patch.object(ws.A, "ee_rpy", lambda: [0.0, 0.0, 0.0]),
            patch.object(ws.A, "check_ik",
                         side_effect=AssertionError("IK should not be called")),
            patch.object(ws, "GEOMETRY_MAX_YAW_DELTA_DEG", 20.0),
        ):
            plan, reason = ws._plan_rpy_trajectory(
                (0.0, 0.0, np.deg2rad(20.11))
            )

        self.assertIsNone(plan)
        self.assertIn("20.11deg", reason)

    def test_rpy_execution_uses_one_continuous_watchdog_bounded_path(self):
        calls = []

        def continuous(poses):
            calls.append(poses)
            return ws.A.MoveResult(
                True, "continuous_pose_minimum_jerk_arrived", poses[-1][:3]
            )

        logs = []
        waypoints = [
            (0.0, 0.0, np.deg2rad(2.0)),
            (0.0, 0.0, np.deg2rad(4.0)),
            (0.0, 0.0, np.deg2rad(6.0)),
        ]
        with patch.object(ws.A, "safe_pose_path", side_effect=continuous):
            ok, reason = ws._execute_rpy_trajectory(
                waypoints, logs.append, fixed_xyz=[0.30, 0.0, 0.25]
            )

        self.assertTrue(ok, reason)
        self.assertEqual(len(calls), 1)
        self.assertEqual(len(calls[0]), 3)
        self.assertEqual(calls[0][0][:3], (0.30, 0.0, 0.25))
        self.assertAlmostEqual(calls[0][-1][-1], np.deg2rad(6.0))
        self.assertTrue(any("watchdog_s=" in line for line in logs))

    def test_safe_pose_path_enters_local_controller_once(self):
        calls = []
        endpoint = (0.30, 0.0, 0.25, 0.0, 0.0, np.deg2rad(4.0))

        class FakeLocal:
            def move_pose_path(self, poses, duration, send_freq):
                calls.append((poses, duration, send_freq))
                return {
                    "end_effector": {
                        "position_m": list(endpoint[:3]),
                        "rpy_rad": list(endpoint[3:]),
                    },
                    "joint_vel_rad_s": [0.0] * 6,
                }

        with (
            patch.object(ws.A, "_local_arm", return_value=FakeLocal()),
            patch.object(ws.A, "ee_pos", return_value=[0.30, 0.0, 0.25]),
            patch.object(ws.A, "ee_rpy", return_value=[0.0, 0.0, 0.0]),
            patch.object(ws.A, "in_bounds", return_value=True),
        ):
            result = ws.A.safe_pose_path([
                (0.30, 0.0, 0.25, 0.0, 0.0, np.deg2rad(2.0)),
                endpoint,
            ])

        self.assertTrue(result.ok, result.reason)
        self.assertEqual(len(calls), 1)
        self.assertEqual(len(calls[0][0]), 2)
        self.assertGreaterEqual(calls[0][1], ws.A.MIN_MOVE_S)

    def test_look_around_observes_after_orientation_before_any_retreat(self):
        """A clipped target must not force the singular level-wrist retreat."""
        detection = SimpleNamespace(
            box=(100, 100, 200, 200), z_mm=200.0, source="sam3"
        )
        frame = (np.zeros((480, 640, 3), dtype=np.uint8), None, None)
        logs = []
        with (
            patch.object(ws.A, "ee_pos", return_value=[0.301, 0.0, 0.249]),
            patch.object(ws.A, "ee_rpy", return_value=[0.0, 0.0, 0.0]),
            patch.object(ws, "_plan_rpy_trajectory",
                         return_value=([(0.0, 0.0, np.deg2rad(12.0))], "ok")),
            patch.object(ws, "_execute_rpy_trajectory",
                         return_value=(True, "ok")),
            patch.object(ws, "_plan_fixed_approach_trajectory",
                         side_effect=AssertionError(
                             "first look-around view must not translate")),
            patch.object(ws, "_settle"),
            patch.object(ws, "_observe",
                         return_value=(detection, frame, "found")),
        ):
            found, found_frame, geometry, detail = ws._look_around_object(
                SimpleNamespace(), "yellow sponge", logs.append,
                require_grasp=False,
            )

        self.assertIs(found, detection)
        self.assertIs(found_frame, frame)
        self.assertIsNone(geometry)
        self.assertTrue(detail["ok"])
        self.assertEqual(detail["pose"], "yaw_left")
        self.assertTrue(any("status=observing" in line for line in logs))

    def test_look_around_retreats_only_after_pitched_views(self):
        partial = SimpleNamespace(
            box=(0, 100, 200, 200), z_mm=200.0, source="sam3"
        )
        complete = SimpleNamespace(
            box=(100, 100, 200, 200), z_mm=200.0, source="sam3"
        )
        frame = (np.zeros((480, 640, 3), dtype=np.uint8), None, None)
        observations = iter([
            (partial, frame, "found"),
            (partial, frame, "found"),
            (partial, frame, "found"),
            (complete, frame, "found"),
        ])
        translations = []

        def execute_translation(path, _log, fixed_rpy=None):
            translations.append((path, fixed_rpy))
            return True, "ok"

        with (
            patch.object(ws.A, "ee_pos", return_value=[0.301, 0.0, 0.249]),
            patch.object(ws.A, "ee_rpy", return_value=[0.0, 0.0, 0.0]),
            patch.object(ws, "_plan_rpy_trajectory",
                         return_value=([(0.0, 0.0, 0.0)], "ok")),
            patch.object(ws, "_execute_rpy_trajectory",
                         return_value=(True, "ok")),
            patch.object(ws, "_plan_fixed_approach_trajectory",
                         return_value=([(0.261, 0.0, 0.249)], "ok")),
            patch.object(ws, "_execute_fixed_approach_trajectory",
                         side_effect=execute_translation),
            patch.object(ws, "_settle"),
            patch.object(ws, "_observe", side_effect=lambda *_: next(observations)),
        ):
            found, _frame, _geometry, detail = ws._look_around_object(
                SimpleNamespace(), "yellow sponge", lambda _line: None,
                require_grasp=False,
            )

        self.assertIs(found, complete)
        self.assertEqual(detail["pose"], "pitched_retreat")
        self.assertEqual(len(translations), 1)
        self.assertAlmostEqual(
            np.rad2deg(translations[0][1][1]), 8.0, places=3
        )

    def test_look_around_rejects_one_lucky_reference_frame(self):
        """A boundary flicker must not prevent trying the next safe view."""
        frame = (np.zeros((480, 640, 3), dtype=np.uint8), None, None)

        def detection(x, *, partial):
            return SimpleNamespace(
                box=(100, 100, 200, 200), z_mm=200.0, source="sam3",
                p_cam_mm=(x, 20.0, 200.0), partial=partial,
            )

        first_lucky = detection(100.0, partial=False)
        clipped = [detection(100.0 + index, partial=True)
                   for index in range(3)]
        stable_a = detection(110.0, partial=False)
        stable_b = detection(111.0, partial=False)
        observations = [
            (item, frame, "found")
            for item in [first_lucky, *clipped, stable_a, stable_b]
        ]
        logs = []
        with (
            patch.object(ws.A, "ee_pos", return_value=[0.301, 0.0, 0.249]),
            patch.object(ws.A, "ee_rpy", return_value=[0.0, 0.0, 0.0]),
            patch.object(ws, "_plan_rpy_trajectory",
                         return_value=([(0.0, 0.0, 0.0)], "ok")),
            patch.object(ws, "_execute_rpy_trajectory",
                         return_value=(True, "ok")),
            patch.object(ws, "_settle"),
            patch.object(ws, "_observe", side_effect=observations),
            patch.object(ws, "_detection_partial",
                         side_effect=lambda det, _frame: (
                             "clipped at image bottom" if det.partial else None
                         )),
            patch.object(ws.time, "sleep"),
        ):
            found, _frame, _geometry, detail = ws._look_around_object(
                SimpleNamespace(), "blue marker", logs.append,
                require_grasp=False,
                observation_consensus_frames=3,
                observation_consensus_max_spread_mm=15.0,
            )

        self.assertIs(found, stable_a)
        self.assertEqual(detail["pose"], "yaw_right")
        self.assertEqual(
            detail["observation_consensus"]["stable_frames"], 2
        )
        self.assertTrue(any(
            "name=yaw_left status=partial" in line for line in logs
        ))
        self.assertTrue(any(
            "name=yaw_right status=accepted" in line for line in logs
        ))

    def test_placement_look_around_skips_view_outside_camera_mapping(self):
        detection = SimpleNamespace(
            box=(100, 100, 200, 200), z_mm=200.0, source="sam3",
            p_cam_mm=(10.0, 20.0, 200.0),
        )
        frame = (np.zeros((480, 640, 3), dtype=np.uint8), None, None)
        logs = []
        with (
            patch.object(ws.A, "ee_pos", return_value=[0.301, 0.0, 0.249]),
            patch.object(ws.A, "ee_rpy", return_value=[0.0, 0.0, 0.0]),
            patch.object(ws, "_plan_rpy_trajectory",
                         return_value=([(0.0, 0.0, 0.0)], "ok")),
            patch.object(ws, "_execute_rpy_trajectory",
                         return_value=(True, "ok")),
            patch.object(ws, "_settle"),
            patch.object(ws, "_observe", return_value=(
                detection, frame, "found"
            )),
            patch.object(ws, "load_saved_camera_jacobian", side_effect=[
                (None, {"reason": "31deg exceeds calibrated envelope"}),
                (np.eye(3) * 1000.0, {"reason": "calibrated"}),
            ]),
        ):
            found, found_frame, geometry, detail = ws._look_around_object(
                SimpleNamespace(), "blue marker", logs.append,
                require_grasp=False,
                require_camera_mapping=True,
            )

        self.assertIs(found, detection)
        self.assertIs(found_frame, frame)
        self.assertIsNone(geometry)
        self.assertEqual(detail["pose"], "yaw_right")
        self.assertTrue(any(
            "look_around_mapping_gate name=yaw_left" in line for line in logs
        ))

    def test_placement_reference_edge_recenter_is_bounded_and_stable(self):
        """A clipped reference may reframe sideways/away/up, never forward/down."""
        intr = (500.0, 500.0, 320.0, 240.0)
        frame = (
            np.zeros((480, 640, 3), dtype=np.uint8),
            np.full((480, 640), 200.0, dtype=np.float32),
            intr,
        )
        clipped = SimpleNamespace(
            box=(580, 180, 639, 300), uv=(639.0, 240.0), z_mm=200.0,
            p_cam_mm=(127.6, 0.0, 200.0), source="sam3",
        )
        complete_a = SimpleNamespace(
            box=(260, 180, 380, 300), uv=(320.0, 240.0), z_mm=200.0,
            p_cam_mm=(0.0, 0.0, 200.0), source="sam3",
        )
        complete_b = SimpleNamespace(
            box=(261, 180, 381, 300), uv=(321.0, 240.0), z_mm=200.0,
            p_cam_mm=(0.4, 0.0, 200.0), source="sam3",
        )
        planned_endpoints = []

        def plan(endpoint, **_kwargs):
            planned_endpoints.append(np.asarray(endpoint, dtype=float))
            return [tuple(endpoint)], "ok"

        with (
            patch.object(ws.A, "ee_pos", return_value=[0.30, 0.01, 0.25]),
            patch.object(ws.A, "ee_rpy", return_value=[0.0, 0.0, 0.0]),
            patch.object(ws, "load_saved_camera_jacobian",
                         return_value=(np.eye(3) * 1000.0, {"reason": "ok"})),
            patch.object(ws, "_plan_fixed_approach_trajectory",
                         side_effect=plan),
            patch.object(ws, "_execute_fixed_approach_trajectory",
                         return_value=(True, "ok")),
            patch.object(ws, "_settle"),
            patch.object(ws, "_observe", side_effect=[
                (complete_a, frame, "found"),
                (complete_b, frame, "found"),
            ]),
            patch.object(ws, "_detection_partial", return_value=None),
            patch.object(ws.time, "sleep"),
        ):
            found, found_frame, detail = \
                ws._recenter_clipped_placement_reference(
                    SimpleNamespace(), "green marker", clipped, frame,
                    lambda _line: None,
                    observation_consensus_frames=3,
                    observation_consensus_max_spread_mm=15.0,
                )

        self.assertIs(found, complete_a)
        self.assertIs(found_frame, frame)
        self.assertTrue(detail["ok"])
        self.assertEqual(detail["observation_consensus"]["stable_frames"], 2)
        self.assertEqual(len(planned_endpoints), 1)
        delta = planned_endpoints[0] - np.array([0.30, 0.01, 0.25])
        self.assertLessEqual(delta[0], 1e-9)
        self.assertGreaterEqual(delta[2], -1e-9)
        self.assertLessEqual(np.linalg.norm(delta), 0.045001)

    def test_placement_reference_recenter_reobserves_between_bounded_legs(self):
        intr = (500.0, 500.0, 320.0, 240.0)
        frame = (
            np.zeros((480, 640, 3), dtype=np.uint8),
            np.full((480, 640), 200.0, dtype=np.float32),
            intr,
        )
        clipped_right = SimpleNamespace(
            box=(580, 180, 639, 300), uv=(639.0, 240.0), z_mm=200.0,
            p_cam_mm=(127.6, 0.0, 200.0), source="sam3",
        )
        clipped_less = SimpleNamespace(
            box=(500, 180, 639, 300), uv=(570.0, 240.0), z_mm=200.0,
            p_cam_mm=(100.0, 0.0, 200.0), source="sam3",
        )
        complete_a = SimpleNamespace(
            box=(300, 180, 420, 300), uv=(360.0, 240.0), z_mm=200.0,
            p_cam_mm=(16.0, 0.0, 200.0), source="sam3",
        )
        complete_b = SimpleNamespace(
            box=(301, 180, 421, 300), uv=(361.0, 240.0), z_mm=200.0,
            p_cam_mm=(16.4, 0.0, 200.0), source="sam3",
        )
        plans = []

        def partial(det, _frame):
            return "semantic envelope is clipped at the image right edge" \
                if det in (clipped_right, clipped_less) else None

        with (
            patch.object(ws.A, "ee_pos", return_value=[0.30, 0.01, 0.25]),
            patch.object(ws.A, "ee_rpy", return_value=[0.0, 0.0, 0.0]),
            patch.object(ws, "load_saved_camera_jacobian",
                         return_value=(np.eye(3) * 1000.0, {"reason": "ok"})),
            patch.object(ws, "_plan_fixed_approach_trajectory",
                         side_effect=lambda endpoint, **_kwargs: (
                             plans.append(tuple(endpoint)) or [tuple(endpoint)], "ok"
                         )),
            patch.object(ws, "_execute_fixed_approach_trajectory",
                         return_value=(True, "ok")),
            patch.object(ws, "_settle"),
            patch.object(ws, "_observe", side_effect=[
                (clipped_less, frame, "found"),
                (clipped_less, frame, "found"),
                (clipped_less, frame, "found"),
                (clipped_less, frame, "found"),
                (clipped_less, frame, "found"),
                (clipped_less, frame, "found"),
                (complete_a, frame, "found"),
                (complete_b, frame, "found"),
            ]),
            patch.object(ws, "_detection_partial", side_effect=partial),
            patch.object(ws.time, "sleep"),
        ):
            found, _found_frame, detail = \
                ws._recenter_clipped_placement_reference(
                    SimpleNamespace(), "green marker", clipped_right, frame,
                    lambda _line: None,
                    observation_consensus_frames=3,
                    observation_consensus_max_spread_mm=15.0,
                    _remaining_corrections=2,
                )

        self.assertIs(found, complete_a)
        self.assertTrue(detail["ok"])
        self.assertEqual(len(plans), 2)

    def test_placement_reference_recenter_rejects_identity_switch_outlier(self):
        intr = (500.0, 500.0, 320.0, 240.0)
        frame = (
            np.zeros((480, 640, 3), dtype=np.uint8),
            np.full((480, 640), 250.0, dtype=np.float32),
            intr,
        )
        clipped = SimpleNamespace(
            box=(580, 180, 639, 300), uv=(639.0, 240.0), z_mm=250.0,
            p_cam_mm=(159.5, 0.0, 250.0), source="sam3",
        )
        wrong_object = SimpleNamespace(
            box=(260, 180, 380, 300), uv=(320.0, 240.0), z_mm=236.0,
            p_cam_mm=(17.0, -25.0, 236.0), source="sam3",
        )
        reference_a = SimpleNamespace(
            box=(260, 180, 380, 300), uv=(320.0, 240.0), z_mm=291.0,
            p_cam_mm=(21.0, 60.0, 291.0), source="sam3",
        )
        reference_b = SimpleNamespace(
            box=(261, 180, 381, 300), uv=(321.0, 240.0), z_mm=291.4,
            p_cam_mm=(21.2, 60.7, 291.4), source="sam3",
        )

        with (
            patch.object(ws.A, "ee_pos", return_value=[0.30, 0.01, 0.25]),
            patch.object(ws.A, "ee_rpy", return_value=[0.0, 0.0, 0.0]),
            patch.object(ws, "load_saved_camera_jacobian",
                         return_value=(np.eye(3) * 1000.0, {"reason": "ok"})),
            patch.object(ws, "_execute_fixed_approach_trajectory",
                         return_value=(True, "ok")),
            patch.object(ws, "_settle"),
            patch.object(ws, "_observe", side_effect=[
                (wrong_object, frame, "found"),
                (reference_a, frame, "found"),
                (reference_b, frame, "found"),
            ]),
            patch.object(ws, "_detection_partial", return_value=None),
            patch.object(ws.time, "sleep"),
        ):
            # Planner returns the normal (waypoints, reason) pair.
            with patch.object(
                ws, "_plan_fixed_approach_trajectory",
                return_value=([(0.26, 0.01, 0.25)], "ok"),
            ):
                found, _found_frame, detail = \
                    ws._recenter_clipped_placement_reference(
                        SimpleNamespace(), "green marker", clipped, frame,
                        lambda _line: None,
                        observation_consensus_frames=3,
                        observation_consensus_max_spread_mm=15.0,
                    )

        self.assertIn(found, (reference_a, reference_b))
        consensus = detail["observation_consensus"]
        self.assertEqual(consensus["stable_frames"], 2)
        self.assertEqual(consensus["rejected_frame_indices"], [0])

    def test_recenter_mapping_restore_rotates_back_before_using_jacobian(self):
        logs = []
        mappings = [
            (None, {"reason": "orientation outside calibration"}),
            (np.eye(3), {"reason": "ok"}),
        ]
        with (
            patch.object(ws.A, "ee_pos", return_value=[0.26, 0.0, 0.25]),
            patch.object(ws.A, "ee_rpy", side_effect=[
                [0.0, 0.14, 0.46],
                [0.0, 0.14, 0.17],
            ]),
            patch.object(ws, "load_saved_camera_jacobian",
                         side_effect=mappings),
            patch.object(ws, "_plan_rpy_trajectory",
                         return_value=([(0.0, 0.14, 0.17)], "ok")),
            patch.object(ws, "_execute_rpy_trajectory",
                         return_value=(True, "arrived")) as execute,
        ):
            ok, reason = ws._restore_camera_mapping_for_recenter(
                [0.26, 0.0, 0.25], [0.0, 0.14, 0.17], logs.append
            )

        self.assertTrue(ok, reason)
        execute.assert_called_once()
        self.assertIn("calibrated camera orientation", reason)

    def test_look_around_keeps_pose_aware_views_and_rejects_wide_section(self):
        detection = SimpleNamespace(
            box=(100, 100, 200, 200), z_mm=200.0, source="sam3"
        )
        frame_a = (np.zeros((480, 640, 3), dtype=np.uint8), None, None)
        frame_b = (np.ones((480, 640, 3), dtype=np.uint8), None, None)
        wide = SimpleNamespace(ok=True, reason="wide", grasp_detection=object())
        narrow = SimpleNamespace(ok=True, reason="narrow", grasp_detection=object())
        captured = []
        logs = []

        with (
            patch.object(ws.A, "ee_pos", return_value=[0.301, 0.0, 0.249]),
            patch.object(ws.A, "ee_rpy", return_value=[0.0, 0.0, 0.0]),
            patch.object(ws, "_plan_rpy_trajectory",
                         return_value=([(0.0, 0.0, 0.0)], "ok")),
            patch.object(ws, "_execute_rpy_trajectory",
                         return_value=(True, "ok")),
            patch.object(ws, "_settle"),
            patch.object(ws, "_observe", side_effect=[
                (detection, frame_a, "found"),
                (detection, frame_b, "found"),
            ]),
            patch.object(ws, "_plan_scene_grasp_geometry",
                         side_effect=[wide, narrow]),
            patch.object(ws, "load_saved_camera_jacobian", side_effect=[
                (np.eye(3) * 1000.0, {"reason": "ok"}),
                (np.eye(3) * 900.0, {"reason": "ok"}),
            ]),
        ):
            found, found_frame, geometry, detail = ws._look_around_object(
                SimpleNamespace(), "yellow sponge", logs.append,
                require_grasp=True,
                geometry_accept=lambda candidate: (
                    candidate is narrow,
                    "cross-section too wide" if candidate is wide else "fits",
                ),
                captured_views=captured,
            )

        self.assertIs(found, detection)
        self.assertIs(found_frame, frame_b)
        self.assertIs(geometry, narrow)
        self.assertEqual(detail["pose"], "yaw_right")
        self.assertEqual([view["name"] for view in captured],
                         ["yaw_left", "yaw_right"])
        self.assertTrue(np.allclose(captured[0]["camera_mapping"],
                                    np.eye(3) * 1000.0))
        self.assertTrue(np.allclose(captured[1]["camera_mapping"],
                                    np.eye(3) * 900.0))
        self.assertTrue(any("grasp entry rejected" in line for line in logs))

    def test_failed_grasp_can_export_pose_aware_discovery_cloud(self):
        frame = (
            np.zeros((4, 5, 3), dtype=np.uint8),
            np.full((4, 5), 200.0, dtype=np.float32),
            (100.0, 100.0, 2.5, 2.0),
        )
        views = [
            {
                "name": "yaw_left", "frame": frame,
                "ee_xyz_m": np.array([0.30, 0.01, 0.25]),
                "camera_mapping": np.eye(3) * 1000.0,
            },
            {
                "name": "pitched_retreat", "frame": frame,
                "ee_xyz_m": np.array([0.26, 0.01, 0.25]),
                "camera_mapping": np.eye(3) * 900.0,
            },
        ]
        lifted = []

        def build_cloud(*_args, **kwargs):
            lifted.append((kwargs["ee_xyz_m"],
                           kwargs["camera_mapping_mm_per_m"],
                           kwargs["jaw_target_cam_mm"]))
            return np.array([[0.3, 0.0, 0.1]]), np.array([[1, 2, 3]])

        with (
            patch.object(ws, "SCENE_COLLISION_EXPORT_VIEW", True),
            patch.object(ws, "load_tcp",
                         return_value=np.array([3.6, 4.0, 66.9])),
            patch.object(ws, "load_safety_calibration",
                         return_value={"table_z_m": 0.012}),
            patch.object(ws.SC, "build_dense_view_cloud",
                         side_effect=build_cloud),
            patch.object(ws.SC, "fuse_dense_view_clouds", return_value=(
                np.array([[0.3, 0.0, 0.1], [0.31, 0.0, 0.1]]),
                np.array([[1, 2, 3], [4, 5, 6]], dtype=np.uint8),
            )) as fuse,
            patch.object(ws.SC, "save_scene_artifacts", return_value={
                "ply": "/tmp/latest_discovery.ply",
                "rerun": "/tmp/latest_discovery.rrd",
            }) as save,
        ):
            result = ws._export_discovery_cloud(views, lambda _line: None)

        self.assertTrue(result["ok"])
        self.assertEqual(result["views"], 2)
        self.assertEqual(result["dense_rgb_points"], 2)
        self.assertEqual(len(lifted), 2)
        self.assertTrue(np.allclose(lifted[0][0], views[0]["ee_xyz_m"]))
        self.assertTrue(np.allclose(lifted[1][1],
                                    views[1]["camera_mapping"]))
        self.assertTrue(np.allclose(lifted[0][2], [3.6, 4.0, 66.9]))
        self.assertEqual(len(fuse.call_args.args[0]), 1)
        self.assertEqual(result["fused_views"], 1)
        self.assertEqual(result["rejected_views"], ["pitched_retreat"])
        self.assertEqual(save.call_args.kwargs["stem"], "latest_discovery")
        self.assertEqual(save.call_args.kwargs["named_view_clouds"], {})
        self.assertFalse(result["per_view_clouds_exported"])

    def test_discovery_target_box_ignores_rejected_geometry_view(self):
        frame = (
            np.zeros((8, 8, 3), dtype=np.uint8),
            np.full((8, 8), 220.0, dtype=np.float32),
            (100.0, 100.0, 4.0, 4.0),
        )
        target = ws.WD.Detection(
            box=(2, 2, 6, 6), uv=(4.0, 4.0), z_mm=220.0,
            p_cam_mm=(0.0, 0.0, 220.0), source="sam3",
            min_width_mm=12.0, max_width_mm=60.0,
            info={"mask_u8": np.ones((8, 8), dtype=np.uint8) * 255},
        )
        rejected = SimpleNamespace(
            ok=False, envelope_detection=target, grasp_detection=target,
        )
        accepted = SimpleNamespace(
            ok=True, envelope_detection=target, grasp_detection=target,
        )
        views = [
            {"name": "rejected", "frame": frame, "geometry": rejected,
             "ee_xyz_m": np.array([0.30, 0.0, 0.25]),
             "camera_mapping": np.eye(3) * 1000.0},
            {"name": "accepted", "frame": frame, "geometry": accepted,
             "ee_xyz_m": np.array([0.30, 0.0, 0.25]),
             "camera_mapping": np.eye(3) * 1000.0},
        ]
        cloud = (np.array([[0.30, 0.0, 0.10]]),
                 np.array([[1, 2, 3]], dtype=np.uint8))
        reports = [
            {"valid_for_fusion": True, "translation_mm": [0.0, 0.0, 0.0]},
            {"valid_for_fusion": True, "translation_mm": [0.0, 0.0, 0.0]},
        ]
        fitted = SimpleNamespace(as_dict=lambda: {
            "views": 1,
            "observed_size_m": [0.01, 0.02, 0.06],
            "orientation_mode": "test",
        })
        with (
            patch.object(ws, "load_tcp", return_value=np.zeros(3)),
            patch.object(ws, "load_safety_calibration",
                         return_value={"table_z_m": 0.012}),
            patch.object(ws.SC, "build_dense_view_cloud", return_value=cloud),
            patch.object(ws.SC, "crop_dense_view_clouds",
                         side_effect=lambda clouds, **_kwargs: clouds),
            patch.object(ws.SC, "register_dense_view_clouds",
                         return_value=([cloud, cloud], reports)),
            patch.object(ws.SC, "fuse_dense_view_clouds", return_value=cloud),
            patch.object(ws.OG3, "masked_points_base",
                         return_value=np.ones((30, 3))) as masked,
            patch.object(ws.OG3, "fit_tabletop_box", return_value=fitted),
        ):
            result = ws._export_discovery_cloud(
                views, lambda _line: None, force_geometry=True
            )

        self.assertTrue(result["ok"])
        self.assertEqual(result["object_box_3d"]["views"], 1)
        masked.assert_called_once()

    def test_metric_recovery_uses_native_mask_from_failed_2d_grasp(self):
        frame = (
            np.zeros((8, 8, 3), dtype=np.uint8),
            np.full((8, 8), 220.0, dtype=np.float32),
            (100.0, 100.0, 4.0, 4.0),
        )
        mask = np.ones((8, 8), dtype=np.uint8) * 255
        target = ws.WD.Detection(
            box=(2, 2, 6, 6), uv=(4.0, 4.0), z_mm=220.0,
            p_cam_mm=(0.0, 0.0, 220.0), source="sam3",
            min_width_mm=12.0, max_width_mm=77.0,
            info={
                "mask_u8": mask,
                "segmentation_source": "sam3:native_mask",
                "identifier_source": "sam3",
            },
        )
        envelope = SimpleNamespace(
            source="rgbd:semantic_envelope",
            info={
                "mask_u8": mask,
                "segmentation_source": "sam3:native_mask",
                "identifier_source": "sam3",
            },
        )
        geometry = SimpleNamespace(
            ok=False, envelope_detection=envelope, grasp_detection=target,
        )
        views = [{
            "name": "complete_oblique_native_mask", "frame": frame,
            "detection": target, "geometry": geometry,
            "ee_xyz_m": np.array([0.30, 0.0, 0.25]),
            "camera_mapping": np.eye(3) * 1000.0,
        }]
        cloud = (
            np.array([[0.30, 0.0, 0.10]]),
            np.array([[1, 2, 3]], dtype=np.uint8),
        )
        reports = [{
            "valid_for_fusion": True, "translation_mm": [0.0, 0.0, 0.0]
        }]
        fitted = SimpleNamespace(as_dict=lambda: {
            "views": 1,
            "observed_size_m": [0.01, 0.02, 0.06],
            "orientation_mode": "test",
        })
        with (
            patch.object(ws, "load_tcp", return_value=np.zeros(3)),
            patch.object(ws, "load_safety_calibration",
                         return_value={"table_z_m": 0.012}),
            patch.object(ws.SC, "build_dense_view_cloud", return_value=cloud),
            patch.object(ws.SC, "crop_dense_view_clouds",
                         side_effect=lambda clouds, **_kwargs: clouds),
            patch.object(ws.SC, "register_dense_view_clouds",
                         return_value=([cloud], reports)),
            patch.object(ws.SC, "fuse_dense_view_clouds", return_value=cloud),
            patch.object(ws.OG3, "masked_points_base",
                         return_value=np.ones((30, 3))) as masked,
            patch.object(ws.OG3, "fit_tabletop_box", return_value=fitted),
        ):
            result = ws._export_discovery_cloud(
                views, lambda _line: None, force_geometry=True
            )

        self.assertTrue(result["ok"])
        self.assertIsNotNone(result["object_box_3d"])
        masked.assert_called_once()

    def test_feature_offset_reacquires_same_section_when_axis_sign_flips(self):
        frame = (
            np.zeros((100, 100, 3), dtype=np.uint8),
            np.full((100, 100), 200.0, dtype=np.float32),
            (100.0, 100.0, 50.0, 50.0),
        )
        feature = ws.WD.Detection(
            box=(35, 40, 65, 60), uv=(50.0, 50.0), z_mm=200.0,
            p_cam_mm=(0.0, 0.0, 200.0), source="sam3",
            min_width_mm=20.0, max_width_mm=80.0,
        )
        # The fresh mask flipped its modulo-180 long-axis sign.  The physical
        # section at u=40 is now labelled +0.32, while signed-fraction matching
        # alone would incorrectly select the opposite end at u=60.
        wrong_default = ws.WD.Detection(
            box=(35, 40, 65, 60), uv=(60.0, 50.0), z_mm=200.0,
            p_cam_mm=(20.0, 0.0, 200.0), source="depth:ranked",
            min_width_mm=20.0, max_width_mm=80.0,
            info={"selected_candidate": {"axis_fraction": -0.32}},
        )
        candidates = [
            {"uv": [40.0, 50.0], "width_mm": 20.0,
             "axis_fraction": 0.32, "score": 1.0},
            {"uv": [60.0, 50.0], "width_mm": 20.0,
             "axis_fraction": -0.32, "score": 1.0},
        ]
        geometry = ws.GG.GraspGeometry(
            True, "ok", wrong_default, feature, (0.0, 0.0, 0.0),
            candidates=candidates,
        )

        selected = ws._geometry_near_feature_offset(
            geometry, frame, feature, [20.0, 0.0, 0.0],
            axis_fraction=-0.32,
        )

        self.assertIsNotNone(selected)
        self.assertEqual(selected.grasp_detection.uv, (40.0, 50.0))
        self.assertAlmostEqual(selected.grasp_detection.p_cam_mm[0], -20.0)

    def test_feature_offset_reacquisition_fails_closed_without_section(self):
        frame = (
            np.zeros((20, 20, 3), dtype=np.uint8),
            np.full((20, 20), 200.0, dtype=np.float32),
            (100.0, 100.0, 10.0, 10.0),
        )
        feature = ws.WD.Detection(
            box=(1, 1, 18, 18), uv=(10.0, 10.0), z_mm=200.0,
            p_cam_mm=(0.0, 0.0, 200.0), source="sam3",
            min_width_mm=20.0, max_width_mm=80.0,
        )
        default = ws.WD.Detection(
            box=(1, 1, 18, 18), uv=(10.0, 10.0), z_mm=200.0,
            p_cam_mm=(0.0, 0.0, 200.0), source="depth:ranked",
            min_width_mm=20.0, max_width_mm=80.0,
        )
        geometry = ws.GG.GraspGeometry(
            True, "ok", default, feature, (0.0, 0.0, 0.0), candidates=[]
        )

        with patch.dict(os.environ, {
            "REBOT_GRASP_SECTION_MATCH_TOL_MM": "nan",
        }):
            selected = ws._geometry_near_feature_offset(
                geometry, frame, feature, [25.0, 0.0, 0.0],
                axis_fraction=0.32,
            )

        self.assertIsNone(selected)

    def test_feature_offset_reacquisition_rejects_nonfinite_body(self):
        frame = (
            np.zeros((20, 20, 3), dtype=np.uint8),
            np.full((20, 20), 200.0, dtype=np.float32),
            (100.0, 100.0, 10.0, 10.0),
        )
        feature = ws.WD.Detection(
            box=(1, 1, 18, 18), uv=(10.0, 10.0), z_mm=200.0,
            p_cam_mm=(0.0, 0.0, 200.0), source="sam3",
            min_width_mm=20.0, max_width_mm=80.0,
        )
        invalid_body = ws.WD.Detection(
            box=(1, 1, 18, 18), uv=(10.0, 10.0), z_mm=200.0,
            p_cam_mm=(float("nan"), 0.0, 200.0), source="depth:ranked",
            min_width_mm=20.0, max_width_mm=80.0,
        )
        geometry = ws.GG.GraspGeometry(
            True, "ok", invalid_body, feature, (0.0, 0.0, 0.0),
            candidates=[],
        )

        selected = ws._geometry_near_feature_offset(
            geometry, frame, feature, [0.0, 0.0, 0.0]
        )

        self.assertIsNone(selected)

    def test_normal_grasp_uses_fresh_checkpoint_section_without_strict_lock(self):
        frame = (
            np.zeros((20, 20, 3), dtype=np.uint8),
            np.full((20, 20), 200.0, dtype=np.float32),
            (100.0, 100.0, 10.0, 10.0),
        )
        feature = ws.WD.Detection(
            box=(1, 1, 18, 18), uv=(10.0, 10.0), z_mm=200.0,
            p_cam_mm=(0.0, 0.0, 200.0), source="sam3",
            min_width_mm=20.0, max_width_mm=80.0,
        )
        fresh_body = ws.WD.Detection(
            box=(1, 1, 18, 18), uv=(10.0, 10.0), z_mm=200.0,
            p_cam_mm=(0.0, 0.0, 200.0), source="depth:ranked",
            min_width_mm=20.0, max_width_mm=80.0,
        )
        geometry = ws.GG.GraspGeometry(
            True, "fresh", fresh_body, feature, (0.0, 0.0, 0.0),
            candidates=[],
        )

        normal = ws._fresh_geometry_for_selected_section(
            geometry, frame, feature, [25.0, 0.0, 0.0],
            section_locked=False,
        )
        alternate = ws._fresh_geometry_for_selected_section(
            geometry, frame, feature, [25.0, 0.0, 0.0],
            section_locked=True, axis_fraction=0.25,
        )

        self.assertIs(normal, geometry)
        self.assertIsNone(alternate)

    def test_endpoint_section_identity_locks_fresh_physical_body(self):
        feature = ws.WD.Detection(
            box=(1, 1, 18, 18), uv=(10.0, 10.0), z_mm=210.0,
            p_cam_mm=(4.0, 30.0, 74.0), source="sam3",
            min_width_mm=20.0, max_width_mm=80.0,
        )
        body = ws.WD.Detection(
            box=(1, 1, 18, 18), uv=(9.0, 11.0), z_mm=201.0,
            p_cam_mm=(2.0, 25.0, 65.0), source="depth:ranked",
            min_width_mm=20.0, max_width_mm=80.0,
            info={"selected_candidate": {"axis_fraction": -0.16}},
        )
        geometry = ws.GG.GraspGeometry(
            True, "fresh", body, feature, (0.0, 0.0, 0.0), candidates=[]
        )

        offset, axis_fraction, detail = ws._endpoint_section_identity({
            "detection": feature,
            "geometry": geometry,
        })

        self.assertTrue(np.allclose(offset, [2.0, 5.0, 9.0]))
        self.assertAlmostEqual(axis_fraction, -0.16)
        self.assertIn("locked", detail["reason"])

    def test_endpoint_section_identity_rejects_nonfinite_geometry(self):
        feature = SimpleNamespace(p_cam_mm=(0.0, 0.0, 70.0))
        body = SimpleNamespace(
            p_cam_mm=(0.0, float("nan"), 65.0), info={}
        )
        geometry = SimpleNamespace(grasp_detection=body)

        offset, axis_fraction, detail = ws._endpoint_section_identity({
            "detection": feature,
            "geometry": geometry,
        })

        self.assertIsNone(offset)
        self.assertIsNone(axis_fraction)
        self.assertIn("non-finite", detail["reason"])

    def test_checkpoint_reselects_stable_fresh_body_when_lock_disappears(self):
        fresh = [
            {"body_cam_mm": [10.0, 20.0, 100.0], "name": "a"},
            {"body_cam_mm": [10.4, 20.1, 100.2], "name": "b"},
            {"body_cam_mm": [35.0, 50.0, 130.0], "name": "outlier"},
        ]

        body, detail, selected = ws._checkpoint_section_consensus(
            [], fresh, section_locked=True,
            min_frames=2, max_spread_mm=4.0,
        )

        self.assertIsNotNone(body, detail)
        self.assertTrue(detail["section_reselected"])
        self.assertIs(selected, fresh)
        self.assertTrue(np.allclose(body, [10.2, 20.05, 100.1]))

    def test_checkpoint_does_not_reselect_without_explicit_section_lock(self):
        fresh = [
            {"body_cam_mm": [10.0, 20.0, 100.0]},
            {"body_cam_mm": [10.2, 20.1, 100.1]},
        ]

        body, detail, selected = ws._checkpoint_section_consensus(
            [], fresh, section_locked=False,
            min_frames=2, max_spread_mm=4.0,
        )

        self.assertIsNone(body)
        self.assertIn("only 0/2", detail["reason"])
        self.assertEqual(selected, [])

    def test_endpoint_measurement_reselects_stable_fresh_close_geometry(self):
        def samples(widths, offset=0.0):
            return [
                {
                    "width_mm": width,
                    "body_cam_mm": [10.0 + offset, 20.0, 100.0 + 0.1 * i],
                    "token": i,
                }
                for i, width in enumerate(widths)
            ]

        result = ws._endpoint_measurement_consensus(
            [], samples([20.8, 20.9, 21.0, 20.9, 20.8]),
            20.0, 22.0, section_locked=True,
        )

        self.assertIsNotNone(result["selected"], result)
        self.assertIsNotNone(result["body_cam"], result)
        self.assertTrue(result["section_reselected"])
        self.assertIn("stable fresh close-range", result["width_detail"]["reason"])

    def test_endpoint_measurement_will_not_reselect_unstable_fresh_width(self):
        fresh = [
            {
                "width_mm": width,
                "body_cam_mm": [10.0, 20.0, 100.0 + 0.1 * index],
            }
            for index, width in enumerate((12.0, 20.0, 28.0, 36.0, 44.0))
        ]

        result = ws._endpoint_measurement_consensus(
            [], fresh, 20.0, 30.0, section_locked=True,
        )

        self.assertIsNone(result["selected"])
        self.assertFalse(result["section_reselected"])

    def test_ready_geometry_fallback_rejects_repeated_jaw_inflation_only(self):
        body = ws.WD.Detection(
            box=(2, 2, 18, 18), uv=(10.0, 10.0), z_mm=100.0,
            p_cam_mm=(10.0, 20.0, 100.0), source="depth:ranked",
            min_width_mm=14.5, max_width_mm=80.0,
        )
        geometry = ws.GG.GraspGeometry(
            True, "stable ready geometry", body, body,
            (0.0, 0.0, 0.0), candidates=[],
        )
        jaw_plan = {
            "ok": True, "min_width_mm": 14.5, "max_width_mm": 80.0,
            "max_aperture_mm": 50.0, "preopen_mm": 50.0,
            "grasp_mm": 11.5, "grasp_rad": 0.6,
        }
        samples = [
            {
                "width_mm": width,
                "body_cam_mm": [10.0, 20.0, 100.0 + 0.1 * index],
                "detection": body, "frame": (None, None, None),
            }
            for index, width in enumerate((24.6, 26.4, 34.2))
        ]
        measurement = {
            "samples": samples,
            "body_cam": np.array([10.0, 20.0, 100.1]),
            "body_detail": {"reason": "3 stable body frames"},
        }
        with patch.object(
            ws, "_validate_object_gripper_fit",
            return_value=(True, {"reason": "fits"}),
        ):
            result = ws._ready_geometry_endpoint_fallback(
                measurement, jaw_plan, geometry, 14.6, None,
            )

        self.assertTrue(result["ok"], result)
        self.assertEqual(result["selected"]["width_mm"], 14.5)
        self.assertTrue(np.allclose(
            result["selected"]["geometry"].grasp_detection.p_cam_mm,
            [10.0, 20.0, 100.1],
        ))
        self.assertIn("jaw occlusion", result["reason"])

    def test_ready_geometry_fallback_refuses_mixed_endpoint_widths(self):
        body = ws.WD.Detection(
            box=(2, 2, 18, 18), uv=(10.0, 10.0), z_mm=100.0,
            p_cam_mm=(10.0, 20.0, 100.0), source="depth:ranked",
            min_width_mm=14.5, max_width_mm=80.0,
        )
        geometry = ws.GG.GraspGeometry(
            True, "stable ready geometry", body, body,
            (0.0, 0.0, 0.0), candidates=[],
        )
        measurement = {
            "samples": [
                {"width_mm": width, "body_cam_mm": [10.0, 20.0, 100.0]}
                for width in (15.0, 26.0, 34.0)
            ],
            "body_cam": np.array([10.0, 20.0, 100.0]),
        }
        result = ws._ready_geometry_endpoint_fallback(
            measurement,
            {"min_width_mm": 14.5}, geometry, 14.6, None,
        )

        self.assertFalse(result["ok"])
        self.assertIn("not consistently inflated", result["reason"])

    def test_collision_aware_grasp_uses_measured_jaw_slack(self):
        frame = (
            np.zeros((20, 20, 3), dtype=np.uint8),
            np.full((20, 20), 200.0, dtype=np.float32),
            (100.0, 100.0, 10.0, 10.0),
        )
        body = ws.WD.Detection(
            box=(2, 2, 18, 18), uv=(10.0, 10.0), z_mm=200.0,
            p_cam_mm=(0.0, 0.0, 200.0), source="depth:ranked",
            min_width_mm=20.0, max_width_mm=80.0,
        )
        geometry = ws.GG.GraspGeometry(
            True, "ok", body, body, (0.0, 0.0, 0.0), candidates=[]
        )
        jaw_plan = {
            "ok": True, "min_width_mm": 20.0, "max_width_mm": 80.0,
            "max_aperture_mm": 50.0, "preopen_mm": 50.0,
        }
        checked_endpoints = []

        def reachable(_body, _mapping, target):
            endpoint = np.asarray(target, dtype=float) / 1000.0
            return np.asarray(target, dtype=float), {
                "full_hover": True,
                "requested_endpoint_m": endpoint.tolist(),
                "reason": "full",
            }

        def endpoint_gate(x, y, z, *_rpy):
            checked_endpoints.append((x, y, z))
            return ws.A.MoveResult(x > 0.0, "clear" if x > 0.0 else "blocked")

        with (
            patch.dict(os.environ, {
                "REBOT_WRIST_JAW_LONG_AXIS_CAMERA_DEG": "-90",
                "REBOT_GRASP_POSE_SLOT_STEP_MM": "5",
                "REBOT_GRASP_POSE_MAX_CANDIDATES": "10",
            }),
            patch.object(ws, "_jaw_slot_fit", return_value=(True, {
                "across_limit_mm": 10.0,
            })),
            patch.object(ws, "_plan_aperture", return_value=jaw_plan),
            patch.object(ws, "_validate_object_gripper_fit",
                         return_value=(True, {"reason": "fits"})),
            patch.object(ws, "_reachable_approach_target",
                         side_effect=reachable),
            patch.object(ws.A, "check_pose_clearance",
                         side_effect=endpoint_gate),
            patch.object(ws, "_plan_clearance_axial_trajectory",
                         return_value=([(0.1, 0.0, 0.1)], "route", {
                             "clearance_strategy": "test",
                         })) as route,
        ):
            result = ws._plan_collision_aware_grasp_pose(
                geometry, frame, np.eye(3) * 1000.0,
                [0.0, 0.0, 70.0], (0.0, 0.0, 0.0),
                jaw_plan, None, lambda _line: None,
            )

        self.assertTrue(result["ok"], result)
        self.assertEqual(result["slot_offset_mm"], 5.0)
        self.assertTrue(np.allclose(result["requested_cam_mm"],
                                    [5.0, 0.0, 70.0]))
        self.assertEqual(len(checked_endpoints), 2)
        route.assert_called_once()

    def test_collision_search_checks_alternate_section_before_dense_offsets(self):
        frame = (
            np.zeros((20, 20, 3), dtype=np.uint8),
            np.full((20, 20), 200.0, dtype=np.float32),
            (100.0, 100.0, 10.0, 10.0),
        )
        body = ws.WD.Detection(
            box=(2, 2, 18, 18), uv=(10.0, 10.0), z_mm=200.0,
            p_cam_mm=(0.0, 0.0, 200.0), source="depth:ranked",
            min_width_mm=20.0, max_width_mm=80.0,
        )
        geometry = ws.GG.GraspGeometry(
            True, "ok", body, body, (0.0, 0.0, 0.0), candidates=[{
                "uv": [12.0, 10.0], "width_mm": 19.0,
                "axis_fraction": 0.25, "score": 2.0,
            }],
        )
        jaw_plan = {
            "ok": True, "min_width_mm": 20.0, "max_width_mm": 80.0,
            "max_aperture_mm": 50.0, "preopen_mm": 50.0,
        }
        checked_bodies = []

        def reachable(candidate, _mapping, target):
            checked_bodies.append(candidate.source)
            return np.asarray(target), {
                "full_hover": True,
                "requested_endpoint_m": [0.35, 0.0, 0.18],
                "reason": "full",
            }

        with (
            patch.dict(os.environ, {
                "REBOT_WRIST_JAW_LONG_AXIS_CAMERA_DEG": "-90",
                "REBOT_GRASP_POSE_SLOT_STEP_MM": "0.5",
                "REBOT_GRASP_POSE_MAX_CANDIDATES": "4",
            }),
            patch.object(ws, "_jaw_slot_fit", return_value=(True, {
                "across_limit_mm": 10.0,
            })),
            patch.object(ws, "_plan_aperture", return_value=jaw_plan),
            patch.object(ws, "_validate_object_gripper_fit",
                         return_value=(True, {"reason": "fits"})),
            patch.object(ws, "_reachable_approach_target",
                         side_effect=reachable),
            patch.object(ws.A, "check_pose_clearance",
                         return_value=ws.A.MoveResult(True, "clear")),
            patch.object(ws, "_plan_clearance_axial_trajectory",
                         return_value=([(0.35, 0.0, 0.18)], "route", {})),
        ):
            result = ws._plan_collision_aware_grasp_pose(
                geometry, frame, np.eye(3) * 1000.0,
                [0.0, 0.0, 70.0], (0.0, 0.0, 0.0),
                jaw_plan, None, lambda _line: None,
            )

        self.assertTrue(result["ok"], result)
        self.assertEqual(result["body_name"], "section_1")
        self.assertEqual(result["slot_offset_mm"], 0.0)
        self.assertEqual(checked_bodies,
                         ["depth:collision_ranked_cross_section"])

    def test_collision_search_uses_bounded_pitch_when_terminal_link_is_blocked(self):
        frame = (
            np.zeros((20, 20, 3), dtype=np.uint8),
            np.full((20, 20), 200.0, dtype=np.float32),
            (100.0, 100.0, 10.0, 10.0),
        )
        body = ws.WD.Detection(
            box=(2, 2, 18, 18), uv=(10.0, 10.0), z_mm=200.0,
            p_cam_mm=(0.0, 0.0, 200.0), source="depth:ranked",
            min_width_mm=20.0, max_width_mm=70.0,
        )
        geometry = ws.GG.GraspGeometry(
            True, "ok", body, body, (0.0, 0.0, 0.0), candidates=[]
        )
        jaw_plan = {
            "ok": True, "min_width_mm": 20.0, "max_width_mm": 70.0,
            "max_aperture_mm": 50.0, "preopen_mm": 50.0,
        }
        object_box = {
            "axes_base": np.eye(3).tolist(),
            "observed_size_m": [0.02, 0.07, 0.03],
            "conservative_size_m": [0.026, 0.076, 0.036],
        }
        checked_pitches = []

        def reachable(_body, _mapping, target):
            return np.asarray(target), {
                "full_hover": True,
                "requested_endpoint_m": [0.40, 0.05, 0.12],
                "reason": "full",
            }

        def endpoint_gate(_x, _y, _z, _roll, pitch, _yaw):
            checked_pitches.append(round(float(np.degrees(pitch)), 1))
            return ws.A.MoveResult(pitch >= np.radians(10.0), "gate")

        with (
            patch.dict(os.environ, {
                "REBOT_WRIST_JAW_LONG_AXIS_CAMERA_DEG": "-90",
                "REBOT_GRASP_COLLISION_PITCH_STEP_DEG": "5",
                "REBOT_GRASP_COLLISION_MAX_PITCH_DEG": "20",
            }),
            patch.object(ws, "_jaw_slot_fit", return_value=(True, {
                "across_limit_mm": 10.0,
            })),
            patch.object(ws, "_plan_aperture", return_value=jaw_plan),
            patch.object(ws, "_validate_object_gripper_fit",
                         return_value=(True, {"reason": "fits"})),
            patch.object(ws, "_reachable_approach_target",
                         side_effect=reachable),
            patch.object(ws, "load_saved_camera_jacobian",
                         return_value=(np.eye(3) * 1000.0, {
                             "reason": "rotated rigid mount",
                         })),
            patch.object(ws, "_projected_3d_jaw_span", return_value={
                "ok": True, "observed_width_mm": 22.0,
                "conservative_width_mm": 28.0, "reason": "fits",
            }),
            patch.object(ws.A, "check_pose_clearance",
                         side_effect=endpoint_gate),
            patch.object(ws, "_plan_clearance_axial_trajectory",
                         return_value=([(0.4, 0.05, 0.12)], "route", {
                             "clearance_strategy": "test",
                         })) as route,
        ):
            result = ws._plan_collision_aware_grasp_pose(
                geometry, frame, np.eye(3) * 1000.0,
                [0.0, 0.0, 70.0], (0.0, 0.0, 0.0),
                jaw_plan, object_box, lambda _line: None,
            )

        self.assertTrue(result["ok"], result)
        self.assertEqual(result["pitch_offset_deg"], 10.0)
        self.assertAlmostEqual(result["approach_rpy"][1], np.radians(10.0))
        self.assertEqual(checked_pitches, [5.0, -5.0, 10.0])
        route.assert_called_once()

    def test_multiview_restore_accepts_fresh_edge_clip_after_safe_obb_projection(self):
        frame = (
            np.zeros((20, 20, 3), dtype=np.uint8),
            np.full((20, 20), 200.0, dtype=np.float32),
            (100.0, 100.0, 10.0, 10.0),
        )
        body = ws.WD.Detection(
            box=(2, 2, 18, 18), uv=(10.0, 10.0), z_mm=200.0,
            p_cam_mm=(0.0, 0.0, 200.0), source="sam3",
            min_width_mm=22.0, max_width_mm=74.0,
        )
        validated = ws.GG.GraspGeometry(
            True, "metric candidate", body, body,
            (0.0, np.radians(8.0), 0.0), yaw_correction_deg=0.0,
        )
        noisy_fresh = ws.GG.GraspGeometry(
            False, "single frame yaw returned outside safe wrist budget", body, body,
            (0.0, np.radians(8.0), np.radians(20.0)),
            yaw_correction_deg=20.0,
        )
        object_box = {
            "axes_base": np.eye(3).tolist(),
            "observed_size_m": [0.074, 0.022, 0.057],
            "conservative_size_m": [0.080, 0.028, 0.063],
        }
        view = {
            "name": "pitched_retreat", "frame": frame,
            "detection": body, "ee_xyz_m": [0.26, 0.0, 0.25],
            "ee_rpy_rad": [0.0, np.radians(8.0), 0.0],
            "camera_mapping": np.eye(3) * 1000.0,
        }
        with (
            patch.object(ws, "_export_discovery_cloud", return_value={
                "ok": True, "object_box_3d": object_box,
            }),
            patch.object(ws, "_metric_horizontal_box_dimensions",
                         return_value=(22.0, 74.0, {"reason": "metric"})),
            patch.object(ws, "_detection_partial", side_effect=[
                None,
                "semantic/depth envelope is clipped at the image right edge",
            ]),
            patch.object(ws, "_exact_semantic_geometry_mask",
                         return_value=(True, "exact native semantic mask")),
            patch.object(ws, "_plan_scene_grasp_geometry",
                         side_effect=[validated, noisy_fresh]),
            patch.object(ws.A, "ee_pos", return_value=[0.26, 0.0, 0.25]),
            patch.object(ws.A, "ee_rpy",
                         return_value=[0.0, np.radians(8.0), 0.0]),
            patch.object(ws, "_plan_rpy_trajectory",
                         return_value=([(0.0, np.radians(8.0), 0.0)], "ok")),
            patch.object(ws, "_execute_rpy_trajectory",
                         return_value=(True, "ok")),
            patch.object(ws, "_plan_fixed_approach_trajectory",
                         return_value=([(0.26, 0.0, 0.25)], "ok")),
            patch.object(ws, "_execute_fixed_approach_trajectory",
                         return_value=(True, "ok")),
            patch.object(ws, "_settle"),
            patch.object(ws, "_observe",
                         return_value=(body, frame, "ok")),
            patch.object(ws, "_projected_3d_jaw_span", return_value={
                "ok": True, "observed_width_mm": 24.0,
                "conservative_width_mm": 32.0, "reason": "safe projection",
            }),
            patch("arm_agent.gripper_model.max_aperture_mm",
                  return_value=50.0),
        ):
            _det, _frame, result, detail = \
                ws._recover_multiview_metric_grasp(
                    object(), "yellow object", [view], lambda _line: None
                )

        self.assertTrue(detail["ok"], detail)
        self.assertIsNotNone(result)
        self.assertTrue(np.allclose(result.rpy_rad, validated.rpy_rad))
        self.assertEqual(result.yaw_correction_deg, 0.0)
        self.assertAlmostEqual(result.grasp_detection.min_width_mm, 24.0)
        self.assertTrue(result.grasp_detection.info[
            "metric_recovery_pose_validated"
        ])

    def test_multiview_restore_uses_bounded_yaw_for_near_boundary_projection(self):
        frame = (
            np.zeros((20, 20, 3), dtype=np.uint8),
            np.full((20, 20), 200.0, dtype=np.float32),
            (100.0, 100.0, 10.0, 10.0),
        )
        body = ws.WD.Detection(
            box=(2, 2, 18, 18), uv=(10.0, 10.0), z_mm=200.0,
            p_cam_mm=(0.0, 0.0, 200.0), source="sam3",
            min_width_mm=28.0, max_width_mm=78.0,
        )
        base_geometry = ws.GG.GraspGeometry(
            True, "metric candidate", body, body,
            (0.0, np.radians(8.0), 0.0), yaw_correction_deg=0.0,
        )
        object_box = {
            "axes_base": np.eye(3).tolist(),
            "observed_size_m": [0.078, 0.028, 0.055],
            "conservative_size_m": [0.084, 0.034, 0.061],
        }
        view = {
            "name": "pitched_retreat", "frame": frame,
            "detection": body, "ee_xyz_m": [0.26, 0.0, 0.25],
            "ee_rpy_rad": [0.0, np.radians(8.0), 0.0],
            "camera_mapping": np.eye(3) * 1000.0,
        }
        logs = []
        with (
            patch.object(ws, "_export_discovery_cloud", return_value={
                "ok": True, "object_box_3d": object_box,
            }),
            patch.object(ws, "_metric_horizontal_box_dimensions",
                         return_value=(28.0, 78.0, {"reason": "metric"})),
            patch.object(ws, "_detection_partial", return_value=(
                "semantic/depth envelope is clipped at the image bottom edge"
            )),
            patch.object(ws, "_exact_semantic_geometry_mask",
                         return_value=(True, "exact native semantic mask")),
            patch.object(ws, "_plan_scene_grasp_geometry",
                         side_effect=[base_geometry, base_geometry, base_geometry]),
            patch.object(ws.A, "ee_pos", return_value=[0.26, 0.0, 0.25]),
            patch.object(ws.A, "ee_rpy",
                         return_value=[0.0, np.radians(8.0), 0.0]),
            patch.object(ws, "_plan_rpy_trajectory",
                         return_value=([(0.0, np.radians(8.0),
                                        np.radians(2.0))], "ok")),
            patch.object(ws, "_execute_rpy_trajectory",
                         return_value=(True, "ok")),
            patch.object(ws, "_plan_fixed_approach_trajectory",
                         return_value=([(0.26, 0.0, 0.25)], "ok")),
            patch.object(ws, "_execute_fixed_approach_trajectory",
                         return_value=(True, "ok")),
            patch.object(ws, "_settle"),
            patch.object(ws, "_observe", return_value=(body, frame, "ok")),
            patch.object(ws, "load_saved_camera_jacobian",
                         return_value=(np.eye(3) * 1000.0, {"reason": "ok"})),
            patch.object(ws, "_projected_3d_jaw_span", side_effect=[
                {"ok": True, "observed_width_mm": 42.8,
                 "conservative_width_mm": 50.1, "reason": "boundary"},
                {"ok": True, "observed_width_mm": 39.0,
                 "conservative_width_mm": 47.0, "reason": "aligned"},
            ]),
            patch("arm_agent.gripper_model.max_aperture_mm",
                  return_value=50.0),
        ):
            _det, _frame, result, detail = \
                ws._recover_multiview_metric_grasp(
                    object(), "yellow sponge", [view], logs.append
                )

        self.assertTrue(detail["ok"], detail)
        self.assertIsNotNone(result)
        self.assertAlmostEqual(np.degrees(result.rpy_rad[2]), 2.0)
        self.assertTrue(any(
            "multiview_metric_grasp_orientation status=accepted" in line
            for line in logs
        ))

    def test_collision_map_uses_metric_target_obb_as_sole_exclusion(self):
        frame = (
            np.zeros((20, 30, 3), dtype=np.uint8),
            np.full((20, 30), 220.0, dtype=np.float32),
            (100.0, 100.0, 15.0, 10.0),
        )
        detection = SimpleNamespace(
            box=(5, 4, 20, 18), p_cam_mm=(0.0, 0.0, 220.0),
            min_width_mm=20.0, max_width_mm=80.0,
            info={
                "long_axis_deg": 0.0,
                # Even an available semantic mask must not erase neighboring
                # pixels once the metric OBB is authoritative.
                "mask_u8": np.ones((20, 30), dtype=np.uint8),
                "segmentation_source": "sam3",
            },
        )
        geometry = SimpleNamespace(
            envelope_detection=detection, grasp_detection=detection,
        )
        object_box = {
            "center_m": [0.38, 0.02, 0.04],
            "axes_base": np.eye(3).tolist(),
            "observed_size_m": [0.08, 0.04, 0.02],
            "conservative_size_m": [0.086, 0.046, 0.026],
        }
        scene = ws.SC.SceneMap(
            boxes=(ws.SC.ObstacleBox((0.45, 0.10, 0.05),
                                      (0.03, 0.03, 0.08), 4),),
            table_z_m=0.012, voxel_m=0.02, inflation_m=0.0075,
            valid_depth_points=100, mapped_obstacle_points=20,
            excluded_target_points=10, closest_obstacle_m=0.1,
        )
        with (
            patch.object(ws, "SCENE_COLLISION_ENABLED", True),
            patch.object(ws, "SCENE_COLLISION_EXPORT_VIEW", False),
            patch.object(ws, "load_tcp",
                         return_value=(3.6, 4.0, 66.9)),
            patch.object(ws, "load_safety_calibration",
                         return_value={"table_z_m": 0.012}),
            patch.object(ws.A, "ee_pos", return_value=[0.30, 0.0, 0.25]),
            patch.object(ws.SC, "build_scene_map", return_value=scene) as build,
            patch.object(ws.A, "filter_scene_boxes_at_current_pose",
                         return_value={"ok": True,
                                       "kept": [scene.boxes[0].as_dict()],
                                       "removed": []}),
            patch.object(ws.A, "configure_collision_scene",
                         return_value=ws.A.MoveResult(True, "configured")),
            patch.object(ws.A, "validate_current_collision_scene",
                         return_value={
                             "ok": True,
                             "reason": "capture pose clear",
                         }),
        ):
            ok, detail = ws._install_scene_collision_map(
                frame, geometry, np.eye(3) * 1000.0,
                object_boxes_3d={"pick_target": object_box},
            )

        self.assertTrue(ok, detail)
        kwargs = build.call_args.kwargs
        self.assertEqual(kwargs["exclude_pixel_boxes"], [])
        self.assertEqual(kwargs["exclude_pixel_masks"], [])
        self.assertEqual(kwargs["exclude_base_capsules"], [])
        self.assertEqual(len(kwargs["exclude_base_obbs"]), 1)
        self.assertTrue(np.allclose(
            kwargs["jaw_target_cam_mm"], [3.6, 4.0, 66.9]
        ))
        self.assertEqual(detail["target_exclusion_method"], "measured_3d_obb")

    def test_collision_map_requires_stationary_majority_occupancy(self):
        frame = (
            np.zeros((20, 30, 3), dtype=np.uint8),
            np.full((20, 30), 220.0, dtype=np.float32),
            (100.0, 100.0, 15.0, 10.0),
        )
        detection = SimpleNamespace(
            box=(5, 4, 20, 18), p_cam_mm=(0.0, 0.0, 220.0),
            min_width_mm=20.0, max_width_mm=80.0,
            info={"long_axis_deg": 0.0},
        )
        geometry = SimpleNamespace(
            envelope_detection=detection, grasp_detection=detection,
        )
        scene = ws.SC.SceneMap(
            boxes=(ws.SC.ObstacleBox((0.45, 0.10, 0.05),
                                      (0.03, 0.03, 0.08), 4),),
            table_z_m=0.012, voxel_m=0.02, inflation_m=0.003,
            valid_depth_points=100, mapped_obstacle_points=20,
            excluded_target_points=10, closest_obstacle_m=0.1,
        )
        with (
            patch.object(ws, "SCENE_COLLISION_ENABLED", True),
            patch.object(ws, "SCENE_COLLISION_EXPORT_VIEW", False),
            patch.object(ws, "load_tcp", return_value=(3.6, 4.0, 66.9)),
            patch.object(ws, "load_safety_calibration",
                         return_value={"table_z_m": 0.012}),
            patch.object(ws.A, "ee_pos", return_value=[0.30, 0.0, 0.25]),
            patch.object(ws.SC, "build_scene_map", return_value=scene),
            patch.object(ws.SC, "fuse_scene_maps", return_value=scene) as fuse,
            patch.object(ws.A, "filter_scene_boxes_at_current_pose",
                         return_value={"ok": True,
                                       "kept": [scene.boxes[0].as_dict()],
                                       "removed": []}),
            patch.object(ws.A, "configure_collision_scene",
                         return_value=ws.A.MoveResult(True, "configured")),
            patch.object(ws.A, "validate_current_collision_scene",
                         return_value={"ok": True, "reason": "clear"}),
        ):
            ok, detail = ws._install_scene_collision_map(
                frame, geometry, np.eye(3) * 1000.0,
                extra_frames=[frame, frame],
            )

        self.assertTrue(ok, detail)
        self.assertEqual(
            fuse.call_args.kwargs["minimum_views_per_column"], 2
        )
        self.assertEqual(detail["temporal_required_frames"], 2)
        self.assertEqual(
            detail["temporal_fusion"],
            "strict_majority_stationary_occupancy",
        )

    def test_clearance_and_axial_legs_are_validated_as_one_chain(self):
        checked_paths = []

        def check_path(waypoints, *_rpy):
            checked_paths.append(list(waypoints))
            return ws.A.MoveResult(True, "path_ok", waypoints[-1])

        start = (0.30, 0.0, 0.25)
        pregrasp = (0.34, 0.01, 0.21)
        endpoint = (0.37, 0.02, 0.18)
        rpy = (0.0, 0.0, 0.1)
        with (
            patch.object(ws.A, "ee_pos", lambda: list(start)),
            patch.object(ws.A, "ee_rpy", lambda: list(rpy)),
            patch.object(ws.A, "check_ik_path", check_path),
            patch.object(ws.A, "X_MIN", 0.15),
            patch.object(ws.A, "X_MAX", 0.49),
            patch.object(ws.A, "Y_MIN", -0.24),
            patch.object(ws.A, "Y_MAX", 0.24),
            patch.object(ws.A, "Z_MIN", 0.01),
            patch.object(ws.A, "Z_MAX", 0.30),
        ):
            first, first_reason = ws._plan_fixed_approach_trajectory(
                pregrasp, fixed_rpy=rpy, validate_ik=False
            )
            second, second_reason = ws._plan_fixed_approach_trajectory(
                endpoint, start_m=pregrasp, fixed_rpy=rpy,
                validate_ik=False,
            )
            self.assertIsNotNone(first, first_reason)
            self.assertIsNotNone(second, second_reason)
            combined = first + second
            ok, reason = ws._validate_fixed_waypoints(combined, rpy)

        self.assertTrue(ok, reason)
        self.assertEqual(len(checked_paths), 1)
        self.assertEqual(checked_paths[0], combined)
        self.assertEqual(combined[-1], endpoint)

    def test_axial_planner_retries_level_clearance_without_relaxing_ik(self):
        start = np.array([0.30, 0.0, 0.25])
        endpoint = np.array([0.40, 0.05, 0.16])
        checked_paths = []

        def generate(target, *, start_m=None, validate_ik=True, **_kwargs):
            self.assertFalse(validate_ik)
            return [tuple(np.asarray(target, dtype=float))], "bounded"

        def validate(path, _rpy):
            checked_paths.append(list(path))
            if len(checked_paths) == 1:
                return False, "synthetic straight-path singularity"
            return True, "complete chain passed"

        with (
            patch.object(ws.A, "ee_pos", lambda: start.tolist()),
            patch.object(ws, "_plan_fixed_approach_trajectory", generate),
            patch.object(ws, "_validate_fixed_waypoints", validate),
            patch.object(ws, "GEOMETRY_FINAL_APPROACH_MM", 40.0),
            patch.object(ws, "GEOMETRY_MIN_FINAL_APPROACH_MM", 20.0),
        ):
            plan, reason, detail = ws._plan_clearance_axial_trajectory(
                endpoint, np.eye(3) * 1000.0, (0.0, 0.0, 0.0)
            )

        self.assertIsNotNone(plan, reason)
        self.assertEqual(len(checked_paths), 2)
        self.assertEqual(detail["clearance_strategy"], "level_clearance")
        self.assertEqual(detail["final_approach_camera_mm"], [0.0, 0.0, 40.0])
        self.assertIn("40mm camera-axial", reason)
        self.assertGreater(len(plan), 2)

    def test_axial_planner_skips_prefixes_when_shared_jaw_entry_is_blocked(self):
        start = np.array([0.30, 0.0, 0.25])
        endpoint = np.array([0.40, 0.05, 0.16])
        checked_paths = []

        def generate(target, *, start_m=None, validate_ik=True, **_kwargs):
            self.assertFalse(validate_ik)
            return [tuple(np.asarray(target, dtype=float))], "bounded"

        def validate(path, _rpy):
            checked_paths.append(list(path))
            return False, (
                "complete path failed continuation IK: local_path_ik_error "
                "Cartesian path collision at segment 2, sample 1/1: "
                "{'collision': True, 'kind': 'environment', "
                "'robot_link': 'gripper_left_0', 'obstacle': 'scene_box_0'}"
            )

        with (
            patch.object(ws.A, "ee_pos", lambda: start.tolist()),
            patch.object(ws, "_plan_fixed_approach_trajectory", generate),
            patch.object(ws, "_validate_fixed_waypoints", validate),
            patch.object(ws.A, "plan_joint_motion") as global_planner,
            patch.object(ws, "GEOMETRY_FINAL_APPROACH_MM", 40.0),
            patch.object(ws, "GEOMETRY_MIN_FINAL_APPROACH_MM", 20.0),
        ):
            plan, reason, detail = ws._plan_clearance_axial_trajectory(
                endpoint, np.eye(3) * 1000.0, (0.0, 0.0, 0.0)
            )

        self.assertIsNone(plan)
        self.assertEqual(len(checked_paths), 3)
        self.assertTrue(detail["terminal_approach_blocked"])
        self.assertEqual(detail["tested_backoffs_mm"], [40.0, 30.0, 20.0])
        self.assertIn("shared final jaw-entry is blocked", reason)
        global_planner.assert_not_called()

    def test_fixed_trajectory_accepts_exact_decimal_workspace_margin(self):
        with (
            patch.object(ws.A, "ee_pos", lambda: [0.46, 0.0, 0.22]),
            patch.object(ws.A, "ee_rpy", lambda: [0.0, 0.0, 0.0]),
            patch.object(
                ws.A, "check_ik",
                lambda x, y, z, *rpy: ws.A.MoveResult(
                    True, "ik_ok", (x, y, z)
                ),
            ),
            patch.object(ws.A, "check_ik_path", None),
            patch.object(ws.A, "X_MIN", 0.15),
            patch.object(ws.A, "X_MAX", 0.480),
            patch.object(ws.A, "Y_MIN", -0.24),
            patch.object(ws.A, "Y_MAX", 0.24),
            patch.object(ws.A, "Z_MIN", 0.01),
            patch.object(ws.A, "Z_MAX", 0.30),
            patch.object(ws, "APPROACH_WORKSPACE_MARGIN_M", 0.015),
        ):
            waypoints, reason = ws._plan_fixed_approach_trajectory(
                [0.465, 0.0, 0.22]
            )

        self.assertIsNotNone(waypoints, reason)
        self.assertEqual(waypoints[-1], (0.465, 0.0, 0.22))

    def test_approach_target_stays_inside_workspace_margin(self):
        det = SimpleNamespace(p_cam_mm=np.array([0.0, 0.0, 0.0]))
        # 1000 camera-mm per base-metre makes a 200 mm camera error request a
        # 200 mm +X tool move: from x=.30 to x=.50, beyond the .44 envelope.
        jacobian = np.eye(3) * 1000.0
        with (
            patch.object(ws.A, "ee_pos", lambda: [0.30, 0.0, 0.22]),
            patch.object(ws.A, "ee_rpy", lambda: [0.0, 0.0, 0.0]),
            patch.object(ws.A, "check_ik", lambda *args:
                         ws.A.MoveResult(True, "ik_ok", tuple(args[:3]))),
            patch.object(ws.A, "X_MIN", 0.15),
            patch.object(ws.A, "X_MAX", 0.44),
            patch.object(ws.A, "Y_MIN", -0.24),
            patch.object(ws.A, "Y_MAX", 0.24),
            patch.object(ws.A, "Z_MIN", 0.01),
            patch.object(ws.A, "Z_MAX", 0.30),
            patch.object(ws, "APPROACH_WORKSPACE_MARGIN_M", 0.015),
        ):
            target, plan = ws._reachable_approach_target(
                det, jacobian, np.array([200.0, 0.0, 0.0])
            )

        self.assertIsNotNone(target)
        self.assertFalse(plan["full_hover"])
        self.assertAlmostEqual(plan["predicted_endpoint_m"][0], 0.425, places=3)
        self.assertLess(plan["path_fraction"], 1.0)

    def test_jacobian_shrinks_an_ik_refused_axis_jog(self):
        base = np.array([0.30, 0.0, 0.22], dtype=float)
        current = base.copy()
        y_attempts = []

        def observe(*_args, **_kwargs):
            p = tuple((-1000.0 * (current - base)).tolist())
            return SimpleNamespace(p_cam_mm=p), None, {}

        def goto(x, y, z, **_kwargs):
            target = np.array([x, y, z], dtype=float)
            y_delta = abs(float(target[1] - base[1]))
            if y_delta > 1e-8:
                y_attempts.append(round(y_delta, 3))
                if y_delta > 0.0101:
                    return ws.A.MoveResult(False, "IK refused")
            current[:] = target
            return ws.A.MoveResult(True, "arrived", tuple(target))

        with (
            patch.object(ws.A, "ee_pos", lambda: list(current)),
            patch.object(ws.A, "safe_goto", goto),
            patch.object(ws, "_observe", observe),
            patch.object(ws, "_settle", lambda *args, **kwargs: None),
        ):
            jacobian, reason = ws.learn_jacobian(SimpleNamespace(), "blue marker")

        self.assertIsNotNone(jacobian, reason)
        self.assertIn(0.020, y_attempts)
        self.assertIn(0.015, y_attempts)
        self.assertIn(0.010, y_attempts)

    def test_jacobian_uses_measured_arm_displacement_not_requested_jog(self):
        base = np.array([0.30, 0.0, 0.22], dtype=float)
        current = base.copy()

        def observe(*_args, **_kwargs):
            p = tuple((-1000.0 * (current - base)).tolist())
            return SimpleNamespace(p_cam_mm=p), None, {}

        def goto(x, y, z, **_kwargs):
            target = np.array([x, y, z], dtype=float)
            if np.allclose(target, base):
                current[:] = base
            else:
                # Model a controller that reports success within tolerance but
                # reaches only 80% of each requested calibration displacement.
                current[:] = base + 0.8 * (target - base)
            return ws.A.MoveResult(True, "arrived", tuple(current))

        with (
            patch.object(ws.A, "ee_pos", lambda: list(current)),
            patch.object(ws.A, "safe_goto", goto),
            patch.object(ws, "_observe", observe),
            patch.object(ws, "_settle", lambda *args, **kwargs: None),
        ):
            jacobian, reason = ws.learn_jacobian(SimpleNamespace(), "sponge")

        self.assertIsNotNone(jacobian, reason)
        assert jacobian is not None
        self.assertTrue(np.allclose(jacobian, -1000.0 * np.eye(3), atol=1e-6))
        rigid, detail = ws._rigid_camera_jacobian(jacobian)
        self.assertIsNotNone(rigid, detail)

    def test_approach_reaches_hover_then_homes_without_gripper_command(self):
        class StopEye:
            camera_kind = "test"

            def frame(self):
                raise KeyboardInterrupt

            def close(self):
                pass

        det = type("Det", (), {
            "source": "test", "z_mm": 300.0,
            "p_cam_mm": np.array([0.0, 0.0, 300.0]),
        })()
        preflight = ws.A.MoveResult(True, "ready")
        recover_args = []
        requested_targets = []

        def reachable_target(*args):
            requested_targets.append(np.asarray(args[2], dtype=float))
            return np.zeros(3), {
                "full_hover": True, "path_fraction": 1.0,
                "predicted_endpoint_m": [0.3, 0.0, 0.2],
                "reason": "reachable",
            }

        with (
            patch.object(ws, "load_tcp", lambda: (3.6, 4.0, 66.9)),
            patch.object(ws.A, "preflight", lambda **kwargs: preflight),
            patch.object(ws.A, "ee_pos", lambda: [0.30, 0.0, 0.22]),
            patch.object(ws, "_approach_ready_lift", lambda *args, **kwargs:
                         ws.A.MoveResult(True, "arrived", (0.30, 0.0, 0.22))),
            patch.object(ws, "_settle", lambda *args, **kwargs: None),
            patch.object(ws, "_observe", lambda *args: (det, None, {})),
            patch.object(ws, "learn_jacobian", lambda *args: (np.eye(3), "ok")),
            patch.object(ws, "_reachable_approach_target", reachable_target),
            patch.object(
                ws, "_plan_fixed_approach_trajectory",
                lambda *args: ([(0.30, 0.0, 0.20)], "prevalidated"),
            ),
            patch.object(
                ws, "_execute_fixed_approach_trajectory",
                lambda *args: (True, "completed"),
            ),
            patch.object(ws, "_freeze_arm", lambda: {"ok": True, "reason": "held"}),
            patch.object(
                ws,
                "_home_after_interrupt",
                lambda **kwargs: recover_args.append(kwargs) or {
                    "ok": True, "gripper_opened": True, "homed": True,
                    "reason": "homed",
                },
            ),
        ):
            result = ws.servo_approach("blue marker", eye=StopEye())

        self.assertTrue(result["ok"])
        self.assertTrue(result["homed"])
        self.assertFalse(result["gripper_commanded"])
        self.assertEqual(recover_args, [{"open_gripper": False}])
        self.assertAlmostEqual(
            requested_targets[0][2],
            max(66.9 + ws.APPROACH_NEAR_TOUCH_GAP_MM, ws.WD.NEAR_MM + 15.0),
            places=3,
        )

    def test_approach_planning_failure_homes_without_gripper_command(self):
        class Eye:
            camera_kind = "test"

            def close(self):
                pass

        det = type("Det", (), {
            "source": "test", "z_mm": 300.0,
            "p_cam_mm": np.array([0.0, 0.0, 300.0]),
        })()
        preflight = ws.A.MoveResult(True, "ready")
        recover_args = []
        with (
            patch.object(ws, "load_tcp", lambda: (3.6, 4.0, 66.9)),
            patch.object(ws.A, "preflight", lambda **kwargs: preflight),
            patch.object(ws.A, "ee_pos", lambda: [0.30, 0.0, 0.22]),
            patch.object(ws, "_approach_ready_lift", lambda *args, **kwargs:
                         ws.A.MoveResult(True, "arrived", (0.30, 0.0, 0.22))),
            patch.object(ws, "_settle", lambda *args, **kwargs: None),
            patch.object(ws, "_observe", lambda *args: (det, None, {})),
            patch.object(ws, "learn_jacobian", lambda *args: (None, "IK refused")),
            patch.object(
                ws,
                "_home_after_interrupt",
                lambda **kwargs: recover_args.append(kwargs) or {
                    "ok": True, "gripper_opened": True, "homed": True,
                    "reason": "homed",
                },
            ),
        ):
            result = ws.servo_approach("blue marker", eye=Eye())

        self.assertFalse(result["ok"])
        self.assertTrue(result["homed"])
        self.assertFalse(result["gripper_commanded"])
        self.assertEqual(recover_args, [{"open_gripper": False}])

    def test_interrupted_look_opens_then_homes(self):
        preflight = ws.A.MoveResult(True, "ready")
        reached = ws.A.MoveResult(True, "arrived", (0.32, 0.0, 0.25))
        recoveries = []
        with (
            patch.object(ws.A, "preflight", lambda **kwargs: preflight),
            patch.object(ws, "_scan_poses", lambda: [(0.32, 0.0, 0.25)]),
            patch.object(ws, "_goto", lambda *args, **kwargs: reached),
            patch.object(ws.time, "sleep", lambda seconds: None),
            patch.object(ws, "watch", lambda *args, **kwargs: {
                "ok": True, "interrupted": True,
            }),
            patch.object(
                ws,
                "_home_after_interrupt",
                lambda: recoveries.append(True) or {
                    "ok": True, "gripper_opened": True, "homed": True,
                },
            ),
        ):
            result = ws.look("blue marker")

        self.assertEqual(recoveries, [True])
        self.assertTrue(result["gripper_opened"])
        self.assertTrue(result["homed"])

    def test_standalone_pick_interrupt_returns_recovery_state(self):
        ws._OBJECT_HELD = False
        ready = (True, "torque verified; table z=0.0120m", 0.012)
        preflight = ws.A.MoveResult(True, "ready")

        class InterruptEye:
            camera_kind = "test"

            def frame(self):
                raise KeyboardInterrupt

            def close(self):
                pass

        with (
            patch.object(ws, "load_tcp", lambda: (3.6, 4.0, 66.9)),
            patch.object(ws, "descent_readiness", lambda **kwargs: ready),
            patch.object(ws.A, "preflight", lambda **kwargs: preflight),
            patch.object(
                ws,
                "_home_after_interrupt",
                lambda: {"ok": True, "gripper_opened": True, "homed": True},
            ),
        ):
            result = ws.servo_pick("blue marker", eye=InterruptEye())

        self.assertFalse(result["ok"])
        self.assertEqual(result["status"], "interrupted")
        self.assertFalse(result["object_held"])
        self.assertFalse(result["safe_to_retry"])
        self.assertTrue(result["gripper_opened"])
        self.assertTrue(result["homed"])

    def test_interrupt_recovery_opens_before_homing(self):
        events = []
        with (
            patch.object(ws, "_gripper", lambda action: events.append(action) or {"ok": True}),
            patch.object(ws, "_home", lambda: events.append("home") or {"ok": True}),
        ):
            result = ws._home_after_interrupt()

        self.assertEqual(events, ["open", "home"])
        self.assertTrue(result["gripper_opened"])
        self.assertTrue(result["homed"])

    def test_failed_freeze_refuses_home_and_gripper_trajectory(self):
        events = []
        with (
            patch.object(ws, "_freeze_arm", lambda: {"ok": False, "reason": "still moving"}),
            patch.object(ws, "_gripper", lambda action: events.append(action) or {"ok": True}),
            patch.object(ws, "_home", lambda: events.append("home") or {"ok": True}),
        ):
            result = ws._home_after_interrupt()

        self.assertFalse(result["ok"])
        self.assertFalse(result["homed"])
        self.assertEqual(events, [])

    def test_pick_place_interrupt_opens_and_homes(self):
        ws._OBJECT_HELD = None
        with (
            patch.object(ws, "servo_pick", side_effect=KeyboardInterrupt),
            patch.object(
                ws,
                "_home_after_interrupt",
                lambda: {"ok": True, "gripper_opened": True, "homed": True},
            ),
        ):
            result = ws.pick_and_place("blue marker")

        self.assertFalse(result["ok"])
        self.assertEqual(result["status"], "interrupted")
        self.assertTrue(result["homed"])
        self.assertTrue(result["gripper_opened"])
        self.assertTrue(result["needs_operator"])

    def test_ordinary_pick_failure_opens_and_homes_before_return(self):
        events = []
        failed_pick = {
            "ok": False,
            "reason": "visual servo unavailable",
            "object_held": False,
        }
        with (
            patch.object(ws, "servo_pick", return_value=failed_pick),
            patch.object(
                ws,
                "_home_after_interrupt",
                lambda: events.append("recover") or {
                    "ok": True, "gripper_opened": True, "homed": True,
                },
            ),
        ):
            result = ws.pick_and_place("blue marker")

        self.assertEqual(events, ["recover"])
        self.assertFalse(result["ok"])
        self.assertTrue(result["homed"])
        self.assertTrue(result["pick"]["gripper_opened"])

    def test_live_place_refuses_unconfirmed_object_before_motion(self):
        ws._OBJECT_HELD = False
        moves = []
        ready = (True, "torque verified", 0.012)
        preflight = ws.A.MoveResult(True, "ready")
        with (
            patch.object(ws, "load_drop_zone", lambda: (0.275, -0.187, 0.059)),
            patch.object(ws, "descent_readiness", lambda **kwargs: ready),
            patch.object(ws.A, "preflight", lambda **kwargs: preflight),
            patch.object(ws.A, "ee_pos", lambda: moves.append("state")),
        ):
            result = ws.place_in_drop_zone()

        self.assertFalse(result["ok"])
        self.assertIn("held object is not confirmed", result["reason"])
        self.assertEqual(moves, [])
