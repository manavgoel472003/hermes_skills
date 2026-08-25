from __future__ import annotations

import os
import json
import math
import sys
import time
from pathlib import Path
from unittest import TestCase, skipUnless
from unittest.mock import patch

import cv2
import numpy as np
from types import SimpleNamespace

from arm_agent import wrist_servo as ws
from arm_agent import wrist_detect as WD
from arm_agent import wrist_sim


class WristWorkflowTest(TestCase):
    def test_simple_jaw_presence_is_position_only(self):
        detections = [
            SimpleNamespace(p_cam_mm=[3.9, 31.7, 65.0]),
            SimpleNamespace(p_cam_mm=[4.2, 31.5, 65.2]),
            SimpleNamespace(p_cam_mm=[40.0, 80.0, 120.0]),
        ]

        position, detail = ws._simple_jaw_presence_consensus(detections)

        self.assertIsNotNone(position, detail)
        self.assertTrue(detail["presence_only"])
        self.assertFalse(detail["geometry_updated"])
        self.assertIn("frozen checkpoint geometry", detail["reason"])

    def test_simple_jaw_presence_uses_physical_votes_not_centroid_alignment(self):
        detections = [
            SimpleNamespace(p_cam_mm=[0.0, 0.0, 65.0]),
            SimpleNamespace(p_cam_mm=[5.0, 10.0, 70.0]),
            SimpleNamespace(p_cam_mm=[-5.0, -10.0, 60.0]),
        ]

        position, detail = ws._simple_jaw_presence_consensus(
            detections, min_frames=3, max_spread_mm=5.0, total_frames=5,
        )

        self.assertIsNotNone(position, detail)
        self.assertEqual(detail["valid_frames"], 3)
        self.assertEqual(detail["total_frames"], 5)
        self.assertGreater(detail["spread_mm_diagnostic"], 5.0)
        self.assertTrue(detail["presence_only"])

    def test_simple_grasp_retries_one_alternate_after_verified_recovery(self):
        failed = {
            "ok": False, "reason": "no jaw occupancy",
            "grasp_retryable": True, "homed": True,
            "object_held": False,
        }
        succeeded = {
            "ok": True, "reason": "picked", "homed": False,
            "object_held": True,
        }
        with (
            patch.object(ws, "SIMPLE_GRASP_FLOW_ENABLED", True),
            patch.object(ws, "SIMPLE_GRASP_MAX_ATTEMPTS", 2),
            patch.object(
                ws, "_fixed_pick_and_place_attempt",
                side_effect=[failed, succeeded],
            ) as execute,
        ):
            result = ws.fixed_pick_and_place(
                "green marker", place_after_pick=False
            )

        self.assertTrue(result["ok"], result)
        self.assertEqual(execute.call_count, 2)
        self.assertEqual(
            execute.call_args_list[0].kwargs["grasp_attempt_index"], 0
        )
        self.assertEqual(
            execute.call_args_list[1].kwargs["grasp_attempt_index"], 1
        )
        self.assertEqual(len(result["grasp_attempts"]), 2)
        self.assertEqual(
            result["grasp_attempts"][1]["strategy"],
            "fresh_primary_then_alternate_if_blocked",
        )

    def test_simple_grasp_never_retries_without_verified_home(self):
        failed = {
            "ok": False, "reason": "recovery incomplete",
            "grasp_retryable": True, "homed": False,
            "object_held": None,
        }
        with (
            patch.object(ws, "SIMPLE_GRASP_FLOW_ENABLED", True),
            patch.object(ws, "SIMPLE_GRASP_MAX_ATTEMPTS", 2),
            patch.object(
                ws, "_fixed_pick_and_place_attempt", return_value=failed,
            ) as execute,
        ):
            result = ws.fixed_pick_and_place("green marker")

        self.assertFalse(result["ok"])
        self.assertEqual(execute.call_count, 1)
        self.assertEqual(len(result["grasp_attempts"]), 1)

    def test_payload_geometry_handoff_does_not_require_collision_map(self):
        box = {
            "center_m": [0.40, 0.02, 0.08],
            "axes_base": np.eye(3).tolist(),
            "conservative_size_m": [0.02, 0.03, 0.09],
        }

        context, detail = ws._payload_box_context_from_object_box(
            box, [0.40, 0.02, 0.10], [0.0, 0.0, 0.0]
        )

        self.assertIsNotNone(context, detail)
        assert context is not None
        self.assertEqual(context["full_size_m"], [0.02, 0.03, 0.09])
        np.testing.assert_allclose(
            context["center_offset_ee_m"], [0.0, 0.0, -0.02]
        )
        self.assertEqual(context["long_axis_ee"], [0.0, 0.0, 1.0])

    def test_multiview_target_support_removes_one_view_mask_spill(self):
        common = np.array([
            [0.400 + x * 0.002, 0.010 + y * 0.002, 0.040]
            for x in range(10) for y in range(10)
        ])
        point_sets = []
        for index, offset in enumerate((0.0, 0.001, -0.001)):
            repeated = common + np.array([offset, 0.0, 0.0])
            spill = np.array([
                [0.52 + 0.03 * index, 0.10 + item * 0.001, 0.05]
                for item in range(30)
            ])
            point_sets.append(np.vstack((repeated, spill)))

        filtered, detail = ws._filter_target_points_by_multiview_support(
            point_sets
        )

        self.assertTrue(detail["ok"], detail)
        self.assertTrue(detail["applied"])
        self.assertGreater(detail["removed"], 0)
        self.assertEqual(len(filtered), 3)
        self.assertLess(max(float(points[:, 0].max()) for points in filtered),
                        0.50)

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

    def test_verified_metric_identity_anchor_cannot_drift_at_checkpoint(self):
        messages = []
        verified_center = np.array([0.4063, -0.0614, 0.0562])
        gate = ws._HybridTargetIdentity(
            description="red marker",
            tool_camera_mm=np.zeros(3), log=messages.append,
            anchor_base_m=verified_center.copy(),
            semantic_points_base_m=[verified_center.copy()],
            anchor_kind="verified_metric_obb_center",
        )
        frame = (np.zeros((100, 100, 3), dtype=np.uint8),
                 np.ones((100, 100), dtype=np.float32),
                 (100.0, 100.0, 50.0, 50.0))
        # These represent accepted-but-contaminated close semantic centroids.
        # They remain inside the 75 mm identity cluster but must not relocate
        # the multi-view metric centre.
        detections = [
            SimpleNamespace(source="sam3", p_cam_mm=(0.0, 0.0, 100.0)),
            SimpleNamespace(source="sam3", p_cam_mm=(0.0, 0.0, 100.0)),
        ]
        shifted = [
            verified_center + np.array([0.060, 0.0, 0.0]),
            verified_center + np.array([0.055, 0.005, 0.0]),
        ]
        with patch.object(gate, "_point_base", side_effect=[
            (shifted[0], "mapped"), (shifted[1], "mapped")
        ]):
            first, _ = gate(detections[0], frame)
            second, _ = gate(detections[1], frame)

        self.assertTrue(first)
        self.assertTrue(second)
        self.assertTrue(np.allclose(gate.anchor_base_m, verified_center))
        self.assertEqual(len(gate.semantic_points_base_m), 1)

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
            tool_camera_mm=np.array([0.0, 0.0, 300.0]),
            log=messages.append,
        )
        color_near = SimpleNamespace(
            source="color:red", p_cam_mm=(-190.0, 0.0, 205.0)
        )
        semantic = SimpleNamespace(
            source="sam3", p_cam_mm=(-200.0, 0.0, 200.0),
            z_mm=200.0, box=(20, 20, 40, 40), info={}
        )
        color_far = SimpleNamespace(
            source="color:red", p_cam_mm=(-200.0, 0.0, 520.0)
        )
        frame = (np.zeros((100, 100, 3), dtype=np.uint8),
                 np.ones((100, 100), dtype=np.float32),
                 (100.0, 100.0, 50.0, 50.0))
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

    def test_hybrid_target_identity_defers_clipped_seed_then_anchors_valid_view(self):
        gate = ws._HybridTargetIdentity(
            description="purple marker",
            tool_camera_mm=np.zeros(3),
            log=lambda _message: None,
        )
        clipped = SimpleNamespace(
            source="sam3", p_cam_mm=(0.0, 0.0, 240.0),
            z_mm=240.0, box=(20, 20, 60, 99), info={},
        )
        complete = SimpleNamespace(
            source="sam3", p_cam_mm=(0.0, 0.0, 240.0),
            z_mm=240.0, box=(20, 20, 60, 70), info={},
        )
        frame = (np.zeros((100, 100, 3), dtype=np.uint8),
                 np.ones((100, 100), dtype=np.float32),
                 (100.0, 100.0, 50.0, 50.0))
        with (
            patch.object(
                gate, "_point_base",
                side_effect=[
                    (np.array([0.53, 0.21, 0.03]), "mapped"),
                    (np.array([0.44, 0.05, 0.03]), "mapped"),
                ],
            ),
            patch.object(
                ws, "load_safety_calibration",
                return_value={"table_z_m": 0.012},
            ),
        ):
            accepted, rejected_detail = gate(clipped, frame)
            self.assertFalse(accepted)
            self.assertTrue(rejected_detail["anchor_deferred"])
            self.assertIn("clipped", rejected_detail["reason"])
            self.assertIsNone(gate.anchor_base_m)

            accepted, accepted_detail = gate(complete, frame)

        self.assertTrue(accepted, accepted_detail)
        self.assertEqual(gate.anchor_kind, "semantic")
        self.assertTrue(np.allclose(gate.anchor_base_m, [0.44, 0.05, 0.03]))

    def test_hybrid_target_identity_defers_seed_below_calibrated_table(self):
        gate = ws._HybridTargetIdentity(
            description="purple marker",
            tool_camera_mm=np.zeros(3),
            log=lambda _message: None,
        )
        detection = SimpleNamespace(
            source="sam3", p_cam_mm=(0.0, 0.0, 240.0),
            z_mm=240.0, box=(20, 20, 60, 70), info={},
        )
        frame = (np.zeros((100, 100, 3), dtype=np.uint8),
                 np.ones((100, 100), dtype=np.float32),
                 (100.0, 100.0, 50.0, 50.0))
        with (
            patch.object(
                gate, "_point_base",
                return_value=(np.array([0.33, 0.11, -0.001]), "mapped"),
            ),
            patch.object(
                ws, "load_safety_calibration",
                return_value={"table_z_m": 0.012},
            ),
        ):
            accepted, detail = gate(detection, frame)

        self.assertFalse(accepted)
        self.assertTrue(detail["anchor_deferred"])
        self.assertIn("below calibrated table", detail["reason"])
        self.assertIsNone(gate.anchor_base_m)

    def test_hybrid_target_identity_defers_seed_outside_arm_workspace(self):
        gate = ws._HybridTargetIdentity(
            description="red marker",
            tool_camera_mm=np.zeros(3),
            log=lambda _message: None,
        )
        detection = SimpleNamespace(
            source="sam3", p_cam_mm=(0.0, 0.0, 568.0),
            z_mm=568.0, box=(20, 20, 60, 70), info={},
        )
        frame = (np.zeros((100, 100, 3), dtype=np.uint8),
                 np.ones((100, 100), dtype=np.float32),
                 (100.0, 100.0, 50.0, 50.0))
        with (
            patch.object(
                gate, "_point_base",
                return_value=(np.array([0.752, -0.3872, 0.0204]), "mapped"),
            ),
            patch.object(
                ws, "load_safety_calibration",
                return_value={"table_z_m": 0.012},
            ),
        ):
            accepted, detail = gate(detection, frame)

        self.assertFalse(accepted)
        self.assertTrue(detail["anchor_deferred"])
        self.assertIn("outside the reachable workspace", detail["reason"])
        self.assertIsNone(gate.anchor_base_m)

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

    def test_hybrid_target_identity_requires_stable_color_seeded_instance(self):
        gate = ws._HybridTargetIdentity(
            description="purple marker",
            tool_camera_mm=np.zeros(3),
            log=lambda _message: None,
        )
        frame = (np.zeros((2, 2, 3), dtype=np.uint8),
                 np.ones((2, 2), dtype=np.float32),
                 (100.0, 100.0, 1.0, 1.0))
        observations = [
            SimpleNamespace(source="rgbd:color_seeded_instance",
                            p_cam_mm=(0.0, 0.0, depth))
            for depth in (230.0, 230.5, 229.8)
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
        self.assertEqual(gate.anchor_kind, "stable_color_seeded_instance")
        self.assertIn("exact-color-seeded", detail["reason"])

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

    def test_3d_axis_refinement_selects_bounded_aligned_yaw(self):
        box = {
            "axes_base": np.eye(3).tolist(),
            "observed_size_m": [0.020, 0.090, 0.030],
            "conservative_size_m": [0.026, 0.096, 0.036],
        }

        def mapping_for_rpy(rpy, **_kwargs):
            marker = np.eye(3) * 1000.0
            marker[0, 0] = np.degrees(rpy[2])
            return marker, {"reason": "bounded rigid camera mapping"}

        def alignment_at(mapping_box, mapping):
            self.assertIs(mapping_box, box)
            yaw_deg = float(mapping[0, 0])
            if abs(yaw_deg - 6.0) > 0.1:
                return {"ok": False, "reason": "axis remains oblique"}
            return {
                "ok": True,
                "observed_width_mm": 20.0,
                "conservative_width_mm": 26.0,
                "image_alignment_error_deg": 0.2,
                "reason": "measured narrow axis aligned",
            }

        with (
            patch.dict(os.environ, {
                "REBOT_3D_AXIS_REFINE_STEP_DEG": "1",
                "REBOT_3D_AXIS_REFINE_MAX_YAW_DEG": "10",
            }),
            patch.object(ws, "load_saved_camera_jacobian",
                         side_effect=mapping_for_rpy),
            patch.object(ws, "_aligned_3d_jaw_width",
                         side_effect=alignment_at),
        ):
            result = ws._find_3d_axis_yaw_refinement(
                box, (0.0, 0.0, 0.0), 50.0
            )

        self.assertTrue(result["ok"], result)
        self.assertAlmostEqual(result["offset_deg"], 6.0)
        self.assertAlmostEqual(np.degrees(result["target_rpy"][2]), 6.0)

    def test_3d_axis_refinement_never_relaxes_jaw_aperture(self):
        box = {
            "axes_base": np.eye(3).tolist(),
            "observed_size_m": [0.044, 0.090, 0.030],
            "conservative_size_m": [0.052, 0.096, 0.036],
        }
        aligned_but_too_wide = {
            "ok": True,
            "observed_width_mm": 44.0,
            "conservative_width_mm": 52.0,
            "image_alignment_error_deg": 0.0,
            "reason": "aligned but too wide",
        }
        with (
            patch.dict(os.environ, {
                "REBOT_3D_AXIS_REFINE_STEP_DEG": "2",
                "REBOT_3D_AXIS_REFINE_MAX_YAW_DEG": "6",
            }),
            patch.object(ws, "load_saved_camera_jacobian", return_value=(
                np.eye(3) * 1000.0, {"reason": "mapping"}
            )),
            patch.object(ws, "_aligned_3d_jaw_width",
                         return_value=aligned_but_too_wide),
        ):
            result = ws._find_3d_axis_yaw_refinement(
                box, (0.0, 0.0, 0.0), 50.0
            )

        self.assertFalse(result["ok"])
        self.assertIn("50.0mm aperture", result["reason"])

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

    def test_locked_identifier_recovers_outside_body_for_bounded_alignment(self):
        geometry = SimpleNamespace(grasp_detection=SimpleNamespace(
            p_cam_mm=np.array([3.9, 31.7, 65.0]),
            min_width_mm=14.0,
            max_width_mm=100.0,
        ))
        outside = {"reason": "along 10.0/6.0mm"}
        with patch.object(
            ws, "_jaw_slot_fit", return_value=(False, outside),
        ):
            result = ws._locked_identifier_body_slot_fallback(
                np.array([5.2, 36.2, 64.3]),
                {"stable_frames": 3, "spread_mm": 1.0},
                np.array([1.3, -5.5, -0.7]),
                np.array([3.9, 31.7, 65.0]),
                {
                    "grasp_mm": 10.0, "preopen_mm": 28.0,
                    "min_width_mm": 14.0,
                },
                geometry,
            )

        self.assertTrue(result["ok"], result)
        self.assertFalse(result["slot_ok"])
        self.assertIn("along 10.0", result["reason"])

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

    def test_elongated_object_can_enter_by_stable_local_subsection(self):
        geometry = SimpleNamespace(grasp_detection=object())
        jaw_plan = {"ok": True, "preopen_mm": 50.0}
        with patch.object(
            ws, "_validate_object_gripper_fit",
            return_value=(True, {"reason": "local section fits"}),
        ):
            safe, detail = ws._long_object_subsection_entry_safe(
                geometry, jaw_plan,
                metric_width_mm=25.9, metric_major_mm=42.4,
            )

        self.assertTrue(safe, detail)
        self.assertGreater(detail["aspect_ratio"], 1.5)
        self.assertIn("continue beyond the fingers", detail["reason"])

    def test_broad_object_cannot_use_subsection_entry_exception(self):
        geometry = SimpleNamespace(grasp_detection=object())
        jaw_plan = {"ok": True, "preopen_mm": 50.0}
        with patch.object(
            ws, "_validate_object_gripper_fit",
            return_value=(True, {"reason": "local section fits"}),
        ) as fit:
            safe, detail = ws._long_object_subsection_entry_safe(
                geometry, jaw_plan,
                metric_width_mm=35.0, metric_major_mm=42.0,
            )

        self.assertFalse(safe)
        self.assertIn("aspect ratio", detail["reason"])
        fit.assert_not_called()

    def test_projected_span_accepts_submillimetre_uncertainty_boundary(self):
        detail = {
            "ok": True,
            "observed_width_mm": 41.93,
            "conservative_width_mm": 50.24,
        }
        with (
            patch.object(ws, "GRASP_MIN_OPEN_CLEARANCE_MM", 8.0),
            patch.object(ws, "GRASP_OPEN_CLEARANCE_BOUNDARY_SLOP_MM", 0.75),
        ):
            safe, gate = ws._projected_3d_span_fits_aperture(detail, 50.0)

        self.assertTrue(safe, gate)
        self.assertAlmostEqual(gate["measured_required_open_mm"], 49.18)
        self.assertAlmostEqual(gate["conservative_excess_mm"], 0.24)

    def test_projected_span_refuses_real_aperture_overrun(self):
        detail = {
            "ok": True,
            "observed_width_mm": 42.8,
            "conservative_width_mm": 51.0,
        }
        with (
            patch.object(ws, "GRASP_MIN_OPEN_CLEARANCE_MM", 8.0),
            patch.object(ws, "GRASP_OPEN_CLEARANCE_BOUNDARY_SLOP_MM", 0.75),
        ):
            safe, gate = ws._projected_3d_span_fits_aperture(detail, 50.0)

        self.assertFalse(safe, gate)
        self.assertGreater(gate["measured_required_open_mm"], 50.0)
        self.assertGreater(gate["conservative_excess_mm"], 0.75)

    def test_projected_span_never_expands_hard_aperture(self):
        detail = {
            "ok": True,
            "observed_width_mm": 50.1,
            "conservative_width_mm": 50.2,
        }
        with (
            patch.object(ws, "GRASP_MIN_OPEN_CLEARANCE_MM", 0.0),
            patch.object(ws, "GRASP_OPEN_CLEARANCE_BOUNDARY_SLOP_MM", 0.75),
        ):
            safe, gate = ws._projected_3d_span_fits_aperture(detail, 50.0)

        self.assertFalse(safe, gate)

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
            info={
                "long_axis_deg": 0.0,
                "mask_u8": np.ones((200, 200), dtype=np.uint8),
                "depth_basis": "native_semantic_mask",
            },
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

    def test_locked_hybrid_checkpoint_uses_position_without_rebuilding_shape(self):
        frame = (
            np.zeros((20, 20, 3), dtype=np.uint8),
            np.full((20, 20), 220.0, dtype=np.float32),
            (100.0, 100.0, 10.0, 10.0),
        )
        features = [
            ws.WD.Detection(
                box=(4, 4, 12, 16), uv=(8.0, 10.0), z_mm=220.0,
                p_cam_mm=(1.0 + delta, 2.0, 220.0), source="sam3",
            )
            for delta in (0.0, 0.4)
        ]
        messages = []
        with (
            patch.object(ws, "_observe", side_effect=[
                (features[0], frame, {}), (features[1], frame, {}),
            ]),
            patch.object(ws, "_plan_scene_grasp_geometry") as shape_builder,
        ):
            found, found_frame, detail = \
                ws._observe_locked_geometry_position_consensus(
                    SimpleNamespace(), "red marker", messages.append,
                    max_frames=2,
        )

        self.assertIsNotNone(found, detail)
        self.assertIsNotNone(found_frame)
        self.assertEqual(found_frame[2], frame[2])
        self.assertEqual(
            found.info["position_basis"],
            "stable_semantic_identity_locked_geometry",
        )
        self.assertTrue(detail["locked_geometry"])
        self.assertFalse(detail["shape_reconstructed"])
        shape_builder.assert_not_called()

    def test_registered_anchor_tracking_reacquires_without_semantic_query(self):
        frame = (
            np.zeros((480, 640, 3), dtype=np.uint8),
            np.full((480, 640), 300.0, dtype=np.float32),
            (400.0, 400.0, 320.0, 240.0),
        )
        eye = SimpleNamespace(frame=lambda: frame)
        box = {
            "center_m": [0.30, 0.0, 0.20],
            "observed_size_m": [0.018, 0.018, 0.085],
        }
        with (
            patch.object(ws.A, "ee_pos", return_value=np.array([0.30, 0.0, 0.20])),
            patch.object(ws.A, "ee_rpy", return_value=np.zeros(3)),
            patch.object(ws, "load_saved_camera_jacobian", return_value=(
                np.eye(3), {"reason": "ok"}
            )),
            patch.object(ws.time, "sleep"),
        ):
            found, found_frame, detail = \
                ws._observe_registered_target_position_consensus(
                    eye, box, [0.0, 0.0, 300.0], lambda _line: None,
                    max_frames=2,
                )

        self.assertIsNotNone(found, detail)
        self.assertIsNotNone(found_frame)
        self.assertEqual(found_frame[2], frame[2])
        self.assertEqual(found.source, "registered:local_rgbd_anchor")
        self.assertEqual(detail["tracking_mode"], "registered_local_rgbd")

    def test_locked_metric_subsection_reprojects_across_camera_pose(self):
        grasp = ws.WD.Detection(
            box=(1, 1, 5, 5), uv=(3.0, 3.0), z_mm=100.0,
            p_cam_mm=(100.0, 100.0, 100.0), source="depth:ranked",
            min_width_mm=18.0, max_width_mm=90.0,
            info={"selected_candidate": {"axis_fraction": 0.0}},
        )
        geometry = ws.GG.GraspGeometry(
            True, "locked", grasp, grasp, (0.0, 0.0, 0.0)
        )
        box = {
            "center_m": [0.0, 0.0, 0.3],
            "observed_size_m": [0.02, 0.02, 0.09],
            "axes_base": np.eye(3).tolist(),
        }

        projected, detail = ws._reproject_locked_geometry_at_pose(
            geometry, box,
            camera_mapping=np.eye(3) * 1000.0,
            ee_xyz_m=[0.0, 0.0, 0.0],
            tool_camera_mm=[0.0, 0.0, 500.0],
            intr=(100.0, 100.0, 10.0, 10.0),
            current_rpy=(0.1, 0.2, 0.3),
        )

        self.assertIsNotNone(projected, detail)
        np.testing.assert_allclose(
            projected.grasp_detection.p_cam_mm, [0.0, 0.0, 200.0]
        )
        self.assertEqual(projected.grasp_detection.min_width_mm, 18.0)
        self.assertEqual(projected.rpy_rad, (0.1, 0.2, 0.3))
        self.assertTrue(detail["pose_reprojection"])
        self.assertIn("no fresh mask", detail["reason"])

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
            info={
                "long_axis_deg": 0.0,
                "mask_u8": np.ones((200, 200), dtype=np.uint8),
                "depth_basis": "native_semantic_mask",
            },
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

    def test_aperture_clips_position_compression_to_the_ceiling(self):
        """An absurd request is clipped, not honoured -- the ceiling is now
        configurable so a round object can be squeezed harder than 4mm."""
        det = SimpleNamespace(min_width_mm=18.5, max_width_mm=28.0)

        with (
            patch.object(ws, "SIZE_SAFETY_MARGIN_MM", 0.0),
            patch.object(ws, "SIZE_COMPRESSION_MM", 20.0),
            patch.object(ws, "GRIPPER_MAX_COMPRESSION_MM", 4.0),
        ):
            plan = ws._plan_aperture(det)

        self.assertTrue(plan["ok"], plan["reason"])
        self.assertAlmostEqual(plan["grasp_mm"], 14.5,
                               delta=ws.SIZE_TARGET_TOL_MM)
        self.assertEqual(plan["planned_compression_mm"], 4.0)

    def test_a_raised_ceiling_squeezes_further(self):
        """The change the round-marker slip needed: bite deeper than 4mm."""
        det = SimpleNamespace(min_width_mm=23.0, max_width_mm=28.0)

        with (
            patch.object(ws, "SIZE_SAFETY_MARGIN_MM", 0.0),
            patch.object(ws, "SIZE_COMPRESSION_MM", 6.0),
            patch.object(ws, "GRIPPER_MAX_COMPRESSION_MM", 8.0),
        ):
            plan = ws._plan_aperture(det)

        self.assertTrue(plan["ok"], plan["reason"])
        self.assertGreater(plan["planned_compression_mm"], 4.0)
        self.assertIn("6.0mm planned position compression", plan["reason"])

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

    def test_object_fit_boundary_allows_logged_point_six_mm_only(self):
        logged, logged_jaw = self._elongated_fit_fixture(width_mm=42.6)
        real_miss, miss_jaw = self._elongated_fit_fixture(width_mm=43.0)
        with (
            patch.object(ws, "GRASP_MIN_OPEN_CLEARANCE_MM", 8.0),
            patch.object(
                ws, "GRASP_OPEN_CLEARANCE_BOUNDARY_SLOP_MM", 0.75
            ),
        ):
            logged_ok, detail = ws._validate_object_gripper_fit(
                logged, logged_jaw
            )
            miss_ok, miss_detail = ws._validate_object_gripper_fit(
                real_miss, miss_jaw
            )

        self.assertTrue(logged_ok, detail)
        self.assertFalse(miss_ok, miss_detail)
        self.assertEqual(detail["effective_clearance_mm"], 7.25)
        self.assertEqual(detail["clearance_boundary_slop_mm"], 0.75)

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

    def test_axis_aware_slot_allows_object_overhang_but_centres_subsection(self):
        geometry, jaw_plan = self._elongated_fit_fixture()
        expected = np.array([0.0, 0.0, 0.0])
        with (
            patch.object(ws, "load_grasp_target_detail", return_value={
                "jaw_long_axis_camera_deg": -90.0,
            }),
            patch.object(ws, "DYNAMIC_SLOT_TOL_MM", 6.0),
        ):
            fits, detail = ws._jaw_slot_fit(
                np.array([-5.66, 5.5, 4.7]), expected,
                jaw_plan, geometry,
            )
            misses, miss_detail = ws._jaw_slot_fit(
                np.array([-5.66, 10.5, 4.7]), expected,
                jaw_plan, geometry,
            )

        self.assertTrue(fits, detail)
        self.assertFalse(misses, miss_detail)
        self.assertGreater(
            detail["subsection_capture"]["major_mm"],
            jaw_plan["max_aperture_mm"],
        )
        self.assertTrue(
            detail["subsection_capture"]["whole_object_overhang_allowed"]
        )
        self.assertEqual(detail["along_limit_mm"], 6.0)
        self.assertLess(detail["across_error_mm"], detail["across_limit_mm"])

    def test_local_subsection_profile_dynamically_tightens_near_object_end(self):
        geometry, jaw_plan = self._elongated_fit_fixture(
            width_mm=20.0, major_mm=100.0, end_fraction=0.68
        )
        selected = {
            "uv": [10.0, 10.0], "width_mm": 20.0,
            "axis_fraction": 0.32, "end_clearance_fraction": 0.68,
        }
        geometry.grasp_detection.info["selected_candidate"] = selected
        geometry.candidates = [
            {"uv": [8.0, 10.0], "width_mm": 20.4,
             "axis_fraction": -0.16},
            {"uv": [9.0, 10.0], "width_mm": 20.2,
             "axis_fraction": 0.0},
            {"uv": [10.0, 10.0], "width_mm": 20.0,
             "axis_fraction": 0.16},
            selected,
        ]

        with patch.dict(os.environ, {
            "REBOT_GRASP_SUBSECTION_CENTER_RESERVE_MM": "2.0"
        }):
            limit, detail = ws._local_subsection_along_limit_mm(
                geometry, jaw_plan,
                {"major_mm": 100.0, "end_clearance_mm": 34.0},
                6.0,
            )

        self.assertAlmostEqual(limit, 3.0)
        self.assertEqual(
            detail["source"], "measured_local_cross_section_profile"
        )
        self.assertTrue(detail["whole_object_overhang_allowed"])
        self.assertLess(detail["right_support_mm"],
                        detail["left_support_mm"])

    def test_central_stable_subsection_uses_calibrated_six_mm_cap(self):
        geometry, jaw_plan = self._elongated_fit_fixture(
            width_mm=20.0, major_mm=100.0
        )
        selected = {
            "uv": [10.0, 10.0], "width_mm": 20.0,
            "axis_fraction": 0.0, "end_clearance_fraction": 1.0,
        }
        geometry.grasp_detection.info["selected_candidate"] = selected
        geometry.candidates = [
            {"uv": [8.0, 10.0], "width_mm": 20.4,
             "axis_fraction": -0.16},
            selected,
            {"uv": [12.0, 10.0], "width_mm": 20.2,
             "axis_fraction": 0.16},
        ]

        limit, detail = ws._local_subsection_along_limit_mm(
            geometry, jaw_plan,
            {"major_mm": 100.0, "end_clearance_mm": 50.0},
            6.0,
        )

        self.assertAlmostEqual(limit, 6.0)
        self.assertGreater(detail["local_half_support_mm"], 8.0)
        self.assertTrue(detail["whole_object_overhang_allowed"])

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

    def test_close_requires_dynamic_lateral_centring_inside_open_jaws(self):
        jaw_plan = {
            "min_width_mm": 18.0,
            "grasp_mm": 14.0,
            "planned_compression_mm": 4.0,
        }
        physical_slot = {
            "reason": "inside open-jaw capture envelope",
            "across_error_mm": 6.0,
            "across_limit_mm": 12.0,
            "fit_ok": True,
        }

        with patch.object(
            ws, "DYNAMIC_CLOSE_CENTER_BOUNDARY_SLOP_MM", 0.5
        ):
            ready, detail = ws._jaw_slot_centered_for_close(
                True, physical_slot, jaw_plan
            )
            centered, centered_detail = ws._jaw_slot_centered_for_close(
                True, {**physical_slot, "across_error_mm": 2.1}, jaw_plan
            )
            outside, outside_detail = ws._jaw_slot_centered_for_close(
                True, {**physical_slot, "across_error_mm": 2.6}, jaw_plan
            )

        self.assertFalse(ready)
        self.assertTrue(centered)
        self.assertFalse(outside)
        self.assertEqual(detail["close_center_nominal_limit_mm"], 2.0)
        self.assertEqual(detail["close_center_limit_mm"], 2.5)
        self.assertEqual(detail["close_center_boundary_slop_mm"], 0.5)
        self.assertIn("close centring", detail["reason"])
        self.assertTrue(centered_detail["close_centered"])
        self.assertFalse(outside_detail["close_centered"])

    def test_close_center_hysteresis_never_exceeds_physical_slot(self):
        jaw_plan = {
            "min_width_mm": 18.0,
            "grasp_mm": 14.0,
            "planned_compression_mm": 4.0,
        }
        physical_slot = {
            "reason": "inside a narrow physical envelope",
            "across_error_mm": 1.9,
            "across_limit_mm": 1.8,
            "fit_ok": True,
        }

        with patch.object(
            ws, "DYNAMIC_CLOSE_CENTER_BOUNDARY_SLOP_MM", 0.5
        ):
            centered, detail = ws._jaw_slot_centered_for_close(
                True, physical_slot, jaw_plan
            )

        self.assertFalse(centered)
        self.assertEqual(detail["close_center_limit_mm"], 1.8)

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

    def test_simple_endpoint_presence_does_not_reapply_subsection_alignment(self):
        geometry, jaw_plan = self._elongated_fit_fixture(
            width_mm=10.9, major_mm=100.0
        )
        jaw_plan["preopen_mm"] = 28.7
        color = np.zeros((100, 100, 3), dtype=np.uint8)
        depth = np.full((100, 100), 120.0, dtype=np.float32)
        depth[35:66, 43:58] = 65.0
        frame = (color, depth, (100.0, 100.0, 50.0, 50.0))

        with (
            patch.object(ws, "load_grasp_target_detail", return_value={
                "jaw_long_axis_camera_deg": -90.0,
            }),
            patch.object(
                ws, "_jaw_slot_fit",
                side_effect=AssertionError(
                    "presence-only detection must not run alignment"
                ),
            ),
        ):
            detection, reason = ws._jaw_slot_depth_occupancy_detection(
                frame, np.array([0.0, 0.0, 65.0]), jaw_plan, geometry,
                physical_presence_only=True,
            )

        self.assertIsNotNone(detection, reason)
        self.assertTrue(detection.info["physical_between_jaws"])
        self.assertIn("physical between-jaw occupancy", reason)

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

    def test_closed_jaw_slot_occupancy_excludes_finger_zones(self):
        geometry, jaw_plan = self._elongated_fit_fixture(
            width_mm=10.9, major_mm=100.0
        )
        jaw_plan["preopen_mm"] = 28.7
        color = np.zeros((100, 100, 3), dtype=np.uint8)
        depth = np.full((100, 100), 120.0, dtype=np.float32)
        # Central object plus two larger same-depth finger-like strips. At
        # z=65mm and fx=100, the strips lie roughly +/-6.5mm across the slot.
        depth[35:66, 47:54] = 65.0
        depth[25:76, 39:43] = 65.0
        depth[25:76, 58:62] = 65.0
        frame = (color, depth, (100.0, 100.0, 50.0, 50.0))

        with patch.object(ws, "load_grasp_target_detail", return_value={
            "jaw_long_axis_camera_deg": -90.0,
        }):
            detection, reason = ws._jaw_slot_depth_occupancy_detection(
                frame, np.array([0.0, 0.0, 65.0]), jaw_plan, geometry,
                occupancy_aperture_mm=11.0,
            )

        self.assertIsNotNone(detection, reason)
        self.assertLess(abs(detection.p_cam_mm[0]), 2.0)
        self.assertEqual(
            detection.info["endpoint_width_basis"],
            "registered_shape_plus_slot_occupancy",
        )

    def test_closed_jaw_slot_occupancy_cannot_use_fingers_as_object(self):
        geometry, jaw_plan = self._elongated_fit_fixture(
            width_mm=10.9, major_mm=100.0
        )
        jaw_plan["preopen_mm"] = 28.7
        color = np.zeros((100, 100, 3), dtype=np.uint8)
        depth = np.full((100, 100), 120.0, dtype=np.float32)
        depth[25:76, 39:43] = 65.0
        depth[25:76, 58:62] = 65.0
        frame = (color, depth, (100.0, 100.0, 50.0, 50.0))

        with patch.object(ws, "load_grasp_target_detail", return_value={
            "jaw_long_axis_camera_deg": -90.0,
        }):
            detection, reason = ws._jaw_slot_depth_occupancy_detection(
                frame, np.array([0.0, 0.0, 65.0]), jaw_plan, geometry,
                occupancy_aperture_mm=11.0,
            )

        self.assertIsNone(detection)
        self.assertIn("no stable-size depth foreground", reason)

    def test_gripper_close_shortfall_counts_only_open_side_error(self):
        from arm_agent import gripper_model as gm

        with patch.object(gm, "OPEN_RAD", 2.6), \
                patch.object(gm, "CLOSE_RAD", 0.0):
            shortfall = ws._gripper_close_target_shortfall_rad(
                {"position_rad": 0.67}, 0.58
            )
            overshot = ws._gripper_close_target_shortfall_rad(
                {"position_rad": 0.50}, 0.58
            )

        self.assertAlmostEqual(shortfall, 0.09)
        self.assertEqual(overshot, 0.0)

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

    def test_hold_observation_keeps_guarded_slot_witness_consistent(self):
        # Full-body detection can reappear at a different centroid after a
        # micro-lift.  If the pre-lift evidence came from the guarded jaw slot,
        # compare it to the post-lift slot evidence instead of mixing them.
        before, after, basis = ws._hold_observation_pair(
            [3.0, 30.0, 60.0], [11.0, 22.0, 62.0],
            None, None,
            before_slot_cam_mm=[3.0, 30.0, 60.0],
            after_slot_cam_mm=[3.4, 29.8, 59.7],
            prefer_slot_witness=True,
        )

        self.assertTrue(np.allclose(before, [3.0, 30.0, 60.0]))
        self.assertTrue(np.allclose(after, [3.4, 29.8, 59.7]))
        self.assertEqual(basis, "guarded_jaw_slot_depth_occupancy")

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

    def test_post_close_baseline_marks_guarded_slot_witness(self):
        baseline = ws._post_close_hold_baseline(
            [3.0, 30.0, 60.0],
            {
                "basis": "guarded_closed_jaw_depth_occupancy",
                "slot_occupancy_cam_mm": [3.0, 30.0, 60.0],
            },
        )

        self.assertTrue(baseline["ok"], baseline)
        self.assertEqual(baseline["witness_kind"], "guarded_jaw_slot_depth")
        self.assertEqual(
            baseline["basis"], "post_close_guarded_jaw_slot_depth"
        )

    def test_post_close_baseline_rejects_when_all_identity_is_occluded(self):
        baseline = ws._post_close_hold_baseline(
            None, {"feature_cam_mm": None}
        )

        self.assertFalse(baseline["ok"])
        self.assertEqual(baseline["basis"], "unobserved")

    def test_post_close_baseline_uses_frozen_preclose_section_when_closed_jaws_occlude_everything(self):
        baseline = ws._post_close_hold_baseline(
            None, {"feature_cam_mm": None},
            frozen_preclose_body_cam_mm=[6.7, 17.8, 57.1],
        )

        self.assertTrue(baseline["ok"], baseline)
        self.assertTrue(np.allclose(
            baseline["body_cam_mm"], [6.7, 17.8, 57.1]
        ))
        self.assertEqual(baseline["basis"], "frozen_preclose_grasp_section")
        self.assertIn("only as the micro-lift baseline", baseline["reason"])

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

    def test_reference_shape_filter_allows_long_axis_completion(self):
        reference = {
            "center_m": [0.40, 0.02, 0.10],
            "axes_base": np.eye(3).T.tolist(),
            "observed_size_m": [0.020, 0.030, 0.070],
        }
        # Extending the measured 70 mm major axis is legitimate completion.
        extension = np.array([
            [0.40, 0.02, 0.16],
            [0.401, 0.021, 0.17],
        ] * 15)

        filtered, detail = ws._filter_target_points_to_reference_shape(
            extension, reference
        )

        self.assertTrue(detail["ok"], detail)
        self.assertEqual(len(filtered), len(extension))
        self.assertEqual(detail["unbounded_axis"], 2)

    def test_reference_shape_filter_removes_nearby_transverse_spill(self):
        reference = {
            "center_m": [0.40, 0.02, 0.10],
            "axes_base": np.eye(3).T.tolist(),
            "observed_size_m": [0.020, 0.030, 0.070],
        }
        target = np.array([
            [0.40 + offset, 0.02, 0.10]
            for offset in np.linspace(-0.009, 0.009, 30)
        ])
        # Still inside a 75 mm centroid sphere, but outside the measured
        # object's transverse cross-section.
        spill = np.array([
            [0.445, 0.02, 0.10 + offset]
            for offset in np.linspace(-0.02, 0.02, 25)
        ])

        filtered, detail = ws._filter_target_points_to_reference_shape(
            np.vstack((target, spill)), reference
        )

        self.assertTrue(detail["ok"], detail)
        self.assertEqual(len(filtered), len(target))
        self.assertEqual(detail["removed"], len(spill))

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

    def test_metric_box_must_stop_growing_before_hybrid_acceptance(self):
        first = {
            "observed_size_m": [0.0155, 0.0235, 0.0300], "views": 1,
        }
        growing = {
            "observed_size_m": [0.0154, 0.0259, 0.0424], "views": 2,
        }
        stable = {
            "observed_size_m": [0.0156, 0.0262, 0.0430], "views": 3,
        }

        converged, detail = ws._metric_box_extent_convergence(first, growing)
        self.assertFalse(converged)
        self.assertIn("still changing", detail["reason"])
        converged, detail = ws._metric_box_extent_convergence(growing, stable)
        self.assertTrue(converged, detail)

    def test_metric_box_accepts_stable_repeat_after_bounded_discovery(self):
        previous = {
            "observed_size_m": [0.0083, 0.0158, 0.0852], "views": 2,
        }
        current = {
            "observed_size_m": [0.0083, 0.0160, 0.0908], "views": 2,
        }

        converged, detail = ws._metric_box_extent_convergence(
            previous, current
        )

        self.assertTrue(converged, detail)
        self.assertTrue(detail["repeated_view_set"])
        self.assertIn("added no new target mask", detail["reason"])

    def test_metric_box_does_not_accept_single_view_repeat(self):
        previous = {
            "observed_size_m": [0.0083, 0.0158, 0.0852], "views": 1,
        }
        current = {
            "observed_size_m": [0.0083, 0.0158, 0.0852], "views": 1,
        }

        converged, detail = ws._metric_box_extent_convergence(
            previous, current
        )

        self.assertFalse(converged, detail)
        self.assertIn("no newly registered target mask", detail["reason"])

    def test_gravity_locked_sponge_converges_by_tabletop_axes(self):
        previous = {
            "observed_size_m": [0.0810, 0.0221, 0.0169],
            "axes_base": [
                [-0.995, -0.099, 0.0],
                [0.099, -0.995, 0.0],
                [0.0, 0.0, 1.0],
            ],
            "orientation_mode": "three_view_gravity_locked_obb",
            "views": 4,
        }
        current = {
            "observed_size_m": [0.0870, 0.0231, 0.0231],
            "axes_base": [
                [-0.995, -0.099, 0.0],
                [0.099, -0.995, 0.0],
                [0.0, 0.0, 1.0],
            ],
            "orientation_mode": "three_view_gravity_locked_obb",
            "views": 5,
        }

        converged, detail = ws._metric_box_extent_convergence(
            previous, current
        )

        self.assertTrue(converged, detail)
        self.assertEqual(
            detail["comparison_mode"], "gravity_locked_tabletop_axes"
        )
        self.assertIn("tabletop axes stable", detail["reason"])

    def test_gravity_snapped_upright_marker_keeps_generic_extent_convergence(self):
        previous = {
            "observed_size_m": [0.0308, 0.0212, 0.0828],
            "axes_base": [
                [0.30, -0.954, 0.0],
                [0.954, 0.30, 0.0],
                [0.0, 0.0, 1.0],
            ],
            "orientation_mode": "gravity_snapped_measured_obb(1.6deg)",
            "views": 4,
        }
        current = {
            "observed_size_m": [0.0314, 0.0223, 0.0862],
            "axes_base": [
                [0.307, -0.952, 0.0],
                [0.952, 0.307, 0.0],
                [0.0, 0.0, 1.0],
            ],
            "orientation_mode": "gravity_snapped_measured_obb(1.4deg)",
            "views": 5,
        }

        converged, detail = ws._metric_box_extent_convergence(
            previous, current
        )

        self.assertTrue(converged, detail)
        self.assertEqual(detail["comparison_mode"], "sorted_3d_extents")
        self.assertIn("all measured box extents stable", detail["reason"])

    def test_elongated_target_accepts_long_axis_completion(self):
        previous = {
            "observed_size_m": [0.0199, 0.0159, 0.0427], "views": 3,
        }
        current = {
            "observed_size_m": [0.0172, 0.0186, 0.0971], "views": 4,
        }

        converged, detail = ws._metric_box_extent_convergence(
            previous, current
        )

        self.assertTrue(converged, detail)
        self.assertEqual(
            detail["comparison_mode"], "elongated_long_axis_completion"
        )
        self.assertIn("long axis completed", detail["reason"])

    def test_elongated_target_rejects_transverse_growth(self):
        previous = {
            "observed_size_m": [0.0160, 0.0200, 0.0430], "views": 3,
        }
        current = {
            "observed_size_m": [0.0230, 0.0320, 0.0970], "views": 4,
        }

        converged, detail = ws._metric_box_extent_convergence(
            previous, current
        )

        self.assertFalse(converged, detail)
        self.assertEqual(detail["comparison_mode"], "sorted_3d_extents")

    def test_elongated_target_needs_four_registered_views_for_completion(self):
        previous = {
            "observed_size_m": [0.0199, 0.0159, 0.0427], "views": 2,
        }
        current = {
            "observed_size_m": [0.0172, 0.0186, 0.0971], "views": 3,
        }

        converged, detail = ws._metric_box_extent_convergence(
            previous, current
        )

        self.assertFalse(converged, detail)
        self.assertEqual(detail["comparison_mode"], "sorted_3d_extents")

    def test_supported_three_view_completion_replaces_partial_single_view(self):
        previous = {
            "observed_size_m": [0.0180, 0.0119, 0.0575], "views": 1,
        }
        current = {
            "observed_size_m": [0.0192, 0.0104, 0.0882], "views": 3,
        }
        support = {
            "ok": True, "applied": True, "views": 3,
            "reason": "points repeated across at least two of 3 views",
        }

        converged, detail = ws._metric_box_extent_convergence(
            previous, current, current_support=support,
        )

        self.assertTrue(converged, detail)
        self.assertEqual(
            detail["comparison_mode"],
            "supported_three_view_long_axis_completion",
        )
        self.assertIn("long axis completed", detail["reason"])

    def test_three_view_completion_without_cross_view_support_is_rejected(self):
        previous = {
            "observed_size_m": [0.0180, 0.0119, 0.0575], "views": 1,
        }
        current = {
            "observed_size_m": [0.0192, 0.0104, 0.0882], "views": 3,
        }

        converged, detail = ws._metric_box_extent_convergence(
            previous, current,
            current_support={"ok": False, "applied": False, "views": 3},
        )

        self.assertFalse(converged, detail)
        self.assertEqual(detail["comparison_mode"], "sorted_3d_extents")

    def test_gravity_locked_sponge_still_rejects_growing_footprint(self):
        previous = {
            "observed_size_m": [0.040, 0.020, 0.016],
            "axes_base": np.eye(3).tolist(),
            "orientation_mode": "three_view_gravity_locked_obb",
            "views": 3,
        }
        current = {
            "observed_size_m": [0.080, 0.030, 0.022],
            "axes_base": np.eye(3).tolist(),
            "orientation_mode": "three_view_gravity_locked_obb",
            "views": 4,
        }

        converged, detail = ws._metric_box_extent_convergence(
            previous, current
        )

        self.assertFalse(converged, detail)
        self.assertEqual(
            detail["comparison_mode"], "gravity_locked_tabletop_axes"
        )

    def test_hybrid_discovery_can_require_translated_camera_baseline(self):
        detection = SimpleNamespace(
            box=(100, 100, 200, 200), z_mm=240.0, source="sam3",
        )
        frame = (np.zeros((480, 640, 3), dtype=np.uint8), None, None)
        current = np.array([0.301, 0.0, 0.249], dtype=float)

        def execute_translation(path, _log, fixed_rpy=None):
            current[:] = np.asarray(path[-1], dtype=float)
            return True, "ok"

        with (
            patch.object(ws.A, "ee_pos", side_effect=lambda: current.tolist()),
            patch.object(ws.A, "ee_rpy", return_value=[0.0, 0.0, 0.0]),
            patch.object(ws, "_plan_rpy_trajectory",
                         return_value=([(0.0, 0.0, 0.0)], "ok")),
            patch.object(ws, "_execute_rpy_trajectory",
                         return_value=(True, "ok")),
            patch.object(ws, "_plan_fixed_approach_trajectory",
                         side_effect=lambda endpoint, **_kwargs:
                         ([tuple(endpoint)], "ok")),
            patch.object(ws, "_execute_fixed_approach_trajectory",
                         side_effect=execute_translation),
            patch.object(ws, "_settle"),
            patch.object(ws, "_observe",
                         return_value=(detection, frame, "found")),
        ):
            found, _frame, _geometry, detail = ws._look_around_object(
                SimpleNamespace(), "yellow sponge", lambda _line: None,
                require_grasp=False,
                minimum_translation_from_m=[0.301, 0.0, 0.249],
                minimum_translation_m=0.025,
            )

        self.assertIs(found, detection)
        self.assertEqual(detail["pose"], "pitched_retreat")
        self.assertGreaterEqual(
            np.linalg.norm(current - np.array([0.301, 0.0, 0.249])), 0.025
        )

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

    def test_registered_identifier_body_can_seed_bounded_alignment(self):
        body, detail = ws._registered_identifier_body_for_alignment(
            [2.8, 34.7, 64.8],
            {"stable_frames": 5, "spread_mm": 0.9},
            [-2.8, 8.0, -9.7],
        )

        self.assertIsNotNone(body, detail)
        np.testing.assert_allclose(body, [5.6, 26.7, 74.5])
        self.assertIn("closure still requires", detail["reason"])

    def test_registered_identifier_body_requires_multiframe_identity(self):
        body, detail = ws._registered_identifier_body_for_alignment(
            [2.8, 34.7, 64.8],
            {"stable_frames": 1},
            [-2.8, 8.0, -9.7],
        )

        self.assertIsNone(body)
        self.assertIn("1/2 stable", detail["reason"])

    def test_registered_identifier_body_rejects_unbounded_offset(self):
        body, detail = ws._registered_identifier_body_for_alignment(
            [2.8, 34.7, 64.8],
            {"stable_frames": 5},
            [0.0, 0.0, 80.0],
        )

        self.assertIsNone(body)
        self.assertIn("75mm", detail["reason"])

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

    def test_propagated_body_uses_tight_fresh_close_width(self):
        def endpoint(width, grasp_mm):
            detection = ws.WD.Detection(
                box=(2, 2, 18, 18), uv=(10.0, 10.0), z_mm=65.0,
                p_cam_mm=(8.0, 31.0, 65.0), source="depth:ranked",
                min_width_mm=width, max_width_mm=70.0,
            )
            geometry = ws.GG.GraspGeometry(
                True, "close geometry", detection, detection,
                (0.0, 0.0, 0.0), candidates=[],
            )
            return {
                "width_mm": width,
                "body_cam_mm": list(detection.p_cam_mm),
                "plan": {
                    "ok": True, "grasp_mm": grasp_mm,
                    "grasp_rad": grasp_mm / 20.0,
                    "preopen_mm": 50.0, "max_aperture_mm": 50.0,
                    "planned_compression_mm": width - grasp_mm,
                },
                "geometry": geometry,
            }

        prior = endpoint(25.6, 21.6)
        close_samples = [
            endpoint(7.97, 5.0), endpoint(7.51, 5.0),
            endpoint(8.12, 5.0),
        ]
        with patch.object(
            ws, "_minimum_commandable_aperture_mm", return_value=4.8
        ):
            selected, detail = \
                ws._endpoint_with_propagated_body_and_fresh_width(
                    prior, close_samples, [1.7, 31.7, 65.0], 22.68, 25.62,
                )

        self.assertTrue(detail["fresh_width_applied"], detail)
        self.assertAlmostEqual(selected["width_mm"], 7.97)
        self.assertAlmostEqual(selected["plan"]["grasp_mm"], 5.0)
        np.testing.assert_allclose(
            selected["geometry"].grasp_detection.p_cam_mm,
            [1.7, 31.7, 65.0],
        )

    def test_propagated_body_retains_prior_width_when_close_width_is_noisy(self):
        prior = {"width_mm": 25.6, "plan": {"grasp_mm": 21.6}}
        noisy = [
            {"width_mm": 7.5}, {"width_mm": 10.0},
            {"width_mm": 13.5},
        ]
        selected, detail = \
            ws._endpoint_with_propagated_body_and_fresh_width(
                prior, noisy, [1.7, 31.7, 65.0], 22.68, 25.62,
            )

        self.assertIs(selected, prior)
        self.assertFalse(detail["fresh_width_applied"])
        self.assertAlmostEqual(selected["plan"]["grasp_mm"], 21.6)

    def test_two_endpoint_batches_must_still_form_three_frame_consensus(self):
        def sample(width, x):
            return {
                "width_mm": width,
                "body_cam_mm": [x, 31.7, 65.0],
                "token": width,
            }

        first = {
            "locked_samples": [
                sample(15.47, 3.60), sample(15.49, 3.62)
            ],
            "fresh_samples": [],
            "identifier_samples": [],
        }
        second = {
            "locked_samples": [sample(15.48, 3.61)],
            "fresh_samples": [],
            "identifier_samples": [],
        }

        merged = ws._merge_endpoint_measurement_batches(
            first, second, 25.19, 28.19,
            section_locked=True, allow_reselection=False,
        )

        self.assertIsNotNone(merged["selected"], merged["width_detail"])
        self.assertIsNotNone(merged["body_cam"], merged["body_detail"])
        self.assertEqual(len(merged["locked_samples"]), 3)
        self.assertEqual(merged["merged_stationary_batches"], 2)
        self.assertTrue(
            merged["width_detail"]["stable_body_endpoint_override"]
        )

    def test_two_endpoint_batches_do_not_accept_only_two_total_frames(self):
        sample = {
            "width_mm": 15.48,
            "body_cam_mm": [3.6, 31.7, 65.0],
        }
        first = {
            "locked_samples": [sample], "fresh_samples": [],
            "identifier_samples": [],
        }
        second = {
            "locked_samples": [dict(sample)], "fresh_samples": [],
            "identifier_samples": [],
        }

        merged = ws._merge_endpoint_measurement_batches(
            first, second, 25.19, 28.19,
            section_locked=True, allow_reselection=False,
        )

        self.assertIsNone(merged["selected"])
        self.assertIsNone(merged["body_cam"])
        self.assertIn("only 2/3", merged["width_detail"]["reason"])

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

    def test_scene_relative_stack_uses_exact_reference_xy(self):
        det = SimpleNamespace(
            min_width_mm=20.0, max_width_mm=40.0, source="sam3"
        )
        attempted = []

        def route(endpoint, _rpy, log=None):
            attempted.append(np.asarray(endpoint, dtype=float))
            return [tuple(float(value) for value in endpoint)], "ik ok"

        with (
            patch.object(ws, "_scene_reference_consensus", return_value=(
                det, np.array([3.0, 32.0, 70.0]), {"reason": "stable"}
            )),
            patch.object(ws, "load_grasp_target",
                         return_value=(3.0, 32.0, 70.0)),
            patch.object(ws, "load_saved_camera_jacobian", return_value=(
                np.eye(3) * 1000.0, {"reason": "rigid"}
            )),
            patch.object(ws, "_stack_reference_top_z", return_value=(
                0.05, {"reason": "measured top"}
            )),
            patch.object(ws.A, "ee_pos", return_value=(0.30, 0.0, 0.22)),
            patch.object(ws.A, "ee_rpy", return_value=(0.0, 0.0, 0.0)),
            patch.object(ws, "_high_release_waypoints", side_effect=route),
        ):
            endpoint, waypoints, detail = ws._plan_scene_relative_release(
                "green marker", "stack", 0.10, {}, SimpleNamespace(),
                lambda _msg: None,
            )

        self.assertIsNotNone(endpoint, detail)
        self.assertIsNotNone(waypoints)
        self.assertEqual(len(attempted), 1)
        self.assertTrue(np.allclose(endpoint, [0.30, 0.0, 0.20]))
        self.assertEqual(detail["selected_relation"], "stack")
        self.assertEqual(detail["offset_m"], 0.0)
        self.assertIn("exact reference XY", detail["distance_basis"])

    def test_stack_payload_centering_compensates_held_object_offset(self):
        context = {
            "payload_box": {
                "center_offset_ee_m": [0.012, -0.008, -0.030],
            }
        }

        endpoint_xy, detail = ws._payload_centered_release_xy(
            [0.45, 0.05], context, (0.0, 0.0, 0.0)
        )

        self.assertIsNotNone(endpoint_xy, detail)
        np.testing.assert_allclose(endpoint_xy, [0.438, 0.058])
        self.assertEqual(
            detail["desired_object_center_xy_m"], [0.45, 0.05]
        )

    def test_stack_reference_top_uses_the_compact_mask_depth_not_table_z(self):
        depth = np.full((20, 20), 120.0, dtype=np.float32)
        depth[:10, :] = 100.0
        detection = SimpleNamespace(info={
            "mask_u8": np.ones((20, 20), dtype=np.uint8)
        })

        top_z, detail = ws._stack_reference_top_z(
            detection,
            (np.zeros((20, 20, 3), dtype=np.uint8), depth,
             (100.0, 100.0, 10.0, 10.0)),
            np.eye(3) * 1000.0,
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 0.30),
        )

        self.assertIsNotNone(top_z, detail)
        self.assertAlmostEqual(float(top_z), 0.20, places=3)
        self.assertGreater(detail["height_mm"], 10.0)

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

    def test_stack_reference_rejects_broad_multi_object_semantic_mask(self):
        frame = (
            np.zeros((480, 640, 3), dtype=np.uint8),
            np.full((480, 640), 334.0, dtype=np.float32),
            (430.0, 430.0, 320.0, 240.0),
        )
        detection = SimpleNamespace(
            source="sam3", box=(185, 210, 608, 385),
            min_width_mm=136.0, max_width_mm=329.0,
            info={
                "depth_basis": "semantic_seeded_rgbd_object",
                "completion_area_ratio": 4.4,
                "semantic_overlap_fraction": 0.80,
                "semantic_to_object_center_delta_mm": 13.0,
            },
        )

        accepted, reason = ws._stack_reference_detection_quality(
            detection, frame
        )

        self.assertFalse(accepted)
        self.assertTrue(
            "image" in reason or "spans" in reason,
            reason,
        )

    def test_stack_reference_accepts_compact_semantic_instance(self):
        frame = (
            np.zeros((480, 640, 3), dtype=np.uint8),
            np.full((480, 640), 300.0, dtype=np.float32),
            (430.0, 430.0, 320.0, 240.0),
        )
        detection = SimpleNamespace(
            source="sam3", box=(450, 275, 555, 410),
            min_width_mm=72.0, max_width_mm=96.0,
            info={
                "depth_basis": "native_semantic_mask",
                "mask_u8": np.ones((480, 640), dtype=np.uint8),
            },
        )

        accepted, reason = ws._stack_reference_detection_quality(
            detection, frame
        )

        self.assertTrue(accepted, reason)
        self.assertIn("compact semantic stack reference", reason)

    def test_stack_reference_recovers_compact_native_mask_from_merged_completion(self):
        depth = np.full((480, 640), 300.0, dtype=np.float32)
        native = np.zeros_like(depth, dtype=np.uint8)
        native[220:330, 420:500] = 255
        completed = np.zeros_like(depth, dtype=np.uint8)
        completed[190:380, 350:590] = 255
        frame = (
            np.zeros((480, 640, 3), dtype=np.uint8), depth,
            (430.0, 430.0, 320.0, 240.0),
        )
        detection = SimpleNamespace(
            source="sam3", box=(350, 190, 590, 380), uv=(460.0, 275.0),
            z_mm=300.0, p_cam_mm=(97.7, 24.4, 300.0), area_px=45600.0,
            min_width_mm=70.0, max_width_mm=150.0,
            info={
                "mask_u8": completed,
                "semantic_seed_mask_u8": native,
                "semantic_anchor_uv": [460.0, 275.0],
                "depth_basis": "semantic_seeded_rgbd_object",
                "completion_area_ratio": 2.4,
                "semantic_overlap_fraction": 0.95,
                "semantic_to_object_center_delta_mm": 55.0,
            },
        )

        accepted, reason = ws._stack_reference_detection_quality(
            detection, frame
        )

        self.assertTrue(accepted, reason)
        self.assertTrue(
            detection.info["stack_reference_compact_native_mask"]
        )
        self.assertEqual(
            detection.info["depth_basis"],
            "native_semantic_compact_stack_reference",
        )
        self.assertLess(detection.area_px, 10000.0)

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
        self.assertEqual(ws._normalize_scene_relation("on top of"), "stack")
        self.assertEqual(ws._normalize_scene_relation("stack on"), "stack")

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
                "across_error_mm": 0.0,
                "across_limit_mm": 12.0,
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
                {"min_width_mm": 18.0, "grasp_mm": 14.0,
                 "planned_compression_mm": 4.0}, SimpleNamespace(),
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

    def test_hybrid_grasp_look_around_scopes_wider_camera_envelope(self):
        detection = SimpleNamespace(
            box=(100, 100, 200, 200), z_mm=200.0, source="sam3",
            p_cam_mm=(0.0, 0.0, 200.0), info={},
        )
        frame = (np.zeros((480, 640, 3), dtype=np.uint8), None, None)
        geometry = SimpleNamespace(ok=True, reason="fits")
        envelope_calls = []
        logs = []

        with (
            patch.object(ws.A, "ee_pos", return_value=[0.301, 0.0, 0.249]),
            patch.object(ws.A, "ee_rpy", return_value=[0.0, 0.0, 0.0]),
            patch.object(ws, "_plan_rpy_trajectory",
                         return_value=([(0.0, 0.0, np.deg2rad(12.0))], "ok")),
            patch.object(ws, "_execute_rpy_trajectory",
                         return_value=(True, "ok")),
            patch.object(ws, "_settle"),
            patch.object(ws, "_observe",
                         return_value=(detection, frame, "found")),
            patch.object(ws, "_plan_scene_grasp_geometry",
                         return_value=geometry),
            patch.object(ws, "_append_pose_aware_discovery_view",
                         return_value=True),
            patch.object(ws, "_set_active_camera_envelope",
                         side_effect=envelope_calls.append),
        ):
            found, _frame, _geometry, detail = ws._look_around_object(
                SimpleNamespace(), "blue marker", logs.append,
                require_grasp=True,
                geometry_accept=lambda _geometry: (True, "fits"),
                captured_views=[],
            )

        self.assertIs(found, detection)
        self.assertTrue(detail["ok"])
        self.assertEqual(envelope_calls, [30.0, None])
        self.assertTrue(any(
            "look_around_camera_envelope status=raised" in line
            for line in logs
        ))

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

    def test_look_around_adds_farther_retreat_after_original_sweep(self):
        """A target clipped in all original views gets a second camera origin."""
        partial = SimpleNamespace(
            box=(0, 100, 200, 200), z_mm=200.0, source="sam3"
        )
        complete = SimpleNamespace(
            box=(100, 100, 200, 200), z_mm=240.0, source="sam3"
        )
        frame = (np.zeros((480, 640, 3), dtype=np.uint8), None, None)
        observations = [
            *((partial, frame, "found") for _ in range(6)),
            (complete, frame, "found"),
        ]
        planned_endpoints = []

        def plan_translation(endpoint, **_kwargs):
            planned_endpoints.append(np.asarray(endpoint, dtype=float))
            return [tuple(endpoint)], "ok"

        with (
            patch.object(ws.A, "ee_pos", return_value=[0.301, 0.0, 0.249]),
            patch.object(ws.A, "ee_rpy", return_value=[0.0, 0.0, 0.0]),
            patch.object(ws, "_plan_rpy_trajectory",
                         return_value=([(0.0, 0.0, 0.0)], "ok")),
            patch.object(ws, "_execute_rpy_trajectory",
                         return_value=(True, "ok")),
            patch.object(ws, "_plan_fixed_approach_trajectory",
                         side_effect=plan_translation),
            patch.object(ws, "_execute_fixed_approach_trajectory",
                         return_value=(True, "ok")),
            patch.object(ws, "_settle"),
            patch.object(ws, "_observe", side_effect=observations),
        ):
            found, _frame, _geometry, detail = ws._look_around_object(
                SimpleNamespace(), "yellow sponge", lambda _line: None,
                require_grasp=False,
            )

        self.assertIs(found, complete)
        self.assertEqual(detail["pose"], "extended_pitched_retreat")
        self.assertTrue(any(
            np.isclose(endpoint[0], 0.301 - 0.075)
            for endpoint in planned_endpoints
        ))

    def test_grasp_discovery_can_use_calibrated_edge_recenter(self):
        """The translated recenter path is available to grasp targets too."""
        partial = SimpleNamespace(
            box=(0, 100, 200, 200), z_mm=200.0, source="sam3",
            p_cam_mm=(100.0, 0.0, 200.0),
        )
        complete = SimpleNamespace(
            box=(100, 100, 200, 200), z_mm=240.0, source="sam3",
            p_cam_mm=(0.0, 0.0, 240.0),
        )
        frame = (np.zeros((480, 640, 3), dtype=np.uint8), None, None)
        geometry = SimpleNamespace(ok=True, reason="fits")
        captured = []

        with (
            patch.object(ws.A, "ee_pos", return_value=[0.301, 0.0, 0.249]),
            patch.object(ws.A, "ee_rpy", return_value=[0.0, 0.0, 0.0]),
            patch.object(ws, "_plan_rpy_trajectory",
                         return_value=([(0.0, 0.0, 0.0)], "ok")),
            patch.object(ws, "_execute_rpy_trajectory",
                         return_value=(True, "ok")),
            patch.object(ws, "_plan_fixed_approach_trajectory",
                         return_value=([(0.25, 0.0, 0.249)], "ok")),
            patch.object(ws, "_execute_fixed_approach_trajectory",
                         return_value=(True, "ok")),
            patch.object(ws, "_settle"),
            patch.object(ws, "_observe", return_value=(partial, frame, "found")),
            patch.object(ws, "load_saved_camera_jacobian",
                         return_value=(np.eye(3) * 1000.0, {"reason": "ok"})),
            patch.object(ws, "_recenter_clipped_placement_reference",
                         return_value=(complete, frame, {"ok": True})),
            patch.object(ws, "_plan_scene_grasp_geometry",
                         return_value=geometry),
            patch.object(ws, "_append_pose_aware_discovery_view",
                         return_value=True) as append_view,
        ):
            found, found_frame, found_geometry, detail = \
                ws._look_around_object(
                    SimpleNamespace(), "yellow sponge", lambda _line: None,
                    require_grasp=True,
                    geometry_accept=lambda _geometry: (True, "fits"),
                    captured_views=captured,
                    allow_edge_recenter=True,
                )

        self.assertIs(found, complete)
        self.assertIs(found_frame, frame)
        self.assertIs(found_geometry, geometry)
        self.assertEqual(detail["pose"], "edge_recenter")
        self.assertEqual(
            append_view.call_args_list[-1].kwargs["name"], "edge_recenter"
        )

    def test_grasp_discovery_uses_rgbd_fallback_only_to_recenter(self):
        """A broad colour/depth component must never become grasp geometry."""
        fallback = SimpleNamespace(
            box=(100, 100, 300, 300), z_mm=240.0,
            source="rgbd:colored_object", p_cam_mm=(20.0, 0.0, 240.0),
        )
        semantic = SimpleNamespace(
            box=(180, 120, 260, 300), z_mm=235.0,
            source="sam3", p_cam_mm=(0.0, 0.0, 235.0),
        )
        frame = (np.zeros((480, 640, 3), dtype=np.uint8), None, None)
        geometry = SimpleNamespace(ok=True, reason="semantic marker fits")
        captured = []
        logs = []

        def semantic_only(det, _frame):
            if det.source == "rgbd:colored_object":
                return False, (
                    "RGB-D fallback is location-only and recenter-only; "
                    "semantic mask required"
                )
            return True, "semantic source"

        with (
            patch.object(ws.A, "ee_pos", return_value=[0.301, 0.0, 0.249]),
            patch.object(ws.A, "ee_rpy", return_value=[0.0, 0.0, 0.0]),
            patch.object(ws, "_plan_rpy_trajectory",
                         return_value=([(0.0, 0.0, 0.0)], "ok")),
            patch.object(ws, "_execute_rpy_trajectory",
                         return_value=(True, "ok")),
            patch.object(ws, "_plan_fixed_approach_trajectory",
                         return_value=([(0.25, 0.0, 0.249)], "ok")),
            patch.object(ws, "_execute_fixed_approach_trajectory",
                         return_value=(True, "ok")),
            patch.object(ws, "_settle"),
            patch.object(ws, "_observe", return_value=(
                fallback, frame, "found"
            )),
            patch.object(ws, "_detection_partial", return_value=None),
            patch.object(ws, "load_saved_camera_jacobian",
                         return_value=(np.eye(3) * 1000.0, {"reason": "ok"})),
            patch.object(ws, "_recenter_clipped_placement_reference",
                         return_value=(semantic, frame, {"ok": True})) as recenter,
            patch.object(ws, "_plan_scene_grasp_geometry",
                         return_value=geometry) as plan_geometry,
            patch.object(ws, "_append_pose_aware_discovery_view",
                         return_value=True) as append_view,
        ):
            found, _found_frame, found_geometry, detail = \
                ws._look_around_object(
                    SimpleNamespace(), "purple marker", logs.append,
                    require_grasp=True,
                    geometry_accept=lambda _geometry: (True, "fits"),
                    captured_views=captured,
                    allow_edge_recenter=True,
                    detection_accept=semantic_only,
                )

        self.assertIs(found, semantic)
        self.assertIs(found_geometry, geometry)
        self.assertEqual(detail["pose"], "edge_recenter")
        self.assertEqual(plan_geometry.call_count, 1)
        self.assertEqual(append_view.call_count, 1)
        self.assertIs(recenter.call_args.kwargs["detection_accept"], semantic_only)
        self.assertTrue(any(
            "location-only fallback cannot contribute" in message
            for message in logs
        ))

    def test_grasp_discovery_can_recenter_deferred_identity_hint(self):
        """A clipped unanchored mask is camera-motion input, not a target."""
        partial = SimpleNamespace(
            box=(100, 300, 200, 479), z_mm=240.0, source="sam3",
            p_cam_mm=(0.0, 80.0, 240.0),
        )
        complete = SimpleNamespace(
            box=(100, 100, 200, 300), z_mm=240.0, source="sam3",
            p_cam_mm=(0.0, 0.0, 240.0),
        )
        frame = (np.zeros((480, 640, 3), dtype=np.uint8), None, None)
        geometry = SimpleNamespace(ok=True, reason="fits")

        class DeferredEye:
            def take_deferred_identity_candidate(self, _description):
                return partial, frame, {
                    "anchor_deferred": True,
                    "reason": (
                        "semantic candidate cannot establish identity because "
                        "semantic/depth envelope is clipped at the image "
                        "bottom edge"
                    ),
                }

        messages = []
        with (
            patch.object(ws.A, "ee_pos", return_value=[0.301, 0.0, 0.249]),
            patch.object(ws.A, "ee_rpy", return_value=[0.0, 0.0, 0.0]),
            patch.object(ws, "_plan_rpy_trajectory",
                         return_value=([(0.0, 0.0, 0.0)], "ok")),
            patch.object(ws, "_execute_rpy_trajectory",
                         return_value=(True, "ok")),
            patch.object(ws, "_plan_fixed_approach_trajectory",
                         return_value=([(0.25, 0.0, 0.249)], "ok")),
            patch.object(ws, "_execute_fixed_approach_trajectory",
                         return_value=(True, "ok")),
            patch.object(ws, "_settle"),
            patch.object(ws, "_observe", return_value=(
                None, None, {"identity_gate": {"anchor_deferred": True}}
            )),
            patch.object(ws, "load_saved_camera_jacobian",
                         return_value=(np.eye(3) * 1000.0, {"reason": "ok"})),
            patch.object(ws, "_recenter_clipped_placement_reference",
                         return_value=(complete, frame, {"ok": True})),
            patch.object(ws, "_plan_scene_grasp_geometry",
                         return_value=geometry),
        ):
            found, _found_frame, found_geometry, detail = \
                ws._look_around_object(
                    DeferredEye(), "red marker", messages.append,
                    require_grasp=True,
                    geometry_accept=lambda _geometry: (True, "fits"),
                    allow_edge_recenter=True,
                )

        self.assertIs(found, complete)
        self.assertIs(found_geometry, geometry)
        self.assertEqual(detail["pose"], "edge_recenter")
        self.assertTrue(any(
            "authority=recenter_only" in message for message in messages
        ))

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

    def test_placement_reference_recenter_uses_short_sideways_leg_when_full_move_fails_ik(self):
        intr = (500.0, 500.0, 320.0, 240.0)
        frame = (
            np.zeros((480, 640, 3), dtype=np.uint8),
            np.full((480, 640), 200.0, dtype=np.float32), intr,
        )
        clipped = SimpleNamespace(
            box=(580, 180, 639, 300), uv=(639.0, 240.0), z_mm=200.0,
            p_cam_mm=(127.6, 0.0, 200.0), source="sam3",
        )
        complete_a = SimpleNamespace(
            box=(280, 180, 400, 300), uv=(340.0, 240.0), z_mm=200.0,
            p_cam_mm=(8.0, 0.0, 200.0), source="sam3",
        )
        complete_b = SimpleNamespace(
            box=(281, 180, 401, 300), uv=(341.0, 240.0), z_mm=200.0,
            p_cam_mm=(8.4, 0.0, 200.0), source="sam3",
        )
        planned = []

        def plan(endpoint, **_kwargs):
            planned.append(np.asarray(endpoint, dtype=float))
            if len(planned) == 1:
                return None, "local IK notch"
            return [tuple(endpoint)], "ok"

        with (
            patch.object(ws.A, "ee_pos", return_value=[0.30, 0.01, 0.25]),
            patch.object(ws.A, "ee_rpy", return_value=[0.0, 0.0, 0.0]),
            patch.object(ws, "load_saved_camera_jacobian",
                         return_value=(np.eye(3) * 1000.0, {"reason": "ok"})),
            patch.object(ws, "_plan_fixed_approach_trajectory", side_effect=plan),
            patch.object(ws, "_execute_fixed_approach_trajectory",
                         return_value=(True, "ok")),
            patch.object(ws, "_settle"),
            patch.object(ws, "_observe", side_effect=[
                (complete_a, frame, "found"),
                (complete_b, frame, "found"),
            ]),
            patch.object(ws, "_detection_partial", side_effect=lambda det, _f: (
                "clipped" if det is clipped else None
            )),
            patch.object(ws.time, "sleep"),
        ):
            found, _found_frame, detail = \
                ws._recenter_clipped_placement_reference(
                    SimpleNamespace(), "white cup", clipped, frame,
                    lambda _line: None,
                    observation_consensus_frames=3,
                    observation_consensus_max_spread_mm=15.0,
                )

        self.assertIs(found, complete_a)
        self.assertTrue(detail["ok"], detail)
        self.assertGreaterEqual(len(planned), 2)
        delta = planned[1] - np.array([0.30, 0.01, 0.25])
        self.assertLessEqual(np.linalg.norm(delta), 0.020001)

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

    @staticmethod
    def _shape_vote(view, sizes_m, major_axis_rows=None):
        return {
            "view": view,
            "box": {
                "center_m": [0.45, 0.0, 0.05],
                "axes_base": (
                    np.eye(3) if major_axis_rows is None
                    else np.asarray(major_axis_rows, dtype=float)
                ).tolist(),
                "observed_size_m": list(sizes_m),
            },
            "points": np.ones((30, 3), dtype=float),
        }

    def test_shape_axis_consensus_ignores_partial_view_axis_swap(self):
        # The clipped first view mistakes the real 45 mm width (base X) for
        # the object's major axis. Three later complete views agree that the
        # true ~81 mm length is base Y.
        candidates = [
            self._shape_vote("partial", [0.045, 0.019, 0.016]),
            self._shape_vote("wide", [0.044, 0.081, 0.020]),
            self._shape_vote("retreat", [0.043, 0.080, 0.021]),
            self._shape_vote("yaw", [0.045, 0.082, 0.020]),
        ]

        selected, detail = ws._target_shape_axis_consensus(candidates)

        self.assertEqual(
            [item["view"] for item in selected],
            ["wide", "retreat", "yaw"],
        )
        self.assertNotIn("partial", detail["views"])
        self.assertLessEqual(detail["maximum_axis_delta_deg"], 0.1)

    def test_shape_axis_consensus_does_not_lock_from_two_views(self):
        candidates = [
            self._shape_vote("partial_a", [0.045, 0.019, 0.016]),
            self._shape_vote("partial_b", [0.046, 0.020, 0.017]),
        ]

        selected, detail = ws._target_shape_axis_consensus(candidates)

        self.assertEqual(selected, [])
        self.assertIn("transverse filtering remains disabled", detail["reason"])

    def test_shape_axis_signature_rejects_nearly_square_view(self):
        axis, detail = ws._target_shape_major_axis_signature({
            "axes_base": np.eye(3).tolist(),
            "observed_size_m": [0.044, 0.048, 0.020],
        })

        self.assertIsNone(axis)
        self.assertIn("ambiguous", detail["reason"])

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

    def test_metric_recovery_cannot_merge_rejected_masks_into_valid_target(self):
        frame = (
            np.zeros((8, 8, 3), dtype=np.uint8),
            np.full((8, 8), 220.0, dtype=np.float32),
            (100.0, 100.0, 4.0, 4.0),
        )
        mask = np.ones((8, 8), dtype=np.uint8) * 255
        target = ws.WD.Detection(
            box=(2, 2, 6, 6), uv=(4.0, 4.0), z_mm=220.0,
            p_cam_mm=(0.0, 0.0, 220.0), source="sam3",
            info={"mask_u8": mask, "depth_basis": "native_semantic_mask"},
        )
        valid = SimpleNamespace(
            ok=True, envelope_detection=target, grasp_detection=target,
        )
        rejected = SimpleNamespace(
            ok=False, envelope_detection=target, grasp_detection=target,
        )
        views = [
            {"name": "valid_narrow", "frame": frame,
             "detection": target, "geometry": valid,
             "ee_xyz_m": np.array([0.30, 0.0, 0.25]),
             "camera_mapping": np.eye(3) * 1000.0},
            {"name": "rejected_broad", "frame": frame,
             "detection": target, "geometry": rejected,
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
            "views": 1, "observed_size_m": [0.015, 0.016, 0.095],
            "orientation_mode": "test",
        })
        logs = []
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
                views, logs.append, force_geometry=True,
                require_semantic_target_masks=True,
            )

        self.assertTrue(result["ok"])
        masked.assert_called_once()
        self.assertTrue(any(
            "grasp_valid_semantic_views_only" in line for line in logs
        ))
        self.assertTrue(any(
            "rejected recovery mask excluded" in line for line in logs
        ))

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

    def test_unstable_body_retains_registered_section_transform(self):
        noisy_sample = {
            "detection": SimpleNamespace(p_cam_mm=(4.0, 30.0, 74.0)),
            "geometry": SimpleNamespace(grasp_detection=SimpleNamespace(
                p_cam_mm=(4.0, 28.7, 71.4), info={}
            )),
        }
        registered = np.array([-9.9, -28.1, 10.6])

        offset, fraction, detail = \
            ws._endpoint_section_lock_from_measurement(
                noisy_sample, None,
                registered_section_locked=True,
                registered_feature_to_body_cam_mm=registered,
                registered_axis_fraction=0.0,
            )

        np.testing.assert_allclose(offset, registered)
        self.assertEqual(fraction, 0.0)
        self.assertTrue(detail["registered_transform_retained"])
        self.assertIn("arbitrary noisy frame", detail["reason"])

    def test_converged_body_may_lock_fresh_section_transform(self):
        sample = {
            "detection": SimpleNamespace(p_cam_mm=(4.0, 30.0, 74.0)),
            "geometry": SimpleNamespace(grasp_detection=SimpleNamespace(
                p_cam_mm=(2.0, 25.0, 65.0),
                info={"selected_candidate": {"axis_fraction": -0.16}},
            )),
        }

        offset, fraction, detail = \
            ws._endpoint_section_lock_from_measurement(
                sample, [2.1, 25.0, 65.1],
                registered_section_locked=True,
                registered_feature_to_body_cam_mm=[20.0, 20.0, 20.0],
                registered_axis_fraction=0.0,
            )

        np.testing.assert_allclose(offset, [2.0, 5.0, 9.0])
        self.assertAlmostEqual(fraction, -0.16)
        self.assertIn("stable close-range", detail["reason"])

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

    def test_strict_checkpoint_lock_never_reselects_spilled_fresh_body(self):
        fresh = [
            {"body_cam_mm": [10.0, 20.0, 100.0]},
            {"body_cam_mm": [10.2, 20.1, 100.1]},
        ]

        body, detail, selected = ws._checkpoint_section_consensus(
            [], fresh, section_locked=True, allow_reselection=False,
            min_frames=2, max_spread_mm=4.0,
        )

        self.assertIsNone(body)
        self.assertIn("only 0/2", detail["reason"])
        self.assertEqual(selected, [])

    def test_registered_central_section_can_recover_fresh_centre_position(self):
        allowed, detail = ws._registered_section_can_use_fresh_centre(0.0)

        self.assertTrue(allowed, detail)
        self.assertIn("inside", detail["reason"])

    def test_registered_implicit_centre_can_recover_fresh_centre_position(self):
        allowed, detail = ws._registered_section_can_use_fresh_centre(
            None, implicit_centre=True,
        )

        self.assertTrue(allowed, detail)
        self.assertTrue(detail["implicit_centre"])
        self.assertIn("implicit measured body centre", detail["reason"])

    def test_missing_fraction_is_not_centre_without_explicit_context(self):
        allowed, detail = ws._registered_section_can_use_fresh_centre(None)

        self.assertFalse(allowed, detail)
        self.assertIn("invalid", detail["reason"])

    def test_registered_noncentral_section_cannot_jump_to_fresh_centre(self):
        allowed, detail = ws._registered_section_can_use_fresh_centre(0.16)

        self.assertFalse(allowed, detail)
        self.assertIn("outside", detail["reason"])

    def test_horizontal_major_consensus_builds_gravity_locked_axes(self):
        votes = [
            {"major_axis": np.array([1.0, 0.02, 0.05])},
            {"major_axis": np.array([-0.99, -0.04, -0.08])},
            {"major_axis": np.array([0.98, -0.03, 0.10])},
        ]

        axes, detail = ws._horizontal_major_consensus_axes(votes)

        self.assertIsNotNone(axes, detail)
        np.testing.assert_allclose(axes[:, 2], [0.0, 0.0, 1.0])
        np.testing.assert_allclose(axes.T @ axes, np.eye(3), atol=1e-6)

    def test_tilted_major_consensus_does_not_force_tabletop_frame(self):
        angle = np.radians(38.0)
        axis = np.array([np.cos(angle), 0.0, np.sin(angle)])
        votes = [{"major_axis": axis.copy()} for _ in range(3)]

        axes, detail = ws._horizontal_major_consensus_axes(votes)

        self.assertIsNone(axes)
        self.assertIn("exceeds", detail["reason"])

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

    def test_strict_checkpoint_presence_ignores_shifting_mask_centroids(self):
        def sample(point):
            return {
                "point_base_m": point,
                "feature": SimpleNamespace(
                    source="sam3", info={"mask_u8": np.ones((2, 2), np.uint8)}
                ),
            }

        accepted, detail = ws._strict_checkpoint_identity_presence([
            sample([0.45, 0.06, 0.11]),
            sample([0.48, 0.09, 0.12]),
            sample([0.44, 0.04, 0.10]),
        ])

        self.assertIsNotNone(accepted, detail)
        self.assertFalse(detail["centroid_steering"])
        self.assertIn("were not used to steer", detail["reason"])

    def test_strict_checkpoint_presence_rejects_nonsemantic_fallbacks(self):
        samples = [
            {
                "point_base_m": [0.45, 0.06, 0.11],
                "feature": SimpleNamespace(
                    source="color:yellow", info={"mask_u8": np.ones((2, 2))}
                ),
            },
            {
                "point_base_m": [0.46, 0.06, 0.11],
                "feature": SimpleNamespace(source="sam3", info={}),
            },
        ]

        accepted, detail = ws._strict_checkpoint_identity_presence(samples)

        self.assertIsNone(accepted)
        self.assertIn("0/2", detail["reason"])

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

    def test_strict_endpoint_lock_never_reselects_spilled_fresh_width(self):
        fresh = [
            {
                "width_mm": width,
                "body_cam_mm": [10.0, 20.0, 100.0 + 0.1 * index],
            }
            for index, width in enumerate((114.8, 115.0, 115.1, 114.9, 115.0))
        ]

        result = ws._endpoint_measurement_consensus(
            [], fresh, 20.0, 45.0,
            section_locked=True, allow_reselection=False,
        )

        self.assertIsNone(result["selected"])
        self.assertFalse(result["section_reselected"])

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

    def test_second_complete_attempt_cannot_reuse_selected_section_offset(self):
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

        with (
            patch.dict(os.environ, {
                "REBOT_WRIST_JAW_LONG_AXIS_CAMERA_DEG": "-90",
            }),
            patch.object(ws, "_jaw_slot_fit", return_value=(True, {
                "across_limit_mm": 10.0,
            })),
            patch.object(ws, "_plan_aperture", return_value=jaw_plan),
            patch.object(ws, "_validate_object_gripper_fit",
                         return_value=(True, {"reason": "fits"})),
            patch.object(ws, "_reachable_approach_target") as reachable,
        ):
            result = ws._plan_collision_aware_grasp_pose(
                geometry, frame, np.eye(3) * 1000.0,
                [0.0, 0.0, 70.0], (0.0, 0.0, 0.0),
                jaw_plan, None, lambda _line: None,
                allow_pitch_search=False,
                require_alternate_section=True,
            )

        self.assertFalse(result["ok"], result)
        self.assertEqual(result["attempts"], 0)
        reachable.assert_not_called()

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

    def test_projected_metric_width_cannot_exceed_stale_cropped_major(self):
        body = ws.WD.Detection(
            box=(2, 2, 18, 18), uv=(10.0, 10.0), z_mm=200.0,
            p_cam_mm=(0.0, 0.0, 200.0), source="sam3",
            min_width_mm=18.0, max_width_mm=29.4,
        )
        geometry = ws.GG.GraspGeometry(
            True, "cropped restored view", body, body,
            (0.0, 0.0, 0.0), candidates=[],
        )

        updated = ws._geometry_with_projected_3d_width(geometry, {
            "ok": True,
            "observed_width_mm": 35.2,
            "conservative_width_mm": 42.0,
            "object_major_mm": 81.6,
            "reason": "measured OBB projection",
        })

        self.assertEqual(updated.grasp_detection.min_width_mm, 35.2)
        self.assertEqual(updated.grasp_detection.max_width_mm, 81.6)
        self.assertGreaterEqual(
            updated.grasp_detection.max_width_mm,
            updated.grasp_detection.min_width_mm,
        )

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
            patch.object(ws, "_ACTIVE_CAMERA_ENVELOPE_DEG", 30.0),
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
        self.assertEqual(
            result.grasp_detection.info[
                "metric_recovery_camera_envelope_deg"
            ],
            30.0,
        )

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

    def test_interrupt_recovery_opens_before_homing_when_object_is_held(self):
        ws._OBJECT_HELD = True
        events = []
        with (
            patch.object(ws, "_gripper", lambda action: events.append(action) or {"ok": True}),
            patch.object(ws, "_home", lambda: events.append("home") or {"ok": True}),
        ):
            result = ws._home_after_interrupt()

        self.assertEqual(events, ["open", "home"])
        self.assertTrue(result["gripper_opened"])
        self.assertTrue(result["homed"])

    def test_interrupt_recovery_homes_before_opening_when_nothing_is_held(self):
        ws._OBJECT_HELD = False
        events = []
        with (
            patch.object(ws, "_gripper", lambda action: events.append(action) or {"ok": True}),
            patch.object(ws, "_home", lambda: events.append("home") or {"ok": True}),
        ):
            result = ws._home_after_interrupt()

        self.assertEqual(events, ["home", "open"])
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


class LockedTargetLivePoseTest(TestCase):
    """Hand-off back into the pick pipeline must not require one nominal pose.

    Discovery recentres precisely when the calibrated ready pose cannot see the
    target. Re-establishing the fused object has to be possible at the pose that
    can see it, or a complete registered 3D target is discarded for a viewpoint
    already proven blind to the object.
    """

    def setUp(self) -> None:
        self.frame = (
            np.zeros((20, 20, 3), dtype=np.uint8),
            np.full((20, 20), 220.0, dtype=np.float32),
            (100.0, 100.0, 10.0, 10.0),
        )
        grasp = ws.WD.Detection(
            box=(1, 1, 5, 5), uv=(3.0, 3.0), z_mm=100.0,
            p_cam_mm=(100.0, 100.0, 100.0), source="depth:ranked",
            min_width_mm=18.0, max_width_mm=90.0,
            info={"selected_candidate": {"axis_fraction": 0.0}},
        )
        self.geometry = ws.GG.GraspGeometry(
            True, "locked", grasp, grasp, (0.0, 0.0, 0.0)
        )
        self.box = {
            "center_m": [0.0, 0.0, 0.3],
            "observed_size_m": [0.02, 0.02, 0.09],
            "axes_base": np.eye(3).tolist(),
        }
        self.features = [
            ws.WD.Detection(
                box=(4, 4, 12, 16), uv=(8.0, 10.0), z_mm=220.0,
                p_cam_mm=(1.0 + delta, 2.0, 220.0), source="sam3",
            )
            for delta in (0.0, 0.4)
        ]

    def _run(self, *, mapping, observations):
        with (
            patch.object(ws.A, "ee_pos", return_value=(0.0, 0.0, 0.0)),
            patch.object(ws.A, "ee_rpy", return_value=(0.1, 0.2, 0.3)),
            patch.object(ws, "load_saved_camera_jacobian",
                         return_value=mapping),
            patch.object(ws, "_observe", side_effect=observations),
        ):
            return ws._establish_locked_target_at_live_pose(
                SimpleNamespace(), "purple marker", lambda _m: None,
                self.geometry, self.box,
                tool_camera_mm=[0.0, 0.0, 500.0], max_frames=2,
            )

    def test_target_is_established_away_from_the_calibrated_ready_pose(self):
        det, frame, geometry, detail = self._run(
            mapping=(np.eye(3) * 1000.0, {"reason": "ok"}),
            observations=[(self.features[0], self.frame, {}),
                          (self.features[1], self.frame, {})],
        )

        self.assertIsNotNone(geometry, detail)
        self.assertIs(frame, self.frame)
        self.assertIsNotNone(det)
        # Shape stays locked to the registered box; only the pose is re-expressed.
        self.assertEqual(geometry.grasp_detection.min_width_mm, 18.0)
        self.assertEqual(geometry.rpy_rad, (0.1, 0.2, 0.3))

    def test_extreme_posture_is_refused_so_the_caller_can_fall_back(self):
        det, frame, geometry, detail = self._run(
            mapping=(None, {"reason": "wrist orientation is 40deg from the "
                                      "saved camera calibration"}),
            observations=[],
        )

        self.assertIsNone(geometry)
        self.assertTrue(detail["camera_envelope_rejected"])
        self.assertIn("camera calibration", detail["reason"])

    def test_unseen_target_is_refused_rather_than_invented(self):
        det, frame, geometry, detail = self._run(
            mapping=(np.eye(3) * 1000.0, {"reason": "ok"}),
            observations=[(None, None, "not found"), (None, None, "not found")],
        )

        self.assertIsNone(geometry)
        self.assertTrue(detail["identity_rejected"])


class CheckpointMetricReprojectionTest(TestCase):
    """The mask centroid is not a fixed physical point.

    The subsection is identified by a camera-frame vector from the semantic
    mask centroid to the body point. That centroid migrates along the object as
    the view scale changes between the planning distance and the much closer
    visible checkpoint, so the offset drifts past its millimetre tolerance even
    while target identity is steady. The registered metric box locates the same
    subsection without any centroid, which is what keeps a converged target from
    being thrown away at the last step.
    """

    def _locked_geometry(self):
        grasp = ws.WD.Detection(
            box=(1, 1, 5, 5), uv=(3.0, 3.0), z_mm=100.0,
            p_cam_mm=(100.0, 100.0, 100.0), source="depth:ranked",
            min_width_mm=18.0, max_width_mm=90.0,
            info={"selected_candidate": {"axis_fraction": 0.0}},
        )
        return ws.GG.GraspGeometry(True, "locked", grasp, grasp, (0.0, 0.0, 0.0))

    def test_registered_box_locates_the_subsection_without_a_centroid(self):
        box = {
            "center_m": [0.0, 0.0, 0.3],
            "observed_size_m": [0.02, 0.02, 0.09],
            "axes_base": np.eye(3).tolist(),
        }

        projected, detail = ws._reproject_locked_geometry_at_pose(
            self._locked_geometry(), box,
            camera_mapping=np.eye(3) * 1000.0,
            ee_xyz_m=[0.0, 0.0, 0.0],
            tool_camera_mm=[0.0, 0.0, 500.0],
            intr=(100.0, 100.0, 10.0, 10.0),
            current_rpy=(0.0, 0.0, 0.0),
        )

        self.assertIsNotNone(projected, detail)
        # Position is recovered from the box; shape and width stay locked.
        self.assertEqual(projected.grasp_detection.min_width_mm, 18.0)
        self.assertEqual(projected.grasp_detection.max_width_mm, 90.0)
        self.assertTrue(detail["pose_reprojection"])

    def test_position_consensus_still_gates_a_moved_object(self):
        """Reprojection is stale geometry if the object moved; that must fail."""
        anchor = np.array([0.40, 0.0, 0.05])
        moved = [
            {"point_base_m": (anchor + np.array([0.30, 0.0, 0.0])).tolist()},
            {"point_base_m": (anchor + np.array([0.31, 0.0, 0.0])).tolist()},
        ]

        median, detail, cluster = ws._checkpoint_target_position_consensus(
            moved, anchor, min_frames=2,
        )

        self.assertIsNone(median, detail)


class CrossSectionJawWidthTest(TestCase):
    """A subsection grasp must not be sized by the axis passing through the jaws.

    On an elongated object the major axis runs through the fingers rather than
    between them. Counting it — as both the 2D silhouette and the full OBB
    support function do — leaves the commanded opening wider than the object,
    and the close grips nothing.
    """

    @staticmethod
    def _axis(image_deg, optical_deg):
        a, o = np.radians(image_deg), np.radians(optical_deg)
        return [np.cos(a) * np.cos(o), np.sin(a) * np.cos(o), np.sin(o)]

    def setUp(self) -> None:
        self.mapping = np.eye(3) * 1000.0
        # Geometry measured on the purple marker: 9.1 x 20.5 x 77.4 mm, with no
        # axis aimed at the jaws (the 20.5mm one missed by 2.7deg).
        self.marker = {
            "axes_base": [self._axis(55.5, 48.1), self._axis(17.7, 14.5),
                          self._axis(84.1, 38.2)],
            "observed_size_m": [0.0091, 0.0205, 0.0774],
            "conservative_size_m": [0.0151, 0.0265, 0.0834],
        }

    def test_major_axis_is_excluded_from_the_jaw_width(self):
        full = ws._projected_3d_jaw_span(self.marker, self.mapping)
        cross = ws._projected_3d_cross_section_span(self.marker, self.mapping)

        self.assertTrue(cross["ok"], cross)
        # The excluded contribution is exactly the difference.
        # axis_contributions is rounded to 5 decimals in the returned dict.
        self.assertAlmostEqual(
            full["observed_width_mm"] - cross["observed_width_mm"],
            cross["excluded_major_contribution_mm"], places=3,
        )
        self.assertLess(cross["observed_width_mm"], full["observed_width_mm"])
        self.assertGreater(cross["elongation_ratio"], 1.5)

    def test_compact_object_keeps_the_full_span(self):
        """Every axis of a compact object is between the jaws."""
        compact = {
            "axes_base": np.eye(3).tolist(),
            "observed_size_m": [0.030, 0.032, 0.034],
            "conservative_size_m": [0.036, 0.038, 0.040],
        }

        cross = ws._projected_3d_cross_section_span(compact, self.mapping)

        self.assertTrue(cross["ok"], cross)
        # Elongation below the gate, so the caller keeps the full span.
        self.assertLess(cross["elongation_ratio"], 1.5)

    def test_cross_section_stays_conservative_within_itself(self):
        cross = ws._projected_3d_cross_section_span(self.marker, self.mapping)

        self.assertGreater(
            cross["conservative_width_mm"], cross["observed_width_mm"]
        )


class SingleViewBaselineConvergenceTest(TestCase):
    """Convergence classification is unchanged; only the collection budget is.

    A 1-view baseline genuinely has not demonstrated stability, so rejecting it
    is correct. The bug was spending the only collection round on that
    comparison, leaving no like-for-like one. These pin the classification that
    the budget change relies on.
    """

    BASE = {
        "axes_base": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        "orientation_mode": "gravity_snapped_measured_obb(5.0deg)",
    }

    def box(self, size_m, views):
        return {**self.BASE, "observed_size_m": size_m, "views": views}

    def test_single_view_baseline_is_still_rejected(self):
        # Measured on the purple marker: 34.1mm across from one view, 17.6mm
        # from three. Rejection is right; the run just needs another round.
        converged, detail = ws._metric_box_extent_convergence(
            self.box([0.0157, 0.0341, 0.0886], 1),
            self.box([0.0116, 0.0176, 0.0752], 3),
        )

        self.assertFalse(converged)
        self.assertIn("still changing", detail["reason"])

    def test_like_for_like_comparison_converges(self):
        converged, detail = ws._metric_box_extent_convergence(
            self.box([0.0116, 0.0176, 0.0752], 3),
            self.box([0.0118, 0.0179, 0.0757], 4),
        )

        self.assertTrue(converged, detail)

    def test_transverse_drift_is_still_refused(self):
        """The jaws close on the transverse axes, so drift there must fail."""
        for size in ([0.0116, 0.0300, 0.0752], [0.0220, 0.0176, 0.0752]):
            converged, detail = ws._metric_box_extent_convergence(
                self.box([0.0116, 0.0176, 0.0752], 3), self.box(size, 4),
            )
            self.assertFalse(converged, f"{size} -> {detail}")


class StallClampTest(TestCase):
    """Position control alone cannot hold a rigid object.

    The jaw plan commands width minus a few millimetres of "position
    compression". A sponge compresses, so that produces real clamping force. A
    marker does not: the jaws stall on contact and hold with only the small
    residual position error the stall produced — enough to touch it, not enough
    to carry it. Commanding further closure against jaws that are already
    blocked converts that stall into grip without changing any controller gain.
    """

    def test_clamp_target_is_tighter_but_respects_the_close_limit(self):
        from arm_agent import gripper_model as gm

        scale = gm.mm_per_rad()
        self.assertIsNotNone(scale)
        close_rad, open_rad = ws._wrist_limits()
        direction = 1.0 if open_rad >= close_rad else -1.0
        grasp_rad = np.radians(47.3)

        clamp_rad = gm.clamp_rad(
            grasp_rad - direction * ws.GRIPPER_CLAMP_EXTRA_MM / float(scale),
            (close_rad, open_rad),
        )

        # Tighter than the validated target...
        self.assertLess(clamp_rad, grasp_rad)
        # ...and never past the calibrated close limit.
        self.assertGreaterEqual(clamp_rad, min(close_rad, open_rad))

    def test_extra_closure_is_bounded(self):
        self.assertLessEqual(ws.GRIPPER_CLAMP_EXTRA_MM, 10.0)
        self.assertGreaterEqual(ws.GRIPPER_CLAMP_EXTRA_MM, 0.0)

    def test_shortfall_detects_a_stall_only_when_jaws_stop_open(self):
        target = np.radians(47.3)
        stalled = ws._gripper_close_target_shortfall_rad(
            {"position_rad": float(np.radians(50.91))}, target,
        )
        reached = ws._gripper_close_target_shortfall_rad(
            {"position_rad": float(target)}, target,
        )

        self.assertGreater(np.degrees(stalled), 2.0)
        self.assertEqual(reached, 0.0)


class SlotOccupancyApertureTest(TestCase):
    """The jaw-slot witness must be sized from where the jaws actually are.

    Its sampled volume keeps a 2mm margin inside the fingers so a finger cannot
    masquerade as a held object during the lift test. Sizing that volume from a
    commanded target the jaws closed past leaves the window straddling the
    fingers, and the object's measured prominence collapses even though the
    grasp is sound.
    """

    def half_width(self, aperture_mm):
        return min(12.0, max(2.0, 0.5 * aperture_mm - 2.0))

    def test_commanded_aperture_would_straddle_the_fingers(self):
        # Measured run: validated target 51.1deg, jaws clamped to 38.45deg.
        from arm_agent import gripper_model as gm

        commanded = gm.aperture_mm(np.radians(51.1))
        actual = gm.aperture_mm(np.radians(38.45))

        # Sizing from the commanded gap samples beyond the real finger faces.
        self.assertGreater(self.half_width(commanded), actual / 2.0)
        # Sizing from the measured gap stays inside them.
        self.assertLess(self.half_width(actual), actual / 2.0)

    def test_measured_window_keeps_its_margin(self):
        from arm_agent import gripper_model as gm

        actual = gm.aperture_mm(np.radians(38.45))
        self.assertAlmostEqual(
            actual / 2.0 - self.half_width(actual), 2.0, places=6
        )


class CameraOpticalCentreBaselineTest(TestCase):
    """Parallax must be measured at the optical centre, not the wrist endpoint.

    The camera is bolted about 67mm off the wrist rotation centre, so rotating
    at a fixed end-effector XYZ still swings it through a real arc. Measuring
    ``A.ee_pos()`` reported ~0.5mm for every discovery view in the field logs
    (0.3, 0.4, 0.9, 1.0mm against a 25mm requirement), which is residual servo
    error rather than the camera's actual motion.
    """

    def setUp(self):
        # Real saved calibration: rigid mount rotation and TCP-in-camera.
        self.mapping_cal = 1000.0 * np.array([
            [-0.016729715, 0.995612283, -0.092066811],
            [0.287418424, 0.092981964, 0.953281178],
            [-0.957659003, 0.010513575, 0.287712876],
        ])
        self.tool_mm = np.array([3.6095055, 3.9656094, 66.8999939])
        self.ee = np.array([0.3013, -0.0014, 0.2196])

    def rpy_matrix(self, roll, pitch, yaw):
        return ws._rpy_rotation_matrix((roll, pitch, yaw))

    def mapping_at(self, rpy):
        # Calibrated at rpy=0, so M_cur = M_cal @ R_cur^T.
        return self.mapping_cal @ self.rpy_matrix(*rpy).T

    def test_optical_centre_is_offset_from_the_endpoint(self):
        origin = ws._camera_origin_base_m(
            self.ee, self.mapping_cal, self.tool_mm
        )
        self.assertIsNotNone(origin)
        offset_mm = np.linalg.norm(origin - self.ee) * 1000.0
        # The mount offset is centimetres, not the ~0.5mm the gate reported.
        self.assertGreater(offset_mm, 50.0)
        self.assertLess(offset_mm, 90.0)

    def test_inverts_the_module_projection_convention(self):
        # p_cam_mm = tool - mapping @ (p_base - ee) must vanish at the origin.
        origin = ws._camera_origin_base_m(
            self.ee, self.mapping_cal, self.tool_mm
        )
        p_cam = self.tool_mm - self.mapping_cal @ (origin - self.ee)
        self.assertTrue(np.allclose(p_cam, np.zeros(3), atol=1e-6))

    def test_orientation_only_poses_carry_real_parallax(self):
        reference = ws._camera_origin_base_m(
            self.ee, self.mapping_at((0.0, 0.0, 0.0)), self.tool_mm
        )
        for pitch_deg, yaw_deg, low_mm, high_mm in (
            (0.0, 12.0, 12.0, 15.0),    # yaw_left
            (0.0, -12.0, 12.0, 15.0),   # yaw_right
            (8.0, 0.0, 8.0, 11.0),      # pitch_wide
        ):
            rpy = (0.0, math.radians(pitch_deg), math.radians(yaw_deg))
            live = ws._camera_origin_base_m(
                self.ee, self.mapping_at(rpy), self.tool_mm
            )
            baseline_mm = np.linalg.norm(live - reference) * 1000.0
            # The endpoint does not move at all for these poses.
            self.assertAlmostEqual(
                float(np.linalg.norm(self.ee - self.ee)), 0.0
            )
            self.assertGreater(baseline_mm, low_mm)
            self.assertLess(baseline_mm, high_mm)

    def test_orientation_only_cannot_reach_the_baseline_floor(self):
        """Why fixed-XYZ poses must not be tried first when parallax is asked.

        The clamp floor for ``REBOT_HYBRID_MIN_CAMERA_BASELINE_M`` is 15mm and
        the widest orientation-only pose gives ~13mm, so those poses are
        arithmetically incapable of terminating such a sweep.
        """
        reference = ws._camera_origin_base_m(
            self.ee, self.mapping_at((0.0, 0.0, 0.0)), self.tool_mm
        )
        best_mm = 0.0
        for pitch_deg, yaw_deg in ((0.0, 12.0), (0.0, -12.0), (8.0, 0.0)):
            rpy = (0.0, math.radians(pitch_deg), math.radians(yaw_deg))
            live = ws._camera_origin_base_m(
                self.ee, self.mapping_at(rpy), self.tool_mm
            )
            best_mm = max(best_mm, np.linalg.norm(live - reference) * 1000.0)
        self.assertLess(best_mm, 15.0)

    def test_retreat_poses_clear_the_default_requirement(self):
        reference = ws._camera_origin_base_m(
            self.ee, self.mapping_at((0.0, 0.0, 0.0)), self.tool_mm
        )
        pitched = (0.0, math.radians(8.0), 0.0)
        for retreat_m, minimum_mm in ((0.040, 25.0), (0.075, 60.0)):
            retreat_ee = self.ee - np.array([retreat_m, 0.0, 0.0])
            live = ws._camera_origin_base_m(
                retreat_ee, self.mapping_at(pitched), self.tool_mm
            )
            baseline_mm = np.linalg.norm(live - reference) * 1000.0
            self.assertGreater(baseline_mm, minimum_mm)

    def test_malformed_inputs_fail_closed(self):
        for ee, mapping, tool in (
            (None, self.mapping_cal, self.tool_mm),
            (self.ee, None, self.tool_mm),
            (self.ee, self.mapping_cal, None),
            (self.ee, np.eye(2), self.tool_mm),
            (np.array([0.0, 0.0]), self.mapping_cal, self.tool_mm),
            (self.ee, self.mapping_cal * np.nan, self.tool_mm),
        ):
            self.assertIsNone(ws._camera_origin_base_m(ee, mapping, tool))


class SweepPosePriorityTest(TestCase):
    """Fixed-XYZ poses are promoted past only for a bottom-edge clip.

    A near object runs off the bottom of the wrist image and yaw/pitch at a
    fixed endpoint cannot pull it back into frame; only retreating can.

    A requested parallax baseline is deliberately NOT a trigger, even though
    fixed-XYZ poses cannot satisfy one. Views are captured before the parallax
    gate, so those poses still feed metric fusion -- failing the baseline only
    stops them ending the sweep. Promoting on that basis regressed a live run
    (2026-08-21): retreated wide-FOV poses ran first, SAM 3 locked onto a
    different marker ~224mm away at every one, and the near poses that would
    have worked were left unreachable at the end of the sweep.
    """

    def partition(self, pending):
        """Mirror of the stable partition in ``promote_translating_poses``."""
        translating = [entry for entry in pending if entry[2]]
        stationary = [entry for entry in pending if not entry[2]]
        if not translating or not stationary:
            return list(pending)
        return translating + stationary

    def sweep(self):
        return [
            ("yaw_left", None, False, None),
            ("yaw_right", None, False, None),
            ("pitch_wide", None, False, None),
            ("pitched_retreat", None, True, None),
            ("pitched_retreat_yaw_left", None, True, None),
            ("pitched_retreat_yaw_right", None, True, None),
            ("extended_pitched_retreat", None, True, None),
        ]

    def test_translating_poses_lead_after_promotion(self):
        order = [entry[0] for entry in self.partition(self.sweep())]
        self.assertEqual(order[0], "pitched_retreat")
        self.assertLess(
            order.index("pitched_retreat"), order.index("yaw_left")
        )

    def test_promotion_drops_nothing_and_keeps_relative_order(self):
        original = self.sweep()
        promoted = self.partition(original)
        self.assertCountEqual(
            [entry[0] for entry in promoted], [entry[0] for entry in original]
        )
        for group in (True, False):
            self.assertEqual(
                [e[0] for e in promoted if e[2] is group],
                [e[0] for e in original if e[2] is group],
            )

    def test_partway_promotion_only_reorders_the_remainder(self):
        # yaw_left already tried and clipped at the bottom edge.
        remaining = self.sweep()[1:]
        order = [entry[0] for entry in self.partition(remaining)]
        self.assertNotIn("yaw_left", order)
        self.assertEqual(order[0], "pitched_retreat")

    def test_all_translating_remainder_is_left_alone(self):
        remaining = self.sweep()[3:]
        self.assertEqual(
            [e[0] for e in self.partition(remaining)],
            [e[0] for e in remaining],
        )

    def test_bottom_edge_reason_is_what_triggers_promotion(self):
        # The trigger reads the reason text the clip gate actually emits.
        clipped = ws._detection_partial(
            SimpleNamespace(box=(100.0, 100.0, 200.0, 480.0)),
            (np.zeros((480, 640, 3), dtype=np.uint8),),
        )
        self.assertIsNotNone(clipped)
        self.assertIn("bottom", clipped)
        # A side clip must not promote: yaw genuinely can reframe it.
        side = ws._detection_partial(
            SimpleNamespace(box=(0.0, 100.0, 200.0, 300.0)),
            (np.zeros((480, 640, 3), dtype=np.uint8),),
        )
        self.assertIsNotNone(side)
        self.assertIn("left", side)
        self.assertNotIn("bottom", side)


class RegisteredBoxOccupancyTest(TestCase):
    """Liveness at the collision-map gate must not need a fresh mask.

    Field run 2026-08-21: SAM 3 located the marker at 11:58:16 with 0.5mm
    sample spread, then missed three times in the following second at the same
    pose. The colour detector behind it matched a different blue object 464mm
    away, and the run died on "target could not be isolated in the
    collision-map frame" -- with a verified base-frame OBB already in hand.
    """

    def setUp(self):
        self.intr = (600.0, 600.0, 320.0, 240.0)
        self.ee = np.array([0.30, 0.0, 0.22])
        self.mapping = 1000.0 * np.array([
            [-0.016729715, 0.995612283, -0.092066811],
            [0.287418424, 0.092981964, 0.953281178],
            [-0.957659003, 0.010513575, 0.287712876],
        ])
        self.tool = np.array([3.6095055, 3.9656094, 66.8999939])
        self.box = {
            "center_m": np.array([0.52, -0.11, 0.061]),
            "axes_base": np.eye(3),
            "observed_size_m": np.array([0.0155, 0.0155, 0.0975]),
            "conservative_size_m": np.array([0.021, 0.021, 0.1035]),
        }
        self.centre_cam = self.tool - self.mapping @ (
            self.box["center_m"] - self.ee
        )
        self.uv = ws.WD.project(self.centre_cam, self.intr)

    def scene(self, depth_mm, patch=100):
        """Paint the object across its real projected extent, not a token dot.

        A 97.5mm marker projects ~180px tall at this range, so a 40px patch was
        smaller than the object it represented -- fine for a centre-only probe,
        misleading for one that samples the whole registered volume.
        """
        depth = np.full((480, 640), 800.0, dtype=np.float32)
        u, v = int(self.uv[0]), int(self.uv[1])
        depth[max(0, v - patch):v + patch, max(0, u - patch):u + patch] = depth_mm
        return (np.zeros((480, 640, 3), np.uint8), depth, self.intr)

    def check(self, frame):
        return ws._registered_box_occupancy(
            frame, self.box, camera_mapping=self.mapping,
            ee_xyz_m=self.ee, tool_camera_mm=self.tool,
        )

    def test_present_object_is_confirmed_without_any_detector(self):
        ok, detail = self.check(self.scene(self.centre_cam[2]))
        self.assertTrue(ok, detail.get("reason"))
        self.assertAlmostEqual(
            detail["observed_depth_mm"], self.centre_cam[2], places=0
        )

    def test_removed_object_refuses_and_is_conclusive(self):
        # Table surface behind where the marker used to stand.
        ok, detail = self.check(self.scene(self.centre_cam[2] + 120.0))
        self.assertFalse(ok)
        self.assertFalse(detail["inconclusive"])
        self.assertIn("no longer where it was registered", detail["reason"])

    def test_nearer_surface_also_refuses(self):
        ok, detail = self.check(self.scene(self.centre_cam[2] - 150.0))
        self.assertFalse(ok)
        self.assertFalse(detail["inconclusive"])

    def test_depth_hole_is_inconclusive_not_a_refusal(self):
        """An RGB-D hole must fall back to detection, never assert absence."""
        frame = (
            np.zeros((480, 640, 3), np.uint8),
            np.zeros((480, 640), np.float32),
            self.intr,
        )
        ok, detail = self.check(frame)
        self.assertFalse(ok)
        self.assertTrue(detail["inconclusive"])

    def span(self, box):
        """Camera-Z span the gate itself reports for a box.

        The marker stands vertically while the camera looks down at it, so the
        span is not any single box axis -- ask the code rather than assume one.
        """
        _ok, detail = ws._registered_box_occupancy(
            self.scene(self.centre_cam[2]), box, camera_mapping=self.mapping,
            ee_xyz_m=self.ee, tool_camera_mm=self.tool,
        )
        return detail["expected_near_mm"], detail["expected_far_mm"]

    def test_tolerance_admits_rgbd_noise_at_the_box_face(self):
        # A reading just inside either face, plus noise, still counts.
        near_mm, far_mm = self.span(self.box)
        for probe in (near_mm + 1.0, 0.5 * (near_mm + far_mm), far_mm - 1.0):
            ok, detail = self.check(self.scene(probe))
            self.assertTrue(ok, f"{probe:.0f}mm: {detail.get('reason')}")

    def test_tolerance_is_bounded_and_rejects_beyond_it(self):
        near_mm, far_mm = self.span(self.box)
        for probe in (near_mm - 5.0, far_mm + 5.0):
            ok, detail = self.check(self.scene(probe))
            self.assertFalse(ok, f"{probe:.0f}mm unexpectedly accepted")
            self.assertFalse(detail["inconclusive"])

    def test_conservative_size_is_preferred_over_observed(self):
        tight = dict(self.box)
        tight.pop("conservative_size_m")
        padded_near, _ = self.span(self.box)
        tight_near, _ = self.span(tight)
        # The padded box must admit strictly more than the tight one.
        self.assertLess(padded_near, tight_near)
        probe = 0.5 * (padded_near + tight_near)
        ok_conservative, _ = self.check(self.scene(probe))
        ok_observed, _ = ws._registered_box_occupancy(
            self.scene(probe), tight, camera_mapping=self.mapping,
            ee_xyz_m=self.ee, tool_camera_mm=self.tool,
        )
        self.assertTrue(ok_conservative)
        self.assertFalse(ok_observed)

    def test_offscreen_projection_is_inconclusive(self):
        far_box = dict(self.box)
        far_box["center_m"] = np.array([0.52, -1.20, 0.061])
        ok, detail = ws._registered_box_occupancy(
            self.scene(self.centre_cam[2]), far_box,
            camera_mapping=self.mapping, ee_xyz_m=self.ee,
            tool_camera_mm=self.tool,
        )
        self.assertFalse(ok)
        self.assertTrue(detail["inconclusive"])

    def test_malformed_box_is_inconclusive(self):
        for bad in (
            {},
            {"center_m": np.array([0.5, 0.0]), "axes_base": np.eye(3),
             "observed_size_m": np.array([0.01, 0.01, 0.1])},
            {"center_m": np.array([0.5, 0.0, 0.06]), "axes_base": np.eye(2),
             "observed_size_m": np.array([0.01, 0.01, 0.1])},
            {"center_m": np.array([np.nan, 0.0, 0.06]),
             "axes_base": np.eye(3),
             "observed_size_m": np.array([0.01, 0.01, 0.1])},
        ):
            ok, detail = ws._registered_box_occupancy(
                self.scene(self.centre_cam[2]), bad,
                camera_mapping=self.mapping, ee_xyz_m=self.ee,
                tool_camera_mm=self.tool,
            )
            self.assertFalse(ok)
            self.assertTrue(detail["inconclusive"])

    def test_a_wrong_instance_half_a_metre_away_cannot_confirm(self):
        """The exact Aug-21 failure: a same-coloured object 464mm off.

        Occupancy is evaluated only where the registered box projects, so a
        different instance elsewhere in the frame is not even consulted.
        """
        depth = np.full((480, 640), 800.0, dtype=np.float32)
        # A decoy surface at plausible marker depth, but far off in the image.
        depth[100:140, 40:80] = self.centre_cam[2]
        frame = (np.zeros((480, 640, 3), np.uint8), depth, self.intr)
        ok, detail = self.check(frame)
        self.assertFalse(ok)
        self.assertFalse(detail["inconclusive"])

    def test_parallax_alone_must_not_promote(self):
        """Regression guard for the 2026-08-21 live failure.

        Asserts the source states the parallax trigger is intentionally gone,
        so a future reader does not "restore" it from the arithmetic argument.
        """
        import inspect

        source = inspect.getsource(ws._look_around_object)
        self.assertIn("promote_translating_poses(\"bottom_edge_clip\")", source)
        self.assertNotIn("parallax_baseline_required", source)

    def test_views_are_captured_before_the_parallax_gate(self):
        """The premise that made the parallax promotion look justified.

        Capture must precede the gate; otherwise failing views really would be
        discarded and skipping those poses would be free.
        """
        import inspect

        source, _start = inspect.getsourcelines(ws._look_around_object)
        body = "".join(source)
        capture_at = body.index("status=saved") if "status=saved" in body \
            else body.index("captured_views is not None and capture_stable")
        gate_at = body.index("look_around_parallax_gate")
        self.assertLess(capture_at, gate_at)



FRAMES = Path(
    "/tmp/claude-1000/-home-manav-rebot-lerobot/"
    "1b94e05b-7601-46f2-a0fe-97b0ab3194ba/scratchpad/wrist_servo"
)


class SemanticColorVetoTest(TestCase):
    """A single named colour may veto a semantic match that lacks it.

    SAM 3 was observed returning whichever object was most salient while
    ignoring the colour word: it offered the same green marker for
    'blue marker' (15:02 run) and 'red marker' (16:55 run) with a real blue
    and a real red marker sitting in frame. Measured separation on this rig is
    wide -- a correct marker reads 30-45%, a wrong one 0-1.4%.
    """

    def detection(self, box, z_mm=340.0):
        return WD.Detection(
            box=box, uv=((box[0] + box[2]) / 2, (box[1] + box[3]) / 2),
            z_mm=z_mm, p_cam_mm=(0.0, 0.0, z_mm), source="sam3",
        )

    def scene(self, color, size=(240, 320)):
        """A single coloured object on a neutral table, at a known depth."""
        img = np.full((*size, 3), 118, np.uint8)
        img[80:160, 140:170] = color
        depth = np.full(size, 340.0, np.float32)
        return img, depth

    def fraction(self, img, depth, box, name):
        return WD.semantic_color_fraction(img, depth, self.detection(box), name)

    # --- measurement ------------------------------------------------------

    def test_matching_colour_scores_high(self):
        img, depth = self.scene((200, 60, 30))       # BGR blue
        self.assertGreater(self.fraction(img, depth, (140, 80, 170, 160), "blue"), 0.8)

    def test_wrong_colour_scores_near_zero(self):
        img, depth = self.scene((40, 40, 40))        # near-black
        self.assertLess(self.fraction(img, depth, (140, 80, 170, 160), "blue"), 0.05)

    def test_too_little_surface_is_no_opinion(self):
        """A degenerate box must not be read as a refusal."""
        img, depth = self.scene((200, 60, 30))
        self.assertIsNone(self.fraction(img, depth, (140, 80, 142, 82), "blue"))

    def test_missing_depth_is_no_opinion(self):
        img, _ = self.scene((200, 60, 30))
        self.assertIsNone(
            WD.semantic_color_fraction(
                img, None, self.detection((140, 80, 170, 160)), "blue"
            )
        )

    def test_depth_shell_excludes_the_background(self):
        """Colour behind the object must not be credited to it."""
        img, depth = self.scene((118, 118, 118))     # grey object
        img[:80, :] = (200, 60, 30)                  # blue wall behind
        depth[:80, :] = 900.0                        # far away
        self.assertLess(self.fraction(img, depth, (100, 0, 200, 160), "blue"), 0.35)

    def test_native_mask_excludes_same_depth_color_beside_target(self):
        img = np.full((120, 180, 3), (40, 210, 40), np.uint8)
        depth = np.full((120, 180), 240.0, np.float32)
        # A blue neighbour occupies most of the semantic rectangle, but the
        # model's exact target mask covers only the green object on the left.
        img[20:100, 80:160] = (220, 50, 20)
        target_mask = np.zeros((120, 180), np.uint8)
        target_mask[20:100, 20:70] = 255
        detection = self.detection((15, 15, 165, 105), z_mm=240.0)
        detection.info = {"mask_u8": target_mask}

        fraction = WD.semantic_color_fraction(
            img, depth, detection, "blue"
        )

        self.assertIsNotNone(fraction)
        self.assertLess(fraction, 0.01)

    # --- policy -----------------------------------------------------------

    def test_veto_is_on_by_default(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("REBOT_COLOR_VETO", None)
            self.assertTrue(WD.color_veto_enabled())

    def test_veto_can_be_disabled(self):
        for value in ("0", "false", "no", "off", "OFF"):
            with patch.dict(os.environ, {"REBOT_COLOR_VETO": value}):
                self.assertFalse(WD.color_veto_enabled(), value)

    def test_threshold_sits_between_the_measured_populations(self):
        self.assertGreater(WD.COLOR_VETO_FRACTION, 0.05)   # above wrong (0-1.4%)
        self.assertLess(WD.COLOR_VETO_FRACTION, 0.30)      # below right (30-45%)

    # --- the real frames --------------------------------------------------

    # The one-time validation against real captures (wrong picks 0.0-1.4%,
    # correct 30-45%) is recorded in the class docstring. The frames themselves
    # lived in the scratchpad and are overwritten by every run, so pinning box
    # coordinates into them made the test assert against whatever scene ran
    # last. The behavioural tests above cover the logic.


class PostDiscoveryOrientationBudgetTest(TestCase):
    """The return from discovery gets the same envelope the sweep was granted.

    Run 2026-08-21 17:26 reached the discovery sweep, registered 4 views and
    converged its metric box, then stranded: the sweep had raised the camera
    envelope to 30deg, wandered to 23.98deg from ready, and the restore planned
    under the default 20deg grasp clamp. Everything upstream had succeeded.
    A sweep must not be able to reach a pose it cannot leave.
    """

    def budget_deg(self, max_delta_deg):
        """The clamp _plan_rpy_trajectory applies for a given request."""
        extended = max_delta_deg is not None
        return (
            float(np.clip(float(max_delta_deg), 0.0, 45.0)) if extended
            else ws.GEOMETRY_MAX_YAW_DELTA_DEG
        )

    def test_the_failing_case_now_plans(self):
        self.assertLess(self.budget_deg(None), 23.98)      # what failed
        self.assertGreater(self.budget_deg(30.0), 23.98)   # what is granted

    def test_no_raised_envelope_keeps_the_default_clamp(self):
        """Runs that never raised the envelope must be untouched."""
        self.assertEqual(self.budget_deg(None), ws.GEOMETRY_MAX_YAW_DELTA_DEG)

    def test_the_sweep_envelope_is_always_reachable_in_return(self):
        """Any pose the sweep can reach must be one the restore can leave."""
        for envelope in (25.0, 30.0, 35.0):   # the clip range of the sweep
            self.assertGreaterEqual(self.budget_deg(envelope), envelope)

    def test_the_budget_is_still_bounded(self):
        """A raised envelope must not become an unbounded rotation."""
        self.assertEqual(self.budget_deg(90.0), 45.0)

    def test_an_extended_request_uses_the_stricter_gate(self):
        """Paying for the wider budget with clearance checking, not just IK."""
        self.assertIsNotNone(getattr(ws.A, "check_pose_clearance", None))


class DiscoveryEnvelopeSurvivesTeardownTest(TestCase):
    """The restore budget must outlive the sweep's own envelope teardown.

    First attempt at this fix read `_ACTIVE_CAMERA_ENVELOPE_DEG` at the restore
    and was a silent no-op: `restore_camera_envelope()` sets that global back to
    None when the sweep returns, which happens BEFORE the post-discovery
    restore runs (17:39:59 raise -> 17:40:04 restore). The run failed with the
    identical 23.96deg-vs-20deg message. The budget therefore has to be
    recorded separately and survive that teardown.
    """

    def setUp(self):
        self.addCleanup(
            setattr, ws, "_ACTIVE_CAMERA_ENVELOPE_DEG",
            ws._ACTIVE_CAMERA_ENVELOPE_DEG,
        )
        self.addCleanup(
            setattr, ws, "_DISCOVERY_CAMERA_ENVELOPE_DEG",
            ws._DISCOVERY_CAMERA_ENVELOPE_DEG,
        )

    def effective_budget(self):
        """What the post-discovery restore will plan against."""
        return (
            ws._ACTIVE_CAMERA_ENVELOPE_DEG
            or ws._DISCOVERY_CAMERA_ENVELOPE_DEG
        )

    def test_budget_survives_the_active_envelope_teardown(self):
        ws._DISCOVERY_CAMERA_ENVELOPE_DEG = 30.0
        ws._set_active_camera_envelope(None)          # what the sweep does
        self.assertEqual(self.effective_budget(), 30.0)

    def test_the_exact_failing_geometry_now_plans(self):
        ws._DISCOVERY_CAMERA_ENVELOPE_DEG = 30.0
        ws._set_active_camera_envelope(None)
        self.assertGreater(self.effective_budget(), 23.96)

    def test_a_run_without_discovery_keeps_the_default_clamp(self):
        ws._DISCOVERY_CAMERA_ENVELOPE_DEG = None
        ws._set_active_camera_envelope(None)
        self.assertIsNone(self.effective_budget())

    def test_the_widest_sweep_envelope_wins(self):
        """Several raises in one sweep: the return must clear the widest."""
        ws._DISCOVERY_CAMERA_ENVELOPE_DEG = None
        for envelope in (25.0, 30.0, 27.0):
            ws._DISCOVERY_CAMERA_ENVELOPE_DEG = max(
                float(ws._DISCOVERY_CAMERA_ENVELOPE_DEG or 0.0), envelope
            )
        self.assertEqual(ws._DISCOVERY_CAMERA_ENVELOPE_DEG, 30.0)

    def test_an_active_envelope_still_takes_precedence(self):
        ws._DISCOVERY_CAMERA_ENVELOPE_DEG = 25.0
        ws._set_active_camera_envelope(35.0)
        self.assertEqual(self.effective_budget(), 35.0)


class RegisteredBoxOccupancySamplingTest(RegisteredBoxOccupancyTest):
    """Occupancy is a question about the object, not about one pixel.

    Run 2026-08-22 13:44 cleared every stage -- fusion converged, jaw plan,
    gripper fit, 3D axis alignment, metric crosscheck -- then died here. The
    gate probed only the projected box centre with a 6px median. The target was
    a 16.3mm marker (~28px wide at 340mm); after a 45mm metric reanchor and a
    21.6deg wrist rotation that probe landed just off it, read the table at
    475mm against a 258-429mm span, and declared the object gone.

    Inherits the parent's real mapping/box/intrinsics so both behaviours are
    tested against identical geometry.
    """

    def test_a_thin_object_survives_a_centre_miss(self):
        """The exact regression: the centre pixel reads background."""
        colour, depth, intr = self.scene(self.centre_cam[2])
        u, v = int(self.uv[0]), int(self.uv[1])
        depth[v - 8:v + 8, u - 8:u + 8] = 800.0
        ok, why = self.check((colour, depth, intr))
        self.assertTrue(ok, why.get("reason"))
        self.assertGreater(why["occupied_samples"], 0)

    def test_removal_still_fails_closed(self):
        ok, why = self.check(self.scene(self.centre_cam[2] + 120.0))
        self.assertFalse(ok)
        self.assertFalse(why["inconclusive"])
        self.assertEqual(why["occupied_samples"], 0)

    def test_partial_occlusion_is_not_removal(self):
        """Most of the body hidden must not read as a vanished target."""
        colour, depth, intr = self.scene(800.0)
        u, v = int(self.uv[0]), int(self.uv[1])
        depth[v - 40:v + 40, u - 12:u + 12] = self.centre_cam[2]
        ok, why = self.check((colour, depth, intr))
        self.assertTrue(ok, why.get("reason"))

    def test_sparse_depth_is_inconclusive_not_a_refusal(self):
        colour, depth, intr = self.scene(self.centre_cam[2])
        depth[:] = 0.0
        ok, why = self.check((colour, depth, intr))
        self.assertFalse(ok)
        self.assertTrue(why["inconclusive"])

    def test_it_reports_how_many_samples_agreed(self):
        ok, why = self.check(self.scene(self.centre_cam[2]))
        self.assertTrue(ok, why.get("reason"))
        self.assertGreaterEqual(why["samples"], 3)
        self.assertLessEqual(why["occupied_samples"], why["samples"])


class StackRgbdReferenceTest(TestCase):
    """A depth-connected RGB-D object may serve as an exact stack destination.

    SAM 3 missed the placement reference for a white cup, a red cup and a green
    mug in turn, while the colour path found each one in every frame. Refusing
    on provenance alone blocked stacking entirely with the reference in plain
    view. The hazard the gate exists for -- a mask spanning several nearby
    objects -- is a property of the mask's extent, which is measured directly.
    """

    def detection(self, source, box=(300, 200, 340, 260),
                  min_mm=60.0, max_mm=90.0):
        return SimpleNamespace(
            source=source, box=box,
            min_width_mm=min_mm, max_width_mm=max_mm, info={},
        )

    def frame(self):
        return (np.zeros((480, 640, 3), np.uint8),
                np.full((480, 640), 340.0, np.float32),
                (600.0, 600.0, 320.0, 240.0))

    def check(self, detection):
        return ws._stack_reference_detection_quality(detection, self.frame())

    def test_a_compact_rgbd_object_is_accepted(self):
        ok, why = self.check(self.detection("rgbd:colored_object"))
        self.assertTrue(ok, why)

    def test_a_semantic_mask_is_still_accepted(self):
        ok, why = self.check(self.detection("sam3"))
        self.assertTrue(ok, why)

    def test_a_sprawling_rgbd_mask_is_still_refused(self):
        """Two touching same-coloured objects must not become one target."""
        ok, why = self.check(self.detection(
            "rgbd:colored_object", min_mm=60.0, max_mm=175.0
        ))
        self.assertFalse(ok)
        self.assertIn("beyond", why)

    def test_rgbd_is_held_to_a_tighter_extent_than_semantic(self):
        """150mm passes as a semantic instance but not as a colour component."""
        wide = dict(min_mm=60.0, max_mm=150.0)
        self.assertTrue(self.check(self.detection("sam3", **wide))[0])
        self.assertFalse(self.check(
            self.detection("rgbd:colored_object", **wide)
        )[0])

    def test_a_huge_rgbd_box_is_refused(self):
        ok, why = self.check(self.detection(
            "rgbd:colored_object", box=(100, 100, 420, 400)
        ))
        self.assertFalse(ok)
        self.assertIn("covers", why)

    def test_an_unknown_source_is_still_refused(self):
        ok, why = self.check(self.detection("guess"))
        self.assertFalse(ok)
        self.assertIn("requires", why)

    def test_the_rgbd_path_can_be_disabled(self):
        with patch.dict(os.environ,
                        {"REBOT_STACK_ALLOW_RGBD_REFERENCE": "0"}):
            ok, why = self.check(self.detection("rgbd:colored_object"))
            self.assertFalse(ok)
            self.assertIn("requires", why)


class GripperCompressionTest(TestCase):
    """The jaws must bite into the measured width, not merely touch it.

    A 23mm round marker closed to 20.4mm (the old 4mm ceiling) was not held --
    the rigid-motion check refused it correctly. A cylinder has no flat to seat
    against, so it rolls out of a grip that only kisses it.
    """

    def test_default_is_unchanged_so_the_sim_stays_valid(self):
        """wrist_sim counts a command >2mm inside the object as a miss, so a
        raised default would report every simulated grasp as unheld. Extra
        compression is opt-in per run on hardware."""
        self.assertEqual(ws.SIZE_COMPRESSION_MM, 0.0)

    def test_the_ceiling_allows_more_than_the_old_four_mm(self):
        self.assertGreater(ws.GRIPPER_MAX_COMPRESSION_MM, 4.0)

    def test_the_ceiling_is_still_bounded(self):
        self.assertLessEqual(ws.GRIPPER_MAX_COMPRESSION_MM, 12.0)

    def test_requests_are_clipped_to_the_ceiling(self):
        clipped = float(np.clip(99.0, 0.0, ws.GRIPPER_MAX_COMPRESSION_MM))
        self.assertEqual(clipped, ws.GRIPPER_MAX_COMPRESSION_MM)

    def test_negative_requests_clip_to_zero(self):
        self.assertEqual(
            float(np.clip(-5.0, 0.0, ws.GRIPPER_MAX_COMPRESSION_MM)), 0.0
        )


class AchromaticColorWordsTest(TestCase):
    """White and black must be nameable colours, not gaps in the vocabulary.

    Neither has a hue, so neither fitted the hue-band table -- and
    `colors_of("white cup")` returned []. That skipped the RGB-D colour path
    entirely, leaving such references wholly dependent on a semantic hit. A
    plainly visible white cup went undetected 0/3 frames on that basis, and the
    same gap blocked "black coffee mug".

    Bands are separated on saturation and value, tuned against measured values
    on this rig: white cup S=24/V=189, black object S=67/V=39, dark green mug
    S=81/V=55, dark mat S=152/V=76.
    """

    def swatch(self, h, s, v, size=(60, 60)):
        hsv = np.zeros((*size, 3), np.uint8)
        hsv[:] = (h, s, v)
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    def fraction(self, img, name):
        return int(np.count_nonzero(WD.color_mask(img, name))) / float(
            img.shape[0] * img.shape[1]
        )

    def test_they_are_recognised_as_colour_words(self):
        self.assertEqual(WD.colors_of("white cup"), ["white"])
        self.assertEqual(WD.colors_of("black coffee mug"), ["black"])

    def test_white_matches_a_bright_desaturated_surface(self):
        self.assertGreater(self.fraction(self.swatch(90, 24, 189), "white"), 0.9)

    def test_black_matches_a_dark_surface(self):
        """Kept clear of the band edge: the HSV->BGR->HSV roundtrip quantises
        S=67 to both 65 and 72, straddling the S<=70 bound, so a swatch built
        there measures rounding rather than the band. The real black object is
        covered by the measured-frame figures in the class docstring."""
        self.assertGreater(self.fraction(self.swatch(60, 40, 30), "black"), 0.9)

    def test_a_shadowed_coloured_object_is_not_black(self):
        """The dark green mug at S=81/V=55 must stay out of the black band."""
        self.assertLess(self.fraction(self.swatch(79, 81, 55), "black"), 0.12)

    def test_a_dark_saturated_surface_is_not_black(self):
        """The mat: dark but strongly coloured."""
        self.assertLess(self.fraction(self.swatch(96, 152, 76), "black"), 0.12)

    def test_white_and_black_are_disjoint(self):
        for h, s, v in ((90, 24, 189), (60, 67, 39), (96, 152, 76)):
            img = self.swatch(h, s, v)
            self.assertLess(
                min(self.fraction(img, "white"), self.fraction(img, "black")),
                0.12, f"H={h} S={s} V={v} matched both",
            )

    def test_a_vivid_colour_is_neither(self):
        for name in ("red", "green", "blue", "purple"):
            lo, _hi = WD.COLORS[name][0]
            img = self.swatch(lo[0] + 5, 220, 200)
            self.assertLess(self.fraction(img, "white"), 0.12, name)
            self.assertLess(self.fraction(img, "black"), 0.12, name)

    def test_existing_hue_words_are_unchanged(self):
        for name in ("red", "orange", "yellow", "green", "blue", "purple", "pink"):
            self.assertIn(name, WD.COLORS)
        self.assertEqual(WD.colors_of("blue marker"), ["blue"])
        self.assertEqual(WD.colors_of("red marker"), ["red"])


class StackReferenceColourFallbackTest(TestCase):
    """A broad semantic mask must not shut out a compact colour detection.

    ``detect`` returns as soon as the semantic path finds anything, so the
    RGB-D colour path beneath it is unreachable whenever SAM 3 answers at all
    -- even badly. Two consecutive runs of the identical command, 2026-08-23:
    at 11:49 SAM 3 missed the white cup, colour froze it cleanly (samples 4mm
    apart) and the stack succeeded; at 11:59 SAM 3 hit, returned a 223-235mm
    mask that the quality gate refused, and the run died 0/3 with a perfectly
    good colour detection never attempted.
    """

    def detection(self, source, min_mm, max_mm, box=(300, 200, 340, 260)):
        return SimpleNamespace(
            source=source, box=box,
            min_width_mm=min_mm, max_width_mm=max_mm, info={},
        )

    def frame(self):
        return (np.zeros((480, 640, 3), np.uint8),
                np.full((480, 640), 340.0, np.float32),
                (600.0, 600.0, 320.0, 240.0))

    def test_a_compact_colour_detection_replaces_a_broad_mask(self):
        broad = self.detection("sam3", 60.0, 230.0)
        compact = self.detection("rgbd:colored_object", 60.0, 90.0)
        with patch.object(ws.WD, "detect", return_value=(compact, {})):
            found, reason = ws._colour_only_reference(
                broad, self.frame(), "white cup"
            )
        self.assertIs(found, compact)
        self.assertIn("compact", reason)

    def test_it_disables_grounding_for_the_retry(self):
        """The retry is only useful if it does not ask the same model again."""
        compact = self.detection("rgbd:colored_object", 60.0, 90.0)
        with patch.object(ws.WD, "detect", return_value=(compact, {})) as det:
            ws._colour_only_reference(
                self.detection("sam3", 60.0, 230.0), self.frame(), "white cup"
            )
        self.assertFalse(det.call_args.kwargs["allow_vlm"])

    def test_an_equally_broad_colour_blob_is_refused(self):
        """The fallback must not smuggle past the gate it exists to serve."""
        sprawl = self.detection("rgbd:colored_object", 60.0, 300.0)
        with patch.object(ws.WD, "detect", return_value=(sprawl, {})):
            found, reason = ws._colour_only_reference(
                self.detection("sam3", 60.0, 230.0), self.frame(), "white cup"
            )
        self.assertIsNone(found)
        self.assertIn("also unusable", reason)

    def test_no_colour_detection_keeps_the_original_refusal(self):
        with patch.object(ws.WD, "detect", return_value=(None, {})):
            found, reason = ws._colour_only_reference(
                self.detection("sam3", 60.0, 230.0), self.frame(), "white cup"
            )
        self.assertIsNone(found)
        self.assertIn("found no reference", reason)

    def test_a_detector_error_never_aborts_the_run(self):
        with patch.object(ws.WD, "detect", side_effect=RuntimeError("cuda")):
            found, reason = ws._colour_only_reference(
                self.detection("sam3", 60.0, 230.0), self.frame(), "white cup"
            )
        self.assertIsNone(found)
        self.assertIn("RuntimeError", reason)

    def test_a_malformed_frame_is_handled(self):
        found, reason = ws._colour_only_reference(
            self.detection("sam3", 60.0, 230.0), None, "white cup"
        )
        self.assertIsNone(found)
        self.assertIn("malformed", reason)
