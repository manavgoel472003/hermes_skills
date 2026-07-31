from __future__ import annotations

import os
import time
from unittest import TestCase
from unittest.mock import patch

import numpy as np
from types import SimpleNamespace

from arm_agent import wrist_servo as ws
from arm_agent import wrist_sim


class WristWorkflowTest(TestCase):
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
                np.zeros((10, 10, 3), dtype=np.uint8),
                np.ones((10, 10), dtype=np.float32) * 250.0,
                (100.0, 100.0, 5.0, 5.0),
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
                         return_value={"ok": True, "reason": "fits"}),
        ):
            result = ws.fixed_pick_and_place(
                "object", dry=True, eye=eye
            )

        self.assertTrue(result["ok"], result["reason"])
        self.assertEqual(
            plan_geometry.call_args.kwargs["current_rpy"],
            (0.0, 0.0, 0.0),
        )

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

    def test_aperture_caps_position_compression_at_three_mm(self):
        det = SimpleNamespace(min_width_mm=18.5, max_width_mm=28.0)

        with (
            patch.object(ws, "SIZE_SAFETY_MARGIN_MM", 0.0),
            patch.object(ws, "SIZE_COMPRESSION_MM", 20.0),
        ):
            plan = ws._plan_aperture(det)

        self.assertTrue(plan["ok"], plan["reason"])
        self.assertAlmostEqual(plan["grasp_mm"], 15.5,
                               delta=ws.SIZE_TARGET_TOL_MM)
        self.assertEqual(plan["planned_compression_mm"], 3.0)
        self.assertIn("3.0mm planned position compression", plan["reason"])

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
