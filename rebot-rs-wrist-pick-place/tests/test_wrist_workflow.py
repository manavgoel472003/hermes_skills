from __future__ import annotations

import os
from unittest import TestCase
from unittest.mock import patch

import numpy as np
from types import SimpleNamespace

from arm_agent import wrist_servo as ws
from arm_agent import wrist_sim


class WristWorkflowTest(TestCase):
    def test_aperture_closes_to_detected_width_without_extra_margin(self):
        det = SimpleNamespace(min_width_mm=18.5, max_width_mm=28.0)

        with (
            patch.dict(os.environ, {"REBOT_GRIPPER_MM_PER_RAD": "19.099"}),
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
            patch.dict(os.environ, {"REBOT_GRIPPER_MM_PER_RAD": "19.099"}),
            patch.object(ws, "SIZE_SAFETY_MARGIN_MM", 0.0),
            patch.object(ws, "SIZE_COMPRESSION_MM", 1.0),
        ):
            plan = ws._plan_aperture(det)

        self.assertTrue(plan["ok"], plan["reason"])
        self.assertAlmostEqual(plan["grasp_mm"], 17.5, delta=ws.SIZE_TARGET_TOL_MM)
        self.assertEqual(plan["planned_compression_mm"], 1.0)
        self.assertIn("1.0mm planned position compression", plan["reason"])

    def test_rigid_camera_jacobian_removes_finite_difference_shear(self):
        measured = np.array([
            [150.0, 1000.0, -150.0],
            [-200.0, 50.0, 950.0],
            [-1100.0, 50.0, 300.0],
        ])

        rigid, detail = ws._rigid_camera_jacobian(measured)

        self.assertIsNotNone(rigid, detail)
        metric = rigid.T @ rigid
        self.assertTrue(np.allclose(metric, np.eye(3) * 1_000_000.0,
                                    atol=1e-6))
        self.assertLess(detail["rigid_fit_relative_error"], 0.35)

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
                "REBOT_GRIPPER_MM_PER_RAD": "19.099",
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
