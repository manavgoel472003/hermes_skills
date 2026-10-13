"""Hardware-free mission invariants and the real wrist integration seams."""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import unittest
from unittest.mock import patch

import numpy as np

from arm_agent import mission_control as mc
from arm_agent.hybrid_mission import MissionRunner, Routine, Task, load_routines, calibration_report
from arm_agent.hybrid_profile import DEFAULTS


class FakeBackend:
    def __init__(self, result=None):
        self.result = result or {"ok": True, "verified": True, "homed": True, "object_held": False}
        self.calls = []

    def execute(self, task, routine, dry):
        self.calls.append((task, routine, dry))
        for stage in ("observe", "plan_grasp", "approach", "checkpoint", "grasp",
                      "verify_pick", "held", "plan_place", "carry", "release", "return_home"):
            mc.stage(stage)
        return copy.deepcopy(self.result)

    def freeze(self):
        self.calls.append("freeze")
        return {"ok": True}


class MissionTests(unittest.TestCase):
    def setUp(self):
        self.routines = load_routines()

    def test_hybrid_profile_uses_exact_camera_aperture_without_squeeze(self):
        self.assertEqual(DEFAULTS["REBOT_REFLEX_COLOR_GRASP_FALLBACK"], "1")
        self.assertEqual(DEFAULTS["REBOT_CLAW_BOX_REFLEX_FALLBACK"], "fallback")
        self.assertEqual(DEFAULTS["REBOT_WRIST_CLOSE_GRIPPER_AT_START"], "0")
        self.assertEqual(DEFAULTS["REBOT_GRIPPER_CLOSE_SETTLE_RETRY"], "0")
        for key in ("REBOT_GRIPPER_SIZE_COMPRESSION_MM",
                    "REBOT_GRIPPER_SIZE_MARGIN_MM",
                    "REBOT_GRIPPER_CLAMP_EXTRA_MM",
                    "REBOT_DYNAMIC_RETRY_CLOSE_MM",
                    "REBOT_DYNAMIC_RETRY_CLOSE_DEG",
                    "REBOT_DYNAMIC_MAX_RETRIES"):
            self.assertEqual(DEFAULTS[key], "0")

    def test_validation_precedes_backend_access(self):
        backend = FakeBackend()
        runner = MissionRunner(self.routines, backend)
        for task in (Task(""), Task("marker", "scene_place", "left"),
                     Task("marker", relation="left"), Task("marker", reference="cup"),
                     Task("marker", "scene_place", "left", "cup", float("nan"))):
            with self.assertRaises(ValueError):
                runner.start(task, dry=False)
        self.assertEqual(backend.calls, [])

    def test_success_requires_verified_empty_gripper_and_home(self):
        for field, value in (("verified", False), ("homed", False), ("object_held", None)):
            backend = FakeBackend()
            backend.result[field] = value
            outcome = MissionRunner(self.routines, backend).start(Task("marker"), dry=False)
            self.assertEqual(outcome["status"], "failed")
            self.assertFalse(outcome["placement_verified"])
            self.assertIn("freeze", backend.calls)

    def test_successful_live_recipe_retains_distinct_verification(self):
        runner = MissionRunner(self.routines, FakeBackend())
        outcome = runner.start(Task("marker"), dry=False)
        self.assertEqual(outcome["status"], "succeeded")
        self.assertTrue(outcome["release_verified"])
        self.assertFalse(outcome["placement_verified"])
        self.assertEqual([e["stage"] for e in outcome["events"]][1:4],
                         ["observe", "plan_grasp", "approach"])
        outcome["events"].clear()
        self.assertTrue(runner.status()["events"])

    def test_dry_run_is_not_a_physical_success(self):
        runner = MissionRunner(self.routines, FakeBackend())
        outcome = runner.start(Task("marker"))
        self.assertEqual(outcome["status"], "planned")
        self.assertFalse(outcome["release_verified"])
        self.assertFalse(outcome["placement_verified"])

    def test_stop_at_checkpoint_never_reaches_grasp_or_place(self):
        backend = FakeBackend()
        runner = MissionRunner(self.routines, backend)
        emit = runner._event
        def stop_at_checkpoint(name, detail):
            emit(name, detail)
            if name == "checkpoint":
                runner.cancel(runner.status()["mission_id"])
        runner._event = stop_at_checkpoint
        outcome = runner.start(Task("marker"), dry=False)
        stages = [e["stage"] for e in outcome["events"]]
        self.assertEqual(outcome["status"], "cancelled")
        self.assertNotIn("grasp", stages)
        self.assertNotIn("release", stages)
        self.assertTrue(outcome["needs_operator"])
        self.assertIn("freeze", backend.calls)
        self.assertFalse(runner.start(Task("another marker"), dry=False)["ok"])

    def test_concurrent_task_and_wrong_stop_id_are_refused(self):
        entered = threading.Event()
        finished = threading.Event()
        class Blocking(FakeBackend):
            def execute(self, *args):
                entered.set()
                try:
                    mc.pause(5)
                finally:
                    finished.set()
        runner = MissionRunner(self.routines, Blocking())
        receipt = runner.start(Task("marker"), background=True)
        self.assertTrue(entered.wait(1))
        self.assertFalse(runner.start(Task("other"))["ok"])
        self.assertFalse(runner.cancel("wrong")["ok"])
        self.assertTrue(runner.cancel(receipt["mission_id"])["ok"])
        self.assertTrue(finished.wait(1))

    def test_failed_attempt_not_retried_by_outer_mission(self):
        backend = FakeBackend({"ok": False, "homed": True, "object_held": False,
                               "reason": "grasp failed", "grasp_retryable": True})
        outcome = MissionRunner(self.routines, backend).start(Task("marker"), dry=False)
        self.assertEqual(len([v for v in backend.calls if isinstance(v, tuple)]), 1)
        self.assertFalse(outcome["needs_operator"])

    def test_taught_route_requires_calibration_binding(self):
        data = {"version": 1, "routines": {"test": {"relations": ["drop_zone"],
                                                     "carry_via_m": [[.3, 0, .2]]}}}
        with patch.object(Path, "read_text", return_value=json.dumps(data)):
            with self.assertRaisesRegex(ValueError, "calibration_id"):
                load_routines()

    def test_artifact_mismatch_refuses_taught_route(self):
        routine = Routine("test", "", ("drop_zone",), ((.3, 0, .2),), "obsolete")
        report = calibration_report(Task("marker"), routine)
        self.assertFalse(report["ok"])
        self.assertTrue(any("changed" in e for e in report["errors"]))

    def test_offline_plan_does_not_import_hardware(self):
        code = """
import sys
from arm_agent.hybrid_mission import Task, load_routines, calibration_report
report = calibration_report(Task('marker'), load_routines()['taught_drop'])
# A fresh clone intentionally has no machine-specific calibration.  The
# offline planner must report that prerequisite without importing a camera,
# CAN driver, or local arm backend.
assert not report['ok'], report
assert any('wrist_tcp.json' in error for error in report['errors']), report
assert 'arm_agent.local_arm' not in sys.modules
assert 'arm_agent.wrist_servo' not in sys.modules
assert 'pyrealsense2' not in sys.modules
"""
        result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_extra_overrides_are_bound_but_not_exposed(self):
        task = Task("marker")
        routine = self.routines["taught_drop"]
        original = calibration_report(task, routine)
        with patch.dict(os.environ, {"REBOT_TEST_PRIVATE_OVERRIDE": "private-value"}):
            changed = calibration_report(task, routine)
        self.assertNotEqual(original["calibration_id"], changed["calibration_id"])
        self.assertNotIn("private-value", json.dumps(changed))

    def test_malformed_calibration_is_reported(self):
        with patch.object(Path, "read_bytes", return_value=b"[]"):
            report = calibration_report(Task("marker"), self.routines["taught_drop"])
        self.assertFalse(report["ok"])
        self.assertTrue(any("Expected a calibration object" in e for e in report["errors"]))


class WristIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # The repository deliberately excludes hardware calibration.  The
        # simulated integration seam needs only a generic aperture curve; keep
        # it isolated from the robot's real calib/ directory.
        fixture = Path(__file__).with_name("fixtures") / "calib"
        cls._old_calib = os.environ.get("REBOT_WRIST_CALIB")
        os.environ["REBOT_WRIST_CALIB"] = str(fixture)
        from arm_agent import wrist_servo
        from arm_agent.local_arm import LocalArm
        cls.ws, cls.LocalArm = wrist_servo, LocalArm

    @classmethod
    def tearDownClass(cls):
        if cls._old_calib is None:
            os.environ.pop("REBOT_WRIST_CALIB", None)
        else:
            os.environ["REBOT_WRIST_CALIB"] = cls._old_calib

    def test_dry_guard_blocks_every_motor_entry_before_access(self):
        arm = object.__new__(self.LocalArm)
        with mc.activate(mc.MissionControl(dry_run=True)):
            calls = (arm.connect, arm.home, lambda: arm.gripper("open"),
                     lambda: arm.move_pose(0, 0, 0, 0, 0, 0, 1),
                     lambda: arm.move_pose_path([], 1), lambda: arm.execute_joint_plan({}))
            for call in calls:
                with self.assertRaisesRegex(RuntimeError, "dry run"):
                    call()

    def test_cancelled_recovery_freezes_without_release_or_home(self):
        active = mc.MissionControl()
        active.cancel()
        with mc.activate(active), \
             patch.object(self.ws, "_freeze_arm", return_value={"ok": True}) as freeze, \
             patch.object(self.ws, "_gripper", side_effect=AssertionError("must not release")), \
             patch.object(self.ws, "_home", side_effect=AssertionError("must not home")):
            result = self.ws._home_after_interrupt()
        freeze.assert_called_once()
        self.assertTrue(result["frozen"])
        self.assertFalse(result["gripper_opened"])
        self.assertFalse(result["homed"])

    def test_preset_stitches_live_start_to_geometry_endpoint_and_validates(self):
        active = mc.MissionControl(carry_via_m=((.3, .1, .2),))
        endpoint = np.array([.4, .1, .1])
        def leg(target, **kwargs):
            return [tuple(target)], "ok"
        with mc.activate(active), \
             patch.object(self.ws.A, "ee_pos", return_value=[.25, 0, .2]), \
             patch.object(self.ws, "_plan_fixed_approach_trajectory", side_effect=leg) as plan, \
             patch.object(self.ws, "_validate_fixed_waypoints", return_value=(True, "validated")) as validate:
            path, reason = self.ws._mission_carry_waypoints(endpoint, (0, 0, 0))
        self.assertEqual(path, [(.3, .1, .2), (.4, .1, .1)])
        np.testing.assert_allclose(plan.call_args_list[0].kwargs["start_m"], [.25, 0, .2])
        validate.assert_called_once_with(path, (0, 0, 0))

    def test_preset_collision_returns_no_executable_path(self):
        with mc.activate(mc.MissionControl(carry_via_m=((.3, .1, .2),))), \
             patch.object(self.ws.A, "ee_pos", return_value=[.25, 0, .2]), \
             patch.object(self.ws, "_plan_fixed_approach_trajectory", return_value=([(.3, .1, .2)], "ok")), \
             patch.object(self.ws, "_validate_fixed_waypoints", return_value=(False, "payload collision")):
            path, reason = self.ws._mission_carry_waypoints(np.array([.4, 0, .2]), (0, 0, 0))
        self.assertIsNone(path)
        self.assertIn("payload collision", reason)

    def test_sender_cancellation_does_not_send_remaining_targets(self):
        from types import SimpleNamespace
        arm = object.__new__(self.LocalArm)
        arm._lock = threading.RLock()
        controller = SimpleNamespace(_q_target=np.zeros(6))
        raw = SimpleNamespace(get_state=lambda: (np.zeros(7), None, None))
        active = mc.MissionControl()
        def stop_after_first(_seconds):
            active.cancel()
            mc.checkpoint()
        with mc.activate(active), \
             patch.object(arm, "_require", return_value=(raw, controller)), \
             patch.object(arm, "solve_ik", return_value={"joints_rad": [.2] * 6}), \
             patch.object(arm, "_assert_collision_free_joint_path"), \
             patch.object(mc, "pause", side_effect=stop_after_first):
            with self.assertRaises(mc.MissionCancelled):
                arm.move_pose(.3, 0, .2, 0, 0, 0, 1)
        self.assertTrue(np.all(controller._q_target < .2))

    def test_mission_context_real_simulated_pick_and_place(self):
        from arm_agent import wrist_sim
        ws = self.ws
        world = wrist_sim.SimWorld(object_bgr=(200, 80, 20), object_radius_m=.008, object_height_m=.045)
        arm = wrist_sim.SimArm(world, kinematic=True, grasp_tol_m=.030)
        original = (ws.A, ws._CAMERA_FACTORY, ws.SETTLE_S)
        events = []
        tcp = ','.join(str(float(v)) for v in world.to_camera(world.ee))
        try:
            with patch.dict(os.environ, {"REBOT_WRIST_TCP": tcp, "REBOT_WRIST_DROP": ".275,-.187,.059"}), \
                 patch.object(ws, "SCENE_COLLISION_ENABLED", False), \
                 mc.activate(mc.MissionControl(notify=lambda name, _: events.append(name))):
                ws.install_backend(arm=arm, camera_factory=lambda _: wrist_sim.SimCamera(world), settle_s=0)
                result = ws.fixed_pick_and_place("blue marker")
        finally:
            ws._OBJECT_HELD = False
            ws._FIXED_HELD_CONTEXT = None
            ws.install_backend(arm=original[0], camera_factory=original[1], settle_s=original[2])
        self.assertTrue(result["ok"], result)
        self.assertTrue(result["pick"]["verified"])
        self.assertFalse(world.held)
        self.assertLess(events.index("held"), events.index("carry"))
        self.assertLess(events.index("carry"), events.index("release"))
        self.assertIn("return_home", events)


if __name__ == "__main__":
    unittest.main()
