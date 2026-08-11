import unittest
import threading
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from arm_agent.local_arm import LocalArm


class _Motor:
    def __init__(self, state):
        self.state = state

    def get_state(self):
        return self.state


class _Arm:
    def __init__(self, states):
        self.joint_names = list(states)
        self._motor_map = {
            name: _Motor(state) for name, state in states.items()
        }

    def get_state(self):
        count = len(self.joint_names)
        return np.zeros(count), np.zeros(count), np.zeros(count)


class _WakeGroup:
    def __init__(self, ok):
        self.ok = ok
        self.calls = 0

    def enable(self):
        self.calls += 1
        if isinstance(self.ok, list):
            return self.ok[min(self.calls - 1, len(self.ok) - 1)]
        return self.ok


class FeedbackGateTest(unittest.TestCase):
    def test_stationary_hold_accepts_stable_encoders_with_noisy_velocity(self):
        local = LocalArm.__new__(LocalArm)
        state = {
            "joints_rad": [0.1] * 6,
            "joint_vel_rad_s": [0.18] * 6,
            "end_effector": {
                "position_m": [0.4, 0.1, 0.15],
                "rpy_rad": [0.0, 0.2, 0.0],
            },
        }
        local.state = lambda: dict(state)

        with patch("arm_agent.local_arm.time.sleep", return_value=None):
            result = local.wait_for_stationary_hold(
                [0.1] * 6, timeout_s=0.2,
                consecutive_samples=3,
            )

        self.assertTrue(result["settlement"]["ok"])
        self.assertEqual(
            result["settlement"]["method"], "stable_joint_encoders"
        )
        self.assertGreater(
            result["settlement"]["max_joint_velocity_rad_s"], 0.10
        )

    def test_stationary_hold_rejects_encoder_motion_with_high_velocity(self):
        local = LocalArm.__new__(LocalArm)
        sample = {"index": 0}

        def moving_state():
            sample["index"] += 1
            q = 0.01 * sample["index"]
            return {
                "joints_rad": [q] * 6,
                "joint_vel_rad_s": [0.18] * 6,
                "end_effector": {
                    "position_m": [0.4, 0.1, 0.15],
                    "rpy_rad": [0.0, 0.2, 0.0],
                },
            }

        local.state = moving_state
        result = local.wait_for_stationary_hold(
            None, timeout_s=0.03, sample_interval_s=0.01,
            consecutive_samples=3,
        )

        self.assertFalse(result["settlement"]["ok"])
        self.assertEqual(result["settlement"]["method"], "timeout")

    def test_payload_escape_reservation_keeps_nominal_support_clearance(self):
        local = LocalArm.__new__(LocalArm)
        local._lock = threading.RLock()
        local._collision_payload_box = {"size_m": np.ones(3)}
        local._collision_payload_radius_m = 0.0
        local._payload_escape_obstacles = set()
        local._payload_escape_reference = None
        local._collision_scene_version = 10
        local._collision_boxes = [{
            "center_m": np.array([0.30, 0.0, 0.13]),
            "size_m": np.array([0.04, 0.04, 0.10]),
        }]

        def collision(_q, *, ignored_payload_obstacles=None):
            ignored = ignored_payload_obstacles or set()
            if "scene_box_0" not in ignored:
                return {
                    "collision": True,
                    "kind": "payload",
                    "obstacle": "scene_box_0",
                    "payload_center_m": [0.30, 0.0, 0.15],
                    "payload_aabb_size_m": [0.10, 0.10, 0.10],
                    "payload_rotation_matrix": np.eye(3).tolist(),
                    "payload_z_bounds_m": [0.10, 0.20],
                }
            return {"collision": False, "minimum_clearance_m": 0.01}

        with (
            patch.object(local, "state", return_value={"joints_rad": [0.0] * 6}),
            patch.object(local, "_configuration_collision", side_effect=collision),
            patch.object(
                local, "_payload_scene_proximity",
                return_value={"scene_box_0": 0.0},
            ),
        ):
            detail = local.begin_payload_vertical_escape()

        self.assertTrue(detail["ok"], detail)
        self.assertGreater(detail["reservation_distance_m"], 0.0)
        self.assertEqual(detail["exact_initial_obstacles"], ["scene_box_0"])
        self.assertEqual(detail["reserved_near_obstacles"], [])
        # The uncertainty tube may reserve names, but it must not lower the
        # measured payload bottom or turn that tube into extra lift distance.
        self.assertEqual(
            detail["cluster"]["payload_bounds_m"],
            [[0.25, -0.05, 0.10], [0.35, 0.05, 0.20]],
        )
        self.assertAlmostEqual(detail["minimum_vertical_clearance_m"], 0.082)

    def test_payload_escape_allows_only_initial_payload_scene_boxes(self):
        local = LocalArm.__new__(LocalArm)
        local._lock = threading.RLock()
        local._collision_payload_box = {"size_m": np.ones(3)}
        local._collision_payload_radius_m = 0.0
        local._payload_escape_obstacles = set()
        local._collision_scene_version = 10
        local._collision_boxes = [
            {"center_m": np.zeros(3), "size_m": np.ones(3) * 0.01}
            for _ in range(7)
        ]
        local._collision_boxes[6] = {
            "center_m": np.array([0.30, 0.0, 0.10]),
            "size_m": np.array([0.04, 0.04, 0.04]),
        }

        def collision(_q, *, ignored_payload_obstacles=None):
            ignored = ignored_payload_obstacles or set()
            if "scene_box_6" not in ignored:
                return {"collision": True, "kind": "payload",
                        "obstacle": "scene_box_6",
                        "payload_center_m": [0.30, 0.0, 0.11],
                        "payload_aabb_size_m": [0.06, 0.06, 0.06],
                        "payload_rotation_matrix": np.eye(3).tolist()}
            return {"collision": False, "minimum_clearance_m": 0.01}

        with (
            patch.object(local, "state", return_value={"joints_rad": [0.0] * 6}),
            patch.object(local, "_configuration_collision",
                         side_effect=collision),
        ):
            detail = local.begin_payload_vertical_escape()

        self.assertTrue(detail["ok"], detail)
        self.assertTrue(detail["active"])
        self.assertEqual(detail["allowed_initial_obstacles"], ["scene_box_6"])
        self.assertEqual(local._payload_escape_obstacles, {"scene_box_6"})

    def test_payload_escape_derives_clearance_from_payload_and_box_tops(self):
        local = LocalArm.__new__(LocalArm)
        local._lock = threading.RLock()
        local._collision_payload_box = {"size_m": np.ones(3)}
        local._collision_payload_radius_m = 0.0
        local._payload_escape_obstacles = set()
        local._collision_scene_version = 10
        local._collision_boxes = [
            {"center_m": np.zeros(3), "size_m": np.ones(3) * 0.01}
            for _ in range(8)
        ]
        local._collision_boxes[2] = {
            "center_m": np.array([0.30, 0.0, 0.10]),
            "size_m": np.array([0.04, 0.04, 0.10]),
        }
        local._collision_boxes[7] = {
            "center_m": np.array([0.30, 0.0, 0.13]),
            "size_m": np.array([0.04, 0.04, 0.10]),
        }

        def collision(_q, *, ignored_payload_obstacles=None):
            ignored = ignored_payload_obstacles or set()
            for name, top in (("scene_box_2", 0.15), ("scene_box_7", 0.18)):
                if name not in ignored:
                    return {
                        "collision": True,
                        "kind": "payload",
                        "obstacle": name,
                        "obstacle_center_m": [0.3, 0.0, top - 0.05],
                        "obstacle_size_m": [0.04, 0.04, 0.10],
                        "payload_center_m": [0.30, 0.0, 0.15],
                        "payload_aabb_size_m": [0.10, 0.10, 0.10],
                        "payload_rotation_matrix": np.eye(3).tolist(),
                        "payload_z_bounds_m": [0.10, 0.20],
                    }
            return {"collision": False, "minimum_clearance_m": 0.01}

        with (
            patch.object(local, "state", return_value={"joints_rad": [0.0] * 6}),
            patch.object(local, "_configuration_collision",
                         side_effect=collision),
        ):
            detail = local.begin_payload_vertical_escape()

        self.assertTrue(detail["ok"], detail)
        self.assertEqual(
            detail["allowed_initial_obstacles"],
            ["scene_box_2", "scene_box_7"],
        )
        self.assertAlmostEqual(detail["minimum_vertical_clearance_m"], 0.082)

    def test_payload_escape_accepts_connected_cluster_larger_than_four_boxes(self):
        local = LocalArm.__new__(LocalArm)
        local._lock = threading.RLock()
        local._collision_payload_box = {"size_m": np.ones(3)}
        local._collision_payload_radius_m = 0.0
        local._payload_escape_obstacles = set()
        local._collision_scene_version = 3
        centers = np.linspace(-0.05, 0.05, 6)
        local._collision_boxes = [{
            "center_m": np.array([x, 0.0, 0.10]),
            "size_m": np.array([0.025, 0.04, 0.04]),
        } for x in centers]

        def collision(_q, *, ignored_payload_obstacles=None):
            ignored = ignored_payload_obstacles or set()
            for index in range(len(local._collision_boxes)):
                name = f"scene_box_{index}"
                if name not in ignored:
                    return {
                        "collision": True,
                        "kind": "payload",
                        "obstacle": name,
                        "payload_center_m": [0.0, 0.0, 0.10],
                        "payload_aabb_size_m": [0.14, 0.06, 0.06],
                        "payload_rotation_matrix": np.eye(3).tolist(),
                    }
            return {"collision": False, "minimum_clearance_m": 0.01}

        with (
            patch.object(local, "state", return_value={"joints_rad": [0.0] * 6}),
            patch.object(local, "_configuration_collision",
                         side_effect=collision),
        ):
            detail = local.begin_payload_vertical_escape()

        self.assertTrue(detail["ok"], detail)
        self.assertTrue(detail["active"])
        self.assertEqual(len(detail["allowed_initial_obstacles"]), 6)
        self.assertEqual(len(detail["cluster"]["discovered_boxes"]), 6)
        self.assertIn("vertically separable", detail["cluster"]["reason"])
        self.assertAlmostEqual(detail["minimum_vertical_clearance_m"], 0.052)

    def test_payload_escape_accepts_target_bridged_disconnected_bands(self):
        local = LocalArm.__new__(LocalArm)
        local._lock = threading.RLock()
        local._collision_payload_box = {"size_m": np.ones(3)}
        local._collision_payload_radius_m = 0.0
        local._payload_escape_obstacles = set()
        local._collision_scene_version = 3
        local._collision_boxes = [{
            "center_m": np.array([x, 0.0, 0.10]),
            "size_m": np.array([0.02, 0.04, 0.04]),
        } for x in (-0.07, 0.07)]

        def collision(_q, *, ignored_payload_obstacles=None):
            ignored = ignored_payload_obstacles or set()
            for index in range(2):
                name = f"scene_box_{index}"
                if name not in ignored:
                    return {
                        "collision": True,
                        "kind": "payload",
                        "obstacle": name,
                        "payload_center_m": [0.0, 0.0, 0.10],
                        "payload_aabb_size_m": [0.20, 0.06, 0.06],
                        "payload_rotation_matrix": np.eye(3).tolist(),
                    }
            return {"collision": False, "minimum_clearance_m": 0.01}

        with (
            patch.object(local, "state", return_value={"joints_rad": [0.0] * 6}),
            patch.object(local, "_configuration_collision",
                         side_effect=collision),
        ):
            detail = local.begin_payload_vertical_escape()

        self.assertTrue(detail["ok"], detail)
        self.assertTrue(detail["active"])
        self.assertEqual(
            local._payload_escape_obstacles,
            {"scene_box_0", "scene_box_1"},
        )
        self.assertEqual(detail["cluster"]["map_component_count"], 2)

    def test_payload_escape_accepts_inflated_perimeter_bands_from_target_removal(self):
        local = LocalArm.__new__(LocalArm)
        local._collision_inflation_m = 0.0075
        # Two table-rooted perimeter bands remain after the selected target's
        # interior columns are removed. Some boxes have long consolidated tails,
        # but every exact contact is cleared by the same fixed-XY +Z sweep.
        geometry = [
            ([0.37, 0.07, 0.03634], [0.035, 0.035, 0.03977]),
            ([0.39, 0.05, 0.03396], [0.035, 0.075, 0.03500]),
            ([0.39, 0.13, 0.05781], [0.035, 0.035, 0.08269]),
            ([0.41, 0.04, 0.03516], [0.035, 0.095, 0.03740]),
            ([0.41, 0.13, 0.06406], [0.035, 0.035, 0.09521]),
            ([0.44, 0.07, 0.04192], [0.055, 0.035, 0.05091]),
            ([0.43, 0.13, 0.05851], [0.035, 0.035, 0.08410]),
            ([0.45, 0.13, 0.05589], [0.035, 0.035, 0.07885]),
            ([0.47, 0.07, 0.04397], [0.035, 0.035, 0.05502]),
        ]
        local._collision_boxes = [
            {"center_m": np.asarray(center), "size_m": np.asarray(size)}
            for center, size in geometry
        ]

        detail = local._payload_escape_cluster_geometry(
            {f"scene_box_{index}" for index in range(len(geometry))},
            [0.41525, 0.10266, 0.07851],
            [0.11024, 0.04596, 0.09862],
        )

        self.assertTrue(detail["ok"], detail)
        self.assertEqual(detail["map_component_count"], 2)
        self.assertAlmostEqual(detail["minimum_vertical_clearance_m"], 0.0845)
        self.assertEqual(detail["scene_geometry_halo_m"], 0.0075)

    def test_payload_escape_rejects_box_outside_payload_geometry(self):
        local = LocalArm.__new__(LocalArm)
        local._lock = threading.RLock()
        local._collision_payload_box = {"size_m": np.ones(3)}
        local._collision_payload_radius_m = 0.0
        local._payload_escape_obstacles = set()
        local._collision_scene_version = 3
        local._collision_boxes = [{
            "center_m": np.array([0.40, 0.0, 0.10]),
            "size_m": np.array([0.02, 0.04, 0.04]),
        }]

        with (
            patch.object(local, "state", return_value={"joints_rad": [0.0] * 6}),
            patch.object(local, "_configuration_collision", side_effect=[{
                "collision": True,
                "kind": "payload",
                "obstacle": "scene_box_0",
                "payload_center_m": [0.0, 0.0, 0.10],
                "payload_aabb_size_m": [0.10, 0.06, 0.06],
                "payload_rotation_matrix": np.eye(3).tolist(),
            }, {"collision": False, "minimum_clearance_m": 0.01}]),
        ):
            detail = local.begin_payload_vertical_escape()

        self.assertFalse(detail["ok"], detail)
        self.assertIn("outside the attached payload geometry", detail["reason"])
        self.assertEqual(local._payload_escape_obstacles, set())
        self.assertEqual(detail["rejected_obstacle"], "scene_box_0")

    def test_payload_escape_rejects_huge_box_that_defines_its_own_extent(self):
        local = LocalArm.__new__(LocalArm)
        local._lock = threading.RLock()
        local._collision_payload_box = {"size_m": np.ones(3)}
        local._collision_payload_radius_m = 0.0
        local._payload_escape_obstacles = set()
        local._collision_scene_version = 3
        local._collision_boxes = [
            {
                "center_m": np.array([0.30, 0.20, 0.05]),
                "size_m": np.array([0.02, 0.02, 0.04]),
            },
            {
                "center_m": np.array([0.0, 0.0, 0.10]),
                "size_m": np.array([0.30, 0.30, 0.08]),
            },
        ]

        with (
            patch.object(local, "state", return_value={"joints_rad": [0.0] * 6}),
            patch.object(local, "_configuration_collision", side_effect=[{
                "collision": True,
                "kind": "payload",
                "obstacle": "scene_box_1",
                "payload_center_m": [0.0, 0.0, 0.10],
                "payload_aabb_size_m": [0.10, 0.08, 0.08],
                "payload_rotation_matrix": np.eye(3).tolist(),
            }, {"collision": False, "minimum_clearance_m": 0.01}]),
        ):
            detail = local.begin_payload_vertical_escape()

        self.assertFalse(detail["ok"], detail)
        self.assertIn("unbounded lateral extent", detail["reason"])
        self.assertEqual(detail["base_scene_cell_xy_m"], [0.02, 0.02])
        self.assertEqual(local._payload_escape_obstacles, set())

    def test_payload_escape_rejects_support_top_far_above_payload(self):
        local = LocalArm.__new__(LocalArm)
        local._lock = threading.RLock()
        local._collision_payload_box = {"size_m": np.ones(3)}
        local._collision_payload_radius_m = 0.0
        local._payload_escape_obstacles = set()
        local._collision_scene_version = 3
        local._collision_boxes = [{
            "center_m": np.array([0.0, 0.0, 0.20]),
            "size_m": np.array([0.02, 0.02, 0.30]),
        }]

        with (
            patch.object(local, "state", return_value={"joints_rad": [0.0] * 6}),
            patch.object(local, "_configuration_collision", side_effect=[{
                "collision": True,
                "kind": "payload",
                "obstacle": "scene_box_0",
                "payload_center_m": [0.0, 0.0, 0.10],
                "payload_aabb_size_m": [0.10, 0.08, 0.08],
                "payload_rotation_matrix": np.eye(3).tolist(),
            }, {"collision": False, "minimum_clearance_m": 0.01}]),
        ):
            detail = local.begin_payload_vertical_escape()

        self.assertFalse(detail["ok"], detail)
        self.assertIn("materially above", detail["reason"])
        self.assertEqual(local._payload_escape_obstacles, set())

    def test_payload_escape_never_ignores_robot_or_table_collision(self):
        local = LocalArm.__new__(LocalArm)
        local._lock = threading.RLock()
        local._collision_payload_box = {"size_m": np.ones(3)}
        local._collision_payload_radius_m = 0.0
        local._payload_escape_obstacles = set()
        local._collision_scene_version = 0

        with (
            patch.object(local, "state", return_value={"joints_rad": [0.0] * 6}),
            patch.object(local, "_configuration_collision", return_value={
                "collision": True, "kind": "environment",
                "robot_link": "gripper_left_0", "obstacle": "scene_box_6",
            }),
        ):
            robot = local.begin_payload_vertical_escape()
        self.assertFalse(robot["ok"])
        self.assertEqual(local._payload_escape_obstacles, set())

        with (
            patch.object(local, "state", return_value={"joints_rad": [0.0] * 6}),
            patch.object(local, "_configuration_collision", return_value={
                "collision": True, "kind": "payload", "obstacle": "table",
            }),
        ):
            table = local.begin_payload_vertical_escape()
        self.assertFalse(table["ok"])
        self.assertIn("table", table["reason"])

    def test_payload_escape_requires_strictly_clear_endpoint_and_restoration(self):
        local = LocalArm.__new__(LocalArm)
        local._lock = threading.RLock()
        local._payload_escape_obstacles = {"scene_box_6"}
        local._collision_scene_version = 4
        solution = {"joints_rad": [0.1] * 6}
        with (
            patch.object(local, "solve_ik", return_value=solution),
            patch.object(local, "_configuration_collision", return_value={
                "collision": False, "minimum_clearance_m": 0.004,
            }) as collision,
        ):
            endpoint = local.validate_payload_escape_endpoint(
                0.3, 0.0, 0.25, 0.0, 0.0, 0.0
            )
        self.assertTrue(endpoint["ok"], endpoint)
        self.assertEqual(
            collision.call_args.kwargs["ignored_payload_obstacles"], set()
        )

        with (
            patch.object(local, "state", return_value={"joints_rad": [0.1] * 6}),
            patch.object(local, "_configuration_collision", return_value={
                "collision": False, "minimum_clearance_m": 0.004,
            }),
        ):
            restored = local.end_payload_vertical_escape()
        self.assertTrue(restored["ok"], restored)
        self.assertEqual(local._payload_escape_obstacles, set())
        self.assertEqual(local._collision_scene_version, 5)

    def test_payload_escape_scope_allows_only_fixed_xy_rotation_and_upward_z(self):
        local = LocalArm.__new__(LocalArm)
        local._payload_escape_obstacles = {"scene_box_2"}
        local._payload_escape_reference = {
            "payload_center_m": np.array([0.30, 0.02, 0.10]),
            "payload_rotation": np.eye(3),
        }

        allowed = local._payload_escape_scope_violation(
            np.array([0.30, 0.02, 0.18]), np.eye(3)
        )
        lateral = local._payload_escape_scope_violation(
            np.array([0.32, 0.02, 0.18]), np.eye(3)
        )
        downward = local._payload_escape_scope_violation(
            np.array([0.30, 0.02, 0.09]), np.eye(3)
        )
        angle = 0.08
        rotated = local._payload_escape_scope_violation(
            np.array([0.30, 0.02, 0.18]),
            np.array([
                [np.cos(angle), -np.sin(angle), 0.0],
                [np.sin(angle), np.cos(angle), 0.0],
                [0.0, 0.0, 1.0],
            ]),
        )

        self.assertIsNone(allowed)
        self.assertEqual(lateral["kind"], "payload_escape_scope")
        self.assertLess(downward["payload_z_progress_m"], 0.0)
        self.assertGreater(rotated["payload_rotation_error_rad"], 0.05)

    def test_move_pose_path_revalidates_fresh_live_start_before_command(self):
        local = LocalArm.__new__(LocalArm)
        local._lock = threading.RLock()
        planned_start = np.zeros(6)
        live_start = np.full(6, 0.02)
        target = np.full(6, 0.10)
        events = []

        class RecordingTarget:
            def __init__(self):
                self.values = planned_start.copy()

            def __setitem__(self, key, value):
                events.append("command")
                self.values[key] = value

        controller = SimpleNamespace(_q_target=RecordingTarget())
        arm = SimpleNamespace(get_state=lambda: (
            live_start.copy(), np.zeros(6), np.zeros(6)
        ))
        solved = [{
            "joints_rad": target.tolist(),
            "end_effector": {
                "position_m": [0.30, 0.0, 0.20],
                "rpy_rad": [0.0, 0.0, 0.0],
            },
        }]

        def validate(points, *, label):
            events.append("validate")
            self.assertTrue(np.allclose(points[0], live_start), label)
            self.assertTrue(np.allclose(points[-1], target), label)
            return {
                "ok": True,
                "checked_configurations": len(points),
                "minimum_clearance_m": 0.01,
            }

        settled = {
            "joints_rad": target.tolist(),
            "joint_vel_rad_s": [0.0] * 6,
            "end_effector": {
                "position_m": [0.30, 0.0, 0.20],
                "rpy_rad": [0.0, 0.0, 0.0],
            },
        }
        with (
            patch.object(local, "solve_ik_path", return_value=solved),
            patch.object(local, "_require", return_value=(arm, controller)),
            patch.object(
                local, "_assert_collision_free_joint_path",
                side_effect=validate,
            ) as collision_gate,
            patch.object(local, "state", return_value=settled),
            patch("arm_agent.local_arm.time.sleep", return_value=None),
        ):
            result = local.move_pose_path(
                [(0.30, 0.0, 0.20, 0.0, 0.0, 0.0)],
                duration=0.25,
                send_freq=20.0,
            )

        collision_gate.assert_called_once()
        self.assertEqual(events[0], "validate")
        self.assertIn("command", events)
        self.assertTrue(np.allclose(controller._q_target.values, target))
        self.assertEqual(result["joints_rad"], target.tolist())

    def test_payload_box_retains_measured_tool_frame_center(self):
        local = LocalArm.__new__(LocalArm)
        local._collision_model = object()
        local._collision_payload_radius_m = 0.0
        local._collision_payload_box = None
        local._collision_scene_version = 3

        detail = local.update_collision_payload_box(
            [0.03, 0.04, 0.12], [0.0, 1.0, 0.0],
            [0.004, 0.028, -0.002],
        )

        self.assertTrue(detail["ok"])
        self.assertTrue(np.allclose(
            local._collision_payload_box["center_offset_ee_m"],
            [0.004, 0.028, -0.002],
        ))
        self.assertEqual(detail["scene_version"], 4)

    def test_payload_box_preserves_full_measured_axes_and_dimensions(self):
        local = LocalArm.__new__(LocalArm)
        local._collision_model = object()
        local._collision_payload_radius_m = 0.0
        local._collision_payload_box = None
        local._collision_scene_version = 0
        angle = np.radians(20.0)
        rotation = np.array([
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ])

        detail = local.update_collision_payload_box(
            [0.026, 0.062, 0.077], rotation[:, 2],
            [0.01, 0.0, -0.03], axes_ee=rotation,
        )

        self.assertTrue(detail["ok"])
        self.assertTrue(np.allclose(
            local._collision_payload_box["size_m"],
            [0.026, 0.062, 0.077],
        ))
        self.assertTrue(np.allclose(
            local._collision_payload_box["rotation_ee"], rotation,
        ))

    def test_collision_reason_names_exact_pair(self):
        self.assertEqual(
            LocalArm._collision_reason({
                "kind": "payload", "obstacle": "scene_box_18",
            }),
            "payload collision with scene_box_18",
        )
        self.assertEqual(
            LocalArm._collision_reason({
                "kind": "environment", "robot_link": "gripper_left_0",
                "obstacle": "scene_box_4",
            }),
            "environment collision: gripper_left_0 with scene_box_4",
        )

    def test_feedback_wake_enable_has_explicit_acknowledgement(self):
        refused = _WakeGroup(False)
        with (
            patch("arm_agent.local_arm.time.sleep"),
            self.assertRaisesRegex(RuntimeError, "enable acknowledgement"),
        ):
            LocalArm._enable_feedback_wake(SimpleNamespace(
                groups={"arm": refused}
            ))
        self.assertEqual(refused.calls, 3)

        accepted = _WakeGroup(True)
        LocalArm._enable_feedback_wake(
            SimpleNamespace(groups={"arm": accepted})
        )
        self.assertEqual(accepted.calls, 1)

    def test_feedback_wake_retries_in_same_session_but_still_requires_ack(self):
        delayed = _WakeGroup([False, True])
        with patch("arm_agent.local_arm.time.sleep") as sleep:
            LocalArm._enable_feedback_wake(
                SimpleNamespace(groups={"arm": delayed})
            )
        self.assertEqual(delayed.calls, 2)
        sleep.assert_called_once()

    def test_wrong_can_bitrate_is_rejected_before_motor_traffic(self):
        completed = SimpleNamespace(
            stdout="can0: <UP>\n    bitrate 500000 sample-point 0.875\n"
        )
        with (
            patch("arm_agent.local_arm.subprocess.run",
                  return_value=completed),
            patch.dict("os.environ", {"REBOT_CAN_BITRATE": "1000000"}),
        ):
            with self.assertRaisesRegex(RuntimeError, "requires 1000000"):
                LocalArm._verify_can_bitrate("can0")

    def test_station_can_bitrate_is_accepted(self):
        completed = SimpleNamespace(
            stdout="can0: <UP>\n    bitrate 1000000 sample-point 0.750\n"
        )
        with (
            patch("arm_agent.local_arm.subprocess.run",
                  return_value=completed),
            patch.dict("os.environ", {"REBOT_CAN_BITRATE": "1000000"}),
        ):
            self.assertEqual(LocalArm._verify_can_bitrate("can0"), 1000000)

    def test_missing_feedback_zeros_are_never_accepted_as_state(self):
        arm = _Arm({"joint1": None, "joint2": None})
        with patch("arm_agent.local_arm.time.sleep", lambda _value: None):
            with self.assertRaisesRegex(RuntimeError, "no feedback from"):
                LocalArm._verified_feedback(arm, attempts=2)

    def test_faulted_motor_is_rejected(self):
        arm = _Arm({
            "joint1": SimpleNamespace(status_code=0),
            "joint2": SimpleNamespace(status_code=7),
        })
        with patch("arm_agent.local_arm.time.sleep", lambda _value: None):
            with self.assertRaisesRegex(RuntimeError, "joint2=fault:7"):
                LocalArm._verified_feedback(arm, attempts=1)

    def test_real_zero_pose_is_allowed_when_every_motor_replied(self):
        arm = _Arm({
            "joint1": SimpleNamespace(status_code=0),
            "joint2": SimpleNamespace(status_code=0),
        })
        q, velocity, torque = LocalArm._verified_feedback(arm, attempts=1)
        self.assertTrue(np.all(q == 0.0))
        self.assertTrue(np.all(velocity == 0.0))
        self.assertTrue(np.all(torque == 0.0))


if __name__ == "__main__":
    unittest.main()
