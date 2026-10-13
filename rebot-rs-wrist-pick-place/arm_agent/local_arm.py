"""In-process RS-arm backend used by ``arm_safe``.

This owns the CAN connection directly from the ``rebot_rs`` interpreter.  It uses
the existing reBotArm controller and Pinocchio IK, but deliberately does not start
FastAPI or require ``rebot_arm_service`` to be running.
"""
from __future__ import annotations

import atexit
import hashlib
import math
import os
import re
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CONTROL_ROOT = ROOT / "reBotArm_control_py_fresh"
if str(CONTROL_ROOT) not in sys.path:
    sys.path.insert(0, str(CONTROL_ROOT))

from reBotArm_control_py.actuator import RebotArm
from reBotArm_control_py.controllers import RebotArmEndPose
from reBotArm_control_py.kinematics import (
    _resolve_urdf, get_end_effector_frame_id, joint_to_pose, load_robot_model,
    pad_q_for_model, pos_rot_to_se3,
)
from reBotArm_control_py.kinematics.inverse_kinematics import IKParams, solve_ik_with_retry

# Gripper limits come from arm_agent.gripper_model, which holds the angles MEASURED
# on this gripper (close 22.92 deg, open 141.13 deg) in radians. Commands target
# those angles directly; there is no percentage-of-travel abstraction.
from . import gripper_model as _gm  # noqa: E402
from . import mission_control as MC

GRIPPER_CLOSE_RAD = _gm.CLOSE_RAD
GRIPPER_OPEN_RAD = _gm.OPEN_RAD
# Back-compat aliases for callers that still import the old names.
OPEN_RAD = GRIPPER_OPEN_RAD
CLOSE_RAD = GRIPPER_CLOSE_RAD
# Treat "already at the close limit" as done rather than re-driving into the stop.
GRIPPER_ALREADY_CLOSED_RAD = float(os.getenv("REBOT_GRIPPER_CLOSED_EPS_RAD", "") or 0.05)
# Jaw moves are RAMPED rather than stepped. Handing the MIT loop a target far from
# the current position makes it apply kp * (large error) all at once, which snaps
# the jaws and slams a burst of saturated commands onto the CAN bus. Walking the
# target across in small increments gives the same end state, smoothly.
def _bounded_timing_env(name: str, default: float, low: float, high: float) -> float:
    try:
        requested = float(os.getenv(name, "") or default)
    except ValueError:
        requested = default
    return min(high, max(low, requested))


# The fast grasp profile may shorten these ramps modestly, but hard floors keep
# the jaws from receiving a large position error or a sparse sequence of targets.
GRIPPER_MOVE_S = _bounded_timing_env(
    "REBOT_GRIPPER_MOVE_S", 1.5, 0.8, 3.0
)
GRIPPER_STEP_S = _bounded_timing_env(
    "REBOT_GRIPPER_STEP_S", 0.05, 0.025, 0.10
)
def _joint_vlim() -> float:
    """Per-joint velocity ceiling (rad/s). Read from arm_safe so the joint ceiling
    and the cartesian pace are tuned together; imported lazily because arm_safe
    pulls this module in itself."""
    try:
        from . import arm_safe
        return float(arm_safe.JOINT_VLIM)
    except Exception:  # noqa: BLE001
        return float(os.getenv("REBOT_JOINT_VLIM", "") or 0.25)
# The bundled numerical IK can plateau just outside its combined SE(3) residual
# threshold even when the positional solution is excellent.  This acceptance
# envelope is deliberately tighter than arm_safe's 6 mm request check.
# Near-converged numerical IK remains tighter than arm_safe's independent 6 mm
# execution envelope. Hardware continuation paths have plateaued at 3.5 mm with
# only 0.0012 rad orientation error near the extended grasp boundary; 4 mm admits
# that numerically stable solution while retaining 2 mm beneath the outer gate.
LOCAL_IK_POS_TOL_M = float(
    os.getenv("REBOT_LOCAL_IK_POS_TOL_M", "") or 0.004
)
LOCAL_IK_RPY_TOL_RAD = 0.01
LOCAL_IK_MAX_ITER = int(os.getenv("REBOT_IK_MAX_ITER", "400"))
PATH_IK_MAX_JOINT_STEP_RAD = float(
    os.getenv("REBOT_IK_PATH_MAX_JOINT_STEP_RAD", "") or 0.25
)
HOME_POS_TOL_RAD = 0.03
HOME_VEL_TOL_RAD_S = 0.10
HOME_SETTLE_TIMEOUT_S = 2.0


def _collision_library():
    """Coal is the current HPP-FCL Python name; retain old-package fallback."""
    try:
        import coal
        return coal
    except ImportError:
        import hppfcl
        return hppfcl


class LocalArm:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._arm: RebotArm | None = None
        self._controller: RebotArmEndPose | None = None
        self._model = load_robot_model()
        self._data = self._model.createData()
        self._end_frame = get_end_effector_frame_id(self._model)
        self._ik = IKParams(
            max_iter=max(200, min(800, LOCAL_IK_MAX_ITER)),
            tolerance=1e-4, step_size=0.5, damping=1e-6,
        )
        self._joint_lo = np.asarray(self._model.lowerPositionLimit[:6], dtype=np.float64)
        self._joint_hi = np.asarray(self._model.upperPositionLimit[:6], dtype=np.float64)
        # Populated only when the wrist skill explicitly enables scene collision
        # checking.  Keeping construction lazy avoids loading mesh geometry for
        # read-only state/gripper commands.
        self._collision_boxes: list[dict[str, Any]] = []
        self._collision_table_z_m: float | None = None
        self._collision_voxel_m: float | None = None
        self._collision_inflation_m: float | None = None
        self._collision_payload_radius_m = 0.0
        self._collision_payload_box: dict[str, np.ndarray] | None = None
        # Exact scene boxes that already overlap an attached payload at the
        # grasp pose.  This allow-list exists only during a bounded vertical
        # extraction and is cleared before any carry planning.
        self._payload_escape_obstacles: set[str] = set()
        # World-frame payload pose at the instant the scoped +Z extraction is
        # armed.  While the allow-list is active, collision checking rejects
        # lateral, downward, or rotational motion before considering any
        # payload/scene exception.
        self._payload_escape_reference: dict[str, np.ndarray] | None = None
        self._collision_model: Any = None
        self._collision_data: Any = None
        self._self_collision_pairs: list[tuple[int, int]] = []
        self._collision_scene_version = 0

    def _ensure_collision_model(self) -> None:
        if self._collision_model is not None:
            return
        try:
            import pinocchio as pin
            urdf_path, package_dir = _resolve_urdf()
            geometry = pin.buildGeomFromUrdf(
                self._model, urdf_path, pin.GeometryType.COLLISION,
                package_dirs=[package_dir],
            )
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                f"URDF collision geometry unavailable: {type(exc).__name__}: {exc}"
            ) from exc
        pairs: list[tuple[int, int]] = []
        parents = self._model.parents
        for first in range(geometry.ngeoms):
            joint_a = int(geometry.geometryObjects[first].parentJoint)
            for second in range(first + 1, geometry.ngeoms):
                joint_b = int(geometry.geometryObjects[second].parentJoint)
                # Directly connected links and sibling jaws overlap by design at
                # their joints; SRDF-style filtering removes those permanent
                # contacts while retaining every non-adjacent pair.
                adjacent = (
                    joint_a == joint_b
                    or int(parents[joint_a]) == joint_b
                    or int(parents[joint_b]) == joint_a
                    or int(parents[joint_a]) == int(parents[joint_b])
                )
                if not adjacent:
                    pairs.append((first, second))
        self._collision_model = geometry
        self._collision_data = pin.GeometryData(geometry)
        self._self_collision_pairs = pairs

    def configure_collision_scene(
        self,
        boxes: list[dict[str, Any]],
        *,
        table_z_m: float,
        payload_radius_m: float = 0.0,
        voxel_m: float | None = None,
        inflation_m: float | None = None,
    ) -> dict[str, Any]:
        """Install a static base-frame scene used by every subsequent motion."""
        normalized: list[dict[str, Any]] = []
        for index, item in enumerate(boxes):
            center = np.asarray(item.get("center_m"), dtype=float)
            size = np.asarray(item.get("size_m"), dtype=float)
            if center.shape != (3,) or size.shape != (3,) \
                    or not np.all(np.isfinite(center)) \
                    or not np.all(np.isfinite(size)) \
                    or np.any(size <= 0.0):
                raise ValueError(f"invalid obstacle box {index}")
            normalized.append({"center_m": center, "size_m": size})
        self._ensure_collision_model()
        self._collision_boxes = normalized
        self._collision_table_z_m = float(table_z_m)
        self._collision_voxel_m = (
            None if voxel_m is None else max(0.0, float(voxel_m))
        )
        self._collision_inflation_m = (
            None if inflation_m is None else max(0.0, float(inflation_m))
        )
        self._collision_payload_radius_m = float(np.clip(
            payload_radius_m, 0.0, 0.12
        ))
        self._collision_payload_box = None
        self._payload_escape_obstacles = set()
        self._payload_escape_reference = None
        self._collision_scene_version += 1
        return {
            "ok": True,
            "obstacle_boxes": len(normalized),
            "table_z_m": self._collision_table_z_m,
            "voxel_m": self._collision_voxel_m,
            "inflation_m": self._collision_inflation_m,
            "payload_radius_m": self._collision_payload_radius_m,
            "self_collision_pairs": len(self._self_collision_pairs),
            "scene_version": self._collision_scene_version,
        }

    def filter_scene_boxes_at_current_pose(
        self, boxes: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """Remove RGB-D returns that overlap the robot in the capture pose."""
        self._ensure_collision_model()
        collision = _collision_library()
        import pinocchio as pin
        q6 = np.asarray(self.state()["joints_rad"], dtype=float)
        q = pad_q_for_model(self._model, q6, controlled_joints=6)
        geometry = self._collision_model
        geometry_data = self._collision_data
        pin.forwardKinematics(self._model, self._data, q)
        pin.updateGeometryPlacements(self._model, self._data, geometry, geometry_data, q)
        request = collision.CollisionRequest()
        kept, removed = [], []
        for box_index, item in enumerate(boxes):
            box = collision.Box(*[float(v) for v in item["size_m"]])
            box_tf = self._fcl_transform(np.eye(3), item["center_m"])
            overlaps = []
            for geom_index, robot_object in enumerate(geometry.geometryObjects):
                if int(robot_object.parentJoint) <= 1:
                    continue
                pose = geometry_data.oMg[geom_index]
                result = collision.CollisionResult()
                if collision.collide(
                    robot_object.geometry,
                    self._fcl_transform(pose.rotation, pose.translation),
                    box, box_tf, request, result,
                ):
                    overlaps.append(robot_object.name)
            if overlaps:
                removed.append({"index": box_index, "robot_links": overlaps})
            else:
                kept.append(item)
        return {"ok": True, "kept": kept, "removed": removed}

    def validate_current_collision_scene(self) -> dict[str, Any]:
        """Require the newly installed map to clear the measured robot pose.

        Self-filtering is performed before scene installation.  A final check
        against fresh joint feedback catches pose drift, quantization, or an
        incorrectly retained robot return.  Such a map is rejected rather than
        letting every subsequent route appear trapped inside an obstacle.
        """
        with self._lock:
            q6 = np.asarray(self.state()["joints_rad"], dtype=float)
            detail = self._configuration_collision(q6)
            if detail.get("collision"):
                return {
                    "ok": False,
                    "reason": (
                        "installed scene collides with the measured capture "
                        f"pose: {self._collision_reason(detail)}"
                    ),
                    "collision": detail,
                }
            return {
                "ok": True,
                "reason": "installed scene clears the measured capture pose",
                "minimum_clearance_m": detail.get("minimum_clearance_m"),
            }

    def update_collision_payload(self, radius_m: float) -> dict[str, Any]:
        if getattr(self, "_collision_model", None) is None:
            return {"ok": False, "reason": "collision scene is not configured"}
        self._collision_payload_radius_m = float(np.clip(radius_m, 0.0, 0.12))
        self._collision_payload_box = None
        self._payload_escape_obstacles = set()
        self._payload_escape_reference = None
        self._collision_scene_version += 1
        return {"ok": True, "payload_radius_m": self._collision_payload_radius_m,
                "scene_version": self._collision_scene_version}

    def update_collision_payload_box(
        self, size_m: Any, long_axis_ee: Any,
        center_offset_ee_m: Any | None = None,
        axes_ee: Any | None = None,
    ) -> dict[str, Any]:
        """Attach an oriented carried-object box to the end-effector frame.

        ``center_offset_ee_m`` is the measured vector from the URDF tool origin
        to the centre of the object in tool coordinates.  Treating the tool
        origin as the object centre makes half of a long payload protrude into
        the wrist and creates false endpoint collisions.
        """
        size = np.asarray(size_m, dtype=float)
        axis = np.asarray(long_axis_ee, dtype=float)
        center = np.zeros(3, dtype=float) if center_offset_ee_m is None else \
            np.asarray(center_offset_ee_m, dtype=float)
        if size.shape != (3,) or axis.shape != (3,) or center.shape != (3,) \
                or np.any(size <= 0.0) or not np.all(np.isfinite(center)):
            raise ValueError("payload box needs positive XYZ size and a 3D axis")
        axis /= max(1e-9, float(np.linalg.norm(axis)))
        if axes_ee is not None:
            measured_rotation = np.asarray(axes_ee, dtype=float)
            if measured_rotation.shape != (3, 3) or not np.all(
                np.isfinite(measured_rotation)
            ):
                raise ValueError("payload box axes need a finite 3x3 rotation")
            u, _singular, vh = np.linalg.svd(measured_rotation)
            rotation_ee = u @ vh
            if float(np.linalg.det(rotation_ee)) < 0.0:
                u[:, -1] *= -1.0
                rotation_ee = u @ vh
            fit_error = float(np.linalg.norm(
                measured_rotation - rotation_ee
            ) / max(1e-9, np.linalg.norm(rotation_ee)))
            if fit_error > 0.08:
                raise ValueError(
                    f"payload box axes are not rigid (fit error {fit_error:.3f})"
                )
        else:
            helper = np.array([1.0, 0.0, 0.0])
            if abs(float(np.dot(helper, axis))) > 0.9:
                helper = np.array([0.0, 1.0, 0.0])
            x_axis = np.cross(helper, axis)
            x_axis /= max(1e-9, float(np.linalg.norm(x_axis)))
            y_axis = np.cross(axis, x_axis)
            rotation_ee = np.column_stack((x_axis, y_axis, axis))
        self._collision_payload_box = {
            "size_m": size,
            "rotation_ee": rotation_ee,
            "center_offset_ee_m": center,
        }
        self._collision_payload_radius_m = 0.0
        self._payload_escape_obstacles = set()
        self._payload_escape_reference = None
        self._collision_scene_version += 1
        return {"ok": True, "payload_box_size_m": size.tolist(),
                "payload_box_center_offset_ee_m": center.tolist(),
                "payload_box_axes_ee": rotation_ee.tolist(),
                "scene_version": self._collision_scene_version}

    def clear_collision_scene(self) -> None:
        self._collision_boxes = []
        self._collision_table_z_m = None
        self._collision_voxel_m = None
        self._collision_inflation_m = None
        self._collision_payload_radius_m = 0.0
        self._collision_payload_box = None
        self._payload_escape_obstacles = set()
        self._payload_escape_reference = None
        self._collision_scene_version += 1

    @property
    def collision_scene_version(self) -> int:
        """Monotonic version covering obstacles, table and attached payload."""
        return int(self._collision_scene_version)

    @staticmethod
    def _fcl_transform(rotation: Any, translation: Any) -> Any:
        collision = _collision_library()
        transform_type = getattr(
            collision, "Transform3f", getattr(collision, "Transform3s")
        )
        return transform_type(
            np.asarray(rotation, dtype=float),
            np.asarray(translation, dtype=float),
        )

    @staticmethod
    def _payload_escape_scope_limits(
        payload_aabb_size_m: Any,
    ) -> dict[str, float]:
        """Numerical pose envelope shared by reservation and execution."""
        payload_size = np.asarray(payload_aabb_size_m, dtype=float)
        if payload_size.shape != (3,) or not np.all(np.isfinite(payload_size)):
            payload_size = np.zeros(3, dtype=float)
        xy_tolerance = max(0.002, 2.0 * LOCAL_IK_POS_TOL_M)
        rotation_tolerance = max(0.005, 2.0 * LOCAL_IK_RPY_TOL_RAD)
        downward_tolerance = max(0.001, LOCAL_IK_POS_TOL_M)
        # A rotation by theta can move any payload point by at most
        # 2*r*sin(theta/2). Fold that into the pre-reserved translational shell
        # so a pose inside the execution scope cannot discover a new initial
        # contact merely because a conservative OBB corner moved slightly.
        radius = 0.5 * float(np.linalg.norm(payload_size))
        rotation_shell = 2.0 * radius * math.sin(rotation_tolerance / 2.0)
        return {
            "xy_tolerance_m": xy_tolerance,
            "rotation_tolerance_rad": rotation_tolerance,
            "downward_tolerance_m": downward_tolerance,
            "rotation_shell_m": rotation_shell,
            "reservation_xy_shell_m": xy_tolerance + rotation_shell,
            "reservation_z_shell_m": downward_tolerance + rotation_shell,
        }

    def _payload_escape_scope_violation(
        self, payload_center: np.ndarray, payload_rotation: np.ndarray
    ) -> dict[str, Any] | None:
        """Reject any use of the temporary allow-list outside a +Z sweep."""
        escape_reference = getattr(self, "_payload_escape_reference", None)
        if not self._payload_escape_obstacles or escape_reference is None:
            return None
        reference_center = np.asarray(
            escape_reference["payload_center_m"], dtype=float
        )
        reference_rotation = np.asarray(
            escape_reference["payload_rotation"], dtype=float
        )
        xy_drift = float(np.linalg.norm(
            np.asarray(payload_center, dtype=float)[:2]
            - reference_center[:2]
        ))
        z_progress = float(payload_center[2] - reference_center[2])
        relative_rotation = reference_rotation.T @ payload_rotation
        rotation_error = float(np.arccos(np.clip(
            (float(np.trace(relative_rotation)) - 1.0) / 2.0,
            -1.0, 1.0,
        )))
        limits = self._payload_escape_scope_limits(
            escape_reference.get("payload_aabb_size_m", np.zeros(3))
        )
        xy_tolerance = limits["xy_tolerance_m"]
        rotation_tolerance = limits["rotation_tolerance_rad"]
        downward_tolerance = limits["downward_tolerance_m"]
        if xy_drift <= xy_tolerance \
                and rotation_error <= rotation_tolerance \
                and z_progress >= -downward_tolerance:
            return None
        return {
            "collision": True,
            "kind": "payload_escape_scope",
            "reason": (
                "payload exception permits only fixed-XY, fixed-orientation, "
                "upward motion"
            ),
            "payload_xy_drift_m": round(xy_drift, 6),
            "payload_z_progress_m": round(z_progress, 6),
            "payload_rotation_error_rad": round(rotation_error, 6),
            "xy_tolerance_m": round(xy_tolerance, 6),
            "rotation_tolerance_rad": round(rotation_tolerance, 6),
        }

    def _payload_scene_proximity(
        self,
        payload_center_m: Any,
        payload_rotation: Any,
        maximum_distance_m: float,
    ) -> dict[str, float]:
        """Return exact OBB/sphere-to-scene distances inside a bounded tube."""
        collision = _collision_library()
        center = np.asarray(payload_center_m, dtype=float)
        rotation = np.asarray(payload_rotation, dtype=float)
        if center.shape != (3,) or rotation.shape != (3, 3) \
                or not np.all(np.isfinite(center)) \
                or not np.all(np.isfinite(rotation)):
            raise ValueError("payload proximity query needs a finite world pose")
        if self._collision_payload_box is not None:
            payload = collision.Box(*[
                float(value)
                for value in self._collision_payload_box["size_m"]
            ])
        elif self._collision_payload_radius_m > 0.0:
            payload = collision.Sphere(
                float(self._collision_payload_radius_m)
            )
        else:
            raise ValueError("payload proximity query has no attached payload")
        payload_tf = self._fcl_transform(rotation, center)
        request = collision.DistanceRequest()
        maximum = max(0.0, float(maximum_distance_m))
        nearby: dict[str, float] = {}
        for index, item in enumerate(self._collision_boxes):
            obstacle = collision.Box(*[
                float(value) for value in item["size_m"]
            ])
            obstacle_tf = self._fcl_transform(
                np.eye(3), item["center_m"]
            )
            result = collision.DistanceResult()
            distance = float(collision.distance(
                payload, payload_tf, obstacle, obstacle_tf, request, result
            ))
            if distance <= maximum + 1e-9:
                nearby[f"scene_box_{index}"] = max(0.0, distance)
        return nearby

    def _configuration_collision(
        self, q6: np.ndarray, *,
        ignored_payload_obstacles: set[str] | None = None,
    ) -> dict[str, Any]:
        """Return the first collision and minimum inflated-scene clearance."""
        if self._collision_model is None:
            return {"collision": False, "minimum_clearance_m": None}
        collision = _collision_library()
        import pinocchio as pin

        geometry = self._collision_model
        geometry_data = self._collision_data
        q = pad_q_for_model(
            self._model, np.asarray(q6, dtype=float), controlled_joints=6
        )
        pin.forwardKinematics(self._model, self._data, q)
        pin.updateFramePlacements(self._model, self._data)
        pin.updateGeometryPlacements(
            self._model, self._data, geometry, geometry_data, q
        )
        request = collision.CollisionRequest()

        for first, second in self._self_collision_pairs:
            result = collision.CollisionResult()
            pose_a = geometry_data.oMg[first]
            pose_b = geometry_data.oMg[second]
            hit = collision.collide(
                geometry.geometryObjects[first].geometry,
                self._fcl_transform(pose_a.rotation, pose_a.translation),
                geometry.geometryObjects[second].geometry,
                self._fcl_transform(pose_b.rotation, pose_b.translation),
                request, result,
            )
            if hit:
                return {
                    "collision": True,
                    "kind": "self",
                    "first": geometry.geometryObjects[first].name,
                    "second": geometry.geometryObjects[second].name,
                }

        obstacle_geometries: list[tuple[str, Any, Any]] = []
        for index, item in enumerate(self._collision_boxes):
            obstacle_geometries.append((
                f"scene_box_{index}",
                collision.Box(*[float(v) for v in item["size_m"]]),
                self._fcl_transform(np.eye(3), item["center_m"]),
            ))
        if self._collision_table_z_m is not None:
            table_depth = 0.10
            obstacle_geometries.append((
                "table",
                collision.Box(1.5, 1.5, table_depth),
                self._fcl_transform(
                    np.eye(3),
                    [0.25, 0.0, self._collision_table_z_m - table_depth / 2.0],
                ),
            ))

        minimum = float("inf")
        distance_request = collision.DistanceRequest()
        for geom_index, robot_object in enumerate(geometry.geometryObjects):
            # The pedestal intersects the table by construction.  It is fixed and
            # cannot sweep into tabletop clutter, so environmental checks begin at
            # link2 while self-collision still covers all non-adjacent links.
            if int(robot_object.parentJoint) <= 1:
                continue
            robot_pose = geometry_data.oMg[geom_index]
            robot_tf = self._fcl_transform(
                robot_pose.rotation, robot_pose.translation
            )
            for obstacle_name, obstacle_geometry, obstacle_tf in obstacle_geometries:
                result = collision.CollisionResult()
                hit = collision.collide(
                    robot_object.geometry, robot_tf,
                    obstacle_geometry, obstacle_tf,
                    request, result,
                )
                if hit:
                    obstacle_index = (
                        int(obstacle_name.rsplit("_", 1)[-1])
                        if obstacle_name.startswith("scene_box_") else None
                    )
                    return {
                        "collision": True,
                        "kind": "environment",
                        "robot_link": robot_object.name,
                        "obstacle": obstacle_name,
                        "obstacle_center_m": (
                            self._collision_boxes[obstacle_index][
                                "center_m"
                            ].round(4).tolist()
                            if obstacle_index is not None else None
                        ),
                        "obstacle_size_m": (
                            self._collision_boxes[obstacle_index][
                                "size_m"
                            ].round(4).tolist()
                            if obstacle_index is not None else None
                        ),
                    }
                distance_result = collision.DistanceResult()
                distance = collision.distance(
                    robot_object.geometry, robot_tf,
                    obstacle_geometry, obstacle_tf,
                    distance_request, distance_result,
                )
                minimum = min(minimum, float(distance))

        payload_box = self._collision_payload_box
        if self._collision_payload_radius_m > 0.0 or payload_box is not None:
            ee_pose = self._data.oMf[self._end_frame]
            payload_center = np.asarray(ee_pose.translation, dtype=float)
            if payload_box is None:
                payload = collision.Sphere(self._collision_payload_radius_m)
                payload_rotation = ee_pose.rotation
                payload_half_z = float(self._collision_payload_radius_m)
                payload_aabb_size = np.full(
                    3, 2.0 * float(self._collision_payload_radius_m),
                    dtype=float,
                )
            else:
                payload = collision.Box(*[float(v) for v in payload_box["size_m"]])
                payload_rotation = ee_pose.rotation @ payload_box["rotation_ee"]
                payload_center = payload_center + ee_pose.rotation @ \
                    payload_box["center_offset_ee_m"]
                payload_aabb_size = np.abs(payload_rotation) @ np.asarray(
                    payload_box["size_m"], dtype=float
                )
                # Projection of an oriented box's half extents onto world +Z.
                # During the scoped extraction the tool orientation and XY are
                # fixed, so this lower face translates monotonically upward.
                payload_half_z = float(np.sum(
                    np.abs(payload_rotation[2, :])
                    * np.asarray(payload_box["size_m"], dtype=float) / 2.0
                ))
            payload_tf = self._fcl_transform(
                payload_rotation, payload_center
            )
            ignored = (
                self._payload_escape_obstacles
                if ignored_payload_obstacles is None
                else ignored_payload_obstacles
            )
            scope_violation = self._payload_escape_scope_violation(
                payload_center, payload_rotation
            )
            if scope_violation is not None:
                return scope_violation
            for obstacle_name, obstacle_geometry, obstacle_tf in obstacle_geometries:
                if obstacle_name in ignored:
                    continue
                result = collision.CollisionResult()
                hit = collision.collide(
                    payload, payload_tf, obstacle_geometry, obstacle_tf,
                    request, result,
                )
                if hit:
                    obstacle_index = (
                        int(obstacle_name.rsplit("_", 1)[-1])
                        if obstacle_name.startswith("scene_box_") else None
                    )
                    if obstacle_index is not None:
                        obstacle_center = np.asarray(
                            self._collision_boxes[obstacle_index]["center_m"],
                            dtype=float,
                        )
                        obstacle_size = np.asarray(
                            self._collision_boxes[obstacle_index]["size_m"],
                            dtype=float,
                        )
                    elif obstacle_name == "table" \
                            and self._collision_table_z_m is not None:
                        obstacle_center = np.array([
                            0.25, 0.0, self._collision_table_z_m - 0.05
                        ])
                        obstacle_size = np.array([1.5, 1.5, 0.10])
                    else:
                        obstacle_center = None
                        obstacle_size = None
                    return {
                        "collision": True,
                        "kind": "payload",
                        "obstacle": obstacle_name,
                        "obstacle_center_m": (
                            obstacle_center.round(4).tolist()
                            if obstacle_center is not None else None
                        ),
                        "obstacle_size_m": (
                            obstacle_size.round(4).tolist()
                            if obstacle_size is not None else None
                        ),
                        "payload_center_m": payload_center.round(4).tolist(),
                        "payload_aabb_size_m":
                            payload_aabb_size.round(4).tolist(),
                        "payload_rotation_matrix":
                            np.asarray(payload_rotation, dtype=float).round(6).tolist(),
                        "payload_z_bounds_m": [
                            round(float(payload_center[2] - payload_half_z), 4),
                            round(float(payload_center[2] + payload_half_z), 4),
                        ],
                    }
                distance_result = collision.DistanceResult()
                distance = collision.distance(
                    payload, payload_tf, obstacle_geometry, obstacle_tf,
                    distance_request, distance_result,
                )
                minimum = min(minimum, float(distance))
        return {
            "collision": False,
            "minimum_clearance_m": None if not np.isfinite(minimum) else minimum,
        }

    def _payload_escape_cluster_geometry(
        self,
        obstacle_names: set[str],
        payload_center_m: Any,
        payload_aabb_size_m: Any,
        measured_payload_aabb_size_m: Any = None,
        strict_obstacle_names: set[str] | None = None,
    ) -> dict[str, Any]:
        """Validate exact initial contacts for a scoped vertical extraction.

        Target-owned collision columns are deliberately removed before a pick.
        Their removal can split the inflated perimeter around a long object into
        several disconnected bands, so mutual box connectivity is not a valid
        safety invariant. Instead, every allowed name must either be an exact
        stationary FCL overlap or intersect the conservative pose-uncertainty
        envelope reserved at that same pose. It must remain support-like in Z
        and have bounded lateral size relative to both the measured payload and
        the map's base cell. The active exception separately enforces fixed XY,
        fixed orientation, and non-decreasing Z; all robot collisions and every
        obstacle outside the reserved envelope remain enabled throughout.
        """
        payload_center = np.asarray(payload_center_m, dtype=float)
        payload_size = np.asarray(payload_aabb_size_m, dtype=float)
        if payload_center.shape != (3,) or payload_size.shape != (3,) \
                or not np.all(np.isfinite(payload_center)) \
                or not np.all(np.isfinite(payload_size)) \
                or np.any(payload_size <= 0.0):
            return {
                "ok": False,
                "reason": "payload overlap lacks finite world-AABB geometry",
                "allowed_initial_obstacles": sorted(obstacle_names),
            }

        records: list[dict[str, Any]] = []
        for name in sorted(obstacle_names):
            match = re.fullmatch(r"scene_box_(\d+)", name)
            if match is None:
                return {
                    "ok": False,
                    "reason": "payload overlap contains a non-scene obstacle",
                    "allowed_initial_obstacles": sorted(obstacle_names),
                    "rejected_obstacle": name,
                }
            index = int(match.group(1))
            if index < 0 or index >= len(self._collision_boxes):
                return {
                    "ok": False,
                    "reason": "payload overlap names an unconfigured scene box",
                    "allowed_initial_obstacles": sorted(obstacle_names),
                    "rejected_obstacle": name,
                    "configured_scene_boxes": len(self._collision_boxes),
                }
            center = np.asarray(
                self._collision_boxes[index]["center_m"], dtype=float
            )
            size = np.asarray(
                self._collision_boxes[index]["size_m"], dtype=float
            )
            if center.shape != (3,) or size.shape != (3,) \
                    or not np.all(np.isfinite(center)) \
                    or not np.all(np.isfinite(size)) or np.any(size <= 0.0):
                return {
                    "ok": False,
                    "reason": "payload overlap has invalid scene-box geometry",
                    "allowed_initial_obstacles": sorted(obstacle_names),
                    "rejected_obstacle": name,
                }
            records.append({
                "name": name,
                "index": index,
                "center_m": center,
                "size_m": size,
                "lower_m": center - size / 2.0,
                "upper_m": center + size / 2.0,
            })

        if not records:
            return {
                "ok": True,
                "reason": "payload grasp pose is already collision-free",
                "allowed_initial_obstacles": [],
                "discovered_boxes": [],
            }

        # The reservation tube widens this envelope so nearby names can be
        # discovered and guarded. That widened box is a *containment* test only.
        # It must not become the reported payload: the caller derives lift
        # distance from the payload bottom, so an inflated bottom silently turns
        # an uncertainty allowance into extra commanded travel. Keep the guarded
        # box for the separation checks and report the measured one.
        guard_lower = payload_center - payload_size / 2.0
        guard_upper = payload_center + payload_size / 2.0
        measured_size = payload_size
        if measured_payload_aabb_size_m is not None:
            candidate_size = np.asarray(
                measured_payload_aabb_size_m, dtype=float
            )
            if candidate_size.shape == (3,) \
                    and np.all(np.isfinite(candidate_size)) \
                    and np.all(candidate_size > 0.0):
                measured_size = candidate_size
        payload_lower = payload_center - measured_size / 2.0
        payload_upper = payload_center + measured_size / 2.0
        scale = max(
            1.0,
            float(np.max(np.abs(payload_center))),
            float(np.max(payload_size)),
            max(float(np.max(record["size_m"])) for record in records),
        )
        epsilon = 128.0 * np.finfo(float).eps * scale

        diagnostic_boxes = [{
            "name": record["name"],
            "center_m": record["center_m"].round(4).tolist(),
            "size_m": record["size_m"].round(4).tolist(),
        } for record in records]
        # An *exact* reported contact that lies outside the payload geometry is
        # a contradiction and still fails closed. A name discovered only by the
        # reservation tube is different: that tube is deliberately generous, so
        # it can surface boxes the payload does not actually touch. Those are
        # reserved-near, not initial contacts, and must simply be dropped rather
        # than abort the extraction — otherwise a distant box anywhere in the
        # scene can veto every escape.
        strict_names = set(strict_obstacle_names or obstacle_names)
        contained: list[dict[str, Any]] = []
        dropped_far_names: list[str] = []
        for record in records:
            separation = np.maximum(
                guard_lower - record["upper_m"],
                record["lower_m"] - guard_upper,
            )
            if np.any(separation > epsilon):
                if record["name"] in strict_names:
                    return {
                        "ok": False,
                        "reason": (
                            "reported payload overlap lies outside the attached "
                            "payload geometry"
                        ),
                        "allowed_initial_obstacles": sorted(obstacle_names),
                        "rejected_obstacle": record["name"],
                        "separation_m": separation.round(6).tolist(),
                        "payload_bounds_m": [
                            payload_lower.round(4).tolist(),
                            payload_upper.round(4).tolist(),
                        ],
                        "discovered_boxes": diagnostic_boxes,
                    }
                dropped_far_names.append(record["name"])
                continue
            contained.append(record)
        if not contained:
            return {
                "ok": True,
                "reason": (
                    "no reserved scene box actually intersects the payload"
                ),
                "allowed_initial_obstacles": [],
                "discovered_boxes": diagnostic_boxes,
                "dropped_far_obstacles": sorted(dropped_far_names),
            }
        records = contained
        obstacle_names = {record["name"] for record in records}

        cluster_lower = np.min(np.stack([
            record["lower_m"] for record in records
        ]), axis=0)
        cluster_upper = np.max(np.stack([
            record["upper_m"] for record in records
        ]), axis=0)
        configured_sizes = [
            np.asarray(item.get("size_m"), dtype=float)
            for item in self._collision_boxes
        ]
        configured_xy = [
            size[:2] for size in configured_sizes
            if size.shape == (3,) and np.all(np.isfinite(size))
            and np.all(size > 0.0)
        ]
        if not configured_xy:
            return {
                "ok": False,
                "reason": "scene map has no finite base-cell geometry",
                "allowed_initial_obstacles": sorted(obstacle_names),
                "discovered_boxes": diagnostic_boxes,
            }
        base_scene_cell_xy = np.min(np.stack(configured_xy), axis=0)
        # Consolidation can produce a long box whose remote tail is irrelevant
        # to a fixed-XY payload sweep. Still reject a single unbounded wall-like
        # box: at most one base cell of extent on either side of the payload is
        # permitted. This bound is derived from measured geometry, not a box
        # count or an object-specific constant.
        maximum_box_xy = payload_size[:2] + 2.0 * base_scene_cell_xy
        oversized = [
            record["name"] for record in records
            if np.any(record["size_m"][:2] > maximum_box_xy + epsilon)
        ]
        # A vertical escape is valid only for table/support-like contacts. A
        # box that begins materially above the payload's lower face or extends
        # materially above its upper face is a side/overhead obstacle and stays
        # non-bypassable.
        configured_inflation = getattr(
            self, "_collision_inflation_m", None
        )
        geometry_halo = (
            float(configured_inflation)
            if configured_inflation is not None
            and np.isfinite(configured_inflation)
            and configured_inflation > 0.0
            else 0.5 * float(np.min(base_scene_cell_xy))
        )
        support_root_halo = geometry_halo
        top_halo = min(geometry_halo, float(np.max(payload_size)))
        maximum_support_top = float(payload_upper[2] + top_halo)
        elevated = [
            record["name"] for record in records
            if float(record["lower_m"][2]) >
            float(payload_lower[2]) + support_root_halo + epsilon
        ]
        too_tall = [
            record["name"] for record in records
            if float(record["upper_m"][2]) > maximum_support_top + epsilon
        ]
        if oversized or elevated or too_tall:
            return {
                "ok": False,
                "reason": (
                    "initial payload overlap is not vertically separable: "
                    + (
                        "a scene box has unbounded lateral extent"
                        if oversized else
                        "a scene box begins materially above the payload"
                        if elevated else
                        "scene-box top is materially above the payload"
                    )
                ),
                "allowed_initial_obstacles": sorted(obstacle_names),
                "oversized_boxes": sorted(oversized),
                "elevated_boxes": sorted(elevated),
                "too_tall_boxes": sorted(too_tall),
                "payload_bounds_m": [
                    payload_lower.round(4).tolist(),
                    payload_upper.round(4).tolist(),
                ],
                "cluster_bounds_m": [
                    cluster_lower.round(4).tolist(),
                    cluster_upper.round(4).tolist(),
                ],
                "base_scene_cell_xy_m":
                    base_scene_cell_xy.round(4).tolist(),
                "scene_geometry_halo_m": round(geometry_halo, 4),
                "maximum_box_xy_m": maximum_box_xy.round(4).tolist(),
                "maximum_support_bottom_m": round(
                    float(payload_lower[2]) + support_root_halo, 4
                ),
                "maximum_support_top_m": round(maximum_support_top, 4),
                "discovered_boxes": diagnostic_boxes,
            }

        # Retain ordinary connectivity only as a diagnostic. Several components
        # are expected when the selected target was removed from the middle of
        # an inflated height map; they are bridged by the measured payload, not
        # by occupied environment geometry.
        remaining = set(range(len(records)))
        components: list[list[str]] = []
        while remaining:
            seed = remaining.pop()
            connected = {seed}
            frontier = [seed]
            while frontier:
                first = frontier.pop()
                for second in list(remaining):
                    separation = np.maximum(
                        records[first]["lower_m"]
                        - records[second]["upper_m"],
                        records[second]["lower_m"]
                        - records[first]["upper_m"],
                    )
                    if np.all(separation <= epsilon):
                        remaining.remove(second)
                        connected.add(second)
                        frontier.append(second)
            components.append(sorted(
                records[index]["name"] for index in connected
            ))

        payload_bottom = float(payload_lower[2])
        minimum_vertical_clearance_m = max(
            float(record["upper_m"][2]) - payload_bottom + 0.002
            for record in records
        )
        return {
            "ok": True,
            "reason": (
                "guarded initial payload contacts are support-like and "
                "vertically separable under the scoped +Z exception"
            ),
            "allowed_initial_obstacles": sorted(obstacle_names),
            "map_components": components,
            "map_component_count": len(components),
            "discovered_boxes": diagnostic_boxes,
            "payload_bounds_m": [
                payload_lower.round(4).tolist(),
                payload_upper.round(4).tolist(),
            ],
            "cluster_bounds_m": [
                cluster_lower.round(4).tolist(),
                cluster_upper.round(4).tolist(),
            ],
            "base_scene_cell_xy_m": base_scene_cell_xy.round(4).tolist(),
            "scene_geometry_halo_m": round(geometry_halo, 4),
            "maximum_box_xy_m": maximum_box_xy.round(4).tolist(),
            "maximum_support_top_m": round(maximum_support_top, 4),
            "minimum_vertical_clearance_m": round(
                max(0.0, minimum_vertical_clearance_m), 4
            ),
        }

    def begin_payload_vertical_escape(self) -> dict[str, Any]:
        """Allow only payload overlaps already present at the grasp pose.

        Robot-link collisions are never ignored.  Table contact is never
        ignored.  The caller must validate a strictly clear endpoint, execute
        only a short +Z motion, then call :meth:`end_payload_vertical_escape`.
        """
        with self._lock:
            if self._collision_payload_box is None \
                    and self._collision_payload_radius_m <= 0.0:
                return {"ok": False, "reason": "no attached collision payload"}
            if self._payload_escape_obstacles:
                return {
                    "ok": False,
                    "reason": "payload vertical escape is already active",
                    "allowed_initial_obstacles": sorted(
                        self._payload_escape_obstacles
                    ),
                }
            q6 = np.asarray(self.state()["joints_rad"], dtype=float)
            discovered: set[str] = set()
            payload_center_m = None
            payload_aabb_size_m = None
            payload_rotation = None
            while True:
                detail = self._configuration_collision(
                    q6, ignored_payload_obstacles=discovered
                )
                if not detail.get("collision"):
                    break
                if detail.get("kind") != "payload":
                    return {
                        "ok": False,
                        "reason": "grasp pose has a non-payload collision",
                        "collision": detail,
                        "allowed_initial_obstacles": sorted(discovered),
                    }
                obstacle = str(detail.get("obstacle"))
                if obstacle == "table":
                    return {
                        "ok": False,
                        "reason": "payload intersects the calibrated table",
                        "collision": detail,
                        "allowed_initial_obstacles": sorted(discovered),
                    }
                match = re.fullmatch(r"scene_box_(\d+)", obstacle)
                if match is None:
                    return {
                        "ok": False,
                        "reason": "payload overlap is not a configured scene box",
                        "collision": detail,
                        "allowed_initial_obstacles": sorted(discovered),
                    }
                obstacle_index = int(match.group(1))
                if obstacle_index >= len(self._collision_boxes):
                    return {
                        "ok": False,
                        "reason": "payload overlap names an unconfigured scene box",
                        "collision": detail,
                        "allowed_initial_obstacles": sorted(discovered),
                        "configured_scene_boxes": len(self._collision_boxes),
                    }
                if obstacle in discovered:
                    return {
                        "ok": False,
                        "reason": "collision query repeated an ignored scene box",
                        "collision": detail,
                        "allowed_initial_obstacles": sorted(discovered),
                    }
                try:
                    current_payload_center = np.asarray(
                        detail["payload_center_m"], dtype=float
                    )
                    current_payload_size = np.asarray(
                        detail["payload_aabb_size_m"], dtype=float
                    )
                except (KeyError, TypeError, ValueError):
                    return {
                        "ok": False,
                        "reason": (
                            "payload overlap lacks metric payload geometry"
                        ),
                        "collision": detail,
                        "allowed_initial_obstacles": sorted(discovered),
                    }
                current_rotation_raw = detail.get("payload_rotation_matrix")
                current_payload_rotation = None
                if current_rotation_raw is not None:
                    current_payload_rotation = np.asarray(
                        current_rotation_raw, dtype=float
                    )
                    if current_payload_rotation.shape != (3, 3) \
                            or not np.all(np.isfinite(
                                current_payload_rotation
                            )):
                        return {
                            "ok": False,
                            "reason": (
                                "payload overlap has invalid orientation "
                                "geometry"
                            ),
                            "collision": detail,
                            "allowed_initial_obstacles": sorted(discovered),
                        }
                if payload_center_m is None:
                    payload_center_m = current_payload_center
                    payload_aabb_size_m = current_payload_size
                    payload_rotation = current_payload_rotation
                elif not np.allclose(
                    current_payload_center, payload_center_m,
                    rtol=1e-6, atol=1e-6,
                ) or not np.allclose(
                    current_payload_size, payload_aabb_size_m,
                    rtol=1e-6, atol=1e-6,
                ) or (
                    payload_rotation is not None
                    and current_payload_rotation is not None
                    and not np.allclose(
                        current_payload_rotation, payload_rotation,
                        rtol=1e-6, atol=1e-6,
                    )
                ):
                    return {
                        "ok": False,
                        "reason": "payload geometry changed during overlap discovery",
                        "collision": detail,
                        "allowed_initial_obstacles": sorted(discovered),
                    }
                discovered.add(obstacle)
            if discovered and payload_rotation is None:
                self._payload_escape_obstacles = set()
                self._payload_escape_reference = None
                return {
                    "ok": False,
                    "reason": (
                        "payload overlap lacks the orientation required to "
                        "enforce a fixed-orientation vertical escape"
                    ),
                    "allowed_initial_obstacles": sorted(discovered),
                }
            exact_initial_obstacles = set(discovered)
            reservation_limits = self._payload_escape_scope_limits(
                payload_aabb_size_m
                if payload_aabb_size_m is not None else np.zeros(3)
            )
            guarded_payload_size = (
                np.asarray(payload_aabb_size_m, dtype=float).copy()
                if payload_aabb_size_m is not None else np.ones(3)
            )
            reserved_near_distances: dict[str, float] = {}
            reservation_distance = (
                math.hypot(
                    reservation_limits["xy_tolerance_m"],
                    reservation_limits["downward_tolerance_m"],
                ) + reservation_limits["rotation_shell_m"]
            )
            if exact_initial_obstacles:
                # Query the actual oriented payload against every scene box.
                # A raw world-AABB shell is much too loose for a long tilted
                # object. The exact FCL distance tube is sized from the same
                # translation/rotation residuals enforced during execution, so
                # a pose inside that scope cannot create an unreserved initial
                # contact at segment zero.
                nearby = self._payload_scene_proximity(
                    payload_center_m, payload_rotation, reservation_distance
                )
                discovered.update(nearby)
                reserved_near_distances = {
                    name: round(float(distance), 6)
                    for name, distance in nearby.items()
                    if name not in exact_initial_obstacles
                }
                # The helper uses AABBs only for cheap sanity checks. Expand
                # that diagnostic envelope by the exact distance-tube radius;
                # candidate membership itself came solely from OBB/FCL.
                guarded_payload_size += 2.0 * reservation_distance
            reserved_near_obstacles = discovered - exact_initial_obstacles
            cluster = self._payload_escape_cluster_geometry(
                discovered,
                payload_center_m if payload_center_m is not None else [0.0] * 3,
                guarded_payload_size,
                measured_payload_aabb_size_m=payload_aabb_size_m,
                strict_obstacle_names=exact_initial_obstacles,
            )
            if not cluster.get("ok"):
                self._payload_escape_obstacles = set()
                self._payload_escape_reference = None
                return cluster
            # The reservation tube may have surfaced boxes the payload does not
            # touch. Only the names the cluster actually retained may have their
            # collisions excepted; guarding a distant box would disable a real
            # obstacle for the whole extraction.
            retained_names = set(cluster.get("allowed_initial_obstacles") or [])
            if retained_names:
                dropped = discovered - retained_names
                discovered = discovered & retained_names
                reserved_near_obstacles = (
                    reserved_near_obstacles & retained_names
                )
                reserved_near_distances = {
                    name: distance
                    for name, distance in reserved_near_distances.items()
                    if name in retained_names
                }
                if dropped:
                    cluster = {
                        **cluster,
                        "dropped_far_obstacles": sorted(dropped),
                    }
            self._payload_escape_obstacles = set(discovered)
            self._payload_escape_reference = (
                {
                    "payload_center_m": np.asarray(
                        payload_center_m, dtype=float
                    ).copy(),
                    "payload_rotation": np.asarray(
                        payload_rotation, dtype=float
                    ).copy(),
                    "payload_aabb_size_m": np.asarray(
                        payload_aabb_size_m, dtype=float
                    ).copy(),
                }
                if discovered and payload_center_m is not None else None
            )
            if discovered:
                self._collision_scene_version += 1
            return {
                "ok": True,
                "active": bool(discovered),
                "allowed_initial_obstacles": sorted(discovered),
                "exact_initial_obstacles": sorted(exact_initial_obstacles),
                "reserved_near_obstacles": sorted(reserved_near_obstacles),
                "reserved_near_distances_m": reserved_near_distances,
                "reservation_distance_m": round(reservation_distance, 6),
                "reservation_xy_shell_m": round(
                    reservation_limits["reservation_xy_shell_m"], 6
                ),
                "reservation_z_shell_m": round(
                    reservation_limits["reservation_z_shell_m"], 6
                ),
                "minimum_vertical_clearance_m": float(
                    cluster.get("minimum_vertical_clearance_m", 0.0)
                ),
                "reason": (
                    f"bounded fixed-XY vertical payload escape enabled for "
                    f"the guarded initial support envelope: "
                    f"{', '.join(sorted(discovered))}"
                    if discovered else "payload grasp pose is already collision-free"
                ),
                "cluster": cluster,
                "scene_version": self._collision_scene_version,
            }

    def validate_payload_escape_endpoint(
        self, x: float, y: float, z: float,
        roll: float, pitch: float, yaw: float,
    ) -> dict[str, Any]:
        """Require the extraction endpoint to clear every scene obstacle."""
        with self._lock:
            try:
                solution = self.solve_ik(x, y, z, roll, pitch, yaw)
            except Exception as exc:
                return {"ok": False, "reason": f"endpoint IK failed: {exc}"}
            detail = self._configuration_collision(
                np.asarray(solution["joints_rad"], dtype=float),
                ignored_payload_obstacles=set(),
            )
            if detail.get("collision"):
                return {
                    "ok": False,
                    "reason": "vertical escape endpoint is not strictly clear",
                    "collision": detail,
                }
            return {"ok": True, "reason": "vertical escape endpoint is strictly clear"}

    def end_payload_vertical_escape(self) -> dict[str, Any]:
        """Restore the full collision scene and verify the measured pose."""
        with self._lock:
            was_active = bool(self._payload_escape_obstacles)
            previous = sorted(self._payload_escape_obstacles)
            self._payload_escape_obstacles.clear()
            self._payload_escape_reference = None
            if was_active:
                self._collision_scene_version += 1
            q6 = np.asarray(self.state()["joints_rad"], dtype=float)
            detail = self._configuration_collision(
                q6, ignored_payload_obstacles=set()
            )
            if detail.get("collision"):
                return {
                    "ok": False,
                    "reason": "payload escape ended but measured pose is not clear",
                    "allowed_initial_obstacles": previous,
                    "collision": detail,
                }
            return {
                "ok": True,
                "reason": "full payload collision checking restored",
                "allowed_initial_obstacles": previous,
                "scene_version": self._collision_scene_version,
            }

    def _assert_collision_free_joint_path(
        self, joint_points: list[np.ndarray], *, label: str
    ) -> dict[str, Any]:
        if getattr(self, "_collision_model", None) is None or not joint_points:
            return {"ok": True, "checked_configurations": 0,
                    "minimum_clearance_m": None}
        checked = 0
        minimum = float("inf")
        previous = np.asarray(joint_points[0], dtype=float)
        for segment_index, target_raw in enumerate(joint_points):
            target = np.asarray(target_raw, dtype=float)
            max_step = float(np.max(np.abs(target - previous)))
            samples = max(1, int(np.ceil(max_step / 0.035)))
            for sample_index in range(1 if segment_index else 0, samples + 1):
                fraction = sample_index / samples
                candidate = previous + (target - previous) * fraction
                detail = self._configuration_collision(candidate)
                checked += 1
                clearance = detail.get("minimum_clearance_m")
                if clearance is not None:
                    minimum = min(minimum, float(clearance))
                if detail.get("collision"):
                    raise RuntimeError(
                        f"{label} collision at segment {segment_index}, "
                        f"sample {sample_index}/{samples}: {detail}"
                    )
            previous = target
        return {
            "ok": True,
            "checked_configurations": checked,
            "minimum_clearance_m": None if not np.isfinite(minimum) else minimum,
        }

    @staticmethod
    def _angle_error(value: float, reference: float) -> float:
        return float((value - reference + np.pi) % (2.0 * np.pi) - np.pi)

    def configuration_valid(
        self,
        q6: Any,
        *,
        workspace_lo: Any | None = None,
        workspace_hi: Any | None = None,
        orientation_ref_rpy: Any | None = None,
        max_tilt_rad: float | None = None,
    ) -> dict[str, Any]:
        """Public read-only validity gate used by sampling-based planners."""
        q = np.asarray(q6, dtype=np.float64)
        if q.shape != (6,) or not np.all(np.isfinite(q)):
            return {"ok": False, "reason": "invalid six-joint configuration"}
        if np.any(q < self._joint_lo) or np.any(q > self._joint_hi):
            return {"ok": False, "reason": "URDF joint hard limit"}
        position, rpy = joint_to_pose(q)
        position = np.asarray(position, dtype=float)
        rpy = np.asarray(rpy, dtype=float)
        if workspace_lo is not None and np.any(
            position < np.asarray(workspace_lo, dtype=float)
        ):
            return {"ok": False, "reason": "end effector below workspace bounds"}
        if workspace_hi is not None and np.any(
            position > np.asarray(workspace_hi, dtype=float)
        ):
            return {"ok": False, "reason": "end effector above workspace bounds"}
        if orientation_ref_rpy is not None and max_tilt_rad is not None:
            reference = np.asarray(orientation_ref_rpy, dtype=float)
            # Roll and pitch constrain payload tilt. Yaw is deliberately free so
            # the planner can route around clutter without tipping the object.
            tilt = max(
                abs(self._angle_error(float(rpy[0]), float(reference[0]))),
                abs(self._angle_error(float(rpy[1]), float(reference[1]))),
            )
            if tilt > float(max_tilt_rad):
                return {"ok": False, "reason": "payload upright constraint"}
        collision = self._configuration_collision(q)
        if collision.get("collision"):
            return {"ok": False, "reason": self._collision_reason(collision),
                    "collision": collision}
        return {
            "ok": True,
            "position_m": position.tolist(),
            "rpy_rad": rpy.tolist(),
            "minimum_clearance_m": collision.get("minimum_clearance_m"),
        }

    @staticmethod
    def _collision_reason(detail: dict[str, Any]) -> str:
        """Keep the exact collider visible at every public planning gate."""
        kind = str(detail.get("kind") or "unknown")
        if kind == "payload":
            return f"payload collision with {detail.get('obstacle', 'unknown obstacle')}"
        if kind == "environment":
            box_detail = ""
            if detail.get("obstacle_center_m") is not None:
                box_detail = (
                    f" center={detail.get('obstacle_center_m')}m"
                    f" size={detail.get('obstacle_size_m')}m"
                )
            return (
                f"environment collision: {detail.get('robot_link', 'robot')} "
                f"with {detail.get('obstacle', 'unknown obstacle')}"
                f"{box_detail}"
            )
        if kind == "self":
            return (
                f"self collision: {detail.get('first', 'robot')} with "
                f"{detail.get('second', 'robot')}"
            )
        return f"{kind} collision"

    def validate_joint_path(
        self,
        joint_path: Any,
        *,
        scene_version: int | None = None,
        max_step_rad: float = 0.03,
        workspace_lo: Any | None = None,
        workspace_hi: Any | None = None,
        orientation_ref_rpy: Any | None = None,
        max_tilt_rad: float | None = None,
    ) -> dict[str, Any]:
        """Densely certify a joint polyline against the current scene."""
        if scene_version is not None and int(scene_version) != self.collision_scene_version:
            return {
                "ok": False,
                "reason": f"stale scene version {scene_version}; current is "
                          f"{self.collision_scene_version}",
            }
        points = [np.asarray(point, dtype=float) for point in joint_path]
        if not points:
            return {"ok": False, "reason": "empty joint path"}
        checked = 0
        minimum = float("inf")
        previous = points[0]
        for segment_index, target in enumerate(points):
            if target.shape != (6,):
                return {"ok": False, "reason": "joint path point is not length six"}
            samples = max(1, int(np.ceil(
                float(np.max(np.abs(target - previous))) /
                max(0.005, float(max_step_rad))
            )))
            for sample_index in range(0 if segment_index == 0 else 1, samples + 1):
                candidate = previous + (target - previous) * (sample_index / samples)
                detail = self.configuration_valid(
                    candidate,
                    workspace_lo=workspace_lo,
                    workspace_hi=workspace_hi,
                    orientation_ref_rpy=orientation_ref_rpy,
                    max_tilt_rad=max_tilt_rad,
                )
                checked += 1
                if not detail.get("ok"):
                    return {
                        "ok": False,
                        "reason": f"joint path invalid at segment {segment_index}, "
                                  f"sample {sample_index}/{samples}: "
                                  f"{detail.get('reason')}",
                        "detail": detail,
                    }
                clearance = detail.get("minimum_clearance_m")
                if clearance is not None:
                    minimum = min(minimum, float(clearance))
            previous = target
        return {
            "ok": True,
            "checked_configurations": checked,
            "minimum_clearance_m": None if not np.isfinite(minimum) else minimum,
            "scene_version": self.collision_scene_version,
        }

    def _solve_pose_from_seed(
        self, pose: tuple[float, float, float, float, float, float], q_seed6: Any
    ) -> np.ndarray:
        x, y, z, roll, pitch, yaw = pose
        seed = pad_q_for_model(
            self._model, np.asarray(q_seed6, dtype=float), controlled_joints=6
        )
        target = pos_rot_to_se3(
            np.array([x, y, z], dtype=float), roll=roll, pitch=pitch, yaw=yaw
        )
        result = solve_ik_with_retry(
            self._model, self._data, self._end_frame, target, seed,
            self._ik, max_retries=0,
        )
        q = np.asarray(result.q[:6], dtype=float)
        actual_pos, actual_rpy = joint_to_pose(result.q)
        pos_error = float(np.linalg.norm(np.asarray(actual_pos) - [x, y, z]))
        rpy_error = float(np.max(np.abs(
            [self._angle_error(a, b) for a, b in zip(actual_rpy, [roll, pitch, yaw])]
        )))
        if not np.all(np.isfinite(q)) or np.any(q < self._joint_lo) or np.any(q > self._joint_hi):
            raise RuntimeError("IK goal violates a joint hard limit")
        if not result.success and not (
            pos_error <= LOCAL_IK_POS_TOL_M and rpy_error <= LOCAL_IK_RPY_TOL_RAD
        ):
            raise RuntimeError(
                f"IK failed (position={pos_error * 1000:.1f}mm, "
                f"orientation={rpy_error:.4f}rad)"
            )
        return q

    @staticmethod
    def _plan_certificate(scene_version: int, start: np.ndarray,
                          path: list[np.ndarray]) -> str:
        digest = hashlib.sha256()
        digest.update(str(int(scene_version)).encode("ascii"))
        digest.update(np.asarray(start, dtype="<f8").tobytes())
        digest.update(np.asarray(path, dtype="<f8").tobytes())
        return digest.hexdigest()

    def plan_joint_motion(
        self,
        terminal_poses: list[tuple[float, float, float, float, float, float]],
        *,
        workspace_lo: Any,
        workspace_hi: Any,
        orientation_ref_rpy: Any | None = None,
        max_tilt_rad: float | None = None,
        timeout_s: float = 4.0,
        max_nodes: int = 4000,
        seed: int = 0,
    ) -> dict[str, Any]:
        """Plan a global joint route followed by exact Cartesian terminal poses."""
        if not terminal_poses:
            return {"ok": False, "reason": "no terminal pose supplied"}
        from .joint_planner import PlannerSettings, plan_rrt_connect

        with self._lock:
            arm, _ = self._require()
            q_live, _, _ = arm.get_state()
            start = np.asarray(q_live[:6], dtype=float)
            version = self.collision_scene_version
            try:
                goal = self._solve_pose_from_seed(terminal_poses[0], start)
            except Exception as exc:
                return {"ok": False, "reason": f"global goal IK failed: {exc}"}

            validity_kwargs = dict(
                workspace_lo=workspace_lo, workspace_hi=workspace_hi,
                orientation_ref_rpy=orientation_ref_rpy,
                max_tilt_rad=max_tilt_rad,
            )
            state_valid = lambda q: bool(
                self.configuration_valid(q, **validity_kwargs).get("ok")
            )
            result = plan_rrt_connect(
                start, goal, self._joint_lo, self._joint_hi, state_valid,
                settings=PlannerSettings(
                    timeout_s=float(np.clip(timeout_s, 0.2, 12.0)),
                    max_nodes=int(np.clip(max_nodes, 100, 12000)),
                    seed=int(seed),
                ),
            )
            if not result.ok:
                return result.as_dict() | {"scene_version": version}
            path = [np.asarray(point, dtype=float) for point in result.path]
            previous = path[-1]
            try:
                for pose in terminal_poses[1:]:
                    target = self._solve_pose_from_seed(pose, previous)
                    if float(np.max(np.abs(target - previous))) > PATH_IK_MAX_JOINT_STEP_RAD:
                        raise RuntimeError("terminal Cartesian continuation changed IK branch")
                    path.append(target)
                    previous = target
            except Exception as exc:
                return {"ok": False, "reason": f"terminal path IK failed: {exc}",
                        "scene_version": version}
            certified = self.validate_joint_path(
                path, scene_version=version, **validity_kwargs
            )
            if not certified.get("ok"):
                return {"ok": False, "reason": certified.get("reason"),
                        "scene_version": version}
            return {
                "ok": True,
                "kind": "certified_joint_plan",
                "joint_path": [point.tolist() for point in path],
                "start_joints_rad": start.tolist(),
                "scene_version": version,
                "certificate": self._plan_certificate(version, start, path),
                "created_monotonic_s": time.monotonic(),
                "validation": certified,
                "planner": result.as_dict(),
                "constraints": {
                    "workspace_lo": np.asarray(workspace_lo, dtype=float).tolist(),
                    "workspace_hi": np.asarray(workspace_hi, dtype=float).tolist(),
                    "orientation_ref_rpy": None if orientation_ref_rpy is None else
                        np.asarray(orientation_ref_rpy, dtype=float).tolist(),
                    "max_tilt_rad": max_tilt_rad,
                },
            }

    def execute_joint_plan(
        self, plan: dict[str, Any], *, duration: float = 0.0,
        send_freq: float = 50.0,
    ) -> dict[str, Any]:
        """Revalidate a certified plan, then execute one minimum-jerk trajectory."""
        MC.motion()
        with self._lock:
            arm, controller = self._require()
            version = int(plan.get("scene_version", -1))
            if version != self.collision_scene_version:
                raise RuntimeError(
                    f"stale motion plan: scene {version}, current "
                    f"{self.collision_scene_version}"
                )
            created = float(plan.get("created_monotonic_s", 0.0))
            max_age = float(np.clip(
                float(os.getenv("REBOT_GLOBAL_PLAN_MAX_AGE_S", "30")), 5.0, 120.0
            ))
            if created <= 0.0 or time.monotonic() - created > max_age:
                raise RuntimeError(
                    f"motion plan is older than the {max_age:.0f}s freshness limit"
                )
            path = [np.asarray(point, dtype=float) for point in plan.get("joint_path", [])]
            planned_start = np.asarray(plan.get("start_joints_rad", []), dtype=float)
            expected = self._plan_certificate(version, planned_start, path)
            if plan.get("certificate") != expected:
                raise RuntimeError("motion plan certificate mismatch")
            q_live, _, _ = arm.get_state()
            q_live = np.asarray(q_live[:6], dtype=float)
            if planned_start.shape != (6,) or float(np.max(np.abs(q_live - planned_start))) > 0.04:
                raise RuntimeError("arm moved since planning; refusing stale start state")
            constraints = dict(plan.get("constraints") or {})
            certified = self.validate_joint_path(
                [q_live, *path], scene_version=version, **constraints
            )
            if not certified.get("ok"):
                raise RuntimeError(f"pre-execution certification failed: {certified.get('reason')}")
            points = [q_live, *path]
            lengths = np.asarray([
                float(np.max(np.abs(b - a))) for a, b in zip(points[:-1], points[1:])
            ])
            cumulative = np.concatenate(([0.0], np.cumsum(lengths)))
            path_length = float(cumulative[-1])
            if path_length > 1e-8:
                total_s = max(float(duration), 1.875 * path_length / max(1e-6, _joint_vlim()), 0.25)
                count = max(2, int(np.ceil(total_s * max(20.0, send_freq))))
                for index in range(1, count + 1):
                    MC.checkpoint()
                    t = index / count
                    distance = (10*t**3 - 15*t**4 + 6*t**5) * path_length
                    segment = min(len(lengths) - 1, int(
                        np.searchsorted(cumulative, distance, side="right") - 1
                    ))
                    fraction = 1.0 if lengths[segment] < 1e-10 else float(np.clip(
                        (distance - cumulative[segment]) / lengths[segment], 0.0, 1.0
                    ))
                    controller._q_target[:] = points[segment] + (
                        points[segment + 1] - points[segment]
                    ) * fraction
                    MC.pause(total_s / count)
                controller._q_target[:] = path[-1]
        deadline = time.monotonic() + 3.0
        state = self.state()
        target = path[-1]
        while time.monotonic() < deadline:
            MC.checkpoint()
            joint_error = max(abs(a - b) for a, b in zip(
                state["joints_rad"], target
            ))
            velocity = max((abs(float(value)) for value in
                            state.get("joint_vel_rad_s", [])), default=float("inf"))
            if joint_error < 0.02 and velocity < 0.10:
                return state
            time.sleep(0.05)
            state = self.state()
        raise RuntimeError("global joint route endpoint did not settle")

    @property
    def connected(self) -> bool:
        return self._arm is not None and self._controller is not None

    def wait_for_stationary_hold(
        self, target_joints: Any | None = None, *, timeout_s: float = 3.0,
        velocity_limit_rad_s: float = 0.10,
        encoder_delta_limit_rad: float = 0.0025,
        consecutive_samples: int = 3,
        sample_interval_s: float = 0.05,
    ) -> dict[str, Any]:
        """Verify a held endpoint using velocity and independent encoders.

        RobStride velocity feedback can remain noisy after the measured joints
        and tool pose have stopped.  Accept either repeated low velocity or a
        stricter run of nearly unchanged encoder samples.  The latter never
        accepts a moving arm: the controller target must remain within 0.02 rad
        and every consecutive encoder delta must stay below 0.0025 rad.
        """
        target = None if target_joints is None else np.asarray(
            target_joints, dtype=float
        )[:6]
        deadline = time.monotonic() + max(0.0, float(timeout_s))
        required = max(2, int(consecutive_samples))
        previous_q: np.ndarray | None = None
        velocity_streak = 0
        encoder_streak = 0
        state = self.state()
        while True:
            MC.checkpoint()
            q = np.asarray(state.get("joints_rad", []), dtype=float)[:6]
            velocities = np.asarray(
                state.get("joint_vel_rad_s", []), dtype=float
            )[:6]
            finite = bool(
                q.shape == (6,) and velocities.shape == (6,)
                and np.all(np.isfinite(q)) and np.all(np.isfinite(velocities))
            )
            target_error = (
                float(np.max(np.abs(q - target)))
                if finite and target is not None and target.shape == (6,)
                else (0.0 if finite and target is None else float("inf"))
            )
            target_held = target_error < 0.02
            max_velocity = (
                float(np.max(np.abs(velocities))) if finite else float("inf")
            )
            velocity_streak = (
                velocity_streak + 1
                if target_held and max_velocity <= velocity_limit_rad_s else 0
            )
            encoder_delta = (
                float(np.max(np.abs(q - previous_q)))
                if finite and previous_q is not None else float("inf")
            )
            encoder_streak = (
                encoder_streak + 1
                if target_held and encoder_delta <= encoder_delta_limit_rad
                else 0
            )
            method = None
            if velocity_streak >= 2:
                method = "velocity_feedback"
            elif encoder_streak >= required:
                method = "stable_joint_encoders"
            if method is not None:
                return {
                    **state,
                    "settlement": {
                        "ok": True,
                        "method": method,
                        "max_joint_velocity_rad_s": max_velocity,
                        "max_encoder_delta_rad": (
                            None if not np.isfinite(encoder_delta)
                            else encoder_delta
                        ),
                        "max_target_error_rad": target_error,
                    },
                }
            if time.monotonic() >= deadline:
                return {
                    **state,
                    "settlement": {
                        "ok": False,
                        "method": "timeout",
                        "max_joint_velocity_rad_s": max_velocity,
                        "max_encoder_delta_rad": (
                            None if not np.isfinite(encoder_delta)
                            else encoder_delta
                        ),
                        "max_target_error_rad": target_error,
                    },
                }
            previous_q = q.copy() if finite else None
            MC.pause(max(0.01, float(sample_interval_s)))
            state = self.state()

    @staticmethod
    def _verify_can_bitrate(can_iface: str) -> int:
        """Refuse an interface configured differently from this RS station."""
        expected = int(os.getenv("REBOT_CAN_BITRATE", "1000000"))
        try:
            result = subprocess.run(
                ["ip", "-details", "link", "show", can_iface],
                check=True, capture_output=True, text=True, timeout=2.0,
            )
        except Exception as exc:
            raise RuntimeError(
                f"cannot verify CAN bitrate for {can_iface!r}: {exc}"
            ) from exc
        match = re.search(r"\bbitrate\s+(\d+)\b", result.stdout)
        if match is None:
            raise RuntimeError(
                f"CAN interface {can_iface!r} does not report a bitrate"
            )
        actual = int(match.group(1))
        if actual != expected:
            raise RuntimeError(
                f"CAN interface {can_iface!r} is configured at {actual} bit/s; "
                f"this RS station requires {expected} bit/s. Refusing to send "
                "motor traffic at the wrong bitrate"
            )
        return actual

    @staticmethod
    def _verified_feedback(
        arm: RebotArm,
        *,
        attempts: int = 10,
        interval_s: float = 0.05,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Require a real, fault-free reply from every configured motor.

        ``RebotArm.get_state`` substitutes zero for a missing motor state. Those
        zeros are useful for display but must never seed or authorize motion.
        Inspect the underlying per-motor reply objects before accepting a state.
        """
        last_missing = list(arm.joint_names)
        last_faults: list[str] = []
        state = (np.zeros(len(last_missing)),) * 3
        for _ in range(max(1, attempts)):
            state = arm.get_state()
            last_missing = []
            last_faults = []
            for name in arm.joint_names:
                motor_state = arm._motor_map[name].get_state()
                if motor_state is None:
                    last_missing.append(name)
                elif int(motor_state.status_code) != 0:
                    last_faults.append(
                        f"{name}=fault:{int(motor_state.status_code)}"
                    )
            arrays = tuple(np.asarray(values, dtype=np.float64)
                           for values in state)
            if not last_missing and not last_faults and all(
                np.all(np.isfinite(values)) for values in arrays
            ):
                return arrays  # type: ignore[return-value]
            time.sleep(max(0.0, interval_s))
        detail = []
        if last_missing:
            detail.append("no feedback from " + ", ".join(last_missing))
        if last_faults:
            detail.append("motor faults " + ", ".join(last_faults))
        if not detail:
            detail.append("non-finite motor state")
        raise RuntimeError(
            "motor feedback unavailable; refusing to enable or move ("
            + "; ".join(detail) + ")"
        )

    @staticmethod
    def _enable_feedback_wake(arm: RebotArm) -> None:
        """Wake RS feedback without starting a controller or sending a target.

        A RobStride drive can begin returning CAN frames just after the native
        enable acknowledgement window closes, especially after an unclean
        process exit or controller power cycle.  Recreating the entire arm
        object for every CLI retry repeatedly hits that same first-window race.
        Retry the acknowledgement in the existing CAN session, but never infer
        success merely from traffic: one complete ``enable`` call must still
        explicitly acknowledge before position control may start.
        """
        arm_group = arm.groups.get("arm")
        if arm_group is None:
            raise RuntimeError(
                "arm motor group is unavailable; refusing to start position "
                "control"
            )
        attempts = int(np.clip(int(os.getenv(
            "REBOT_ENABLE_ACK_ATTEMPTS", "3"
        )), 1, 5))
        interval_s = float(np.clip(float(os.getenv(
            "REBOT_ENABLE_ACK_RETRY_S", "0.15"
        )), 0.05, 0.50))
        for attempt in range(1, attempts + 1):
            if arm_group.enable():
                if attempt > 1:
                    print(
                        f"[arm/enable] acknowledgement recovered on "
                        f"attempt {attempt}/{attempts}"
                    )
                return
            if attempt < attempts:
                print(
                    f"[arm/enable] acknowledgement missing on attempt "
                    f"{attempt}/{attempts}; retrying without sending a pose "
                    "target"
                )
                time.sleep(interval_s)
        raise RuntimeError(
            "motor enable acknowledgement failed after "
            f"{attempts} attempts while waking feedback; refusing to start "
            "position control"
        )

    def connect(self) -> None:
        MC.motion()
        with self._lock:
            if self.connected:
                return
            can_iface = os.getenv("REBOT_CAN_IFACE", "can0")
            operstate = Path(f"/sys/class/net/{can_iface}/operstate")
            try:
                is_up = operstate.read_text().strip().lower() == "up"
            except OSError as exc:
                raise RuntimeError(f"CAN interface {can_iface!r} is unavailable") from exc
            if not is_up:
                raise RuntimeError(
                    f"CAN interface {can_iface!r} is down; bring it up before connecting the arm"
                )
            self._verify_can_bitrate(can_iface)
            cfg = ROOT / "reBotArm_control_py_fresh" / "config" / "rebotarm_rs.yaml"
            arm = RebotArm(str(cfg))
            # Read the physical pose before enabling the position controller.  A
            # fresh controller otherwise starts with a zero joint target, which is
            # unsafe for a diagnostic connection when the arm is parked elsewhere.
            arm.connect()
            enabled_for_feedback = False
            try:
                try:
                    # Some RS drives answer while disabled, so prefer the fully
                    # passive read first.
                    q, _, _ = self._verified_feedback(
                        arm, attempts=3, interval_s=0.05
                    )
                except RuntimeError:
                    # RobStride feedback is not reliable while disabled. Enabling
                    # sends no pose target; it only wakes the drives so their
                    # measured state can be read. Position control is not started
                    # until that measured pose has been latched below.
                    enabled_for_feedback = True
                    self._enable_feedback_wake(arm)
                    q, _, _ = self._verified_feedback(arm)
            except BaseException:
                # No pose target or control loop has started. If the feedback wake
                # succeeded partially, make one bounded disable attempt before
                # closing the transport.
                if enabled_for_feedback:
                    try:
                        arm.disable_all()
                    except BaseException:
                        pass
                for ctrl in arm._ctrl_map.values():
                    try:
                        ctrl.shutdown()
                    except BaseException:
                        pass
                    try:
                        ctrl.close()
                    except BaseException:
                        pass
                raise
            controller = RebotArmEndPose(arm)
            controller._q_target = np.asarray(q[:6], dtype=np.float64).copy()
            if len(q) > 6:
                controller._gripper_target = float(q[6])
            try:
                controller.start()
            except BaseException:
                try:
                    arm.stop_control_loop()
                except BaseException:
                    pass
                for ctrl in arm._ctrl_map.values():
                    try:
                        ctrl.shutdown()
                    except BaseException:
                        pass
                    try:
                        ctrl.close()
                    except BaseException:
                        pass
                raise
            # Keep local visual-servo moves deliberately slow even if the hardware
            # YAML has higher per-joint limits. Sourced from arm_safe so the joint
            # ceiling and the cartesian pace are tuned from one place.
            controller._vlim_override = np.full(6, _joint_vlim(), dtype=np.float64)
            self._arm, self._controller = arm, controller

    def freeze(self) -> dict[str, Any]:
        """Cancel trajectory generation and hold the measured joint pose.

        This is the first response to every software failure. It does not disable
        any motor and therefore does not intentionally release the arm to gravity.
        """
        with self._lock:
            arm, controller = self._require()
            controller._stop_send.set()
            sender = controller._send_thread
            if sender is not None and sender.is_alive():
                sender.join(timeout=2.0)
            q, _, _ = arm.get_state()
            held = np.asarray(q[:6], dtype=np.float64).copy()
            controller._q_target = held
            controller._qd_target = np.zeros_like(held)
            controller._moving = False
        state = self.wait_for_stationary_hold(
            held, timeout_s=2.0, consecutive_samples=3
        )
        settlement = dict(state.get("settlement") or {})
        stopped = bool(settlement.get("ok"))
        return {
            "ok": stopped,
            "reason": (
                "measured pose latched; motor control remains enabled; "
                f"settled by {settlement.get('method')}"
                if stopped else
                "pose target latched but neither velocity nor joint encoders "
                "verified a stationary hold"
            ),
            "max_joint_velocity_rad_s": settlement.get(
                "max_joint_velocity_rad_s", float("inf")
            ),
            "settlement": settlement,
            "state": state,
        }

    def disconnect(self, *, require_home: bool = True) -> dict[str, Any]:
        """Park before disabling. Refuse an elevated/unverified disconnect.

        The old implementation unconditionally stopped the control loop and called
        ``disable_all``. On a planning failure at the scan pose that removed
        holding torque and the arm fell. Disabling is now reachable only after a
        verified home state.
        """
        with self._lock:
            if not self.connected:
                return {"ok": True, "reason": "already disconnected", "homed": True}

        parked: dict[str, Any] | None = None
        if require_home:
            try:
                frozen = self.freeze()
                parked = self.home() if frozen.get("ok") else {
                    "ok": False,
                    "reason": f"freeze_not_confirmed: {frozen.get('reason')}",
                }
            except BaseException as exc:  # keep control alive; do not disable
                parked = {"ok": False, "reason": f"park_exception {exc}"}
            if not parked.get("ok"):
                message = (
                    "CRITICAL: home was not verified; refusing motor disable. "
                    "The controller is still holding the last measured pose. "
                    f"Reason: {parked.get('reason')}"
                )
                print(message, file=sys.stderr, flush=True)
                return {"ok": False, "reason": message, "homed": False,
                        "holding": True, "home": parked}

        with self._lock:
            arm, controller = self._arm, self._controller
            self._arm = self._controller = None
        try:
            # RebotArm.disconnect owns orderly loop stop, motor disable, and CAN
            # teardown. It is safe here because home was independently verified.
            assert arm is not None
            arm.disconnect()
        except BaseException as exc:
            return {"ok": False, "reason": f"disconnect_error {exc}",
                    "homed": bool(parked and parked.get("ok"))}
        return {"ok": True, "reason": "home verified, then motors disabled",
                "homed": bool(parked and parked.get("ok")) if require_home else None}

    def _require(self) -> tuple[RebotArm, RebotArmEndPose]:
        if self._arm is None or self._controller is None:
            raise RuntimeError("RS arm is not connected")
        return self._arm, self._controller

    def state(self) -> dict[str, Any]:
        with self._lock:
            arm, _ = self._require()
            q, vel, torque = arm.get_state()
            q6 = np.asarray(q[:6], dtype=np.float64)
            pos, rpy = joint_to_pose(q6)
            return {"connected": True, "joints_rad": q6.tolist(),
                    "joint_torque": np.asarray(torque[:6], dtype=float).tolist(),
                    "joint_vel_rad_s": np.asarray(vel[:6], dtype=float).tolist(),
                    "gripper_position_rad": float(q[6]) if len(q) > 6 else None,
                    "end_effector": {"position_m": [float(v) for v in pos],
                                     "rpy_rad": [float(v) for v in rpy]}}

    def solve_ik(self, x: float, y: float, z: float, roll: float, pitch: float, yaw: float) -> dict[str, Any]:
        with self._lock:
            if self.connected:
                q_seed, _, _ = self._require()[0].get_state()
                q_seed = np.asarray(q_seed[:6], dtype=np.float64)
            else:
                q_seed = np.zeros(6, dtype=np.float64)
            q_seed = pad_q_for_model(self._model, q_seed, controlled_joints=6)
            target = pos_rot_to_se3(np.array([x, y, z], dtype=np.float64), roll=roll, pitch=pitch, yaw=yaw)
            # Never use the helper's random fallback on live hardware.  A random
            # seed can find a mathematically valid but physically distant elbow
            # branch; local motion must either converge from the current joints or
            # be refused by the caller.
            result = solve_ik_with_retry(self._model, self._data, self._end_frame, target, q_seed,
                                         self._ik, max_retries=0)
            target_q = np.asarray(result.q[:6], dtype=np.float64)
            if not np.all(np.isfinite(target_q)):
                raise RuntimeError("IK produced non-finite joint values")
            if np.any(target_q < self._joint_lo) or np.any(target_q > self._joint_hi):
                raise RuntimeError("IK solution violates a URDF joint hard limit")
            pos, rpy = joint_to_pose(result.q)
            pos_error = float(np.linalg.norm(np.asarray(pos) - np.array([x, y, z])))
            rpy_error = float(np.max(np.abs(np.asarray(rpy) - np.array([roll, pitch, yaw]))))
            if not result.success and not (
                pos_error <= LOCAL_IK_POS_TOL_M and rpy_error <= LOCAL_IK_RPY_TOL_RAD
            ):
                raise RuntimeError(
                    f"IK failed (error={result.error:.3e}, position={pos_error*1000:.1f}mm, "
                    f"orientation={rpy_error:.4f}rad)"
                )
            return {"joints_rad": target_q.tolist(),
                    "end_effector": {"position_m": [float(v) for v in pos],
                                     "rpy_rad": [float(v) for v in rpy]},
                    "near_solution": not result.success,
                    "position_error_m": pos_error,
                    "orientation_error_rad": rpy_error}

    def validate_pose_configuration(
        self, x: float, y: float, z: float,
        roll: float, pitch: float, yaw: float,
    ) -> dict[str, Any]:
        """Read-only IK plus collision check for one proposed endpoint."""
        try:
            solution = self.solve_ik(x, y, z, roll, pitch, yaw)
        except Exception as exc:
            return {"ok": False, "reason": f"endpoint IK failed: {exc}"}
        detail = self.configuration_valid(solution["joints_rad"])
        if not detail.get("ok"):
            return {
                "ok": False,
                "reason": f"endpoint configuration invalid: {detail.get('reason')}",
                "detail": detail,
            }
        return {"ok": True, "reason": "endpoint IK and collision check passed",
                "detail": detail, "joints_rad": solution["joints_rad"]}

    def solve_ik_path(
        self,
        poses: list[tuple[float, float, float, float, float, float]],
    ) -> list[dict[str, Any]]:
        """Deterministically solve a Cartesian path without moving hardware.

        Each waypoint is seeded from the preceding waypoint's solution. This
        models the continuous path that the controller will execute and avoids
        both the false negatives of solving every distant waypoint from the
        initial joints and the unsafe branch jumps of random-seed IK retries.
        """
        with self._lock:
            arm, _ = self._require()
            q_live, _, _ = arm.get_state()
            q_seed = pad_q_for_model(
                self._model,
                np.asarray(q_live[:6], dtype=np.float64),
                controlled_joints=6,
            )
            previous_q = np.asarray(q_live[:6], dtype=np.float64)
            solved: list[dict[str, Any]] = []
            for index, (x, y, z, roll, pitch, yaw) in enumerate(poses, 1):
                target = pos_rot_to_se3(
                    np.array([x, y, z], dtype=np.float64),
                    roll=roll, pitch=pitch, yaw=yaw,
                )
                result = solve_ik_with_retry(
                    self._model, self._data, self._end_frame, target,
                    q_seed.copy(), self._ik, max_retries=0,
                )
                target_q = np.asarray(result.q[:6], dtype=np.float64)
                if not np.all(np.isfinite(target_q)):
                    raise RuntimeError(
                        f"path waypoint {index} produced non-finite joints"
                    )
                if np.any(target_q < self._joint_lo) or \
                        np.any(target_q > self._joint_hi):
                    raise RuntimeError(
                        f"path waypoint {index} violates a URDF joint hard limit"
                    )
                joint_step = float(np.max(np.abs(target_q - previous_q)))
                if joint_step > PATH_IK_MAX_JOINT_STEP_RAD:
                    raise RuntimeError(
                        f"path waypoint {index} changes a joint by "
                        f"{joint_step:.3f}rad, beyond the "
                        f"{PATH_IK_MAX_JOINT_STEP_RAD:.3f}rad continuity cap"
                    )
                pos, rpy = joint_to_pose(result.q)
                pos_error = float(np.linalg.norm(
                    np.asarray(pos) - np.array([x, y, z])
                ))
                rpy_error = float(np.max(np.abs(
                    np.asarray(rpy) - np.array([roll, pitch, yaw])
                )))
                if not result.success and not (
                    pos_error <= LOCAL_IK_POS_TOL_M
                    and rpy_error <= LOCAL_IK_RPY_TOL_RAD
                ):
                    raise RuntimeError(
                        f"path waypoint {index} IK failed "
                        f"(error={result.error:.3e}, "
                        f"position={pos_error * 1000:.1f}mm, "
                        f"orientation={rpy_error:.4f}rad)"
                    )
                solved.append({
                    "joints_rad": target_q.tolist(),
                    "end_effector": {
                        "position_m": [float(v) for v in pos],
                        "rpy_rad": [float(v) for v in rpy],
                    },
                    "near_solution": not result.success,
                    "position_error_m": pos_error,
                    "orientation_error_rad": rpy_error,
                })
                q_seed = pad_q_for_model(
                    self._model, target_q, controlled_joints=6
                )
                previous_q = target_q
            self._assert_collision_free_joint_path(
                [np.asarray(q_live[:6], dtype=float)]
                + [np.asarray(item["joints_rad"], dtype=float) for item in solved],
                label="Cartesian path",
            )
            return solved

    def move_pose(self, x: float, y: float, z: float, roll: float, pitch: float, yaw: float,
                  duration: float) -> dict[str, Any]:
        MC.motion()
        solution = self.solve_ik(x, y, z, roll, pitch, yaw)
        with self._lock:
            arm, controller = self._require()
            start, _, _ = arm.get_state()
            start = np.asarray(start[:6], dtype=np.float64)
            target = np.asarray(solution["joints_rad"], dtype=np.float64)
            self._assert_collision_free_joint_path(
                [start, target], label="single-pose move"
            )
            steps = max(12, int(max(0.8, duration) * 12))
            for i in range(1, steps + 1):
                MC.checkpoint()
                # Quintic minimum-jerk time scaling starts and ends each short
                # Cartesian segment with zero target velocity and acceleration.
                # This is materially smoother than the previous linear ramp,
                # especially when the faster profile crosses many waypoints.
                t = i / steps
                blend = 10.0 * t**3 - 15.0 * t**4 + 6.0 * t**5
                controller._q_target = (start + (target - start) * blend).copy()
                MC.pause(duration / steps)
        deadline = time.monotonic() + max(3.0, duration)
        while time.monotonic() < deadline:
            MC.checkpoint()
            state = self.state()
            if max(abs(a - b) for a, b in zip(state["joints_rad"], target)) < 0.02:
                break
            time.sleep(0.1)
        return self.state()

    def move_pose_path(
        self,
        poses: list[tuple[float, float, float, float, float, float]],
        duration: float,
        send_freq: float = 50.0,
    ) -> dict[str, Any]:
        """Execute a fully solved Cartesian chain as one minimum-jerk path.

        ``solve_ik_path`` validates every pose, joint hard limit, and inter-pose
        joint continuity before this method sends the first target.  The global
        minimum-jerk clock then advances through the dense joint-space polyline
        without stopping at each Cartesian waypoint.
        """
        MC.motion()
        if not poses:
            return self.state()
        solutions = self.solve_ik_path(poses)
        targets = [
            np.asarray(item["joints_rad"], dtype=np.float64)
            for item in solutions
        ]
        with self._lock:
            arm, controller = self._require()
            q_live, _, _ = arm.get_state()
            points = [np.asarray(q_live[:6], dtype=np.float64), *targets]
            # IK validation begins from the planning snapshot.  Rebuild the
            # chain from fresh motor feedback and collision-check it immediately
            # before the first command, so pose drift or a newly installed scene
            # cannot turn an otherwise valid preplan into an unsafe motion.
            self._assert_collision_free_joint_path(
                points,
                label="continuous Cartesian execution path",
            )
            segment_lengths = np.asarray([
                float(np.max(np.abs(b - a)))
                for a, b in zip(points[:-1], points[1:])
            ], dtype=np.float64)
            cumulative = np.concatenate(([0.0], np.cumsum(segment_lengths)))
            path_length = float(cumulative[-1])
            if path_length < 1e-8:
                controller._q_target[:] = targets[-1]
                return self.state()

            # The peak derivative of 10s^3-15s^4+6s^5 is 1.875.  This lower
            # bound keeps the commanded joint path within the configured velocity
            # ceiling even at the midpoint of the minimum-jerk profile.
            velocity_duration = 1.875 * path_length / max(1e-6, _joint_vlim())
            total_s = max(float(duration), velocity_duration, 0.25)
            count = max(2, int(np.ceil(total_s * max(20.0, send_freq))))
            interval = total_s / count

            for i in range(1, count + 1):
                MC.checkpoint()
                t = i / count
                blend = 10.0 * t**3 - 15.0 * t**4 + 6.0 * t**5
                distance = blend * path_length
                segment = min(
                    len(segment_lengths) - 1,
                    int(np.searchsorted(cumulative, distance, side="right") - 1),
                )
                local_length = float(segment_lengths[segment])
                local_fraction = (
                    1.0 if local_length < 1e-10 else
                    float(np.clip(
                        (distance - cumulative[segment]) / local_length,
                        0.0, 1.0,
                    ))
                )
                controller._q_target[:] = (
                    points[segment]
                    + (points[segment + 1] - points[segment]) * local_fraction
                )
                MC.pause(interval)
            controller._q_target[:] = targets[-1]

        return self.wait_for_stationary_hold(
            targets[-1], timeout_s=3.0, consecutive_samples=3
        )

    def gripper(self, action: str, target_rad: float | None = None,
                limit_rads: tuple[float, float] | None = None,
                kp: float | None = None, kd: float | None = None,
                stop_on_contact: bool | None = None) -> dict[str, Any]:
        """Open/close the jaws, optionally to an explicit angle.

        ``target_rad`` lets a caller open only as wide as the object needs (see
        arm_agent.gripper_model). It is advisory: the clamp below still applies.

        ``limit_rads`` is a per-call (close, open) pair in RADIANS, so one skill can
        be more conservative than the measured limits without imposing that on every
        other user of this backend. Omitted => the measured limits.
        """
        MC.motion()
        if action not in {"open", "close"}:
            raise ValueError("gripper action must be open or close")
        with self._lock:
            arm, controller = self._require()
            close_rad, open_rad = _gm.limits(*(reversed(limit_rads) if limit_rads else (None, None)))
            if limit_rads:
                close_rad, open_rad = limit_rads
            requested = (float(target_rad) if target_rad is not None
                         else _gm.target_for(action, (close_rad, open_rad)))
            # Full open is negative in this B601-RS gripper frame; closing moves
            # toward +rad.  This hard clamp is applied immediately before every
            # motor target assignment, so neither end can ever be over-driven --
            # closing stops at the 5% travel mark rather than at the mechanical
            # stop, which the motor reports as a fault if pushed into.
            target = _gm.clamp_rad(requested, (close_rad, open_rad))

            # Re-commanding a close onto jaws that are already shut drives the motor
            # against its stop and faults it. If we are already at least this closed,
            # there is nothing to do.
            q_now, _, _ = arm.get_state()
            current = float(q_now[6]) if len(q_now) > 6 else None
            # There is no validated gripper force/contact signal. Never infer
            # contact from encoder stall: a stall may be the object, a mechanism
            # fault, or the hard stop. Callers must provide a depth-sized position.
            if stop_on_contact:
                return {
                    "ok": False,
                    "action": action,
                    "reason": "contact-aware closing is disabled; command an explicit "
                              "depth-sized target instead",
                    "connected": True,
                }

            # A reading well beyond the closed stop is not a closed gripper: the
            # motor has stalled into the stop or lost its zero reference. Say so
            # instead of reporting "already closed", which hides the fault AND
            # blocks the close command that might otherwise recover it.
            if current is not None and _gm.out_of_range(current, (close_rad, open_rad)):
                return {"ok": False, "action": action,
                        "position_rad": round(current, 4),
                        "position_deg": round(_gm.deg(current), 2),
                        "target_rad": round(target, 4),
                        "target_deg": round(_gm.deg(target), 2),
                        "connected": True,
                        "reason": f"gripper reads {_gm.deg(current):.2f} deg, outside the "
                                  f"limits [{_gm.deg(close_rad):.2f}, {_gm.deg(open_rad):.2f}] "
                                  f"by more than {_gm.deg(_gm.SLOP_RAD):.0f} deg -- stalled, or "
                                  "the position reference has moved. Re-measure before "
                                  "commanding it again."}
            if (action == "close" and current is not None
                    and abs(current - close_rad) <= GRIPPER_ALREADY_CLOSED_RAD):
                return {"ok": True, "action": action, "target_pos_rad": target,
                        "requested_target_pos_rad": requested,
                        "open_limit_deg": round(_gm.deg(open_rad), 2),
                        "close_limit_deg": round(_gm.deg(close_rad), 2),
                        "position_rad": round(current, 4),
                        "position_deg": round(_gm.deg(current), 2),
                        "connected": True,
                        "skipped": f"already at the close limit "
                                   f"({_gm.deg(close_rad):.2f} deg)"}

            # Optional MIT gain override for the duration of this move. The config
            # ships the gripper at kp=50, but the routine that demonstrably moves
            # these jaws uses kp=2 and the known-good RS gripper config uses kp=8 --
            # so being able to try a softer gain without editing yaml is the fastest
            # way to tell a gain problem from a comms one.
            group = getattr(controller, "_gripper_group", None)
            saved: tuple[Any, Any] | None = None
            effective_kp = kp
            effective_kd = kd
            if group is not None and (effective_kp is not None or effective_kd is not None):
                saved = (group._mit_kp.copy(), group._mit_kd.copy())
                if effective_kp is not None:
                    group._mit_kp.fill(float(effective_kp))
                if effective_kd is not None:
                    group._mit_kd.fill(float(effective_kd))
            try:
                # Ramp the target instead of jumping to it, so the jaws ease open and
                # shut rather than snapping.
                start = current if current is not None else target
                travel = abs(target - start)
                n_steps = max(1, int(GRIPPER_MOVE_S / max(1e-3, GRIPPER_STEP_S)))
                for i in range(1, n_steps + 1):
                    MC.checkpoint()
                    command = start + (target - start) * i / n_steps
                    controller.set_gripper_target(command)
                    MC.pause(GRIPPER_STEP_S)
            finally:
                if saved is not None and group is not None:
                    group._mit_kp[:], group._mit_kd[:] = saved
        time.sleep(0.4)
        q, _, _ = arm.get_state()
        position = float(q[6]) if len(q) > 6 else None
        return {"ok": True, "action": action,
                "target_rad": round(target, 4), "target_deg": round(_gm.deg(target), 2),
                "target_pos_rad": target,
                "position_deg": None if position is None else round(_gm.deg(position), 2),
                "open_limit_deg": round(_gm.deg(open_rad), 2),
                "close_limit_deg": round(_gm.deg(close_rad), 2),
                "position_rad": position, "connected": True,
                "mit_kp": None if effective_kp is None else float(effective_kp),
                "mit_kd": None if effective_kd is None else float(effective_kd),
                "contact_aware": False}

    def home(self) -> dict[str, Any]:
        MC.motion()
        with self._lock:
            arm, controller = self._require()
            q_start, _, _ = arm.get_state()
            q_start = np.asarray(q_start[:6], dtype=np.float64)
            # Home is still a trajectory through the scene. Validate the direct
            # joint-space return before sending it; on failure the caller's
            # recovery path freezes and keeps motor holding torque enabled.
            self._assert_collision_free_joint_path(
                [q_start, np.zeros(6, dtype=np.float64)],
                label="home path",
            )
            # Recovery remains at the previously tested 2x ceiling even when a
            # normal trajectory opts into the faster 3x profile.
            max_vel = min(_joint_vlim(), 0.60)
            # safe_home uses a minimum-jerk trajectory lasting
            # 2*max_joint_error/max_vel. Its old fixed 15 s timeout was shorter
            # than a valid scan->home path at the deliberate 0.25 rad/s limit.
            trajectory_s = 2.0 * float(np.max(np.abs(q_start))) / max(1e-6, max_vel)
            if MC.current() is None:
                confirmed = controller.safe_home(
                    max_vel=max_vel,
                    timeout=max(15.0, trajectory_s + 5.0),
                )
            else:
                # The vendor safe_home owns a blocking sender. Hybrid missions
                # use the same validated joint return with an interruptible
                # minimum-jerk loop; endpoint feedback below remains authoritative.
                duration = max(0.25, trajectory_s)
                count = max(2, int(math.ceil(duration * 50)))
                for index in range(1, count + 1):
                    MC.checkpoint()
                    t = index / count
                    blend = 10 * t**3 - 15 * t**4 + 6 * t**5
                    controller._q_target = q_start * (1.0 - blend)
                    MC.pause(duration / count)
                confirmed = False
            settle_deadline = time.monotonic() + HOME_SETTLE_TIMEOUT_S
            state = self.state()
            while True:
                MC.checkpoint()
                joints = [float(v) for v in state.get("joints_rad", [])]
                velocities = [abs(float(v)) for v in state.get("joint_vel_rad_s", [])]
                max_error = max((abs(v) for v in joints), default=float("inf"))
                max_velocity = max(velocities, default=float("inf"))
                if max_error < HOME_POS_TOL_RAD and max_velocity < HOME_VEL_TOL_RAD_S:
                    break
                if time.monotonic() >= settle_deadline:
                    break
                time.sleep(0.05)
                state = self.state()
            # safe_home's internal confirmation uses a tighter instantaneous
            # threshold and can time out a few encoder counts before zero. The
            # state sampled through the settle window is the authoritative,
            # independently measured postcondition for parking.
            measured_home = (
                max_error < HOME_POS_TOL_RAD
                and max_velocity < HOME_VEL_TOL_RAD_S
            )
            ok = bool(measured_home)
            return {
                "ok": ok,
                "reason": (
                    "home_confirmed"
                    if confirmed else
                    "home_confirmed_by_settled_state_after_controller_timeout"
                ) if ok else
                          f"home_not_confirmed max_joint_error={max_error:.4f}rad "
                          f"max_joint_velocity={max_velocity:.4f}rad/s",
                "max_joint_error_rad": max_error,
                "max_joint_velocity_rad_s": max_velocity,
                "state": state,
            }


_local: LocalArm | None = None


def get_local_arm() -> LocalArm:
    global _local
    if _local is None:
        _local = LocalArm()
        atexit.register(_local.disconnect)
    return _local
