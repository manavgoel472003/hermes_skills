"""Conservative RGB-D obstacle mapping for the wrist pick/place controller.

The motion controller does not need to recognize every object in order to avoid
it.  This module converts the valid depth image into base-frame obstacle columns;
semantic detections are used only to remove the explicitly allowed grasp target.
Everything else that rises above the calibrated table is occupied space.

The saved wrist-camera translation mapping has units of camera millimetres per
metre of base translation.  If ``jaw_target_cam_mm`` is the camera coordinate of
the robot tool point, a scene point maps to base coordinates as::

    p_base = ee_base + solve(mapping, jaw_target_cam_mm - p_camera_mm)

This is the same measured geometry used by the proven visual approach and avoids
inventing a second camera convention.  Mapping uncertainty is handled by
inflating every obstacle column before collision checking.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np


@dataclass(frozen=True)
class ObstacleBox:
    """Axis-aligned occupied box in the robot base frame, in metres."""

    center_m: tuple[float, float, float]
    size_m: tuple[float, float, float]
    point_count: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "center_m": [round(float(v), 4) for v in self.center_m],
            "size_m": [round(float(v), 4) for v in self.size_m],
            "point_count": int(self.point_count),
        }


@dataclass(frozen=True)
class SceneMap:
    boxes: tuple[ObstacleBox, ...]
    table_z_m: float
    voxel_m: float
    inflation_m: float
    valid_depth_points: int
    mapped_obstacle_points: int
    excluded_target_points: int
    closest_obstacle_m: float | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "collision_model": True,
            "obstacle_columns": len(self.boxes),
            "valid_depth_points": int(self.valid_depth_points),
            "mapped_obstacle_points": int(self.mapped_obstacle_points),
            "excluded_target_points": int(self.excluded_target_points),
            "voxel_mm": round(self.voxel_m * 1000.0, 1),
            "inflation_mm": round(self.inflation_m * 1000.0, 1),
            "table_z_m": round(float(self.table_z_m), 4),
            "closest_obstacle_mm": (
                None if self.closest_obstacle_m is None
                else round(self.closest_obstacle_m * 1000.0, 1)
            ),
        }


def _intrinsics(intr: Any) -> tuple[float, float, float, float]:
    if isinstance(intr, (tuple, list)) and len(intr) >= 4:
        return tuple(float(v) for v in intr[:4])  # type: ignore[return-value]
    names = ("fx", "fy", "ppx", "ppy")
    try:
        return tuple(float(getattr(intr, name)) for name in names)  # type: ignore[return-value]
    except (AttributeError, TypeError) as exc:
        raise ValueError("camera intrinsics must provide fx, fy, ppx, ppy") from exc


def _pixel_exclusion_mask(
    shape: tuple[int, int], boxes: Iterable[tuple[int, int, int, int]], pad_px: int
) -> np.ndarray:
    height, width = shape
    mask = np.zeros((height, width), dtype=bool)
    for x1, y1, x2, y2 in boxes:
        xa, ya = max(0, int(x1) - pad_px), max(0, int(y1) - pad_px)
        xb, yb = min(width, int(x2) + pad_px), min(height, int(y2) + pad_px)
        if xb > xa and yb > ya:
            mask[ya:yb, xa:xb] = True
    return mask


def build_scene_map(
    depth_mm: np.ndarray,
    intr: Any,
    *,
    ee_xyz_m: Iterable[float],
    camera_mapping_mm_per_m: np.ndarray,
    jaw_target_cam_mm: Iterable[float],
    table_z_m: float,
    workspace_lo_m: Iterable[float],
    workspace_hi_m: Iterable[float],
    exclude_pixel_boxes: Iterable[tuple[int, int, int, int]] = (),
    near_mm: float = 100.0,
    far_mm: float = 650.0,
    stride_px: int = 4,
    voxel_m: float = 0.020,
    inflation_m: float = 0.025,
    table_margin_m: float = 0.012,
    target_pad_px: int = 8,
    minimum_points_per_column: int = 2,
) -> SceneMap:
    """Create inflated vertical obstacle columns from one aligned depth frame.

    A column representation is deliberately conservative for table manipulation:
    if an object occupies an XY cell, the entire space from the table to its
    highest observed point is blocked.  That prevents paths from slipping through
    depth holes or beneath partially observed objects.
    """
    depth = np.asarray(depth_mm, dtype=float)
    if depth.ndim != 2:
        raise ValueError("depth image must be two-dimensional")
    mapping = np.asarray(camera_mapping_mm_per_m, dtype=float)
    if mapping.shape != (3, 3) or not np.all(np.isfinite(mapping)):
        raise ValueError("camera mapping must be a finite 3x3 matrix")
    if abs(float(np.linalg.det(mapping))) < 1e-6:
        raise ValueError("camera mapping is singular")
    ee = np.asarray(tuple(ee_xyz_m), dtype=float)
    jaw = np.asarray(tuple(jaw_target_cam_mm), dtype=float)
    lower = np.asarray(tuple(workspace_lo_m), dtype=float)
    upper = np.asarray(tuple(workspace_hi_m), dtype=float)
    if any(value.shape != (3,) for value in (ee, jaw, lower, upper)):
        raise ValueError("EE, jaw target, and workspace bounds must be XYZ triplets")

    step = max(1, int(stride_px))
    rows = np.arange(0, depth.shape[0], step, dtype=int)
    cols = np.arange(0, depth.shape[1], step, dtype=int)
    vv, uu = np.meshgrid(rows, cols, indexing="ij")
    zz = depth[vv, uu]
    valid = np.isfinite(zz) & (zz >= float(near_mm)) & (zz <= float(far_mm))
    valid_depth_points = int(np.count_nonzero(valid))

    excluded_full = _pixel_exclusion_mask(
        depth.shape, exclude_pixel_boxes, max(0, int(target_pad_px))
    )
    excluded = excluded_full[vv, uu] & valid
    valid &= ~excluded
    excluded_count = int(np.count_nonzero(excluded))
    if not np.any(valid):
        return SceneMap((), float(table_z_m), float(voxel_m),
                        float(inflation_m), valid_depth_points, 0,
                        excluded_count, None)

    fx, fy, ppx, ppy = _intrinsics(intr)
    z = zz[valid]
    u = uu[valid].astype(float)
    v = vv[valid].astype(float)
    camera_points = np.column_stack((
        (u - ppx) * z / max(1e-9, fx),
        (v - ppy) * z / max(1e-9, fy),
        z,
    ))
    # Solve all points together: mapping @ delta = jaw - camera_point.
    deltas_m = np.linalg.solve(mapping, (jaw - camera_points).T).T
    base_points = ee + deltas_m

    # Restrict mapping to the task volume.  A small lateral halo lets an inflated
    # obstacle just outside the nominal workspace still protect its boundary.
    halo = max(float(inflation_m), float(voxel_m))
    inside = np.all(base_points >= lower - halo, axis=1) & \
        np.all(base_points <= upper + halo, axis=1)
    above_table = base_points[:, 2] > float(table_z_m) + float(table_margin_m)
    points = base_points[inside & above_table]
    if points.size == 0:
        return SceneMap((), float(table_z_m), float(voxel_m),
                        float(inflation_m), valid_depth_points, 0,
                        excluded_count, None)

    voxel = float(np.clip(voxel_m, 0.010, 0.040))
    inflation = float(np.clip(inflation_m, 0.010, 0.060))
    xy_index = np.floor(points[:, :2] / voxel).astype(np.int64)
    unique, inverse, counts = np.unique(
        xy_index, axis=0, return_inverse=True, return_counts=True
    )
    boxes: list[ObstacleBox] = []
    for column_index, (ix, iy) in enumerate(unique):
        count = int(counts[column_index])
        if count < max(1, int(minimum_points_per_column)):
            continue
        column = points[inverse == column_index]
        z_top = float(np.max(column[:, 2]))
        z_bottom = float(table_z_m) + float(table_margin_m)
        height = max(voxel, z_top - z_bottom)
        center = (
            (float(ix) + 0.5) * voxel,
            (float(iy) + 0.5) * voxel,
            z_bottom + height / 2.0,
        )
        size = (
            voxel + 2.0 * inflation,
            voxel + 2.0 * inflation,
            height + 2.0 * inflation,
        )
        boxes.append(ObstacleBox(center, size, count))

    closest = None
    if boxes:
        closest = min(float(np.linalg.norm(
            np.asarray(box.center_m, dtype=float) - ee
        )) for box in boxes)
    return SceneMap(
        tuple(boxes), float(table_z_m), voxel, inflation,
        valid_depth_points, int(points.shape[0]), excluded_count, closest,
    )


def payload_radius_m(context: dict[str, Any] | None) -> float:
    """Conservative spherical carried-object envelope from grasp metadata."""
    if not context:
        return 0.0
    geometry = context.get("grasp_geometry") or {}
    envelope = geometry.get("envelope_detection") or {}
    grasp = geometry.get("grasp_detection") or {}
    candidates = [
        envelope.get("max_width_mm"), envelope.get("min_width_mm"),
        grasp.get("max_width_mm"), grasp.get("min_width_mm"),
        (context.get("jaw_plan") or {}).get("object_width_mm"),
    ]
    widths = []
    for value in candidates:
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(number) and number > 0.0:
            widths.append(number)
    if not widths:
        return 0.035
    # Radius is capped because a long marker is carried lengthwise through the
    # gripper; using its complete length as a sphere would reject every route.
    return float(np.clip(max(widths) / 2000.0 + 0.015, 0.025, 0.080))
