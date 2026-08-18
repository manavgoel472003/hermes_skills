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

from dataclasses import dataclass, replace
from pathlib import Path
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
    points_m: tuple[tuple[float, float, float], ...] = ()
    excluded_target_columns: int = 0
    unconsolidated_obstacle_columns: int = 0

    def as_dict(self) -> dict[str, Any]:
        before = max(
            len(self.boxes), int(self.unconsolidated_obstacle_columns)
        )
        return {
            "collision_model": True,
            "obstacle_columns": len(self.boxes),
            "unconsolidated_obstacle_columns": before,
            "consolidated_obstacle_columns": len(self.boxes),
            "consolidated_obstacle_reduction": before - len(self.boxes),
            "valid_depth_points": int(self.valid_depth_points),
            "mapped_obstacle_points": int(self.mapped_obstacle_points),
            "excluded_target_points": int(self.excluded_target_points),
            "excluded_target_columns": int(self.excluded_target_columns),
            "voxel_mm": round(self.voxel_m * 1000.0, 1),
            "inflation_mm": round(self.inflation_m * 1000.0, 1),
            "table_z_m": round(float(self.table_z_m), 4),
            "closest_obstacle_mm": (
                None if self.closest_obstacle_m is None
                else round(self.closest_obstacle_m * 1000.0, 1)
            ),
        }


def _box_bounds(box: ObstacleBox) -> tuple[np.ndarray, np.ndarray]:
    center = np.asarray(box.center_m, dtype=float)
    half = np.asarray(box.size_m, dtype=float) / 2.0
    return center - half, center + half


def _bounding_box(first: ObstacleBox, second: ObstacleBox) -> ObstacleBox:
    first_lo, first_hi = _box_bounds(first)
    second_lo, second_hi = _box_bounds(second)
    lower = np.minimum(first_lo, second_lo)
    upper = np.maximum(first_hi, second_hi)
    return ObstacleBox(
        tuple(float(value) for value in (lower + upper) / 2.0),
        tuple(float(value) for value in upper - lower),
        int(first.point_count) + int(second.point_count),
    )


def consolidate_obstacle_boxes(
    boxes: Iterable[ObstacleBox], *,
    minimum_overlap_fraction: float = 0.15,
    maximum_added_volume_fraction: float = 0.02,
) -> tuple[ObstacleBox, ...]:
    """Consolidate redundant AABBs without removing occupied volume.

    Every accepted replacement is the bounding box of its inputs, so it is a
    conservative superset of both.  The added-volume limit prevents a merge
    from filling meaningful concavities or gaps merely to reduce box count.
    Inflated neighbouring height-map cells overlap strongly and normally have
    a common lower/upper Z extent, making their merge exact or nearly exact.
    """
    pending = list(boxes)
    if len(pending) < 2:
        return tuple(pending)
    overlap_limit = float(np.clip(minimum_overlap_fraction, 0.0, 1.0))
    addition_limit = max(0.0, float(maximum_added_volume_fraction))
    epsilon = 1e-12

    while True:
        best: tuple[float, float, int, int, ObstacleBox] | None = None
        for first_index in range(len(pending)):
            first = pending[first_index]
            first_lo, first_hi = _box_bounds(first)
            first_size = first_hi - first_lo
            first_volume = float(np.prod(first_size))
            for second_index in range(first_index + 1, len(pending)):
                second = pending[second_index]
                second_lo, second_hi = _box_bounds(second)
                second_size = second_hi - second_lo
                second_volume = float(np.prod(second_size))
                if first_volume <= epsilon or second_volume <= epsilon:
                    continue

                intersection_size = np.maximum(
                    0.0, np.minimum(first_hi, second_hi)
                    - np.maximum(first_lo, second_lo)
                )
                intersection_volume = float(np.prod(intersection_size))
                merged = _bounding_box(first, second)
                merged_volume = float(np.prod(merged.size_m))
                union_volume = (
                    first_volume + second_volume - intersection_volume
                )
                added_fraction = max(
                    0.0, (merged_volume - union_volume)
                    / max(epsilon, union_volume)
                )

                # Containment is always an exact geometry reduction. Otherwise
                # require genuine three-dimensional overlap as well as tightly
                # bounded conservative fill.
                contained = bool(
                    np.all(first_lo >= second_lo - epsilon)
                    and np.all(first_hi <= second_hi + epsilon)
                ) or bool(
                    np.all(second_lo >= first_lo - epsilon)
                    and np.all(second_hi <= first_hi + epsilon)
                )
                overlap_fraction = intersection_volume / min(
                    first_volume, second_volume
                )
                if not contained and (
                    overlap_fraction + epsilon < overlap_limit
                    or added_fraction > addition_limit + epsilon
                ):
                    continue
                candidate = (
                    added_fraction, -overlap_fraction,
                    first_index, second_index, merged,
                )
                if best is None or candidate[:4] < best[:4]:
                    best = candidate
        if best is None:
            break
        _, _, first_index, second_index, merged = best
        pending[first_index] = merged
        del pending[second_index]
    return tuple(pending)


def consolidate_scene_map(
    scene: SceneMap, *, minimum_overlap_fraction: float = 0.15,
    maximum_added_volume_fraction: float = 0.02,
) -> SceneMap:
    """Return a collision-equivalent-or-more-conservative compact scene."""
    before = max(
        len(scene.boxes), int(scene.unconsolidated_obstacle_columns)
    )
    boxes = consolidate_obstacle_boxes(
        scene.boxes,
        minimum_overlap_fraction=minimum_overlap_fraction,
        maximum_added_volume_fraction=maximum_added_volume_fraction,
    )
    return replace(
        scene, boxes=boxes, unconsolidated_obstacle_columns=before
    )


def fuse_scene_maps(
    scenes: Iterable[SceneMap], *, ee_xyz_m: Iterable[float] | None = None,
    minimum_views_per_column: int = 1,
) -> SceneMap:
    """Fuse base-frame maps without averaging away occupied observations.

    Inputs may come from different camera poses because ``points_m`` are already
    expressed in the robot base frame. A column is retained when observed in the
    requested number of views; the default is conservative union occupancy.
    """
    maps = list(scenes)
    if not maps:
        raise ValueError("at least one scene map is required")
    table_z = float(maps[0].table_z_m)
    voxel = float(maps[0].voxel_m)
    inflation = float(maps[0].inflation_m)
    for scene in maps[1:]:
        if abs(scene.table_z_m - table_z) > 0.005:
            raise ValueError("scene maps use incompatible table heights")
        if abs(scene.voxel_m - voxel) > 1e-9 or \
                abs(scene.inflation_m - inflation) > 1e-9:
            raise ValueError("scene maps use incompatible voxel parameters")
    per_view: list[dict[tuple[int, int], np.ndarray]] = []
    all_points = []
    for scene in maps:
        points = np.asarray(scene.points_m, dtype=float).reshape(-1, 3)
        all_points.append(points)
        columns: dict[tuple[int, int], np.ndarray] = {}
        if len(points):
            indices = np.floor(points[:, :2] / voxel).astype(np.int64)
            for key_raw in np.unique(indices, axis=0):
                key = (int(key_raw[0]), int(key_raw[1]))
                columns[key] = points[np.all(indices == key_raw, axis=1)]
        per_view.append(columns)
    keys = set().union(*(columns.keys() for columns in per_view))
    boxes = []
    retained_points = []
    required = max(1, min(len(maps), int(minimum_views_per_column)))
    z_bottom = table_z + 0.012
    for ix, iy in sorted(keys):
        observations = [columns[(ix, iy)] for columns in per_view
                        if (ix, iy) in columns]
        if len(observations) < required:
            continue
        column = np.vstack(observations)
        retained_points.append(column)
        z_top = float(np.max(column[:, 2]))
        height = max(voxel, z_top - z_bottom)
        boxes.append(ObstacleBox(
            ((ix + 0.5) * voxel, (iy + 0.5) * voxel,
             z_bottom + height / 2.0),
            (voxel + 2 * inflation, voxel + 2 * inflation,
             height + 2 * inflation),
            int(len(column)),
        ))
    fused_points = (
        np.vstack(retained_points) if retained_points
        else np.empty((0, 3), dtype=float)
    )
    closest = None
    if boxes and ee_xyz_m is not None:
        ee = np.asarray(tuple(ee_xyz_m), dtype=float)
        closest = min(float(np.linalg.norm(np.asarray(box.center_m) - ee))
                      for box in boxes)
    return SceneMap(
        tuple(boxes), table_z, voxel, inflation,
        sum(scene.valid_depth_points for scene in maps), int(len(fused_points)),
        sum(scene.excluded_target_points for scene in maps), closest,
        tuple(tuple(float(value) for value in point) for point in fused_points),
        sum(scene.excluded_target_columns for scene in maps),
    )


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


def _normalize_exclusion_obb(
    center_raw: Iterable[float],
    axes_raw: np.ndarray,
    half_size_raw: Iterable[float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Validate and rigidize one measured target OBB."""
    center = np.asarray(tuple(center_raw), dtype=float)
    axes = np.asarray(axes_raw, dtype=float)
    half_size = np.asarray(tuple(half_size_raw), dtype=float)
    if center.shape != (3,) or axes.shape != (3, 3) \
            or half_size.shape != (3,) or np.any(half_size <= 0.0):
        raise ValueError("target exclusion OBB has invalid dimensions")
    u, _singular, vh = np.linalg.svd(axes)
    rigid_axes = u @ vh
    if float(np.linalg.det(rigid_axes)) < 0.0:
        u[:, -1] *= -1.0
        rigid_axes = u @ vh
    fit_error = float(np.linalg.norm(axes - rigid_axes) /
                      max(1e-9, np.linalg.norm(rigid_axes)))
    if fit_error > 0.08:
        raise ValueError("target exclusion OBB axes are not rigid")
    return center, rigid_axes, half_size


def _column_owned_by_target_obb(
    center_xy_m: Iterable[float],
    z_top_m: float,
    voxel_m: float,
    target_obb: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> bool:
    """Return whether a raw XY voxel is a boundary fragment of the target.

    Point-centre tests leave fragments whenever an object boundary and a coarse
    collision voxel straddle one another.  Test the *cell footprint* against the
    measured OBB instead.  The height gate is deliberately limited to one raw
    voxel above the measured box: this removes quantisation spill from the
    selected object, but preserves a materially taller neighbour/support.
    """
    center, axes, half_size = target_obb
    center_xy = np.asarray(tuple(center_xy_m), dtype=float)
    if center_xy.shape != (2,):
        raise ValueError("column centre must be an XY pair")
    cell_center = np.array([center_xy[0], center_xy[1], center[2]], dtype=float)
    local_center = (cell_center - center) @ axes.T
    half_voxel = max(0.0, float(voxel_m)) / 2.0
    # Projection of an axis-aligned XY square onto each OBB local axis.
    footprint_expansion = half_voxel * (
        np.abs(axes[:, 0]) + np.abs(axes[:, 1])
    )
    footprint_intersects = bool(np.all(
        np.abs(local_center) <= half_size + footprint_expansion + 1e-9
    ))
    if not footprint_intersects:
        return False
    base_z_extent = float(np.sum(np.abs(axes[:, 2]) * half_size))
    target_top_m = float(center[2]) + base_z_extent
    # One coarse voxel covers quantisation at the object boundary. The extra
    # bounded 5 mm covers RealSense depth/fusion spread observed between the
    # stationary OBB fit and the collision burst. It is intentionally much less
    # than another voxel, so a stacked/taller object remains occupied.
    sensor_height_allowance_m = min(0.005, float(voxel_m) / 2.0)
    return float(z_top_m) <= (
        target_top_m + float(voxel_m) + sensor_height_allowance_m + 1e-9
    )


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
    exclude_pixel_masks: Iterable[np.ndarray] = (),
    exclude_base_capsules: Iterable[
        tuple[Iterable[float], Iterable[float], float, float]
    ] = (),
    exclude_base_obbs: Iterable[
        tuple[Iterable[float], np.ndarray, Iterable[float]]
    ] = (),
    near_mm: float = 100.0,
    far_mm: float = 650.0,
    stride_px: int = 4,
    voxel_m: float = 0.020,
    inflation_m: float = 0.003,
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
    for mask_raw in exclude_pixel_masks:
        mask = np.asarray(mask_raw)
        if mask.shape != depth.shape:
            raise ValueError("target exclusion mask does not match depth image")
        excluded_full |= mask > 0
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

    # Pixel masks can leave depth-edge fragments around the selected object.
    # Remove only points inside its measured 3D capsule, preserving unrelated
    # obstacles that happen to be nearby in the image.
    excluded_3d = np.zeros(len(base_points), dtype=bool)
    target_capsules = list(exclude_base_capsules)
    target_obbs = [
        _normalize_exclusion_obb(center, axes, half_size)
        for center, axes, half_size in exclude_base_obbs
    ]
    for center_raw, axis_raw, half_length_m, radius_m in target_capsules:
        center = np.asarray(tuple(center_raw), dtype=float)
        axis = np.asarray(tuple(axis_raw), dtype=float)
        axis /= max(1e-9, float(np.linalg.norm(axis)))
        relative = base_points - center
        axial = relative @ axis
        clipped = np.clip(axial, -float(half_length_m), float(half_length_m))
        radial = np.linalg.norm(relative - clipped[:, None] * axis, axis=1)
        excluded_3d |= radial <= float(radius_m)
    # The complete measured target OBB is more authoritative than the 2D
    # silhouette capsule when the object is broad, tilted, or partly occluded.
    # Axes are supplied as three base-frame row vectors. A point is excluded
    # only when it lies inside all three measured half-extents; nearby objects
    # outside the box remain occupied even when they touch its AABB.
    for center, rigid_axes, half_size in target_obbs:
        local = (base_points - center) @ rigid_axes.T
        excluded_3d |= np.all(
            np.abs(local) <= half_size[None, :] + 1e-9, axis=1
        )
    excluded_count += int(np.count_nonzero(excluded_3d))

    # Restrict mapping to the task volume.  A small lateral halo lets an inflated
    # obstacle just outside the nominal workspace still protect its boundary.
    halo = max(float(inflation_m), float(voxel_m))
    inside = np.all(base_points >= lower - halo, axis=1) & \
        np.all(base_points <= upper + halo, axis=1)
    above_table = base_points[:, 2] > float(table_z_m) + float(table_margin_m)
    points = base_points[inside & above_table & ~excluded_3d]
    if points.size == 0:
        return SceneMap((), float(table_z_m), float(voxel_m),
                        float(inflation_m), valid_depth_points, 0,
                        excluded_count, None)

    voxel = float(np.clip(voxel_m, 0.010, 0.040))
    # Keep a nonzero sensor/model uncertainty band, while allowing callers to
    # use a tighter map than the former 7.5 mm hard floor. The voxel itself
    # still covers its complete measured cell before this band is added.
    inflation = float(np.clip(inflation_m, 0.001, 0.060))
    xy_index = np.floor(points[:, :2] / voxel).astype(np.int64)
    unique, inverse, counts = np.unique(
        xy_index, axis=0, return_inverse=True, return_counts=True
    )
    boxes: list[ObstacleBox] = []
    excluded_target_columns = 0
    target_owned_points = np.zeros(len(points), dtype=bool)
    for column_index, (ix, iy) in enumerate(unique):
        count = int(counts[column_index])
        column = points[inverse == column_index]
        z_top = float(np.max(column[:, 2]))
        z_bottom = float(table_z_m) + float(table_margin_m)
        height = max(voxel, z_top - z_bottom)
        center = (
            (float(ix) + 0.5) * voxel,
            (float(iy) + 0.5) * voxel,
            z_bottom + height / 2.0,
        )
        # A handful of depth-edge returns can survive the point-level mask and
        # otherwise regenerate the selected object as a full-height column.
        # Suppress columns inside the target's measured XY footprint. Use its
        # circumscribed radius because the 2D column representation has already
        # discarded object orientation.
        target_column = False
        center_xy = np.asarray(center[:2], dtype=float)
        for target_center_raw, _axis_raw, half_length_m, radius_m in \
                target_capsules:
            target_center = np.asarray(tuple(target_center_raw), dtype=float)
            footprint_radius = max(float(half_length_m), float(radius_m))
            if float(np.linalg.norm(center_xy - target_center[:2])) <= \
                    footprint_radius:
                target_column = True
                break
        if not target_column:
            target_column = any(
                _column_owned_by_target_obb(center_xy, z_top, voxel, target_obb)
                for target_obb in target_obbs
            )
        if target_column:
            excluded_target_columns += 1
            # Scene fusion consumes ``points_m`` and reconstructs columns from
            # it. Remove the owned raw returns too, otherwise a target column
            # deleted in every frame is resurrected by the fusion pass.
            target_owned_points[inverse == column_index] = True
            continue
        if count < max(1, int(minimum_points_per_column)):
            continue
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
    retained_points = points[~target_owned_points]
    return SceneMap(
        tuple(boxes), float(table_z_m), voxel, inflation,
        valid_depth_points, int(retained_points.shape[0]), excluded_count, closest,
        points_m=tuple(
            tuple(float(value) for value in point) for point in retained_points
        ),
        excluded_target_columns=excluded_target_columns,
    )


def build_dense_view_cloud(
    color_bgr: np.ndarray,
    depth_mm: np.ndarray,
    intr: Any,
    *,
    ee_xyz_m: Iterable[float],
    camera_mapping_mm_per_m: np.ndarray,
    jaw_target_cam_mm: Iterable[float],
    workspace_lo_m: Iterable[float],
    workspace_hi_m: Iterable[float],
    table_z_m: float | None = None,
    near_mm: float = 60.0,
    far_mm: float = 650.0,
    stride_px: int = 1,
    voxel_m: float = 0.002,
    bounds_margin_m: float = 0.08,
    minimum_neighbor_support: int = 2,
    crop_to_workspace: bool = True,
    crop_below_table: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Dense RGB cloud for inspection, independent of the safety voxel map.

    Depth-edge speckles are removed by requiring local depth support, then a
    fine 3 mm RGB voxel average reduces noise without producing the blocky
    20 mm columns used by collision planning.
    """
    color = np.asarray(color_bgr)
    depth = np.asarray(depth_mm, dtype=float)
    if depth.ndim != 2 or color.shape[:2] != depth.shape:
        raise ValueError("aligned color and depth frames must have equal dimensions")
    mapping = np.asarray(camera_mapping_mm_per_m, dtype=float)
    ee = np.asarray(tuple(ee_xyz_m), dtype=float)
    jaw = np.asarray(tuple(jaw_target_cam_mm), dtype=float)
    lower = np.asarray(tuple(workspace_lo_m), dtype=float)
    upper = np.asarray(tuple(workspace_hi_m), dtype=float)
    if mapping.shape != (3, 3) or abs(float(np.linalg.det(mapping))) < 1e-6:
        raise ValueError("camera mapping must be an invertible 3x3 matrix")

    valid = np.isfinite(depth) & (depth >= float(near_mm)) \
        & (depth <= float(far_mm))
    # Flying depth pixels have no similarly deep neighbor. Requiring two of the
    # four direct neighbors preserves object boundaries while removing isolated
    # streaks and RealSense edge spray.
    support = np.zeros(depth.shape, dtype=np.uint8)
    threshold_mm = 25.0
    for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        shifted = np.roll(depth, (dy, dx), axis=(0, 1))
        shifted_valid = np.roll(valid, (dy, dx), axis=(0, 1))
        supported = shifted_valid & (np.abs(shifted - depth) <= threshold_mm)
        if dy == -1:
            supported[-1, :] = False
        elif dy == 1:
            supported[0, :] = False
        elif dx == -1:
            supported[:, -1] = False
        else:
            supported[:, 0] = False
        support += supported.astype(np.uint8)
    valid &= support >= int(np.clip(minimum_neighbor_support, 0, 4))

    step = max(1, int(stride_px))
    vv, uu = np.indices(depth.shape)
    valid &= (vv % step == 0) & (uu % step == 0)
    if not np.any(valid):
        return np.empty((0, 3), dtype=float), np.empty((0, 3), dtype=np.uint8)
    fx, fy, ppx, ppy = _intrinsics(intr)
    z = depth[valid]
    u = uu[valid].astype(float)
    v = vv[valid].astype(float)
    camera_points = np.column_stack((
        (u - ppx) * z / max(1e-9, fx),
        (v - ppy) * z / max(1e-9, fy),
        z,
    ))
    base_points = ee + np.linalg.solve(
        mapping, (jaw - camera_points).T
    ).T
    # Preserve the raw calibrated base-frame transform. Older code snapped
    # each view independently to its strongest near-table Z histogram mode.
    # When an object top won that histogram, the whole view moved by up to
    # 40 mm and multi-view fusion rendered a second copy of the scene.
    margin = float(np.clip(bounds_margin_m, 0.03, 0.30))
    inside = np.ones(len(base_points), dtype=bool)
    if crop_to_workspace:
        inside &= np.all(base_points >= lower - margin, axis=1) \
            & np.all(base_points <= upper + margin, axis=1)
    if table_z_m is not None and crop_below_table:
        inside &= base_points[:, 2] >= float(table_z_m) - 0.015
    points = base_points[inside]
    # OpenCV frames are BGR; PLY/Rerun expect RGB.
    colors = color[valid][inside][:, ::-1].astype(np.uint8)
    if not points.size:
        return points.reshape(0, 3), colors.reshape(0, 3)

    voxel = float(np.clip(voxel_m, 0.0015, 0.008))
    keys = np.floor(points / voxel).astype(np.int64)
    _, inverse, counts = np.unique(
        keys, axis=0, return_inverse=True, return_counts=True
    )
    count = len(counts)
    point_sum = np.zeros((count, 3), dtype=float)
    color_sum = np.zeros((count, 3), dtype=float)
    np.add.at(point_sum, inverse, points)
    np.add.at(color_sum, inverse, colors.astype(float))
    averaged_points = point_sum / counts[:, None]
    averaged_colors = np.clip(
        color_sum / counts[:, None], 0, 255
    ).astype(np.uint8)
    return averaged_points, averaged_colors


def median_depth_image(depth_images_mm: Iterable[np.ndarray]) -> np.ndarray:
    """Return one stable depth image from a stationary RGB-D burst.

    Re-exporting every stationary frame does not add a new viewpoint; it only
    gives RealSense depth jitter several chances to draw parallel copies of an
    edge.  Collapse those frames per pixel before lifting the diagnostic cloud.
    Zero, negative, and non-finite samples are treated as missing.  This helper
    is deliberately separate from the conservative collision map, which still
    consumes each original frame independently.
    """
    images = [np.asarray(image, dtype=float) for image in depth_images_mm]
    if not images:
        raise ValueError("at least one depth image is required")
    shape = images[0].shape
    if len(shape) != 2 or any(image.shape != shape for image in images):
        raise ValueError("stationary depth images must share one 2D shape")
    stack = np.stack(images, axis=0)
    valid = np.isfinite(stack) & (stack > 0.0)
    # Sorting with +inf sentinels avoids nanmedian's all-NaN warnings. Use the
    # lower observed middle sample for an even count rather than inventing a
    # depth halfway between two disagreeing physical returns.
    ordered = np.sort(np.where(valid, stack, np.inf), axis=0)
    counts = np.sum(valid, axis=0)
    median_index = np.maximum(0, (counts - 1) // 2)
    median = np.take_along_axis(
        ordered, median_index[None, ...], axis=0
    )[0]
    median[counts == 0] = 0.0
    return median.astype(np.float32)


def crop_dense_view_clouds(
    clouds: Iterable[tuple[np.ndarray, np.ndarray]], *,
    table_z_m: float,
    workspace_lo_m: Iterable[float],
    workspace_hi_m: Iterable[float],
    bounds_margin_m: float = 0.08,
    crop_below_table: bool = True,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Crop raw FK/hand-eye clouds without moving their geometry.

    ``build_dense_view_cloud`` historically snapped each view to whichever
    horizontal Z histogram mode was strongest in that image.  A wrist-angle
    change could make an object top win in one frame and the table win in the
    next, translating one complete scene by several centimetres.  A fused
    cloud then showed two copies of the same objects.

    Diagnostic multi-view fusion therefore preserves the raw calibrated base
    coordinates. The table calibration is used only as a crop plane, matching
    the collision map. A future cosmetic alignment, if needed, must be one
    externally measured transform shared by clouds, target points, and OBBs.
    """
    prepared = [
        (np.asarray(points, dtype=float).reshape(-1, 3),
         np.asarray(colors, dtype=np.uint8).reshape(-1, 3))
        for points, colors in clouds
    ]
    expected = float(table_z_m)
    lower = np.asarray(tuple(workspace_lo_m), dtype=float)
    upper = np.asarray(tuple(workspace_hi_m), dtype=float)
    if lower.shape != (3,) or upper.shape != (3,):
        raise ValueError("workspace bounds must be XYZ triplets")
    margin = float(np.clip(bounds_margin_m, 0.03, 0.30))
    cropped: list[tuple[np.ndarray, np.ndarray]] = []
    for points, colors in prepared:
        inside = np.isfinite(points).all(axis=1)
        inside &= np.all(points >= lower - margin, axis=1) \
            & np.all(points <= upper + margin, axis=1)
        if crop_below_table:
            inside &= points[:, 2] >= expected - 0.015
        cropped.append((points[inside], colors[inside]))
    return cropped


def fuse_dense_view_clouds(
    clouds: Iterable[tuple[np.ndarray, np.ndarray]], *,
    voxel_m: float = 0.0015,
) -> tuple[np.ndarray, np.ndarray]:
    """Union camera-colored diagnostic clouds in their shared base frame."""
    valid = [
        (np.asarray(points, dtype=float).reshape(-1, 3),
         np.asarray(colors, dtype=np.uint8).reshape(-1, 3))
        for points, colors in clouds if np.asarray(points).size
    ]
    if not valid:
        return np.empty((0, 3), dtype=float), np.empty((0, 3), dtype=np.uint8)
    points = np.vstack([item[0] for item in valid])
    colors = np.vstack([item[1] for item in valid])
    voxel = float(np.clip(voxel_m, 0.001, 0.008))
    keys = np.floor(points / voxel).astype(np.int64)
    _, inverse, counts = np.unique(
        keys, axis=0, return_inverse=True, return_counts=True
    )
    point_sum = np.zeros((len(counts), 3), dtype=float)
    color_sum = np.zeros((len(counts), 3), dtype=float)
    np.add.at(point_sum, inverse, points)
    np.add.at(color_sum, inverse, colors.astype(float))
    return (
        point_sum / counts[:, None],
        np.clip(color_sum / counts[:, None], 0, 255).astype(np.uint8),
    )


def register_dense_view_clouds(
    clouds: Iterable[tuple[np.ndarray, np.ndarray]], *,
    table_z_m: float,
    voxel_m: float = 0.004,
    max_translation_m: float = 0.012,
    adaptive_max_translation_m: float | None = None,
    minimum_overlap: float = 0.15,
    minimum_improvement: float = 0.015,
) -> tuple[list[tuple[np.ndarray, np.ndarray]], list[dict[str, Any]]]:
    """Remove small base-frame translation ghosts before diagnostic fusion.

    Wrist FK plus the saved rigid camera rotation provides the authoritative
    first transform.  Different wrist angles can nevertheless disagree by a
    few millimetres because of joint tracking and hand-eye calibration error.
    Search only a tightly bounded translation against *above-table* geometry;
    never deform an individual view and never align using the featureless table
    plane.  Consequently two objects stacked in one RGB-D frame retain their
    physical separation, while duplicate copies of the complete scene align.

    This is intentionally diagnostic registration.  The fail-closed collision
    map uses stationary frames and does not consume these corrected clouds.

    ``accepted`` means that a non-zero correction was justified. A view can
    also be ``valid_for_fusion`` when its uncorrected overlap is already strong.
    Conversely, a view which cannot be registered is returned unchanged for an
    explicitly enabled per-view debugger, but marked invalid so callers do not
    blindly union it into the fused cloud. Blindly unioning those failed views
    was the source of duplicated and fanned scene geometry.
    """
    prepared = [
        (np.asarray(points, dtype=float).reshape(-1, 3),
         np.asarray(colors, dtype=np.uint8).reshape(-1, 3))
        for points, colors in clouds
    ]
    if not prepared:
        return [], []
    voxel = float(np.clip(voxel_m, 0.003, 0.008))
    maximum = float(np.clip(max_translation_m, 0.0, 0.015))
    max_step = int(np.floor(maximum / voxel + 1e-9))
    adaptive_maximum = maximum
    if adaptive_max_translation_m is not None:
        adaptive_maximum = float(np.clip(
            adaptive_max_translation_m, maximum, 0.024
        ))
    adaptive_max_step = int(np.floor(adaptive_maximum / voxel + 1e-9))
    table_clearance = float(table_z_m) + max(0.010, 2.5 * voxel)

    def occupied(points: np.ndarray) -> np.ndarray:
        selected = points[np.isfinite(points).all(axis=1)]
        selected = selected[selected[:, 2] >= table_clearance]
        if not len(selected):
            return np.empty((0, 3), dtype=np.int64)
        keys = np.unique(np.floor(selected / voxel).astype(np.int64), axis=0)
        # Bound registration work without introducing random run-to-run drift.
        if len(keys) > 20000:
            keys = keys[np.linspace(0, len(keys) - 1, 20000).astype(int)]
        return keys

    reference_keys = occupied(prepared[0][0])
    reference_set = {tuple(int(value) for value in key)
                     for key in reference_keys}
    registered = [prepared[0]]
    reports: list[dict[str, Any]] = [{
        "view": 0,
        "accepted": True,
        "valid_for_fusion": True,
        "translation_mm": [0.0, 0.0, 0.0],
        "reason": "anchor view",
        "above_table_voxels": int(len(reference_keys)),
    }]
    shifts = [
        np.array([dx, dy, dz], dtype=np.int64)
        for dx in range(-max_step, max_step + 1)
        for dy in range(-max_step, max_step + 1)
        for dz in range(-max_step, max_step + 1)
    ] if max_step else [np.zeros(3, dtype=np.int64)]
    shifts.sort(key=lambda shift: (int(np.dot(shift, shift)),
                                   int(abs(shift[2])),
                                   tuple(int(v) for v in shift)))

    for view_index, (points, colors) in enumerate(prepared[1:], 1):
        moving_keys = occupied(points)
        if len(reference_set) < 40 or len(moving_keys) < 40:
            registered.append((points, colors))
            reports.append({
                "view": view_index, "accepted": False,
                "valid_for_fusion": False,
                "translation_mm": [0.0, 0.0, 0.0],
                "reason": "insufficient shared above-table geometry",
                "above_table_voxels": int(len(moving_keys)),
            })
            continue

        def overlap(shift: np.ndarray) -> int:
            return sum(
                (int(key[0] + shift[0]), int(key[1] + shift[1]),
                 int(key[2] + shift[2])) in reference_set
                for key in moving_keys
            )

        zero_count = overlap(np.zeros(3, dtype=np.int64))
        best_shift = np.zeros(3, dtype=np.int64)
        best_count = zero_count
        for shift in shifts[1:]:
            count = overlap(shift)
            if count > best_count:
                best_count = count
                best_shift = shift
        denominator = float(max(1, min(len(reference_set), len(moving_keys))))
        zero_ratio = zero_count / denominator
        best_ratio = best_count / denominator
        initial_best_ratio = best_ratio
        initial_pinned = bool(
            max_step and int(np.max(np.abs(best_shift))) >= max_step
        )
        adaptive_search = False
        # A boundary optimum can be a real hand-eye/FK translation just beyond
        # the normal diagnostic window. Expand the search only when the first
        # pass has strong occupancy evidence, and only for callers that opted
        # into the larger bound. This keeps the common path fast and prevents a
        # weakly related scene cluster from being dragged across the workspace.
        strong_overlap = max(float(minimum_overlap) + 0.10, 0.25)
        strong_improvement = max(4.0 * float(minimum_improvement), 0.08)
        if (
            initial_pinned
            and adaptive_max_step > max_step
            and best_ratio >= strong_overlap
            and best_ratio - zero_ratio >= strong_improvement
        ):
            adaptive_search = True
            extended_shifts = [
                np.array([dx, dy, dz], dtype=np.int64)
                for dx in range(-adaptive_max_step, adaptive_max_step + 1)
                for dy in range(-adaptive_max_step, adaptive_max_step + 1)
                for dz in range(-adaptive_max_step, adaptive_max_step + 1)
                if max(abs(dx), abs(dy), abs(dz)) > max_step
            ]
            extended_shifts.sort(key=lambda shift: (
                int(np.dot(shift, shift)), int(abs(shift[2])),
                tuple(int(v) for v in shift)
            ))
            for shift in extended_shifts:
                count = overlap(shift)
                if count > best_count:
                    best_count = count
                    best_shift = shift
            best_ratio = best_count / denominator
        # A translation sitting on the edge of the search grid is a clamp, not a
        # fit: the best alignment found is the furthest one allowed, so the real
        # optimum lies outside the bounded search and this view is simply not
        # registered. Accepting it drags the view to the bound in every pinned
        # axis, and the object box fitted to the union then inflates by the
        # spread between views. One such view took a measured 87x35x25mm object
        # to 97x105x91mm and put every axis beyond the jaw aperture.
        effective_max_step = adaptive_max_step if adaptive_search else max_step
        effective_maximum = effective_max_step * voxel
        pinned_to_bound = bool(
            effective_max_step
            and int(np.max(np.abs(best_shift))) >= effective_max_step
        )
        accepted = bool(
            np.any(best_shift)
            and not pinned_to_bound
            and best_ratio >= float(minimum_overlap)
            and best_ratio - zero_ratio >= float(minimum_improvement)
        )
        # A zero correction is not a failed registration when the views already
        # share enough occupied geometry. Keep that case distinct from a view
        # whose FK/hand-eye transform has no registration consensus at all.
        valid_for_fusion = bool(
            accepted or zero_ratio >= float(minimum_overlap)
        )
        translation = best_shift.astype(float) * voxel if accepted else np.zeros(3)
        corrected = points + translation
        registered.append((corrected, colors))
        reports.append({
            "view": view_index,
            "accepted": accepted,
            "valid_for_fusion": valid_for_fusion,
            "translation_mm": (translation * 1000.0).round(1).tolist(),
            "zero_overlap": round(zero_ratio, 4),
            "best_overlap": round(best_ratio, 4),
            "above_table_voxels": int(len(moving_keys)),
            "pinned_to_bound": pinned_to_bound,
            "initial_pinned_to_bound": initial_pinned,
            "adaptive_search": adaptive_search,
            "initial_best_overlap": round(initial_best_ratio, 4),
            "search_bound_mm": round(effective_maximum * 1000.0, 1),
            "reason": (
                "adaptive bounded above-table occupancy registration accepted"
                if accepted and adaptive_search else
                "bounded above-table occupancy registration accepted"
                if accepted else
                "raw FK/hand-eye transform already has sufficient overlap"
                if valid_for_fusion else
                "view excluded from fusion; best translation sat on the "
                f"{effective_maximum * 1000.0:.0f}mm search bound, so the alignment did "
                "not converge"
                if pinned_to_bound else
                "view excluded from fusion; FK/hand-eye transform and bounded "
                "translation lacked registration consensus"
            ),
        })
    return registered, reports


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


def save_scene_artifacts(
    scene: SceneMap, output_dir: str | Path, *, stem: str = "scene_collision",
    view_points_m: np.ndarray | None = None,
    view_colors_rgb: np.ndarray | None = None,
    named_view_clouds: dict[str, tuple[np.ndarray, np.ndarray]] | None = None,
    object_boxes_3d: dict[str, dict[str, Any]] | None = None,
    show_box_labels: bool = False,
) -> dict[str, Any]:
    """Save a universal PLY and, when available, an interactive Rerun recording."""
    outdir = Path(output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    ply_path = outdir / f"{stem}.ply"
    dense_points = np.asarray(
        scene.points_m if view_points_m is None else view_points_m, dtype=float
    ).reshape(-1, 3)
    if view_colors_rgb is None:
        dense_colors = np.tile(
            np.array([[255, 170, 0]], dtype=np.uint8), (len(dense_points), 1)
        )
    else:
        dense_colors = np.asarray(view_colors_rgb, dtype=np.uint8).reshape(-1, 3)
        if len(dense_colors) != len(dense_points):
            raise ValueError("view point and RGB color counts differ")
    vertices: list[tuple[float, float, float, int, int, int]] = [
        (float(point[0]), float(point[1]), float(point[2]),
         int(rgb[0]), int(rgb[1]), int(rgb[2]))
        for point, rgb in zip(dense_points, dense_colors)
    ]
    # Red corners show the actual inflated collision volume rather than only the
    # raw orange depth returns.
    for box in scene.boxes:
        center = np.asarray(box.center_m, dtype=float)
        half = np.asarray(box.size_m, dtype=float) / 2.0
        for sx in (-1.0, 1.0):
            for sy in (-1.0, 1.0):
                for sz in (-1.0, 1.0):
                    point = center + half * np.array([sx, sy, sz])
                    vertices.append((*[float(v) for v in point], 255, 40, 40))
    valid_object_boxes: list[tuple[str, np.ndarray, np.ndarray, np.ndarray]] = []
    for name, detail in (object_boxes_3d or {}).items():
        try:
            center = np.asarray(detail["center_m"], dtype=float)
            size = np.asarray(detail["conservative_size_m"], dtype=float)
            axes = np.asarray(detail["axes_base"], dtype=float).T
            if center.shape != (3,) or size.shape != (3,) or axes.shape != (3, 3):
                raise ValueError("invalid OBB dimensions")
            if not np.all(np.isfinite(center)) or not np.all(size > 0):
                raise ValueError("invalid OBB values")
        except (KeyError, TypeError, ValueError):
            continue
        valid_object_boxes.append((str(name), center, size, axes))
        half = size / 2.0
        for sx in (-1.0, 1.0):
            for sy in (-1.0, 1.0):
                for sz in (-1.0, 1.0):
                    point = center + axes @ (half * np.array([sx, sy, sz]))
                    vertices.append((*[float(v) for v in point], 40, 255, 80))
    # A sparse blue table grid gives the cloud an immediately readable reference.
    if scene.boxes:
        centres = np.asarray([box.center_m for box in scene.boxes], dtype=float)
        x_min, y_min = np.min(centres[:, :2], axis=0) - 0.05
        x_max, y_max = np.max(centres[:, :2], axis=0) + 0.05
        for x in np.linspace(x_min, x_max, 12):
            for y in np.linspace(y_min, y_max, 12):
                vertices.append((float(x), float(y), scene.table_z_m,
                                 50, 120, 255))
    header = [
        "ply", "format ascii 1.0", f"element vertex {len(vertices)}",
        "property float x", "property float y", "property float z",
        "property uchar red", "property uchar green", "property uchar blue",
        "end_header",
    ]
    body = [
        f"{x:.6f} {y:.6f} {z:.6f} {red} {green} {blue}"
        for x, y, z, red, green, blue in vertices
    ]
    ply_path.write_text("\n".join([*header, *body]) + "\n")

    result: dict[str, Any] = {
        "ply": str(ply_path),
        "ply_vertices": len(vertices),
        "colors": {
            "rgb": "dense denoised camera-colored scene points",
            "red": "inflated obstacle-box corners",
            "green": "metric semantic object OBB corners",
            "blue": "calibrated table plane",
        },
        "object_boxes": [name for name, *_ in valid_object_boxes],
        "collision_box_labels_shown": bool(show_box_labels),
    }
    try:
        import rerun as rr

        rrd_path = outdir / f"{stem}.rrd"
        rr.init("rebot_rs_collision_scene", spawn=False)
        rr.save(str(rrd_path))
        if len(dense_points):
            rr.log(
                "scene/dense_rgb_cloud",
                rr.Points3D(dense_points, colors=dense_colors, radii=0.0015),
                static=True,
            )
        logged_views = []
        for name, (view_points, view_colors) in (named_view_clouds or {}).items():
            points = np.asarray(view_points, dtype=float).reshape(-1, 3)
            colors = np.asarray(view_colors, dtype=np.uint8).reshape(-1, 3)
            if not len(points) or len(points) != len(colors):
                continue
            safe_name = "".join(
                character if character.isalnum() or character in "_-" else "_"
                for character in str(name)
            )
            rr.log(
                f"scene/views/{safe_name}",
                rr.Points3D(points, colors=colors, radii=0.0012),
                static=True,
            )
            logged_views.append(str(name))
        if scene.points_m:
            rr.log(
                "collision/raw_obstacle_returns",
                rr.Points3D(scene.points_m, colors=[255, 170, 0], radii=0.0025),
                static=True,
            )
        if scene.boxes:
            box_labels = [f"scene_box_{index}"
                          for index in range(len(scene.boxes))]
            box_colors = [
                [255, 0, 255, 150] if index == 0 else [255, 40, 40, 90]
                for index in range(len(scene.boxes))
            ]
            rr.log(
                "collision/inflated_obstacles",
                rr.Boxes3D(
                    centers=[box.center_m for box in scene.boxes],
                    sizes=[box.size_m for box in scene.boxes],
                    colors=box_colors,
                    labels=box_labels,
                    show_labels=bool(show_box_labels),
                    fill_mode="solid",
                ),
                static=True,
            )
        for name, center, size, axes in valid_object_boxes:
            yaw = float(np.arctan2(axes[1, 0], axes[0, 0]))
            safe_name = "".join(
                character if character.isalnum() or character in "_-" else "_"
                for character in name
            )
            rr.log(
                f"objects/{safe_name}",
                rr.Boxes3D(
                    centers=[center.tolist()], sizes=[size.tolist()],
                    rotation_axis_angles=[rr.RotationAxisAngle(
                        [0.0, 0.0, 1.0], radians=yaw
                    )],
                    colors=[[40, 255, 80, 110]], labels=[name],
                    show_labels=True, fill_mode="solid",
                ),
                static=True,
            )
        table_centers = np.asarray([
            box.center_m for box in scene.boxes
        ], dtype=float)
        if table_centers.size:
            lo = np.min(table_centers[:, :2], axis=0) - 0.05
            hi = np.max(table_centers[:, :2], axis=0) + 0.05
            rr.log(
                "scene/table",
                rr.Boxes3D(
                    centers=[[(lo[0] + hi[0]) / 2.0,
                              (lo[1] + hi[1]) / 2.0,
                              scene.table_z_m - 0.0025]],
                    sizes=[[hi[0] - lo[0], hi[1] - lo[1], 0.005]],
                    colors=[50, 120, 255, 100], fill_mode="solid",
                ),
                static=True,
            )
        rr.log(
            "robot_frame/axes",
            rr.Arrows3D(
                origins=[[0.0, 0.0, 0.0]] * 3,
                vectors=[[0.10, 0.0, 0.0], [0.0, 0.10, 0.0],
                         [0.0, 0.0, 0.10]],
                colors=[[255, 0, 0], [0, 255, 0], [0, 100, 255]],
                labels=["+X front", "+Y left", "+Z up"],
            ),
            static=True,
        )
        result["rerun"] = str(rrd_path)
        result["open_command"] = f"rerun {rrd_path}"
        result["named_views"] = logged_views
    except Exception as exc:  # noqa: BLE001 - PLY remains a valid artifact
        result["rerun_error"] = f"{type(exc).__name__}: {exc}"
    return result
