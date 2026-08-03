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
        points_m=tuple(tuple(float(value) for value in point) for point in points),
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
    near_mm: float = 60.0,
    far_mm: float = 650.0,
    stride_px: int = 1,
    voxel_m: float = 0.002,
    bounds_margin_m: float = 0.15,
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
    valid &= support >= 2

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
    margin = float(np.clip(bounds_margin_m, 0.03, 0.30))
    inside = np.all(base_points >= lower - margin, axis=1) \
        & np.all(base_points <= upper + margin, axis=1)
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
            "blue": "calibrated table plane",
        },
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
        if scene.points_m:
            rr.log(
                "collision/raw_obstacle_returns",
                rr.Points3D(scene.points_m, colors=[255, 170, 0], radii=0.0025),
                static=True,
            )
        if scene.boxes:
            rr.log(
                "collision/inflated_obstacles",
                rr.Boxes3D(
                    centers=[box.center_m for box in scene.boxes],
                    sizes=[box.size_m for box in scene.boxes],
                    colors=[255, 40, 40, 90],
                    fill_mode="solid",
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
    except Exception as exc:  # noqa: BLE001 - PLY remains a valid artifact
        result["rerun_error"] = f"{type(exc).__name__}: {exc}"
    return result
