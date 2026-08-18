"""Metric, conservative 3D object geometry for table manipulation.

Semantic models decide which pixels belong to an object.  Aligned RealSense
depth and the calibrated wrist transform remain authoritative for scale.  A
learned completion prior may expand missing geometry, but can never shrink an
extent already observed by RGB-D.
"""
from __future__ import annotations

import base64
import os
from dataclasses import dataclass
from typing import Any, Iterable

import cv2
import numpy as np


@dataclass(frozen=True)
class ObjectBox3D:
    center_m: tuple[float, float, float]
    axes_base: tuple[tuple[float, float, float], ...]
    observed_size_m: tuple[float, float, float]
    conservative_size_m: tuple[float, float, float]
    uncertainty_m: tuple[float, float, float]
    support_z_m: float
    point_count: int
    views: int
    source: str = "multiframe_rgbd_semantic_mask"
    prior: dict[str, Any] | None = None
    support_completed: bool = False
    orientation_mode: str = "measured_3d_pca"

    def as_dict(self) -> dict[str, Any]:
        return {
            "center_m": [round(float(v), 5) for v in self.center_m],
            "axes_base": [[round(float(v), 6) for v in axis]
                          for axis in self.axes_base],
            "observed_size_m": [round(float(v), 5)
                                for v in self.observed_size_m],
            "conservative_size_m": [round(float(v), 5)
                                    for v in self.conservative_size_m],
            "uncertainty_m": [round(float(v), 5) for v in self.uncertainty_m],
            "support_z_m": round(float(self.support_z_m), 5),
            "point_count": int(self.point_count),
            "views": int(self.views),
            "source": self.source,
            "prior": self.prior,
            "support_completed": bool(self.support_completed),
            "orientation_mode": self.orientation_mode,
        }


def _intrinsics(intr: Any) -> tuple[float, float, float, float]:
    if isinstance(intr, (tuple, list)) and len(intr) >= 4:
        return tuple(float(v) for v in intr[:4])  # type: ignore[return-value]
    return tuple(float(getattr(intr, name))
                 for name in ("fx", "fy", "ppx", "ppy"))  # type: ignore[return-value]


def _depth_connected_box_fallback(
    depth: np.ndarray, detection: Any, *, near_mm: float, far_mm: float,
) -> np.ndarray:
    """Recover a bounded component when a legacy detection has no pixel mask.

    This compatibility path keeps simulation and older detectors working, but
    does not repeat the unsafe historical behaviour of treating every valid
    depth pixel in an axis-aligned box as object geometry.  Growth is confined
    to the detection box and cannot cross a sharp local depth discontinuity.
    """
    h, w = depth.shape
    x1, y1, x2, y2 = (int(v) for v in detection.box)
    inside = np.zeros((h, w), dtype=bool)
    inside[max(0, y1):min(h, y2), max(0, x1):min(w, x2)] = True
    eligible = inside & np.isfinite(depth) \
        & (depth >= float(near_mm)) & (depth <= float(far_mm))
    if np.count_nonzero(eligible) < 30:
        raise ValueError("legacy detection box has fewer than 30 valid depth pixels")
    seed_u = int(np.clip(round(float(detection.uv[0])), 0, w - 1))
    seed_v = int(np.clip(round(float(detection.uv[1])), 0, h - 1))
    if not eligible[seed_v, seed_u]:
        vv, uu = np.nonzero(eligible)
        nearest = int(np.argmin((uu - seed_u) ** 2 + (vv - seed_v) ** 2))
        seed_u, seed_v = int(uu[nearest]), int(vv[nearest])
    edge_mm = float(np.clip(float(os.getenv(
        "REBOT_OBJECT_BOX_DEPTH_EDGE_MM", "12"
    )), 5.0, 25.0))
    flood_mask = np.ones((h + 2, w + 2), dtype=np.uint8)
    flood_mask[1:-1, 1:-1][eligible] = 0
    cv2.floodFill(
        np.asarray(depth, dtype=np.float32).copy(), flood_mask,
        (seed_u, seed_v), 0.0,
        loDiff=edge_mm, upDiff=edge_mm,
        flags=8 | cv2.FLOODFILL_MASK_ONLY | (255 << 8),
    )
    component = flood_mask[1:-1, 1:-1] == 255
    if np.count_nonzero(component) < 30:
        raise ValueError("legacy detection depth component has fewer than 30 pixels")
    return component


def masked_points_base(
    frame: tuple[np.ndarray, np.ndarray, Any], detection: Any, *,
    ee_xyz_m: Iterable[float], camera_mapping_mm_per_m: np.ndarray,
    tool_camera_mm: Iterable[float], near_mm: float = 60.0,
    far_mm: float = 650.0,
) -> np.ndarray:
    """Lift one semantic object mask into the calibrated robot base frame."""
    _color, depth_raw, intr = frame
    depth = np.asarray(depth_raw, dtype=float)
    mask = detection.info.get("mask_u8") if hasattr(detection, "info") else None
    if mask is None:
        # Compatibility for legacy/simulation detections.  Use only the
        # target-seeded depth component inside the box, never the entire box.
        mask = _depth_connected_box_fallback(
            depth, detection, near_mm=near_mm, far_mm=far_mm
        ).astype(np.uint8) * 255
    mask = np.asarray(mask) > 0
    if mask.shape != depth.shape:
        raise ValueError("semantic mask and depth image dimensions differ")
    # Defence in depth: metric geometry may only consume pixels inside the
    # cleaned detection returned by wrist_detect.  This prevents a stale or
    # malformed backend mask from silently producing an enormous payload box.
    x1, y1, x2, y2 = (int(v) for v in detection.box)
    inside_box = np.zeros(depth.shape, dtype=bool)
    inside_box[max(0, y1):min(depth.shape[0], y2),
               max(0, x1):min(depth.shape[1], x2)] = True
    mask &= inside_box
    # Erode only one pixel to reject mixed foreground/background depth edges.
    mask = cv2.erode(mask.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    valid = mask & np.isfinite(depth) & (depth >= float(near_mm)) \
        & (depth <= float(far_mm))
    if np.count_nonzero(valid) < 30:
        raise ValueError("semantic mask has fewer than 30 valid depth pixels")
    fx, fy, cx, cy = _intrinsics(intr)
    vv, uu = np.nonzero(valid)
    z = depth[vv, uu]
    points_cam = np.column_stack((
        (uu.astype(float) - cx) * z / fx,
        (vv.astype(float) - cy) * z / fy,
        z,
    ))
    mapping = np.asarray(camera_mapping_mm_per_m, dtype=float)
    tool = np.asarray(tuple(tool_camera_mm), dtype=float)
    ee = np.asarray(tuple(ee_xyz_m), dtype=float)
    if mapping.shape != (3, 3) or abs(float(np.linalg.det(mapping))) < 1e-6:
        raise ValueError("camera mapping is singular")
    return ee + np.linalg.solve(mapping, (tool - points_cam).T).T


def consistent_view_extent_indices(
    point_sets_m: Iterable[np.ndarray], *, table_z_m: float,
    absolute_tolerance_m: float = 0.008,
    relative_tolerance: float = 0.25,
) -> tuple[list[int], dict[str, Any]]:
    """Select a repeatable metric extent cluster from stationary RGB-D views.

    Extents are sorted so harmless PCA axis permutations do not look like
    shape changes.  At least two views must agree; a contaminated mask cannot
    become payload/collision geometry merely because its centroid was stable.
    """
    point_sets = [np.asarray(points, dtype=float).reshape(-1, 3)
                  for points in point_sets_m]
    signatures: list[np.ndarray | None] = []
    failures: list[str] = []
    for index, points in enumerate(point_sets):
        try:
            single = fit_tabletop_box([points], table_z_m=table_z_m)
            signatures.append(np.sort(np.asarray(
                single.observed_size_m, dtype=float
            )))
        except Exception as exc:  # noqa: BLE001
            signatures.append(None)
            failures.append(f"view {index + 1}: {type(exc).__name__}: {exc}")
    best: list[int] = []
    for anchor_index, anchor in enumerate(signatures):
        if anchor is None:
            continue
        cluster = []
        for candidate_index, candidate in enumerate(signatures):
            if candidate is None:
                continue
            limit = np.maximum(
                float(absolute_tolerance_m),
                float(relative_tolerance) * np.maximum(anchor, candidate),
            )
            if np.all(np.abs(candidate - anchor) <= limit):
                cluster.append(candidate_index)
        if len(cluster) > len(best):
            best = cluster
    detail = {
        "ok": len(best) >= 2,
        "accepted_indices": best,
        "extent_mm": [
            None if signature is None else
            np.round(signature * 1000.0, 1).tolist()
            for signature in signatures
        ],
        "absolute_tolerance_mm": float(absolute_tolerance_m) * 1000.0,
        "relative_tolerance": float(relative_tolerance),
        "failures": failures,
    }
    detail["reason"] = (
        f"{len(best)}/{len(point_sets)} stationary views have consistent "
        "orientation-independent metric extents"
        if detail["ok"] else
        "fewer than two stationary views agree on metric object extents"
    )
    return best, detail


def _rectangle_axes(points_xy_m: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rect = cv2.minAreaRect((points_xy_m * 1000.0).astype(np.float32))
    corners = cv2.boxPoints(rect).astype(float) / 1000.0
    edge_a = corners[1] - corners[0]
    edge_b = corners[2] - corners[1]
    length_a = float(np.linalg.norm(edge_a))
    length_b = float(np.linalg.norm(edge_b))
    if min(length_a, length_b) < 1e-4:
        raise ValueError("object footprint is degenerate")
    axis_a = edge_a / length_a
    axis_b = edge_b / length_b
    if np.linalg.det(np.array([
        [axis_a[0], axis_b[0]], [axis_a[1], axis_b[1]]
    ])) < 0:
        axis_b = -axis_b
    return np.asarray(rect[0], dtype=float) / 1000.0, \
        np.array([axis_a[0], axis_a[1], 0.0]), \
        np.array([axis_b[0], axis_b[1], 0.0])


def fit_tabletop_box(
    point_sets_m: Iterable[np.ndarray], *, table_z_m: float,
    prior_vertices: np.ndarray | None = None,
) -> ObjectBox3D:
    """Fit a measured-only 3D OBB to one or more metric RGB-D views.

    ``table_z_m`` is a rejection boundary, not a shape-completion prior.  Older
    code forced every target down to that plane, which could turn a partial or
    elevated observation into a very tall synthetic box.  Hidden geometry is
    now represented only by bounded uncertainty; it is never invented.
    """
    views = [np.asarray(points, dtype=float).reshape(-1, 3)
             for points in point_sets_m if np.asarray(points).size]
    if not views:
        raise ValueError("no metric object observations")
    points = np.vstack(views)
    points = points[np.all(np.isfinite(points), axis=1)]
    points = points[points[:, 2] >= float(table_z_m) - 0.004]
    if len(points) < 50:
        raise ValueError("fewer than 50 above-support object points")
    # Robustly trim isolated RealSense flying pixels, then fit all three axes
    # from the measurements.  A full 3-D PCA box handles standing and tilted
    # objects; fixing one axis to table Z was the source of the projected-box
    # spill seen on long objects.
    median = np.median(points, axis=0)
    radial = np.linalg.norm(points - median, axis=1)
    points = points[radial <= np.percentile(radial, 99.0)]
    centered = points - np.mean(points, axis=0)
    covariance = centered.T @ centered / max(1, len(centered) - 1)
    eigenvalues, axes = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    axes = axes[:, order]
    if float(np.linalg.det(axes)) < 0.0:
        axes[:, 2] *= -1.0
    orientation_mode = "measured_3d_pca"
    # Gravity is a real calibrated scene direction, not an inferred object
    # dimension.  If PCA already finds an axis close to gravity, snap that one
    # axis to the table normal and refit the two measured horizontal axes. This
    # removes unstable 10--20 degree box lean caused by one-sided RGB-D views,
    # while genuinely tilted objects remain unconstrained.
    vertical_index = int(np.argmax(np.abs(axes[2, :])))
    vertical_alignment = float(np.clip(
        abs(axes[2, vertical_index]), 0.0, 1.0
    ))
    vertical_error_deg = float(np.degrees(np.arccos(vertical_alignment)))
    snap_limit_deg = float(np.clip(float(os.getenv(
        "REBOT_OBJECT_BOX_GRAVITY_SNAP_DEG", "25"
    )), 5.0, 30.0))
    if vertical_error_deg <= snap_limit_deg:
        _center_xy, axis_a, axis_b = _rectangle_axes(points[:, :2])
        vertical_sign = 1.0 if axes[2, vertical_index] >= 0.0 else -1.0
        vertical = np.array([0.0, 0.0, vertical_sign])
        axes = np.column_stack((axis_a, axis_b, vertical))
        if float(np.linalg.det(axes)) < 0.0:
            axes[:, 1] *= -1.0
        orientation_mode = (
            f"gravity_snapped_measured_obb({vertical_error_deg:.1f}deg)"
        )
    projections = points @ axes
    lo = np.percentile(projections, 1.0, axis=0)
    hi = np.percentile(projections, 99.0, axis=0)
    observed = np.maximum(hi - lo, 0.003)
    center = axes @ ((lo + hi) / 2.0)
    # Repeated depth frames reduce random error but do not reveal a hidden side.
    # Keep at least 3mm/axis uncertainty even with multiple views.
    uncertainty = np.full(3, 0.005 if len(views) == 1 else 0.003, dtype=float)
    conservative = observed + 2.0 * uncertainty
    prior_detail = None
    if prior_vertices is not None:
        conservative, prior_detail = expand_with_shape_prior(
            observed, conservative, prior_vertices
        )
    return ObjectBox3D(
        center_m=tuple(float(value) for value in center),
        axes_base=tuple(tuple(float(v) for v in axes[:, index])
                        for index in range(3)),
        observed_size_m=tuple(float(v) for v in observed),
        conservative_size_m=tuple(float(v) for v in conservative),
        uncertainty_m=tuple(float(v) for v in uncertainty),
        support_z_m=float(table_z_m), point_count=int(len(points)),
        views=len(views), prior=prior_detail,
        source="multiframe_rgbd_measured_obb", support_completed=False,
        orientation_mode=orientation_mode,
    )


def fit_tabletop_box_with_axes(
    point_sets_m: Iterable[np.ndarray], *, table_z_m: float,
    axes_base: np.ndarray, orientation_mode: str,
) -> ObjectBox3D:
    """Refit measured extents while preserving a trusted orthonormal frame.

    Multi-view translation registration can shear a small object enough for a
    fresh PCA to rotate its axes even though independent views already agreed
    on their physical direction.  This helper does not invent dimensions: it
    projects the same robustly trimmed measured points onto the supplied frame
    and recomputes every extent and the centre from those measurements.
    """
    views = [np.asarray(points, dtype=float).reshape(-1, 3)
             for points in point_sets_m if np.asarray(points).size]
    if not views:
        raise ValueError("no metric object observations")
    points = np.vstack(views)
    points = points[np.all(np.isfinite(points), axis=1)]
    points = points[points[:, 2] >= float(table_z_m) - 0.004]
    if len(points) < 50:
        raise ValueError("fewer than 50 above-support object points")
    axes = np.asarray(axes_base, dtype=float)
    if axes.shape != (3, 3) or not np.all(np.isfinite(axes)):
        raise ValueError("trusted object axes are invalid")
    gram = axes.T @ axes
    if not np.allclose(gram, np.eye(3), atol=1e-3) \
            or abs(float(np.linalg.det(axes))) < 0.99:
        raise ValueError("trusted object axes are not orthonormal")
    median = np.median(points, axis=0)
    radial = np.linalg.norm(points - median, axis=1)
    points = points[radial <= np.percentile(radial, 99.0)]
    projections = points @ axes
    lo = np.percentile(projections, 1.0, axis=0)
    hi = np.percentile(projections, 99.0, axis=0)
    observed = np.maximum(hi - lo, 0.003)
    center = axes @ ((lo + hi) / 2.0)
    uncertainty = np.full(3, 0.005 if len(views) == 1 else 0.003, dtype=float)
    conservative = observed + 2.0 * uncertainty
    return ObjectBox3D(
        center_m=tuple(float(value) for value in center),
        axes_base=tuple(tuple(float(v) for v in axes[:, index])
                        for index in range(3)),
        observed_size_m=tuple(float(v) for v in observed),
        conservative_size_m=tuple(float(v) for v in conservative),
        uncertainty_m=tuple(float(v) for v in uncertainty),
        support_z_m=float(table_z_m), point_count=int(len(points)),
        views=len(views), prior=None,
        source="multiframe_rgbd_measured_obb", support_completed=False,
        orientation_mode=str(orientation_mode),
    )


def expand_with_shape_prior(
    observed_size_m: np.ndarray, conservative_size_m: np.ndarray,
    vertices: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Scale a shape-only mesh without ever shrinking measured dimensions."""
    verts = np.asarray(vertices, dtype=float).reshape(-1, 3)
    if len(verts) < 20 or not np.all(np.isfinite(verts)):
        raise ValueError("shape prior contains too few finite vertices")
    centered = verts - np.median(verts, axis=0)
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    projected = centered @ vh.T
    prior_extent = np.ptp(np.percentile(projected, [1.0, 99.0], axis=0), axis=0)
    prior_sorted = np.sort(prior_extent)
    observed_order = np.argsort(observed_size_m)
    observed_sorted = observed_size_m[observed_order]
    if np.min(prior_sorted) <= 1e-8:
        raise ValueError("shape prior has a degenerate axis")
    scale = float(np.max(observed_sorted / prior_sorted))
    expanded_sorted = prior_sorted * scale
    # Reject a wildly incompatible generative completion instead of allowing it
    # to turn the whole scene into occupied payload space.
    if np.any(expanded_sorted > np.maximum(0.30, observed_sorted * 2.5)):
        raise ValueError("shape prior is incompatible with metric RGB-D extents")
    expanded = np.asarray(conservative_size_m, dtype=float).copy()
    expanded[observed_order] = np.maximum(
        expanded[observed_order], expanded_sorted
    )
    return expanded, {
        "used": True, "metric_scale": scale,
        "normalized_prior_extents": prior_extent.round(5).tolist(),
        "rule": "prior may expand but never shrink RGB-D extents",
    }


def request_sam3d_prior(color_bgr: np.ndarray, mask_u8: np.ndarray) -> np.ndarray | None:
    """Optional external SAM 3D service; disabled on the local 8GB GPU.

    The endpoint must return JSON ``{"vertices": [[x,y,z], ...]}``. Keeping it
    out of the arm process prevents a 32GB model from starving control/vision.
    """
    endpoint = os.getenv("REBOT_SAM3D_ENDPOINT", "").strip()
    if not endpoint:
        return None
    import httpx
    ok_image, image_bytes = cv2.imencode(".jpg", color_bgr)
    ok_mask, mask_bytes = cv2.imencode(".png", np.asarray(mask_u8, dtype=np.uint8))
    if not ok_image or not ok_mask:
        raise RuntimeError("could not encode SAM 3D request")
    payload = {
        "image_jpeg_base64": base64.b64encode(image_bytes).decode("ascii"),
        "mask_png_base64": base64.b64encode(mask_bytes).decode("ascii"),
    }
    # This is advisory geometry completion inside a live robot process. Never
    # let a remote model hold planning (and therefore operator recovery) for a
    # long inference timeout; metric RGB-D remains the safe fallback.
    timeout_s = float(np.clip(
        float(os.getenv("REBOT_SAM3D_TIMEOUT_S", "12")), 2.0, 30.0
    ))
    response = httpx.post(
        endpoint, json=payload,
        timeout=httpx.Timeout(timeout_s, connect=min(3.0, timeout_s)),
    )
    response.raise_for_status()
    vertices = np.asarray(response.json().get("vertices"), dtype=float)
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise RuntimeError("SAM 3D service returned invalid vertices")
    return vertices
