"""Object detection for the wrist (eye-in-hand) RGB-D camera.

Named objects use a purpose-built Grounding DINO detector when GPU vision is
enabled; SAM 2.1 supplies their whole-object masks.  A generous HSV band plus a
depth gate remains the fast deterministic fallback and the default when GPU
vision is disabled.

The depth gate is what makes this trustworthy on a *wrist* camera.  An eye-in-hand
view is dominated by two things a top-down view never contains: the gripper's own
jaws (very close) and the room beyond the table edge (far).  Restricting detection
to a plausible grasp range removes both before contours are ever considered.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np

Intrinsics = tuple[float, float, float, float]

# Generous HSV bands for common object colours (OpenCV H is 0-179).  Red wraps the
# hue circle and therefore needs two bands.  Shared vocabulary with
# ``reachy_color_pick.COLORS``; kept local so this module has no cross-demo import.
COLORS: dict[str, list[tuple[tuple[int, int, int], tuple[int, int, int]]]] = {
    "red":    [((0, 110, 80), (10, 255, 255)), ((170, 110, 80), (180, 255, 255))],
    "orange": [((8, 130, 130), (22, 255, 255))],
    "yellow": [((22, 110, 110), (34, 255, 255))],
    "green":  [((38, 70, 50), (85, 255, 255))],
    # The D405's auto white balance gives neutral walls/tables a cyan cast around
    # H=96, S=120. Keep that out: the blue marker is vivid (roughly H=103, S=255).
    "blue":   [((98, 150, 50), (130, 255, 255))],
    "purple": [((130, 60, 50), (160, 255, 255))],
    "pink":   [((160, 80, 120), (172, 255, 255))],
}
# Words that name a colour without being one.
_ALIASES = {"magenta": "purple", "violet": "purple", "cyan": "blue",
            "lime": "green", "gold": "yellow", "amber": "orange"}


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, "") or default)
    except ValueError:
        return default


# A D405 on the wrist works from ~70 mm; the jaws sit nearer than that when open.
NEAR_MM = _env_float("REBOT_WRIST_NEAR_MM", 90.0)
FAR_MM = _env_float("REBOT_WRIST_FAR_MM", 600.0)
MIN_AREA_PX = _env_float("REBOT_WRIST_MIN_AREA_PX", 200.0)


@dataclass
class Detection:
    """Where an object is, in pixels and in the camera's own 3-D frame (mm).

    ``min_width_mm`` / ``max_width_mm`` are the object's real size, from the minimum
    area rectangle scaled by its depth. They let the gripper open to fit the object
    rather than to a fixed fraction, and let an impossible pick be refused before
    the arm moves at all.
    """

    box: tuple[int, int, int, int]          # x1, y1, x2, y2
    uv: tuple[float, float]                 # centroid pixel
    z_mm: float                             # robust depth at the object
    p_cam_mm: tuple[float, float, float]    # deprojected camera-frame position
    source: str                             # "color:red" | "locate_anything"
    area_px: float = 0.0
    min_width_mm: float | None = None       # narrow side (best grasp axis)
    max_width_mm: float | None = None       # wide side (worst-case jaw span)
    info: dict[str, Any] = field(default_factory=dict)


def _metric_size(w_px: float, h_px: float, z_mm: float,
                 intr: Intrinsics) -> tuple[float, float]:
    """Pixel extents at a known depth -> millimetres, via the pinhole model."""
    fx, fy, _, _ = intr
    a = abs(w_px) * z_mm / max(1e-6, fx)
    b = abs(h_px) * z_mm / max(1e-6, fy)
    return (min(a, b), max(a, b))


def rect_geometry(contour: np.ndarray) -> dict[str, Any]:
    """Return stable image-plane geometry for a segmented grasp region.

    OpenCV's rectangle angle changes convention when width and height swap.
    ``long_axis_deg`` is normalized modulo 180 degrees, which is the symmetry a
    parallel-jaw gripper actually cares about.
    """
    (cu, cv), (rw, rh), angle = cv2.minAreaRect(contour)
    long_axis = float(angle if rw >= rh else angle + 90.0)
    long_axis = ((long_axis + 90.0) % 180.0) - 90.0
    area = float(cv2.contourArea(contour))
    perimeter = float(cv2.arcLength(contour, True))
    circularity = (
        4.0 * np.pi * area / (perimeter * perimeter)
        if perimeter > 1e-6 else 0.0
    )
    return {
        "rect_center_uv": [round(float(cu), 2), round(float(cv), 2)],
        "rect_size_px": [round(float(rw), 2), round(float(rh), 2)],
        "long_axis_deg": round(long_axis, 2),
        "circularity": round(float(circularity), 3),
    }


def local_cross_section_width_mm(
    component: np.ndarray,
    centre_uv: tuple[float, float],
    long_axis_deg: float,
    z_mm: float,
    intr: Intrinsics,
) -> float | None:
    """Width through the actual grasp centre, perpendicular to the long axis.

    A global minimum-area rectangle uses the widest transverse part anywhere on
    an object. That is wrong for a marker with a wide cap when the jaws will
    close around its narrower body. Measure a thin central slice instead, using
    robust endpoints to ignore isolated RealSense/morphology pixels.
    """
    ys, xs = np.nonzero(component > 0)
    if xs.size < 20:
        return None
    theta = np.deg2rad(float(long_axis_deg))
    long_vec = np.array([np.cos(theta), np.sin(theta)], dtype=float)
    short_vec = np.array([-np.sin(theta), np.cos(theta)], dtype=float)
    points = np.column_stack((xs.astype(float), ys.astype(float)))
    rel = points - np.asarray(centre_uv, dtype=float)
    along = rel @ long_vec
    across = rel @ short_vec
    long_span = float(np.ptp(along))
    half_slice_px = max(2.0, 0.06 * long_span)
    local = across[np.abs(along) <= half_slice_px]
    if local.size < 12:
        return None
    lo, hi = np.percentile(local, [1.0, 99.0])
    width_px = float(hi - lo)
    if not np.isfinite(width_px) or width_px <= 1.0:
        return None
    fx, fy, _, _ = intr
    mm_per_short_px = float(z_mm) * float(np.hypot(
        short_vec[0] / max(1e-6, fx),
        short_vec[1] / max(1e-6, fy),
    ))
    width_mm = width_px * mm_per_short_px
    return width_mm if np.isfinite(width_mm) and width_mm > 0 else None


def cross_section_profile(
    component: np.ndarray,
    contour: np.ndarray,
    z_mm: float,
    intr: Intrinsics,
) -> list[dict[str, Any]]:
    """Sample grasp candidates along an object's medial long axis."""
    geometry = rect_geometry(contour)
    centre = np.asarray(geometry["rect_center_uv"], dtype=float)
    theta = np.deg2rad(float(geometry["long_axis_deg"]))
    long_vec = np.array([np.cos(theta), np.sin(theta)], dtype=float)
    short_vec = np.array([-np.sin(theta), np.cos(theta)], dtype=float)
    ys, xs = np.nonzero(component > 0)
    if xs.size < 30:
        return []
    points = np.column_stack((xs.astype(float), ys.astype(float)))
    rel = points - centre
    along = rel @ long_vec
    across = rel @ short_vec
    lo_long, hi_long = np.percentile(along, [1.0, 99.0])
    half_span = max(1.0, 0.5 * float(hi_long - lo_long))
    candidates: list[dict[str, Any]] = []
    for fraction in (-0.32, -0.16, 0.0, 0.16, 0.32):
        station = float(fraction * half_span)
        half_slice = max(2.0, 0.05 * (2.0 * half_span))
        section = np.abs(along - station) <= half_slice
        local_across = across[section]
        if local_across.size < 12:
            continue
        edge_lo, edge_hi = np.percentile(local_across, [1.0, 99.0])
        cross_mid = 0.5 * float(edge_lo + edge_hi)
        uv = centre + long_vec * station + short_vec * cross_mid
        width = local_cross_section_width_mm(
            component, (float(uv[0]), float(uv[1])),
            float(geometry["long_axis_deg"]), z_mm, intr,
        )
        if width is None:
            continue
        clearance_fraction = max(0.0, 1.0 - abs(float(fraction)))
        candidates.append({
            "uv": [round(float(uv[0]), 2), round(float(uv[1]), 2)],
            "width_mm": round(float(width), 2),
            "axis_fraction": float(fraction),
            "end_clearance_fraction": round(clearance_fraction, 3),
        })
    widths = [float(item["width_mm"]) for item in candidates]
    for index, item in enumerate(candidates):
        neighbours = widths[max(0, index - 1):min(len(widths), index + 2)]
        stability = max(neighbours) - min(neighbours) if neighbours else 0.0
        item["width_variation_mm"] = round(float(stability), 2)
        item["score"] = round(
            2.0 * float(item["end_clearance_fraction"])
            - 0.08 * float(stability)
            - 0.15 * abs(float(item["axis_fraction"])),
            3,
        )
    return candidates


def deproject(u: float, v: float, z_mm: float, intr: Intrinsics) -> tuple[float, float, float]:
    fx, fy, cx, cy = intr
    return ((u - cx) * z_mm / fx, (v - cy) * z_mm / fy, z_mm)


def project(p_cam_mm, intr: Intrinsics) -> tuple[float, float] | None:
    fx, fy, cx, cy = intr
    x, y, z = (float(v) for v in p_cam_mm)
    if z <= 1e-6:
        return None
    return (x * fx / z + cx, y * fy / z + cy)


def median_depth(depth: np.ndarray, u: float, v: float, r: int = 6) -> float:
    """Median of the valid depths in a small window -- robust to RealSense holes."""
    h, w = depth.shape[:2]
    x0, x1 = max(0, int(u) - r), min(w, int(u) + r + 1)
    y0, y1 = max(0, int(v) - r), min(h, int(v) + r + 1)
    patch = depth[y0:y1, x0:x1]
    valid = patch[patch > 0]
    return float(np.median(valid)) if valid.size else 0.0


def color_of(description: str) -> str | None:
    """Pull a known colour word out of a free-text object name, or None."""
    for word in str(description).lower().replace("-", " ").split():
        word = word.strip(",.'\"")
        if word in COLORS:
            return word
        if word in _ALIASES:
            return _ALIASES[word]
    return None


def depth_gate(depth: np.ndarray, near_mm: float = NEAR_MM, far_mm: float = FAR_MM) -> np.ndarray:
    """Pixels at a plausible grasp range: excludes the jaws and the far room."""
    return (depth > near_mm) & (depth < far_mm)


def color_mask(color_bgr: np.ndarray, name: str) -> np.ndarray:
    hsv = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2HSV)
    mask = np.zeros(hsv.shape[:2], np.uint8)
    for lo, hi in COLORS[name]:
        mask = cv2.bitwise_or(mask, cv2.inRange(hsv, np.array(lo, np.uint8), np.array(hi, np.uint8)))
    return mask


def detect_color(color_bgr: np.ndarray, depth_mm: np.ndarray, intr: Intrinsics, name: str,
                 *, near_mm: float = NEAR_MM, far_mm: float = FAR_MM,
                 min_area_px: float = MIN_AREA_PX) -> Detection | None:
    """Largest depth-gated blob of the named colour, or None."""
    if name not in COLORS:
        return None
    mask = cv2.bitwise_and(color_mask(color_bgr, name),
                           depth_gate(depth_mm, near_mm, far_mm).astype(np.uint8) * 255)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [c for c in contours if cv2.contourArea(c) >= min_area_px]
    if not contours:
        return None
    best = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(best)
    moments = cv2.moments(best)
    if moments["m00"] <= 0:
        return None
    u, v = moments["m10"] / moments["m00"], moments["m01"] / moments["m00"]

    # Depth from the blob's own pixels, not a window around the centroid: a
    # concave or ring-shaped object can have background showing at its centre.
    blob = np.zeros(mask.shape, np.uint8)
    cv2.drawContours(blob, [best], -1, 255, -1)
    values = depth_mm[(blob > 0) & depth_gate(depth_mm, near_mm, far_mm)]
    if values.size < 10:
        return None
    z = float(np.median(values))
    # minAreaRect gives the object's true minor/major extents, not the axis-aligned
    # box: a diagonal object would otherwise look far wider than it is.
    (_, _), (rw, rh), _ = cv2.minAreaRect(best)
    min_mm, max_mm = _metric_size(rw, rh, z, intr)
    geometry = rect_geometry(best)
    return Detection(box=(x, y, x + w, y + h), uv=(float(u), float(v)), z_mm=z,
                     p_cam_mm=deproject(u, v, z, intr), source=f"color:{name}",
                     area_px=float(cv2.contourArea(best)),
                     min_width_mm=min_mm, max_width_mm=max_mm,
                     info={"depth_px": int(values.size), **geometry})


# Words that mean "whatever is there", rather than naming a particular object.
GENERIC_NAMES = {"", "any", "anything", "object", "an object", "the object",
                 "any object", "it", "something", "thing", "nearest", "closest"}


def is_generic(description: str | None) -> bool:
    return str(description or "").strip().lower() in GENERIC_NAMES


def fit_plane(points: np.ndarray, iterations: int = 3) -> tuple[np.ndarray, float] | None:
    """Least-squares plane through a point cloud, refined by rejecting outliers.

    The table dominates a downward wrist view, so a plain fit lands on it; two or
    three reweighting passes then shake off the objects standing on it.
    """
    if points.shape[0] < 50:
        return None
    p = points
    normal = np.array([0.0, 0.0, 1.0])
    offset = 0.0
    for _ in range(iterations):
        centroid = p.mean(axis=0)
        _, _, vh = np.linalg.svd(p - centroid, full_matrices=False)
        normal = vh[-1]
        offset = float(normal @ centroid)
        residual = np.abs(points @ normal - offset)
        keep = residual < max(6.0, 2.5 * float(np.median(residual)))
        if keep.sum() < 50:
            break
        p = points[keep]
    return normal, offset


def segment_objects(color_bgr: np.ndarray, depth_mm: np.ndarray, intr: Intrinsics, *,
                    near_mm: float = NEAR_MM, far_mm: float = FAR_MM,
                    min_height_mm: float = 8.0, max_height_mm: float = 120.0,
                    min_area_px: float = MIN_AREA_PX) -> list[Detection]:
    """Find every object standing proud of the table -- no colour, no model, no name.

    This is what makes "pick that" work for arbitrary objects: a depth camera does
    not need to recognise something to know it is there. Fit the table plane, take
    everything at least ``min_height_mm`` above it, and each blob is an object with
    a real position and size.
    """
    fx, fy, cx, cy = intr
    valid = depth_gate(depth_mm, near_mm, far_mm)
    if valid.sum() < 200:
        return []
    vs, us = np.nonzero(valid)
    zs = depth_mm[vs, us]
    pts = np.stack([(us - cx) * zs / fx, (vs - cy) * zs / fy, zs], axis=-1)

    plane = fit_plane(pts)
    if plane is None:
        return []
    normal, offset = plane
    # Orient the normal to point back at the camera (camera looks along +z, so the
    # table's outward normal has a negative z component). Then a positive signed
    # distance means "nearer than the table" == standing on it. Deciding this from
    # the median of the signed distances does NOT work: a plane fit puts the median
    # at ~0, so its sign is noise.
    if normal[2] > 0:
        normal, offset = -normal, -offset
    signed = pts @ normal - offset

    # Bounded band, not just "above the table": the gripper's own jaws hang far
    # proud of the plane in an eye-in-hand view and would otherwise be returned as
    # the nearest object every time.
    above = (signed > min_height_mm) & (signed < max_height_mm)
    mask = np.zeros(depth_mm.shape, np.uint8)
    mask[vs[above], us[above]] = 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out: list[Detection] = []
    for contour in contours:
        if cv2.contourArea(contour) < min_area_px:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        moments = cv2.moments(contour)
        if moments["m00"] <= 0:
            continue
        u, v = moments["m10"] / moments["m00"], moments["m01"] / moments["m00"]
        blob = np.zeros(mask.shape, np.uint8)
        cv2.drawContours(blob, [contour], -1, 255, -1)
        values = depth_mm[(blob > 0) & valid]
        if values.size < 10:
            continue
        z = float(np.median(values))
        (_, _), (rw, rh), _ = cv2.minAreaRect(contour)
        min_mm, max_mm = _metric_size(rw, rh, z, intr)
        geometry = rect_geometry(contour)
        patch = color_bgr[y:y + h, x:x + w].reshape(-1, 3)
        out.append(Detection(
            box=(x, y, x + w, y + h), uv=(float(u), float(v)), z_mm=z,
            p_cam_mm=deproject(u, v, z, intr), source="depth:object",
            area_px=float(cv2.contourArea(contour)),
            min_width_mm=min_mm, max_width_mm=max_mm,
            info={"mean_bgr": [int(c) for c in patch.mean(axis=0)]
                  if patch.size else None, **geometry}))
    return out


def segment_target_envelope(
    color_bgr: np.ndarray,
    depth_mm: np.ndarray,
    intr: Intrinsics,
    target: Detection,
    *,
    near_mm: float = NEAR_MM,
    far_mm: float = FAR_MM,
    nearer_slop_mm: float = 20.0,
    farther_span_mm: float = 120.0,
) -> Detection | None:
    """Grow a full object envelope outward from a detected visual feature.

    A table-plane segmenter can intentionally exclude tall objects. Conversely,
    choosing any 2-D box that contains a colored cap can associate distant
    background. This seed-based path admits only depth-connected pixels near the
    target surface, so a cap can lead into its body but not into the far table.
    """
    h, w = depth_mm.shape[:2]
    x1, y1, x2, y2 = target.box
    bw, bh = max(1, x2 - x1), max(1, y2 - y1)
    pad_x = max(40, 3 * bw)
    pad_y = max(60, 5 * bh)
    rx1, ry1 = max(0, x1 - pad_x), max(0, y1 - pad_y)
    rx2, ry2 = min(w, x2 + pad_x), min(h, y2 + pad_y)
    z0 = float(target.z_mm)
    valid = (
        (depth_mm >= max(float(near_mm), z0 - float(nearer_slop_mm)))
        & (depth_mm <= min(float(far_mm), z0 + float(farther_span_mm)))
    )
    roi = np.zeros((h, w), np.uint8)
    roi[ry1:ry2, rx1:rx2] = 255
    mask = (valid.astype(np.uint8) * 255) & roi
    mask = cv2.morphologyEx(
        mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8)
    )
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        (mask > 0).astype(np.uint8), connectivity=8
    )
    if count <= 1:
        return None
    seed = np.zeros((h, w), np.uint8)
    seed[max(0, y1):min(h, y2), max(0, x1):min(w, x2)] = 1
    overlaps = [
        int(np.count_nonzero((labels == label) & (seed > 0)))
        for label in range(1, count)
    ]
    if not overlaps or max(overlaps) == 0:
        return None
    label = int(np.argmax(overlaps)) + 1
    component = (labels == label).astype(np.uint8) * 255
    contours, _ = cv2.findContours(
        component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    area = float(cv2.contourArea(contour))
    if area < max(20.0, 0.5 * float(target.area_px)):
        return None
    moments = cv2.moments(contour)
    if moments["m00"] <= 0:
        return None
    u = float(moments["m10"] / moments["m00"])
    v = float(moments["m01"] / moments["m00"])
    values = depth_mm[(component > 0) & valid]
    if values.size < 10:
        return None
    z = float(np.median(values))
    x, y, rw_box, rh_box = cv2.boundingRect(contour)
    (_, _), (rw, rh), _ = cv2.minAreaRect(contour)
    min_mm, max_mm = _metric_size(rw, rh, z, intr)
    geometry = rect_geometry(contour)
    local_width_mm = local_cross_section_width_mm(
        component, (u, v), float(geometry["long_axis_deg"]), z, intr
    )
    return Detection(
        box=(x, y, x + rw_box, y + rh_box),
        uv=(u, v), z_mm=z, p_cam_mm=deproject(u, v, z, intr),
        source="depth:target_envelope", area_px=area,
        min_width_mm=min_mm, max_width_mm=max_mm,
        info={
            "seed_source": target.source,
            "depth_span_mm": [
                round(float(np.min(values)), 1),
                round(float(np.max(values)), 1),
            ],
            "global_min_width_mm": round(float(min_mm), 2),
            "local_grasp_width_mm": (
                None if local_width_mm is None
                else round(float(local_width_mm), 2)
            ),
            "cross_section_candidates": cross_section_profile(
                component, contour, z, intr
            ),
            **geometry,
        },
    )


def segment_semantic_envelope(
    color_bgr: np.ndarray,
    depth_mm: np.ndarray,
    intr: Intrinsics,
    semantic_box: Detection,
    identifier: Detection,
    *,
    near_mm: float = NEAR_MM,
    far_mm: float = FAR_MM,
) -> Detection | None:
    """RGB foreground mask inside a semantic whole-object box.

    This handles low objects whose depth is indistinguishable from the table.
    The semantic model supplies only a tight box; GrabCut supplies the contour,
    while the original color/VLM identifier is a definite foreground seed.
    """
    h, w = color_bgr.shape[:2]
    x1, y1, x2, y2 = semantic_box.box
    x1, y1 = max(0, int(x1)), max(0, int(y1))
    x2, y2 = min(w, int(x2)), min(h, int(y2))
    if x2 - x1 < 8 or y2 - y1 < 8:
        return None
    pad_x = max(4, int(0.10 * (x2 - x1)))
    pad_y = max(4, int(0.06 * (y2 - y1)))
    rx1, ry1 = max(0, x1 - pad_x), max(0, y1 - pad_y)
    rx2, ry2 = min(w, x2 + pad_x), min(h, y2 + pad_y)
    ix1, iy1, ix2, iy2 = identifier.box
    # A grounded SAM backend supplies a true promptable object mask.  Retain the
    # deterministic GrabCut path as a fallback for cached LocateAnything users.
    from .local_vision import segment_box as segment_local_box
    component = segment_local_box(
        color_bgr, (x1, y1, x2, y2)
    )
    segmentation_source = "sam2.1"
    if component is None:
        segmentation_source = "grabcut"
        mask = np.full((h, w), cv2.GC_BGD, dtype=np.uint8)
        mask[ry1:ry2, rx1:rx2] = cv2.GC_PR_BGD
        mask[y1:y2, x1:x2] = cv2.GC_PR_FGD
        mask[max(0, iy1):min(h, iy2), max(0, ix1):min(w, ix2)] = cv2.GC_FGD
        bg_model = np.zeros((1, 65), np.float64)
        fg_model = np.zeros((1, 65), np.float64)
        try:
            cv2.grabCut(color_bgr, mask, None, bg_model, fg_model, 5,
                        cv2.GC_INIT_WITH_MASK)
        except cv2.error:
            return None
        component = np.where(
            (mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0
        ).astype(np.uint8)
    # Never let foreground escape the semantic model's padded object region.
    roi = np.zeros_like(component)
    roi[ry1:ry2, rx1:rx2] = 255
    component &= roi
    component = cv2.morphologyEx(
        component, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8)
    )
    count, labels, _, _ = cv2.connectedComponentsWithStats(
        (component > 0).astype(np.uint8), connectivity=8
    )
    if count <= 1:
        return None
    seed = np.zeros_like(component)
    seed[max(0, iy1):min(h, iy2), max(0, ix1):min(w, ix2)] = 1
    overlaps = [
        int(np.count_nonzero((labels == label) & (seed > 0)))
        for label in range(1, count)
    ]
    label = int(np.argmax(overlaps)) + 1 if overlaps and max(overlaps) else int(
        1 + np.argmax([np.count_nonzero(labels == value)
                       for value in range(1, count)])
    )
    component = (labels == label).astype(np.uint8) * 255
    contours, _ = cv2.findContours(
        component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    area = float(cv2.contourArea(contour))
    if area < max(30.0, 0.4 * float(identifier.area_px)):
        return None
    moments = cv2.moments(contour)
    if moments["m00"] <= 0:
        return None
    u = float(moments["m10"] / moments["m00"])
    v = float(moments["m01"] / moments["m00"])
    valid = depth_gate(depth_mm, near_mm, far_mm)
    values = depth_mm[(component > 0) & valid]
    if values.size < 10:
        return None
    # The nearer half suppresses table pixels retained around a flush object.
    cutoff = float(np.percentile(values, 60.0))
    object_values = values[values <= cutoff]
    z = float(np.median(object_values if object_values.size >= 10 else values))
    x, y, bw, bh = cv2.boundingRect(contour)
    (_, _), (rw, rh), _ = cv2.minAreaRect(contour)
    min_mm, max_mm = _metric_size(rw, rh, z, intr)
    geometry = rect_geometry(contour)
    profile = cross_section_profile(component, contour, z, intr)
    local_width = local_cross_section_width_mm(
        component, (u, v), float(geometry["long_axis_deg"]), z, intr
    )
    return Detection(
        box=(x, y, x + bw, y + bh), uv=(u, v), z_mm=z,
        p_cam_mm=deproject(u, v, z, intr),
        source="rgbd:semantic_envelope", area_px=area,
        min_width_mm=min_mm, max_width_mm=max_mm,
        info={
            "semantic_box": list(semantic_box.box),
            "segmentation_source": segmentation_source,
            "identifier_source": identifier.source,
            "local_grasp_width_mm": local_width,
            "cross_section_candidates": profile,
            **geometry,
        },
    )
def detect_any(color_bgr: np.ndarray, depth_mm: np.ndarray, intr: Intrinsics, *,
               near_mm: float = NEAR_MM, far_mm: float = FAR_MM,
               max_width_mm: float | None = None) -> tuple[Detection | None, dict[str, Any]]:
    """Pick the best graspable object off the table with no name at all.

    Candidates too wide for the jaws are dropped when the jaw scale is known, then
    the most central one wins -- that is the object the camera is pointed at.
    """
    found = segment_objects(color_bgr, depth_mm, intr, near_mm=near_mm, far_mm=far_mm)
    why: dict[str, Any] = {"candidates": len(found), "used": None}
    if not found:
        why["reason"] = "nothing standing above the table plane"
        return None, why
    fits = found
    if max_width_mm is not None:
        fits = [d for d in found if (d.min_width_mm or 0.0) <= max_width_mm]
        why["too_wide"] = len(found) - len(fits)
        if not fits:
            why["reason"] = (f"all {len(found)} objects are wider than the "
                             f"{max_width_mm:.0f}mm jaw opening")
            return None, why
    h, w = depth_mm.shape[:2]
    centre = np.array([w / 2.0, h / 2.0])
    best = min(fits, key=lambda d: float(np.hypot(*(np.asarray(d.uv) - centre))))
    why["used"] = best.source
    why["sizes_mm"] = [[round(d.min_width_mm or 0), round(d.max_width_mm or 0)] for d in fits]
    return best, why


def detect_vlm(color_bgr: np.ndarray, depth_mm: np.ndarray, intr: Intrinsics,
               description: str, *, near_mm: float = NEAR_MM,
               far_mm: float = FAR_MM) -> Detection | None:
    """Open-vocabulary fallback via LocateAnything (opt-in; see local_vision)."""
    from .local_vision import locate as locate_local

    result = locate_local(color_bgr, description)
    if not result or not result.get("found"):
        return None
    x1, y1, x2, y2 = (int(v) for v in result["box"])
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(depth_mm.shape[1], x2), min(depth_mm.shape[0], y2)
    if x2 <= x1 or y2 <= y1:
        return None
    patch = depth_mm[y1:y2, x1:x2]
    values = patch[depth_gate(patch, near_mm, far_mm)]
    if values.size < 10:
        return None
    z = float(np.median(values))
    u, v = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    # Only an axis-aligned box is available here, so the extents are an upper bound.
    min_mm, max_mm = _metric_size(x2 - x1, y2 - y1, z, intr)
    backend = str(result.get("backend") or "locate_anything")
    return Detection(box=(x1, y1, x2, y2), uv=(u, v), z_mm=z,
                     p_cam_mm=deproject(u, v, z, intr), source=backend,
                     area_px=float((x2 - x1) * (y2 - y1)),
                     min_width_mm=min_mm, max_width_mm=max_mm,
                     info={"confidence": result.get("confidence"),
                           "detector_backend": backend,
                           "box_axis_aligned": True})


def detect(color_bgr: np.ndarray, depth_mm: np.ndarray, intr: Intrinsics, description: str,
           *, near_mm: float = NEAR_MM, far_mm: float = FAR_MM,
           allow_vlm: bool = True) -> tuple[Detection | None, dict[str, Any]]:
    """Locate ``description`` in a wrist RGB-D frame.

    With GPU vision enabled, Grounding DINO is primary for named objects so a
    colored cap cannot silently stand in for the whole requested object.  The
    inexpensive color/depth detector remains the deterministic fallback.

    Returns ``(detection | None, why)`` where ``why`` records what each detector
    did, so a failure is explainable to the planner instead of just empty.
    """
    why: dict[str, Any] = {"description": description}
    # No name given (or a generic one): take whatever is on the table. Works for
    # ANY object -- a depth camera does not have to recognise something to see it.
    if is_generic(description):
        from . import gripper_model as gm
        found, generic_why = detect_any(color_bgr, depth_mm, intr, near_mm=near_mm,
                                        far_mm=far_mm, max_width_mm=gm.max_aperture_mm())
        why.update(generic_why)
        return found, why
    vision_enabled = os.getenv(
        "REBOT_ENABLE_LOCATE_ANYTHING", "0"
    ).strip().lower() in {"1", "true", "yes", "on"}
    vision_primary = os.getenv(
        "REBOT_VISION_PRIMARY", "0"
    ).strip().lower() in {"1", "true", "yes", "on"}
    if allow_vlm and vision_enabled and vision_primary:
        found = detect_vlm(
            color_bgr, depth_mm, intr, description,
            near_mm=near_mm, far_mm=far_mm,
        )
        why["semantic_result"] = found.source if found else "not found"
        if found is not None:
            why["used"] = found.source
            return found, why
    name = color_of(description)
    why["color_word"] = name
    if name is not None:
        found = detect_color(color_bgr, depth_mm, intr, name, near_mm=near_mm, far_mm=far_mm)
        if found is not None:
            why["used"] = found.source
            return found, why
        why["color_result"] = "no depth-gated blob of that colour"
    if not allow_vlm:
        why["used"] = None
        return None, why
    found = detect_vlm(color_bgr, depth_mm, intr, description,
                       near_mm=near_mm, far_mm=far_mm)
    why["used"] = found.source if found else None
    if found is None:
        why["vlm_result"] = "not found (or LocateAnything disabled)"
    return found, why
