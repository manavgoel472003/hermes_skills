"""Object detection for the wrist (eye-in-hand) RGB-D camera.

Colour first, VLM second.  On this table a generous HSV band plus a depth gate is
both faster and more reliable than a 3B grounding model, and it needs no GPU -- so
it is the primary detector and ``LocateAnything`` is the open-vocabulary fallback
for names that are not a colour ("the sponge", "the cable").

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
    (_, _), (rw, rh), angle = cv2.minAreaRect(best)
    min_mm, max_mm = _metric_size(rw, rh, z, intr)
    return Detection(box=(x, y, x + w, y + h), uv=(float(u), float(v)), z_mm=z,
                     p_cam_mm=deproject(u, v, z, intr), source=f"color:{name}",
                     area_px=float(cv2.contourArea(best)),
                     min_width_mm=min_mm, max_width_mm=max_mm,
                     info={"depth_px": int(values.size), "angle_deg": round(float(angle), 1)})


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
        patch = color_bgr[y:y + h, x:x + w].reshape(-1, 3)
        out.append(Detection(
            box=(x, y, x + w, y + h), uv=(float(u), float(v)), z_mm=z,
            p_cam_mm=deproject(u, v, z, intr), source="depth:object",
            area_px=float(cv2.contourArea(contour)),
            min_width_mm=min_mm, max_width_mm=max_mm,
            info={"mean_bgr": [int(c) for c in patch.mean(axis=0)] if patch.size else None}))
    return out


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
    return Detection(box=(x1, y1, x2, y2), uv=(u, v), z_mm=z,
                     p_cam_mm=deproject(u, v, z, intr), source="locate_anything",
                     area_px=float((x2 - x1) * (y2 - y1)),
                     min_width_mm=min_mm, max_width_mm=max_mm,
                     info={"confidence": result.get("confidence"), "box_axis_aligned": True})


def detect(color_bgr: np.ndarray, depth_mm: np.ndarray, intr: Intrinsics, description: str,
           *, near_mm: float = NEAR_MM, far_mm: float = FAR_MM,
           allow_vlm: bool = True) -> tuple[Detection | None, dict[str, Any]]:
    """Locate ``description`` in a wrist RGB-D frame. Colour first, VLM fallback.

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
    found = detect_vlm(color_bgr, depth_mm, intr, description, near_mm=near_mm, far_mm=far_mm)
    why["used"] = found.source if found else None
    if found is None:
        why["vlm_result"] = "not found (or LocateAnything disabled)"
    return found, why
