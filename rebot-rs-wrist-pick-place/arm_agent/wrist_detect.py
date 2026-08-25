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
import itertools
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
    # Achromatic. White and black have no hue, so they span every H and are
    # separated on saturation and value instead. Without these, "white cup" and
    # "black mug" yield no colour word at all and the RGB-D colour path is
    # skipped entirely, leaving them wholly dependent on a semantic hit -- which
    # is how a plainly visible white cup went undetected 0/3 frames.
    # Bounds measured on this rig: white cup S=24/V=189, pale table S=29/V=204,
    # dark mat S=152/V=76, black object S=67/V=39, dark green mug S=81/V=55.
    # Black stays under V=45/S=70: at V=50/S=80 the dark green mug read 15%
    # black, above the 12% veto threshold, and tightening drops it to 9% while
    # the genuinely black object still reads 46%. A shadowed coloured object is
    # the real hazard here, not the table.
    # the pale table matches "white" but cannot be selected, because the
    # detector resolves a depth-connected object above the table plane first
    # and only then reads colour inside it.
    "white":  [((0, 0, 150), (180, 60, 255))],
    "black":  [((0, 0, 0), (180, 70, 45))],
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


def colors_of(description: str) -> list[str]:
    """Return every distinct known colour in a free-text object name.

    Slash-separated alternatives such as ``red/pink marker`` are common operator
    descriptions.  Treating that as one token silently bypassed stable colour
    tracking and made the changing semantic object box drive camera calibration.
    """
    normalized = str(description).lower()
    for separator in ("-", "/", "|", "+"):
        normalized = normalized.replace(separator, " ")
    found: list[str] = []
    for word in normalized.split():
        word = word.strip(",.'\"")
        if word in COLORS:
            name = word
        elif word in _ALIASES:
            name = _ALIASES[word]
        else:
            continue
        if name not in found:
            found.append(name)
    return found


def color_of(description: str) -> str | None:
    """Pull the first known colour word out of a free-text object name, or None."""
    names = colors_of(description)
    return names[0] if names else None


def depth_gate(depth: np.ndarray, near_mm: float = NEAR_MM, far_mm: float = FAR_MM) -> np.ndarray:
    """Pixels at a plausible grasp range: excludes the jaws and the far room."""
    return (depth > near_mm) & (depth < far_mm)


def color_mask(color_bgr: np.ndarray, name: str) -> np.ndarray:
    hsv = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2HSV)
    mask = np.zeros(hsv.shape[:2], np.uint8)
    for lo, hi in COLORS[name]:
        mask = cv2.bitwise_or(mask, cv2.inRange(hsv, np.array(lo, np.uint8), np.array(hi, np.uint8)))
    return mask


def _detect_mask(
    mask: np.ndarray,
    depth_mm: np.ndarray,
    intr: Intrinsics,
    source: str,
    *,
    near_mm: float,
    far_mm: float,
    min_area_px: float,
    close_kernel: int = 7,
) -> Detection | None:
    mask = cv2.bitwise_and(
        mask, depth_gate(depth_mm, near_mm, far_mm).astype(np.uint8) * 255
    )
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    mask = cv2.morphologyEx(
        mask, cv2.MORPH_CLOSE,
        np.ones((max(3, close_kernel), max(3, close_kernel)), np.uint8),
    )
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
                     p_cam_mm=deproject(u, v, z, intr), source=source,
                     area_px=float(cv2.contourArea(best)),
                     min_width_mm=min_mm, max_width_mm=max_mm,
                     info={"depth_px": int(values.size), **geometry})


def detect_color(color_bgr: np.ndarray, depth_mm: np.ndarray, intr: Intrinsics, name: str,
                 *, near_mm: float = NEAR_MM, far_mm: float = FAR_MM,
                 min_area_px: float = MIN_AREA_PX) -> Detection | None:
    """Largest depth-gated blob of the named colour, or None."""
    if name not in COLORS:
        return None
    return _detect_mask(
        color_mask(color_bgr, name), depth_mm, intr, f"color:{name}",
        near_mm=near_mm, far_mm=far_mm, min_area_px=min_area_px,
    )


def detect_color_union(
    color_bgr: np.ndarray,
    depth_mm: np.ndarray,
    intr: Intrinsics,
    names: list[str],
    *,
    near_mm: float = NEAR_MM,
    far_mm: float = FAR_MM,
    min_area_px: float = MIN_AREA_PX,
) -> Detection | None:
    """Treat a multi-colour description as one physical visual feature.

    Returning the first named colour made ``green and yellow sponge`` alternate
    between different patches as perspective changed. Their union is a much more
    stable identifier and remains depth-gated to the grasp workspace.
    """
    valid_names = [name for name in names if name in COLORS]
    if not valid_names:
        return None
    mask = np.zeros(color_bgr.shape[:2], dtype=np.uint8)
    for name in valid_names:
        mask = cv2.bitwise_or(mask, color_mask(color_bgr, name))
    return _detect_mask(
        mask, depth_mm, intr, f"color:{'+'.join(valid_names)}",
        near_mm=near_mm, far_mm=far_mm, min_area_px=min_area_px,
        close_kernel=11,
    )


def _requires_all_colors(description: str) -> bool:
    """True for descriptions such as ``green and yellow sponge``.

    A slash normally means alternatives (``red/pink marker``), while ``and``
    describes a compound appearance.  Mixing those meanings caused the largest
    single green object to satisfy a green-and-yellow request.
    """
    words = str(description).lower().replace("-", " ").split()
    return "and" in words and len(colors_of(description)) > 1


def _box_has_all_colors(
    color_bgr: np.ndarray,
    box: tuple[int, int, int, int],
    names: list[str],
    *,
    min_pixels: int = 25,
) -> bool:
    """Validate that every requested colour occurs inside a semantic box."""
    h, w = color_bgr.shape[:2]
    x1, y1, x2, y2 = box
    pad_x = max(3, int(0.08 * max(1, x2 - x1)))
    pad_y = max(3, int(0.08 * max(1, y2 - y1)))
    x1, y1 = max(0, x1 - pad_x), max(0, y1 - pad_y)
    x2, y2 = min(w, x2 + pad_x), min(h, y2 + pad_y)
    return all(
        int(np.count_nonzero(color_mask(color_bgr, name)[y1:y2, x1:x2]))
        >= min_pixels
        for name in names
    )


#: Fraction of a detection's own surface that must show the single colour the
#: prompt names.  Measured on this rig: a correct marker reads 30-32% (the
#: barrel is white, only the cap and label carry the colour) while a
#: wrong-coloured one reads 0-3%.  12% sits well clear of both, leaving room
#: for a shaded or edge-on face without admitting an outright wrong colour.
COLOR_VETO_FRACTION = float(
    os.getenv("REBOT_COLOR_VETO_FRACTION", "0.12")
)


def color_veto_enabled() -> bool:
    """Whether a single-colour prompt may veto a semantic detection."""
    return os.getenv("REBOT_COLOR_VETO", "1").strip().lower() not in {
        "0", "false", "no", "off"
    }


def _globally_distinctive_color(name: str) -> bool:
    """Whether colour alone can safely identify an object in the full frame.

    Chromatic HSV bands constrain both hue and saturation, so they are useful
    deterministic fallbacks after a semantic noun miss. Achromatic bands span
    essentially every hue (white/black); using those globally discards the noun
    and can turn ``white cup`` into any white marker, paper, or table patch.
    """
    bands = COLORS.get(name, [])
    if not bands:
        return False
    return all(
        (int(hi[0]) - int(lo[0])) < 170 and int(lo[1]) >= 50
        for lo, hi in bands
    )


def semantic_color_fraction(
    color_bgr: np.ndarray,
    depth_mm: np.ndarray,
    detection: "Detection",
    name: str,
) -> float | None:
    """How much of ``detection``'s own surface actually shows ``name``.

    Returns ``None`` when there is too little surface to judge, which callers
    must treat as "no opinion" rather than as a refusal.

    The measurement is confined to the object's depth shell.  A box alone would
    include the table behind the object, which both dilutes the colour of a
    correct object and can lend a wrong one the colour of whatever it sits on.
    """
    if not isinstance(depth_mm, np.ndarray) or depth_mm.ndim != 2:
        return None
    height, width = color_bgr.shape[:2]
    x1, y1, x2, y2 = detection.box
    x1, y1 = max(0, int(x1)), max(0, int(y1))
    x2, y2 = min(width, int(x2)), min(height, int(y2))
    if x2 - x1 < 2 or y2 - y1 < 2:
        return None
    pixels = np.zeros((height, width), dtype=bool)
    pixels[y1:y2, x1:x2] = True
    native_mask_value = (getattr(detection, "info", {}) or {}).get(
        "mask_u8"
    )
    native_mask = (
        np.asarray(native_mask_value, dtype=np.uint8)
        if native_mask_value is not None else np.empty((0, 0), np.uint8)
    )
    # A semantic box can include a neighbouring marker, cup, or background at
    # nearly the same depth.  Those pixels previously lent the wrong instance
    # enough requested colour to pass the veto.  When the detector supplies a
    # native mask, measure only that exact object surface; retain the bounded
    # box behavior solely for older detectors without masks.
    if native_mask.shape == pixels.shape and np.any(native_mask > 0):
        pixels &= native_mask > 0
    if depth_mm.shape == pixels.shape:
        shell = pixels & (
            (depth_mm > detection.z_mm - 45.0)
            & (depth_mm < detection.z_mm + 45.0)
        )
        if int(np.count_nonzero(shell)) >= 40:
            pixels = shell
    total = int(np.count_nonzero(pixels))
    if total < 40:
        return None
    hits = int(np.count_nonzero((color_mask(color_bgr, name) > 0) & pixels))
    return hits / float(total)


def detect_color_conjunction(
    color_bgr: np.ndarray,
    depth_mm: np.ndarray,
    intr: Intrinsics,
    names: list[str],
    *,
    near_mm: float = NEAR_MM,
    far_mm: float = FAR_MM,
    min_area_px: float = MIN_AREA_PX,
) -> Detection | None:
    """Find one nearby RGB-D object containing *every* named colour.

    Components need not touch in the image—the coloured faces of a sponge can
    be separated by foam or glare—but they must be close in metric 3-D and at a
    similar depth.  Consequently a standalone green marker cannot win a
    ``green and yellow sponge`` request.
    """
    valid_names = [name for name in names if name in COLORS]
    if len(valid_names) < 2:
        return None
    valid_depth = depth_gate(depth_mm, near_mm, far_mm)
    groups: list[list[dict[str, Any]]] = []
    component_min = max(25.0, 0.20 * float(min_area_px))
    for name in valid_names:
        mask = cv2.bitwise_and(
            color_mask(color_bgr, name), valid_depth.astype(np.uint8) * 255
        )
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        components: list[dict[str, Any]] = []
        for contour in contours:
            area = float(cv2.contourArea(contour))
            if area < component_min:
                continue
            blob = np.zeros(mask.shape, np.uint8)
            cv2.drawContours(blob, [contour], -1, 255, -1)
            values = depth_mm[(blob > 0) & valid_depth]
            moments = cv2.moments(contour)
            if values.size < 8 or moments["m00"] <= 0:
                continue
            u = float(moments["m10"] / moments["m00"])
            v = float(moments["m01"] / moments["m00"])
            z = float(np.median(values))
            components.append({
                "contour": contour, "area": area, "uv": (u, v), "z": z,
                "p": np.asarray(deproject(u, v, z, intr), dtype=float),
            })
        if not components:
            return None
        groups.append(components)

    max_separation_mm = _env_float("REBOT_COLOR_CONJUNCTION_MAX_SEPARATION_MM", 90.0)
    max_depth_delta_mm = _env_float("REBOT_COLOR_CONJUNCTION_MAX_DEPTH_DELTA_MM", 55.0)
    candidates: list[tuple[float, tuple[dict[str, Any], ...]]] = []
    for combination in itertools.product(*groups):
        distances = [
            float(np.linalg.norm(a["p"] - b["p"]))
            for a, b in itertools.combinations(combination, 2)
        ]
        depth_deltas = [
            abs(float(a["z"]) - float(b["z"]))
            for a, b in itertools.combinations(combination, 2)
        ]
        if max(distances, default=0.0) > max_separation_mm:
            continue
        if max(depth_deltas, default=0.0) > max_depth_delta_mm:
            continue
        total_area = sum(float(item["area"]) for item in combination)
        score = total_area / (1.0 + max(distances, default=0.0))
        candidates.append((score, combination))
    if not candidates:
        return None
    _, chosen = max(candidates, key=lambda item: item[0])
    component_mask = np.zeros(depth_mm.shape, np.uint8)
    all_points = []
    for item in chosen:
        cv2.drawContours(component_mask, [item["contour"]], -1, 255, -1)
        all_points.append(item["contour"])
    points = np.concatenate(all_points, axis=0)
    hull = cv2.convexHull(points)
    values = depth_mm[(component_mask > 0) & valid_depth]
    moments = cv2.moments(component_mask, binaryImage=True)
    if values.size < 10 or moments["m00"] <= 0:
        return None
    u = float(moments["m10"] / moments["m00"])
    v = float(moments["m01"] / moments["m00"])
    z = float(np.median(values))
    x, y, bw, bh = cv2.boundingRect(hull)
    (_, _), (rw, rh), _ = cv2.minAreaRect(hull)
    min_mm, max_mm = _metric_size(rw, rh, z, intr)
    geometry = rect_geometry(hull)
    return Detection(
        box=(x, y, x + bw, y + bh), uv=(u, v), z_mm=z,
        p_cam_mm=deproject(u, v, z, intr),
        source=f"color:{'&'.join(valid_names)}",
        area_px=sum(float(item["area"]) for item in chosen),
        min_width_mm=min_mm, max_width_mm=max_mm,
        info={
            "required_colors": valid_names,
            "component_centres_uv": [
                [round(item["uv"][0], 1), round(item["uv"][1], 1)]
                for item in chosen
            ],
            "max_component_separation_mm": round(max(
                (float(np.linalg.norm(a["p"] - b["p"]))
                 for a, b in itertools.combinations(chosen, 2)),
                default=0.0,
            ), 1),
            **geometry,
        },
    )


def track_detection_flow(
    before: tuple[np.ndarray, np.ndarray, Intrinsics],
    after: tuple[np.ndarray, np.ndarray, Intrinsics],
    detection: Detection,
    description: str,
) -> Detection | None:
    """Track one detected instance across a short arm jog using RGB-D flow.

    This is an association layer, not a replacement detector. Forward/backward
    Lucas-Kanade consistency rejects drifting points, while the initial object
    colours and depth keep features off the table/background.
    """
    color0, depth0, _intr0 = before
    color1, depth1, intr1 = after
    gray0 = cv2.cvtColor(color0, cv2.COLOR_BGR2GRAY)
    gray1 = cv2.cvtColor(color1, cv2.COLOR_BGR2GRAY)
    h, w = gray0.shape
    x1, y1, x2, y2 = detection.box
    pad_x = max(4, int(0.12 * max(1, x2 - x1)))
    pad_y = max(4, int(0.12 * max(1, y2 - y1)))
    roi = np.zeros((h, w), dtype=np.uint8)
    roi[max(0, y1 - pad_y):min(h, y2 + pad_y),
        max(0, x1 - pad_x):min(w, x2 + pad_x)] = 255
    depth_seed = (
        (depth0 > max(NEAR_MM, detection.z_mm - 45.0))
        & (depth0 < min(FAR_MM, detection.z_mm + 45.0))
    ).astype(np.uint8) * 255
    mask = cv2.bitwise_and(roi, depth_seed)
    names = colors_of(description)
    if names:
        color_union = np.zeros_like(mask)
        for name in names:
            color_union = cv2.bitwise_or(color_union, color_mask(color0, name))
        colored = cv2.bitwise_and(mask, color_union)
        if int(np.count_nonzero(colored)) >= 30:
            mask = colored
    points0 = cv2.goodFeaturesToTrack(
        gray0, maxCorners=80, qualityLevel=0.01, minDistance=4,
        mask=mask, blockSize=5,
    )
    if points0 is None or len(points0) < 5:
        return None
    lk = dict(winSize=(31, 31), maxLevel=3,
              criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                        30, 0.01))
    points1, status1, _ = cv2.calcOpticalFlowPyrLK(gray0, gray1, points0, None, **lk)
    if points1 is None or status1 is None:
        return None
    points_back, status2, _ = cv2.calcOpticalFlowPyrLK(
        gray1, gray0, points1, None, **lk
    )
    if points_back is None or status2 is None:
        return None
    p0 = points0.reshape(-1, 2)
    p1 = points1.reshape(-1, 2)
    pb = points_back.reshape(-1, 2)
    good = (status1.reshape(-1) > 0) & (status2.reshape(-1) > 0)
    good &= np.linalg.norm(pb - p0, axis=1) <= 1.5
    good &= (p1[:, 0] >= 0) & (p1[:, 0] < w) \
        & (p1[:, 1] >= 0) & (p1[:, 1] < h)
    if int(np.count_nonzero(good)) < 4:
        return None
    shifts = p1[good] - p0[good]
    median_shift = np.median(shifts, axis=0)
    residual = np.linalg.norm(shifts - median_shift, axis=1)
    mad = float(np.median(residual))
    consistent = residual <= max(1.5, 3.0 * mad)
    if int(np.count_nonzero(consistent)) < 4:
        return None
    p1_good = p1[good][consistent]
    tracked_u = float(detection.uv[0] + np.median(shifts[consistent, 0]))
    tracked_v = float(detection.uv[1] + np.median(shifts[consistent, 1]))
    us = np.clip(np.rint(p1_good[:, 0]).astype(int), 0, w - 1)
    vs = np.clip(np.rint(p1_good[:, 1]).astype(int), 0, h - 1)
    depths = depth1[vs, us]
    depths = depths[
        (depths > max(NEAR_MM, detection.z_mm - 70.0))
        & (depths < min(FAR_MM, detection.z_mm + 70.0))
    ]
    if depths.size < 3:
        return None
    z = float(np.median(depths))
    dx, dy = float(median_shift[0]), float(median_shift[1])
    bx1 = int(np.clip(round(x1 + dx), 0, w - 1))
    by1 = int(np.clip(round(y1 + dy), 0, h - 1))
    bx2 = int(np.clip(round(x2 + dx), bx1 + 1, w))
    by2 = int(np.clip(round(y2 + dy), by1 + 1, h))
    return Detection(
        box=(bx1, by1, bx2, by2), uv=(tracked_u, tracked_v), z_mm=z,
        p_cam_mm=deproject(tracked_u, tracked_v, z, intr1),
        source=f"flow:{detection.source}", area_px=detection.area_px,
        min_width_mm=detection.min_width_mm,
        max_width_mm=detection.max_width_mm,
        info={**detection.info, "flow_points": int(np.count_nonzero(consistent)),
              "flow_shift_px": [round(dx, 2), round(dy, 2)]},
    )


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
            info={
                "mean_bgr": [int(c) for c in patch.mean(axis=0)]
                if patch.size else None,
                # Preserve the actual depth component.  Metric 3-D geometry
                # must never substitute the component's rectangular bounding
                # box, which also contains table/background depth.
                "mask_u8": blob,
                "segmentation_source": "table_plane_depth_component",
                **geometry,
            }))
    return out


def detect_colored_tabletop_object(
    color_bgr: np.ndarray,
    depth_mm: np.ndarray,
    intr: Intrinsics,
    names: list[str],
    *,
    require_all: bool = False,
    near_mm: float = NEAR_MM,
    far_mm: float = FAR_MM,
) -> Detection | None:
    """Match requested colour inside an independently segmented 3D object.

    This is deliberately stronger than selecting the largest HSV contour. The
    table plane and depth connectivity define the physical object first; colour
    only identifies which of those objects the description refers to. Distant
    red furniture therefore cannot stand in for a red tabletop marker.
    """
    valid_names = [name for name in names if name in COLORS]
    if not valid_names:
        return None
    objects = segment_objects(
        color_bgr, depth_mm, intr,
        near_mm=near_mm, far_mm=far_mm,
        # Upright markers and bottles can exceed the generic 120 mm object
        # ceiling. This remains bounded below the wrist/camera workspace depth.
        max_height_mm=220.0,
        min_area_px=max(80.0, 0.4 * float(MIN_AREA_PX)),
    )
    color_masks = {
        name: color_mask(color_bgr, name) > 0 for name in valid_names
    }
    minimum_pixels = int(np.clip(_env_float(
        "REBOT_COLORED_OBJECT_MIN_PIXELS", 25.0
    ), 12.0, 200.0))
    candidates: list[tuple[float, float, Detection, dict[str, int]]] = []
    for detected in objects:
        height, width = depth_mm.shape[:2]
        x1, y1, x2, y2 = detected.box
        edge_margin = 4
        # An object cut by the camera boundary has unknown dimensions and its
        # centroid shifts as more of it enters the image. It cannot establish
        # hybrid identity or authorize jaw sizing; bounded live look-around is
        # responsible for finding a complete view.
        if x1 <= edge_margin or y1 <= edge_margin \
                or x2 >= width - edge_margin \
                or y2 >= height - edge_margin:
            continue
        object_mask = np.asarray(
            detected.info.get("mask_u8"), dtype=np.uint8
        ) > 0
        if object_mask.shape != depth_mm.shape:
            continue
        counts = {
            name: int(np.count_nonzero(object_mask & color_masks[name]))
            for name in valid_names
        }
        present = [count >= minimum_pixels for count in counts.values()]
        color_ok = all(present) if require_all else any(present)
        if not color_ok:
            continue
        object_pixels = max(1, int(np.count_nonzero(object_mask)))
        coverage = float(sum(counts.values())) / float(object_pixels)
        candidates.append((float(detected.z_mm), -coverage, detected, counts))
    if not candidates:
        return None
    # A wrist camera sees the grasp workspace before the room background. Among
    # independently segmented matching objects, prefer the nearest physical
    # object and use colour coverage only as a tie-breaker.
    _depth, _negative_coverage, selected, selected_counts = min(
        candidates, key=lambda item: (item[0], item[1])
    )
    return Detection(
        box=selected.box,
        uv=selected.uv,
        z_mm=selected.z_mm,
        p_cam_mm=selected.p_cam_mm,
        source="rgbd:colored_object",
        area_px=selected.area_px,
        min_width_mm=selected.min_width_mm,
        max_width_mm=selected.max_width_mm,
        info={
            **selected.info,
            "identifier_colors": valid_names,
            "identifier_color_pixels": selected_counts,
            "identifier_requires_all_colors": bool(require_all),
            "identity_basis": "table_plane_depth_component_with_color",
        },
    )


def detect_color_seeded_instance(
    color_bgr: np.ndarray,
    depth_mm: np.ndarray,
    intr: Intrinsics,
    name: str,
    *,
    near_mm: float = NEAR_MM,
    far_mm: float = FAR_MM,
) -> Detection | None:
    """Build one physical object mask from an exact requested-colour seed.

    Text grounding can choose the wrong same-class object while ignoring its
    colour adjective.  Conversely, a raw HSV mask often covers only a marker
    cap or label.  This path uses the deterministic colour pixels to select an
    independently segmented tabletop object, then retains only the connected
    physical component containing those pixels.  The resulting mask—not its
    rectangle—is authoritative for identity and metric geometry.
    """
    if name not in COLORS:
        return None
    physical = detect_colored_tabletop_object(
        color_bgr, depth_mm, intr, [name],
        near_mm=near_mm, far_mm=far_mm,
    )
    if physical is None:
        return None
    raw_mask = np.asarray(
        (getattr(physical, "info", {}) or {}).get("mask_u8"),
        dtype=np.uint8,
    )
    if raw_mask.shape != depth_mm.shape or not np.any(raw_mask > 0):
        return None
    requested = color_mask(color_bgr, name)
    requested = cv2.morphologyEx(
        requested, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)
    )
    seed = np.where(
        (raw_mask > 0) & (requested > 0), 255, 0
    ).astype(np.uint8)
    minimum_pixels = int(np.clip(_env_float(
        "REBOT_COLOR_SEED_MIN_PIXELS", 25.0
    ), 12.0, 200.0))
    if int(np.count_nonzero(seed)) < minimum_pixels:
        return None
    instance_mask, separation = _separate_semantic_seeded_component(
        raw_mask, seed
    )
    valid = (instance_mask > 0) & depth_gate(
        depth_mm, near_mm, far_mm
    )
    values = np.asarray(depth_mm[valid], dtype=float)
    if values.size < max(40, minimum_pixels):
        return None
    contours, _ = cv2.findContours(
        instance_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    moments = cv2.moments(contour)
    if moments["m00"] <= 0:
        return None
    u = float(moments["m10"] / moments["m00"])
    v = float(moments["m01"] / moments["m00"])
    z_mm = float(np.median(values))
    x, y, width, height = cv2.boundingRect(contour)
    (_, _), (rect_width, rect_height), _ = cv2.minAreaRect(contour)
    min_width_mm, max_width_mm = _metric_size(
        rect_width, rect_height, z_mm, intr
    )
    seed_pixels = int(np.count_nonzero(seed))
    return Detection(
        box=(x, y, x + width, y + height),
        uv=(u, v), z_mm=z_mm,
        p_cam_mm=deproject(u, v, z_mm, intr),
        source="rgbd:color_seeded_instance",
        area_px=float(np.count_nonzero(instance_mask)),
        min_width_mm=min_width_mm,
        max_width_mm=max_width_mm,
        info={
            **(getattr(physical, "info", {}) or {}),
            "mask_u8": instance_mask,
            # Reuse the established seed-mask field consumed by registered
            # multi-view fusion.  Its kind is explicit so diagnostics do not
            # misrepresent this deterministic mask as a semantic-model mask.
            "semantic_seed_mask_u8": seed,
            "identity_seed_kind": "direct_color_mask",
            "identifier_colors": [name],
            "identifier_color_pixels": {name: seed_pixels},
            "semantic_seeded_instance_separation": separation,
            "segmentation_source":
                "color_seeded_rgbd_object_completion:native_mask",
            "depth_basis": "color_seeded_rgbd_object",
            "object_completion_validated": True,
            "semantic_identity_source": "rgbd:color_seeded_instance",
            "identity_basis":
                "exact_color_seed_inside_table_plane_depth_component",
            **rect_geometry(contour),
        },
    )


def complete_semantic_with_rgbd_object(
    color_bgr: np.ndarray,
    depth_mm: np.ndarray,
    intr: Intrinsics,
    semantic: Detection,
    *,
    near_mm: float = NEAR_MM,
    far_mm: float = FAR_MM,
) -> tuple[Detection | None, dict[str, Any]]:
    """Associate a semantic part mask with its complete physical RGB-D object.

    Text segmentation often returns the named/coloured part of an object (for
    example a purple cap) rather than a differently coloured connected body.
    SAM remains the identity authority, while the table-plane depth component
    supplies the whole physical mask only when it contains a substantial part
    of that exact semantic mask at compatible depth.  No class-specific names,
    colours, or dimensions are used.
    """
    semantic_mask = np.asarray(
        (getattr(semantic, "info", {}) or {}).get("mask_u8"),
        dtype=np.uint8,
    )
    if semantic_mask.shape != depth_mm.shape or not np.any(semantic_mask > 0):
        return None, {"reason": "semantic detection has no native pixel mask"}
    semantic_pixels = semantic_mask > 0
    semantic_count = int(np.count_nonzero(semantic_pixels))
    objects = segment_objects(
        color_bgr, depth_mm, intr,
        near_mm=near_mm, far_mm=far_mm,
        max_height_mm=220.0,
        min_area_px=max(60.0, 0.3 * float(MIN_AREA_PX)),
    )
    candidates: list[
        tuple[float, float, float, Detection, int, float]
    ] = []
    for physical in objects:
        raw_physical_mask = np.asarray(
            (getattr(physical, "info", {}) or {}).get("mask_u8"),
            dtype=np.uint8,
        )
        if raw_physical_mask.shape != depth_mm.shape:
            continue
        physical_mask, separation = \
            _separate_semantic_seeded_component(
                raw_physical_mask, semantic_mask
            )
        overlap = int(np.count_nonzero(
            semantic_pixels & (physical_mask > 0)
        ))
        overlap_fraction = overlap / max(1, semantic_count)
        if overlap < 12 or overlap_fraction < 0.20:
            continue
        depth_delta = abs(float(physical.z_mm) - float(semantic.z_mm))
        if depth_delta > 45.0:
            continue
        mask_area = int(np.count_nonzero(physical_mask > 0))
        area_ratio = float(mask_area) / max(1.0, float(semantic.area_px))
        contours, _ = cv2.findContours(
            physical_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            continue
        contour = max(contours, key=cv2.contourArea)
        moments = cv2.moments(contour)
        if moments["m00"] <= 0:
            continue
        u = float(moments["m10"] / moments["m00"])
        v = float(moments["m01"] / moments["m00"])
        valid_pixels = (physical_mask > 0) & depth_gate(
            depth_mm, near_mm, far_mm
        )
        values = np.asarray(depth_mm[valid_pixels], dtype=float)
        if values.size < 10:
            continue
        z_mm = float(np.median(values))
        x, y, bw, bh = cv2.boundingRect(contour)
        (_, _), (rw, rh), _ = cv2.minAreaRect(contour)
        min_width_mm, max_width_mm = _metric_size(rw, rh, z_mm, intr)
        physical = Detection(
            box=(x, y, x + bw, y + bh), uv=(u, v), z_mm=z_mm,
            p_cam_mm=deproject(u, v, z_mm, intr), source=physical.source,
            area_px=float(mask_area), min_width_mm=min_width_mm,
            max_width_mm=max_width_mm,
            info={
                **(getattr(physical, "info", {}) or {}),
                "mask_u8": physical_mask,
                "semantic_seeded_instance_separation": separation,
                **rect_geometry(contour),
            },
        )
        # The completion may be much larger than a cap/label, but an enormous
        # component is more likely a merged support or scene surface.
        center_delta_mm = float(np.linalg.norm(
            np.asarray(physical.p_cam_mm, dtype=float)
            - np.asarray(semantic.p_cam_mm, dtype=float)
        ))
        if area_ratio > 10.0 or center_delta_mm > 75.0:
            continue
        candidates.append((
            -overlap_fraction, center_delta_mm, depth_delta, physical,
            overlap, area_ratio,
        ))
    if not candidates:
        return None, {
            "reason": "no depth-connected tabletop object contained the semantic mask",
            "physical_candidates": len(objects),
        }
    _negative_overlap, center_delta_mm, depth_delta, physical, overlap, \
        area_ratio = min(
        candidates, key=lambda item: (item[0], item[1], item[2])
    )
    physical_info = getattr(physical, "info", {}) or {}
    completed = Detection(
        box=physical.box,
        uv=physical.uv,
        z_mm=physical.z_mm,
        p_cam_mm=physical.p_cam_mm,
        source=semantic.source,
        area_px=physical.area_px,
        min_width_mm=physical.min_width_mm,
        max_width_mm=physical.max_width_mm,
        info={
            **(getattr(semantic, "info", {}) or {}),
            **physical_info,
            "mask_u8": np.asarray(physical_info["mask_u8"], dtype=np.uint8),
            # Preserve the model-authoritative identity pixels separately from
            # the larger RGB-D completion.  Multi-view metric fusion uses this
            # seed to stop a depth component shared with a neighbour from
            # becoming part of the requested object's 3-D box.  The completed
            # mask remains the ordinary detection/grasp envelope.
            "semantic_seed_mask_u8": np.asarray(
                semantic_mask, dtype=np.uint8
            ).copy(),
            "segmentation_source": "semantic_seeded_rgbd_object_completion",
            "depth_basis": "semantic_seeded_rgbd_object",
            "object_completion_validated": True,
            "semantic_identity_source": semantic.source,
            "semantic_box": list(semantic.box),
            "semantic_anchor_uv": [float(v) for v in semantic.uv],
            "semantic_anchor_xyz_mm": [float(v) for v in semantic.p_cam_mm],
            "semantic_overlap_pixels": overlap,
            "semantic_overlap_fraction": round(-_negative_overlap, 4),
            "semantic_to_object_depth_delta_mm": round(depth_delta, 2),
            "semantic_to_object_center_delta_mm": round(center_delta_mm, 2),
            "completion_area_ratio": round(area_ratio, 3),
        },
    )
    return completed, {
        "reason": "semantic identity expanded to its overlapping RGB-D object component",
        "overlap_fraction": round(-_negative_overlap, 4),
        "area_ratio": round(area_ratio, 3),
        "depth_delta_mm": round(depth_delta, 2),
        "center_delta_mm": round(center_delta_mm, 2),
    }


def _separate_semantic_seeded_component(
    physical_mask_u8: np.ndarray,
    semantic_mask_u8: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Split thin bridges and retain only the object containing the SAM seed."""
    physical = np.where(
        np.asarray(physical_mask_u8) > 0, 255, 0
    ).astype(np.uint8)
    semantic = np.asarray(semantic_mask_u8) > 0
    original_area = int(np.count_nonzero(physical))
    best = physical
    best_area = original_area
    best_iterations = 0
    kernel = np.ones((3, 3), np.uint8)
    for iterations in range(1, 5):
        eroded = cv2.erode(physical, kernel, iterations=iterations)
        count, labels = cv2.connectedComponents(
            (eroded > 0).astype(np.uint8), connectivity=8
        )
        if count <= 1:
            continue
        overlaps = [
            int(np.count_nonzero((labels == label) & semantic))
            for label in range(1, count)
        ]
        if not overlaps or max(overlaps) <= 0:
            continue
        selected_label = int(np.argmax(overlaps)) + 1
        core = (labels == selected_label).astype(np.uint8) * 255
        restored = cv2.dilate(core, kernel, iterations=iterations)
        restored &= physical
        semantic_overlap = int(np.count_nonzero((restored > 0) & semantic))
        if semantic_overlap < max(12, int(0.20 * np.count_nonzero(semantic))):
            continue
        area = int(np.count_nonzero(restored))
        if 0 < area < best_area:
            best, best_area, best_iterations = restored, area, iterations
    return best, {
        "applied": best_iterations > 0,
        "erosion_iterations": best_iterations,
        "pixels_before": original_area,
        "pixels_after": best_area,
        "removed_fraction": round(
            1.0 - best_area / max(1, original_area), 4
        ),
    }


def validate_full_object_semantic_query(
    identifier: Detection,
    complete: Detection | None,
) -> tuple[Detection | None, dict[str, Any]]:
    """Require a complete-instance query to contain the identifier instance."""
    if complete is None:
        return None, {"reason": "complete-object semantic query found nothing"}
    identifier_mask = np.asarray(
        (getattr(identifier, "info", {}) or {}).get("mask_u8"),
        dtype=np.uint8,
    )
    complete_mask = np.asarray(
        (getattr(complete, "info", {}) or {}).get("mask_u8"),
        dtype=np.uint8,
    )
    if identifier_mask.shape != complete_mask.shape \
            or identifier_mask.ndim != 2:
        return None, {"reason": "semantic query masks are unavailable or mismatched"}
    identifier_pixels = identifier_mask > 0
    complete_pixels = complete_mask > 0
    identifier_count = int(np.count_nonzero(identifier_pixels))
    complete_count = int(np.count_nonzero(complete_pixels))
    overlap = int(np.count_nonzero(identifier_pixels & complete_pixels))
    overlap_fraction = overlap / max(1, identifier_count)
    area_ratio = complete_count / max(1, identifier_count)
    depth_delta = abs(float(complete.z_mm) - float(identifier.z_mm))
    center_delta = float(np.linalg.norm(
        np.asarray(complete.p_cam_mm, dtype=float)
        - np.asarray(identifier.p_cam_mm, dtype=float)
    ))
    accepted = bool(
        overlap >= 12
        and overlap_fraction >= 0.35
        and 1.10 <= area_ratio <= 10.0
        and depth_delta <= 45.0
        and center_delta <= 75.0
    )
    detail = {
        "reason": (
            "complete-object semantic mask contains the identifier instance"
            if accepted else
            "complete-object semantic mask did not safely match the identifier"
        ),
        "overlap_fraction": round(overlap_fraction, 4),
        "area_ratio": round(area_ratio, 3),
        "depth_delta_mm": round(depth_delta, 2),
        "center_delta_mm": round(center_delta, 2),
    }
    if not accepted:
        return None, detail
    return Detection(
        box=complete.box, uv=complete.uv, z_mm=complete.z_mm,
        p_cam_mm=complete.p_cam_mm, source=identifier.source,
        area_px=complete.area_px,
        min_width_mm=complete.min_width_mm,
        max_width_mm=complete.max_width_mm,
        info={
            **(getattr(complete, "info", {}) or {}),
            "object_completion_validated": True,
            "semantic_identity_source": identifier.source,
            "segmentation_source": "complete_object_semantic_query:native_mask",
            "depth_basis": "native_semantic_mask",
            "semantic_box": list(identifier.box),
            "semantic_anchor_uv": [float(v) for v in identifier.uv],
            "semantic_anchor_xyz_mm": [float(v) for v in identifier.p_cam_mm],
            "completion_query_overlap_fraction": round(overlap_fraction, 4),
            "completion_query_area_ratio": round(area_ratio, 3),
        },
    ), detail


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
    eligible = valid & (roi > 0)
    if not np.any(eligible):
        return None
    # Absolute depth gating alone is insufficient: two touching surfaces at
    # 205 mm and 310 mm both fit the historical 120 mm band and therefore
    # became one binary component. Grow from the identifier using a bounded
    # *local* depth edge. This can traverse the cap/body step of a marker while
    # refusing the much larger jump onto a support or background surface.
    seed_u = int(np.clip(round(float(target.uv[0])), 0, w - 1))
    seed_v = int(np.clip(round(float(target.uv[1])), 0, h - 1))
    if not eligible[seed_v, seed_u]:
        vv, uu = np.nonzero(eligible)
        seed_region = (
            (uu >= max(0, x1)) & (uu < min(w, x2))
            & (vv >= max(0, y1)) & (vv < min(h, y2))
        )
        if np.any(seed_region):
            uu_seed, vv_seed = uu[seed_region], vv[seed_region]
        else:
            uu_seed, vv_seed = uu, vv
        nearest = int(np.argmin(
            (uu_seed - seed_u) ** 2 + (vv_seed - seed_v) ** 2
        ))
        seed_u, seed_v = int(uu_seed[nearest]), int(vv_seed[nearest])
    depth_edge_mm = float(np.clip(_env_float(
        "REBOT_TARGET_DEPTH_EDGE_MM", 30.0
    ), 15.0, 35.0))
    flood_mask = np.ones((h + 2, w + 2), dtype=np.uint8)
    flood_mask[1:-1, 1:-1][eligible] = 0
    cv2.floodFill(
        np.asarray(depth_mm, dtype=np.float32).copy(), flood_mask,
        (seed_u, seed_v), 0.0,
        loDiff=depth_edge_mm, upDiff=depth_edge_mm,
        flags=8 | cv2.FLOODFILL_MASK_ONLY | (255 << 8),
    )
    component = (flood_mask[1:-1, 1:-1] == 255).astype(np.uint8) * 255
    component = cv2.morphologyEx(
        component, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8)
    )
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
            # This exact component is the authoritative input to metric 3-D
            # fitting.  Previously it was discarded here, so the downstream
            # fitter filled the whole bounding rectangle and turned a slender
            # colour-identified marker into a large chunk of the scene.
            "mask_u8": component,
            "segmentation_source": "target_seeded_depth_component",
            "depth_edge_mm": depth_edge_mm,
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
    diagnostics: dict[str, Any] | None = None,
) -> Detection | None:
    """RGB foreground mask inside a semantic whole-object box.

    This handles low objects whose depth is indistinguishable from the table.
    The semantic model supplies only a tight box; GrabCut supplies the contour,
    while the original color/VLM identifier is a definite foreground seed.
    """
    diag = diagnostics if diagnostics is not None else {}
    diag.clear()
    diag["semantic_source"] = semantic_box.source
    diag["semantic_box"] = list(semantic_box.box)
    h, w = color_bgr.shape[:2]
    x1, y1, x2, y2 = semantic_box.box
    x1, y1 = max(0, int(x1)), max(0, int(y1))
    x2, y2 = min(w, int(x2)), min(h, int(y2))
    if x2 - x1 < 8 or y2 - y1 < 8:
        diag.update(stage="semantic_box", reason="box smaller than 8 pixels")
        return None
    pad_x = max(4, int(0.10 * (x2 - x1)))
    pad_y = max(4, int(0.06 * (y2 - y1)))
    rx1, ry1 = max(0, x1 - pad_x), max(0, y1 - pad_y)
    rx2, ry2 = min(w, x2 + pad_x), min(h, y2 + pad_y)
    ix1, iy1, ix2, iy2 = identifier.box
    # A grounded SAM backend supplies a true promptable object mask.  Retain the
    # deterministic GrabCut path as a fallback for cached LocateAnything users.
    from .local_vision import segment_box as segment_local_box
    embedded_mask = (getattr(semantic_box, "info", {}) or {}).get("mask_u8")
    component = None
    if embedded_mask is not None:
        embedded_mask = np.asarray(embedded_mask, dtype=np.uint8)
        if embedded_mask.shape == depth_mm.shape:
            component = np.where(embedded_mask > 0, 255, 0).astype(np.uint8)
    if component is None:
        component = segment_local_box(
            color_bgr, (x1, y1, x2, y2)
        )
    segmentation_source = str(
        (getattr(semantic_box, "info", {}) or {}).get(
            "segmentation_source", f"{semantic_box.source}:native_mask"
        )
    )
    diag["native_mask_available"] = component is not None
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
            diag.update(stage="grabcut", reason="OpenCV GrabCut failed")
            return None
        component = np.where(
            (mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0
        ).astype(np.uint8)
    # A native instance mask must remain inside the detector box.  The padded
    # region exists only for GrabCut, which needs explicit background context.
    # Allowing a native mask into that padding made occasional SAM spill attach
    # a neighbouring support or table surface to a small target.
    roi = np.zeros_like(component)
    if segmentation_source == "grabcut":
        roi[ry1:ry2, rx1:rx2] = 255
    else:
        roi[y1:y2, x1:x2] = 255
    component &= roi
    diag["mask_pixels_before_cleanup"] = int(np.count_nonzero(component))
    component = cv2.morphologyEx(
        component, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8)
    )
    # Select the surface reached from the detector's target point without
    # crossing a physical RGB-D edge.  Plain 2-D connected components cannot
    # distinguish a target from background that a learned mask has joined to
    # it by one or two pixels.  OpenCV's floating-range flood fill compares
    # neighbouring depths, so sloped/rounded objects remain connected while a
    # sharp object/support boundary stops the fill.
    valid = depth_gate(depth_mm, near_mm, far_mm)
    eligible = (component > 0) & valid
    if not np.any(eligible):
        diag.update(
            stage="mask_depth", reason="fewer than 10 valid depth pixels",
            mask_pixels=int(np.count_nonzero(component)),
            valid_depth_pixels=0, valid_depth_fraction=0.0,
        )
        return None
    seed_u = int(round(float(identifier.uv[0])))
    seed_v = int(round(float(identifier.uv[1])))
    seed_u = int(np.clip(seed_u, 0, w - 1))
    seed_v = int(np.clip(seed_v, 0, h - 1))
    if not eligible[seed_v, seed_u]:
        vv, uu = np.nonzero(eligible)
        nearest = int(np.argmin((uu - seed_u) ** 2 + (vv - seed_v) ** 2))
        seed_u, seed_v = int(uu[nearest]), int(vv[nearest])
    depth_edge_mm = float(np.clip(float(os.getenv(
        "REBOT_SEMANTIC_DEPTH_EDGE_MM", "12"
    )), 5.0, 35.0))
    flood_mask = np.ones((h + 2, w + 2), dtype=np.uint8)
    flood_mask[1:-1, 1:-1][eligible] = 0
    cv2.floodFill(
        np.asarray(depth_mm, dtype=np.float32).copy(), flood_mask,
        (seed_u, seed_v), 0.0,
        loDiff=depth_edge_mm, upDiff=depth_edge_mm,
        flags=8 | cv2.FLOODFILL_MASK_ONLY | (255 << 8),
    )
    depth_component = flood_mask[1:-1, 1:-1] == 255
    retained = int(np.count_nonzero(depth_component))
    before_depth = int(np.count_nonzero(component))
    if retained < max(20, int(0.15 * max(1.0, float(identifier.area_px)))):
        diag.update(
            stage="depth_connectivity",
            reason="target-seeded depth component is too small",
            depth_component_pixels=retained,
            depth_edge_mm=depth_edge_mm,
        )
        return None
    component = depth_component.astype(np.uint8) * 255
    component = cv2.morphologyEx(
        component, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8)
    )
    diag["depth_edge_mm"] = depth_edge_mm
    diag["depth_component_pixels"] = retained
    diag["depth_rejected_pixels"] = before_depth - retained
    count, labels, _, _ = cv2.connectedComponentsWithStats(
        (component > 0).astype(np.uint8), connectivity=8
    )
    if count <= 1:
        diag.update(stage="mask_components", reason="mask has no foreground component")
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
        diag.update(stage="mask_contour", reason="foreground has no contour")
        return None
    contour = max(contours, key=cv2.contourArea)
    area = float(cv2.contourArea(contour))
    if area < max(30.0, 0.4 * float(identifier.area_px)):
        diag.update(
            stage="mask_area", reason="foreground mask is too small",
            contour_area_px=area,
            minimum_area_px=max(30.0, 0.4 * float(identifier.area_px)),
        )
        return None
    moments = cv2.moments(contour)
    if moments["m00"] <= 0:
        diag.update(stage="mask_moments", reason="foreground centroid is undefined")
        return None
    u = float(moments["m10"] / moments["m00"])
    v = float(moments["m01"] / moments["m00"])
    values = depth_mm[(component > 0) & valid]
    diag["mask_pixels"] = int(np.count_nonzero(component))
    diag["valid_depth_pixels"] = int(values.size)
    diag["valid_depth_fraction"] = round(
        float(values.size) / max(1, int(np.count_nonzero(component))), 3
    )
    if values.size < 10:
        diag.update(
            stage="mask_depth", reason="fewer than 10 valid depth pixels"
        )
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
    diag.update(
        ok=True, stage="complete", reason="semantic mask with robust in-mask depth",
        segmentation_source=segmentation_source,
        depth_median_mm=round(z, 1), contour_area_px=round(area, 1),
        width_mm=[round(float(min_mm), 1), round(float(max_mm), 1)],
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
            "semantic_identity_source": (
                (getattr(semantic_box, "info", {}) or {}).get(
                    "semantic_identity_source", identifier.source
                )
            ),
            "object_completion_validated": bool(
                (getattr(semantic_box, "info", {}) or {}).get(
                    "object_completion_validated", False
                )
            ),
            "depth_edge_mm": depth_edge_mm,
            "mask_pixels_before_cleanup": diag.get(
                "mask_pixels_before_cleanup"
            ),
            "depth_component_pixels": diag.get("depth_component_pixels"),
            "depth_rejected_pixels": diag.get("depth_rejected_pixels"),
            # Kept in-memory only. GraspGeometry.summary deliberately omits it.
            # The metric 3D box fitter needs the exact semantic pixels rather
            # than the axis-aligned detector box.
            "mask_u8": component,
            **({
                "semantic_seed_mask_u8": np.asarray(
                    (getattr(semantic_box, "info", {}) or {}).get(
                        "semantic_seed_mask_u8"
                    ),
                    dtype=np.uint8,
                ).copy(),
            } if (getattr(semantic_box, "info", {}) or {}).get(
                "semantic_seed_mask_u8"
            ) is not None else {}),
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
    from .local_vision import locate as locate_local, segment_box as segment_local_box

    result = locate_local(color_bgr, description)
    if not result or not result.get("found"):
        return None
    x1, y1, x2, y2 = (int(v) for v in result["box"])
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(depth_mm.shape[1], x2), min(depth_mm.shape[0], y2)
    if x2 <= x1 or y2 <= y1:
        return None
    backend = str(result.get("backend") or "locate_anything")
    native_mask = segment_local_box(color_bgr, (x1, y1, x2, y2))
    mask_u8 = None
    depth_basis = "axis_aligned_box"
    if native_mask is not None:
        native_mask = np.asarray(native_mask)
        if native_mask.shape != depth_mm.shape:
            # A semantic mask exists, so silently reverting to its much larger
            # axis-aligned box would allow background depth to establish target
            # identity. Fail this observation and let the bounded view retry.
            return None
        mask_u8 = np.where(native_mask > 0, 255, 0).astype(np.uint8)
        roi = np.zeros_like(mask_u8)
        roi[y1:y2, x1:x2] = 255
        mask_u8 &= roi
        valid_mask = (mask_u8 > 0) & depth_gate(
            depth_mm, near_mm, far_mm
        )
        rows, cols = np.nonzero(valid_mask)
        if rows.size < 10:
            # SAM supplied pixel authority but those pixels have no trustworthy
            # range. Never substitute table/background pixels from the box.
            return None
        values = np.asarray(depth_mm[valid_mask], dtype=float)
        z = float(np.median(values))
        # Locate the 3-D anchor at the median of the ranged mask pixels. This
        # remains on the visible object even when its detector box is clipped
        # against an image edge and mostly contains distant background.
        u, v = float(np.median(cols)), float(np.median(rows))
        contours, _ = cv2.findContours(
            mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            return None
        contour = max(contours, key=cv2.contourArea)
        (_, _), (rw, rh), _ = cv2.minAreaRect(contour)
        min_mm, max_mm = _metric_size(rw, rh, z, intr)
        area_px = float(np.count_nonzero(mask_u8))
        depth_basis = "native_semantic_mask"
    else:
        # LocateAnything and older box-only backends have no pixel mask. Keep
        # their legacy behavior; hybrid SAM3 runs take the mask-authoritative
        # branch above.
        patch = depth_mm[y1:y2, x1:x2]
        values = patch[depth_gate(patch, near_mm, far_mm)]
        if values.size < 10:
            return None
        z = float(np.median(values))
        u, v = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        min_mm, max_mm = _metric_size(x2 - x1, y2 - y1, z, intr)
        area_px = float((x2 - x1) * (y2 - y1))
    return Detection(box=(x1, y1, x2, y2), uv=(u, v), z_mm=z,
                     p_cam_mm=deproject(u, v, z, intr), source=backend,
                     area_px=area_px,
                     min_width_mm=min_mm, max_width_mm=max_mm,
                     info={"confidence": result.get("confidence"),
                           "detector_backend": backend,
                           "model_version": result.get("model_version"),
                           "box_axis_aligned": True,
                           "depth_basis": depth_basis,
                           **({"mask_u8": mask_u8}
                              if mask_u8 is not None else {})})


def detect(color_bgr: np.ndarray, depth_mm: np.ndarray, intr: Intrinsics, description: str,
           *, near_mm: float = NEAR_MM, far_mm: float = FAR_MM,
           allow_vlm: bool = True,
           semantic_noun_first: bool = False) -> tuple[Detection | None, dict[str, Any]]:
    """Locate ``description`` in a wrist RGB-D frame.

    Hybrid mode resolves the semantic noun first and verifies its exact mask
    against any requested colour. If that semantic result fails, a distinctive
    chromatic colour may seed one physical RGB-D component as a fallback;
    achromatic colour alone is never allowed to discard the noun identity.

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
    hybrid_identity = os.getenv(
        "REBOT_OBJECT_GEOMETRY_MODE", "legacy"
    ).strip().lower() == "hybrid"
    names = colors_of(description)
    require_all_colors = _requires_all_colors(description)
    color_seeded_fallback_enabled = os.getenv(
        "REBOT_COLOR_SEEDED_PRIMARY", "1"
    ).strip().lower() not in {"0", "false", "no", "off"}
    distinctive_names = [
        name for name in names if _globally_distinctive_color(name)
    ]
    color_seeded_fallback_allowed = bool(
        hybrid_identity
        and color_seeded_fallback_enabled
        and len(names) == 1
        and not require_all_colors
        and distinctive_names
    )
    # Pick targets retain the proven deterministic single-colour path. Exact
    # placement references opt into noun-first mode because their semantic
    # class (for example cup versus marker) defines the destination itself.
    if color_seeded_fallback_allowed and not semantic_noun_first:
        found = detect_color_seeded_instance(
            color_bgr, depth_mm, intr, names[0],
            near_mm=near_mm, far_mm=far_mm,
        )
        why["color_seeded_primary_result"] = (
            "found" if found is not None else "not found"
        )
        if found is not None:
            why["used"] = found.source
            why["color_seeded_primary"] = {
                "color": names[0],
                "reason": (
                    "pick-target policy: exact requested-colour pixels "
                    "selected one physical RGB-D object before semantics"
                ),
            }
            return found, why
    # Noun-first references resolve the semantic object and verify its exact
    # mask against the requested colour. Only a distinctive chromatic colour
    # may recover a semantic miss; achromatic colour never replaces the noun.
    if allow_vlm and vision_enabled and vision_primary:
        found = detect_vlm(
            color_bgr, depth_mm, intr, description,
            near_mm=near_mm, far_mm=far_mm,
        )
        why["semantic_result"] = found.source if found else "not found"
        if found is None:
            from .local_vision import tracking_lost as semantic_tracking_lost
            if semantic_tracking_lost(description):
                why.update(
                    semantic_tracking_lost=True,
                    reason=(
                        "SAM3 video lost the locked target instance; "
                        "only a distinctive exact-colour seed may recover it"
                    ),
                )
                if not color_seeded_fallback_allowed:
                    why["used"] = None
                    return None, why
        if found is not None:
            if hybrid_identity:
                complete_query = detect_vlm(
                    color_bgr, depth_mm, intr,
                    f"the complete physical {description}, including its "
                    "entire body and all connected parts",
                    near_mm=near_mm, far_mm=far_mm,
                )
                completed, completion_why = \
                    validate_full_object_semantic_query(
                        found, complete_query
                    )
                why["semantic_complete_object_query"] = completion_why
                if completed is None:
                    completed, completion_why = \
                        complete_semantic_with_rgbd_object(
                            color_bgr, depth_mm, intr, found,
                            near_mm=near_mm, far_mm=far_mm,
                        )
                    why["semantic_rgbd_object_completion"] = completion_why
                if completed is not None:
                    found = completed
            if require_all_colors:
                # Semantic grounding understands the noun (for example,
                # ``sponge``); HSV does not reliably see every named colour on
                # a shaded or edge-on face. Keep this as explainable telemetry,
                # not a veto over a stable semantic-primary result.
                why["semantic_color_check"] = (
                    "all requested colours visible in box"
                    if _box_has_all_colors(color_bgr, found.box, names)
                    else "incomplete under HSV; accepted semantic noun match"
                )
            elif len(names) == 1 and color_veto_enabled():
                # A single named colour is the one case where HSV is stronger
                # evidence than the noun match. SAM 3 has been observed to
                # return whichever object is most salient and ignore the colour
                # word entirely -- offering a green marker for "red marker",
                # "blue marker" and back. A near-total absence of the named
                # colour on the object's own surface is positive evidence of
                # the wrong instance, so refuse it and let the colour-primary
                # RGB-D path below answer instead. Deliberately not applied to
                # multi-colour prompts, where a shaded face routinely hides one
                # of the named colours.
                fraction = semantic_color_fraction(
                    color_bgr, depth_mm, found, names[0]
                )
                if fraction is None:
                    why["semantic_color_check"] = (
                        "too little measurable surface; accepted noun match"
                    )
                elif fraction < COLOR_VETO_FRACTION:
                    why["semantic_color_veto"] = {
                        "color": names[0],
                        "fraction": round(fraction, 4),
                        "required": COLOR_VETO_FRACTION,
                        "reason": (
                            f"semantic match shows {fraction:.1%} "
                            f"{names[0]}; below the "
                            f"{COLOR_VETO_FRACTION:.0%} required"
                        ),
                    }
                    found = None
                else:
                    why["semantic_color_check"] = (
                        f"{fraction:.1%} {names[0]} on the object"
                    )
            if found is not None:
                why["used"] = found.source
                return found, why
    if color_seeded_fallback_allowed:
        found = detect_color_seeded_instance(
            color_bgr, depth_mm, intr, names[0],
            near_mm=near_mm, far_mm=far_mm,
        )
        why["color_seeded_fallback_result"] = (
            "found" if found is not None else "not found"
        )
        if found is not None:
            why["used"] = found.source
            why["color_seeded_fallback"] = {
                "color": names[0],
                "reason": (
                    "semantic noun was unavailable or failed exact-mask "
                    "colour verification; distinctive requested-colour "
                    "pixels selected one physical RGB-D object"
                ),
            }
            return found, why
    # Hybrid identity needs a physical object candidate even when open-vocabulary
    # grounding misses. The depth/table component is authoritative here; HSV is
    # only evidence inside that component. Legacy detection order is untouched.
    if hybrid_identity and names and distinctive_names:
        found = detect_colored_tabletop_object(
            color_bgr, depth_mm, intr, names,
            require_all=require_all_colors,
            near_mm=near_mm, far_mm=far_mm,
        )
        why["rgbd_colored_object_result"] = (
            "found" if found is not None else "not found"
        )
        if found is not None:
            why["used"] = found.source
            return found, why
    elif hybrid_identity and names:
        why["rgbd_colored_object_result"] = (
            "skipped: achromatic colour cannot replace semantic noun identity"
        )
        why["used"] = None
        return None, why
    why["color_word"] = names[0] if len(names) == 1 else None
    why["color_words"] = names
    if names:
        found = (
            detect_color_conjunction(
                color_bgr, depth_mm, intr, names,
                near_mm=near_mm, far_mm=far_mm,
            ) if require_all_colors else
            detect_color_union(
                color_bgr, depth_mm, intr, names,
                near_mm=near_mm, far_mm=far_mm,
            ) if len(names) > 1 else None
        )
        if found is not None:
            why["used"] = found.source
            return found, why
        if require_all_colors:
            why["color_result"] = (
                "no nearby RGB-D object contained every requested colour"
            )
            # Never degrade a conjunction to one colour: that is precisely how
            # a green marker was mistaken for a green-and-yellow sponge.
            if not allow_vlm:
                why["used"] = None
                return None, why
            found = detect_vlm(color_bgr, depth_mm, intr, description,
                               near_mm=near_mm, far_mm=far_mm)
            if found is not None:
                why["fallback_semantic_color_check"] = (
                    "all requested colours visible in box"
                    if _box_has_all_colors(color_bgr, found.box, names)
                    else "incomplete under HSV; accepted semantic noun match"
                )
            why["used"] = found.source if found else None
            return found, why
        for name in names:
            found = detect_color(
                color_bgr, depth_mm, intr, name,
                near_mm=near_mm, far_mm=far_mm,
            )
            if found is not None:
                why["used"] = found.source
                return found, why
        why["color_result"] = "no depth-gated blob of the requested colour(s)"
    if not allow_vlm:
        why["used"] = None
        return None, why
    found = detect_vlm(color_bgr, depth_mm, intr, description,
                       near_mm=near_mm, far_mm=far_mm)
    why["used"] = found.source if found else None
    if found is None:
        why["vlm_result"] = "not found (or LocateAnything disabled)"
    return found, why
