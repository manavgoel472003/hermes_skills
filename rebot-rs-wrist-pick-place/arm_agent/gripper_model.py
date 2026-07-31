"""Gripper limits, in the units they were measured in: degrees and radians.

The jaw positions were measured on the real gripper and are used verbatim:

    close     0.00 deg   (set_zero fired with the jaws closed)
    open    338.63 deg   (hand-opened until the reading stopped climbing)

Those two angles ARE the limits. Every open/close command targets one of them, and
every read-back is reported as radians and degrees -- there is no percentage-of-
travel abstraction, because it only forced the real numbers to be translated back.

Opening is the more positive direction on this gripper, but nothing here assumes
that: the clamp orders the two limits by value, so re-measuring with the sign the
other way needs no code change.

Commands are additionally clamped to a HARD, non-configurable range of
15-150 deg, so no skill or env override can drive the jaws to either stop.

Override the measured limits without editing code:
    REBOT_GRIPPER_OPEN_DEG / REBOT_GRIPPER_CLOSE_DEG      (degrees)
    REBOT_GRIPPER_OPEN_RAD / REBOT_GRIPPER_CLOSE_RAD      (radians, take priority)
"""
from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
CALIB_DIR = Path(os.getenv("REBOT_WRIST_CALIB", str(ROOT / "calib")))
CALIB_FILE = CALIB_DIR / "wrist_gripper.json"


def _limit(rad_env: str, deg_env: str, default_deg: float) -> float:
    """Resolve a limit from an explicit radian env, else a degree env, else the
    measured default (given in degrees)."""
    raw = os.getenv(rad_env)
    if raw:
        try:
            return float(raw)
        except ValueError:
            pass
    raw = os.getenv(deg_env)
    if raw:
        try:
            return math.radians(float(raw))
        except ValueError:
            pass
    return math.radians(default_deg)


# The measured limits.
# From the free-drive sweep 0 -> 48.10 -> 133.00 -> 186.16 -> 272.60 -> 338.63 (stop).
# An earlier pair (22.92 / 141.13) came from a DIFFERENT set_zero frame and does not
# apply: with the jaws closed the gripper now reads ~0 deg, matching this frame.
CLOSE_RAD = _limit("REBOT_GRIPPER_CLOSE_RAD", "REBOT_GRIPPER_CLOSE_DEG", 0.0)
OPEN_RAD = _limit("REBOT_GRIPPER_OPEN_RAD", "REBOT_GRIPPER_OPEN_DEG", 338.63)
# HARD command bounds. Not configurable: no command, from any skill, with any env
# override, is ever sent outside this range. The measured limits above can sit
# outside it (the jaws really do reach 0 and 338.63 deg) -- those are used to judge
# whether a READING is sane, while these bound what we ever DRIVE the motor to.
HARD_MIN_DEG = 15.0
HARD_MAX_DEG = 150.0
HARD_MIN_RAD = math.radians(HARD_MIN_DEG)
HARD_MAX_RAD = math.radians(HARD_MAX_DEG)
# How far outside the two limits a reading may sit before it is treated as a fault
# (stalled, or the position reference has moved) rather than a jaw position.
SLOP_RAD = math.radians(float(os.getenv("REBOT_GRIPPER_SLOP_DEG", "") or 15.0))
# Extra gap either side of an object when opening to fit it (mm).
CLEARANCE_MM = float(os.getenv("REBOT_GRIPPER_CLEARANCE_MM", "") or 12.0)


def deg(rad: float) -> float:
    return math.degrees(float(rad))


def limits(open_rad: float | None = None, close_rad: float | None = None
           ) -> tuple[float, float]:
    """(close, open) in radians, defaulting to the measured pair."""
    return (CLOSE_RAD if close_rad is None else float(close_rad),
            OPEN_RAD if open_rad is None else float(open_rad))


def clamp_rad(rad: float, limit_rads: tuple[float, float] | None = None) -> float:
    """Clamp a target between the close and open limits, then to the HARD bounds.

    The limit pair is ordered by value rather than assuming which is numerically
    larger, so it can be re-measured with either sign convention. The hard bounds
    are applied last and unconditionally, so nothing can drive the jaws below
    HARD_MIN_DEG or above HARD_MAX_DEG.
    """
    close_rad, open_rad = limit_rads or (CLOSE_RAD, OPEN_RAD)
    lo, hi = min(close_rad, open_rad), max(close_rad, open_rad)
    value = max(lo, min(hi, float(rad)))
    return max(HARD_MIN_RAD, min(HARD_MAX_RAD, value))


def out_of_range(rad: float, limit_rads: tuple[float, float] | None = None,
                 slop: float | None = None) -> bool:
    """True if a reading sits outside the limits by more than the slop."""
    close_rad, open_rad = limit_rads or (CLOSE_RAD, OPEN_RAD)
    lo, hi = min(close_rad, open_rad), max(close_rad, open_rad)
    margin = SLOP_RAD if slop is None else float(slop)
    return not (lo - margin <= float(rad) <= hi + margin)


def target_for(action: str, limit_rads: tuple[float, float] | None = None) -> float:
    """The commanded angle for 'open' or 'close', inside the hard bounds."""
    close_rad, open_rad = limit_rads or (CLOSE_RAD, OPEN_RAD)
    return clamp_rad(open_rad if action == "open" else close_rad, (close_rad, open_rad))


# -- millimetres per radian (optional; only needed to open to an object's size) ------
def load_calibration() -> dict[str, Any] | None:
    raw = os.getenv("REBOT_GRIPPER_MM_PER_RAD")
    if raw:
        try:
            return {"mm_per_rad": float(raw), "source": "env"}
        except ValueError:
            pass
    try:
        return json.loads(CALIB_FILE.read_text())
    except Exception:  # noqa: BLE001
        return None


def save_calibration(mm_per_rad: float, detail: dict[str, Any] | None = None) -> Path:
    CALIB_DIR.mkdir(parents=True, exist_ok=True)
    CALIB_FILE.write_text(json.dumps({
        "mm_per_rad": float(mm_per_rad),
        "close_rad": CLOSE_RAD, "close_deg": round(deg(CLOSE_RAD), 2),
        "open_rad": OPEN_RAD, "open_deg": round(deg(OPEN_RAD), 2),
        "max_aperture_mm": round(abs(float(mm_per_rad) * (OPEN_RAD - CLOSE_RAD)), 2),
        "saved_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "detail": detail or {},
    }, indent=2))
    return CALIB_FILE


def _curve_points(calib: dict[str, Any] | None = None) -> list[tuple[float, float]]:
    """Validated ``(angle_rad, aperture_mm)`` points, including closed zero."""
    calib = load_calibration() if calib is None else calib
    if not calib or not isinstance(calib.get("points"), list):
        return []
    points: list[tuple[float, float]] = []
    for item in calib["points"]:
        try:
            rad = float(item["angle_rad"])
            mm = float(item["aperture_mm"])
        except (KeyError, TypeError, ValueError):
            return []
        if not (math.isfinite(rad) and math.isfinite(mm) and mm >= 0):
            return []
        points.append((rad, mm))
    points.append((float(CLOSE_RAD), 0.0))
    points = sorted(set(points), key=lambda pair: pair[0])
    if len(points) < 2:
        return []
    direction = 1.0 if OPEN_RAD >= CLOSE_RAD else -1.0
    ordered = sorted(points, key=lambda pair: direction * (pair[0] - CLOSE_RAD))
    if any(ordered[i + 1][1] <= ordered[i][1]
           for i in range(len(ordered) - 1)):
        return []
    return ordered


def _interp(x: float, points: list[tuple[float, float]], *, inverse: bool = False
            ) -> float:
    pairs = [(b, a) for a, b in points] if inverse else list(points)
    pairs.sort(key=lambda pair: pair[0])
    if x <= pairs[0][0]:
        lo, hi = pairs[0], pairs[1]
    elif x >= pairs[-1][0]:
        lo, hi = pairs[-2], pairs[-1]
    else:
        lo, hi = pairs[0], pairs[-1]
        for left, right in zip(pairs, pairs[1:]):
            if left[0] <= x <= right[0]:
                lo, hi = left, right
                break
    span = hi[0] - lo[0]
    if abs(span) <= 1e-9:
        return float(lo[1])
    return float(lo[1] + (x - lo[0]) * (hi[1] - lo[1]) / span)


def add_calibration_point(angle_deg: float, aperture_mm_value: float) -> dict[str, Any]:
    """Add a ruler-measured empty-jaw point; this never commands the gripper."""
    angle_rad = math.radians(float(angle_deg))
    if not HARD_MIN_RAD <= angle_rad <= HARD_MAX_RAD:
        return {"ok": False, "reason": f"angle must be within the hard "
                f"{HARD_MIN_DEG:.0f}-{HARD_MAX_DEG:.0f}deg command limits"}
    if not math.isfinite(aperture_mm_value) or aperture_mm_value <= 0:
        return {"ok": False, "reason": "aperture must be a positive millimetre value"}

    old = load_calibration() or {}
    raw_points = list(old.get("points") or [])
    # Seed a legacy one-point calibration from its original measured span, not
    # from arbitrary values extrapolated by the old linear model.
    if not raw_points:
        detail = old.get("detail") if isinstance(old.get("detail"), dict) else {}
        try:
            legacy_rad = float(detail["measured_rad"])
            legacy_mm = float(detail["aperture_mm"])
            raw_points.append({"angle_rad": legacy_rad,
                               "aperture_mm": legacy_mm})
        except (KeyError, TypeError, ValueError):
            pass
    raw_points = [
        item for item in raw_points
        if abs(float(item.get("angle_rad", 1e9)) - angle_rad) > math.radians(0.1)
    ]
    raw_points.append({"angle_rad": angle_rad,
                       "aperture_mm": float(aperture_mm_value)})
    raw_points.sort(key=lambda item: float(item["angle_rad"]))
    trial = {"points": raw_points}
    points = _curve_points(trial)
    if not points:
        return {"ok": False, "reason": "point would make the angle-to-aperture "
                "curve non-monotonic; check the ruler measurement"}
    CALIB_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": "piecewise_linear",
        "points": [
            {"angle_rad": float(item["angle_rad"]),
             "angle_deg": round(deg(float(item["angle_rad"])), 2),
             "aperture_mm": float(item["aperture_mm"])}
            for item in raw_points
        ],
        "close_rad": CLOSE_RAD, "close_deg": round(deg(CLOSE_RAD), 2),
        "open_rad": OPEN_RAD, "open_deg": round(deg(OPEN_RAD), 2),
        "saved_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "detail": {"method": "manual_empty_jaw_curve",
                   "gripper_commanded": False},
    }
    CALIB_FILE.write_text(json.dumps(payload, indent=2))
    return {"ok": True, "model": "piecewise_linear",
            "point_added": {"angle_deg": round(float(angle_deg), 2),
                            "aperture_mm": round(float(aperture_mm_value), 2)},
            "points": payload["points"], "saved": str(CALIB_FILE),
            "note": "calibration was saved only; no gripper command was sent"}


def mm_per_rad() -> float | None:
    calib = load_calibration()
    if not calib:
        return None
    try:
        value = float(calib["mm_per_rad"])
    except (KeyError, TypeError, ValueError):
        return None
    return value if value > 0 else None


def aperture_mm(rad: float) -> float | None:
    """Jaw opening (mm) at an angle, measured from the close limit."""
    curve = _curve_points()
    if curve:
        return max(0.0, _interp(float(rad), curve))
    scale = mm_per_rad()
    if scale is None:
        return None
    return max(0.0, scale * abs(float(rad) - CLOSE_RAD))


def rad_for_aperture(width_mm: float,
                     limit_rads: tuple[float, float] | None = None) -> float | None:
    """Angle that opens the jaws to ``width_mm``, clamped to the limits."""
    close_rad, open_rad = limit_rads or (CLOSE_RAD, OPEN_RAD)
    curve = _curve_points()
    if curve:
        return clamp_rad(_interp(float(width_mm), curve, inverse=True),
                         (close_rad, open_rad))
    scale = mm_per_rad()
    if scale is None:
        return None
    direction = 1.0 if open_rad >= close_rad else -1.0
    return clamp_rad(close_rad + direction * float(width_mm) / scale,
                     (close_rad, open_rad))


def max_aperture_mm(limit_rads: tuple[float, float] | None = None) -> float | None:
    """Widest object admitted at the largest angle software may command.

    The measured open stop can lie beyond ``HARD_MAX_RAD``. Using that raw stop
    here would claim that objects fit even though every live command is clamped
    earlier, so compute aperture at the clamped open target instead.
    """
    close_rad, open_rad = limit_rads or (CLOSE_RAD, OPEN_RAD)
    commanded_open = clamp_rad(open_rad, (close_rad, open_rad))
    return aperture_mm(commanded_open)


def fit_from_grasp(object_width_mm: float, contact_rad: float) -> dict[str, Any]:
    """Calibrate mm-per-radian from one grip on an object of measured width."""
    travel = abs(float(contact_rad) - CLOSE_RAD)
    if travel <= 1e-6:
        return {"ok": False, "reason": "jaws did not travel -- was anything gripped?"}
    if object_width_mm <= 1.0:
        return {"ok": False, "reason": f"implausible object width {object_width_mm:.1f}mm"}
    scale = float(object_width_mm) / travel
    path = save_calibration(scale, {"object_width_mm": round(float(object_width_mm), 2),
                                    "contact_rad": float(contact_rad),
                                    "contact_deg": round(deg(contact_rad), 2)})
    return {"ok": True, "mm_per_rad": round(scale, 3),
            "max_aperture_mm": round(max_aperture_mm() or 0.0, 1),
            "saved": str(path)}


def set_span(aperture_mm_value: float, at_fraction: float | None = None) -> dict[str, Any]:
    """Set the scale from a measured jaw gap at a known travel fraction.

    With no fraction, the gap is assumed to have been measured after the normal
    ``gripper open`` command, i.e. at the hard-clamped safe open target. An
    explicit fraction refers to the measured close-to-open mechanical travel.
    """
    if aperture_mm_value <= 1.0:
        return {"ok": False, "reason": f"implausible aperture {aperture_mm_value:.1f}mm"}
    full_travel = abs(OPEN_RAD - CLOSE_RAD)
    if full_travel <= 1e-6:
        return {"ok": False, "reason": "gripper close/open limits are identical"}
    if at_fraction is None:
        measured_rad = clamp_rad(OPEN_RAD, (CLOSE_RAD, OPEN_RAD))
        measured_travel = abs(measured_rad - CLOSE_RAD)
        fraction = measured_travel / full_travel
    else:
        fraction = float(at_fraction)
        if not 0.0 < fraction <= 1.0:
            return {"ok": False, "reason": "at_fraction must be greater than 0 and at most 1"}
        measured_travel = full_travel * fraction
        measured_rad = CLOSE_RAD + (OPEN_RAD - CLOSE_RAD) * fraction
    if measured_travel <= 1e-6:
        return {"ok": False, "reason": "measured jaw travel is zero"}
    scale = float(aperture_mm_value) / measured_travel
    path = save_calibration(scale, {"method": "measured_span",
                                    "aperture_mm": float(aperture_mm_value),
                                    "measured_rad": float(measured_rad),
                                    "measured_deg": round(deg(measured_rad), 2),
                                    "at_fraction": float(fraction)})
    return {"ok": True, "mm_per_rad": round(scale, 3),
            "max_aperture_mm": round(max_aperture_mm() or 0.0, 1),
            "measured_at_deg": round(deg(measured_rad), 2),
            "saved": str(path)}


def describe() -> dict[str, Any]:
    scale = mm_per_rad()
    out: dict[str, Any] = {
        "close_rad": round(CLOSE_RAD, 5), "close_deg": round(deg(CLOSE_RAD), 2),
        "open_rad": round(OPEN_RAD, 5), "open_deg": round(deg(OPEN_RAD), 2),
        "slop_deg": round(deg(SLOP_RAD), 2),
        "hard_min_deg": HARD_MIN_DEG, "hard_max_deg": HARD_MAX_DEG,
        "commands_open_deg": round(deg(target_for("open")), 2),
        "commands_close_deg": round(deg(target_for("close")), 2),
        "calibrated_mm": scale is not None,
    }
    if scale is not None:
        out["mm_per_rad"] = round(scale, 3)
        out["max_aperture_mm"] = round(max_aperture_mm() or 0.0, 1)
    curve = _curve_points()
    if curve:
        out["aperture_model"] = "piecewise_linear"
        out["aperture_points"] = [
            {"angle_deg": round(deg(rad), 2), "aperture_mm": round(mm, 2)}
            for rad, mm in curve
        ]
    return out
