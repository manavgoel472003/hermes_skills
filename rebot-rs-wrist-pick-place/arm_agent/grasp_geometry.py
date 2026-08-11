"""Geometry-only grasp selection for the wrist RGB-D picker.

The target detector may identify a small visual feature (for example a green
bottle cap).  Motion safety must still use the complete depth envelope around
that feature.  This module associates those two observations and produces a
bounded, explainable parallel-jaw grasp proposal without object-class rules.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, asdict, field, replace
from typing import Any

import numpy as np

from .wrist_detect import (
    Detection,
    deproject,
    segment_objects,
    segment_target_envelope,
)


def _angle_delta_deg(target: float, reference: float) -> float:
    """Smallest signed difference for axes that are symmetric modulo 180°."""
    return ((float(target) - float(reference) + 90.0) % 180.0) - 90.0


def _contains(box: tuple[int, int, int, int], uv: tuple[float, float],
              pad_px: int = 8) -> bool:
    x1, y1, x2, y2 = box
    u, v = uv
    return x1 - pad_px <= u <= x2 + pad_px and y1 - pad_px <= v <= y2 + pad_px


def associate_envelope(target: Detection,
                       envelopes: list[Detection]) -> Detection | None:
    """Find the full depth object containing a target feature."""
    candidates = [item for item in envelopes if _contains(item.box, target.uv)]
    if not candidates:
        return None
    # Prefer a containing region, then the nearest centre.  Large table/gripper
    # fragments lose to the compact object centred on the target feature.
    return min(
        candidates,
        key=lambda item: (
            math.hypot(item.uv[0] - target.uv[0], item.uv[1] - target.uv[1]),
            item.area_px,
        ),
    )


@dataclass(frozen=True)
class GraspGeometry:
    ok: bool
    reason: str
    grasp_detection: Detection | None
    envelope_detection: Detection | None
    rpy_rad: tuple[float, float, float] | None
    yaw_correction_deg: float = 0.0
    final_approach_mm: float = 40.0
    used_full_envelope: bool = False
    candidates: list[dict[str, Any]] = field(default_factory=list)
    object_box_3d: dict[str, Any] | None = None

    def summary(self) -> dict[str, Any]:
        data = asdict(self)
        for key in ("grasp_detection", "envelope_detection"):
            det = data.get(key)
            if det is not None:
                data[key] = {
                    "source": det["source"],
                    "uv": [round(float(v), 1) for v in det["uv"]],
                    "z_mm": round(float(det["z_mm"]), 1),
                    "min_width_mm": (
                        None if det["min_width_mm"] is None
                        else round(float(det["min_width_mm"]), 1)
                    ),
                    "max_width_mm": (
                        None if det["max_width_mm"] is None
                        else round(float(det["max_width_mm"]), 1)
                    ),
                    "long_axis_deg": det["info"].get("long_axis_deg"),
                }
        if data["rpy_rad"] is not None:
            data["rpy_rad"] = [round(float(v), 4) for v in data["rpy_rad"]]
        return data


def plan_grasp_geometry(
    color: np.ndarray,
    depth: np.ndarray,
    intr: tuple[float, float, float, float],
    target: Detection,
    *,
    current_rpy: tuple[float, float, float],
    max_aperture_mm: float,
    jaw_long_axis_camera_deg: float | None,
    max_yaw_delta_deg: float = 10.0,
    yaw_delta_bounds_deg: tuple[float, float] | None = None,
    yaw_measurement_slop_deg: float = 0.5,
    max_residual_yaw_deg: float = 0.0,
    allow_aperture_derived_residual: bool = False,
    final_approach_mm: float = 40.0,
    min_feature_prominence_mm: float = 25.0,
    require_axis_for_elongated: bool = True,
    semantic_envelope: Detection | None = None,
    metric_jaw_width_mm: float | None = None,
    metric_major_mm: float | None = None,
) -> GraspGeometry:
    """Choose a grasp region and a heavily bounded wrist-yaw adjustment.

    A substantially larger depth envelope replaces a small color/VLM feature
    whenever that envelope itself fits the jaws.  If it does not fit, the
    feature is accepted only when it is clearly proud of the envelope in depth;
    otherwise the gripper would have to pass through the body to reach it.
    """
    # A promptable whole-object mask with valid in-mask depth is authoritative.
    # Do not run table-plane segmentation eagerly: aside from added latency, its
    # downward-camera assumption is unreliable in the nearly horizontal ready
    # view. Seed growth and plane separation are fallback sources only.
    if semantic_envelope is not None and _contains(
        semantic_envelope.box, target.uv
    ):
        envelope = semantic_envelope
    else:
        seeded = segment_target_envelope(color, depth, intr, target)
        # The seed route recovers tall features that plane segmentation can miss,
        # but on a low object it can depth-connect into the table. Association
        # prefers the compact candidate centred on the identifier feature.
        envelopes = segment_objects(color, depth, intr)
        if seeded is not None:
            envelopes.append(seeded)
        envelope = associate_envelope(target, envelopes)
    grasp = target
    used_envelope = False
    feature_clearance_note: str | None = None

    if envelope is None and target.source != "depth:object":
        return GraspGeometry(
            False,
            "the visual target was found, but its complete depth envelope could "
            "not be separated from the table; refusing feature-only motion",
            None, None, None,
        )
    if envelope is None:
        envelope = target
    envelope_candidates = (
        [dict(item) for item in envelope.info.get("cross_section_candidates", [])]
        if isinstance(envelope.info.get("cross_section_candidates"), list)
        else []
    )
    metric_width: float | None = None
    metric_major: float | None = None
    if metric_jaw_width_mm is not None:
        try:
            metric_width = float(metric_jaw_width_mm)
            metric_major = float(
                metric_major_mm if metric_major_mm is not None
                else envelope.max_width_mm or metric_width
            )
        except (TypeError, ValueError):
            metric_width, metric_major = math.nan, math.nan
        if not math.isfinite(metric_width) or metric_width <= 0.0 \
                or not math.isfinite(metric_major) \
                or metric_major < metric_width:
            return GraspGeometry(
                False, "metric 3D jaw dimensions are invalid",
                grasp, envelope, None, candidates=envelope_candidates,
            )

    if envelope.min_width_mm is not None:
        try:
            local_width = float(envelope.info.get("local_grasp_width_mm"))
            if not math.isfinite(local_width) or local_width <= 0:
                local_width = float(envelope.min_width_mm)
        except (TypeError, ValueError):
            local_width = float(envelope.min_width_mm)
        profile = envelope.info.get("cross_section_candidates")
        usable_candidates = []
        if isinstance(profile, list):
            for item in profile:
                if not isinstance(item, dict):
                    continue
                try:
                    candidate_width = float(item.get("width_mm", 0))
                except (TypeError, ValueError):
                    continue
                if 0 < candidate_width <= float(max_aperture_mm):
                    usable_candidates.append(dict(item))
            usable_candidates.sort(
                key=lambda item: float(item.get("score", -1e9)), reverse=True
            )
        chosen_candidate = usable_candidates[0] if usable_candidates else None
        # A fused metric OBB is stronger evidence than the perspective-stretched
        # 2D cross-section. Apply it before deciding that the envelope is too
        # large; applying it later leaves the feature-prominence rejection in
        # control and discards the very 3D evidence intended to resolve it.
        if metric_width is not None:
            local_width = metric_width
            chosen_candidate = None
        area_ratio = envelope.area_px / max(1.0, target.area_px)
        # A color blob is merely an identifier, so its associated envelope must
        # be materially larger before replacing it.  A Grounding DINO result is
        # already a whole-object *box*; its SAM mask is intentionally smaller
        # and more precise, and must always supersede the coarse box geometry.
        semantic_whole_object_box = target.source in {
            "grounding_dino_sam2", "locate_anything", "sam3"
        }
        substantially_larger = area_ratio >= 1.35 or semantic_whole_object_box
        envelope_fits = (
            chosen_candidate is not None
            or local_width <= float(max_aperture_mm)
        )
        if substantially_larger and envelope_fits:
            if chosen_candidate is not None:
                local_width = float(chosen_candidate["width_mm"])
                candidate_uv = tuple(float(value) for value in chosen_candidate["uv"])
                grasp = replace(
                    envelope,
                    uv=candidate_uv,
                    p_cam_mm=deproject(candidate_uv[0], candidate_uv[1],
                                      float(envelope.z_mm), intr),
                    min_width_mm=local_width,
                    source="depth:ranked_cross_section",
                    info={**envelope.info, "jaw_sizing_basis":
                          "ranked local cross-section",
                          "selected_candidate": chosen_candidate},
                )
            if not math.isclose(local_width, float(envelope.min_width_mm),
                                abs_tol=0.05):
                if chosen_candidate is None:
                    grasp = replace(
                        envelope,
                        min_width_mm=local_width,
                        max_width_mm=(
                            metric_major if metric_major is not None
                            else envelope.max_width_mm
                        ),
                        source="depth:local_grasp_section",
                        info={**envelope.info, "jaw_sizing_basis":
                              (
                                  "multiview metric 3D horizontal axes"
                                  if metric_width is not None else
                                  "local cross-section at grasp centre"
                              )},
                    )
            else:
                if chosen_candidate is None:
                    grasp = envelope
            used_envelope = True
        elif substantially_larger and not envelope_fits:
            prominence = float(envelope.z_mm) - float(target.z_mm)
            if prominence < float(min_feature_prominence_mm):
                return GraspGeometry(
                    False,
                    "the detected feature belongs to a larger object that does "
                    f"not fit the {max_aperture_mm:.1f}mm jaws, and it is only "
                    f"{prominence:.1f}mm proud of the body (minimum finger "
                    f"clearance {min_feature_prominence_mm:.1f}mm); no "
                    "collision-free feature grasp can be justified",
                    None, envelope, None,
                )
            feature_clearance_note = (
                f"oversized {float(envelope.min_width_mm):.1f}mm envelope "
                f"cleared because the grasp feature is {prominence:.1f}mm "
                f"proud of it (minimum {min_feature_prominence_mm:.1f}mm)"
            )

    width = grasp.min_width_mm
    if metric_width is not None and metric_major is not None:
        grasp = replace(
            grasp,
            min_width_mm=metric_width,
            max_width_mm=metric_major,
            info={
                **grasp.info,
                "jaw_sizing_basis": "multiview metric 3D horizontal axes",
                "metric_3d_jaw_width_mm": metric_width,
                "metric_3d_major_mm": metric_major,
            },
        )
        width = metric_width
    if width is None or not math.isfinite(float(width)) or float(width) <= 0:
        return GraspGeometry(False, "no reliable metric grasp width", None,
                             envelope, None)
    if float(width) > float(max_aperture_mm):
        return GraspGeometry(
            False,
            f"best geometric grasp is {float(width):.1f}mm across, beyond the "
            f"{max_aperture_mm:.1f}mm jaw limit",
            None, envelope, None,
        )

    yaw_delta = 0.0
    yaw_clamp_note: str | None = None
    yaw_residual_deg = 0.0
    raw_grasp_width_mm = float(width)
    rpy = tuple(float(v) for v in current_rpy)
    object_axis = grasp.info.get("long_axis_deg")
    aspect_ratio = float(grasp.max_width_mm or width) / max(
        1e-6, float(width)
    )
    try:
        circularity = float(grasp.info.get("circularity", 0.0))
    except (TypeError, ValueError):
        circularity = 0.0
    rotation_invariant = aspect_ratio <= 1.6 or circularity >= 0.62
    if jaw_long_axis_camera_deg is not None and object_axis is not None:
        required = _angle_delta_deg(float(object_axis),
                                    float(jaw_long_axis_camera_deg))
        yaw_limit = max(0.0, float(max_yaw_delta_deg))
        yaw_lo, yaw_hi = (-yaw_limit, yaw_limit)
        if yaw_delta_bounds_deg is not None:
            yaw_lo = max(yaw_lo, float(yaw_delta_bounds_deg[0]))
            yaw_hi = min(yaw_hi, float(yaw_delta_bounds_deg[1]))
        if yaw_lo > yaw_hi:
            return GraspGeometry(
                False, "camera calibration leaves no safe wrist-yaw budget",
                grasp, envelope, None, candidates=envelope_candidates,
            )
        measurement_slop = max(0.0, float(yaw_measurement_slop_deg))
        residual_allowance = max(
            measurement_slop, max(0.0, float(max_residual_yaw_deg))
        )
        outside_fixed_allowance = bool(
            required < yaw_lo - residual_allowance
            or required > yaw_hi + residual_allowance
        )
        if outside_fixed_allowance and not allow_aperture_derived_residual:
            limit_text = (
                f"beyond the clamped {yaw_limit:.1f}deg limit"
                if yaw_delta_bounds_deg is None else
                f"outside the safe [{yaw_lo:.1f}, {yaw_hi:.1f}]deg budget"
            )
            allowance_text = (
                f"{measurement_slop:.1f}deg vision uncertainty"
                if float(max_residual_yaw_deg) <= measurement_slop else
                f"the bounded {residual_allowance:.1f}deg residual allowance"
            )
            return GraspGeometry(
                False,
                f"grasp needs {required:.1f}deg of wrist yaw to align the jaws, "
                f"{limit_text} plus {allowance_text}",
                grasp, envelope, None,
                candidates=envelope_candidates,
            )
        yaw_delta = min(yaw_hi, max(yaw_lo, required))
        if not math.isclose(yaw_delta, required, abs_tol=1e-9):
            yaw_residual_deg = abs(float(required) - float(yaw_delta))
            # A small residual angle makes an elongated object appear wider
            # between parallel jaws.  Account for that geometry instead of
            # treating measurement slop as free alignment error.  This keeps
            # the commanded wrist orientation inside its existing calibrated
            # bound and refuses the grasp if the resulting span cannot fit.
            major_mm = max(float(width), float(grasp.max_width_mm or width))
            residual_rad = math.radians(yaw_residual_deg)
            projected_width_mm = (
                float(width) * abs(math.cos(residual_rad))
                + major_mm * abs(math.sin(residual_rad))
            )
            if projected_width_mm > float(max_aperture_mm):
                return GraspGeometry(
                    False,
                    f"grasp yaw is {yaw_residual_deg:.1f}deg outside the safe "
                    f"wrist budget; its conservative projected jaw span is "
                    f"{projected_width_mm:.1f}mm, beyond the "
                    f"{max_aperture_mm:.1f}mm jaw limit",
                    grasp, envelope, None,
                    candidates=envelope_candidates,
                )
            grasp_info = dict(grasp.info)
            grasp_info.update({
                "raw_grasp_width_mm": raw_grasp_width_mm,
                "yaw_residual_deg": yaw_residual_deg,
                "yaw_projected_width_mm": projected_width_mm,
            })
            grasp = replace(
                grasp,
                min_width_mm=projected_width_mm,
                info=grasp_info,
            )
            width = projected_width_mm
            yaw_clamp_note = (
                f"vision requested {required:.1f}deg; commanded "
                f"{yaw_delta:.1f}deg at the physical clamp; compensated "
                f"{yaw_residual_deg:.1f}deg residual with "
                f"{projected_width_mm:.1f}mm projected jaw span"
                + (
                    " using aperture-derived residual acceptance"
                    if outside_fixed_allowance else ""
                )
            )
        rpy = (rpy[0], rpy[1], rpy[2] + math.radians(yaw_delta))
    elif not rotation_invariant and require_axis_for_elongated:
        return GraspGeometry(
            False,
            f"the grasp region is elongated (aspect ratio {aspect_ratio:.2f}) "
            "but the camera-to-jaw axis is not calibrated; refusing an "
            "orientation-blind grasp",
            grasp, envelope, None,
            candidates=envelope_candidates,
        )

    if grasp.info.get("jaw_sizing_basis"):
        basis = "full depth envelope with local body cross-section"
    else:
        basis = "full depth envelope" if used_envelope else "detected grasp feature"
    orientation = (
        (
            f"yaw correction {yaw_delta:.1f}deg"
            + (f" ({yaw_clamp_note})" if yaw_clamp_note else "")
        )
        if jaw_long_axis_camera_deg is not None
        else (
            "previously aligned wrist orientation preserved"
            if not require_axis_for_elongated else
            f"rotation-invariant grasp region (circularity {circularity:.2f}); "
            "wrist orientation unchanged"
        )
    )
    clearance = (
        f"; {feature_clearance_note}" if feature_clearance_note else ""
    )
    return GraspGeometry(
        True,
        f"{basis}, width {float(width):.1f}mm{clearance}; {orientation}; "
        f"{final_approach_mm:.0f}mm axial final approach",
        grasp, envelope, rpy,
        yaw_correction_deg=yaw_delta,
        final_approach_mm=float(final_approach_mm),
        used_full_envelope=used_envelope,
        candidates=envelope_candidates,
    )
