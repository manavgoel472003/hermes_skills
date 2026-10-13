"""Markerless RGB-D end-effector localisation from gripper motion.

Instead of relying on a coloured fingertip marker, capture a closed-gripper frame
and an open-gripper frame while the arm itself is stationary.  The jaw motion is
the only intentional local change, so RGB *and* aligned-depth differencing isolate
the fingertip region.  The centre of the two moved jaws is the grasp/TCP pixel.

This command is deliberately opt-in: ``--run`` performs only a bounded
close/open/close gripper cycle; it never moves an arm joint or Cartesian pose.
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from . import arm_safe as A
from .depth_camera import Intrinsics, open_depth_camera


@dataclass
class MotionTip:
    pixel: tuple[float, float]
    camera_xyz_mm: tuple[float, float, float]
    mask: np.ndarray
    info: dict[str, Any]


def _deproject(u: float, v: float, z_mm: float, intr: Intrinsics) -> tuple[float, float, float]:
    fx, fy, cx, cy = intr
    return ((u - cx) * z_mm / fx, (v - cy) * z_mm / fy, z_mm)


def detect_motion_tip(closed_color: np.ndarray, closed_depth: np.ndarray,
                      open_color: np.ndarray, open_depth: np.ndarray,
                      intr: Intrinsics, *, diagnostics: dict[str, Any] | None = None,
                      min_depth_mm: float = 130.0,
                      max_depth_mm: float = 1500.0) -> MotionTip | None:
    """Extract the jaw midpoint from a bounded open/close RGB-D pair.

    The depth term makes this work for a black/unpainted gripper; the RGB term
    helps at depth discontinuities.  Requiring a valid closed-frame depth avoids
    background flicker being mistaken for a jaw.
    """
    diag = diagnostics if diagnostics is not None else {}
    a = cv2.cvtColor(closed_color, cv2.COLOR_BGR2GRAY)
    b = cv2.cvtColor(open_color, cv2.COLOR_BGR2GRAY)
    rgb_change = cv2.absdiff(a, b) >= 18
    raw_depth_valid = (closed_depth > 0) & (open_depth > 0)
    # The gripper is close to the camera.  A background point cannot be the TCP,
    # and this gate prevents a far depth plane from winning a frame-difference.
    depth_valid = (raw_depth_valid & (closed_depth >= min_depth_mm)
                   & (closed_depth <= max_depth_mm) & (open_depth >= min_depth_mm)
                   & (open_depth <= max_depth_mm))
    depth_change = np.abs(closed_depth - open_depth) >= 12.0
    raw_depth_change = depth_change & raw_depth_valid
    raw_valid_count = int(raw_depth_valid.sum())
    global_depth_fraction = (float(raw_depth_change.sum()) / raw_valid_count
                             if raw_valid_count else 1.0)
    # A depth discontinuity should be local to the two jaws.  When much of the
    # image changes, it is a bad aligned RealSense frame (as opposed to motion),
    # so rely strictly on local RGB change or report no result.
    global_depth_rejected = global_depth_fraction > 0.05
    if global_depth_rejected:
        moved = (rgb_change & depth_valid).astype(np.uint8) * 255
    else:
        moved = ((rgb_change | depth_change) & depth_valid).astype(np.uint8) * 255
    diag.update({
        "rgb_changed_px": int(rgb_change.sum()),
        "depth_changed_px": int(raw_depth_change.sum()),
        "depth_changed_fraction": round(global_depth_fraction, 5),
        "depth_global_rejected": global_depth_rejected,
        "depth_range_mm": [min_depth_mm, max_depth_mm],
    })
    moved = cv2.morphologyEx(moved, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    moved = cv2.morphologyEx(moved, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

    contours, _ = cv2.findContours(moved, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [c for c in contours if 25 <= cv2.contourArea(c) <= 6000]
    if not contours:
        diag.update({"motion_px": 0, "components": 0, "failure": "no_local_near_motion"})
        return None
    # Two jaws normally form two components; keep a few in case a jaw is split by
    # a depth hole, but never admit global background motion.
    selected = sorted(contours, key=cv2.contourArea, reverse=True)[:4]
    mask = np.zeros(moved.shape, np.uint8)
    cv2.drawContours(mask, selected, -1, 255, -1)

    # Closed jaws are the nearer part of the difference field.  If the physical
    # geometry moves sideways rather than away/towards the camera, use all moved
    # pixels as a conservative fallback.
    ys, xs = np.where((mask > 0) & depth_valid & (closed_depth <= open_depth - 8.0))
    fallback = False
    if len(xs) < 30:
        ys, xs = np.where((mask > 0) & (closed_depth > 0))
        fallback = True
    if len(xs) < 30:
        diag.update({"motion_px": int((mask > 0).sum()), "components": len(selected),
                     "failure": "insufficient_near_motion"})
        return None

    # The grasp point is the midpoint of moved jaw pixels, not the gripper body.
    u, v = float(np.mean(xs)), float(np.mean(ys))
    z_mm = float(np.median(closed_depth[ys, xs]))
    info = {
        **diag,
        "motion_px": int((mask > 0).sum()),
        "components": len(selected),
        "fallback_all_motion": fallback,
        "depth_mm": z_mm,
    }
    return MotionTip((u, v), _deproject(u, v, z_mm, intr), mask, info)


def _annotate(color: np.ndarray, depth: np.ndarray, result: MotionTip | None,
              note: str | None = None) -> np.ndarray:
    norm = np.clip((depth - 200.0) / 700.0, 0.0, 1.0)
    norm[depth <= 0] = 0
    depth_view = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    depth_view[depth <= 0] = (0, 0, 0)
    left = color.copy()
    if result is None:
        label = note or "motion tip: not found"
        cv2.putText(left, label[:70], (12, 28), cv2.FONT_HERSHEY_SIMPLEX, .55, (0, 0, 255), 2)
        return np.hstack([left, depth_view])
    for image in (left, depth_view):
        image[result.mask > 0] = (0, 165, 255)
        u, v = result.pixel
        cv2.drawMarker(image, (round(u), round(v)), (0, 255, 0), cv2.MARKER_CROSS, 42, 3)
    x, y, z = result.camera_xyz_mm
    cv2.putText(left, f"markerless tip: {x:.0f}, {y:.0f}, {z:.0f} mm", (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX, .58, (0, 255, 0), 2)
    cv2.putText(depth_view, "closed-frame aligned depth", (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX, .58, (255, 255, 255), 2)
    return np.hstack([left, depth_view])


def _stable_frame(camera: Any, count: int = 5) -> tuple[np.ndarray, np.ndarray, Intrinsics]:
    """Median-combine consecutive frames to reject one-frame RGB-D glitches."""
    colors: list[np.ndarray] = []
    depths: list[np.ndarray] = []
    intr: Intrinsics | None = None
    for _ in range(count):
        color, depth, intr = camera.frame()
        colors.append(color)
        depths.append(depth)
    assert intr is not None
    color = np.median(np.stack(colors), axis=0).astype(np.uint8)
    stack = np.stack(depths).astype(np.float32)
    # Zero means invalid.  Do not allow it to drag an otherwise-valid median down.
    stack[stack <= 0] = np.nan
    depth = np.nanmedian(stack, axis=0)
    depth[~np.isfinite(depth)] = 0.0
    return color, depth.astype(np.float32), intr


def run(camera_kind: str = "realsense", settle_s: float = 0.8) -> dict[str, Any]:
    """Run the user-authorized gripper-only markerless calibration."""
    preflight = A.preflight(connect=True)
    if not preflight.ok:
        return {"ok": False, "reason": preflight.reason}
    camera = open_depth_camera(camera_kind)
    try:
        if not A.gripper("close"):
            return {"ok": False, "reason": "could_not_close_gripper"}
        time.sleep(settle_s)
        closed_color, closed_depth, intr = _stable_frame(camera)
        if not A.gripper("open"):
            return {"ok": False, "reason": "could_not_open_gripper"}
        time.sleep(settle_s)
        open_color, open_depth, _ = _stable_frame(camera)
        # Restore a closed, compact gripper after calibration.
        A.gripper("close")
        time.sleep(settle_s)
        diagnostics: dict[str, Any] = {}
        result = detect_motion_tip(closed_color, closed_depth, open_color, open_depth, intr,
                                   diagnostics=diagnostics)
        output = Path("outputs/motion_tip_latest.jpg").resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        closed_output = output.with_name("motion_tip_closed.jpg")
        open_output = output.with_name("motion_tip_open.jpg")
        mask_output = output.with_name("motion_tip_mask.jpg")
        cv2.imwrite(str(closed_output), closed_color)
        cv2.imwrite(str(open_output), open_color)
        cv2.imwrite(str(mask_output), result.mask if result is not None
                    else np.zeros(closed_depth.shape, dtype=np.uint8))
        cv2.imwrite(str(output), _annotate(closed_color, closed_depth, result,
                                           note=diagnostics.get("failure")))
        images = {"overlay": str(output), "closed": str(closed_output),
                  "open": str(open_output), "mask": str(mask_output)}
        if result is None:
            return {"ok": False, "reason": diagnostics.get("failure", "no_local_jaw_motion"),
                    "detail": diagnostics, "images": images}
        return {"ok": True, "tip_pixel": list(result.pixel),
                "tip_camera_xyz_mm": list(result.camera_xyz_mm),
                "detail": result.info, "images": images}
    finally:
        camera.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Markerless RGB-D gripper-tip calibration")
    parser.add_argument("--camera", choices=("realsense", "orbbec", "auto"), default="realsense")
    parser.add_argument("--settle", type=float, default=0.8)
    parser.add_argument("--run", action="store_true", help="authorize the bounded close/open/close cycle")
    args = parser.parse_args()
    if not args.run:
        parser.error("this calibration changes only the gripper; pass --run to authorize it")
    print(json.dumps(run(args.camera, args.settle), indent=2))


if __name__ == "__main__":
    main()
