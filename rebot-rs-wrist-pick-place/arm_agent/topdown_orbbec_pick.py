"""Experimental top-down Orbbec RGB-D grasp guidance.

This is deliberately separate from :mod:`wrist_servo`.  It uses the overhead
Orbbec's *aligned* depth and a local gripper segmentation model to observe the
actual gripper and a requested object in the same camera frame.  It does not
touch the gripper while registering the camera and refuses live Cartesian
motion until the camera-to-arm transform was measured and validated.

The first live capability is an approach to a hover point, not an automatic
grasp.  A palm-and-jaws segmentation does not yet identify the empty jaw slot
accurately enough to close safely; that needs a separately labelled jaw-slot
model or a one-time visual slot calibration.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

# The committed branch can be run directly from the hardware workspace.  That
# workspace owns the private RS controller package, while this repository keeps
# the experimental module itself.  Normal ``python -m`` use remains relative.
if __package__:
    from .depth_camera import Intrinsics, open_depth_camera
    from . import wrist_detect as WD
else:  # pragma: no cover - used only by the hardware launcher
    sys.path.insert(0, "/home/manav/rebot_lerobot")
    from arm_agent.depth_camera import Intrinsics, open_depth_camera
    from arm_agent import wrist_detect as WD


ROOT = Path("/home/manav/rebot_lerobot")
CALIBRATION = Path(os.getenv(
    "REBOT_TOPDOWN_ORBBEC_CALIB",
    str(ROOT / "calib" / "topdown_orbbec_gripper_to_base.json"),
))
JAW_SLOT_CALIBRATION = Path(os.getenv(
    "REBOT_TOPDOWN_JAW_SLOT_CALIB",
    str(ROOT / "calib" / "topdown_orbbec_jaw_slot.json"),
))
DEBUG = Path(os.getenv("REBOT_TOPDOWN_DEBUG_DIR", "/tmp/rebot-topdown-orbbec"))
MODEL = Path(os.getenv(
    "REBOT_TOPDOWN_GRIPPER_MODEL",
    str(ROOT / "runs/gripper_segmentation/topdown_yolo11n_orbbec_finetune/weights/best.pt"),
))


def _arm_backend():
    """Import hardware code only for an explicit motion command.

    This keeps ``--watch`` and the pure registration-math tests usable on a
    machine which has the camera/model environment but not the RS CAN package.
    """
    if __package__:
        from .local_arm import get_local_arm
    else:  # pragma: no cover - hardware launcher route
        from arm_agent.local_arm import get_local_arm
    return get_local_arm()


def _truth(name: str, default: bool = False) -> bool:
    return os.getenv(name, str(int(default))).strip().lower() in {"1", "true", "yes", "on"}


def _f(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except ValueError:
        return default


def _deproject(u: float, v: float, z_mm: float, intr: Intrinsics) -> np.ndarray:
    fx, fy, cx, cy = intr
    return np.array([(u - cx) * z_mm / fx, (v - cy) * z_mm / fy, z_mm], dtype=float)


def _rigid_fit(camera_mm: np.ndarray, base_m: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Fit base metres = R @ (camera millimetres / 1000) + t."""
    if camera_mm.ndim != 2 or camera_mm.shape != base_m.shape or camera_mm.shape[0] < 4:
        raise ValueError("need at least four corresponding 3-D camera and arm points")
    source = camera_mm / 1000.0
    source_c, target_c = source.mean(axis=0), base_m.mean(axis=0)
    h = (source - source_c).T @ (base_m - target_c)
    u, _, vt = np.linalg.svd(h)
    r = vt.T @ u.T
    if np.linalg.det(r) < 0:
        vt[-1, :] *= -1
        r = vt.T @ u.T
    t = target_c - r @ source_c
    residual_mm = float(np.sqrt(np.mean(np.sum(((source @ r.T + t) - base_m) ** 2, axis=1))) * 1000.0)
    return r, t, residual_mm


def _load_transform() -> tuple[np.ndarray, np.ndarray, dict[str, Any]] | None:
    try:
        data = json.loads(CALIBRATION.read_text())
        r = np.asarray(data["rotation_camera_to_base"], dtype=float)
        t = np.asarray(data["translation_base_m"], dtype=float)
        if r.shape != (3, 3) or t.shape != (3,) or not np.all(np.isfinite(r)) or not np.all(np.isfinite(t)):
            raise ValueError("invalid matrix shape")
        if not np.isclose(np.linalg.det(r), 1.0, atol=0.03):
            raise ValueError("rotation is not proper")
        return r, t, data
    except Exception:
        return None


def _base_point(p_camera_mm: np.ndarray, transform: tuple[np.ndarray, np.ndarray, dict[str, Any]]) -> np.ndarray:
    r, t, _ = transform
    return r @ (np.asarray(p_camera_mm, dtype=float) / 1000.0) + t


@dataclass
class Observation:
    color: np.ndarray
    depth_mm: np.ndarray
    intr: Intrinsics
    gripper_mask: np.ndarray | None
    gripper_camera_mm: np.ndarray | None
    target: WD.Detection | None
    target_why: dict[str, Any]


class TopDownEye:
    def __init__(self) -> None:
        if not MODEL.is_file():
            raise RuntimeError(f"top-down gripper model is missing: {MODEL}")
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError("ultralytics is not installed in rebot_rs") from exc
        self.camera = open_depth_camera("orbbec")
        self.model = YOLO(str(MODEL))

    def close(self) -> None:
        self.camera.close()

    def _gripper(self, color: np.ndarray, depth: np.ndarray, intr: Intrinsics) -> tuple[np.ndarray | None, np.ndarray | None]:
        # The live-Orbbec fine-tune is deliberately accepted at 0.20: on its
        # held-object validation image the real gripper is a 0.224 detection.
        # This remains a visual calibration aid, never a permission to close.
        result = self.model.predict(color, verbose=False, conf=_f("REBOT_TOPDOWN_GRIPPER_CONF", 0.20), imgsz=640)[0]
        if result.masks is None or result.boxes is None:
            return None, None
        masks = result.masks.data.detach().cpu().numpy()
        scores = result.boxes.conf.detach().cpu().numpy()
        order = np.argsort(scores)[::-1]
        near, far = _f("REBOT_TOPDOWN_GRIPPER_NEAR_MM", 120), _f("REBOT_TOPDOWN_GRIPPER_FAR_MM", 1400)
        for index in order:
            mask = cv2.resize(masks[index].astype(np.uint8), (depth.shape[1], depth.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
            valid = mask & np.isfinite(depth) & (depth >= near) & (depth <= far)
            if int(valid.sum()) < 100:
                continue
            rows, cols = np.nonzero(valid)
            z = float(np.median(depth[valid]))
            return valid, _deproject(float(np.median(cols)), float(np.median(rows)), z, intr)
        return None, None

    def observe(self, description: str) -> Observation:
        color, depth, intr = self.camera.frame()
        mask, gripper = self._gripper(color, depth, intr)
        target, why = WD.detect(
            color, depth, intr, description,
            near_mm=_f("REBOT_TOPDOWN_NEAR_MM", 120),
            far_mm=_f("REBOT_TOPDOWN_FAR_MM", 1400),
        )
        return Observation(color, depth, intr, mask, gripper, target, why)


def _draw(observation: Observation, description: str, path: Path) -> None:
    frame = observation.color.copy()
    if observation.gripper_mask is not None:
        frame[observation.gripper_mask] = (0.35 * frame[observation.gripper_mask] + np.array([255, 255, 0]) * 0.65).astype(np.uint8)
    if observation.gripper_camera_mm is not None:
        p = observation.gripper_camera_mm
        cv2.putText(frame, f"gripper {p[2]:.0f}mm", (12, 26), cv2.FONT_HERSHEY_SIMPLEX, .55, (255,255,0), 2)
    if observation.target is not None:
        x1,y1,x2,y2 = observation.target.box
        target_mask = observation.target.info.get("mask_u8")
        if target_mask is not None:
            target_mask = np.asarray(target_mask) > 0
            if target_mask.shape == frame.shape[:2]:
                # A translucent fill makes a thin marker/cap visible at a
                # glance; a bare rectangle was too easy to miss in the wide
                # top-view image.
                overlay = frame[target_mask].astype(np.float32)
                frame[target_mask] = (0.28 * overlay + 0.72 * np.array([0, 255, 0])).astype(np.uint8)
        cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
        u, v = (int(round(observation.target.uv[0])), int(round(observation.target.uv[1])))
        cv2.drawMarker(frame, (u, v), (0, 0, 255), cv2.MARKER_CROSS, 18, 2)
        cv2.putText(frame, f"TARGET: {description} [{observation.target.source}]", (max(5, x1), max(20,y1-8)), cv2.FONT_HERSHEY_SIMPLEX, .53, (0,255,0), 2)
    else:
        cv2.putText(frame, f"TARGET NOT DETECTED: {description}", (12, frame.shape[0] - 14), cv2.FONT_HERSHEY_SIMPLEX, .60, (0,0,255), 2)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), frame)


def _watch(description: str) -> dict[str, Any]:
    eye = TopDownEye()
    try:
        count = 0
        print("[topdown] watching Orbbec RGB-D; Ctrl+C exits. Overlay: /tmp/rebot-topdown-orbbec/latest.jpg", flush=True)
        while True:
            obs = eye.observe(description)
            _draw(obs, description, DEBUG / "latest.jpg")
            count += 1
            target_detail = (
                "no" if obs.target is None else
                f"yes source={obs.target.source} uv=({obs.target.uv[0]:.0f},{obs.target.uv[1]:.0f}) depth_mm={obs.target.z_mm:.0f}"
            )
            print(f"[topdown] frame={count} gripper={'yes' if obs.gripper_camera_mm is not None else 'no'} target={target_detail}", flush=True)
            time.sleep(.15)
    except KeyboardInterrupt:
        return {"ok": True, "frames": count, "overlay": str(DEBUG / "latest.jpg")}
    finally:
        eye.close()


def _calibrate_jaw_slot(description: str) -> dict[str, Any]:
    """Record a visual gripper-reference -> open-jaw-slot offset without motion."""
    eye = TopDownEye()
    try:
        samples: list[np.ndarray] = []
        count = max(3, min(8, int(_f("REBOT_TOPDOWN_SLOT_FRAMES", 5))))
        print(
            "[topdown] jaw-slot calibration is camera-only: arm/gripper are not "
            "connected, read, enabled, or commanded",
            flush=True,
        )
        for index in range(count):
            obs = eye.observe(description)
            _draw(obs, description, DEBUG / "latest.jpg")
            if obs.gripper_camera_mm is None or obs.target is None:
                stamp = time.strftime("%Y%m%d-%H%M%S")
                image = DEBUG / f"jaw_slot_rejected_{stamp}.jpg"
                detail = DEBUG / f"jaw_slot_rejected_{stamp}.json"
                _draw(obs, description, image)
                detail.write_text(json.dumps({
                    "description": description,
                    "sample": index + 1,
                    "gripper_detected": obs.gripper_camera_mm is not None,
                    "target_detected": obs.target is not None,
                    "target_why": obs.target_why,
                    "image": str(image),
                }, indent=2) + "\n")
                return {
                    "ok": False,
                    "reason": "the top camera must see both the segmented gripper and the object held centrally between its open jaws",
                    "sample": index + 1,
                    "diagnostic_image": str(image),
                    "diagnostic_json": str(detail),
                    "gripper_commanded": False,
                    "arm_connected": False,
                }
            # This is the vector the later planner subtracts from the detected
            # object to put the *jaw slot*, rather than the palm centroid, over it.
            samples.append(np.asarray(obs.target.p_cam_mm, dtype=float) - obs.gripper_camera_mm)
            print(f"[topdown] jaw-slot sample={index + 1}/{count} captured", flush=True)
            time.sleep(.15)
        offsets = np.asarray(samples)
        median = np.median(offsets, axis=0)
        spread = float(np.max(np.linalg.norm(offsets - median, axis=1)))
        magnitude = float(np.linalg.norm(median))
        max_spread = _f("REBOT_TOPDOWN_SLOT_MAX_SPREAD_MM", 6.0)
        if spread > max_spread:
            return {"ok": False, "reason": f"jaw-slot observations were not stationary (spread {spread:.1f}mm > {max_spread:.1f}mm); hold still and retry", "gripper_commanded": False, "arm_connected": False}
        if not 8.0 <= magnitude <= 180.0:
            return {"ok": False, "reason": f"jaw-slot offset {magnitude:.1f}mm is implausible; ensure the object is between the jaws, not elsewhere in the view", "gripper_commanded": False, "arm_connected": False}
        JAW_SLOT_CALIBRATION.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": 1,
            "camera": "orbbec_topdown",
            "description": description,
            "gripper_reference_to_open_jaw_slot_camera_mm": median.tolist(),
            "spread_mm": spread,
            "samples": count,
            "model": str(MODEL),
            "arm_connected": False,
            "gripper_commanded": False,
        }
        JAW_SLOT_CALIBRATION.write_text(json.dumps(payload, indent=2) + "\n")
        return {"ok": True, "saved": str(JAW_SLOT_CALIBRATION), "jaw_slot_offset_camera_mm": np.round(median, 2).tolist(), "spread_mm": round(spread, 2), "arm_connected": False, "gripper_commanded": False}
    finally:
        eye.close()


def _calibrate(run: bool) -> dict[str, Any]:
    if not run:
        return {"ok": False, "reason": "calibration moves the arm; rerun with --run. The gripper is never read, opened, or closed."}
    arm = _arm_backend()
    moved_from_home = False
    eye = TopDownEye()
    try:
        arm.connect()
        # Registration must not be based on an arbitrary manually parked pose:
        # the local IK solver deliberately has no distant/random branch search.
        # First use the established home configuration, then sample only the
        # offline-verified forward/lateral/upward volume reachable from it.
        homed = arm.home()
        if not homed.get("ok"):
            raise RuntimeError(f"could not establish the safe home calibration start: {homed.get('reason')}")
        home = arm.state()["end_effector"]
        xyz = np.asarray(home["position_m"], dtype=float)
        rpy = np.asarray(home["rpy_rad"], dtype=float)
        step = min(.025, max(.015, _f("REBOT_TOPDOWN_CALIB_STEP_M", .025)))
        # Do not include negative X: behind home is outside this arm's local IK
        # branch.  The home sample plus elevated X/Y samples are non-coplanar.
        offsets = np.array([
            [0.0, 0.0, 0.0],
            [0.0, 0.0, step],
            [step, 0.0, step],
            [0.0, step, step],
            [0.0, -step, step],
            [0.8 * step, 0.8 * step, 1.35 * step],
        ], dtype=float)
        sample_poses = [tuple(np.r_[xyz + d, rpy]) for d in offsets[1:]]
        # The RS safety layer caps each solved joint transition at 0.250 rad.
        # Keep that invariant: interpolate the *calibration-only* Cartesian
        # route instead of weakening the global cap for a camera measurement.
        max_segment_m = _f("REBOT_TOPDOWN_CALIB_SEGMENT_M", .008)
        dense_legs: list[list[tuple[float, float, float, float, float, float]]] = []
        previous_xyz = xyz.copy()
        for pose in sample_poses:
            target_xyz = np.asarray(pose[:3], dtype=float)
            count = max(1, int(np.ceil(np.linalg.norm(target_xyz - previous_xyz) /
                                       max(.003, max_segment_m))))
            leg = [tuple(np.r_[previous_xyz + (target_xyz - previous_xyz) * (j / count), rpy])
                   for j in range(1, count + 1)]
            dense_legs.append(leg)
            previous_xyz = target_xyz
        dense_poses = [pose for leg in dense_legs for pose in leg]
        # Validate the exact complete calibration chain at the confirmed home
        # pose before the first sample motion.
        arm.solve_ik_path(dense_poses)
        camera_points, base_points = [], []
        for index, offset in enumerate(offsets, 1):
            if index > 1:
                leg = dense_legs[index - 2]
                arm.move_pose_path(leg, duration=max(.8, .30 * len(leg)))
                moved_from_home = True
            time.sleep(.3)
            obs = eye.observe("object")
            if obs.gripper_camera_mm is None:
                raise RuntimeError(f"gripper segmentation unavailable at sample {index}; nothing else will move")
            measured = np.asarray(arm.state()["end_effector"]["position_m"], dtype=float)
            camera_points.append(obs.gripper_camera_mm)
            base_points.append(measured)
            print(f"[topdown] registration sample={index}/{len(offsets)} captured", flush=True)
        r, t, rmse = _rigid_fit(np.asarray(camera_points), np.asarray(base_points))
        limit = _f("REBOT_TOPDOWN_CALIB_MAX_RMSE_MM", 12.0)
        if rmse > limit:
            raise RuntimeError(f"registration residual {rmse:.1f}mm exceeds {limit:.1f}mm; calibration not saved")
        CALIBRATION.parent.mkdir(parents=True, exist_ok=True)
        payload = {"version": 1, "camera": "orbbec_topdown", "model": str(MODEL), "rotation_camera_to_base": r.tolist(), "translation_base_m": t.tolist(), "rmse_mm": rmse, "samples": len(camera_points), "gripper_commanded": False}
        CALIBRATION.write_text(json.dumps(payload, indent=2) + "\n")
        return {"ok": True, "saved": str(CALIBRATION), "rmse_mm": round(rmse, 2), "gripper_commanded": False}
    finally:
        # Return to the known safe home pose; no gripper operation or disable.
        try:
            if arm.connected and moved_from_home:
                frozen = arm.freeze()
                if not frozen.get("ok"):
                    raise RuntimeError(f"could not verify a stationary hold: {frozen.get('reason')}")
                parked = arm.home()
                if not parked.get("ok"):
                    raise RuntimeError(parked.get("reason", "home not verified"))
        except Exception as exc:
            print(f"[topdown] WARNING: could not safely return calibration to home: {exc}", flush=True)
        eye.close()


def _plan(description: str, run: bool) -> dict[str, Any]:
    transform = _load_transform()
    if transform is None:
        return {"ok": False, "reason": f"no validated top-down registration at {CALIBRATION}; run --calibrate-camera --run first"}
    eye = TopDownEye()
    try:
        obs = eye.observe(description)
        _draw(obs, description, DEBUG / "latest.jpg")
        if obs.gripper_camera_mm is None or obs.target is None:
            return {"ok": False, "reason": "top-down view must see both the gripper and the requested object", "detection": obs.target_why}
        target_base = _base_point(np.asarray(obs.target.p_cam_mm), transform)
        gripper_base = _base_point(obs.gripper_camera_mm, transform)
        hover = target_base.copy(); hover[2] += _f("REBOT_TOPDOWN_HOVER_M", .075)
        arm = _arm_backend(); arm.connect()
        state = arm.state()["end_effector"]
        current_xyz = np.asarray(state["position_m"], dtype=float)
        rpy = tuple(float(v) for v in state["rpy_rad"])
        limits_lo = np.array([_f("REBOT_TOPDOWN_WS_X_MIN", .150),
                              _f("REBOT_TOPDOWN_WS_Y_MIN", -.300),
                              _f("REBOT_TOPDOWN_WS_Z_MIN", .000)])
        limits_hi = np.array([_f("REBOT_TOPDOWN_WS_X_MAX", .540),
                              _f("REBOT_TOPDOWN_WS_Y_MAX", .300),
                              _f("REBOT_TOPDOWN_WS_Z_MAX", .360)])
        if np.any(hover < limits_lo) or np.any(hover > limits_hi):
            return {"ok": False, "reason": "registered hover point is outside the configured top-down workspace", "hover_base_m": np.round(hover, 4).tolist(), "workspace": {"min": limits_lo.tolist(), "max": limits_hi.tolist()}}
        # This maps the observed gripper reference to a hover over the object.
        # Move via a high traverse rather than drawing one long diagonal through
        # the table plane.  This is still not scene collision avoidance; it is
        # a bounded, obstacle-agnostic approach test and deliberately stops
        # before the object's height and before every gripper action.
        travel_z = min(limits_hi[2] - .01, max(current_xyz[2], hover[2]) + _f("REBOT_TOPDOWN_TRAVEL_CLEARANCE_M", .060))
        if travel_z < max(current_xyz[2], hover[2]):
            return {"ok": False, "reason": "no vertical clearance remains within the configured top-down workspace"}
        key_xyz = [
            np.array([current_xyz[0], current_xyz[1], travel_z]),
            np.array([hover[0], hover[1], travel_z]),
            hover,
        ]
        max_segment_m = max(.003, _f("REBOT_TOPDOWN_APPROACH_SEGMENT_M", .008))
        path: list[tuple[float, float, float, float, float, float]] = []
        previous = current_xyz.copy()
        for endpoint in key_xyz:
            count = max(1, int(np.ceil(np.linalg.norm(endpoint - previous) / max_segment_m)))
            path.extend(tuple(np.r_[previous + (endpoint - previous) * (j / count), rpy])
                        for j in range(1, count + 1))
            previous = endpoint
        arm.solve_ik_path(path)
        out = {"ok": True, "reason": "top-down RGB-D hover plan validated; high-clearance segmented route, no gripper action", "gripper_camera_mm": np.round(obs.gripper_camera_mm,1).tolist(), "object_camera_mm": [round(float(v),1) for v in obs.target.p_cam_mm], "observed_gripper_base_m": np.round(gripper_base,4).tolist(), "hover_base_m": np.round(hover,4).tolist(), "travel_z_m": round(float(travel_z), 4), "waypoints": len(path), "overlay": str(DEBUG / "latest.jpg"), "gripper_commanded": False}
        if run:
            arm.move_pose_path(path, duration=max(2.0, .10 * len(path)))
            out["executed"] = True
        return out
    finally:
        eye.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--watch", metavar="OBJECT")
    mode.add_argument("--calibrate-camera", action="store_true")
    mode.add_argument("--calibrate-jaw-slot", metavar="OBJECT")
    mode.add_argument("--approach", metavar="OBJECT")
    parser.add_argument("--run", action="store_true", help="allow the explicit calibration or hover motion")
    args = parser.parse_args()
    try:
        if args.watch:
            result = _watch(args.watch)
        elif args.calibrate_camera:
            result = _calibrate(args.run)
        elif args.calibrate_jaw_slot:
            result = _calibrate_jaw_slot(args.calibrate_jaw_slot)
        else:
            result = _plan(args.approach, args.run)
    except Exception as exc:  # Camera/model preflight is intentionally before CAN.
        result = {
            "ok": False,
            "reason": f"top-down Orbbec preflight failed: {type(exc).__name__}: {exc}",
            "arm_connected": False,
            "gripper_commanded": False,
        }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
