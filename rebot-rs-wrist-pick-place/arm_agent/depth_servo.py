"""Closed-loop visual-servo pick & place with an overhead RGB-D camera.

Fresh, calibration-free, and arm-agnostic. Works for whichever arm the rebot arm
service is configured for -- the RobStride "RS" arm or the Damiao "DM" arm -- because
EVERY motion goes through the arm service over HTTP (via ``arm_safe``): the service
owns the URDF / IK / gripper for the active arm, so nothing here is tied to one robot.

Runs wholly in the ``rebot_rs`` conda environment: RealSense, LocateAnything, Pinocchio
IK and the RS CAN controller are all local. RealSense is the default camera; the older
Orbbec bridge remains an explicit compatibility fallback for its incompatible SDK.

(This is the DEPTH-camera, modular, MCP-wired sibling of the older single-camera
``visual_servo.py``, which pixel-servos off the Reachy RGB head and only picks.)

The idea (no hand-eye calibration, ever)
----------------------------------------
The SAME depth camera sees both the object and the gripper tip each iteration. We
drive their 3-D camera-frame positions together using an online-learned 3x3 Jacobian
(jog each base axis a little, watch how the tip moves in the camera), then descend by
feel (torque-gated) and grasp. Only the *relative* tip<->object error matters, so no
camera->base transform is ever needed. The base-vertical direction even falls out of
the Jacobian's z-column, so we hover a fixed height above the object and drop straight
down without knowing the camera's pose.

Two independent, composable skills (a small ENPIRE-style validated skill library;
the self-improvement / policy-evolution loop is intentionally NOT built here). Each
skill is precondition -> arm_safe-gated action -> success check -> safe reset, and
returns a structured ``{ok, reason, steps, verified}`` the planner can chain on.
  * ``servo_pick(object)``            -> leaves the object held
  * ``servo_place(target=|x=,y=)``    -> releases the held object

Visuals
-------
Pass ``--view`` to save an annotated frame per step (object box + depth, tip-marker
crosshair + depth, target, error, and a depth-colormap inset) to REBOT_SERVO_DEBUG,
and ``--mjpeg PORT`` to also stream them live to a browser (the robot host is usually
headless). ``--snapshot`` grabs one annotated live frame (great for aiming the
camera), and ``--demo`` renders the annotation on a synthetic scene with NO hardware.

CLI (``conda activate rebot_rs`` first)::

    python -m arm_agent.depth_servo --demo                       # no hardware
    python -m arm_agent.depth_servo --snapshot --object "red block"
    python -m arm_agent.depth_servo --pick  "red block" --dry --view
    python -m arm_agent.depth_servo --pick  "red block" --run --view --mjpeg 8730
    python -m arm_agent.depth_servo --place --target "the bowl" --run --view
    python -m arm_agent.depth_servo --place --x 0.30 --y -0.10  --run

Tip detection tracks a coloured marker on the gripper fingertips (default: orange).
If the active arm's tip is a different colour, set
``REBOT_TIP_HSV="hlo,slo,vlo,hhi,shi,vhi"`` (OpenCV HSV, H in 0-179). Nothing else is
arm-specific: the scan pose and workspace are read live from ``arm_safe`` and the
gripper / IK come from the service.

Env:
  REBOT_DEPTH_CAMERA realsense (default), orbbec, or auto
  REBOT_ORBBEC_URL   compatibility bridge URL when REBOT_DEPTH_CAMERA=orbbec
  REBOT_TIP_HSV      6 comma-separated ints for the tip marker HSV band
  REBOT_SERVO_SCAN   "x,y,z" scan-pose override (else derived from arm_safe bounds)
  REBOT_SERVO_DEBUG  where annotated frames are written (default: session scratchpad)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
import webbrowser
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from arm_agent import arm_safe as A  # shared, IK-verified, arm-agnostic motion gate
from arm_agent.depth_camera import DepthCamera, open_depth_camera
from arm_agent.local_vision import locate as locate_local

# ---- tunables (env-overridable; defaults suit a table-top overhead Orbbec) --------
CAMERA_KIND = os.getenv("REBOT_DEPTH_CAMERA", "realsense")
JAC_STEP = 0.02          # base jog per axis when learning the Jacobian (m)
APPROACH_M = 0.06        # hover this far above the object before dropping (m)
PLACE_CLEAR_M = 0.05     # hover this far above the place target before lowering (m)
ALIGN_TOL_MM = 12.0      # 3-D camera-frame tip<->target error we call "aligned"
GAIN = 0.5               # servo step = GAIN * Jinv @ error (then clamped by arm_safe)
MAX_ITERS = 24           # servo iterations before giving up
SETTLE_S = 0.35          # let the arm settle before re-observing
GRASP_GAP_RAD = 0.09     # gripper stopped >this short of full close => holding an object
GRIPPER_NEAR_RAD = 0.12  # gripper within this of the open target => actually opened
DEPTH_WIN = 6            # median-depth window half-size (px)
DEPTH_VIS_MM = (300.0, 1000.0)  # depth-colormap range for the inset

DEBUG_DIR = Path(os.getenv(
    "REBOT_SERVO_DEBUG",
    "/tmp/claude-1000/-home-manav-rebot-lerobot/"
    "0ccac054-06ed-4d4c-ae38-775608f311a9/scratchpad/depth_servo"))

# Default gripper-tip marker: tight orange (skips red plates <=6 and yellow >=28).
_DEFAULT_TIP_HSV = (8, 120, 110, 24, 255, 255)


def _env_tip_hsv() -> tuple[np.ndarray, np.ndarray]:
    raw = os.getenv("REBOT_TIP_HSV")
    vals: tuple[int, ...] = _DEFAULT_TIP_HSV
    if raw:
        try:
            parsed = tuple(int(v) for v in raw.split(","))
            if len(parsed) == 6:
                vals = parsed
        except ValueError:
            pass
    return (np.array(vals[:3], np.uint8), np.array(vals[3:], np.uint8))


# =====================================================================================
# Geometry
# =====================================================================================
def _deproject(u: float, v: float, Z_mm: float, intr) -> np.ndarray:
    """Pinhole back-projection: pixel (u,v) + depth Z (mm) -> camera-frame XYZ (mm)."""
    fx, fy, cx, cy = intr
    return np.array([(u - cx) * Z_mm / fx, (v - cy) * Z_mm / fy, Z_mm], float)


def _project(p_cam_mm, intr):
    """Camera-frame XYZ (mm) -> pixel (u, v). Inverse of _deproject."""
    fx, fy, cx, cy = intr
    X, Y, Z = p_cam_mm
    if Z <= 1e-6:
        return None
    return (fx * X / Z + cx, fy * Y / Z + cy)


def _median_depth(depth: np.ndarray, u: float, v: float, r: int = DEPTH_WIN) -> float:
    """Robust depth (mm) around a pixel; 0.0 if too few valid samples."""
    y, x = int(round(v)), int(round(u))
    win = depth[max(0, y - r): y + r, max(0, x - r): x + r]
    win = win[win > 0]
    return float(np.median(win)) if win.size >= 8 else 0.0


# =====================================================================================
# Visualization (annotated frames + optional browser MJPEG stream / cv2 window)
# =====================================================================================
def _depth_inset(depth: np.ndarray, w: int = 200) -> np.ndarray:
    lo, hi = DEPTH_VIS_MM
    d = np.clip((depth - lo) / max(1.0, hi - lo), 0, 1)
    d[depth <= 0] = 0
    g = cv2.applyColorMap((d * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    g[depth <= 0] = (0, 0, 0)
    h = int(w * depth.shape[0] / depth.shape[1])
    return cv2.resize(g, (w, h))


def annotate(color, depth=None, intr=None, *, tip_uv=None, obj_box=None, obj_uv=None,
             target_cam=None, err_mm=None, note=None, title=None):
    """Draw everything the servo 'sees' onto a copy of the color frame."""
    vis = color.copy()
    lo, hi = _env_tip_hsv()
    # highlight the tip marker mask
    hsv = cv2.cvtColor(color, cv2.COLOR_BGR2HSV)
    m = cv2.morphologyEx(cv2.inRange(hsv, lo, hi), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    vis[m > 0] = (0, 165, 255)

    def _dtxt(uv):
        return "" if depth is None else f" {_median_depth(depth, uv[0], uv[1]):.0f}mm"

    if obj_box is not None:
        b = [int(v) for v in obj_box]
        cv2.rectangle(vis, (b[0], b[1]), (b[2], b[3]), (80, 220, 80), 2)
    if obj_uv is not None:
        p = (int(obj_uv[0]), int(obj_uv[1]))
        cv2.circle(vis, p, 7, (80, 220, 80), -1)
        cv2.putText(vis, "object" + _dtxt(obj_uv), (p[0] + 10, p[1]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (80, 220, 80), 2)
    tgt_uv = _project(target_cam, intr) if (target_cam is not None and intr is not None) else None
    if tgt_uv is not None:
        t = (int(tgt_uv[0]), int(tgt_uv[1]))
        cv2.drawMarker(vis, t, (255, 0, 255), cv2.MARKER_TILTED_CROSS, 26, 2)
        cv2.putText(vis, "target", (t[0] + 10, t[1] + 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
    if tip_uv is not None:
        p = (int(tip_uv[0]), int(tip_uv[1]))
        cv2.drawMarker(vis, p, (0, 0, 255), cv2.MARKER_CROSS, 34, 3)
        cv2.putText(vis, "tip" + _dtxt(tip_uv), (p[0] + 10, p[1] + 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)
        if tgt_uv is not None:
            cv2.arrowedLine(vis, p, (int(tgt_uv[0]), int(tgt_uv[1])), (0, 255, 255), 2, tipLength=0.2)

    banner = title or "depth_servo"
    if err_mm is not None:
        banner += f"   err={err_mm:.0f}mm"
    if note:
        banner += f"   {note}"
    cv2.rectangle(vis, (0, 0), (vis.shape[1], 26), (0, 0, 0), -1)
    cv2.putText(vis, banner, (8, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    if depth is not None:
        inset = _depth_inset(depth)
        ih, iw = inset.shape[:2]
        vis[vis.shape[0] - ih:, vis.shape[1] - iw:] = inset
        cv2.putText(vis, "depth", (vis.shape[1] - iw + 6, vis.shape[0] - ih + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return vis


class _MjpegStream:
    """Minimal single-client-friendly MJPEG server for the latest annotated frame."""

    def __init__(self, port: int, *, auto_open: bool = False):
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
        self._jpg: bytes | None = None
        self._lock = threading.Lock()
        outer = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):  # quiet
                pass

            def do_GET(self):
                self.send_response(200)
                self.send_header("Content-Type",
                                 "multipart/x-mixed-replace; boundary=frame")
                self.end_headers()
                while True:
                    with outer._lock:
                        buf = outer._jpg
                    if buf:
                        try:
                            self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n\r\n")
                            self.wfile.write(buf)
                            self.wfile.write(b"\r\n")
                        except (BrokenPipeError, ConnectionResetError):
                            return
                    time.sleep(0.1)

        self._srv = ThreadingHTTPServer(("0.0.0.0", port), H)
        threading.Thread(target=self._srv.serve_forever, daemon=True).start()
        url = f"http://127.0.0.1:{port}/"
        print(f"[viz] MJPEG stream -> {url}  (open in a browser)")
        if auto_open:
            # ``webbrowser`` delegates to the desktop's registered browser.  Run
            # it off-thread: a missing GUI/browser must never delay the camera
            # stream or the safety-critical motion path.
            def open_browser() -> None:
                try:
                    opened = webbrowser.open(url, new=2, autoraise=True)
                    if not opened:
                        print(f"[viz] could not auto-open browser; use {url}")
                except Exception as exc:  # desktop integration is optional
                    print(f"[viz] browser auto-open unavailable: {exc}; use {url}")
            threading.Thread(target=open_browser, daemon=True,
                             name="mjpeg-browser-open").start()

    def update(self, vis) -> None:
        ok, buf = cv2.imencode(".jpg", vis)
        if ok:
            with self._lock:
                self._jpg = buf.tobytes()


@dataclass
class Viz:
    outdir: Path = DEBUG_DIR
    window: bool = False
    mjpeg_port: int | None = None
    mjpeg_auto_open: bool = False
    # Sibling servos (e.g. the eye-in-hand wrist_servo) draw a different overlay
    # but want this file/MJPEG plumbing; leave unset for the overhead annotation.
    annotator: Any = None
    title: str = "depth_servo"
    _n: int = 0
    _stream: Any = None

    def __post_init__(self):
        self.outdir = Path(self.outdir)
        self.outdir.mkdir(parents=True, exist_ok=True)
        if self.mjpeg_port:
            self._stream = _MjpegStream(
                self.mjpeg_port, auto_open=self.mjpeg_auto_open
            )

    def show(self, color, depth=None, intr=None, *, tag="frame", save=True, **kw):
        """Annotate a frame; write it to disk unless save=False.

        ``save=False`` is for live streaming, where frames arrive many times per
        second and writing every one would bury the real per-step captures.
        """
        vis = (self.annotator or annotate)(color, depth, intr, **kw)
        if save:
            cv2.imwrite(str(self.outdir / f"{self._n:03d}_{tag}.jpg"), vis)
            self._n += 1
        if self._stream:
            self._stream.update(vis)
        if self.window:
            try:
                cv2.imshow(self.title, vis)
                cv2.waitKey(1)
            except Exception:  # headless -- stop trying
                self.window = False
        return vis


_VIZ: Viz | None = None  # set by the CLI; None in the MCP path (no viz overhead)


def set_viz(viz: Viz | None) -> None:
    global _VIZ
    _VIZ = viz


# =====================================================================================
# Overhead depth camera / local detector
# =====================================================================================
@dataclass
class DepthEye:
    """Owns the in-process depth camera and local open-vocabulary detector."""

    camera_kind: str = field(default_factory=lambda: CAMERA_KIND)
    _camera: DepthCamera | None = field(default=None, repr=False)
    _tip_lo: np.ndarray = field(default=None, repr=False)  # type: ignore[assignment]
    _tip_hi: np.ndarray = field(default=None, repr=False)  # type: ignore[assignment]

    def __post_init__(self) -> None:
        self._tip_lo, self._tip_hi = _env_tip_hsv()

    # -- raw frame ---------------------------------------------------------------
    def frame(self):
        if self._camera is None:
            self._camera = open_depth_camera(self.camera_kind)
        return self._camera.frame()

    def close(self) -> None:
        """Release the depth-camera stream; safe even if opening failed."""
        if self._camera is not None:
            self._camera.close()
            self._camera = None

    # -- object (open-vocabulary VLM box + robust box depth) ---------------------
    def object(self, name: str, color=None, depth=None, intr=None):
        """Return (p_cam_mm | None, info). info has box/uv/Z for visualization.
        Detection runs in this process on the exact acquired frame."""
        if color is None:
            color, depth, intr = self.frame()
        r = locate_local(color, name)
        if not r or not r.get("found"):
            return None, {}
        b = [int(v) for v in r["box"]]
        u, v = (b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0
        box = depth[b[1]:b[3], b[0]:b[2]]
        valid = box[box > 0]
        info = {"box": b, "uv": (u, v)}
        if valid.size < 10:
            return None, info
        Z = float(np.median(valid))
        info["Z"] = Z
        return _deproject(u, v, Z, intr), info

    def object_cam(self, name, color=None, depth=None, intr=None):
        return self.object(name, color, depth, intr)[0]

    # -- gripper tip (coloured marker; configurable for any arm) -----------------
    def tip_pixel(self, color):
        """Grasp-centre pixel of the tip marker(s): midpoint of the (up to two)
        largest coloured blobs. Works open (two fins) or closed (one blob)."""
        hsv = cv2.cvtColor(color, cv2.COLOR_BGR2HSV)
        m = cv2.inRange(hsv, self._tip_lo, self._tip_hi)
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = sorted((c for c in cnts if cv2.contourArea(c) > 60),
                      key=cv2.contourArea, reverse=True)[:2]
        cents = []
        for c in cnts:
            M = cv2.moments(c)
            if M["m00"] > 0:
                cents.append((M["m10"] / M["m00"], M["m01"] / M["m00"]))
        if not cents:
            return None
        return (float(np.mean([c[0] for c in cents])), float(np.mean([c[1] for c in cents])))

    def tip_cam(self, color=None, depth=None, intr=None):
        if color is None:
            color, depth, intr = self.frame()
        uv = self.tip_pixel(color)
        if uv is None:
            return None
        Z = _median_depth(depth, uv[0], uv[1])
        if Z <= 0:
            return None
        return _deproject(uv[0], uv[1], Z, intr)


# =====================================================================================
# Motion / gripper helpers (thin wrappers over the arm service, via arm_safe)
# =====================================================================================
def _scan_pose() -> tuple[float, float, float]:
    """Scan pose derived from arm_safe's live workspace so it fits whatever arm the
    service is running (RS or DM). Overridable with REBOT_SERVO_SCAN='x,y,z'."""
    raw = os.getenv("REBOT_SERVO_SCAN")
    if raw:
        try:
            x, y, z = (float(v) for v in raw.split(","))
            return (x, y, z)
        except ValueError:
            pass
    x = (A.X_MIN + A.X_MAX) / 2.0
    y = (A.Y_MIN + A.Y_MAX) / 2.0
    z = min(A.Z_MAX, A.Z_MIN + 0.145)
    return (x, y, z)


def _carry_z() -> float:
    return min(A.Z_MAX, A.Z_MIN + 0.19)


def _gripper(action: str) -> dict[str, Any]:
    """Open/close through the selected arm backend and return its readback."""
    assert action in ("open", "close")
    if A.BACKEND == "local":
        try:
            from arm_agent.local_arm import get_local_arm
            return get_local_arm().gripper(action)
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "reason": f"gripper_error {exc}"}
    try:
        import httpx
        r = httpx.post(f"{A.ARM}/gripper", json={"action": action}, timeout=25)
        if r.status_code == 200:
            return r.json()
        return {"ok": False, "reason": f"http_{r.status_code}", "body": r.text[:120]}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "reason": f"gripper_error {exc}"}


def _verify_grasp(close_resp: dict[str, Any]):
    """True if the gripper could NOT reach its full-close target -> something is held.
    None if the gripper didn't report a position (verification unavailable)."""
    pos, tgt = close_resp.get("position_rad"), close_resp.get("target_pos_rad")
    if pos is None or tgt is None:
        return None
    return abs(float(pos) - float(tgt)) > GRASP_GAP_RAD


def _verify_release(open_resp: dict[str, Any]):
    """True if the gripper reached its open target -> nothing jamming it (released)."""
    pos, tgt = open_resp.get("position_rad"), open_resp.get("target_pos_rad")
    if pos is None or tgt is None:
        return None
    return abs(float(pos) - float(tgt)) < GRIPPER_NEAR_RAD


# =====================================================================================
# Online Jacobian + servo core (calibration-free)
# =====================================================================================
def learn_jacobian(eye: DepthEye):
    """Learn the 3x3 J mapping a base-frame EE step (dx,dy,dz in m) to the tip's
    shift in the CAMERA frame (dx,dy,dz in mm): jog each base axis JAC_STEP, watch
    the tip, return to start. Self-calibrating -- no camera pose needed. Returns
    (J, "ok...") or (None, reason). J's z-column is base-up expressed in camera mm."""
    base = tuple(A.ee_pos())
    cols = []
    for axis in range(3):
        p0 = eye.tip_cam()
        if p0 is None:
            A.safe_goto(*base)
            return None, f"tip not visible before jogging axis {axis}"
        step = JAC_STEP
        d = [0.0, 0.0, 0.0]
        d[axis] = step
        r = A.safe_step(dx=d[0], dy=d[1], dz=d[2], cur=base)
        if not r.ok:  # near a workspace edge -- try the other direction
            step = -JAC_STEP
            d[axis] = step
            r = A.safe_step(dx=d[0], dy=d[1], dz=d[2], cur=base)
            if not r.ok:
                A.safe_goto(*base)
                return None, f"could not jog axis {axis}: {r.reason}"
        time.sleep(SETTLE_S)
        p1 = eye.tip_cam()
        if _VIZ is not None:
            c, d2, i2 = eye.frame()
            _VIZ.show(c, d2, i2, tag=f"jac_axis{axis}", tip_uv=eye.tip_pixel(c),
                      note=f"learn jacobian: jog base axis {axis}")
        A.safe_goto(*base, cur=r.reached)  # restore the start pose
        time.sleep(SETTLE_S)
        if p1 is None:
            return None, f"tip lost after jogging axis {axis}"
        cols.append((np.asarray(p1) - np.asarray(p0)) / step)  # mm per m, signed
    J = np.array(cols).T  # rows = camera xyz, cols = base xyz
    if min(float(np.linalg.norm(c)) for c in cols) < 3.0:
        return None, "a base jog produced no visible tip motion (tip occluded?)"
    cond = float(np.linalg.cond(J))
    if cond > 80.0:
        return None, f"Jacobian ill-conditioned (cond={cond:.0f})"
    return J, f"ok (cond={cond:.1f})"


def _servo_to(eye: DepthEye, Jinv: np.ndarray, target_cam: np.ndarray,
              tol_mm: float, log: Callable[[str], None], tag: str = "servo"):
    """Step the base EE until the tip reaches ``target_cam`` (camera frame, mm).
    Every step is arm_safe-gated (bounds + IK-verify + step cap)."""
    err = float("inf")
    for it in range(MAX_ITERS):
        color, depth, intr = eye.frame()
        uv = eye.tip_pixel(color)
        if uv is None:
            if _VIZ is not None:
                _VIZ.show(color, depth, intr, tag=f"{tag}{it:02d}",
                          target_cam=target_cam, note="tip lost")
            return False, f"tip lost during servo (it {it})"
        Z = _median_depth(depth, uv[0], uv[1])
        if Z <= 0:
            return False, f"tip depth missing (it {it})"
        tip = _deproject(uv[0], uv[1], Z, intr)
        e = np.asarray(target_cam, float) - tip  # mm, camera frame
        err = float(np.linalg.norm(e))
        if _VIZ is not None:
            _VIZ.show(color, depth, intr, tag=f"{tag}{it:02d}", tip_uv=uv,
                      target_cam=target_cam, err_mm=err, title=tag)
        if err < tol_mm:
            return True, f"aligned err={err:.0f}mm in {it} steps"
        raw = GAIN * (Jinv @ e)  # -> metres, base frame
        step = (float(np.clip(raw[0], -A.MAX_XY_STEP, A.MAX_XY_STEP)),
                float(np.clip(raw[1], -A.MAX_XY_STEP, A.MAX_XY_STEP)),
                float(np.clip(raw[2], -A.MAX_Z_STEP, A.MAX_Z_STEP)))
        r = A.safe_step(dx=step[0], dy=step[1], dz=step[2])
        log(f"  [it {it}] err={err:.0f}mm step=({step[0]*1000:+.0f},"
            f"{step[1]*1000:+.0f},{step[2]*1000:+.0f})mm -> {r.reason}")
        if not r.ok:
            return False, f"step refused (it {it}): {r.reason}"
        time.sleep(SETTLE_S)
    return False, f"did not converge (err={err:.0f}mm after {MAX_ITERS} steps)"


# =====================================================================================
# Result plumbing
# =====================================================================================
def _result(ok: bool, reason: str, steps: list[tuple[str, bool, str]],
            verified: bool | None = None) -> dict[str, Any]:
    out = {
        "ok": bool(ok),
        "reason": reason,
        "verified": verified,
        "steps": [{"step": s, "ok": bool(o), "reason": r} for (s, o, r) in steps],
    }
    if _VIZ is not None:
        out["frames"] = str(_VIZ.outdir)
    return out


def _printer(verbose: bool) -> Callable[[str], None]:
    return (lambda m: print(m)) if verbose else (lambda m: None)


# =====================================================================================
# SKILL 1: pick (independent, composable)
# =====================================================================================
def servo_pick(object_name: str, *, dry: bool = False, verbose: bool = False) -> dict[str, Any]:
    """Pick a named object with the overhead depth camera, calibration-free.

    pre:   arm ready + object visible with valid depth.
    act:   scan pose -> learn Jacobian -> servo tip to hover above object -> descend
           by feel (torque-gated) -> close.
    post:  verify the gripper stopped short of full close (object held, not slipped).
    reset: lift to a carry pose. Leaves the object HELD; knows nothing of placing.
    """
    log = _printer(verbose)
    steps: list[tuple[str, bool, str]] = []
    pf = A.preflight(connect=not dry)
    steps.append(("preflight", pf.ok, pf.reason))
    if not pf.ok:
        return _result(False, "arm not ready", steps)

    eye = DepthEye()
    try:
        color, depth, intr = eye.frame()
    except Exception as exc:  # noqa: BLE001
        steps.append(("camera", False, str(exc)))
        return _result(False, f"overhead camera unavailable ({CAMERA_KIND})", steps)

    p_obj, info = eye.object(object_name, color, depth, intr)
    if _VIZ is not None:
        _VIZ.show(color, depth, intr, tag="detect_object", obj_box=info.get("box"),
                  obj_uv=info.get("uv"), note=f"detect '{object_name}'")
    steps.append(("detect_object", p_obj is not None,
                  object_name if p_obj is not None else "not found / no depth"))
    if p_obj is None:
        return _result(False, f"'{object_name}' not found or has no depth", steps)
    if dry:
        return _result(True, "object localized in camera frame; safe to --run", steps)

    _gripper("open")
    r = A.safe_goto(*_scan_pose())
    steps.append(("scan_pose", r.ok, r.reason))
    if not r.ok:
        return _result(False, "could not reach scan pose (check arm_safe bounds for this arm)", steps)

    # Fixed camera + static object: measure the object ONCE from the scan view; then
    # servo only the (cheap) tip each iteration toward that fixed target.
    color, depth, intr = eye.frame()
    p_obj, _ = eye.object(object_name, color, depth, intr)
    if p_obj is None:
        A.safe_goto(*_scan_pose())
        return _result(False, f"'{object_name}' lost at scan pose", steps)

    J, why = learn_jacobian(eye)
    steps.append(("learn_jacobian", J is not None, why))
    if J is None:
        A.safe_goto(*_scan_pose())
        return _result(False, f"visual servo unavailable: {why}", steps)
    Jinv = np.linalg.inv(J)

    target = np.asarray(p_obj) + APPROACH_M * J[:, 2]  # hover APPROACH above object (base-up)
    ok, why = _servo_to(eye, Jinv, target, ALIGN_TOL_MM, log, tag="pick_align")
    steps.append(("align_above_object", ok, why))
    if not ok:
        A.safe_goto(*_scan_pose())
        return _result(False, f"could not align over object: {why}", steps)

    cur = tuple(A.ee_pos())
    res = A.safe_descend(cur[0], cur[1], A.Z_MIN)  # straight down, stops on torque / stall
    steps.append(("descend", res.ok, res.reason))
    if not res.ok:
        return _result(False, f"descent failed: {res.reason}", steps)
    cur = res.reached or cur

    close = _gripper("close")
    time.sleep(0.8)
    held = _verify_grasp(close)
    steps.append(("verify_grasp", bool(held),
                  "held" if held else ("empty/slipped" if held is False else "unknown")))
    if _VIZ is not None:
        try:
            c, d2, i2 = eye.frame()
            _VIZ.show(c, d2, i2, tag="after_grasp", tip_uv=eye.tip_pixel(c),
                      note=f"grasp verified={held}")
        except Exception:  # noqa: BLE001
            pass

    lift = A.safe_goto(cur[0], cur[1], _carry_z(), cur=cur)  # reset to carry pose
    steps.append(("lift_to_carry", lift.ok, lift.reason))

    return _result(bool(held),
                   "picked" if held else "grasp not confirmed",
                   steps, verified=held)


# =====================================================================================
# SKILL 2: place (independent, composable)
# =====================================================================================
def servo_place(*, target: str | None = None, x: float | None = None,
                y: float | None = None, dry: bool = False,
                verbose: bool = False) -> dict[str, Any]:
    """Place the CURRENTLY held object -- two modes, both independent of how it was picked.

    - target=<label>: visually servo the held object above the detected target
      (e.g. "the bowl"), then lower by feel and release. Calibration-free.
    - x,y=<metres>:   go straight to explicit table coords (no vision) and release.

    pre:   arm ready (assumes something is held).
    post:  verify the gripper reached its open target (released).
    reset: retreat to a carry pose.
    """
    log = _printer(verbose)
    steps: list[tuple[str, bool, str]] = []
    pf = A.preflight(connect=not dry)
    steps.append(("preflight", pf.ok, pf.reason))
    if not pf.ok:
        return _result(False, "arm not ready", steps)
    if target is None and (x is None or y is None):
        return _result(False, "need a place target label or explicit (x, y)", steps)

    if target is not None:
        eye = DepthEye()
        try:
            color, depth, intr = eye.frame()
        except Exception as exc:  # noqa: BLE001
            steps.append(("camera", False, str(exc)))
            return _result(False, f"overhead camera unavailable ({CAMERA_KIND})", steps)
        p_tgt, info = eye.object(target, color, depth, intr)
        if _VIZ is not None:
            _VIZ.show(color, depth, intr, tag="detect_target", obj_box=info.get("box"),
                      obj_uv=info.get("uv"), note=f"place target '{target}'")
        steps.append(("detect_target", p_tgt is not None,
                      target if p_tgt is not None else "not found / no depth"))
        if p_tgt is None:
            return _result(False, f"place target '{target}' not found", steps)
        if dry:
            return _result(True, "target localized in camera frame; safe to --run", steps)

        r = A.safe_goto(*_scan_pose())
        steps.append(("scan_pose", r.ok, r.reason))
        if not r.ok:
            return _result(False, "could not reach scan pose", steps)
        color, depth, intr = eye.frame()
        p_tgt, _ = eye.object(target, color, depth, intr)
        if p_tgt is None:
            return _result(False, f"target '{target}' lost at scan pose", steps)

        J, why = learn_jacobian(eye)
        steps.append(("learn_jacobian", J is not None, why))
        if J is None:
            return _result(False, f"visual servo unavailable: {why}", steps)
        Jinv = np.linalg.inv(J)

        above = np.asarray(p_tgt) + PLACE_CLEAR_M * J[:, 2]
        ok, why = _servo_to(eye, Jinv, above, ALIGN_TOL_MM, log, tag="place_align")
        steps.append(("align_above_target", ok, why))
        if not ok:
            return _result(False, f"could not align over target: {why}", steps)
        cur = tuple(A.ee_pos())
        res = A.safe_descend(cur[0], cur[1], A.Z_MIN + 0.005)  # lower held object onto target
        steps.append(("lower", res.ok, res.reason))
        if not res.ok:
            return _result(False, f"place descent failed: {res.reason}", steps)
        cur = res.reached or cur
    else:
        if dry:
            ik = A.check_ik(float(x), float(y), A.Z_MIN + 0.05)
            steps.append(("ik_check", ik.ok, ik.reason))
            return _result(ik.ok, "dry IK check for explicit place", steps)
        r = A.safe_goto(float(x), float(y), min(A.Z_MAX, A.Z_MIN + 0.10))
        steps.append(("goto_place_xy", r.ok, r.reason))
        if not r.ok:
            return _result(False, f"could not reach ({x}, {y})", steps)
        cur = tuple(A.ee_pos())
        res = A.safe_descend(cur[0], cur[1], A.Z_MIN + 0.01)
        steps.append(("lower", res.ok, res.reason))
        if not res.ok:
            return _result(False, f"place descent failed: {res.reason}", steps)
        cur = res.reached or cur

    opn = _gripper("open")
    time.sleep(0.5)
    released = _verify_release(opn)
    steps.append(("verify_release", bool(released),
                  "released" if released else "uncertain"))

    up = A.safe_goto(cur[0], cur[1], _carry_z(), cur=cur)  # reset to carry pose
    steps.append(("retreat", up.ok, up.reason))

    return _result(True, "placed", steps, verified=released)


# =====================================================================================
# Standalone visuals: live snapshot + synthetic demo (no arm motion)
# =====================================================================================
def snapshot(object_name: str | None = None) -> dict[str, Any]:
    """Grab ONE live overhead frame, annotate the tip (and object if given), save it.
    No arm motion -- great for aiming the camera and checking detection/depth."""
    eye = DepthEye()
    try:
        color, depth, intr = eye.frame()
    except Exception as exc:  # noqa: BLE001 -- bridge down / unreachable
        return {"ok": False, "reason": f"overhead camera unavailable ({CAMERA_KIND}): {exc}"}
    kw: dict[str, Any] = {"tip_uv": eye.tip_pixel(color)}
    if object_name:
        _, info = eye.object(object_name, color, depth, intr)
        kw.update(obj_box=info.get("box"), obj_uv=info.get("uv"))
    vis = annotate(color, depth, intr, note=object_name or "snapshot",
                   title="depth_servo snapshot", **kw)
    DEBUG_DIR.mkdir(parents=True, exist_ok=True)
    out = DEBUG_DIR / "snapshot.jpg"
    cv2.imwrite(str(out), vis)
    return {"ok": True, "image": str(out)}


def _demo_scene():
    """A synthetic overhead scene (gray table, a red block, two orange fingertips)
    with a matching depth map -- lets the annotation be viewed with NO hardware."""
    H, W = 480, 640
    color = np.full((H, W, 3), (70, 70, 70), np.uint8)
    depth = np.full((H, W), 720.0, np.float32)          # table at 720 mm
    cv2.rectangle(color, (298, 250), (360, 312), (40, 40, 205), -1)  # red block
    depth[250:312, 298:360] = 655.0                     # block top nearer
    cv2.rectangle(color, (300, 150), (322, 205), (0, 150, 255), -1)  # orange fin L
    cv2.rectangle(color, (344, 150), (366, 205), (0, 150, 255), -1)  # orange fin R
    depth[150:205, 300:366] = 560.0                     # gripper nearest
    intr = (600.0, 600.0, 320.0, 240.0)
    return color, depth, intr


def demo() -> str:
    """Render the annotation on the synthetic scene and save it -- no camera/arm."""
    color, depth, intr = _demo_scene()
    eye = DepthEye()
    tip_uv = eye.tip_pixel(color)
    obj_box = [298, 250, 360, 312]
    obj_uv = ((obj_box[0] + obj_box[2]) / 2.0, (obj_box[1] + obj_box[3]) / 2.0)
    # fake a learned Jacobian's base-up (camera -z) and hover target above the block
    obj_cam = _deproject(obj_uv[0], obj_uv[1], _median_depth(depth, *obj_uv), intr)
    up_cam = np.array([0.0, 0.0, -1.0]) * (700.0)       # ~ base-up in camera mm / m
    target_cam = obj_cam + APPROACH_M * up_cam
    tip_cam = _deproject(tip_uv[0], tip_uv[1], _median_depth(depth, *tip_uv), intr) if tip_uv else None
    err = float(np.linalg.norm(target_cam - tip_cam)) if tip_cam is not None else None
    vis = annotate(color, depth, intr, tip_uv=tip_uv, obj_box=obj_box, obj_uv=obj_uv,
                   target_cam=target_cam, err_mm=err, note="synthetic demo (no hardware)",
                   title="depth_servo demo")
    DEBUG_DIR.mkdir(parents=True, exist_ok=True)
    out = DEBUG_DIR / "demo.jpg"
    cv2.imwrite(str(out), vis)
    return {"ok": True, "image": str(out)}


# =====================================================================================
# CLI (hardware bring-up; validate each skill standalone before wiring MCP)
# =====================================================================================
def main() -> None:
    ap = argparse.ArgumentParser(description="Depth-camera visual-servo pick & place")
    ap.add_argument("--pick", metavar="OBJECT")
    ap.add_argument("--place", action="store_true")
    ap.add_argument("--target", metavar="OBJECT", help="place onto/into this detected object")
    ap.add_argument("--object", metavar="OBJECT", help="object to highlight in --snapshot")
    ap.add_argument("--x", type=float, help="explicit place x (m)")
    ap.add_argument("--y", type=float, help="explicit place y (m)")
    ap.add_argument("--dry", action="store_true", help="localize + validate only; no motion")
    ap.add_argument("--run", action="store_true", help="actually move the arm")
    ap.add_argument("--view", action="store_true", help="save an annotated frame per step")
    ap.add_argument("--window", action="store_true", help="also show a live cv2 window")
    ap.add_argument("--mjpeg", type=int, metavar="PORT", help="stream annotated frames to a browser")
    ap.add_argument("--camera", choices=("realsense", "orbbec", "auto"),
                    help="depth camera backend (default: REBOT_DEPTH_CAMERA or realsense)")
    ap.add_argument("--snapshot", action="store_true", help="one annotated live frame; no motion")
    ap.add_argument("--demo", action="store_true", help="annotate a synthetic scene; NO hardware")
    args = ap.parse_args()

    global CAMERA_KIND
    if args.camera:
        CAMERA_KIND = args.camera

    if args.view or args.window or args.mjpeg:
        set_viz(Viz(window=args.window, mjpeg_port=args.mjpeg))

    if args.demo:
        print(json.dumps(demo(), indent=2))
        return
    if args.snapshot:
        print(json.dumps(snapshot(args.object), indent=2))
        return
    if args.pick:
        out = servo_pick(args.pick, dry=not args.run, verbose=True)
    elif args.place:
        out = servo_place(target=args.target, x=args.x, y=args.y,
                          dry=not args.run, verbose=True)
    else:
        ap.error("use --demo | --snapshot | --pick OBJECT | --place [...]  (+ --dry/--run)")
        return
    print(json.dumps(out, indent=2))
    if args.mjpeg:
        print("[viz] leaving MJPEG stream up; Ctrl-C to exit.")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
