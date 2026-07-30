"""Offline simulator for the eye-in-hand wrist servo -- no camera, no arm, no CAN.

``wrist_servo`` is easy to get subtly wrong: with the camera bolted to the gripper,
a static object appears to move *opposite* to the arm, so every sign in the learned
Jacobian is inverted relative to the fixed-overhead ``depth_servo``.  A bench test
would find that with the arm already lunging the wrong way.

So the geometry here is built from first principles -- a pinhole camera rigidly
mounted to the tool, ray-casting a table plane and a cylindrical object -- and
knows nothing about how the servo solves anything.  If the servo converges against
this, its sign conventions are right.  ``SimArm`` mirrors the ``arm_safe`` surface
(bounds, step cap, torque-gated descent, gripper read-back) and ``SimCamera``
mirrors ``DepthCamera``, so the real servo code runs unmodified.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from . import gripper_model as _gm

# The RS arm's real envelope (arm_agent.arm_safe derives these from the shared
# hand-follow constraints).  Imported when available so the simulated workspace
# cannot drift from the hardware one; the literals are the documented fallback.
try:  # pragma: no cover - depends on the hardware checkout being present
    from . import arm_safe as _A
    _BOUNDS = (_A.X_MIN, _A.X_MAX, _A.Y_MIN, _A.Y_MAX, _A.Z_MIN, _A.Z_MAX)
    _STEP = (_A.MAX_XY_STEP, _A.MAX_Z_STEP)
except Exception:  # noqa: BLE001
    _BOUNDS = (0.21, 0.43, -0.1445, 0.1445, 0.18275, 0.32725)
    _STEP = (0.009, 0.009)


@dataclass
class MoveResult:
    ok: bool
    reason: str = ""
    reached: tuple[float, float, float] | None = None
    status: str = ""


def _rot_x(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], float)


def _rot_z(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], float)


@dataclass
class SimWorld:
    """Ground truth: where things are, and how the wrist camera is bolted on.

    ``R_bc`` holds the camera axes as columns in the base frame.  The nominal mount
    looks straight down with image-right along base +y and image-down along base +x;
    a deliberate tilt and roll are added so the servo cannot pass by assuming an
    axis-aligned camera.
    """

    object_xy: tuple[float, float] = (0.31, 0.055)
    object_radius_m: float = 0.022
    object_height_m: float = 0.045
    object_bgr: tuple[int, int, int] = (40, 40, 200)          # a red block
    table_z_m: float = _BOUNDS[4] + 0.004                     # just above the Z floor
    table_bgr: tuple[int, int, int] = (150, 150, 148)
    # Camera pose relative to the TCP, in the base frame: behind and above the jaws.
    mount_offset_m: tuple[float, float, float] = (-0.030, 0.0, 0.110)
    tilt_deg: float = 8.0
    roll_deg: float = 5.0
    intr: tuple[float, float, float, float] = (394.07, 393.66, 319.48, 245.50)  # real D405
    width: int = 640
    height: int = 480
    depth_noise_mm: float = 1.5
    dropout: float = 0.01
    seed: int = 0

    # Start pose, near the URDF's neutral EE. It must be IK-reachable at the working
    # pitch: parking the arm at the top of the workspace box makes even the first
    # interpolated sub-step unreachable, which reads as a puzzling scan failure.
    ee: np.ndarray = field(default_factory=lambda: np.array([0.30, 0.0, 0.215]))
    gripper_rad: float = 0.0
    held: bool = False
    _rng: np.random.Generator = field(default=None, repr=False)  # type: ignore[assignment]

    def __post_init__(self) -> None:
        self._rng = np.random.default_rng(self.seed)
        base = np.array([[0.0, 1.0, 0.0],    # camera x (image right) = base +y
                         [1.0, 0.0, 0.0],    # camera y (image down)  = base +x
                         [0.0, 0.0, -1.0]])  # camera z (optical)     = base -z
        self.R_bc = _rot_z(math.radians(self.roll_deg)) @ base @ _rot_x(math.radians(self.tilt_deg))

    # -- ground truth the servo is not allowed to see -------------------------
    @property
    def R_cb(self) -> np.ndarray:
        return self.R_bc.T

    @property
    def cam_origin(self) -> np.ndarray:
        return self.ee + np.asarray(self.mount_offset_m, float)

    def object_center(self) -> np.ndarray:
        """Object centre-top in the base frame (it rides along once grasped)."""
        if self.held:
            return np.array([self.ee[0], self.ee[1], self.ee[2]])
        return np.array([self.object_xy[0], self.object_xy[1],
                         self.table_z_m + self.object_height_m])

    def to_camera(self, p_base: np.ndarray) -> np.ndarray:
        """Base-frame point -> camera frame, in millimetres."""
        return 1000.0 * (self.R_cb @ (np.asarray(p_base, float) - self.cam_origin))

    # -- rendering ------------------------------------------------------------
    def render(self) -> tuple[np.ndarray, np.ndarray, tuple[float, float, float, float]]:
        """Ray-cast the table plane and the object into an aligned RGB-D frame."""
        fx, fy, cx, cy = self.intr
        us, vs = np.meshgrid(np.arange(self.width, dtype=np.float32),
                             np.arange(self.height, dtype=np.float32))
        # Ray directions with a unit camera-frame z, so the ray parameter t IS the
        # camera-frame depth of the hit point.
        dirs_cam = np.stack([(us - cx) / fx, (vs - cy) / fy, np.ones_like(us)], axis=-1)
        dirs_base = dirs_cam @ self.R_bc.T
        origin = self.cam_origin

        color = np.zeros((self.height, self.width, 3), np.uint8)
        color[:] = self.table_bgr
        depth = np.zeros((self.height, self.width), np.float32)

        def hit_plane(z_plane: float) -> tuple[np.ndarray, np.ndarray]:
            dz = dirs_base[..., 2]
            with np.errstate(divide="ignore", invalid="ignore"):
                t = (z_plane - origin[2]) / dz
            valid = np.isfinite(t) & (t > 0) & (dz < -1e-9)
            return t, valid

        t_table, table_ok = hit_plane(self.table_z_m)
        depth[table_ok] = t_table[table_ok] * 1000.0

        # Object: a flat disc at its top face (the only surface a top-down wrist
        # camera sees), rendered in front of the table wherever it is hit.
        obj = self.object_center()
        t_top, top_ok = hit_plane(obj[2])
        hit = origin + t_top[..., None] * dirs_base
        radial = np.hypot(hit[..., 0] - obj[0], hit[..., 1] - obj[1])
        on_object = top_ok & (radial <= self.object_radius_m)
        color[on_object] = self.object_bgr
        depth[on_object] = t_top[on_object] * 1000.0

        # The jaws: dark, close, either side of the tool axis.  They exist so the
        # detector's depth gate and colour filter are exercised, as on the real arm.
        for sign in (-1.0, 1.0):
            jaw = np.array([self.ee[0], self.ee[1] + sign * 0.022, self.ee[2] + 0.012])
            p = self.to_camera(jaw)
            if p[2] > 1.0:
                u = int(p[0] * fx / p[2] + cx)
                v = int(p[1] * fy / p[2] + cy)
                x0, x1 = max(0, u - 12), min(self.width, u + 12)
                y0, y1 = max(0, v - 20), min(self.height, v + 20)
                if x1 > x0 and y1 > y0:
                    color[y0:y1, x0:x1] = (45, 45, 45)
                    depth[y0:y1, x0:x1] = p[2]

        noise = self._rng.normal(0.0, self.depth_noise_mm, depth.shape).astype(np.float32)
        depth = np.where(depth > 0, depth + noise, 0.0).astype(np.float32)
        if self.dropout > 0:
            depth[self._rng.random(depth.shape) < self.dropout] = 0.0
        color = np.clip(color.astype(np.int16)
                        + self._rng.normal(0, 3, color.shape).astype(np.int16), 0, 255).astype(np.uint8)
        return color, depth, self.intr


class SimCamera:
    """``DepthCamera``-compatible view onto a :class:`SimWorld`."""

    def __init__(self, world: SimWorld) -> None:
        self.world = world

    def frame(self):
        return self.world.render()

    def close(self) -> None:
        pass


class SimArm:
    """``arm_safe``-compatible motion gate over a :class:`SimWorld`.

    Enforces the same guarantees the real gate does -- workspace AABB, per-move step
    cap, contact-stopped descent -- so a servo bug that only shows up against those
    limits (a step the real arm would refuse) still shows up here.
    """

    BACKEND = "sim"
    ARM = "sim://wrist"
    X_MIN, X_MAX, Y_MIN, Y_MAX, Z_MIN, Z_MAX = _BOUNDS
    MAX_XY_STEP, MAX_Z_STEP = _STEP
    # Matches local_arm gripper semantics: closing onto an object stops short of the
    # full-close target, which is how a grasp is verified. The open target tracks the
    # same capped travel fraction so the sim cannot model a wider jaw than hardware
    # will actually give (testing default: 50%).
    # Endpoints and caps come from gripper_model, so the sim opens and closes to
    # exactly the angles the hardware backend would command -- including the 5%
    # close limit that keeps the motor off its mechanical stop.
    OPEN_RAD = _gm.OPEN_RAD
    CLOSE_RAD = _gm.CLOSE_RAD

    def __init__(self, world: SimWorld, *, grasp_tol_m: float = 0.018,
                 trace: bool = False, kinematic: bool = True,
                 pitch: float = 1.0) -> None:
        self.world = world
        self.grasp_tol_m = grasp_tol_m
        # Tool pitch held for the whole run; see wrist_servo.WORK_PITCH for why it
        # is not arm_safe's near-vertical DEF_PITCH.
        self.pitch = pitch
        self.moves = 0
        self.refused: list[str] = []
        # Bounds alone are far too permissive: on this arm the whole upper half of
        # the workspace box is unreachable at the servo's down-pitch. Delegating to
        # the SAME Pinocchio IK the hardware path uses makes the simulator reject
        # exactly what the real gate would, so unreachable scan poses surface here
        # rather than on the bench. Falls back to bounds-only if the model is absent.
        self._ik: Any = None
        if kinematic:
            try:
                from .local_arm import get_local_arm
                self._ik = get_local_arm()
            except Exception:  # noqa: BLE001
                self._ik = None
        self.kinematic = self._ik is not None
        # Waypoint log for the 3-D replay (arm_agent.wrist_replay); off by default
        # so a plain --sim run stays allocation-free.
        self.trace: list[dict[str, Any]] | None = [] if trace else None
        if self.trace is not None:
            self._record("start")

    def _record(self, label: str) -> None:
        if self.trace is None:
            return
        w = self.world
        self.trace.append({
            "label": label,
            "ee": [float(v) for v in w.ee],
            "gripper_rad": float(w.gripper_rad),
            "held": bool(w.held),
            "object_xy": [float(v) for v in w.object_xy],
        })

    # -- state ---------------------------------------------------------------
    def ee_pos(self) -> list[float]:
        return [float(v) for v in self.world.ee]

    def ee_rpy(self) -> list[float]:
        return [0.0, self.pitch, 0.0]

    def joint_torques(self) -> list[float]:
        return [0.0] * 6

    def connected(self) -> bool:
        return True

    def in_bounds(self, x: float, y: float, z: float) -> bool:
        return (self.X_MIN <= x <= self.X_MAX and self.Y_MIN <= y <= self.Y_MAX
                and self.Z_MIN <= z <= self.Z_MAX)

    def preflight(self, *, connect: bool = False) -> MoveResult:
        return MoveResult(True, "ready", tuple(self.ee_pos()))

    def check_ik(self, x, y, z, roll=0.0, pitch=None, yaw=0.0) -> MoveResult:
        pitch = self.pitch if pitch is None else pitch
        if not self.in_bounds(x, y, z):
            return MoveResult(False, f"out_of_bounds ({x:.3f},{y:.3f},{z:.3f})")
        if self._ik is None:
            return MoveResult(True, "ik_ok (bounds only)", (x, y, z))
        try:
            solution = self._ik.solve_ik(x, y, z, roll, pitch, yaw)
        except Exception as exc:  # noqa: BLE001 - same refusal the hardware gate gives
            return MoveResult(False, f"ik_unreachable ({x:.3f},{y:.3f},{z:.3f}): "
                                     f"{str(exc)[:60]}")
        p = solution["end_effector"]["position_m"]
        off = max(abs(p[0] - x), abs(p[1] - y), abs(p[2] - z))
        if off > 0.006:
            return MoveResult(False, f"ik_did_not_reach (off {off*1000:.1f}mm)")
        return MoveResult(True, "ik_ok", (float(p[0]), float(p[1]), float(p[2])))

    # -- motion --------------------------------------------------------------
    def safe_move(self, x, y, z, *, cur=None, dry=False, roll=None, pitch=None,
                  yaw=None, **kw) -> MoveResult:
        cur = tuple(self.ee_pos()) if cur is None else cur
        dx, dy, dz = x - cur[0], y - cur[1], z - cur[2]
        eps = 1e-5
        if abs(dx) > self.MAX_XY_STEP + eps or abs(dy) > self.MAX_XY_STEP + eps \
                or abs(dz) > self.MAX_Z_STEP + eps:
            reason = f"step_too_large dx={dx*1000:.0f} dy={dy*1000:.0f} dz={dz*1000:.0f}mm"
            self.refused.append(reason)
            return MoveResult(False, reason)
        # Mirror arm_safe: an unspecified orientation preserves the live one.
        ik = self.check_ik(x, y, z, 0.0 if roll is None else roll,
                           self.pitch if pitch is None else pitch,
                           0.0 if yaw is None else yaw)
        if not ik.ok:
            self.refused.append(ik.reason)
            return ik
        if dry:
            return MoveResult(True, "dry_ok", (x, y, z))
        self.world.ee = np.array([x, y, z], float)
        self.moves += 1
        self._record("move")
        return MoveResult(True, "arrived", (x, y, z))

    def safe_step(self, dx=0.0, dy=0.0, dz=0.0, *, cur=None, **kw) -> MoveResult:
        cur = tuple(self.ee_pos()) if cur is None else cur
        return self.safe_move(cur[0] + dx, cur[1] + dy, cur[2] + dz, cur=cur, **kw)

    def safe_goto(self, x, y, z, *, cur=None, dry=False, **kw) -> MoveResult:
        cur = tuple(self.ee_pos()) if cur is None else cur
        dx, dy, dz = x - cur[0], y - cur[1], z - cur[2]
        n = max(1, int(math.ceil(max(abs(dx) / self.MAX_XY_STEP, abs(dy) / self.MAX_XY_STEP,
                                     abs(dz) / self.MAX_Z_STEP))))
        res = MoveResult(True, "noop", cur)
        for i in range(1, n + 1):
            prev = (cur[0] + dx * (i - 1) / n, cur[1] + dy * (i - 1) / n, cur[2] + dz * (i - 1) / n)
            tgt = (cur[0] + dx * i / n, cur[1] + dy * i / n, cur[2] + dz * i / n)
            res = self.safe_move(*tgt, cur=prev, dry=dry, **kw)
            if not res.ok:
                return res
        return res

    def safe_descend(self, x, y, z_floor, *, step=0.004, **kw) -> MoveResult:
        """Descend until the jaws straddle the object, the table blocks, or the floor."""
        w = self.world
        contact_z = w.table_z_m + 0.004
        if not w.held:
            over_object = math.hypot(x - w.object_xy[0], y - w.object_xy[1])
            if over_object <= w.object_radius_m + 0.03:
                # Jaws close around the object rather than resting on the table.
                contact_z = w.table_z_m + w.object_height_m * 0.45
        z = float(w.ee[2])
        last = tuple(self.ee_pos())
        while z - z_floor > 1e-4:
            z_cmd = max(z_floor, z - step)
            r = self.safe_move(x, y, z_cmd, cur=(x, y, z))
            if not r.ok:
                return MoveResult(False, f"move_refused: {r.reason}", last, "move_refused")
            last = r.reached or last
            z = last[2]
            if z <= contact_z:
                return MoveResult(True, f"contact:stall at z={z:.4f}", last, "contact_stall")
        return MoveResult(True, f"reached floor z={z:.4f}", last, "floor_reached")

    def home(self) -> dict[str, Any]:
        """Park at the neutral pose, mirroring local_arm.home()'s safe_home."""
        target = (0.30, 0.0, 0.215)
        r = self.safe_goto(*target)
        self._record("home")
        return {"ok": r.ok, "reason": r.reason, "position_m": self.ee_pos()}

    # -- gripper -------------------------------------------------------------
    def gripper_readback(self, action: str, target_rad: float | None = None,
                         limit_rads: tuple[float, float] | None = None) -> dict[str, Any]:
        """Open/close with the position read-back the servo verifies grasps from.

        ``target_rad`` (a sized aperture) and ``limit_rads`` are accepted and clamped
        exactly as the hardware backend does, so the sim exercises the same path.
        """
        w = self.world
        if target_rad is not None and action == "open":
            target_rad = _gm.clamp_rad(float(target_rad), limit_rads)
            w.gripper_rad = target_rad
            w.held = False
            self._record("gripper_open")
            return {"ok": True, "position_rad": target_rad, "target_pos_rad": target_rad}
        if action == "open":
            if w.held:  # released here: the object stays where it was let go
                w.object_xy = (float(w.ee[0]), float(w.ee[1]))
            w.held = False
            open_rad = (limit_rads or (_gm.CLOSE_RAD, _gm.OPEN_RAD))[1]
            w.gripper_rad = open_rad
            self._record("gripper_open")
            return {"ok": True, "position_rad": open_rad, "target_pos_rad": open_rad}
        obj = w.object_center()
        # The jaw fingers extend around the TCP, so their grasp volume has finite
        # vertical extent; requiring the TCP itself to pass below the object's top
        # incorrectly rejects a deliberate small non-contact stand-off.
        aligned = (
            math.hypot(float(w.ee[0]) - obj[0], float(w.ee[1]) - obj[1])
            <= self.grasp_tol_m
            and abs(float(w.ee[2]) - float(obj[2])) <= self.grasp_tol_m
        )
        position = (_gm.clamp_rad(float(target_rad), limit_rads)
                    if target_rad is not None else _gm.target_for("close", limit_rads))
        # A sized close is considered capable of enclosing the simulated object's
        # diameter. This models position control only; no contact/stall signal.
        target_mm = _gm.aperture_mm(position)
        object_mm = 2.0 * w.object_radius_m * 1000.0
        w.held = bool(aligned and target_mm is not None and target_mm >= object_mm - 2.0)
        w.gripper_rad = position
        self._record("gripper_close")
        return {"ok": True, "position_rad": position, "target_pos_rad": position,
                "target_rad": position, "position_deg": math.degrees(position),
                "target_deg": math.degrees(position), "contact_aware": False}

    def gripper(self, action: str) -> bool:
        return bool(self.gripper_readback(action).get("ok"))
