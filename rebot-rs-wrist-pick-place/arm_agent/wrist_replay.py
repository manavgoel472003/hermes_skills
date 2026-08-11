"""Watch the ARM do the wrist-camera pick & place, in 3-D, with no hardware.

``wrist_servo --sim --view`` shows what the *camera* saw. This shows the *robot*:
the simulated run's tool path is pushed back through the real Pinocchio IK and the
real URDF, then streamed to the rerun viewer as an articulated arm you can orbit
and scrub through time -- with the synchronised wrist view beside it.

Nothing here invents motion. Every pose comes from the trajectory
``arm_agent.wrist_sim`` actually executed under ``arm_safe``'s limits, so what you
watch is exactly what the servo commanded. Joint angles come from the same IK the
hardware path uses (``solve_ik_with_retry`` on ``00-arm-rs_asm-v3.urdf``), seeded
from the previous frame so the replay is continuous rather than jumping between
elbow branches. The FK residual is reported: if it is sub-millimetre, the arm you
see is on the trajectory the servo drove.

    conda activate rebot_rs
    python -m arm_agent.wrist_replay                  # writes an .rrd, prints how to view
    python -m arm_agent.wrist_replay --spawn          # open the rerun viewer directly
    python -m arm_agent.wrist_replay --object-xy 0.26,-0.10 --tcp-error-mm 5
    python -m arm_agent.wrist_replay --no-camera      # skip the wrist-view panel (faster)
"""
from __future__ import annotations

import argparse
import struct
import sys
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pinocchio as pin

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
CONTROL_ROOT = ROOT / "reBotArm_control_py_fresh"
if str(CONTROL_ROOT) not in sys.path:
    sys.path.insert(0, str(CONTROL_ROOT))

from reBotArm_control_py.kinematics import (  # noqa: E402
    get_end_effector_frame_id, load_robot_model, pad_q_for_model, pos_rot_to_se3,
)
from reBotArm_control_py.kinematics.inverse_kinematics import (  # noqa: E402
    IKParams, solve_ik_with_retry,
)

DEFAULT_OUT = ROOT / "outputs" / "wrist_sim" / "wrist_replay.rrd"


def _replay_rpy() -> tuple[float, float, float]:
    """The orientation the run was actually executed at. Read from wrist_servo so
    the replay cannot drift from the servo's working pitch (it is NOT arm_safe's
    near-vertical DEF_PITCH -- see WORK_PITCH)."""
    from arm_agent.wrist_servo import WORK_PITCH

    return (0.0, WORK_PITCH, 0.0)


# =====================================================================================
# URDF visual geometry
# =====================================================================================
@dataclass
class Visual:
    link: str
    vertices: np.ndarray          # (3T, 3) triangle soup, link frame, metres
    color: tuple[float, float, float, float]


def _urdf_paths() -> tuple[Path, Path]:
    """The URDF the ARM ITSELF loads, so the replay cannot drift from the robot."""
    from reBotArm_control_py.kinematics import robot_model

    path, pkg = robot_model._resolve_urdf(None)  # no public accessor for the path
    return Path(path), Path(pkg)


def _load_stl(path: Path) -> np.ndarray:
    """Binary-STL triangle soup. These meshes carry no scale attribute and are
    already in metres (verified against the arm's real link extents)."""
    raw = path.read_bytes()
    count = struct.unpack("<I", raw[80:84])[0]
    if len(raw) != 84 + 50 * count:
        raise ValueError(f"{path.name} is not a binary STL")
    rows = np.frombuffer(raw[84:84 + 50 * count], dtype=np.uint8).reshape(count, 50)
    floats = rows[:, :48].copy().view("<f4").reshape(count, 12)
    return floats[:, 3:12].reshape(count * 3, 3).astype(np.float32)


def _rpy_matrix(rpy: tuple[float, float, float]) -> np.ndarray:
    r, p, y = rpy
    cr, sr, cp, sp, cy, sy = (np.cos(r), np.sin(r), np.cos(p),
                              np.sin(p), np.cos(y), np.sin(y))
    return np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp,     cp * sr,                cp * cr],
    ])


def load_visuals() -> list[Visual]:
    """Every <visual> mesh in the URDF, baked into its link frame."""
    urdf, pkg = _urdf_paths()
    root = ET.parse(urdf).getroot()
    out: list[Visual] = []
    for link in root.findall("link"):
        name = link.get("name") or "?"
        for visual in link.findall("visual"):
            mesh = visual.find("geometry/mesh")
            if mesh is None:
                continue
            filename = mesh.get("filename", "")
            path = (pkg / filename) if not Path(filename).is_absolute() else Path(filename)
            if not path.exists():                      # tolerate package:// style refs
                path = pkg / Path(filename).name
                if not path.exists():
                    continue
            try:
                verts = _load_stl(path)
            except Exception:  # noqa: BLE001 - a bad mesh must not kill the replay
                continue
            origin = visual.find("origin")
            if origin is not None:
                xyz = np.fromstring(origin.get("xyz", "0 0 0"), sep=" ")
                rpy = np.fromstring(origin.get("rpy", "0 0 0"), sep=" ")
                verts = verts @ _rpy_matrix(tuple(rpy)).T + xyz
            rgba = visual.find("material/color")
            color = (0.75, 0.75, 0.78, 1.0)
            if rgba is not None:
                parsed = np.fromstring(rgba.get("rgba", ""), sep=" ")
                if parsed.size == 4 and parsed[:3].sum() > 0:
                    color = tuple(float(v) for v in parsed)  # type: ignore[assignment]
            out.append(Visual(name, verts.astype(np.float32), color))
    return out


# =====================================================================================
# Trajectory -> joint angles
# =====================================================================================
def solve_trajectory(trace: list[dict[str, Any]]) -> tuple[np.ndarray, dict[str, float]]:
    """IK every recorded tool pose, seeded from the previous frame for continuity.

    Returns (q per frame, stats). ``stats['max_fk_error_mm']`` is the check that
    matters: it is how far the rendered arm's tool sits from the commanded path.
    """
    model = load_robot_model()
    data = model.createData()
    end_frame = get_end_effector_frame_id(model)
    params = IKParams(max_iter=200, tolerance=1e-4, step_size=0.5, damping=1e-6)

    roll, pitch, yaw = _replay_rpy()
    qs, errors, failures = [], [], 0
    seed = pad_q_for_model(model, np.zeros(6), controlled_joints=6)
    for point in trace:
        x, y, z = point["ee"]
        target = pos_rot_to_se3(np.array([x, y, z]), roll=roll, pitch=pitch, yaw=yaw)
        try:
            result = solve_ik_with_retry(model, data, end_frame, target, seed,
                                         params, max_retries=0)
            q = np.asarray(result.q, dtype=np.float64)
        except Exception:  # noqa: BLE001 - hold the last good pose, keep the replay going
            failures += 1
            q = seed.copy()
        pin.forwardKinematics(model, data, q)
        pin.updateFramePlacements(model, data)
        errors.append(float(np.linalg.norm(data.oMf[end_frame].translation - np.array([x, y, z]))))
        seed = q.copy()

        # Jaws: map the sim's gripper travel onto the URDF's two prismatic joints.
        fraction = min(1.0, abs(point["gripper_rad"]) / 4.712388980384690)
        q = q.copy()
        if model.nq >= 8:
            q[6] = fraction * float(model.upperPositionLimit[6])
            q[7] = fraction * float(model.upperPositionLimit[7])
        qs.append(q)

    stats = {
        "frames": float(len(qs)),
        "max_fk_error_mm": float(np.max(errors) * 1000) if errors else 0.0,
        "mean_fk_error_mm": float(np.mean(errors) * 1000) if errors else 0.0,
        "ik_failures": float(failures),
    }
    return np.array(qs), stats


# =====================================================================================
# rerun logging
# =====================================================================================
def _set_frame(rr: Any, index: int) -> None:
    """rerun renamed the timeline API; support both spellings."""
    try:
        rr.set_time("frame", sequence=index)
    except (AttributeError, TypeError):
        rr.set_time_sequence("frame", index)


def replay(result: dict[str, Any], out: Path = DEFAULT_OUT, *, spawn: bool = False,
           camera: bool = True, verbose: bool = True) -> dict[str, Any]:
    """Stream a traced ``run_sim`` result to rerun as an articulated 3-D arm."""
    import rerun as rr

    from arm_agent import wrist_sim

    sim = result.get("sim") or {}
    trace = sim.get("trace")
    if not trace:
        raise ValueError("run_sim(...) was not traced; pass trace=True")
    world_info = sim["world"]

    qs, stats = solve_trajectory(trace)
    if verbose:
        print(f"[replay] {len(trace)} waypoints  "
              f"max FK error {stats['max_fk_error_mm']:.2f} mm  "
              f"mean {stats['mean_fk_error_mm']:.2f} mm  "
              f"IK failures {int(stats['ik_failures'])}")

    model = load_robot_model()
    data = model.createData()
    visuals = load_visuals()
    if verbose:
        print(f"[replay] {len(visuals)} URDF meshes across "
              f"{len({v.link for v in visuals})} links")

    rr.init("rebot_wrist_pick_place", spawn=spawn)
    rr.log("/", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)

    # Static scene: meshes (logged once), workspace box, table, drop zone.
    for i, vis in enumerate(visuals):
        rr.log(f"world/robot/{vis.link}_{i}",
               rr.Mesh3D(vertex_positions=vis.vertices,
                         triangle_indices=np.arange(len(vis.vertices)).reshape(-1, 3),
                         albedo_factor=vis.color),
               static=True)

    x0, x1, y0, y1, z0, z1 = world_info["bounds"]
    rr.log("world/workspace",
           rr.Boxes3D(centers=[[(x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2]],
                      half_sizes=[[(x1 - x0) / 2, (y1 - y0) / 2, (z1 - z0) / 2]],
                      colors=[[90, 140, 255, 40]], labels=["safe workspace"]),
           static=True)
    table_z = world_info["table_z_m"]
    rr.log("world/table",
           rr.Boxes3D(centers=[[(x0 + x1) / 2, 0.0, table_z - 0.005]],
                      half_sizes=[[0.28, 0.28, 0.005]], colors=[[120, 120, 118, 120]]),
           static=True)
    drop = world_info["drop_xyz_m"]
    rr.log("world/drop_zone",
           rr.Boxes3D(centers=[[drop[0], drop[1], table_z + 0.02]],
                      half_sizes=[[0.05, 0.05, 0.02]], colors=[[255, 170, 40, 90]],
                      labels=["drop zone"]),
           static=True)

    # Per-frame: link transforms, object, tool path, and the wrist view.
    sim_world = wrist_sim.SimWorld()
    sim_world.object_xy = tuple(trace[0]["object_xy"])  # type: ignore[assignment]
    radius = world_info["object_radius_m"]
    height = world_info["object_height_m"]
    path_points: list[list[float]] = []

    for index, (point, q) in enumerate(zip(trace, qs)):
        _set_frame(rr, index)
        pin.forwardKinematics(model, data, q)
        pin.updateFramePlacements(model, data)
        for i, vis in enumerate(visuals):
            fid = model.getFrameId(vis.link)
            placement = data.oMf[fid]
            rr.log(f"world/robot/{vis.link}_{i}",
                   rr.Transform3D(translation=placement.translation,
                                  mat3x3=placement.rotation))

        ox, oy = point["object_xy"]
        oz = (point["ee"][2] if point["held"] else table_z + height / 2)
        rr.log("world/object",
               rr.Boxes3D(centers=[[ox, oy, oz]],
                          half_sizes=[[radius, radius, height / 2]],
                          colors=[[220, 50, 50, 255]],
                          labels=["held" if point["held"] else "red block"]))

        path_points.append([float(v) for v in point["ee"]])
        rr.log("world/tool_path", rr.LineStrips3D([path_points],
                                                  colors=[[80, 220, 120, 255]]))
        rr.log("phase", rr.TextLog(point["label"]))

        if camera:
            sim_world.ee = np.array(point["ee"], float)
            sim_world.object_xy = (ox, oy)
            sim_world.held = point["held"]
            color_img, depth_img, _ = sim_world.render()
            rr.log("wrist_camera/rgb", rr.Image(color_img[:, :, ::-1]))
            rr.log("wrist_camera/depth", rr.DepthImage(depth_img, meter=1000.0))

    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if not spawn:
        rr.save(str(out))
    stats["output"] = str(out)  # type: ignore[assignment]
    return stats


def main() -> None:
    from arm_agent import wrist_servo as W

    p = argparse.ArgumentParser(description="3-D replay of the wrist pick & place")
    p.add_argument("--object", default="red block")
    p.add_argument("--object-xy", help="'x,y' in metres")
    p.add_argument("--tcp-error-mm", type=float, default=0.0)
    p.add_argument("--out", default=str(DEFAULT_OUT))
    p.add_argument("--spawn", action="store_true", help="open the rerun viewer now")
    p.add_argument("--no-camera", action="store_true", help="skip the wrist-view panels")
    args = p.parse_args()

    xy = None
    if args.object_xy:
        parts = [float(v) for v in args.object_xy.split(",")]
        xy = (parts[0], parts[1])

    started = time.time()
    result = W.run_sim(args.object, object_xy=xy, tcp_error_mm=args.tcp_error_mm,
                       verbose=False, trace=True)
    print(f"[sim] {result['reason']}  (ok={result['ok']})")
    stats = replay(result, Path(args.out), spawn=args.spawn, camera=not args.no_camera)
    print(f"[replay] built in {time.time() - started:.1f}s")
    if not args.spawn:
        print(f"\nOpen it with:\n    rerun {stats['output']}")


if __name__ == "__main__":
    main()
