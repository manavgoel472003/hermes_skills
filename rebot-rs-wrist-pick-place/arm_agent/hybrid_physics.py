"""Contact-physics benchmark, not the kinematic wrist_sim or animation renderer.

Run in a fresh process. Only reset initializes qpos; actions change actuator ctrl.
Ground truth is reserved for the evaluator and recorded-state visualization.
The policy sees the usual RGB-D camera and arm encoder interfaces.
"""
from __future__ import annotations

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
import os
from pathlib import Path
import threading
import time
import shutil
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SIM_CAMERA_FOVY_DEG = 80.0
os.environ.setdefault("MUJOCO_GL", "egl")
# Keep physics optional for the hardware environment.  This local dependency is
# deliberately not installed into its site-packages or deployed to the robot.
_MUJOCO_VENDOR = ROOT / ".vendor" / "hybrid_physics_deps"
if _MUJOCO_VENDOR.is_dir():
    import sys
    sys.path.insert(0, str(_MUJOCO_VENDOR))
import mujoco


def simulation_calibration(directory):
    """Calibrate a declared virtual wrist mount, never the task's object pose.

    The hardware's saved camera is occluded by this CAD gripper. A virtual mount
    above the jaw tips is part of the sim model, with its own calibration folder.
    Physical calibration artifacts are never changed.
    """
    directory.mkdir(parents=True)
    for source in (ROOT / "calib").glob("wrist_*.json"):
        shutil.copy2(source, directory / source.name)
    # Forward/above the wrist, aimed at the work area rather than through the
    # CAD palm. This is a declared *virtual* mounting geometry for the benchmark.
    position = np.array([.100, -.080, .180])
    # Fixed wrist-frame look vector aimed below and ahead of the jaws; it does
    # not encode either scene object's pose.
    zaxis = np.array([.065, -.070, -.200]) - position
    zaxis /= np.linalg.norm(zaxis)
    xaxis = np.cross(np.array([1., 0, 0]), zaxis)
    xaxis /= np.linalg.norm(xaxis)
    yaxis = np.cross(zaxis, xaxis)
    rotation = np.column_stack([xaxis, yaxis, zaxis])
    tcp_cam = -rotation.T @ position * 1000
    payloads = {
        "wrist_camera_rotation.json": {"base_to_camera_rotation": (-rotation.T).tolist(),
                                       "calibrated_tool_rpy_rad": [0, 0, 0]},
        "wrist_tcp.json": {"tcp_camera_xyz_mm": tcp_cam.tolist()},
        "wrist_grasp_target.json": {"object_camera_xyz_mm": tcp_cam.tolist()},
    }
    for name, data in payloads.items():
        (directory / name).write_text(json.dumps({**data, "camera": "realsense",
                    "method": "simulated_mount_geometry", "physical_calibration": False}, indent=2))
    return position.tolist()


def build_scene(object_kind="ball", seed=0):
    """RS URDF, dynamic object and genuinely hollow, dynamic cup."""
    robot_dir = ROOT / "reBotArm_control_py_fresh/urdf/00-arm-rs_asm-v3"
    spec = mujoco.MjSpec.from_file(str(robot_dir / "urdf/00-arm-rs_asm-v3.urdf"))
    spec.meshdir = str(robot_dir)
    spec.compiler.fusestatic = False
    spec.option.timestep = .002
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    spec.option.iterations = 80
    spec.option.gravity = [0, 0, -9.81]
    spec.visual.global_.offwidth = 640
    spec.visual.global_.offheight = 480
    # Ideal model-based robot gravity compensation, NOT object compensation.
    # Gains/friction below are explicit simulation assumptions, not measured fits.
    for body in spec.bodies:
        if body.name != "world":
            body.gravcomp = 1
    for geom in spec.geoms:
        geom.friction = [0.8, .005, .0001]
        geom.rgba = [.48, .53, .58, 1]
    for joint in spec.joints:
        jaw = joint.name.startswith("joint_")
        joint.damping = [2 if jaw else .5, 0, 0]
        joint.armature = .002 if jaw else .02
        kp, kd, force = (1800, 30, 12) if jaw else (150, 12, 12)
        spec.add_actuator(name=joint.name, target=joint.name,
                          trntype=mujoco.mjtTrn.mjTRN_JOINT,
                          gaintype=mujoco.mjtGain.mjGAIN_FIXED, gainprm=[kp]+[0]*9,
                          biastype=mujoco.mjtBias.mjBIAS_AFFINE, biasprm=[0, -kp, -kd]+[0]*7,
                          ctrllimited=True, ctrlrange=joint.range,
                          forcelimited=True, forcerange=[-force, force])
    # Match the planner's adjacent-body exclusions, not scene exclusions.
    for body in spec.bodies:
        if body.name != "world" and body.parent.name != "world":
            spec.add_exclude(bodyname1=body.parent.name, bodyname2=body.name)
    spec.add_exclude(bodyname1="gripper_left", bodyname2="gripper_right")
    spec.worldbody.add_light(pos=[.3, -.3, 1.3], dir=[0, 0, -1], diffuse=[.8]*3)
    spec.worldbody.add_light(pos=[-.4, .5, .9], dir=[.3, -.2, -1], diffuse=[.5]*3)
    calibration = Path(os.environ.get("REBOT_WRIST_CALIB", ROOT / "calib"))
    table_z = float(json.loads((calibration / "wrist_safety.json").read_text())["table_z_m"])
    spec.worldbody.add_geom(name="table", type=mujoco.mjtGeom.mjGEOM_BOX,
                            pos=[.42, 0, table_z - .025], size=[.26, .32, .025],
                            rgba=[.55, .48, .36, 1], friction=[.8, .005, .0001])
    rng = np.random.default_rng(seed)
    jitter = rng.uniform(-.012, .012, 4)
    obj_xy = np.array([.32, .025]) + jitter[:2]
    cup_xy = np.array([.38, -.095]) + jitter[2:]
    # A capped marker: a 32 mm flat base lets it stand under gravity, while the
    # 24 mm body is still well within the 50 mm physical jaw aperture.
    radius, half_length = (.014, 0) if object_kind == "ball" else (.012, .025)
    initial_height = radius if object_kind == "ball" else half_length
    obj = spec.worldbody.add_body(name="target", pos=[*obj_xy, table_z + initial_height + .001])
    obj.add_freejoint(name="target_free")
    obj.add_geom(name="target_geom", type=(mujoco.mjtGeom.mjGEOM_SPHERE if object_kind == "ball"
                                          else mujoco.mjtGeom.mjGEOM_CYLINDER),
                 size=[radius, half_length, 0],
                 mass=.020, rgba=[.03, .16, .95, 1], friction=[.65, .005, .0001],
                 condim=4)
    if object_kind == "marker":
        obj.add_geom(name="target_flat_cap", type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                     pos=[0, 0, -half_length + .003], size=[.016, .003, 0],
                     mass=.004, rgba=[.025, .11, .68, 1], friction=[.7, .005, .0001],
                     condim=4)
    inner, wall, height, bottom = .045, .004, .065, .005
    cup = spec.worldbody.add_body(name="cup", pos=[*cup_xy, table_z + .002])
    cup.add_freejoint(name="cup_free")
    cup.add_geom(name="cup_bottom", type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                 size=[inner+wall, bottom/2, 0], pos=[0, 0, bottom/2],
                 mass=.04, rgba=[.85, .035, .025, 1], friction=[.7, .005, .0001])
    for i in range(32):
        angle = i * 2 * math.pi / 32
        cup.add_geom(name=f"cup_wall_{i}", type=mujoco.mjtGeom.mjGEOM_BOX,
                     pos=[(inner+wall/2)*math.cos(angle), (inner+wall/2)*math.sin(angle), height/2],
                     quat=[math.cos(angle/2), 0, 0, math.sin(angle/2)],
                     size=[wall/2, (inner+wall)*math.tan(math.pi/32), height/2],
                     mass=.003, rgba=[.85, .035, .025, 1], friction=[.7, .005, .0001])
    spec.worldbody.add_geom(name="obstacle", type=mujoco.mjtGeom.mjGEOM_BOX,
                            pos=[.42, .10, table_z+.035], size=[.025, .025, .035],
                            rgba=[.25, .55, .22, 1])
    # Optical transform is calibrated robot geometry, never an object pose.
    rotation = np.array(json.loads((calibration / "wrist_camera_rotation.json").read_text())[
        "base_to_camera_rotation"])
    u, _, vt = np.linalg.svd(rotation)
    # Saved matrix is the apparent-object-motion Jacobian / 1000 = -R_camera_base.
    # Its determinant is -1; treating it as a rotation points the camera backward.
    camera_to_tool = -(u @ vt).T
    tcp_camera = np.array(json.loads((calibration / "wrist_tcp.json").read_text())[
        "tcp_camera_xyz_mm"]) / 1000
    mj_camera_rotation = camera_to_tool @ np.diag([1, -1, -1])
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, mj_camera_rotation.ravel())
    spec.body("gripper_end").add_camera(name="wrist", pos=-camera_to_tool @ tcp_camera,
                                        quat=quat, fovy=SIM_CAMERA_FOVY_DEG)
    model = spec.compile()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    meta = dict(object_kind=object_kind, seed=seed, radius=radius, half_length=half_length,
                cup_inner_radius=inner, cup_height=height, cup_bottom=bottom,
                table_z=table_z, object_initial_xyz=data.body("target").xpos.tolist(),
                calibration=str(calibration), urdf=str(robot_dir / "urdf/00-arm-rs_asm-v3.urdf"))
    return model, data, meta, spec.to_xml()


def evaluate_sample(model, data, meta):
    """Privileged evaluator. Never imported/called by the control policy."""
    cup, obj = data.body("cup"), data.body("target")
    cup_r = cup.xmat.reshape(3, 3)
    relative = cup_r.T @ (obj.xpos - cup.xpos)
    axis_world = obj.xmat.reshape(3, 3) @ np.array([0., 0, 1.])
    axis = cup_r.T @ axis_world
    ends = np.array([relative - meta["half_length"]*axis,
                     relative + meta["half_length"]*axis])
    radial = float(np.max(np.linalg.norm(ends[:, :2], axis=1)) + meta["radius"])
    zlo, zhi = float(ends[:, 2].min()-meta["radius"]), float(ends[:, 2].max()+meta["radius"])
    object_id = model.body("target").id
    jaw_ids = {model.body(n).id for n in ("gripper_left", "gripper_right", "gripper_end")}
    touching_gripper = any(
        object_id in (int(model.geom_bodyid[c.geom1]), int(model.geom_bodyid[c.geom2]))
        and bool(jaw_ids.intersection((int(model.geom_bodyid[c.geom1]), int(model.geom_bodyid[c.geom2]))))
        for c in data.contact)
    dof = int(model.jnt_dofadr[model.joint("target_free").id])
    speed = float(np.linalg.norm(data.qvel[dof:dof+3]))
    inside = (radial < meta["cup_inner_radius"]-.001 and zlo >= meta["cup_bottom"]-.002
              and zhi < meta["cup_height"] and cup_r[2, 2] > .98)
    # Supported task: robot-left of cup, nominal 120 mm centre spacing. Existing
    # planner may expand same-side clearance to 180 mm; never accept opposite side.
    delta = obj.xpos - cup.xpos
    vertical_extent = (meta["radius"] if meta["object_kind"] == "ball" else
                       meta["half_length"]*abs(axis_world[2])
                       + meta["radius"]*math.sqrt(max(0., 1-axis_world[2]**2)))
    on_table = abs(float(obj.xpos[2])-vertical_extent-meta["table_z"]) < .01
    left_of_cup = (abs(float(delta[0])) < .04 and .085 < float(delta[1]) < .205 and on_table)
    return dict(inside=bool(inside), left_of_cup=bool(left_of_cup), released=not touching_gripper, speed_m_s=speed,
                settled=bool(speed < .015), radial_extent_m=radial, z_extent_m=[zlo, zhi],
                cup_upright=bool(cup_r[2, 2] > .98), object_xyz=obj.xpos.tolist())


class PhysicsWorld:
    """Private simulator state. Public action API contains actuator targets only."""
    def __init__(self, kind="ball", seed=0):
        self.model, self.data, self.meta, self.scene_xml = build_scene(kind, seed)
        self.lock = threading.RLock()
        self.stopped = threading.Event()
        self.thread = None
        self.history = []
        self.commands = []
        self.failure = None
        self.arm_jids = [self.model.joint(f"joint{i}").id for i in range(1, 7)]
        self.qadr = self.model.jnt_qposadr[self.arm_jids]
        self.vadr = self.model.jnt_dofadr[self.arm_jids]
        self.jaw_jids = [self.model.joint(n).id for n in ("joint_left", "joint_right")]
        self.jaw_qadr = self.model.jnt_qposadr[self.jaw_jids]
        self.arm_act = [self.model.actuator(f"joint{i}").id for i in range(1, 7)]
        self.jaw_act = [self.model.actuator(n).id for n in ("joint_left", "joint_right")]
        self.renderer = None
        self.target_source = None
        self.rng = np.random.default_rng(seed + 10000)

    def step(self, count=1):
        with self.lock:
            if self.target_source is not None:
                target = np.asarray(self.target_source(), float).copy()
                if not np.array_equal(target, self.data.ctrl[self.arm_act]):
                    self.command_arm(target)
            for _ in range(count):
                mujoco.mj_step(self.model, self.data)
                if not np.isfinite(self.data.qpos).all():
                    raise RuntimeError("Non-finite physics state")
            self.history.append((float(self.data.time), self.data.qpos.copy(), self.data.qvel.copy()))

    def start(self):
        if self.thread is not None and self.thread.is_alive():
            return
        def loop():
            try:
                while not self.stopped.is_set():
                    began = time.monotonic()
                    self.step(10)
                    self.stopped.wait(max(0, .02-(time.monotonic()-began)))
            except BaseException as exc:
                self.failure = str(exc)
                self.stopped.set()
        self.thread = threading.Thread(target=loop, daemon=True, name="mujoco-physics")
        self.thread.start()

    def watch(self) -> None:
        """Open MuJoCo's interactive viewer over the same live physics data."""
        try:
            from mujoco import viewer
            self.start()
            with viewer.launch_passive(self.model, self.data) as live:
                while live.is_running() and not self.stopped.is_set():
                    with self.lock:
                        live.sync()
                    time.sleep(0.02)
        except Exception as exc:
            raise RuntimeError(
                "Could not open the MuJoCo viewer. Run from a graphical desktop "
                "with DISPLAY set (for example DISPLAY=:0), not a headless shell."
            ) from exc

    def serve_web_viewer(self) -> tuple[str, callable]:
        """Serve live MuJoCo-rendered frames without the native GLFW viewer."""
        import cv2
        world = self
        image_lock = threading.Lock()
        latest = {"jpeg": b""}
        rendering = threading.Event()
        rendering.set()

        def render_loop():
            # EGL contexts are thread-affine: create, use, and close this one
            # exclusively from the producer thread. HTTP threads only read bytes.
            renderer = mujoco.Renderer(world.model, 480, 640)
            camera = mujoco.MjvCamera()
            camera.lookat[:] = [.30, 0, .10]
            camera.distance, camera.azimuth, camera.elevation = .82, 135, -32
            try:
                while rendering.is_set() and not world.stopped.is_set():
                    with world.lock:
                        renderer.update_scene(world.data, camera)
                        frame = renderer.render().copy()
                    ok, encoded = cv2.imencode(".jpg", frame[:, :, ::-1],
                                               [cv2.IMWRITE_JPEG_QUALITY, 85])
                    if ok:
                        with image_lock:
                            latest["jpeg"] = encoded.tobytes()
                    time.sleep(.08)
            finally:
                renderer.close()

        render_thread = threading.Thread(target=render_loop, daemon=True, name="mujoco-web-render")
        render_thread.start()

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, _format, *_args):
                pass

            def do_GET(self):
                if self.path == "/":
                    page = b"""<!doctype html><title>MuJoCo hybrid physics</title>
<style>body{margin:0;background:#111;color:#ddd;font:16px sans-serif;text-align:center}img{max-width:100vw;max-height:92vh}</style>
<p>Live MuJoCo physics: upright blue marker, red cup, green obstacle</p><img id=f src=/frame.jpg>
<script>setInterval(()=>f.src='/frame.jpg?'+Date.now(),80)</script>"""
                    self.send_response(200); self.send_header("Content-Type", "text/html")
                    self.send_header("Content-Length", str(len(page))); self.end_headers(); self.wfile.write(page)
                    return
                if not self.path.startswith("/frame.jpg"):
                    self.send_error(404); return
                with image_lock:
                    payload = latest["jpeg"]
                if not payload:
                    self.send_error(503); return
                self.send_response(200); self.send_header("Content-Type", "image/jpeg")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(payload))); self.end_headers()
                try:
                    self.wfile.write(payload)
                except (BrokenPipeError, ConnectionResetError):
                    # Browser frame refreshes commonly cancel stale requests.
                    pass

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True, name="mujoco-web-viewer")
        thread.start()
        url = f"http://127.0.0.1:{server.server_port}"

        def close():
            rendering.clear(); server.shutdown(); server.server_close()
            thread.join(2); render_thread.join(2)
        return url, close

    def close(self):
        self.stopped.set()
        if self.thread:
            self.thread.join(3)
        if self.renderer:
            self.renderer.close()

    def command_arm(self, q):
        q = np.asarray(q, float)
        if q.shape != (6,) or not np.isfinite(q).all():
            raise ValueError("Expected six finite joint targets")
        with self.lock:
            lohi = self.model.jnt_range[self.arm_jids]
            if np.any(q < lohi[:, 0]-1e-6) or np.any(q > lohi[:, 1]+1e-6):
                raise ValueError("Joint target outside URDF limits")
            self.data.ctrl[self.arm_act] = q
            self.commands.append({"t": float(self.data.time), "joint_targets": q.tolist()})

    def command_gripper(self, angle):
        from . import gripper_model as gm
        aperture = gm.aperture_mm(float(angle))
        if aperture is None:
            raise ValueError("No gripper aperture calibration")
        with self.lock:
            # URDF's independent prismatic jaws approximate the physical linkage.
            self.data.ctrl[self.jaw_act] = np.clip(aperture/2000, 0, .05)
            self.commands.append({"t": float(self.data.time), "jaw_target_m": aperture/2000})

    def encoders(self):
        from . import gripper_model as gm
        with self.lock:
            measured_span = float(np.sum(self.data.qpos[self.jaw_qadr])*1000)
            angle = gm.rad_for_aperture(max(0., measured_span))
            q = np.r_[self.data.qpos[self.qadr], angle if angle is not None else 0.]
            v = np.r_[self.data.qvel[self.vadr], 0.]
            # Actual generalized actuator effort; not a fabricated grasp flag.
            effort = np.r_[self.data.qfrc_actuator[self.vadr], 0.]
            return q.copy(), v.copy(), effort.copy()

    def frame(self):
        if self.renderer is None:
            self.renderer = mujoco.Renderer(self.model, 480, 640)
        with self.lock:
            self.renderer.disable_depth_rendering()
            self.renderer.update_scene(self.data, camera="wrist")
            rgb = self.renderer.render().copy()
            self.renderer.enable_depth_rendering()
            self.renderer.update_scene(self.data, camera="wrist")
            depth = self.renderer.render().copy()*1000
            self.renderer.disable_depth_rendering()
        depth += self.rng.normal(0, 1.5, depth.shape)
        depth[(depth < 60) | (depth > 600) | (self.rng.random(depth.shape) < .01)] = 0
        focal = 240 / math.tan(math.radians(SIM_CAMERA_FOVY_DEG/2))
        return rgb[:, :, ::-1].copy(), depth.astype(np.float32), (focal, focal, 319.5, 239.5)

    def verdict(self):
        """Check one full second after release, not a single lucky frame."""
        self.step(500)
        samples = []
        for _ in range(50):
            self.step(10)
            samples.append(evaluate_sample(self.model, self.data, self.meta))
        lifted = any(q[self.model.jnt_qposadr[self.model.joint("target_free").id]+2]
                     > self.meta["object_initial_xyz"][2]+.025 for _, q, _ in self.history)
        displaced = float(np.linalg.norm(np.asarray(samples[-1]["object_xyz"])[:2]
                                         - np.asarray(self.meta["object_initial_xyz"])[:2])) > .04
        return {"success": bool(lifted and displaced and all(s["left_of_cup"] and s["released"] and s["settled"]
                                               for s in samples) and self.failure is None),
                "lift_observed": bool(lifted), "displaced_over_40mm": displaced, "stable_window_s": 1.,
                "final": samples[-1], "physics_error": self.failure}

    def export(self, output):
        """Video is a replay of recorded physics, never input to the evaluator."""
        import imageio.v2 as imageio
        output.mkdir(parents=True, exist_ok=True)
        (output / "scene.xml").write_text(self.scene_xml)
        # XML refers to source mesh files; absolute meshdir is retained.
        np.savez_compressed(output / "physics_trace.npz",
                            time=[h[0] for h in self.history],
                            qpos=[h[1] for h in self.history], qvel=[h[2] for h in self.history])
        (output / "commands.json").write_text(json.dumps(self.commands))
        replay = mujoco.MjData(self.model)
        renderer = mujoco.Renderer(self.model, 480, 640)
        camera = mujoco.MjvCamera()
        camera.lookat[:] = [.25, 0, .10]
        camera.distance, camera.azimuth, camera.elevation = .85, 135, -35
        stride = max(1, len(self.history)//120)
        frames = []
        for _, q, _ in self.history[::stride]:
            replay.qpos[:] = q  # Replay-only data, never self.data.
            mujoco.mj_forward(self.model, replay)
            renderer.update_scene(replay, camera)
            frames.append(renderer.render().copy())
        if frames:
            imageio.mimsave(output / "physics_replay.gif", frames, duration=.10, loop=0)
            imageio.imwrite(output / "scene.png", frames[0])
        renderer.close()


def simulated_backend(world):
    """Reuse LocalArm IK/path gates/senders; replace only CAN driver/controller."""
    from . import local_arm as la, wrist_servo as ws, arm_safe

    class Driver:
        def get_state(self):
            return world.encoders()

    class Controller:
        _send_thread = None
        _moving = False
        _gripper_group = None
        _qd_target = np.zeros(6)
        _stop_send = threading.Event()
        _target = world.encoders()[0][:6].copy()

        @property
        def _q_target(self):
            return self._target

        @_q_target.setter
        def _q_target(self, q):
            self._target = np.asarray(q, float).copy()
            world.command_arm(q)

        def set_gripper_target(self, angle):
            world.command_gripper(angle)

    class SimLocalArm(la.LocalArm):
        def connect(self):
            if self._arm is None:
                self._arm, self._controller = Driver(), Controller()

        def disconnect(self, **_kwargs):
            # No physical shutdown handler, no CAN object ever constructed.
            return {"ok": True, "simulated": True}

    local = SimLocalArm()
    local.connect()
    world.target_source = lambda: local._controller._q_target
    # Prevent ANY accidentally missed local or service hardware route.
    def refuse_hardware(*_args, **_kwargs):
        raise RuntimeError("Physical hardware/service access prohibited in physics benchmark")
    la.RebotArm = refuse_hardware
    la.get_local_arm = lambda: local
    la._local = local
    arm_safe.BACKEND = "local"
    camera = SimpleNamespace(frame=world.frame, close=lambda: None)
    ws.install_backend(arm=arm_safe, camera_factory=lambda _kind: camera, settle_s=.25)

    class Backend:
        def execute(self, task, routine, dry):
            return ws.fixed_pick_and_place(task.object, dry=dry, verbose=True,
                                           place_reference=task.reference,
                                           place_relation=task.relation,
                                           place_distance_m=task.distance_m)

        def freeze(self):
            return local.freeze()

    return Backend()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--object", choices=["ball", "marker"], default="marker")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mode", choices=["scene", "hybrid"], default="scene")
    ap.add_argument("--perception", choices=["rgbd", "sam3"], default="rgbd",
                    help="RGB-D colour/geometry fallback or the installed SAM 3.1 model")
    ap.add_argument("--timeout", type=float, default=120)
    ap.add_argument("--output", type=Path)
    ap.add_argument("--viewer", action="store_true",
                    help="open a live interactive MuJoCo window; close it to finish")
    ap.add_argument("--web-viewer", action="store_true",
                    help="serve live MuJoCo frames in a local browser (works without GLFW)")
    args = ap.parse_args()
    if args.timeout <= 0 or not math.isfinite(args.timeout):
        ap.error("timeout must be positive and finite")
    output = args.output or ROOT / "outputs/hybrid_physics" / f"{args.object}_{args.seed}_{time.time_ns()}"
    if output.exists():
        ap.error("output already exists; choose a new directory to preserve previous evidence")
    output = output.resolve()
    mount = simulation_calibration(output / "calibration")
    os.environ["REBOT_WRIST_CALIB"] = str(output / "calibration")
    # Both choices use only rendered RGB-D. ``rgbd`` exercises the production
    # distinctive-colour physical-component fallback, avoiding a model download
    # during a repeatable contact benchmark. ``sam3`` keeps the normal profile.
    if args.perception == "rgbd":
        os.environ["REBOT_OBJECT_GEOMETRY_MODE"] = "legacy"
        os.environ["REBOT_VISION_PRIMARY"] = "0"
        os.environ["REBOT_ENABLE_LOCATE_ANYTHING"] = "0"
    # The generated scene has exactly one red reference. This explicitly opts
    # into the wrist pipeline's RGB-D fallback after a semantic cup miss.
    os.environ["REBOT_SCENE_REFERENCE_COLOR_FALLBACK"] = "1"
    from .hybrid_profile import configure
    configure()
    from .hybrid_mission import MissionRunner, Task, load_routines
    world = PhysicsWorld(args.object, args.seed)
    close_web_viewer = None
    outcome = None
    report = {"task": f"{args.object}_left_of_cup", "mode": args.mode,
              "perception": args.perception, "scene": world.meta,
              "controller_relation": "left", "distance_m": .12,
              "oracle_policy_inputs": False, "scripted_attachment": False,
              "reference_colour_fallback": True,
              "physical_validation": False,
              "limitations": ["URDF convex collision meshes and prismatic jaw approximation",
                              "Unidentified actuator/friction dynamics; ideal robot gravity compensation",
                              "Explicit virtual wrist mount; exact model calibration, not measured hardware calibration"],
              "virtual_camera_mount_xyz_m": mount}
    try:
        # Let gravity settle the reset, with all robot targets at their home pose.
        world.step(500)
        output.mkdir(parents=True, exist_ok=True)
        # Do not create an off-screen EGL/GL renderer before an interactive GLFW
        # viewer: mixed contexts crash on several NVIDIA desktop configurations.
        if args.mode == "scene" and args.viewer:
            world.watch()
        if args.web_viewer:
            url, close_web_viewer = world.serve_web_viewer()
            print(f"LIVE_MUJOCO_WEB_VIEWER={url}", flush=True)
            world.start()
            if args.mode == "scene":
                # A scene viewer is intentionally persistent. Ctrl+C ends the
                # server and writes its trace/replay evidence.
                while not world.stopped.wait(.25):
                    pass
        color, depth, intr = world.frame()
        import imageio.v2 as imageio
        imageio.imwrite(output / "wrist_initial.png", color[:, :, ::-1])
        np.savez_compressed(output / "wrist_initial_rgbd.npz", bgr=color, depth_mm=depth, intr=intr)
        if args.mode == "hybrid":
            backend = simulated_backend(world)
            runner = MissionRunner(load_routines(), backend=backend)
            world.start()
            def timeout():
                active = runner.status().get("mission_id")
                if active:
                    runner.cancel(active)
            timer = threading.Timer(args.timeout, timeout)
            timer.start()
            task = Task(f"blue {args.object}", "scene_place", "left", "red cup", .12)
            try:
                if args.viewer:
                    holder = {}
                    worker = threading.Thread(
                        target=lambda: holder.setdefault("outcome", runner.start(task, dry=False)),
                        daemon=False, name="hybrid-physics-mission",
                    )
                    worker.start()
                    world.watch()
                    if worker.is_alive():
                        active = runner.status().get("mission_id")
                        if active:
                            runner.cancel(active)
                    worker.join(timeout=15)
                    outcome = holder.get("outcome", runner.status())
                else:
                    outcome = runner.start(task, dry=False)
            finally:
                timer.cancel()
                world.stopped.set()
                world.thread.join(3)
        report["mission"] = outcome
        report["evaluation"] = world.verdict()
        report["hybrid_verified"] = bool(args.mode == "hybrid" and outcome and outcome.get("ok")
                                          and report["evaluation"]["success"])
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        report["hybrid_verified"] = False
    finally:
        world.stopped.set()
        if world.thread:
            world.thread.join(3)
        output.mkdir(parents=True, exist_ok=True)
        (output / "report.json").write_text(json.dumps(report, indent=2, default=str))
        try:
            world.export(output)
        finally:
            if close_web_viewer:
                close_web_viewer()
            world.close()
    print(json.dumps({"output": str(output), **report}, indent=2, default=str))
    return 0 if args.mode == "scene" or report.get("hybrid_verified") else 2


if __name__ == "__main__":
    raise SystemExit(main())
