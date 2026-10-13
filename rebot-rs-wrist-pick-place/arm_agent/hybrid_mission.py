"""Wrist-only mission recipes, stage telemetry and persistent MCP execution.

`--plan` is offline. `--dry` reads the camera but sends no motor commands.
`--run` explicitly enables the existing wrist manipulation pipeline.
"""
from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import sys
import threading
import time
from typing import Any
import uuid

from . import mission_control as control

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RECIPES = ROOT / "config" / "hybrid_missions.json"
RELATIONS = {"drop_zone", "left", "right", "front", "behind", "next_to", "stack"}


@dataclass(frozen=True)
class Routine:
    name: str
    description: str
    relations: tuple[str, ...]
    carry_via_m: tuple[tuple[float, float, float], ...] = ()
    calibration_id: str | None = None


def load_routines(path: Path = DEFAULT_RECIPES) -> dict[str, Routine]:
    data = json.loads(Path(path).read_text())
    if data.get("version") != 1 or not isinstance(data.get("routines"), dict):
        raise ValueError("Expected version 1 and a routines object")
    routines = {}
    for name, spec in data["routines"].items():
        if set(spec) - {"description", "relations", "carry_via_m", "calibration_id"}:
            raise ValueError(f"Unknown routine fields: {name}")
        relations = spec.get("relations", [])
        if not isinstance(relations, list) or not relations or not all(r in RELATIONS for r in relations):
            raise ValueError(f"Invalid relations for {name}")
        points = spec.get("carry_via_m", [])
        if not isinstance(points, list) or len(points) > 8:
            raise ValueError("At most eight carry waypoints are supported")
        for point in points:
            if (not isinstance(point, list) or len(point) != 3 or
                any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in point)):
                raise ValueError("Carry waypoints must be finite base-frame XYZ metres")
        if points and not spec.get("calibration_id"):
            raise ValueError("Taught carry waypoints require a calibration_id from --plan")
        routines[name] = Routine(name, str(spec.get("description", "")), tuple(relations),
                                 tuple(tuple(float(v) for v in p) for p in points),
                                 spec.get("calibration_id"))
    return routines


@dataclass(frozen=True)
class Task:
    object: str
    routine: str = "taught_drop"
    relation: str = "drop_zone"
    reference: str | None = None
    distance_m: float | None = None

    def validate(self, routines: dict[str, Routine]) -> Routine:
        if not isinstance(self.object, str) or not self.object.strip():
            raise ValueError("A specific object description is required")
        routine = routines.get(self.routine)
        if routine is None or self.relation not in routine.relations:
            raise ValueError("Unknown routine or unsupported placement relation")
        if self.relation != "drop_zone" and (not isinstance(self.reference, str) or not self.reference.strip()):
            raise ValueError("Scene placement requires a reference object")
        if self.relation == "drop_zone" and (self.reference is not None or self.distance_m is not None):
            raise ValueError("Taught drop does not accept a scene reference or distance")
        if self.relation == "stack" and self.distance_m is not None:
            raise ValueError("Stack uses the reference centre, not a lateral distance")
        if self.distance_m is not None and (
            isinstance(self.distance_m, bool) or not math.isfinite(self.distance_m)
            or not 0.06 <= self.distance_m <= 0.18
        ):
            raise ValueError("distance_m must be between 0.06 and 0.18 metres")
        return routine


def calibration_report(task: Task, routine: Routine) -> dict[str, Any]:
    """Inspect artifacts without importing a robot/camera driver."""
    from .hybrid_profile import DEFAULTS
    directory = Path(os.environ.get("REBOT_WRIST_CALIB", ROOT / "calib"))
    required = {
        "wrist_tcp.json": "tcp_camera_xyz_mm",
        "wrist_grasp_target.json": "object_camera_xyz_mm",
        "wrist_camera_rotation.json": "base_to_camera_rotation",
        "wrist_gripper.json": "mm_per_rad",
        "wrist_safety.json": "table_z_m",
    }
    if task.relation == "drop_zone":
        required["wrist_drop_zone.json"] = "release_xyz_m"
    errors, files = [], {}
    digest = hashlib.sha256()
    # Include all calibration artifacts so one ID serves both built-in recipes.
    for filename in sorted(set(required) | {"wrist_drop_zone.json"}):
        path = directory / filename
        try:
            raw = path.read_bytes()
            data = json.loads(raw)
            if not isinstance(data, dict):
                raise ValueError("Expected a calibration object")
            digest.update(filename.encode() + raw)
            files[filename] = str(path)
            key = required.get(filename)
            if key and key not in data:
                errors.append(f"{filename}: missing {key}")
            elif key:
                value = data[key]
                flat = ([item for row in value for item in row]
                        if key == "base_to_camera_rotation" and isinstance(value, list)
                        and len(value) == 3 and all(isinstance(row, list) and len(row) == 3 for row in value)
                        else value)
                expected = 9 if key == "base_to_camera_rotation" else 3
                if key in {"table_z_m", "mm_per_rad"}:
                    flat = [value]
                    expected = 1
                if (not isinstance(flat, list) or len(flat) != expected or
                    any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in flat)):
                    errors.append(f"{filename}: malformed or non-finite {key}")
                elif key == "mm_per_rad" and value <= 0:
                    errors.append(f"{filename}: gripper scale must be positive")
            if data.get("camera", "realsense") != "realsense":
                errors.append(f"{filename}: not a wrist RealSense calibration")
        except (OSError, ValueError, TypeError) as exc:
            if filename in required:
                errors.append(f"{filename}: {exc}")
    effective = {key: os.environ.get(key, value) for key, value in DEFAULTS.items()}
    # Capture the serial too; changing camera selection invalidates taught routes.
    effective["REBOT_REALSENSE_SERIAL"] = os.environ.get("REBOT_REALSENSE_SERIAL", "unspecified")
    # Bind additional overrides without exposing potentially sensitive values in
    # the report. Normalize defaults so configure() does not change the identity.
    identity_settings = {**{key: value for key, value in os.environ.items()
                           if key.startswith("REBOT_")}, **effective}
    digest.update(json.dumps(identity_settings, sort_keys=True).encode())
    calibration_id = digest.hexdigest()
    if routine.calibration_id and routine.calibration_id != calibration_id:
        errors.append("Taught route calibration/profile changed; revalidate the route")
    for point in routine.carry_via_m:
        for axis, value in zip("XYZ", point):
            if not float(effective[f"REBOT_WS_{axis}_MIN"]) <= value <= float(effective[f"REBOT_WS_{axis}_MAX"]):
                errors.append(f"Carry waypoint {point} outside configured workspace")
    return {"ok": not errors, "errors": errors, "files": files,
            "calibration_id": calibration_id, "profile": effective,
            "physical_calibration_verified": False}


class WristBackend:
    """Adapter keeps the combined pick/place path and its pre-pick scene plan."""
    def __init__(self) -> None:
        import fcntl
        # Lifetime lock also covers an interrupted, unparked controller. Legacy
        # launchers do not take this lock and must not run concurrently.
        (ROOT / "logs").mkdir(exist_ok=True)
        self._lockfile = open(ROOT / "logs" / "hybrid_controller.lock", "a+")
        try:
            fcntl.flock(self._lockfile, fcntl.LOCK_EX | fcntl.LOCK_NB)
            from .hybrid_profile import configure
            configure()
            from . import wrist_servo
            self.ws = wrist_servo
            if os.getenv("REBOT_HYBRID_LIVE", "").strip().lower() in {
                    "1", "true", "yes", "on"}:
                # Configure visualization only after the hybrid profile has
                # selected the wrist backend.  The stream is diagnostic-only;
                # it never supplies a motion target.
                self.ws.set_viz(self.ws.make_viz(
                    mjpeg_port=8731, mjpeg_auto_open=True
                ))
                print("[hybrid] LIVE wrist feed -> http://127.0.0.1:8731/",
                      flush=True)
        except BaseException:
            self._lockfile.close()
            raise
        self.calibration_id: str | None = None

    def execute(self, task: Task, routine: Routine, dry: bool) -> dict[str, Any]:
        if not dry and self.ws._OBJECT_HELD is not False:
            raise RuntimeError("Held-object state is not empty; operator recovery is required")
        return self.ws.fixed_pick_and_place(
            task.object, dry=dry, verbose=False, place_reference=task.reference,
            place_relation=task.relation, place_distance_m=task.distance_m,
        )

    def freeze(self) -> dict[str, Any]:
        from . import local_arm
        if local_arm._local is None or not local_arm._local.connected:
            return {"ok": True, "connected": False, "reason": "no hardware connection"}
        with control.recovery():
            return local_arm._local.freeze()


class MissionRunner:
    """Single task owner; no detached stage can continue after a cancellation."""
    def __init__(self, routines: dict[str, Routine], backend: Any = None) -> None:
        self.routines, self.backend = routines, backend
        self._lock = threading.RLock()
        self._control: control.MissionControl | None = None
        self._state: dict[str, Any] = {"status": "idle", "events": [], "needs_operator": False}

    def status(self) -> dict[str, Any]:
        with self._lock:
            return copy.deepcopy(self._state)

    def cancel(self, mission_id: str) -> dict[str, Any]:
        with self._lock:
            if self._state.get("mission_id") != mission_id or self._control is None:
                return {"ok": False, "reason": "No matching active mission"}
            self._control.cancel()
            self._state["stop_requested"] = True
            return {"ok": True, "mission_id": mission_id, "status": "stop_requested"}

    def _event(self, name: str, detail: dict[str, Any]) -> None:
        with self._lock:
            self._state["stage"] = name
            self._state["events"].append({"stage": name, "detail": detail,
                                          "elapsed_s": round(time.monotonic() - self._began, 3)})
            # Retain a bounded history in a persistent controller process.
            self._state["events"] = self._state["events"][-200:]

    def start(self, task: Task, *, dry: bool = True, background: bool = False) -> dict[str, Any]:
        routine = task.validate(self.routines)
        with self._lock:
            if self._control is not None:
                return {"ok": False, "reason": "Robot busy", "mission_id": self._state["mission_id"]}
            if self._state.get("needs_operator"):
                return {"ok": False, "reason": "Previous task requires physical recovery; new motion refused"}
            self._began = time.monotonic()
            self._control = control.MissionControl(dry_run=dry, carry_via_m=routine.carry_via_m, notify=self._event)
            self._state = {"status": "running", "stage": "queued", "mission_id": uuid.uuid4().hex,
                           "dry_run": dry, "object": task.object, "routine": routine.name,
                           "relation": task.relation, "reference": task.reference,
                           "events": [], "needs_operator": False, "stop_requested": False}
            receipt = {"ok": True, "mission_id": self._state["mission_id"], "status": "running"}
        if background:
            # Non-daemon: exiting the client must not silently kill an active task.
            threading.Thread(target=self._run, args=(task, routine, dry), daemon=False,
                             name="hybrid-mission").start()
            return receipt
        self._run(task, routine, dry)
        return self.status()

    def _run(self, task: Task, routine: Routine, dry: bool) -> None:
        active = self._control
        assert active is not None
        result: dict[str, Any] = {}
        status = "failed"
        try:
            with control.activate(active):
                control.stage("preflight")
                if self.backend is None or isinstance(self.backend, WristBackend):
                    report = calibration_report(task, routine)
                    if not report["ok"]:
                        raise ValueError("; ".join(report["errors"]))
                    if self.backend is None:
                        self.backend = WristBackend()
                        self.backend.calibration_id = report["calibration_id"]
                    elif self.backend.calibration_id != report["calibration_id"]:
                        raise ValueError("Calibration/profile changed in a running controller; restart and revalidate")
                control.checkpoint()
                result = self.backend.execute(task, routine, dry)
                control.checkpoint()
                if dry:
                    status = "planned" if result.get("ok") else "failed"
                elif (result.get("ok") and result.get("verified") is True
                      and result.get("homed") is True and result.get("object_held") is False):
                    status = "succeeded"
                else:
                    status = "failed"
        except (control.MissionCancelled, KeyboardInterrupt) as exc:
            active.cancel(str(exc) or "operator interrupted")
            status = "cancelled"
            result = {**result, "ok": False, "reason": active.reason}
        except Exception as exc:
            result = {"ok": False, "reason": f"{type(exc).__name__}: {exc}"}
        finally:
            if active.stop.is_set():
                status = "cancelled"
            if status in {"failed", "cancelled"} and not dry and self.backend is not None:
                try:
                    result["freeze"] = self.backend.freeze()
                except BaseException as exc:
                    result["freeze"] = {"ok": False, "reason": str(exc)}
            # No automatic restart or outer retry: wrist already owns bounded
            # grasp recovery. An uncertain physical state blocks the next task.
            needs_operator = bool(not dry and self.backend is not None and status != "succeeded"
                                  and not (result.get("homed") is True and result.get("object_held") is False)
                                  and result.get("freeze", {}).get("connected") is not False)
            with self._lock:
                self._state.update(status=status, ok=status in {"succeeded", "planned"},
                                   result=result, needs_operator=needs_operator,
                                   placement_verified=False,
                                   release_verified=bool(not dry and result.get("verified") is True
                                                         and result.get("object_held") is False))
                self._event(status, {"reason": result.get("reason", "")})
                self._control = None


def serve_mcp(runner: MissionRunner, *, allow_motion: bool) -> None:
    from mcp.server.fastmcp import FastMCP
    mcp = FastMCP("rebot-hybrid-wrist")

    @mcp.tool()
    def hybrid_start(object: str, routine: str = "taught_drop", relation: str = "drop_zone",
                     reference: str | None = None, distance_m: float | None = None,
                     dry_run: bool = True) -> dict:
        """Start one wrist-only mission; poll hybrid_status. +X front, +Y robot-left.
        taught_drop uses the saved zone; scene_place needs a reference. A live
        start requires the server's --allow-motion flag as well as dry_run=False.
        Stop freezes; it does not release or home. No placement success inference
        should be made from release alone."""
        if not dry_run and not allow_motion:
            return {"ok": False, "reason": "Server is configured for dry runs only"}
        return runner.start(Task(object, routine, relation, reference, distance_m),
                            dry=dry_run, background=True)

    @mcp.tool()
    def hybrid_status() -> dict:
        """Current mission, actual stage, verification evidence, and any operator requirement."""
        return runner.status()

    @mcp.tool()
    def hybrid_stop(mission_id: str) -> dict:
        """Request cooperative cancellation for this mission; poll until cancelled.
        No release or automatic home. Inference calls may delay observation of stop."""
        return runner.cancel(mission_id)

    # Detector/control diagnostics print to stdout; reserve it for MCP protocol.
    import anyio
    import contextlib
    from mcp.server.stdio import stdio_server
    protocol_stdout = anyio.wrap_file(sys.stdout)

    async def run_stdio() -> None:
        with contextlib.redirect_stdout(sys.stderr):
            async with stdio_server(stdout=protocol_stdout) as (read_stream, write_stream):
                await mcp._mcp_server.run(
                    read_stream, write_stream, mcp._mcp_server.create_initialization_options()
                )

    try:
        anyio.run(run_stdio)
    finally:
        active = runner.status()
        if active.get("status") == "running":
            runner.cancel(active["mission_id"])
            while runner.status()["status"] == "running":
                time.sleep(0.1)
        if runner.status().get("needs_operator") and isinstance(runner.backend, WristBackend):
            runner.backend.ws._hold_unparked_controller("MCP disconnected after an unparked mission")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("object", nargs="?", default="")
    ap.add_argument("--routine", default="taught_drop")
    ap.add_argument("--relation", default="drop_zone", choices=sorted(RELATIONS))
    ap.add_argument("--reference")
    ap.add_argument("--distance-m", type=float)
    ap.add_argument("--recipes", type=Path, default=DEFAULT_RECIPES)
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--plan", action="store_true", help="offline artifact/config inspection (default)")
    mode.add_argument("--dry", action="store_true", help="camera preview and existing dry checks, no motion")
    mode.add_argument("--run", action="store_true", help="execute on the robot")
    mode.add_argument("--mcp", action="store_true", help="persistent stdio MCP server")
    ap.add_argument("--allow-motion", action="store_true", help="allow live tasks in MCP server")
    ap.add_argument("--live", action="store_true",
                    help="auto-open the diagnostic wrist RGB-D browser feed on port 8731")
    args = ap.parse_args()
    if args.allow_motion and not args.mcp:
        ap.error("--allow-motion is only used with --mcp; CLI motion uses --run")
    if args.live and args.mcp:
        ap.error("--live is a one-shot CLI option; use the browser feed only with --dry or --run")
    if args.live:
        os.environ["REBOT_HYBRID_LIVE"] = "1"
    routines = load_routines(args.recipes)
    runner = MissionRunner(routines)
    if args.mcp:
        serve_mcp(runner, allow_motion=args.allow_motion)
        return
    task = Task(args.object, args.routine, args.relation, args.reference, args.distance_m)
    routine = task.validate(routines)
    if not (args.dry or args.run):
        report = calibration_report(task, routine)
        print(json.dumps({"mode": "offline_plan", "routine": routine.name,
                          "stages": ["observe", "plan_grasp", "approach", "checkpoint",
                                     "grasp", "verify_pick", "carry", "release", "return_home"],
                          "calibration": report, "trajectory_validated": False}, indent=2))
        raise SystemExit(0 if report["ok"] else 1)
    previous = signal.getsignal(signal.SIGINT)
    def stop(_signum, _frame):
        active = runner.status().get("mission_id")
        if active:
            runner.cancel(active)
    signal.signal(signal.SIGINT, stop)
    try:
        outcome = runner.start(task, dry=not args.run)
    finally:
        signal.signal(signal.SIGINT, previous)
    print(json.dumps(outcome, indent=2, default=str))
    if outcome["needs_operator"] and isinstance(runner.backend, WristBackend):
        runner.backend.ws._hold_unparked_controller("Hybrid mission interrupted; inspect result before recovery.")
    raise SystemExit(0 if outcome["ok"] else 1)


if __name__ == "__main__":
    main()
