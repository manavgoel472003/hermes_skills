"""MCP server exposing the reBot B601 arm as tools for an external agent.

This lets a capable agent (e.g. the `hermes` CLI backed by OpenAI Codex) be the
planner/brain, calling the arm's skills as Model-Context-Protocol tools. The
depth-camera visual-servo tools run directly in ``rebot_rs`` (camera, detector,
IK and CAN); legacy scene-graph tools still use ``rebot_arm_service`` if enabled.

Register with hermes (one time)::

    hermes mcp add rebot-arm \
      --command /home/manav/miniconda3/envs/rebot_rs/bin/python \
      --args -m arm_agent.mcp_server \
      --env REBOT_ARM_SERVICE_URL=http://127.0.0.1:8000

(The depth-camera visual-servo tools offload detection to the arm service, so the clean
rebot_rs env only needs httpx + mcp -- no torch/transformers here.)

Then drive it from ``hermes chat`` in natural language.

Env:
  REBOT_ARM_SERVICE_URL   arm service base URL (default http://127.0.0.1:8000)
  REBOT_MCP_DRY_RUN=1     plan + validate IK only; never move the arm
"""

from __future__ import annotations

import os
from pathlib import Path
import sys
from typing import Any

# Hermes starts this module directly rather than through run_object_pick.sh.  Give
# its wrist tools the same proven profile as that script while still allowing an
# explicit MCP environment value to override any setting.
_ROOT = Path(__file__).resolve().parent.parent
_SAM3_VENDOR = _ROOT / ".vendor" / "sam3_transformers"
if _SAM3_VENDOR.is_dir() and str(_SAM3_VENDOR) not in sys.path:
    sys.path.insert(0, str(_SAM3_VENDOR))

_WRIST_PICK_PROFILE = {
    "REBOT_DYNAMIC_GRASP": "1",
    "REBOT_WRIST_CLOSE_GRIPPER_AT_START": "1",
    "REBOT_CAN_BITRATE": "1000000",
    "REBOT_ENABLE_LOCATE_ANYTHING": "1",
    "REBOT_VISION_BACKEND": "sam3_1",
    "REBOT_VISION_TRACKER": "none",
    "REBOT_SAM31_BOOTSTRAP_ONLY": "1",
    "REBOT_VISION_PRIMARY": "1",
    "REBOT_SAM3_MODEL": "facebook/sam3",
    "REBOT_VISION_DEVICE": "cuda",
    "REBOT_VISION_DTYPE": "bfloat16",
    "REBOT_SAM3_SCORE_THRESHOLD": "0.30",
    "REBOT_SAM3_MASK_THRESHOLD": "0.50",
    "REBOT_WS_X_MIN": "0.150",
    "REBOT_WS_X_MAX": "0.540",
    "REBOT_WS_Y_MIN": "-0.300",
    "REBOT_WS_Y_MAX": "0.300",
    "REBOT_WS_Z_MIN": "0.000",
    "REBOT_WS_Z_MAX": "0.360",
    "REBOT_WRIST_NEAR_MM": "60",
    "REBOT_WRIST_FAR_MM": "600",
    "REBOT_WRIST_PICK_CLOSER_MM": "4",
    "REBOT_WRIST_PICK_Y_OFFSET_MM": "8",
    "REBOT_WRIST_PICK_TARGET_TOL_MM": "20",
    "REBOT_WRIST_PICK_MAX_CORRECTIONS": "3",
    "REBOT_WRIST_PICK_MAX_CORRECTION_M": "0.012",
    "REBOT_WRIST_FINAL_ALIGN_GAIN": "0.85",
    "REBOT_WRIST_RIGID_FIT_MAX_ERROR": "0.15",
    "REBOT_WRIST_USE_SAVED_CAMERA_ROTATION": "1",
    "REBOT_WRIST_SAVED_ROTATION_MAX_RPY_DELTA_DEG": "20.0",
    "REBOT_WRIST_PICK_OPEN_BEFORE_END_M": "0.040",
    "REBOT_WRIST_JAW_LONG_AXIS_CAMERA_DEG": "-90",
    "REBOT_WRIST_GRASP_MAX_YAW_DELTA_DEG": "16",
    "REBOT_WRIST_GRASP_YAW_MEASUREMENT_SLOP_DEG": "1.5",
    "REBOT_WRIST_GRASP_MAX_RESIDUAL_YAW_DEG": "6",
    "REBOT_WRIST_GRASP_RPY_STEP_DEG": "2",
    "REBOT_WRIST_REACH_PITCH_BIAS_DEG": "10",
    "REBOT_WRIST_GRASP_FINAL_APPROACH_MM": "40",
    "REBOT_WRIST_GRASP_MIN_FINAL_APPROACH_MM": "20",
    "REBOT_WRIST_GRASP_MIN_FEATURE_PROMINENCE_MM": "25",
    "REBOT_DYNAMIC_GRIPPER_SLOT_MARGIN_MM": "8",
    "REBOT_DYNAMIC_GRIPPER_REMEASURE_RADIUS_MM": "6",
    "REBOT_DYNAMIC_GRIPPER_CLOSE_STEP_DEG": "18",
    "REBOT_GRIPPER_SIZE_MARGIN_MM": "0",
    "REBOT_GRIPPER_SIZE_COMPRESSION_MM": "3",
    "REBOT_GRASP_MICRO_LIFT_M": "0.015",
    "REBOT_GRASP_HOLD_FRAMES": "6",
    "REBOT_GRASP_RETRY_CLOSE_DEG": "5",
    "REBOT_GRASP_MAX_RETRIES": "3",
    "REBOT_LOCAL_IK_POS_TOL_M": "0.004",
    "REBOT_IK_MAX_ITER": "400",
    "REBOT_ARM_SPEED_M_S": "0.036",
    "REBOT_ARM_MIN_MOVE_S": "0.42",
    "REBOT_ARM_ANGULAR_SPEED_RAD_S": "0.60",
    "REBOT_JOINT_VLIM": "0.60",
    "REBOT_GRIPPER_MOVE_S": "0.95",
    "REBOT_GRIPPER_STEP_S": "0.035",
}
for _profile_key, _profile_value in _WRIST_PICK_PROFILE.items():
    os.environ.setdefault(_profile_key, _profile_value)

import httpx
from mcp.server.fastmcp import FastMCP

from .reachy_client import ReachyBridgeClient
from .reachy_mcp import register as _register_reachy_tools
from .service_client import ArmServiceClient

_URL = os.getenv("REBOT_ARM_SERVICE_URL", "http://127.0.0.1:8000")
_DRY_RUN = os.getenv("REBOT_MCP_DRY_RUN", "0").strip().lower() in {"1", "true", "yes"}

mcp = FastMCP("rebot-arm")
_client = ArmServiceClient(base_url=_URL)
_connected = False

# Reachable workspace (metres, base frame). Surfaced in tool docs so the planner
# never proposes an out-of-reach coordinate (those are rejected by the service).
_WORKSPACE = "x in [0.18, 0.49], y in [-0.18, 0.18]; objects sit near x[0.20,0.48] y[-0.15,0.15]"


def _ensure_connected() -> None:
    global _connected
    if _connected:
        return
    try:
        httpx.post(f"{_URL}/connect", json={}, timeout=20.0)
    except Exception:
        pass  # the service may already hold the connection
    _connected = True


def _trim_steps(result: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {"step": s.get("label"), "ok": s.get("ok"), "reason": s.get("reason")}
        for s in result.get("steps", [])
    ]


def _visual(name: str, call: Any) -> dict[str, Any]:
    """Run a depth-camera visual-servo skill (in-process: overhead camera + detector +
    arm_safe live here, not in the arm service). Lazily imported so a missing camera
    dep can't break the rest of the server."""
    try:
        from . import depth_servo as _vs
        from . import arm_safe as _arm
        if not _DRY_RUN and _arm.BACKEND == "local":
            from .local_arm import get_local_arm
            get_local_arm().connect()
        elif _arm.BACKEND != "local":
            _ensure_connected()
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": f"depth_servo import failed: {exc}", "skill": name}
    try:
        out = call(_vs)
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": str(exc), "skill": name}
    out.setdefault("skill", name)
    out["dry_run"] = _DRY_RUN
    return out


def _wrist(name: str, call: Any) -> dict[str, Any]:
    """Run an eye-in-hand wrist-camera skill (in-process: the wrist RealSense, the
    colour/VLM detector and arm_safe all live here). Lazily imported so a missing
    camera dependency cannot break the rest of the server."""
    try:
        from . import wrist_servo as _ws
        from . import arm_safe as _arm
        if not _DRY_RUN and _arm.BACKEND == "local":
            from .local_arm import get_local_arm
            get_local_arm().connect()
        elif _arm.BACKEND != "local":
            _ensure_connected()
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": f"wrist_servo import failed: {exc}", "skill": name}
    try:
        out = call(_ws)
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": str(exc), "skill": name}
    out.setdefault("skill", name)
    out["dry_run"] = _DRY_RUN
    return out


def _local_direct(action: str) -> dict[str, Any] | None:
    """Handle the small direct-control surface without starting the HTTP service."""
    try:
        from . import arm_safe as arm
        if arm.BACKEND != "local":
            return None
        from .local_arm import get_local_arm
        backend = get_local_arm()
        backend.connect()
        if action in {"open", "close"}:
            return backend.gripper(action)
        if action == "home":
            backend.home()
            return {"ok": True, "state": backend.state()}
        if action == "state":
            return backend.state()
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": str(exc)}
    return None


def _skill(name: str, **params: Any) -> dict[str, Any]:
    """Run a service skill, returning a compact result for the planner."""
    _ensure_connected()
    params = {k: v for k, v in params.items() if v is not None}
    try:
        r = _client.run_skill(name, params, dry_run=_DRY_RUN)
    except Exception as exc:
        return {"ok": False, "error": str(exc), "skill": name}
    out: dict[str, Any] = {
        "ok": bool(r.get("ok")),
        "skill": r.get("skill") or name,
        "reason": r.get("reason"),
        "dry_run": _DRY_RUN,
    }
    steps = _trim_steps(r)
    if steps:
        out["steps"] = steps
    det = (r.get("detail") or {}).get("detection") if isinstance(r.get("detail"), dict) else None
    if isinstance(det, dict) and det.get("grasp_pose"):
        out["grasp_pose_m"] = det["grasp_pose"].get("position_m")
    return out


# -- perception -------------------------------------------------------------
# NOTE: tool text that interpolates _WORKSPACE goes in `description=`, NOT in the
# docstring. `"""...""" % _WORKSPACE` is an expression, not a string literal, so
# Python sets __doc__ to None and the planner would receive an EMPTY description.
@mcp.tool(description=f"""Look at the table with the top camera and list every
    detected object with its base-frame position (metres) and size. Call this
    FIRST, and again after actions, to see the current world.
    Workspace: {_WORKSPACE}.""")
def scan_scene() -> dict[str, Any]:
    _ensure_connected()
    try:
        scene = _client.scene()
    except Exception as exc:
        return {"ok": False, "error": str(exc)}
    objs = [
        {
            "label": o.get("label"),
            "position_m": o.get("position_m") or o.get("position"),
            "size_m": o.get("size_m") or o.get("size"),
        }
        for o in scene.get("objects", [])
    ]
    return {"ok": True, "count": len(objs), "objects": objs, "relations": scene.get("relations", [])}


@mcp.tool()
def detect_object(object: str) -> dict[str, Any]:
    """Locate a single named object on the table (no motion). Returns its grasp
    pose and whether it fits the gripper. Use to verify an object is present
    before picking. Distinct/coloured objects detect most reliably."""
    return _skill("detect_object", object=object)


# -- pick & place -----------------------------------------------------------
@mcp.tool(description=f"""Pick up a named object, and optionally place it in the
    SAME call.
    - pick & hold: pick_object(object="green sponge")
    - place at a spot: pick_object(object=..., place_x=0.30, place_y=-0.10)
      (inside {_WORKSPACE})
    - hand it to the user: pick_object(object=..., place="my hand")
    - place onto another object: pick_object(object=..., place="<other object label>")
    Grasp localization is best for distinct/coloured objects.""")
def pick_object(
    object: str,
    place_x: float | None = None,
    place_y: float | None = None,
    place: str | None = None,
) -> dict[str, Any]:
    return _skill("pick_object", object=object, place_x=place_x, place_y=place_y, place=place)


@mcp.tool(description=f"""Place the CURRENTLY held object at table coordinate
    (x, y) in metres (inside {_WORKSPACE}). You must have picked something first.""")
def place_at(x: float, y: float) -> dict[str, Any]:
    return _skill("place_at", x=x, y=y)


# -- depth-camera visual servo (calibration-free; overhead Orbbec) -----------
@mcp.tool(description=f"""Pick a named object using the OVERHEAD DEPTH CAMERA
    (Orbbec) with closed-loop visual servoing -- NO calibration needed and
    drift-immune. The arm watches its own gripper tip AND the object in the depth
    camera and drives them together, then descends by feel (torque-gated) and
    grasps. Leaves the object held -- call visual_place next to put it down. Best
    for distinct/coloured objects on the table. This is the depth-camera
    alternative to pick_object (which uses the flat-table homography).
    Workspace: {_WORKSPACE}.""")
def visual_pick(object: str) -> dict[str, Any]:
    return _visual("visual_pick", lambda vs: vs.servo_pick(object, dry=_DRY_RUN))


@mcp.tool(description=f"""Place the CURRENTLY held object using the overhead depth
    camera. Either:
    - onto/into a detected target: visual_place(target="the bowl") -- the arm
      visually servos the held object over it and lowers by feel, or
    - at explicit table coords: visual_place(x=0.30, y=-0.10)
      (metres, inside {_WORKSPACE}).
    Pick something with visual_pick first.""")
def visual_place(
    target: str | None = None,
    x: float | None = None,
    y: float | None = None,
) -> dict[str, Any]:
    return _visual("visual_place", lambda vs: vs.servo_place(target=target, x=x, y=y, dry=_DRY_RUN))


# -- wrist camera / eye-in-hand (proven fixed trajectory + taught drop zone) --
@mcp.tool(description=f"""Pick and lift a named object using the proven RS wrist
    camera pipeline, then HOLD it at the carry pose. Detection, geometry, jaw sizing,
    bounded alignment, slow position closing, and camera-only micro-lift verification
    are completed before this tool returns. There is no contact inference and no
    torque-gated descent. Call wrist_place next in the SAME Hermes controller
    session. Do not restart the arm service between the two calls. The object must
    be visible and inside the calibrated jaw/workspace limits ({_WORKSPACE}).""")
def wrist_pick(object: str = "any") -> dict[str, Any]:
    return _wrist(
        "wrist_pick",
        lambda ws: ws.fixed_pick(object, dry=_DRY_RUN, verbose=not _DRY_RUN),
    )


@mcp.tool()
def wrist_place(
    reference: str | None = None,
    relation: str = "drop_zone",
    distance_m: float | None = None,
) -> dict[str, Any]:
    """Place the object held by wrist_pick, verify release, and home. For a taught
    release use relation='drop_zone'. For scene-relative placement provide a
    specific reference and relation: left, right, front, behind, or next_to.
    Directions are in the robot/table frame: +X front/away from the base and +Y
    robot-left. next_to selects the reachable side with the largest workspace-edge
    clearance. relation='free_space' conservatively uses the taught table point
    because obstacle mapping is not enabled. Optional distance_m is clamped to
    0.06-0.18m. The reference must be visible from the carry pose. This refuses
    motion unless the same MCP process completed wrist_pick."""
    return _wrist(
        "wrist_place",
        lambda ws: ws.fixed_place(
            reference=reference, relation=relation, distance_m=distance_m,
            dry=_DRY_RUN, verbose=not _DRY_RUN,
        ),
    )


@mcp.tool()
def wrist_pick_and_place(
    object: str,
    reference: str | None = None,
    relation: str = "drop_zone",
    distance_m: float | None = None,
) -> dict[str, Any]:
    """Complete pick, verified lift, place, and home in one call. The original
    taught-zone behavior remains relation='drop_zone'. Scene-relative placement
    accepts reference plus left/right/front/behind/next_to. Directions use the
    robot/table frame. relation='free_space' uses the taught safe table point while
    collision mapping is disabled. Scene localization happens only after the pick
    is visually lift-verified, from the carry pose, before any place trajectory."""
    return _wrist(
        "wrist_pick_and_place",
        lambda ws: ws.fixed_pick_and_place(
            object, dry=_DRY_RUN, verbose=not _DRY_RUN,
            place_reference=reference, place_relation=relation,
            place_distance_m=distance_m,
        ),
    )


@mcp.tool()
def wrist_stop() -> dict[str, Any]:
    """Emergency software stop for the RS wrist workflow: freeze the measured pose,
    open the gripper, and home. Use when the user says stop, cancel, abort, or quit
    after a split wrist_pick has returned holding an object. Active MCP calls use
    MCP cancellation; this tool is the Hermes-safe equivalent of Q during the hold.
    MCP stdio is never used as a raw keyboard channel."""
    return _wrist(
        "wrist_stop",
        lambda ws: ws.abort_fixed_hold("Hermes wrist_stop requested"),
    )


@mcp.tool()
def wrist_look(object: str | None = None) -> dict[str, Any]:
    """Look through the wrist camera WITHOUT moving (one frame). Reports whether the
    named object is visible from where the arm is now, which detector found it, and
    whether the TCP calibration and drop zone are set up. Use this to check the demo
    is ready, or to answer "can you see the red block?" A "no" means it is not in
    the current view; the fixed picker intentionally does not use scan poses."""
    return _wrist("wrist_look", lambda ws: ws.snapshot(object))


@mcp.tool()
def wrist_teach_drop_zone() -> dict[str, Any]:
    """Record where the arm is RIGHT NOW as the drop zone that wrist_place uses.
    Moves nothing -- it only reads the current pose. Ask the user to position the
    gripper over the bin/tray at release height before you call this."""
    return _wrist("wrist_teach_drop_zone", lambda ws: ws.teach_drop_zone())


@mcp.tool()
def place_on(object: str) -> dict[str, Any]:
    """Place the currently held object on top of another object (by label).
    Pick something first."""
    return _skill("place_on", object=object)


@mcp.tool()
def place_beside(object: str, side: str = "left") -> dict[str, Any]:
    """RS wrist scene placement beside a visible reference. side is left, right,
    front, behind, or next_to in the robot/table frame. Pick something first in
    this same Hermes session."""
    return _wrist(
        "place_beside",
        lambda ws: ws.fixed_place(
            reference=object, relation=side,
            dry=_DRY_RUN, verbose=not _DRY_RUN,
        ),
    )


@mcp.tool()
def stack(object: str, on: str) -> dict[str, Any]:
    """Pick up `object` and stack it on top of `on` (both by label) in one call."""
    return _skill("stack", object=object, on=on)


@mcp.tool()
def hand_over(object: str) -> dict[str, Any]:
    """Pick up a named object and present it to the user to take."""
    return _skill("pick_object", object=object, place="my hand")


# -- non-prehensile & gestures ---------------------------------------------
@mcp.tool()
def push(object: str, direction: str = "forward", distance: float = 0.06) -> dict[str, Any]:
    """Push a named object across the table. direction: forward, backward, left,
    right (forward = away from base / +x). distance in metres (default 0.06)."""
    return _skill("push", object=object, direction=direction, distance=distance)


@mcp.tool()
def point_at(object: str) -> dict[str, Any]:
    """Point the gripper at a named object (no grasp)."""
    return _skill("point_at", object=object)


# -- direct control ---------------------------------------------------------
@mcp.tool()
def open_gripper() -> dict[str, Any]:
    """Open the gripper (releases any held object)."""
    if local := _local_direct("open"):
        return local
    return _skill("open_gripper")


@mcp.tool()
def close_gripper() -> dict[str, Any]:
    """Close the gripper."""
    if local := _local_direct("close"):
        return local
    return _skill("close_gripper")


@mcp.tool()
def home() -> dict[str, Any]:
    """Move the arm to its safe home pose."""
    if local := _local_direct("home"):
        return local
    _ensure_connected()
    try:
        return {"ok": True, **_client.home()}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


@mcp.tool()
def get_state() -> dict[str, Any]:
    """Report the arm's current joint angles and end-effector pose."""
    if local := _local_direct("state"):
        return local
    _ensure_connected()
    try:
        return _client.state()
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


# -- reachy vision/head (eyes for the arm; see reachy_mcp) ------------------
_reachy = ReachyBridgeClient()
_register_reachy_tools(mcp, arm_skill=_skill, reachy=_reachy, workspace=_WORKSPACE)


def main() -> None:
    mcp.run()  # stdio transport


if __name__ == "__main__":
    main()
