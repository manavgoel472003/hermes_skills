"""Reachy Mini vision/head tools for the B601 arm's MCP planner surface.

One planner orchestrates both robots: the arm's existing pick/place tools plus
these ``reachy_*`` tools (Reachy is the eyes + a steerable head). The intended
demo flow is::

    reachy_calibrate()                      # once, after Reachy/bench moves
    reachy_look_around("red block")         # find it (head may be off grasp view)
    reachy_detect("red block")              # -> grasp pose in B601 base frame
    pick_object("red block", place="my hand")
    reachy_look_at_person(); reachy_track_hand()   # aim the handover at the hand

SCAFFOLD status: the planner-facing shape is real and dry-run testable now
(``REACHY_MOCK=1`` returns marked mock data). The bits that need the live bridge
+ calibration are flagged ``scaffold_not_implemented`` and land in the transport
and handover phases.
"""

from __future__ import annotations

import threading
import time
from typing import Any, Callable

from .reachy_client import ReachyBridgeClient


def register(mcp, *, arm_skill: Callable[..., dict[str, Any]], reachy: ReachyBridgeClient, workspace: str) -> None:
    """Attach the reachy_* tools onto an existing FastMCP instance."""

    def _down() -> dict[str, Any]:
        return {"ok": False, "reason": "reachy_bridge_unreachable",
                "hint": "start reachy_camera_bridge in reachy_mini_env, or set REACHY_MOCK=1"}

    # Calibration takes ~3-4 min -- longer than the MCP client's tool timeout --
    # so it runs in a BACKGROUND thread and is polled via reachy_calibrate_status.
    _calib: dict[str, Any] = {"running": False, "result": None, "log": [], "started": 0.0}

    @mcp.tool()
    def reachy_calibrate() -> dict[str, Any]:
        """START calibrating the Reachy camera -> B601 base-frame transform so picks
        land accurately (auto-aims Reachy at the arm, steps the TCP through a grid,
        locates the orange gripper tips, solves+saves the homography). Run this after
        Reachy is moved OR the arm re-zeros (USB re-enumeration) -- otherwise picks
        are consistently 'way off'.

        This RETURNS IMMEDIATELY and runs ~3-4 min in the background (the arm moves).
        Then poll reachy_calibrate_status every ~20-30s until it reports done, and
        read the reprojection error (aim for <15mm). Do NOT call other arm/reachy
        tools while a calibration is running."""
        if not reachy.available():
            return _down()
        if reachy.mock:
            _calib.update(running=False, result={"ok": True, "mock": True, "reproj_error_mm": 4.0})
            return {"ok": True, "mock": True, "started": True}
        if _calib["running"]:
            return {"ok": True, "already_running": True,
                    "note": "a calibration is already in progress; poll reachy_calibrate_status"}

        def _run():
            from . import handme_calib as calib
            try:
                res = calib.calibrate(reachy, log=lambda m: _calib["log"].append(m))
            except Exception as exc:  # noqa: BLE001
                res = {"ok": False, "reason": f"exception: {exc}"}
            if res.get("ok"):
                _state["cal"] = None; _state["gv"] = None   # force reload of the new calibration
            _calib["result"] = res
            _calib["running"] = False

        _calib.update(running=True, result=None, log=[], started=time.time())
        threading.Thread(target=_run, daemon=True).start()
        return {"ok": True, "started": True, "eta_min": 4,
                "note": "calibration running in the background; poll reachy_calibrate_status until done"}

    @mcp.tool()
    def reachy_calibrate_status() -> dict[str, Any]:
        """Check on a calibration started by reachy_calibrate. Returns running/done,
        the elapsed time, the latest progress lines, and (when finished) the result
        with reproj_error_mm. Poll this every ~20-30s until running=False."""
        elapsed = round(time.time() - _calib["started"], 0) if _calib["started"] else 0
        out: dict[str, Any] = {"ok": True, "running": bool(_calib["running"]),
                               "elapsed_s": elapsed, "recent": _calib["log"][-4:]}
        if not _calib["running"] and _calib["result"] is not None:
            out["result"] = _calib["result"]
        return out

    @mcp.tool()
    def reachy_look_around(object: str | None = None) -> dict[str, Any]:
        """Sweep Reachy's head across scan views to FIND things when the object
        may be outside the grasp view. Optionally name what to look for. Returns
        which views were covered and any objects seen. Follow with reachy_detect
        to get a grasp pose (it snaps back to the calibrated grasp view)."""
        if not reachy.available():
            return _down()
        views = ["scan_left", "scan_center", "scan_right"]
        for view in views:
            reachy.head(preset=view, duration=0.7)
            # transport phase: run the VLM detector on reachy.frame_jpeg(view=view)
        if reachy.mock:
            return {"ok": True, "mock": True, "views": views,
                    "found": bool(object), "seen_objects": [object] if object else []}
        return {"ok": True, "views": views, "seen_objects": [],
                "note": "per-view detection wiring lands in the transport phase"}

    # Interpolated tool text must go in `description=`: a `"""...""" % workspace`
    # docstring is an expression, so __doc__ is None and the planner gets nothing.
    @mcp.tool(description=f"""Snap Reachy's head to the calibrated GRASP_VIEW, detect
        `object` in Reachy's camera, and return its grasp pose IN THE B601 BASE
        FRAME (metres), ready to hand to pick_object. Requires reachy_calibrate to
        have succeeded. Workspace: {workspace}.""")
    def reachy_detect(object: str, color: str | None = None) -> dict[str, Any]:
        if not reachy.available():
            return _down()
        reachy.head(preset="grasp_view", duration=1.0)
        if reachy.mock:
            return {"ok": True, "mock": True, "object": object, "camera": "reachy",
                    "grasp_pose_m": [0.30, -0.04, 0.02], "fits_gripper": True}
        # Real path reuses ALL existing detection/grasp code: the arm service
        # detects on the "reachy" camera source and applies the reachy homography.
        return arm_skill("detect_object", object=object, color=color, camera="reachy")

    # --- real hand-me pipeline (headless core; head parks at pick view) --------
    _state: dict[str, Any] = {"cal": None, "gv": None, "bearing": None, "holding": None}

    def _cal_gv():
        if _state["cal"] is None:
            from .handme_core import load_calib_gv
            _state["cal"], _state["gv"] = load_calib_gv()
        return _state["cal"], _state["gv"]

    @mcp.tool()
    def reachy_pick(object: str, color: str | None = None) -> dict[str, Any]:
        """Detect `object` in Reachy's camera at the fixed calibrated PICK VIEW and
        grasp it precisely (orientation-aware; opens the gripper only to its safe
        cap; verifies the object left the table and retries once). The arm ends
        HOLDING the object. `color` (e.g. 'green') sharpens the grasp point on a
        coloured object. Follow with reachy_find_palm + reachy_present, then
        reachy_release once the user confirms."""
        if not reachy.available():
            return _down()
        if reachy.mock:
            return {"ok": True, "mock": True, "object": object, "grasp_m": [0.30, -0.04]}
        from . import handme_core as hc
        cal, gv = _cal_gv()
        res = hc.pick(reachy, cal, gv, object, color)
        if res.get("ok"):
            _state["holding"] = object
        return res

    @mcp.tool()
    def reachy_find_palm() -> dict[str, Any]:
        """Sweep Reachy's head to find the user's OPEN palm held up, then keep the
        head centred on it. Returns whether a palm was found and its bearing. This
        is the ONLY step that moves Reachy's head. Grasp something first."""
        if not reachy.available():
            return _down()
        if reachy.mock:
            _state["bearing"] = 0.0
            return {"ok": True, "mock": True, "found": True, "bearing_deg": 0.0}
        from . import handme_core as hc
        res = hc.find_palm(reachy)
        if res.get("found"):
            _state["bearing"] = res.get("bearing_deg")
        return res

    @mcp.tool()
    def reachy_present(bearing_deg: float | None = None) -> dict[str, Any]:
        """Reach the held object toward the user's palm (approximate; toward the
        bearing from reachy_find_palm). Does NOT release -- ask the user to confirm,
        then call reachy_release. Pass bearing_deg to override the last find_palm."""
        if not reachy.available():
            return _down()
        if reachy.mock:
            return {"ok": True, "mock": True, "awaiting": "reachy_release"}
        from . import handme_core as hc
        _cal, gv = _cal_gv()
        b = bearing_deg if bearing_deg is not None else _state["bearing"]
        return hc.present(reachy, gv, b)

    @mcp.tool()
    def reachy_release() -> dict[str, Any]:
        """Open the gripper to release the held object into the user's hand. Call
        this ONLY after the user has confirmed their palm is in place."""
        if reachy.mock:
            return {"ok": True, "mock": True, "released": True}
        from . import handme_core as hc
        _state["holding"] = None
        return hc.release()

    @mcp.tool()
    def reachy_hand_me(object: str, color: str | None = None) -> dict[str, Any]:
        """High-level: pick `object`, look around for the user's open palm, and
        present it toward that palm -- all in one call. Stops WITHOUT releasing and
        returns awaiting=reachy_release, so you can confirm with the user ('shall I
        let go?') and then call reachy_release. `color` sharpens a coloured grasp."""
        if not reachy.available():
            return _down()
        if reachy.mock:
            return {"ok": True, "mock": True, "object": object, "awaiting": "reachy_release"}
        from . import handme_core as hc
        cal, gv = _cal_gv()
        pr = hc.pick(reachy, cal, gv, object, color)
        if not pr.get("ok"):
            return {"stage": "pick", **pr}
        _state["holding"] = object
        fp = hc.find_palm(reachy)
        pres = hc.present(reachy, gv, fp.get("bearing_deg"))
        return {"ok": bool(pres.get("ok")), "object": object, "pick": pr,
                "palm_found": bool(fp.get("found")), "present": pres,
                "awaiting": "reachy_release",
                "note": "confirm with the user, then call reachy_release"}

    @mcp.tool()
    def reachy_look_at_person() -> dict[str, Any]:
        """Turn Reachy's head up toward the user, to begin a handover."""
        if not reachy.available():
            return _down()
        res = reachy.head(preset="look_person", duration=0.7)
        return {"ok": True, "head_pose": res.get("head_pose"), **({"mock": True} if reachy.mock else {})}

    # (reachy_track_hand removed -- superseded by the real reachy_find_palm above)
