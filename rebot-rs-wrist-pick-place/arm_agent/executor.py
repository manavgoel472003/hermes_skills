"""The agent loop: plan -> act -> verify -> replan, with memory.

An :class:`AgentSession` holds the running conversation with the planner and the
connection to the arm service. ``run(task)`` drives the tool-calling loop until
the planner calls ``done`` (finished), calls ``ask`` (needs the user), or the step
budget is hit. The planner sees each skill's structured result (including failure
reasons like ``grasp_missed``) and is told to re-perceive and retry -- that feedback
loop is what makes long-horizon, multi-object tasks work.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Callable

from .service_client import ArmServiceClient
from .hermes_client import Planner, iter_tool_calls
from .tools import build_tool_schema, execute_tool

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are the planner for a 6-DOF reBot robot arm with a parallel gripper and a \
top-down camera. You carry out the user's task by calling tools (skills). You never \
control motors directly.

World frame (metres): +x points forward / away from the base, +y points to the arm's LEFT, \
+z is up. The reachable workspace is x in [0.10, 0.37], y in [-0.28, 0.28]; objects sit on the \
table near x in [0.20, 0.36], y in [-0.18, 0.18]. NEVER invent a coordinate outside this box -- \
out-of-range targets are rejected. "scan_scene" returns every object with its base-frame position \
and relations (left_of, right_of, nearer_base_than, farther_from_base_than, near).

How to work:
- Call scan_scene FIRST to see what is on the table before acting. Refer to objects by the labels it reports.
- If the user only ASKS what is present (e.g. "what is on the table?", "what do you see?"), call scan_scene \
and then report the objects you found with done() -- do NOT call ask() for a perception question.
- pick_object is your primary tool: name the object (no colour needed) and it localizes (VLM box + \
top-camera homography + RGB refine) and grasps it. It can ALSO place in the SAME call -- pick ONE form by the task:
    * "pick up / grab / lift X"  (no destination)        -> pick_object(object="X")              [holds it; do NOT add a place]
    * "hand me / give me X"                               -> pick_object(object="X", place="my hand")
    * "put / move X to <a free spot / coordinates>"       -> pick_object(object="X", place_x=<x>, place_y=<y>)  (inside the box above)
    * "put X on / onto Y"  or  "stack X on Y"             -> stack(object="X", on="Y")
- For placing NEXT TO another object ("to the left/right/front/behind of Y"), it is TWO steps: \
first pick_object(object="X") to grasp X, THEN place_beside(object="Y", side="left|right|front|behind"). \
place_beside / place_on / place_at place whatever is CURRENTLY held, so you MUST pick first.
- "push X <direction>" -> push(object="X", direction="forward|backward|left|right", distance=<m>). point_at / rotate_wrist / home are direct.
- After any action that can fail (grasp/pick/place), check the result; if it failed \
(reason "grasp_missed", "not_found", "ik_unreachable", "out_of_workspace", "place_failed"), \
re-scan and retry a sensible alternative -- do NOT repeat the same failing call or fall back to a made-up coordinate.
- Use scan_scene again to VERIFY the world changed as intended before moving on.
- If the task is ambiguous or a needed object is not on the table, call ask().
- Do EXACTLY what the task asks and nothing more: do not pick up, move, or rearrange objects the task \
did not mention. "Keep going until done" means finish the requested task, not invent extra steps.
- When the requested task is complete, call done() with a short summary. Do not stop before it is done.

{calibration_note}
Available skills: {skill_summary}
"""


@dataclass
class AgentSession:
    client: ArmServiceClient
    planner: Planner
    dry_run: bool = False
    max_steps: int = 24
    messages: list[dict[str, Any]] = field(default_factory=list)
    executed: list[dict[str, Any]] = field(default_factory=list)
    observer: Callable[[str, dict[str, Any]], None] | None = None
    _tools: list[dict[str, Any]] = field(default_factory=list)
    _skill_names: set[str] = field(default_factory=set)
    _started: bool = False
    _current_task: str = ""

    # -- observability ------------------------------------------------------
    def _emit(self, event: str, payload: dict[str, Any]) -> None:
        """Notify an attached observer (e.g. the TUI) of a loop event.

        Events: ``task`` (new task), ``planner`` (assistant message/thought),
        ``tool_call`` (about to run a skill), ``tool_result`` (skill returned),
        ``scene`` (scan_scene objects), ``final`` (task finished). Best-effort:
        an observer error never breaks the control loop.
        """
        if self.observer is None:
            return
        try:
            self.observer(event, payload)
        except Exception:
            logger.debug("observer raised on event %s", event, exc_info=True)

    # -- setup --------------------------------------------------------------
    def _ensure_started(self) -> None:
        if self._started:
            return
        listing = self.client.skill_list()
        specs = listing.get("skills", {})
        self._skill_names = set(specs)
        self._tools = build_tool_schema(specs)
        try:
            calib = self.client.calibration()
            calib_note = (
                "Object positions are available (camera is calibrated)."
                if calib.get("calibrated")
                else "WARNING: the camera is NOT calibrated, so object positions are unavailable; "
                "ask the user to run calibration before manipulation."
            )
        except Exception:
            calib_note = ""
        skill_summary = ", ".join(f"{n}({', '.join(s.get('params', {}))})" for n, s in specs.items())
        self.messages.append({
            "role": "system",
            "content": SYSTEM_PROMPT.format(calibration_note=calib_note, skill_summary=skill_summary),
        })
        self._started = True

    # -- public -------------------------------------------------------------
    def run(self, task: str) -> dict[str, Any]:
        self._ensure_started()
        self._current_task = task
        self._emit("task", {"task": task, "dry_run": self.dry_run})
        self.messages.append({"role": "user", "content": task})
        result = self._loop()
        self._emit("final", result)
        return result

    def provide_answer(self, answer: str) -> dict[str, Any]:
        """Resume after an ``ask`` with the user's reply."""
        self.messages.append({"role": "user", "content": answer})
        result = self._loop()
        self._emit("final", result)
        return result

    # -- core loop ----------------------------------------------------------
    def _loop(self) -> dict[str, Any]:
        empty_turns = 0
        for _ in range(self.max_steps):
            msg = self.planner.chat(self.messages, self._tools)
            self.messages.append(_as_assistant(msg))
            calls = iter_tool_calls(msg)
            self._emit("planner", {
                "content": (msg.get("content") or "").strip(),
                "tool_calls": [c["name"] for c in calls],
            })

            if not calls:
                content = (msg.get("content") or "").strip()
                empty_turns += 1
                if empty_turns >= 2:
                    return {"status": "done", "summary": content or "(no action taken)",
                            "executed": self.executed}
                self.messages.append({
                    "role": "user",
                    "content": "Use a tool to act, or call done() if the task is complete.",
                })
                continue
            empty_turns = 0

            for call in calls:
                name, args = call["name"], call["arguments"]
                self._emit("tool_call", {"name": name, "args": args})
                result = execute_tool(
                    self.client, name, args,
                    skill_names=self._skill_names, dry_run=self.dry_run,
                )
                if name == "scan_scene":
                    self._emit("scene", {
                        "objects": result.get("objects", []),
                        "relations": result.get("relations", []),
                    })
                self._emit("tool_result", {
                    "name": name,
                    "skill": result.get("skill") or name,
                    "ok": result.get("ok"),
                    "reason": result.get("reason"),
                    "control": result.get("_control"),
                    "result": result,
                })
                control = result.get("_control")
                if control == "done":
                    return {"status": "done", "summary": result.get("summary", ""),
                            "executed": self.executed}
                if control == "ask":
                    return {"status": "ask", "question": result.get("question", ""),
                            "executed": self.executed}
                actual_skill = result.get("skill") or name
                self.executed.append({"skill": actual_skill, "args": args,
                                      "ok": result.get("ok"), "reason": result.get("reason")})
                self.messages.append({
                    "role": "tool",
                    "tool_name": name,
                    "content": _summarize_result(name, result),
                })
                if (
                    actual_skill in ("pick_object", "pick_marker")
                    and result.get("ok")
                    and _is_simple_pick(self._current_task)
                ):
                    return {
                        "status": "done",
                        "summary": "object picked",
                        "executed": self.executed,
                    }
        return {"status": "max_steps", "summary": "step budget exhausted", "executed": self.executed}


def _as_assistant(msg: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {"role": "assistant", "content": msg.get("content", "") or ""}
    if msg.get("tool_calls"):
        out["tool_calls"] = msg["tool_calls"]
    return out


def _summarize_result(name: str, result: dict[str, Any], max_len: int = 1500) -> str:
    """Compact JSON of a tool result for the planner (trims huge scenes)."""
    trimmed = dict(result)
    text = json.dumps(trimmed, default=str)
    if len(text) > max_len:
        # keep the most useful keys
        keep = {k: trimmed[k] for k in ("ok", "reason", "objects", "relations", "calibrated", "summary")
                if k in trimmed}
        text = json.dumps(keep, default=str)[:max_len]
    return text


def _is_simple_pick(task: str) -> bool:
    """A bare 'pick up X' with no placement -- finish as soon as the grasp
    succeeds so the planner doesn't invent a place step."""
    text = (task or "").lower()
    if not any(word in text for word in ("pick", "pickup", "pick up", "grab", "lift")):
        return False
    return not any(word in text for word in ("place", "put", "move", "stack", "beside", "on top", "into", "hand"))
