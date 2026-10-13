"""Planner tool schema + dispatch.

The planner's action space is: one tool per service skill (fetched live from
``/skill/list`` so the brain and service stay decoupled), plus a few built-ins:
``scan_scene`` (perceive), ``get_state`` (proprioception), ``done`` (finish), and
``ask`` (request clarification). Tool calls are dispatched to the service client.
"""

from __future__ import annotations

from typing import Any

from .service_client import ArmServiceClient

_TYPE_MAP = {"str": "string", "float": "number", "bool": "boolean", "int": "integer"}

BUILTIN_TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "scan_scene",
            "description": (
                "Look at the table and return every object with its base-frame "
                "position (metres) and spatial relations (left_of, right_of, "
                "nearer_base_than, farther_from_base_than, near). Call this first, "
                "and again after actions, to see the current world."
            ),
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_state",
            "description": "Get the arm's current joint and end-effector state.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "done",
            "description": "Call when the task is fully complete. Provide a short summary.",
            "parameters": {
                "type": "object",
                "properties": {"summary": {"type": "string", "description": "what was accomplished"}},
                "required": ["summary"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "ask",
            "description": "Ask the user a clarifying question when the task is ambiguous or an object is missing.",
            "parameters": {
                "type": "object",
                "properties": {"question": {"type": "string", "description": "the question for the user"}},
                "required": ["question"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_code",
            "description": (
                "Escape-hatch for motions no skill covers. Provide Python that calls the "
                "available functions: scan_scene(), locate(desc), move_to(x,y,z[,roll,pitch,yaw]), "
                "gripper('open'|'close'), and any skill (pick, place_at, push, ...). No imports, "
                "files, or network. Prefer skills; use this only when necessary."
            ),
            "parameters": {
                "type": "object",
                "properties": {"code": {"type": "string", "description": "Python to execute in the sandbox"}},
                "required": ["code"],
            },
        },
    },
]

_CONTROL_TOOLS = {"done", "ask"}


def _skill_tool(name: str, spec: dict[str, Any]) -> dict[str, Any]:
    props: dict[str, Any] = {}
    required: list[str] = []
    for pname, pspec in spec.get("params", {}).items():
        props[pname] = {
            "type": _TYPE_MAP.get(pspec.get("type", "str"), "string"),
            "description": pspec.get("desc", ""),
        }
        if pspec.get("required"):
            required.append(pname)
    fn: dict[str, Any] = {
        "name": name,
        "description": spec.get("doc", name),
        "parameters": {"type": "object", "properties": props},
    }
    if required:
        fn["parameters"]["required"] = required
    return {"type": "function", "function": fn}


def build_tool_schema(skill_specs: dict[str, Any]) -> list[dict[str, Any]]:
    """Merge per-skill tools with the built-ins."""
    tools = [_skill_tool(name, spec) for name, spec in skill_specs.items()]
    return tools + BUILTIN_TOOLS


def execute_tool(
    client: ArmServiceClient,
    name: str,
    args: dict[str, Any],
    *,
    skill_names: set[str],
    dry_run: bool = False,
) -> dict[str, Any]:
    """Run a tool call. Returns a result dict; control tools add a ``_control`` key."""
    args = dict(args or {})
    obj = str(args.get("object", "")).strip()
    # Route generic pick/grasp requests through pick_object. It uses the shared
    # VLM + homography + local-refinement detector and returns a size/orientation
    # aware table grasp; colour is optional, not required.
    if name in {"pick", "grasp"} and "pick_object" in skill_names:
        skill_args = {"object": obj} if obj else {}
        result = client.run_skill("pick_object", skill_args, dry_run=dry_run)
        result["_routed_from"] = name
        return result
    if name == "close_gripper" and args:
        return {
            "ok": True,
            "skill": "close_gripper",
            "reason": "skipped_extra_args",
            "note": "close_gripper takes no object args; ignoring redundant planner call",
        }
    if name == "done":
        return {"_control": "done", "summary": args.get("summary", "")}
    if name == "ask":
        return {"_control": "ask", "question": args.get("question", "")}
    if name == "scan_scene":
        return client.scene()
    if name == "run_code":
        return client.run_code(args.get("code", ""), dry_run=dry_run)
    if name == "get_state":
        try:
            return client.state()
        except Exception as exc:
            return {"ok": False, "error": str(exc)}
    if name in skill_names:
        return client.run_skill(name, args, dry_run=dry_run)
    return {"ok": False, "error": f"unknown tool '{name}'"}
