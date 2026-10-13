"""Command-line front-end for the arm agent.

Examples:
  python -m arm_agent.cli                         # interactive REPL
  python -m arm_agent.cli --once "stack the red block on the blue block"
  python -m arm_agent.cli --dry-run --once "put the cup to the right of the marker"
"""

from __future__ import annotations

import argparse
import json
import sys

from .service_client import ArmServiceClient
from .hermes_client import OllamaHermesClient
from .executor import AgentSession


def _preflight(client: ArmServiceClient, planner: OllamaHermesClient) -> bool:
    ok = True
    try:
        h = client.healthz()
        print(f"service: ok (arm connected={h.get('connected')})")
    except Exception as exc:
        print(f"service: UNREACHABLE at {client.base_url} ({exc})")
        return False
    try:
        cal = client.calibration()
        print(f"camera calibration: {'present' if cal.get('calibrated') else 'MISSING (run scene.calibrate)'}")
    except Exception as exc:
        print(f"camera calibration: unknown ({exc})")
    vh = planner.health()
    if vh.get("ok") and vh.get("pulled"):
        print(f"planner: {planner.model} ready on {planner.host}")
    elif vh.get("ok"):
        print(f"planner: WARNING model '{planner.model}' not pulled. Run: ollama pull {planner.model}")
        ok = False
    else:
        print(f"planner: Ollama unreachable ({vh.get('error')}). Run: ollama serve")
        ok = False
    return ok


def _run_task(session: AgentSession, task: str) -> None:
    result = session.run(task)
    while result.get("status") == "ask":
        print(f"\n[arm asks] {result['question']}")
        try:
            answer = input("you> ").strip()
        except EOFError:
            answer = "stop"
        result = session.provide_answer(answer)
    print(f"\n[{result['status']}] {result.get('summary', '')}")
    if result.get("executed"):
        print("steps:")
        for s in result["executed"]:
            print(f"  - {s['skill']}({json.dumps(s['args'])}) -> ok={s['ok']} {s.get('reason','')}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Hermes-driven reBot arm agent.")
    ap.add_argument("--url", default=None, help="rebot_arm_service base URL.")
    ap.add_argument("--model", default=None, help="Ollama model for the planner (default hermes3).")
    ap.add_argument("--once", default=None, help="Run a single task and exit.")
    ap.add_argument("--dry-run", action="store_true", help="Plan/validate skills without moving the arm.")
    ap.add_argument("--max-steps", type=int, default=24)
    args = ap.parse_args(argv)

    client = ArmServiceClient(base_url=args.url)
    planner = OllamaHermesClient(model=args.model)
    ready = _preflight(client, planner)
    if not ready and not args.once:
        print("\nFix the warnings above, or continue at your own risk.")

    session = AgentSession(client=client, planner=planner, dry_run=args.dry_run, max_steps=args.max_steps)

    if args.once:
        _run_task(session, args.once)
        return 0

    print("\nType a task (or 'reset' to clear memory, 'quit' to exit). Ctrl-D to quit.")
    while True:
        try:
            task = input("\ntask> ").strip()
        except EOFError:
            break
        if not task:
            continue
        if task.lower() in {"quit", "exit"}:
            break
        if task.lower() == "reset":
            session = AgentSession(client=client, planner=planner, dry_run=args.dry_run, max_steps=args.max_steps)
            print("(memory cleared)")
            continue
        try:
            _run_task(session, task)
        except Exception as exc:
            print(f"error: {exc}")
    client.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
