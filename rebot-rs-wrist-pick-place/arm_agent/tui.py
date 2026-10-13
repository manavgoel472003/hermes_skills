"""Rich terminal UI for the Hermes arm agent.

A friendlier front-end than ``cli.py`` that *shows its work*: the planner's
thoughts, every tool call with its arguments, what ``scan_scene`` actually saw
(as a table), and per-step progress with pass/fail. It attaches to
``AgentSession`` through the session's event observer hook, so the control loop
is unchanged.

Run:
    python -m arm_agent.tui                          # interactive
    python -m arm_agent.tui --model qwen3.5:9b
    python -m arm_agent.tui --dry-run --once "pick up the green sponge"
"""

from __future__ import annotations

import argparse
import sys
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from .service_client import ArmServiceClient
from .hermes_client import OllamaHermesClient
from .executor import AgentSession

# Built-in tools that are not physical skills (don't show in the step recap).
_NON_SKILL_TOOLS = {"scan_scene", "get_state", "done", "ask"}


def _fmt_args(args: dict[str, Any]) -> str:
    """Compact k=v rendering of tool arguments."""
    if not args:
        return ""
    parts = []
    for k, v in args.items():
        if isinstance(v, float):
            v = round(v, 3)
        parts.append(f"{k}={v!r}" if isinstance(v, str) else f"{k}={v}")
    return ", ".join(parts)


class TaskView:
    """Renders one task's loop events to the console and tracks step results."""

    def __init__(self, console: Console, task: str, dry_run: bool) -> None:
        self.console = console
        self.task = task
        self.dry_run = dry_run
        self.steps: list[dict[str, Any]] = []
        self.status: Any = None  # rich Status, set by the runner
        self._n = 0

    # -- the observer entry point ------------------------------------------
    def on_event(self, event: str, payload: dict[str, Any]) -> None:
        handler = getattr(self, f"_on_{event}", None)
        if handler:
            handler(payload)

    def _on_task(self, p: dict[str, Any]) -> None:
        mode = "[yellow]DRY-RUN[/]" if self.dry_run else "[green]LIVE[/]"
        self.console.print(
            Panel(Text(self.task, style="bold white"), title=f"task · {mode}",
                  border_style="blue", title_align="left")
        )

    def _on_planner(self, p: dict[str, Any]) -> None:
        thought = (p.get("content") or "").strip()
        if thought:
            self.console.print(Text(f"🧠 {thought}", style="dim italic"))
        calls = p.get("tool_calls") or []
        if calls and self.status is not None:
            self.status.update(f"[cyan]running {', '.join(calls)}…[/]")

    def _on_tool_call(self, p: dict[str, Any]) -> None:
        self._n += 1
        name, args = p.get("name", "?"), _fmt_args(p.get("args") or {})
        self.console.print(Text.assemble(
            (f"  {self._n:>2} ", "dim"), ("🔧 ", ""),
            (name, "bold cyan"), (f"({args})", "cyan"),
        ))

    def _on_scene(self, p: dict[str, Any]) -> None:
        objects = p.get("objects") or []
        t = Table(title=f"👁  scan_scene → {len(objects)} object(s)",
                  title_style="magenta", border_style="magenta", title_justify="left")
        t.add_column("label", style="bold")
        t.add_column("position (x, y, z) m", style="green")
        t.add_column("size (m)", style="dim")
        for o in objects:
            pos = o.get("position_m") or o.get("position") or []
            size = o.get("size_m") or o.get("size") or []
            pos_s = ", ".join(f"{v:+.3f}" for v in pos) if pos else "—"
            size_s = " × ".join(f"{v:.3f}" for v in size) if size else "—"
            t.add_row(str(o.get("label", "?")), pos_s, size_s)
        if not objects:
            t.add_row("[dim]nothing detected[/]", "—", "—")
        self.console.print(t)

    def _on_tool_result(self, p: dict[str, Any]) -> None:
        name = p.get("name", "?")
        control = p.get("control")
        if name == "scan_scene" or control in {"done", "ask"}:
            return  # scene rendered separately; done/ask shown in final
        ok = p.get("ok")
        reason = p.get("reason") or ""
        if name not in _NON_SKILL_TOOLS:
            self.steps.append({"skill": p.get("skill") or name, "ok": ok, "reason": reason})
        if ok:
            self.console.print(Text.assemble(("     ✓ ", "bold green"),
                                              (f"{name}", "green"), (f"  {reason}", "dim")))
        else:
            self.console.print(Text.assemble(("     ✗ ", "bold red"),
                                              (f"{name}", "red"), (f"  {reason}", "yellow")))

    def _on_final(self, p: dict[str, Any]) -> None:
        status = p.get("status", "?")
        if status == "ask":
            return  # handled by the runner's prompt loop
        ok_n = sum(1 for s in self.steps if s["ok"])
        recap = Table.grid(padding=(0, 1))
        recap.add_column()
        recap.add_column()
        for s in self.steps:
            mark = "[green]✓[/]" if s["ok"] else "[red]✗[/]"
            recap.add_row(mark, f"{s['skill']} [dim]{s['reason']}[/]")
        body = Text(p.get("summary", ""), style="bold")
        border = {"done": "green", "max_steps": "red"}.get(status, "yellow")
        self.console.print(Panel(body, title=f"{status} · {ok_n}/{len(self.steps)} steps ok",
                                 border_style=border, title_align="left"))
        if self.steps:
            self.console.print(recap)


def _preflight(console: Console, client: ArmServiceClient, planner: OllamaHermesClient) -> bool:
    rows = Table.grid(padding=(0, 2))
    rows.add_column(style="bold")
    rows.add_column()
    ok = True
    try:
        h = client.healthz()
        rows.add_row("service", f"[green]ok[/] (arm connected={h.get('connected')})")
    except Exception as exc:
        rows.add_row("service", f"[red]UNREACHABLE[/] at {client.base_url} ({exc})")
        console.print(Panel(rows, title="preflight", border_style="red"))
        return False
    try:
        cal = client.calibration()
        rows.add_row("camera", "[green]calibrated[/]" if cal.get("calibrated") else "[red]NOT calibrated[/]")
    except Exception as exc:
        rows.add_row("camera", f"[yellow]unknown[/] ({exc})")
    vh = planner.health()
    if vh.get("ok") and vh.get("pulled"):
        rows.add_row("planner", f"[green]{planner.model}[/] ready on {planner.host}")
    elif vh.get("ok"):
        rows.add_row("planner", f"[yellow]model '{planner.model}' not pulled[/] — ollama pull {planner.model}")
        ok = False
    else:
        rows.add_row("planner", f"[red]Ollama unreachable[/] ({vh.get('error')})")
        ok = False
    console.print(Panel(rows, title="preflight", border_style="blue" if ok else "yellow"))
    return ok


def _run_task(console: Console, session: AgentSession, task: str, dry_run: bool) -> None:
    view = TaskView(console, task, dry_run)
    session.observer = view.on_event
    with console.status("[cyan]Hermes thinking…[/]", spinner="dots") as status:
        view.status = status
        result = session.run(task)
    # ``ask`` needs user input, which must happen outside the spinner.
    while result.get("status") == "ask":
        console.print(Panel(Text(result["question"], style="yellow"), title="arm asks", border_style="yellow"))
        try:
            answer = console.input("[bold yellow]you ›[/] ").strip()
        except EOFError:
            console.print("[dim](no input — stopping)[/]")
            break
        if answer.lower() in {"stop", "quit", "cancel"}:
            break
        with console.status("[cyan]Hermes thinking…[/]", spinner="dots") as status:
            view.status = status
            result = session.provide_answer(answer)
    session.observer = None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Hermes arm agent — rich TUI.")
    ap.add_argument("--url", default=None, help="rebot_arm_service base URL.")
    ap.add_argument("--model", default=None, help="Ollama planner model (default hermes3; recommend qwen3.5:9b).")
    ap.add_argument("--once", default=None, help="Run a single task and exit.")
    ap.add_argument("--dry-run", action="store_true", help="Plan/validate IK without moving the arm.")
    ap.add_argument("--max-steps", type=int, default=24)
    args = ap.parse_args(argv)

    console = Console()
    console.print(Panel.fit("[bold]reBot B601 · Hermes control[/]", border_style="blue"))

    client = ArmServiceClient(base_url=args.url)
    planner = OllamaHermesClient(model=args.model)
    ready = _preflight(console, client, planner)
    if not ready and not args.once:
        console.print("[yellow]Fix the warnings above, or continue at your own risk.[/]")

    def new_session() -> AgentSession:
        return AgentSession(client=client, planner=planner, dry_run=args.dry_run, max_steps=args.max_steps)

    session = new_session()

    if args.once:
        _run_task(console, session, args.once, args.dry_run)
        client.close()
        return 0

    console.print("\nType a task. Commands: [bold]reset[/] (clear memory), [bold]quit[/]. Ctrl-D to exit.")
    while True:
        try:
            task = console.input("\n[bold blue]task ›[/] ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not task:
            continue
        if task.lower() in {"quit", "exit"}:
            break
        if task.lower() == "reset":
            session = new_session()
            console.print("[dim](memory cleared)[/]")
            continue
        try:
            _run_task(console, session, task, args.dry_run)
        except Exception as exc:
            console.print(f"[red]error:[/] {exc}")
    client.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
