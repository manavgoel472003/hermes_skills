"""Opt-in mission context shared by wrist skills and the local motion sender.

No hardware imports. Outside a mission every hook is a no-op. Cancellation is a
BaseException so existing `except Exception` fallbacks cannot turn a stop into
another motion candidate. Wrist recovery handles it by freezing, without release.
"""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
import threading
import time
from typing import Any, Callable, Iterator


class MissionCancelled(BaseException):
    pass


@dataclass
class MissionControl:
    dry_run: bool = False
    carry_via_m: tuple[tuple[float, float, float], ...] = ()
    notify: Callable[[str, dict[str, Any]], None] | None = None
    stop: threading.Event = field(default_factory=threading.Event)
    reason: str = "operator requested stop"

    def cancel(self, reason: str = "operator requested stop") -> None:
        self.reason = reason
        self.stop.set()


_active: ContextVar[MissionControl | None] = ContextVar("robot_mission", default=None)
_suspended: ContextVar[bool] = ContextVar("robot_mission_recovery", default=False)


def current() -> MissionControl | None:
    return _active.get()


def cancelled() -> bool:
    control = current()
    return bool(control and control.stop.is_set())


def checkpoint() -> None:
    if not _suspended.get() and cancelled():
        raise MissionCancelled(current().reason)


def motion() -> None:
    """Last-mile dry-run guard before connection or actuator commands."""
    checkpoint()
    if current() is not None and current().dry_run:
        raise RuntimeError("Hybrid dry run cannot connect or command actuators")


def stage(name: str, **detail: Any) -> None:
    checkpoint()
    control = current()
    if control and control.notify:
        control.notify(name, detail)
    checkpoint()


def pause(seconds: float) -> None:
    """Interruptible wait; does not reduce required settling time."""
    checkpoint()
    control = current()
    if control and not _suspended.get():
        control.stop.wait(max(0.0, seconds))
    else:
        time.sleep(max(0.0, seconds))
    checkpoint()


@contextmanager
def activate(control: MissionControl) -> Iterator[None]:
    token = _active.set(control)
    try:
        yield
    finally:
        _active.reset(token)


@contextmanager
def recovery() -> Iterator[None]:
    """Allow measured-pose freeze/feedback while a cancellation is pending."""
    token = _suspended.set(True)
    try:
        yield
    finally:
        _suspended.reset(token)
