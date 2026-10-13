"""Planner clients: Nous Hermes over Ollama, plus a scriptable mock.

A planner takes the running message history and the tool schema and returns the
assistant's next message (which may contain tool calls), in Ollama's message dict
shape so it can be appended straight back into the history.
"""

from __future__ import annotations

import os
from typing import Any, Iterable, Protocol


class Planner(Protocol):
    def chat(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> dict[str, Any]:
        ...


def iter_tool_calls(message: dict[str, Any]) -> list[dict[str, Any]]:
    """Normalise a message's tool calls to [{'name':..., 'arguments': {...}}]."""
    calls = message.get("tool_calls") or []
    out: list[dict[str, Any]] = []
    for c in calls:
        fn = c.get("function", c) if isinstance(c, dict) else c
        name = fn.get("name")
        args = fn.get("arguments", {})
        if isinstance(args, str):
            import json
            try:
                args = json.loads(args)
            except Exception:
                args = {}
        out.append({"name": name, "arguments": args or {}})
    return out


class OllamaHermesClient:
    def __init__(self, model: str | None = None, host: str | None = None,
                 temperature: float = 0.2, timeout: float = 180.0) -> None:
        self.model = model or os.getenv("REBOT_HERMES_MODEL", "hermes3")
        self.host = host or os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")
        if not self.host.startswith("http"):
            self.host = "http://" + self.host
        self.temperature = temperature
        self.timeout = timeout
        self._client = None

    def _get_client(self):
        if self._client is None:
            import ollama
            self._client = ollama.Client(host=self.host, timeout=self.timeout)
        return self._client

    def chat(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> dict[str, Any]:
        client = self._get_client()
        resp = client.chat(
            model=self.model,
            messages=messages,
            tools=tools,
            options={"temperature": self.temperature},
        )
        msg = resp["message"]
        # ollama returns a pydantic Message; normalise to a plain dict
        if hasattr(msg, "model_dump"):
            return msg.model_dump()
        return dict(msg)

    def health(self) -> dict[str, Any]:
        try:
            client = self._get_client()
            tags = client.list()
            names = [m.get("model") or m.get("name") for m in tags.get("models", [])]
            present = any((n or "").split(":")[0] == self.model.split(":")[0] for n in names)
            return {"ok": True, "model": self.model, "host": self.host,
                    "pulled": present, "available_models": names}
        except Exception as exc:
            return {"ok": False, "error": str(exc), "model": self.model, "host": self.host}


class MockPlanner:
    """Replays a scripted sequence of assistant messages. For tests/offline use.

    ``script`` is a list of messages; each is either {'content': str} or
    {'tool_calls': [{'name':..., 'arguments': {...}}]} (sugar that is expanded to
    the Ollama function-call shape).
    """

    def __init__(self, script: Iterable[dict[str, Any]]) -> None:
        self._script = list(script)
        self._i = 0

    def chat(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> dict[str, Any]:
        if self._i >= len(self._script):
            return {"role": "assistant", "content": "",
                    "tool_calls": [{"function": {"name": "done", "arguments": {"summary": "script ended"}}}]}
        step = self._script[self._i]
        self._i += 1
        msg: dict[str, Any] = {"role": "assistant", "content": step.get("content", "")}
        if "tool_calls" in step:
            msg["tool_calls"] = [
                {"function": {"name": c["name"], "arguments": c.get("arguments", {})}}
                for c in step["tool_calls"]
            ]
        return msg
