"""Tiny shared WebSocket client for the relay link, used by both nodes.

Runs a background reader thread that auto-reconnects, and exposes thread-safe
send_text / send_frame. Incoming TEXT frames are handed to an on_text callback.
Wraps websocket-client (`pip install websocket-client`).
"""
from __future__ import annotations

import threading
import time
from typing import Callable

import websocket
from websocket import ABNF


class RelayLink:
    def __init__(self, url: str, on_text: Callable[[str], None] | None = None,
                 name: str = "relay") -> None:
        self.url = url
        self.on_text = on_text
        self.name = name
        self.ws: websocket.WebSocketApp | None = None
        self._send_lock = threading.Lock()
        self.connected = threading.Event()
        self._stop = False

    # ── callbacks ──
    def _on_open(self, ws):
        self.connected.set()
        print(f"[{self.name}] relay connected: {self.url}")

    def _on_close(self, ws, code, msg):
        self.connected.clear()
        print(f"[{self.name}] relay disconnected (code={code}); reconnecting…")

    def _on_error(self, ws, err):
        print(f"[{self.name}] relay error: {err}")

    def _on_message(self, ws, message):
        if isinstance(message, (bytes, bytearray)):
            return
        if self.on_text is not None:
            self.on_text(message)

    # ── lifecycle ──
    def run_forever(self):
        while not self._stop:
            self.ws = websocket.WebSocketApp(
                self.url, on_open=self._on_open, on_close=self._on_close,
                on_error=self._on_error, on_message=self._on_message)
            try:
                self.ws.run_forever(ping_interval=20, ping_timeout=10)
            except Exception as e:   # noqa: BLE001
                print(f"[{self.name}] ws loop error: {e}")
            if not self._stop:
                time.sleep(2.0)

    def start(self) -> threading.Thread:
        t = threading.Thread(target=self.run_forever, name=self.name, daemon=True)
        t.start()
        return t

    def wait_connected(self, timeout: float) -> bool:
        return self.connected.wait(timeout=max(0.0, timeout))

    def send_frame(self, jpeg: bytes) -> bool:
        return self._send(jpeg, ABNF.OPCODE_BINARY)

    def send_text(self, text: str) -> bool:
        return self._send(text, ABNF.OPCODE_TEXT)

    def _send(self, data, opcode) -> bool:
        if not self.connected.is_set() or self.ws is None:
            return False
        try:
            with self._send_lock:
                self.ws.send(data, opcode=opcode)
            return True
        except Exception:   # noqa: BLE001 - reader thread reconnects
            self.connected.clear()
            return False

    def stop(self):
        self._stop = True
        if self.ws is not None:
            try:
                self.ws.close()
            except Exception:   # noqa: BLE001
                pass
