"""Shared wire protocol + helpers for the remote (over-the-internet) teleop.

Three programs speak this protocol, all connecting *out* to one self-hosted
relay (so nothing needs port-forwarding / a public IP except the relay):

    operator_node.py  --(WS: target JSON)-->  relay_server.py  <--(WS: JPEG + state)--  arm_node.py
                                                    |
                                                    +--(HTTP MJPEG)-->  browser viewers

The link is a single WebSocket per node to
    ws://RELAY_HOST:PORT/ws?room=ROOM&role=ROLE
carrying:
  * TEXT frames  = JSON control/state messages (see TYPE_* below)
  * BINARY frames = a single JPEG image (arm node -> relay, the video)

Everything here is dependency-light (numpy + cv2 only) so both the arm venv
(Py3.10 CAN stack) and the operator venv can import it unchanged.
"""
from __future__ import annotations

import json
import time
import uuid
from urllib.parse import quote

import cv2
import numpy as np

# ── Defaults ────────────────────────────────────────────────────────────────
DEFAULT_PORT = 8765
DEFAULT_ROOM = "b601"
JPEG_QUALITY = 70            # 0-100; 70 is a good size/quality tradeoff for a demo

# ── Message types (the "type" field of a JSON text frame) ─────────────────────
TYPE_TARGET = "target"       # operator -> arm : desired EE pose + engage/grip (gesture path)
TYPE_ACTION = "action"       # leader -> follower : lerobot joint-position action dict
TYPE_STATE = "state"         # arm/follower -> operator/viewers : live status
TYPE_HELLO = "hello"         # node -> relay : announce role (also in query string)
TYPE_PING = "ping"
PROTOCOL_VERSION = 2


def new_session_id() -> str:
    """Return a short process-unique id used to reject stale/out-of-order actions."""
    return uuid.uuid4().hex


def action_msg(action: dict, t: float | None = None, *, seq: int = 0,
               session: str = "legacy") -> str:
    """Leader -> follower: a lerobot action dict, e.g. {'shoulder_pan.pos': 12.3, …}
    (motor positions in degrees). Kept as plain floats so it is JSON-safe."""
    return json.dumps({
        "type": TYPE_ACTION,
        "v": PROTOCOL_VERSION,
        "session": str(session),
        "seq": int(seq),
        "a": {k: float(v) for k, v in action.items()},
        "t": float(time.time() if t is None else t),
    })


def build_ws_url(relay: str, room: str, role: str) -> str:
    """Turn a relay base ('host:port', 'ws://host:port', 'wss://host') plus a
    room + role into a full WebSocket URL. Accepts bare host:port for convenience.
    """
    r = relay.strip()
    if not r.startswith(("ws://", "wss://")):
        # allow http(s):// too, map to ws(s)://
        if r.startswith("http://"):
            r = "ws://" + r[len("http://"):]
        elif r.startswith("https://"):
            r = "wss://" + r[len("https://"):]
        else:
            r = "ws://" + r
    r = r.rstrip("/")
    return f"{r}/ws?room={quote(room)}&role={quote(role)}"


def target_msg(x: float, y: float, z: float, grip: float, engaged: bool,
               t: float) -> str:
    return json.dumps({
        "type": TYPE_TARGET,
        "x": float(x), "y": float(y), "z": float(z),
        "grip": float(grip), "engaged": bool(engaged), "t": float(t),
    })


def state_msg(connected: bool, engaged: bool, ee, target, err_mm: float,
              t: float) -> str:
    return json.dumps({
        "type": TYPE_STATE,
        "connected": bool(connected), "engaged": bool(engaged),
        "ee": [round(float(v), 4) for v in ee],
        "target": [round(float(v), 4) for v in target],
        "err_mm": round(float(err_mm), 1), "t": float(t),
    })


def parse_msg(text: str) -> dict | None:
    try:
        m = json.loads(text)
        return m if isinstance(m, dict) else None
    except (ValueError, TypeError):
        return None


def encode_jpeg(frame: np.ndarray, quality: int = JPEG_QUALITY) -> bytes | None:
    """BGR uint8 image -> JPEG bytes (None on failure)."""
    ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    return buf.tobytes() if ok else None


def draw_hud(frame: np.ndarray, lines: list[str], active: bool,
             active_label: str = "ENGAGED", idle_label: str = "HOLD") -> np.ndarray:
    """Overlay status text on the frame so browser viewers see everything with a
    plain <img> (no extra data channel needed). Draws top-left, plus a green/blue
    active/idle chip top-right. Mutates and returns the frame."""
    h, w = frame.shape[:2]
    # translucent banner
    band_h = 22 * len(lines) + 12
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, band_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)
    for i, ln in enumerate(lines):
        cv2.putText(frame, ln, (10, 24 + i * 22), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (255, 255, 255), 1, cv2.LINE_AA)
    chip = active_label if active else idle_label
    color = (60, 220, 60) if active else (60, 200, 240)
    (tw, _), _ = cv2.getTextSize(chip, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    cv2.putText(frame, chip, (w - tw - 12, 26), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, color, 2, cv2.LINE_AA)
    return frame
