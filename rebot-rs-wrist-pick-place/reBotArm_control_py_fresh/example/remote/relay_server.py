#!/usr/bin/env python3
"""Self-hosted relay for remote B601 teleop — the ONLY component that needs a
public IP. Run it on any cloud VM; the arm node, operator node, and browser
viewers all connect *out* to it, so nothing behind a home router needs port
forwarding.

What it does, per "room" (an arbitrary session name):
  * relays operator -> arm control messages (WebSocket TEXT, target JSON)
  * relays arm -> operator status messages (WebSocket TEXT, state JSON)
  * receives the arm's camera as JPEG frames (WebSocket BINARY) and re-serves
    them to any number of browser viewers as an MJPEG stream (plain HTTP <img>)

Endpoints:
  GET  /                     landing page (enter a room, get the links)
  GET  /view?room=ROOM       browser viewer page (open this from anywhere)
  GET  /stream?room=ROOM     multipart/x-mixed-replace MJPEG (the <img> src)
  WS   /ws?room=ROOM&role=R  node link (role = arm | operator | viewer)

Run:
    pip install aiohttp
    python relay_server.py --port 8765
    # then, from anywhere:  http://YOUR_VM_IP:8765/view?room=b601
"""
from __future__ import annotations

import argparse
import asyncio
import time
import weakref
from pathlib import Path

from aiohttp import WSMsgType, web

ROLE_ARM = "arm"
ROLE_OPERATOR = "operator"
ROLE_VIEWER = "viewer"

VIEWER_HTML = Path(__file__).with_name("viewer.html")
MJPEG_BOUNDARY = "frame"


class Room:
    """Live state + connection sets for one teleop session."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.arms: weakref.WeakSet = weakref.WeakSet()
        self.operators: weakref.WeakSet = weakref.WeakSet()
        self.viewers: weakref.WeakSet = weakref.WeakSet()
        self.latest_frame: bytes | None = None
        self.frame_seq = 0
        self.last_frame_t = 0.0
        self.cond = asyncio.Condition()   # notifies MJPEG streamers of new frames

    def counts(self) -> str:
        return (f"arms={len(self.arms)} operators={len(self.operators)} "
                f"viewers={len(self.viewers)}")

    async def set_frame(self, data: bytes) -> None:
        async with self.cond:
            self.latest_frame = data
            self.frame_seq += 1
            self.last_frame_t = time.time()
            self.cond.notify_all()


class Relay:
    def __init__(self) -> None:
        self.rooms: dict[str, Room] = {}

    def room(self, name: str) -> Room:
        room = self.rooms.get(name)
        if room is None:
            room = Room(name)
            self.rooms[name] = room
        return room


relay = Relay()


# ── WebSocket node link ───────────────────────────────────────────────────────
async def ws_handler(request: web.Request) -> web.WebSocketResponse:
    room_name = request.query.get("room", "default")
    role = request.query.get("role", ROLE_VIEWER)
    room = relay.room(room_name)
    # Exactly one command source and one physical arm are allowed per room.
    # Two leaders must never be able to fight over the same follower.
    existing = room.arms if role == ROLE_ARM else room.operators if role == ROLE_OPERATOR else None
    if existing is not None and any(not peer.closed for peer in list(existing)):
        raise web.HTTPConflict(text=f"room {room_name!r} already has an active {role}")
    ws = web.WebSocketResponse(heartbeat=20, max_msg_size=8 * 1024 * 1024)
    await ws.prepare(request)

    reg = {ROLE_ARM: room.arms, ROLE_OPERATOR: room.operators}.get(role, room.viewers)
    reg.add(ws)
    peer = request.remote
    print(f"[relay] + {role} joined room={room_name!r} ({peer}) :: {room.counts()}")

    try:
        async for msg in ws:
            if msg.type == WSMsgType.TEXT:
                if role == ROLE_OPERATOR:
                    # operator -> all arms in the room
                    await _broadcast(room.arms, msg.data, ws)
                elif role == ROLE_ARM:
                    # arm status -> operators + ws viewers
                    await _broadcast(room.operators, msg.data, ws)
                    await _broadcast(room.viewers, msg.data, ws)
            elif msg.type == WSMsgType.BINARY:
                if role == ROLE_ARM:
                    await room.set_frame(msg.data)      # the camera frame
            elif msg.type == WSMsgType.ERROR:
                print(f"[relay] ws error room={room_name!r}: {ws.exception()}")
    finally:
        reg.discard(ws)
        print(f"[relay] - {role} left room={room_name!r} :: {room.counts()}")
    return ws


async def _broadcast(conns: weakref.WeakSet, data: str, exclude) -> None:
    dead = []
    for c in list(conns):
        if c is exclude or c.closed:
            continue
        try:
            await c.send_str(data)
        except (ConnectionError, RuntimeError):
            dead.append(c)
    for c in dead:
        conns.discard(c)


# ── MJPEG video for browser viewers ───────────────────────────────────────────
async def stream_handler(request: web.Request) -> web.StreamResponse:
    room = relay.room(request.query.get("room", "default"))
    resp = web.StreamResponse(status=200, headers={
        "Content-Type": f"multipart/x-mixed-replace; boundary={MJPEG_BOUNDARY}",
        "Cache-Control": "no-cache, no-store, must-revalidate",
        "Connection": "close",
    })
    resp.enable_chunked_encoding()
    await resp.prepare(request)

    last_seq = -1
    try:
        while True:
            # wait (holding the condition lock) for a frame newer than last_seq;
            # time out every 10s so a stalled stream loops and re-checks liveness.
            async with room.cond:
                if room.frame_seq == last_seq:
                    try:
                        await asyncio.wait_for(
                            room.cond.wait_for(lambda: room.frame_seq != last_seq),
                            timeout=10.0)
                    except asyncio.TimeoutError:
                        continue
                frame = room.latest_frame
                last_seq = room.frame_seq
            if not frame:
                continue
            await resp.write(
                b"--" + MJPEG_BOUNDARY.encode() + b"\r\n"
                b"Content-Type: image/jpeg\r\n"
                b"Content-Length: " + str(len(frame)).encode() + b"\r\n\r\n"
                + frame + b"\r\n")
    except (asyncio.TimeoutError, ConnectionResetError, asyncio.CancelledError):
        pass
    except Exception as e:   # noqa: BLE001 - keep the server alive on any client hiccup
        print(f"[relay] stream ended: {type(e).__name__}: {e}")
    return resp


# ── HTML pages ────────────────────────────────────────────────────────────────
async def view_handler(request: web.Request) -> web.Response:
    if VIEWER_HTML.exists():
        return web.Response(text=VIEWER_HTML.read_text(), content_type="text/html")
    return web.Response(text="viewer.html missing next to relay_server.py", status=500)


async def index_handler(request: web.Request) -> web.Response:
    rooms = "".join(
        f'<li><a href="/view?room={r}">{r}</a> &mdash; {rm.counts()}</li>'
        for r, rm in sorted(relay.rooms.items())) or "<li><i>none yet</i></li>"
    html = f"""<!doctype html><meta charset=utf-8>
<title>B601 remote teleop relay</title>
<style>body{{font-family:system-ui,sans-serif;max-width:640px;margin:40px auto;padding:0 16px}}
input,button{{font-size:16px;padding:8px}}</style>
<h1>B601 remote teleop relay</h1>
<p>Open a room to watch the live teleop:</p>
<form action="/view"><input name=room placeholder=room value=b601>
<button>Watch</button></form>
<h3>Active rooms</h3><ul>{rooms}</ul>
<p style="color:#666">Nodes connect to
<code>ws://THIS_HOST/ws?room=ROOM&amp;role=arm|operator</code>.</p>"""
    return web.Response(text=html, content_type="text/html")


async def health_handler(request: web.Request) -> web.Response:
    return web.json_response({"ok": True, "rooms": {
        r: {"arms": len(rm.arms), "operators": len(rm.operators),
            "viewers": len(rm.viewers), "fps_frame_age_s": round(time.time() - rm.last_frame_t, 1)
            if rm.last_frame_t else None}
        for r, rm in relay.rooms.items()}})


def make_app() -> web.Application:
    app = web.Application()
    app.add_routes([
        web.get("/", index_handler),
        web.get("/view", view_handler),
        web.get("/stream", stream_handler),
        web.get("/ws", ws_handler),
        web.get("/health", health_handler),
    ])
    return app


def main() -> None:
    ap = argparse.ArgumentParser(description="Self-hosted relay for remote teleop")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8765)
    args = ap.parse_args()
    print(f"[relay] listening on http://{args.host}:{args.port}  "
          f"(viewer: /view?room=ROOM)")
    web.run_app(make_app(), host=args.host, port=args.port, print=None)


if __name__ == "__main__":
    main()
