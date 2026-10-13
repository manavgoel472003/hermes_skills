"""RGB-D bridge for the Orbbec Gemini 336 -- serves color + aligned depth +
intrinsics over localhost HTTP, so the arm stack (lerobot env, Py3.12) can use the
camera even though the Orbbec SDK only installs in the wall-touch-demo venv
(Py3.13). Mirrors the Reachy camera-bridge pattern.

Run WITH the wall-touch-demo venv (has pyorbbecsdk2):
    /home/manav/rebot_lerobot/wall-touch-demo/.venv/bin/python \
        /home/manav/rebot_lerobot/arm_agent/orbbec_bridge.py --port 8720

Endpoints:
    GET /health          -> {ok, intrinsics:{fx,fy,cx,cy,w,h}, stream}
    GET /frame.npz       -> npz bytes: color(HxWx3 uint8 BGR), depth(HxW float32 mm),
                            fx,fy,cx,cy  (a fresh synchronized, HW-aligned frame)
"""
from __future__ import annotations

import argparse
import io
import os
import sys
import threading
from pathlib import Path
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

# The Orbbec SDK is installed in LightTouch-OpenCV's Python 3.13 virtualenv.
# ``wall-touch-demo`` was an earlier workspace name and no longer exists on this
# machine.  Keep this bridge self-contained so the arm process can use the
# aligned RGB-D stream through localhost without importing the SDK itself.
ORBBEC_APP_ROOT = Path(os.getenv(
    "REBOT_ORBBEC_APP_ROOT", "/home/manav/rebot_lerobot/LightTouch-OpenCV"
))
if not (ORBBEC_APP_ROOT / "wall_touch_orbbec.py").is_file():
    raise RuntimeError(
        f"Orbbec application root is invalid: {ORBBEC_APP_ROOT}; "
        "set REBOT_ORBBEC_APP_ROOT to the directory containing wall_touch_orbbec.py"
    )
sys.path.insert(0, str(ORBBEC_APP_ROOT))
from wall_touch_orbbec import OrbbecCamera


class Bridge:
    def __init__(self):
        self.cam = OrbbecCamera()
        p = self.cam.pipeline.get_camera_param().rgb_intrinsic
        self.intr = dict(fx=float(p.fx), fy=float(p.fy), cx=float(p.cx), cy=float(p.cy),
                         w=int(p.width), h=int(p.height))
        self.lock = threading.Lock()
        # warm up
        for _ in range(5):
            self.cam.read()
        print("Orbbec bridge:", self.cam.stream_description, self.intr)

    def frame_npz(self) -> bytes:
        with self.lock:
            f = self.cam.read()
        buf = io.BytesIO()
        np.savez_compressed(buf, color=f.color_bgr, depth=f.depth_mm.astype(np.float32),
                            fx=self.intr["fx"], fy=self.intr["fy"],
                            cx=self.intr["cx"], cy=self.intr["cy"])
        return buf.getvalue()


def make_handler(bridge: Bridge):
    import json

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, code, ctype, body: bytes):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path.startswith("/health"):
                self._send(200, "application/json",
                           json.dumps({"ok": True, "intrinsics": bridge.intr,
                                       "stream": bridge.cam.stream_description}).encode())
                return
            if self.path.startswith("/frame.npz"):
                try:
                    self._send(200, "application/octet-stream", bridge.frame_npz())
                except Exception as e:  # noqa: BLE001
                    self._send(500, "text/plain", f"frame error: {e}".encode())
                return
            self._send(404, "text/plain", b"not found")

    return H


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8720)
    args = ap.parse_args()
    bridge = Bridge()
    print(f"serving on http://127.0.0.1:{args.port}  (/health, /frame.npz)")
    ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(bridge)).serve_forever()


if __name__ == "__main__":
    main()
