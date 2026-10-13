"""Live DepthAnything viewer + bbox-on-depth localizer for the workspace camera.

Shows, side by side in your browser, the RGB frame and the DepthAnything-V2 depth
map, with LocateAnything detection boxes drawn on BOTH and each object's depth
sampled from the depth map inside its box (the "impose the bbox on the depth"
logic). Serves an MJPEG stream so you can watch it live.

Camera source (default = Reachy bridge; falls back to a local webcam):
    python -m arm_agent.depth_viewer                 # Reachy bridge :8712, head=grasp_view
    python -m arm_agent.depth_viewer --webcam 0      # local /dev/video0
    python -m arm_agent.depth_viewer --port 8090
then open http://127.0.0.1:8090 in a browser.

Detection is slow (~2 s) so it runs every --detect-every seconds; depth runs every
frame (~50-100 ms on GPU). Query objects with --objects "red marker,green cube".
"""

from __future__ import annotations

import argparse
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "rebot_arm_service" / "src"))
sys.path.insert(0, str(ROOT))

from rebot_arm_service.scene.depth_anything import DepthEstimator


class _Source:
    """Frame source: Reachy bridge or a local webcam."""

    def __init__(self, webcam: int | None, bridge_url: str, head_view: str | None):
        self.webcam = webcam
        if webcam is None:
            from arm_agent.reachy_client import ReachyBridgeClient
            self.reachy = ReachyBridgeClient(base_url=bridge_url)
            self.head_view = head_view
            self._aimed = False
        else:
            self.cap = cv2.VideoCapture(webcam, cv2.CAP_V4L2)
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    def read(self):
        if self.webcam is None:
            if not self._aimed and self.head_view:
                try:
                    self.reachy.head(preset=self.head_view, duration=1.0)
                    self._aimed = True
                except Exception:
                    pass
            return self.reachy.frame_bgr(warmup=2)
        ok, f = self.cap.read()
        return f if ok else None


class _State:
    def __init__(self):
        self.jpeg: bytes | None = None
        self.lock = threading.Lock()


def _colorize(depth: np.ndarray) -> np.ndarray:
    d = depth.astype(np.float32)
    lo, hi = np.percentile(d, 2), np.percentile(d, 98)
    dn = np.clip((d - lo) / max(1e-6, hi - lo), 0, 1)
    return cv2.applyColorMap((dn * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)


def _worker(state: _State, src: _Source, objects: list[str], detect_every: float):
    de = DepthEstimator()
    la = None
    if objects:
        from rebot_arm_service.scene import LocateAnythingLocator
        la = LocateAnythingLocator()
    dets: list[tuple[str, list[float]]] = []
    last_detect = 0.0
    while True:
        frame = src.read()
        if frame is None:
            time.sleep(0.1)
            continue
        depth = de.depth(frame)
        dcol = _colorize(depth)
        if la is not None and time.time() - last_detect > detect_every:
            new = []
            for obj in objects:
                try:
                    r = la.locate(frame, obj)
                    if r.get("found"):
                        new.append((obj, r["box"]))
                except Exception:
                    pass
            dets = new
            last_detect = time.time()
        rgb = frame.copy()
        for name, b in dets:
            d = DepthEstimator.box_depth(depth, b)
            bi = [int(v) for v in b]
            for canvas, tag in ((rgb, f"{name}"), (dcol, f"{name} d={d:.1f}")):
                cv2.rectangle(canvas, (bi[0], bi[1]), (bi[2], bi[3]), (0, 255, 0), 2)
                cv2.putText(canvas, tag, (bi[0], max(16, bi[1] - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        h = min(rgb.shape[0], dcol.shape[0])
        comp = np.hstack([cv2.resize(rgb, (int(rgb.shape[1] * h / rgb.shape[0]), h)),
                          cv2.resize(dcol, (int(dcol.shape[1] * h / dcol.shape[0]), h))])
        cv2.putText(comp, "RGB", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        cv2.putText(comp, "DepthAnything (bright=near)", (comp.shape[1] // 2 + 10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        ok, buf = cv2.imencode(".jpg", comp, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        if ok:
            with state.lock:
                state.jpeg = buf.tobytes()


def _make_handler(state: _State):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_GET(self):
            if self.path == "/":
                self.send_response(200)
                self.send_header("Content-Type", "text/html")
                self.end_headers()
                self.wfile.write(b"<html><body style='margin:0;background:#111'>"
                                 b"<img src='/stream' style='width:100%'></body></html>")
                return
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.end_headers()
            while True:
                with state.lock:
                    jpg = state.jpeg
                if jpg:
                    self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n")
                time.sleep(0.08)
    return H


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--webcam", type=int, default=None, help="use /dev/videoN instead of the Reachy bridge")
    ap.add_argument("--bridge", default="http://127.0.0.1:8712")
    ap.add_argument("--head-view", default="grasp_view")
    ap.add_argument("--objects", default="", help="comma-separated things to detect+measure depth for")
    ap.add_argument("--detect-every", type=float, default=3.0)
    ap.add_argument("--port", type=int, default=8090)
    args = ap.parse_args()

    objects = [o.strip() for o in args.objects.split(",") if o.strip()]
    state = _State()
    src = _Source(args.webcam, args.bridge, args.head_view)
    threading.Thread(target=_worker, args=(state, src, objects, args.detect_every), daemon=True).start()
    print(f"open http://127.0.0.1:{args.port}  (source={'webcam '+str(args.webcam) if args.webcam is not None else 'reachy bridge'})")
    ThreadingHTTPServer(("127.0.0.1", args.port), _make_handler(state)).serve_forever()


if __name__ == "__main__":
    main()
