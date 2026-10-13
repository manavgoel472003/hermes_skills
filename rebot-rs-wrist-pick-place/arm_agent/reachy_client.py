"""Arm-side client to the Reachy service (fully separated Reachy).

All Reachy hardware -- camera AND head -- lives behind ONE service running in
``reachy_mini_env`` (``reachy_camera_bridge.py``). The arm stack (this env) never
touches Reachy devices or the ``reachy_mini`` SDK; it only speaks HTTP:

  GET  /health                          -> {ok, head_pose, camera}
  GET  /frame.jpg[?view=grasp_view]     -> JPEG (optionally move head first)
  POST /head  {preset | yaw/pitch/... | look_pixel:[u,v]}

This keeps the two systems cleanly decoupled and sidesteps the dependency
conflict (reachy_mini needs huggingface_hub>=1.17; transformers needs <1.0).

Env:
  REACHY_BRIDGE_URL     Reachy service base URL (default http://127.0.0.1:8712)
  REACHY_BRIDGE_TOKEN   optional bearer token
  REACHY_MOCK=1         no network; return mock frames/poses (dry-run)
"""

from __future__ import annotations

import os
from typing import Any

import httpx

# Named head poses the service understands. grasp_view is the single fixed pose
# the pixel->base calibration is tied to.
HEAD_PRESETS = ("grasp_view", "neutral", "scan_left", "scan_center", "scan_right", "look_person")


class ReachyBridgeClient:
    def __init__(self, base_url: str | None = None, token: str | None = None, mock: bool | None = None) -> None:
        self.base_url = (base_url or os.getenv("REACHY_BRIDGE_URL", "http://127.0.0.1:8712")).rstrip("/")
        self.token = token or os.getenv("REACHY_BRIDGE_TOKEN")
        self.mock = (os.getenv("REACHY_MOCK", "0").strip().lower() in {"1", "true", "yes"}) if mock is None else mock
        self._headers = {"Authorization": f"Bearer {self.token}"} if self.token else {}

    # -- availability -------------------------------------------------------
    def available(self) -> bool:
        if self.mock:
            return True
        try:
            httpx.get(f"{self.base_url}/health", headers=self._headers, timeout=3.0).raise_for_status()
            return True
        except Exception:
            return False

    def health(self) -> dict[str, Any]:
        if self.mock:
            return {"ok": True, "mock": True, "head_pose": {"preset": "neutral"}, "camera": [1280, 720]}
        r = httpx.get(f"{self.base_url}/health", headers=self._headers, timeout=5.0)
        r.raise_for_status()
        return r.json()

    # -- head control (remote) ---------------------------------------------
    def head(self, *, preset: str | None = None, pose: dict[str, float] | None = None,
             look_pixel: tuple[float, float] | None = None, duration: float = 0.7) -> dict[str, Any]:
        """Move the head: a named preset, an explicit pose (deg), or a proportional
        servo toward an image pixel (hand tracking)."""
        if self.mock:
            return {"ok": True, "mock": True, "head_pose": {"preset": preset, "pose": pose, "look_pixel": look_pixel}}
        body: dict[str, Any] = {"duration": duration}
        if preset is not None:
            body["preset"] = preset
        elif look_pixel is not None:
            body["look_pixel"] = list(look_pixel)
        elif pose is not None:
            body.update(pose)
        # The Reachy daemon occasionally drops its hardware link for a beat
        # ("Lost connection with the server" -> HTTP 500); retry a few times so a
        # transient hiccup doesn't crash a run that's mid-pick.
        import time as _time
        last: Exception | None = None
        for attempt in range(4):
            try:
                r = httpx.post(f"{self.base_url}/head", json=body, headers=self._headers, timeout=15.0)
                r.raise_for_status()
                return r.json()
            except Exception as exc:  # noqa: BLE001
                last = exc
                _time.sleep(0.6 * (attempt + 1))
        raise last  # type: ignore[misc]

    # -- camera (remote) ----------------------------------------------------
    def frame_jpeg(self, *, view: str | None = None, warmup: int = 4) -> bytes:
        """Fetch a JPEG frame, optionally moving the head to ``view`` first."""
        if self.mock:
            return _mock_jpeg()
        params: dict[str, Any] = {"warmup": warmup}
        if view is not None:
            params["view"] = view
        r = httpx.get(f"{self.base_url}/frame.jpg", params=params, headers=self._headers, timeout=15.0)
        r.raise_for_status()
        return r.content

    def frame_bgr(self, *, view: str | None = None, warmup: int = 4):
        """Decode the fetched frame to a BGR numpy array (for the VLM/calibration)."""
        import cv2
        import numpy as np
        if self.mock:
            return np.full((480, 640, 3), 127, np.uint8)
        return cv2.imdecode(np.frombuffer(self.frame_jpeg(view=view, warmup=warmup), np.uint8), cv2.IMREAD_COLOR)


def _mock_jpeg() -> bytes:
    """A small valid gray JPEG for dry-run/mock paths (encoded on the fly)."""
    import cv2
    import numpy as np
    ok, buf = cv2.imencode(".jpg", np.full((480, 640, 3), 127, np.uint8))
    return buf.tobytes() if ok else b""
