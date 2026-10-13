"""In-process RGB-D camera backends for :mod:`arm_agent.depth_servo`.

``realsense`` is the default because ``pyrealsense2`` has Python 3.12 wheels and
therefore runs wholly inside ``rebot_rs``.  ``orbbec`` deliberately remains a
small HTTP bridge fallback: its installed SDK is built for a different Python ABI.
Both backends return BGR colour, millimetre depth and (fx, fy, cx, cy).
"""
from __future__ import annotations

import io
import os
import sys
import time
from dataclasses import dataclass
from typing import Protocol

import httpx
import numpy as np


Intrinsics = tuple[float, float, float, float]


class DepthCamera(Protocol):
    def frame(self) -> tuple[np.ndarray, np.ndarray, Intrinsics]: ...

    def close(self) -> None: ...


@dataclass
class OrbbecBridgeCamera:
    """Compatibility client for ``arm_agent.orbbec_bridge``."""

    url: str = os.getenv("REBOT_ORBBEC_URL", "http://127.0.0.1:8720")

    def frame(self) -> tuple[np.ndarray, np.ndarray, Intrinsics]:
        response = httpx.get(f"{self.url.rstrip('/')}/frame.npz", timeout=15)
        response.raise_for_status()
        data = np.load(io.BytesIO(response.content))
        return (
            data["color"], data["depth"].astype(np.float32, copy=False),
            (float(data["fx"]), float(data["fy"]), float(data["cx"]), float(data["cy"])),
        )

    def close(self) -> None:
        pass


class RealSenseCamera:
    """Aligned RealSense colour/depth stream owned by this Python process."""

    def __init__(self) -> None:
        try:
            import pyrealsense2 as rs
        except ImportError as exc:  # pragma: no cover - environment-specific
            raise RuntimeError(
                f"pyrealsense2 is not installed in the active Python ({sys.executable}); "
                "run this command with /home/manav/miniconda3/envs/rebot_rs/bin/python"
            ) from exc

        self._rs = rs
        width = int(os.getenv("REBOT_REALSENSE_WIDTH", "640"))
        height = int(os.getenv("REBOT_REALSENSE_HEIGHT", "480"))
        fps = int(os.getenv("REBOT_REALSENSE_FPS", "30"))
        serial = os.getenv("REBOT_REALSENSE_SERIAL")
        # Fail synchronously, with a useful message, before a caller starts any
        # robot motion.  In particular, do not leave device discovery to a
        # background video thread where the exception can be invisible.
        devices = rs.context().query_devices()
        if len(devices) == 0:
            raise RuntimeError(
                "no RealSense camera is connected; connect the wrist camera "
                "and rerun (no robot motion was started)"
            )
        if serial:
            connected_serials = {
                device.get_info(rs.camera_info.serial_number)
                for device in devices
            }
            if serial not in connected_serials:
                raise RuntimeError(
                    f"configured RealSense serial {serial!r} is not connected; "
                    f"available serials: {sorted(connected_serials)}"
                )
        config = rs.config()
        if serial:
            config.enable_device(serial)
        config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
        self._pipeline = rs.pipeline()
        profile = self._pipeline.start(config)
        self._align = rs.align(rs.stream.color)
        depth_sensor = profile.get_device().first_depth_sensor()
        self._depth_scale_mm = float(depth_sensor.get_depth_scale()) * 1000.0
        # Let auto exposure and the depth stream stabilize before the first frame.
        # A D405 can take longer than one frame timeout immediately after USB
        # enumeration/reset even though the stream is otherwise sound.  Retrying
        # here is camera-only startup; no caller can issue a robot command until
        # this constructor has returned successfully.
        try:
            startup_timeout_ms = int(os.getenv(
                "REBOT_REALSENSE_STARTUP_TIMEOUT_MS", "15000"
            ))
        except ValueError:
            startup_timeout_ms = 15000
        startup_timeout_ms = max(5000, startup_timeout_ms)
        deadline = time.monotonic() + startup_timeout_ms / 1000.0
        warmed_frames = 0
        last_error: RuntimeError | None = None
        while warmed_frames < 10:
            remaining_ms = int((deadline - time.monotonic()) * 1000)
            if remaining_ms <= 0:
                self.close()
                detail = f" ({last_error})" if last_error else ""
                raise RuntimeError(
                    "RealSense did not stabilize within "
                    f"{startup_timeout_ms} ms{detail}"
                )
            try:
                self._pipeline.wait_for_frames(timeout_ms=min(3000, remaining_ms))
                warmed_frames += 1
            except RuntimeError as exc:
                # The next wait often succeeds once the USB/UVC stream is ready.
                last_error = exc

    def frame(self) -> tuple[np.ndarray, np.ndarray, Intrinsics]:
        frames = self._align.process(self._pipeline.wait_for_frames(timeout_ms=5000))
        color = frames.get_color_frame()
        depth = frames.get_depth_frame()
        if not color or not depth:
            raise RuntimeError("RealSense returned an incomplete colour/depth frame")
        intr = color.profile.as_video_stream_profile().intrinsics
        return (
            np.asanyarray(color.get_data()).copy(),
            np.asanyarray(depth.get_data()).astype(np.float32) * self._depth_scale_mm,
            (float(intr.fx), float(intr.fy), float(intr.ppx), float(intr.ppy)),
        )

    def close(self) -> None:
        try:
            self._pipeline.stop()
        except Exception:
            pass


def open_depth_camera(kind: str | None = None) -> DepthCamera:
    """Open ``realsense``, ``orbbec``, or ``auto`` (RealSense then Orbbec bridge)."""
    selected = (kind or os.getenv("REBOT_DEPTH_CAMERA", "realsense")).strip().lower()
    if selected == "realsense":
        return RealSenseCamera()
    if selected in {"orbbec", "orbbec_bridge"}:
        return OrbbecBridgeCamera()
    if selected == "auto":
        try:
            return RealSenseCamera()
        except Exception as realsense_error:
            try:
                return OrbbecBridgeCamera()
            except Exception as orbbec_error:
                raise RuntimeError(
                    f"no depth camera available (RealSense: {realsense_error}; "
                    f"Orbbec bridge: {orbbec_error})"
                ) from orbbec_error
    raise ValueError("REBOT_DEPTH_CAMERA must be realsense, orbbec, or auto")
