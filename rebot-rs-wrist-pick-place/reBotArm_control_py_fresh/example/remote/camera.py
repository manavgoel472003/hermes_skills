"""Shared camera source for the remote-teleop nodes: a real cv2 device, or a
synthetic 'test' pattern so the whole pipeline can run with no hardware."""
from __future__ import annotations

import time

import cv2
import numpy as np


class Camera:
    def __init__(self, spec: str, width: int, height: int) -> None:
        self.test = (spec == "test")
        self.width, self.height = width, height
        self.cap = None
        if not self.test:
            idx = int(spec) if str(spec).isdigit() else spec
            self.cap = cv2.VideoCapture(idx)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            if not self.cap.isOpened():
                raise RuntimeError(f"could not open camera {spec!r}")

    def read(self) -> np.ndarray | None:
        """Return one BGR frame, or None if unavailable."""
        if self.test:
            t = time.time()
            img = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            img[:] = (30, 30, 40)
            cx = int((0.5 + 0.4 * np.sin(t)) * self.width)
            cy = int((0.5 + 0.4 * np.cos(t * 0.7)) * self.height)
            cv2.circle(img, (cx, cy), 26, (60, 200, 240), -1)
            cv2.putText(img, "TEST PATTERN (no camera)", (10, self.height - 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1, cv2.LINE_AA)
            time.sleep(1 / 30)
            return img
        ok, frame = self.cap.read()
        return frame if ok else None

    def release(self):
        if self.cap is not None:
            self.cap.release()
