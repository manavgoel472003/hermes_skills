from __future__ import annotations

from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import patch

import numpy as np

from arm_agent import wrist_servo as ws


class _Eye:
    camera_kind = "test"
    last_object_kwargs = None

    def __init__(self) -> None:
        self.closed = False

    def frame(self):
        return (
            np.zeros((8, 8, 3), dtype=np.uint8),
            np.full((8, 8), 100, dtype=np.uint16),
            SimpleNamespace(),
        )

    def object(self, name, color, depth, intr, **kwargs):
        type(self).last_object_kwargs = kwargs
        return (
            SimpleNamespace(
                p_cam_mm=(10.0, 20.0, 80.0),
                uv=(4, 4),
                source="test",
                z_mm=80.0,
            ),
            {},
        )

    def close(self) -> None:
        self.closed = True


def _ready():
    return SimpleNamespace(ok=True, reason="ready")


class HeldTcpCalibrationTest(TestCase):
    def test_wrist_annotation_renders_depth_inset(self):
        color = np.zeros((480, 640, 3), dtype=np.uint8)
        depth = np.full((480, 640), 120.0, dtype=np.float32)

        rendered = ws.annotate(color, depth)

        self.assertEqual(rendered.shape, color.shape)

    def test_never_touches_arm_or_gripper(self):
        arm_calls = []
        _Eye.last_object_kwargs = None
        with (
            patch.object(
                ws.A,
                "preflight",
                lambda *args, **kwargs: arm_calls.append(("preflight", args, kwargs)),
            ),
            patch.object(ws, "WristEye", _Eye),
            patch.object(
                ws,
                "_local_state",
                lambda: arm_calls.append(("state",)),
            ),
            patch.object(
                ws,
                "_gripper",
                lambda *args, **kwargs: arm_calls.append(("gripper", args, kwargs)),
            ),
            patch.object(ws, "_show", lambda *args, **kwargs: None),
            patch.object(ws, "save_tcp", lambda xyz, detail=None: "tcp.json"),
        ):
            result = ws.calibrate_tcp_by_grasp("blue marker", close_first=False)

        self.assertTrue(result["ok"])
        self.assertEqual(result["tcp_camera_xyz_mm"], [10.0, 20.0, 80.0])
        self.assertEqual(arm_calls, [])
        self.assertEqual(result["steps"][0]["step"], "camera_only")
        self.assertIn("not connected", result["steps"][0]["reason"])
        self.assertEqual(_Eye.last_object_kwargs, {"near_mm": 90.0, "far_mm": 250.0})

    def test_implausible_detection_does_not_save_or_touch_hardware(self):
        arm_calls = []
        saved = []
        original = _Eye.object

        def far_object(eye, name, color, depth, intr, **kwargs):
            found, why = original(eye, name, color, depth, intr, **kwargs)
            found.p_cam_mm = (0.0, 0.0, 300.0)
            found.z_mm = 300.0
            return found, why

        with (
            patch.object(
                ws.A,
                "preflight",
                lambda *args, **kwargs: arm_calls.append(("preflight", args, kwargs)),
            ),
            patch.object(ws, "WristEye", _Eye),
            patch.object(_Eye, "object", far_object),
            patch.object(ws, "_local_state", lambda: arm_calls.append(("state",))),
            patch.object(ws, "_gripper", lambda *a, **k: arm_calls.append(("gripper",))),
            patch.object(ws, "_show", lambda *args, **kwargs: None),
            patch.object(ws, "save_tcp", lambda *a, **k: saved.append((a, k))),
        ):
            result = ws.calibrate_tcp_by_grasp("blue marker", close_first=False)

        self.assertFalse(result["ok"])
        self.assertIn("too far", result["reason"])
        self.assertEqual(arm_calls, [])
        self.assertEqual(saved, [])
