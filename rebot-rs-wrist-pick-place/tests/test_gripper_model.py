from __future__ import annotations

import math
from unittest import TestCase
from unittest.mock import patch

from arm_agent import gripper_model as gm


class GripperSpanTest(TestCase):
    def test_default_span_is_measured_at_safe_command_cap(self):
        stored = {}

        def save(scale, detail=None):
            stored["scale"] = scale
            stored["detail"] = detail
            return "jaw.json"

        with (
            patch.object(gm, "CLOSE_RAD", 0.0),
            patch.object(gm, "OPEN_RAD", math.radians(338.63)),
            patch.object(gm, "save_calibration", save),
            patch.object(gm, "mm_per_rad", lambda: stored.get("scale")),
        ):
            result = gm.set_span(42.0)

        self.assertTrue(result["ok"])
        self.assertEqual(result["measured_at_deg"], 150.0)
        self.assertAlmostEqual(result["max_aperture_mm"], 42.0, places=1)
        self.assertAlmostEqual(stored["detail"]["at_fraction"], 150.0 / 338.63)

    def test_max_aperture_uses_clamped_open_target(self):
        with (
            patch.object(gm, "mm_per_rad", lambda: 10.0),
            patch.object(gm, "CLOSE_RAD", 0.0),
            patch.object(gm, "OPEN_RAD", math.radians(338.63)),
        ):
            aperture = gm.max_aperture_mm()

        self.assertAlmostEqual(aperture, 10.0 * math.radians(150.0))

    def test_fraction_must_be_valid(self):
        self.assertFalse(gm.set_span(42.0, 0.0)["ok"])
        self.assertFalse(gm.set_span(42.0, 1.1)["ok"])
