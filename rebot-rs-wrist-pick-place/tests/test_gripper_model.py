from __future__ import annotations

import math
import tempfile
from pathlib import Path
from unittest import TestCase
from unittest.mock import patch

from arm_agent import gripper_model as gm


class GripperSpanTest(TestCase):
    def test_piecewise_curve_interpolates_and_inverts_small_apertures(self):
        calib = {
            "points": [
                {"angle_rad": math.radians(30), "aperture_mm": 7.0},
                {"angle_rad": math.radians(60), "aperture_mm": 17.0},
                {"angle_rad": math.radians(150), "aperture_mm": 50.0},
            ]
        }
        with (
            patch.object(gm, "CLOSE_RAD", 0.0),
            patch.object(gm, "OPEN_RAD", math.radians(338.63)),
            patch.object(gm, "load_calibration", return_value=calib),
        ):
            self.assertAlmostEqual(gm.aperture_mm(math.radians(45)), 12.0)
            self.assertAlmostEqual(math.degrees(gm.rad_for_aperture(12.0)), 45.0)
            self.assertAlmostEqual(gm.max_aperture_mm(), 50.0)

    def test_manual_curve_point_saves_without_motion_and_rejects_nonmonotonic(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "jaw.json"
            legacy = {
                "mm_per_rad": 19.1,
                "detail": {"measured_rad": math.radians(150),
                           "aperture_mm": 50.0},
            }
            with (
                patch.object(gm, "CALIB_FILE", path),
                patch.object(gm, "CALIB_DIR", Path(directory)),
                patch.object(gm, "load_calibration", return_value=legacy),
            ):
                result = gm.add_calibration_point(60.0, 18.0)
            self.assertTrue(result["ok"], result.get("reason"))
            self.assertEqual(len(result["points"]), 2)
            self.assertIn("no gripper command", result["note"])

            bad = {"points": result["points"]}
            with (
                patch.object(gm, "CALIB_FILE", path),
                patch.object(gm, "CALIB_DIR", Path(directory)),
                patch.object(gm, "load_calibration", return_value=bad),
            ):
                refused = gm.add_calibration_point(90.0, 10.0)
            self.assertFalse(refused["ok"])
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
