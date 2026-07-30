from __future__ import annotations

from unittest import TestCase

import numpy as np

from arm_agent.wrist_detect import detect_color


class BlueDetectionTest(TestCase):
    def test_vivid_marker_is_not_merged_with_cyan_white_balance_cast(self):
        # Observed D405 colours from the failed dry-run frame.
        color = np.full((480, 640, 3), (136, 122, 69), dtype=np.uint8)
        color[360:440, 300:324] = (131, 76, 0)
        depth = np.full((480, 640), 423.0, dtype=np.float32)
        intr = (394.07, 393.66, 319.48, 245.50)

        found = detect_color(color, depth, intr, "blue", near_mm=60, far_mm=600)

        self.assertIsNotNone(found)
        assert found is not None
        self.assertLess(found.min_width_mm or 999, 30)
        self.assertLess(found.max_width_mm or 999, 100)
        self.assertGreater(found.uv[0], 295)
        self.assertLess(found.uv[0], 330)
