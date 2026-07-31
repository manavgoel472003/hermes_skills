from __future__ import annotations

from unittest import TestCase

import numpy as np

from arm_agent.wrist_detect import (
    Detection,
    detect_color,
    segment_target_envelope,
)


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

    def test_target_seeded_envelope_does_not_merge_far_background(self):
        color = np.zeros((240, 320, 3), dtype=np.uint8)
        depth = np.full((240, 320), 520.0, dtype=np.float32)
        # A close cap touching a slightly deeper body.
        depth[80:110, 145:175] = 220.0
        depth[105:190, 140:180] = 245.0
        target = Detection(
            box=(145, 80, 175, 110), uv=(160.0, 95.0), z_mm=220.0,
            p_cam_mm=(0.0, 0.0, 220.0), source="color:green",
            area_px=800.0, min_width_mm=16.0, max_width_mm=18.0,
            info={"long_axis_deg": 0.0},
        )

        envelope = segment_target_envelope(
            color, depth, (400.0, 400.0, 160.0, 120.0), target,
            near_mm=60.0, far_mm=600.0,
        )

        self.assertIsNotNone(envelope)
        assert envelope is not None
        self.assertLess(envelope.z_mm, 300.0)
        self.assertGreater(envelope.area_px, target.area_px)
        self.assertLess(envelope.box[3], 200)

    def test_local_cross_section_ignores_wider_cap_at_object_end(self):
        color = np.zeros((240, 320, 3), dtype=np.uint8)
        depth = np.full((240, 320), 520.0, dtype=np.float32)
        # Narrow body with a wider cap at one end, all depth-connected.
        depth[70:105, 140:180] = 220.0
        depth[100:200, 150:170] = 225.0
        target = Detection(
            box=(140, 70, 180, 105), uv=(160.0, 87.0), z_mm=220.0,
            p_cam_mm=(0.0, 0.0, 220.0), source="color:blue",
            area_px=1200.0, min_width_mm=22.0, max_width_mm=25.0,
            info={"long_axis_deg": 0.0},
        )

        envelope = segment_target_envelope(
            color, depth, (400.0, 400.0, 160.0, 120.0), target,
            near_mm=60.0, far_mm=600.0,
        )

        self.assertIsNotNone(envelope)
        assert envelope is not None
        local = envelope.info.get("local_grasp_width_mm")
        self.assertIsNotNone(local)
        self.assertLess(float(local), float(envelope.min_width_mm) * 0.75)
