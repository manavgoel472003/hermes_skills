from __future__ import annotations

from unittest import TestCase

import cv2
import numpy as np

from arm_agent.wrist_detect import (
    Detection,
    detect,
    detect_color,
    detect_color_conjunction,
    detect_color_union,
    segment_target_envelope,
    track_detection_flow,
)


class BlueDetectionTest(TestCase):
    def test_slash_separated_colour_alternatives_use_stable_tracker(self):
        color = np.zeros((240, 320, 3), dtype=np.uint8)
        # HSV H=165 lies in the pink mask but outside the red mask.
        import cv2
        pink = cv2.cvtColor(
            np.uint8([[[165, 220, 220]]]), cv2.COLOR_HSV2BGR
        )[0, 0]
        color[80:150, 145:175] = pink
        depth = np.full((240, 320), 220.0, dtype=np.float32)

        found, why = detect(
            color, depth, (400.0, 400.0, 160.0, 120.0),
            "red/pink marker with white body", near_mm=60, far_mm=600,
            allow_vlm=False,
        )

        self.assertIsNotNone(found)
        assert found is not None
        self.assertEqual(found.source, "color:red+pink")
        self.assertEqual(why["color_words"], ["red", "pink"])

    def test_multicolour_sponge_is_one_union_detection(self):
        color = np.zeros((200, 300, 3), dtype=np.uint8)
        color[60:140, 80:130] = (0, 210, 0)
        color[60:140, 130:180] = (0, 220, 220)
        depth = np.full((200, 300), 520.0, dtype=np.float32)
        depth[60:140, 80:180] = 220.0

        found = detect_color_union(
            color, depth, (400.0, 400.0, 150.0, 100.0),
            ["green", "yellow"], near_mm=60, far_mm=600,
        )

        self.assertIsNotNone(found)
        assert found is not None
        self.assertEqual(found.source, "color:green+yellow")
        self.assertGreater(found.area_px, 7000)
        self.assertAlmostEqual(found.uv[0], 129.5, delta=2.0)

    def test_colour_conjunction_rejects_larger_green_marker(self):
        color = np.zeros((240, 360, 3), dtype=np.uint8)
        depth = np.full((240, 360), 520.0, dtype=np.float32)
        # Distractor: a larger standalone green marker-like region.
        color[40:210, 35:65] = (0, 220, 0)
        depth[40:210, 35:65] = 230.0
        # Target: separated green/yellow faces belonging to one nearby sponge.
        color[80:170, 190:235] = (0, 210, 0)
        color[80:170, 245:300] = (0, 220, 220)
        depth[80:170, 190:300] = 245.0
        intr = (400.0, 400.0, 180.0, 120.0)

        found, why = detect(
            color, depth, intr, "green and yellow sponge",
            near_mm=60, far_mm=600, allow_vlm=False,
        )

        self.assertIsNotNone(found)
        assert found is not None
        self.assertEqual(found.source, "color:green&yellow")
        self.assertGreater(found.uv[0], 190.0)
        self.assertEqual(why["used"], "color:green&yellow")

    def test_colour_conjunction_refuses_green_only_scene(self):
        color = np.zeros((180, 280, 3), dtype=np.uint8)
        color[40:150, 100:135] = (0, 220, 0)
        depth = np.full((180, 280), 240.0, dtype=np.float32)

        found, why = detect(
            color, depth, (400.0, 400.0, 140.0, 90.0),
            "green and yellow sponge", near_mm=60, far_mm=600,
            allow_vlm=False,
        )

        self.assertIsNone(found)
        self.assertIn("every requested colour", why["color_result"])

    def test_rgbd_flow_preserves_identity_across_short_jog(self):
        h, w = 200, 300
        color0 = np.zeros((h, w, 3), dtype=np.uint8)
        # Same-hue checker texture gives LK features without splitting the color
        # identifier into unrelated objects.
        for row in range(60, 140, 10):
            for col in range(80, 180, 10):
                scale = 170 if ((row + col) // 10) % 2 else 240
                color0[row:row + 10, col:col + 10] = (
                    (0, scale, 0) if col < 130 else (0, scale, scale)
                )
        depth0 = np.full((h, w), 520.0, dtype=np.float32)
        depth0[60:140, 80:180] = 220.0
        transform = np.float32([[1, 0, 12], [0, 1, -5]])
        color1 = cv2.warpAffine(color0, transform, (w, h))
        depth1 = cv2.warpAffine(
            depth0, transform, (w, h), flags=cv2.INTER_NEAREST,
            borderValue=520.0,
        )
        intr = (400.0, 400.0, 150.0, 100.0)
        det = detect_color_union(
            color0, depth0, intr, ["green", "yellow"],
            near_mm=60, far_mm=600,
        )
        self.assertIsNotNone(det)
        assert det is not None

        tracked = track_detection_flow(
            (color0, depth0, intr), (color1, depth1, intr), det,
            "green and yellow sponge",
        )

        self.assertIsNotNone(tracked)
        assert tracked is not None
        self.assertAlmostEqual(tracked.uv[0] - det.uv[0], 12.0, delta=1.0)
        self.assertAlmostEqual(tracked.uv[1] - det.uv[1], -5.0, delta=1.0)
        self.assertGreaterEqual(tracked.info["flow_points"], 4)

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
