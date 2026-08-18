import unittest
from unittest.mock import Mock

import torch

from arm_agent.local_vision import Sam3TextSegmenter, Sam31TextSegmenter


class Sam3VideoIdentityTests(unittest.TestCase):
    def test_initial_selection_uses_best_matching_prompt_score(self):
        result = {
            "object_ids": torch.tensor([4, 8, 12]),
            "scores": torch.tensor([0.95, 0.70, 0.80]),
            "prompt_to_obj_ids": {"red marker": [8, 12]},
        }
        row, object_id = Sam3TextSegmenter._select_tracked_instance(
            result, "red marker", None
        )
        self.assertEqual(row, 2)
        self.assertEqual(object_id, 12)

    def test_locked_identity_cannot_switch_to_higher_scoring_instance(self):
        result = {
            "object_ids": torch.tensor([4, 8]),
            "scores": torch.tensor([0.99, 0.20]),
            "prompt_to_obj_ids": {"red marker": [4, 8]},
        }
        row, object_id = Sam3TextSegmenter._select_tracked_instance(
            result, "red marker", 8
        )
        self.assertEqual(row, 1)
        self.assertEqual(object_id, 8)

    def test_missing_locked_identity_fails_closed(self):
        result = {
            "object_ids": torch.tensor([4, 12]),
            "scores": torch.tensor([0.99, 0.90]),
            "prompt_to_obj_ids": {"red marker": [4, 12]},
        }
        row, object_id = Sam3TextSegmenter._select_tracked_instance(
            result, "red marker", 8
        )
        self.assertIsNone(row)
        self.assertEqual(object_id, 8)


class Sam31MaskTests(unittest.TestCase):
    def test_box_matching_retains_native_mask(self):
        segmenter = Sam31TextSegmenter()
        segmenter._last_box = (10, 20, 40, 60)
        segmenter._last_mask = torch.ones(80, 100, dtype=torch.uint8).numpy()
        self.assertIsNotNone(segmenter.segment_box(
            segmenter._last_mask, (10, 20, 40, 60)
        ))
        self.assertIsNone(segmenter.segment_box(
            segmenter._last_mask, (60, 20, 90, 60)
        ))

    def test_completed_bootstrap_delegates_later_geometry_queries(self):
        segmenter = Sam31TextSegmenter()
        fallback = Mock()
        fallback.locate.return_value = {
            "ok": True, "found": True, "backend": "sam3"
        }
        segmenter._bootstrap_complete = True
        segmenter._geometry_backend = fallback
        result = segmenter.locate(
            torch.zeros(4, 4, 3, dtype=torch.uint8).numpy(), "yellow sponge"
        )
        fallback.locate.assert_called_once()
        self.assertEqual(
            result["model_version"], "sam3_after_sam3.1_anchor"
        )

if __name__ == "__main__":
    unittest.main()
