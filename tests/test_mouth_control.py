import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEPLOY = ROOT / "deploy"
if str(DEPLOY) not in sys.path:
    sys.path.insert(0, str(DEPLOY))

MODULE_PATH = DEPLOY / "xvideo" / "serving" / "mouth_control.py"
SPEC = importlib.util.spec_from_file_location("mouth_control", MODULE_PATH)
MOUTH_CONTROL = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MOUTH_CONTROL
SPEC.loader.exec_module(MOUTH_CONTROL)


def anatomy(**evidence):
    regions = {"lips": 0.8, "teeth": 0.0, "tongue": 0.0, "oral_cavity": 0.0}
    regions.update(evidence)
    return {
        "schema_version": 1,
        "method": "landmark_aligned_feature_encoder_v2",
        "available": True,
        "roi_confidence": 0.9,
        "region_evidence": regions,
        "appearance_motion": 0.0,
        "significant": False,
    }


def meta(sequence, *, roi=None, anatomy_payload=None, blendshapes=None, significant=False):
    return {
        "mouth_landmark_available": True,
        "mouth_landmark_seq": sequence,
        "mouth_roi": roi or {"x": 0.4, "y": 0.42, "width": 0.2, "height": 0.16},
        "mouth_anatomy": anatomy_payload or anatomy(),
        "mouth_blendshapes": blendshapes or {},
        "mouth_geometry": {"motion": 0.0},
        "mouth_event_significant": significant,
    }


class MouthControlTests(unittest.TestCase):
    def test_visible_anatomy_activates_bounded_deduplicated_roi_control(self):
        first = meta(10, anatomy_payload=anatomy(teeth=0.9))
        duplicate = dict(first)
        second = meta(
            11,
            roi={"x": 0.38, "y": 0.4, "width": 0.24, "height": 0.2},
            anatomy_payload=anatomy(tongue=0.82),
        )
        control = MOUTH_CONTROL.build_mouth_control(
            [first, duplicate, second], enabled=True, max_gain=1.35
        )
        self.assertTrue(control.active)
        self.assertEqual(control.sample_count, 2)
        self.assertEqual(control.roi, (0.38, 0.4, 0.24, 0.2))
        self.assertGreater(control.gain, 1.2)
        self.assertLessEqual(control.gain, 1.35)

    def test_round_open_mouth_and_smile_activate_without_pixel_segmentation(self):
        round_open = meta(20, anatomy_payload=anatomy(oral_cavity=0.84))
        smile = meta(
            21,
            blendshapes={"mouthSmileLeft": 0.8, "mouthSmileRight": 0.9},
        )
        for sample in (round_open, smile):
            with self.subTest(sequence=sample["mouth_landmark_seq"]):
                control = MOUTH_CONTROL.build_mouth_control(
                    [sample], enabled=True, max_gain=1.35
                )
                self.assertTrue(control.active)
                self.assertGreater(control.gain, 1.0)

    def test_disabled_closed_unavailable_and_invalid_roi_are_exact_noops(self):
        cases = (
            (False, [meta(1)]),
            (True, [meta(2)]),
            (True, [{**meta(3), "mouth_landmark_available": False}]),
            (True, [meta(4, roi={"x": 0.4, "y": 0.4, "width": 0.0, "height": 0.1})]),
            (True, [meta(5, roi={"x": 0.4, "y": 0.4, "width": True, "height": 0.1})]),
        )
        for enabled, samples in cases:
            with self.subTest(enabled=enabled, sequence=samples[0]["mouth_landmark_seq"]):
                control = MOUTH_CONTROL.build_mouth_control(
                    samples, enabled=enabled, max_gain=1.35
                )
                self.assertFalse(control.active)
                scale = MOUTH_CONTROL.mouth_control_token_scale(
                    control, temporal_tokens=1, height_tokens=12, width_tokens=20
                )
                np.testing.assert_array_equal(scale, np.ones((1, 240), dtype=np.float32))

    def test_unhashable_tracker_sequence_does_not_break_control(self):
        sample = meta(["malformed"], anatomy_payload=anatomy(teeth=0.9))
        control = MOUTH_CONTROL.build_mouth_control(
            [sample], enabled=True, max_gain=1.35
        )
        self.assertTrue(control.active)
        self.assertEqual(control.sample_count, 1)

    def test_feathered_scale_only_changes_tokens_near_the_mouth(self):
        control = MOUTH_CONTROL.MouthControl(
            active=True,
            roi=(0.4, 0.4, 0.2, 0.2),
            gain=1.35,
            strength=1.0,
            confidence=1.0,
            sample_count=1,
            reason="active",
        )
        scale = MOUTH_CONTROL.mouth_control_token_scale(
            control, temporal_tokens=1, height_tokens=20, width_tokens=20
        ).reshape(20, 20)
        self.assertEqual(float(scale[0, 0]), 1.0)
        self.assertAlmostEqual(float(scale[10, 10]), 1.35, places=5)
        self.assertGreater(float(scale[7, 10]), 1.0)
        self.assertLess(float(scale[7, 10]), 1.35)


if __name__ == "__main__":
    unittest.main()
