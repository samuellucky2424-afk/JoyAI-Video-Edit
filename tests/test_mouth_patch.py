import base64
import io
import sys
import unittest
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
DEPLOY = ROOT / "deploy"
if str(DEPLOY) not in sys.path:
    sys.path.insert(0, str(DEPLOY))

from xvideo.serving.mouth_patch import (  # noqa: E402
    MAX_MOUTH_PATCH_BYTES,
    apply_mouth_detail_patch,
)


def _jpeg_data_url(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=96)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def _active_meta(patch: str):
    return {
        "mouth_landmark_available": True,
        "mouth_landmark_seq": 12,
        "mouth_roi": {"x": 0.4, "y": 0.4, "width": 0.2, "height": 0.2},
        "mouth_anatomy": {
            "schema_version": 1,
            "method": "landmark_aligned_feature_encoder_v2",
            "available": True,
            "roi_confidence": 0.95,
            "region_evidence": {
                "lips": 0.9,
                "teeth": 0.92,
                "tongue": 0.0,
                "oral_cavity": 0.7,
            },
            "appearance_motion": 0.8,
            "significant": True,
        },
        "mouth_blendshapes": {"jawOpen": 0.7},
        "mouth_geometry": {"motion": 0.2},
        "mouth_event_significant": True,
        "mouth_patch": patch,
    }


class MouthPatchTests(unittest.TestCase):
    def test_active_patch_is_feathered_into_source_before_vae(self):
        frame = Image.new("RGB", (200, 100), (20, 30, 40))
        patch = Image.new("RGB", (40, 20), (240, 20, 20))
        conditioned, profile = apply_mouth_detail_patch(
            frame,
            _active_meta(_jpeg_data_url(patch)),
            enabled=True,
            max_gain=1.35,
        )
        self.assertIsNot(conditioned, frame)
        self.assertEqual(profile["mouth_patch_applied"], 1)
        self.assertEqual(profile["mouth_patch_reason"], "applied_before_vae")
        center = conditioned.getpixel((100, 50))
        self.assertGreater(center[0], 210)
        self.assertLess(center[1], 45)
        self.assertEqual(conditioned.getpixel((0, 0)), (20, 30, 40))
        self.assertEqual(frame.getpixel((100, 50)), (20, 30, 40))

    def test_disabled_neutral_and_invalid_patches_are_exact_noops(self):
        frame = Image.new("RGB", (200, 100), (20, 30, 40))
        valid = _active_meta(_jpeg_data_url(Image.new("RGB", (40, 20), "red")))
        neutral = {**valid, "mouth_event_significant": False}
        neutral["mouth_anatomy"] = {
            **valid["mouth_anatomy"],
            "appearance_motion": 0.0,
            "significant": False,
            "region_evidence": {"lips": 0.8, "teeth": 0.0, "tongue": 0.0, "oral_cavity": 0.0},
        }
        neutral["mouth_blendshapes"] = {}
        neutral["mouth_geometry"] = {"motion": 0.0}
        cases = (
            (False, valid),
            (True, neutral),
            (True, {**valid, "mouth_patch": "not-an-image"}),
            (
                True,
                {
                    **valid,
                    "mouth_patch": "data:image/jpeg;base64,"
                    + ("A" * (MAX_MOUTH_PATCH_BYTES * 2)),
                },
            ),
        )
        for enabled, meta in cases:
            with self.subTest(enabled=enabled, patch=str(meta["mouth_patch"])[:20]):
                result, profile = apply_mouth_detail_patch(
                    frame,
                    meta,
                    enabled=enabled,
                    max_gain=1.35,
                )
                self.assertIs(result, frame)
                self.assertEqual(profile["mouth_patch_applied"], 0)


if __name__ == "__main__":
    unittest.main()
