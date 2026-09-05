import sys
import unittest
from pathlib import Path

try:
    import torch
except ModuleNotFoundError:
    torch = None


ROOT = Path(__file__).resolve().parents[1]
DEPLOY = ROOT / "deploy"
if str(DEPLOY) not in sys.path:
    sys.path.insert(0, str(DEPLOY))

from xvideo.serving.mouth_latent_control import (  # noqa: E402
    MOUTH_LATENT_MAX_GAIN,
    _latent_box,
    apply_mouth_latent_control,
)


def _meta(*, significant=True, tongue=0.9, roi=None):
    return {
        "mouth_landmark_available": True,
        "mouth_landmark_age_ms": 0,
        "mouth_landmark_seq": 42,
        "mouth_roi": roi
        or {"x": 0.25, "y": 0.25, "width": 0.5, "height": 0.5},
        "mouth_anatomy": {
            "schema_version": 1,
            "method": "landmark_aligned_feature_encoder_v2",
            "available": True,
            "roi_confidence": 0.95,
            "region_evidence": {
                "lips": 0.9,
                "teeth": 0.0,
                "tongue": tongue,
                "oral_cavity": 0.8 if significant else 0.0,
            },
            "appearance_motion": 0.8 if significant else 0.0,
            "significant": significant,
        },
        "mouth_blendshapes": {"jawOpen": 0.8} if significant else {},
        "mouth_geometry": {"motion": 0.2 if significant else 0.0},
        "mouth_event_significant": significant,
    }


class MouthLatentGeometryTests(unittest.TestCase):
    def test_normalized_roi_maps_to_a_bounded_nonempty_latent_box(self):
        self.assertEqual(
            _latent_box((0.25, 0.25, 0.5, 0.5), height=8, width=12),
            (3, 2, 9, 6),
        )
        self.assertEqual(
            _latent_box((0.99, 0.99, 0.01, 0.01), height=8, width=12),
            (11, 7, 12, 8),
        )
        self.assertIsNone(
            _latent_box((0.25, 0.25, 0.5, 0.5), height=0, width=12)
        )

    def test_latent_gain_cap_is_conservative(self):
        self.assertGreater(MOUTH_LATENT_MAX_GAIN, 1.0)
        self.assertLessEqual(MOUTH_LATENT_MAX_GAIN, 1.125)


@unittest.skipIf(
    torch is None,
    "torch is not installed in the lightweight test environment",
)
class MouthLatentControlTests(unittest.TestCase):
    def test_active_control_changes_only_bounded_roi_and_preserves_contract(self):
        latent = torch.linspace(-2.0, 2.0, 1 * 4 * 1 * 8 * 12).reshape(
            1, 4, 1, 8, 12
        )
        original = latent.clone()
        profile = {}
        controlled = apply_mouth_latent_control(
            latent,
            [_meta()],
            enabled=True,
            max_gain=1.5,
            profile=profile,
        )

        self.assertIsNot(controlled, latent)
        self.assertEqual(controlled.shape, latent.shape)
        self.assertEqual(controlled.dtype, latent.dtype)
        self.assertEqual(controlled.device, latent.device)
        self.assertEqual(profile["mouth_latent_control_applied"], 1)
        self.assertEqual(
            profile["mouth_latent_control_reason"],
            "applied_to_ref_video_latent",
        )
        self.assertLessEqual(
            profile["mouth_latent_control_gain"], MOUTH_LATENT_MAX_GAIN
        )

        left, top, right, bottom = profile["mouth_latent_control_box"]
        outside = torch.ones_like(latent, dtype=torch.bool)
        outside[..., top:bottom, left:right] = False
        self.assertTrue(torch.equal(controlled[outside], latent[outside]))
        self.assertTrue(
            torch.any(
                controlled[..., top:bottom, left:right]
                != latent[..., top:bottom, left:right]
            )
        )
        self.assertTrue(torch.equal(latent, original))

    def test_bfloat16_control_preserves_dtype_and_finite_values(self):
        latent = torch.randn(1, 8, 2, 12, 20, dtype=torch.float32).to(
            torch.bfloat16
        )
        result = apply_mouth_latent_control(
            latent,
            [_meta()],
            enabled=True,
            max_gain=1.35,
        )

        self.assertEqual(result.dtype, torch.bfloat16)
        self.assertEqual(result.shape, latent.shape)
        self.assertTrue(bool(torch.isfinite(result).all().item()))

    def test_disabled_and_neutral_metadata_are_exact_tensor_noops(self):
        latent = torch.randn(1, 4, 1, 8, 12)
        cases = (
            (False, _meta()),
            (True, _meta(significant=False, tongue=0.0)),
            (True, {**_meta(), "mouth_landmark_available": False}),
        )
        for enabled, sample in cases:
            with self.subTest(enabled=enabled):
                profile = {}
                result = apply_mouth_latent_control(
                    latent,
                    [sample],
                    enabled=enabled,
                    max_gain=1.35,
                    profile=profile,
                )
                self.assertIs(result, latent)
                self.assertEqual(profile["mouth_latent_control_applied"], 0)

    def test_non_finite_cpu_tensor_is_an_exact_noop(self):
        latent = torch.zeros(1, 4, 1, 8, 12)
        latent[..., 3, 4] = float("nan")
        profile = {}
        result = apply_mouth_latent_control(
            latent,
            [_meta()],
            enabled=True,
            max_gain=1.35,
            profile=profile,
        )
        self.assertIs(result, latent)
        self.assertEqual(profile["mouth_latent_control_reason"], "non_finite_tensor")

    def test_tiny_valid_roi_maps_to_at_least_one_latent_cell(self):
        latent = torch.arange(1 * 2 * 1 * 4 * 4, dtype=torch.float32).reshape(
            1, 2, 1, 4, 4
        )
        profile = {}
        result = apply_mouth_latent_control(
            latent,
            [
                _meta(
                    roi={
                        "x": 0.49,
                        "y": 0.49,
                        "width": 0.02,
                        "height": 0.02,
                    }
                )
            ],
            enabled=True,
            max_gain=1.35,
            profile=profile,
        )
        self.assertIsNot(result, latent)
        left, top, right, bottom = profile["mouth_latent_control_box"]
        self.assertGreaterEqual(right - left, 1)
        self.assertGreaterEqual(bottom - top, 1)


if __name__ == "__main__":
    unittest.main()
