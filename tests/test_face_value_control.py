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

from xvideo.serving.face_value_control import (  # noqa: E402
    EYE_VALUE_MAX_GAIN,
    MOUTH_VALUE_MAX_GAIN,
    build_face_value_scale,
)
from xvideo.serving.mouth_anatomy import (  # noqa: E402
    MOUTH_ANATOMY_METHOD,
    MOUTH_ANATOMY_SCHEMA_VERSION,
)


def _meta(*, anatomy=None, mouth=None, eyes=None, eye_rois=None):
    return {
        "mouth_landmark_available": True,
        "mouth_landmark_seq": 21,
        "mouth_roi": {"x": 0.4, "y": 0.58, "width": 0.2, "height": 0.16},
        "mouth_anatomy": {
            "schema_version": MOUTH_ANATOMY_SCHEMA_VERSION,
            "method": MOUTH_ANATOMY_METHOD,
            "available": True,
            "roi_confidence": 0.95,
            "region_evidence": anatomy
            or {
                "lips": 0.9,
                "teeth": 0.0,
                "tongue": 0.0,
                "oral_cavity": 0.0,
            },
            "appearance_motion": 0.0,
            "significant": False,
        },
        "mouth_blendshapes": mouth or {},
        "mouth_geometry": {"motion": 0.0},
        "eye_landmark_available": True,
        "eye_rois": eye_rois
        or {
            "left": {"x": 0.32, "y": 0.35, "width": 0.13, "height": 0.09},
            "right": {"x": 0.55, "y": 0.35, "width": 0.13, "height": 0.09},
        },
        "eye_blendshapes": eyes or {},
    }


class FaceValueControlGeometryTests(unittest.TestCase):
    def test_attention_value_caps_are_local_and_conservative(self):
        self.assertGreater(MOUTH_VALUE_MAX_GAIN, 1.0)
        self.assertLessEqual(MOUTH_VALUE_MAX_GAIN, 1.125)
        self.assertGreater(EYE_VALUE_MAX_GAIN, 1.0)
        self.assertLess(EYE_VALUE_MAX_GAIN, MOUTH_VALUE_MAX_GAIN)


@unittest.skipIf(
    torch is None,
    "torch is not installed in the lightweight test environment",
)
class FaceValueControlTensorTests(unittest.TestCase):
    def test_teeth_tongue_and_round_mouth_build_a_bounded_local_scale(self):
        latent = torch.zeros(1, 8, 1, 20, 35, dtype=torch.bfloat16)
        cases = (
            ("teeth", _meta(anatomy={"lips": 0.9, "teeth": 0.95, "tongue": 0.0, "oral_cavity": 0.4})),
            ("tongue", _meta(anatomy={"lips": 0.9, "teeth": 0.0, "tongue": 0.9, "oral_cavity": 0.8})),
            ("round", _meta(mouth={"jawOpen": 0.9, "mouthFunnel": 0.88})),
        )
        for event, sample in cases:
            with self.subTest(event=event):
                profile = {}
                scale = build_face_value_scale(
                    latent,
                    [sample],
                    enabled=True,
                    max_gain=1.35,
                    patch_size=(1, 1, 1),
                    profile=profile,
                )
                self.assertIsNotNone(scale)
                self.assertEqual(scale.shape, (1, 20 * 35))
                self.assertEqual(scale.dtype, latent.dtype)
                self.assertGreater(float(scale.max()), 1.0)
                self.assertLessEqual(float(scale.max()), MOUTH_VALUE_MAX_GAIN)
                self.assertIn(event, profile["face_value_control_mouth_events"])
                self.assertTrue(bool(torch.any(scale == 1).item()))

    def test_left_and_right_eye_events_are_independent_and_bounded(self):
        latent = torch.zeros(1, 8, 1, 20, 35)
        profile = {}
        scale = build_face_value_scale(
            latent,
            [_meta(eyes={"eyeBlinkLeft": 0.96})],
            enabled=True,
            max_gain=1.35,
            patch_size=(1, 1, 1),
            profile=profile,
        )
        self.assertIsNotNone(scale)
        self.assertLessEqual(float(scale.max()), EYE_VALUE_MAX_GAIN)
        self.assertEqual(profile["face_value_control_eye_sides"], ["left"])
        self.assertGreater(profile["face_value_control_changed_fraction"], 0.0)

    def test_disabled_neutral_and_unavailable_metadata_are_exact_noops(self):
        latent = torch.zeros(1, 8, 1, 20, 35)
        neutral = _meta()
        unavailable = {**_meta(eyes={"eyeBlinkLeft": 0.9}), "eye_landmark_available": False}
        for enabled, metas in (
            (False, [_meta(anatomy={"lips": 0.8, "teeth": 0.9, "tongue": 0.0, "oral_cavity": 0.0})]),
            (True, [neutral]),
            (True, [unavailable]),
        ):
            with self.subTest(enabled=enabled):
                profile = {}
                scale = build_face_value_scale(
                    latent,
                    metas,
                    enabled=enabled,
                    max_gain=1.35,
                    patch_size=(1, 1, 1),
                    profile=profile,
                )
                self.assertIsNone(scale)
                self.assertEqual(profile["face_value_control_applied"], 0)

    def test_scale_shape_tracks_transformer_patch_tokens(self):
        latent = torch.zeros(2, 4, 2, 12, 20)
        scale = build_face_value_scale(
            latent,
            [_meta(anatomy={"lips": 0.9, "teeth": 0.9, "tongue": 0.0, "oral_cavity": 0.0})],
            enabled=True,
            max_gain=1.35,
            patch_size=(1, 2, 2),
        )
        self.assertEqual(scale.shape, (2, 2 * 6 * 10))


if __name__ == "__main__":
    unittest.main()
