import importlib.util
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "deploy" / "xvideo" / "serving" / "mouth_anatomy.py"
SPEC = importlib.util.spec_from_file_location("mouth_anatomy", MODULE_PATH)
MOUTH_ANATOMY = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MOUTH_ANATOMY)


def valid_payload():
    return {
        "schema_version": 1,
        "method": "landmark_aligned_feature_encoder_v2",
        "available": True,
        "roi_confidence": 0.92,
        "region_evidence": {
            "lips": 0.88,
            "teeth": 0.64,
            "tongue": 0.21,
            "oral_cavity": 0.71,
        },
        "appearance_motion": 0.37,
        "significant": True,
    }


class MouthAnatomyContractTests(unittest.TestCase):
    def test_contract_names_all_requested_anatomy_regions(self):
        self.assertEqual(
            MOUTH_ANATOMY.MOUTH_ANATOMY_REGIONS,
            ("lips", "teeth", "tongue", "oral_cavity"),
        )
        self.assertEqual(
            set(MOUTH_ANATOMY.MOUTH_ANATOMY_REGION_MEANINGS),
            set(MOUTH_ANATOMY.MOUTH_ANATOMY_REGIONS),
        )

    def test_normalizes_valid_scores_to_a_stable_json_shape(self):
        normalized = MOUTH_ANATOMY.normalize_mouth_anatomy(valid_payload())

        self.assertEqual(normalized, valid_payload())
        self.assertEqual(
            tuple(normalized["region_evidence"]),
            MOUTH_ANATOMY.MOUTH_ANATOMY_REGIONS,
        )

    def test_rejects_missing_or_unknown_anatomy_regions(self):
        missing = valid_payload()
        del missing["region_evidence"]["tongue"]
        with self.assertRaisesRegex(
            MOUTH_ANATOMY.MouthAnatomyContractError, "missing: tongue"
        ):
            MOUTH_ANATOMY.normalize_mouth_anatomy(missing)

        unknown = valid_payload()
        unknown["region_evidence"]["skin"] = 0.4
        with self.assertRaisesRegex(
            MOUTH_ANATOMY.MouthAnatomyContractError, "unknown regions: skin"
        ):
            MOUTH_ANATOMY.normalize_mouth_anatomy(unknown)

    def test_rejects_out_of_range_and_non_finite_scores(self):
        cases = (-0.01, 1.01, float("nan"), float("inf"), True)
        for value in cases:
            with self.subTest(value=value):
                payload = valid_payload()
                payload["region_evidence"]["teeth"] = value
                with self.assertRaises(MOUTH_ANATOMY.MouthAnatomyContractError):
                    MOUTH_ANATOMY.normalize_mouth_anatomy(payload)

    def test_unavailable_roi_cannot_influence_frame_selection(self):
        unavailable = {
            "schema_version": 1,
            "method": "landmark_aligned_feature_encoder_v2",
            "available": False,
            "roi_confidence": 0.0,
            "region_evidence": {
                "lips": 0.0,
                "teeth": 0.0,
                "tongue": 0.0,
                "oral_cavity": 0.0,
            },
            "appearance_motion": 0.0,
            "significant": False,
        }
        self.assertEqual(
            MOUTH_ANATOMY.normalize_mouth_anatomy(unavailable), unavailable
        )

        unavailable["significant"] = True
        with self.assertRaisesRegex(
            MOUTH_ANATOMY.MouthAnatomyContractError,
            "unavailable mouth anatomy must have zero scores",
        ):
            MOUTH_ANATOMY.normalize_mouth_anatomy(unavailable)


if __name__ == "__main__":
    unittest.main()
