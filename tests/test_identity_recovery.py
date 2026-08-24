import importlib.util
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "deploy" / "xvideo" / "serving" / "identity_recovery.py"
SPEC = importlib.util.spec_from_file_location("identity_recovery", MODULE_PATH)
IDENTITY_RECOVERY = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = IDENTITY_RECOVERY
SPEC.loader.exec_module(IDENTITY_RECOVERY)


def metas(*risks, overlap=0.0):
    return [
        {
            "identity_occlusion_risk": risk,
            "identity_occlusion_overlap": overlap,
        }
        for risk in risks
    ]


class IdentityRecoveryControllerTests(unittest.TestCase):
    def test_risky_chunks_reuse_last_clean_anchor_until_hysteresis_releases(self):
        controller = IDENTITY_RECOVERY.IdentityRecoveryController(
            enabled=True,
            clean_chunks_to_release=2,
        )

        clean = controller.observe_chunk(0, metas(False))
        risky = controller.observe_chunk(1, metas(True, True, False))
        still_risky = controller.observe_chunk(2, metas(True, True, True))
        first_clean = controller.observe_chunk(3, metas(False, False, False))
        released = controller.observe_chunk(4, metas(False, False, False))
        normal = controller.observe_chunk(5, metas(False, False, False))

        self.assertFalse(clean.recovery_active)
        self.assertEqual(risky.anchor_chunk_id, 0)
        self.assertEqual(still_risky.anchor_chunk_id, 0)
        self.assertEqual(first_clean.anchor_chunk_id, 0)
        self.assertFalse(first_clean.release_after_chunk)
        self.assertEqual(released.anchor_chunk_id, 0)
        self.assertTrue(released.release_after_chunk)
        self.assertFalse(normal.recovery_active)
        self.assertIsNone(normal.anchor_chunk_id)

    def test_one_noisy_flag_is_ignored_but_strong_overlap_is_not(self):
        controller = IDENTITY_RECOVERY.IdentityRecoveryController(enabled=True)
        controller.observe_chunk(0, metas(False))

        noisy = controller.observe_chunk(1, metas(True, False, False, overlap=0.02))
        strong = controller.observe_chunk(2, metas(False, False, False, overlap=0.25))

        self.assertFalse(noisy.occlusion_risk)
        self.assertTrue(strong.occlusion_risk)
        self.assertEqual(strong.anchor_chunk_id, 1)

    def test_disabled_mode_never_changes_history(self):
        controller = IDENTITY_RECOVERY.IdentityRecoveryController(enabled=False)
        decision = controller.observe_chunk(0, metas(True, overlap=0.5))
        self.assertTrue(decision.occlusion_risk)
        self.assertFalse(decision.recovery_active)
        self.assertIsNone(decision.anchor_chunk_id)

    def test_invalid_overlap_is_safely_ignored(self):
        controller = IDENTITY_RECOVERY.IdentityRecoveryController(enabled=True)
        for value in (None, "bad", float("nan"), float("inf")):
            with self.subTest(value=value):
                decision = controller.observe_chunk(
                    0,
                    [{
                        "identity_occlusion_risk": False,
                        "identity_occlusion_overlap": value,
                    }],
                )
                self.assertEqual(decision.max_overlap, 0.0)


if __name__ == "__main__":
    unittest.main()
