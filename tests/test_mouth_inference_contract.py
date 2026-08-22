import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
HTML_PATH = ROOT / "deploy" / "static" / "index.html"
SERVER_PATH = (
    ROOT / "deploy" / "xvideo" / "serving" / "serve_joyomni_streaming.py"
)
STREAMING_PATH = ROOT / "deploy" / "xvideo" / "serving" / "joyomni_streaming.py"

class MouthInferenceContractTests(unittest.TestCase):
    def test_high_quality_mouth_patch_reaches_source_conditioning_boundary(self) -> None:
        """Require the patch to be applied before the source frame enters VAE."""
        html = HTML_PATH.read_text(encoding="utf-8")
        server = SERVER_PATH.read_text(encoding="utf-8")

        self.assertIn("mouth_anatomy: available ?", html)
        self.assertIn("function mouthDetailPatchMeta(meta)", html)
        self.assertIn('mouth_patch: mouthPatchCapture.toDataURL("image/jpeg", 0.96)', html)
        self.assertIn('"mouth_anatomy": payload.get("mouth_anatomy")', server)
        self.assertIn('"mouth_patch": payload.get("mouth_patch")', server)
        self.assertIn(
            '"mouth_event_significant": payload.get(',
            server,
            "the browser's significant-mouth signal is dropped before inference",
        )
        patch_call = server.index("frame, mouth_patch_profile = apply_mouth_detail_patch(")
        inference_call = server.index("return session.push_frame(frame, frame_meta=frame_meta)")
        self.assertLess(
            patch_call,
            inference_call,
            "the high-quality source crop must be merged before VAE inference",
        )

    def test_mouth_control_does_not_modify_transformer_attention_values(self) -> None:
        streaming = STREAMING_PATH.read_text(encoding="utf-8")
        graph_runner = (
            ROOT / "deploy" / "xvideo" / "serving" / "graph_runner.py"
        ).read_text(encoding="utf-8")
        dit = (
            ROOT / "deploy" / "xvideo" / "models" / "dit" / "dit.py"
        ).read_text(encoding="utf-8")

        self.assertNotIn("_mouth_ref_video_value_scale", streaming)
        self.assertNotIn("ref_video_value_scale", streaming)
        self.assertNotIn("in_ref_value_scale", graph_runner)
        self.assertNotIn("ref_video_value_scale", dit)
        self.assertNotIn("img_value_scale", dit)


if __name__ == "__main__":
    unittest.main()
