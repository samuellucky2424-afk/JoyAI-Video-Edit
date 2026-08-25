import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
HTML_PATH = ROOT / "deploy" / "static" / "index.html"
SERVER_PATH = (
    ROOT / "deploy" / "xvideo" / "serving" / "serve_joyomni_streaming.py"
)
STREAMING_PATH = ROOT / "deploy" / "xvideo" / "serving" / "joyomni_streaming.py"
LATENT_CONTROL_PATH = (
    ROOT / "deploy" / "xvideo" / "serving" / "mouth_latent_control.py"
)
FACE_VALUE_CONTROL_PATH = (
    ROOT / "deploy" / "xvideo" / "serving" / "face_value_control.py"
)
SERVER_LAUNCHER_PATH = ROOT / "deploy" / "run_server.sh"


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

    def test_face_metadata_reaches_edit_condition_attention_values(self) -> None:
        """Require regional source-value control without touching identity KV."""
        streaming = STREAMING_PATH.read_text(encoding="utf-8")
        graph_runner = (
            ROOT / "deploy" / "xvideo" / "serving" / "graph_runner.py"
        ).read_text(encoding="utf-8")
        dit = (
            ROOT / "deploy" / "xvideo" / "models" / "dit" / "dit.py"
        ).read_text(encoding="utf-8")

        self.assertTrue(
            FACE_VALUE_CONTROL_PATH.exists(),
            "eye/mouth metadata still stops before attention values",
        )
        self.assertIn("face_value_control_enabled: bool = False", streaming)
        self.assertIn("build_face_value_scale(", streaming)
        self.assertIn("ref_video_value_scale=ref_video_value_scale", streaming)
        self.assertIn("in_ref_value_scale", graph_runner)
        self.assertIn("ref_video_value_scale=self.in_ref_value_scale", graph_runner)
        self.assertIn("runner.in_ref_value_scale.fill_(1.0)", streaming)
        self.assertIn("ref_video_value_scale: Optional[torch.Tensor] = None", dit)
        self.assertIn("img_value_scale=visual_value_scale", dit)
        self.assertIn("img_v = img_v * img_value_scale", dit)

        # The uploaded reference image remains the unmodified identity anchor.
        self.assertNotIn(
            "ref_video_value_scale=",
            streaming[
                streaming.index("def _encode_ref_image_latent"):
                streaming.index("def _chunk_last_frame_gray")
            ],
        )

    def test_eye_and_anatomy_metadata_cross_the_browser_server_contract(self) -> None:
        worker = (
            ROOT / "deploy" / "static" / "mediapipe-mouth-worker.js"
        ).read_text(encoding="utf-8")
        html = HTML_PATH.read_text(encoding="utf-8")
        server = SERVER_PATH.read_text(encoding="utf-8")

        self.assertIn("const EYE_BLENDSHAPES", worker)
        self.assertIn("function eyeRois(", worker)
        self.assertIn("eyeBlendshapes", worker)
        self.assertIn("eyeRois", worker)
        self.assertIn("eye_rois:", html)
        self.assertIn("eye_blendshapes:", html)
        self.assertIn("face_value_control: faceValueControl", html)
        self.assertIn('"eye_rois": payload.get("eye_rois")', server)
        self.assertIn(
            '"eye_blendshapes": payload.get("eye_blendshapes")',
            server,
        )

    def test_mouth_metadata_reaches_dit_source_conditioning_latent(self) -> None:
        """Require bounded ROI control after VAE encode and before DiT inference."""
        self.assertTrue(
            LATENT_CONTROL_PATH.exists(),
            "mouth metadata still stops before the inference tensor boundary",
        )
        streaming = STREAMING_PATH.read_text(encoding="utf-8")
        latent_control = LATENT_CONTROL_PATH.read_text(encoding="utf-8")

        self.assertIn("mouth_latent_control_enabled: bool = False", streaming)
        self.assertIn("def apply_mouth_latent_control(", latent_control)
        self.assertIn("build_mouth_control(metas", latent_control)
        self.assertIn("ref_chunk_latent = apply_mouth_latent_control(", streaming)

        encode_call = streaming.index(
            "ref_chunk_latent = self._encode_reference_chunk("
        )
        control_call = streaming.index(
            "ref_chunk_latent = apply_mouth_latent_control("
        )
        inference_call = streaming.index(
            "current_chunk_latents = self._denoise_chunk("
        )
        self.assertLess(encode_call, control_call)
        self.assertLess(
            control_call,
            inference_call,
            "mouth ROI control must modify source conditioning before DiT",
        )

        server = SERVER_PATH.read_text(encoding="utf-8")
        launcher = SERVER_LAUNCHER_PATH.read_text(encoding="utf-8")
        self.assertIn('"--mouth-latent-control"', server)
        self.assertIn("JOYOMNI_MOUTH_LATENT_CONTROL", launcher)
        self.assertIn("EXTRA_ARGS+=(--mouth-latent-control)", launcher)
        self.assertIn('"--face-value-control"', server)
        self.assertIn("JOYOMNI_FACE_VALUE_CONTROL", launcher)
        self.assertIn("EXTRA_ARGS+=(--face-value-control)", launcher)


if __name__ == "__main__":
    unittest.main()
