import importlib.util
import unittest
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "deploy" / "xvideo" / "serving" / "pe.py"
SPEC = importlib.util.spec_from_file_location("prompt_enhancer", MODULE_PATH)
PE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(PE)


class PromptEnhancerTests(unittest.TestCase):
    def _enhancer_with_capture(self):
        enhancer = PE.PromptEnhancer.__new__(PE.PromptEnhancer)
        captured = {}

        def fake_chat(system_prompt, user_text, images_b64, raw_fallback=""):
            captured["system_prompt"] = system_prompt
            captured["user_text"] = user_text
            captured["images_b64"] = images_b64
            captured["raw_fallback"] = raw_fallback
            return "enhanced"

        enhancer._chat = fake_chat
        return enhancer, captured

    def test_reference_identity_precedes_source_performance_frame(self):
        enhancer, captured = self._enhancer_with_capture()
        reference = Image.new("RGB", (8, 8), (210, 120, 80))
        source = Image.new("RGB", (8, 8), (20, 40, 60))

        result = enhancer(
            "rv2v",
            "Replace the subject.",
            video=[source],
            images=[reference],
        )

        self.assertEqual(result, "enhanced")
        self.assertEqual(len(captured["images_b64"]), 2)
        self.assertEqual(captured["images_b64"][0], PE._pil_to_b64(reference))
        self.assertEqual(captured["images_b64"][1], PE._pil_to_b64(source))
        self.assertIn("Image 1: the identity reference", captured["user_text"])
        self.assertIn("Image 2: the current source-video performance frame", captured["user_text"])
        self.assertIn("precise mouth shape and lip articulation", captured["user_text"])
        self.assertIn("natural skin tone and visible skin texture", captured["user_text"])

    def test_generic_v2v_without_reference_keeps_original_route(self):
        enhancer, captured = self._enhancer_with_capture()
        source = Image.new("RGB", (8, 8), (20, 40, 60))

        result = enhancer("v2v", "Add a hat.", video=[source])

        self.assertEqual(result, "enhanced")
        self.assertEqual(len(captured["images_b64"]), 1)
        self.assertIn("Visual Reference: Provided source video frame", captured["user_text"])
        self.assertNotIn("Image 1: the identity reference", captured["user_text"])

    def test_message_image_labels_are_one_based(self):
        messages = PE._build_messages("system", "user", ["aaa", "bbb"])
        labels = [
            item["text"]
            for item in messages[1]["content"]
            if item["type"] == "text" and item["text"].startswith("\n[Image")
        ]
        self.assertEqual(labels, ["\n[Image 1]:", "\n[Image 2]:"])


if __name__ == "__main__":
    unittest.main()
