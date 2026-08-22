import ast
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
HTML_PATH = ROOT / "deploy" / "static" / "index.html"
SERVER_PATH = (
    ROOT / "deploy" / "xvideo" / "serving" / "serve_joyomni_streaming.py"
)
STREAMING_PATH = ROOT / "deploy" / "xvideo" / "serving" / "joyomni_streaming.py"


def _class_method(tree: ast.AST, class_name: str, method_name: str) -> ast.FunctionDef:
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for child in node.body:
                if isinstance(child, ast.FunctionDef) and child.name == method_name:
                    return child
    raise AssertionError(f"missing {class_name}.{method_name}")


class MouthInferenceContractTests(unittest.TestCase):
    def test_mouth_metadata_reaches_denoise_inference_boundary(self) -> None:
        """Require the browser mouth signal to cross every runtime boundary.

        This intentionally fails until the DIT worker passes the chunk's source
        metadata into ``_denoise_chunk``.  It does not prescribe how the model
        should consume that signal; that interface must be chosen only after
        the existing inference controls have been inspected.
        """
        html = HTML_PATH.read_text(encoding="utf-8")
        server = SERVER_PATH.read_text(encoding="utf-8")
        streaming_source = STREAMING_PATH.read_text(encoding="utf-8")
        streaming_tree = ast.parse(streaming_source)

        self.assertIn("mouth_anatomy: available ?", html)
        self.assertIn('"mouth_anatomy": payload.get("mouth_anatomy")', server)
        self.assertIn(
            '"mouth_event_significant": payload.get(',
            server,
            "the browser's significant-mouth signal is dropped before inference",
        )

        dit_worker = _class_method(
            streaming_tree, "JoyOmniV2VStreamingSession", "_dit_worker"
        )
        denoise_chunk = _class_method(
            streaming_tree, "JoyOmniV2VStreamingSession", "_denoise_chunk"
        )
        denoise_calls = [
            node
            for node in ast.walk(dit_worker)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "_denoise_chunk"
        ]
        self.assertEqual(len(denoise_calls), 1)

        source_meta_keywords = [
            keyword
            for keyword in denoise_calls[0].keywords
            if keyword.arg == "source_metas"
        ]
        self.assertEqual(
            len(source_meta_keywords),
            1,
            "mouth metadata stops in _ChunkJob.source_metas: _dit_worker does "
            "not pass it into _denoise_chunk",
        )
        self.assertEqual(
            ast.unparse(source_meta_keywords[0].value),
            "encoded.job.source_metas",
        )
        self.assertIn(
            "source_metas",
            [argument.arg for argument in denoise_chunk.args.kwonlyargs],
            "the denoise inference boundary does not accept source metadata",
        )

    def test_roi_scale_reaches_eager_and_cuda_graph_attention_values(self) -> None:
        streaming = STREAMING_PATH.read_text(encoding="utf-8")
        graph_runner = (
            ROOT / "deploy" / "xvideo" / "serving" / "graph_runner.py"
        ).read_text(encoding="utf-8")
        dit = (
            ROOT / "deploy" / "xvideo" / "models" / "dit" / "dit.py"
        ).read_text(encoding="utf-8")

        self.assertIn("def _mouth_ref_video_value_scale(", streaming)
        self.assertIn("ref_video_value_scale=ref_video_value_scale", streaming)
        self.assertIn("runner.in_ref_value_scale.copy_", streaming)
        self.assertIn("self.in_ref_value_scale = torch.ones(", graph_runner)
        self.assertIn("ref_video_value_scale=self.in_ref_value_scale", graph_runner)
        self.assertIn("ref_video_value_scale: Optional[torch.Tensor] = None", dit)
        self.assertIn("img_value_scale: Optional[torch.Tensor] = None", dit)
        self.assertIn("img_v = img_v * img_value_scale.to(", dit)


if __name__ == "__main__":
    unittest.main()
