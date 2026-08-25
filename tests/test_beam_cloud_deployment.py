import ast
import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def load_module(name: str, relative_path: str):
    spec = importlib.util.spec_from_file_location(name, REPOSITORY_ROOT / relative_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class BeamCloudDeploymentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = load_module("beam_cloud_config", "beam_cloud/config.py")
        cls.download = load_module(
            "beam_cloud_download_models", "beam_cloud/download_models.py"
        )

    def test_runtime_uses_immutable_validated_image(self):
        self.assertEqual(
            self.config.IMAGE_URI,
            "ghcr.io/samuellucky2424-afk/joyai-video-edit@"
            "sha256:a18a738d656daba513ff744bb670aef02be45549486be58bd9ee773b4fbe790b",
        )
        self.assertEqual(
            self.config.IMAGE_REVISION,
            "5bea2f6ba296fe34fdfaaa4da1497e3de19d4008",
        )
        self.assertNotIn(":latest", self.config.IMAGE_URI)

    def test_runtime_keeps_validated_blackwell_profile(self):
        env = self.config.RUNTIME_ENV
        self.assertEqual(self.config.GPU_TYPE, "RTXPro6000")
        self.assertEqual(env["JOYOMNI_EXPECTED_CUDA_CAPABILITY"], "12.0")
        self.assertEqual(env["JOYOMNI_WIDTH"], "840")
        self.assertEqual(env["JOYOMNI_HEIGHT"], "480")
        self.assertEqual(env["JOYOMNI_FPS"], "24")
        self.assertEqual(env["JOYOMNI_NUM_INFERENCE_STEPS"], "2")
        self.assertEqual(env["JOYOMNI_FP8_IMG"], "1")
        self.assertEqual(env["JOYOMNI_FP8_TXT"], "1")
        self.assertEqual(env["JOYOMNI_MOUTH_LATENT_CONTROL"], "1")

    def test_model_revisions_and_dit_digest_are_pinned(self):
        self.assertEqual(
            self.download.JOYAI_REVISION,
            "eda14f342ef99c52485bbb8dc271c29b42298089",
        )
        self.assertEqual(
            self.download.MIMO_REVISION,
            "4bfb270765825d2fa059011deb4c96fdd579be6f",
        )
        self.assertEqual(
            self.download.JOYAI_DIT_SHA256,
            "b3904b6fda53d13b230918bb616f322d12cfb2337b0e8d9dc203cdabc36605ba",
        )

    def test_ready_marker_is_written_only_after_verification_contract(self):
        source = (REPOSITORY_ROOT / "beam_cloud/download_models.py").read_text()
        tree = ast.parse(source)
        main = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "main"
        )
        calls = [
            node.func.id
            for node in ast.walk(main)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        ]
        self.assertIn("verify_sha256", calls)
        self.assertIn("write_ready_marker", calls)
        self.assertLess(calls.index("verify_sha256"), calls.index("write_ready_marker"))

    def test_hash_verification_rejects_modified_data(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "model.bin"
            path.write_bytes(b"validated")
            expected = hashlib.sha256(b"validated").hexdigest()
            self.download.verify_sha256(path, expected)
            path.write_bytes(b"changed")
            with self.assertRaises(RuntimeError):
                self.download.verify_sha256(path, expected)

    def test_required_layout_rejects_partial_download(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            with self.assertRaisesRegex(RuntimeError, "Required model files"):
                self.download.verify_required_layout(Path(temporary_directory))

    def test_ready_marker_contains_immutable_release_manifest(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            self.download.write_ready_marker(root)
            payload = json.loads((root / self.download.READY_MARKER).read_text())
            self.assertEqual(payload["status"], "current")
            self.assertEqual(payload["joyai_revision"], self.download.JOYAI_REVISION)
            self.assertEqual(payload["mimo_revision"], self.download.MIMO_REVISION)
            self.assertEqual(
                payload["joyai_dit_sha256"], self.download.JOYAI_DIT_SHA256
            )

    def test_gpu_entrypoint_refuses_unverified_volume(self):
        command = " ".join(self.config.joyai_entrypoint())
        self.assertIn(str(self.config.MODEL_READY_MARKER), command)
        self.assertIn("test -f", command)
        self.assertIn("beam_cloud/README.md", command)
        self.assertIn("exec python3 /opt/joyai/vast/start.py", command)

    def test_readme_documents_native_beam_handler_paths(self):
        readme = (REPOSITORY_ROOT / "beam_cloud/README.md").read_text()
        self.assertIn(r"beam_cloud\app.py:model_download", readme)
        self.assertIn("beam_cloud/app.py:model_download", readme)
        self.assertIn(r"beam_cloud\app.py:joyai", readme)
        self.assertIn("beam_cloud/app.py:joyai", readme)

    def test_model_download_pod_cannot_be_stopped_as_immediately_idle(self):
        source = (REPOSITORY_ROOT / "beam_cloud/app.py").read_text()
        tree = ast.parse(source)
        assignment = next(
            node
            for node in tree.body
            if isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "model_download"
                for target in node.targets
            )
        )
        self.assertIsInstance(assignment.value, ast.Call)
        keyword = next(
            item
            for item in assignment.value.keywords
            if item.arg == "keep_warm_seconds"
        )
        self.assertEqual(ast.literal_eval(keyword.value), -1)

    def test_runtime_pod_respects_beam_system_memory_limit(self):
        source = (REPOSITORY_ROOT / "beam_cloud/app.py").read_text()
        tree = ast.parse(source)
        assignment = next(
            node
            for node in tree.body
            if isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "joyai"
                for target in node.targets
            )
        )
        self.assertIsInstance(assignment.value, ast.Call)
        keyword = next(
            item for item in assignment.value.keywords if item.arg == "memory"
        )
        self.assertEqual(ast.literal_eval(keyword.value), "64Gi")


if __name__ == "__main__":
    unittest.main()
