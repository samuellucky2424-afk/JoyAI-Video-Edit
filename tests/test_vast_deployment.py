import py_compile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class VastDeploymentContractTests(unittest.TestCase):
    def test_vast_python_entrypoints_compile(self) -> None:
        for name in ("start.py", "download_models.py", "verify_checkpoint.py"):
            py_compile.compile(str(ROOT / "vast" / name), doraise=True)

    def test_vast_dockerfile_is_provider_specific(self) -> None:
        dockerfile = (ROOT / "Dockerfile.vast").read_text(encoding="utf-8")
        self.assertIn("nvidia/cuda:12.8.1-devel-ubuntu22.04", dockerfile)
        self.assertIn("TORCH_CUDA_ARCH_LIST=12.0", dockerfile)
        self.assertIn("JOYOMNI_EXPECTED_CUDA_CAPABILITY=12.0", dockerfile)
        self.assertIn("/workspace/joyai/checkpoints", dockerfile)
        self.assertIn('["python3", "/opt/joyai/vast/start.py"]', dockerfile)
        self.assertNotIn("/runpod-volume", dockerfile)
        self.assertNotIn("runpod/start.py", dockerfile)

    def test_vast_start_uses_the_original_server(self) -> None:
        start = (ROOT / "vast" / "start.py").read_text(encoding="utf-8")
        self.assertIn('repository_root / "deploy" / "run_server.sh"', start)
        self.assertIn('"--preload" if preload else "--no-preload"', start)
        self.assertNotIn("health_server.py", start)

    def test_vast_downloader_pins_the_validated_rv2v_release(self) -> None:
        downloader = (ROOT / "vast" / "download_models.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("eda14f342ef99c52485bbb8dc271c29b42298089", downloader)
        self.assertIn("/workspace/joyai/checkpoints", downloader)

    def test_vast_workflow_is_manual_and_uses_the_vast_dockerfile(self) -> None:
        workflow = (
            ROOT / ".github" / "workflows" / "build-vast-image.yml"
        ).read_text(encoding="utf-8")
        self.assertIn("workflow_dispatch:", workflow)
        self.assertNotIn("\n  push:", workflow)
        self.assertIn("file: ./Dockerfile.vast", workflow)
        self.assertIn(":vast-rtx-pro-6000", workflow)


if __name__ == "__main__":
    unittest.main()
