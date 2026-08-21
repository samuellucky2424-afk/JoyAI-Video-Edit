import os
import signal
import subprocess
import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEPLOY_ROOT = REPOSITORY_ROOT / "deploy"
if str(DEPLOY_ROOT) not in sys.path:
    sys.path.insert(0, str(DEPLOY_ROOT))

from xvideo.checkpoint_status import checkpoint_status


TRUE_VALUES = {"1", "true", "yes", "on"}
FALSE_VALUES = {"0", "false", "no", "off"}


def env_enabled(name: str, *, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default

    normalized = value.strip().lower()
    if normalized in TRUE_VALUES:
        return True
    if normalized in FALSE_VALUES:
        return False

    choices = ", ".join(sorted(TRUE_VALUES | FALSE_VALUES))
    raise ValueError(f"{name} must be one of: {choices}")


def cuda_capability_matches(
    actual: tuple[int, int] | None = None,
    device_name: str | None = None,
) -> bool:
    """Reject the wrong GPU before loading the 32.5 GB DiT checkpoint."""
    expected = os.getenv("JOYOMNI_EXPECTED_CUDA_CAPABILITY", "").strip()
    if not expected:
        return True

    if actual is None:
        import torch

        if not torch.cuda.is_available():
            print("This image requires a CUDA GPU, but CUDA is unavailable.", flush=True)
            return False
        actual = torch.cuda.get_device_capability(0)
        device_name = torch.cuda.get_device_name(0)

    actual_text = f"{actual[0]}.{actual[1]}"
    print(
        "JoyAI CUDA target check: "
        f"expected sm_{expected.replace('.', '')}, "
        f"got sm_{actual_text.replace('.', '')} "
        f"({device_name or 'unknown GPU'}).",
        flush=True,
    )
    if actual_text == expected:
        return True

    print("Refusing to load this image on incompatible hardware.", flush=True)
    return False


def required_checkpoint_items(checkpoint_root: Path) -> list[Path]:
    return [
        checkpoint_root
        / "JoyAI-Video-Edit"
        / "dit"
        / "joyai_video_edit_dit_0811.pth",
        checkpoint_root / "JoyAI-Video-Edit" / "vae" / "config.json",
        checkpoint_root
        / "JoyAI-Video-Edit"
        / "vae"
        / "diffusion_pytorch_model.safetensors",
        checkpoint_root / "MiMo-VL-7B-RL-2508",
        checkpoint_root / "face_detection_yunet_2023mar.onnx",
    ]


def build_model_command(repository_root: Path, *, preload: bool) -> list[str]:
    return [
        "bash",
        str(repository_root / "deploy" / "run_server.sh"),
        "--preload" if preload else "--no-preload",
    ]


def main() -> int:
    if not cuda_capability_matches():
        return 1

    checkpoint_root = Path(
        os.getenv("JOYOMNI_CKPT_ROOT", "/workspace/joyai/checkpoints")
    )
    checkpoint_root.mkdir(parents=True, exist_ok=True)

    required_items = required_checkpoint_items(checkpoint_root)
    missing_items = [str(item) for item in required_items if not item.exists()]
    if missing_items:
        print("Required model files are missing:", flush=True)
        for item in missing_items:
            print(f" - {item}", flush=True)
        print("\nDownload them once onto the attached Vast volume:", flush=True)
        print("python3 /opt/joyai/vast/download_models.py", flush=True)
        return 1

    dit_status = checkpoint_status(required_items[0])
    status = dit_status["status"]
    print(
        f"JoyAI RV2V checkpoint status: {status} "
        f"({dit_status['verification']}).",
        flush=True,
    )
    if status == "stale":
        print(
            "The mounted volume contains different 0811 DiT weights. "
            "Refusing to start with a stale checkpoint.",
            flush=True,
        )
        print("Run: python3 /opt/joyai/vast/download_models.py", flush=True)
        return 1
    if status == "unknown":
        print(
            "Checkpoint metadata is unavailable. For a one-time full "
            "verification run: python3 /opt/joyai/vast/verify_checkpoint.py "
            "--full-hash",
            flush=True,
        )

    try:
        preload = env_enabled("JOYOMNI_PRELOAD", default=True)
    except ValueError as error:
        print(f"Invalid startup configuration: {error}", flush=True)
        return 1

    environment = os.environ.copy()
    model_port = environment.get("PORT", environment.get("JOYOMNI_PORT", "8080"))
    environment["JOYOMNI_PORT"] = model_port

    command = build_model_command(REPOSITORY_ROOT, preload=preload)
    print(f"Starting JoyAI on Vast port {model_port}...", flush=True)
    print("Routes: GET /, GET /health, POST /load, WS /ws", flush=True)
    if preload:
        print("Preloading the validated JoyAI runtime...", flush=True)

    model_process = subprocess.Popen(
        command,
        cwd=REPOSITORY_ROOT,
        env=environment,
    )

    def forward_signal(signum: int, _frame: object) -> None:
        if model_process.poll() is None:
            model_process.send_signal(signum)

    signal.signal(signal.SIGTERM, forward_signal)
    signal.signal(signal.SIGINT, forward_signal)
    return model_process.wait()


if __name__ == "__main__":
    raise SystemExit(main())
