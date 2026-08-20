import os
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
    """Read a strict boolean environment flag.

    A typo must fail during startup instead of silently disabling model preload
    and leaving RunPod's load balancer waiting forever for a ready worker.
    """
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
    """Fail early when a GPU-specific image is assigned to the wrong GPU.

    CUDA extension failures appear much later in startup and still consume
    billed GPU time.  A targeted image can set
    ``JOYOMNI_EXPECTED_CUDA_CAPABILITY`` to make the mismatch explicit before
    the 32.5 GB checkpoint is loaded.
    """
    expected = os.getenv("JOYOMNI_EXPECTED_CUDA_CAPABILITY", "").strip()
    if not expected:
        return True

    if actual is None:
        import torch

        if not torch.cuda.is_available():
            print(
                "This image requires a CUDA GPU, but CUDA is unavailable.",
                flush=True,
            )
            return False
        actual = torch.cuda.get_device_capability(0)
        device_name = torch.cuda.get_device_name(0)

    actual_text = f"{actual[0]}.{actual[1]}"
    print(
        "JoyAI CUDA target check: "
        f"expected sm_{expected.replace('.', '')}, got sm_{actual_text.replace('.', '')} "
        f"({device_name or 'unknown GPU'}).",
        flush=True,
    )
    if actual_text == expected:
        return True

    print(
        "Refusing to load a GPU-specific image on incompatible hardware.",
        flush=True,
    )
    return False


def required_checkpoint_items(checkpoint_root: Path) -> list[Path]:
    return [
        checkpoint_root
        / "JoyAI-Video-Edit"
        / "dit"
        / "joyai_video_edit_dit_0811.pth",
        checkpoint_root
        / "JoyAI-Video-Edit"
        / "vae"
        / "config.json",
        checkpoint_root
        / "JoyAI-Video-Edit"
        / "vae"
        / "diffusion_pytorch_model.safetensors",
        checkpoint_root / "MiMo-VL-7B-RL-2508",
    ]


def build_model_command(repository_root: Path, *, preload: bool) -> list[str]:
    return [
        "bash",
        str(repository_root / "deploy" / "run_server.sh"),
        "--preload" if preload else "--no-preload",
    ]


def main() -> int:
    repository_root = REPOSITORY_ROOT
    if not cuda_capability_matches():
        return 1

    checkpoint_root = Path(
        os.getenv(
            "JOYOMNI_CKPT_ROOT",
            "/runpod-volume/joyai/checkpoints",
        )
    )

    missing_items = [
        str(item)
        for item in required_checkpoint_items(checkpoint_root)
        if not item.exists()
    ]

    if missing_items:
        print("Required model files are missing:", flush=True)
        for item in missing_items:
            print(f" - {item}", flush=True)

        print(
            "\nRun this command once on the attached RunPod volume:",
            flush=True,
        )
        print(
            "python3 /opt/joyai/runpod/download_models.py",
            flush=True,
        )
        return 1

    dit_checkpoint = required_checkpoint_items(checkpoint_root)[0]
    dit_status = checkpoint_status(dit_checkpoint)
    status = dit_status["status"]
    print(
        "JoyAI RV2V checkpoint status: "
        f"{status} ({dit_status['verification']}).",
        flush=True,
    )
    if status == "stale":
        print(
            "The mounted volume contains a different 0811 DiT than the "
            "upgraded RV2V release. Refusing to load stale weights.",
            flush=True,
        )
        print(
            "Update the volume with: "
            "python3 /opt/joyai/runpod/download_models.py",
            flush=True,
        )
        return 1
    if status == "unknown":
        print(
            "Hugging Face metadata is unavailable, so this checkpoint cannot "
            "be identified without a one-time full hash. Run: "
            "python3 /opt/joyai/runpod/verify_checkpoint.py --full-hash",
            flush=True,
        )

    environment = os.environ.copy()

    # RunPod supplies PORT for public traffic. PORT_HEALTH is reserved for the
    # internal /ping server and must not be used by clients.
    model_port = environment.get(
        "PORT",
        environment.get("JOYOMNI_PORT", "8080"),
    )
    environment["JOYOMNI_PORT"] = model_port

    try:
        preload = env_enabled("JOYOMNI_PRELOAD", default=True)
    except ValueError as error:
        print(f"Invalid startup configuration: {error}", flush=True)
        return 1

    print(f"Starting JoyAI on public port {model_port}...", flush=True)
    print(
        "Routes: GET /, GET /health, POST /load, WS /ws; "
        "RunPod health: GET /ping on PORT_HEALTH.",
        flush=True,
    )
    if preload:
        print(
            "Preloading the JoyAI runtime before RunPod marks this worker ready...",
            flush=True,
        )

    health_process = subprocess.Popen(
        [
            sys.executable,
            str(repository_root / "runpod" / "health_server.py"),
        ],
        cwd=repository_root,
        env=environment,
    )

    exit_code = 1
    try:
        model_process = subprocess.Popen(
            build_model_command(repository_root, preload=preload),
            cwd=repository_root,
            env=environment,
        )
        exit_code = model_process.wait()
    finally:
        health_process.terminate()
        try:
            health_process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            health_process.kill()

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
