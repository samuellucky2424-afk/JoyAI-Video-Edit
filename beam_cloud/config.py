"""Immutable Beam.cloud deployment settings for the validated JoyAI runtime."""

from pathlib import PurePosixPath


APP_NAME = "joyai-video-edit"
MODEL_DOWNLOAD_APP_NAME = "joyai-model-download"

# This digest was published from repository commit 5bea2f6ba296fe34fdfaaa4da1497e3de19d4008.
# It adds bounded face-occlusion recovery without modifying the validated checkpoint.
IMAGE_URI = (
    "ghcr.io/samuellucky2424-afk/joyai-video-edit@"
    "sha256:a18a738d656daba513ff744bb670aef02be45549486be58bd9ee773b4fbe790b"
)
IMAGE_REVISION = "5bea2f6ba296fe34fdfaaa4da1497e3de19d4008"

GPU_TYPE = "RTXPro6000"
GPU_POOL = "joyai-rtx-pro-6000"
MODEL_VOLUME_NAME = "joyai-models-v1"
MODEL_VOLUME_MOUNT = "/workspace/joyai"
CHECKPOINT_ROOT = PurePosixPath(MODEL_VOLUME_MOUNT) / "checkpoints"
MODEL_READY_MARKER = CHECKPOINT_ROOT / "beam_models_ready.json"
PORT = 8080

# Keep the validated 840x480 / 24 FPS runtime. A 720p switch would change the
# compiled graph and substantially increase latency and VRAM pressure.
RUNTIME_ENV = {
    "PORT": str(PORT),
    "JOYOMNI_PORT": str(PORT),
    "JOYOMNI_CKPT_ROOT": str(CHECKPOINT_ROOT),
    "JOYOMNI_CACHE_ROOT": (
        f"{MODEL_VOLUME_MOUNT}/cache/"
        "rtx-pro-6000-blackwell-torch291-cu128-oomfix-v2"
    ),
    "JOYOMNI_CACHE_READY_MARKER": (
        f"{MODEL_VOLUME_MOUNT}/cache/"
        "rtx-pro-6000-blackwell-torch291-cu128-oomfix-v2/ready.json"
    ),
    "JOYOMNI_EXPECTED_CUDA_CAPABILITY": "12.0",
    "JOYOMNI_PRELOAD": "1",
    "JOYOMNI_WIDTH": "840",
    "JOYOMNI_HEIGHT": "480",
    "JOYOMNI_FPS": "24",
    "JOYOMNI_NUM_INFERENCE_STEPS": "2",
    "JOYOMNI_FP8_IMG": "1",
    "JOYOMNI_FP8_TXT": "1",
    "JOYOMNI_CUDA_GRAPH": "1",
    "JOYOMNI_SAGE_ATTN": "0",
    "JOYOMNI_TXT_PARALLEL": "0",
    "JOYOMNI_FP8_FAST_ACCUM": "0",
    "JOYOMNI_VAE_COMPILE": "1",
    "JOYOMNI_VAE_COMPILE_STRICT": "1",
    "JOYOMNI_LOAD_WARMUP_STRICT": "1",
    "JOYOMNI_FULL_WARMUP_TIMEOUT_SECONDS": "300",
    "JOYOMNI_WARMUP_BOTH_ORIENTATIONS": "0",
    "JOYOMNI_WARMUP_REFERENCE_BUCKETS": "1",
    "JOYOMNI_RECORD_ENABLED": "0",
    "JOYOMNI_ONLINE_GATE_ENABLED": "0",
}

DOWNLOAD_ENV = {
    "JOYOMNI_CKPT_ROOT": str(CHECKPOINT_ROOT),
    "HF_HOME": f"{MODEL_VOLUME_MOUNT}/huggingface",
    "HF_XET_HIGH_PERFORMANCE": "1",
}


def joyai_entrypoint() -> list[str]:
    """Fail before loading the GPU if the verified model volume is not ready."""
    marker = str(MODEL_READY_MARKER)
    command = (
        f'test -f "{marker}" || '
        f'{{ echo "Missing verified Beam model marker: {marker}"; '
        'echo "Run the model_download command for your OS from beam_cloud/README.md"; '
        "exit 1; }; "
        "exec python3 /opt/joyai/vast/start.py"
    )
    return ["bash", "-lc", command]
