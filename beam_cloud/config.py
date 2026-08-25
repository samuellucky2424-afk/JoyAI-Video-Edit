"""Immutable Beam.cloud deployment settings for the validated JoyAI runtime."""

from pathlib import PurePosixPath


APP_NAME = "joyai-video-edit"
MODEL_DOWNLOAD_APP_NAME = "joyai-model-download"

# This digest was published from repository commit 7b3df0a51a438ff7f625e2811afa6782f8e58410.
# It adds bounded eye/mouth source-value conditioning without modifying the
# validated checkpoint or the uploaded reference-image identity KV.
IMAGE_URI = (
    "ghcr.io/samuellucky2424-afk/joyai-video-edit@"
    "sha256:4f3f77ba936f8e2d82e702fcd6502795ae547d697f268246c5025eefcfbdb700"
)
IMAGE_REVISION = "7b3df0a51a438ff7f625e2811afa6782f8e58410"

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
    # Isolate the regional attention-value experiment for the next live test.
    # It touches only source edit-condition values; checkpoint weights and the
    # uploaded reference-image identity KV remain unchanged.
    "JOYOMNI_MOUTH_LATENT_CONTROL": "0",
    "JOYOMNI_FACE_VALUE_CONTROL": "1",
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
