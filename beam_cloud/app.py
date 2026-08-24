"""Beam.cloud Pods for downloading and serving the validated JoyAI runtime."""

from beam import GpuType, Image, Pod, Volume

from beam_cloud.config import (
    APP_NAME,
    DOWNLOAD_ENV,
    GPU_POOL,
    GPU_TYPE,
    IMAGE_URI,
    MODEL_DOWNLOAD_APP_NAME,
    MODEL_VOLUME_MOUNT,
    MODEL_VOLUME_NAME,
    PORT,
    RUNTIME_ENV,
    joyai_entrypoint,
)


model_volume = Volume(
    name=MODEL_VOLUME_NAME,
    mount_path=MODEL_VOLUME_MOUNT,
)

# This small CPU image is used only for the one-time model download. It avoids
# reserving a paid GPU while roughly 50 GB of immutable model data is fetched.
download_image = Image(
    python_version="python3.11",
    python_packages=[
        "huggingface_hub==0.36.0",
        "hf-xet==1.2.0",
    ],
)

model_download = Pod(
    app=MODEL_DOWNLOAD_APP_NAME,
    name=MODEL_DOWNLOAD_APP_NAME,
    entrypoint=["python3", "beam_cloud/download_models.py"],
    cpu=4,
    memory="8Gi",
    image=download_image,
    volumes=[model_volume],
    env=DOWNLOAD_ENV,
    # This Pod has no HTTP connection to keep it "active" while it downloads.
    # Zero lets Beam's scheduler stop it immediately (exit code 558). Keep it
    # alive until the downloader process exits normally after verification.
    keep_warm_seconds=-1,
    authorized=True,
)

# The image is imported by immutable digest; Beam does not rebuild the CUDA,
# CUTLASS, PyTorch, or JoyAI runtime layers.
joyai = Pod(
    app=APP_NAME,
    name=APP_NAME,
    entrypoint=joyai_entrypoint(),
    ports=[PORT],
    cpu=8,
    # Beam currently accepts at most 64 GiB of Pod system RAM. Model weights
    # remain on the RTX PRO 6000's 96 GB VRAM; this does not alter the model.
    memory="64Gi",
    gpu=GpuType.RTXPro6000,
    gpu_count=1,
    image=Image.from_registry(IMAGE_URI),
    volumes=[model_volume],
    env=RUNTIME_ENV,
    pool=GPU_POOL,
    keep_warm_seconds=-1,
    # The current browser client opens a native WebSocket and cannot attach a
    # Beam bearer header. Keep the test URL public and release it after use.
    authorized=False,
)

assert GPU_TYPE == GpuType.RTXPro6000.value
