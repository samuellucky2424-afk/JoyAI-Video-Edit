"""Download and verify JoyAI's immutable model set on a Beam persistent volume."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import urllib.request
from pathlib import Path

JOYAI_REPO = "jdopensource/JoyAI-Video-Edit"
JOYAI_REVISION = "eda14f342ef99c52485bbb8dc271c29b42298089"
JOYAI_DIT_RELATIVE_PATH = Path("JoyAI-Video-Edit/dit/joyai_video_edit_dit_0811.pth")
JOYAI_DIT_SHA256 = "b3904b6fda53d13b230918bb616f322d12cfb2337b0e8d9dc203cdabc36605ba"

MIMO_REPO = "XiaomiMiMo/MiMo-VL-7B-RL-2508"
MIMO_REVISION = "4bfb270765825d2fa059011deb4c96fdd579be6f"

YUNET_FILENAME = "face_detection_yunet_2023mar.onnx"
YUNET_REVISION = "f12e12798e8314f7c074a6656816c048dcc95b7a"
YUNET_URL = (
    "https://media.githubusercontent.com/media/opencv/opencv_zoo/"
    f"{YUNET_REVISION}/models/face_detection_yunet/{YUNET_FILENAME}"
)
YUNET_SHA256 = "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4"
READY_MARKER = "beam_models_ready.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_sha256(path: Path, expected: str) -> None:
    actual = sha256_file(path)
    if actual != expected:
        raise RuntimeError(
            f"SHA256 mismatch for {path}: expected {expected}, got {actual}"
        )
    print(f"Verified {path.name}: {actual}", flush=True)


def verify_required_layout(checkpoint_root: Path) -> None:
    required_files = [
        checkpoint_root / JOYAI_DIT_RELATIVE_PATH,
        checkpoint_root / "JoyAI-Video-Edit/vae/config.json",
        checkpoint_root
        / "JoyAI-Video-Edit/vae/diffusion_pytorch_model.safetensors",
        checkpoint_root / "MiMo-VL-7B-RL-2508/config.json",
        checkpoint_root / YUNET_FILENAME,
    ]
    missing = [str(path) for path in required_files if not path.is_file()]
    mimo_directory = checkpoint_root / "MiMo-VL-7B-RL-2508"
    if not any(mimo_directory.glob("*.safetensors")):
        missing.append(f"{mimo_directory}/*.safetensors")
    if missing:
        raise RuntimeError("Required model files are missing:\n - " + "\n - ".join(missing))


def download_yunet(destination: Path) -> None:
    if destination.is_file() and sha256_file(destination) == YUNET_SHA256:
        print(f"YuNet already verified: {destination}", flush=True)
        return

    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f"{destination.name}.",
        suffix=".download",
        dir=destination.parent,
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    try:
        urllib.request.urlretrieve(YUNET_URL, temporary_path)
        verify_sha256(temporary_path, YUNET_SHA256)
        temporary_path.replace(destination)
    finally:
        temporary_path.unlink(missing_ok=True)


def write_ready_marker(checkpoint_root: Path) -> None:
    marker = checkpoint_root / READY_MARKER
    payload = {
        "status": "current",
        "joyai_repo": JOYAI_REPO,
        "joyai_revision": JOYAI_REVISION,
        "joyai_dit_sha256": JOYAI_DIT_SHA256,
        "mimo_repo": MIMO_REPO,
        "mimo_revision": MIMO_REVISION,
        "yunet_revision": YUNET_REVISION,
        "yunet_sha256": YUNET_SHA256,
    }
    temporary = marker.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(marker)
    print(f"Verified model marker written: {marker}", flush=True)


def main() -> int:
    from huggingface_hub import snapshot_download

    checkpoint_root = Path(
        os.getenv("JOYOMNI_CKPT_ROOT", "/workspace/joyai/checkpoints")
    )
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    (checkpoint_root / READY_MARKER).unlink(missing_ok=True)
    hf_token = os.getenv("HF_TOKEN") or None

    print("Downloading the pinned JoyAI 0811 RV2V model and VAE...", flush=True)
    snapshot_download(
        repo_id=JOYAI_REPO,
        revision=JOYAI_REVISION,
        local_dir=checkpoint_root / "JoyAI-Video-Edit",
        allow_patterns=[
            "dit/joyai_video_edit_dit_0811.pth",
            "vae/*",
        ],
        token=hf_token,
        max_workers=8,
    )

    print("Downloading the pinned MiMo-VL text and vision encoder...", flush=True)
    snapshot_download(
        repo_id=MIMO_REPO,
        revision=MIMO_REVISION,
        local_dir=checkpoint_root / "MiMo-VL-7B-RL-2508",
        token=hf_token,
        max_workers=8,
    )

    print("Downloading the pinned YuNet face detector...", flush=True)
    download_yunet(checkpoint_root / YUNET_FILENAME)

    verify_required_layout(checkpoint_root)
    dit_path = checkpoint_root / JOYAI_DIT_RELATIVE_PATH
    verify_sha256(dit_path, JOYAI_DIT_SHA256)
    write_ready_marker(checkpoint_root)
    print(f"All required model files are ready at {checkpoint_root}.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
