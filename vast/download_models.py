import os
import urllib.request
from pathlib import Path

from huggingface_hub import snapshot_download


JOYAI_RV2V_REVISION = "eda14f342ef99c52485bbb8dc271c29b42298089"


def main() -> int:
    checkpoint_root = Path(
        os.getenv("JOYOMNI_CKPT_ROOT", "/workspace/joyai/checkpoints")
    )
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    hf_token = os.getenv("HF_TOKEN") or None

    print("Downloading the pinned JoyAI 0811 RV2V model and VAE...", flush=True)
    snapshot_download(
        repo_id="jdopensource/JoyAI-Video-Edit",
        revision=JOYAI_RV2V_REVISION,
        local_dir=checkpoint_root / "JoyAI-Video-Edit",
        allow_patterns=[
            "dit/joyai_video_edit_dit_0811.pth",
            "vae/*",
        ],
        token=hf_token,
        max_workers=8,
    )

    print("Downloading the MiMo-VL text and vision encoder...", flush=True)
    snapshot_download(
        repo_id="XiaomiMiMo/MiMo-VL-7B-RL-2508",
        local_dir=checkpoint_root / "MiMo-VL-7B-RL-2508",
        token=hf_token,
        max_workers=8,
    )

    face_model = checkpoint_root / "face_detection_yunet_2023mar.onnx"
    if not face_model.exists():
        print("Downloading the YuNet face detector...", flush=True)
        temporary_file = face_model.with_suffix(".download")
        urllib.request.urlretrieve(
            "https://media.githubusercontent.com/media/opencv/"
            "opencv_zoo/main/models/face_detection_yunet/"
            "face_detection_yunet_2023mar.onnx",
            temporary_file,
        )
        temporary_file.replace(face_model)

    print("All required model files are ready.", flush=True)
    print(f"Checkpoint location: {checkpoint_root}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
