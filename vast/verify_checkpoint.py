import argparse
import json
import os
import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEPLOY_ROOT = REPOSITORY_ROOT / "deploy"
if str(DEPLOY_ROOT) not in sys.path:
    sys.path.insert(0, str(DEPLOY_ROOT))

from xvideo.checkpoint_status import checkpoint_status


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verify the mounted JoyAI upgraded RV2V DiT checkpoint."
    )
    parser.add_argument(
        "--full-hash",
        action="store_true",
        help="Read and SHA-256 hash the complete 32.5 GB file.",
    )
    args = parser.parse_args()

    checkpoint_root = Path(
        os.getenv("JOYOMNI_CKPT_ROOT", "/workspace/joyai/checkpoints")
    )
    checkpoint = (
        checkpoint_root
        / "JoyAI-Video-Edit"
        / "dit"
        / "joyai_video_edit_dit_0811.pth"
    )
    report = checkpoint_status(checkpoint, full_hash=args.full_hash)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "current" else 2


if __name__ == "__main__":
    raise SystemExit(main())
