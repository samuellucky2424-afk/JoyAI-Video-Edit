from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any


JOYAI_DIT_FILENAME = "dit/joyai_video_edit_dit_0811.pth"
JOYAI_DIT_RELEASE_COMMIT = "eda14f342ef99c52485bbb8dc271c29b42298089"
JOYAI_DIT_XET_HASH = "86a577acfe936e9b56ae7c89f04d2db61d5b69f97f6fa6496da8f7cfcc47305f"
JOYAI_DIT_SHA256 = "b3904b6fda53d13b230918bb616f322d12cfb2337b0e8d9dc203cdabc36605ba"


def _metadata_path(checkpoint_path: Path) -> Path:
    """Return huggingface_hub's local_dir metadata path for the DiT file."""
    local_dir = checkpoint_path.parent.parent
    return local_dir / ".cache" / "huggingface" / "download" / "dit" / (
        checkpoint_path.name + ".metadata"
    )


def _read_metadata(checkpoint_path: Path) -> tuple[str | None, str | None, str | None]:
    metadata_path = _metadata_path(checkpoint_path)
    if not metadata_path.is_file():
        return None, None, None

    try:
        with metadata_path.open(encoding="utf-8") as metadata:
            revision = metadata.readline().strip() or None
            etag = metadata.readline().strip().strip('"') or None
            metadata.readline()  # timestamp; validate only by successful reading
    except (OSError, UnicodeError) as error:
        return None, None, f"{type(error).__name__}: {error}"

    return revision, etag, None


def checkpoint_status(
    checkpoint: str | Path,
    *,
    full_hash: bool = False,
) -> dict[str, Any]:
    """Inspect the RV2V DiT without reading 32.5 GB unless explicitly requested."""
    checkpoint_path = Path(checkpoint)
    report: dict[str, Any] = {
        "status": "missing",
        "path": str(checkpoint_path),
        "expected_release_commit": JOYAI_DIT_RELEASE_COMMIT,
        "expected_xet_hash": JOYAI_DIT_XET_HASH,
        "expected_sha256": JOYAI_DIT_SHA256,
        "metadata_revision": None,
        "metadata_etag": None,
        "sha256": None,
        "size_bytes": None,
        "verification": "none",
    }

    if not checkpoint_path.is_file():
        return report

    try:
        report["size_bytes"] = checkpoint_path.stat().st_size
    except OSError:
        pass

    revision, etag, metadata_error = _read_metadata(checkpoint_path)
    report["metadata_revision"] = revision
    report["metadata_etag"] = etag
    if metadata_error is not None:
        report["metadata_error"] = metadata_error

    if etag in {JOYAI_DIT_XET_HASH, JOYAI_DIT_SHA256}:
        report["status"] = "current"
        report["verification"] = "huggingface_metadata"
    elif etag:
        report["status"] = "stale"
        report["verification"] = "huggingface_metadata"
    else:
        report["status"] = "unknown"

    if full_hash:
        digest = hashlib.sha256()
        with checkpoint_path.open("rb") as checkpoint_file:
            for block in iter(lambda: checkpoint_file.read(16 * 1024 * 1024), b""):
                digest.update(block)
        actual_sha256 = digest.hexdigest()
        report["sha256"] = actual_sha256
        report["verification"] = "sha256"
        report["status"] = (
            "current" if actual_sha256 == JOYAI_DIT_SHA256 else "stale"
        )

    return report
