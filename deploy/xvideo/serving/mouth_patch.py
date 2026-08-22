"""Preserve source-mouth evidence without changing JoyAI model behavior.

The browser can send a tiny, high-quality JPEG crop from the same source frame
as the normal H.264 uplink.  This module validates and feather-pastes that crop
back into the decoded source frame before the existing VAE conditioning path.
It never composites source pixels into generated output frames.
"""

from __future__ import annotations

import base64
import binascii
import io
import math
from typing import Any, Mapping

from PIL import Image, ImageDraw, ImageFilter, UnidentifiedImageError

from xvideo.serving.mouth_control import build_mouth_control


MAX_MOUTH_PATCH_BYTES = 64 * 1024
MAX_MOUTH_PATCH_DIMENSION = 320
MIN_MOUTH_PATCH_DIMENSION = 4
_JPEG_DATA_PREFIX = "data:image/jpeg;base64,"
_LANCZOS = getattr(Image, "Resampling", Image).LANCZOS


def _profile(control, *, applied: bool, reason: str, size=None, byte_count=0):
    fields = control.profile_fields()
    fields.update(
        {
            "mouth_patch_applied": int(applied),
            "mouth_patch_reason": reason,
            "mouth_patch_bytes": int(byte_count),
            "mouth_patch_size": list(size) if size is not None else None,
        }
    )
    return fields


def _decode_patch(value: Any) -> tuple[Image.Image | None, int, str]:
    if not isinstance(value, str) or not value.startswith(_JPEG_DATA_PREFIX):
        return None, 0, "missing_or_invalid_patch"
    encoded = value[len(_JPEG_DATA_PREFIX):]
    # Reject oversized text before allocating its decoded representation.
    if not encoded or len(encoded) > math.ceil(MAX_MOUTH_PATCH_BYTES * 4 / 3) + 4:
        return None, 0, "patch_too_large"
    try:
        payload = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError):
        return None, 0, "invalid_patch_base64"
    if not payload or len(payload) > MAX_MOUTH_PATCH_BYTES:
        return None, len(payload), "patch_too_large"
    try:
        with Image.open(io.BytesIO(payload)) as image:
            width, height = image.size
            if (
                width < MIN_MOUTH_PATCH_DIMENSION
                or height < MIN_MOUTH_PATCH_DIMENSION
                or width > MAX_MOUTH_PATCH_DIMENSION
                or height > MAX_MOUTH_PATCH_DIMENSION
            ):
                return None, len(payload), "invalid_patch_dimensions"
            patch = image.convert("RGB")
    except (OSError, UnidentifiedImageError, ValueError):
        return None, len(payload), "invalid_patch_image"
    return patch, len(payload), "ok"


def _pixel_box(
    roi: tuple[float, float, float, float], frame_size: tuple[int, int]
) -> tuple[int, int, int, int] | None:
    x, y, width, height = roi
    frame_width, frame_height = frame_size
    left = max(0, min(frame_width - 1, int(math.floor(x * frame_width))))
    top = max(0, min(frame_height - 1, int(math.floor(y * frame_height))))
    right = max(left + 1, min(frame_width, int(math.ceil((x + width) * frame_width))))
    bottom = max(top + 1, min(frame_height, int(math.ceil((y + height) * frame_height))))
    if right - left < MIN_MOUTH_PATCH_DIMENSION or bottom - top < MIN_MOUTH_PATCH_DIMENSION:
        return None
    return left, top, right, bottom


def _feather_mask(size: tuple[int, int]) -> Image.Image:
    width, height = size
    feather = max(2, min(12, int(round(min(width, height) * 0.08))))
    mask = Image.new("L", size, 0)
    draw = ImageDraw.Draw(mask)
    if width > feather * 2 and height > feather * 2:
        draw.rectangle(
            (feather, feather, width - feather - 1, height - feather - 1),
            fill=255,
        )
        return mask.filter(ImageFilter.GaussianBlur(radius=max(1.0, feather * 0.65)))
    return Image.new("L", size, 255)


def apply_mouth_detail_patch(
    frame: Image.Image,
    frame_meta: Mapping[str, Any],
    *,
    enabled: bool,
    max_gain: float,
) -> tuple[Image.Image, dict[str, Any]]:
    """Return the conditioned source frame and bounded diagnostic fields.

    Invalid, stale, neutral, disabled, or absent patches return the original
    ``frame`` object unchanged.  This makes the legacy inference path an exact
    no-op whenever mouth detail cannot be applied safely.
    """

    control = build_mouth_control([frame_meta], enabled=enabled, max_gain=max_gain)
    if not control.active or control.roi is None:
        return frame, _profile(control, applied=False, reason=control.reason)

    patch, byte_count, decode_reason = _decode_patch(frame_meta.get("mouth_patch"))
    if patch is None:
        return frame, _profile(
            control,
            applied=False,
            reason=decode_reason,
            byte_count=byte_count,
        )

    box = _pixel_box(control.roi, frame.size)
    if box is None:
        return frame, _profile(
            control,
            applied=False,
            reason="invalid_target_roi",
            size=patch.size,
            byte_count=byte_count,
        )

    left, top, right, bottom = box
    target_size = (right - left, bottom - top)
    if patch.size != target_size:
        patch = patch.resize(target_size, _LANCZOS)
    conditioned = frame.convert("RGB").copy()
    conditioned.paste(patch, (left, top), _feather_mask(target_size))
    return conditioned, _profile(
        control,
        applied=True,
        reason="applied_before_vae",
        size=target_size,
        byte_count=byte_count,
    )
