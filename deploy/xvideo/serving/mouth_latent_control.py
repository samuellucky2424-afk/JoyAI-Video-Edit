"""Bounded mouth-ROI emphasis for JoyAI's existing source conditioning latent.

This module does not change, replace, or train any model weight.  It operates on
the already encoded source ``ref_video_latent`` immediately before DiT receives
it.  A validated, significant MediaPipe mouth event selects a small spatial ROI;
the signal in that ROI is then emphasized relative to each latent channel's
spatial baseline.  Shape, dtype, device, and every value outside the ROI remain
unchanged.

Disabled, neutral, malformed, non-finite, or unsupported inputs are exact no-ops
and return the original tensor object.
"""

from __future__ import annotations

import math
from typing import Any, Iterable, Mapping, MutableMapping

from xvideo.serving.mouth_control import build_mouth_control


# The browser gain may reach 1.5 for the source JPEG preservation stage.  A
# latent is much more sensitive than RGB pixels, so independently cap the
# effective tensor gain at 1.125.  This is deliberately conservative for the
# first live experiment and protects reference identity around the mouth.
MOUTH_LATENT_MAX_GAIN = 1.125


def _profile(
    profile: MutableMapping[str, Any] | None,
    *,
    active: bool,
    applied: bool,
    reason: str,
    requested_gain: float = 1.0,
    effective_gain: float = 1.0,
    roi: tuple[float, float, float, float] | None = None,
    box: tuple[int, int, int, int] | None = None,
    changed_fraction: float = 0.0,
    delta_rms: float = 0.0,
) -> None:
    if profile is None:
        return
    profile.update(
        {
            "mouth_latent_control_active": int(active),
            "mouth_latent_control_applied": int(applied),
            "mouth_latent_control_reason": reason,
            "mouth_latent_control_requested_gain": round(
                float(requested_gain), 6
            ),
            "mouth_latent_control_gain": round(float(effective_gain), 6),
            "mouth_latent_control_roi": list(roi) if roi is not None else None,
            "mouth_latent_control_box": list(box) if box is not None else None,
            "mouth_latent_control_changed_fraction": round(
                float(changed_fraction), 8
            ),
            "mouth_latent_control_delta_rms": round(float(delta_rms), 8),
        }
    )


def _latent_box(
    roi: tuple[float, float, float, float],
    *,
    height: int,
    width: int,
) -> tuple[int, int, int, int] | None:
    if height < 1 or width < 1:
        return None
    x, y, roi_width, roi_height = roi
    left = max(0, min(width - 1, int(math.floor(x * width))))
    top = max(0, min(height - 1, int(math.floor(y * height))))
    right = max(
        left + 1,
        min(width, int(math.ceil((x + roi_width) * width))),
    )
    bottom = max(
        top + 1,
        min(height, int(math.ceil((y + roi_height) * height))),
    )
    if right <= left or bottom <= top:
        return None
    return left, top, right, bottom


def _inside_feather(
    height: int,
    width: int,
    *,
    device: Any,
) -> Any:
    """Return a feather that is one for tiny ROIs and fades only inside edges."""

    import torch

    def axis(size: int) -> Any:
        if size <= 2:
            return torch.ones(size, device=device, dtype=torch.float32)
        positions = torch.arange(size, device=device, dtype=torch.float32)
        edge_distance = torch.minimum(positions, (size - 1) - positions)
        feather_cells = max(1.0, min(2.0, float(size - 1) / 2.0))
        # Keep a non-zero edge contribution because the mouth ROI is commonly
        # only 2-4 cells high at the validated 840x480 latent resolution.
        return 0.35 + 0.65 * torch.clamp(edge_distance / feather_cells, 0.0, 1.0)

    return axis(height).unsqueeze(1) * axis(width).unsqueeze(0)


def apply_mouth_latent_control(
    ref_video_latent: Any,
    metas: Iterable[Mapping[str, Any]],
    *,
    enabled: bool,
    max_gain: float,
    profile: MutableMapping[str, Any] | None = None,
) -> Any:
    """Emphasize a validated mouth ROI in the existing DiT condition tensor.

    The returned tensor has exactly the same shape, dtype, and device.  Exact
    no-op cases return ``ref_video_latent`` itself rather than a clone.
    """

    import torch

    with torch.no_grad():
        return _apply_mouth_latent_control_no_grad(
            ref_video_latent,
            metas,
            enabled=enabled,
            max_gain=max_gain,
            profile=profile,
        )


def _apply_mouth_latent_control_no_grad(
    ref_video_latent: Any,
    metas: Iterable[Mapping[str, Any]],
    *,
    enabled: bool,
    max_gain: float,
    profile: MutableMapping[str, Any] | None,
) -> Any:
    import torch

    control = build_mouth_control(metas, enabled=enabled, max_gain=max_gain)
    if not control.active or control.roi is None:
        _profile(
            profile,
            active=False,
            applied=False,
            reason=control.reason,
            requested_gain=control.gain,
        )
        return ref_video_latent

    if not isinstance(ref_video_latent, torch.Tensor):
        _profile(
            profile,
            active=True,
            applied=False,
            reason="not_a_tensor",
            requested_gain=control.gain,
            roi=control.roi,
        )
        return ref_video_latent
    if ref_video_latent.ndim != 5 or not ref_video_latent.is_floating_point():
        _profile(
            profile,
            active=True,
            applied=False,
            reason="unsupported_tensor",
            requested_gain=control.gain,
            roi=control.roi,
        )
        return ref_video_latent
    # Avoid a CUDA host synchronization in the streaming encode worker.  The
    # CPU branch remains fully defensive for tests and non-CUDA callers; CUDA
    # VAE output finiteness is already a prerequisite of the existing path.
    if (
        ref_video_latent.device.type == "cpu"
        and not bool(torch.isfinite(ref_video_latent).all().item())
    ):
        _profile(
            profile,
            active=True,
            applied=False,
            reason="non_finite_tensor",
            requested_gain=control.gain,
            roi=control.roi,
        )
        return ref_video_latent

    height, width = int(ref_video_latent.shape[-2]), int(ref_video_latent.shape[-1])
    box = _latent_box(control.roi, height=height, width=width)
    if box is None:
        _profile(
            profile,
            active=True,
            applied=False,
            reason="invalid_latent_roi",
            requested_gain=control.gain,
            roi=control.roi,
        )
        return ref_video_latent

    effective_gain = min(float(control.gain), MOUTH_LATENT_MAX_GAIN)
    if effective_gain <= 1.0:
        _profile(
            profile,
            active=True,
            applied=False,
            reason="unit_latent_gain",
            requested_gain=control.gain,
            effective_gain=effective_gain,
            roi=control.roi,
            box=box,
        )
        return ref_video_latent

    left, top, right, bottom = box
    roi_height = bottom - top
    roi_width = right - left
    feather = _inside_feather(
        roi_height,
        roi_width,
        device=ref_video_latent.device,
    ).view(1, 1, 1, roi_height, roi_width)

    # Work in float32 for predictable bounded arithmetic, then restore the
    # original precision.  The reduction casts in-kernel and only the small ROI
    # is materialized as float32; converting the full latent would add avoidable
    # VRAM traffic to every streaming chunk.  The channel baseline is used only
    # as a reference and every value outside the selected ROI stays untouched.
    spatial_center = ref_video_latent.mean(
        dim=(-2, -1),
        keepdim=True,
        dtype=torch.float32,
    )
    roi_source = ref_video_latent[..., top:bottom, left:right].to(torch.float32)
    roi_delta = roi_source - spatial_center
    roi_adjustment = (effective_gain - 1.0) * feather * roi_delta

    controlled = ref_video_latent.clone()
    controlled[..., top:bottom, left:right] = (
        roi_source + roi_adjustment
    ).to(dtype=ref_video_latent.dtype)

    changed_fraction = (roi_height * roi_width) / float(height * width)
    delta_rms = 0.0
    if ref_video_latent.device.type == "cpu":
        delta_rms = float(
            torch.sqrt(torch.mean(roi_adjustment.square())).item()
        )
    _profile(
        profile,
        active=True,
        applied=True,
        reason="applied_to_ref_video_latent",
        requested_gain=control.gain,
        effective_gain=effective_gain,
        roi=control.roi,
        box=box,
        changed_fraction=changed_fraction,
        delta_rms=delta_rms,
    )
    return controlled
