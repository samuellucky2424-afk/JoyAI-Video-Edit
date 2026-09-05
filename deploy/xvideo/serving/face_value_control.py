"""Bounded eye/mouth control for JoyAI edit-condition attention values.

The uploaded reference image remains the long-lived identity KV anchor.  This
module only builds a same-shape, per-token scale for the *source video*
conditioning values already consumed by the DiT.  It does not change model
weights, attention keys/logits, generated-history KV, or reference-image KV.

Disabled, neutral, stale, malformed, and unsupported inputs return ``None`` so
the legacy transformer path is an exact no-op.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, MutableMapping, Sequence

from xvideo.serving.mouth_control import (
    build_mouth_control, fresh_face_meta, normalize_mouth_roi,
)


# Attention values are more direct than latent amplitude (which is normalized
# inside every transformer block), so keep the live-test caps small.  Anatomy
# gets a slightly stronger cap only inside the lip interior; the surrounding
# lip shape and both eye regions retain the original conservative caps.
MOUTH_VALUE_MAX_GAIN = 1.125
MOUTH_INTERIOR_VALUE_MAX_GAIN = 1.16
EYE_VALUE_MAX_GAIN = 1.075
FACE_EVENT_ACTIVE_THRESHOLD = 0.15


@dataclass(frozen=True)
class _Region:
    name: str
    roi: tuple[float, float, float, float]
    gain: float
    latent_frames: tuple[int, ...]
    sample_index: int
    sample_weight: float


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _unit_score(value: Any) -> float:
    if isinstance(value, bool):
        return 0.0
    try:
        score = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(score):
        return 0.0
    return max(0.0, min(1.0, score))


def _temporal_meta_buckets(
    metas: Sequence[Mapping[str, Any]],
    *,
    latent_frames: int,
) -> list[list[Mapping[str, Any]]]:
    """Map ordered camera metadata onto the VAE's latent time slices.

    The live runtime retains ONE latent time slice per chunk. A scale cannot
    express sub-frame timing within that slice. Average per-source-frame maps
    inside each bucket instead of treating a brief event as a sustained one.
    Multi-slice inputs remain supported without assuming a fixed VAE ratio.
    """

    if latent_frames < 1:
        return []
    ordered = [_mapping(meta) for meta in metas]
    if not ordered:
        return [[] for _ in range(latent_frames)]
    buckets: list[list[Mapping[str, Any]]] = []
    sample_count = len(ordered)
    for latent_index in range(latent_frames):
        start = (latent_index * sample_count) // latent_frames
        end = ((latent_index + 1) * sample_count) // latent_frames
        if end <= start:
            start = min(start, sample_count - 1)
            end = start + 1
        buckets.append(ordered[start:end])
    return buckets


def _mouth_events(meta: Mapping[str, Any]) -> set[str]:
    events: set[str] = set()
    anatomy = _mapping(meta.get("mouth_anatomy"))
    evidence = _mapping(anatomy.get("region_evidence"))
    if _unit_score(evidence.get("teeth")) >= FACE_EVENT_ACTIVE_THRESHOLD:
        events.add("teeth")
    if _unit_score(evidence.get("tongue")) >= FACE_EVENT_ACTIVE_THRESHOLD:
        events.add("tongue")
    if _unit_score(evidence.get("oral_cavity")) >= FACE_EVENT_ACTIVE_THRESHOLD:
        events.add("oral_cavity")

    blend = _mapping(meta.get("mouth_blendshapes"))
    jaw_open = _unit_score(blend.get("jawOpen"))
    funnel = max(
        _unit_score(blend.get("mouthFunnel")),
        _unit_score(blend.get("mouthPucker")),
    )
    smile = 0.5 * (
        _unit_score(blend.get("mouthSmileLeft"))
        + _unit_score(blend.get("mouthSmileRight"))
    )
    if jaw_open >= FACE_EVENT_ACTIVE_THRESHOLD:
        events.add("open")
    if (
        jaw_open >= FACE_EVENT_ACTIVE_THRESHOLD
        and funnel >= FACE_EVENT_ACTIVE_THRESHOLD
    ):
        events.add("round")
    if smile >= FACE_EVENT_ACTIVE_THRESHOLD:
        events.add("smile")
    return events


_EYE_KEYS = {
    "left": (
        "eyeBlinkLeft",
        "eyeSquintLeft",
        "eyeWideLeft",
    ),
    "right": (
        "eyeBlinkRight",
        "eyeSquintRight",
        "eyeWideRight",
    ),
}


def _eye_strength(meta: Mapping[str, Any], side: str) -> float:
    blend = _mapping(meta.get("eye_blendshapes"))
    return max((_unit_score(blend.get(key)) for key in _EYE_KEYS[side]), default=0.0)


def _mouth_interior_roi(
    roi: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    """Return a conservative inner-mouth box for teeth/tongue/cavity values."""

    x, y, width, height = roi
    return (
        round(x + 0.14 * width, 6),
        round(y + 0.18 * height, 6),
        round(0.72 * width, 6),
        round(0.64 * height, 6),
    )


def _build_regions(
    metas: Sequence[Mapping[str, Any]],
    *,
    max_gain: float,
    latent_frames: int,
) -> tuple[list[_Region], list[str], list[str]]:
    regions: list[_Region] = []
    mouth_events: set[str] = set()
    eye_sides: set[str] = set()
    bounded_eye_gain = min(float(max_gain), EYE_VALUE_MAX_GAIN)
    for latent_index, bucket in enumerate(
        _temporal_meta_buckets(metas, latent_frames=latent_frames)
    ):
        # Include neutral, unavailable and malformed positions in the divisor.
        # Removing them would turn a single observation into a whole-chunk event.
        for sample_index, meta in enumerate(bucket):
            if not fresh_face_meta(meta):
                continue
            common = dict(
                latent_frames=(latent_index,),
                sample_index=sample_index,
                sample_weight=1.0 / len(bucket),
            )
            mouth = build_mouth_control([meta], enabled=True, max_gain=max_gain)
            if mouth.active and mouth.roi is not None:
                events = _mouth_events(meta)
                mouth_events.update(events)
                regions.append(_Region(
                    name="mouth", roi=mouth.roi,
                    gain=min(float(mouth.gain), MOUTH_VALUE_MAX_GAIN), **common,
                ))
                if events.intersection({"teeth", "tongue", "oral_cavity"}):
                    regions.append(_Region(
                        name="mouth_interior", roi=_mouth_interior_roi(mouth.roi),
                        gain=min(float(mouth.gain), MOUTH_INTERIOR_VALUE_MAX_GAIN),
                        **common,
                    ))
            for side in ("left", "right"):
                if meta.get("eye_landmark_available") is not True:
                    continue
                roi = normalize_mouth_roi(_mapping(meta.get("eye_rois")).get(side))
                strength = _eye_strength(meta, side)
                if roi is None or strength <= FACE_EVENT_ACTIVE_THRESHOLD:
                    continue
                normalized = (strength - FACE_EVENT_ACTIVE_THRESHOLD) / (
                    1.0 - FACE_EVENT_ACTIVE_THRESHOLD
                )
                regions.append(_Region(
                    name=f"eye_{side}", roi=roi,
                    gain=1.0 + (bounded_eye_gain - 1.0) * normalized, **common,
                ))
                eye_sides.add(side)
    return regions, sorted(mouth_events), sorted(eye_sides)


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
    right = max(left + 1, min(width, int(math.ceil((x + roi_width) * width))))
    bottom = max(top + 1, min(height, int(math.ceil((y + roi_height) * height))))
    if right <= left or bottom <= top:
        return None
    return left, top, right, bottom


def _profile(
    profile: MutableMapping[str, Any] | None,
    *,
    applied: bool,
    reason: str,
    mouth_events: Sequence[str] = (),
    eye_sides: Sequence[str] = (),
    regions: Sequence[_Region] = (),
    changed_fraction: float = 0.0,
) -> None:
    if profile is None:
        return
    profile.update(
        {
            "face_value_control_active": int(bool(regions)),
            "face_value_control_applied": int(applied),
            "face_value_control_reason": reason,
            "face_value_control_mouth_events": list(mouth_events),
            "face_value_control_eye_sides": list(eye_sides),
            "face_value_control_regions": [
                {
                    "name": region.name,
                    "roi": list(region.roi),
                    "gain": round(float(region.gain), 6),
                    "latent_frames": list(region.latent_frames),
                    "sample_index": region.sample_index,
                    "sample_weight": region.sample_weight,
                }
                for region in regions
            ],
            "face_value_control_changed_fraction": round(
                float(changed_fraction), 8
            ),
        }
    )


def build_face_value_scale(
    ref_video_latent: Any,
    metas: Iterable[Mapping[str, Any]],
    *,
    enabled: bool,
    max_gain: float,
    patch_size: Sequence[int],
    profile: MutableMapping[str, Any] | None = None,
) -> Any | None:
    """Return a flattened edit-condition value scale or ``None`` for no-op."""

    if not enabled:
        _profile(profile, applied=False, reason="disabled")
        return None
    try:
        bounded_gain = float(max_gain)
    except (TypeError, ValueError):
        _profile(profile, applied=False, reason="invalid_gain")
        return None
    if not math.isfinite(bounded_gain) or bounded_gain <= 1.0:
        _profile(profile, applied=False, reason="invalid_or_unit_gain")
        return None

    try:
        import torch
        import torch.nn.functional as functional
    except ModuleNotFoundError:
        _profile(profile, applied=False, reason="torch_unavailable")
        return None

    if (
        not isinstance(ref_video_latent, torch.Tensor)
        or ref_video_latent.ndim != 5
        or not ref_video_latent.is_floating_point()
    ):
        _profile(profile, applied=False, reason="unsupported_tensor")
        return None
    if len(tuple(patch_size)) != 3:
        _profile(profile, applied=False, reason="invalid_patch_size")
        return None
    try:
        pt, ph, pw = (int(value) for value in patch_size)
    except (TypeError, ValueError):
        _profile(profile, applied=False, reason="invalid_patch_size")
        return None
    batch, _, frames, height, width = ref_video_latent.shape
    if (
        min(pt, ph, pw) < 1
        or frames % pt
        or height % ph
        or width % pw
    ):
        _profile(profile, applied=False, reason="incompatible_patch_size")
        return None

    ordered_metas = [_mapping(meta) for meta in metas]
    regions, mouth_events, eye_sides = _build_regions(
        ordered_metas,
        max_gain=bounded_gain,
        latent_frames=frames,
    )
    if not regions:
        _profile(
            profile,
            applied=False,
            reason="no_significant_face_region",
            mouth_events=mouth_events,
            eye_sides=eye_sides,
        )
        return None

    # Rasterize the small ROI maps on CPU, then transfer once. Launching many
    # tiny GPU operations for every camera sample would stall the live pipeline.
    sample_maps: dict[tuple[int, int], dict[int, float]] = {}
    weights: dict[tuple[int, int], float] = {}
    applied_regions: list[_Region] = []
    for region in regions:
        box = _latent_box(region.roi, height=height, width=width)
        if box is None:
            continue
        left, top, right, bottom = box
        roi_height, roi_width = bottom - top, right - left
        y_width = max(1.0, min(2.0, max(1, roi_height - 1) / 2.0))
        x_width = max(1.0, min(2.0, max(1, roi_width - 1) / 2.0))
        for latent_index in region.latent_frames:
            key = (latent_index, region.sample_index)
            sample_map = sample_maps.setdefault(key, {})
            weights[key] = region.sample_weight
            for row in range(top, bottom):
                fy = 0.35 + 0.65 * min(1.0, min(row-top, bottom-1-row) / y_width)
                for col in range(left, right):
                    fx = 0.35 + 0.65 * min(1.0, min(col-left, right-1-col) / x_width)
                    cell = (latent_index * height + row) * width + col
                    delta = (region.gain - 1.0) * fy * fx
                    # Mouth + interior overlap within one frame uses max, not
                    # addition. Different source frames contribute their mean.
                    sample_map[cell] = max(sample_map.get(cell, 0.0), delta)
        applied_regions.append(region)
    values = [1.0] * (frames * height * width)
    changed_cells: set[int] = set()
    for key, sample_map in sample_maps.items():
        for cell, delta in sample_map.items():
            values[cell] += delta * weights[key]
            changed_cells.add(cell)
    scale_map = torch.tensor(values, device=ref_video_latent.device, dtype=torch.float32)
    scale_map = scale_map.view(1, 1, frames, height, width).expand(batch, -1, -1, -1, -1)
    if profile is not None:
        profile["face_value_control_temporal_mode"] = "mean_source_frame_maps"
        profile["face_value_control_latent_frames"] = frames
        profile["face_value_control_source_frames"] = len(ordered_metas)
        profile["face_value_control_max_gain"] = max(values)

    if not applied_regions:
        _profile(
            profile,
            applied=False,
            reason="invalid_face_regions",
            mouth_events=mouth_events,
            eye_sides=eye_sides,
        )
        return None

    if (pt, ph, pw) != (1, 1, 1):
        scale_map = functional.avg_pool3d(
            scale_map,
            kernel_size=(pt, ph, pw),
            stride=(pt, ph, pw),
        )
    token_scale = scale_map.flatten(2).squeeze(1).to(dtype=ref_video_latent.dtype)
    _profile(
        profile,
        applied=True,
        reason="applied_to_edit_condition_values",
        mouth_events=mouth_events,
        eye_sides=eye_sides,
        regions=applied_regions,
        changed_fraction=len(changed_cells) / float(frames * height * width),
    )
    return token_scale
