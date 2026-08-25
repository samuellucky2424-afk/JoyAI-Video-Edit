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

from xvideo.serving.mouth_control import build_mouth_control, normalize_mouth_roi


# Attention values are more direct than latent amplitude (which is normalized
# inside every transformer block), so keep the first live-test caps small.
MOUTH_VALUE_MAX_GAIN = 1.125
EYE_VALUE_MAX_GAIN = 1.075
FACE_EVENT_ACTIVE_THRESHOLD = 0.15


@dataclass(frozen=True)
class _Region:
    name: str
    roi: tuple[float, float, float, float]
    gain: float


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


def _unique_fresh_metas(
    metas: Iterable[Mapping[str, Any]],
) -> list[Mapping[str, Any]]:
    unique: dict[tuple[str, Any], Mapping[str, Any]] = {}
    for index, meta in enumerate(metas):
        if not isinstance(meta, Mapping):
            continue
        sequence = meta.get("mouth_landmark_seq")
        try:
            hash(sequence)
        except TypeError:
            sequence = None
        key = ("seq", sequence) if sequence is not None else ("index", index)
        unique[key] = meta
    return list(unique.values())


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


def _union_roi(
    values: Sequence[tuple[float, float, float, float]],
) -> tuple[float, float, float, float]:
    left = min(roi[0] for roi in values)
    top = min(roi[1] for roi in values)
    right = max(roi[0] + roi[2] for roi in values)
    bottom = max(roi[1] + roi[3] for roi in values)
    return (
        round(left, 6),
        round(top, 6),
        round(right - left, 6),
        round(bottom - top, 6),
    )


def _build_regions(
    metas: list[Mapping[str, Any]],
    *,
    max_gain: float,
) -> tuple[list[_Region], list[str], list[str]]:
    regions: list[_Region] = []
    mouth_events: set[str] = set()
    mouth = build_mouth_control(metas, enabled=True, max_gain=max_gain)
    if mouth.active and mouth.roi is not None:
        for meta in metas:
            if bool(meta.get("mouth_landmark_available")):
                mouth_events.update(_mouth_events(meta))
        regions.append(
            _Region(
                name="mouth",
                roi=mouth.roi,
                gain=min(float(mouth.gain), MOUTH_VALUE_MAX_GAIN),
            )
        )

    eye_sides: list[str] = []
    bounded_eye_gain = min(float(max_gain), EYE_VALUE_MAX_GAIN)
    for side in ("left", "right"):
        candidates: list[tuple[tuple[float, float, float, float], float]] = []
        for meta in metas:
            if not bool(meta.get("eye_landmark_available")):
                continue
            roi = normalize_mouth_roi(_mapping(meta.get("eye_rois")).get(side))
            strength = _eye_strength(meta, side)
            if roi is None or strength < FACE_EVENT_ACTIVE_THRESHOLD:
                continue
            candidates.append((roi, strength))
        if not candidates:
            continue
        strength = max(item[1] for item in candidates)
        normalized = max(
            0.0,
            min(
                1.0,
                (strength - FACE_EVENT_ACTIVE_THRESHOLD)
                / (1.0 - FACE_EVENT_ACTIVE_THRESHOLD),
            ),
        )
        gain = 1.0 + (bounded_eye_gain - 1.0) * normalized
        if gain <= 1.0:
            continue
        regions.append(
            _Region(
                name=f"eye_{side}",
                roi=_union_roi([item[0] for item in candidates]),
                gain=gain,
            )
        )
        eye_sides.append(side)
    return regions, sorted(mouth_events), eye_sides


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

    unique = _unique_fresh_metas(metas)
    regions, mouth_events, eye_sides = _build_regions(
        unique,
        max_gain=bounded_gain,
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

    scale_map = torch.ones(
        (batch, 1, frames, height, width),
        device=ref_video_latent.device,
        dtype=torch.float32,
    )
    changed_cells: set[tuple[int, int]] = set()
    applied_regions: list[_Region] = []
    for region in regions:
        box = _latent_box(region.roi, height=height, width=width)
        if box is None:
            continue
        left, top, right, bottom = box
        roi_height = bottom - top
        roi_width = right - left
        y = torch.arange(roi_height, device=scale_map.device, dtype=torch.float32)
        x = torch.arange(roi_width, device=scale_map.device, dtype=torch.float32)
        y_distance = torch.minimum(y, (roi_height - 1) - y)
        x_distance = torch.minimum(x, (roi_width - 1) - x)
        y_width = max(1.0, min(2.0, float(max(1, roi_height - 1)) / 2.0))
        x_width = max(1.0, min(2.0, float(max(1, roi_width - 1)) / 2.0))
        y_feather = 0.35 + 0.65 * torch.clamp(y_distance / y_width, 0.0, 1.0)
        x_feather = 0.35 + 0.65 * torch.clamp(x_distance / x_width, 0.0, 1.0)
        feather = (y_feather.unsqueeze(1) * x_feather.unsqueeze(0)).view(
            1, 1, 1, roi_height, roi_width
        )
        candidate = 1.0 + (float(region.gain) - 1.0) * feather
        target = scale_map[..., top:bottom, left:right]
        scale_map[..., top:bottom, left:right] = torch.maximum(target, candidate)
        changed_cells.update(
            (row, column)
            for row in range(top, bottom)
            for column in range(left, right)
        )
        applied_regions.append(region)

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
        changed_fraction=len(changed_cells) / float(height * width),
    )
    return token_scale
