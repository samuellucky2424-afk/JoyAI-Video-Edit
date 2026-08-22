"""Safe runtime control derived from validated browser mouth metadata.

The control never creates anatomy pixels and never changes model weights.  It
only describes where a meaningful mouth event exists so the streaming runtime
can give the already-encoded source-video mouth condition a bounded attention
value boost.  Missing, stale, neutral, or malformed metadata is an exact no-op.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

import numpy as np

from xvideo.serving.mouth_anatomy import (
    MouthAnatomyContractError,
    normalize_mouth_anatomy,
)


MOUTH_CONTROL_MIN_GAIN = 1.0
MOUTH_CONTROL_MAX_GAIN = 1.5
MOUTH_CONTROL_ACTIVE_THRESHOLD = 0.15


@dataclass(frozen=True)
class MouthControl:
    active: bool
    roi: tuple[float, float, float, float] | None = None
    gain: float = 1.0
    strength: float = 0.0
    confidence: float = 0.0
    sample_count: int = 0
    reason: str = "inactive"

    def profile_fields(self) -> dict[str, Any]:
        return {
            "mouth_control_active": int(self.active),
            "mouth_control_gain": round(float(self.gain), 6),
            "mouth_control_strength": round(float(self.strength), 6),
            "mouth_control_confidence": round(float(self.confidence), 6),
            "mouth_control_samples": int(self.sample_count),
            "mouth_control_roi": list(self.roi) if self.roi is not None else None,
            "mouth_control_reason": self.reason,
        }


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


def normalize_mouth_roi(value: Any) -> tuple[float, float, float, float] | None:
    if not isinstance(value, Mapping):
        return None
    raw_values = tuple(value.get(field) for field in ("x", "y", "width", "height"))
    if any(isinstance(item, bool) for item in raw_values):
        return None
    try:
        x, y, width, height = (float(item) for item in raw_values)
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(item) for item in (x, y, width, height)):
        return None
    if width <= 0.0 or height <= 0.0:
        return None
    left = max(0.0, min(1.0, x))
    top = max(0.0, min(1.0, y))
    right = max(left, min(1.0, x + width))
    bottom = max(top, min(1.0, y + height))
    if right - left < 1e-4 or bottom - top < 1e-4:
        return None
    return left, top, right - left, bottom - top


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _meta_strength(meta: Mapping[str, Any]) -> tuple[float, float]:
    strength = 1.0 if bool(meta.get("mouth_event_significant")) else 0.0
    confidence = 0.75 if strength > 0.0 else 0.0

    anatomy_payload = meta.get("mouth_anatomy")
    try:
        anatomy = normalize_mouth_anatomy(anatomy_payload)
    except (MouthAnatomyContractError, TypeError):
        anatomy = None
    if anatomy is not None and anatomy["available"]:
        evidence = anatomy["region_evidence"]
        # Lips alone are always present and must not activate the control.  The
        # other regions correspond to the details JoyAI currently misses.
        strength = max(
            strength,
            float(anatomy["appearance_motion"]),
            float(evidence["teeth"]),
            float(evidence["tongue"]),
            float(evidence["oral_cavity"]),
            1.0 if anatomy["significant"] else 0.0,
        )
        confidence = max(confidence, float(anatomy["roi_confidence"]))

    blendshapes = _mapping(meta.get("mouth_blendshapes"))
    smile = 0.5 * (
        _unit_score(blendshapes.get("mouthSmileLeft"))
        + _unit_score(blendshapes.get("mouthSmileRight"))
    )
    strength = max(
        strength,
        _unit_score(blendshapes.get("jawOpen")),
        _unit_score(blendshapes.get("mouthFunnel")),
        _unit_score(blendshapes.get("mouthPucker")),
        smile,
    )

    geometry = _mapping(meta.get("mouth_geometry"))
    strength = max(strength, min(1.0, _unit_score(geometry.get("motion")) * 4.0))
    return strength, confidence


def build_mouth_control(
    metas: Iterable[Mapping[str, Any]],
    *,
    enabled: bool,
    max_gain: float,
) -> MouthControl:
    if not enabled:
        return MouthControl(active=False, reason="disabled")
    try:
        bounded_gain = float(max_gain)
    except (TypeError, ValueError):
        return MouthControl(active=False, reason="invalid_gain")
    if not math.isfinite(bounded_gain):
        return MouthControl(active=False, reason="invalid_gain")
    bounded_gain = max(MOUTH_CONTROL_MIN_GAIN, min(MOUTH_CONTROL_MAX_GAIN, bounded_gain))
    if bounded_gain <= 1.0:
        return MouthControl(active=False, reason="unit_gain")

    # The browser tracker runs more slowly than frame delivery, so the same
    # landmark sample can be attached to several source frames.  Keep one copy
    # of each sample before taking a per-VAE-chunk ROI union.
    unique: dict[tuple[str, Any], Mapping[str, Any]] = {}
    for index, raw_meta in enumerate(metas):
        if not isinstance(raw_meta, Mapping):
            continue
        if not bool(raw_meta.get("mouth_landmark_available")):
            continue
        sequence = raw_meta.get("mouth_landmark_seq")
        try:
            hash(sequence)
        except TypeError:
            sequence = None
        key = ("seq", sequence) if sequence is not None else ("index", index)
        unique[key] = raw_meta

    candidates: list[tuple[tuple[float, float, float, float], float, float]] = []
    for meta in unique.values():
        roi = normalize_mouth_roi(meta.get("mouth_roi"))
        if roi is None:
            continue
        strength, confidence = _meta_strength(meta)
        if strength < MOUTH_CONTROL_ACTIVE_THRESHOLD:
            continue
        candidates.append((roi, strength, confidence))

    if not candidates:
        return MouthControl(active=False, reason="no_significant_fresh_roi")

    left = min(item[0][0] for item in candidates)
    top = min(item[0][1] for item in candidates)
    right = max(item[0][0] + item[0][2] for item in candidates)
    bottom = max(item[0][1] + item[0][3] for item in candidates)
    strength = max(item[1] for item in candidates)
    confidence = max(item[2] for item in candidates)
    if confidence <= 0.0:
        confidence = 0.75
    effective_strength = max(0.0, min(1.0, strength * confidence))
    normalized_strength = max(
        0.0,
        min(
            1.0,
            (effective_strength - MOUTH_CONTROL_ACTIVE_THRESHOLD)
            / (1.0 - MOUTH_CONTROL_ACTIVE_THRESHOLD),
        ),
    )
    gain = 1.0 + (bounded_gain - 1.0) * normalized_strength
    if gain <= 1.0:
        return MouthControl(active=False, reason="below_gain_threshold")
    return MouthControl(
        active=True,
        roi=tuple(round(item, 6) for item in (left, top, right - left, bottom - top)),
        gain=gain,
        strength=strength,
        confidence=confidence,
        sample_count=len(candidates),
        reason="active",
    )


def mouth_control_token_scale(
    control: MouthControl,
    *,
    temporal_tokens: int,
    height_tokens: int,
    width_tokens: int,
) -> np.ndarray:
    shape = (1, temporal_tokens * height_tokens * width_tokens)
    if (
        not control.active
        or control.roi is None
        or temporal_tokens <= 0
        or height_tokens <= 0
        or width_tokens <= 0
    ):
        return np.ones(shape, dtype=np.float32)

    x, y, width, height = control.roi
    center_x = x + width * 0.5
    center_y = y + height * 0.5
    half_width = width * 0.5
    half_height = height * 0.5
    feather_x = max(1.0 / width_tokens, width * 0.35)
    feather_y = max(1.0 / height_tokens, height * 0.35)
    grid_x = (np.arange(width_tokens, dtype=np.float32) + 0.5) / width_tokens
    grid_y = (np.arange(height_tokens, dtype=np.float32) + 0.5) / height_tokens
    mask_x = np.clip(
        (half_width + feather_x - np.abs(grid_x - center_x)) / feather_x,
        0.0,
        1.0,
    )
    mask_y = np.clip(
        (half_height + feather_y - np.abs(grid_y - center_y)) / feather_y,
        0.0,
        1.0,
    )
    spatial_mask = mask_y[:, None] * mask_x[None, :]
    spatial_scale = 1.0 + (float(control.gain) - 1.0) * spatial_mask
    scale = np.broadcast_to(
        spatial_scale,
        (temporal_tokens, height_tokens, width_tokens),
    ).reshape(1, -1)
    return np.ascontiguousarray(scale, dtype=np.float32)
