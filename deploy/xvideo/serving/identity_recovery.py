from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class IdentityRecoveryDecision:
    """Per-chunk decision for bounded identity-history recovery."""

    occlusion_risk: bool
    recovery_active: bool
    anchor_chunk_id: int | None
    release_after_chunk: bool
    risky_frames: int
    max_overlap: float
    event: str


class IdentityRecoveryController:
    """Keep risky face/hand chunks out of long-lived generated history.

    The controller is deliberately model-independent.  It consumes browser
    metadata, remembers the last clean generated chunk, and applies a short
    clean-frame hysteresis before allowing normal autoregressive history to
    resume.  It never changes model weights or source pixels.
    """

    def __init__(
        self,
        *,
        enabled: bool,
        clean_chunks_to_release: int = 2,
        min_risky_frames: int = 2,
        strong_overlap: float = 0.12,
    ) -> None:
        self.enabled = bool(enabled)
        self.clean_chunks_to_release = max(1, int(clean_chunks_to_release))
        self.min_risky_frames = max(1, int(min_risky_frames))
        self.strong_overlap = max(0.0, min(1.0, float(strong_overlap)))
        self.last_safe_chunk_id: int | None = None
        self.recovery_anchor_id: int | None = None
        self.recovery_active = False
        self.clean_chunks = 0

    @staticmethod
    def _overlap(meta: Mapping[str, Any]) -> float:
        try:
            value = float(meta.get("identity_occlusion_overlap", 0.0))
        except (TypeError, ValueError):
            return 0.0
        if not math.isfinite(value):
            return 0.0
        return max(0.0, min(1.0, value))

    def _risk_summary(
        self, metas: Sequence[Mapping[str, Any]]
    ) -> tuple[bool, int, float]:
        risky_frames = sum(
            1 for meta in metas if meta.get("identity_occlusion_risk") is True
        )
        max_overlap = max((self._overlap(meta) for meta in metas), default=0.0)
        required = 1 if len(metas) <= 1 else min(self.min_risky_frames, len(metas))
        risk = risky_frames >= required or max_overlap >= self.strong_overlap
        return risk, risky_frames, max_overlap

    def observe_chunk(
        self,
        chunk_idx: int,
        metas: Sequence[Mapping[str, Any]],
    ) -> IdentityRecoveryDecision:
        chunk_idx = int(chunk_idx)
        risk, risky_frames, max_overlap = self._risk_summary(metas)

        if not self.enabled:
            self.last_safe_chunk_id = chunk_idx
            return IdentityRecoveryDecision(
                risk,
                False,
                None,
                False,
                risky_frames,
                max_overlap,
                "disabled",
            )

        if risk:
            event = "occlusion_continues" if self.recovery_active else "occlusion_started"
            if not self.recovery_active:
                self.recovery_anchor_id = self.last_safe_chunk_id
            self.recovery_active = True
            self.clean_chunks = 0
            return IdentityRecoveryDecision(
                True,
                True,
                self.recovery_anchor_id,
                False,
                risky_frames,
                max_overlap,
                event,
            )

        if self.recovery_active:
            self.clean_chunks += 1
            release = self.clean_chunks >= self.clean_chunks_to_release
            anchor = self.recovery_anchor_id
            event = "recovery_released" if release else "clean_hysteresis"
            if release:
                self.recovery_active = False
                self.recovery_anchor_id = None
                self.clean_chunks = 0
                self.last_safe_chunk_id = chunk_idx
            return IdentityRecoveryDecision(
                False,
                True,
                anchor,
                release,
                risky_frames,
                max_overlap,
                event,
            )

        self.last_safe_chunk_id = chunk_idx
        return IdentityRecoveryDecision(
            False,
            False,
            None,
            False,
            risky_frames,
            max_overlap,
            "clean",
        )
