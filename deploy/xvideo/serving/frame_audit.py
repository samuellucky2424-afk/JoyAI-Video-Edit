from __future__ import annotations

import threading
from collections import Counter, deque
from typing import Any, Iterable


def _int_or_none(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


class FrameAudit:
    """Bounded end-to-end accounting for frames selected by the browser.

    Sequence values are more useful than totals alone: a stage can process the
    expected number of frames while still skipping a short smile or mouth shape.
    The audit therefore records missing sequence positions and non-monotonic
    values at every stage without retaining the video frames themselves.
    """

    CLIENT_FIELDS = (
        "capture_seq",
        "camera_frame_seq",
        "client_skip_total",
        "client_uplink_drop_total",
        "client_drain_factor",
    )

    def __init__(self, recent_limit: int = 32) -> None:
        self._lock = threading.Lock()
        self._stages: dict[str, dict[str, int | None]] = {}
        self._drops: Counter[str] = Counter()
        self._recent_drops: deque[dict[str, Any]] = deque(maxlen=max(1, recent_limit))
        self._latest_client: dict[str, Any] = {}

    @staticmethod
    def _new_stage() -> dict[str, int | None]:
        return {
            "count": 0,
            "first_seq": None,
            "last_seq": None,
            "gap_frames": 0,
            "non_monotonic": 0,
        }

    def _observe_sequences(self, stage: str, sequences: Iterable[int | None]) -> None:
        state = self._stages.setdefault(stage, self._new_stage())
        for seq in sequences:
            if seq is None:
                continue
            state["count"] = int(state["count"] or 0) + 1
            if state["first_seq"] is None:
                state["first_seq"] = seq
            last = state["last_seq"]
            if last is not None:
                delta = seq - int(last)
                if delta > 1:
                    state["gap_frames"] = int(state["gap_frames"] or 0) + delta - 1
                elif delta <= 0:
                    state["non_monotonic"] = int(state["non_monotonic"] or 0) + 1
            state["last_seq"] = seq

    def observe(
        self,
        stage: str,
        metas: Iterable[dict[str, Any]],
        *,
        valid_count: int | None = None,
    ) -> None:
        selected = list(metas)
        if valid_count is not None:
            selected = selected[: max(0, int(valid_count))]
        with self._lock:
            self._observe_sequences(stage, (_int_or_none(meta.get("seq")) for meta in selected))
            if stage == "wire":
                self._observe_sequences(
                    "browser_capture",
                    (_int_or_none(meta.get("capture_seq")) for meta in selected),
                )
                for meta in selected:
                    for field in self.CLIENT_FIELDS:
                        if meta.get(field) is not None:
                            self._latest_client[field] = meta[field]

    def drop(self, reason: str, meta: dict[str, Any] | None = None) -> None:
        reason = str(reason or "unknown")
        meta = meta or {}
        with self._lock:
            self._drops[reason] += 1
            self._recent_drops.append(
                {
                    "reason": reason,
                    "seq": _int_or_none(meta.get("seq")),
                    "capture_seq": _int_or_none(meta.get("capture_seq")),
                }
            )

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            stages = {name: dict(state) for name, state in self._stages.items()}
            drops = dict(self._drops)
            recent_drops = list(self._recent_drops)
            latest_client = dict(self._latest_client)
        wire = int((stages.get("wire") or {}).get("count") or 0)
        decoded = int((stages.get("decoded") or {}).get("count") or 0)
        admitted = int((stages.get("admitted") or {}).get("count") or 0)
        inferred = int((stages.get("inference") or {}).get("count") or 0)

        def ratio(numerator: int, denominator: int) -> float | None:
            return round(numerator / denominator, 6) if denominator else None

        return {
            "stages": stages,
            "drops": drops,
            "recent_drops": recent_drops,
            "latest_client": latest_client,
            "coverage": {
                "wire_to_decoded": ratio(decoded, wire),
                "wire_to_inference": ratio(inferred, wire),
                "admitted_to_inference": ratio(inferred, admitted),
            },
        }
