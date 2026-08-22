from __future__ import annotations

import threading
from collections import Counter, deque
from typing import Any, Iterable


def _int_or_none(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _compact_ranges(values: Iterable[int]) -> list[list[int]]:
    """Return sorted integer values as inclusive [start, end] ranges."""
    ordered = sorted(set(values))
    if not ordered:
        return []
    ranges: list[list[int]] = []
    start = previous = ordered[0]
    for value in ordered[1:]:
        if value == previous + 1:
            previous = value
            continue
        ranges.append([start, previous])
        start = previous = value
    ranges.append([start, previous])
    return ranges


class FrameAudit:
    """Bounded end-to-end accounting for frames selected by the browser.

    Sequence values are more useful than totals alone: a stage can process the
    expected number of frames while still skipping a short smile or mouth shape.
    The audit therefore records missing sequence positions and non-monotonic
    values at every stage without retaining the video frames themselves.

    ``sequence_window`` keeps exact recent sequence identifiers so the report can
    state which received frames have not reached inference. This is diagnostic
    only; it does not alter queueing, scheduling, or inference behavior.
    """

    CLIENT_FIELDS = (
        "capture_seq",
        "camera_frame_seq",
        "client_skip_total",
        "client_uplink_drop_total",
        "client_drain_factor",
        "mouth_landmark_seq",
        "mouth_landmark_age_ms",
        "mouth_landmark_available",
        "mouth_roi",
        "mouth_geometry",
        "mouth_blendshapes",
        "mouth_anatomy",
        "mouth_event_significant",
        "mouth_event_camera_frame_seq",
        "mouth_event_preserved",
        "mouth_event_preserved_total",
        "mouth_tracker_processed_total",
        "mouth_tracker_drop_total",
    )

    MOUTH_FIELDS = (
        "camera_frame_seq",
        "t_capture_ms",
        "face_present",
        "delegate",
        "inference_ms",
        "roi",
        "geometry",
        "blendshapes",
        "anatomy",
        "significant",
        "processed_total",
        "drop_total",
    )

    def __init__(self, recent_limit: int = 32, sequence_window: int = 512) -> None:
        self._lock = threading.Lock()
        self._stages: dict[str, dict[str, int | None]] = {}
        self._drops: Counter[str] = Counter()
        self._recent_drops: deque[dict[str, Any]] = deque(maxlen=max(1, recent_limit))
        self._latest_client: dict[str, Any] = {}
        self._sequence_window = max(1, int(sequence_window))
        self._recent_sequences: dict[str, deque[int]] = {}

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
        recent = self._recent_sequences.setdefault(
            stage, deque(maxlen=self._sequence_window)
        )
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
            recent.append(seq)

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
                self._observe_sequences(
                    "camera_wire",
                    (_int_or_none(meta.get("camera_frame_seq")) for meta in selected),
                )
                self._observe_sequences(
                    "mouth_attached_wire",
                    (_int_or_none(meta.get("mouth_landmark_seq")) for meta in selected),
                )
                for meta in selected:
                    for field in self.CLIENT_FIELDS:
                        if meta.get(field) is not None:
                            self._latest_client[field] = meta[field]
            elif stage == "inference":
                self._observe_sequences(
                    "camera_inference",
                    (_int_or_none(meta.get("camera_frame_seq")) for meta in selected),
                )
            elif stage == "output":
                self._observe_sequences(
                    "camera_output",
                    (_int_or_none(meta.get("camera_frame_seq")) for meta in selected),
                )

    def observe_mouth(self, meta: dict[str, Any]) -> None:
        """Record browser MediaPipe telemetry without retaining image pixels."""
        sequence = _int_or_none(meta.get("camera_frame_seq"))
        with self._lock:
            self._observe_sequences("mouth_landmark", [sequence])
            if bool(meta.get("significant")):
                self._observe_sequences("mouth_event", [sequence])
            for field in self.MOUTH_FIELDS:
                if meta.get(field) is not None:
                    self._latest_client[f"mouth_{field}"] = meta[field]

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

    @staticmethod
    def _stage_count(stages: dict[str, dict[str, int | None]], stage: str) -> int:
        return int((stages.get(stage) or {}).get("count") or 0)

    @staticmethod
    def _missing_ranges(
        recent_sequences: dict[str, list[int]], source: str, target: str
    ) -> tuple[int, list[list[int]]]:
        missing = set(recent_sequences.get(source, ())) - set(
            recent_sequences.get(target, ())
        )
        return len(missing), _compact_ranges(missing)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            stages = {name: dict(state) for name, state in self._stages.items()}
            drops = dict(self._drops)
            recent_drops = list(self._recent_drops)
            latest_client = dict(self._latest_client)
            recent_sequences = {
                name: list(sequences)
                for name, sequences in self._recent_sequences.items()
            }
        wire = self._stage_count(stages, "wire")
        decoded = self._stage_count(stages, "decoded")
        admitted = self._stage_count(stages, "admitted")
        inferred = self._stage_count(stages, "inference")
        output = self._stage_count(stages, "output")

        def ratio(numerator: int, denominator: int) -> float | None:
            return round(numerator / denominator, 6) if denominator else None

        wire_missing_count, wire_missing_ranges = self._missing_ranges(
            recent_sequences, "wire", "inference"
        )
        admitted_missing_count, admitted_missing_ranges = self._missing_ranges(
            recent_sequences, "admitted", "inference"
        )
        inference_missing_count, inference_missing_ranges = self._missing_ranges(
            recent_sequences, "inference", "output"
        )
        unexpected_inference_count, unexpected_inference_ranges = self._missing_ranges(
            recent_sequences, "inference", "wire"
        )
        mouth_event_wire_count, mouth_event_wire_ranges = self._missing_ranges(
            recent_sequences, "mouth_event", "camera_wire"
        )
        mouth_event_inference_count, mouth_event_inference_ranges = self._missing_ranges(
            recent_sequences, "mouth_event", "camera_inference"
        )
        mouth_inference_output_count, mouth_inference_output_ranges = self._missing_ranges(
            recent_sequences, "camera_inference", "camera_output"
        )
        recent_wire = set(recent_sequences.get("wire", ()))
        truncated_stages = sorted(
            name
            for name, sequences in recent_sequences.items()
            if self._stage_count(stages, name) > len(sequences)
        )

        return {
            "stages": stages,
            "drops": drops,
            "recent_drops": recent_drops,
            "latest_client": latest_client,
            "coverage": {
                "wire_to_decoded": ratio(decoded, wire),
                "wire_to_inference": ratio(inferred, wire),
                "admitted_to_inference": ratio(inferred, admitted),
                "wire_to_output": ratio(output, wire),
                "inference_to_output": ratio(output, inferred),
            },
            "sequence_audit": {
                "window_limit": self._sequence_window,
                "window_truncated": bool(truncated_stages),
                "window_truncated_stages": truncated_stages,
                "recent_wire_unique": len(recent_wire),
                "every_recent_wire_frame_reached_inference": (
                    not wire_missing_count if recent_wire else None
                ),
                "recent_wire_not_in_inference_count": wire_missing_count,
                "recent_wire_not_in_inference_ranges": wire_missing_ranges,
                "recent_admitted_not_in_inference_count": admitted_missing_count,
                "recent_admitted_not_in_inference_ranges": admitted_missing_ranges,
                "recent_inference_not_in_output_count": inference_missing_count,
                "recent_inference_not_in_output_ranges": inference_missing_ranges,
                "recent_inference_not_in_wire_count": unexpected_inference_count,
                "recent_inference_not_in_wire_ranges": unexpected_inference_ranges,
            },
            "mouth_audit": {
                "recent_landmark_samples": self._stage_count(stages, "mouth_landmark"),
                "recent_significant_events": self._stage_count(stages, "mouth_event"),
                "recent_events_not_on_wire_count": mouth_event_wire_count,
                "recent_events_not_on_wire_ranges": mouth_event_wire_ranges,
                "recent_events_not_in_inference_count": mouth_event_inference_count,
                "recent_events_not_in_inference_ranges": mouth_event_inference_ranges,
                "recent_inference_not_in_output_count": mouth_inference_output_count,
                "recent_inference_not_in_output_ranges": mouth_inference_output_ranges,
            },
        }
