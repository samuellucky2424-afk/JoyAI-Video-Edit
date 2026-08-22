"""Validated metadata contract for landmark-aligned mouth appearance features.

This module deliberately does not inspect frames.  It defines the validation
boundary for the browser feature encoder so that every stage uses the same
names and score semantics before metadata is used for frame retention or the
optional bounded runtime mouth control.

The four region values are evidence scores in the MediaPipe-aligned mouth ROI;
they are not pixel-perfect semantic masks.  The runtime may derive a source
token scale from them, but this module does not alter checkpoints or pixels.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any


MOUTH_ANATOMY_SCHEMA_VERSION = 1
MOUTH_ANATOMY_METHOD = "landmark_aligned_feature_encoder_v2"
MOUTH_ANATOMY_REGIONS = ("lips", "teeth", "tongue", "oral_cavity")

MOUTH_ANATOMY_REGION_MEANINGS = {
    "lips": "visible lip-ring evidence between the outer and inner lip contours",
    "teeth": "bright, low-chroma evidence inside the inner lip contour",
    "tongue": "pink/red evidence inside the inner lip contour",
    "oral_cavity": "dark interior evidence inside the inner lip contour",
}


class MouthAnatomyContractError(ValueError):
    """Raised when mouth-anatomy metadata is incomplete or unsafe to consume."""


def _unit_score(value: Any, field: str) -> float:
    if isinstance(value, bool):
        raise MouthAnatomyContractError(f"{field} must be a number in [0, 1]")
    try:
        score = float(value)
    except (TypeError, ValueError) as error:
        raise MouthAnatomyContractError(
            f"{field} must be a number in [0, 1]"
        ) from error
    if not math.isfinite(score) or not 0.0 <= score <= 1.0:
        raise MouthAnatomyContractError(f"{field} must be a finite number in [0, 1]")
    return round(score, 6)


def normalize_mouth_anatomy(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and return the canonical JSON-safe mouth-anatomy structure.

    ``available`` means the landmark-aligned ROI was sufficiently reliable to
    analyze.  An unavailable result must contain zero evidence and cannot mark a
    frame as significant.  This prevents a failed tracker or poor crop from
    influencing future frame-retention decisions.
    """

    if not isinstance(payload, Mapping):
        raise MouthAnatomyContractError("mouth anatomy payload must be an object")

    if payload.get("schema_version") != MOUTH_ANATOMY_SCHEMA_VERSION:
        raise MouthAnatomyContractError(
            f"schema_version must be {MOUTH_ANATOMY_SCHEMA_VERSION}"
        )
    if payload.get("method") != MOUTH_ANATOMY_METHOD:
        raise MouthAnatomyContractError(
            f"method must be {MOUTH_ANATOMY_METHOD!r}"
        )

    available = payload.get("available")
    significant = payload.get("significant")
    if not isinstance(available, bool):
        raise MouthAnatomyContractError("available must be a boolean")
    if not isinstance(significant, bool):
        raise MouthAnatomyContractError("significant must be a boolean")

    evidence = payload.get("region_evidence")
    if not isinstance(evidence, Mapping):
        raise MouthAnatomyContractError("region_evidence must be an object")
    missing = [region for region in MOUTH_ANATOMY_REGIONS if region not in evidence]
    unknown = sorted(set(evidence) - set(MOUTH_ANATOMY_REGIONS))
    if missing:
        raise MouthAnatomyContractError(
            f"region_evidence is missing: {', '.join(missing)}"
        )
    if unknown:
        raise MouthAnatomyContractError(
            f"region_evidence contains unknown regions: {', '.join(unknown)}"
        )

    normalized_evidence = {
        region: _unit_score(evidence[region], f"region_evidence.{region}")
        for region in MOUTH_ANATOMY_REGIONS
    }
    roi_confidence = _unit_score(payload.get("roi_confidence"), "roi_confidence")
    appearance_motion = _unit_score(
        payload.get("appearance_motion"), "appearance_motion"
    )

    if not available and (
        roi_confidence != 0.0
        or appearance_motion != 0.0
        or significant
        or any(normalized_evidence.values())
    ):
        raise MouthAnatomyContractError(
            "unavailable mouth anatomy must have zero scores and significant=false"
        )

    return {
        "schema_version": MOUTH_ANATOMY_SCHEMA_VERSION,
        "method": MOUTH_ANATOMY_METHOD,
        "available": available,
        "roi_confidence": roi_confidence,
        "region_evidence": normalized_evidence,
        "appearance_motion": appearance_motion,
        "significant": significant,
    }
