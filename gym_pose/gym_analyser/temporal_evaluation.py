"""Temporal and event evaluation engine for Gym Pose Analyser (Task 45).

Pure Python standard-library module and CLI for evaluating fine-grained temporal
repetition boundaries, turnaround proximity, Temporal IoU (tIoU), and movement
category agreement against human-reviewed ground truth from ReviewStore.

Adheres strictly to docs/gym/TEMPORAL_EVAL_CONTRACT_45.md and Review 46 specifications:
- Pure Python standard library (no torch, ultralytics, cv2, numpy, or third-party ML).
- Read-only provenance consuming accepted review_evaluation / review_execution artifacts.
- Reuse of review_execution.load_and_validate_plan and verify_plan_inputs.
- Mandatory execution receipt verification and multi-model common completed cohort binding.
- Unresolved ancestor and symlink/junction checks prior to resolve().
- Deterministic greedy 1-to-1 matching with strict tie-breaking (honestly declared).
- Honest null semantics: missing turnaround remains null, never an interval midpoint.
- Ambiguous human events (ambiguous_or_unclear) never imply false positive or false negative.
- Supported Precision=0 and Recall=0 yields F1=0.0; genuinely unsupported remain null.
- Cohort segregation: primary eligible vs exploratory/unreviewed cohorts kept isolated.
- Fail-closed security: no-clobber output directory (FileExistsError on non-empty).
"""

from __future__ import annotations

import argparse
import copy
import csv
import datetime
import hashlib
import json
import logging
import math
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Mapping,
    NamedTuple,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
)

from .review_evaluation import (
    CHUNK_SIZE,
    SAFE_MODEL_NAME_PATTERN,
    compute_file_sha256,
    reject_linked_ancestors,
    validate_model_name,
)
from .review_execution import (
    compute_plan_sha256 as exec_compute_plan_sha256,
    load_and_validate_plan,
    verify_plan_inputs,
    normalize_declared_configuration,
)
from .review_store import (
    ALLOWED_CYCLES,
    ALLOWED_EVENT_CLASSES,
    ALLOWED_INTENTS,
    ALLOWED_SCOPES,
    CaseCorruptedError,
    CaseNotFoundError,
    InvalidCaseIdError,
    ReviewStore,
    SHA256_HEX_PATTERN,
    _reject_json_constant,
    detect_duplicate_pairs,
    is_symlink,
    is_valid_uuid4,
    validate_review_payload,
)

logger = logging.getLogger(__name__)

# ==============================================================================
# Constants & Defaults
# ==============================================================================

CONTRACT_VERSION: str = "1.0.0"
REPORT_VERSION: str = "1.0"
DEFAULT_TOLERANCE_SECONDS: float = 1.0
MIN_TOLERANCE_SECONDS: float = 0.1

CSV_FIELDNAMES: Sequence[str] = (
    "model_name",
    "case_id",
    "pair_status",
    "human_event_id",
    "human_event_class",
    "human_start_time_s",
    "human_turnaround_time_s",
    "human_end_time_s",
    "human_uncertainty_notes",
    "predicted_attempt_id",
    "predicted_status",
    "predicted_category",
    "predicted_reason",
    "predicted_start_time_s",
    "predicted_turnaround_time_s",
    "predicted_turnaround_source",
    "predicted_end_time_s",
    "classification_label",
    "delta_start_s",
    "delta_turnaround_s",
    "delta_end_s",
    "temporal_iou",
    "temporal_distance",
    "in_primary_cohort",
    "notes",
)


# ==============================================================================
# Custom Exceptions
# ==============================================================================


class TemporalEvaluationError(Exception):
    """Base exception for temporal evaluation failures."""


class CorruptedArtifactError(TemporalEvaluationError, ValueError):
    """Raised when an input artifact (plan, receipt, prediction, case) is corrupt."""


class HashMismatchError(TemporalEvaluationError, ValueError, RuntimeError):
    """Raised when an artifact or source file SHA256 digest fails verification."""


# ==============================================================================
# Input & Configuration Validators
# ==============================================================================


def validate_temporal_tolerance(tolerance: Any) -> float:
    """Validate configurable matching tolerance window.

    Args:
        tolerance: Tolerance window in seconds (float >= 0.1).

    Returns:
        Validated float tolerance.

    Raises:
        TypeError: If tolerance is bool or cannot be converted to float.
        ValueError: If tolerance is not finite or < 0.1.
    """
    if isinstance(tolerance, bool):
        raise TypeError(f"tolerance_seconds cannot be a boolean, got {tolerance!r}")
    try:
        val = float(tolerance)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"tolerance_seconds must be a numeric float, got {tolerance!r}") from exc

    if not math.isfinite(val) or val < MIN_TOLERANCE_SECONDS:
        raise ValueError(
            f"tolerance_seconds must be a finite float >= {MIN_TOLERANCE_SECONDS}, got {val}"
        )
    return val


def compute_plan_sha256(plan: Dict[str, Any]) -> str:
    """Compute the canonical SHA256 digest of an evaluation plan payload."""
    return exec_compute_plan_sha256(plan)


def validate_evaluation_output_dir(
    output_dir: Path,
    plan_path: Path,
    execution_dir: Path,
    review_root: Optional[Path],
) -> None:
    """Validate that output_dir does not overlap with inputs or protected dirs, and is empty."""
    reject_linked_ancestors(output_dir, label="Output directory")
    reject_linked_ancestors(output_dir.parent, label="Output parent directory")

    if is_symlink(output_dir):
        raise ValueError(f"Output directory cannot be a symlink or reparse point: {output_dir}")

    out_res = output_dir.resolve()
    plan_res = plan_path.resolve()
    exec_res = execution_dir.resolve()

    pkg_dir = Path(__file__).resolve().parent
    app_root = pkg_dir.parent
    protected_paths = [
        pkg_dir,
        (app_root / "tests").resolve(),
        (app_root / "models").resolve(),
    ]

    for prot in protected_paths:
        if out_res == prot:
            raise ValueError(f"Output overlaps protected directory: {prot}")
        try:
            out_res.relative_to(prot)
            raise ValueError(f"Output overlaps protected directory: {prot}")
        except ValueError as exc:
            if "overlaps protected" in str(exc):
                raise

    if out_res == plan_res or out_res == plan_res.parent:
        raise ValueError(f"Output directory cannot be identical to or contain plan: {plan_path}")
    try:
        plan_res.relative_to(out_res)
        raise ValueError(f"Plan cannot be inside output directory: {plan_path}")
    except ValueError as exc:
        if "cannot be inside" in str(exc):
            raise

    if out_res == exec_res:
        raise ValueError(
            f"Output directory cannot be identical to execution directory: {execution_dir}"
        )
    try:
        out_res.relative_to(exec_res)
        raise ValueError(f"Output directory cannot be inside execution directory: {execution_dir}")
    except ValueError as exc:
        if "cannot be inside" in str(exc):
            raise
    try:
        exec_res.relative_to(out_res)
        raise ValueError(f"Execution directory cannot be inside output directory: {execution_dir}")
    except ValueError as exc:
        if "cannot be inside" in str(exc):
            raise

    if review_root is not None:
        rev_res = review_root.resolve()
        if out_res == rev_res:
            raise ValueError(f"Output directory cannot be identical to review root: {review_root}")
        try:
            out_res.relative_to(rev_res)
            raise ValueError(f"Output directory cannot be inside review root: {review_root}")
        except ValueError as exc:
            if "cannot be inside" in str(exc):
                raise
        try:
            rev_res.relative_to(out_res)
            raise ValueError(f"Review root cannot be inside output directory: {review_root}")
        except ValueError as exc:
            if "cannot be inside" in str(exc):
                raise

    if output_dir.exists():
        if not output_dir.is_dir():
            raise NotADirectoryError(f"Output path exists and is not a directory: {output_dir}")
        if any(output_dir.iterdir()):
            raise FileExistsError(
                f"Refusing to overwrite existing non-empty output directory: {output_dir}"
            )


# ==============================================================================
# Domain Data Classes
# ==============================================================================


class HumanEvent(NamedTuple):
    """Normalized human ground-truth repetition event."""

    event_id: str
    event_class: str
    start_time_s: Optional[float] = None
    turnaround_time_s: Optional[float] = None
    end_time_s: Optional[float] = None
    uncertainty_notes: Optional[str] = None
    visible_posture_observations: Any = None
    is_legacy: bool = False
    raw_payload: Optional[Dict[str, Any]] = None


class PredictedAttempt(NamedTuple):
    """Normalized algorithmic predicted squat attempt."""

    attempt_id: Union[int, str]
    status: str = "uncounted"
    category: str = ""
    reason: str = ""
    start_time_s: Optional[float] = None
    turnaround_time_s: Optional[float] = None
    turnaround_source: str = "missing"
    end_time_s: Optional[float] = None
    min_knee_angle_deg: Optional[float] = None
    minimum_observed_frame: Optional[int] = None
    raw_payload: Optional[Dict[str, Any]] = None


class MatchedPair(NamedTuple):
    """Result of a deterministic 1-to-1 match between a human event and predicted attempt."""

    human: HumanEvent
    predicted: PredictedAttempt
    distance: float
    delta_start_s: Optional[float]
    delta_turnaround_s: Optional[float]
    delta_end_s: Optional[float]
    temporal_iou: Optional[float]
    classification_label: str


# ==============================================================================
# Event & Prediction Normalizers
# ==============================================================================


def parse_human_events(events_raw: Any) -> List[HumanEvent]:
    """Parse and validate human annotation events into normalized HumanEvent objects.

    Supports both modern Schema Version 1 events (event_id, event_class, start_time_s,
    turnaround_time_s, end_time_s, uncertainty_notes, visible_posture_observations) and
    legacy events (event_index, start_time_s, end_time_s, intent, cycle, observation).

    Missing turnaround remains strictly null; interval midpoint is NEVER imputed.

    Args:
        events_raw: Stored events payload (list of dicts or None).

    Returns:
        List of validated HumanEvent objects.

    Raises:
        ValueError: If event data is malformed, has duplicate IDs, or violates constraints.
    """
    if events_raw is None:
        return []
    if not isinstance(events_raw, list):
        raise ValueError(f"events must be a list or None, got {type(events_raw).__name__}")

    parsed: List[HumanEvent] = []
    seen_ids: Set[str] = set()

    for idx, item in enumerate(events_raw):
        if not isinstance(item, dict):
            raise ValueError(f"Event at index {idx} must be a dict, got {type(item).__name__}")

        if "event_id" in item:
            # Modern Version 1 Schema
            raw_id = item.get("event_id")
            if not isinstance(raw_id, str) or not raw_id.strip():
                raise ValueError(f"Event at index {idx} has invalid event_id: {raw_id!r}")
            event_id = raw_id.strip()
            if event_id in seen_ids:
                raise ValueError(f"Duplicate event_id detected in human events: {event_id!r}")
            seen_ids.add(event_id)

            event_class = item.get("event_class")
            if not isinstance(event_class, str) or event_class not in ALLOWED_EVENT_CLASSES:
                raise ValueError(
                    f"Event '{event_id}' has invalid event_class {event_class!r}; "
                    f"must be one of {sorted(ALLOWED_EVENT_CLASSES)}"
                )

            st = item.get("start_time_s")
            tt = item.get("turnaround_time_s")
            et = item.get("end_time_s")

            for name, val in (("start_time_s", st), ("turnaround_time_s", tt), ("end_time_s", et)):
                if val is not None:
                    if isinstance(val, bool) or not isinstance(val, (int, float)):
                        raise ValueError(f"Event '{event_id}' {name} must be numeric or None, got {val!r}")
                    if not math.isfinite(val) or val < 0.0:
                        raise ValueError(f"Event '{event_id}' {name} must be finite and >= 0.0, got {val!r}")

            f_st = float(st) if st is not None else None
            f_tt = float(tt) if tt is not None else None
            f_et = float(et) if et is not None else None

            if f_st is not None and f_tt is not None and f_tt < f_st:
                raise ValueError(
                    f"Event '{event_id}' turnaround_time_s ({f_tt}) is earlier than start_time_s ({f_st})"
                )
            if f_tt is not None and f_et is not None and f_et < f_tt:
                raise ValueError(
                    f"Event '{event_id}' end_time_s ({f_et}) is earlier than turnaround_time_s ({f_tt})"
                )
            if f_st is not None and f_et is not None and f_et < f_st:
                raise ValueError(
                    f"Event '{event_id}' end_time_s ({f_et}) is earlier than start_time_s ({f_st})"
                )

            uncertainty = item.get("uncertainty_notes")
            if uncertainty is not None and not isinstance(uncertainty, str):
                raise ValueError(f"Event '{event_id}' uncertainty_notes must be string or None")

            posture = item.get("visible_posture_observations")

            parsed.append(
                HumanEvent(
                    event_id=event_id,
                    event_class=event_class,
                    start_time_s=f_st,
                    turnaround_time_s=f_tt,
                    end_time_s=f_et,
                    uncertainty_notes=uncertainty,
                    visible_posture_observations=posture,
                    is_legacy=False,
                    raw_payload=item,
                )
            )

        elif "event_index" in item:
            # Legacy Schema
            raw_idx = item.get("event_index")
            if isinstance(raw_idx, bool) or not isinstance(raw_idx, int) or raw_idx < 0:
                raise ValueError(f"Legacy event at index {idx} has invalid event_index: {raw_idx!r}")
            event_id = f"legacy_{raw_idx}"
            if event_id in seen_ids:
                raise ValueError(f"Duplicate legacy event_index detected: {raw_idx}")
            seen_ids.add(event_id)

            st_leg = item.get("start_time_s")
            et_leg = item.get("end_time_s")
            if isinstance(st_leg, bool) or not isinstance(st_leg, (int, float)) or not math.isfinite(st_leg) or st_leg < 0.0:
                raise ValueError(f"Legacy event '{event_id}' start_time_s must be finite >= 0.0")
            if isinstance(et_leg, bool) or not isinstance(et_leg, (int, float)) or not math.isfinite(et_leg) or et_leg < st_leg:
                raise ValueError(f"Legacy event '{event_id}' end_time_s must be finite >= start_time_s")

            intent = item.get("intent")
            cycle = item.get("cycle")
            if not isinstance(intent, str) or intent not in ALLOWED_INTENTS:
                raise ValueError(f"Legacy event '{event_id}' invalid intent: {intent!r}")
            if not isinstance(cycle, str) or cycle not in ALLOWED_CYCLES:
                raise ValueError(f"Legacy event '{event_id}' invalid cycle: {cycle!r}")

            # Map legacy semantics cleanly
            if intent == "intentional_squat" and cycle == "complete":
                event_class = "completed_rep"
            elif intent == "setup_return":
                event_class = "setup_or_weight_return"
            elif cycle == "incomplete":
                event_class = "partial_or_unfinished"
            else:
                event_class = "ambiguous_or_unclear"

            obs = item.get("observation")

            parsed.append(
                HumanEvent(
                    event_id=event_id,
                    event_class=event_class,
                    start_time_s=float(st_leg),
                    turnaround_time_s=None,  # Legacy annotations lack turnaround timestamps
                    end_time_s=float(et_leg),
                    uncertainty_notes=obs if isinstance(obs, str) else None,
                    visible_posture_observations=None,
                    is_legacy=True,
                    raw_payload=item,
                )
            )
        else:
            raise ValueError(
                f"Event at index {idx} does not match modern or legacy event schema: {item!r}"
            )

    return parsed


def parse_predicted_attempts(
    attempts_data: Any,
    fps: Optional[float] = None,
) -> List[PredictedAttempt]:
    """Parse attempts.json payload into normalized PredictedAttempt objects.

    Missing turnaround remains strictly null; interval midpoint is NEVER imputed.
    Only explicit documented timestamp or frame-plus-fps may supply turnaround.

    Args:
        attempts_data: Dict containing 'events' or a list of attempt dicts.
        fps: Optional video frame rate for deriving turnaround timestamp.

    Returns:
        List of validated PredictedAttempt objects.

    Raises:
        ValueError: If attempts payload is malformed or has duplicate IDs.
    """
    if isinstance(attempts_data, dict) and "events" in attempts_data:
        raw_list = attempts_data["events"]
    elif isinstance(attempts_data, list):
        raw_list = attempts_data
    else:
        raise ValueError(
            f"Unexpected attempts payload structure; expected dict with 'events' or list, "
            f"got {type(attempts_data).__name__}"
        )

    if not isinstance(raw_list, list):
        raise ValueError(f"Attempts events must be a list, got {type(raw_list).__name__}")

    parsed: List[PredictedAttempt] = []
    seen_ids: Set[Union[int, str]] = set()

    for idx, item in enumerate(raw_list):
        if not isinstance(item, dict):
            raise ValueError(f"Attempt at index {idx} must be a dict, got {type(item).__name__}")

        att_id = item.get("attempt_id")
        if att_id is None:
            raise ValueError("Missing attempt_id")
        if isinstance(att_id, bool):
            raise ValueError(f"Attempt at index {idx} has boolean attempt_id: {att_id!r}")
        if isinstance(att_id, int):
            if not math.isfinite(att_id):
                raise ValueError(f"Attempt at index {idx} has non-finite attempt_id: {att_id!r}")
            att_id = int(att_id)
        elif isinstance(att_id, str):
            if not att_id.strip():
                raise ValueError(f"Attempt at index {idx} has empty attempt_id string")
            att_id = att_id.strip()
        else:
            raise ValueError(f"Attempt at index {idx} has invalid attempt_id type: {type(att_id).__name__}")

        if att_id in seen_ids:
            raise ValueError(f"Duplicate attempt_id detected in predictions: {att_id!r}")
        seen_ids.add(att_id)

        st = item.get("start_time_s")
        et = item.get("end_time_s")

        if st is not None:
            if isinstance(st, bool) or not isinstance(st, (int, float)) or not math.isfinite(st) or st < 0.0:
                raise ValueError(f"Attempt '{att_id}' start_time_s must be finite numeric >= 0.0, got {st!r}")
            f_st = float(st)
        else:
            f_st = None

        if et is not None:
            if isinstance(et, bool) or not isinstance(et, (int, float)) or not math.isfinite(et) or et < 0.0:
                raise ValueError(f"Attempt '{att_id}' end_time_s must be finite numeric >= 0.0, got {et!r}")
            f_et = float(et)
        else:
            f_et = None

        if f_st is not None and f_et is not None and f_et < f_st:
            raise ValueError(f"Attempt '{att_id}' end_time_s ({f_et}) is earlier than start_time_s ({f_st})")

        min_frame = item.get("minimum_observed_frame")
        if isinstance(min_frame, bool) or (min_frame is not None and not isinstance(min_frame, int)):
            raise ValueError("minimum_observed_frame must be an integer or null")
        if min_frame is not None and min_frame < 0:
            raise ValueError("minimum_observed_frame must be nonnegative")
        if fps is not None and (isinstance(fps, bool) or not isinstance(fps, (int, float))
                                or not math.isfinite(fps) or fps <= 0):
            raise ValueError("fps must be finite positive numeric or null")

        # Turnaround calculation: strictly explicit timestamp or frame-plus-fps ONLY.
        # NEVER invent an interval midpoint!
        explicit_tt = item.get("turnaround_time_s")
        if explicit_tt is not None and (isinstance(explicit_tt, bool)
                or not isinstance(explicit_tt, (int, float)) or not math.isfinite(explicit_tt)
                or explicit_tt < 0):
            raise ValueError("turnaround_time_s must be finite nonnegative numeric or null")
        if (
            explicit_tt is not None
            and not isinstance(explicit_tt, bool)
            and isinstance(explicit_tt, (int, float))
            and math.isfinite(explicit_tt)
            and explicit_tt >= 0.0
        ):
            tt = float(explicit_tt)
            tt_source = "explicit_timestamp"
        elif (
            min_frame is not None
            and fps is not None
            and fps > 0.0
            and math.isfinite(fps)
        ):
            tt = float(min_frame) / float(fps)
            tt_source = "minimum_observed_frame"
        else:
            tt = None
            tt_source = "missing"

        if tt is not None:
            if f_st is not None and tt < f_st:
                raise ValueError(
                    f"Attempt '{att_id}' turnaround_time_s ({tt}) is earlier than start_time_s ({f_st})"
                )
            if f_et is not None and f_et < tt:
                raise ValueError(
                    f"Attempt '{att_id}' end_time_s ({f_et}) is earlier than turnaround_time_s ({tt})"
                )

        raw_status = item.get("status")
        if not isinstance(raw_status, str):
            raise ValueError(f"Attempt '{att_id}' status must be a string, got {type(raw_status).__name__}")
        status = raw_status.strip().lower()
        if status not in ("counted", "uncounted"):
            raise ValueError("Unsupported prediction status")

        category = str(item.get("category", "")).strip().lower()
        reason = str(item.get("reason", "")).strip()

        min_angle = item.get("min_knee_angle_deg")
        if min_angle is not None:
            if isinstance(min_angle, bool) or not isinstance(min_angle, (int, float)) or not math.isfinite(min_angle):
                f_min_angle = None
            else:
                f_min_angle = float(min_angle)
        else:
            f_min_angle = None

        parsed.append(
            PredictedAttempt(
                attempt_id=att_id,
                status=status,
                category=category,
                reason=reason,
                start_time_s=f_st,
                turnaround_time_s=tt,
                turnaround_source=tt_source,
                end_time_s=f_et,
                min_knee_angle_deg=f_min_angle,
                minimum_observed_frame=min_frame,
                raw_payload=item,
            )
        )

    return parsed


# ==============================================================================
# Deterministic Greedy Matching
# ==============================================================================


def is_candidate_match(
    h: HumanEvent,
    p: PredictedAttempt,
    tolerance_seconds: float,
) -> bool:
    """Check if predicted attempt and human event meet temporal candidate criteria.

    A predicted attempt P is a candidate match for human event H if:
    1. Overlap constraint: max(0, min(P_end, H_end) - max(P_start, H_start)) > 0, OR
    2. Turnaround proximity: |P_turn - H_turn| <= tolerance_seconds (if both present), OR
    3. Center proximity: |P_mid - H_mid| <= tolerance_seconds when turnaround is missing.
    """
    # 1. Overlap constraint
    if (
        p.start_time_s is not None
        and p.end_time_s is not None
        and h.start_time_s is not None
        and h.end_time_s is not None
    ):
        overlap = min(p.end_time_s, h.end_time_s) - max(p.start_time_s, h.start_time_s)
        if overlap > 0.0:
            return True

    # 2. Turnaround proximity
    if p.turnaround_time_s is not None and h.turnaround_time_s is not None:
        if abs(p.turnaround_time_s - h.turnaround_time_s) <= tolerance_seconds:
            return True

    # 3. Center proximity when turnaround is missing
    turnaround_missing = (p.turnaround_time_s is None or h.turnaround_time_s is None)
    if turnaround_missing:
        if (
            p.start_time_s is not None
            and p.end_time_s is not None
            and h.start_time_s is not None
            and h.end_time_s is not None
        ):
            p_mid = (p.start_time_s + p.end_time_s) / 2.0
            h_mid = (h.start_time_s + h.end_time_s) / 2.0
            if abs(p_mid - h_mid) <= tolerance_seconds:
                return True

    return False


def compute_temporal_distance(
    h: HumanEvent,
    p: PredictedAttempt,
    w_turn: float = 1.0,
    w_start: float = 1.0,
    w_end: float = 1.0,
) -> float:
    """Compute normalized temporal distance across available boundaries.

    Missing boundaries are assigned weight 0 and normalized by sum of active weights.

    Returns:
        Normalized non-negative float distance, or float('inf') if no boundaries are shared.
    """
    active_weights = 0.0
    weighted_err = 0.0

    if h.turnaround_time_s is not None and p.turnaround_time_s is not None:
        weighted_err += w_turn * abs(p.turnaround_time_s - h.turnaround_time_s)
        active_weights += w_turn

    if h.start_time_s is not None and p.start_time_s is not None:
        weighted_err += w_start * abs(p.start_time_s - h.start_time_s)
        active_weights += w_start

    if h.end_time_s is not None and p.end_time_s is not None:
        weighted_err += w_end * abs(p.end_time_s - h.end_time_s)
        active_weights += w_end

    if active_weights > 0.0:
        return weighted_err / active_weights
    return float("inf")


def match_events_1to1(
    human_events: Sequence[HumanEvent],
    predicted_attempts: Sequence[PredictedAttempt],
    tolerance_seconds: float = DEFAULT_TOLERANCE_SECONDS,
    w_turn: float = 1.0,
    w_start: float = 1.0,
    w_end: float = 1.0,
) -> Tuple[List[MatchedPair], List[HumanEvent], List[PredictedAttempt]]:
    """Perform deterministic greedy 1-to-1 matching for a single video clip.

    Matching policy:
    1. Candidate pairs meeting proximity/overlap criteria are identified.
    2. Distance metric is computed per pair across active boundaries.
    3. Pairs are sorted by:
       - Distance (ascending)
       - Earlier human start time (ascending)
       - Smaller predicted start time (ascending)
       - Lower human event_id (ascending)
       - Lower predicted attempt_id (ascending)
    4. Pairs are matched greedily without replacement.
       Greedy matching is deterministic and reproducible; it is NOT described as globally optimal.

    Ambiguous human events (ambiguous_or_unclear) are matched if candidate criteria permit,
    but are NEVER labeled as FP or FN.
    """
    candidate_pairs: List[Tuple[HumanEvent, PredictedAttempt, float]] = []

    for h in human_events:
        for p in predicted_attempts:
            if is_candidate_match(h, p, tolerance_seconds):
                dist = compute_temporal_distance(h, p, w_turn, w_start, w_end)
                candidate_pairs.append((h, p, dist))

    def sort_key(item: Tuple[HumanEvent, PredictedAttempt, float]) -> Tuple[float, float, float, str, Tuple[int, Any]]:
        h, p, dist = item
        h_st = h.start_time_s if h.start_time_s is not None else float("inf")
        p_st = p.start_time_s if p.start_time_s is not None else float("inf")
        h_id = str(h.event_id)
        try:
            p_id_sort = (0, int(p.attempt_id))
        except (ValueError, TypeError):
            p_id_sort = (1, str(p.attempt_id))
        return (dist, h_st, p_st, h_id, p_id_sort)

    candidate_pairs.sort(key=sort_key)

    matched_pairs: List[MatchedPair] = []
    matched_h_ids: Set[str] = set()
    matched_p_ids: Set[Union[int, str]] = set()

    for h, p, dist in candidate_pairs:
        if h.event_id in matched_h_ids or p.attempt_id in matched_p_ids:
            continue

        matched_h_ids.add(h.event_id)
        matched_p_ids.add(p.attempt_id)

        dt_start = (
            (p.start_time_s - h.start_time_s)
            if (p.start_time_s is not None and h.start_time_s is not None)
            else None
        )
        dt_turn = (
            (p.turnaround_time_s - h.turnaround_time_s)
            if (p.turnaround_time_s is not None and h.turnaround_time_s is not None)
            else None
        )
        dt_end = (
            (p.end_time_s - h.end_time_s)
            if (p.end_time_s is not None and h.end_time_s is not None)
            else None
        )

        tiou = None
        if (
            p.start_time_s is not None
            and p.end_time_s is not None
            and h.start_time_s is not None
            and h.end_time_s is not None
        ):
            inter = max(0.0, min(p.end_time_s, h.end_time_s) - max(p.start_time_s, h.start_time_s))
            union = max(p.end_time_s, h.end_time_s) - min(p.start_time_s, h.start_time_s)
            tiou = (inter / union) if union > 0.0 else 0.0

        # Classification label assignment
        # Ambiguous events NEVER imply false positive or false negative
        if h.event_class == "completed_rep" and p.status == "counted":
            lbl = "TP"
        elif h.event_class == "completed_rep" and p.status != "counted":
            lbl = "FN_rep_uncounted"
        elif h.event_class == "ambiguous_or_unclear":
            lbl = f"matched_ambiguous_{p.status}"
        elif h.event_class in ("setup_or_weight_return", "partial_or_unfinished") and p.status == "counted":
            lbl = f"FP_{h.event_class}_as_counted"
        else:
            lbl = f"matched_{h.event_class}_{p.status}"

        matched_pairs.append(
            MatchedPair(
                human=h,
                predicted=p,
                distance=dist,
                delta_start_s=dt_start,
                delta_turnaround_s=dt_turn,
                delta_end_s=dt_end,
                temporal_iou=tiou,
                classification_label=lbl,
            )
        )

    unmatched_human = [h for h in human_events if h.event_id not in matched_h_ids]
    unmatched_predicted = [p for p in predicted_attempts if p.attempt_id not in matched_p_ids]

    return matched_pairs, unmatched_human, unmatched_predicted


# ==============================================================================
# Aggregate Metric Computations
# ==============================================================================


def compute_boundary_error_statistics(errors: Sequence[float]) -> Dict[str, Optional[float]]:
    """Compute mean error (bias), MAE, and RMSE for a sequence of float errors.

    Returns:
        Dict with keys: n, mean_error_s, mae_s, rmse_s. Explicit None when n == 0.
    """
    n = len(errors)
    if n == 0:
        return {
            "n": 0,
            "mean_error_s": None,
            "mae_s": None,
            "rmse_s": None,
        }
    mean_err = sum(errors) / n
    mae = sum(abs(e) for e in errors) / n
    rmse = math.sqrt(sum(e * e for e in errors) / n)
    return {
        "n": n,
        "mean_error_s": round(mean_err, 4),
        "mae_s": round(mae, 4),
        "rmse_s": round(rmse, 4),
    }


def compute_cohort_metrics(
    all_matched_pairs: Sequence[MatchedPair],
    all_unmatched_human: Sequence[HumanEvent],
    all_unmatched_predicted: Sequence[PredictedAttempt],
    *,
    case_exhaustive_flags: Optional[Mapping[str, bool]] = None,
    case_id_for_prediction: Optional[Callable[[Union[int, str]], Optional[str]]] = None,
) -> Dict[str, Any]:
    """Compute full repetition, boundary, and category metrics across an evaluated cohort.

    Semantics:
    - True Positives (TP): Human completed_rep matched to Predicted counted attempt.
    - False Positives (FP):
      * Predicted counted attempt matching human non-rep (setup or partial).
      * Unmatched predicted counted attempt IF the case has exhaustive scorable support.
      * Ambiguous human events (ambiguous_or_unclear) are NEVER counted as FP.
    - False Negatives (FN):
      * Human completed_rep not matched to any predicted counted attempt.
      * Ambiguous human events (ambiguous_or_unclear) are NEVER counted as FN.
    - Precision: TP / (TP + FP) or None (when denominator is 0).
    - Recall: TP / (TP + FN) or None (when denominator is 0).
    - F1-Score:
      * When both Precision and Recall are 0.0 on supported non-zero denominators, F1 is 0.0.
      * When either denominator is 0, F1 is None.
    """
    # 1. Repetition Metrics
    tp = sum(1 for m in all_matched_pairs if m.classification_label == "TP")

    # FP Misclassification: predicted counted matching known non-rep (setup or partial)
    # Excludes ambiguous_or_unclear strictly!
    fp_misclass = sum(
        1 for m in all_matched_pairs
        if m.predicted.status == "counted"
        and m.human.event_class in ("setup_or_weight_return", "partial_or_unfinished")
    )

    # FP Unmatched: unmatched predicted counted attempts
    # If case_exhaustive_flags is provided, only treat unmatched as FP when exhaustive scorable support exists.
    fp_unmatched = 0
    unscored_unmatched = 0
    for p in all_unmatched_predicted:
        if p.status == "counted":
            cid = case_id_for_prediction(p.attempt_id) if case_id_for_prediction else None
            is_exhaustive = False
            if case_exhaustive_flags is not None and cid in case_exhaustive_flags:
                is_exhaustive = case_exhaustive_flags[cid]

            if is_exhaustive:
                fp_unmatched += 1
            else:
                unscored_unmatched += 1

    fp = fp_misclass + fp_unmatched

    # FN: human completed_rep not matched to predicted counted attempt
    # Excludes ambiguous_or_unclear strictly!
    scorable_unmatched_human = [h for h in all_unmatched_human
        if h.turnaround_time_s is not None or
        (h.start_time_s is not None and h.end_time_s is not None)]
    fn_unmatched = sum(h.event_class == "completed_rep" for h in scorable_unmatched_human)
    fn_uncounted = sum(
        1 for m in all_matched_pairs
        if m.human.event_class == "completed_rep" and m.predicted.status != "counted"
    )
    fn = fn_unmatched + fn_uncounted

    # Ambiguous matches (tracked transparently, never counted in FP or FN)
    ambiguous_counted_matches = sum(
        1 for m in all_matched_pairs
        if m.human.event_class == "ambiguous_or_unclear" and m.predicted.status == "counted"
    )

    total_human_reps = tp + fn
    total_predicted_counted = tp + fp + ambiguous_counted_matches + unscored_unmatched

    # Explicit denominators
    denom_p = tp + fp
    denom_r = tp + fn

    precision = (tp / denom_p) if denom_p > 0 else None
    recall = (tp / denom_r) if denom_r > 0 else None

    if precision is not None and recall is not None:
        if (precision + recall) > 0.0:
            f1 = (2.0 * precision * recall) / (precision + recall)
        else:
            # Precision=0 and Recall=0 on supported non-zero denominators yields F1=0.0
            f1 = 0.0
    else:
        # Genuinely unsupported denominator remains None
        f1 = None

    repetition_metrics = {
        "metric_scope": "Agreement on scorable annotated events; not exhaustive video detection accuracy",
        "unscored_unmatched_human_events": len(all_unmatched_human) - len(scorable_unmatched_human),
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "total_human_completed_reps": total_human_reps,
        "total_predicted_counted_attempts": total_predicted_counted,
        "precision": round(precision, 4) if precision is not None else None,
        "recall": round(recall, 4) if recall is not None else None,
        "f1_score": round(f1, 4) if f1 is not None else None,
    }
    if unscored_unmatched > 0:
        repetition_metrics["unscored_unmatched_predictions"] = unscored_unmatched

    # 2. Per-Boundary Errors on True Positives
    tp_pairs = [m for m in all_matched_pairs if m.classification_label == "TP"]
    start_errors = [m.delta_start_s for m in tp_pairs if m.delta_start_s is not None]
    turn_errors = [m.delta_turnaround_s for m in tp_pairs if m.delta_turnaround_s is not None]
    end_errors = [m.delta_end_s for m in tp_pairs if m.delta_end_s is not None]
    tious = [m.temporal_iou for m in tp_pairs if m.temporal_iou is not None]

    boundary_errors = {
        "start": compute_boundary_error_statistics(start_errors),
        "turnaround": compute_boundary_error_statistics(turn_errors),
        "end": compute_boundary_error_statistics(end_errors),
    }

    n_tiou = len(tious)
    temporal_iou = {
        "n": n_tiou,
        "mean_tiou": round(sum(tious) / n_tiou, 4) if n_tiou > 0 else None,
    }

    # 3. Movement Category Agreement
    setup_matched_uncounted = sum(
        1 for m in all_matched_pairs
        if m.human.event_class == "setup_or_weight_return" and m.predicted.status != "counted"
    )
    setup_matched_counted = sum(
        1 for m in all_matched_pairs
        if m.human.event_class == "setup_or_weight_return" and m.predicted.status == "counted"
    )
    setup_unmatched = sum(1 for h in all_unmatched_human if h.event_class == "setup_or_weight_return")
    total_setup = setup_matched_uncounted + setup_matched_counted + setup_unmatched

    partial_matched_uncounted = sum(
        1 for m in all_matched_pairs
        if m.human.event_class == "partial_or_unfinished" and m.predicted.status != "counted"
    )
    partial_matched_counted = sum(
        1 for m in all_matched_pairs
        if m.human.event_class == "partial_or_unfinished" and m.predicted.status == "counted"
    )
    partial_unmatched = sum(1 for h in all_unmatched_human if h.event_class == "partial_or_unfinished")
    total_partial = partial_matched_uncounted + partial_matched_counted + partial_unmatched

    ambig_matched_uncounted = sum(
        1 for m in all_matched_pairs
        if m.human.event_class == "ambiguous_or_unclear" and m.predicted.status != "counted"
    )
    ambig_matched_counted = sum(
        1 for m in all_matched_pairs
        if m.human.event_class == "ambiguous_or_unclear" and m.predicted.status == "counted"
    )
    ambig_unmatched = sum(1 for h in all_unmatched_human if h.event_class == "ambiguous_or_unclear")
    total_ambig = ambig_matched_uncounted + ambig_matched_counted + ambig_unmatched

    movement_agreement = {
        "setup_or_weight_return": {
            "human_event_count": total_setup,
            "matched_uncounted": setup_matched_uncounted,
            "matched_counted": setup_matched_counted,
            "unmatched": setup_unmatched,
            "agreement_rate": (
                round(setup_matched_uncounted / (setup_matched_uncounted + setup_matched_counted), 4)
                if setup_matched_uncounted + setup_matched_counted > 0 else None
            ),
        },
        "partial_or_unfinished": {
            "human_event_count": total_partial,
            "matched_uncounted": partial_matched_uncounted,
            "matched_counted": partial_matched_counted,
            "unmatched": partial_unmatched,
            "agreement_rate": (
                round(partial_matched_uncounted / (partial_matched_uncounted + partial_matched_counted), 4)
                if partial_matched_uncounted + partial_matched_counted > 0 else None
            ),
        },
        "ambiguous_or_unclear": {
            "human_event_count": total_ambig,
            "matched_uncounted": ambig_matched_uncounted,
            "matched_counted": ambig_matched_counted,
            "unmatched": ambig_unmatched,
        },
    }

    return {
        "repetition_metrics": repetition_metrics,
        "boundary_errors": boundary_errors,
        "temporal_iou": temporal_iou,
        "movement_category_agreement": movement_agreement,
    }


# ==============================================================================
# Artifact Builders
# ==============================================================================


def build_matches_csv_rows(
    model_name: str,
    case_id: str,
    matched_pairs: Sequence[MatchedPair],
    unmatched_human: Sequence[HumanEvent],
    unmatched_predicted: Sequence[PredictedAttempt],
    in_primary_cohort: bool,
) -> List[Dict[str, Any]]:
    """Build flattened CSV dictionary records for matched and unmatched events."""
    rows: List[Dict[str, Any]] = []

    # Matched pairs
    for m in matched_pairs:
        rows.append({
            "model_name": model_name,
            "case_id": case_id,
            "pair_status": "matched",
            "human_event_id": m.human.event_id,
            "human_event_class": m.human.event_class,
            "human_start_time_s": f"{m.human.start_time_s:.4f}" if m.human.start_time_s is not None else "",
            "human_turnaround_time_s": f"{m.human.turnaround_time_s:.4f}" if m.human.turnaround_time_s is not None else "",
            "human_end_time_s": f"{m.human.end_time_s:.4f}" if m.human.end_time_s is not None else "",
            "human_uncertainty_notes": m.human.uncertainty_notes or "",
            "predicted_attempt_id": str(m.predicted.attempt_id),
            "predicted_status": m.predicted.status,
            "predicted_category": m.predicted.category,
            "predicted_reason": m.predicted.reason,
            "predicted_start_time_s": f"{m.predicted.start_time_s:.4f}" if m.predicted.start_time_s is not None else "",
            "predicted_turnaround_time_s": f"{m.predicted.turnaround_time_s:.4f}" if m.predicted.turnaround_time_s is not None else "",
            "predicted_turnaround_source": m.predicted.turnaround_source,
            "predicted_end_time_s": f"{m.predicted.end_time_s:.4f}" if m.predicted.end_time_s is not None else "",
            "classification_label": m.classification_label if in_primary_cohort else "diagnostic_" + m.classification_label,
            "delta_start_s": f"{m.delta_start_s:.4f}" if m.delta_start_s is not None else "",
            "delta_turnaround_s": f"{m.delta_turnaround_s:.4f}" if m.delta_turnaround_s is not None else "",
            "delta_end_s": f"{m.delta_end_s:.4f}" if m.delta_end_s is not None else "",
            "temporal_iou": f"{m.temporal_iou:.4f}" if m.temporal_iou is not None else "",
            "temporal_distance": f"{m.distance:.4f}" if math.isfinite(m.distance) else "",
            "in_primary_cohort": in_primary_cohort,
            "notes": "",
        })

    # Unmatched human events
    for h in unmatched_human:
        scorable = (h.turnaround_time_s is not None or
                    (h.start_time_s is not None and h.end_time_s is not None))
        lbl = ("FN" if in_primary_cohort and scorable and h.event_class == "completed_rep"
               else f"unscored_unmatched_human_{h.event_class}")
        rows.append({
            "model_name": model_name,
            "case_id": case_id,
            "pair_status": "unmatched_human",
            "human_event_id": h.event_id,
            "human_event_class": h.event_class,
            "human_start_time_s": f"{h.start_time_s:.4f}" if h.start_time_s is not None else "",
            "human_turnaround_time_s": f"{h.turnaround_time_s:.4f}" if h.turnaround_time_s is not None else "",
            "human_end_time_s": f"{h.end_time_s:.4f}" if h.end_time_s is not None else "",
            "human_uncertainty_notes": h.uncertainty_notes or "",
            "predicted_attempt_id": "",
            "predicted_status": "",
            "predicted_category": "",
            "predicted_reason": "",
            "predicted_start_time_s": "",
            "predicted_turnaround_time_s": "",
            "predicted_turnaround_source": "",
            "predicted_end_time_s": "",
            "classification_label": lbl,
            "delta_start_s": "",
            "delta_turnaround_s": "",
            "delta_end_s": "",
            "temporal_iou": "",
            "temporal_distance": "",
            "in_primary_cohort": in_primary_cohort,
            "notes": "Unmatched human annotation event",
        })

    # Unmatched predicted attempts
    for p in unmatched_predicted:
        lbl = f"unscored_unmatched_predicted_{p.status}"
        rows.append({
            "model_name": model_name,
            "case_id": case_id,
            "pair_status": "unmatched_predicted",
            "human_event_id": "",
            "human_event_class": "",
            "human_start_time_s": "",
            "human_turnaround_time_s": "",
            "human_end_time_s": "",
            "human_uncertainty_notes": "",
            "predicted_attempt_id": str(p.attempt_id),
            "predicted_status": p.status,
            "predicted_category": p.category,
            "predicted_reason": p.reason,
            "predicted_start_time_s": f"{p.start_time_s:.4f}" if p.start_time_s is not None else "",
            "predicted_turnaround_time_s": f"{p.turnaround_time_s:.4f}" if p.turnaround_time_s is not None else "",
            "predicted_turnaround_source": p.turnaround_source,
            "predicted_end_time_s": f"{p.end_time_s:.4f}" if p.end_time_s is not None else "",
            "classification_label": lbl,
            "delta_start_s": "",
            "delta_turnaround_s": "",
            "delta_end_s": "",
            "temporal_iou": "",
            "temporal_distance": "",
            "in_primary_cohort": in_primary_cohort,
            "notes": "Unmatched algorithmic predicted attempt",
        })

    return rows


def format_temporal_summary_text(report: Dict[str, Any]) -> str:
    """Format human-readable text summary of temporal event evaluation."""
    lines: List[str] = [
        "=" * 64,
        "          GYM POSE TEMPORAL EVENT EVALUATION REPORT",
        "=" * 64,
        f"Cohort Status           : {report['cohort_status']}",
        f"Generated (UTC)         : {report['generated_utc']}",
        f"Plan Path               : {report['plan_path']}",
        f"Plan SHA256             : {report['plan_sha256']}",
        f"Execution Path          : {report['execution_path']}",
        f"Execution Receipt SHA256: {report.get('execution_receipt_sha256') or 'none'}",
        f"Review Root             : {report['review_root']}",
        f"Output Directory        : {report['output_dir']}",
        f"Tolerance Window        : {report['configuration']['tolerance_seconds']} s",
        f"Allow Unreviewed        : {report['configuration']['allow_unreviewed']}",
        "",
        "Cohort Breakdown        :",
        f"  - Primary eligible    : {report['cohort_summary']['primary_eligible_cases_count']}",
        f"  - Primary evaluated   : {len(report['cohort_summary']['primary_cases'])}",
        f"  - Secondary (seen)    : {report['cohort_summary']['secondary_cases_count']}",
        f"  - Excluded cases      : {report['cohort_summary']['excluded_cases_count']}",
        f"  - Incomplete decoding : {len(report['cohort_summary']['incomplete_decoding_cases'])}",
        "",
    ]

    if report["cohort_status"] == "not_ready":
        lines.extend([
            "Primary Cohort Status   : NOT READY (0 eligible complete cases with event annotations)",
            "Reason                  : " + (report.get("not_ready_reason") or "Eligible cohort is empty or lacks annotations."),
            "",
            "Aggregate metrics remain null. No accuracy or posture grading claimed.",
            "=" * 64,
        ])
        return "\n".join(lines)

    lines.append("Per-Model Evaluation Results:")
    for m_name, m_res in report["per_model_results"].items():
        lines.extend([
            "-" * 64,
            f"Model: {m_name}",
            "-" * 64,
        ])
        rep = m_res.get("repetition_metrics") or {}
        lines.extend([
            "Repetition Classification (Completed Reps):",
            f"  - True Positives (TP) : {rep.get('true_positives')}",
            f"  - False Positives (FP): {rep.get('false_positives')}",
            f"  - False Negatives (FN): {rep.get('false_negatives')}",
            f"  - Total Human Reps    : {rep.get('total_human_completed_reps')}",
            f"  - Total Counted Preds : {rep.get('total_predicted_counted_attempts')}",
            f"  - Precision           : {rep.get('precision')}",
            f"  - Recall              : {rep.get('recall')}",
            f"  - F1-Score            : {rep.get('f1_score')}",
            "",
            "Per-Boundary Temporal Errors (on True Positives):",
        ])
        b_err = m_res.get("boundary_errors") or {}
        for b_name in ("start", "turnaround", "end"):
            stats = b_err.get(b_name) or {}
            lines.append(
                f"  - {b_name.capitalize():11}: Mean = {stats.get('mean_error_s')}s, "
                f"MAE = {stats.get('mae_s')}s, RMSE = {stats.get('rmse_s')}s (N = {stats.get('n')})"
            )

        tiou = m_res.get("temporal_iou") or {}
        lines.extend([
            "",
            f"Temporal IoU (tIoU) on True Positives: Mean = {tiou.get('mean_tiou')} (N = {tiou.get('n')})",
            "",
            "Movement Category Agreement:",
        ])
        agr = m_res.get("movement_category_agreement") or {}
        for cat_name, cat_stats in agr.items():
            lines.append(
                f"  - {cat_name}: total={cat_stats.get('human_event_count')}, "
                f"matched_uncounted={cat_stats.get('matched_uncounted')}, "
                f"matched_counted={cat_stats.get('matched_counted')}, "
                f"unmatched={cat_stats.get('unmatched')}"
            )
        lines.append("")

    lines.extend([
        "=" * 64,
        "Methodological Disclaimers:",
        f"  - {report['limitations_and_disclaimers']['gold_standard_note']}",
        f"  - {report['limitations_and_disclaimers']['source_independence_note']}",
        f"  - {report['limitations_and_disclaimers']['exercise_intent_note']}",
        f"  - {report['limitations_and_disclaimers']['incomplete_decoding_note']}",
        "=" * 64,
    ])
    return "\n".join(lines)


# ==============================================================================
# Core Engine: evaluate_temporal_events
# ==============================================================================


def _validate_completed_summary(summary, prediction, case, review_root, model_path, plan, receipt):
    """Require the real pipeline's completed-output contract, without running inference."""
    if summary.get("status") != "completed" or summary.get("error"):
        raise ValueError("Summary is not successfully completed")
    source = summary.get("source_video")
    config = summary.get("configuration")
    if not isinstance(source, dict) or not isinstance(config, dict):
        raise ValueError("Missing source/configuration")
    raw = review_root / case["case_id"] / ("video" + case["extension"])
    if not isinstance(source.get("path"), str) or Path(source["path"]).resolve() != raw.resolve():
        raise ValueError("Source video path mismatch")
    if not isinstance(summary.get("model_name"), str) or Path(summary["model_name"]).resolve() != model_path:
        raise ValueError("Model checkpoint path mismatch")
    processed, declared = summary.get("processed_frames"), source.get("declared_frame_count")
    if type(processed) is not int or processed <= 0:
        raise ValueError("Missing positive processed frame count")
    if declared is not None and (type(declared) is not int or declared <= 0 or processed < declared):
        raise ValueError("Incomplete/invalid declared frame count")
    fps = source.get("fps")
    if isinstance(fps, bool) or not isinstance(fps, (int, float)) or not math.isfinite(fps) or fps <= 0:
        raise ValueError("Missing valid source fps")
    count = summary.get("completed_reps")
    if (type(count) is not int or count < 0 or not isinstance(prediction, dict)
            or prediction.get("status") != "completed" or prediction.get("completed_reps") != count):
        raise ValueError("Summary count differs from completed receipt prediction")
    expected = normalize_declared_configuration(plan.get("declared_configuration"))
    fields = {"down_angle":"down_angle", "up_angle":"up_angle", "min_joint_confidence":"min_confidence",
              "stable_frames":"stable_frames", "departure_frames":"departure_frames", "max_gap_seconds":"max_gap"}
    for actual, key in fields.items():
        value = config.get(actual)
        if (isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
                or (actual in ("stable_frames", "departure_frames") and type(value) is not int)
                or not math.isclose(value, expected[key], rel_tol=1e-12, abs_tol=1e-12)):
            raise ValueError(f"Configuration mismatch: {actual}")
    device = receipt.get("runtime_provenance", {}).get("device")
    if config.get("tracker") != expected["tracker"] or not isinstance(device, str) or config.get("device") != device:
        raise ValueError("Configuration tracker/device mismatch")


def evaluate_temporal_events(
    plan_path_or_dir: Union[str, Path],
    execution_dir: Union[str, Path],
    output_dir: Union[str, Path],
    tolerance_seconds: float = DEFAULT_TOLERANCE_SECONDS,
    allow_unreviewed: bool = False,
    *,
    _seam_hook: Optional[Callable[[], None]] = None,
) -> Dict[str, Any]:
    """Execute deterministic temporal event evaluation against human ground truth.

    Pure-stdlib function implementing docs/gym/TEMPORAL_EVAL_CONTRACT_45.md:
    1. Loads and validates evaluation plan, models, review store cases, and execution receipts.
    2. Validates SHA256 integrity of all input files and rejects symlinks / path overlap.
    3. Segregates primary eligible cohort (zero prior exposure, reviewed count, raw video ack)
       from secondary/unreviewed cohorts.
    4. Matches predicted attempts to human events using deterministic 1-to-1 greedy matching
       with strict tie-breaking.
    5. Computes repetition classification (TP, FP, FN, Precision, Recall, F1), per-boundary raw
       and absolute errors with explicit non-null denominators, tIoU, and movement category agreement.
    6. Flushes artifacts (JSON report, CSV matches, TXT summary) atomically via a temporary staging dir.

    Args:
        plan_path_or_dir: Path to plan.json or directory containing it.
        execution_dir: Path to execution receipt or directory containing attempts.json.
        output_dir: Fresh output directory for temporal evaluation artifacts.
        tolerance_seconds: Proximity tolerance window in seconds (default 1.0s, >= 0.1s).
        allow_unreviewed: If True, evaluates cases with annotations separately as exploratory cohort.

    Returns:
        Full audit report dictionary matching temporal_evaluation_report.json.

    Raises:
        FileNotFoundError: If plan, execution, or required inputs are missing.
        FileExistsError: If output_dir exists and is non-empty (fail-closed no-clobber).
        ValueError: If arguments, paths, or schema structures are invalid.
        CorruptedArtifactError: If input JSON files or metadata are corrupted.
        HashMismatchError: If file SHA256 digests differ from recorded expectations.
    """
    tol_s = validate_temporal_tolerance(tolerance_seconds)
    p_plan_input = Path(plan_path_or_dir)
    p_exec_input = Path(execution_dir)
    p_out = Path(output_dir)

    # 1. Unresolved ancestor and symlink checks BEFORE resolve()
    reject_linked_ancestors(p_plan_input, label="Plan path")
    reject_linked_ancestors(p_exec_input, label="Execution path")
    reject_linked_ancestors(p_out, label="Output directory")

    if is_symlink(p_plan_input):
        raise ValueError(f"Plan path cannot be a symlink: {p_plan_input}")
    if is_symlink(p_exec_input):
        raise ValueError(f"Execution path cannot be a symlink: {p_exec_input}")
    if is_symlink(p_out):
        raise ValueError(f"Output directory cannot be a symlink: {p_out}")

    # 2. Output directory no-clobber check
    if p_out.exists():
        if not p_out.is_dir():
            raise NotADirectoryError(f"Output path exists and is not a directory: {p_out}")
        if any(p_out.iterdir()):
            raise FileExistsError(
                f"Refusing to overwrite existing non-empty output directory: {p_out}"
            )

    # 3. Resolve Plan File
    if p_plan_input.is_dir():
        if (p_plan_input / "plan.json").is_file():
            p_plan = p_plan_input / "plan.json"
        elif (p_plan_input / "evaluation_plan.json").is_file():
            p_plan = p_plan_input / "evaluation_plan.json"
        else:
            raise FileNotFoundError(f"No plan.json or evaluation_plan.json found in {p_plan_input}")
    elif p_plan_input.is_file():
        p_plan = p_plan_input
    else:
        raise FileNotFoundError(f"Plan file not found: {p_plan_input}")

    reject_linked_ancestors(p_plan, label="Plan file")
    if is_symlink(p_plan):
        raise ValueError(f"Plan file cannot be a symlink: {p_plan}")

    # Load & validate plan using review_execution.load_and_validate_plan
    try:
        plan_dict, computed_plan_sha = load_and_validate_plan(p_plan)
    except Exception as exc:
        err_msg = str(exc).lower()
        if "hash mismatch" in err_msg or "sha256 mismatch" in err_msg:
            raise HashMismatchError(f"Plan hash validation failed: {exc}") from exc
        elif isinstance(exc, (ValueError, CaseCorruptedError)):
            raise CorruptedArtifactError(f"Plan validation failed: {exc}") from exc
        raise

    # 4. Resolve Execution Path & Receipt
    if p_exec_input.is_file():
        receipt_file = p_exec_input
        exec_root = p_exec_input.parent
    elif p_exec_input.is_dir():
        exec_root = p_exec_input
        if (exec_root / "evaluation_receipt.json").is_file():
            receipt_file = exec_root / "evaluation_receipt.json"
        elif (exec_root / "receipt.json").is_file():
            receipt_file = exec_root / "receipt.json"
        else:
            receipt_file = None
    else:
        raise FileNotFoundError(f"Execution path not found: {p_exec_input}")

    reject_linked_ancestors(exec_root, label="Execution root")
    if is_symlink(exec_root):
        raise ValueError(f"Execution root cannot be a symlink: {exec_root}")

    # Validate execution_receipt.json existence and validity
    if receipt_file is None or not receipt_file.is_file():
        raise FileNotFoundError(
            f"Execution receipt file (evaluation_receipt.json) not found in {exec_root}"
        )

    reject_linked_ancestors(receipt_file, label="Execution receipt file")
    if is_symlink(receipt_file):
        raise ValueError(f"Receipt file cannot be a symlink: {receipt_file}")

    input_hashes: Dict[str, str] = {
        str(p_plan.resolve()): compute_file_sha256(p_plan),
        str(Path(__file__).resolve()): compute_file_sha256(Path(__file__)),
    }

    receipt_bytes = receipt_file.read_bytes()
    receipt_sha = hashlib.sha256(receipt_bytes).hexdigest().lower()
    input_hashes[str(receipt_file.resolve())] = receipt_sha

    try:
        receipt_dict = json.loads(
            receipt_bytes.decode("utf-8"),
            object_pairs_hook=detect_duplicate_pairs,
            parse_constant=_reject_json_constant,
        )
    except Exception as exc:
        raise CorruptedArtifactError(f"Failed to parse receipt JSON: {exc}") from exc

    if not isinstance(receipt_dict, dict):
        raise CorruptedArtifactError(
            f"Execution receipt must be a JSON object, got {type(receipt_dict).__name__}"
        )

    # Validate receipt plan_sha256
    rec_plan_sha = receipt_dict.get("plan_sha256")
    if not rec_plan_sha or not isinstance(rec_plan_sha, str):
        raise CorruptedArtifactError("Execution receipt is missing mandatory plan_sha256")
    if rec_plan_sha.strip().lower() != computed_plan_sha:
        raise HashMismatchError(
            f"Execution receipt plan_sha256 ({rec_plan_sha}) mismatch with plan canonical hash ({computed_plan_sha})"
        )

    if receipt_dict.get("receipt_version") != "1.0":
        raise CorruptedArtifactError("Unsupported or missing execution receipt_version")
    receipt_status = receipt_dict.get("status")
    if receipt_status not in ("completed", "not_ready", "failed"):
        raise CorruptedArtifactError("Unsupported execution receipt status")
    receipt_failed = receipt_status != "completed"
    plan_models = plan_dict["models"]
    if receipt_dict.get("models") != plan_models:
        raise HashMismatchError("Receipt models must exactly match the frozen plan models")
    if receipt_dict.get("declared_configuration") != plan_dict.get("declared_configuration", {}):
        raise HashMismatchError("Receipt configuration differs from frozen plan")
    frozen_review_root = Path(plan_dict["review_root"])
    if not frozen_review_root.is_absolute():
        frozen_review_root = p_plan.parent / frozen_review_root
    for key, expected in (("plan_path", p_plan), ("review_root", frozen_review_root),
                          ("output_dir", exec_root)):
        value = receipt_dict.get(key)
        if not isinstance(value, str) or Path(value).resolve() != expected.resolve():
            raise CorruptedArtifactError(f"Receipt {key} does not match its input")
    eligible_ids = {c["case_id"] for c in plan_dict["eligible_cases"]}
    raw_common = receipt_dict.get("common_completed_cohort")
    if (not isinstance(raw_common, list) or any(not isinstance(c, str) for c in raw_common)
            or len(raw_common) != len(set(raw_common)) or not set(raw_common) <= eligible_ids
            or type(receipt_dict.get("common_completed_cohort_count")) is not int
            or receipt_dict["common_completed_cohort_count"] != len(raw_common)):
        raise CorruptedArtifactError("Invalid receipt common completed cohort")
    rec_predictions = receipt_dict.get("per_clip_predictions")
    if not isinstance(rec_predictions, dict) or set(rec_predictions) != set(plan_models):
        raise CorruptedArtifactError("Missing per-model receipt predictions")
    for model, clips in rec_predictions.items():
        if not isinstance(clips, dict) or set(clips) != eligible_ids:
            raise CorruptedArtifactError("Receipt predictions differ from frozen eligible cases")
        for cid in raw_common:
            pred = clips[cid]
            if (not isinstance(pred, dict) or pred.get("status") != "completed"
                    or type(pred.get("completed_reps")) is not int or pred["completed_reps"] < 0
                    or pred.get("error")):
                raise CorruptedArtifactError("Common cohort contains invalid execution predictions")
    validated_models, _, _ = verify_plan_inputs(plan_dict, p_plan)
    for path in validated_models.values():
        input_hashes[str(path)] = compute_file_sha256(path)
    rev_root_raw = plan_dict.get("review_root")
    if not rev_root_raw or not isinstance(rev_root_raw, str):
        raise CorruptedArtifactError("Plan is missing or has invalid 'review_root'")
    review_root = Path(rev_root_raw)
    if not review_root.is_absolute():
        review_root = (p_plan.parent / review_root).resolve()
    else:
        review_root = review_root.resolve()

    if review_root.exists():
        reject_linked_ancestors(review_root, label="Review root")
        if is_symlink(review_root):
            raise CaseCorruptedError(f"Review root cannot be a symlink: {review_root}")
        if not review_root.is_dir():
            raise NotADirectoryError(f"Review root is not a directory: {review_root}")

    # Output directory safety check
    validate_evaluation_output_dir(p_out, p_plan, exec_root, review_root if review_root.exists() else None)

    # 6. Models contract
    plan_models = plan_dict.get("models", {})
    if isinstance(plan_models, dict) and plan_models:
        model_names = sorted(list(plan_models.keys()))
    elif isinstance(receipt_dict.get("models"), dict) and receipt_dict["models"]:
        model_names = sorted(list(receipt_dict["models"].keys()))
    else:
        sub_dirs = [d.name for d in exec_root.iterdir() if d.is_dir() and not d.name.startswith(".")]
        model_names = sorted(sub_dirs) if sub_dirs else ["default"]

    for m in model_names:
        validate_model_name(m)

    # 7. Frozen Cases from Plan & Receipt
    # Strictly evaluate cases declared in plan.json and receipt. NEVER scan review_root.iterdir()!
    plan_eligible = plan_dict.get("eligible_cases", [])
    plan_secondary = plan_dict.get("secondary_cases", [])
    plan_excluded = plan_dict.get("excluded_cases", [])
    source_ann_hashes = plan_dict.get("source_annotation_hashes", {})

    cases_pool: Dict[str, Dict[str, Any]] = {}
    for c in plan_eligible:
        if isinstance(c, dict) and "case_id" in c:
            cases_pool[c["case_id"]] = {**c, "_cohort_origin": "primary_eligible"}
    for c in plan_secondary:
        if isinstance(c, dict) and "case_id" in c:
            if c["case_id"] in cases_pool:
                raise CorruptedArtifactError("Case appears in multiple frozen cohorts")
            cases_pool[c["case_id"]] = {**c, "_cohort_origin": "secondary_seen"}
    for c in plan_excluded:
        if isinstance(c, dict) and "case_id" in c:
            if c["case_id"] in cases_pool:
                raise CorruptedArtifactError("Case appears in multiple frozen cohorts")
            cases_pool[c["case_id"]] = {**c, "_cohort_origin": "excluded"}

    # Load and verify declared cases from disk
    if review_root.exists():
        store = ReviewStore(root=review_root)
        for cid in sorted(cases_pool.keys()):
            case_dir = review_root / cid
            reject_linked_ancestors(case_dir, label=f"Case directory '{cid}'")
            if is_symlink(case_dir):
                raise CaseCorruptedError(f"Case directory cannot be a symlink: {case_dir}")
            if not case_dir.exists():
                raise FileNotFoundError(f"Case directory not found: {case_dir}")
            if not case_dir.is_dir():
                raise ValueError(f"Case directory is not a directory: {case_dir}")

            case_json = case_dir / "case.json"
            reject_linked_ancestors(case_json, label=f"case.json for '{cid}'")
            if is_symlink(case_json):
                raise CaseCorruptedError(f"case.json cannot be a symlink: {case_json}")
            if not case_json.is_file():
                raise FileNotFoundError(f"case.json not found for case '{cid}'")

            curr_ann_sha = compute_file_sha256(case_json)
            input_hashes[str(case_json.resolve())] = curr_ann_sha

            if cid in source_ann_hashes:
                exp_ann_sha = source_ann_hashes[cid].strip().lower()
                if curr_ann_sha != exp_ann_sha:
                    raise HashMismatchError(
                        f"Annotation hash mismatch for case '{cid}': expected {exp_ann_sha}, got {curr_ann_sha}"
                    )

            if cid not in source_ann_hashes:
                raise CorruptedArtifactError(f"Missing frozen annotation identity for {cid}")
            case_data = store.load_persisted_case(cid, verify_raw_sha=True)
            raw_path = case_dir / ("video" + case_data["extension"])
            input_hashes[str(raw_path.resolve())] = compute_file_sha256(raw_path)
            origin = cases_pool[cid].get("_cohort_origin", "untracked")
            cases_pool[cid] = {**case_data, "_cohort_origin": origin}

    # Parse human events and check scorable support
    cases_events: Dict[str, List[HumanEvent]] = {}
    case_exhaustive_flags: Dict[str, bool] = {}

    for cid, case_data in sorted(cases_pool.items()):
        review = case_data.get("review", {})
        raw_events = review.get("events")
        human_count = review.get("human_count")

        try:
            parsed_events = parse_human_events(raw_events)
        except Exception as exc:
            raise CorruptedArtifactError(f"Corrupted events in case '{cid}': {exc}") from exc

        cases_events[cid] = parsed_events

        # Stored optional events do not certify exhaustive temporal annotation.
        # A matching total count does not prove all non-reps/uncertain intervals were labelled.
        case_exhaustive_flags[cid] = False

    # 8. Per-case Outputs Verification across Models
    # For each model and clip, summary.json and attempts.json MUST exist and be valid.
    # If missing or failed, exclude from common completed cohort.
    all_case_attempts: Dict[str, Dict[str, List[PredictedAttempt]]] = {m: {} for m in model_names}
    clip_valid_across_all_models: Dict[str, bool] = {cid: True for cid in cases_pool}
    clip_failure_reasons: Dict[str, List[str]] = {cid: [] for cid in cases_pool}
    incomplete_decoding_cases: List[Dict[str, Any]] = []

    for m in model_names:
        for cid in sorted(cases_pool.keys()):
            clip_dir = exec_root / m / cid
            if not clip_dir.is_dir() and len(model_names) == 1:
                alt_dir = exec_root / cid
                if alt_dir.is_dir():
                    clip_dir = alt_dir

            sum_path = clip_dir / "summary.json"
            att_path = clip_dir / "attempts.json"

            # Verify summary.json
            if not sum_path.is_file():
                clip_valid_across_all_models[cid] = False
                clip_failure_reasons[cid].append(f"Model '{m}' missing summary.json")
                continue

            reject_linked_ancestors(sum_path, label=f"Summary file ({m}/{cid})")
            if is_symlink(sum_path):
                raise ValueError(f"Summary file cannot be a symlink: {sum_path}")

            sum_sha = compute_file_sha256(sum_path)
            input_hashes[str(sum_path.resolve())] = sum_sha

            try:
                sum_dict = json.loads(
                    sum_path.read_text(encoding="utf-8"),
                    object_pairs_hook=detect_duplicate_pairs,
                    parse_constant=_reject_json_constant,
                )
            except Exception as exc:
                clip_valid_across_all_models[cid] = False
                clip_failure_reasons[cid].append(f"Model '{m}' malformed summary.json: {exc}")
                continue

            if not isinstance(sum_dict, dict):
                clip_valid_across_all_models[cid] = False
                clip_failure_reasons[cid].append(f"Model '{m}' summary must be an object")
                continue
            sum_status = str(sum_dict.get("status", "")).strip().lower()
            exit_code = sum_dict.get("exit_code")
            if sum_status != "completed" or (exit_code is not None and (type(exit_code) is not int or exit_code != 0)):
                clip_valid_across_all_models[cid] = False
                clip_failure_reasons[cid].append(f"Model '{m}' summary status is {sum_status!r} (exit {exit_code})")

            rec_clip_pred = rec_predictions.get(m, {}).get(cid, {})
            try:
                _validate_completed_summary(sum_dict, rec_clip_pred, cases_pool[cid], review_root,
                                            validated_models[m], plan_dict, receipt_dict)
            except (ValueError, TypeError, KeyError) as exc:
                clip_valid_across_all_models[cid] = False
                clip_failure_reasons[cid].append(f"Model '{m}' invalid completed summary: {exc}")
            if rec_clip_pred and rec_clip_pred.get("status") != "completed":
                clip_valid_across_all_models[cid] = False
                clip_failure_reasons[cid].append(f"Model '{m}' receipt per_clip_prediction status is {rec_clip_pred.get('status')!r}")

            src_v = sum_dict.get("source_video", {})
            fps = src_v.get("fps") or sum_dict.get("fps")
            decl_frames = src_v.get("declared_frame_count") or sum_dict.get("declared_frame_count")
            proc_frames = sum_dict.get("processed_frames")

            if (
                decl_frames is not None
                and proc_frames is not None
                and isinstance(decl_frames, int)
                and isinstance(proc_frames, int)
                and proc_frames < decl_frames
            ):
                clip_valid_across_all_models[cid] = False
                reason_str = f"Incomplete decoding: {proc_frames}/{decl_frames} frames"
                clip_failure_reasons[cid].append(reason_str)
                incomplete_decoding_cases.append({
                    "model": m,
                    "case_id": cid,
                    "processed_frames": proc_frames,
                    "declared_frame_count": decl_frames,
                    "reason": reason_str,
                })

            # Verify attempts.json
            if not att_path.is_file():
                clip_valid_across_all_models[cid] = False
                clip_failure_reasons[cid].append(f"Model '{m}' missing attempts.json")
                continue

            reject_linked_ancestors(att_path, label=f"Attempts file ({m}/{cid})")
            if is_symlink(att_path):
                raise ValueError(f"Attempts file cannot be a symlink: {att_path}")

            att_sha = compute_file_sha256(att_path)
            input_hashes[str(att_path.resolve())] = att_sha

            try:
                att_json = json.loads(
                    att_path.read_text(encoding="utf-8"),
                    object_pairs_hook=detect_duplicate_pairs,
                    parse_constant=_reject_json_constant,
                )
            except Exception as exc:
                raise CorruptedArtifactError(f"Failed to parse attempts.json for {m}/{cid}: {exc}") from exc

            try:
                parsed_att = parse_predicted_attempts(att_json, fps=fps)
            except Exception as exc:
                raise CorruptedArtifactError(f"Invalid attempts in {m}/{cid}: {exc}") from exc

            counted = sum(a.status == "counted" for a in parsed_att)
            if counted != sum_dict.get("completed_reps"):
                clip_valid_across_all_models[cid] = False
                clip_failure_reasons[cid].append(f"Model '{m}' attempt count differs from summary")
            all_case_attempts[m][cid] = parsed_att

    # 9. Cohort Segregation: Primary Eligible vs Secondary / Excluded
    receipt_common_cohort: Set[str] = set()
    raw_receipt_cohort = receipt_dict.get("common_completed_cohort")
    if isinstance(raw_receipt_cohort, list):
        receipt_common_cohort = set(str(c) for c in raw_receipt_cohort)

    primary_candidate_cases: List[str] = []
    secondary_candidate_cases: List[str] = []
    excluded_cases_diag: List[Dict[str, Any]] = []

    for cid, case_data in sorted(cases_pool.items()):
        review = case_data.get("review", {})
        group = case_data.get("group", "")
        scope = review.get("scope", "")
        human_count = review.get("human_count")
        raw_ack = review.get("raw_video_reviewed", False)
        reviewer = review.get("reviewer", "")
        seen = review.get("predictions_previously_seen")
        raw_events = review.get("events")

        reasons: List[str] = []
        if case_data.get("_cohort_origin") != "primary_eligible":
            reasons.append("Not in frozen primary eligible cohort")
        if group != "candidate":
            reasons.append(f"group '{group}' != 'candidate'")
        if scope != "in_scope":
            reasons.append(f"scope '{scope}' != 'in_scope'")
        if human_count is None:
            reasons.append("human_count is null")
        elif isinstance(human_count, bool) or not isinstance(human_count, int) or human_count < 0:
            reasons.append(f"human_count {human_count!r} not non-negative int")
        if raw_ack is not True:
            reasons.append("raw_video_reviewed is False")
        if not isinstance(reviewer, str) or not reviewer.strip():
            reasons.append("reviewer is empty")
        if seen is not False and seen is not True:
            reasons.append(f"predictions_previously_seen is invalid: {seen!r}")
        if raw_events is None:
            reasons.append("event annotations are unreviewed/missing (events is null)")
        elif len(raw_events) == 0 and isinstance(human_count, int) and human_count > 0:
            reasons.append(f"events list is empty but human_count is {human_count}")

        if not any(h.event_class != "ambiguous_or_unclear" and
                   (h.turnaround_time_s is not None or
                    (h.start_time_s is not None and h.end_time_s is not None))
                   for h in cases_events.get(cid, [])):
            reasons.append("No temporally scorable known human event")

        # Execution completion check
        if not clip_valid_across_all_models.get(cid, False):
            fail_reasons = clip_failure_reasons.get(cid, ["Failed execution"])
            reasons.append("; ".join(fail_reasons))
        if cid not in receipt_common_cohort:
            reasons.append("Excluded from execution receipt common completed cohort")

        if reasons:
            excluded_cases_diag.append({
                "case_id": cid,
                "reasons": "; ".join(reasons),
                "group": group,
                "scope": scope,
                "human_count": human_count,
                "predictions_previously_seen": seen,
                "event_count": len(cases_events.get(cid, [])),
            })
        elif seen is True:
            secondary_candidate_cases.append(cid)
        else:
            primary_candidate_cases.append(cid)

    final_primary_cases = sorted(primary_candidate_cases) if not receipt_failed else []

    # 10. Status Determination & Cohort Isolation
    # Receipt failure strictly prevents completed status
    if receipt_failed:
        cohort_status = "not_ready"
        not_ready_reason = f"Execution receipt reported failed status: {receipt_status!r}."
    elif len(final_primary_cases) == 0:
        cohort_status = "not_ready"
        not_ready_reason = (
            f"Primary cohort has 0 eligible fully decoded cases with event annotations and verified outputs. "
            f"Candidate pools: {len(primary_candidate_cases)} primary, "
            f"{len(secondary_candidate_cases)} secondary/seen, {len(excluded_cases_diag)} excluded."
        )
    else:
        cohort_status = "completed"
        not_ready_reason = None

    # Primary evaluated cases are strictly final_primary_cases
    primary_evaluated_cases = final_primary_cases

    # 11. Matching & Metrics Computation
    per_model_results: Dict[str, Any] = {}
    all_csv_rows: List[Dict[str, Any]] = []

    # Map attempt IDs to case IDs for scorable evaluation
    attempt_case_map: Dict[Union[int, str], str] = {}
    for m in model_names:
        for cid, atts in all_case_attempts.get(m, {}).items():
            for a in atts:
                attempt_case_map[a.attempt_id] = cid

    for m in model_names:
        m_matched_pairs: List[MatchedPair] = []
        m_unmatched_human: List[HumanEvent] = []
        m_unmatched_predicted: List[PredictedAttempt] = []
        per_case_matching: Dict[str, Any] = {}

        for cid in sorted(cases_pool.keys()):
            h_events = cases_events.get(cid, [])
            p_attempts = all_case_attempts.get(m, {}).get(cid, [])
            in_primary = (cid in final_primary_cases)

            # Deterministic greedy matching
            matched, un_h, un_p = match_events_1to1(
                human_events=h_events,
                predicted_attempts=p_attempts,
                tolerance_seconds=tol_s,
            )

            # Record CSV audit rows
            rows = build_matches_csv_rows(
                model_name=m,
                case_id=cid,
                matched_pairs=matched,
                unmatched_human=un_h,
                unmatched_predicted=un_p,
                in_primary_cohort=in_primary,
            )
            all_csv_rows.extend(rows)

            per_case_matching[cid] = {
                "in_primary_cohort": in_primary,
                "human_event_count": len(h_events),
                "predicted_attempt_count": len(p_attempts),
                "matched_count": len(matched),
                "unmatched_human_count": len(un_h),
                "unmatched_predicted_count": len(un_p),
                "matches": [
                    {
                        "human_event_id": mp.human.event_id,
                        "human_event_class": mp.human.event_class,
                        "predicted_attempt_id": mp.predicted.attempt_id,
                        "predicted_status": mp.predicted.status,
                        "classification_label": mp.classification_label,
                        "distance": round(mp.distance, 4) if math.isfinite(mp.distance) else None,
                        "delta_start_s": round(mp.delta_start_s, 4) if mp.delta_start_s is not None else None,
                        "delta_turnaround_s": round(mp.delta_turnaround_s, 4) if mp.delta_turnaround_s is not None else None,
                        "delta_end_s": round(mp.delta_end_s, 4) if mp.delta_end_s is not None else None,
                        "temporal_iou": round(mp.temporal_iou, 4) if mp.temporal_iou is not None else None,
                    }
                    for mp in matched
                ],
            }

            # Accumulate into primary cohort
            if cid in primary_evaluated_cases:
                m_matched_pairs.extend(matched)
                m_unmatched_human.extend(un_h)
                m_unmatched_predicted.extend(un_p)

        if cohort_status == "completed":
            m_metrics = compute_cohort_metrics(
                all_matched_pairs=m_matched_pairs,
                all_unmatched_human=m_unmatched_human,
                all_unmatched_predicted=m_unmatched_predicted,
                case_exhaustive_flags=case_exhaustive_flags,
                case_id_for_prediction=lambda att_id: attempt_case_map.get(att_id),
            )
        else:
            m_metrics = None

        per_model_results[m] = {
            "model_name": m,
            "cohort_status": cohort_status,
            "evaluated_cases_count": len(primary_evaluated_cases),
            "repetition_metrics": m_metrics["repetition_metrics"] if m_metrics else None,
            "boundary_errors": m_metrics["boundary_errors"] if m_metrics else None,
            "temporal_iou": m_metrics["temporal_iou"] if m_metrics else None,
            "movement_category_agreement": (
                m_metrics["movement_category_agreement"] if m_metrics else None
            ),
            "per_case_matching": per_case_matching,
        }

    # Primary overall metrics: from first model or strictly None if not ready
    primary_metrics = None
    if cohort_status == "completed" and model_names:
        first_m_res = per_model_results[model_names[0]]
        first_rep = first_m_res.get("repetition_metrics") or {}
        first_b_err = first_m_res.get("boundary_errors") or {}
        primary_metrics = {
            **first_rep,
            "repetition_metrics": first_rep,
            "boundary_errors": first_b_err,
            "temporal_iou": first_m_res.get("temporal_iou"),
            "movement_category_agreement": first_m_res.get("movement_category_agreement"),
            "mean_error_start_s": first_b_err.get("start", {}).get("mean_error_s"),
            "mean_error_turn_s": first_b_err.get("turnaround", {}).get("mean_error_s"),
            "mean_error_end_s": first_b_err.get("end", {}).get("mean_error_s"),
            "mae_start_s": first_b_err.get("start", {}).get("mae_s"),
            "mae_turn_s": first_b_err.get("turnaround", {}).get("mae_s"),
            "mae_end_s": first_b_err.get("end", {}).get("mae_s"),
        }

    # 12. Optional Exploratory Cohort (with --allow-unreviewed)
    # Strictly isolated from primary cohort! Primary metrics remain strictly null if primary is not_ready.
    exploratory_cohort_summary = None
    if allow_unreviewed:
        exploratory_cases = [
            cid for cid, evts in cases_events.items()
            if len(evts) > 0 and cid not in final_primary_cases and not receipt_failed
            and cid in receipt_common_cohort and clip_valid_across_all_models.get(cid, False)
        ]
        exploratory_status = "completed" if exploratory_cases else "not_ready"
        exploratory_results: Dict[str, Any] = {}

        for m in model_names:
            e_matched: List[MatchedPair] = []
            e_un_h: List[HumanEvent] = []
            e_un_p: List[PredictedAttempt] = []

            for cid in exploratory_cases:
                matched, un_h, un_p = match_events_1to1(
                    human_events=cases_events.get(cid, []),
                    predicted_attempts=all_case_attempts.get(m, {}).get(cid, []),
                    tolerance_seconds=tol_s,
                )
                e_matched.extend(matched)
                e_un_h.extend(un_h)
                e_un_p.extend(un_p)

            if exploratory_status == "completed":
                e_metrics = compute_cohort_metrics(
                    all_matched_pairs=e_matched,
                    all_unmatched_human=e_un_h,
                    all_unmatched_predicted=e_un_p,
                    case_exhaustive_flags=case_exhaustive_flags,
                    case_id_for_prediction=lambda att_id: attempt_case_map.get(att_id),
                )
            else:
                e_metrics = None

            exploratory_results[m] = {
                "cohort_status": exploratory_status,
                "evaluated_cases_count": len(exploratory_cases),
                "repetition_metrics": e_metrics["repetition_metrics"] if e_metrics else None,
                "boundary_errors": e_metrics["boundary_errors"] if e_metrics else None,
                "temporal_iou": e_metrics["temporal_iou"] if e_metrics else None,
                "movement_category_agreement": (
                    e_metrics["movement_category_agreement"] if e_metrics else None
                ),
            }

        exploratory_cohort_summary = {
            "cohort_status": exploratory_status,
            "evaluated_cases": exploratory_cases,
            "evaluated_cases_count": len(exploratory_cases),
            "per_model_results": exploratory_results,
        }

    limitations = {
        "prediction_integrity_note": "Input hashes identify consumed bytes. Upstream receipts do not sign/hash individual prediction files; this is not proof of untouched historical outputs.",
        "metric_scope": "Agreement conditional on scorable annotated events and matched movement categories; unmatched/unknown coverage is reported separately.",
        "gold_standard_claim": False,
        "gold_standard_note": (
            "Human-reviewed video labels reflect single-reviewer observations "
            "and are not claimed as gold-standard ground truth."
        ),
        "source_independence_claim": False,
        "source_independence_note": (
            "Source candidate videos are not claimed to be an independent test set; "
            "unknown person and source independence."
        ),
        "exercise_intent_claim": False,
        "exercise_intent_note": (
            "Detector reason codes and attempt boundaries are software-defined thresholds, "
            "not a validated human intent classifier."
        ),
        "incomplete_decoding_note": (
            "Incompletely decoded clips are explicitly flagged and excluded from primary "
            "comparisons; never imputed as zero."
        ),
    }

    report: Dict[str, Any] = {
        "report_version": REPORT_VERSION,
        "contract_version": CONTRACT_VERSION,
        "cohort_status": cohort_status,
        "not_ready_reason": not_ready_reason,
        "generated_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "plan_path": str(p_plan.resolve()),
        "plan_sha256": plan_dict.get("plan_sha256") or computed_plan_sha,
        "execution_path": str(exec_root.resolve()),
        "execution_receipt_sha256": receipt_sha,
        "review_root": str(review_root),
        "output_dir": str(p_out.resolve()),
        "configuration": {
            "tolerance_seconds": tol_s,
            "allow_unreviewed": allow_unreviewed,
            "weights": {"turnaround": 1.0, "start": 1.0, "end": 1.0},
        },
        "cohort_summary": {
            "primary_eligible_cases_count": len(primary_candidate_cases),
            "primary_cases": final_primary_cases,
            "secondary_cases_count": len(secondary_candidate_cases),
            "secondary_cases": secondary_candidate_cases,
            "excluded_cases_count": len(excluded_cases_diag),
            "excluded_cases": excluded_cases_diag,
            "incomplete_decoding_cases": incomplete_decoding_cases,
            "evaluated_cases": primary_evaluated_cases,
        },
        "incomplete_cases": incomplete_decoding_cases,
        "excluded_from_common_cohort": [c["case_id"] for c in excluded_cases_diag],
        "models": model_names,
        "per_model_results": per_model_results,
        "repetition_metrics": primary_metrics.get("repetition_metrics") if primary_metrics else None,
        "movement_category_agreement": primary_metrics.get("movement_category_agreement") if primary_metrics else None,
        "boundary_errors": primary_metrics.get("boundary_errors") if primary_metrics else None,
        "temporal_iou": primary_metrics.get("temporal_iou") if primary_metrics else None,
        "metrics": primary_metrics,
        "limitations_and_disclaimers": limitations,
        "input_hashes": input_hashes,
    }

    if exploratory_cohort_summary is not None:
        report["exploratory_cohort"] = exploratory_cohort_summary

    # 13. Staged Atomic Publication (No-Clobber)
    stage_dir: Optional[Path] = None
    try:
        p_out.parent.mkdir(parents=True, exist_ok=True)
        reject_linked_ancestors(p_out.parent, label="Output parent directory")
        stage_dir = Path(
            tempfile.mkdtemp(prefix=f".{p_out.name}.temporal-", dir=p_out.parent.resolve())
        ).resolve()

        # Write temporal_evaluation_report.json
        report_file = stage_dir / "temporal_evaluation_report.json"
        with open(report_file, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, allow_nan=False)
            f.flush()
            os.fsync(f.fileno())

        # Write temporal_matches.csv
        csv_file = stage_dir / "temporal_matches.csv"
        with open(csv_file, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES)
            writer.writeheader()
            for r in all_csv_rows:
                writer.writerow(r)
            f.flush()
            os.fsync(f.fileno())

        # Write temporal_summary.txt
        summary_text = format_temporal_summary_text(report)
        txt_file = stage_dir / "temporal_summary.txt"
        with open(txt_file, "w", encoding="utf-8") as f:
            f.write(summary_text + "\n")
            f.flush()
            os.fsync(f.fileno())

        if _seam_hook is not None:
            _seam_hook()

        # Verify the exact consumed bytes and source truth again after optional hook.
        verify_plan_inputs(plan_dict, p_plan)
        for source_path, expected_hash in input_hashes.items():
            path = Path(source_path)
            reject_linked_ancestors(path, label="Frozen temporal input")
            if not path.is_file() or compute_file_sha256(path) != expected_hash:
                raise HashMismatchError(f"Temporal input changed before publication: {path}")

        # Final no-clobber check before atomic swap
        validate_evaluation_output_dir(p_out, p_plan, exec_root, review_root if review_root.exists() else None)

        if p_out.exists():
            if any(p_out.iterdir()):
                raise FileExistsError(
                    f"Refusing to overwrite existing non-empty output directory: {p_out}"
                )
            p_out.rmdir()

        stage_dir.rename(p_out)
        stage_dir = None

    finally:
        if stage_dir is not None and stage_dir.exists():
            shutil.rmtree(stage_dir, ignore_errors=True)

    return report


# ==============================================================================
# CLI Entrypoint
# ==============================================================================


def build_parser() -> argparse.ArgumentParser:
    """Build CLI argument parser for temporal evaluation."""
    parser = argparse.ArgumentParser(
        prog="python -m gym_analyser.temporal_evaluation",
        description="Temporal repetition event evaluation engine (Task 45).",
    )
    parser.add_argument(
        "--plan",
        type=Path,
        required=True,
        help="Path to evaluation plan JSON or directory containing plan.json.",
    )
    parser.add_argument(
        "--execution",
        type=Path,
        required=True,
        help="Path to execution receipt JSON or directory containing model predictions.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Fresh destination directory for temporal evaluation report and CSV.",
    )
    parser.add_argument(
        "--tolerance-seconds",
        type=float,
        default=DEFAULT_TOLERANCE_SECONDS,
        help=f"Matching proximity tolerance in seconds (default {DEFAULT_TOLERANCE_SECONDS}s, >= {MIN_TOLERANCE_SECONDS}s).",
    )
    parser.add_argument(
        "--allow-unreviewed",
        action="store_true",
        default=False,
        help="Allow evaluating cases with annotations separately as exploratory cohort.",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI execution entrypoint."""
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        report = evaluate_temporal_events(
            plan_path_or_dir=args.plan,
            execution_dir=args.execution,
            output_dir=args.output_dir,
            tolerance_seconds=args.tolerance_seconds,
            allow_unreviewed=args.allow_unreviewed,
        )
        print(format_temporal_summary_text(report))
        print(f"\nReport written to: {args.output_dir / 'temporal_evaluation_report.json'}")
        print(f"Matches CSV written to: {args.output_dir / 'temporal_matches.csv'}")
        print(f"Summary text written to: {args.output_dir / 'temporal_summary.txt'}")
        return 0
    except Exception as exc:
        sys.stderr.write(f"Temporal evaluation error: {exc}\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
