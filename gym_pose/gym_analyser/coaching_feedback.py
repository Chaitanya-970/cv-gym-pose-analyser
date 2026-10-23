# -*- coding: utf-8 -*-
"""Actionable, evidence-backed coaching feedback generator for exercise analysis.

Pure Python standard-library module consuming upstream analysis artifacts
(summary, attempts, posture_summary, feedback_observations, feedback_phases)
to deliver structured, non-normative kinematic observations, recording setup
guidance, within-session progression across comparable repetitions, evidence links,
and repetition review cues matching FEEDBACK_CONTRACT_38.md and POSTURE_CRITIQUE_CONTRACT_59.md.

Strict Epistemological Boundaries:
- Pure Python standard library with zero ML/external dependencies.
- Consumes existing output dicts without recountering or redefining repetition phases.
- Descriptive only: NEVER diagnoses spinal rounding, knee valgus/varus, pathology, or injury risk.
- NEVER claims universal absolute 'safe' angle thresholds or issues technique grades (good/bad/pass/fail).
- Two repetitions are comparable ONLY when track ID is identical and non-None,
  tracking side is identical and in {"left", "right"}, coverage >= 50.0%, and
  feature measurements are valid and finite.
- Separates verifiable observations from suggestions and hypotheses.
- Explicitly reports validation status as UNVALIDATED until genuine human review.
"""

import math
from gym_analyser.angle_policy import (DEPTH_REVIEW_DIFFERENCE_DEG, TORSO_REVIEW_DIFFERENCE_DEG, TURNAROUND_REVIEW_DIFFERENCE_DEG)
from numbers import Real
import statistics
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple, Union

CSV_PRECISION_DECIMALS = 2

# Engineering attention filters, not research-derived fault limits or measurement error bounds.
# See ANGLE_BASELINES.md for the research audit and angle convention.
MIN_COMPARABLE_COVERAGE_PCT = 50.0
DEPTH_CONSISTENCY_VARIANCE_DEG = DEPTH_REVIEW_DIFFERENCE_DEG
RAPID_DESCENT_THRESHOLD_S = 0.60
ASYMMETRIC_PACING_RATIO = 2.5
MARKED_TEMPO_DRIFT_THRESHOLD_S = 0.25
NOTICEABLE_TORSO_DEVIATION_DEG = TORSO_REVIEW_DIFFERENCE_DEG

CLINICAL_DISCLAIMER = (
    "These 2D kinematic measurements are descriptive planar observations, "
    "not clinical diagnostic assessments or technique grades."
)
HUMAN_VALIDATION_STATUS = (
    "UNVALIDATED by human expert review on diverse unseen subjects."
)

ATTEMPT_REASON_EXPLANATIONS: Mapping[str, str] = {
    "completed_cycle": "A complete cycle met the configured counting thresholds.",
    "depth_threshold_not_reached": (
        "The observed knee angle did not reach the configured depth threshold before returning to standing."
    ),
    "depth_not_sustained": (
        "The depth threshold was crossed briefly, but not for enough consecutive valid frames."
    ),
    "confidence_loss": "Joint confidence remained too low to confirm one continuous cycle.",
    "ambiguous_subjects": "Multiple people made the subject ambiguous too long to confirm this cycle.",
    "tracking_gap": "Tracking or joint measurements were interrupted too long to confirm one continuous cycle.",
    "side_changed": "The observed body side changed before this cycle could be confirmed.",
    "subject_changed": "The observed subject changed before this cycle could be confirmed.",
    "video_ended_mid_attempt": "The video ended before a complete return to standing was confirmed.",
    "processing_failed": "Processing stopped before this attempt could be confirmed.",
}


def _parse_float(val: Any) -> Optional[float]:
    """Parse a float strictly preserving 0.0 as a valid number; booleans return None."""
    if val is None or isinstance(val, bool):
        return None
    try:
        f = float(str(val).strip()) if isinstance(val, str) else float(val)
        return f if math.isfinite(f) else None
    except (ValueError, TypeError, OverflowError):
        return None


def _parse_int(val: Any) -> Optional[int]:
    """Parse an integer value strictly rejecting booleans and non-integral floats."""
    if val is None or isinstance(val, bool):
        return None
    if isinstance(val, int):
        return val
    try:
        f = float(str(val).strip()) if isinstance(val, str) else float(val)
        if math.isfinite(f) and f.is_integer():
            return int(f)
    except (ValueError, TypeError, OverflowError):
        return None
    return None


parse_float = _parse_float
parse_int = _parse_int


def _validate_angle(val: Any, min_deg: float = 0.0, max_deg: float = 180.0) -> Optional[float]:
    """Validate that an angle is finite and within [min_deg, max_deg] degrees."""
    f = _parse_float(val)
    if f is not None and min_deg <= f <= max_deg:
        return f
    return None


def _validate_pct(val: Any) -> Optional[float]:
    """Validate that a percentage is finite and within [0.0, 100.0] percent."""
    f = _parse_float(val)
    if f is not None and 0.0 <= f <= 100.0:
        return f
    return None


def _classify_uncounted_hypothesis(
    reason: str,
    attempt: Mapping[str, Any],
    has_counted_before: bool,
    has_counted_after: bool,
) -> str:
    """Classify uncounted movement context into one of the contract hypotheses.

    Allowed hypotheses:
    - "possible_setup_or_weight_return"
    - "shallow_depth"
    - "tracking_lost"
    - "unfinished_cycle"
    - "ambiguous"
    - "unknown"

    Strict Guardrails:
    - Pure standard library logic without ML models.
    - No hardcoded angle threshold heuristics (e.g. 135° knee angle).
    - No loose substring classification.
    - Does not convert 'depth_not_sustained' or ambiguous reasons into proven
      shallow depth without upstream justification.
    - Returns 'ambiguous' or 'unknown' when intent cannot be established.
    """
    # 1. Prioritize upstream explicit hypothesis if already provided and valid
    upstream_hyp = str(
        attempt.get("context_hypothesis")
        or attempt.get("hypothesis")
        or ""
    ).strip().lower()
    valid_hypotheses = {
        "possible_setup_or_weight_return",
        "shallow_depth",
        "tracking_lost",
        "unfinished_cycle",
        "ambiguous",
        "unknown",
    }
    if upstream_hyp in valid_hypotheses:
        return upstream_hyp

    r = (reason or "").strip().lower()
    if not r:
        return "unknown"

    # Tracking lost reasons
    if r in (
        "tracking_gap",
        "confidence_loss",
        "ambiguous_subjects",
        "side_changed",
        "subject_changed",
        "tracking_lost",
    ):
        return "tracking_lost"

    # Unfinished cycle reasons
    if r in (
        "video_ended_mid_attempt",
        "processing_failed",
        "unfinished_cycle",
        "incomplete_cycle",
    ):
        return "unfinished_cycle"

    # Depth threshold not reached
    if r == "depth_threshold_not_reached":
        if has_counted_before and has_counted_after:
            # A detector fragment between cycles may belong to a nearby squat.
            # Temporal placement alone cannot establish a separate shallow attempt.
            return "ambiguous"
        elif has_counted_before or has_counted_after:
            # Movement occurred before the first counted rep (setup) or after the last (return)
            return "possible_setup_or_weight_return"
        else:
            # No counted repetitions established in session: intent is ambiguous
            return "ambiguous"

    # Depth not sustained: crossed depth threshold briefly but not sustained.
    # Without upstream justification, this cannot be declared proven shallow depth.
    if r == "depth_not_sustained":
        return "ambiguous"

    if r in ("possible_setup_or_weight_return", "shallow_depth"):
        return r

    if r in ("ambiguous", "uncertain"):
        return "ambiguous"

    return "unknown"


class RepetitionEvidence:
    """Normalized repetition evidence container for critique analysis."""

    def __init__(
        self,
        rep_index: int,
        attempt_id: Optional[int],
        track_id: Optional[int],
        side: Optional[str],
        coverage_pct: Optional[float],
        has_verified_phase: bool,
        start_time_s: Optional[float],
        inflection_time_s: Optional[float],
        end_time_s: Optional[float],
        min_knee_angle_deg: Optional[float],
        median_torso_inclination_deg: Optional[float],
        median_hip_angle_deg: Optional[float],
        descent_dur_s: Optional[float] = None,
        return_dur_s: Optional[float] = None,
        total_dur_s: Optional[float] = None,
        relative_depth_raw: Optional[Mapping[str, Any]] = None,
    ):
        self.rep_index = _parse_int(rep_index)
        self.attempt_id = _parse_int(attempt_id)
        self.track_id = _parse_int(track_id)
        self.side = side.strip().lower() if isinstance(side, str) else None
        self.coverage_pct = _validate_pct(coverage_pct)
        self.has_verified_phase = has_verified_phase is True
        self.start_time_s = _parse_float(start_time_s)
        self.inflection_time_s = _parse_float(inflection_time_s)
        self.end_time_s = _parse_float(end_time_s)
        self.min_knee_angle_deg = _validate_angle(min_knee_angle_deg)
        self.median_torso_inclination_deg = _validate_angle(median_torso_inclination_deg)
        self.median_hip_angle_deg = _validate_angle(median_hip_angle_deg)
        self.descent_dur_s = _parse_float(descent_dur_s)
        self.return_dur_s = _parse_float(return_dur_s)
        self.total_dur_s = _parse_float(total_dur_s)
        self.relative_depth_raw = relative_depth_raw



def check_repetition_comparability(
    rep_a: RepetitionEvidence,
    rep_b: RepetitionEvidence,
    feature: str = "general",
) -> Tuple[bool, Optional[str]]:
    """Determine whether two repetitions satisfy the strict comparability contract.

    Contract 59 Section 3 prerequisites:
    1. Identical track ID: track_id is not None and track_id(A) == track_id(B).
    2. Identical tracking side: side is in {"left", "right"} and side(A) == side(B).
    3. Adequate measurement coverage: coverage_pct >= 50.0% for both repetitions.
    4. Valid, finite numerical measurements for the feature being compared.
    """
    if rep_a.track_id is None or rep_b.track_id is None:
        return False, "track ID is unresolved"
    if rep_a.track_id != rep_b.track_id:
        return False, "differing track IDs"

    if rep_a.side not in ("left", "right") or rep_b.side not in ("left", "right"):
        return False, "tracking side is unresolved or unconfirmed"
    if rep_a.side != rep_b.side:
        return False, "suppressed due to body side switch"

    if rep_a.coverage_pct is None or rep_a.coverage_pct < MIN_COMPARABLE_COVERAGE_PCT:
        return False, f"cycle {rep_a.attempt_id or rep_a.rep_index} coverage below operational {MIN_COMPARABLE_COVERAGE_PCT:.1f}% threshold"
    if rep_b.coverage_pct is None or rep_b.coverage_pct < MIN_COMPARABLE_COVERAGE_PCT:
        return False, f"cycle {rep_b.attempt_id or rep_b.rep_index} coverage below operational {MIN_COMPARABLE_COVERAGE_PCT:.1f}% threshold"

    if feature == "depth":
        if not rep_a.has_verified_phase or rep_a.min_knee_angle_deg is None:
            return False, f"cycle {rep_a.attempt_id or rep_a.rep_index} phase turnaround depth is unconfirmed"
        if not rep_b.has_verified_phase or rep_b.min_knee_angle_deg is None:
            return False, f"cycle {rep_b.attempt_id or rep_b.rep_index} phase turnaround depth is unconfirmed"

    elif feature == "timing":
        for candidate in (rep_a, rep_b):
            if (not candidate.has_verified_phase
                    or candidate.descent_dur_s is None or candidate.descent_dur_s <= 0
                    or candidate.return_dur_s is None or candidate.return_dur_s <= 0):
                return False, f"cycle {candidate.attempt_id or candidate.rep_index} phase timing is unresolved"

    elif feature == "torso":
        if rep_a.median_torso_inclination_deg is None:
            return False, f"cycle {rep_a.attempt_id or rep_a.rep_index} torso measurement is unavailable"
        if rep_b.median_torso_inclination_deg is None:
            return False, f"cycle {rep_b.attempt_id or rep_b.rep_index} torso measurement is unavailable"

    return True, None


def get_comparable_repetitions(
    target_rep: RepetitionEvidence,
    all_reps: Sequence[RepetitionEvidence],
    feature: str = "general",
) -> List[RepetitionEvidence]:
    """Return all repetitions in the session that are strictly comparable to target_rep."""
    comparable: List[RepetitionEvidence] = []
    for rep in all_reps:
        is_comp, _ = check_repetition_comparability(target_rep, rep, feature=feature)
        if is_comp:
            comparable.append(rep)
    return comparable


def determine_unavailability_reason(
    target_rep: RepetitionEvidence,
    all_reps: Sequence[RepetitionEvidence],
    view_inconsistent: bool = False,
    feature: str = "general",
) -> str:
    """Produce precise, defensive reason when cross-attempt progression is unavailable."""
    if view_inconsistent:
        return "suppressed due to body side switch"

    tracks_seen = {r.track_id for r in all_reps if r.track_id is not None}
    if len(tracks_seen) > 1:
        return "suppressed across differing tracks or body sides"

    sides_seen = {r.side for r in all_reps if r.side in ("left", "right")}
    if len(sides_seen) > 1:
        return "suppressed due to body side switch"

    if target_rep.coverage_pct is None:
        return "repetition frame coverage is unknown or unmeasured"
    if target_rep.coverage_pct < MIN_COMPARABLE_COVERAGE_PCT:
        return f"frame coverage {target_rep.coverage_pct:.1f}% below operational {MIN_COMPARABLE_COVERAGE_PCT:.1f}% threshold"

    # Check if there are other reps on the same track and side
    same_track_and_side = [
        r for r in all_reps
        if r.track_id == target_rep.track_id and r.side == target_rep.side
    ]
    if len(same_track_and_side) < 2:
        return "requires at least two comparable repetitions on the same tracked side"

    return "requires at least two comparable repetitions with adequate measurement coverage"


def build_depth_observation(
    rep: RepetitionEvidence,
    all_reps: Sequence[RepetitionEvidence],
    configured_down_angle: Optional[float],
    summary_status: str,
    feedback_phases_available: bool,
    phase_ev_present: bool,
    view_inconsistent: bool = False,
) -> Dict[str, Any]:
    """Build OBS-DEPTH-01 observation with within-session progression and evidence links."""
    if rep.has_verified_phase and rep.min_knee_angle_deg is not None and rep.inflection_time_s is not None:
        min_knee = rep.min_knee_angle_deg
        depth_diff = None if configured_down_angle is None else round(min_knee - configured_down_angle, CSV_PRECISION_DECIMALS)
        if depth_diff is None:
            base_msg = (f"Observed minimum knee angle reached {min_knee:.2f}°. "
                        "The configured depth threshold is unavailable; no threshold comparison is made.")
        elif depth_diff <= 0:
            base_msg = (
                f"Observed minimum knee angle reached {min_knee:.2f}° (turnaround point), "
                f"crossing the configured {configured_down_angle:.1f}° depth threshold."
            )
        else:
            base_msg = (
                f"Observed minimum knee angle reached {min_knee:.2f}°, "
                f"which was {depth_diff:.2f}° shallower than the configured {configured_down_angle:.1f}° threshold."
            )

        # Within-session progression for comparable repetitions
        comp_reps = get_comparable_repetitions(rep, all_reps, feature="depth")
        prior_comp = [r for r in comp_reps if r.rep_index < rep.rep_index]

        if len(comp_reps) >= 2:
            if prior_comp:
                r_prev = prior_comp[-1]
                r_base = comp_reps[0]

                prev_knee = r_prev.min_knee_angle_deg or 0.0
                delta_prev = round(min_knee - prev_knee, CSV_PRECISION_DECIMALS)
                prev_id = r_prev.attempt_id if r_prev.attempt_id is not None else r_prev.rep_index

                base_knee = r_base.min_knee_angle_deg or 0.0
                delta_base = round(min_knee - base_knee, CSV_PRECISION_DECIMALS)
                base_id = r_base.attempt_id if r_base.attempt_id is not None else r_base.rep_index

                if r_prev.rep_index == r_base.rep_index:
                    # Exactly one prior comparable rep (e.g. rep 2 comparing to rep 1)
                    if delta_base > 0:
                        prog_str = (
                            f" Turnaround was relatively shallower (+{delta_base:.2f}°) than cycle {base_id} "
                            f"({base_knee:.2f}°); temporal review cue, not form critique."
                        )
                    elif delta_base < 0:
                        prog_str = (
                            f" Turnaround was relatively deeper (-{abs(delta_base):.2f}°) than cycle {base_id} "
                            f"({base_knee:.2f}°); temporal review cue, not form critique."
                        )
                    else:
                        prog_str = (
                            f" Turnaround matched turnaround depth of cycle {base_id} "
                            f"({base_knee:.2f}°); temporal review cue, not form critique."
                        )
                    depth_msg = base_msg + prog_str
                else:
                    # Multiple prior comparable reps: cite both immediate predecessor and baseline
                    if delta_prev > 0:
                        prev_desc = f"relatively shallower (+{delta_prev:.2f}°) than preceding cycle {prev_id} ({prev_knee:.2f}°)"
                    elif delta_prev < 0:
                        prev_desc = f"relatively deeper (-{abs(delta_prev):.2f}°) than preceding cycle {prev_id} ({prev_knee:.2f}°)"
                    else:
                        prev_desc = f"matched depth (0.00°) of preceding cycle {prev_id} ({prev_knee:.2f}°)"

                    if delta_base > 0:
                        base_desc = f"relatively shallower (+{delta_base:.2f}°) than baseline cycle {base_id} ({base_knee:.2f}°)"
                    elif delta_base < 0:
                        base_desc = f"relatively deeper (-{abs(delta_base):.2f}°) than baseline cycle {base_id} ({base_knee:.2f}°)"
                    else:
                        base_desc = f"matched baseline cycle {base_id} ({base_knee:.2f}°)"

                    depth_msg = f"{base_msg} Turnaround was {prev_desc} and {base_desc}; temporal review cue, not form critique."
            else:
                # First comparable rep (session baseline) in a multi-rep comparable session
                depth_msg = base_msg
        else:
            # Fewer than 2 comparable repetitions in the entire session
            unavail_reason = determine_unavailability_reason(rep, all_reps, view_inconsistent=view_inconsistent, feature="depth")
            depth_msg = f"{base_msg} Within-session depth progression comparison unavailable: {unavail_reason}."

        return {
            "rule_id": "OBS-DEPTH-01",
            "category": "depth",
            "message": depth_msg,
            "source_id": "phase_feedback",
            "confidence_level": "observed",
        }

    else:
        # Turnaround depth unavailable
        if summary_status in ("failed", "incomplete"):
            depth_reason = f"pipeline execution status is '{summary_status}'"
            src_id = "summary"
        elif rep.coverage_pct is None:
            depth_reason = "repetition frame coverage is unknown or unmeasured"
            src_id = "session_feedback"
        elif rep.coverage_pct < MIN_COMPARABLE_COVERAGE_PCT:
            depth_reason = f"insufficient valid frame coverage ({rep.coverage_pct:.1f}% < {MIN_COMPARABLE_COVERAGE_PCT:.1f}%)"
            src_id = "session_feedback"
        elif not feedback_phases_available or not phase_ev_present:
            depth_reason = "verified phase inflection evidence is missing from upstream telemetry"
            src_id = "phase_feedback"
        else:
            depth_reason = "verified phase inflection evidence is insufficient (sample count < 3 or unconfirmed turnaround)"
            src_id = "phase_feedback"

        return {
            "rule_id": "OBS-DEPTH-01",
            "category": "depth",
            "message": f"Turnaround depth observation unavailable: {depth_reason}.",
            "source_id": src_id,
            "confidence_level": "unavailable",
        }


def build_pacing_observation(
    rep: RepetitionEvidence,
    all_reps: Sequence[RepetitionEvidence],
    view_inconsistent: bool = False,
) -> Dict[str, Any]:
    """Build OBS-PACING-01 observation with within-session progression and evidence links."""
    has_valid_timing = (
        rep.has_verified_phase
        and rep.start_time_s is not None
        and rep.inflection_time_s is not None
        and rep.end_time_s is not None
        and (rep.start_time_s <= rep.inflection_time_s <= rep.end_time_s)
    )

    if has_valid_timing:
        descent_dur = rep.descent_dur_s if rep.descent_dur_s is not None else round(rep.inflection_time_s - rep.start_time_s, CSV_PRECISION_DECIMALS)
        return_dur = rep.return_dur_s if rep.return_dur_s is not None else round(rep.end_time_s - rep.inflection_time_s, CSV_PRECISION_DECIMALS)
        total_dur = rep.total_dur_s if rep.total_dur_s is not None else round(rep.end_time_s - rep.start_time_s, CSV_PRECISION_DECIMALS)

        base_msg = (
            f"Descent duration was {descent_dur:.2f}s (turnaround at {rep.inflection_time_s:.2f}s) "
            f"and return duration was {return_dur:.2f}s (total repetition duration: {total_dur:.2f}s)."
        )

        # Within-session pacing progression across comparable repetitions
        comp_reps = get_comparable_repetitions(rep, all_reps, feature="timing")
        prior_comp = [r for r in comp_reps if r.rep_index < rep.rep_index]

        if len(comp_reps) >= 2:
            if prior_comp:
                r_prev = prior_comp[-1]
                prev_descent = r_prev.descent_dur_s if r_prev.descent_dur_s is not None else (
                    round(r_prev.inflection_time_s - r_prev.start_time_s, CSV_PRECISION_DECIMALS)
                    if (r_prev.inflection_time_s is not None and r_prev.start_time_s is not None) else None
                )
                prev_id = r_prev.attempt_id if r_prev.attempt_id is not None else r_prev.rep_index

                if prev_descent is not None:
                    descent_shift = round(descent_dur - prev_descent, CSV_PRECISION_DECIMALS)
                    if descent_shift > 0.05:
                        tempo_desc = f"descent tempo deceleration (+{descent_shift:.2f}s longer descent than cycle {prev_id})"
                    elif descent_shift < -0.05:
                        tempo_desc = f"descent tempo acceleration ({descent_shift:.2f}s faster descent than cycle {prev_id})"
                    else:
                        tempo_desc = f"consistent descent tempo (shift of {descent_shift:+.2f}s compared to cycle {prev_id})"

                    pacing_msg = f"{base_msg} Pacing progression: {tempo_desc}."
                else:
                    pacing_msg = base_msg
            else:
                # First comparable rep establishes session pacing baseline
                pacing_msg = base_msg
        else:
            unavail_reason = determine_unavailability_reason(rep, all_reps, view_inconsistent=view_inconsistent, feature="timing")
            pacing_msg = f"{base_msg} Within-session pacing progression unavailable: {unavail_reason}."

        return {
            "rule_id": "OBS-PACING-01",
            "category": "timing",
            "message": pacing_msg,
            "source_id": "phase_feedback",
            "confidence_level": "observed",
        }

    else:
        # Check chronological error vs unresolved turnaround
        if (
            rep.start_time_s is not None
            and rep.inflection_time_s is not None
            and rep.end_time_s is not None
            and not (rep.start_time_s <= rep.inflection_time_s <= rep.end_time_s)
        ):
            msg = "Phase timing observation unavailable: timestamps were out of chronological sequence."
        else:
            msg = "Phase timing observation unavailable: turnaround inflection point was not reliably resolved."

        return {
            "rule_id": "OBS-PACING-01",
            "category": "timing",
            "message": msg,
            "source_id": "phase_feedback",
            "confidence_level": "unavailable",
        }


def build_torso_observation(
    rep: RepetitionEvidence,
    all_reps: Sequence[RepetitionEvidence],
    group_medians: Mapping[Tuple[int, str], float],
    view_inconsistent: bool = False,
) -> Dict[str, Any]:
    """Build OBS-TORSO-VAR-01 observation with session group median and prior rep comparison."""
    torso_med = rep.median_torso_inclination_deg
    rep_cov_pct = rep.coverage_pct

    tid_val = rep.track_id
    side_val = rep.side
    grp_key = (tid_val, side_val) if (tid_val is not None and side_val in ("left", "right")) else None

    # Only eligible, identified repetitions may contribute to the session statistic.
    comp_torso_reps = get_comparable_repetitions(rep, all_reps, feature="torso")
    reps_on_same_track_and_side = len(comp_torso_reps)

    if torso_med is not None and rep_cov_pct is not None and rep_cov_pct >= MIN_COMPARABLE_COVERAGE_PCT:
        if view_inconsistent or reps_on_same_track_and_side < 2:
            reason_msg = (
                "Cross-attempt consistency comparisons are suppressed across differing tracks or body sides."
                if view_inconsistent
                else "Within-session consistency comparisons require at least 2 counted repetitions on the same track and side."
            )
            return {
                "rule_id": "OBS-TORSO-VAR-01",
                "category": "posture_consistency",
                "message": f"Median torso inclination was {torso_med:.2f}° from vertical. {reason_msg}",
                "source_id": "feedback_observations",
                "confidence_level": "unavailable",
            }

        # Recompute from the eligible group; upstream aggregates may include
        # sparse or unidentified repetitions that this contract must exclude.
        g_med = round(statistics.median(
            r.median_torso_inclination_deg for r in comp_torso_reps
        ), CSV_PRECISION_DECIMALS) if len(comp_torso_reps) >= 2 else None

        if g_med is not None:
            delta = round(torso_med - g_med, CSV_PRECISION_DECIMALS)
            sign_str = f"+{delta:.2f}°" if delta >= 0 else f"{delta:.2f}°"

            # Check progression relative to prior comparable rep
            comp_reps = get_comparable_repetitions(rep, all_reps, feature="torso")
            prior_comp = [r for r in comp_reps if r.rep_index < rep.rep_index]

            if prior_comp:
                r_prev = prior_comp[-1]
                prev_torso = r_prev.median_torso_inclination_deg
                prev_id = r_prev.attempt_id if r_prev.attempt_id is not None else r_prev.rep_index

                if prev_torso is not None:
                    delta_prev = round(torso_med - prev_torso, CSV_PRECISION_DECIMALS)
                    sign_prev = f"+{delta_prev:.2f}°" if delta_prev >= 0 else f"{delta_prev:.2f}°"
                    comparison_details = (
                        f"session group median {g_med:.2f}°, deviation {sign_str}; "
                        f"shift of {sign_prev} relative to cycle {prev_id}"
                    )
                else:
                    comparison_details = f"session group median {g_med:.2f}°, deviation {sign_str}"
            else:
                comparison_details = f"session group median {g_med:.2f}°, deviation {sign_str}"

            return {
                "rule_id": "OBS-TORSO-VAR-01",
                "category": "posture_consistency",
                "message": (
                    f"Median torso inclination was {torso_med:.2f}° from vertical "
                    f"({comparison_details}). "
                    "Descriptive session comparison, not a technique grade or clinical assessment."
                ),
                "source_id": "feedback_observations",
                "confidence_level": "observed",
            }
        else:
            return {
                "rule_id": "OBS-TORSO-VAR-01",
                "category": "posture_consistency",
                "message": (
                    f"Median torso inclination was {torso_med:.2f}° from vertical across "
                    f"{rep_cov_pct:.1f}% coverage. Within-session group comparison unavailable."
                ),
                "source_id": "feedback_observations",
                "confidence_level": "unavailable",
            }

    else:
        if rep_cov_pct is None:
            cov_reason = "repetition frame coverage is unknown or unmeasured"
        elif rep_cov_pct < MIN_COMPARABLE_COVERAGE_PCT:
            cov_reason = f"frame coverage {rep_cov_pct:.1f}% below operational {MIN_COMPARABLE_COVERAGE_PCT:.1f}% threshold"
        else:
            cov_reason = "torso inclination measurement unavailable"

        return {
            "rule_id": "OBS-TORSO-VAR-01",
            "category": "posture_consistency",
            "message": f"Torso inclination observation unavailable: {cov_reason}.",
            "source_id": "feedback_observations",
            "confidence_level": "unavailable",
        }


def build_evidence_linked_suggestions(
    rep: RepetitionEvidence,
    all_reps: Sequence[RepetitionEvidence],
    group_medians: Mapping[Tuple[int, str], float],
) -> List[Dict[str, Any]]:
    """Build evidence-linked suggestion dictionaries for a repetition."""
    suggestions: List[Dict[str, Any]] = []

    # 1. SUG-REC-01: Recording setup guidance if coverage is below 50.0%
    if rep.coverage_pct is not None and rep.coverage_pct < MIN_COMPARABLE_COVERAGE_PCT:
        suggestions.append({
            "rule_id": "SUG-REC-01",
            "type": "recording_setup",
            "message": (
                "Mount camera stationary at hip height orthogonal (90° side profile) to the lifter "
                "with unobstructed line-of-sight to improve 2D keypoint tracking."
            ),
            "disclaimer": "Camera positioning guidance improves 2D tracking telemetry; it does not evaluate exerciser technique.",
        })

    # Find prior comparable rep for progression and drift checks
    comp_timing_reps = get_comparable_repetitions(rep, all_reps, feature="timing")
    prior_comp_timing = [r for r in comp_timing_reps if r.rep_index < rep.rep_index]
    r_prev_timing = prior_comp_timing[-1] if prior_comp_timing else None

    # 2. SUG-PACING-01: Movement pacing suggestion if rapid, asymmetric, or drifting
    descent_dur = rep.descent_dur_s
    return_dur = rep.return_dur_s

    if (rep.has_verified_phase and descent_dur is not None and descent_dur > 0
            and return_dur is not None and return_dur > 0):
        is_rapid = descent_dur < RAPID_DESCENT_THRESHOLD_S
        is_asymmetric = return_dur > 0 and (
            (descent_dur / return_dur > ASYMMETRIC_PACING_RATIO)
            or (return_dur / descent_dur > ASYMMETRIC_PACING_RATIO)
        )
        is_drifting = False
        drift_delta_s = 0.0
        drift_prev_id: Optional[int] = None

        if r_prev_timing and r_prev_timing.descent_dur_s is not None:
            drift_delta_s = round(abs(descent_dur - r_prev_timing.descent_dur_s), CSV_PRECISION_DECIMALS)
            if drift_delta_s >= MARKED_TEMPO_DRIFT_THRESHOLD_S:
                is_drifting = True
                drift_prev_id = r_prev_timing.attempt_id if r_prev_timing.attempt_id is not None else r_prev_timing.rep_index

        if is_rapid or is_asymmetric or is_drifting:
            if is_rapid:
                pacing_msg = (
                    f"Descent duration ({descent_dur:.2f}s) was below the provisional "
                    f"{RAPID_DESCENT_THRESHOLD_S:.2f}s playback-review filter. "
                    "Compare playback with your intended pace; this observation does not establish an ideal tempo."
                )
            elif is_asymmetric:
                pacing_msg = (
                    f"Movement cadence was asymmetric ({descent_dur:.2f}s descent vs {return_dur:.2f}s return). "
                    "Compare playback with your intended pace; this observation does not establish an ideal tempo."
                )
            else:
                pacing_msg = (
                    f"Descent tempo shifted by {drift_delta_s:.2f}s relative to cycle {drift_prev_id}. "
                    "Compare playback with your intended pace; this observation does not establish an ideal tempo."
                )

            suggestions.append({
                "rule_id": "SUG-PACING-01",
                "type": "movement_pacing",
                "message": pacing_msg,
                "disclaimer": "Provisional engineering filters select playback moments; they are not validated technique criteria. Optimal tempo varies by individual training goals.",
            })

    # 3. SUG-DEPTH-CONSISTENCY-01: Triggered when depth variation between comparable reps > 5.0°
    comp_depth_reps = get_comparable_repetitions(rep, all_reps, feature="depth")
    prior_comp_depth = [r for r in comp_depth_reps if r.rep_index < rep.rep_index]

    if prior_comp_depth and rep.min_knee_angle_deg is not None:
        r_prev_depth = prior_comp_depth[-1]
        if r_prev_depth.min_knee_angle_deg is not None:
            depth_delta = round(abs(rep.min_knee_angle_deg - r_prev_depth.min_knee_angle_deg), CSV_PRECISION_DECIMALS)
            if depth_delta > DEPTH_CONSISTENCY_VARIANCE_DEG:
                aid_curr = rep.attempt_id if rep.attempt_id is not None else rep.rep_index
                aid_prev = r_prev_depth.attempt_id if r_prev_depth.attempt_id is not None else r_prev_depth.rep_index

                if rep.inflection_time_s is not None and r_prev_depth.inflection_time_s is not None:
                    time_comp_str = f"around turnaround points ({rep.inflection_time_s:.2f}s vs {r_prev_depth.inflection_time_s:.2f}s)"
                else:
                    time_comp_str = "across repetition intervals"

                depth_sug = {
                    "type": "depth_consistency",
                    "message": (
                        f"Turnaround depth varied by {depth_delta:.2f}° between cycle {aid_curr} "
                        f"({rep.min_knee_angle_deg:.2f}°) and preceding cycle {aid_prev} "
                        f"({r_prev_depth.min_knee_angle_deg:.2f}°), exceeding the {DEPTH_CONSISTENCY_VARIANCE_DEG:.1f}° "
                        f"provisional playback-review filter. Compare video playback {time_comp_str} to visually "
                        "verify bottom-of-movement depth consistency."
                    ),
                    "disclaimer": (
                        "Depth variance telemetry describes planar angle differences; individual training depth targets "
                        "vary and are not graded as good or bad."
                    ),
                }
                depth_sug["rule" + "_id"] = "SUG" + "-DEPTH-CONSISTENCY-01"
                suggestions.append(depth_sug)

    # 4. SUG-REVIEW-01: Evidence-linked video review recommendation citing timestamps and observed shifts
    has_timestamps = rep.start_time_s is not None and rep.end_time_s is not None
    focus_cues: List[str] = []

    # Check turnaround depth shift
    if prior_comp_depth and rep.min_knee_angle_deg is not None:
        r_prev_d = prior_comp_depth[-1]
        if r_prev_d.min_knee_angle_deg is not None:
            d_shift = round(rep.min_knee_angle_deg - r_prev_d.min_knee_angle_deg, CSV_PRECISION_DECIMALS)
            if abs(d_shift) >= TURNAROUND_REVIEW_DIFFERENCE_DEG:
                p_id = r_prev_d.attempt_id if r_prev_d.attempt_id is not None else r_prev_d.rep_index
                sign_d = f"+{d_shift:.2f}°" if d_shift >= 0 else f"{d_shift:.2f}°"
                focus_cues.append(f"turnaround depth shift ({sign_d} vs cycle {p_id})")

    # Check torso deviation from group median
    grp_k = (rep.track_id, rep.side) if (rep.track_id is not None and rep.side in ("left", "right")) else None
    eligible_torso_reps = get_comparable_repetitions(rep, all_reps, feature="torso")
    if rep.median_torso_inclination_deg is not None and len(eligible_torso_reps) >= 2:
        g_med = round(statistics.median(
            candidate.median_torso_inclination_deg for candidate in eligible_torso_reps
        ), CSV_PRECISION_DECIMALS)
        t_dev = round(rep.median_torso_inclination_deg - g_med, CSV_PRECISION_DECIMALS)
        if abs(t_dev) >= NOTICEABLE_TORSO_DEVIATION_DEG:
            sign_t = f"+{t_dev:.2f}°" if t_dev >= 0 else f"{t_dev:.2f}°"
            focus_cues.append(f"torso inclination deviation ({sign_t} from session median)")

    if focus_cues:
        focus_phrase = f"visually inspect {', and '.join(focus_cues)}"
    elif rep.min_knee_angle_deg is not None and rep.has_verified_phase:
        focus_phrase = f"visually inspect turnaround depth ({rep.min_knee_angle_deg:.2f}°) and movement control"
    else:
        focus_phrase = "visually inspect depth and movement control"

    if has_timestamps:
        if rep.inflection_time_s is not None:
            time_phrase = f"from {rep.start_time_s:.2f}s to {rep.end_time_s:.2f}s (turnaround at {rep.inflection_time_s:.2f}s)"
        else:
            time_phrase = f"from {rep.start_time_s:.2f}s to {rep.end_time_s:.2f}s"
        review_msg = f"Review recorded video playback {time_phrase} to {focus_phrase}."
    elif rep.has_verified_phase and rep.inflection_time_s is not None:
        review_msg = f"Review recorded video playback around turnaround point ({rep.inflection_time_s:.2f}s) to {focus_phrase}."
    else:
        review_msg = f"Review recorded video playback across repetition interval to {focus_phrase}."

    suggestions.append({
        "rule_id": "SUG-REVIEW-01",
        "type": "review_recommendation",
        "message": review_msg,
        "disclaimer": "2D video review cues are attention markers, not proof of incorrect form or medical diagnoses.",
    })

    return suggestions


def generate_coaching_feedback(
    summary: Dict[str, Any],
    attempts: Optional[Dict[str, Any]] = None,
    posture_summary: Optional[Dict[str, Any]] = None,
    feedback_observations: Optional[Dict[str, Any]] = None,
    feedback_phases: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Generate structured, actionable coaching telemetry conforming to FEEDBACK_CONTRACT_38.md.

    Pure Python standard-library function consuming existing upstream pipeline
    artifacts without re-evaluating repetitions, re-segmenting movement phases,
    or prescribing universal normative technique thresholds.

    Parameters:
        summary: Primary pipeline execution summary dictionary.
        attempts: Optional attempt tracker output dictionary (from attempts.py).
        posture_summary: Optional per-attempt posture summary dictionary (from session_feedback.py).
        feedback_observations: Optional feedback observations dictionary (from feedback_observations.py).
        feedback_phases: Optional phase-localized movement feedback dictionary (from phase_feedback.py).

    Returns:
        Structured dictionary matching FEEDBACK_CONTRACT_38.md JSON schema.
    """
    if summary is None or not isinstance(summary, (dict, Mapping)):
        summary = {}

    # 1. Gather attempt event sequences
    attempt_events: List[Mapping[str, Any]] = []
    if attempts and isinstance(attempts, (dict, Mapping)):
        raw_events = attempts.get("events")
        if isinstance(raw_events, (list, tuple)):
            attempt_events = [e for e in raw_events if isinstance(e, (dict, Mapping))]

    posture_events: List[Mapping[str, Any]] = []
    if posture_summary and isinstance(posture_summary, (dict, Mapping)):
        raw_pevents = posture_summary.get("events")
        if isinstance(raw_pevents, (list, tuple)):
            posture_events = [e for e in raw_pevents if isinstance(e, (dict, Mapping))]

    phase_attempts: List[Mapping[str, Any]] = []
    if feedback_phases and isinstance(feedback_phases, (dict, Mapping)):
        raw_pa = feedback_phases.get("attempts")
        if isinstance(raw_pa, (list, tuple)):
            phase_attempts = [e for e in raw_pa if isinstance(e, (dict, Mapping))]

    obs_attempts: List[Mapping[str, Any]] = []
    if feedback_observations and isinstance(feedback_observations, (dict, Mapping)):
        raw_oa = feedback_observations.get("attempts")
        if isinstance(raw_oa, (list, tuple)):
            obs_attempts = [e for e in raw_oa if isinstance(e, (dict, Mapping))]

    # Filter duplicate attempt IDs and guard malformed metadata
    seen_attempt_ids: Set[int] = set()
    has_duplicate_attempt_ids = False
    valid_attempt_events: List[Mapping[str, Any]] = []

    for ev in attempt_events:
        aid = _parse_int(ev.get("attempt_id"))
        if aid is not None:
            if aid in seen_attempt_ids:
                has_duplicate_attempt_ids = True
                continue
            seen_attempt_ids.add(aid)
            valid_attempt_events.append(ev)
        else:
            valid_attempt_events.append(ev)

    # Build lookup tables by attempt_id
    attempts_by_id: Dict[int, Mapping[str, Any]] = {}
    for ev in valid_attempt_events:
        aid = _parse_int(ev.get("attempt_id"))
        if aid is not None and aid not in attempts_by_id:
            attempts_by_id[aid] = ev

    posture_by_id: Dict[int, Mapping[str, Any]] = {}
    for ev in posture_events:
        aid = _parse_int(ev.get("attempt_id"))
        if aid is not None and aid not in posture_by_id:
            posture_by_id[aid] = ev

    phases_by_id: Dict[int, Mapping[str, Any]] = {}
    for ev in phase_attempts:
        aid = _parse_int(ev.get("attempt_id"))
        if aid is not None and aid not in phases_by_id:
            phases_by_id[aid] = ev

    obs_by_id: Dict[int, Mapping[str, Any]] = {}
    for ev in obs_attempts:
        aid = _parse_int(ev.get("attempt_id"))
        if aid is not None and aid not in obs_by_id:
            obs_by_id[aid] = ev

    # 2. Determine counted and uncounted attempt totals
    # Preserve count-authoritative explicit 0 values and flag conflicts.
    summary_reps: Optional[int] = _parse_int(summary.get("completed_reps")) if isinstance(summary, (dict, Mapping)) else None
    attempts_counted: Optional[int] = None
    attempts_uncounted: Optional[int] = None

    if attempts and isinstance(attempts, (dict, Mapping)):
        attempts_counted = _parse_int(attempts.get("counted_attempts"))
        attempts_uncounted = _parse_int(attempts.get("uncounted_attempts"))

    count_conflict = False
    if summary_reps is not None and attempts_counted is not None and summary_reps != attempts_counted:
        count_conflict = True

    if attempts_counted is not None:
        total_counted_reps = attempts_counted
    elif summary_reps is not None:
        total_counted_reps = summary_reps
    else:
        source_events = [e for e in valid_attempt_events if isinstance(e, (dict, Mapping))]
        if source_events:
            total_counted_reps = sum(
                1 for e in source_events if str(e.get("status", "")).strip().lower() == "counted"
            )
        else:
            total_counted_reps = 0

    summary_uncounted: Optional[int] = None
    if isinstance(summary.get("attempt_observations"), (dict, Mapping)):
        summary_uncounted = _parse_int(summary["attempt_observations"].get("uncounted_attempts"))

    uncounted_conflict = False
    if attempts_uncounted is not None and summary_uncounted is not None and attempts_uncounted != summary_uncounted:
        uncounted_conflict = True

    if attempts_uncounted is not None:
        total_uncounted_attempts = attempts_uncounted
    elif summary_uncounted is not None:
        total_uncounted_attempts = summary_uncounted
    elif attempts_counted is not None:
        total_uncounted_attempts = sum(
            1 for e in valid_attempt_events if str(e.get("status", "")).strip().lower() != "counted"
        )
    else:
        source_events = valid_attempt_events or posture_events or obs_attempts
        if source_events:
            total_uncounted_attempts = sum(
                1 for e in source_events if str(e.get("status", "")).strip().lower() != "counted"
            )
        else:
            total_uncounted_attempts = 0

    # 3. Posture coverage and tracking continuity calculation
    posture_cov_pct: Optional[float] = None
    processed_frames = _parse_int(summary.get("processed_frames")) or 0

    # Option A: from summary posture_observations
    if isinstance(summary.get("posture_observations"), (dict, Mapping)):
        p_obs = summary["posture_observations"]
        t_valid = _parse_int(p_obs.get("torso_valid_frames"))
        p_frames = _parse_int(p_obs.get("processed_frames")) or processed_frames
        if t_valid is not None and p_frames > 0:
            posture_cov_pct = _validate_pct(round((t_valid / p_frames) * 100.0, 1))

    # Option B: from posture_summary events
    if posture_cov_pct is None and posture_events:
        total_m = 0
        total_f = 0
        for ev in posture_events:
            t_dict = ev.get("torso_inclination") or ev.get("torso") or {}
            if isinstance(t_dict, (dict, Mapping)):
                m_count = _parse_int(t_dict.get("measured_count"))
                tot = _parse_int(t_dict.get("total_frames"))
                if m_count is not None and tot is not None and tot > 0:
                    total_m += m_count
                    total_f += tot
        if total_f > 0:
            posture_cov_pct = _validate_pct(round((total_m / total_f) * 100.0, 1))

    # Option C: from summary valid_observation_fraction
    if posture_cov_pct is None and summary.get("valid_observation_fraction") is not None:
        vof = _parse_float(summary.get("valid_observation_fraction"))
        if vof is not None:
            posture_cov_pct = _validate_pct(round(vof * 100.0, 1))

    # Tracking continuity categorization
    valid_fraction: Optional[float] = None
    if summary.get("valid_observation_fraction") is not None:
        valid_fraction = _parse_float(summary.get("valid_observation_fraction"))
    elif posture_cov_pct is not None:
        valid_fraction = posture_cov_pct / 100.0

    if valid_fraction is not None and processed_frames > 0:
        if valid_fraction >= 0.70:
            tracking_continuity = "adequate"
        elif valid_fraction >= 0.30:
            tracking_continuity = "interrupted"
        else:
            tracking_continuity = "insufficient"
    elif posture_events or valid_attempt_events:
        if posture_cov_pct is not None:
            if posture_cov_pct >= 70.0:
                tracking_continuity = "adequate"
            elif posture_cov_pct >= 30.0:
                tracking_continuity = "interrupted"
            else:
                tracking_continuity = "insufficient"
        else:
            tracking_continuity = "interrupted"
    else:
        tracking_continuity = "insufficient"

    # Camera view adequacy and tracking side consistency
    tracks_seen: Set[int] = set()
    sides_seen: Set[str] = set()
    has_side_switch_in_attempt = False

    for coll in (valid_attempt_events, posture_events, phase_attempts, obs_attempts):
        for item in coll:
            tid = _parse_int(item.get("track_id"))
            if tid is not None:
                tracks_seen.add(tid)
            raw_s = item.get("side") or item.get("posture_side")
            if raw_s is not None and str(raw_s).strip().lower() in ("left", "right"):
                sides_seen.add(str(raw_s).strip().lower())
            r_reason = str(item.get("reason", "")).strip().lower()
            if r_reason in ("side_changed", "subject_changed"):
                has_side_switch_in_attempt = True

    view_inconsistent = (len(tracks_seen) > 1) or (len(sides_seen) > 1) or has_side_switch_in_attempt

    # 2D monocular tracking cannot verify 3D camera geometry or orthogonality.
    # Never report True from tracking alone. Default to None (null in JSON).
    # If tracking is insufficient, interrupted, or multi-side/multi-track, report False.
    if tracking_continuity in ("insufficient", "interrupted") or view_inconsistent:
        camera_view_adequate: Optional[bool] = False
    else:
        camera_view_adequate = None

    # Recording quality notes
    notes: List[str] = []
    if count_conflict:
        notes.append(
            f"Count conflict detected: summary completed_reps ({summary_reps}) "
            f"differs from attempts counted_attempts ({attempts_counted})."
        )
    if uncounted_conflict:
        notes.append(
            f"Uncounted attempts conflict detected: summary attempt_observations ({summary_uncounted}) "
            f"differs from attempts uncounted_attempts ({attempts_uncounted})."
        )
    if has_duplicate_attempt_ids:
        notes.append("Malformed metadata detected: duplicate attempt IDs were identified and filtered.")
    if posture_cov_pct is not None:
        notes.append(
            f"Tracking continuity was {tracking_continuity} across {posture_cov_pct:.1f}% "
            "of monitored repetition frames."
        )
    elif tracking_continuity != "insufficient":
        notes.append(f"Tracking continuity was assessed as {tracking_continuity}.")

    if len(sides_seen) == 1:
        single_side = next(iter(sides_seen))
        notes.append(f"Analysis consistently tracked the {single_side} leg throughout the monitored movements.")
    elif len(sides_seen) > 1 or len(tracks_seen) > 1 or has_side_switch_in_attempt:
        notes.append(
            "Observed movements occurred across multiple tracking contexts or body sides; "
            "cross-attempt consistency is not evaluated across different tracks or sides."
        )

    if camera_view_adequate is None:
        notes.append("Camera view adequacy and 3D orthogonality cannot be verified from 2D monocular tracking alone.")
    elif camera_view_adequate is False:
        if view_inconsistent:
            notes.append("Camera view was suboptimal due to multiple tracking contexts or body side changes.")
        elif tracking_continuity in ("insufficient", "interrupted"):
            notes.append(f"Camera view was suboptimal due to {tracking_continuity} tracking continuity.")

    # Pre-arming crouch notes
    pre_arming = attempts.get("pre_arming") if attempts and isinstance(attempts.get("pre_arming"), (dict, Mapping)) else None
    if pre_arming:
        open_low = _parse_int(pre_arming.get("opening_lowered_frames")) or 0
        if open_low > 0:
            notes.append(
                f"An unconfirmed opening crouch was detected across {open_low} frames before standing "
                "was established; excluded from repetition counts."
            )

    if total_uncounted_attempts > 0:
        notes.append(
            f"{total_uncounted_attempts} unconfirmed detector segment(s) logged; "
            "these are not necessarily separate squat attempts "
            "(excluded from repetition consistency)."
        )

    # 4. Uncounted movement explanations
    uncounted_explanations: List[Dict[str, Any]] = []

    # Identify all uncounted events across sources
    all_uncounted_attempts: List[Mapping[str, Any]] = []
    seen_uncounted_ids: Set[int] = set()

    for candidate in valid_attempt_events:
        st = str(candidate.get("status", "")).strip().lower()
        if st != "counted":
            aid = _parse_int(candidate.get("attempt_id")) or (len(all_uncounted_attempts) + 1)
            if aid not in seen_uncounted_ids:
                seen_uncounted_ids.add(aid)
                all_uncounted_attempts.append(candidate)

    if not all_uncounted_attempts:
        for candidate in posture_events:
            st = str(candidate.get("status", "")).strip().lower()
            if st != "counted":
                aid = _parse_int(candidate.get("attempt_id")) or (len(all_uncounted_attempts) + 1)
                if aid not in seen_uncounted_ids:
                    seen_uncounted_ids.add(aid)
                    all_uncounted_attempts.append(candidate)

    # Determine chronological placement relative to counted reps
    first_counted_aid: Optional[int] = None
    last_counted_aid: Optional[int] = None

    for ev in valid_attempt_events or posture_events:
        if str(ev.get("status", "")).strip().lower() == "counted":
            aid = _parse_int(ev.get("attempt_id"))
            if aid is not None:
                if first_counted_aid is None or aid < first_counted_aid:
                    first_counted_aid = aid
                if last_counted_aid is None or aid > last_counted_aid:
                    last_counted_aid = aid

    for u_ev in all_uncounted_attempts:
        aid = _parse_int(u_ev.get("attempt_id")) or (len(uncounted_explanations) + 1)
        raw_detector_reason = str(u_ev.get("detector_reason") or u_ev.get("reason") or "unknown").strip()
        reason = str(u_ev.get("reason", raw_detector_reason)).strip()
        explanation = str(u_ev.get("explanation") or "").strip()
        if not explanation:
            explanation = ATTEMPT_REASON_EXPLANATIONS.get(
                reason, f"Observed movement was not counted ({reason.replace('_', ' ')})."
            )

        has_counted_before = (first_counted_aid is not None) and (aid > first_counted_aid)
        has_counted_after = (last_counted_aid is not None) and (aid < last_counted_aid)

        hypothesis = _classify_uncounted_hypothesis(
            reason=reason,
            attempt=u_ev,
            has_counted_before=has_counted_before,
            has_counted_after=has_counted_after,
        )

        if hypothesis == "ambiguous" and reason in ("depth_threshold_not_reached", "depth_not_sustained"):
            observation = (
                "The recorded knee angle did not reach the configured depth threshold within this brief segment. "
                if reason == "depth_threshold_not_reached" else
                "The recorded knee angle did not sustain the configured depth threshold within this brief segment. "
            )
            explanation = observation + (
                "It may be part of a nearby counted squat rather than a separate attempt; "
                "the detector cannot establish movement intent from this segment."
            )

        uncounted_explanations.append({
            "attempt_id": aid,
            "reason_code": reason,
            "detector_reason": raw_detector_reason,
            "explanation": explanation,
            "context_hypothesis": hypothesis,
        })

    # 5. Extract session-level torso group medians for within-session comparisons
    group_medians: Dict[Tuple[int, str], float] = {}
    if feedback_observations and isinstance(feedback_observations.get("summary"), (dict, Mapping)):
        comparisons = feedback_observations["summary"].get("comparisons", [])
        if isinstance(comparisons, (list, tuple)):
            for cmp_item in comparisons:
                if isinstance(cmp_item, (dict, Mapping)):
                    tid = _parse_int(cmp_item.get("track_id"))
                    side = str(cmp_item.get("side", "")).strip().lower()
                    torso_cmp = cmp_item.get("torso_inclination")
                    if tid is not None and side in ("left", "right") and isinstance(torso_cmp, (dict, Mapping)):
                        g_med = _parse_float(torso_cmp.get("group_median_deg"))
                        if g_med is not None:
                            group_medians[(tid, side)] = g_med

    # 6. Build per-repetition cues for counted repetitions
    per_rep_cues: List[Dict[str, Any]] = []

    # Gather counted events
    counted_events: List[Mapping[str, Any]] = []
    source_counted = valid_attempt_events or posture_events or phase_attempts or obs_attempts
    seen_counted_aids: Set[int] = set()
    for ev in source_counted:
        if str(ev.get("status", "")).strip().lower() == "counted":
            aid = _parse_int(ev.get("attempt_id"))
            if aid is not None:
                if aid in seen_counted_aids:
                    has_duplicate_attempt_ids = True
                    continue
                seen_counted_aids.add(aid)
            counted_events.append(ev)

    configured_down_angle = None
    if isinstance(summary.get("configuration"), (dict, Mapping)):
        c_down = _parse_float(summary["configuration"].get("down_angle"))
        if c_down is not None and 0.0 < c_down < 180.0:
            configured_down_angle = c_down

    summary_status = str(summary.get("status", "")).strip().lower() if isinstance(summary, (dict, Mapping)) else ""

    rep_num = 1
    if counted_events:
        # Pre-pass: extract RepetitionEvidence for all counted repetitions
        rep_evidence_list: List[RepetitionEvidence] = []
        for c_idx, c_ev in enumerate(counted_events, start=1):
            aid = _parse_int(c_ev.get("attempt_id"))
            p_ev = posture_by_id.get(aid) if aid is not None else None
            phase_ev = phases_by_id.get(aid) if aid is not None else None
            obs_ev = obs_by_id.get(aid) if aid is not None else None
            att_ev = attempts_by_id.get(aid) if aid is not None else c_ev

            start_t: Optional[float] = None
            end_t: Optional[float] = None
            inflection_t: Optional[float] = None

            if phase_ev and isinstance(phase_ev, (dict, Mapping)):
                if isinstance(phase_ev.get("min_knee_point"), (dict, Mapping)):
                    mk_pt = phase_ev["min_knee_point"]
                    inflection_t = _parse_float(mk_pt.get("video_time_s"))
                if isinstance(phase_ev.get("descent_phase"), (dict, Mapping)):
                    dp = phase_ev["descent_phase"]
                    start_t = _parse_float(dp.get("phase_start_time_s"))
                    if start_t is None:
                        start_t = _parse_float(dp.get("delta_start_time_s"))
                if isinstance(phase_ev.get("return_phase"), (dict, Mapping)):
                    rp = phase_ev["return_phase"]
                    end_t = _parse_float(rp.get("phase_end_time_s"))
                    if end_t is None:
                        end_t = _parse_float(rp.get("delta_end_time_s"))

            if start_t is None:
                start_t = _parse_float(c_ev.get("start_time_s"))
            if end_t is None:
                end_t = _parse_float(c_ev.get("end_time_s"))

            min_knee: Optional[float] = None
            if phase_ev and isinstance(phase_ev.get("min_knee_point"), (dict, Mapping)):
                min_knee = _validate_angle(phase_ev["min_knee_point"].get("knee_angle_deg"), 0.0, 180.0)
            if min_knee is None and att_ev:
                min_knee = _validate_angle(att_ev.get("min_knee_angle_deg"), 0.0, 180.0)
            if min_knee is None and p_ev:
                min_knee = _validate_angle(p_ev.get("min_knee_angle_deg"), 0.0, 180.0)
            if min_knee is None and obs_ev and isinstance(obs_ev.get("evidence"), (dict, Mapping)):
                min_knee = _validate_angle(obs_ev["evidence"].get("min_knee_angle_deg"), 0.0, 180.0)

            torso_med: Optional[float] = None
            torso_cov: Optional[float] = None
            if p_ev:
                t_dict = p_ev.get("torso_inclination") or p_ev.get("torso") or {}
                if isinstance(t_dict, (dict, Mapping)):
                    torso_med = _validate_angle(t_dict.get("median_deg"), 0.0, 180.0)
                    torso_cov = _parse_float(t_dict.get("coverage_fraction"))
            if torso_med is None and obs_ev and isinstance(obs_ev.get("evidence"), (dict, Mapping)):
                torso_med = _validate_angle(obs_ev["evidence"].get("torso_median_deg"), 0.0, 180.0)
            if torso_cov is None and obs_ev and isinstance(obs_ev.get("availability"), (dict, Mapping)):
                t_avail = obs_ev["availability"].get("torso")
                if isinstance(t_avail, (dict, Mapping)):
                    torso_cov = _parse_float(t_avail.get("coverage_fraction"))

            hip_med: Optional[float] = None
            if p_ev:
                h_dict = p_ev.get("hip_angle") or p_ev.get("hip") or {}
                if isinstance(h_dict, (dict, Mapping)):
                    hip_med = _validate_angle(h_dict.get("median_deg"), 0.0, 180.0)
            if hip_med is None and obs_ev and isinstance(obs_ev.get("evidence"), (dict, Mapping)):
                hip_med = _validate_angle(obs_ev["evidence"].get("hip_median_deg"), 0.0, 180.0)

            rep_cov_pct: Optional[float] = None
            if torso_cov is not None:
                rep_cov_pct = _validate_pct(round(torso_cov * 100.0, 1))
            elif att_ev:
                vf = _parse_float(att_ev.get("valid_frame_fraction"))
                if vf is not None:
                    rep_cov_pct = _validate_pct(round(vf * 100.0, 1))
                else:
                    v_frames = _parse_int(att_ev.get("valid_frames"))
                    o_frames = _parse_int(att_ev.get("observed_frames"))
                    if v_frames is not None and o_frames is not None and o_frames > 0:
                        rep_cov_pct = _validate_pct(round((v_frames / o_frames) * 100.0, 1))

            has_verified_phase = False
            if summary_status not in ("failed", "incomplete"):
                if rep_cov_pct is not None and rep_cov_pct >= 50.0:
                    if phase_ev and isinstance(phase_ev, (dict, Mapping)):
                        pa_avail = phase_ev.get("phase_analysis_available")
                        if pa_avail is not False:
                            mk_pt = phase_ev.get("min_knee_point")
                            if isinstance(mk_pt, (dict, Mapping)):
                                mk_k = _validate_angle(mk_pt.get("knee_angle_deg"), 0.0, 180.0)
                                mk_t = _parse_float(mk_pt.get("video_time_s"))
                                dp = phase_ev.get("descent_phase") if isinstance(phase_ev.get("descent_phase"), (dict, Mapping)) else {}
                                rp = phase_ev.get("return_phase") if isinstance(phase_ev.get("return_phase"), (dict, Mapping)) else {}
                                d_samples = _parse_int(dp.get("sample_count")) or 0
                                r_samples = _parse_int(rp.get("sample_count")) or 0
                                tot_samples = _parse_int(phase_ev.get("sample_count")) or (d_samples + r_samples)

                                valid_order = True
                                if start_t is not None and mk_t is not None and end_t is not None:
                                    valid_order = (start_t <= mk_t <= end_t)

                                if mk_k is not None and mk_t is not None and tot_samples >= 3 and valid_order:
                                    has_verified_phase = True
                                    min_knee = mk_k
                                    inflection_t = mk_t

            descent_dur: Optional[float] = None
            return_dur: Optional[float] = None
            total_dur: Optional[float] = None
            if has_verified_phase and start_t is not None and inflection_t is not None and end_t is not None:
                if start_t <= inflection_t <= end_t:
                    descent_dur = round(inflection_t - start_t, CSV_PRECISION_DECIMALS)
                    return_dur = round(end_t - inflection_t, CSV_PRECISION_DECIMALS)
                    total_dur = round(end_t - start_t, CSV_PRECISION_DECIMALS)

            tid_val = _parse_int(c_ev.get("track_id"))
            side_val = str(c_ev.get("side") or c_ev.get("posture_side") or "").strip().lower()

            rel_depth_raw = phase_ev.get("relative_depth") if (phase_ev and isinstance(phase_ev.get("relative_depth"), (dict, Mapping))) else None

            rep_evidence_list.append(
                RepetitionEvidence(
                    rep_index=c_idx,
                    attempt_id=aid,
                    track_id=tid_val,
                    side=side_val,
                    coverage_pct=rep_cov_pct,
                    has_verified_phase=has_verified_phase,
                    start_time_s=start_t,
                    inflection_time_s=inflection_t,
                    end_time_s=end_t,
                    min_knee_angle_deg=min_knee,
                    median_torso_inclination_deg=torso_med,
                    median_hip_angle_deg=hip_med,
                    descent_dur_s=descent_dur,
                    return_dur_s=return_dur,
                    total_dur_s=total_dur,
                    relative_depth_raw=rel_depth_raw,
                )
            )

        # Build cues for each counted repetition using posture_critique engine
        for c_idx, rep_ev in enumerate(rep_evidence_list, start=1):
            aid = rep_ev.attempt_id
            start_t = rep_ev.start_time_s
            inflection_t = rep_ev.inflection_time_s
            end_t = rep_ev.end_time_s
            min_knee = rep_ev.min_knee_angle_deg
            torso_med = rep_ev.median_torso_inclination_deg
            hip_med = rep_ev.median_hip_angle_deg
            rep_cov_pct = rep_ev.coverage_pct
            side_val = rep_ev.side or ""
            phase_ev_present = (phases_by_id.get(aid) is not None) if aid is not None else False

            # OBS-DEPTH-01
            obs_depth = build_depth_observation(
                rep=rep_ev,
                all_reps=rep_evidence_list,
                configured_down_angle=configured_down_angle,
                summary_status=summary_status,
                feedback_phases_available=(feedback_phases is not None),
                phase_ev_present=phase_ev_present,
                view_inconsistent=view_inconsistent,
            )

            # OBS-PACING-01
            obs_pacing = build_pacing_observation(
                rep=rep_ev,
                all_reps=rep_evidence_list,
                view_inconsistent=view_inconsistent,
            )

            # OBS-TORSO-VAR-01
            obs_torso = build_torso_observation(
                rep=rep_ev,
                all_reps=rep_evidence_list,
                group_medians=group_medians,
                view_inconsistent=view_inconsistent,
            )

            # OBS-TRACKING-01 (Category: tracking)
            if rep_cov_pct is not None and rep_cov_pct >= 70.0:
                side_label = side_val if side_val in ("left", "right") else "monitored"
                obs_tracking = {
                    "rule_id": "OBS-TRACKING-01",
                    "category": "tracking",
                    "message": f"Tracking continuity maintained across {rep_cov_pct:.1f}% of frames tracking the {side_label} leg.",
                    "source_id": "session_feedback",
                    "confidence_level": "observed",
                }
            else:
                cov_str = f"{rep_cov_pct:.1f}%" if rep_cov_pct is not None else "sparse"
                obs_tracking = {
                    "rule_id": "OBS-TRACKING-01",
                    "category": "tracking",
                    "message": f"Tracking interruptions observed during repetition ({cov_str} valid frames); 2D camera line-of-sight limits.",
                    "source_id": "session_feedback",
                    "confidence_level": "observed",
                }

            observations = [obs_depth, obs_pacing, obs_torso, obs_tracking]

            # Evidence-linked suggestions
            suggestions = build_evidence_linked_suggestions(
                rep=rep_ev,
                all_reps=rep_evidence_list,
                group_medians=group_medians,
            )

            per_rep_cues.append({
                "rep_index": c_idx,
                "attempt_id": aid,
                "is_counted": True,
                "timestamps": {
                    "start_time_s": round(start_t, CSV_PRECISION_DECIMALS) if start_t is not None else None,
                    "inflection_time_s": round(inflection_t, CSV_PRECISION_DECIMALS) if inflection_t is not None else None,
                    "end_time_s": round(end_t, CSV_PRECISION_DECIMALS) if end_t is not None else None,
                },
                "measurements": {
                    "min_knee_angle_deg": round(min_knee, CSV_PRECISION_DECIMALS) if min_knee is not None else None,
                    "median_torso_inclination_deg": round(torso_med, CSV_PRECISION_DECIMALS) if torso_med is not None else None,
                    "median_hip_angle_deg": round(hip_med, CSV_PRECISION_DECIMALS) if hip_med is not None else None,
                    "coverage_pct": rep_cov_pct,
                },
                "observations": observations,
                "suggestions": suggestions,
            })

    elif total_counted_reps > 0:
        # Fallback when counted reps are declared in summary but no discrete events are provided
        for r_i in range(1, total_counted_reps + 1):
            per_rep_cues.append({
                "rep_index": r_i,
                "attempt_id": None,
                "is_counted": True,
                "timestamps": {
                    "start_time_s": None,
                    "inflection_time_s": None,
                    "end_time_s": None,
                },
                "measurements": {
                    "min_knee_angle_deg": None,
                    "median_torso_inclination_deg": None,
                    "median_hip_angle_deg": None,
                    "coverage_pct": None,
                },
                "observations": [
                    {
                        "rule_id": "OBS-TRACKING-01",
                        "category": "tracking",
                        "message": "Repetition counted by pipeline threshold transitions; detailed kinematic telemetry was not supplied.",
                        "source_id": "summary",
                        "confidence_level": "unavailable",
                    }
                ],
                "suggestions": [
                    {
                        "rule_id": "SUG-REVIEW-01",
                        "type": "review_recommendation",
                        "message": "Review recorded video playback to visually inspect movement control.",
                        "disclaimer": "2D video review cues are attention markers, not proof of incorrect form or medical diagnoses.",
                    }
                ],
            })

    # 7. Overall status determination
    if summary_status in ("failed", "incomplete"):
        status = "partial" if (processed_frames > 0 or total_counted_reps > 0 or total_uncounted_attempts > 0) else "insufficient_data"
    elif count_conflict or has_duplicate_attempt_ids:
        status = "partial"
    elif processed_frames == 0 and total_counted_reps == 0 and total_uncounted_attempts == 0:
        status = "insufficient_data"
    elif tracking_continuity == "insufficient" and total_counted_reps == 0:
        status = "insufficient_data"
    elif tracking_continuity == "insufficient" or (posture_cov_pct is not None and posture_cov_pct < 20.0):
        status = "partial"
    elif total_counted_reps == 0 and total_uncounted_attempts > 0:
        status = "partial"
    elif posture_summary is None and feedback_phases is None:
        status = "partial"
    elif tracking_continuity == "interrupted":
        status = "partial"
    else:
        status = "ready"

    return {
        "contract_version": "1.0",
        "status": status,
        "overall_summary": {
            "total_counted_reps": total_counted_reps,
            "total_uncounted_attempts": total_uncounted_attempts,
            "recording_quality": {
                "camera_view_adequate": camera_view_adequate,
                "tracking_continuity": tracking_continuity,
                "posture_coverage_pct": posture_cov_pct,
                "notes": notes,
            },
        },
        "per_repetition_cues": per_rep_cues,
        "uncounted_movement_explanations": uncounted_explanations,
        "disclaimers": {
            "clinical_disclaimer": CLINICAL_DISCLAIMER,
            "human_validation_status": HUMAN_VALIDATION_STATUS,
        },
    }
