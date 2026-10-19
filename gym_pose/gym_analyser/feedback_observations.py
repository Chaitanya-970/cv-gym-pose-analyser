"""Practical supported posture observations and session feedback aggregation.

Delivers plain-language, evidence-backed kinematic observations across discrete
movement attempts without normative grading, clinical diagnoses, or unvalidated
universal standards.

Operates purely on standard library primitives, consuming existing summary
schemas from session_feedback.py and event mappings from attempts.py.
"""

import math
import statistics
from decimal import Decimal, InvalidOperation
from dataclasses import dataclass
from numbers import Real
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple, Union

# Upstream CSV precision limit (two decimal places)
CSV_PRECISION_DECIMALS = 2

# Standard biomechanical and scope limitations for 2D monocular lateral kinematics
SCOPE_LIMITATIONS: Tuple[str, ...] = (
    "Kinematic measurements represent projected 2D coordinates from a monocular lateral view; "
    "individual anthropometry, camera angle, height, and roll influence observed angles.",
    "Torso inclination is an unsigned 2D angle relative to upward vertical; it does not indicate "
    "signed forward versus backward lean direction or spinal curvature.",
    "Frontal and transverse plane mechanics (such as dynamic knee valgus, pelvic rotation, and lateral "
    "hip shift) are not observable from a single 2D side camera view.",
    "Technique quality, clinical safety, injury risk, and form correctness are not evaluated or implied; "
    "observations describe observed motion telemetry only.",
    "Observed movement events are software-defined threshold crossings and can include equipment setup, "
    "weight pickup, and weight return; exercise intent is not classified.",
    "Cross-attempt comparisons reflect this individual's recorded session consistency across matching "
    "track and side views, not a normative, medical, or universal standard.",
)

# Canonical explanations for attempt reasons (reusing attempts.py semantics)
ATTEMPT_REASON_EXPLANATIONS: Dict[str, str] = {
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


@dataclass(frozen=True)
class ObservationConfig:
    """Operational configuration for feedback observation filtering.

    Attributes:
        min_coverage_fraction: Minimum valid frame coverage ratio [0.0, 1.0] required
            for an attempt's posture metrics to be considered reliable for cross-attempt
            comparisons. Default is 0.5 (50% coverage).
        min_attempts_for_comparison: Minimum number of reliable counted attempts on the same
            track and side required to evaluate cross-attempt spread. Must be >= 2 because
            a single attempt cannot establish consistency. Default is 2.

    Note:
        These parameters are software operational filters to avoid spurious telemetry
        comparisons across sparse or interrupted sequences. They do NOT represent
        biomechanically or clinically validated standards.
    """

    min_coverage_fraction: float = 0.5
    min_attempts_for_comparison: int = 2

    def __post_init__(self) -> None:
        if (
            isinstance(self.min_coverage_fraction, bool)
            or not isinstance(self.min_coverage_fraction, Real)
            or not math.isfinite(self.min_coverage_fraction)
            or not (0.0 <= self.min_coverage_fraction <= 1.0)
        ):
            raise ValueError(
                f"min_coverage_fraction must be a finite float in [0.0, 1.0], got: {self.min_coverage_fraction!r}"
            )
        if (
            isinstance(self.min_attempts_for_comparison, bool)
            or not isinstance(self.min_attempts_for_comparison, int)
            or self.min_attempts_for_comparison < 2
        ):
            raise ValueError(
                f"min_attempts_for_comparison must be an integer >= 2, got: {self.min_attempts_for_comparison!r}"
            )


def _parse_int(val: Any, name: str, allow_negative: bool = False) -> int:
    """Parse an integer value with strict rejection of booleans and non-integers."""
    if isinstance(val, bool):
        raise ValueError(f"{name} cannot be a boolean value: {val!r}")
    if isinstance(val, int):
        parsed = val
    elif isinstance(val, float):
        if not math.isfinite(val) or not val.is_integer():
            raise ValueError(f"{name} must be an integer, got non-integer float: {val!r}")
        parsed = int(val)
    elif isinstance(val, str):
        cleaned = val.strip()
        if not cleaned:
            raise ValueError(f"{name} cannot be empty string")
        try:
            exact = Decimal(cleaned)
            if not exact.is_finite() or exact != exact.to_integral_value():
                raise ValueError
            parsed = int(exact)
        except (InvalidOperation, ValueError, OverflowError):
            raise ValueError(f"{name} must be an integer, got: {val!r}")
    else:
        raise ValueError(f"{name} has invalid type {type(val).__name__}: {val!r}")

    if not allow_negative and parsed < 0:
        raise ValueError(f"{name} must be non-negative, got: {parsed}")
    return parsed


def _parse_angle_metric(val: Any, name: str) -> Optional[float]:
    """Parse an angle value strictly in [0.0, 180.0]; blanks/None return None, booleans/out-of-range raise."""
    if val is None:
        return None
    if isinstance(val, bool):
        raise ValueError(f"{name} cannot be a boolean value: {val!r}")
    if isinstance(val, str):
        cleaned = val.strip()
        if not cleaned or cleaned.lower() in ("none", "nan", "null"):
            return None
        try:
            parsed = float(cleaned)
        except ValueError:
            raise ValueError(f"{name} must be a valid float or blank, got: {val!r}")
    elif isinstance(val, (int, float)):
        parsed = float(val)
    else:
        raise ValueError(f"{name} has invalid type {type(val).__name__}: {val!r}")

    if not math.isfinite(parsed):
        return None

    if not (0.0 <= parsed <= 180.0):
        raise ValueError(f"{name} angle must be in [0.0, 180.0], got: {parsed}")
    return parsed


def _format_unavailable_reasons(reasons: Mapping[str, int]) -> str:
    """Format dictionary of unavailable reason counts into human-readable string."""
    if not reasons:
        return "none reported"
    return ", ".join(f"{k} ({v}f)" for k, v in sorted(reasons.items()))


def _categorize_event(status: str, reason: str) -> str:
    """Categorize event into completed, movement_threshold, measurement, or stream_end."""
    if status == "counted":
        return "completed"
    if reason.startswith("depth_"):
        return "movement_threshold"
    if reason == "video_ended_mid_attempt":
        return "stream_end"
    if reason == "processing_failed":
        return "processing_failed"
    return "measurement"


def _generate_context_note(status: str, category: str, reason: str) -> str:
    """Generate neutral context distinguishing incomplete motion from tracking/setup."""
    if status == "counted":
        return "Counted repetition meeting configured depth and return-to-standing thresholds."
    if reason == "processing_failed":
        return "Processing halted before this attempt could be confirmed; the system cannot determine exercise intent."
    if reason == "video_ended_mid_attempt":
        return (
            "Video recording ended before return to standing was confirmed; "
            "the system cannot determine exercise intent (uncounted motion may reflect a set completion, "
            "weight return, or unfinished movement)."
        )
    if category == "movement_threshold":
        return (
            "Observed movement did not meet configured depth or stability thresholds. "
            "The system cannot determine exercise intent; uncounted motions may reflect setup, bar adjustment, "
            "weight pickup/return, or an incomplete cycle."
        )
    if category == "measurement":
        known_measurement = (
            "tracking_gap",
            "confidence_loss",
            "ambiguous_subjects",
            "side_changed",
            "subject_changed",
        )
        if reason in known_measurement:
            return (
                f"Observation was uncounted due to tracking or visibility interruption ({reason.replace('_', ' ')}). "
                "This reflects 2D camera line-of-sight limits or multi-person tracking ambiguity, "
                "not exerciser movement technique."
            )
        return (
            f"Observation was uncounted (reason: {reason.replace('_', ' ')}). "
            "The system cannot determine exercise intent or form quality from this uncounted event."
        )
    return "Observed movement was not counted by threshold criteria; the system cannot determine exercise intent."


def build_feedback_observations(
    posture_summary: Dict[str, Any],
    attempt_events: Optional[Union[Sequence[Dict[str, Any]], Dict[str, Any]]] = None,
    config: Optional[ObservationConfig] = None,
) -> Dict[str, Any]:
    """Aggregate per-attempt kinematic posture observations and cross-attempt comparisons.

    Consumes the posture summary output produced by summarise_attempt_posture()
    (session_feedback.py) and optional attempt event dictionaries from attempts.py.

    Produces neutral, plain-language observations highlighting:
    - Within-track/side torso inclination differences across reliably measured COUNTED cycles.
    - Movement depth (projected knee flexion angle) consistency across reliably measured COUNTED cycles.
    - Explicit measurement availability flags and reason codes for unavailable metrics.
    - Clear distinction between incomplete movements, tracking/visibility failures,
      and equipment setup / weight return ambiguity.
    - Scope limitations emphasizing 2D monocular lateral-view constraints.

    Parameters:
        posture_summary: Dictionary conforming to session_feedback summary schema,
            containing 'summary' and 'events' mappings.
        attempt_events: Optional sequence or dict containing 'events' from attempts.json,
            providing additional attempt metadata (e.g. min_knee_angle_deg, valid_frames, observed_frames).
        config: Optional ObservationConfig overriding operational filtering thresholds.

    Returns:
        Structured dictionary containing:
        - 'summary': High-level counts, cross-attempt comparisons, and availability totals.
        - 'attempts': Detailed per-attempt plain-language observations and evidence dictionaries.
        - 'configuration': Effective operational filtering parameters.
        - 'scope_limitations': Caveats on 2D monocular lateral-view kinematics.

    Raises:
        ValueError: If posture_summary or attempt_events contain malformed IDs, bounds,
            angles, invalid shapes, or metadata mismatches.
    """
    if config is None:
        config = ObservationConfig()
    elif not isinstance(config, ObservationConfig):
        raise ValueError(f"config must be an ObservationConfig instance, got: {type(config).__name__}")

    if not isinstance(posture_summary, (dict, Mapping)):
        raise ValueError(f"posture_summary must be a dictionary/mapping, got {type(posture_summary).__name__}")

    if "events" not in posture_summary:
        raise ValueError("posture_summary must contain an 'events' list")

    raw_posture_events = posture_summary["events"]
    if not isinstance(raw_posture_events, (list, tuple)):
        raise ValueError(f"posture_summary['events'] must be a sequence, got {type(raw_posture_events).__name__}")

    # Validate and index supplementary attempt event telemetry by attempt_id if provided
    attempt_meta_by_id: Dict[int, Dict[str, Any]] = {}
    if attempt_events is not None:
        if isinstance(attempt_events, (dict, Mapping)) and "events" in attempt_events:
            event_seq = attempt_events["events"]
        elif isinstance(attempt_events, (list, tuple)):
            event_seq = attempt_events
        else:
            raise ValueError(
                f"attempt_events must be a sequence or dict containing 'events', got {type(attempt_events).__name__}"
            )

        seen_supp_ids: Set[int] = set()
        for idx, a_ev in enumerate(event_seq):
            if not isinstance(a_ev, (dict, Mapping)):
                raise ValueError(f"attempt_events[{idx}] must be a dict/mapping, got {type(a_ev).__name__}")
            if "attempt_id" not in a_ev:
                raise ValueError(f"attempt_events[{idx}] missing required 'attempt_id'")
            s_aid = _parse_int(a_ev["attempt_id"], f"attempt_events[{idx}].attempt_id")
            if s_aid < 1:
                raise ValueError(f"attempt_events[{idx}].attempt_id must be a positive integer, got: {s_aid}")
            if s_aid in seen_supp_ids:
                raise ValueError(f"Duplicate attempt_id {s_aid} in supplementary attempt_events at index {idx}")
            seen_supp_ids.add(s_aid)
            attempt_meta_by_id[s_aid] = dict(a_ev)

    processed_attempts: List[Dict[str, Any]] = []

    # Availability counters
    torso_avail_count = 0
    torso_unavail_count = 0
    hip_avail_count = 0
    hip_unavail_count = 0
    depth_avail_count = 0
    depth_unavail_count = 0

    # Grouping attempts by (track_id, side) for safe within-track/side comparisons
    # Key: (track_id, side) -> list of processed attempt dicts
    grouped_attempts: Dict[Tuple[int, str], List[Dict[str, Any]]] = {}
    seen_posture_ids: Set[int] = set()

    for idx, p_ev in enumerate(raw_posture_events):
        if not isinstance(p_ev, (dict, Mapping)):
            raise ValueError(f"posture_summary['events'][{idx}] must be a dict/mapping, got {type(p_ev).__name__}")

        # Strictly validate attempt_id (no fabricated fallback IDs)
        if "attempt_id" not in p_ev:
            raise ValueError(f"posture_summary['events'][{idx}] missing required 'attempt_id'")
        attempt_id = _parse_int(p_ev["attempt_id"], f"events[{idx}].attempt_id")
        if attempt_id < 1:
            raise ValueError(f"events[{idx}].attempt_id must be a positive integer, got: {attempt_id}")
        if attempt_id in seen_posture_ids:
            raise ValueError(f"Duplicate attempt_id {attempt_id} detected in posture_summary['events'] at index {idx}")
        seen_posture_ids.add(attempt_id)

        status = str(p_ev.get("status", "uncounted")).strip().lower()
        if status not in ("counted", "uncounted"):
            raise ValueError(f"events[{idx}].status must be 'counted' or 'uncounted', got: {status!r}")

        reason = str(p_ev.get("reason", "unknown")).strip()

        # Track and side identity
        raw_tid = p_ev.get("track_id")
        if raw_tid is not None and str(raw_tid).strip() != "":
            track_id = _parse_int(raw_tid, f"events[{idx}].track_id")
        else:
            track_id = None

        raw_side = p_ev.get("side") or p_ev.get("posture_side")
        if raw_side is not None and str(raw_side).strip() != "":
            clean_side = str(raw_side).strip().lower()
            side = clean_side if clean_side in ("left", "right") else "unknown"
        else:
            side = None

        # Frame boundaries validation
        if "start_frame" not in p_ev or "end_frame" not in p_ev:
            raise ValueError(f"events[{idx}] missing required 'start_frame' or 'end_frame'")
        start_frame = _parse_int(p_ev["start_frame"], f"events[{idx}].start_frame")
        end_frame = _parse_int(p_ev["end_frame"], f"events[{idx}].end_frame")
        if end_frame < start_frame:
            raise ValueError(f"events[{idx}] end_frame ({end_frame}) < start_frame ({start_frame})")

        expected_total_frames = end_frame - start_frame + 1
        if "total_frames" in p_ev and p_ev["total_frames"] is not None:
            declared_total = _parse_int(p_ev["total_frames"], f"events[{idx}].total_frames")
            if declared_total != expected_total_frames:
                raise ValueError(
                    f"events[{idx}] total_frames ({declared_total}) inconsistent with "
                    f"bounds [{start_frame}, {end_frame}] (expected {expected_total_frames})"
                )
        total_frames = expected_total_frames

        # Validate supplementary event metadata matching
        supp = attempt_meta_by_id.get(attempt_id)
        if supp is not None:
            # Identity consistency check
            if "track_id" in supp and supp["track_id"] is not None and str(supp["track_id"]).strip() != "":
                s_tid = _parse_int(supp["track_id"], f"attempt_events[{attempt_id}].track_id")
                if track_id is not None and s_tid != track_id:
                    raise ValueError(
                        f"Attempt {attempt_id} track_id mismatch between posture ({track_id}) and supplement ({s_tid})"
                    )
            if "side" in supp and supp["side"] is not None and str(supp["side"]).strip() != "":
                s_side = str(supp["side"]).strip().lower()
                if side is not None and side in ("left", "right") and s_side != side:
                    raise ValueError(
                        f"Attempt {attempt_id} side mismatch between posture ({side}) and supplement ({s_side})"
                    )
            if "start_frame" in supp:
                s_sf = _parse_int(supp["start_frame"], f"attempt_events[{attempt_id}].start_frame")
                if s_sf != start_frame:
                    raise ValueError(
                        f"Attempt {attempt_id} start_frame mismatch between posture ({start_frame}) and supplement ({s_sf})"
                    )
            if "end_frame" in supp:
                s_ef = _parse_int(supp["end_frame"], f"attempt_events[{attempt_id}].end_frame")
                if s_ef != end_frame:
                    raise ValueError(
                        f"Attempt {attempt_id} end_frame mismatch between posture ({end_frame}) and supplement ({s_ef})"
                    )

        if supp is not None:
            for identity_key, expected in (("status", status), ("reason", reason)):
                if identity_key in supp and supp[identity_key] != expected:
                    raise ValueError(f"Supplement {identity_key} does not match posture event")
        category = _categorize_event(status, reason)
        depth_confirmed = bool(supp.get("depth_confirmed", False)) if supp else False
        explanation_text = (supp.get("explanation") if supp else None) or ATTEMPT_REASON_EXPLANATIONS.get(
            reason, f"Movement status: {reason.replace('_', ' ')}."
        )

        # Torso inclination evaluation and strict validation
        t_dict = p_ev.get("torso_inclination") or p_ev.get("torso") or {}
        if not isinstance(t_dict, (dict, Mapping)):
            raise ValueError(f"events[{idx}] torso_inclination must be a mapping")

        t_count = _parse_int(t_dict.get("measured_count", 0), f"events[{idx}].torso.measured_count")
        if t_count > total_frames:
            raise ValueError(f"events[{idx}].torso.measured_count ({t_count}) exceeds total_frames ({total_frames})")

        raw_t_cov = t_dict.get("coverage_fraction", 0.0)
        if (
            isinstance(raw_t_cov, bool)
            or not isinstance(raw_t_cov, Real)
            or not math.isfinite(raw_t_cov)
            or not (0.0 <= raw_t_cov <= 1.0)
        ):
            raise ValueError(f"events[{idx}].torso.coverage_fraction must be a finite float in [0.0, 1.0], got: {raw_t_cov!r}")
        t_cov = float(raw_t_cov)
        if not math.isclose(t_cov, t_count / total_frames, abs_tol=1e-4):
            raise ValueError("Metric coverage disagrees with measured_count/total_frames")

        t_med = _parse_angle_metric(t_dict.get("median_deg"), f"events[{idx}].torso.median_deg")
        t_min = _parse_angle_metric(t_dict.get("min_deg"), f"events[{idx}].torso.min_deg")
        t_max = _parse_angle_metric(t_dict.get("max_deg"), f"events[{idx}].torso.max_deg")
        t_unavail = t_dict.get("unavailable_reasons", {})

        if t_min is not None and t_med is not None and t_max is not None:
            if not (t_min <= t_med <= t_max):
                raise ValueError(
                    f"events[{idx}] torso angles must be ordered min <= median <= max, "
                    f"got min={t_min}, med={t_med}, max={t_max}"
                )

        if (
            t_med is not None
            and t_min is not None
            and t_max is not None
            and t_count > 0
            and t_cov >= config.min_coverage_fraction
        ):
            torso_available = True
            torso_unavail_reason = None
            torso_avail_count += 1
        else:
            torso_available = False
            torso_unavail_count += 1
            if t_count == 0:
                reasons_str = _format_unavailable_reasons(t_unavail)
                torso_unavail_reason = f"no valid frames measured ({reasons_str})"
            elif t_cov < config.min_coverage_fraction:
                torso_unavail_reason = (
                    f"frame coverage {t_cov * 100:.1f}% below operational threshold "
                    f"{config.min_coverage_fraction * 100:.1f}%"
                )
            elif t_med is None or t_min is None or t_max is None:
                torso_unavail_reason = "incomplete angle distribution (missing min, median, or max)"
            else:
                torso_unavail_reason = "measurement unavailable"

        # Hip angle evaluation and strict validation
        h_dict = p_ev.get("hip_angle") or p_ev.get("hip") or {}
        if not isinstance(h_dict, (dict, Mapping)):
            raise ValueError(f"events[{idx}] hip_angle must be a mapping")

        h_count = _parse_int(h_dict.get("measured_count", 0), f"events[{idx}].hip.measured_count")
        if h_count > total_frames:
            raise ValueError(f"events[{idx}].hip.measured_count ({h_count}) exceeds total_frames ({total_frames})")

        raw_h_cov = h_dict.get("coverage_fraction", 0.0)
        if (
            isinstance(raw_h_cov, bool)
            or not isinstance(raw_h_cov, Real)
            or not math.isfinite(raw_h_cov)
            or not (0.0 <= raw_h_cov <= 1.0)
        ):
            raise ValueError(f"events[{idx}].hip.coverage_fraction must be a finite float in [0.0, 1.0], got: {raw_h_cov!r}")
        h_cov = float(raw_h_cov)
        if not math.isclose(h_cov, h_count / total_frames, abs_tol=1e-4):
            raise ValueError("Metric coverage disagrees with measured_count/total_frames")

        h_med = _parse_angle_metric(h_dict.get("median_deg"), f"events[{idx}].hip.median_deg")
        h_min = _parse_angle_metric(h_dict.get("min_deg"), f"events[{idx}].hip.min_deg")
        h_max = _parse_angle_metric(h_dict.get("max_deg"), f"events[{idx}].hip.max_deg")
        h_unavail = h_dict.get("unavailable_reasons", {})

        if h_min is not None and h_med is not None and h_max is not None:
            if not (h_min <= h_med <= h_max):
                raise ValueError(
                    f"events[{idx}] hip angles must be ordered min <= median <= max, "
                    f"got min={h_min}, med={h_med}, max={h_max}"
                )

        if (
            h_med is not None
            and h_min is not None
            and h_max is not None
            and h_count > 0
            and h_cov >= config.min_coverage_fraction
        ):
            hip_available = True
            hip_unavail_reason = None
            hip_avail_count += 1
        else:
            hip_available = False
            hip_unavail_count += 1
            if h_count == 0:
                reasons_str = _format_unavailable_reasons(h_unavail)
                hip_unavail_reason = f"no valid frames measured ({reasons_str})"
            elif h_cov < config.min_coverage_fraction:
                hip_unavail_reason = (
                    f"frame coverage {h_cov * 100:.1f}% below operational threshold "
                    f"{config.min_coverage_fraction * 100:.1f}%"
                )
            elif h_med is None or h_min is None or h_max is None:
                hip_unavail_reason = "incomplete angle distribution (missing min, median, or max)"
            else:
                hip_unavail_reason = "measurement unavailable"

        # Movement depth (knee angle) evaluation
        # Depth is only considered reliable for comparison when actual attempt metadata
        # valid_frames/observed_frames demonstrates adequate coverage.
        raw_knee_depth = (supp.get("min_knee_angle_deg") if supp else None)
        if raw_knee_depth is None:
            raw_knee_depth = p_ev.get("min_knee_angle_deg")
        min_knee_angle = _parse_angle_metric(raw_knee_depth, f"events[{idx}].min_knee_angle_deg")

        if supp is not None and "valid_frames" in supp and "observed_frames" in supp:
            v_frames = _parse_int(supp["valid_frames"], f"attempt_events[{attempt_id}].valid_frames")
            o_frames = _parse_int(supp["observed_frames"], f"attempt_events[{attempt_id}].observed_frames")
            if not (0 <= v_frames <= o_frames <= total_frames):
                raise ValueError(
                    f"Attempt {attempt_id} frame counts inconsistent: valid_frames ({v_frames}) "
                    f"must be <= observed_frames ({o_frames}) <= total_frames ({total_frames})"
                )
            knee_cov = v_frames / total_frames
            if min_knee_angle is not None and knee_cov >= config.min_coverage_fraction:
                depth_available = True
                depth_unavail_reason = None
                depth_avail_count += 1
            else:
                depth_available = False
                depth_unavail_count += 1
                if min_knee_angle is not None:
                    depth_unavail_reason = (
                        f"unverified: valid frame coverage {knee_cov * 100:.1f}% "
                        f"below operational threshold {config.min_coverage_fraction * 100:.1f}%"
                    )
                else:
                    depth_unavail_reason = "minimum knee flexion angle not recorded during attempt"
        else:
            # Without attempt metadata valid/observed frames, do not invent coverage from torso or defaults
            depth_available = False
            depth_unavail_count += 1
            if min_knee_angle is not None:
                depth_unavail_reason = "unverified: attempt metadata lacks verified valid_frames/observed_frames coverage"
            else:
                depth_unavail_reason = "minimum knee flexion angle not recorded during attempt"

        # Construct attempt-level observations
        attempt_obs: List[str] = []

        if torso_available and t_med is not None and t_min is not None and t_max is not None:
            attempt_obs.append(
                f"Torso inclination relative to upward vertical was median {t_med:.2f}° "
                f"(observed range {t_min:.2f}°–{t_max:.2f}°) across {t_count}/{total_frames} frames "
                f"({round(t_cov * 100, 1)}% coverage)."
            )
        else:
            attempt_obs.append(f"Torso inclination observation unavailable: {torso_unavail_reason}.")

        if hip_available and h_med is not None and h_min is not None and h_max is not None:
            attempt_obs.append(
                f"Hip joint angle was median {h_med:.2f}° "
                f"(observed range {h_min:.2f}°–{h_max:.2f}°) across {h_count}/{total_frames} frames "
                f"({round(h_cov * 100, 1)}% coverage)."
            )
        else:
            attempt_obs.append(f"Hip angle observation unavailable: {hip_unavail_reason}.")

        if depth_available and min_knee_angle is not None:
            attempt_obs.append(f"Minimum observed projected knee angle reached {min_knee_angle:.2f}°.")
        elif min_knee_angle is not None:
            attempt_obs.append(f"Movement depth observation unverified ({min_knee_angle:.2f}° recorded): {depth_unavail_reason}.")
        else:
            attempt_obs.append(f"Movement depth observation unavailable: {depth_unavail_reason}.")

        context_note = _generate_context_note(status, category, reason)

        attempt_dict: Dict[str, Any] = {
            "attempt_id": attempt_id,
            "status": status,
            "reason": reason,
            "track_id": track_id,
            "side": side,
            "is_counted": (status == "counted"),
            "start_frame": start_frame,
            "end_frame": end_frame,
            "total_frames": total_frames,
            "availability": {
                "torso": {
                    "available": torso_available,
                    "coverage_fraction": round(t_cov, 4),
                    "measured_frames": t_count,
                    "total_frames": total_frames,
                    "reason": torso_unavail_reason,
                },
                "hip": {
                    "available": hip_available,
                    "coverage_fraction": round(h_cov, 4),
                    "measured_frames": h_count,
                    "total_frames": total_frames,
                    "reason": hip_unavail_reason,
                },
                "movement_depth": {
                    "available": depth_available,
                    "min_knee_angle_deg": round(min_knee_angle, CSV_PRECISION_DECIMALS)
                    if min_knee_angle is not None
                    else None,
                    "reason": depth_unavail_reason,
                },
            },
            "evidence": {
                "torso_median_deg": round(t_med, CSV_PRECISION_DECIMALS) if t_med is not None else None,
                "torso_min_deg": round(t_min, CSV_PRECISION_DECIMALS) if t_min is not None else None,
                "torso_max_deg": round(t_max, CSV_PRECISION_DECIMALS) if t_max is not None else None,
                "hip_median_deg": round(h_med, CSV_PRECISION_DECIMALS) if h_med is not None else None,
                "hip_min_deg": round(h_min, CSV_PRECISION_DECIMALS) if h_min is not None else None,
                "hip_max_deg": round(h_max, CSV_PRECISION_DECIMALS) if h_max is not None else None,
                "min_knee_angle_deg": round(min_knee_angle, CSV_PRECISION_DECIMALS)
                if min_knee_angle is not None
                else None,
                "depth_confirmed": depth_confirmed,
                "category": category,
                "start_frame": start_frame,
                "end_frame": end_frame,
                "total_frames": total_frames,
            },
            "movement_explanation": {
                "category": category,
                "primary_reason": reason,
                "explanation": explanation_text,
                "context_note": context_note,
            },
            "observations": attempt_obs,
        }

        processed_attempts.append(attempt_dict)

        # Only group when track_id is valid integer and side is explicitly 'left' or 'right'
        # Unknown/None side must NEVER become a comparison group
        if track_id is not None and side in ("left", "right"):
            group_key = (track_id, side)
            grouped_attempts.setdefault(group_key, []).append(attempt_dict)

    # Cross-attempt comparisons within matching (track_id, side) groups
    comparisons: List[Dict[str, Any]] = []
    session_observations: List[str] = []

    total_attempts_count = len(processed_attempts)
    counted_count = sum(1 for a in processed_attempts if a["is_counted"])
    uncounted_count = total_attempts_count - counted_count

    if total_attempts_count == 0:
        session_observations.append("No discrete movement attempts were observed in this session.")
    elif total_attempts_count == 1:
        session_observations.append(
            "Session contains only 1 observed movement attempt. "
            "Cross-attempt consistency observations cannot be evaluated from a single attempt."
        )

    # Note if attempts occurred across multiple tracks or body sides
    if len(grouped_attempts) > 1:
        views_str = ", ".join(f"track {tid} ({s} side)" for tid, s in sorted(grouped_attempts.keys()))
        session_observations.append(
            f"Observed attempts occurred across multiple tracking contexts or body sides: {views_str}. "
            "Kinematic measurements are never compared across different sides or subjects."
        )

    for (grp_tid, grp_side), grp_list in sorted(grouped_attempts.items()):
        grp_obs: List[str] = []
        grp_limits: List[str] = []

        grp_attempt_ids = [a["attempt_id"] for a in grp_list]
        counted_in_grp = [a for a in grp_list if a["is_counted"]]
        uncounted_in_grp = [a for a in grp_list if not a["is_counted"]]

        if uncounted_in_grp:
            grp_limits.append(
                f"{len(uncounted_in_grp)} uncounted movement(s) on track {grp_tid} ({grp_side} side) "
                "excluded from cross-attempt consistency comparisons."
            )

        # 1. Torso inclination comparison: ONLY reliably measured COUNTED cycles
        torso_eligible = [
            a
            for a in counted_in_grp
            if a["availability"]["torso"]["available"] and a["evidence"]["torso_median_deg"] is not None
        ]

        torso_cmp: Dict[str, Any] = {
            "available": False,
            "attempt_ids": [a["attempt_id"] for a in torso_eligible],
            "min_median_deg": None,
            "max_median_deg": None,
            "spread_deg": None,
            "group_median_deg": None,
        }

        if len(torso_eligible) >= config.min_attempts_for_comparison:
            t_medians = [a["evidence"]["torso_median_deg"] for a in torso_eligible if a["evidence"]["torso_median_deg"] is not None]
            min_t = round(float(min(t_medians)), CSV_PRECISION_DECIMALS)
            max_t = round(float(max(t_medians)), CSV_PRECISION_DECIMALS)
            spread_t = round(max_t - min_t, CSV_PRECISION_DECIMALS)
            grp_t_med = round(float(statistics.median(t_medians)), CSV_PRECISION_DECIMALS)

            torso_cmp.update(
                available=True,
                min_median_deg=min_t,
                max_median_deg=max_t,
                spread_deg=spread_t,
                group_median_deg=grp_t_med,
            )

            grp_obs.append(
                f"Across {len(torso_eligible)} counted cycles on track {grp_tid} ({grp_side} side), "
                f"median torso inclination relative to vertical differed by {spread_t:.2f}° "
                f"(ranging from {min_t:.2f}° to {max_t:.2f}°, group median {grp_t_med:.2f}°)."
            )
        elif len(torso_eligible) == 1:
            grp_limits.append(
                f"Only 1 counted cycle on track {grp_tid} ({grp_side} side) met torso reliability criteria; "
                f"evaluating consistency requires at least {config.min_attempts_for_comparison} counted cycles."
            )
        else:
            grp_limits.append(
                f"No counted cycles on track {grp_tid} ({grp_side} side) had reliable torso measurement coverage."
            )

        # 2. Movement depth (knee angle) comparison: ONLY reliably measured COUNTED cycles
        depth_eligible = [
            a
            for a in counted_in_grp
            if a["availability"]["movement_depth"]["available"] and a["evidence"]["min_knee_angle_deg"] is not None
        ]

        depth_cmp: Dict[str, Any] = {
            "available": False,
            "attempt_ids": [a["attempt_id"] for a in depth_eligible],
            "min_knee_angle_deg": None,
            "max_knee_angle_deg": None,
            "spread_deg": None,
            "group_median_deg": None,
        }

        if len(depth_eligible) >= config.min_attempts_for_comparison:
            k_angles = [a["evidence"]["min_knee_angle_deg"] for a in depth_eligible if a["evidence"]["min_knee_angle_deg"] is not None]
            min_k = round(float(min(k_angles)), CSV_PRECISION_DECIMALS)
            max_k = round(float(max(k_angles)), CSV_PRECISION_DECIMALS)
            spread_k = round(max_k - min_k, CSV_PRECISION_DECIMALS)
            grp_k_med = round(float(statistics.median(k_angles)), CSV_PRECISION_DECIMALS)

            depth_cmp.update(
                available=True,
                min_knee_angle_deg=min_k,
                max_knee_angle_deg=max_k,
                spread_deg=spread_k,
                group_median_deg=grp_k_med,
            )

            grp_obs.append(
                f"Across {len(depth_eligible)} counted cycles on track {grp_tid} ({grp_side} side), "
                f"minimum observed projected knee angle varied by {spread_k:.2f}° "
                f"(ranging from {min_k:.2f}° to {max_k:.2f}°, group median {grp_k_med:.2f}°)."
            )
        elif len(depth_eligible) == 1:
            grp_limits.append(
                f"Only 1 counted cycle on track {grp_tid} ({grp_side} side) met depth reliability criteria; "
                f"evaluating depth consistency requires at least {config.min_attempts_for_comparison} counted cycles."
            )
        else:
            grp_limits.append(
                f"No counted cycles on track {grp_tid} ({grp_side} side) had reliable movement depth."
            )

        comparisons.append(
            {
                "track_id": grp_tid,
                "side": grp_side,
                "attempt_ids": grp_attempt_ids,
                "total_attempts_in_group": len(grp_list),
                "counted_attempts_in_group": len(counted_in_grp),
                "uncounted_attempts_in_group": len(uncounted_in_grp),
                "torso_inclination": torso_cmp,
                "movement_depth": depth_cmp,
                "observations": grp_obs,
                "limitations": grp_limits,
            }
        )

    # Attach group-relative comparisons into per-attempt observations for counted cycles
    for cmp_info in comparisons:
        grp_tid = cmp_info["track_id"]
        grp_side = cmp_info["side"]
        t_cmp = cmp_info["torso_inclination"]

        if t_cmp["available"] and t_cmp["group_median_deg"] is not None:
            for a in processed_attempts:
                if (
                    a["is_counted"]
                    and a["track_id"] == grp_tid
                    and a["side"] == grp_side
                    and a["availability"]["torso"]["available"]
                ):
                    a_med = a["evidence"]["torso_median_deg"]
                    if a_med is not None:
                        diff = round(abs(a_med - t_cmp["group_median_deg"]), CSV_PRECISION_DECIMALS)
                        a["observations"].append(
                            f"Torso inclination median was {diff:.2f}° from this session's "
                            f"counted group median ({t_cmp['group_median_deg']:.2f}° on track {grp_tid}, {grp_side} side)."
                        )

    return {
        "summary": {
            "total_attempt_events": total_attempts_count,
            "counted_events": counted_count,
            "uncounted_events": uncounted_count,
            "session_observations": session_observations,
            "comparisons": comparisons,
            "availability_summary": {
                "torso_available_attempts": torso_avail_count,
                "torso_unavailable_attempts": torso_unavail_count,
                "hip_available_attempts": hip_avail_count,
                "hip_unavailable_attempts": hip_unavail_count,
                "depth_available_attempts": depth_avail_count,
                "depth_unavailable_attempts": depth_unavail_count,
            },
        },
        "attempts": processed_attempts,
        "configuration": {
            "min_coverage_fraction": config.min_coverage_fraction,
            "min_attempts_for_comparison": config.min_attempts_for_comparison,
            "telemetry_precision_deg": 10**-CSV_PRECISION_DECIMALS,
            "interpretation": (
                "Operational telemetry filtering thresholds for reliable aggregation; "
                "not clinical or biomechanical form standards."
            ),
        },
        "scope_limitations": list(SCOPE_LIMITATIONS),
    }


# Convenience alias for American English callers
summarize_feedback_observations = build_feedback_observations


def build_feedback_observations_from_raw(
    frame_rows: Sequence[Dict[str, Any]],
    attempt_events: Union[Sequence[Dict[str, Any]], Dict[str, Any]],
    config: Optional[ObservationConfig] = None,
) -> Dict[str, Any]:
    """Convenience helper computing posture summaries from raw CSV rows and generating observations."""
    try:
        from .session_feedback import summarise_attempt_posture
    except (ImportError, ValueError):
        from gym_analyser.session_feedback import summarise_attempt_posture

    posture_summary = summarise_attempt_posture(frame_rows, attempt_events)
    return build_feedback_observations(
        posture_summary=posture_summary,
        attempt_events=attempt_events,
        config=config,
    )
