# -*- coding: utf-8 -*-
"""Phase-localized movement feedback module for gym exercise analysis.

Provides pure Python standard-library localization of 2D kinematic observations
to movement phases (descent and return) separated by the observed minimum knee
flexion turnaround candidate, without normative grading or unvalidated clinical claims.
"""

from dataclasses import dataclass
import math
from numbers import Real
import statistics
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

try:
    from .session_feedback import (
        _parse_angle,
        _parse_int,
        _validate_events,
        _validate_frame_rows,
    )
except (ImportError, ValueError):
    from gym_analyser.session_feedback import (
        _parse_angle,
        _parse_int,
        _validate_events,
        _validate_frame_rows,
    )

CSV_PRECISION_DECIMALS = 2

def _valid_angle(value):
    angle = _parse_angle(value)
    return angle if angle is not None and 0.0 <= angle <= 180.0 else None


LIMITATIONS = [
    "Kinematics reflect projected 2D coordinates from a monocular lateral view.",
    "Observed minimum knee angle is a turnaround candidate, not true anatomical bottom or derivative inflection.",
    "Torso inclination is an unsigned 2D angle from upward vertical; cannot differentiate forward/backward lean.",
    "Frontal/transverse plane motions (e.g. knee valgus, hip shift) are not observable from side view.",
    "Temporal review cues are attention guides for video inspection, not evidence of wrong form.",
    "Uncounted attempts (setup, bar adjustment, weight return) are excluded from repetition consistency.",
]


@dataclass(frozen=True)
class PhaseFeedbackConfig:
    """Operational parameters for phase-localized feedback analysis."""

    min_phase_samples: int = 3
    max_gap_seconds: float = 0.5
    min_coverage_fraction: float = 0.5

    def __post_init__(self) -> None:
        if isinstance(self.min_phase_samples, bool) or not isinstance(self.min_phase_samples, int) or self.min_phase_samples < 1:
            raise ValueError("min_phase_samples must be an integer >= 1")
        if isinstance(self.max_gap_seconds, bool) or not isinstance(self.max_gap_seconds, Real) or not (math.isfinite(self.max_gap_seconds) and self.max_gap_seconds > 0.0):
            raise ValueError("max_gap_seconds must be a finite float > 0")
        if isinstance(self.min_coverage_fraction, bool) or not isinstance(self.min_coverage_fraction, Real) or not (math.isfinite(self.min_coverage_fraction) and 0.0 <= self.min_coverage_fraction <= 1.0):
            raise ValueError("min_coverage_fraction must be a finite float in [0.0, 1.0]")


def _parse_float(val: Any) -> Optional[float]:
    if val is None or isinstance(val, bool):
        return None
    try:
        f = float(str(val).strip()) if isinstance(val, str) else float(val)
        return f if math.isfinite(f) else None
    except (ValueError, TypeError):
        return None


def _parse_bool(val: Any) -> bool:
    return val if isinstance(val, bool) else str(val).strip().lower() in ("true", "1", "yes")


def _compute_phase_stats(rows: List[Mapping[str, Any]], target_side: str) -> Dict[str, Any]:
    valid_torso: List[Tuple[int, Optional[float], float]] = []
    for r in rows:
        p_side = str(r.get("posture_side") or "").strip().lower()
        t_reason = str(r.get("torso_reason") or "").strip().lower()
        if p_side == target_side and t_reason == "ok":
            angle = _valid_angle(r.get("torso_inclination_deg"))
            if angle is not None:
                f_idx = _parse_int(r["frame_index"], "f")
                t_val = _parse_float(r.get("video_time_s"))
                valid_torso.append((f_idx, t_val, angle))

    sample_count, total_frames = len(valid_torso), len(rows)
    coverage = round(sample_count / total_frames, 4) if total_frames > 0 else 0.0
    torso_vals = [entry[2] for entry in valid_torso]

    phase_start_f = _parse_int(rows[0]["frame_index"], "start_f") if rows else None
    phase_end_f = _parse_int(rows[-1]["frame_index"], "end_f") if rows else None
    phase_start_t = _parse_float(rows[0].get("video_time_s")) if rows else None
    phase_end_t = _parse_float(rows[-1].get("video_time_s")) if rows else None

    if len(valid_torso) >= 2:
        d_start_f, d_start_t = valid_torso[0][0], valid_torso[0][1]
        d_start_deg = round(valid_torso[0][2], CSV_PRECISION_DECIMALS)
        d_end_f, d_end_t = valid_torso[-1][0], valid_torso[-1][1]
        d_end_deg = round(valid_torso[-1][2], CSV_PRECISION_DECIMALS)
        delta_deg = round(d_end_deg - d_start_deg, CSV_PRECISION_DECIMALS)
    else:
        d_start_f = d_start_t = d_start_deg = d_end_f = d_end_t = d_end_deg = delta_deg = None

    return {
        "sample_count": sample_count, "total_frames": total_frames, "coverage_fraction": coverage,
        "min_deg": round(min(torso_vals), CSV_PRECISION_DECIMALS) if torso_vals else None,
        "max_deg": round(max(torso_vals), CSV_PRECISION_DECIMALS) if torso_vals else None,
        "median_deg": round(statistics.median(torso_vals), CSV_PRECISION_DECIMALS) if torso_vals else None,
        "delta_deg": delta_deg, "delta_start_deg": d_start_deg, "delta_end_deg": d_end_deg,
        "delta_start_frame": d_start_f, "delta_end_frame": d_end_f,
        "delta_start_time_s": d_start_t, "delta_end_time_s": d_end_t,
        "phase_start_frame": phase_start_f, "phase_end_frame": phase_end_f,
        "phase_start_time_s": phase_start_t, "phase_end_time_s": phase_end_t,
    }


def _process_attempt(
    ev: Mapping[str, Any],
    frame_by_index: Dict[int, Mapping[str, Any]],
    total_video_rows: int,
    cfg: PhaseFeedbackConfig,
) -> Dict[str, Any]:
    raw_aid = ev.get("attempt_id")
    aid = _parse_int(raw_aid, "event.attempt_id") if raw_aid is not None and str(raw_aid).strip() != "" else None
    raw_tid = ev.get("track_id")
    target_track = _parse_int(raw_tid, "event.track_id") if raw_tid is not None and str(raw_tid).strip() != "" else None
    raw_side = ev.get("side") or ev.get("posture_side")
    target_side = str(raw_side).strip().lower() if raw_side is not None and str(raw_side).strip() != "" else None

    sf, ef = _parse_int(ev["start_frame"], "start_frame"), _parse_int(ev["end_frame"], "end_frame")
    status, reason = ev.get("status", "uncounted"), ev.get("reason", "unknown")

    base: Dict[str, Any] = {
        "attempt_id": aid, "status": status, "reason": reason, "explanation": ev.get("explanation", ""),
        "track_id": target_track, "side": target_side, "start_frame": sf, "end_frame": ef,
        "phase_analysis_available": False, "suppression_reason": None, "min_knee_point": None,
        "descent_phase": None, "return_phase": None, "phase_difference": None,
        "review_cues": [], "context_note": None, "relative_depth": None,
    }

    if status != "counted":
        base["suppression_reason"] = "uncounted_attempt"
        base["context_note"] = (
            f"Observed movement was not a counted cycle ({reason}); the system cannot determine exercise intent "
            "(may represent setup, bar racking, or incomplete cycle). Excluded from repetition consistency."
        )
        return base

    if aid is None or target_track is None or target_side not in ("left", "right"):
        base["suppression_reason"] = "missing_event_identity"
        return base

    expected_count = ef - sf + 1
    if expected_count > total_video_rows:
        base["suppression_reason"] = "frame_gap_detected"
        return base

    attempt_rows, prev_t = [], None
    for f in range(sf, ef + 1):
        if f not in frame_by_index:
            base["suppression_reason"] = "frame_gap_detected"
            return base
        r = frame_by_index[f]
        r_tid = _parse_int(r.get("track_id"), "row.track_id") if r.get("track_id") not in (None, "") else None
        if r_tid != target_track:
            base["suppression_reason"] = "track_mismatch"
            return base
        r_sel_side = str(r.get("selected_side") or "").strip().lower() or None
        if r_sel_side != target_side:
            base["suppression_reason"] = "side_mismatch"
            return base
        if not _parse_bool(r.get("valid")) or _valid_angle(r.get("knee_angle_deg")) is None:
            base["suppression_reason"] = "tracking_gap_exceeded"
            return base
        vt = _parse_float(r.get("video_time_s"))
        if vt is not None:
            if prev_t is not None and (vt <= prev_t or (vt - prev_t) > cfg.max_gap_seconds):
                base["suppression_reason"] = "tracking_gap_exceeded"
                return base
            prev_t = vt
        attempt_rows.append(r)

    min_row = min(attempt_rows, key=lambda x: _valid_angle(x.get("knee_angle_deg")))  # type: ignore
    min_f, min_t = _parse_int(min_row["frame_index"], "min_frame"), _parse_float(min_row.get("video_time_s"))
    min_k = _valid_angle(min_row.get("knee_angle_deg"))
    if min_f == sf or min_f == ef:
        base["suppression_reason"] = "single_phase_only"
        return base

    d_rows = [r for r in attempt_rows if _parse_int(r["frame_index"], "f") <= min_f]
    r_rows = [r for r in attempt_rows if _parse_int(r["frame_index"], "f") >= min_f]
    d_stats = _compute_phase_stats(d_rows, target_side)
    r_stats = _compute_phase_stats(r_rows, target_side)

    if d_stats["sample_count"] < cfg.min_phase_samples or r_stats["sample_count"] < cfg.min_phase_samples:
        base["suppression_reason"] = "insufficient_phase_samples"
        return base
    if d_stats["coverage_fraction"] < cfg.min_coverage_fraction or r_stats["coverage_fraction"] < cfg.min_coverage_fraction:
        base["suppression_reason"] = "insufficient_coverage"
        return base

    base.update({
        "phase_analysis_available": True,
        "min_knee_point": {
            "frame_index": min_f, "video_time_s": min_t,
            "knee_angle_deg": round(min_k, CSV_PRECISION_DECIMALS),  # type: ignore
            "interpretation": "Observed minimum/turnaround candidate; not claimed true anatomical bottom or mathematical inflection.",
        },
        "descent_phase": d_stats, "return_phase": r_stats,
    })
    if r_stats["median_deg"] is not None and d_stats["median_deg"] is not None:
        base["phase_difference"] = {"torso_median_delta_deg": round(r_stats["median_deg"] - d_stats["median_deg"], CSV_PRECISION_DECIMALS)}

    t_str = f" ({min_t:.2f}s)" if min_t is not None else ""
    base["review_cues"].append(
        f"Observed minimum/turnaround candidate of {min_k:.2f}° located at frame {min_f}{t_str}; review video around this turnaround point."
    )
    if r_stats["delta_deg"] is not None:
        st_val, et_val = r_stats["delta_start_time_s"], r_stats["delta_end_time_s"]
        time_part = f" ({st_val:.2f}s–{et_val:.2f}s)" if st_val is not None and et_val is not None else ""
        base["review_cues"].append(
            f"Torso inclination changed by {abs(r_stats['delta_deg']):.2f}° "
            f"(from {r_stats['delta_start_deg']:.2f}° to {r_stats['delta_end_deg']:.2f}°) "
            f"during the observed return phase; review measured frames {r_stats['delta_start_frame']}–{r_stats['delta_end_frame']}{time_part} "
            f"across {r_stats['sample_count']} valid frames (phase interval: frames {r_stats['phase_start_frame']}–{r_stats['phase_end_frame']})."
        )
    return base


def build_phase_feedback(
    frame_rows: Sequence[Mapping[str, Any]],
    attempt_events: Union[Sequence[Mapping[str, Any]], Mapping[str, Any]],
    config: Optional[PhaseFeedbackConfig] = None,
) -> Dict[str, Any]:
    """Build reviewable phase-localized movement feedback from frame rows and attempt events."""
    cfg = config if config is not None else PhaseFeedbackConfig()
    if not isinstance(cfg, PhaseFeedbackConfig):
        raise ValueError(f"config must be a PhaseFeedbackConfig instance, got: {type(cfg).__name__}")

    rows = _validate_frame_rows(frame_rows)
    events = _validate_events(attempt_events)
    frame_by_index = {_parse_int(r["frame_index"], "row.frame_index"): r for r in rows}
    total_video_rows = len(rows)

    processed = [_process_attempt(ev, frame_by_index, total_video_rows, cfg) for ev in events]

    groups: Dict[Tuple[int, str], List[Dict[str, Any]]] = {}
    for a in processed:
        if a["phase_analysis_available"] and a["track_id"] is not None and a["side"] in ("left", "right"):
            groups.setdefault((a["track_id"], a["side"]), []).append(a)

    comparisons: List[Dict[str, Any]] = []
    for (tid, side), grp in groups.items():
        if len(grp) >= 2:
            deepest = min(grp, key=lambda x: x["min_knee_point"]["knee_angle_deg"])
            d_knee, d_id = deepest["min_knee_point"]["knee_angle_deg"], deepest["attempt_id"]
            d_frame, d_time = deepest["min_knee_point"]["frame_index"], deepest["min_knee_point"]["video_time_s"]
            for a in grp:
                a_knee = a["min_knee_point"]["knee_angle_deg"]
                a_frame, a_time = a["min_knee_point"]["frame_index"], a["min_knee_point"]["video_time_s"]
                diff = round(a_knee - d_knee, CSV_PRECISION_DECIMALS)
                is_shallower = diff > 0.0
                a["relative_depth"] = {
                    "is_relatively_shallower": is_shallower, "difference_deg": diff,
                    "compared_to_attempt_id": d_id, "reference_min_knee_deg": d_knee,
                }
                if is_shallower:
                    a_t_str = f" ({a_time:.2f}s)" if a_time is not None else ""
                    d_t_str = f" ({d_time:.2f}s)" if d_time is not None else ""
                    cue = (
                        f"Observed minimum knee angle was {a_knee:.2f}° at frame {a_frame}{a_t_str}, "
                        f"which was relatively shallower (+{diff:.2f}°) than cycle {d_id} ({d_knee:.2f}° at frame {d_frame}{d_t_str}) "
                        f"within the same track {tid}, {side} side context. "
                        "Temporal review cues are attention guides, not evidence of wrong form."
                    )
                    a["review_cues"].append(cue)
                    comparisons.append({"attempt_id": a["attempt_id"], "track_id": tid, "side": side, "observation": cue})

    counted_cnt = sum(1 for e in events if e.get("status") == "counted")
    return {
        "summary": {
            "total_attempt_events": len(events), "counted_events": counted_cnt,
            "uncounted_events": len(events) - counted_cnt,
            "phase_analyzed_events": sum(1 for a in processed if a["phase_analysis_available"]),
            "comparisons": comparisons,
        },
        "attempts": processed,
        "configuration": {
            "min_phase_samples": cfg.min_phase_samples, "max_gap_seconds": cfg.max_gap_seconds,
            "min_coverage_fraction": cfg.min_coverage_fraction,
            "interpretation": "Operational telemetry filters; not biomechanical or clinical technique thresholds.",
        },
        "biomechanical_limitations": list(LIMITATIONS),
    }


summarize_phase_feedback = build_phase_feedback
summarise_phase_feedback = build_phase_feedback
