"""Standalone per-attempt posture summary and session feedback module.

Provides pure Python standard-library summarisation of 2D kinematic posture
metrics (torso inclination from upward vertical and projected hip joint angle)
for discrete exercise attempt events.

Accepts exported CSV row mappings (e.g. from angles.csv) and attempt event
mappings (e.g. from attempts.json). Performs rigorous input validation,
track/side consistency filtering, and produces neutral descriptive telemetry
without normative form judgements or unvalidated clinical safety claims.
"""

import csv
import json
import math
import statistics
from decimal import Decimal, InvalidOperation
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

# CSV angle formatting precision in the upstream pipeline
CSV_ANGLE_PRECISION_DECIMALS = 2


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


def _parse_angle(val: Any) -> Optional[float]:
    """Parse an angle value from CSV string, float, or None.

    A measured 0.0 degrees is a valid angle (e.g. perfectly vertical torso).
    Blanks, empty strings, None, and non-finite values return None.
    Booleans return None.
    """
    if val is None:
        return None
    if isinstance(val, bool):
        return None
    if isinstance(val, (int, float)):
        try:
            return float(val) if math.isfinite(val) else None
        except OverflowError:
            return None
    if isinstance(val, str):
        cleaned = val.strip()
        if not cleaned or cleaned.lower() in ("none", "nan", "null"):
            return None
        try:
            parsed = float(cleaned)
            if not math.isfinite(parsed):
                return None
            return parsed
        except ValueError:
            return None
    return None


def _validate_frame_rows(frame_rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Validate structure, finiteness, and strictly increasing order of frame rows."""
    if not isinstance(frame_rows, (list, tuple)):
        raise ValueError(
            f"frame_rows must be a sequence of mappings, got {type(frame_rows).__name__}"
        )

    prev_frame: Optional[int] = None
    previous_time = None
    for idx, row in enumerate(frame_rows):
        if not isinstance(row, (dict, Mapping)):
            raise ValueError(
                f"frame_rows[{idx}] must be a dictionary/mapping, got {type(row).__name__}"
            )
        if "frame_index" not in row:
            raise ValueError(f"frame_rows[{idx}] is missing required 'frame_index'")

        frame_idx = _parse_int(row["frame_index"], f"frame_rows[{idx}].frame_index")
        if prev_frame is not None:
            if frame_idx == prev_frame:
                raise ValueError(f"Duplicate frame_index {frame_idx} at row {idx}")
            if frame_idx < prev_frame:
                raise ValueError(f"Descending frame_index: {frame_idx} < {prev_frame} at row {idx}")
        prev_frame = frame_idx

        for key in ("track_id", "attempt_id"):
            value = row.get(key)
            if value is not None and str(value).strip():
                _parse_int(value, f"frame_rows[{idx}].{key}")
        validity = row.get("valid")
        if validity is not None and not isinstance(validity, bool) and str(validity).strip().lower() not in {"", "true", "false"}:
            raise ValueError(f"frame_rows[{idx}].valid must be a boolean or CSV True/False")
        # Knee validity never gates independently available torso/hip measurements.
        # Validate finiteness of video_time_s if present
        if "video_time_s" in row:
            vt = row["video_time_s"]
            if vt is not None and str(vt).strip() != "":
                if isinstance(vt, bool):
                    raise ValueError(f"frame_rows[{idx}].video_time_s cannot be boolean")
                try:
                    vt_val = float(vt)
                except (ValueError, TypeError, OverflowError):
                    raise ValueError(
                        f"frame_rows[{idx}].video_time_s must be a valid float, got {vt!r}"
                    )
                if not math.isfinite(vt_val) or vt_val < 0.0:
                    raise ValueError(
                        f"frame_rows[{idx}].video_time_s must be finite and non-negative, got {vt!r}"
                    )

                if previous_time is not None and vt_val < previous_time:
                    raise ValueError("Descending video_time_s")
                previous_time = vt_val

    return list(frame_rows)


def _validate_events(attempt_events: Any) -> List[Dict[str, Any]]:
    """Validate event intervals: ordered, non-overlapping, and end_frame >= start_frame."""
    if isinstance(attempt_events, dict) and "events" in attempt_events:
        attempt_events = attempt_events["events"]

    if not isinstance(attempt_events, (list, tuple)):
        raise ValueError(
            f"attempt_events must be a sequence of mappings, got {type(attempt_events).__name__}"
        )

    prev_end: Optional[int] = None
    prev_start: Optional[int] = None
    validated = []
    for idx, ev in enumerate(attempt_events):
        if not isinstance(ev, (dict, Mapping)):
            raise ValueError(
                f"attempt_events[{idx}] must be a dict/mapping, got {type(ev).__name__}"
            )
        if "start_frame" not in ev or "end_frame" not in ev:
            raise ValueError(
                f"attempt_events[{idx}] missing required 'start_frame' or 'end_frame'"
            )

        sf = _parse_int(ev["start_frame"], f"attempt_events[{idx}].start_frame")
        ef = _parse_int(ev["end_frame"], f"attempt_events[{idx}].end_frame")

        if ef < sf:
            raise ValueError(
                f"attempt_events[{idx}] end_frame ({ef}) < start_frame ({sf})"
            )

        if prev_end is not None:
            if sf <= prev_end:
                if sf < (prev_start if prev_start is not None else 0):
                    raise ValueError(
                        f"attempt_events[{idx}] out of order: start_frame {sf} < previous start_frame {prev_start}"
                    )
                raise ValueError(
                    f"Overlapping event intervals: [{sf}, {ef}] overlaps with previous [{prev_start}, {prev_end}]"
                )
        prev_start = sf
        prev_end = ef
        validated.append(ev)

    seen_ids = set()
    for idx, ev in enumerate(validated):
        aid = _parse_int(ev.get("attempt_id", idx + 1), f"event[{idx}].attempt_id")
        if aid < 1 or aid in seen_ids:
            raise ValueError("Event attempt_id must be positive and unique")
        seen_ids.add(aid)
        if ev.get("status", "uncounted") not in ("counted", "uncounted"):
            raise ValueError("Event status must be counted or uncounted")
        side = ev.get("side") or ev.get("posture_side")
        if side is not None and str(side).strip() and str(side).strip().lower() not in {"left", "right"}:
            raise ValueError("Event side must be left or right")
    return validated


def summarise_attempt_posture(
    frame_rows: Sequence[Dict[str, Any]],
    attempt_events: Union[Sequence[Dict[str, Any]], Dict[str, Any]],
) -> Dict[str, Any]:
    """Compute per-event posture summaries from exported CSV rows and attempts JSON.

    Accepts exported angles.csv row mappings and attempts.json event mappings.
    Produces per-event kinematic distributions (min, max, median, coverage fraction)
    for torso inclination and hip angle, with detailed unavailable reason counts.

    Parameters:
        frame_rows: Sequence of dictionary mappings representing exported angles.csv.
        attempt_events: Sequence of event dictionaries (or attempts.json mapping containing 'events').

    Returns:
        Structured dictionary containing:
        - 'summary': High-level session totals and coverage counts.
        - 'events': List of detailed per-event posture statistics and descriptive observations.
        - 'telemetry_limitations': Precision limits of input telemetry.
        - 'biomechanical_limitations': Caveats on 2D monocular lateral-view kinematics.

    Raises:
        ValueError: If input frame indices are descending/duplicate, event intervals overlap,
            end_frame < start_frame, frame indices are negative, or incompatible event IDs are detected.
    """
    rows = _validate_frame_rows(frame_rows)
    events = _validate_events(attempt_events)

    # Fast indexed lookup by integer frame index
    frame_by_index: Dict[int, Dict[str, Any]] = {}
    for r in rows:
        f_idx = _parse_int(r["frame_index"], "row.frame_index")
        frame_by_index[f_idx] = r

    summarised_events: List[Dict[str, Any]] = []

    full_torso_count = 0
    partial_torso_count = 0
    no_torso_count = 0

    full_hip_count = 0
    partial_hip_count = 0
    no_hip_count = 0

    for ev_idx, ev in enumerate(events):
        sf = _parse_int(ev["start_frame"], f"event[{ev_idx}].start_frame")
        ef = _parse_int(ev["end_frame"], f"event[{ev_idx}].end_frame")
        total_frames = ef - sf + 1

        raw_aid = ev.get("attempt_id")
        event_aid = _parse_int(raw_aid, f"event[{ev_idx}].attempt_id") if raw_aid is not None and str(raw_aid).strip() != "" else ev_idx + 1

        target_track = ev.get("track_id")
        if target_track is not None and str(target_track).strip() != "":
            target_track = _parse_int(target_track, f"event[{ev_idx}].track_id")
        else:
            target_track = None

        target_side = ev.get("side") or ev.get("posture_side")
        if target_side is not None and str(target_side).strip() != "":
            target_side = str(target_side).strip().lower()
        else:
            target_side = None

        torso_measured: List[float] = []
        hip_measured: List[float] = []
        torso_unavail: Counter = Counter()
        hip_unavail: Counter = Counter()

        for f_idx in range(sf, ef + 1):
            if f_idx not in frame_by_index:
                torso_unavail["missing_frame"] += 1
                hip_unavail["missing_frame"] += 1
                continue

            r = frame_by_index[f_idx]

            # Incompatible event ID check on non-blank attempt_id
            row_aid_raw = r.get("attempt_id")
            if row_aid_raw is not None and str(row_aid_raw).strip() != "":
                row_aid = _parse_int(row_aid_raw, f"frame_rows[{f_idx}].attempt_id")
                if event_aid is not None and row_aid != event_aid:
                    raise ValueError(
                        f"Row at frame {f_idx} has attempt_id {row_aid} incompatible with event attempt_id {event_aid}"
                    )

            row_reason = str(r.get("reason", "") or "").strip()

            if target_track is None or target_side is None:
                torso_unavail["missing_event_identity"] += 1
                hip_unavail["missing_event_identity"] += 1
                continue
            if row_reason.startswith(("track_id_mismatch", "missing_track_id")):
                torso_unavail["track_mismatch"] += 1
                hip_unavail["track_mismatch"] += 1
                continue

            # Ambiguous subjects suppression
            if row_reason == "multiple_subjects_ambiguous" or "ambiguous" in row_reason:
                torso_unavail["multiple_subjects_ambiguous"] += 1
                hip_unavail["multiple_subjects_ambiguous"] += 1
                continue

            # No subject detected
            if row_reason == "no_subject_detected":
                torso_unavail["no_subject_detected"] += 1
                hip_unavail["no_subject_detected"] += 1
                continue

            # Track ID consistency check
            raw_r_tid = r.get("track_id")
            r_tid = None
            if raw_r_tid is not None and str(raw_r_tid).strip() != "":
                r_tid = _parse_int(raw_r_tid, f"frame_rows[{f_idx}].track_id")

            if target_track is not None:
                if r_tid is None:
                    torso_unavail["missing_track_id"] += 1
                    hip_unavail["missing_track_id"] += 1
                    continue
                if r_tid != target_track:
                    torso_unavail["track_mismatch"] += 1
                    hip_unavail["track_mismatch"] += 1
                    continue

            # Side consistency check
            # Only explicit measurement-side metadata establishes the measured side.
            r_side = r.get("posture_side")
            if r_side is not None and str(r_side).strip() != "":
                r_side = str(r_side).strip().lower()
            else:
                r_side = None

            if target_side is not None:
                if r_side is None:
                    torso_unavail["missing_side"] += 1
                    hip_unavail["missing_side"] += 1
                    continue
                if r_side != target_side:
                    torso_unavail["side_mismatch"] += 1
                    hip_unavail["side_mismatch"] += 1
                    continue

            # Evaluate torso inclination independently of knee or hip validity
            torso_val = _parse_angle(r.get("torso_inclination_deg"))
            t_reason = str(r.get("torso_reason", "") or "").strip()
            if t_reason and t_reason != "ok":
                torso_unavail[t_reason] += 1
            elif torso_val is not None:
                if 0.0 <= torso_val <= 180.0:
                    torso_measured.append(torso_val)
                else:
                    torso_unavail["invalid_angle_range"] += 1
            else:
                code = t_reason if t_reason and t_reason != "ok" else "missing_measurement"
                torso_unavail[code] += 1

            # Evaluate hip angle independently of torso or knee validity
            hip_val = _parse_angle(r.get("hip_angle_deg"))
            h_reason = str(r.get("hip_reason", "") or "").strip()
            if h_reason and h_reason != "ok":
                hip_unavail[h_reason] += 1
            elif hip_val is not None:
                if 0.0 <= hip_val <= 180.0:
                    hip_measured.append(hip_val)
                else:
                    hip_unavail["invalid_angle_range"] += 1
            else:
                code = h_reason if h_reason and h_reason != "ok" else "missing_measurement"
                hip_unavail[code] += 1

        # Torso distribution
        t_count = len(torso_measured)
        t_cov = round(t_count / total_frames, 4) if total_frames > 0 else 0.0
        t_med = round(float(statistics.median(torso_measured)), CSV_ANGLE_PRECISION_DECIMALS) if torso_measured else None
        t_min = round(float(min(torso_measured)), CSV_ANGLE_PRECISION_DECIMALS) if torso_measured else None
        t_max = round(float(max(torso_measured)), CSV_ANGLE_PRECISION_DECIMALS) if torso_measured else None

        if t_count == total_frames:
            full_torso_count += 1
        elif t_count > 0:
            partial_torso_count += 1
        else:
            no_torso_count += 1

        # Hip distribution
        h_count = len(hip_measured)
        h_cov = round(h_count / total_frames, 4) if total_frames > 0 else 0.0
        h_med = round(float(statistics.median(hip_measured)), CSV_ANGLE_PRECISION_DECIMALS) if hip_measured else None
        h_min = round(float(min(hip_measured)), CSV_ANGLE_PRECISION_DECIMALS) if hip_measured else None
        h_max = round(float(max(hip_measured)), CSV_ANGLE_PRECISION_DECIMALS) if hip_measured else None

        if h_count == total_frames:
            full_hip_count += 1
        elif h_count > 0:
            partial_hip_count += 1
        else:
            no_hip_count += 1

        # Neutral descriptive observations
        obs: List[str] = []
        if torso_measured:
            obs.append(
                f"Torso inclination relative to upward image vertical ranged from {t_min:.2f}° to {t_max:.2f}° "
                f"(median {t_med:.2f}°) across {t_count}/{total_frames} frames ({round(t_cov * 100, 1)}% coverage)."
            )
        else:
            reasons_str = ", ".join(f"{k} ({v}f)" for k, v in sorted(torso_unavail.items()))
            obs.append(
                f"Torso inclination was not measurable across {total_frames} frames. "
                f"Unavailable reasons: {reasons_str or 'none reported'}."
            )

        if hip_measured:
            obs.append(
                f"Hip joint angle ranged from {h_min:.2f}° to {h_max:.2f}° "
                f"(median {h_med:.2f}°) across {h_count}/{total_frames} frames ({round(h_cov * 100, 1)}% coverage)."
            )
        else:
            reasons_str = ", ".join(f"{k} ({v}f)" for k, v in sorted(hip_unavail.items()))
            obs.append(
                f"Hip angle was not measurable across {total_frames} frames. "
                f"Unavailable reasons: {reasons_str or 'none reported'}."
            )

        side_name = target_side if target_side else "unspecified side"
        obs.append(
            f"Kinematic observations reflect projected 2D measurements from tracked {side_name} joints; this does not establish camera orientation; "
            "individual morphology, camera roll, and stance geometry influence projected angles without indicating movement quality."
        )

        torso_dict = {
            "measured_count": t_count,
            "total_frames": total_frames,
            "coverage_fraction": t_cov,
            "median_deg": t_med,
            "min_deg": t_min,
            "max_deg": t_max,
            "unavailable_reasons": dict(torso_unavail),
        }

        hip_dict = {
            "measured_count": h_count,
            "total_frames": total_frames,
            "coverage_fraction": h_cov,
            "median_deg": h_med,
            "min_deg": h_min,
            "max_deg": h_max,
            "unavailable_reasons": dict(hip_unavail),
        }

        event_summary: Dict[str, Any] = {
            "attempt_id": event_aid,
            "status": ev.get("status", "uncounted"),
            "reason": ev.get("reason", "unknown"),
            "start_frame": sf,
            "end_frame": ef,
            "total_frames": total_frames,
            "track_id": target_track,
            "side": target_side,
            "torso_inclination": torso_dict,
            "hip_angle": hip_dict,
            # Aliases for caller flexibility
            "torso": torso_dict,
            "hip": hip_dict,
            "torso_unavailable_reasons": torso_dict["unavailable_reasons"],
            "hip_unavailable_reasons": hip_dict["unavailable_reasons"],
            "observations": obs,
        }

        summarised_events.append(event_summary)

    counted_count = sum(1 for ev in events if ev.get("status") == "counted")
    uncounted_count = len(events) - counted_count

    return {
        "summary": {
            "total_attempt_events": len(events),
            "counted_events": counted_count,
            "uncounted_events": uncounted_count,
            "events_with_full_torso_coverage": full_torso_count,
            "events_with_partial_torso_coverage": partial_torso_count,
            "events_with_no_torso_coverage": no_torso_count,
            "events_with_full_hip_coverage": full_hip_count,
            "events_with_partial_hip_coverage": partial_hip_count,
            "events_with_no_hip_coverage": no_hip_count,
        },
        "events": summarised_events,
        "telemetry_limitations": {
            "angle_precision_deg": 0.01,
            "precision_note": (
                "Exported angles from CSV have at most two decimal places of precision (.2f). "
                "Reported medians and bounds preserve this precision limit."
            ),
        },
        "biomechanical_limitations": [
            "Observed movement events can include setup and weight return; the system does not establish squat intent.",
            "Projected 2D angles from monocular lateral view; sensitive to camera height, pitch, and roll.",
            "Measurements reflect visible side only; cannot evaluate bilateral symmetry or contralateral limb.",
            "Frontal and transverse plane motions (e.g., dynamic knee valgus, pelvic rotation, lateral hip shift) are not observable in 2D side view.",
            "Descriptive kinematic observations only; no universal form score, technique grading, or clinical safety claims are made.",
            "Individual anthropometry (e.g., femur-to-torso ratio) naturally varies torso inclination and hip flexion.",
        ],
    }


# Standard American English alias
summarize_attempt_posture = summarise_attempt_posture


def load_csv_frame_rows(csv_path: Union[str, Path]) -> List[Dict[str, Any]]:
    """Convenience helper to read exported angles.csv into a list of row mappings."""
    path = Path(csv_path)
    if not path.is_file():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    with open(path, mode="r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        return list(reader)


def load_attempts_events(json_path: Union[str, Path]) -> List[Dict[str, Any]]:
    """Convenience helper to read attempts.json and extract its event list."""
    path = Path(json_path)
    if not path.is_file():
        raise FileNotFoundError(f"Attempts JSON file not found: {json_path}")
    with open(path, mode="r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and "events" in data:
        return data["events"]
    if isinstance(data, list):
        return data
    raise ValueError(f"Unexpected JSON structure in {json_path}; expected dict with 'events' or list")
