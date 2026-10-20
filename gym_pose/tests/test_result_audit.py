"""Comprehensive unit tests for gym_analyser.result_audit.

Task 52 test suite verifying:
- Clean, valid results directory passes audit.
- Contradictory repetition counts across summary, attempts, feedback, and coaching fail audit.
- Inverted or non-chronological attempt timestamps fail audit.
- Missing required artifacts (summary.json, attempts.json) fail audit.
- Missing optional artifacts report UNAVAILABLE honestly and do NOT fail as corrupt.
- Decoded prefix runs (incomplete video) are clearly distinguished from whole completion.
- Duplicate JSON keys and duplicate attempt IDs fail audit.
- Dangling attempt ID references fail audit.
- Non-finite numeric constants (NaN/Inf) and booleans in integer fields fail audit.
- Symlinks, directory junctions, and path traversal are rejected.
- 100% byte-for-byte read-only input immutability is preserved.
- No-clobber policy on output report destination is enforced.
- CLI arguments, exit codes, and output modes (quiet, strict).
- Valid zero-repetition runs.
- Mismatched configuration parameters fail audit.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import patch

# Ensure gym_pose directory is in sys.path
_GYM_POSE_DIR = Path(__file__).resolve().parent.parent
if str(_GYM_POSE_DIR) not in sys.path:
    sys.path.insert(0, str(_GYM_POSE_DIR))

from gym_analyser.result_audit import (
    CHECK_FAILED,
    CHECK_PASSED,
    CHECK_UNAVAILABLE,
    CHECK_WARNING,
    DISCLAIMER_TEXT,
    ResultAuditor,
    STATUS_FAILED,
    STATUS_PASSED,
    STATUS_WARNING,
    audit_results_directory,
    compute_sha256,
    is_symlink_or_reparse,
    load_json_strict,
    main,
)


def _write_json(path: Path, data: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def make_test_fixtures(
    completed_reps: int = 2,
    uncounted_attempts: int = 0,
    fps: float = 30.0,
    declared_frames: int = 200,
    processed_frames: int = 200,
    status: str = "completed",
) -> Dict[str, Any]:
    """Generate consistent test artifact dictionaries."""
    events: List[Dict[str, Any]] = []
    fb_attempts: List[Dict[str, Any]] = []
    cues: List[Dict[str, Any]] = []
    phase_events: List[Dict[str, Any]] = []

    for i in range(1, completed_reps + 1):
        sf = (i - 1) * 60 + 20
        turn_f = sf + 25
        ef = sf + 50
        st = round(sf / fps, 3)
        turn_t = round(turn_f / fps, 3)
        et = round(ef / fps, 3)
        min_knee = 42.0 + i

        events.append({
            "attempt_id": i,
            "status": "counted",
            "reason": "completed_cycle",
            "explanation": "A complete cycle met the configured counting thresholds.",
            "start_frame": sf,
            "end_frame": ef,
            "minimum_observed_frame": turn_f,
            "start_time_s": st,
            "end_time_s": et,
            "min_knee_angle_deg": min_knee,
            "category": "completed",
        })

        fb_attempts.append({
            "attempt_id": i,
            "status": "counted",
            "is_counted": True,
            "start_frame": sf,
            "end_frame": ef,
            "evidence": {
                "min_knee_angle_deg": min_knee,
                "torso_median_deg": 22.0,
            },
        })

        phase_events.append({
            "attempt_id": i,
            "status": "counted",
            "start_frame": sf,
            "end_frame": ef,
            "start_time_s": st,
            "end_time_s": et,
            "min_knee_point": {
                "frame_index": turn_f,
                "video_time_s": turn_t,
                "knee_angle_deg": min_knee,
            },
        })

        cues.append({
            "rep_index": i,
            "attempt_id": i,
            "is_counted": True,
            "timestamps": {
                "start_time_s": st,
                "inflection_time_s": turn_t,
                "end_time_s": et,
            },
            "measurements": {
                "min_knee_angle_deg": min_knee,
                "median_torso_inclination_deg": 22.0,
            },
        })

    for u in range(1, uncounted_attempts + 1):
        aid = completed_reps + u
        sf = (completed_reps + u - 1) * 60 + 20
        ef = sf + 30
        st = round(sf / fps, 3)
        et = round(ef / fps, 3)
        events.append({
            "attempt_id": aid,
            "status": "uncounted",
            "reason": "depth_threshold_not_reached",
            "explanation": "Depth threshold not reached.",
            "start_frame": sf,
            "end_frame": ef,
            "start_time_s": st,
            "end_time_s": et,
            "min_knee_angle_deg": 125.0,
            "category": "movement_threshold",
        })

    config_block = {
        "down_angle": 110.0,
        "up_angle": 150.0,
        "min_joint_confidence": 0.3,
        "stable_frames": 3,
        "departure_frames": 2,
        "max_gap_seconds": 0.5,
    }

    summary = {
        "status": status,
        "model_name": "test_model.pt",
        "read_status_note": f"Processed {processed_frames} frames",
        "processed_frames": processed_frames,
        "processed_duration_s": round(processed_frames / fps, 2),
        "completed_reps": completed_reps,
        "configuration": dict(config_block),
        "source_video": {
            "declared_frame_count": declared_frames,
            "fps": fps,
        },
        "attempt_observations": {
            "counted_attempts": completed_reps,
            "uncounted_attempts": uncounted_attempts,
        },
        "limitations": [
            "Projected 2D joint angles, not calibrated 3D anatomical angles.",
            "Repetition counts reflect threshold transitions, not validated clinical accuracy.",
        ],
    }

    attempts = {
        "counter_completed_reps": completed_reps,
        "counted_attempts": completed_reps,
        "uncounted_attempts": uncounted_attempts,
        "configuration": dict(config_block),
        "events": events,
        "limitations": [
            "Attempts are software-defined observed movements, not expert annotations.",
        ],
    }

    feedback_obs = {
        "summary": {
            "total_attempt_events": completed_reps + uncounted_attempts,
            "counted_events": completed_reps,
            "uncounted_events": uncounted_attempts,
        },
        "attempts": fb_attempts,
    }

    feedback_phases = {
        "summary": {
            "total_attempt_events": completed_reps + uncounted_attempts,
            "counted_events": completed_reps,
        },
        "attempts": phase_events,
    }

    coaching = {
        "contract_version": "1.0",
        "status": "ready",
        "overall_summary": {
            "total_counted_reps": completed_reps,
            "total_uncounted_attempts": uncounted_attempts,
        },
        "per_repetition_cues": cues,
        "disclaimers": {
            "clinical_disclaimer": "Descriptive observations only.",
        },
    }

    return {
        "summary.json": summary,
        "attempts.json": attempts,
        "feedback_observations.json": feedback_obs,
        "feedback_phases.json": feedback_phases,
        "coaching_feedback.json": coaching,
    }


def populate_results_dir(target_dir: Path, fixtures: Dict[str, Any], include_media: bool = True) -> None:
    """Write fixtures and dummy media files to target directory."""
    for filename, data in fixtures.items():
        _write_json(target_dir / filename, data)

    # Add dummy CSV
    csv_path = target_dir / "angles.csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as cf:
        cf.write("frame_index,video_time_s,completed_reps,attempt_id\n")
        cf.write("10,0.33,0,\n")
        reps = fixtures.get("summary.json", {}).get("completed_reps", 0)
        cf.write(f"50,1.67,{reps},1\n")

    if include_media:
        (target_dir / "angle_plot.png").write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 32)
        (target_dir / "annotated.mp4").write_bytes(b"\x00\x00\x00\x18ftypmp42" + b"\x00" * 32)


class TestResultAudit(unittest.TestCase):
    """Test suite for result_audit module and CLI."""

    def setUp(self) -> None:
        self.tmp_dir_obj = tempfile.TemporaryDirectory()
        self.tmp_dir = Path(self.tmp_dir_obj.name)

    def tearDown(self) -> None:
        try:
            self.tmp_dir_obj.cleanup()
        except Exception:
            pass

    # -----------------------------------------------------------------------
    # 1. Clean, Valid Directory
    # -----------------------------------------------------------------------

    def test_clean_valid_directory_passes(self) -> None:
        fixtures = make_test_fixtures(completed_reps=2, uncounted_attempts=1)
        populate_results_dir(self.tmp_dir, fixtures)

        report = audit_results_directory(self.tmp_dir)

        self.assertEqual(report["status"], STATUS_PASSED)
        self.assertFalse(report["is_incomplete_prefix"])
        self.assertEqual(report["summary"]["failed"], 0)
        self.assertEqual(report["summary"]["warning"], 0)
        self.assertGreater(report["summary"]["passed"], 5)
        self.assertEqual(len(report["issues"]), 0)
        self.assertTrue(report["input_integrity"]["byte_for_byte_verified"])
        self.assertEqual(report["disclaimer"], DISCLAIMER_TEXT)

    # -----------------------------------------------------------------------
    # 2. Contradictory Repetition Counts Fail
    # -----------------------------------------------------------------------

    def test_contradictory_rep_counts_summary_vs_attempts(self) -> None:
        fixtures = make_test_fixtures(completed_reps=2)
        # Contradiction: summary declares 2 reps, attempts declares 3
        fixtures["attempts.json"]["counter_completed_reps"] = 3
        populate_results_dir(self.tmp_dir, fixtures)

        report = audit_results_directory(self.tmp_dir)

        self.assertEqual(report["status"], STATUS_FAILED)
        self.assertGreater(report["summary"]["failed"], 0)
        issues = [i for i in report["issues"] if i["check_id"] == "CHK_REP_COUNT_AGREEMENT"]
        self.assertTrue(len(issues) > 0)
        self.assertIn("Repetition count contradictions detected", issues[0]["message"])

    def test_contradictory_rep_counts_attempts_events_mismatch(self) -> None:
        fixtures = make_test_fixtures(completed_reps=2)
        # Contradiction: summary completed_reps=2, but only 1 counted event in events list
        fixtures["attempts.json"]["events"][1]["status"] = "uncounted"
        populate_results_dir(self.tmp_dir, fixtures)

        report = audit_results_directory(self.tmp_dir)

        self.assertEqual(report["status"], STATUS_FAILED)
        issues = [i for i in report["issues"] if i["check_id"] == "CHK_REP_COUNT_AGREEMENT"]
        self.assertTrue(len(issues) > 0)

    def test_contradictory_rep_counts_coaching_feedback(self) -> None:
        fixtures = make_test_fixtures(completed_reps=2)
        fixtures["coaching_feedback.json"]["overall_summary"]["total_counted_reps"] = 99
        populate_results_dir(self.tmp_dir, fixtures)

        report = audit_results_directory(self.tmp_dir)

        self.assertEqual(report["status"], STATUS_FAILED)
        issues = [i for i in report["issues"] if i["check_id"] == "CHK_REP_COUNT_AGREEMENT"]
        self.assertTrue(len(issues) > 0)
        self.assertIn("coaching_feedback.json", issues[0]["message"])

    def test_contradictory_rep_counts_feedback_observations(self) -> None:
        fixtures = make_test_fixtures(completed_reps=2)
        fixtures["feedback_observations.json"]["summary"]["counted_events"] = 5
        populate_results_dir(self.tmp_dir, fixtures)

        report = audit_results_directory(self.tmp_dir)

        self.assertEqual(report["status"], STATUS_FAILED)
        issues = [i for i in report["issues"] if i["check_id"] == "CHK_REP_COUNT_AGREEMENT"]
        self.assertTrue(len(issues) > 0)

    # -----------------------------------------------------------------------
    # 3. Inverted Timestamps Fail
    # -----------------------------------------------------------------------

    def test_inverted_attempt_timestamps_fail(self) -> None:
        fixtures = make_test_fixtures(completed_reps=1)
        # End time before start time
        fixtures["attempts.json"]["events"][0]["start_time_s"] = 5.0
        fixtures["attempts.json"]["events"][0]["end_time_s"] = 2.0
        populate_results_dir(self.tmp_dir, fixtures)

        report = audit_results_directory(self.tmp_dir)

        self.assertEqual(report["status"], STATUS_FAILED)
        issues = [i for i in report["issues"] if i["check_id"] == "CHK_CHRONOLOGICAL_TIMESTAMPS"]
        self.assertTrue(len(issues) > 0)
        self.assertIn("end_time_s (2.0) < start_time_s (5.0)", issues[0]["message"])

    def test_inverted_coaching_inflection_timestamp_fail(self) -> None:
        fixtures = make_test_fixtures(completed_reps=1)
        # Inflection outside [start, end]
        cues = fixtures["coaching_feedback.json"]["per_repetition_cues"]
        cues[0]["timestamps"]["inflection_time_s"] = 999.0
        populate_results_dir(self.tmp_dir, fixtures)

        report = audit_results_directory(self.tmp_dir)

        self.assertEqual(report["status"], STATUS_FAILED)
        issues = [i for i in report["issues"] if i["check_id"] == "CHK_CHRONOLOGICAL_TIMESTAMPS"]
        self.assertTrue(len(issues) > 0)
        self.assertIn("inflection (999.0) outside", issues[0]["message"])

    def test_inverted_phase_min_knee_point_fail(self) -> None:
        fixtures = make_test_fixtures(completed_reps=1)
        mkp = fixtures["feedback_phases.json"]["attempts"][0]["min_knee_point"]
        mkp["video_time_s"] = 0.05  # before start_time_s
        populate_results_dir(self.tmp_dir, fixtures)

        report = audit_results_directory(self.tmp_dir)

        self.assertEqual(report["status"], STATUS_FAILED)
        issues = [i for i in report["issues"] if i["check_id"] == "CHK_CHRONOLOGICAL_TIMESTAMPS"]
        self.assertTrue(len(issues) > 0)

    # -----------------------------------------------------------------------
    # 4. Missing Required Artifacts Fail
    # -----------------------------------------------------------------------

    def test_missing_summary_json_fails(self) -> None:
        fixtures = make_test_fixtures(completed_reps=1)
        del fixtures["summary.json"]
        populate_results_dir(self.tmp_dir, fixtures)

        report = audit_results_directory(self.tmp_dir)

        self.assertEqual(report["status"], STATUS_FAILED)
        issues = [i for i in report["issues"] if i["check_id"] == "CHK_REQUIRED_ARTIFACTS"]
        self.assertTrue(len(issues) > 0)
        self.assertIn("summary.json", issues[0]["message"])

    def test_missing_attempts_json_fails(self) -> None:
        fixtures = make_test_fixtures(completed_reps=1)
        del fixtures["attempts.json"]
        populate_results_dir(self.tmp_dir, fixtures)

        report = audit_results_directory(self.tmp_dir)

        self.assertEqual(report["status"], STATUS_FAILED)
        issues = [i for i in report["issues"] if i["check_id"] == "CHK_REQUIRED_ARTIFACTS"]
        self.assertTrue(len(issues) > 0)
        self.assertIn("attempts.json", issues[0]["message"])

    # -----------------------------------------------------------------------
    # 5. Missing Optional Artifacts Report UNAVAILABLE Honestly
    # -----------------------------------------------------------------------

    def test_missing_optional_artifacts_reports_unavailable(self) -> None:
        # Only summary.json and attempts.json provided
        fixtures = {
            "summary.json": make_test_fixtures(completed_reps=1)["summary.json"],
            "attempts.json": make_test_fixtures(completed_reps=1)["attempts.json"],
        }
        populate_results_dir(self.tmp_dir, fixtures, include_media=False)

        report = audit_results_directory(self.tmp_dir)

        self.assertEqual(report["status"], STATUS_PASSED)
        self.assertEqual(report["artifacts"]["coaching_feedback.json"]["status"], CHECK_UNAVAILABLE)
        self.assertEqual(report["artifacts"]["feedback_observations.json"]["status"], CHECK_UNAVAILABLE)
        self.assertEqual(report["artifacts"]["feedback_phases.json"]["status"], CHECK_UNAVAILABLE)
        self.assertEqual(report["artifacts"]["posture_summary.json"]["status"], CHECK_UNAVAILABLE)
        self.assertEqual(report["artifacts"]["preview.mp4"]["status"], CHECK_UNAVAILABLE)
        self.assertEqual(report["summary"]["failed"], 0)

    def test_coaching_feedback_status_unavailable_handled_gracefully(self) -> None:
        fixtures = make_test_fixtures(completed_reps=2)
        fixtures["coaching_feedback.json"] = {
            "contract_version": "1.0",
            "status": "unavailable",
            "overall_summary": {
                "total_counted_reps": 2,
                "total_uncounted_attempts": 0,
            },
            "per_repetition_cues": [],
            "disclaimers": {"clinical_disclaimer": "test"},
        }
        populate_results_dir(self.tmp_dir, fixtures)

        report = audit_results_directory(self.tmp_dir)
        self.assertEqual(report["status"], STATUS_PASSED)

    # -----------------------------------------------------------------------
    # 6. Decoded Prefix Runs
    # -----------------------------------------------------------------------

    def test_incomplete_video_run_clarity(self) -> None:
        fixtures = make_test_fixtures(
            completed_reps=2,
            declared_frames=500,
            processed_frames=450,
            status="incomplete",
        )
        populate_results_dir(self.tmp_dir, fixtures)

        # In non-strict mode: warning status, exit 0
        report = audit_results_directory(self.tmp_dir, strict=False)
        self.assertEqual(report["status"], STATUS_WARNING)
        self.assertTrue(report["is_incomplete_prefix"])
        self.assertEqual(report["prefix_details"]["declared_frame_count"], 500)
        self.assertEqual(report["prefix_details"]["processed_frames"], 450)
        self.assertGreater(report["summary"]["warning"], 0)

        # In strict mode: elevates warning to failed
        report_strict = audit_results_directory(self.tmp_dir, strict=True)
        self.assertEqual(report_strict["status"], STATUS_FAILED)

    # -----------------------------------------------------------------------
    # 7. Duplicate JSON Keys & Duplicate IDs Fail
    # -----------------------------------------------------------------------

    def test_duplicate_json_keys_detected_and_failed(self) -> None:
        fixtures = make_test_fixtures(completed_reps=1)
        populate_results_dir(self.tmp_dir, fixtures)

        # Write invalid raw JSON with duplicate keys to summary.json
        raw_json_with_duplicate = '{\n  "status": "completed",\n  "completed_reps": 1,\n  "completed_reps": 2\n}'
        (self.tmp_dir / "summary.json").write_text(raw_json_with_duplicate, encoding="utf-8")

        report = audit_results_directory(self.tmp_dir)

        self.assertEqual(report["status"], STATUS_FAILED)
        issues = [i for i in report["issues"] if i["check_id"] == "CHK_JSON_SYNTAX"]
        self.assertTrue(len(issues) > 0)
        self.assertIn("Duplicate JSON key", issues[0]["message"])

    def test_duplicate_attempt_ids_fail(self) -> None:
        fixtures = make_test_fixtures(completed_reps=2)
        # Duplicate attempt_id: both have attempt_id = 1
        fixtures["attempts.json"]["events"][1]["attempt_id"] = 1
        populate_results_dir(self.tmp_dir, fixtures)

        report = audit_results_directory(self.tmp_dir)

        self.assertEqual(report["status"], STATUS_FAILED)
        issues = [i for i in report["issues"] if i["check_id"] == "CHK_ATTEMPT_ID_INTEGRITY"]
        self.assertTrue(len(issues) > 0)
        self.assertIn("Duplicate attempt_id", issues[0]["message"])

    def test_dangling_attempt_id_reference_fails(self) -> None:
        fixtures = make_test_fixtures(completed_reps=1)
        # Coaching cue points to non-existent attempt_id 999
        fixtures["coaching_feedback.json"]["per_repetition_cues"][0]["attempt_id"] = 999
        populate_results_dir(self.tmp_dir, fixtures)

        report = audit_results_directory(self.tmp_dir)

        self.assertEqual(report["status"], STATUS_FAILED)
        issues = [i for i in report["issues"] if i["check_id"] == "CHK_ATTEMPT_ID_INTEGRITY"]
        self.assertTrue(len(issues) > 0)
        self.assertIn("Dangling attempt_id references", issues[0]["message"])

    # -----------------------------------------------------------------------
    # 8. Strict Numeric & Type Validation
    # -----------------------------------------------------------------------

    def test_boolean_in_completed_reps_fails(self) -> None:
        fixtures = make_test_fixtures(completed_reps=1)
        # In Python True is an int instance, but audit must strictly reject it
        fixtures["summary.json"]["completed_reps"] = True
        populate_results_dir(self.tmp_dir, fixtures)

        report = audit_results_directory(self.tmp_dir)

        self.assertEqual(report["status"], STATUS_FAILED)
        issues = [i for i in report["issues"] if i["check_id"] == "CHK_REP_COUNT_AGREEMENT"]
        self.assertTrue(len(issues) > 0)

    def test_nan_or_inf_in_json_fails(self) -> None:
        fixtures = make_test_fixtures(completed_reps=1)
        populate_results_dir(self.tmp_dir, fixtures)

        # Write NaN directly in summary.json
        raw_json_with_nan = '{\n  "status": "completed",\n  "completed_reps": NaN\n}'
        (self.tmp_dir / "summary.json").write_text(raw_json_with_nan, encoding="utf-8")

        report = audit_results_directory(self.tmp_dir)

        self.assertEqual(report["status"], STATUS_FAILED)
        issues = [i for i in report["issues"] if i["check_id"] == "CHK_JSON_SYNTAX"]
        self.assertTrue(len(issues) > 0)

    # -----------------------------------------------------------------------
    # 9. Path Safety & Symlink Rejection
    # -----------------------------------------------------------------------

    def test_symlink_directory_rejected(self) -> None:
        fixtures = make_test_fixtures(completed_reps=1)
        populate_results_dir(self.tmp_dir, fixtures)

        # Create symlink pointing to tmp_dir
        symlink_dir = self.tmp_dir.parent / "symlink_test_dir"
        try:
            os.symlink(self.tmp_dir, symlink_dir, target_is_directory=True)
        except (OSError, NotImplementedError):
            self.skipTest("Symlinks not supported or permitted on this platform/configuration")

        try:
            report = audit_results_directory(symlink_dir)
            self.assertEqual(report["status"], STATUS_FAILED)
            issues = [i for i in report["issues"] if i["check_id"] == "CHK_PATH_SAFETY"]
            self.assertTrue(len(issues) > 0)
        finally:
            if symlink_dir.is_symlink() or symlink_dir.exists():
                symlink_dir.unlink()

    def test_mock_symlink_detection(self) -> None:
        fixtures = make_test_fixtures(completed_reps=1)
        populate_results_dir(self.tmp_dir, fixtures)

        with patch("gym_analyser.result_audit.is_symlink_or_reparse", return_value=True):
            report = audit_results_directory(self.tmp_dir)
            self.assertEqual(report["status"], STATUS_FAILED)
            issues = [i for i in report["issues"] if i["check_id"] == "CHK_PATH_SAFETY"]
            self.assertTrue(len(issues) > 0)

    # -----------------------------------------------------------------------
    # 10. Byte-for-byte Input Immutability
    # -----------------------------------------------------------------------

    def test_input_immutability_preserved(self) -> None:
        fixtures = make_test_fixtures(completed_reps=2, uncounted_attempts=1)
        populate_results_dir(self.tmp_dir, fixtures)

        # Record hashes before audit
        before_hashes = {}
        for root, _, files in os.walk(self.tmp_dir):
            for f in files:
                fp = Path(root) / f
                before_hashes[str(fp.relative_to(self.tmp_dir))] = compute_sha256(fp)

        # Run multiple audits
        audit_results_directory(self.tmp_dir, strict=False)
        audit_results_directory(self.tmp_dir, strict=True)

        # Verify hashes after audit are 100% byte-identical
        after_hashes = {}
        for root, _, files in os.walk(self.tmp_dir):
            for f in files:
                fp = Path(root) / f
                after_hashes[str(fp.relative_to(self.tmp_dir))] = compute_sha256(fp)

        self.assertEqual(before_hashes, after_hashes)

    # -----------------------------------------------------------------------
    # 11. No-Clobber Output Report
    # -----------------------------------------------------------------------

    def test_no_clobber_existing_file_rejected(self) -> None:
        fixtures = make_test_fixtures(completed_reps=1)
        populate_results_dir(self.tmp_dir, fixtures)

        existing_report = self.tmp_dir.parent / "existing_report.json"
        existing_report.write_text("existing content", encoding="utf-8")

        try:
            with self.assertRaises(FileExistsError):
                audit_results_directory(self.tmp_dir, output_report=existing_report)
        finally:
            if existing_report.exists():
                existing_report.unlink()

    def test_report_inside_results_dir_rejected(self) -> None:
        fixtures = make_test_fixtures(completed_reps=1)
        populate_results_dir(self.tmp_dir, fixtures)

        inside_report = self.tmp_dir / "audit_report.json"
        with self.assertRaises(ValueError):
            audit_results_directory(self.tmp_dir, output_report=inside_report)

    def test_fresh_report_written_successfully(self) -> None:
        fixtures = make_test_fixtures(completed_reps=1)
        populate_results_dir(self.tmp_dir, fixtures)

        fresh_report = self.tmp_dir.parent / f"fresh_report_{os.getpid()}.json"
        try:
            report = audit_results_directory(self.tmp_dir, output_report=fresh_report)
            self.assertTrue(fresh_report.exists())
            loaded, err = load_json_strict(fresh_report)
            self.assertIsNone(err)
            self.assertEqual(loaded["status"], STATUS_PASSED)
            self.assertEqual(loaded["results_dir"], str(self.tmp_dir.resolve()))
        finally:
            if fresh_report.exists():
                fresh_report.unlink()

    # -----------------------------------------------------------------------
    # 12. CLI Arguments & Exit Codes
    # -----------------------------------------------------------------------

    def test_cli_exit_code_passed(self) -> None:
        fixtures = make_test_fixtures(completed_reps=1)
        populate_results_dir(self.tmp_dir, fixtures)

        exit_code = main(["--results-dir", str(self.tmp_dir), "--quiet"])
        self.assertEqual(exit_code, 0)

    def test_cli_exit_code_failed(self) -> None:
        fixtures = make_test_fixtures(completed_reps=1)
        fixtures["attempts.json"]["counter_completed_reps"] = 99
        populate_results_dir(self.tmp_dir, fixtures)

        exit_code = main(["--results-dir", str(self.tmp_dir), "--quiet"])
        self.assertEqual(exit_code, 1)

    def test_cli_exit_code_error_on_missing_dir(self) -> None:
        non_existent = self.tmp_dir / "non_existent_folder_xyz"
        exit_code = main(["--results-dir", str(non_existent), "--quiet"])
        self.assertEqual(exit_code, 1)  # failed status due to missing dir

    def test_cli_exit_code_clobber_error(self) -> None:
        fixtures = make_test_fixtures(completed_reps=1)
        populate_results_dir(self.tmp_dir, fixtures)

        existing = self.tmp_dir.parent / "already_exists.json"
        existing.write_text("{}", encoding="utf-8")
        try:
            exit_code = main(["--results-dir", str(self.tmp_dir), "--output-report", str(existing)])
            self.assertEqual(exit_code, 2)
        finally:
            if existing.exists():
                existing.unlink()

    # -----------------------------------------------------------------------
    # 13. Valid Zero-Repetition Runs
    # -----------------------------------------------------------------------

    def test_zero_completed_reps_valid(self) -> None:
        fixtures = make_test_fixtures(completed_reps=0, uncounted_attempts=2)
        populate_results_dir(self.tmp_dir, fixtures)

        report = audit_results_directory(self.tmp_dir)

        self.assertEqual(report["status"], STATUS_PASSED)
        self.assertEqual(report["summary"]["failed"], 0)

    # -----------------------------------------------------------------------
    # 14. Mismatched Configuration Parameters
    # -----------------------------------------------------------------------

    def test_mismatched_configuration_fails(self) -> None:
        fixtures = make_test_fixtures(completed_reps=1)
        fixtures["summary.json"]["configuration"]["up_angle"] = 150.0
        fixtures["attempts.json"]["configuration"]["up_angle"] = 175.0
        populate_results_dir(self.tmp_dir, fixtures)

        report = audit_results_directory(self.tmp_dir)

        self.assertEqual(report["status"], STATUS_FAILED)
        issues = [i for i in report["issues"] if i["check_id"] == "CHK_CONFIGURATION_IDENTITY"]
        self.assertTrue(len(issues) > 0)

    # -----------------------------------------------------------------------
    # 15. Review 48 False-Pass Independent Probes
    # -----------------------------------------------------------------------

    def test_review48_probe_null_required_fails(self) -> None:
        """Review 48 probe: summary.json = null with valid empty attempts must FAIL."""
        # Write null to summary.json
        (self.tmp_dir / "summary.json").write_text("null", encoding="utf-8")
        # Write valid empty attempts
        attempts = {
            "counter_completed_reps": 0,
            "counted_attempts": 0,
            "uncounted_attempts": 0,
            "events": [],
            "configuration": {
                "down_angle": 110.0,
                "up_angle": 150.0,
                "stable_frames": 3,
                "departure_frames": 2,
                "max_gap_seconds": 0.5,
            },
            "limitations": ["Attempts are software-defined."],
        }
        _write_json(self.tmp_dir / "attempts.json", attempts)

        report = audit_results_directory(self.tmp_dir)

        self.assertEqual(report["status"], STATUS_FAILED)
        self.assertGreater(report["summary"]["failed"], 0)
        issues = [i for i in report["issues"] if i["check_id"] == "CHK_REQUIRED_ARTIFACTS"]
        self.assertTrue(len(issues) > 0)
        self.assertIn("summary.json must be a valid JSON object", issues[0]["message"])

    def test_review48_probe_failed_status_fails(self) -> None:
        """Review 48 probe: summary.json declaring status 'failed' must FAIL."""
        fixtures = make_test_fixtures(completed_reps=0, uncounted_attempts=0, status="failed")
        populate_results_dir(self.tmp_dir, fixtures)

        report = audit_results_directory(self.tmp_dir)

        self.assertEqual(report["status"], STATUS_FAILED)
        self.assertGreater(report["summary"]["failed"], 0)
        issues = [i for i in report["issues"] if i["check_id"] == "CHK_RUN_STATUS"]
        self.assertTrue(len(issues) > 0)
        self.assertIn("declared failure or interruption", issues[0]["message"])

    def test_review48_probe_interrupted_or_unknown_status_fails(self) -> None:
        """Review 48 probe: status 'interrupted' or unknown must FAIL."""
        fixtures = make_test_fixtures(completed_reps=0, status="interrupted")
        populate_results_dir(self.tmp_dir, fixtures)

        report = audit_results_directory(self.tmp_dir)
        self.assertEqual(report["status"], STATUS_FAILED)
        issues = [i for i in report["issues"] if i["check_id"] == "CHK_RUN_STATUS"]
        self.assertTrue(len(issues) > 0)

        # Unknown status
        fixtures["summary.json"]["status"] = "something_unknown_and_random"
        populate_results_dir(self.tmp_dir, fixtures)
        report2 = audit_results_directory(self.tmp_dir)
        self.assertEqual(report2["status"], STATUS_FAILED)
        issues2 = [i for i in report2["issues"] if i["check_id"] == "CHK_RUN_STATUS"]
        self.assertTrue(len(issues2) > 0)

    def test_review48_probe_wrong_optional_coaching_list_fails(self) -> None:
        """Review 48 probe: coaching_feedback.json = [] (wrong shape) must FAIL."""
        fixtures = make_test_fixtures(completed_reps=1)
        populate_results_dir(self.tmp_dir, fixtures)

        # Overwrite coaching_feedback.json with an empty list []
        (self.tmp_dir / "coaching_feedback.json").write_text("[]", encoding="utf-8")

        report = audit_results_directory(self.tmp_dir)

        self.assertEqual(report["status"], STATUS_FAILED)
        self.assertGreater(report["summary"]["failed"], 0)
        issues = [i for i in report["issues"] if i["check_id"] == "CHK_OPTIONAL_SCHEMA"]
        self.assertTrue(len(issues) > 0)
        self.assertIn("coaching_feedback.json has invalid type", issues[0]["message"])

    def test_review48_probe_wrong_optional_primitive_fails(self) -> None:
        """coaching_feedback.json as primitive int or string must FAIL."""
        fixtures = make_test_fixtures(completed_reps=1)
        populate_results_dir(self.tmp_dir, fixtures)

        (self.tmp_dir / "coaching_feedback.json").write_text("12345", encoding="utf-8")
        report = audit_results_directory(self.tmp_dir)
        self.assertEqual(report["status"], STATUS_FAILED)
        issues = [i for i in report["issues"] if i["check_id"] == "CHK_OPTIONAL_SCHEMA"]
        self.assertTrue(len(issues) > 0)

    def test_review48_probe_wrong_optional_posture_list_fails(self) -> None:
        """posture_summary.json as [] must FAIL."""
        fixtures = make_test_fixtures(completed_reps=1)
        populate_results_dir(self.tmp_dir, fixtures)
        (self.tmp_dir / "posture_summary.json").write_text("[]", encoding="utf-8")

        report = audit_results_directory(self.tmp_dir)
        self.assertEqual(report["status"], STATUS_FAILED)
        issues = [i for i in report["issues"] if i["check_id"] == "CHK_OPTIONAL_SCHEMA"]
        self.assertTrue(len(issues) > 0)

    def test_review48_probe_wrong_optional_feedback_obs_list_fails(self) -> None:
        """feedback_observations.json as [] must FAIL."""
        fixtures = make_test_fixtures(completed_reps=1)
        populate_results_dir(self.tmp_dir, fixtures)
        (self.tmp_dir / "feedback_observations.json").write_text("[]", encoding="utf-8")

        report = audit_results_directory(self.tmp_dir)
        self.assertEqual(report["status"], STATUS_FAILED)
        issues = [i for i in report["issues"] if i["check_id"] == "CHK_OPTIONAL_SCHEMA"]
        self.assertTrue(len(issues) > 0)

    def test_review48_probe_wrong_optional_phases_list_fails(self) -> None:
        """feedback_phases.json as [] must FAIL."""
        fixtures = make_test_fixtures(completed_reps=1)
        populate_results_dir(self.tmp_dir, fixtures)
        (self.tmp_dir / "feedback_phases.json").write_text("[]", encoding="utf-8")

        report = audit_results_directory(self.tmp_dir)
        self.assertEqual(report["status"], STATUS_FAILED)
        issues = [i for i in report["issues"] if i["check_id"] == "CHK_OPTIONAL_SCHEMA"]
        self.assertTrue(len(issues) > 0)

    def test_review48_probe_missing_or_null_coaching_reports_unavailable(self) -> None:
        """coaching_feedback.json as null reports UNAVAILABLE honestly and passes valid run."""
        fixtures = make_test_fixtures(completed_reps=1)
        populate_results_dir(self.tmp_dir, fixtures)
        (self.tmp_dir / "coaching_feedback.json").write_text("null", encoding="utf-8")

        report = audit_results_directory(self.tmp_dir)
        self.assertEqual(report["status"], STATUS_PASSED)
        self.assertEqual(report["artifacts"]["coaching_feedback.json"]["status"], CHECK_UNAVAILABLE)

    # -----------------------------------------------------------------------
    # 16. Authentic Exported Pipeline Fixtures
    # -----------------------------------------------------------------------

    def test_authentic_exported_pipeline_fixtures(self) -> None:
        """Audit authentic pipeline output directory exp42_video_eval_base_squat_demo."""
        exp42_dir = _GYM_POSE_DIR / "output" / "experiments" / "exp42_video_eval_base_squat_demo"
        if not exp42_dir.exists():
            self.skipTest(f"exp42 directory not found at {exp42_dir}")

        report = audit_results_directory(exp42_dir)

        self.assertEqual(report["status"], STATUS_PASSED)
        self.assertEqual(report["summary"]["failed"], 0)
        self.assertTrue(report["input_integrity"]["byte_for_byte_verified"])
        self.assertEqual(report["artifacts"]["preview.mp4"]["status"], CHECK_UNAVAILABLE)

    # -----------------------------------------------------------------------
    # 17. Path Safety & Ancestor / Overlap Protection
    # -----------------------------------------------------------------------

    def test_output_overlap_with_input_dir_rejected(self) -> None:
        """Output destination overlapping with input directory must be rejected."""
        fixtures = make_test_fixtures(completed_reps=1)
        populate_results_dir(self.tmp_dir, fixtures)

        # Output path is parent of results_dir
        with self.assertRaises(ValueError):
            audit_results_directory(self.tmp_dir, output_report=self.tmp_dir.parent)

    # -----------------------------------------------------------------------
    # 18. True No-Clobber Exclusive Creation & Race Safety
    # -----------------------------------------------------------------------

    def test_true_no_clobber_exclusive_creation(self) -> None:
        """Existing file at report path must trigger FileExistsError without modifying it."""
        fixtures = make_test_fixtures(completed_reps=1)
        populate_results_dir(self.tmp_dir, fixtures)

        existing_dest = self.tmp_dir.parent / f"dest_{os.getpid()}.json"
        original_bytes = b'{"sentinel": "do_not_overwrite"}'
        existing_dest.write_bytes(original_bytes)

        try:
            with self.assertRaises(FileExistsError):
                audit_results_directory(self.tmp_dir, output_report=existing_dest)
            # Guarantee file was not overwritten
            self.assertEqual(existing_dest.read_bytes(), original_bytes)
        finally:
            if existing_dest.exists():
                existing_dest.unlink()

    # -----------------------------------------------------------------------
    # 19. Precise Input Hashing (Media files stat-checked, not double-read)
    # -----------------------------------------------------------------------

    def test_precise_input_hashing_scope(self) -> None:
        """Verify that JSONs and CSVs are byte-hashed while media files are stat-checked."""
        fixtures = make_test_fixtures(completed_reps=1)
        populate_results_dir(self.tmp_dir, fixtures, include_media=True)

        report = audit_results_directory(self.tmp_dir)

        integ = report["input_integrity"]
        self.assertTrue(integ["byte_for_byte_verified"])
        # JSON and CSV files should be hashed
        self.assertGreater(integ["consumed_files_hashed"], 3)
        # Media files (png, mp4) should be stat-checked
        self.assertGreater(integ["other_files_stat_checked"], 0)
        self.assertEqual(
            integ["total_files_audited"],
            integ["consumed_files_hashed"] + integ["other_files_stat_checked"]
        )

    # -----------------------------------------------------------------------
    # 20. Malformed Event Elements & Robust Type Validation
    # -----------------------------------------------------------------------

    def test_malformed_event_elements_no_unhandled_type_error(self) -> None:
        """Events with non-dict elements, string timestamps, or null timestamps must report error cleanly."""
        fixtures = make_test_fixtures(completed_reps=1)
        # Injected malformed timestamps: string instead of float
        fixtures["attempts.json"]["events"][0]["start_time_s"] = "invalid_string_timestamp"
        populate_results_dir(self.tmp_dir, fixtures)

        report = audit_results_directory(self.tmp_dir)

        self.assertEqual(report["status"], STATUS_FAILED)
        issues = [i for i in report["issues"] if i["check_id"] == "CHK_CHRONOLOGICAL_TIMESTAMPS"]
        self.assertTrue(len(issues) > 0)
        self.assertIn("start_time_s invalid", issues[0]["message"])

    def test_non_dict_event_in_attempts_fails_cleanly(self) -> None:
        """Non-dict element in attempts.json events list must fail with ERROR."""
        fixtures = make_test_fixtures(completed_reps=1)
        fixtures["attempts.json"]["events"].append("not_a_dictionary")
        populate_results_dir(self.tmp_dir, fixtures)

        report = audit_results_directory(self.tmp_dir)
        self.assertEqual(report["status"], STATUS_FAILED)

    def test_duplicate_cue_attempt_ids_in_coaching_fails(self) -> None:
        """Duplicate attempt_id in coaching cues must fail."""
        fixtures = make_test_fixtures(completed_reps=2)
        cues = fixtures["coaching_feedback.json"]["per_repetition_cues"]
        cues[1]["attempt_id"] = cues[0]["attempt_id"]  # Duplicate attempt_id
        populate_results_dir(self.tmp_dir, fixtures)

        report = audit_results_directory(self.tmp_dir)
        self.assertEqual(report["status"], STATUS_FAILED)
        issues = [i for i in report["issues"] if i["check_id"] == "CHK_ATTEMPT_ID_INTEGRITY"]
        self.assertTrue(len(issues) > 0)

    def test_coaching_cues_strict_is_counted(self) -> None:
        """is_counted missing or False must not be counted as a completed rep."""
        fixtures = make_test_fixtures(completed_reps=2)
        cues = fixtures["coaching_feedback.json"]["per_repetition_cues"]
        # Mark second cue as is_counted: False
        cues[1]["is_counted"] = False
        populate_results_dir(self.tmp_dir, fixtures)

        report = audit_results_directory(self.tmp_dir)
        self.assertEqual(report["status"], STATUS_FAILED)
        issues = [i for i in report["issues"] if i["check_id"] == "CHK_REP_COUNT_AGREEMENT"]
        self.assertTrue(len(issues) > 0)


class TestReview49ConsoleRegression(unittest.TestCase):
    def test_invalid_required_artifact_cannot_print_full_completion(self):
        from gym_analyser.result_audit import format_console_report
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)
            (p/"summary.json").write_text("null")
            (p/"attempts.json").write_text('{"events": []}')
            result=audit_results_directory(p)
            text=format_console_report(result)
            self.assertNotIn("PREFIX STATUS: FULL COMPLETION",text)
            self.assertIn("PREFIX STATUS: NOT VERIFIED",text)

    def test_missing_frame_counts_do_not_verify_completion(self):
        from gym_analyser.result_audit import format_console_report
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)
            (p/"summary.json").write_text('{"status":"completed","completed_reps":0}')
            (p/"attempts.json").write_text('{"events": []}')
            result=audit_results_directory(p)
            check=next(c for c in result["checks"] if c["check_id"]=="CHK_DECODED_PREFIX")
            self.assertEqual(check["status"],"UNAVAILABLE")
            self.assertNotIn("PREFIX STATUS: FULL COMPLETION",format_console_report(result))

    def test_null_optional_artifact_prints_unavailable(self):
        from gym_analyser.result_audit import format_console_report
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)
            (p/"summary.json").write_text('{"status":"completed","completed_reps":0}')
            (p/"attempts.json").write_text('{"events": []}')
            (p/"coaching_feedback.json").write_text("null")
            result=audit_results_directory(p)
            row=next(line for line in format_console_report(result).splitlines() if "coaching_feedback.json" in line)
            self.assertIn("UNAVAILABLE",row)


if __name__ == "__main__":
    unittest.main()
