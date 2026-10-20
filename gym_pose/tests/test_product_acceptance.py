"""End-to-end product acceptance test suite for 2D Gym Pose Analyser.

Task 41 Acceptance Test Suite.
Verifies cross-module product behavior, contracts, and safety boundaries:
1. Pure static & markup checks (executable in any environment without web dependencies):
   - Real INDEX_HTML structure, navigation link to /review/, accessible live regions
   - Client-side validation parameters (200MB limit, allowed video extensions)
   - Disclaimers: 2D descriptive observations, unvalidated by human expert review
2. End-to-End Analysis Workflow:
   - Video upload -> HTTP 202 Accepted with job ID
   - Single-worker execution and lifecycle transitions (queued -> running -> completed)
   - Result retrieval: repetition counts, summary, attempts, posture, phase telemetry, and coaching feedback
   - Comprehensive artifact download validation across all allowed media types
3. Strict Validation & Error Handling:
   - Oversized upload rejection (> 200MB) -> HTTP 413 / HTTP 400
   - Disallowed file extension rejection -> HTTP 400 with descriptive detail
   - Empty input rejection -> HTTP 400
   - Corrupt video or worker crash -> status: failed reported gracefully in job metadata, busy lock released
4. Preserved API Contract:
   - GET /, POST /api/jobs, GET /api/status, GET /api/jobs/{id}, GET /api/jobs/{id}/results, GET /api/jobs/{id}/download/{filename}
   - Standard HTTP status codes (200, 202, 400, 404, 413, 429)
   - Path-traversal and disallowed artifact download protection
5. Review Workspace Isolation:
   - Mount under /review/ serves review interface independently
   - Human review operations trigger NO ML model loading or inference
   - Storage isolation: review cases remain in review_storage_root, web jobs remain in jobs storage
   - Distinct root storage enforcement (disallow overlapping paths)
6. Cross-Module Feedback & Repetition Consistency:
   - Strict mapping invariants between summary.json, attempts.json, feedback_phases.json, and coaching_feedback.json
   - Repetition count matching, attempt ID alignment, timestamp monotonicity
   - Rejection/detection of inconsistent telemetry
"""

from __future__ import annotations

import io
import json
import math
import os
import re
import shutil
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import patch

# Ensure gym_pose directory is in sys.path
_GYM_POSE_DIR = Path(__file__).resolve().parent.parent
if str(_GYM_POSE_DIR) not in sys.path:
    sys.path.insert(0, str(_GYM_POSE_DIR))

# Always importable core components
from gym_analyser.web_jobs import (
    ALLOWED_ARTIFACTS,
    ALLOWED_EXTENSIONS,
    DEFAULT_MAX_UPLOAD_BYTES,
    JobBusyError,
    JobManager,
    STATUS_COMPLETED,
    STATUS_FAILED,
    STATUS_INCOMPLETE,
    STATUS_INTERRUPTED,
    STATUS_QUEUED,
    STATUS_RUNNING,
    is_valid_job_id,
)

# Static INDEX_HTML is always importable even without FastAPI
try:
    from gym_analyser.web import INDEX_HTML
except ImportError:
    INDEX_HTML = ""

# Optional web dependencies for TestClient
try:
    from fastapi.testclient import TestClient
    from gym_analyser.web import create_app
    from gym_analyser.review_web import create_review_app
    HAS_WEB_DEPS = True
except (ImportError, TypeError):
    TestClient = None  # type: ignore
    create_app = None  # type: ignore
    create_review_app = None  # type: ignore
    HAS_WEB_DEPS = False

# Optional coaching_feedback module (from Agent 3 / Task 38)
try:
    from gym_analyser.coaching_feedback import generate_coaching_feedback
    HAS_COACHING_FEEDBACK_MODULE = True
except ImportError:
    generate_coaching_feedback = None  # type: ignore
    HAS_COACHING_FEEDBACK_MODULE = False


# ---------------------------------------------------------------------------
# Fixture Generators & Cross-Module Consistency Validators
# ---------------------------------------------------------------------------

def make_consistent_fixtures(completed_reps: int = 3, uncounted_attempts: int = 1) -> Dict[str, Any]:
    """Build mutually consistent mock payloads for summary, attempts, phases, and coaching feedback."""
    fps = 30.0
    frame_per_rep = 45
    events: List[Dict[str, Any]] = []
    cues: List[Dict[str, Any]] = []
    phase_events: List[Dict[str, Any]] = []

    # Counted repetitions
    for i in range(1, completed_reps + 1):
        start_frame = (i - 1) * frame_per_rep + 10
        inflection_frame = start_frame + 18
        end_frame = start_frame + 36
        start_time_s = round(start_frame / fps, 3)
        inflection_time_s = round(inflection_frame / fps, 3)
        end_time_s = round(end_frame / fps, 3)
        min_knee_deg = 88.0 + (i * 2.0)

        attempt_event = {
            "attempt_id": i,
            "status": "counted",
            "reason": "completed_cycle",
            "explanation": "A complete cycle met the configured counting thresholds.",
            "start_frame": start_frame,
            "end_frame": end_frame,
            "start_time_s": start_time_s,
            "end_time_s": end_time_s,
            "min_knee_angle_deg": min_knee_deg,
            "category": "completed",
        }
        events.append(attempt_event)

        phase_events.append({
            "rep_index": i,
            "attempt_id": i,
            "start_time_s": start_time_s,
            "inflection_time_s": inflection_time_s,
            "end_time_s": end_time_s,
            "duration_s": round(end_time_s - start_time_s, 3),
            "descent_duration_s": round(inflection_time_s - start_time_s, 3),
            "ascent_duration_s": round(end_time_s - inflection_time_s, 3),
        })

        cues.append({
            "rep_index": i,
            "attempt_id": i,
            "is_counted": True,
            "timestamps": {
                "start_time_s": start_time_s,
                "inflection_time_s": inflection_time_s,
                "end_time_s": end_time_s,
            },
            "measurements": {
                "min_knee_angle_deg": min_knee_deg,
                "median_torso_inclination_deg": 28.5,
                "median_hip_angle_deg": 82.0,
                "coverage_pct": 98.5,
            },
            "observations": [
                {
                    "rule_id": "OBS_DEPTH_MET",
                    "category": "depth",
                    "message": f"Observed minimum knee angle reached {min_knee_deg:.1f} degrees (configured threshold: 110.0 deg).",
                    "source_id": "counter_event",
                    "confidence_level": "observed",
                }
            ],
            "suggestions": [
                {
                    "rule_id": "SUGG_PACING",
                    "type": "movement_pacing",
                    "message": "Consistent repetition cadence observed throughout ascent and descent.",
                    "disclaimer": "Descriptive kinematic timing, not diagnostic technique assessment.",
                }
            ],
        })

    # Uncounted movement attempt (e.g. shallow warm-up or weight pickup)
    uncounted_explanations: List[Dict[str, Any]] = []
    if uncounted_attempts > 0:
        uncounted_id = completed_reps + 1
        uncounted_start_frame = completed_reps * frame_per_rep + 10
        uncounted_end_frame = uncounted_start_frame + 20
        uncounted_start_s = round(uncounted_start_frame / fps, 3)
        uncounted_end_s = round(uncounted_end_frame / fps, 3)

        events.append({
            "attempt_id": uncounted_id,
            "status": "uncounted",
            "reason": "depth_threshold_not_reached",
            "explanation": "The observed knee angle (124.0 deg) did not reach the configured 110-degree depth threshold.",
            "start_frame": uncounted_start_frame,
            "end_frame": uncounted_end_frame,
            "start_time_s": uncounted_start_s,
            "end_time_s": uncounted_end_s,
            "min_knee_angle_deg": 124.0,
            "category": "movement_threshold",
        })

        uncounted_explanations.append({
            "attempt_id": uncounted_id,
            "reason_code": "depth_threshold_not_reached",
            "explanation": "Observed flexion remained above configured 110-degree depth threshold.",
            "context_hypothesis": "shallow_depth",
        })

    total_frames = (completed_reps + uncounted_attempts) * frame_per_rep + 20

    summary = {
        "status": "completed",
        "model_name": "yolo11n-pose.pt",
        "read_status_note": "",
        "processed_frames": total_frames,
        "processed_duration_s": round(total_frames / fps, 3),
        "completed_reps": completed_reps,
        "attempt_observations": {
            "counted_attempts": completed_reps,
            "uncounted_attempts": uncounted_attempts,
            "pre_arming": {"unarmed_frames": 10, "opening_lowered_frames": 0},
            "interpretation": "Software-defined observed attempts, not posture judgements",
        },
        "posture_observations": {
            "torso_valid_frames": total_frames - 5,
            "hip_valid_frames": total_frames - 5,
            "processed_frames": total_frames,
            "interpretation": "Projected 2D measurements; unsigned torso angle from upward vertical",
        },
        "limitations": [
            "Projected 2D joint angles, not calibrated 3D anatomical angles.",
            "Repetition counts reflect threshold transitions, not validated clinical accuracy.",
            "Torso inclination is unsigned and depends on camera roll; posture observations do not establish safe or correct form.",
        ],
    }

    attempts = {
        "counter_completed_reps": completed_reps,
        "counted_attempts": completed_reps,
        "uncounted_attempts": uncounted_attempts,
        "events": events,
        "limitations": ["Observed movement events can include setup and weight return; squat intent is not classified."],
    }

    feedback_phases = {
        "schema_version": "1.0",
        "total_completed_reps": completed_reps,
        "repetition_phases": phase_events,
    }

    coaching_feedback = {
        "contract_version": "1.0",
        "status": "ready",
        "overall_summary": {
            "total_counted_reps": completed_reps,
            "total_uncounted_attempts": uncounted_attempts,
            "recording_quality": {
                "camera_view_adequate": None,
                "tracking_continuity": "adequate",
                "posture_coverage_pct": 98.2,
                "notes": ["Clear side-view visibility observed; 3D camera geometry unmeasured."],
            },
        },
        "per_repetition_cues": cues,
        "uncounted_movement_explanations": uncounted_explanations,
        "disclaimers": {
            "clinical_disclaimer": "These 2D kinematic measurements are descriptive planar observations, not clinical diagnostic assessments or technique grades.",
            "human_validation_status": "UNVALIDATED by human expert review on diverse unseen subjects.",
        },
    }

    return {
        "summary": summary,
        "attempts": attempts,
        "feedback_phases": feedback_phases,
        "coaching_feedback": coaching_feedback,
    }


def verify_feedback_mapping_consistency(
    summary: Dict[str, Any],
    attempts: Dict[str, Any],
    feedback_phases: Optional[Dict[str, Any]] = None,
    coaching_feedback: Optional[Dict[str, Any]] = None,
) -> None:
    """Validate cross-module mapping invariants across summary, attempts, phases, and coaching feedback.

    Raises ValueError with descriptive diagnostics if any invariant is violated.
    """
    if not isinstance(summary, dict) or not isinstance(attempts, dict):
        raise ValueError("summary and attempts must be valid dictionaries")

    completed_reps = summary.get("completed_reps")
    if not isinstance(completed_reps, int) or completed_reps < 0:
        raise ValueError(f"Invalid summary completed_reps: {completed_reps}")

    # Check repetition count invariants across summary and attempts
    if attempts.get("counter_completed_reps") != completed_reps:
        raise ValueError(
            f"Repetition count mismatch: summary completed_reps ({completed_reps}) "
            f"!= attempts counter_completed_reps ({attempts.get('counter_completed_reps')})"
        )

    if attempts.get("counted_attempts") != completed_reps:
        raise ValueError(
            f"Counted attempts mismatch: summary completed_reps ({completed_reps}) "
            f"!= attempts counted_attempts ({attempts.get('counted_attempts')})"
        )

    attempt_obs = summary.get("attempt_observations")
    if isinstance(attempt_obs, dict):
        if attempt_obs.get("counted_attempts") != completed_reps:
            raise ValueError(
                f"summary attempt_observations.counted_attempts ({attempt_obs.get('counted_attempts')}) "
                f"!= completed_reps ({completed_reps})"
            )

    # Validate attempts events list
    events = attempts.get("events")
    if not isinstance(events, list):
        raise ValueError("attempts['events'] must be a list")

    counted_events = [e for e in events if e.get("status") == "counted"]
    if len(counted_events) != completed_reps:
        raise ValueError(
            f"Counted events count in attempts ({len(counted_events)}) != completed_reps ({completed_reps})"
        )

    # Check timestamps monotonicity and non-negativity in attempts
    for ev in counted_events:
        st = ev.get("start_time_s")
        et = ev.get("end_time_s")
        if st is None or et is None or st < 0 or et < st:
            raise ValueError(f"Invalid timestamp interval in attempt {ev.get('attempt_id')}: start={st}, end={et}")

    # Check phase feedback consistency if provided
    if feedback_phases is not None and isinstance(feedback_phases, dict):
        rep_phases = feedback_phases.get("repetition_phases") or feedback_phases.get("phase_events") or []
        if len(rep_phases) != completed_reps:
            raise ValueError(
                f"feedback_phases repetition count ({len(rep_phases)}) != completed_reps ({completed_reps})"
            )
        for i, phase in enumerate(rep_phases):
            ev = counted_events[i]
            if "attempt_id" in phase and phase["attempt_id"] != ev.get("attempt_id"):
                raise ValueError(
                    f"feedback_phases rep {i+1} attempt_id ({phase.get('attempt_id')}) "
                    f"!= attempts event attempt_id ({ev.get('attempt_id')})"
                )
            pst = phase.get("start_time_s")
            pet = phase.get("end_time_s")
            if pst is not None and ev.get("start_time_s") is not None:
                if not math.isclose(pst, ev["start_time_s"], abs_tol=0.05):
                    raise ValueError(f"feedback_phases rep {i+1} start_time_s mismatch with attempt event")
            if pet is not None and ev.get("end_time_s") is not None:
                if not math.isclose(pet, ev["end_time_s"], abs_tol=0.05):
                    raise ValueError(f"feedback_phases rep {i+1} end_time_s mismatch with attempt event")

    # Check coaching feedback consistency if provided
    if coaching_feedback is not None and isinstance(coaching_feedback, dict):
        overall = coaching_feedback.get("overall_summary")
        if isinstance(overall, dict):
            c_reps = overall.get("total_counted_reps")
            if c_reps is not None and c_reps != completed_reps:
                raise ValueError(
                    f"coaching_feedback overall total_counted_reps ({c_reps}) != completed_reps ({completed_reps})"
                )
            u_attempts = attempts.get("uncounted_attempts")
            c_uncounted = overall.get("total_uncounted_attempts")
            if u_attempts is not None and c_uncounted is not None and u_attempts != c_uncounted:
                raise ValueError(
                    f"coaching_feedback total_uncounted_attempts ({c_uncounted}) != attempts uncounted_attempts ({u_attempts})"
                )

        cues = coaching_feedback.get("per_repetition_cues")
        if isinstance(cues, list):
            counted_cues = [c for c in cues if c.get("is_counted", True)]
            if len(counted_cues) != completed_reps:
                raise ValueError(
                    f"coaching_feedback counted cues count ({len(counted_cues)}) != completed_reps ({completed_reps})"
                )
            for i, cue in enumerate(counted_cues):
                ev = counted_events[i]
                if cue.get("attempt_id") != ev.get("attempt_id"):
                    raise ValueError(
                        f"coaching cue rep {i+1} attempt_id ({cue.get('attempt_id')}) "
                        f"!= attempts event attempt_id ({ev.get('attempt_id')})"
                    )
                c_ts = cue.get("timestamps", {})
                c_st = c_ts.get("start_time_s")
                c_inf = c_ts.get("inflection_time_s")
                c_et = c_ts.get("end_time_s")
                if c_st is not None and c_et is not None:
                    if c_et < c_st:
                        raise ValueError(f"coaching cue rep {i+1} has inverted timestamps: start={c_st}, end={c_et}")
                    if c_inf is not None and not (c_st <= c_inf <= c_et):
                        raise ValueError(
                            f"coaching cue rep {i+1} inflection ({c_inf}) outside [start, end] [{c_st}, {c_et}]"
                        )
                # Check min knee angle measurement consistency
                meas = cue.get("measurements", {})
                cue_knee = meas.get("min_knee_angle_deg")
                ev_knee = ev.get("min_knee_angle_deg")
                if cue_knee is not None and ev_knee is not None:
                    if not math.isclose(cue_knee, ev_knee, abs_tol=0.1):
                        raise ValueError(
                            f"coaching cue rep {i+1} min_knee_angle_deg ({cue_knee}) != attempts event ({ev_knee})"
                        )

        # Validate uncounted movement explanations hypotheses
        uncounted_exps = coaching_feedback.get("uncounted_movement_explanations")
        if isinstance(uncounted_exps, list):
            valid_hypotheses = {
                "possible_setup_or_weight_return",
                "shallow_depth",
                "tracking_lost",
                "unfinished_cycle",
                "ambiguous",
                "unknown",
            }
            for uexp in uncounted_exps:
                if isinstance(uexp, dict) and "context_hypothesis" in uexp:
                    hyp = uexp["context_hypothesis"]
                    if hyp is not None and hyp not in valid_hypotheses:
                        raise ValueError(f"Invalid context_hypothesis: '{hyp}'")


# ---------------------------------------------------------------------------
# Test Suite 1: Pure-Static & Markup Acceptance (Zero Dependencies)
# ---------------------------------------------------------------------------

class TestProductStaticAcceptance(unittest.TestCase):
    """Pure static and markup acceptance checks executable in any Python environment.

    Does not require FastAPI, httpx, or network/subprocess execution.
    Inspects real INDEX_HTML from gym_analyser.web.
    """

    def test_index_html_loads_and_has_required_sections(self) -> None:
        """Verify INDEX_HTML contains all required layout cards, headers, and UI containers."""
        self.assertTrue(len(INDEX_HTML) > 1000, "INDEX_HTML must be non-empty and substantially populated")
        self.assertIn("2D Gym Pose Analyser", INDEX_HTML)
        self.assertIn("Upload Squat Video", INDEX_HTML)
        self.assertIn("Analysis Job Status", INDEX_HTML)
        self.assertIn("Analysis Results", INDEX_HTML)
        self.assertIn("Recording Guidance", INDEX_HTML)
        self.assertIn("Observed Movement Events", INDEX_HTML)
        self.assertIn("Exported Artifacts", INDEX_HTML)

    def test_index_html_review_navigation_link(self) -> None:
        """Verify INDEX_HTML contains a prominent, accessible navigation link to /review/."""
        self.assertIn("/review/", INDEX_HTML, "INDEX_HTML must contain navigation link to /review/")
        self.assertIn("Human Review", INDEX_HTML)

    def test_index_html_accessibility_attributes(self) -> None:
        """Verify ARIA live regions, roles, and focus-visible indicators for accessibility."""
        self.assertIn('role="status"', INDEX_HTML)
        self.assertIn('aria-live="polite"', INDEX_HTML)
        self.assertIn(':focus-visible', INDEX_HTML)
        self.assertIn('aria-label=', INDEX_HTML)

    def test_index_html_client_validation_constraints(self) -> None:
        """Verify client-side validation logic matches authoritative server parameters."""
        self.assertIn("MAX_UPLOAD_BYTES", INDEX_HTML)
        self.assertIn("200 * 1024 * 1024", INDEX_HTML)
        self.assertIn(".mp4", INDEX_HTML)
        self.assertIn(".mov", INDEX_HTML)
        self.assertIn(".avi", INDEX_HTML)
        self.assertIn("validateClientFile", INDEX_HTML)

    def test_index_html_safety_and_non_normative_disclaimers(self) -> None:
        """Verify safety and non-normative disclaimers are present in the markup."""
        html_lower = INDEX_HTML.lower()
        self.assertIn("2d", html_lower)
        self.assertIn("kinematic", html_lower)
        self.assertIn("do not grade technique", html_lower)
        self.assertIn("safety", html_lower)
        self.assertIn("limitations", html_lower)


# ---------------------------------------------------------------------------
# Test Suite 2: End-to-End Workflow & Download Verification
# ---------------------------------------------------------------------------

@unittest.skipUnless(HAS_WEB_DEPS, "Web dependencies (fastapi, httpx) not installed")
class TestProductE2EWorkflow(unittest.TestCase):
    """End-to-end integration tests: video upload -> processing -> polling -> results -> downloads."""

    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.storage_dir = Path(self.temp_dir.name) / "web_jobs"
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.review_dir = Path(self.temp_dir.name) / "reviews"
        self.review_dir.mkdir(parents=True, exist_ok=True)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_full_e2e_video_upload_to_results_and_downloads(self) -> None:
        """Verify full end-to-end lifecycle:

        1. Upload valid video via POST /api/jobs -> HTTP 202 Accepted.
        2. Status polling via GET /api/jobs/{id} reflects completion.
        3. Results retrieval via GET /api/jobs/{id}/results contains complete telemetry.
        4. Downloads via GET /api/jobs/{id}/download/{filename} succeed for all allowed artifacts.
        """
        fixtures = make_consistent_fixtures(completed_reps=3, uncounted_attempts=1)

        def complete_runner(job_id: str, input_path: Path, results_dir: Path, log_path: Path, **kwargs: Any) -> Dict[str, Any]:
            results_dir.mkdir(parents=True, exist_ok=True)
            # Write all artifacts matching the shared contract
            (results_dir / "summary.json").write_text(json.dumps(fixtures["summary"]), encoding="utf-8")
            (results_dir / "attempts.json").write_text(json.dumps(fixtures["attempts"]), encoding="utf-8")
            (results_dir / "posture_summary.json").write_text(json.dumps({
                "torso_inclination_valid_frames": 175,
                "hip_angle_valid_frames": 175,
                "processed_frames": 180,
                "interpretation": "Descriptive 2D planar angles",
            }), encoding="utf-8")
            (results_dir / "feedback_observations.json").write_text(json.dumps({
                "schema_version": "1.0",
                "observations": [],
            }), encoding="utf-8")
            (results_dir / "feedback_phases.json").write_text(json.dumps(fixtures["feedback_phases"]), encoding="utf-8")
            (results_dir / "coaching_feedback.json").write_text(json.dumps(fixtures["coaching_feedback"]), encoding="utf-8")
            (results_dir / "angles.csv").write_text("frame_index,knee_angle_deg,hip_angle_deg\n0,165.0,170.0\n", encoding="utf-8")
            (results_dir / "angle_plot.png").write_bytes(b"\x89PNG\r\n\x1a\nfake-plot-bytes")
            (results_dir / "annotated.mp4").write_bytes(b"dummy-annotated-mp4-stream")
            return {"status": STATUS_COMPLETED, "exit_code": 0}

        def fake_create_preview(src: Path, dst: Path, **kw: Any) -> Dict[str, Any]:
            dst.write_bytes(b"dummy-preview-mp4-stream")
            return {"available": True, "warning": None}

        manager = JobManager(storage_dir=self.storage_dir, runner=complete_runner)
        app = create_app(job_manager=manager, enable_review=True, review_storage_root=self.review_dir)
        client = TestClient(app)

        with patch("gym_analyser.web_media.create_preview", side_effect=fake_create_preview):
            # 1. Video upload
            fake_video_bytes = b"ftypmp42fake-acceptance-video-stream-payload-98765"
            res_upload = client.post(
                "/api/jobs",
                files={"file": ("acceptance_test.mp4", io.BytesIO(fake_video_bytes), "video/mp4")},
            )
            self.assertEqual(res_upload.status_code, 202, f"Expected 202 Accepted, got: {res_upload.status_code}")
            upload_data = res_upload.json()
            self.assertIn("job_id", upload_data)
            self.assertEqual(upload_data["status"], "queued")
            self.assertEqual(upload_data["filename"], "acceptance_test.mp4")
            self.assertEqual(upload_data["file_size_bytes"], len(fake_video_bytes))
            job_id = upload_data["job_id"]
            self.assertTrue(is_valid_job_id(job_id))

            # Wait briefly for synchronous fake runner execution
            time.sleep(0.15)

            # 2. Status polling
            res_status = client.get(f"/api/jobs/{job_id}")
            self.assertEqual(res_status.status_code, 200)
            status_data = res_status.json()
            self.assertEqual(status_data["status"], STATUS_COMPLETED)
            self.assertIsNotNone(status_data["started_at"])
            self.assertIsNotNone(status_data["finished_at"])
            self.assertIsNone(status_data["error"])

            # 3. Results retrieval
            res_results = client.get(f"/api/jobs/{job_id}/results")
            self.assertEqual(res_results.status_code, 200)
            results_data = res_results.json()
            self.assertEqual(results_data["job_id"], job_id)
            self.assertEqual(results_data["status"], STATUS_COMPLETED)
            self.assertEqual(results_data["completed_reps"], 3)
            self.assertIsNotNone(results_data["summary"])
            self.assertIsNotNone(results_data["attempts"])

            # Validate that results pass cross-module consistency validation
            verify_feedback_mapping_consistency(
                summary=results_data["summary"],
                attempts=results_data["attempts"],
                feedback_phases=fixtures["feedback_phases"],
                coaching_feedback=fixtures["coaching_feedback"],
            )

            # 4. Downloads validation for all available artifacts
            available_artifacts = results_data.get("available_artifacts", {})
            self.assertTrue(len(available_artifacts) > 0, "available_artifacts must be populated")

            expected_mime_types = {
                "summary.json": "application/json",
                "attempts.json": "application/json",
                "posture_summary.json": "application/json",
                "feedback_observations.json": "application/json",
                "feedback_phases.json": "application/json",
                "angles.csv": "text/csv",
                "angle_plot.png": "image/png",
                "annotated.mp4": "video/mp4",
                "preview.mp4": "video/mp4",
            }

            for artifact_name, expected_media in expected_mime_types.items():
                if artifact_name in ALLOWED_ARTIFACTS:
                    res_dl = client.get(f"/api/jobs/{job_id}/download/{artifact_name}")
                    self.assertEqual(
                        res_dl.status_code,
                        200,
                        f"Download for '{artifact_name}' failed with status {res_dl.status_code}",
                    )
                    content_type = res_dl.headers.get("content-type", "")
                    self.assertIn(expected_media, content_type)
                    # Verify disposition: inline for video/image, attachment for json/csv
                    disposition = res_dl.headers.get("content-disposition", "")
                    if expected_media.startswith(("video/", "image/")):
                        self.assertIn("inline", disposition)
                    else:
                        self.assertIn("attachment", disposition)

    def test_coaching_feedback_download_contract(self) -> None:
        """Verify coaching_feedback.json download contract when present in allowed artifacts."""
        fixtures = make_consistent_fixtures(completed_reps=2, uncounted_attempts=0)

        def runner_with_cf(job_id: str, results_dir: Path, **kwargs: Any) -> Dict[str, Any]:
            results_dir.mkdir(parents=True, exist_ok=True)
            (results_dir / "summary.json").write_text(json.dumps(fixtures["summary"]), encoding="utf-8")
            (results_dir / "attempts.json").write_text(json.dumps(fixtures["attempts"]), encoding="utf-8")
            (results_dir / "coaching_feedback.json").write_text(json.dumps(fixtures["coaching_feedback"]), encoding="utf-8")
            return {"status": STATUS_COMPLETED, "exit_code": 0}

        manager = JobManager(storage_dir=self.storage_dir, runner=runner_with_cf)
        app = create_app(job_manager=manager, enable_review=True, review_storage_root=self.review_dir)
        client = TestClient(app)

        res_upload = client.post("/api/jobs", files={"file": ("squat.mp4", b"dummy video", "video/mp4")})
        self.assertEqual(res_upload.status_code, 202)
        job_id = res_upload.json()["job_id"]
        time.sleep(0.15)

        # Test download with ALLOWED_ARTIFACTS containing coaching_feedback.json
        extended_artifacts = dict(ALLOWED_ARTIFACTS)
        extended_artifacts["coaching_feedback.json"] = "application/json"

        with patch("gym_analyser.web.ALLOWED_ARTIFACTS", extended_artifacts):
            with patch("gym_analyser.web_jobs.ALLOWED_ARTIFACTS", extended_artifacts):
                res_dl = client.get(f"/api/jobs/{job_id}/download/coaching_feedback.json")
                self.assertEqual(res_dl.status_code, 200)
                self.assertIn("application/json", res_dl.headers.get("content-type", ""))
                dl_data = res_dl.json()
                self.assertEqual(dl_data.get("contract_version"), "1.0")
                self.assertEqual(dl_data.get("overall_summary", {}).get("total_counted_reps"), 2)


# ---------------------------------------------------------------------------
# Test Suite 3: Validation Rejections & Fault Resilience
# ---------------------------------------------------------------------------

@unittest.skipUnless(HAS_WEB_DEPS, "Web dependencies (fastapi, httpx) not installed")
class TestProductValidationRejections(unittest.TestCase):
    """Validation rejections: oversized uploads, invalid extensions, empty files, corrupt input."""

    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.storage_dir = Path(self.temp_dir.name) / "web_jobs"
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.review_dir = Path(self.temp_dir.name) / "reviews"
        self.review_dir.mkdir(parents=True, exist_ok=True)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_oversized_file_rejected_with_413_or_400(self) -> None:
        """Verify oversized upload rejection (> limit):

        1. POST /review/api/upload returns HTTP 413.
        2. POST /api/jobs returns HTTP 413 or HTTP 400 and cleans up temporary files.
        """
        # Configure small upload limit for fast deterministic testing
        small_limit = 512
        manager = JobManager(storage_dir=self.storage_dir, max_upload_bytes=small_limit)
        app = create_app(
            job_manager=manager,
            enable_review=True,
            review_storage_root=self.review_dir,
            review_max_upload_bytes=small_limit,
        )
        client = TestClient(app)

        oversized_data = b"X" * (small_limit + 100)

        # Test review upload returns 413 Payload Too Large
        res_review = client.post(
            "/review/api/upload",
            files={"file": ("large_review.mp4", io.BytesIO(oversized_data), "video/mp4")},
        )
        self.assertEqual(
            res_review.status_code,
            413,
            f"Review upload exceeding limit must return 413, got {res_review.status_code}",
        )

        # Test analysis job upload rejection (413 or 400 with exceeds detail)
        res_job = client.post(
            "/api/jobs",
            files={"file": ("large_job.mp4", io.BytesIO(oversized_data), "video/mp4")},
        )
        self.assertIn(
            res_job.status_code,
            (413, 400),
            f"Job upload exceeding limit must return 413 or 400, got {res_job.status_code}",
        )
        if res_job.status_code == 400:
            self.assertIn("exceeds", res_job.json().get("detail", "").lower())

        # Verify no orphaned job directory remains in storage_dir
        subdirs = [p for p in self.storage_dir.iterdir() if p.is_dir()]
        self.assertEqual(len(subdirs), 0, "Failed upload must clean up any created job directory")

    def test_disallowed_file_extension_rejected_with_400(self) -> None:
        """Verify non-video file extensions are rejected with HTTP 400 Bad Request."""
        manager = JobManager(storage_dir=self.storage_dir)
        app = create_app(job_manager=manager, enable_review=True, review_storage_root=self.review_dir)
        client = TestClient(app)

        disallowed_filenames = [
            "exploit.exe",
            "script.sh",
            "malware.py",
            "notes.txt",
            "config.json",
            "video_without_extension",
        ]

        for fname in disallowed_filenames:
            res = client.post(
                "/api/jobs",
                files={"file": (fname, io.BytesIO(b"some content payload"), "application/octet-stream")},
            )
            self.assertEqual(
                res.status_code,
                400,
                f"File '{fname}' with disallowed extension must return 400 Bad Request",
            )
            detail = res.json().get("detail", "").lower()
            self.assertTrue(
                "unsupported" in detail or "allowed" in detail or "format" in detail,
                f"Detail must explain unsupported format, got: {detail}",
            )

        # Ensure no directories leaked
        self.assertEqual(len(list(self.storage_dir.iterdir())), 0)

    def test_empty_input_file_rejected_with_400(self) -> None:
        """Verify empty (0 bytes) upload is rejected with HTTP 400 Bad Request."""
        manager = JobManager(storage_dir=self.storage_dir)
        app = create_app(job_manager=manager, enable_review=True, review_storage_root=self.review_dir)
        client = TestClient(app)

        res = client.post(
            "/api/jobs",
            files={"file": ("empty.mp4", io.BytesIO(b""), "video/mp4")},
        )
        self.assertEqual(res.status_code, 400)
        self.assertIn("empty", res.json().get("detail", "").lower())
        self.assertEqual(len(list(self.storage_dir.iterdir())), 0)

    def test_corrupt_video_failure_reported_gracefully(self) -> None:
        """Verify corrupt input causing worker error reports status: failed gracefully.

        Verifies:
        - Job metadata records status: 'failed' and non-empty error message.
        - GET /api/jobs/{id} returns 200 with error details.
        - JobManager releases busy state; server remains usable for subsequent jobs.
        """
        def failing_runner(job_id: str, input_path: Path, results_dir: Path, log_path: Path, **kwargs: Any) -> Dict[str, Any]:
            return {
                "status": STATUS_FAILED,
                "exit_code": 1,
                "error": "Corrupt video stream: unable to decode frame 0 header bytes",
            }

        manager = JobManager(storage_dir=self.storage_dir, runner=failing_runner)
        app = create_app(job_manager=manager, enable_review=True, review_storage_root=self.review_dir)
        client = TestClient(app)

        # Submit corrupted video job
        res_upload = client.post(
            "/api/jobs",
            files={"file": ("corrupt.mp4", io.BytesIO(b"garbage-bytes-not-a-video"), "video/mp4")},
        )
        self.assertEqual(res_upload.status_code, 202)
        job_id = res_upload.json()["job_id"]

        time.sleep(0.15)

        # Check job status endpoint
        res_status = client.get(f"/api/jobs/{job_id}")
        self.assertEqual(res_status.status_code, 200)
        status_data = res_status.json()
        self.assertEqual(status_data["status"], STATUS_FAILED)
        self.assertIn("corrupt", status_data["error"].lower())

        # Check server status endpoint: busy lock MUST be released
        res_server = client.get("/api/status")
        self.assertEqual(res_server.status_code, 200)
        self.assertFalse(res_server.json()["busy"], "Busy lock must be released after job failure")
        self.assertIsNone(res_server.json()["active_job_id"])

        # Subsequent job submission must be accepted without 429
        res_next = client.post(
            "/api/jobs",
            files={"file": ("next_video.mp4", io.BytesIO(b"valid bytes"), "video/mp4")},
        )
        self.assertEqual(res_next.status_code, 202, "Server must accept new jobs after previous job failure")


# ---------------------------------------------------------------------------
# Test Suite 4: Preserved Original API Contract
# ---------------------------------------------------------------------------

@unittest.skipUnless(HAS_WEB_DEPS, "Web dependencies (fastapi, httpx) not installed")
class TestProductPreservedAPIContract(unittest.TestCase):
    """Verify exact preservation of all original routes, schemas, and status codes."""

    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.storage_dir = Path(self.temp_dir.name) / "web_jobs"
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.review_dir = Path(self.temp_dir.name) / "reviews"
        self.review_dir.mkdir(parents=True, exist_ok=True)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_preserved_routes_existence_and_methods(self) -> None:
        """Verify original routes: GET /, POST /api/jobs, GET /api/status,

        GET /api/jobs/{id}, GET /api/jobs/{id}/results, GET /api/jobs/{id}/download/{filename}.
        """
        manager = JobManager(storage_dir=self.storage_dir)
        app = create_app(job_manager=manager, enable_review=True, review_storage_root=self.review_dir)
        client = TestClient(app)

        # 1. GET / -> 200, text/html
        res_root = client.get("/")
        self.assertEqual(res_root.status_code, 200)
        self.assertIn("text/html", res_root.headers.get("content-type", ""))

        # 2. GET /api/status -> 200, {busy: bool, active_job_id: Optional[str]}
        res_status = client.get("/api/status")
        self.assertEqual(res_status.status_code, 200)
        data_status = res_status.json()
        self.assertIn("busy", data_status)
        self.assertIsInstance(data_status["busy"], bool)
        self.assertIn("active_job_id", data_status)

        # 3. Invalid Job ID handling across endpoints -> 400 Bad Request
        invalid_ids = ["not-a-uuid", "12345", "not-uuid-formatted-string"]
        for bad_id in invalid_ids:
            res_bad_job = client.get(f"/api/jobs/{bad_id}")
            self.assertEqual(res_bad_job.status_code, 400)
            res_bad_results = client.get(f"/api/jobs/{bad_id}/results")
            self.assertEqual(res_bad_results.status_code, 400)
            res_bad_dl = client.get(f"/api/jobs/{bad_id}/download/summary.json")
            self.assertEqual(res_bad_dl.status_code, 400)

        # 4. Non-existent valid UUID handling across endpoints -> 404 Not Found
        non_existent_uuid = "00000000-0000-4000-8000-000000000000"
        res_404_job = client.get(f"/api/jobs/{non_existent_uuid}")
        self.assertEqual(res_404_job.status_code, 404)
        res_404_results = client.get(f"/api/jobs/{non_existent_uuid}/results")
        self.assertEqual(res_404_results.status_code, 404)
        res_404_dl = client.get(f"/api/jobs/{non_existent_uuid}/download/summary.json")
        self.assertEqual(res_404_dl.status_code, 404)

    def test_download_security_disallowed_and_path_traversal(self) -> None:
        """Verify strict download allowlist: path traversal and unlisted files return 404."""
        fixtures = make_consistent_fixtures(completed_reps=1)

        def runner(job_id: str, results_dir: Path, log_path: Path, **kwargs: Any) -> Dict[str, Any]:
            results_dir.mkdir(parents=True, exist_ok=True)
            (results_dir / "summary.json").write_text(json.dumps(fixtures["summary"]), encoding="utf-8")
            return {"status": STATUS_COMPLETED, "exit_code": 0}

        manager = JobManager(storage_dir=self.storage_dir, runner=runner)
        app = create_app(job_manager=manager, enable_review=True, review_storage_root=self.review_dir)
        client = TestClient(app)

        res_upload = client.post("/api/jobs", files={"file": ("sec_test.mp4", b"video payload", "video/mp4")})
        job_id = res_upload.json()["job_id"]
        time.sleep(0.15)

        # Traversal attempts
        traversal_names = [
            "../job.json",
            "../../job.json",
            "..\\job.json",
            "../input/source.mp4",
            "run.log",
            "secret.env",
            "not_an_artifact.txt",
        ]

        for unsafe_name in traversal_names:
            res_unsafe = client.get(f"/api/jobs/{job_id}/download/{unsafe_name}")
            self.assertIn(
                res_unsafe.status_code,
                (404, 400),
                f"Attempting to download '{unsafe_name}' must return 404 or 400, got {res_unsafe.status_code}",
            )

    def test_single_worker_concurrency_lock_returns_429(self) -> None:
        """Verify that concurrent submission returns HTTP 429 when worker is busy."""
        unblock_event = threading.Event()

        def slow_runner(**kwargs: Any) -> Dict[str, Any]:
            unblock_event.wait(timeout=5.0)
            return {"status": STATUS_COMPLETED, "exit_code": 0}

        manager = JobManager(storage_dir=self.storage_dir, runner=slow_runner)
        app = create_app(job_manager=manager, enable_review=True, review_storage_root=self.review_dir)
        client = TestClient(app)

        responses: List[Any] = []
        worker_thread = threading.Thread(
            target=lambda: responses.append(
                client.post("/api/jobs", files={"file": ("first.mp4", b"first video stream", "video/mp4")})
            )
        )
        worker_thread.start()

        try:
            # Wait until manager enters busy state
            deadline = time.monotonic() + 3.0
            while not manager.is_busy() and time.monotonic() < deadline:
                time.sleep(0.01)

            self.assertTrue(manager.is_busy(), "JobManager must report busy during active execution")

            # Second submission must be rejected with 429
            res_second = client.post(
                "/api/jobs",
                files={"file": ("second.mp4", b"second video stream", "video/mp4")},
            )
            self.assertEqual(res_second.status_code, 429)
            self.assertIn("busy", res_second.json().get("detail", "").lower())
        finally:
            unblock_event.set()
            worker_thread.join(timeout=5.0)

        self.assertEqual(responses[0].status_code, 202)


# ---------------------------------------------------------------------------
# Test Suite 5: Review Workspace Isolation & Storage Separation
# ---------------------------------------------------------------------------

@unittest.skipUnless(HAS_WEB_DEPS, "Web dependencies (fastapi, httpx) not installed")
class TestProductReviewIsolation(unittest.TestCase):
    """Verify human review workspace isolation: no ML inference, separated storage, clean mounting."""

    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.jobs_storage = Path(self.temp_dir.name) / "web_jobs"
        self.jobs_storage.mkdir(parents=True, exist_ok=True)
        self.review_storage = Path(self.temp_dir.name) / "human_reviews"
        self.review_storage.mkdir(parents=True, exist_ok=True)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_review_mount_serves_independently(self) -> None:
        """Verify /review/ serves human review workspace without triggering ML model download."""
        manager = JobManager(storage_dir=self.jobs_storage)
        app = create_app(job_manager=manager, enable_review=True, review_storage_root=self.review_storage)
        client = TestClient(app)

        res_review = client.get("/review/")
        self.assertEqual(res_review.status_code, 200)
        self.assertIn("text/html", res_review.headers.get("content-type", ""))
        self.assertIn("Review Workspace", res_review.text)

        # GET /review/ must not set analysis worker busy
        self.assertFalse(manager.is_busy())

    def test_review_storage_and_jobs_storage_isolation(self) -> None:
        """Verify strict storage isolation: review cases never enter web_jobs, and vice-versa."""
        manager = JobManager(storage_dir=self.jobs_storage)
        app = create_app(job_manager=manager, enable_review=True, review_storage_root=self.review_storage)
        client = TestClient(app)

        # 1. Upload review case
        raw_review_video = b"dummy-raw-human-review-video-bytes-443322"
        res_rev_upload = client.post(
            "/review/api/upload",
            files={"file": ("raw_eval_clip.mp4", io.BytesIO(raw_review_video), "video/mp4")},
        )
        self.assertEqual(res_rev_upload.status_code, 201)
        rev_case_id = res_rev_upload.json()["case_id"]

        # Check case exists in review_storage, NOT in jobs_storage
        self.assertTrue((self.review_storage / rev_case_id).is_dir())
        self.assertFalse((self.jobs_storage / rev_case_id).exists())

        # 2. Upload analysis job
        res_job_upload = client.post(
            "/api/jobs",
            files={"file": ("analysis_job_clip.mp4", io.BytesIO(b"job video content"), "video/mp4")},
        )
        self.assertEqual(res_job_upload.status_code, 202)
        job_id = res_job_upload.json()["job_id"]

        # Check job exists in jobs_storage, NOT in review_storage
        self.assertTrue((self.jobs_storage / job_id).is_dir())
        self.assertFalse((self.review_storage / job_id).exists())

    def test_review_operations_never_trigger_analysis_worker(self) -> None:
        """Verify submitting reviews or querying cases never triggers the ML analysis worker."""
        worker_called = threading.Event()

        def guarded_runner(**kwargs: Any) -> Dict[str, Any]:
            worker_called.set()
            return {"status": STATUS_COMPLETED, "exit_code": 0}

        manager = JobManager(storage_dir=self.jobs_storage, runner=guarded_runner)
        app = create_app(job_manager=manager, enable_review=True, review_storage_root=self.review_storage)
        client = TestClient(app)

        # Upload review case
        res_up = client.post(
            "/review/api/upload",
            files={"file": ("case_no_ml.mp4", io.BytesIO(b"raw review video"), "video/mp4")},
        )
        case_id = res_up.json()["case_id"]

        # Submit human review assessment
        review_payload = {
            "human_count": 5,
            "reviewer": "expert_evaluator",
            "raw_video_reviewed": True,
            "predictions_previously_seen": False,
            "notes": "Clear side-view squats evaluated manually.",
            "scope": "in_scope",
        }
        res_rev = client.post(f"/review/api/cases/{case_id}/review", json=review_payload)
        self.assertEqual(res_rev.status_code, 200)

        # List cases and export
        client.get("/review/api/cases")
        client.get("/review/api/export/counts")

        # Verify runner was NEVER invoked
        self.assertFalse(worker_called.is_set(), "Analysis runner must NEVER be called by review endpoints")
        self.assertFalse(manager.is_busy())

    def test_overlapping_storage_roots_rejected(self) -> None:
        """Verify that configuring review_storage_root overlapping with jobs_storage raises ValueError."""
        manager = JobManager(storage_dir=self.jobs_storage)
        with self.assertRaises(ValueError):
            create_app(job_manager=manager, enable_review=True, review_storage_root=self.jobs_storage)

    def test_enable_review_false_disables_mount(self) -> None:
        """Verify enable_review=False unmounts /review and strips review links from GET /."""
        manager = JobManager(storage_dir=self.jobs_storage)
        app = create_app(job_manager=manager, enable_review=False)
        client = TestClient(app)

        res_review = client.get("/review/")
        self.assertEqual(res_review.status_code, 404)

        res_root = client.get("/")
        self.assertNotIn('<nav class="review-nav"', res_root.text)


# ---------------------------------------------------------------------------
# Test Suite 6: Cross-Module Feedback & Repetition Consistency
# ---------------------------------------------------------------------------

class TestProductFeedbackMappingConsistency(unittest.TestCase):
    """Verify strict cross-module mapping invariants between summary, attempts, phases, and coaching feedback."""

    def test_consistent_telemetry_passes_all_checks(self) -> None:
        """Verify that a mutually consistent telemetry fixture passes all cross-module invariant checks."""
        fixtures = make_consistent_fixtures(completed_reps=3, uncounted_attempts=1)
        # Should execute cleanly without raising ValueError
        verify_feedback_mapping_consistency(
            summary=fixtures["summary"],
            attempts=fixtures["attempts"],
            feedback_phases=fixtures["feedback_phases"],
            coaching_feedback=fixtures["coaching_feedback"],
        )

    def test_repetition_count_mismatch_detected(self) -> None:
        """Verify that a discrepancy between summary and attempts repetition counts is detected."""
        fixtures = make_consistent_fixtures(completed_reps=3, uncounted_attempts=0)
        # Mutate attempts to have 2 reps instead of 3
        fixtures["attempts"]["counter_completed_reps"] = 2
        with self.assertRaises(ValueError) as ctx:
            verify_feedback_mapping_consistency(
                summary=fixtures["summary"],
                attempts=fixtures["attempts"],
            )
        self.assertIn("Repetition count mismatch", str(ctx.exception))

    def test_inverted_timestamps_detected(self) -> None:
        """Verify that inverted repetition timestamps (end < start) are detected and rejected."""
        fixtures = make_consistent_fixtures(completed_reps=2, uncounted_attempts=0)
        # Invert timestamp in first attempt event
        fixtures["attempts"]["events"][0]["start_time_s"] = 5.0
        fixtures["attempts"]["events"][0]["end_time_s"] = 2.0
        with self.assertRaises(ValueError) as ctx:
            verify_feedback_mapping_consistency(
                summary=fixtures["summary"],
                attempts=fixtures["attempts"],
            )
        self.assertIn("Invalid timestamp interval", str(ctx.exception))

    def test_coaching_cue_attempt_id_mismatch_detected(self) -> None:
        """Verify that attempt_id discrepancy between attempts and coaching cues is detected."""
        fixtures = make_consistent_fixtures(completed_reps=2, uncounted_attempts=0)
        # Mutate attempt_id in coaching cue
        fixtures["coaching_feedback"]["per_repetition_cues"][0]["attempt_id"] = 999
        with self.assertRaises(ValueError) as ctx:
            verify_feedback_mapping_consistency(
                summary=fixtures["summary"],
                attempts=fixtures["attempts"],
                coaching_feedback=fixtures["coaching_feedback"],
            )
        self.assertIn("attempt_id", str(ctx.exception))

    def test_live_coaching_feedback_generation_if_available(self) -> None:
        """If coaching_feedback module is present, test generate_coaching_feedback produces consistent outputs."""
        if not HAS_COACHING_FEEDBACK_MODULE:
            self.skipTest("coaching_feedback module not yet integrated in workspace")

        fixtures = make_consistent_fixtures(completed_reps=3, uncounted_attempts=1)
        generated = generate_coaching_feedback(
            summary=fixtures["summary"],
            attempts=fixtures["attempts"],
            feedback_phases=fixtures["feedback_phases"],
        )
        self.assertIsInstance(generated, dict)
        self.assertIn("overall_summary", generated)
        # Verify generated output passes cross-module consistency validation
        verify_feedback_mapping_consistency(
            summary=fixtures["summary"],
            attempts=fixtures["attempts"],
            feedback_phases=fixtures["feedback_phases"],
            coaching_feedback=generated,
        )

    def test_camera_view_adequate_null_and_supported_hypotheses(self) -> None:
        """Verify null camera adequacy and all contract context hypotheses pass validation."""
        fixtures = make_consistent_fixtures(completed_reps=2, uncounted_attempts=1)
        rq = fixtures["coaching_feedback"]["overall_summary"]["recording_quality"]
        rq["camera_view_adequate"] = None

        for hyp in (
            "possible_setup_or_weight_return",
            "shallow_depth",
            "tracking_lost",
            "unfinished_cycle",
            "ambiguous",
            "unknown",
        ):
            fixtures["coaching_feedback"]["uncounted_movement_explanations"][0]["context_hypothesis"] = hyp
            verify_feedback_mapping_consistency(
                summary=fixtures["summary"],
                attempts=fixtures["attempts"],
                feedback_phases=fixtures["feedback_phases"],
                coaching_feedback=fixtures["coaching_feedback"],
            )

    def test_invalid_context_hypothesis_rejected(self) -> None:
        """Verify unrecognized context hypothesis raises ValueError."""
        fixtures = make_consistent_fixtures(completed_reps=2, uncounted_attempts=1)
        fixtures["coaching_feedback"]["uncounted_movement_explanations"][0]["context_hypothesis"] = "invalid_hypothesis"
        with self.assertRaises(ValueError) as ctx:
            verify_feedback_mapping_consistency(
                summary=fixtures["summary"],
                attempts=fixtures["attempts"],
                coaching_feedback=fixtures["coaching_feedback"],
            )
        self.assertIn("context_hypothesis", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
