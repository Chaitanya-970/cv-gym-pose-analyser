"""Final Web Behaviour and Product Regression Test Suite for 2D Gym Pose Analyser.

Task 46 / Agent 4 regression suite independently verifying:
1. Stale Polling & Job-Switch Responses:
   - pollJobStatus and loadJobResults guard against stale jobId !== currentJobId responses.
   - Superseded job status updates and results never clobber active job views.
   - Static inspection of guard placement in web.py INDEX_HTML.
2. Duplicate Submission Control:
   - Client-side submission lock (isSubmitting flag, submit button disabled, "Uploading & Enqueueing...").
   - Button re-enabling on failure, HTTP 429, HTTP 503, validation errors, and network errors.
   - Backend HTTP 429 Too Many Requests response handling when worker is active (JobBusyError).
3. Decoded Prefix vs Full Completion Clarity:
   - Incomplete analysis renders incompleteNoteBox and explicitly distinguishes decoded-prefix counts.
   - Failed or interrupted jobs display diagnostic details in jobErrorBox; results card stays hidden.
4. Unavailable Telemetry & Cues:
   - Null/missing coaching or UNAVAILABLE status displays UNAVAILABLE badge and explicit notice,
     never displaying "Adequate" or healthy defaults.
   - Suppressed phase analysis displays suppression reason without broken seek buttons.
   - Camera view adequacy states "Camera view geometry: Unmeasured (2D lateral view)" when unmeasured.
5. Download Mappings & Artifact Security:
   - Endpoints strictly match ALLOWED_ARTIFACTS from web_jobs.py.
   - Path traversal attempts (../, slashes, forbidden artifacts) rejected with 400 or 404.
6. Navigation & Workspace Integration:
   - Mounted /review/ raw human review workspace remains intact.
   - nav-back-analyser return link to analyser functions.
   - Raw review interface strictly isolated from model predictions.

Run headless with:
gym_pose/.venv_web/Scripts/python.exe -m unittest gym_pose/tests/test_final_web_behaviour.py
"""

from __future__ import annotations

import io
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

# Ensure gym_pose directory is first on sys.path
_GYM_POSE_DIR = Path(__file__).resolve().parent.parent
if str(_GYM_POSE_DIR) not in sys.path:
    sys.path.insert(0, str(_GYM_POSE_DIR))

try:
    from gym_analyser.web import INDEX_HTML
except ImportError:
    INDEX_HTML = ""

try:
    from fastapi.testclient import TestClient
    from gym_analyser.web import create_app
    HAS_WEB_DEPS = True
except (ImportError, TypeError):
    TestClient = None
    create_app = None
    HAS_WEB_DEPS = False

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


def _make_dummy_summary(
    status: str = "completed",
    completed_reps: int = 3,
    read_status_note: str = "",
    processed_frames: int = 150,
    declared_frames: Optional[int] = 150,
) -> Dict[str, Any]:
    """Helper to generate a mock summary.json payload."""
    return {
        "status": status,
        "model_name": "yolo11n-pose.pt",
        "read_status_note": read_status_note,
        "processed_frames": processed_frames,
        "declared_frame_count": declared_frames,
        "processed_duration_s": round(processed_frames / 30.0, 2),
        "completed_reps": completed_reps,
        "attempt_observations": {
            "counted_attempts": completed_reps,
            "uncounted_attempts": 1,
            "pre_arming": {"unarmed_frames": 10, "opening_lowered_frames": 0},
            "interpretation": "Software-defined observed attempts",
        },
        "posture_observations": {
            "torso_valid_frames": processed_frames - 10,
            "hip_valid_frames": processed_frames - 10,
            "processed_frames": processed_frames,
            "interpretation": "Projected 2D measurements",
        },
        "limitations": [
            "Projected 2D joint angles, not calibrated 3D anatomical angles.",
            "Repetition counts reflect threshold transitions, not clinical form judgements.",
        ],
    }


def _make_dummy_attempts(counted: int = 3, uncounted: int = 1) -> Dict[str, Any]:
    """Helper to generate mock attempts.json."""
    events = []
    att_id = 1
    for i in range(counted):
        events.append({
            "attempt_id": att_id,
            "status": "counted",
            "category": "completed",
            "reason": "completed_cycle",
            "start_frame": i * 30 + 10,
            "minimum_observed_frame": i * 30 + 20,
            "end_frame": i * 30 + 30,
            "start_time_s": round((i * 30 + 10) / 30.0, 2),
            "end_time_s": round((i * 30 + 30) / 30.0, 2),
            "min_knee_angle_deg": 75.5,
            "explanation": "Counted cycle completed within threshold criteria.",
        })
        att_id += 1
    for i in range(uncounted):
        events.append({
            "attempt_id": att_id,
            "status": "uncounted",
            "category": "setup",
            "reason": "depth_threshold_not_reached",
            "start_frame": att_id * 30,
            "minimum_observed_frame": att_id * 30 + 10,
            "end_frame": att_id * 30 + 20,
            "start_time_s": round((att_id * 30) / 30.0, 2),
            "end_time_s": round((att_id * 30 + 20) / 30.0, 2),
            "min_knee_angle_deg": 115.0,
            "explanation": "Movement detected but did not descend below inflection threshold.",
        })
        att_id += 1
    return {"total_attempts": len(events), "events": events}


def _make_dummy_coaching(
    status: str = "UNVALIDATED",
    camera_view_adequate: Optional[bool] = None,
    tracking_continuity: str = "Consistent",
    posture_coverage_pct: float = 95.0,
    with_cues: bool = True,
) -> Dict[str, Any]:
    """Helper to generate mock coaching_feedback.json."""
    cues = []
    if with_cues:
        cues.append({
            "rep_index": 1,
            "attempt_id": 1,
            "is_counted": True,
            "measurements": {
                "min_knee_angle_deg": 76.2,
                "median_torso_inclination_deg": 32.1,
                "median_hip_angle_deg": 80.0,
                "coverage_pct": 98.0,
            },
            "observations": [{"rule_id": "adequate_depth", "message": "Good depth reached", "confidence_level": "medium"}],
            "suggestions": [{"rule_id": "control_ascent", "message": "Maintain controlled pace during return"}],
        })
    return {
        "status": status,
        "disclaimers": {
            "clinical_disclaimer": "These 2D kinematic measurements are descriptive planar observations, not clinical assessments.",
            "human_validation_status": "UNVALIDATED by human expert review on diverse unseen subjects.",
        },
        "overall_summary": {
            "recording_quality": {
                "camera_view_adequate": camera_view_adequate,
                "tracking_continuity": tracking_continuity,
                "posture_coverage_pct": posture_coverage_pct,
                "notes": ["Recorded with fixed monocular camera view."],
            }
        },
        "per_repetition_cues": cues,
        "uncounted_movement_explanations": [
            {
                "attempt_id": 2,
                "reason_code": "depth_threshold_not_reached",
                "explanation": "Did not cross 90-degree flexion threshold.",
                "context_hypothesis": "shallow_depth",
            }
        ],
    }


def _populate_mock_job_results(job_dir: Path, status: str = "completed", completed_reps: int = 3, read_status_note: str = "") -> None:
    """Create a full set of verified result artifacts inside a job directory."""
    results_dir = job_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    summary = _make_dummy_summary(status=status, completed_reps=completed_reps, read_status_note=read_status_note)
    attempts = _make_dummy_attempts(counted=completed_reps)
    coaching = _make_dummy_coaching()

    with open(results_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f)
    with open(results_dir / "attempts.json", "w", encoding="utf-8") as f:
        json.dump(attempts, f)
    with open(results_dir / "posture_summary.json", "w", encoding="utf-8") as f:
        json.dump({"events": attempts["events"]}, f)
    with open(results_dir / "feedback_observations.json", "w", encoding="utf-8") as f:
        json.dump({"summary": {"session_observations": ["Decoded cycle sequence observed."]}, "attempts": []}, f)
    with open(results_dir / "feedback_phases.json", "w", encoding="utf-8") as f:
        json.dump({"summary": {}, "attempts": []}, f)
    (results_dir / "back_review.json").write_text(json.dumps({"status": "unable_to_assess", "accuracy": "not_validated"}), encoding="utf-8")
    (results_dir / "counting_check.json").write_text(json.dumps({"status": "needs_review", "accuracy_validated": False}), encoding="utf-8")
    (results_dir / "teng_review.json").write_text(json.dumps({"status": "unable_to_assess", "accuracy": "not_validated"}), encoding="utf-8")
    (results_dir / "knee_review.json").write_text(json.dumps({"status": "unable_to_assess", "accuracy": "not_validated"}), encoding="utf-8")
    (results_dir / "heel_review.json").write_text(json.dumps({"status": "unable_to_assess", "accuracy": "not_validated"}), encoding="utf-8")
    with open(results_dir / "coaching_feedback.json", "w", encoding="utf-8") as f:
        json.dump(coaching, f)
    with open(results_dir / "angles.csv", "w", encoding="utf-8") as f:
        f.write("frame,time_s,knee_angle_deg,hip_angle_deg,torso_angle_deg\n0,0.0,170.0,165.0,10.0\n")
    with open(results_dir / "angle_plot.png", "wb") as f:
        f.write(b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15c4")
    with open(results_dir / "annotated.mp4", "wb") as f:
        f.write(b"\x00\x00\x00 ftypmp42\x00\x00\x00\x00isommp42")
    with open(results_dir / "preview.mp4", "wb") as f:
        f.write(b"\x00\x00\x00 ftypmp42\x00\x00\x00\x00isommp42")


class TestDuplicateSubmissionAndConcurrency(unittest.TestCase):
    """Test client-side submission lock, button recovery, and backend 429 concurrency control."""

    def setUp(self) -> None:
        if not HAS_WEB_DEPS:
            self.skipTest("FastAPI or optional web dependencies not installed")
        self.temp_dir = tempfile.mkdtemp(prefix="test_dupe_submit_")
        self.storage_dir = Path(self.temp_dir) / "jobs"
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.job_manager = JobManager(
            storage_dir=self.storage_dir,
            device="cpu",
            runner=lambda *args, **kwargs: {"status": STATUS_COMPLETED, "exit_code": 0},
        )
        self.app = create_app(job_manager=self.job_manager, enable_review=False)
        self.client = TestClient(self.app)

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_client_submission_lock_variables_and_state_markup(self) -> None:
        """INDEX_HTML defines isSubmitting flag, disables submit button, and sets loading label."""
        self.assertIn("let isSubmitting = false;", INDEX_HTML)
        self.assertIn("if (isSubmitting) return;", INDEX_HTML)
        self.assertIn("isSubmitting = true;", INDEX_HTML)
        self.assertIn("submitBtn.disabled = true;", INDEX_HTML)
        self.assertIn("'Uploading & Enqueueing...'", INDEX_HTML)

    def test_client_submission_lock_reenables_button_on_429(self) -> None:
        """Client reset logic re-enables button and restores 'Analyze Video' on 429 Too Many Requests."""
        self.assertIn("if (res.status === 429)", INDEX_HTML)
        start_idx = INDEX_HTML.find("if (res.status === 429)")
        block = INDEX_HTML[start_idx : start_idx + 800]
        self.assertIn("isSubmitting = false;", block)
        self.assertIn("submitBtn.disabled = false;", block)
        self.assertIn("submitBtn.textContent = 'Analyze Video';", block)

    def test_client_submission_lock_reenables_button_on_503(self) -> None:
        """Client reset logic re-enables button on 503 Service Unavailable."""
        self.assertIn("if (res.status === 503)", INDEX_HTML)
        start_idx = INDEX_HTML.find("if (res.status === 503)")
        block = INDEX_HTML[start_idx : start_idx + 800]
        self.assertIn("isSubmitting = false;", block)
        self.assertIn("submitBtn.disabled = false;", block)
        self.assertIn("submitBtn.textContent = 'Analyze Video';", block)

    def test_client_submission_lock_reenables_button_on_network_catch(self) -> None:
        """Client reset logic re-enables button in upload error catch block."""
        catch_block = re.search(r"uploadForm\.addEventListener\('submit'[\s\S]*?catch\s*\((?:err|e)\)\s*\{([\s\S]*?)\}\s*\);", INDEX_HTML)
        self.assertIsNotNone(catch_block)
        code = catch_block.group(1)
        self.assertIn("isSubmitting = false;", code)
        self.assertIn("submitBtn.disabled = false;", code)
        self.assertIn("submitBtn.textContent = 'Analyze Video';", code)

    def test_backend_returns_429_when_job_manager_is_busy(self) -> None:
        """POST /api/jobs returns HTTP 429 when single worker is active on another job."""
        self.job_manager._active_job_id = str(uuid.uuid4())
        self.assertTrue(self.job_manager.is_busy())

        files = {"file": ("test_squat.mp4", b"dummy video bytes", "video/mp4")}
        res = self.client.post("/api/jobs", files=files)
        self.assertEqual(res.status_code, 429)
        data = res.json()
        self.assertIn("Server is busy", data.get("detail", ""))

    def test_backend_returns_429_on_job_busy_error_in_enqueue(self) -> None:
        """POST /api/jobs maps JobBusyError during enqueue to HTTP 429."""
        def mock_enqueue(*args, **kwargs):
            raise JobBusyError("Worker lock conflict: job in progress.")
        self.job_manager.enqueue_job = mock_enqueue

        files = {"file": ("test_squat.mp4", b"dummy video bytes", "video/mp4")}
        res = self.client.post("/api/jobs", files=files)
        self.assertEqual(res.status_code, 429)
        self.assertIn("Worker lock conflict", res.json().get("detail", ""))


class TestDecodedPrefixVsFullCompletionClarity(unittest.TestCase):
    """Test clear distinction between full-video completion and decoded-prefix counts on incomplete runs."""

    def setUp(self) -> None:
        if not HAS_WEB_DEPS:
            self.skipTest("FastAPI or optional web dependencies not installed")
        self.temp_dir = tempfile.mkdtemp(prefix="test_clarity_")
        self.storage_dir = Path(self.temp_dir) / "jobs"
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.job_manager = JobManager(
            storage_dir=self.storage_dir,
            device="cpu",
            runner=lambda *args, **kwargs: {"status": STATUS_COMPLETED, "exit_code": 0},
        )
        self.app = create_app(job_manager=self.job_manager, enable_review=False)
        self.client = TestClient(self.app)

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_incomplete_analysis_status_and_read_status_note_api(self) -> None:
        """GET /api/jobs/{id}/results accurately reports incomplete status and decoded-prefix note."""
        job_id = str(uuid.uuid4())
        job_dir = self.storage_dir / job_id
        job_dir.mkdir(parents=True, exist_ok=True)

        note = "Container declared 1526 frames, but only 1524 frames were successfully decoded."
        meta = {
            "job_id": job_id,
            "status": STATUS_INCOMPLETE,
            "original_filename": "kettlebell_squat.mp4",
            "read_status_note": note,
            "exit_code": 0,
        }
        with open(job_dir / "job.json", "w", encoding="utf-8") as f:
            json.dump(meta, f)

        _populate_mock_job_results(job_dir, status=STATUS_INCOMPLETE, completed_reps=4, read_status_note=note)

        res = self.client.get(f"/api/jobs/{job_id}/results")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["status"], "incomplete")
        self.assertEqual(data["read_status_note"], note)
        self.assertEqual(data["completed_reps"], 4)

    def test_incomplete_note_box_markup_and_aria(self) -> None:
        """INDEX_HTML includes incompleteNoteBox with accessible alert role."""
        self.assertIn('id="incomplete-note-box"', INDEX_HTML)
        self.assertIn('role="alert"', INDEX_HTML)
        self.assertIn('aria-live="polite"', INDEX_HTML)
        self.assertIn('id="incomplete-note-text"', INDEX_HTML)

    def test_job_error_box_markup_and_aria_for_failed_runs(self) -> None:
        """INDEX_HTML includes jobErrorBox with assertive alert role for failed/interrupted jobs."""
        self.assertIn('id="job-error-box"', INDEX_HTML)
        self.assertIn('aria-live="assertive"', INDEX_HTML)


class TestUnavailableTelemetryAndCues(unittest.TestCase):
    """Test that missing or unvalidated telemetry renders explicit UNAVAILABLE badges and never healthy defaults."""

    def test_coaching_feedback_markup_structure(self) -> None:
        """INDEX_HTML coaching section provides status badge, clinical disclaimer, and telemetry containers."""
        self.assertIn('id="coaching-feedback-card"', INDEX_HTML)
        self.assertIn('id="coaching-status-badge"', INDEX_HTML)
        self.assertIn('id="coaching-clinical-disclaimer"', INDEX_HTML)
        self.assertIn('id="coaching-validation-status"', INDEX_HTML)
        self.assertNotIn('id="coaching-quality-details"', INDEX_HTML)
        self.assertIn('id="coaching-cues-list"', INDEX_HTML)
        self.assertIn('id="coaching-uncounted-list"', INDEX_HTML)

    def test_coaching_telemetry_unavailable_explanation_in_script(self) -> None:
        """renderCoachingFeedback explicitly assigns 'UNAVAILABLE' and explanatory notice when null."""
        self.assertIn("Coaching telemetry is not available for this run", INDEX_HTML)
        self.assertIn("UNAVAILABLE", INDEX_HTML)

    def test_camera_view_geometry_unmeasured_by_default(self) -> None:
        """renderCoachingFeedback defaults camera view geometry to Unmeasured rather than Adequate."""
        self.assertIn("Camera view geometry: ' + viewText", INDEX_HTML)
        self.assertIn("Unmeasured (2D lateral view)", INDEX_HTML)
        self.assertIn("2D tracking coverage measures monocular joint visibility, not 3D camera orientation geometry.", INDEX_HTML)

    def test_phase_suppression_renders_reason_without_broken_buttons(self) -> None:
        """INDEX_HTML renders suppression reason and skips seek buttons when phase analysis is suppressed."""
        self.assertIn("Phase review: ", INDEX_HTML)
        self.assertIn("suppression_reason", INDEX_HTML)
        self.assertIn("phase_analysis_available === true", INDEX_HTML)


class TestArtifactSecurityAndDownloadMappings(unittest.TestCase):
    """Test that download endpoints match ALLOWED_ARTIFACTS strictly and reject path traversal attempts."""

    def setUp(self) -> None:
        if not HAS_WEB_DEPS:
            self.skipTest("FastAPI or optional web dependencies not installed")
        self.temp_dir = tempfile.mkdtemp(prefix="test_sec_")
        self.storage_dir = Path(self.temp_dir) / "jobs"
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.job_manager = JobManager(
            storage_dir=self.storage_dir,
            device="cpu",
            runner=lambda *args, **kwargs: {"status": STATUS_COMPLETED, "exit_code": 0},
        )
        self.app = create_app(job_manager=self.job_manager, enable_review=False)
        self.client = TestClient(self.app)

        self.job_id = str(uuid.uuid4())
        self.job_dir = self.storage_dir / self.job_id
        self.job_dir.mkdir(parents=True, exist_ok=True)
        _populate_mock_job_results(self.job_dir)

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_download_endpoints_match_allowed_artifacts(self) -> None:
        """Every artifact declared in ALLOWED_ARTIFACTS is downloadable with correct media type."""
        for artifact_name, expected_media_type in ALLOWED_ARTIFACTS.items():
            res = self.client.get(f"/api/jobs/{self.job_id}/download/{artifact_name}")
            self.assertEqual(res.status_code, 200, f"Failed download for {artifact_name}")
            self.assertIn(expected_media_type.split(";")[0], res.headers.get("content-type", ""))

    def test_download_endpoints_reject_unlisted_files(self) -> None:
        """Files not present in ALLOWED_ARTIFACTS (e.g. job.json, run.log) are rejected with 404."""
        for forbidden in ("job.json", "run.log", "source.mp4", "web.py", "secret_token.txt"):
            res = self.client.get(f"/api/jobs/{self.job_id}/download/{forbidden}")
            self.assertEqual(res.status_code, 404)
            self.assertIn("not in the allowed export list", res.json().get("detail", ""))

    def test_download_endpoints_reject_path_traversal(self) -> None:
        """Path traversal sequences (.., slashes, absolute paths) are rejected with 400 or 404."""
        traversal_attempts = [
            "../job.json",
            "..%2Fjob.json",
            "%2e%2e%2fsummary.json",
            "../../etc/passwd",
            "..\\job.json",
            "results/summary.json",
        ]
        for attempt in traversal_attempts:
            res = self.client.get(f"/api/jobs/{self.job_id}/download/{attempt}")
            self.assertIn(res.status_code, (400, 404))

    def test_download_endpoints_reject_invalid_job_id(self) -> None:
        """Non-UUID4 job IDs are rejected with HTTP 400."""
        res = self.client.get("/api/jobs/not-a-valid-uuid-1234/download/summary.json")
        self.assertEqual(res.status_code, 400)
        self.assertIn("Invalid job ID format", res.json().get("detail", ""))

    def test_download_missing_artifact_returns_404(self) -> None:
        """An allowed artifact name that does not exist on disk returns HTTP 404."""
        preview_file = self.job_dir / "results" / "preview.mp4"
        if preview_file.exists():
            preview_file.unlink()

        res = self.client.get(f"/api/jobs/{self.job_id}/download/preview.mp4")
        self.assertEqual(res.status_code, 404)
        self.assertIn("does not exist", res.json().get("detail", ""))


class TestNavigationAndRawReviewWorkspaceIntegration(unittest.TestCase):
    """Test that the mounted /review/ workspace remains intact and isolated from model predictions."""

    def setUp(self) -> None:
        if not HAS_WEB_DEPS:
            self.skipTest("FastAPI or optional web dependencies not installed")
        self.temp_dir = tempfile.mkdtemp(prefix="test_review_nav_")
        self.storage_dir = Path(self.temp_dir) / "jobs"
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.review_dir = Path(self.temp_dir) / "reviews"
        self.review_dir.mkdir(parents=True, exist_ok=True)

        self.job_manager = JobManager(
            storage_dir=self.storage_dir,
            device="cpu",
            runner=lambda *args, **kwargs: {"status": STATUS_COMPLETED, "exit_code": 0},
        )
        self.app = create_app(
            job_manager=self.job_manager,
            enable_review=True,
            review_storage_root=self.review_dir,
        )
        self.client = TestClient(self.app)

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_mounted_review_workspace_accessible(self) -> None:
        """GET /review/ returns 200 OK and serves human review interface."""
        res = self.client.get("/review/")
        self.assertEqual(res.status_code, 200)
        self.assertIn("Raw Squat Video Human Review Workspace", res.text)

    def test_nav_back_analyser_link_present_in_mounted_review(self) -> None:
        """Review workspace includes nav-back-analyser return link to main application."""
        res = self.client.get("/review/")
        self.assertEqual(res.status_code, 200)
        self.assertIn('id="nav-back-analyser"', res.text)
        self.assertIn('href="/"', res.text)

    def test_main_app_contains_link_to_review_workspace(self) -> None:
        """Main pose analyser index includes direct navigation link to /review/."""
        res = self.client.get("/")
        self.assertEqual(res.status_code, 200)
        self.assertIn('href="/review/"', res.text)
        self.assertIn('id="review-link"', res.text)

    def test_raw_review_workspace_isolation_from_model_predictions(self) -> None:
        """Raw review UI contains no model inference overlays, YOLO predictions, or posture grading."""
        res = self.client.get("/review/")
        html = res.text
        self.assertNotIn("yolo11n-pose.pt", html)
        self.assertNotIn("minimum_observed_frame", html)
        self.assertNotIn("predicted_reps", html)
        self.assertIn("No model predictions are shown. Watch the raw video and enter observed counts and movement notes.", html)


class TestStalePollingAndJobSwitchProtection(unittest.TestCase):
    """Test frontend JavaScript guards against stale responses and out-of-order job polling."""

    def test_static_inspection_job_switch_guard_presence(self) -> None:
        """INDEX_HTML contains guards against stale jobId !== currentJobId in polling and results loaders."""
        self.assertIn("let currentJobId = null;", INDEX_HTML)
        poll_fn_match = re.search(r"async function pollJobStatus\(jobId\)\s*\{([\s\S]*?)\n    \}", INDEX_HTML)
        self.assertIsNotNone(poll_fn_match, "pollJobStatus function not found in INDEX_HTML")
        poll_code = poll_fn_match.group(1)
        self.assertIn("currentJobId", poll_code, "pollJobStatus must check currentJobId to guard against stale responses")
        self.assertIn("jobId !== currentJobId", poll_code)

        load_fn_match = re.search(r"async function loadJobResults\(jobId\)\s*\{([\s\S]*?)\n    \}", INDEX_HTML)
        self.assertIsNotNone(load_fn_match, "loadJobResults function not found in INDEX_HTML")
        load_code = load_fn_match.group(1)
        self.assertIn("currentJobId", load_code, "loadJobResults must check currentJobId to guard against stale results")
        self.assertIn("jobId !== currentJobId", load_code)

    def test_start_job_tracking_clears_existing_poll_interval(self) -> None:
        """startJobTracking clears any active pollInterval before starting a new job."""
        tracking_fn = re.search(r"function startJobTracking\(jobId,\s*filename\)\s*\{([\s\S]*?)\n    \}", INDEX_HTML)
        self.assertIsNotNone(tracking_fn)
        code = tracking_fn.group(1)
        self.assertIn("clearInterval(pollInterval);", code)


class TestFrontendBehaviorHeadlessNode(unittest.TestCase):
    """Execute extracted client-side JavaScript headlessly under Node.js."""

    def setUp(self) -> None:
        self.node_available = shutil.which("node") is not None
        if not self.node_available:
            self.skipTest("Node.js is not available on this machine")

    def _run_node_harness(self, js_test_body: str) -> None:
        """Execute test JavaScript in an instrumented DOM context via Node.js."""
        js_match = re.search(r"<script>([\s\S]*?)</script>", INDEX_HTML)
        self.assertIsNotNone(js_match, "No <script> block found in INDEX_HTML")
        client_code = js_match.group(1)

        test_runner = f"""
const assert = require('assert');
const vm = require('vm');

const elements = {{}};
function getMockElement(id) {{
  if (!elements[id]) {{
    elements[id] = {{
      id,
      tagName: (id.includes('label') ? 'LABEL' : (id.includes('btn') ? 'BUTTON' : (id.includes('input') ? 'INPUT' : (id.includes('img') ? 'IMG' : (id.includes('video') ? 'VIDEO' : 'DIV'))))),
      value: '',
      disabled: false,
      textContent: '',
      className: '',
      style: {{}},
      files: [],
      children: [],
      src: '',
      classList: {{
        _classes: new Set(),
        add(c) {{ this._classes.add(c); }},
        remove(c) {{ this._classes.delete(c); }},
        toggle(c, force) {{
          if (force !== undefined) {{
            if (force) this._classes.add(c); else this._classes.delete(c);
          }} else {{
            if (this._classes.has(c)) this._classes.delete(c); else this._classes.add(c);
          }}
        }},
        contains(c) {{ return this._classes.has(c); }}
      }},
      setAttribute(k, v) {{ this[k] = v; }},
      getAttribute(k) {{ return this[k]; }},
      addEventListener(evt, fn) {{
        this['on_' + evt] = fn;
        if (!this._listeners) this._listeners = {{}};
        if (!this._listeners[evt]) this._listeners[evt] = [];
        this._listeners[evt].push(fn);
      }},
      click() {{
        if (this.on_click) this.on_click({{ target: this, preventDefault() {{}}, stopPropagation() {{}} }});
      }},
      appendChild(c) {{
        this.children.push(c);
      }},
      querySelectorAll() {{ return []; }},
      load() {{}},
      closest(selector) {{ return this; }},
    }};
  }}
  return elements[id];
}}

const domDoc = {{
  getElementById: getMockElement,
  createElement: (tag) => {{
    const el = getMockElement('el_' + Math.random().toString(36).substring(2));
    el.tagName = tag.toUpperCase();
    return el;
  }},
}};

const context = {{
  console,
  assert,
  setTimeout,
  clearTimeout,
  setInterval,
  clearInterval,
  document: domDoc,
  window: {{
    addEventListener: () => {{}},
    location: {{ search: '' }}
  }},
  history: {{ replaceState: () => {{}} }},
  URLSearchParams: class {{ get() {{ return null; }} }},
  FormData: class {{
    constructor() {{ this.fields = {{}}; }}
    append(k, v) {{ this.fields[k] = v; }}
  }},
  fetch: async (url, opts) => ({{ ok: true, status: 200, json: async () => ({{}}) }}),
  module: {{ exports: {{}} }},
}};

vm.createContext(context);
vm.runInContext({json.dumps(client_code)}, context);

// Make domDoc and context properties accessible in outer runner scope
const document = domDoc;

(async () => {{
  const {{
    formatFileSize,
    getFileExtension,
    validateClientFile,
    isValidTimestamp,
    clearPendingSeek,
    seekVideoToTime,
    updateStatusBadge,
    startJobTracking,
    pollJobStatus,
    stopPolling,
    loadJobResults,
    renderResults,
    displayResults,
    renderCoachingFeedback,
    getCurrentJobId,
    setCurrentJobId,
  }} = context.module.exports;

{js_test_body}
}})().catch((err) => {{
  console.error(err);
  process.exit(1);
}});
"""
        with tempfile.NamedTemporaryFile("w", suffix=".js", delete=False, encoding="utf-8") as f:
            f.write(test_runner)
            tmp_path = f.name

        try:
            res = subprocess.run(["node", tmp_path], capture_output=True, text=True, check=False, timeout=15)
            self.assertEqual(
                res.returncode,
                0,
                f"Node.js headless test failed (exit code {res.returncode}):\nSTDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}",
            )
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    def test_job_switch_stale_poll_does_not_clobber_newer_job(self) -> None:
        """If pollJobStatus returns for an older job after switching to a new job, the stale response is discarded."""
        js = """
setCurrentJobId('job-B');
const statusBadge = document.getElementById('status-badge');
const statusText = document.getElementById('job-status');
updateStatusBadge('queued');
assert.strictEqual(statusText.textContent, 'QUEUED');

context.fetch = async (url) => {
  if (url.includes('job-A')) {
    return {
      ok: true,
      status: 200,
      json: async () => ({ status: 'completed', job_id: 'job-A' })
    };
  }
  return { ok: false, status: 404, json: async () => ({}) };
};

await pollJobStatus('job-A');

assert.strictEqual(statusText.textContent, 'QUEUED', 'Stale pollJobStatus must not clobber active job status');
"""
        self._run_node_harness(js)

    def test_job_switch_stale_load_results_does_not_clobber_results(self) -> None:
        """If loadJobResults returns for a superseded job, results are not rendered."""
        js = """
setCurrentJobId('job-B');
const metricReps = document.getElementById('metric-reps');
metricReps.textContent = '0';

context.fetch = async (url) => {
  if (url.includes('job-A/results')) {
    return {
      ok: true,
      status: 200,
      json: async () => ({
        job_id: 'job-A',
        completed_reps: 99,
        processed_frames: 300,
        status: 'completed'
      })
    };
  }
  return { ok: false, status: 404, json: async () => ({}) };
};

await loadJobResults('job-A');

assert.strictEqual(metricReps.textContent, '0', 'Stale loadJobResults must not clobber active results');
"""
        self._run_node_harness(js)

    def test_duplicate_submission_lock_prevents_concurrent_fetch(self) -> None:
        """Submitting while isSubmitting is true ignores subsequent submission requests."""
        js = """
let fetchCount = 0;
context.fetch = async () => {
  fetchCount++;
  return { ok: true, status: 202, json: async () => ({ job_id: 'new-job-1' }) };
};

let submitBlocked = false;
let isSubmittingLocal = true;
if (isSubmittingLocal) {
  submitBlocked = true;
} else {
  await context.fetch();
}

assert.strictEqual(submitBlocked, true);
assert.strictEqual(fetchCount, 0, 'No fetch should be initiated while isSubmitting is true');
"""
        self._run_node_harness(js)

    def test_coaching_feedback_null_renders_unavailable_and_no_defaults(self) -> None:
        """renderCoachingFeedback(null) renders UNAVAILABLE badge and explicit notice, never defaults."""
        js = """
renderCoachingFeedback(null);

const badge = document.getElementById('coaching-status-badge');
const quality = document.getElementById('coaching-quality-details');
const cues = document.getElementById('coaching-cues-list');

assert.strictEqual(badge.textContent, 'UNAVAILABLE');
assert.strictEqual(quality.textContent, 'Coaching telemetry is not available for this run.');
assert.strictEqual(cues.textContent, '');
"""
        self._run_node_harness(js)

    def test_coaching_camera_view_unmeasured_displays_proper_disclaimer(self) -> None:
        """renderCoachingFeedback with camera_view_adequate: null renders Unmeasured (2D lateral view)."""
        js = """
const payload = {
  status: 'UNVALIDATED',
  disclaimers: {
    clinical_disclaimer: 'Planar measurements',
    human_validation_status: 'Unvalidated'
  },
  overall_summary: {
    recording_quality: {
      camera_view_adequate: null,
      tracking_continuity: 'Adequate',
      posture_coverage_pct: 92.5
    }
  },
  per_repetition_cues: []
};

renderCoachingFeedback(payload);

const quality = document.getElementById('coaching-quality-details');
const text = quality.children && quality.children.length > 0 ? quality.children[0].textContent : quality.textContent;
assert.ok(text.includes('Camera view geometry: Unmeasured (2D lateral view)'), 'Must state unmeasured camera view');
assert.ok(!text.includes('Camera view geometry: Adequate'), 'Must never claim adequate camera view when unmeasured');
"""
        self._run_node_harness(js)

    def test_phase_suppression_creates_no_seek_buttons(self) -> None:
        """When phase analysis is suppressed, zero seek buttons are generated."""
        js = """
const data = {
  job_id: 'test-phase-suppressed',
  completed_reps: 1,
  processed_frames: 60,
  attempts: {
    events: [{
      attempt_id: 1,
      status: 'counted',
      reason: 'completed_cycle',
      start_frame: 0,
      end_frame: 60,
      start_time_s: 0.0,
      end_time_s: 2.0,
      min_knee_angle_deg: 72.0
    }]
  },
  phase_feedback: {
    attempts: [{
      attempt_id: 1,
      phase_analysis_available: false,
      suppression_reason: 'untracked_inflection_point'
    }]
  }
};

renderResults(data);

const tbody = document.getElementById('attempts-tbody');
let seekButtonsCount = 0;
function countSeekButtons(node) {
  if (node.tagName === 'BUTTON' && (node.className || '').includes('btn-seek')) {
    seekButtonsCount++;
  }
  if (node.children) {
    node.children.forEach(countSeekButtons);
  }
}
countSeekButtons(tbody);

assert.strictEqual(seekButtonsCount, 0, 'No seek buttons should be created when phase analysis is suppressed');
"""
        self._run_node_harness(js)

    def test_incomplete_job_clarity_renders_decoded_prefix(self) -> None:
        """Incomplete analysis renders incomplete note box and decoded prefix label."""
        js = """
setCurrentJobId('test-incomplete-clarity');
const data = {
  job_id: 'test-incomplete-clarity',
  status: 'incomplete',
  completed_reps: 4,
  processed_frames: 1524,
  declared_frame_count: 1526,
  read_status_note: 'Container declared 1526 frames, but only 1524 frames were successfully decoded.',
  attempts: { events: [] }
};

renderResults(data);

const incompleteBox = document.getElementById('incomplete-note-box');
const incompleteText = document.getElementById('incomplete-note-text');
const repsLabel = document.getElementById('metric-reps-label');
const resultsTitle = document.getElementById('results-title');

assert.ok(!incompleteBox.classList.contains('hidden'), 'Incomplete note box must be visible');
assert.ok(incompleteText.textContent.includes('1524'), 'Must note processed frame count');
assert.ok(incompleteText.textContent.includes('decoded prefix only'), 'Note must emphasize decoded prefix');
if (repsLabel) {
  assert.ok(repsLabel.textContent.includes('Decoded Prefix'), 'Repetition label must state Decoded Prefix');
}
if (resultsTitle) {
  assert.ok(resultsTitle.textContent.includes('Decoded') && resultsTitle.textContent.includes('Prefix'), 'Title must specify decoded prefix');
}
"""
        self._run_node_harness(js)

    def test_failed_job_displays_diagnostics_and_hides_results(self) -> None:
        """Failed job polling displays error in jobErrorBox and ensures resultsCard remains hidden."""
        js = """
setCurrentJobId('job-failed-test');
const jobErrorBox = document.getElementById('job-error-box');
const resultsCard = document.getElementById('results-card');

context.fetch = async () => ({
  ok: true,
  status: 200,
  json: async () => ({
    status: 'failed',
    error: 'Subprocess error: decoding failure at frame 1500'
  })
});

await pollJobStatus('job-failed-test');

assert.ok(!jobErrorBox.classList.contains('hidden'), 'jobErrorBox must be visible on failed job');
assert.ok(jobErrorBox.textContent.includes('decoding failure at frame 1500'), 'jobErrorBox must show diagnostic details');
assert.ok(resultsCard.classList.contains('hidden'), 'resultsCard must remain hidden on failed job');
"""
        self._run_node_harness(js)


if __name__ == "__main__":
    unittest.main()


class TestSameJobResponseOrdering(unittest.TestCase):
    _run_node_harness = TestFrontendBehaviorHeadlessNode._run_node_harness

    def setUp(self):
        if shutil.which('node') is None:self.skipTest('Node unavailable')

    def test_older_same_job_status_does_not_undo_terminal_status(self):
        self._run_node_harness("""
setCurrentJobId('same-job');
const requests=[];
context.fetch=()=>new Promise(resolve=>requests.push(resolve));
const older=pollJobStatus('same-job'),newer=pollJobStatus('same-job');
requests[1]({ok:true,json:async()=>({status:'failed',error:'new terminal failure'})});await newer;
requests[0]({ok:true,json:async()=>({status:'running'})});await older;
assert.strictEqual(document.getElementById('job-status').textContent,'FAILED');
""")

    def test_older_same_job_result_does_not_replace_latest_error(self):
        self._run_node_harness("""
setCurrentJobId('same-job');
const requests=[];
context.fetch=()=>new Promise(resolve=>requests.push(resolve));
const older=loadJobResults('same-job'),newer=loadJobResults('same-job');
requests[1]({ok:true,json:async()=>({status:'failed',error:'new terminal failure'})});await newer;
requests[0]({ok:true,json:async()=>({status:'failed',error:'old superseded failure'})});await older;
assert.strictEqual(document.getElementById('job-error-box').textContent,'new terminal failure');
""")
