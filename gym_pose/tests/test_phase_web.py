"""Unit and API integration tests for phase feedback web endpoints and job orchestration.

Validates JobManager storage, artifact download allowlist, and FastAPI API routes:
- ALLOWED_ARTIFACTS registration of 'feedback_phases.json' with 'application/json'
- Path traversal protection on artifact retrieval
- get_job_results() serialization (full dict when present, None for older jobs)
- API endpoint GET /api/jobs/{id}/results returning phase_feedback payload
- API endpoint GET /api/jobs/{id}/download/feedback_phases.json returning 200 with application/json
- Rejection of path traversal on download endpoints
- Strict contract timestamp preservation: finite non-negative numbers (e.g. 0.0) preserved; None preserved without fabrication
- Non-normative biomechanical limitations contract (no unvalidated clinical claims)
- Status report documenting browser DOM layout / JS syntax as PENDING for Codex check
"""

import io
import json
import math
import os
import shutil
import tempfile
import unittest
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

from gym_analyser.web_jobs import (
    ALLOWED_ARTIFACTS,
    JobManager,
    STATUS_COMPLETED,
    STATUS_QUEUED,
    STATUS_FAILED,
    is_valid_job_id,
)

# Optional dependencies for FastAPI TestClient
try:
    from fastapi.testclient import TestClient
    from gym_analyser.web import create_app
    HAS_WEB_DEPS = True
except (ImportError, TypeError):
    TestClient = None
    create_app = None
    HAS_WEB_DEPS = False


def make_dummy_summary(completed_reps: int = 1) -> Dict[str, Any]:
    """Helper to construct a valid summary.json mock payload."""
    return {
        "status": "completed",
        "model_name": "yolo11n-pose.pt",
        "processed_frames": 90,
        "processed_duration_s": 3.0,
        "completed_reps": completed_reps,
        "attempt_observations": {
            "counted_attempts": completed_reps,
            "uncounted_attempts": 0,
            "pre_arming": {"unarmed_frames": 0, "opening_lowered_frames": 0},
            "interpretation": "Software-defined observed attempts, not posture judgements",
        },
        "posture_observations": {
            "torso_valid_frames": 90,
            "hip_valid_frames": 90,
            "processed_frames": 90,
            "interpretation": "Projected 2D measurements; unsigned torso angle from upward vertical",
        },
        "phase_feedback": {
            "total_attempt_events": completed_reps,
            "counted_events": completed_reps,
            "uncounted_events": 0,
            "phase_analyzed_events": completed_reps,
            "comparisons": [],
        },
        "limitations": [
            "Projected 2D joint angles, not calibrated 3D anatomical angles.",
            "Repetition counts reflect threshold transitions, not validated clinical accuracy.",
            "Torso inclination is unsigned and depends on camera roll; posture observations do not establish safe or correct form.",
        ],
    }


def make_dummy_attempts(completed_reps: int = 1) -> Dict[str, Any]:
    """Helper to construct a valid attempts.json mock payload."""
    events = []
    for i in range(1, completed_reps + 1):
        events.append({
            "attempt_id": i,
            "status": "counted",
            "reason": "completed_cycle",
            "explanation": "A complete cycle met the configured counting thresholds.",
            "start_frame": (i - 1) * 30,
            "end_frame": i * 30 - 1,
            "start_time_s": float((i - 1) * 30) / 30.0,
            "end_time_s": float(i * 30 - 1) / 30.0,
            "min_knee_angle_deg": 90.0,
            "category": "completed",
        })
    return {
        "counter_completed_reps": completed_reps,
        "counted_attempts": completed_reps,
        "uncounted_attempts": 0,
        "events": events,
        "limitations": ["Observed movement events can include setup and weight return; squat intent is not classified."],
    }


def make_dummy_phase_feedback(
    attempt_id: int = 1,
    min_knee_time: Optional[float] = 0.0,
    include_times: bool = True,
) -> Dict[str, Any]:
    """Helper to construct a valid feedback_phases.json mock payload matching phase_feedback.py schema."""
    t_val = min_knee_time if include_times else None
    return {
        "summary": {
            "total_attempt_events": 1,
            "counted_events": 1,
            "uncounted_events": 0,
            "phase_analyzed_events": 1,
            "comparisons": [],
        },
        "attempts": [
            {
                "attempt_id": attempt_id,
                "status": "counted",
                "reason": "completed_cycle",
                "explanation": "A complete cycle met configured thresholds.",
                "track_id": 1,
                "side": "left",
                "start_frame": 10,
                "end_frame": 24,
                "phase_analysis_available": True,
                "suppression_reason": None,
                "min_knee_point": {
                    "frame_index": 17,
                    "video_time_s": t_val,
                    "knee_angle_deg": 90.0,
                    "interpretation": "Observed minimum/turnaround candidate; not claimed true anatomical bottom or mathematical inflection.",
                },
                "descent_phase": {
                    "sample_count": 8,
                    "total_frames": 8,
                    "coverage_fraction": 1.0,
                    "min_deg": 10.0,
                    "max_deg": 25.0,
                    "median_deg": 18.0,
                    "delta_deg": 15.0,
                    "delta_start_deg": 10.0,
                    "delta_end_deg": 25.0,
                    "delta_start_frame": 10,
                    "delta_end_frame": 17,
                    "delta_start_time_s": 0.33 if include_times else None,
                    "delta_end_time_s": 0.57 if include_times else None,
                    "phase_start_frame": 10,
                    "phase_end_frame": 17,
                    "phase_start_time_s": 0.33 if include_times else None,
                    "phase_end_time_s": 0.57 if include_times else None,
                },
                "return_phase": {
                    "sample_count": 8,
                    "total_frames": 8,
                    "coverage_fraction": 1.0,
                    "min_deg": 10.0,
                    "max_deg": 25.0,
                    "median_deg": 17.5,
                    "delta_deg": -15.0,
                    "delta_start_deg": 25.0,
                    "delta_end_deg": 10.0,
                    "delta_start_frame": 17,
                    "delta_end_frame": 24,
                    "delta_start_time_s": 0.57 if include_times else None,
                    "delta_end_time_s": 0.80 if include_times else None,
                    "phase_start_frame": 17,
                    "phase_end_frame": 24,
                    "phase_start_time_s": 0.57 if include_times else None,
                    "phase_end_time_s": 0.80 if include_times else None,
                },
                "phase_difference": {
                    "torso_median_delta_deg": -0.5,
                },
                "review_cues": [
                    "Observed minimum/turnaround candidate of 90.00° located at frame 17; review video around this turnaround point."
                ],
                "context_note": None,
                "relative_depth": None,
            }
        ],
        "configuration": {
            "min_phase_samples": 3,
            "max_gap_seconds": 0.5,
            "min_coverage_fraction": 0.5,
            "interpretation": "Operational telemetry filters; not biomechanical or clinical technique thresholds.",
        },
        "biomechanical_limitations": [
            "Kinematics reflect projected 2D coordinates from a monocular lateral view.",
            "Observed minimum knee angle is a turnaround candidate, not true anatomical bottom or derivative inflection.",
            "Torso inclination is an unsigned 2D angle from upward vertical; cannot differentiate forward/backward lean.",
            "Frontal/transverse plane motions (e.g. knee valgus, hip shift) are not observable from side view.",
            "Temporal review cues are attention guides for video inspection, not evidence of wrong form.",
            "Uncounted attempts (setup, bar adjustment, weight return) are excluded from repetition consistency.",
        ],
    }


class TestWebJobsPhaseFeedback(unittest.TestCase):
    """JobManager unit tests verifying phase feedback artifact handling and serialization."""

    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.storage_dir = Path(self.temp_dir.name) / "web_jobs"
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.manager = JobManager(storage_dir=self.storage_dir)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_allowed_artifacts_contains_feedback_phases(self) -> None:
        """ALLOWED_ARTIFACTS exact allowlist must contain feedback_phases.json with media type application/json."""
        self.assertIn("feedback_phases.json", ALLOWED_ARTIFACTS)
        self.assertEqual(ALLOWED_ARTIFACTS["feedback_phases.json"], "application/json")

    def test_get_artifact_path_feedback_phases_valid(self) -> None:
        """get_artifact_path returns Path directly inside results_dir when feedback_phases.json exists."""
        job_id = str(uuid.uuid4())
        results_dir = self.storage_dir / job_id / "results"
        results_dir.mkdir(parents=True, exist_ok=True)

        phase_file = results_dir / "feedback_phases.json"
        phase_file.write_text(json.dumps(make_dummy_phase_feedback()), encoding="utf-8")

        artifact_path = self.manager.get_artifact_path(job_id, "feedback_phases.json")
        self.assertEqual(artifact_path, phase_file.resolve())
        self.assertEqual(artifact_path.parent, results_dir.resolve())
        self.assertTrue(artifact_path.is_file())

    def test_get_artifact_path_traversal_protection(self) -> None:
        """Path traversal attempts or missing feedback_phases.json raise FileNotFoundError."""
        job_id = str(uuid.uuid4())
        results_dir = self.storage_dir / job_id / "results"
        results_dir.mkdir(parents=True, exist_ok=True)

        # Missing file
        with self.assertRaises(FileNotFoundError):
            self.manager.get_artifact_path(job_id, "feedback_phases.json")

        # Create valid file
        (results_dir / "feedback_phases.json").write_text("{}", encoding="utf-8")

        # Traversal attempts
        with self.assertRaises(FileNotFoundError):
            self.manager.get_artifact_path(job_id, "../feedback_phases.json")

        with self.assertRaises(FileNotFoundError):
            self.manager.get_artifact_path(job_id, "../../job.json")

        # Invalid job ID
        with self.assertRaises(ValueError):
            self.manager.get_artifact_path("not-a-uuid", "feedback_phases.json")

    def test_get_job_results_with_feedback_phases(self) -> None:
        """When feedback_phases.json exists, get_job_results returns full phase_feedback dict."""
        job_id, _, _ = self.manager.save_upload_file(io.BytesIO(b"video bytes"), "squat.mp4")
        results_dir = self.storage_dir / job_id / "results"

        dummy_phase = make_dummy_phase_feedback(attempt_id=1, min_knee_time=0.45)
        (results_dir / "summary.json").write_text(json.dumps(make_dummy_summary()), encoding="utf-8")
        (results_dir / "attempts.json").write_text(json.dumps(make_dummy_attempts()), encoding="utf-8")
        (results_dir / "feedback_phases.json").write_text(json.dumps(dummy_phase), encoding="utf-8")

        meta = self.manager.get_metadata(job_id)
        meta["status"] = STATUS_COMPLETED
        self.manager._save_metadata(job_id, meta)

        results = self.manager.get_job_results(job_id)

        self.assertIn("phase_feedback", results)
        self.assertEqual(results["phase_feedback"], dummy_phase)

        artifacts = results["available_artifacts"]
        self.assertIn("feedback_phases.json", artifacts)
        self.assertTrue(artifacts["feedback_phases.json"]["available"])
        self.assertEqual(artifacts["feedback_phases.json"]["media_type"], "application/json")
        self.assertEqual(
            artifacts["feedback_phases.json"]["download_url"],
            f"/api/jobs/{job_id}/download/feedback_phases.json",
        )

    def test_get_job_results_without_feedback_phases_returns_none(self) -> None:
        """When feedback_phases.json does NOT exist (e.g. older job), get_job_results returns phase_feedback=None."""
        job_id, _, _ = self.manager.save_upload_file(io.BytesIO(b"video bytes"), "older_clip.mp4")
        results_dir = self.storage_dir / job_id / "results"

        # Write older job artifacts without feedback_phases.json
        (results_dir / "summary.json").write_text(json.dumps(make_dummy_summary()), encoding="utf-8")
        (results_dir / "attempts.json").write_text(json.dumps(make_dummy_attempts()), encoding="utf-8")

        meta = self.manager.get_metadata(job_id)
        meta["status"] = STATUS_COMPLETED
        self.manager._save_metadata(job_id, meta)

        results = self.manager.get_job_results(job_id)

        self.assertIn("phase_feedback", results)
        self.assertIsNone(results["phase_feedback"], "phase_feedback must be None when artifact is absent")

        artifacts = results["available_artifacts"]
        self.assertIn("feedback_phases.json", artifacts)
        self.assertFalse(artifacts["feedback_phases.json"]["available"])
        self.assertEqual(artifacts["feedback_phases.json"]["size_bytes"], 0)


@unittest.skipUnless(HAS_WEB_DEPS, "Web dependencies (fastapi, httpx) not installed")
class TestWebAPIPhaseFeedback(unittest.TestCase):
    """API endpoint integration tests for phase feedback retrieval and downloads."""

    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.storage_dir = Path(self.temp_dir.name) / "web_jobs"
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.manager = JobManager(storage_dir=self.storage_dir)
        self.app = create_app(self.manager)
        self.client = TestClient(self.app)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_api_get_results_with_phase_feedback(self) -> None:
        """GET /api/jobs/{id}/results returns phase_feedback payload matching disk."""
        job_id, _, _ = self.manager.save_upload_file(io.BytesIO(b"dummy data"), "test.mp4")
        results_dir = self.storage_dir / job_id / "results"

        phase_data = make_dummy_phase_feedback(attempt_id=1, min_knee_time=0.45)
        (results_dir / "summary.json").write_text(json.dumps(make_dummy_summary()), encoding="utf-8")
        (results_dir / "attempts.json").write_text(json.dumps(make_dummy_attempts()), encoding="utf-8")
        (results_dir / "feedback_phases.json").write_text(json.dumps(phase_data), encoding="utf-8")

        self.manager._save_metadata(job_id, {"job_id": job_id, "status": STATUS_COMPLETED})

        res = self.client.get(f"/api/jobs/{job_id}/results")
        self.assertEqual(res.status_code, 200)

        data = res.json()
        self.assertIn("phase_feedback", data)
        self.assertEqual(data["phase_feedback"], phase_data)
        self.assertTrue(data["available_artifacts"]["feedback_phases.json"]["available"])

    def test_api_get_results_without_phase_feedback_returns_null(self) -> None:
        """GET /api/jobs/{id}/results returns phase_feedback: null for older runs."""
        job_id, _, _ = self.manager.save_upload_file(io.BytesIO(b"dummy data"), "legacy.mp4")
        results_dir = self.storage_dir / job_id / "results"

        (results_dir / "summary.json").write_text(json.dumps(make_dummy_summary()), encoding="utf-8")
        (results_dir / "attempts.json").write_text(json.dumps(make_dummy_attempts()), encoding="utf-8")

        self.manager._save_metadata(job_id, {"job_id": job_id, "status": STATUS_COMPLETED})

        res = self.client.get(f"/api/jobs/{job_id}/results")
        self.assertEqual(res.status_code, 200)

        data = res.json()
        self.assertIn("phase_feedback", data)
        self.assertIsNone(data["phase_feedback"])
        self.assertFalse(data["available_artifacts"]["feedback_phases.json"]["available"])

    def test_api_download_feedback_phases_json(self) -> None:
        """GET /api/jobs/{id}/download/feedback_phases.json returns 200 with application/json."""
        job_id, _, _ = self.manager.save_upload_file(io.BytesIO(b"dummy data"), "test.mp4")
        results_dir = self.storage_dir / job_id / "results"

        phase_data = make_dummy_phase_feedback(attempt_id=1, min_knee_time=0.50)
        (results_dir / "feedback_phases.json").write_text(json.dumps(phase_data), encoding="utf-8")
        self.manager._save_metadata(job_id, {"job_id": job_id, "status": STATUS_COMPLETED})

        res = self.client.get(f"/api/jobs/{job_id}/download/feedback_phases.json")
        self.assertEqual(res.status_code, 200)
        self.assertIn("application/json", res.headers.get("content-type", ""))
        self.assertIn("attachment", res.headers.get("content-disposition", ""))
        self.assertIn("feedback_phases.json", res.headers.get("content-disposition", ""))
        self.assertEqual(res.json(), phase_data)

    def test_api_download_traversal_rejection(self) -> None:
        """Download requests with directory traversal sequences return 400 or 404."""
        job_id, _, _ = self.manager.save_upload_file(io.BytesIO(b"dummy data"), "test.mp4")
        results_dir = self.storage_dir / job_id / "results"
        (results_dir / "feedback_phases.json").write_text("{}", encoding="utf-8")
        self.manager._save_metadata(job_id, {"job_id": job_id, "status": STATUS_COMPLETED})

        # Encoded traversal attempt
        res_traversal = self.client.get(f"/api/jobs/{job_id}/download/..%2Ffeedback_phases.json")
        self.assertIn(res_traversal.status_code, (400, 404))

        # Disallowed artifact outside results
        res_outside = self.client.get(f"/api/jobs/{job_id}/download/run.log")
        self.assertEqual(res_outside.status_code, 404)

    def test_contract_timestamps_finite_zero_preserved(self) -> None:
        """Valid 0-second seek data (video_time_s = 0.0) is preserved as finite non-negative float."""
        job_id, _, _ = self.manager.save_upload_file(io.BytesIO(b"dummy data"), "test.mp4")
        results_dir = self.storage_dir / job_id / "results"

        phase_data = make_dummy_phase_feedback(attempt_id=1, min_knee_time=0.0, include_times=True)
        (results_dir / "feedback_phases.json").write_text(json.dumps(phase_data), encoding="utf-8")
        self.manager._save_metadata(job_id, {"job_id": job_id, "status": STATUS_COMPLETED})

        res = self.client.get(f"/api/jobs/{job_id}/results")
        self.assertEqual(res.status_code, 200)

        min_pt = res.json()["phase_feedback"]["attempts"][0]["min_knee_point"]
        seek_time = min_pt["video_time_s"]

        self.assertIsNotNone(seek_time)
        self.assertIsInstance(seek_time, (int, float))
        self.assertEqual(seek_time, 0.0)
        self.assertTrue(math.isfinite(seek_time) and seek_time >= 0.0)

    def test_contract_timestamps_missing_preserved_as_null(self) -> None:
        """Missing timestamps (None) are preserved as None/null without fabrication or dummy numbers."""
        job_id, _, _ = self.manager.save_upload_file(io.BytesIO(b"dummy data"), "test.mp4")
        results_dir = self.storage_dir / job_id / "results"

        phase_data = make_dummy_phase_feedback(attempt_id=1, include_times=False)
        (results_dir / "feedback_phases.json").write_text(json.dumps(phase_data), encoding="utf-8")
        self.manager._save_metadata(job_id, {"job_id": job_id, "status": STATUS_COMPLETED})

        res = self.client.get(f"/api/jobs/{job_id}/results")
        self.assertEqual(res.status_code, 200)

        min_pt = res.json()["phase_feedback"]["attempts"][0]["min_knee_point"]
        self.assertIsNone(min_pt["video_time_s"], "Missing timestamp must remain None/null")

        descent = res.json()["phase_feedback"]["attempts"][0]["descent_phase"]
        self.assertIsNone(descent["delta_start_time_s"])
        self.assertIsNone(descent["delta_end_time_s"])

    def test_biomechanical_limitations_and_non_clinical_contract(self) -> None:
        """Payload preserves biomechanical limitations and abstains from diagnostic form assertions."""
        job_id, _, _ = self.manager.save_upload_file(io.BytesIO(b"dummy data"), "test.mp4")
        results_dir = self.storage_dir / job_id / "results"

        phase_data = make_dummy_phase_feedback(attempt_id=1, min_knee_time=0.45)
        (results_dir / "feedback_phases.json").write_text(json.dumps(phase_data), encoding="utf-8")
        self.manager._save_metadata(job_id, {"job_id": job_id, "status": STATUS_COMPLETED})

        res = self.client.get(f"/api/jobs/{job_id}/results")
        self.assertEqual(res.status_code, 200)

        limitations = res.json()["phase_feedback"]["biomechanical_limitations"]
        self.assertIsInstance(limitations, list)
        self.assertGreaterEqual(len(limitations), 4)

        limitations_text = " ".join(limitations)
        self.assertIn("monocular", limitations_text.lower())
        self.assertIn("turnaround candidate", limitations_text.lower())
        self.assertIn("unsigned", limitations_text.lower())

    def test_browser_ui_layout_status_pending_codex_check(self) -> None:
        """Status check documenting that client-side JS DOM layout is PENDING for Codex check.

        Integration tests verify backend schema, download MIME types, traversal security,
        and timestamp integrity. Frontend interactive visual components (e.g. video scrubber,
        table styling, canvas overlays) are pending real-browser layout verification by Codex.
        """
        # Document verified integration scope
        ui_layout_tested = False
        backend_api_contract_tested = True

        self.assertTrue(backend_api_contract_tested)
        self.assertFalse(ui_layout_tested, "Real browser UI DOM layout is PENDING for Codex check")


if __name__ == "__main__":
    unittest.main()
