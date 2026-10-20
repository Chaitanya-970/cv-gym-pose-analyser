"""Integration tests for coaching feedback across pipeline, web worker, and local web UI."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from gym_analyser.counter import CounterConfig
from gym_analyser.web_jobs import (
    ALLOWED_ARTIFACTS,
    STATUS_COMPLETED,
    JobManager,
)


def create_minimal_test_video(path: Path, num_frames: int = 5, fps: float = 10.0) -> None:
    """Create minimal MP4 video file for testing pipeline execution."""
    import cv2
    import numpy as np

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (64, 64))
    for _ in range(num_frames):
        frame = np.zeros((64, 64, 3), dtype=np.uint8)
        writer.write(frame)
    writer.release()


def create_mock_pose_model() -> MagicMock:
    """Create a mock YOLO pose model conforming to the Ultralytics API."""
    import numpy as np
    import torch

    mock_model = MagicMock()
    mock_model.task = "pose"
    mock_res = MagicMock()
    mock_res.boxes = MagicMock()
    mock_res.boxes.id = torch.tensor([1])
    kpt_array = np.zeros((1, 17, 3), dtype=np.float32)
    # Standing left leg keypoints
    kpt_array[0, 11] = [32.0, 16.0, 0.95]
    kpt_array[0, 13] = [32.0, 32.0, 0.95]
    kpt_array[0, 15] = [32.0, 48.0, 0.95]
    # Torso keypoint (shoulder)
    kpt_array[0, 5] = [32.0, 8.0, 0.95]
    mock_res.keypoints = MagicMock()
    mock_res.keypoints.data = torch.from_numpy(kpt_array)
    mock_res.plot.return_value = np.zeros((64, 64, 3), dtype=np.uint8)
    mock_model.track.return_value = [mock_res]
    return mock_model


class TestPipelineCoachingIntegration(unittest.TestCase):
    """Verify pipeline writes coaching_feedback.json alongside other artifacts."""

    def setUp(self):
        try:
            import cv2
            import numpy as np
            import torch
            from gym_analyser.pipeline import run_pipeline
            self.run_pipeline = run_pipeline
        except ImportError:
            self.skipTest("cv2, numpy, torch, or pipeline dependencies not installed in this environment")

    def test_pipeline_generates_coaching_feedback_artifact(self):
        config = CounterConfig()
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            video_path = tmp / "input.mp4"
            out_dir = tmp / "output"
            create_minimal_test_video(video_path, num_frames=5, fps=10.0)
            mock_model = create_mock_pose_model()

            summary = self.run_pipeline(video_path, out_dir, config, model_name=mock_model)

            # 1. Check all standard output artifacts exist
            for expected_file in (
                "annotated.mp4",
                "angles.csv",
                "summary.json",
                "angle_plot.png",
                "attempts.json",
                "posture_summary.json",
                "feedback_observations.json",
                "feedback_phases.json",
                "coaching_feedback.json",
            ):
                artifact_path = out_dir / expected_file
                self.assertTrue(artifact_path.is_file(), f"Expected artifact {expected_file} was not created")
                self.assertGreater(artifact_path.stat().st_size, 0, f"Artifact {expected_file} is empty")

            # 2. Check summary output_paths includes coaching_feedback_json
            self.assertIn("coaching_feedback_json", summary["output_paths"])
            self.assertEqual(
                summary["output_paths"]["coaching_feedback_json"],
                str(out_dir / "coaching_feedback.json"),
            )

            # 3. Inspect coaching_feedback.json contents against schema contract
            coaching_path = out_dir / "coaching_feedback.json"
            with open(coaching_path, "r", encoding="utf-8") as f:
                coaching_data = json.load(f)

            self.assertIn("contract_version", coaching_data)
            self.assertIn("status", coaching_data)
            self.assertIn("overall_summary", coaching_data)
            self.assertIn("per_repetition_cues", coaching_data)
            self.assertIn("uncounted_movement_explanations", coaching_data)
            self.assertIn("disclaimers", coaching_data)

            # Check recording quality fields: 2D tracking does not establish 3D camera geometry
            rq = coaching_data["overall_summary"]["recording_quality"]
            self.assertIn("camera_view_adequate", rq)
            self.assertIsNone(rq["camera_view_adequate"])
            self.assertIn(rq["tracking_continuity"], ("adequate", "interrupted", "insufficient"))

            disclaimers = coaching_data["disclaimers"]
            self.assertIn("clinical_disclaimer", disclaimers)
            self.assertIn("human_validation_status", disclaimers)
            self.assertIn("UNVALIDATED", disclaimers["human_validation_status"])
            self.assertIn("descriptive planar observations", disclaimers["clinical_disclaimer"])

    def test_pipeline_overwrite_replaces_stale_coaching_feedback(self):
        config = CounterConfig()
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            video_path = tmp / "input.mp4"
            out_dir = tmp / "output"
            out_dir.mkdir(parents=True, exist_ok=True)
            create_minimal_test_video(video_path, num_frames=5, fps=10.0)

            # Write a stale coaching feedback file
            stale_file = out_dir / "coaching_feedback.json"
            stale_file.write_text(json.dumps({"stale": True}), encoding="utf-8")

            mock_model = create_mock_pose_model()
            self.run_pipeline(video_path, out_dir, config, model_name=mock_model, overwrite=True)

            with open(stale_file, "r", encoding="utf-8") as f:
                refreshed = json.load(f)

            self.assertNotIn("stale", refreshed)
            self.assertIn("contract_version", refreshed)


class TestWebJobsCoachingIntegration(unittest.TestCase):
    """Verify web_jobs.py handles coaching_feedback.json in allowlist and get_job_results()."""

    def test_allowed_artifacts_contains_coaching_feedback(self):
        self.assertIn("coaching_feedback.json", ALLOWED_ARTIFACTS)
        self.assertEqual(ALLOWED_ARTIFACTS["coaching_feedback.json"], "application/json")

    def test_get_job_results_includes_coaching_feedback_when_present(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            storage = Path(tmp_dir)
            manager = JobManager(storage_dir=storage)
            job_id = "11111111-2222-4333-8444-555555555555"
            job_dir = storage / job_id
            results_dir = job_dir / "results"
            results_dir.mkdir(parents=True, exist_ok=True)

            meta = {
                "job_id": job_id,
                "status": STATUS_COMPLETED,
                "created_at": "2026-10-20T10:00:00Z",
            }
            (job_dir / "job.json").write_text(json.dumps(meta), encoding="utf-8")

            sample_coaching = {
                "contract_version": "1.0",
                "status": "ready",
                "overall_summary": {
                    "total_counted_reps": 3,
                    "total_uncounted_attempts": 1,
                    "recording_quality": {
                        "camera_view_adequate": None,
                        "tracking_continuity": "adequate",
                        "posture_coverage_pct": 95.0,
                        "notes": ["2D lateral view observed; 3D camera adequacy unmeasured"],
                    },
                },
                "per_repetition_cues": [],
                "uncounted_movement_explanations": [],
                "disclaimers": {
                    "clinical_disclaimer": "These 2D kinematic measurements are descriptive planar observations, not clinical diagnostic assessments or technique grades.",
                    "human_validation_status": "UNVALIDATED by human expert review on diverse unseen subjects.",
                },
            }
            (results_dir / "coaching_feedback.json").write_text(json.dumps(sample_coaching), encoding="utf-8")

            results = manager.get_job_results(job_id)
            self.assertIn("coaching_feedback", results)
            self.assertEqual(results["coaching_feedback"], sample_coaching)
            self.assertIn("coaching_feedback.json", results["available_artifacts"])
            self.assertTrue(results["available_artifacts"]["coaching_feedback.json"]["available"])

    def test_get_job_results_missing_coaching_feedback_returns_none(self):
        """Older jobs without coaching_feedback.json must return coaching_feedback=None cleanly."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            storage = Path(tmp_dir)
            manager = JobManager(storage_dir=storage)
            job_id = "22222222-3333-4444-8555-666666666666"
            job_dir = storage / job_id
            results_dir = job_dir / "results"
            results_dir.mkdir(parents=True, exist_ok=True)

            meta = {
                "job_id": job_id,
                "status": STATUS_COMPLETED,
                "created_at": "2026-10-19T10:00:00Z",
            }
            (job_dir / "job.json").write_text(json.dumps(meta), encoding="utf-8")

            results = manager.get_job_results(job_id)
            self.assertIn("coaching_feedback", results)
            self.assertIsNone(results["coaching_feedback"])
            self.assertIn("coaching_feedback.json", results["available_artifacts"])
            self.assertFalse(results["available_artifacts"]["coaching_feedback.json"]["available"])


class TestWebEndpointCoachingIntegration(unittest.TestCase):
    """Verify web.py endpoints serve coaching_feedback.json and include it in UI responses."""

    def setUp(self):
        try:
            from fastapi.testclient import TestClient
            from gym_analyser.web import create_app
            self.TestClient = TestClient
            self.create_app = create_app
        except ImportError:
            self.skipTest("FastAPI or dependencies not available in current test environment")

    def test_download_coaching_feedback_endpoint_and_results(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            storage = Path(tmp_dir)
            manager = JobManager(storage_dir=storage)
            job_id = "33333333-4444-4555-8666-777777777777"
            job_dir = storage / job_id
            results_dir = job_dir / "results"
            results_dir.mkdir(parents=True, exist_ok=True)

            meta = {
                "job_id": job_id,
                "status": STATUS_COMPLETED,
                "created_at": "2026-10-20T10:00:00Z",
            }
            (job_dir / "job.json").write_text(json.dumps(meta), encoding="utf-8")

            sample_coaching = {
                "contract_version": "1.0",
                "status": "ready",
                "overall_summary": {
                    "total_counted_reps": 2,
                    "total_uncounted_attempts": 0,
                    "recording_quality": {
                        "camera_view_adequate": None,
                        "tracking_continuity": "adequate",
                        "posture_coverage_pct": None,
                        "notes": [],
                    },
                },
                "per_repetition_cues": [],
                "uncounted_movement_explanations": [],
                "disclaimers": {
                    "clinical_disclaimer": "These 2D kinematic measurements are descriptive planar observations, not clinical diagnostic assessments or technique grades.",
                    "human_validation_status": "UNVALIDATED by human expert review on diverse unseen subjects.",
                },
            }
            (results_dir / "coaching_feedback.json").write_text(json.dumps(sample_coaching), encoding="utf-8")

            app = self.create_app(job_manager=manager, enable_review=False)
            client = self.TestClient(app)

            # Test download endpoint
            resp_dl = client.get(f"/api/jobs/{job_id}/download/coaching_feedback.json")
            self.assertEqual(resp_dl.status_code, 200)
            self.assertIn("application/json", resp_dl.headers.get("content-type", ""))
            self.assertEqual(resp_dl.json(), sample_coaching)

            # Test results endpoint
            resp_res = client.get(f"/api/jobs/{job_id}/results")
            self.assertEqual(resp_res.status_code, 200)
            res_json = resp_res.json()
            self.assertIn("coaching_feedback", res_json)
            self.assertEqual(res_json["coaching_feedback"], sample_coaching)

    def test_download_missing_coaching_feedback_returns_404(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            storage = Path(tmp_dir)
            manager = JobManager(storage_dir=storage)
            job_id = "44444444-5555-4666-8777-888888888888"
            job_dir = storage / job_id
            results_dir = job_dir / "results"
            results_dir.mkdir(parents=True, exist_ok=True)

            meta = {
                "job_id": job_id,
                "status": STATUS_COMPLETED,
                "created_at": "2026-10-20T10:00:00Z",
            }
            (job_dir / "job.json").write_text(json.dumps(meta), encoding="utf-8")

            app = self.create_app(job_manager=manager, enable_review=False)
            client = self.TestClient(app)

            # Download endpoint must 404
            resp_dl = client.get(f"/api/jobs/{job_id}/download/coaching_feedback.json")
            self.assertEqual(resp_dl.status_code, 404)

            # Results endpoint must return coaching_feedback=None
            resp_res = client.get(f"/api/jobs/{job_id}/results")
            self.assertEqual(resp_res.status_code, 200)
            self.assertIsNone(resp_res.json().get("coaching_feedback"))

    def test_index_html_contains_coaching_feedback_section(self):
        try:
            from gym_analyser.web import INDEX_HTML
        except ImportError:
            self.skipTest("web module dependencies not available")

        # HTML elements check
        self.assertIn('id="coaching-feedback-card"', INDEX_HTML)
        self.assertIn('id="coaching-status-badge"', INDEX_HTML)
        self.assertIn('id="coaching-clinical-disclaimer"', INDEX_HTML)
        self.assertIn('id="coaching-validation-status"', INDEX_HTML)
        self.assertIn("UNVALIDATED", INDEX_HTML)
        self.assertIn("descriptive planar observations", INDEX_HTML)
        self.assertIn("displayResults", INDEX_HTML)
        self.assertIn("renderCoachingFeedback", INDEX_HTML)


if __name__ == "__main__":
    unittest.main()
