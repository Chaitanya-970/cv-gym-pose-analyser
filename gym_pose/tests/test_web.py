"""Unit and API tests for gym_analyser web interface and job orchestration.

Uses standard-library unittest and tempfile. Does not require GPU, model downloads,
or active YOLO inference. All runner execution is mocked or injected.
Compatible with standard unittest discovery under Python 3.10.
"""

import io
import json
import os
import shutil
import tempfile
import threading
import time
import unittest
import uuid
from pathlib import Path
from typing import Any, Dict

from gym_analyser.web_jobs import (
    ALLOWED_ARTIFACTS,
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

# Optional dependencies for FastAPI TestClient
try:
    from fastapi.testclient import TestClient
    from gym_analyser.web import create_app
    HAS_WEB_DEPS = True
except (ImportError, TypeError):
    TestClient = None
    create_app = None
    HAS_WEB_DEPS = False


def make_dummy_summary(
    status: str = "completed",
    completed_reps: int = 3,
    read_status_note: str = "",
) -> Dict[str, Any]:
    """Helper to construct a valid summary.json mock payload."""
    return {
        "status": status,
        "model_name": "yolo11n-pose.pt",
        "read_status_note": read_status_note,
        "processed_frames": 150,
        "processed_duration_s": 5.0,
        "completed_reps": completed_reps,
        "attempt_observations": {
            "counted_attempts": completed_reps,
            "uncounted_attempts": 1,
            "pre_arming": {"unarmed_frames": 10, "opening_lowered_frames": 0},
            "interpretation": "Software-defined observed attempts, not posture judgements",
        },
        "posture_observations": {
            "torso_valid_frames": 140,
            "hip_valid_frames": 140,
            "processed_frames": 150,
            "interpretation": "Projected 2D measurements; unsigned torso angle from upward vertical",
        },
        "limitations": [
            "Projected 2D joint angles, not calibrated 3D anatomical angles.",
            "Repetition counts reflect threshold transitions, not validated clinical accuracy.",
            "Torso inclination is unsigned and depends on camera roll; posture observations do not establish safe or correct form.",
        ],
    }


def make_dummy_attempts(completed_reps: int = 3) -> Dict[str, Any]:
    """Helper to construct a valid attempts.json mock payload."""
    events = []
    for i in range(1, completed_reps + 1):
        events.append({
            "attempt_id": i,
            "status": "counted",
            "reason": "completed_cycle",
            "explanation": "A complete cycle met the configured counting thresholds.",
            "start_frame": (i - 1) * 40 + 5,
            "end_frame": i * 40,
            "start_time_s": float((i - 1) * 40 + 5) / 30.0,
            "end_time_s": float(i * 40) / 30.0,
            "min_knee_angle_deg": 95.5,
            "category": "completed",
        })
    # Add an uncounted attempt (e.g. setup)
    events.append({
        "attempt_id": completed_reps + 1,
        "status": "uncounted",
        "reason": "depth_threshold_not_reached",
        "explanation": "The observed knee angle did not reach the configured 110-degree depth threshold.",
        "start_frame": completed_reps * 40 + 5,
        "end_frame": completed_reps * 40 + 25,
        "start_time_s": float(completed_reps * 40 + 5) / 30.0,
        "end_time_s": float(completed_reps * 40 + 25) / 30.0,
        "min_knee_angle_deg": 125.0,
        "category": "movement_threshold",
    })
    return {
        "counter_completed_reps": completed_reps,
        "counted_attempts": completed_reps,
        "uncounted_attempts": 1,
        "events": events,
        "limitations": ["Observed movement events can include setup and weight return; squat intent is not classified."],
    }


class TestJobManager(unittest.TestCase):
    """Unit tests for JobManager storage, concurrency locking, and status handling."""

    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.storage_dir = Path(self.temp_dir.name) / "web_jobs"
        self.storage_dir.mkdir(parents=True, exist_ok=True)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_job_id_validation(self) -> None:
        """Validate UUID4 string format enforcement and traversal rejection."""
        valid_uuid = str(uuid.uuid4())
        self.assertTrue(is_valid_job_id(valid_uuid))
        self.assertTrue(is_valid_job_id(valid_uuid.upper()))

        self.assertFalse(is_valid_job_id("not-a-uuid"))
        self.assertFalse(is_valid_job_id("../../etc/passwd"))
        self.assertFalse(is_valid_job_id(""))
        self.assertFalse(is_valid_job_id(12345))  # type: ignore
        self.assertFalse(is_valid_job_id(None))  # type: ignore

    def test_upload_save_normal(self) -> None:
        """Validate normal upload chunking, directory creation, and metadata."""
        manager = JobManager(storage_dir=self.storage_dir)
        content = b"fake-video-bytes-stream-payload-12345"
        stream = io.BytesIO(content)

        job_id, saved_path, size = manager.save_upload_file(stream, "sample_clip.mp4")

        self.assertTrue(is_valid_job_id(job_id))
        self.assertTrue(saved_path.is_file())
        self.assertEqual(size, len(content))
        self.assertEqual(saved_path.suffix, ".mp4")

        meta = manager.get_metadata(job_id)
        self.assertIsNotNone(meta)
        self.assertEqual(meta["job_id"], job_id)
        self.assertEqual(meta["status"], STATUS_QUEUED)
        self.assertEqual(meta["original_filename"], "sample_clip.mp4")
        self.assertEqual(meta["file_size_bytes"], len(content))

    def test_upload_save_empty_rejected(self) -> None:
        """Validate that empty (0 bytes) upload is rejected and directory cleaned up."""
        manager = JobManager(storage_dir=self.storage_dir)
        stream = io.BytesIO(b"")

        with self.assertRaises(ValueError) as ctx:
            manager.save_upload_file(stream, "empty.mp4")
        self.assertIn("empty", str(ctx.exception).lower())

        # Check no dangling job directory remains
        subdirs = list(self.storage_dir.iterdir())
        self.assertEqual(len(subdirs), 0)

    def test_upload_save_oversize_rejected(self) -> None:
        """Validate that file exceeding max upload bytes is rejected and cleaned up."""
        manager = JobManager(storage_dir=self.storage_dir, max_upload_bytes=100)
        stream = io.BytesIO(b"X" * 200)

        with self.assertRaises(ValueError) as ctx:
            manager.save_upload_file(stream, "big.mp4")
        self.assertIn("exceeds", str(ctx.exception).lower())

        subdirs = list(self.storage_dir.iterdir())
        self.assertEqual(len(subdirs), 0)

    def test_upload_disallowed_extension(self) -> None:
        """Validate that non-video extensions are rejected immediately."""
        manager = JobManager(storage_dir=self.storage_dir)
        stream = io.BytesIO(b"some content")

        for ext in ("script.sh", "malicious.exe", "data.json", "payload.py"):
            with self.assertRaises(ValueError) as ctx:
                manager.save_upload_file(stream, ext)
            self.assertIn("unsupported video format", str(ctx.exception).lower())

    def test_busy_locking_and_release_on_success(self) -> None:
        """Validate single-worker busy locking and release on job completion."""
        completion_event = threading.Event()

        def fake_runner(job_id, input_path, results_dir, log_path, **kwargs):
            completion_event.wait(timeout=5.0)
            summary_path = results_dir / "summary.json"
            with open(summary_path, "w", encoding="utf-8") as f:
                json.dump(make_dummy_summary(status="completed", completed_reps=4), f)
            return {"status": STATUS_COMPLETED, "exit_code": 0}

        manager = JobManager(storage_dir=self.storage_dir, runner=fake_runner)
        job_id, _, _ = manager.save_upload_file(io.BytesIO(b"dummy video"), "test.mp4")

        self.assertFalse(manager.is_busy())
        manager.enqueue_job(job_id)

        # Worker should now be busy
        self.assertTrue(manager.is_busy())
        self.assertEqual(manager.get_active_job_id(), job_id)

        # Submitting second job while busy must raise JobBusyError
        job_id2, _, _ = manager.save_upload_file(io.BytesIO(b"dummy video 2"), "test2.mp4")
        with self.assertRaises(JobBusyError):
            manager.enqueue_job(job_id2)

        # Unblock worker and wait for finish
        completion_event.set()
        time.sleep(0.15)

        self.assertFalse(manager.is_busy())
        meta = manager.get_metadata(job_id)
        self.assertEqual(meta["status"], STATUS_COMPLETED)

    def test_busy_release_after_failure(self) -> None:
        """Validate that busy state is released when worker runner raises an exception."""
        def failing_runner(job_id, input_path, results_dir, log_path, **kwargs):
            raise RuntimeError("Fatal worker engine crash")

        manager = JobManager(storage_dir=self.storage_dir, runner=failing_runner)
        job_id, _, _ = manager.save_upload_file(io.BytesIO(b"dummy video"), "test.mp4")

        manager.enqueue_job(job_id)
        time.sleep(0.15)

        self.assertFalse(manager.is_busy())
        meta = manager.get_metadata(job_id)
        self.assertEqual(meta["status"], STATUS_FAILED)
        self.assertIn("Fatal worker engine crash", meta["error"])

    def test_runner_timeout(self) -> None:
        """Validate that worker timeout marks job as failed and releases busy state."""
        def timing_out_runner(**kwargs):
            raise TimeoutError("Analysis timed out after 10 seconds")

        manager = JobManager(storage_dir=self.storage_dir, runner=timing_out_runner)
        job_id, _, _ = manager.save_upload_file(io.BytesIO(b"dummy video"), "test.mp4")

        manager.enqueue_job(job_id)
        time.sleep(0.15)

        self.assertFalse(manager.is_busy())
        meta = manager.get_metadata(job_id)
        self.assertEqual(meta["status"], STATUS_FAILED)
        self.assertIn("timed out", meta["error"].lower())

    def test_incomplete_status_handling(self) -> None:
        """Validate that non-completed pipeline output sets status to incomplete honestly."""
        def incomplete_runner(job_id, input_path, results_dir, log_path, **kwargs):
            summary_path = results_dir / "summary.json"
            note = "Video ended early: 25 decoded vs 50 declared frames"
            with open(summary_path, "w", encoding="utf-8") as f:
                json.dump(make_dummy_summary(status="incomplete", completed_reps=1, read_status_note=note), f)
            return {"status": STATUS_INCOMPLETE, "read_status_note": note, "exit_code": 1}

        manager = JobManager(storage_dir=self.storage_dir, runner=incomplete_runner)
        job_id, _, _ = manager.save_upload_file(io.BytesIO(b"dummy video"), "test.mp4")

        manager.enqueue_job(job_id)
        time.sleep(0.15)

        meta = manager.get_metadata(job_id)
        self.assertEqual(meta["status"], STATUS_INCOMPLETE)
        self.assertIn("Video ended early", meta["read_status_note"])

    def test_server_restart_interruption_recovery(self) -> None:
        """Validate that dangling queued and running jobs are marked interrupted on restart."""
        # Create dummy interrupted jobs on disk directly
        queued_id = str(uuid.uuid4())
        running_id = str(uuid.uuid4())
        done_id = str(uuid.uuid4())

        for jid, st in ((queued_id, STATUS_QUEUED), (running_id, STATUS_RUNNING), (done_id, STATUS_COMPLETED)):
            jdir = self.storage_dir / jid
            jdir.mkdir(parents=True)
            with open(jdir / "job.json", "w", encoding="utf-8") as f:
                json.dump({"job_id": jid, "status": st, "created_at": "2026-10-19T00:00:00Z"}, f)

        # Re-initialize JobManager to trigger recover_interrupted_jobs
        manager = JobManager(storage_dir=self.storage_dir)

        meta_queued = manager.get_metadata(queued_id)
        meta_running = manager.get_metadata(running_id)
        meta_done = manager.get_metadata(done_id)

        self.assertEqual(meta_queued["status"], STATUS_INTERRUPTED)
        self.assertEqual(meta_running["status"], STATUS_INTERRUPTED)
        self.assertEqual(meta_done["status"], STATUS_COMPLETED)

    def test_artifact_allowlist_and_traversal_protection(self) -> None:
        """Validate artifact exact allowlist, file existence, and path traversal block."""
        manager = JobManager(storage_dir=self.storage_dir)
        job_id = str(uuid.uuid4())
        job_dir = self.storage_dir / job_id
        results_dir = job_dir / "results"
        results_dir.mkdir(parents=True)

        # Create allowed and disallowed files in results
        summary_file = results_dir / "summary.json"
        summary_file.write_text("{}", encoding="utf-8")

        secret_file = results_dir / "passwords.txt"
        secret_file.write_text("secret", encoding="utf-8")

        log_file = job_dir / "run.log"
        log_file.write_text("log output", encoding="utf-8")

        # Allowed artifact exists
        path = manager.get_artifact_path(job_id, "summary.json")
        self.assertEqual(path, summary_file.resolve())

        # Disallowed artifact in results directory
        with self.assertRaises(FileNotFoundError):
            manager.get_artifact_path(job_id, "passwords.txt")

        # Disallowed artifact outside results directory
        with self.assertRaises(FileNotFoundError):
            manager.get_artifact_path(job_id, "run.log")

        # Path traversal attempts
        with self.assertRaises(FileNotFoundError):
            manager.get_artifact_path(job_id, "../run.log")

        with self.assertRaises(FileNotFoundError):
            manager.get_artifact_path(job_id, "../../job.json")

    def test_get_job_results_serialization(self) -> None:
        """Validate results aggregation from summary, attempts, and posture files."""
        manager = JobManager(storage_dir=self.storage_dir)
        job_id, _, _ = manager.save_upload_file(io.BytesIO(b"video data"), "test.mp4")
        job_dir = self.storage_dir / job_id
        results_dir = job_dir / "results"

        # Populate realistic mock result files
        with open(results_dir / "summary.json", "w", encoding="utf-8") as f:
            json.dump(make_dummy_summary(completed_reps=5), f)
        with open(results_dir / "attempts.json", "w", encoding="utf-8") as f:
            json.dump(make_dummy_attempts(completed_reps=5), f)
        (results_dir / "angles.csv").write_text("frame_index,knee_angle_deg\n0,160.0\n", encoding="utf-8")
        (results_dir / "angle_plot.png").write_bytes(b"dummy png bytes")

        # Update metadata to completed
        meta = manager.get_metadata(job_id)
        meta["status"] = STATUS_COMPLETED
        manager._save_metadata(job_id, meta)

        results = manager.get_job_results(job_id)
        self.assertEqual(results["job_id"], job_id)
        self.assertEqual(results["status"], STATUS_COMPLETED)
        self.assertEqual(results["completed_reps"], 5)
        self.assertEqual(results["processed_frames"], 150)
        self.assertIsNotNone(results["attempts"])
        self.assertEqual(len(results["attempts"]["events"]), 6)  # 5 counted + 1 uncounted

        artifacts = results["available_artifacts"]
        self.assertTrue(artifacts["summary.json"]["available"])
        self.assertTrue(artifacts["angles.csv"]["available"])
        self.assertTrue(artifacts["angle_plot.png"]["available"])
        self.assertFalse(artifacts["annotated.mp4"]["available"])


@unittest.skipUnless(HAS_WEB_DEPS, "Web dependencies (fastapi, httpx) not installed")
class TestWebAPI(unittest.TestCase):
    """API integration tests using FastAPI TestClient with injected runners."""

    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.storage_dir = Path(self.temp_dir.name) / "web_jobs"
        self.storage_dir.mkdir(parents=True, exist_ok=True)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_get_index_html(self) -> None:
        """Validate that GET / returns the HTML application page."""
        manager = JobManager(storage_dir=self.storage_dir)
        app = create_app(manager)
        client = TestClient(app)

        res = client.get("/")
        self.assertEqual(res.status_code, 200)
        self.assertIn("text/html", res.headers.get("content-type", ""))
        self.assertIn("2D Gym Pose Analyser", res.text)
        self.assertIn("Recording Guidance", res.text)
        self.assertIn("Uncounted Movements", res.text)

    def test_get_server_status_idle(self) -> None:
        """Validate GET /api/status returns idle state when no job is running."""
        manager = JobManager(storage_dir=self.storage_dir)
        app = create_app(manager)
        client = TestClient(app)

        res = client.get("/api/status")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertFalse(data["busy"])
        self.assertIsNone(data["active_job_id"])

    def test_submit_job_and_poll_results(self) -> None:
        """Validate upload submission, status polling, and results retrieval."""
        def sync_runner(job_id, input_path, results_dir, log_path, **kwargs):
            with open(results_dir / "summary.json", "w", encoding="utf-8") as f:
                json.dump(make_dummy_summary(status="completed", completed_reps=2), f)
            with open(results_dir / "attempts.json", "w", encoding="utf-8") as f:
                json.dump(make_dummy_attempts(completed_reps=2), f)
            (results_dir / "annotated.mp4").write_bytes(b"dummy mp4 video bytes")
            return {"status": STATUS_COMPLETED, "exit_code": 0}

        manager = JobManager(storage_dir=self.storage_dir, runner=sync_runner)
        app = create_app(manager)
        client = TestClient(app)

        video_bytes = b"fake video content for testing"
        res = client.post(
            "/api/jobs",
            files={"file": ("squat_sample.mp4", video_bytes, "video/mp4")},
        )
        self.assertEqual(res.status_code, 202)
        submit_data = res.json()
        self.assertIn("job_id", submit_data)
        job_id = submit_data["job_id"]

        # Wait briefly for synchronous fake runner execution
        time.sleep(0.15)

        # Check status endpoint
        res_status = client.get(f"/api/jobs/{job_id}")
        self.assertEqual(res_status.status_code, 200)
        status_data = res_status.json()
        self.assertEqual(status_data["status"], STATUS_COMPLETED)

        # Check results endpoint
        res_results = client.get(f"/api/jobs/{job_id}/results")
        self.assertEqual(res_results.status_code, 200)
        res_data = res_results.json()
        self.assertEqual(res_data["completed_reps"], 2)
        self.assertTrue(res_data["available_artifacts"]["annotated.mp4"]["available"])

        # Check download endpoint for allowed artifact
        res_dl = client.get(f"/api/jobs/{job_id}/download/annotated.mp4")
        self.assertEqual(res_dl.status_code, 200)
        self.assertEqual(res_dl.content, b"dummy mp4 video bytes")
        self.assertIn("inline", res_dl.headers.get("content-disposition", ""))

    def test_submit_job_busy_rejection(self) -> None:
        """Validate that concurrent submission returns HTTP 429 when worker is busy."""
        pause_event = threading.Event()

        def slow_runner(**kwargs):
            pause_event.wait(timeout=5.0)
            return {"status": STATUS_COMPLETED, "exit_code": 0}

        manager = JobManager(storage_dir=self.storage_dir, runner=slow_runner)
        app = create_app(manager)
        client = TestClient(app)

        responses = []
        first = threading.Thread(target=lambda: responses.append(client.post(
            "/api/jobs", files={"file": ("first.mp4", b"dummy video 1", "video/mp4")})))
        first.start()
        try:
            deadline = time.monotonic() + 3
            while not manager.is_busy() and time.monotonic() < deadline:
                time.sleep(0.01)
            self.assertTrue(manager.is_busy())
            res2 = client.post("/api/jobs", files={"file": ("second.mp4", b"dummy video 2", "video/mp4")})
            self.assertEqual(res2.status_code, 429)
        finally:
            pause_event.set()
            first.join(timeout=5)
        self.assertFalse(first.is_alive())
        self.assertEqual(responses[0].status_code, 202)

    def test_submit_empty_file_rejected(self) -> None:
        """Validate that 0-byte upload returns HTTP 400."""
        manager = JobManager(storage_dir=self.storage_dir)
        app = create_app(manager)
        client = TestClient(app)

        res = client.post(
            "/api/jobs",
            files={"file": ("empty.mp4", b"", "video/mp4")},
        )
        self.assertEqual(res.status_code, 400)
        self.assertIn("empty", res.json()["detail"].lower())

    def test_submit_disallowed_file_extension(self) -> None:
        """Validate that non-video file formats return HTTP 400."""
        manager = JobManager(storage_dir=self.storage_dir)
        app = create_app(manager)
        client = TestClient(app)

        res = client.post(
            "/api/jobs",
            files={"file": ("payload.exe", b"binary content", "application/octet-stream")},
        )
        self.assertEqual(res.status_code, 400)
        self.assertIn("unsupported video format", res.json()["detail"].lower())

    def test_invalid_and_nonexistent_job_ids(self) -> None:
        """Validate HTTP 400 for malformed job IDs and HTTP 404 for missing IDs."""
        manager = JobManager(storage_dir=self.storage_dir)
        app = create_app(manager)
        client = TestClient(app)

        # Malformed ID
        res_bad = client.get("/api/jobs/not-a-valid-uuid")
        self.assertEqual(res_bad.status_code, 400)

        # Nonexistent valid UUID
        fake_uuid = str(uuid.uuid4())
        res_not_found = client.get(f"/api/jobs/{fake_uuid}")
        self.assertEqual(res_not_found.status_code, 404)

    def test_download_artifact_allowlist_and_traversal(self) -> None:
        """Validate that download endpoint rejects disallowed files and traversals."""
        manager = JobManager(storage_dir=self.storage_dir)
        job_id = str(uuid.uuid4())
        jdir = self.storage_dir / job_id / "results"
        jdir.mkdir(parents=True)
        (jdir / "summary.json").write_text("{}", encoding="utf-8")
        (jdir.parent / "run.log").write_text("private log", encoding="utf-8")

        # Save metadata for this job
        manager._save_metadata(job_id, {"job_id": job_id, "status": STATUS_COMPLETED})

        app = create_app(manager)
        client = TestClient(app)

        # Allowed artifact
        res_ok = client.get(f"/api/jobs/{job_id}/download/summary.json")
        self.assertEqual(res_ok.status_code, 200)

        # Disallowed artifact (private log)
        res_disallowed = client.get(f"/api/jobs/{job_id}/download/run.log")
        self.assertEqual(res_disallowed.status_code, 404)

        # Path traversal
        res_traversal = client.get(f"/api/jobs/{job_id}/download/..%2Frun.log")
        self.assertIn(res_traversal.status_code, (400, 404))



class TestWebReviewRegressions(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.manager = JobManager(storage_dir=Path(self.tmp.name)/'jobs')

    def test_busy_released_if_metadata_write_fails(self):
        from unittest.mock import patch
        jid, _, _ = self.manager.save_upload_file(io.BytesIO(b'video'), 'x.mp4')
        self.manager._active_job_id = jid
        with patch.object(self.manager, '_save_metadata', side_effect=OSError('disk locked')):
            with self.assertRaises(OSError):
                self.manager._execute_job(jid)
        self.assertFalse(self.manager.is_busy())

    def test_invalid_timeout_configuration(self):
        for timeout in (0, -1, float('nan'), float('inf')):
            with self.subTest(timeout=timeout), self.assertRaises(ValueError):
                JobManager(storage_dir=Path(self.tmp.name)/'bad', timeout_seconds=timeout)

    def test_runner_without_terminal_state_fails(self):
        self.manager.runner = lambda **kwargs: {'exit_code': 0}
        jid, _, _ = self.manager.save_upload_file(io.BytesIO(b'video'), 'x.mp4')
        self.manager._active_job_id = jid
        self.manager._execute_job(jid)
        self.assertEqual(self.manager.get_metadata(jid)['status'], STATUS_FAILED)
        self.assertFalse(self.manager.is_busy())

    def test_result_directory_redirect_rejected(self):
        from unittest.mock import patch
        jid, _, _ = self.manager.save_upload_file(io.BytesIO(b'video'), 'x.mp4')
        original = Path.resolve
        expected = self.manager._get_job_dir(jid)/'results'
        outside = Path(self.tmp.name)/'private'
        def redirect(path, *args, **kwargs):
            return outside if path == expected else original(path, *args, **kwargs)
        with patch.object(Path, 'resolve', redirect):
            with self.assertRaises(ValueError):
                self.manager.get_artifact_path(jid, 'summary.json')

class SchedulingFailureTests(unittest.TestCase):
    def test_scheduling_failure_releases_worker(self):
        with tempfile.TemporaryDirectory() as temp:
            manager = JobManager(Path(temp)/'jobs')
            jid,_,_ = manager.save_upload_file(io.BytesIO(b'video'),'x.mp4')
            class RejectSchedule:
                def add_task(self,*args): raise OSError('scheduler unavailable')
            with self.assertRaises(OSError): manager.enqueue_job(jid,RejectSchedule())
            self.assertFalse(manager.is_busy())

class MetadataRetryTests(unittest.TestCase):
    def test_transient_windows_file_lock_is_retried(self):
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as temp:
            manager=JobManager(Path(temp)/'jobs')
            jid,_,_=manager.save_upload_file(io.BytesIO(b'video'),'x.mp4')
            original=Path.replace
            attempts=[]
            def flaky(path,target):
                attempts.append(path)
                if len(attempts)==1: raise PermissionError('OneDrive transient file lock')
                return original(path,target)
            with patch.object(Path,'replace',flaky),patch('gym_analyser.web_jobs.time.sleep'):
                manager._save_metadata(jid,{'job_id':jid,'status':'running'})
            self.assertEqual(len(attempts),2)
            self.assertEqual(manager.get_metadata(jid)['status'],'running')

if __name__ == "__main__":
    unittest.main()
