"""Final Web Integration and Product Acceptance Test Suite for 2D Gym Pose Analyser.

Task 63 Acceptance Suite:
1. Upload validation:
   - Valid formats (.mp4, .mov, .avi, .mkv, .webm, .m4v) accepted and queued.
   - Disallowed formats (.exe, .txt, .py, .tar.gz, etc.) fail gracefully with HTTP 400.
   - Missing, empty, or whitespace-only filenames rejected with HTTP 400/422.
   - Empty (0-byte) payloads fail gracefully with HTTP 400.
   - Oversized files exceeding limits fail gracefully with HTTP 400.
   - Corrupted video inputs fail gracefully in worker with status: 'failed' without server crashes.
2. Job isolation & path traversal security:
   - Dedicated UUID4 storage per job; files never leak across jobs.
   - Traversal attempts in job ID or artifact download path return HTTP 400 or 404.
   - Single-worker concurrency busy state returns HTTP 429.
   - Server restart recovery marks dangling jobs as interrupted.
3. Results & Artifact agreement:
   - Harmonization of completed_reps across summary, attempts, coaching_feedback.
   - Agreement on rep boundaries (timestamps and frame boundaries).
   - Accurate decoded-prefix reporting for incomplete runs.
   - All allowed artifacts available and downloadable.
4. Real saved personal-video artifacts:
   - Exercises local HTTP endpoints using actual saved artifacts from
     misc/user_video_review/a2857b52f2cd/execution_01/baseline/3fefbab9-65c6-4b3e-8822-09e31b009806/.
   - Validates playable H.264 preview.mp4 headers, media type, and inline disposition.
   - Validates HTTP Range requests (206 Partial Content, Accept-Ranges, Content-Range) for streaming.
5. Headless UI acceptance:
   - Bounded phase seeking clamped within actual video duration.
   - Stale job switching cleans up media elements and discards out-of-date responses.
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
import time
import unittest
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

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

# Real saved baseline artifacts directory
_BASELINE_ARTIFACTS_DIR = (
    _GYM_POSE_DIR.parent
    / "misc"
    / "user_video_review"
    / "a2857b52f2cd"
    / "execution_01"
    / "baseline"
    / "3fefbab9-65c6-4b3e-8822-09e31b009806"
)


# ---------------------------------------------------------------------------
# Node.js Headless DOM Harness
# ---------------------------------------------------------------------------

class _NodeAcceptanceHarness(unittest.TestCase):
    """Base class for running headless JavaScript acceptance tests under Node.js."""

    def setUp(self) -> None:
        self.node_available = shutil.which("node") is not None
        if not self.node_available:
            self.skipTest("Node.js is not available on this machine")

    def run_node_script(self, js_body: str) -> None:
        """Run JavaScript in a virtual DOM environment under Node.js."""
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
      duration: 10.0,
      currentTime: 0.0,
      readyState: 4,
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
      removeAttribute(k) {{ delete this[k]; }},
      addEventListener(evt, fn) {{
        this['on_' + evt] = fn;
        if (!this._listeners) this._listeners = {{}};
        if (!this._listeners[evt]) this._listeners[evt] = [];
        this._listeners[evt].push(fn);
      }},
      removeEventListener(evt, fn) {{
        if (this._listeners && this._listeners[evt]) {{
          this._listeners[evt] = this._listeners[evt].filter(f => f !== fn);
        }}
      }},
      dispatchEvent(evt) {{
        const type = typeof evt === 'string' ? evt : evt.type;
        const listeners = (this._listeners && this._listeners[type]) || [];
        listeners.forEach(fn => fn(evt));
      }},
      click() {{
        if (this.on_click) this.on_click({{ target: this, type: 'click', preventDefault() {{}}, stopPropagation() {{}} }});
      }},
      appendChild(c) {{
        this.children.push(c);
      }},
      pause() {{ this._paused = true; }},
      load() {{ this._loaded = true; }},
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

const document = domDoc;

(async () => {{
  const {{
    formatFileSize,
    getFileExtension,
    validateClientFile,
    isValidTimestamp,
    clearPendingSeek,
    seekVideoToTime,
    announceSeekStatus,
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
    getActiveVideoDuration,
    setActiveVideoDuration,
  }} = context.module.exports;

{js_body}
  if (typeof stopPolling === 'function') stopPolling();
  process.exit(0);
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


# ---------------------------------------------------------------------------
# Test Suite 1: Upload Validation & Graceful Error Handling
# ---------------------------------------------------------------------------

@unittest.skipUnless(HAS_WEB_DEPS, "Web dependencies not installed")
class TestWebUploadAndValidation(unittest.TestCase):
    """Test video upload validation, supported formats, limits, and corrupted input handling."""

    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.storage_dir = Path(self.temp_dir.name) / "web_jobs"
        self.storage_dir.mkdir(parents=True, exist_ok=True)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_valid_video_formats_accepted_and_queued(self) -> None:
        """All supported extensions (.mp4, .mov, .avi, .mkv, .webm, .m4v) are accepted with HTTP 202."""
        for ext in ALLOWED_EXTENSIONS:
            manager = JobManager(storage_dir=self.storage_dir, runner=lambda **kw: {"status": STATUS_COMPLETED})
            app = create_app(job_manager=manager, enable_review=False)
            client = TestClient(app)

            filename = f"valid_sample{ext}"
            res = client.post("/api/jobs", files={"file": (filename, io.BytesIO(b"dummy video data"), "video/mp4")})
            self.assertEqual(res.status_code, 202, f"Format {ext} should return 202 Accepted")
            data = res.json()
            self.assertTrue(is_valid_job_id(data["job_id"]))
            self.assertEqual(data["status"], "queued")
            self.assertEqual(data["filename"], filename)

    def test_disallowed_extensions_rejected_with_400(self) -> None:
        """Unsupported extensions (.exe, .txt, .py, .tar.gz, .sh) are rejected with HTTP 400 Bad Request."""
        disallowed = ["malicious.exe", "notes.txt", "exploit.py", "archive.tar.gz", "script.sh"]
        manager = JobManager(storage_dir=self.storage_dir)
        app = create_app(job_manager=manager, enable_review=False)
        client = TestClient(app)

        for fname in disallowed:
            res = client.post("/api/jobs", files={"file": (fname, io.BytesIO(b"payload"), "application/octet-stream")})
            self.assertEqual(res.status_code, 400, f"Disallowed filename '{fname}' must return 400")
            detail = res.json().get("detail", "").lower()
            self.assertTrue("unsupported" in detail or "allowed" in detail or "valid" in detail)

    def test_missing_and_empty_filename_rejected_with_400_or_422(self) -> None:
        """Missing, empty string, or extensionless filename is rejected with HTTP 400 or 422."""
        invalid_names = ["", "   ", "barefilename_without_ext"]
        manager = JobManager(storage_dir=self.storage_dir)
        app = create_app(job_manager=manager, enable_review=False)
        client = TestClient(app)

        for name in invalid_names:
            res = client.post("/api/jobs", files={"file": (name, io.BytesIO(b"payload"), "video/mp4")})
            self.assertIn(res.status_code, (400, 422), f"Invalid filename '{name}' must return 400 or 422")

    def test_empty_payload_rejected_with_400(self) -> None:
        """Uploading an empty (0-byte) video file returns HTTP 400 Bad Request."""
        manager = JobManager(storage_dir=self.storage_dir)
        app = create_app(job_manager=manager, enable_review=False)
        client = TestClient(app)

        res = client.post("/api/jobs", files={"file": ("empty.mp4", io.BytesIO(b""), "video/mp4")})
        self.assertEqual(res.status_code, 400)
        self.assertIn("empty", res.json().get("detail", "").lower())
        # Verify no job directories leaked
        self.assertEqual(len(list(self.storage_dir.iterdir())), 0)

    def test_oversized_file_rejected_and_cleaned_up(self) -> None:
        """Files exceeding configured max_upload_bytes fail with HTTP 400 and clean up disk state."""
        manager = JobManager(storage_dir=self.storage_dir, max_upload_bytes=500 * 1024)
        app = create_app(job_manager=manager, enable_review=False)
        client = TestClient(app)

        large_payload = b"X" * (600 * 1024)
        res = client.post("/api/jobs", files={"file": ("large.mp4", io.BytesIO(large_payload), "video/mp4")})
        self.assertEqual(res.status_code, 400)
        self.assertIn("exceeds", res.json().get("detail", "").lower())
        self.assertEqual(len(list(self.storage_dir.iterdir())), 0)

    def test_corrupt_video_fails_gracefully_with_diagnostics(self) -> None:
        """Corrupted video input triggers worker failure gracefully, releases busy state, and leaves server healthy."""
        def failing_runner(job_id: str, input_path: Path, results_dir: Path, log_path: Path, **kwargs: Any) -> Dict[str, Any]:
            return {
                "status": STATUS_FAILED,
                "exit_code": 1,
                "error": "OpenCV error: corrupted bitstream at frame 0, unable to read packet headers.",
            }

        manager = JobManager(storage_dir=self.storage_dir, runner=failing_runner)
        app = create_app(job_manager=manager, enable_review=False)
        client = TestClient(app)

        res_upload = client.post("/api/jobs", files={"file": ("corrupt.mp4", io.BytesIO(b"corrupted-bytes"), "video/mp4")})
        self.assertEqual(res_upload.status_code, 202)
        job_id = res_upload.json()["job_id"]

        time.sleep(0.15)

        # GET /api/jobs/{id} returns failed status with diagnostics
        res_status = client.get(f"/api/jobs/{job_id}")
        self.assertEqual(res_status.status_code, 200)
        meta = res_status.json()
        self.assertEqual(meta["status"], STATUS_FAILED)
        self.assertIn("corrupted", meta["error"].lower())

        # Verify busy lock is released
        res_server = client.get("/api/status")
        self.assertEqual(res_server.status_code, 200)
        self.assertFalse(res_server.json()["busy"])
        self.assertIsNone(res_server.json()["active_job_id"])

        # Server immediately accepts subsequent jobs
        res_next = client.post("/api/jobs", files={"file": ("next.mp4", io.BytesIO(b"next bytes"), "video/mp4")})
        self.assertEqual(res_next.status_code, 202)


# ---------------------------------------------------------------------------
# Test Suite 2: Job Isolation & Security
# ---------------------------------------------------------------------------

@unittest.skipUnless(HAS_WEB_DEPS, "Web dependencies not installed")
class TestWebJobIsolationAndSecurity(unittest.TestCase):
    """Test dedicated storage per job, strict path traversal blocking, and restart recovery."""

    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.storage_dir = Path(self.temp_dir.name) / "web_jobs"
        self.storage_dir.mkdir(parents=True, exist_ok=True)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_job_directories_strictly_isolated(self) -> None:
        """Each job creates an isolated UUID directory under web_jobs without leaking files."""
        job_id_1 = str(uuid.uuid4())
        job_id_2 = str(uuid.uuid4())

        dir_1 = self.storage_dir / job_id_1 / "results"
        dir_2 = self.storage_dir / job_id_2 / "results"
        dir_1.mkdir(parents=True, exist_ok=True)
        dir_2.mkdir(parents=True, exist_ok=True)

        (dir_1 / "summary.json").write_text(json.dumps({"job": 1, "completed_reps": 5}), encoding="utf-8")
        (dir_2 / "summary.json").write_text(json.dumps({"job": 2, "completed_reps": 8}), encoding="utf-8")
        (self.storage_dir / job_id_1 / "job.json").write_text(json.dumps({"job_id": job_id_1, "status": STATUS_COMPLETED}), encoding="utf-8")
        (self.storage_dir / job_id_2 / "job.json").write_text(json.dumps({"job_id": job_id_2, "status": STATUS_COMPLETED}), encoding="utf-8")

        manager = JobManager(storage_dir=self.storage_dir)
        app = create_app(job_manager=manager, enable_review=False)
        client = TestClient(app)

        res1 = client.get(f"/api/jobs/{job_id_1}/results")
        res2 = client.get(f"/api/jobs/{job_id_2}/results")

        self.assertEqual(res1.status_code, 200)
        self.assertEqual(res2.status_code, 200)
        self.assertEqual(res1.json()["completed_reps"], 5)
        self.assertEqual(res2.json()["completed_reps"], 8)

    def test_path_traversal_in_job_id_rejected_with_400_or_404(self) -> None:
        """Non-canonical or traversal job IDs are rejected securely with HTTP 400 or 404."""
        manager = JobManager(storage_dir=self.storage_dir)
        app = create_app(job_manager=manager, enable_review=False)
        client = TestClient(app)

        traversals = [
            "../secret",
            "..%2F..%2Fetc",
            "not-a-valid-uuid",
            "3fefbab9-65c6-4b3e-8822-09e31b009806/extra",
        ]
        for bad_id in traversals:
            res_stat = client.get(f"/api/jobs/{bad_id}")
            self.assertIn(res_stat.status_code, (400, 404))
            res_res = client.get(f"/api/jobs/{bad_id}/results")
            self.assertIn(res_res.status_code, (400, 404))
            res_dl = client.get(f"/api/jobs/{bad_id}/download/summary.json")
            self.assertIn(res_dl.status_code, (400, 404))

    def test_path_traversal_in_artifact_name_rejected_with_404(self) -> None:
        """Artifact download rejects traversal patterns (..%, slashes) and non-allowlisted files with 404."""
        job_id = str(uuid.uuid4())
        results_dir = self.storage_dir / job_id / "results"
        results_dir.mkdir(parents=True, exist_ok=True)
        (results_dir / "summary.json").write_text("{}", encoding="utf-8")
        (self.storage_dir / job_id / "job.json").write_text(json.dumps({"job_id": job_id, "status": STATUS_COMPLETED}), encoding="utf-8")
        # Write private run.log in job root
        (self.storage_dir / job_id / "run.log").write_text("private execution log", encoding="utf-8")

        manager = JobManager(storage_dir=self.storage_dir)
        app = create_app(job_manager=manager, enable_review=False)
        client = TestClient(app)

        # 1. Traversal attempt to read run.log
        res_traversal = client.get(f"/api/jobs/{job_id}/download/..%2Frun.log")
        self.assertEqual(res_traversal.status_code, 404)

        # 2. Direct attempt to read disallowed file run.log
        res_disallowed = client.get(f"/api/jobs/{job_id}/download/run.log")
        self.assertEqual(res_disallowed.status_code, 404)

        # 3. Arbitrary file outside
        res_outside = client.get(f"/api/jobs/{job_id}/download/passwords.txt")
        self.assertEqual(res_outside.status_code, 404)

    def test_nonexistent_job_returns_404(self) -> None:
        """Valid UUID that does not exist returns HTTP 404 on status, results, and downloads."""
        random_id = str(uuid.uuid4())
        manager = JobManager(storage_dir=self.storage_dir)
        app = create_app(job_manager=manager, enable_review=False)
        client = TestClient(app)

        self.assertEqual(client.get(f"/api/jobs/{random_id}").status_code, 404)
        self.assertEqual(client.get(f"/api/jobs/{random_id}/results").status_code, 404)
        self.assertEqual(client.get(f"/api/jobs/{random_id}/download/summary.json").status_code, 404)

    def test_single_worker_concurrency_busy_429(self) -> None:
        """Simultaneous upload while worker is active returns HTTP 429 Too Many Requests."""
        manager = JobManager(storage_dir=self.storage_dir)
        app = create_app(job_manager=manager, enable_review=False)
        client = TestClient(app)

        # Simulate active job
        with manager._lock:
            manager._active_job_id = "test-active-job"

        res = client.post("/api/jobs", files={"file": ("video.mp4", io.BytesIO(b"payload"), "video/mp4")})
        self.assertEqual(res.status_code, 429)
        self.assertIn("busy", res.json().get("detail", "").lower())

        # Release
        with manager._lock:
            manager._active_job_id = None

    def test_server_restart_recovers_interrupted_jobs(self) -> None:
        """Jobs left in queued or running states during server shutdown are marked interrupted on restart."""
        job_id = str(uuid.uuid4())
        job_dir = self.storage_dir / job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        (job_dir / "job.json").write_text(json.dumps({
            "job_id": job_id,
            "status": STATUS_RUNNING,
            "created_at": "2026-10-21T10:00:00Z",
            "started_at": "2026-10-21T10:00:05Z",
        }), encoding="utf-8")

        # Initializing JobManager simulates server restart
        manager = JobManager(storage_dir=self.storage_dir)
        meta = manager.get_metadata(job_id)
        self.assertIsNotNone(meta)
        self.assertEqual(meta["status"], STATUS_INTERRUPTED)
        self.assertIn("interrupted", meta["error"].lower())


# ---------------------------------------------------------------------------
# Test Suite 3: Results & Artifact Agreement
# ---------------------------------------------------------------------------

@unittest.skipUnless(HAS_WEB_DEPS, "Web dependencies not installed")
class TestWebResultsAndArtifactAgreement(unittest.TestCase):
    """Verify agreement across summary, attempts, posture, coaching, and download endpoints."""

    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.storage_dir = Path(self.temp_dir.name) / "web_jobs"
        self.storage_dir.mkdir(parents=True, exist_ok=True)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_results_rep_counts_and_boundaries_agreement(self) -> None:
        """Ensure coaching_feedback, attempts breakdown, metrics, and summary agree on counts and rep boundaries."""
        job_id = str(uuid.uuid4())
        job_dir = self.storage_dir / job_id
        results_dir = job_dir / "results"
        results_dir.mkdir(parents=True, exist_ok=True)

        # 3 counted reps, 1 uncounted attempt
        summary_payload = {
            "status": "completed",
            "completed_reps": 3,
            "processed_frames": 300,
            "processed_duration_s": 10.0,
            "attempt_observations": {
                "counted_attempts": 3,
                "uncounted_attempts": 1,
            },
        }
        attempts_payload = {
            "counter_completed_reps": 3,
            "counted_attempts": 3,
            "uncounted_attempts": 1,
            "events": [
                {
                    "attempt_id": 1,
                    "status": "counted",
                    "reason": "completed_cycle",
                    "start_frame": 30,
                    "end_frame": 90,
                    "start_time_s": 1.0,
                    "end_time_s": 3.0,
                    "min_knee_angle_deg": 75.0,
                },
                {
                    "attempt_id": 2,
                    "status": "counted",
                    "reason": "completed_cycle",
                    "start_frame": 120,
                    "end_frame": 180,
                    "start_time_s": 4.0,
                    "end_time_s": 6.0,
                    "min_knee_angle_deg": 72.5,
                },
                {
                    "attempt_id": 3,
                    "status": "counted",
                    "reason": "completed_cycle",
                    "start_frame": 200,
                    "end_frame": 260,
                    "start_time_s": 6.67,
                    "end_time_s": 8.67,
                    "min_knee_angle_deg": 70.0,
                },
                {
                    "attempt_id": 4,
                    "status": "uncounted",
                    "reason": "shallow_depth",
                    "start_frame": 270,
                    "end_frame": 295,
                    "start_time_s": 9.0,
                    "end_time_s": 9.83,
                    "min_knee_angle_deg": 115.0,
                },
            ],
        }
        coaching_payload = {
            "status": "ready",
            "overall_summary": {
                "total_counted_reps": 3,
                "total_uncounted_attempts": 1,
                "recording_quality": {
                    "camera_view_adequate": True,
                    "tracking_continuity": "adequate",
                    "posture_coverage_pct": 99.0,
                },
            },
            "per_repetition_cues": [
                {
                    "rep_index": 1,
                    "attempt_id": 1,
                    "is_counted": True,
                    "timestamps": {"start_time_s": 1.0, "inflection_time_s": 2.0, "end_time_s": 3.0},
                    "measurements": {"min_knee_angle_deg": 75.0},
                },
                {
                    "rep_index": 2,
                    "attempt_id": 2,
                    "is_counted": True,
                    "timestamps": {"start_time_s": 4.0, "inflection_time_s": 5.0, "end_time_s": 6.0},
                    "measurements": {"min_knee_angle_deg": 72.5},
                },
                {
                    "rep_index": 3,
                    "attempt_id": 3,
                    "is_counted": True,
                    "timestamps": {"start_time_s": 6.67, "inflection_time_s": 7.67, "end_time_s": 8.67},
                    "measurements": {"min_knee_angle_deg": 70.0},
                },
            ],
            "uncounted_movement_explanations": [
                {
                    "attempt_id": 4,
                    "reason_code": "shallow_depth",
                    "explanation": "Attempt did not reach threshold depth.",
                }
            ],
        }

        (results_dir / "summary.json").write_text(json.dumps(summary_payload), encoding="utf-8")
        (results_dir / "attempts.json").write_text(json.dumps(attempts_payload), encoding="utf-8")
        (results_dir / "coaching_feedback.json").write_text(json.dumps(coaching_payload), encoding="utf-8")
        (job_dir / "job.json").write_text(json.dumps({"job_id": job_id, "status": STATUS_COMPLETED}), encoding="utf-8")

        manager = JobManager(storage_dir=self.storage_dir)
        app = create_app(job_manager=manager, enable_review=False)
        client = TestClient(app)

        res = client.get(f"/api/jobs/{job_id}/results")
        self.assertEqual(res.status_code, 200)
        data = res.json()

        # Check count agreement
        self.assertEqual(data["completed_reps"], 3)
        self.assertEqual(data["summary"]["completed_reps"], 3)
        self.assertEqual(data["coaching_feedback"]["overall_summary"]["total_counted_reps"], 3)

        # Check rep boundaries agreement between attempts and coaching cues
        cues = data["coaching_feedback"]["per_repetition_cues"]
        events = [e for e in data["attempts"]["events"] if e["status"] == "counted"]
        self.assertEqual(len(cues), len(events))
        for cue, event in zip(cues, events):
            self.assertEqual(cue["attempt_id"], event["attempt_id"])
            self.assertAlmostEqual(cue["timestamps"]["start_time_s"], event["start_time_s"], places=1)
            self.assertAlmostEqual(cue["timestamps"]["end_time_s"], event["end_time_s"], places=1)
            self.assertAlmostEqual(cue["measurements"]["min_knee_angle_deg"], event["min_knee_angle_deg"], places=1)

    def test_incomplete_analysis_status_and_notes(self) -> None:
        """Status incomplete correctly distinguishes decoded-prefix repetitions and surfaces read_status_note."""
        job_id = str(uuid.uuid4())
        job_dir = self.storage_dir / job_id
        results_dir = job_dir / "results"
        results_dir.mkdir(parents=True, exist_ok=True)

        note = "Container declared 300 frames, but only 120 frames were successfully decoded before EOF."
        summary_payload = {
            "status": "incomplete",
            "completed_reps": 1,
            "processed_frames": 120,
            "declared_frame_count": 300,
            "read_status_note": note,
        }
        (results_dir / "summary.json").write_text(json.dumps(summary_payload), encoding="utf-8")
        (job_dir / "job.json").write_text(json.dumps({
            "job_id": job_id,
            "status": STATUS_INCOMPLETE,
            "read_status_note": note,
        }), encoding="utf-8")

        manager = JobManager(storage_dir=self.storage_dir)
        app = create_app(job_manager=manager, enable_review=False)
        client = TestClient(app)

        res = client.get(f"/api/jobs/{job_id}/results")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["status"], STATUS_INCOMPLETE)
        self.assertEqual(data["completed_reps"], 1)
        self.assertEqual(data["processed_frames"], 120)
        self.assertEqual(data["read_status_note"], note)

    def test_all_allowed_artifacts_availability(self) -> None:
        """All allowed artifacts are listed in available_artifacts and individually downloadable."""
        job_id = str(uuid.uuid4())
        job_dir = self.storage_dir / job_id
        results_dir = job_dir / "results"
        results_dir.mkdir(parents=True, exist_ok=True)

        for name in ALLOWED_ARTIFACTS:
            if name.endswith(".json"):
                (results_dir / name).write_text("{}", encoding="utf-8")
            elif name.endswith(".csv"):
                (results_dir / name).write_text("frame,angle\n1,120\n", encoding="utf-8")
            elif name.endswith(".png"):
                (results_dir / name).write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 32)
            elif name.endswith(".mp4"):
                (results_dir / name).write_bytes(b"\x00\x00\x00\x1cftypisom" + b"\x00" * 64)

        (job_dir / "job.json").write_text(json.dumps({"job_id": job_id, "status": STATUS_COMPLETED}), encoding="utf-8")

        manager = JobManager(storage_dir=self.storage_dir)
        app = create_app(job_manager=manager, enable_review=False)
        client = TestClient(app)

        res = client.get(f"/api/jobs/{job_id}/results")
        self.assertEqual(res.status_code, 200)
        artifacts = res.json()["available_artifacts"]
        self.assertEqual(set(artifacts), set(ALLOWED_ARTIFACTS))

        for name, expected_media in ALLOWED_ARTIFACTS.items():
            self.assertIn(name, artifacts)
            self.assertTrue(artifacts[name]["available"])
            self.assertGreater(artifacts[name]["size_bytes"], 0)

            # Test download endpoint
            dl_res = client.get(f"/api/jobs/{job_id}/download/{name}")
            self.assertEqual(dl_res.status_code, 200)
            self.assertEqual(dl_res.headers.get("content-type"), expected_media)


# ---------------------------------------------------------------------------
# Test Suite 4: Real Saved Personal-Video Artifact Verification
# ---------------------------------------------------------------------------

@unittest.skipUnless(HAS_WEB_DEPS, "Web dependencies not installed")
class TestRealSavedPersonalVideoArtifacts(unittest.TestCase):
    """Exercise HTTP endpoints with actual saved personal-video artifacts from Task 61/62 runs."""

    def setUp(self) -> None:
        if not _BASELINE_ARTIFACTS_DIR.is_dir():
            self.skipTest(f"Baseline artifacts directory not found: {_BASELINE_ARTIFACTS_DIR}")

        self.temp_dir = tempfile.TemporaryDirectory()
        self.storage_dir = Path(self.temp_dir.name) / "web_jobs"
        self.job_id = "3fefbab9-65c6-4b3e-8822-09e31b009806"
        self.job_dir = self.storage_dir / self.job_id
        self.results_dir = self.job_dir / "results"
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # Copy real artifacts to test results folder
        for file_path in _BASELINE_ARTIFACTS_DIR.iterdir():
            if file_path.is_file():
                shutil.copy(file_path, self.results_dir / file_path.name)

        # Create job metadata
        meta = {
            "job_id": self.job_id,
            "status": STATUS_COMPLETED,
            "created_at": "2026-10-20T17:40:00Z",
            "started_at": "2026-10-20T17:40:05Z",
            "finished_at": "2026-10-20T17:43:00Z",
            "original_filename": "video.mp4",
            "file_size_bytes": 13036088,
        }
        (self.job_dir / "job.json").write_text(json.dumps(meta), encoding="utf-8")

        self.manager = JobManager(storage_dir=self.storage_dir)
        self.app = create_app(job_manager=self.manager, enable_review=False)
        self.client = TestClient(self.app)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_real_saved_video_results_schema_and_counts(self) -> None:
        """GET /api/jobs/{id}/results with real baseline artifacts returns 9 counted reps and 717 frames."""
        res = self.client.get(f"/api/jobs/{self.job_id}/results")
        self.assertEqual(res.status_code, 200)
        data = res.json()

        self.assertEqual(data["status"], "completed")
        self.assertEqual(data["completed_reps"], 9)
        self.assertEqual(data["processed_frames"], 717)
        self.assertEqual(len(data["attempts"]["events"]), 9)
        self.assertEqual(len(data["coaching_feedback"]["per_repetition_cues"]), 9)
        self.assertEqual(data["coaching_feedback"]["overall_summary"]["total_counted_reps"], 9)

        # Historical base artifacts exist; optional heel review was not run.
        self.assertEqual(set(data["available_artifacts"]), set(ALLOWED_ARTIFACTS))
        self.assertFalse(data["available_artifacts"]["heel_review.json"]["available"])
        self.assertFalse(data["available_artifacts"]["knee_review.json"]["available"])
        self.assertFalse(data["available_artifacts"]["back_review.json"]["available"])
        for name, art_info in data["available_artifacts"].items():
            if name in {"heel_review.json", "knee_review.json", "back_review.json"}:
                self.assertEqual(art_info["size_bytes"], 0)
                continue
            self.assertTrue(art_info["available"], f"Artifact {name} must be available on disk")
            self.assertGreater(art_info["size_bytes"], 0)

    def test_real_saved_preview_video_h264_playback_and_headers(self) -> None:
        """GET preview.mp4 returns 200, Content-Type video/mp4, inline disposition, Accept-Ranges, and H.264 stream."""
        res = self.client.get(f"/api/jobs/{self.job_id}/download/preview.mp4")
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.headers.get("content-type"), "video/mp4")
        self.assertIn("inline", res.headers.get("content-disposition", "").lower())
        self.assertEqual(res.headers.get("accept-ranges"), "bytes")
        self.assertEqual(int(res.headers.get("content-length", 0)), 3908758)

        # Inspect H.264 codec using ffprobe if available
        if shutil.which("ffprobe"):
            probe_cmd = [
                "ffprobe", "-v", "error",
                "-show_entries", "stream=codec_name",
                "-of", "default=noprint_wrappers=1",
                str(self.results_dir / "preview.mp4"),
            ]
            probe_out = subprocess.check_output(probe_cmd, text=True)
            self.assertIn("codec_name=h264", probe_out)

    def test_real_saved_video_range_request_streaming(self) -> None:
        """Preview video supports HTTP Range requests returning HTTP 206 Partial Content for smooth browser seeking."""
        # Request first 1KB
        res_range = self.client.get(
            f"/api/jobs/{self.job_id}/download/preview.mp4",
            headers={"Range": "bytes=0-1023"},
        )
        self.assertEqual(res_range.status_code, 206)
        self.assertEqual(res_range.headers.get("content-type"), "video/mp4")
        self.assertEqual(res_range.headers.get("content-range"), "bytes 0-1023/3908758")
        self.assertEqual(res_range.headers.get("content-length"), "1024")
        self.assertEqual(res_range.headers.get("accept-ranges"), "bytes")
        self.assertEqual(len(res_range.content), 1024)

    def test_real_saved_auxiliary_artifacts_downloads(self) -> None:
        """Angles CSV, angle plot, summary, and coaching downloads succeed with exact headers."""
        # angles.csv
        res_csv = self.client.get(f"/api/jobs/{self.job_id}/download/angles.csv")
        self.assertEqual(res_csv.status_code, 200)
        self.assertIn("text/csv", res_csv.headers.get("content-type", ""))
        self.assertIn("frame_index,video_time_s", res_csv.text[:50])

        # angle_plot.png
        res_png = self.client.get(f"/api/jobs/{self.job_id}/download/angle_plot.png")
        self.assertEqual(res_png.status_code, 200)
        self.assertEqual(res_png.headers.get("content-type"), "image/png")
        self.assertTrue(res_png.content.startswith(b"\x89PNG\r\n\x1a\n"))

        # coaching_feedback.json
        res_coaching = self.client.get(f"/api/jobs/{self.job_id}/download/coaching_feedback.json")
        self.assertEqual(res_coaching.status_code, 200)
        self.assertEqual(res_coaching.headers.get("content-type"), "application/json")
        coaching_json = res_coaching.json()
        self.assertEqual(coaching_json["status"], "ready")
        self.assertEqual(coaching_json["overall_summary"]["total_counted_reps"], 9)


# ---------------------------------------------------------------------------
# Test Suite 5: Frontend Bounded Seek & Stale Job Switching
# ---------------------------------------------------------------------------

class TestFrontendBoundedSeekAndJobSwitching(_NodeAcceptanceHarness):
    """Headless Node.js tests verifying client-side phase seeking bounds and job-switch isolation."""

    def test_seek_clamped_to_video_duration_headless(self) -> None:
        """Seeking beyond video duration clamps currentTime to duration and announces accurately."""
        js = """
const video = document.getElementById('annotated-video');
video.readyState = 4;
video.duration = 10.0;
video.currentTime = 0.0;

const seekStatus = document.getElementById('video-seek-status');

// Target time 18.5s exceeds duration 10.0s
seekVideoToTime(18.5);

assert.strictEqual(video.currentTime, 10.0, 'Video currentTime must be clamped to video duration (10.0s)');
assert.ok(seekStatus.textContent.includes('10.00'), 'Status must confirm seek to clamped 10.00s');
"""
        self.run_node_script(js)

    def test_seek_bounded_by_active_video_duration(self) -> None:
        """When video element duration is NaN (metadata loading), activeVideoDuration bounds target time."""
        js = """
const video = document.getElementById('annotated-video');
video.readyState = 0; // metadata loading
video.duration = NaN;
video.currentTime = 0.0;

setActiveVideoDuration(12.5);

const seekStatus = document.getElementById('video-seek-status');
seekVideoToTime(25.0);

assert.ok(seekStatus.textContent.includes('queued') || seekStatus.textContent.includes('loading'),
  'Must queue seek while metadata loads');

// Simulate metadata loaded with duration 12.5s
video.readyState = 1;
video.duration = 12.5;
video.dispatchEvent('loadedmetadata');

assert.strictEqual(video.currentTime, 12.5, 'Video must seek to clamped duration (12.5s) upon load');
"""
        self.run_node_script(js)

    def test_negative_and_nan_seek_handled_gracefully(self) -> None:
        """Negative or non-numeric seek values announce 'Invalid seek target timestamp' without throwing."""
        js = """
const seekStatus = document.getElementById('video-seek-status');

seekVideoToTime(-1.0);
assert.strictEqual(seekStatus.textContent, 'Invalid seek target timestamp.');

seekVideoToTime('invalid-string');
assert.strictEqual(seekStatus.textContent, 'Invalid seek target timestamp.');

seekVideoToTime(NaN);
assert.strictEqual(seekStatus.textContent, 'Invalid seek target timestamp.');
"""
        self.run_node_script(js)

    def test_job_switch_resets_media_state_and_ignores_stale(self) -> None:
        """Switching from Job A to Job B pauses playback, clears src, clears seek status, and drops stale Job A responses."""
        js = """
const video = document.getElementById('annotated-video');
const videoSrc = document.getElementById('video-source');
const anglePlot = document.getElementById('angle-plot-img');
const seekStatus = document.getElementById('video-seek-status');

// Simulate Job A active and loaded
startJobTracking('job-aaaa-1111', 'first_video.mp4');
videoSrc.src = '/api/jobs/job-aaaa-1111/download/preview.mp4';
anglePlot.src = '/api/jobs/job-aaaa-1111/download/angle_plot.png';
seekStatus.textContent = 'Seeked video to 5.00 seconds.';

// Switch to Job B
startJobTracking('job-bbbb-2222', 'second_video.mp4');

assert.strictEqual(getCurrentJobId(), 'job-bbbb-2222');
assert.strictEqual(videoSrc.src, undefined, 'Video source must be cleared on job switch');
assert.strictEqual(anglePlot.src, undefined, 'Angle plot must be cleared on job switch');
assert.strictEqual(seekStatus.textContent, '', 'Seek status must be cleared on job switch');
assert.strictEqual(getActiveVideoDuration(), null, 'Active duration must be reset on job switch');

// Stale response from Job A arriving now must be dropped
context.fetch = async (url) => {
  if (url.includes('job-aaaa-1111')) {
    return { ok: true, status: 200, json: async () => ({ status: 'completed', completed_reps: 99 }) };
  }
  return { ok: true, status: 200, json: async () => ({ status: 'queued' }) };
};

await pollJobStatus('job-aaaa-1111');

// Job B status must remain QUEUED
assert.strictEqual(document.getElementById('job-status').textContent, 'QUEUED',
  'Stale Job A response must be ignored and not clobber Job B');

stopPolling();
"""
        self.run_node_script(js)


if __name__ == "__main__":
    unittest.main()
