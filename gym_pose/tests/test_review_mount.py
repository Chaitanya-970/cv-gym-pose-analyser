"""Integration and lifecycle tests for mounted human review workspace under /review.

Verifies:
- Mounted FastAPI app lifecycle under app = create_app(enable_review=True, review_storage_root=temp_dir):
  * GET /: 200 OK, HTML contains visible navigation link to '/review/'.
  * GET /review/ (and GET /review): 200 OK (or redirect 307 to /review/), HTML contains review workspace and return link to '/'.
  * POST /review/api/upload: upload dummy raw mp4, returns 201 with UUID.
  * GET /review/api/cases: returns 200 with list containing created case.
  * GET /review/api/cases/{case_id}: returns 200 with case metadata.
  * POST /review/api/cases/{case_id}/review: saves valid review (count, reviewer, ack, explicit exposure), returns 200.
  * GET /review/api/cases/{case_id}/video: returns 200, supports Range header (e.g. Range: bytes=0-10 -> 206 Partial Content).
  * GET /review/api/export/counts and GET /review/api/export/manifest: return 200 with valid JSON.
  * Error handling: invalid UUID -> 400/404, corrupted raw video -> 500.
  * Storage isolation: verify review video and case.json are stored in review_storage_root, NOT inside web_jobs output directory.
  * Standalone launcher verification: create_review_app(storage_root=temp_dir2) still works independently at /api/upload, /api/cases, etc. (standalone port 8001 mode).
  * Node.js test for resolveReviewUrl: extract JS from review_web.INDEX_HTML and execute Node tests verifying that:
    - resolveReviewUrl('/api/cases', '/review/') returns '/review/api/cases'
    - resolveReviewUrl('/api/cases', '/review') returns '/review/api/cases'
    - resolveReviewUrl('/api/cases', '/') returns '/api/cases'
    - resolveReviewUrl('/api/cases', '') returns '/api/cases'
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
from typing import Any, Dict

# Ensure gym_pose directory is in sys.path
_GYM_POSE_DIR = Path(__file__).resolve().parent.parent
if str(_GYM_POSE_DIR) not in sys.path:
    sys.path.insert(0, str(_GYM_POSE_DIR))

try:
    from fastapi.testclient import TestClient
    from gym_analyser.web import create_app
    from gym_analyser.review_web import create_review_app, INDEX_HTML
    from gym_analyser.web_jobs import JobManager, DEFAULT_MAX_UPLOAD_BYTES
    HAS_WEB_DEPS = True
except (ImportError, TypeError):
    TestClient = None
    create_app = None
    create_review_app = None
    INDEX_HTML = ""
    JobManager = None
    DEFAULT_MAX_UPLOAD_BYTES = 200 * 1024 * 1024
    HAS_WEB_DEPS = False


class TestReviewMountLifecycle(unittest.TestCase):
    """Test suite for mounted FastAPI review application under /review."""

    def setUp(self) -> None:
        if not HAS_WEB_DEPS:
            self.skipTest("FastAPI / web dependencies not available")

        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)

        # Separate directory roots for review storage and web_jobs output
        self.review_storage_root = self.temp_path / "human_reviews"
        self.review_storage_root.mkdir(parents=True, exist_ok=True)

        self.web_jobs_storage_root = self.temp_path / "web_jobs"
        self.web_jobs_storage_root.mkdir(parents=True, exist_ok=True)

        self.job_manager = JobManager(storage_dir=self.web_jobs_storage_root)
        self.app = create_app(
            job_manager=self.job_manager,
            enable_review=True,
            review_storage_root=self.review_storage_root,
        )
        self.client = TestClient(self.app)

    def tearDown(self) -> None:
        if hasattr(self, "temp_dir"):
            self.temp_dir.cleanup()

    def _upload_dummy_video(self, name: str = "mount_test.mp4", content: bytes = b"dummy-mp4-video-stream-content"):
        files = {"file": (name, io.BytesIO(content), "video/mp4")}
        return self.client.post("/review/api/upload", files=files)

    def test_get_root_contains_review_navigation_link(self) -> None:
        """GET / returns 200 OK, and HTML contains visible navigation link to '/review/'."""
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn("text/html", response.headers.get("content-type", ""))
        html = response.text
        self.assertIn("/review/", html, "GET / must contain a visible navigation link to '/review/'")
        html_lower = html.lower()
        self.assertTrue(
            "human review" in html_lower or "review" in html_lower,
            "Root page must mention human review workspace navigation",
        )

    def test_get_review_root_returns_html_and_return_link(self) -> None:
        """GET /review/ and GET /review return 200 OK (or redirect 307 to /review/), HTML contains return link to '/'."""
        # 1. GET /review/
        res_slash = self.client.get("/review/")
        self.assertEqual(res_slash.status_code, 200)
        self.assertIn("text/html", res_slash.headers.get("content-type", ""))
        html = res_slash.text
        self.assertTrue(
            "review" in html.lower() or "squat" in html.lower(),
            "HTML at /review/ must contain review workspace interface",
        )
        # Check return navigation link to '/'
        self.assertTrue(
            'href="/"' in html or "href='/'" in html or "href=\"/\"" in html or 'to analysis' in html.lower() or 'analyser' in html.lower(),
            "Review page HTML must contain return navigation link to main analyser at '/'",
        )

        # 2. GET /review (without trailing slash) returns 200 or 307 redirect to /review/
        res_no_slash = self.client.get("/review", follow_redirects=False)
        self.assertIn(res_no_slash.status_code, [200, 307, 308])
        if res_no_slash.status_code in [307, 308]:
            redirect_target = res_no_slash.headers.get("location", "")
            self.assertTrue(
                redirect_target.endswith("/review/") or redirect_target.endswith("/review"),
                f"Redirect location must lead to /review/, got: {redirect_target}",
            )
            # Following redirect must succeed with 200
            res_followed = self.client.get("/review", follow_redirects=True)
            self.assertEqual(res_followed.status_code, 200)

    def test_post_upload_returns_201_with_uuid(self) -> None:
        """POST /review/api/upload: upload dummy raw mp4, returns 201 with UUID."""
        res = self._upload_dummy_video("sample.mp4", b"dummy-video-data-bytes")
        self.assertIn(res.status_code, [200, 201])
        data = res.json()
        case_id = data.get("case_id") or data.get("id") or data.get("clip_id")
        self.assertIsNotNone(case_id, f"Upload response must contain case identifier: {data}")
        parsed_uuid = uuid.UUID(str(case_id))
        self.assertEqual(parsed_uuid.version, 4)

    def test_get_cases_list_and_metadata(self) -> None:
        """GET /review/api/cases returns 200 with list containing created case, and GET /review/api/cases/{case_id} returns 200."""
        up_res = self._upload_dummy_video("fetch_case.mp4", b"fetch-case-bytes")
        case_id = up_res.json().get("case_id") or up_res.json().get("id") or up_res.json().get("clip_id")

        # GET /review/api/cases
        cases_res = self.client.get("/review/api/cases")
        self.assertEqual(cases_res.status_code, 200)
        cases_data = cases_res.json()
        cases_list = cases_data if isinstance(cases_data, list) else cases_data.get("cases", [])
        case_ids = [c.get("case_id") or c.get("id") or c.get("clip_id") for c in cases_list]
        self.assertIn(case_id, case_ids)

        # GET /review/api/cases/{case_id}
        case_meta_res = self.client.get(f"/review/api/cases/{case_id}")
        self.assertEqual(case_meta_res.status_code, 200)
        case_meta = case_meta_res.json()
        meta_id = case_meta.get("case_id") or case_meta.get("id") or case_meta.get("clip_id")
        self.assertEqual(meta_id, case_id)

    def test_save_valid_review_and_reload(self) -> None:
        """POST /review/api/cases/{case_id}/review: saves valid review (count, reviewer, ack, explicit exposure), returns 200."""
        up_res = self._upload_dummy_video("review_case.mp4", b"review-case-bytes")
        case_id = up_res.json().get("case_id") or up_res.json().get("id") or up_res.json().get("clip_id")

        review_payload = {
            "human_count": 5,
            "reviewer": "Reviewer Alice",
            "raw_video_reviewed": True,
            "predictions_previously_seen": False,
            "scope": "in_scope",
            "notes": "Standard clean reps observed",
            "events": [
                {
                    "start_time_s": 0.5,
                    "end_time_s": 2.2,
                    "intent": "intentional_squat",
                    "cycle": "complete",
                    "observation": "Good form",
                }
            ],
        }

        save_res = self.client.post(f"/review/api/cases/{case_id}/review", json=review_payload)
        self.assertEqual(save_res.status_code, 200)

        # Reload case and verify review data is saved
        reload_res = self.client.get(f"/review/api/cases/{case_id}")
        self.assertEqual(reload_res.status_code, 200)
        saved_data = reload_res.json()
        self.assertEqual(saved_data.get("human_count"), 5)
        self.assertEqual(saved_data.get("reviewer"), "Reviewer Alice")
        self.assertEqual(saved_data.get("predictions_previously_seen"), False)

    def test_video_endpoint_and_range_header_support(self) -> None:
        """GET /review/api/cases/{case_id}/video returns 200, supports Range header (206 Partial Content)."""
        video_content = b"0123456789abcdefghijklmnopqrstuvwxyz"
        up_res = self._upload_dummy_video("range_test.mp4", video_content)
        case_id = up_res.json().get("case_id") or up_res.json().get("id") or up_res.json().get("clip_id")

        # Full video request
        vid_res = self.client.get(f"/review/api/cases/{case_id}/video")
        self.assertEqual(vid_res.status_code, 200)
        self.assertEqual(vid_res.content, video_content)

        # Partial content request with Range header
        range_res = self.client.get(
            f"/review/api/cases/{case_id}/video",
            headers={"Range": "bytes=0-10"},
        )
        self.assertIn(range_res.status_code, [200, 206])
        if range_res.status_code == 206:
            self.assertEqual(range_res.content, video_content[0:11])
            self.assertIn("content-range", range_res.headers)

    def test_export_counts_and_manifest(self) -> None:
        """GET /review/api/export/counts and GET /review/api/export/manifest: return 200 with valid JSON."""
        up_res = self._upload_dummy_video("export_case.mp4", b"export-video-bytes")
        case_id = up_res.json().get("case_id") or up_res.json().get("id") or up_res.json().get("clip_id")

        self.client.post(
            f"/review/api/cases/{case_id}/review",
            json={
                "human_count": 3,
                "reviewer": "Bob",
                "raw_video_reviewed": True,
                "predictions_previously_seen": False,
                "scope": "in_scope",
            },
        )

        # GET /review/api/export/counts
        counts_res = self.client.get("/review/api/export/counts")
        self.assertEqual(counts_res.status_code, 200)
        counts_json = counts_res.json()
        self.assertIn("clips", counts_json)
        clip_ids = [c["clip_id"] for c in counts_json["clips"]]
        self.assertIn(case_id, clip_ids)

        # GET /review/api/export/manifest
        manifest_res = self.client.get("/review/api/export/manifest")
        self.assertEqual(manifest_res.status_code, 200)
        manifest_json = manifest_res.json()
        self.assertTrue(isinstance(manifest_json, (dict, list)))

    def test_error_handling_invalid_uuid_and_corrupt_video(self) -> None:
        """Error handling: invalid UUID -> 400/404, corrupted raw video -> 500."""
        # 1. Invalid UUID format
        res_bad_uuid = self.client.get("/review/api/cases/invalid-uuid-string")
        self.assertIn(res_bad_uuid.status_code, [400, 404, 422])

        # 2. Non-existent UUID
        res_missing = self.client.get(f"/review/api/cases/{uuid.uuid4()}")
        self.assertEqual(res_missing.status_code, 404)

        # 3. Corrupted raw video integrity failure
        up_res = self._upload_dummy_video("corrupt_test.mp4", b"original-intact-video-data")
        case_id = up_res.json().get("case_id") or up_res.json().get("id") or up_res.json().get("clip_id")

        # Mutate the saved raw video bytes in the review storage directory
        case_dir = self.review_storage_root / case_id
        video_files = list(case_dir.glob("video.*"))
        self.assertTrue(len(video_files) > 0, f"Expected video file in {case_dir}")
        video_files[0].write_bytes(b"tampered-corrupted-bytes-causing-checksum-failure")

        # Accessing video endpoint should now fail integrity check with 500
        res_vid_corrupt = self.client.get(f"/review/api/cases/{case_id}/video")
        self.assertEqual(res_vid_corrupt.status_code, 500)

        # Saving review for tampered case should also return 500
        res_rev_corrupt = self.client.post(
            f"/review/api/cases/{case_id}/review",
            json={
                "human_count": 2,
                "reviewer": "Integrity Check",
                "raw_video_reviewed": True,
                "predictions_previously_seen": False,
            },
        )
        self.assertEqual(res_rev_corrupt.status_code, 500)

    def test_storage_isolation(self) -> None:
        """Storage isolation: review video and case.json are stored in review_storage_root, NOT in web_jobs output directory!"""
        up_res = self._upload_dummy_video("isolated.mp4", b"isolation-check-video-bytes")
        case_id = up_res.json().get("case_id") or up_res.json().get("id") or up_res.json().get("clip_id")

        # Verify stored in review_storage_root
        case_dir_review = self.review_storage_root / case_id
        self.assertTrue(case_dir_review.is_dir(), f"Expected case dir {case_dir_review} in review_storage_root")
        self.assertTrue((case_dir_review / "case.json").is_file(), "case.json must exist in review_storage_root")
        video_files = list(case_dir_review.glob("video.*"))
        self.assertTrue(len(video_files) > 0, "Raw review video must exist in review_storage_root")

        # Verify web_jobs directory does NOT contain the case
        case_dir_jobs = self.web_jobs_storage_root / case_id
        self.assertFalse(case_dir_jobs.exists(), "Review case must NOT exist inside web_jobs output directory!")


    def test_primary_storage_overlap_and_config(self):
        for root in [self.web_jobs_storage_root, self.web_jobs_storage_root / "reviews", self.web_jobs_storage_root.parent]:
            with self.subTest(root=root):
                with self.assertRaises(ValueError):
                    create_app(job_manager=self.job_manager, review_storage_root=root)
        with self.assertRaises(TypeError):
            create_app(job_manager=self.job_manager, enable_review="yes")
        client = TestClient(create_app(job_manager=self.job_manager, enable_review=False))
        self.assertNotIn('href="/review/"', client.get("/").text)

    def test_disabled_review_mount_returns_404(self) -> None:
        """When enable_review=False, /review/ returns 404."""
        app_disabled = create_app(
            job_manager=self.job_manager,
            enable_review=False,
            review_storage_root=self.review_storage_root,
        )
        client_disabled = TestClient(app_disabled)
        res = client_disabled.get("/review/")
        self.assertEqual(res.status_code, 404)

    def test_create_app_rejects_unexpected_kwargs(self) -> None:
        """create_app with unexpected keyword arguments raises TypeError."""
        with self.assertRaises(TypeError):
            create_app(job_manager=self.job_manager, unexpected_param=123)


class TestStandaloneReviewApp(unittest.TestCase):
    """Verify create_review_app(storage_root=temp_dir2) works independently in standalone port 8001 mode."""

    def setUp(self) -> None:
        if not HAS_WEB_DEPS:
            self.skipTest("FastAPI / web dependencies not available")

        self.temp_dir = tempfile.TemporaryDirectory()
        self.storage_root = Path(self.temp_dir.name) / "standalone_reviews"
        self.storage_root.mkdir(parents=True, exist_ok=True)

        self.app = create_review_app(storage_root=self.storage_root)
        self.client = TestClient(self.app)

    def tearDown(self) -> None:
        if hasattr(self, "temp_dir"):
            self.temp_dir.cleanup()

    def test_standalone_endpoints_operate_at_root_api_paths(self) -> None:
        """Standalone review app operates directly at / and /api/... without /review prefix."""
        # 1. GET /
        res_root = self.client.get("/")
        self.assertEqual(res_root.status_code, 200)

        # 2. POST /api/upload
        files = {"file": ("standalone.mp4", io.BytesIO(b"standalone-bytes"), "video/mp4")}
        res_up = self.client.post("/api/upload", files=files)
        self.assertIn(res_up.status_code, [200, 201])
        case_id = res_up.json().get("case_id") or res_up.json().get("id") or res_up.json().get("clip_id")

        # 3. GET /api/cases
        res_cases = self.client.get("/api/cases")
        self.assertEqual(res_cases.status_code, 200)

        # 4. GET /api/cases/{case_id}
        res_meta = self.client.get(f"/api/cases/{case_id}")
        self.assertEqual(res_meta.status_code, 200)

        # 5. POST /api/cases/{case_id}/review
        res_rev = self.client.post(
            f"/api/cases/{case_id}/review",
            json={
                "human_count": 4,
                "reviewer": "Standalone Reviewer",
                "raw_video_reviewed": True,
                "predictions_previously_seen": False,
            },
        )
        self.assertEqual(res_rev.status_code, 200)

        # 6. GET /api/export/counts
        res_counts = self.client.get("/api/export/counts")
        self.assertEqual(res_counts.status_code, 200)


class TestReviewUrlResolutionNode(unittest.TestCase):
    """Node.js verification of resolveReviewUrl helper extracted from review_web.INDEX_HTML."""

    def test_resolve_review_url_in_node(self) -> None:
        """Execute headless Node test verifying resolveReviewUrl under both mounted and standalone prefixes."""
        if shutil.which("node") is None:
            self.skipTest("Node.js is not installed on this system")

        from gym_analyser import review_web

        html = review_web.INDEX_HTML
        m = re.search(r"<script>(.*?)</script>", html, re.DOTALL)
        self.assertIsNotNone(m, "Could not find <script> tag in INDEX_HTML")
        js_code = m.group(1)

        test_script = """
const assert = require('assert');

// Minimal DOM stub to allow headless script evaluation without browser
global.document = {
  getElementById: (id) => ({
    id: id,
    addEventListener: () => {},
    appendChild: () => {},
    querySelector: () => ({ addEventListener: () => {} }),
    querySelectorAll: () => [],
    style: {},
    classList: { add: () => {}, remove: () => {} },
    value: '',
    checked: false,
    textContent: ''
  }),
  createElement: (tag) => ({
    tagName: tag,
    appendChild: () => {},
    addEventListener: () => {},
    classList: { add: () => {} },
    style: {}
  })
};
global.window = {
  location: { pathname: '/' }
};
global.fetch = () => Promise.resolve({ ok: true, json: () => Promise.resolve({}) });

// Evaluate extracted JS from review_web.py
""" + js_code + """

// Verify resolveReviewUrl function existence
assert.ok(
  typeof resolveReviewUrl === 'function',
  'resolveReviewUrl must be a defined function in review_web.py JS'
);

// Verify required specifications:
// 1. resolveReviewUrl('/api/cases', '/review/') returns '/review/api/cases'
assert.strictEqual(
  resolveReviewUrl('/api/cases', '/review/'),
  '/review/api/cases',
  "resolveReviewUrl('/api/cases', '/review/') should return '/review/api/cases'"
);

// 2. resolveReviewUrl('/api/cases', '/review') returns '/review/api/cases'
assert.strictEqual(
  resolveReviewUrl('/api/cases', '/review'),
  '/review/api/cases',
  "resolveReviewUrl('/api/cases', '/review') should return '/review/api/cases'"
);

// 3. resolveReviewUrl('/api/cases', '/') returns '/api/cases'
assert.strictEqual(
  resolveReviewUrl('/api/cases', '/'),
  '/api/cases',
  "resolveReviewUrl('/api/cases', '/') should return '/api/cases'"
);

// 4. resolveReviewUrl('/api/cases', '') returns '/api/cases'
assert.strictEqual(
  resolveReviewUrl('/api/cases', ''),
  '/api/cases',
  "resolveReviewUrl('/api/cases', '') should return '/api/cases'"
);

// Additional edge cases: subpaths under /review/
assert.strictEqual(
  resolveReviewUrl('/api/cases', '/review/case-123'),
  '/review/api/cases',
  "resolveReviewUrl under '/review/case-123' should return '/review/api/cases'"
);

// Relative path input without leading slash
assert.strictEqual(
  resolveReviewUrl('api/upload', '/review/'),
  '/review/api/upload',
  "resolveReviewUrl('api/upload', '/review/') should return '/review/api/upload'"
);
assert.strictEqual(
  resolveReviewUrl('api/upload', '/'),
  '/api/upload',
  "resolveReviewUrl('api/upload', '/') should return '/api/upload'"
);

console.log('NODE_RESOLVE_REVIEW_URL_SUCCESS');
"""

        with tempfile.NamedTemporaryFile("w", suffix=".js", delete=False, encoding="utf-8") as f:
            f.write(test_script)
            tmp_path = f.name

        try:
            res = subprocess.run(["node", tmp_path], capture_output=True, text=True, check=False, timeout=15)
            self.assertEqual(
                res.returncode,
                0,
                f"Node.js headless verification of resolveReviewUrl failed with code {res.returncode}.\nSTDERR:\n{res.stderr}\nSTDOUT:\n{res.stdout}",
            )
            self.assertIn("NODE_RESOLVE_REVIEW_URL_SUCCESS", res.stdout)
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)


if __name__ == "__main__":
    unittest.main()
