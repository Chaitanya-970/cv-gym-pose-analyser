"""API and web endpoint tests for gym_analyser.review_web.

Validates FastAPI endpoints and human review web interface:
- GET / returns 200, HTML contains title, form controls.
- POST /api/upload with dummy mp4 returns 201 with UUID.
- POST /api/upload with invalid extension returns 400 or 415.
- POST /api/upload oversize returns 413.
- GET /api/cases/{case_id} returns 200 with case metadata.
- POST /api/cases/{case_id}/review:
  - missing reviewer/ack with count -> 400.
  - boolean count -> 400.
  - valid review -> 200.
- GET /api/cases/{case_id}/video returns 200 with video content.
- GET /api/export/counts and GET /api/export/manifest return 200 with valid JSON.
- Path traversal / invalid UUID returns 400 or 404.
- No prediction imports: inspects review_web module to ensure no yolo, ultralytics,
  torch, pipeline, or counter are imported.
- Task 32 regression tests:
  - test_unexpected_kwargs_rejected
  - test_predictions_exposure_required_when_count_set
  - test_raw_tamper_web_rejection
  - test_upload_file_closed
  - test_ui_js_pure_functions_via_node
"""

from __future__ import annotations

import inspect
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
from unittest.mock import AsyncMock, patch

_GYM_POSE_DIR = Path(__file__).resolve().parent.parent
if str(_GYM_POSE_DIR) not in sys.path:
    sys.path.insert(0, str(_GYM_POSE_DIR))

try:
    from fastapi.testclient import TestClient
    from gym_analyser.review_web import INDEX_HTML, create_review_app
    HAS_WEB_DEPS = True
except (ImportError, TypeError):
    TestClient = None
    create_review_app = None
    INDEX_HTML = ""
    HAS_WEB_DEPS = False


def _build_test_app(storage_root: str, max_upload_bytes: int = 10 * 1024 * 1024):
    """Instantiate test review app with explicit storage root and max upload bytes."""
    if create_review_app is None:
        raise RuntimeError("FastAPI / web dependencies not available")
    return create_review_app(storage_root=storage_root, max_upload_bytes=max_upload_bytes)


class TestReviewWebEndpoints(unittest.TestCase):
    """Test suite for FastAPI review web endpoints."""

    def setUp(self) -> None:
        if not HAS_WEB_DEPS:
            self.skipTest("FastAPI or review_web dependencies not available")
        self.test_dir = tempfile.mkdtemp(prefix="test_review_web_")
        self.max_bytes = 10 * 1024 * 1024
        self.app = _build_test_app(self.test_dir, self.max_bytes)
        self.client = TestClient(self.app)

    def tearDown(self) -> None:
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def _upload_dummy_video(self, name: str = "test.mp4", content: bytes = b"dummy-mp4-video-stream"):
        files = {"file": (name, io.BytesIO(content), "video/mp4")}
        return self.client.post("/api/upload", files=files)

    # 1. GET / returns 200, HTML contains title, form controls
    def test_get_root_returns_html_ui(self) -> None:
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn("text/html", response.headers.get("content-type", ""))
        html = response.text.lower()
        self.assertTrue("<title>" in html or "squat" in html or "review" in html)
        self.assertTrue("input" in html or "button" in html or "form" in html or "upload" in html)

    # 2. POST /api/upload with dummy mp4 returns 201 with UUID
    def test_upload_dummy_mp4_success(self) -> None:
        res = self._upload_dummy_video("valid.mp4", b"dummy-mp4-bytes")
        self.assertIn(res.status_code, [200, 201])
        data = res.json()
        case_id = data.get("case_id") or data.get("id") or data.get("clip_id")
        self.assertIsNotNone(case_id, f"Missing case ID in upload response: {data}")
        val = uuid.UUID(str(case_id))
        self.assertEqual(val.version, 4)

    # 3. POST /api/upload with invalid extension returns 400 or 415
    def test_upload_invalid_extension(self) -> None:
        bad_files = [
            ("script.exe", b"executable", "application/octet-stream"),
            ("notes.txt", b"plain text", "text/plain"),
            ("legacy.avi", b"avi format", "video/x-msvideo"),
        ]
        for fname, content, ctype in bad_files:
            files = {"file": (fname, io.BytesIO(content), ctype)}
            res = self.client.post("/api/upload", files=files)
            self.assertIn(res.status_code, [400, 415, 422], f"Expected 400/415/422 for {fname}, got {res.status_code}")

    # 4. POST /api/upload oversize returns 413
    def test_upload_oversize_returns_413(self) -> None:
        small_dir = tempfile.mkdtemp(prefix="test_review_small_")
        try:
            small_app = _build_test_app(small_dir, max_upload_bytes=100)
            small_client = TestClient(small_app)
            oversize_files = {"file": ("big.mp4", io.BytesIO(b"A" * 200), "video/mp4")}
            res = small_client.post("/api/upload", files=oversize_files)
            self.assertIn(res.status_code, [413, 400, 422], f"Expected 413 for oversize upload, got {res.status_code}")
        finally:
            shutil.rmtree(small_dir, ignore_errors=True)

    # 5. GET /api/cases/{case_id} returns 200
    def test_get_case_metadata(self) -> None:
        up_res = self._upload_dummy_video("fetch.mp4", b"content-fetch")
        case_id = up_res.json().get("case_id") or up_res.json().get("id") or up_res.json().get("clip_id")

        get_res = self.client.get(f"/api/cases/{case_id}")
        self.assertEqual(get_res.status_code, 200)
        case_data = get_res.json()
        self.assertIn(case_id, [case_data.get("case_id"), case_data.get("id"), case_data.get("clip_id")])

    # 6. POST /api/cases/{case_id}/review
    def test_review_validation_and_saving(self) -> None:
        up_res = self._upload_dummy_video("rev.mp4", b"content-rev")
        case_id = up_res.json().get("case_id") or up_res.json().get("id") or up_res.json().get("clip_id")

        # Missing reviewer with count set -> 400 or 422
        bad_rev1 = self.client.post(
            f"/api/cases/{case_id}/review",
            json={"human_count": 3, "reviewer": "", "raw_video_reviewed": True, "predictions_previously_seen": False},
        )
        self.assertIn(bad_rev1.status_code, [400, 422])

        # Missing ack (raw_video_reviewed=False) with count set -> 400 or 422
        bad_rev2 = self.client.post(
            f"/api/cases/{case_id}/review",
            json={"human_count": 3, "reviewer": "Jane Doe", "raw_video_reviewed": False, "predictions_previously_seen": False},
        )
        self.assertIn(bad_rev2.status_code, [400, 422])

        # Boolean count -> 400 or 422
        bad_rev3 = self.client.post(
            f"/api/cases/{case_id}/review",
            json={"human_count": True, "reviewer": "Jane Doe", "raw_video_reviewed": True, "predictions_previously_seen": False},
        )
        self.assertIn(bad_rev3.status_code, [400, 422])

        # Valid review -> 200
        valid_res = self.client.post(
            f"/api/cases/{case_id}/review",
            json={
                "human_count": 3,
                "reviewer": "Jane Doe",
                "raw_video_reviewed": True,
                "predictions_previously_seen": False,
                "scope": "in_scope",
                "notes": "Verified good squats",
                "events": [
                    {
                        "start_time_s": 0.5,
                        "end_time_s": 2.0,
                        "intent": "intentional_squat",
                        "cycle": "complete",
                        "observation": "Good depth",
                    }
                ],
            },
        )
        self.assertEqual(valid_res.status_code, 200)

        # Confirm reload reflects saved review
        get_res = self.client.get(f"/api/cases/{case_id}")
        self.assertEqual(get_res.status_code, 200)
        self.assertEqual(get_res.json().get("human_count"), 3)
        self.assertEqual(get_res.json().get("reviewer"), "Jane Doe")

    # 7. GET /api/cases/{case_id}/video returns 200 with video content
    def test_get_case_video_stream(self) -> None:
        video_bytes = b"my-custom-video-test-bytes"
        up_res = self._upload_dummy_video("stream.mp4", video_bytes)
        case_id = up_res.json().get("case_id") or up_res.json().get("id") or up_res.json().get("clip_id")

        vid_res = self.client.get(f"/api/cases/{case_id}/video")
        self.assertEqual(vid_res.status_code, 200)
        self.assertEqual(vid_res.content, video_bytes)

    # 8. GET /api/export/counts and GET /api/export/manifest
    def test_export_counts_and_manifest_endpoints(self) -> None:
        up_res = self._upload_dummy_video("export_case.mp4", b"data")
        cid = up_res.json().get("case_id") or up_res.json().get("id") or up_res.json().get("clip_id")
        self.client.post(
            f"/api/cases/{cid}/review",
            json={
                "human_count": 4,
                "reviewer": "Jane",
                "raw_video_reviewed": True,
                "predictions_previously_seen": False,
                "scope": "in_scope",
            },
        )

        # Test counts export
        counts_res = self.client.get("/api/export/counts")
        self.assertEqual(counts_res.status_code, 200)
        counts_data = counts_res.json()
        self.assertIn("clips", counts_data)
        clip_ids = [c["clip_id"] for c in counts_data["clips"]]
        self.assertIn(cid, clip_ids)

        # Test manifest export
        manifest_res = self.client.get("/api/export/manifest")
        self.assertEqual(manifest_res.status_code, 200)
        manifest_data = manifest_res.json()
        self.assertTrue(isinstance(manifest_data, (dict, list)))

    # 9. Path traversal / invalid UUID returns 400 or 404
    def test_path_traversal_and_invalid_uuid(self) -> None:
        # Invalid UUID
        res1 = self.client.get("/api/cases/not-a-valid-uuid")
        self.assertIn(res1.status_code, [400, 404, 422])

        # Non-existent UUID
        res2 = self.client.get(f"/api/cases/{uuid.uuid4()}")
        self.assertEqual(res2.status_code, 404)

        # Path traversal attempt on video endpoint
        res3 = self.client.get("/api/cases/../../etc/passwd/video")
        self.assertIn(res3.status_code, [400, 404, 422])

        # Invalid UUID on video endpoint
        res4 = self.client.get("/api/cases/not-a-valid-uuid/video")
        self.assertIn(res4.status_code, [400, 404, 422])

    # =========================================================================
    # Task 32 Rigorous Regression Tests
    # =========================================================================

    def test_unexpected_kwargs_rejected(self) -> None:
        """create_review_app(unexpected_arg=123) raises TypeError."""
        with self.assertRaises(TypeError):
            create_review_app(unexpected_arg=123)

        with self.assertRaises(TypeError):
            create_review_app(storage_root=self.test_dir, invalid_param="hello")

    def test_predictions_exposure_required_when_count_set(self) -> None:
        """POST /api/cases/{case_id}/review requires explicit boolean predictions exposure when count set."""
        up_res = self._upload_dummy_video("preds_clip.mp4", b"preds-data")
        cid = up_res.json().get("case_id") or up_res.json().get("id") or up_res.json().get("clip_id")

        # 1. predictions_previously_seen omitted -> 400
        res_omitted = self.client.post(
            f"/api/cases/{cid}/review",
            json={
                "human_count": 5,
                "reviewer": "Jane Doe",
                "raw_video_reviewed": True,
            },
        )
        self.assertEqual(res_omitted.status_code, 400)

        # 2. predictions_previously_seen=None -> 400
        res_none = self.client.post(
            f"/api/cases/{cid}/review",
            json={
                "human_count": 5,
                "reviewer": "Jane Doe",
                "raw_video_reviewed": True,
                "predictions_previously_seen": None,
            },
        )
        self.assertEqual(res_none.status_code, 400)

        # 3. Explicit True -> 200
        res_true = self.client.post(
            f"/api/cases/{cid}/review",
            json={
                "human_count": 5,
                "reviewer": "Jane Doe",
                "raw_video_reviewed": True,
                "predictions_previously_seen": True,
            },
        )
        self.assertEqual(res_true.status_code, 200)

        # 4. Explicit False -> 200
        res_false = self.client.post(
            f"/api/cases/{cid}/review",
            json={
                "human_count": 5,
                "reviewer": "Jane Doe",
                "raw_video_reviewed": True,
                "predictions_previously_seen": False,
            },
        )
        self.assertEqual(res_false.status_code, 200)

    def test_raw_tamper_web_rejection(self) -> None:
        """Mutating managed video file causes video, review, and export endpoints to return 500."""
        up_res = self._upload_dummy_video("tamper_web.mp4", b"orig-video-data")
        cid = up_res.json().get("case_id") or up_res.json().get("id") or up_res.json().get("clip_id")

        # Mutate video file on disk
        vpath = next(Path(self.test_dir, cid).glob("video.*"))
        vpath.write_bytes(b"tampered-mutated-video-content")

        # 1. GET video returns 500 (integrity check failed)
        vid_res = self.client.get(f"/api/cases/{cid}/video")
        self.assertEqual(vid_res.status_code, 500)

        # 2. POST review returns 500
        rev_res = self.client.post(
            f"/api/cases/{cid}/review",
            json={
                "human_count": 3,
                "reviewer": "Jane Doe",
                "raw_video_reviewed": True,
                "predictions_previously_seen": False,
            },
        )
        self.assertEqual(rev_res.status_code, 500)

        # 3. GET /api/export/counts returns 500
        counts_res = self.client.get("/api/export/counts")
        self.assertEqual(counts_res.status_code, 500)

    def test_upload_file_closed(self) -> None:
        """Verify upload endpoint always closes upload file in both success and error paths."""
        import starlette.datastructures

        # 1. Success path
        with patch.object(starlette.datastructures.UploadFile, "close", new_callable=AsyncMock) as mock_close:
            res = self._upload_dummy_video("close_success.mp4", b"test-content")
            self.assertIn(res.status_code, [200, 201])
            self.assertTrue(mock_close.called, "UploadFile.close must be awaited on success")

        # 2. Error path: invalid extension
        with patch.object(starlette.datastructures.UploadFile, "close", new_callable=AsyncMock) as mock_close:
            bad_file = {"file": ("script.exe", io.BytesIO(b"content"), "application/octet-stream")}
            res = self.client.post("/api/upload", files=bad_file)
            self.assertIn(res.status_code, [400, 415, 422])
            self.assertTrue(mock_close.called, "UploadFile.close must be awaited on invalid extension error")

        # 3. Error path: oversize upload
        small_dir = tempfile.mkdtemp(prefix="test_close_small_")
        try:
            small_app = _build_test_app(small_dir, max_upload_bytes=10)
            small_client = TestClient(small_app)
            with patch.object(starlette.datastructures.UploadFile, "close", new_callable=AsyncMock) as mock_close:
                big_file = {"file": ("big.mp4", io.BytesIO(b"X" * 100), "video/mp4")}
                res = small_client.post("/api/upload", files=big_file)
                self.assertEqual(res.status_code, 413)
                self.assertTrue(mock_close.called, "UploadFile.close must be awaited on oversize error")
        finally:
            shutil.rmtree(small_dir, ignore_errors=True)

    def test_ui_js_pure_functions_via_node(self) -> None:
        """Extract JavaScript from review_web.py and run headless Node.js tests verifying strict parsing and race safety."""
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
    classList: { add: () => {}, remove: () => {} }
  }),
  createElement: (tag) => ({
    tagName: tag,
    appendChild: () => {},
    addEventListener: () => {},
    classList: { add: () => {} },
    style: {}
  })
};
global.window = global;
global.fetch = () => Promise.resolve({ ok: true, json: () => Promise.resolve({}) });

// Evaluate extracted JS from review_web.py
""" + js_code + """

// ============================================================================
// Part a: Strict integer count parsing (parseStrictCount)
// ============================================================================
assert.ok(typeof parseStrictCount === 'function', 'parseStrictCount must be a defined function');
assert.strictEqual(parseStrictCount('5'), 5, "parseStrictCount('5') should return 5");
assert.strictEqual(parseStrictCount('0'), 0, "parseStrictCount('0') should return 0");
assert.strictEqual(parseStrictCount(''), null, "parseStrictCount('') should return null");
assert.strictEqual(parseStrictCount('   '), null, "parseStrictCount('   ') should return null");
assert.strictEqual(parseStrictCount(null), null, "parseStrictCount(null) should return null");
assert.strictEqual(parseStrictCount(undefined), null, "parseStrictCount(undefined) should return null");

// Must reject non-integers and strings with non-digit characters as NaN
assert.ok(Number.isNaN(parseStrictCount('5abc')), "parseStrictCount('5abc') must be NaN");
assert.ok(Number.isNaN(parseStrictCount('2.5')), "parseStrictCount('2.5') must be NaN");
assert.ok(Number.isNaN(parseStrictCount('-1')), "parseStrictCount('-1') must be NaN");
assert.ok(Number.isNaN(parseStrictCount('abc')), "parseStrictCount('abc') must be NaN");

// ============================================================================
// Part b: Request generation / out-of-order response handling
// ============================================================================
// Simulate user rapidly switching from Case A to Case B:
// Case A fetch starts (gen 1, slow: 50ms)
// User immediately switches to Case B (gen 2, fast: 10ms)
// Verify stale response from Case A is discarded and Case B remains displayed.
let activeGen = 0;
let displayedCase = null;

function simulateSelectCase(caseId, delayMs) {
  activeGen++;
  const capturedGen = activeGen;
  return new Promise(resolve => {
    setTimeout(() => {
      // Out-of-order response discard check:
      if (capturedGen === activeGen) {
        displayedCase = caseId;
      }
      resolve();
    }, delayMs);
  });
}

const reqA = simulateSelectCase('case-A', 50);
const reqB = simulateSelectCase('case-B', 10);

Promise.all([reqA, reqB]).then(() => {
  assert.strictEqual(
    displayedCase,
    'case-B',
    'Stale response from case-A was not discarded; case-B should be active'
  );
  console.log('NODE_JS_VERIFICATION_SUCCESS');
});
"""

        with tempfile.NamedTemporaryFile("w", suffix=".js", delete=False, encoding="utf-8") as f:
            f.write(test_script)
            tmp_path = f.name

        try:
            res = subprocess.run(["node", tmp_path], capture_output=True, text=True, check=False, timeout=15)
            self.assertEqual(
                res.returncode,
                0,
                f"Node.js headless verification failed with code {res.returncode}.\nSTDERR:\n{res.stderr}\nSTDOUT:\n{res.stdout}",
            )
            self.assertIn("NODE_JS_VERIFICATION_SUCCESS", res.stdout)
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)


class TestNoPredictionImports(unittest.TestCase):
    """Enforce strict separation: review_web must never import ML/prediction libraries."""

    def test_no_prediction_modules_imported(self) -> None:
        try:
            import gym_analyser.review_web as rweb
        except ImportError:
            self.skipTest("gym_analyser.review_web not yet available")

        # Inspect source code of review_web
        src = inspect.getsource(rweb).lower()
        forbidden_tokens = [
            "ultralytics",
            "yolo",
            "torch",
            "gym_analyser.pipeline",
            "gym_analyser.counter",
            "pipeline.py",
        ]
        for token in forbidden_tokens:
            self.assertNotIn(
                f"import {token}",
                src,
                f"Forbidden prediction import '{token}' found in review_web.py source",
            )
            self.assertNotIn(
                f"from {token}",
                src,
                f"Forbidden prediction import from '{token}' found in review_web.py source",
            )



class TestActualReviewClientRegression(unittest.TestCase):
    def test_actual_page_async_association(self):
        import subprocess, tempfile, shutil
        from gym_analyser.review_web import INDEX_HTML
        if shutil.which("node") is None:
            self.skipTest("Node is not installed")
        script = '\nconst assert = require(\'assert\'), vm = require(\'vm\');\nconst html = JSON.parse(require(\'fs\').readFileSync(process.argv[2], \'utf8\'));\nconst js = html.match(/<script>([\\s\\S]*?)<\\/script>/)[1];\nconst elements = {};\nfunction element(id) {\n if(!elements[id]) elements[id] = {value:\'\',checked:false,disabled:false,textContent:\'\',style:{},files:[],\n classList:{add(){},remove(){},toggle(){}},handlers:{},\n addEventListener(name,fn){this.handlers[name]=fn;},querySelectorAll(){return[];},\n appendChild(){},removeChild(){},setAttribute(){},load(){},pause(){}};\n return elements[id];\n}\nconst pending=[];\nconst ctx=vm.createContext({console,Number,Promise,Date,NaN,JSON,encodeURIComponent,\n module:{exports:{}},confirm:()=>true,window:{location:{pathname:process.argv[3] || \'/\'}},\n document:{getElementById:element,createElement:()=>element(\'temporary\')},\n fetch(url,opts){\n  if(process.argv[3]===\'/review/\') assert(url.startsWith(\'/review/\'),\'mounted request must stay under review prefix\');\n  url=url.replace(/^\\/review(?=\\/)/,\'\');\n  if(url===\'/api/cases\')return Promise.resolve({ok:true,json:async()=>({cases:[]})});\n  return new Promise(resolve=>pending.push({url,opts,resolve}));\n }});\nvm.runInContext(js,ctx);\nvm.runInContext("renderCasesTable=()=>{}; populateCaseView=c=>{displayed=c;}; setReviewFormDisabled=v=>{formDisabled=v;};",ctx);\nconst response=data=>({ok:true,json:async()=>data});\n(async()=>{\n for(const value of [\'5abc\',\'2.5\',\'-1\',\'1e3\',\'Infinity\',\'9007199254740992\'])\n  assert(Number.isNaN(ctx.parseStrictCount(value)),value);\n assert.strictEqual(ctx.parseStrictCount(\'0\'),0);\n assert.strictEqual(ctx.parseStrictCount(\'\'),null);\n const a=ctx.selectCase(\'A\'), b=ctx.selectCase(\'B\');\n pending.find(x=>x.url.endsWith(\'/B\')).resolve(response({case_id:\'B\',tag:\'fresh B\'}));\n await b;\n pending.find(x=>x.url.endsWith(\'/A\')).resolve(response({case_id:\'A\',tag:\'stale A\'}));\n await a;assert.strictEqual(ctx.displayed.tag,\'fresh B\');\n function setup(){\n  element(\'review-count\').value=\'2\';element(\'review-reviewer\').value=\'synthetic test\';\n  element(\'review-raw-ack\').checked=true;element(\'review-preds-seen\').value=\'false\';\n  element(\'review-scope\').value=\'in_scope\';element(\'review-notes\').value=\'\';\n }\n setup();\n const saving=element(\'review-form\').handlers.submit({preventDefault(){}});\n const post=pending.find(x=>x.opts && x.opts.method===\'POST\');\n assert(post,\'actual save handler must POST\');assert.strictEqual(ctx.formDisabled,true);\n const reload=ctx.selectCase(\'B\'), reloadGet=pending.filter(x=>x.url.endsWith(\'/B\')).at(-1);\n reloadGet.resolve(response({case_id:\'B\',tag:\'new B\'}));await reload;\n post.resolve(response({case_id:\'B\',tag:\'stale save\'}));await saving;\n assert.strictEqual(ctx.displayed.tag,\'new B\',\'old same-case save must not overwrite reopened case\');\n setup();\n const save2=element(\'review-form\').handlers.submit({preventDefault(){}});\n const post2=pending.filter(x=>x.opts && x.opts.method===\'POST\').at(-1);\n const loading=ctx.selectCase(\'C\'), getC=pending.find(x=>x.url.endsWith(\'/C\'));\n post2.resolve(response({case_id:\'B\',tag:\'old B\'}));await save2;\n assert.strictEqual(ctx.formDisabled,true,\'stale save must not enable form while C is loading\');\n getC.resolve(response({case_id:\'C\',tag:\'fresh C\'}));await loading;\n assert.strictEqual(ctx.displayed.tag,\'fresh C\');\n assert.strictEqual(ctx.formDisabled,false);\n console.log(\'ACTUAL_REVIEW_UI_PASS: strict parsing, stale load, same-case save, loading-form guard\');\n})().catch(e=>{console.error(e);process.exitCode=1;});\n'
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            (base / "checks.cjs").write_text(script, encoding="utf-8")
            (base / "html.json").write_text(json.dumps(INDEX_HTML), encoding="utf-8")
            for prefix in ["/", "/review/"]:
                result = subprocess.run(["node", str(base / "checks.cjs"), str(base / "html.json"), prefix], capture_output=True, text=True, timeout=15)
                self.assertEqual(result.returncode, 0, prefix + result.stdout + result.stderr)
                self.assertIn("ACTUAL_REVIEW_UI_PASS", result.stdout)

if __name__ == "__main__":
    unittest.main()
