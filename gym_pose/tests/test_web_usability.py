"""Usability, accessibility, client validation, and endpoint regression tests for 2D Gym Pose Analyser.

Validates:
- Accessibility markup in INDEX_HTML: explicit label associations, visible keyboard focus (:focus-visible),
  touch target sizes (>= 44px), ARIA live regions and roles (role="status", role="alert", aria-live),
  and keyboard-activatable drop zone (tabindex="0", Enter/Space keydown handlers).
- Responsive mobile layout, viewport settings, and overflow-safe data tables.
- Client validation logic matching authoritative server constraints (<= 200MB, allowed extensions: .mp4, .mov, .avi, .mkv, .webm, .m4v).
- Duplicate form submission prevention and robust error recovery across busy-server (HTTP 429),
  service unavailable (HTTP 503), client validation, and network failure states.
- Seamless backend endpoint interoperability: GET /, POST /api/jobs, GET /api/status,
  GET /api/jobs/{id}, GET /api/jobs/{id}/download/{filename}, and the mounted /review/ workspace.
- Safe text rendering preventing XSS vulnerabilities.
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

_GYM_POSE_DIR = Path(__file__).resolve().parent.parent
if str(_GYM_POSE_DIR) not in sys.path:
    sys.path.insert(0, str(_GYM_POSE_DIR))

try:
    from gym_analyser.web import INDEX_HTML
except ImportError:
    INDEX_HTML = ""

from gym_analyser.web_jobs import (
    ALLOWED_ARTIFACTS,
    ALLOWED_EXTENSIONS,
    DEFAULT_MAX_UPLOAD_BYTES,
    JobManager,
    STATUS_COMPLETED,
)

try:
    from fastapi.testclient import TestClient
    from gym_analyser.web import create_app
    HAS_WEB_DEPS = True
except (ImportError, TypeError):
    TestClient = None
    create_app = None
    HAS_WEB_DEPS = False


class TestAccessibilityAndMarkup(unittest.TestCase):
    """Verify accessibility (a11y), responsive styling, and markup structure in INDEX_HTML."""

    def test_explicit_label_for_video_input(self) -> None:
        """Video file input must be explicitly associated with a label element."""
        self.assertIn('id="video-input"', INDEX_HTML)
        self.assertIn('<label for="video-input"', INDEX_HTML)
        self.assertIn('id="file-label"', INDEX_HTML)

    def test_keyboard_activatable_drop_zone(self) -> None:
        """Drop zone must be keyboard-focusable and activate on Enter or Space."""
        self.assertIn('id="drop-zone"', INDEX_HTML)
        self.assertIn('tabindex="0"', INDEX_HTML)
        self.assertIn('role="button"', INDEX_HTML)
        self.assertIn('aria-controls="video-input"', INDEX_HTML)
        self.assertIn('aria-label=', INDEX_HTML)

        # Verify keydown handler checks for Enter and Space and prevents default
        self.assertIn("e.key === 'Enter'", INDEX_HTML)
        self.assertIn("e.key === ' '", INDEX_HTML)
        self.assertIn("fileInput.click()", INDEX_HTML)
        self.assertIn("e.preventDefault()", INDEX_HTML)

    def test_visible_keyboard_focus_indicators(self) -> None:
        """CSS must define high-contrast :focus-visible outlines."""
        self.assertIn("*:focus-visible", INDEX_HTML)
        self.assertIn("outline:", INDEX_HTML)
        self.assertIn("outline-offset:", INDEX_HTML)
        self.assertIn(".file-input-wrapper:focus-visible", INDEX_HTML)

    def test_touch_target_sizes_min_44px(self) -> None:
        """Interactive controls must comply with touch target size guidelines (>= 44px)."""
        self.assertIn("--touch-min: 44px;", INDEX_HTML)
        self.assertIn("var(--touch-min)", INDEX_HTML)

        # Check button styles reference touch minimum
        self.assertRegex(INDEX_HTML, r"\.btn\s*\{[^}]*min-height:\s*var\(--touch-min\)")
        self.assertRegex(INDEX_HTML, r"\.btn-secondary\s*\{[^}]*min-height:\s*var\(--touch-min\)")
        self.assertRegex(INDEX_HTML, r"\.btn-seek\s*\{[^}]*min-height:\s*var\(--touch-min\)")
        self.assertRegex(INDEX_HTML, r"\.review-nav-link\s*\{[^}]*min-height:\s*var\(--touch-min\)")
        self.assertRegex(INDEX_HTML, r"\.file-label\s*\{[^}]*min-height:\s*var\(--touch-min\)")

    def test_accessible_status_and_error_announcements(self) -> None:
        """Dynamic status changes and alerts must use ARIA live regions and roles."""
        self.assertIn('id="status-badge"', INDEX_HTML)
        self.assertIn('id="job-status"', INDEX_HTML)
        self.assertIn('role="status"', INDEX_HTML)
        self.assertIn('aria-live="polite"', INDEX_HTML)

        # Error regions must have role="alert" and assertive or polite live attributes
        self.assertIn('id="upload-error"', INDEX_HTML)
        self.assertIn('id="job-error-box"', INDEX_HTML)
        self.assertIn('role="alert"', INDEX_HTML)
        self.assertIn('aria-live="assertive"', INDEX_HTML)

        # Results live announcement
        self.assertIn('id="results-announcement"', INDEX_HTML)

    def test_responsive_mobile_viewport_and_layout(self) -> None:
        """HTML must include viewport meta tag, responsive container, and table horizontal scrolling."""
        self.assertIn('<meta name="viewport" content="width=device-width, initial-scale=1.0">', INDEX_HTML)
        self.assertIn("@media (max-width: 640px)", INDEX_HTML)
        self.assertIn("overflow-x: auto", INDEX_HTML)
        self.assertIn("table-responsive", INDEX_HTML)

    def test_preserved_dom_ids(self) -> None:
        """Preserve all DOM IDs and test hooks referenced by existing tests and specs."""
        required_ids = [
            "video-input",
            "file-label",
            "drop-zone",
            "file-name-display",
            "upload-form",
            "submit-btn",
            "upload-error",
            "status-card",
            "status-badge",
            "job-status",
            "job-id-display",
            "job-filename-display",
            "job-error-box",
            "job-warning-box",
            "results-card",
            "metric-reps",
            "metric-attempts",
            "metric-uncounted",
            "metric-frames",
            "incomplete-note-box",
            "incomplete-note-text",
            "session-observations",
            "attempts-table",
            "attempts-tbody",
            "plot-box",
            "angle-plot-img",
            "video-box",
            "annotated-video",
            "preview-player",
            "video-source",
            "video-fallback-notice",
            "download-links",
            "downloads-container",
            "review-link",
        ]
        for dom_id in required_ids:
            self.assertIn(f'id="{dom_id}"', INDEX_HTML, f"Missing required DOM element id='{dom_id}'")

    def test_honest_guidance_and_wording(self) -> None:
        """Maintain honest clinical non-diagnostic disclaimers and raw-review-first guidance."""
        self.assertIn("2D Gym Pose Analyser", INDEX_HTML)
        self.assertIn("Recording Guidance", INDEX_HTML)
        self.assertIn("Uncounted Movements", INDEX_HTML)
        self.assertIn("Human Review Workspace", INDEX_HTML)
        self.assertIn("projected 2D coordinates", INDEX_HTML)
        self.assertIn("do not constitute clinical safety evaluations", INDEX_HTML)

    def test_safe_text_rendering_no_dynamic_innerhtml(self) -> None:
        """Dynamic content must be rendered safely via textContent and DOM APIs without innerHTML."""
        js_match = re.search(r"<script>([\s\S]*?)</script>", INDEX_HTML)
        self.assertIsNotNone(js_match, "INDEX_HTML must contain inline JavaScript")
        js_content = js_match.group(1)
        self.assertNotIn(".innerHTML =", js_content, "JavaScript must not assign to innerHTML")
        self.assertIn(".textContent =", js_content, "JavaScript must use textContent for safe text rendering")


class TestClientValidationAndStateRecoveryNode(unittest.TestCase):
    """Test client-side validation, error recovery, and duplicate submission prevention via Node.js."""

    def setUp(self) -> None:
        self.node_available = shutil.which("node") is not None
        if not self.node_available:
            self.skipTest("Node.js is not available on this machine")

    def _run_node_test(self, js_body: str) -> None:
        """Execute a JavaScript snippet using Node.js against the client-side code extracted from INDEX_HTML."""
        js_match = re.search(r"<script>([\s\S]*?)</script>", INDEX_HTML)
        self.assertIsNotNone(js_match)
        client_code = js_match.group(1)

        test_runner = f"""
const assert = require('assert');
const vm = require('vm');

const elements = {{}};
function getMockElement(id) {{
  if (!elements[id]) {{
    elements[id] = {{
      id,
      tagName: (id.includes('label') ? 'LABEL' : (id.includes('btn') ? 'BUTTON' : (id.includes('input') ? 'INPUT' : 'DIV'))),
      value: '',
      disabled: false,
      textContent: '',
      className: '',
      style: {{}},
      files: [],
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
      click() {{ if (this.on_click) this.on_click({{ target: this, preventDefault() {{}}, stopPropagation() {{}} }}); }},
      appendChild(c) {{}},
      querySelectorAll() {{ return []; }},
      load() {{}},
      closest(selector) {{
        let el = this;
        while (el) {{
          if (selector === 'label[for="video-input"]') {{
            if ((el.tagName === 'LABEL' || el.id === 'file-label') && (el.getAttribute('for') === 'video-input' || el['for'] === 'video-input')) {{
              return el;
            }}
          }} else if (selector === 'label[for]') {{
            if ((el.tagName === 'LABEL' || el.id === 'file-label') && (el.getAttribute('for') || el['for'])) {{
              return el;
            }}
          }} else if (selector === 'label') {{
            if (el.tagName === 'LABEL' || el.id === 'file-label') return el;
          }} else if (selector === '#drop-zone' || selector === '.file-input-wrapper') {{
            if (el.id === 'drop-zone') return el;
          }}
          el = el.parentElement;
        }}
        return null;
      }},
    }};
  }}
  return elements[id];
}}

// Initialize and link drop-zone DOM hierarchy
const mockDropZone = getMockElement('drop-zone');
mockDropZone.tagName = 'DIV';
const mockFileInput = getMockElement('video-input');
mockFileInput.tagName = 'INPUT';
mockFileInput.parentElement = mockDropZone;
const mockFileLabel = getMockElement('file-label');
mockFileLabel.tagName = 'LABEL';
mockFileLabel.setAttribute('for', 'video-input');
mockFileLabel.parentElement = mockDropZone;
const mockFileNameDisplay = getMockElement('file-name-display');
mockFileNameDisplay.tagName = 'DIV';
mockFileNameDisplay.parentElement = mockDropZone;

const domDoc = {{
  getElementById: getMockElement,
  createElement: (tag) => getMockElement('temp_' + Math.random()),
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
  module: {{ exports: {{}} }},
}};

vm.createContext(context);
vm.runInContext({json.dumps(client_code)}, context);

(async () => {{
{js_body}
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
                f"Node.js validation failed (code {res.returncode}):\nSTDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}",
            )
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    def test_file_size_formatting(self) -> None:
        """formatFileSize helper accurately formats bytes into readable units."""
        js = """
const { formatFileSize } = context.module.exports;
assert.strictEqual(formatFileSize(0), '0 B');
assert.strictEqual(formatFileSize(500), '500 B');
assert.strictEqual(formatFileSize(1024), '1.0 KB');
assert.strictEqual(formatFileSize(1048576), '1.00 MB');
assert.strictEqual(formatFileSize(200 * 1024 * 1024), '200.00 MB');
assert.strictEqual(formatFileSize(-10), '0 B');
assert.strictEqual(formatFileSize(NaN), '0 B');
"""
        self._run_node_test(js)

    def test_client_validation_file_constraints(self) -> None:
        """Client validation matches server upload rules (<= 200MB, allowed video extensions)."""
        js = """
const { validateClientFile, MAX_UPLOAD_BYTES, ALLOWED_EXTENSIONS } = context.module.exports;

assert.strictEqual(MAX_UPLOAD_BYTES, 200 * 1024 * 1024);
const allowedList = Array.from(ALLOWED_EXTENSIONS).sort();
assert.deepStrictEqual(allowedList, ['.avi', '.m4v', '.mkv', '.mov', '.mp4', '.webm']);

// 1. Missing file
const resNull = validateClientFile(null);
assert.strictEqual(resNull.valid, false);

// 2. Empty file (0 bytes)
const resEmpty = validateClientFile({ name: 'test.mp4', size: 0 });
assert.strictEqual(resEmpty.valid, false);
assert(resEmpty.error.includes('empty'));

// 3. Oversize file (> 200MB)
const resOver = validateClientFile({ name: 'test.mp4', size: MAX_UPLOAD_BYTES + 1 });
assert.strictEqual(resOver.valid, false);
assert(resOver.error.includes('exceeds maximum upload limit'));

// 4. Disallowed extensions
const resBadExt = validateClientFile({ name: 'malicious.exe', size: 1024 });
assert.strictEqual(resBadExt.valid, false);
assert(resBadExt.error.includes('Unsupported video format'));

const resText = validateClientFile({ name: 'notes.txt', size: 1024 });
assert.strictEqual(resText.valid, false);

// 5. Valid files
for (const ext of allowedList) {
  const resValid = validateClientFile({ name: 'sample' + ext, size: 5 * 1024 * 1024 });
  assert.strictEqual(resValid.valid, true);
  assert.strictEqual(resValid.error, null);
}
"""
        self._run_node_test(js)

    def test_duplicate_submission_prevention_logic(self) -> None:
        """Form submission must disable the submit button and prevent concurrent submissions."""
        js = """
const submitBtn = context.document.getElementById('submit-btn');
const uploadForm = context.document.getElementById('upload-form');
const videoInput = context.document.getElementById('video-input');

let fetchCalls = 0;
let resolveFetch;
const fetchPromise = new Promise((resolve) => {
  resolveFetch = resolve;
});

context.fetch = (url, opts) => {
  fetchCalls++;
  return fetchPromise;
};

// Stage a valid file
const validFile = { name: 'good.mp4', size: 1024 * 1024 };
videoInput.files = [validFile];

// Trigger first submission
const sub1 = uploadForm.on_submit({ preventDefault() {} });

assert.strictEqual(submitBtn.disabled, true, 'Submit button should be disabled upon submission');
assert.strictEqual(fetchCalls, 1, 'First submission must invoke fetch');

// Trigger immediate second submission while in-flight
const sub2 = uploadForm.on_submit({ preventDefault() {} });
assert.strictEqual(fetchCalls, 1, 'Concurrent submission while in-flight must be ignored');

// Complete the in-flight request
resolveFetch({
  ok: true,
  json: async () => ({ job_id: '11111111-2222-3333-4444-555555555555', filename: 'good.mp4' })
});

await sub1;
await sub2;

// Stop polling interval so node event loop exits cleanly
context.module.exports.stopPolling();
"""
        self._run_node_test(js)

    def test_error_state_recovery_re_enables_submit(self) -> None:
        """Controls are re-enabled after 429 busy server, 503 unavailable, network error, or invalid file."""
        js = """
const submitBtn = context.document.getElementById('submit-btn');
const uploadForm = context.document.getElementById('upload-form');
const uploadError = context.document.getElementById('upload-error');
const videoInput = context.document.getElementById('video-input');

async function testRecovery(fetchMock, expectedErrorSubstring) {
  context.fetch = fetchMock;
  videoInput.files = [{ name: 'test.mp4', size: 1024 * 1024 }];
  await uploadForm.on_submit({ preventDefault() {} });
  
  assert.strictEqual(submitBtn.disabled, false, 'Submit button must be re-enabled on error');
  assert.strictEqual(submitBtn.textContent, 'Analyze Video', 'Submit button label must be reset');
  assert.strictEqual(uploadError.classList.contains('hidden'), false, 'Error box must be visible');
  assert(
    uploadError.textContent.toLowerCase().includes(expectedErrorSubstring.toLowerCase()),
    `Expected '${expectedErrorSubstring}' in '${uploadError.textContent}'`
  );
}

// 1. Busy server (HTTP 429)
await testRecovery(
  async () => ({ status: 429, ok: false, json: async () => ({ detail: 'Server is busy processing another job' }) }),
  'busy'
);

// 2. Service unavailable (HTTP 503)
await testRecovery(
  async () => ({ status: 503, ok: false, json: async () => ({ detail: 'Service temporarily unavailable' }) }),
  'unavailable'
);

// 3. Network error / connection loss
await testRecovery(
  async () => { throw new Error('Network connection failed'); },
  'Network'
);
"""
        self._run_node_test(js)

    def test_label_activation_and_drop_zone_click_simulation(self) -> None:
        """Verify label default-action simulation triggers exactly 1 activation, and dropzone outside label triggers 1 activation."""
        js = """
const fileInput = context.document.getElementById('video-input');
const dropZone = context.document.getElementById('drop-zone');
const fileLabel = context.document.getElementById('file-label');
const fileNameDisplay = context.document.getElementById('file-name-display');

let activationCount = 0;
// Track activation calls to fileInput.click
fileInput.click = () => {
  activationCount++;
};

// Simulate browser event dispatch: native label activation and event bubbling
function simulateBrowserClick(target) {
  // 1. Browser native label default activation behavior:
  // When a <label for="id"> is clicked, browser natively triggers click on associated control
  const label = target.closest ? target.closest('label[for="video-input"]') : null;
  if (label && (label.getAttribute('for') === 'video-input' || label['for'] === 'video-input')) {
    fileInput.click();
  }

  // 2. Click event bubbles up the DOM tree through ancestors
  let curr = target;
  const evt = {
    target: target,
    currentTarget: null,
    defaultPrevented: false,
    preventDefault() { this.defaultPrevented = true; },
    stopPropagation() { this._stopped = true; },
    _stopped: false,
  };
  while (curr) {
    evt.currentTarget = curr;
    if (curr.on_click) {
      curr.on_click(evt);
    }
    if (evt._stopped) break;
    curr = curr.parentElement;
  }
}

// 1. Click on the label element:
// Native label-for association activates file input, but dropZone click handler ignores it (no duplicate activation).
activationCount = 0;
simulateBrowserClick(fileLabel);
assert.strictEqual(
  activationCount,
  1,
  `Clicking <label for="video-input"> must trigger exactly 1 file input activation, got ${activationCount}`
);

// 2. Click on dropZone outside the label:
// No native label activation; dropZone click listener activates fileInput.
activationCount = 0;
simulateBrowserClick(dropZone);
assert.strictEqual(
  activationCount,
  1,
  `Clicking drop-zone outside label must trigger exactly 1 file input activation, got ${activationCount}`
);

// 3. Click on filename display text inside dropZone (outside label):
// Bubbles to dropZone; dropZone click listener activates fileInput.
activationCount = 0;
simulateBrowserClick(fileNameDisplay);
assert.strictEqual(
  activationCount,
  1,
  `Clicking file-name-display must trigger exactly 1 file input activation, got ${activationCount}`
);

// 4. Keyboard activation via Enter or Space keydown on dropZone
activationCount = 0;
let defaultPrevented = false;
dropZone.on_keydown({
  key: 'Enter',
  preventDefault() { defaultPrevented = true; }
});
assert.strictEqual(defaultPrevented, true, 'Enter keydown must preventDefault');
assert.strictEqual(
  activationCount,
  1,
  `Pressing Enter on drop-zone must trigger exactly 1 file input activation, got ${activationCount}`
);

activationCount = 0;
defaultPrevented = false;
dropZone.on_keydown({
  key: ' ',
  preventDefault() { defaultPrevented = true; }
});
assert.strictEqual(defaultPrevented, true, 'Space keydown must preventDefault');
assert.strictEqual(
  activationCount,
  1,
  `Pressing Space on drop-zone must trigger exactly 1 file input activation, got ${activationCount}`
);
"""
        self._run_node_test(js)


@unittest.skipUnless(HAS_WEB_DEPS, "FastAPI or web dependencies not installed")
class TestWebEndpointsIntegration(unittest.TestCase):
    """Verify backend FastAPI routes, download allowlists, and review mount."""

    def setUp(self) -> None:
        if not HAS_WEB_DEPS:
            self.skipTest("FastAPI or web dependencies not installed")
        self.temp_dir = tempfile.mkdtemp(prefix="test_web_usability_")
        self.storage_dir = Path(self.temp_dir) / "web_jobs"
        self.review_dir = Path(self.temp_dir) / "review_store"
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.review_dir.mkdir(parents=True, exist_ok=True)

        self.manager = JobManager(storage_dir=self.storage_dir, max_upload_bytes=DEFAULT_MAX_UPLOAD_BYTES)
        self.app = create_app(
            job_manager=self.manager,
            enable_review=True,
            review_storage_root=self.review_dir,
        )
        self.client = TestClient(self.app)

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_get_root_serves_html(self) -> None:
        """GET / returns 200 with HTML content-type and page headings."""
        res = self.client.get("/")
        self.assertEqual(res.status_code, 200)
        self.assertIn("text/html", res.headers.get("content-type", ""))
        self.assertIn("2D Gym Pose Analyser", res.text)
        self.assertIn("Upload Squat Video", res.text)

    def test_get_server_status_idle(self) -> None:
        """GET /api/status returns 200 with busy: False and active_job_id: None."""
        res = self.client.get("/api/status")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertFalse(data.get("busy"))
        self.assertIsNone(data.get("active_job_id"))

    def test_submit_job_accepted(self) -> None:
        """POST /api/jobs accepts valid uploads and returns 202."""
        dummy_video = b"\x00\x00\x00\x18ftypmp42\x00\x00\x00\x00mp42isom"

        res = self.client.post(
            "/api/jobs",
            files={"file": ("test_clip1.mp4", io.BytesIO(dummy_video), "video/mp4")},
        )
        self.assertEqual(res.status_code, 202)
        data = res.json()
        self.assertTrue(uuid.UUID(data["job_id"]))
        self.assertEqual(data["status"], "queued")

    def test_submit_job_validation_rejections(self) -> None:
        """Server rejects empty files and disallowed extensions with 400 Bad Request."""
        # Empty file
        res_empty = self.client.post(
            "/api/jobs",
            files={"file": ("empty.mp4", io.BytesIO(b""), "video/mp4")},
        )
        self.assertEqual(res_empty.status_code, 400)
        self.assertIn("empty", res_empty.json()["detail"].lower())

        # Disallowed extension
        res_bad_ext = self.client.post(
            "/api/jobs",
            files={"file": ("script.sh", io.BytesIO(b"echo hi"), "text/plain")},
        )
        self.assertEqual(res_bad_ext.status_code, 400)
        self.assertIn("unsupported video format", res_bad_ext.json()["detail"].lower())

    def test_get_job_status(self) -> None:
        """Job status retrieval works via GET /api/jobs/{id}."""
        job_id, _, _ = self.manager.save_upload_file(io.BytesIO(b"dummy data"), "test.mp4")

        # GET /api/jobs/{id}
        res1 = self.client.get(f"/api/jobs/{job_id}")
        self.assertEqual(res1.status_code, 200)
        self.assertEqual(res1.json()["job_id"], job_id)

        # Invalid UUID
        self.assertEqual(self.client.get("/api/jobs/not-a-uuid").status_code, 400)

        # Nonexistent UUID
        non_existent = str(uuid.uuid4())
        self.assertEqual(self.client.get(f"/api/jobs/{non_existent}").status_code, 404)

    def test_download_artifact_allowlist_and_traversal(self) -> None:
        """Artifact downloads adhere to strict allowlist and reject path traversal."""
        job_id, _, _ = self.manager.save_upload_file(io.BytesIO(b"dummy data"), "test.mp4")
        results_dir = self.storage_dir / job_id / "results"
        results_dir.mkdir(parents=True, exist_ok=True)
        summary_file = results_dir / "summary.json"
        summary_file.write_text(json.dumps({"status": STATUS_COMPLETED}), encoding="utf-8")

        # Allowed artifact
        res = self.client.get(f"/api/jobs/{job_id}/download/summary.json")
        self.assertEqual(res.status_code, 200)
        self.assertIn("application/json", res.headers.get("content-type", ""))

        # Disallowed artifact outside allowlist
        res_disallowed = self.client.get(f"/api/jobs/{job_id}/download/secret_key.pem")
        self.assertEqual(res_disallowed.status_code, 404)

        # Path traversal
        res_traversal = self.client.get(f"/api/jobs/{job_id}/download/..%2Fsummary.json")
        self.assertIn(res_traversal.status_code, (400, 404))

    def test_review_mount_integration(self) -> None:
        """Human review workspace is mounted at /review/ and toggleable via enable_review."""
        # With review enabled
        res_mounted = self.client.get("/review/")
        self.assertEqual(res_mounted.status_code, 200)
        self.assertIn("text/html", res_mounted.headers.get("content-type", ""))

        # With review disabled
        app_disabled = create_app(
            job_manager=self.manager,
            enable_review=False,
        )
        client_disabled = TestClient(app_disabled)
        self.assertEqual(client_disabled.get("/review/").status_code, 404)
        self.assertNotIn('href="/review/"', client_disabled.get("/").text)


if __name__ == "__main__":
    unittest.main()
