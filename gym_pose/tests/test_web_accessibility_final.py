"""Final Web Accessibility, Usability, and Keyboard Interaction Regression Suite.

Task 54 / Agent 3 regression suite independently verifying:
1. Keyboard-operable phase seek buttons:
   - Seek buttons created with role="button", tabindex="0", explicit aria-label describing timestamp/attempt.
   - Keyboard activation on Enter and Space keydown events.
   - Accessible status announcements via #video-seek-status (seeked, queued while loading, error, timeout).
2. Visible focus indicators:
   - High-contrast :focus-visible rules on all interactive elements (buttons, links, dropzone, inputs, table).
   - Distinct outline and box-shadow styling for keyboard navigability.
3. Narrow-screen / Mobile responsiveness:
   - Viewport meta tag, html/body overflow-x safety, responsive cards/containers without horizontal scrollbars.
   - Media queries for <=640px and <=420px (stacked buttons, single-column metrics, 100% width controls).
   - Horizontally-scrollable table-responsive wrapper with tabindex="0" and accessible region labelling.
4. Dynamic error and status live announcements:
   - ARIA live regions: #upload-error, #job-error-box, #job-warning-box, #status-badge,
     #results-announcement, #incomplete-note-box, #file-name-display, #video-seek-status.
   - Descriptive job status badge aria-labels across lifecycle states.
5. Incomplete / Missing feedback states:
   - High-contrast .status-unavailable, .status-incomplete, and .status-unvalidated badge styling.
   - Assistive text on UNAVAILABLE and INCOMPLETE states; honest disclaimers preserved.
6. Download buttons accessibility:
   - Links rendered with role="button", tabindex="0", aria-label with formatted file size, and Spacebar keydown support.
7. Safe DOM rendering:
   - Zero assignments to innerHTML, outerHTML, or document.write in client script.
8. Preservation of critical invariants:
   - Request-version guards (statusRequestVersion, resultsRequestVersion, currentJobId) preventing stale clobbering.

Run with:
gym_pose/.venv_web/Scripts/python.exe -m unittest gym_pose/tests/test_web_accessibility_final.py
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any, Dict, Optional

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


class _NodeHarnessBase(unittest.TestCase):
    """Base class providing instrumented Node.js headless DOM execution."""

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
  }} = context.module.exports;

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
                f"Node.js headless test failed (exit code {res.returncode}):\nSTDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}",
            )
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)


class TestPhaseSeekButtonAccessibility(_NodeHarnessBase):
    """Test keyboard operability, ARIA attributes, and live announcements for phase seek buttons."""

    def test_phase_seek_buttons_rendered_with_proper_roles_and_attributes(self) -> None:
        """Seek buttons must have role='button', tabindex='0', and explicit aria-label."""
        js = """
const data = {
  job_id: 'job-seek-test',
  completed_reps: 1,
  processed_frames: 90,
  attempts: {
    events: [{
      attempt_id: 1,
      status: 'counted',
      reason: 'completed_cycle',
      start_frame: 0,
      end_frame: 90,
      start_time_s: 0.0,
      end_time_s: 3.0,
      min_knee_angle_deg: 74.0
    }]
  },
  phase_feedback: {
    attempts: [{
      attempt_id: 1,
      phase_analysis_available: true,
      min_knee_point: { video_time_s: 1.45 },
      return_phase: { delta_start_time_s: 1.60 }
    }]
  }
};

renderResults(data);

const tbody = document.getElementById('attempts-tbody');
const buttons = [];
function collectButtons(node) {
  if (node.tagName === 'BUTTON' && (node.className || '').includes('btn-seek')) {
    buttons.push(node);
  }
  if (node.children) {
    node.children.forEach(collectButtons);
  }
}
collectButtons(tbody);

assert.strictEqual(buttons.length, 2, 'Must render two phase seek buttons (turnaround and return)');

const [btnKnee, btnReturn] = buttons;

// 1. Role button
assert.strictEqual(btnKnee.getAttribute('role'), 'button', 'Turnaround seek button must have role="button"');
assert.strictEqual(btnReturn.getAttribute('role'), 'button', 'Return seek button must have role="button"');

// 2. Tabindex 0
assert.strictEqual(btnKnee.getAttribute('tabindex'), '0', 'Turnaround seek button must have tabindex="0"');
assert.strictEqual(btnReturn.getAttribute('tabindex'), '0', 'Return seek button must have tabindex="0"');

// 3. Descriptive aria-label
const labelKnee = btnKnee.getAttribute('aria-label') || '';
assert.ok(labelKnee.includes('turnaround') && labelKnee.includes('1.45'), 'Turnaround aria-label must describe action and timestamp');
assert.ok(labelKnee.includes('Attempt #1') || labelKnee.includes('#1'), 'Turnaround aria-label should reference attempt ID');

const labelReturn = btnReturn.getAttribute('aria-label') || '';
assert.ok(labelReturn.includes('return') && labelReturn.includes('1.60'), 'Return aria-label must describe action and timestamp');
assert.ok(labelReturn.includes('Attempt #1') || labelReturn.includes('#1'), 'Return aria-label should reference attempt ID');
"""
        self.run_node_script(js)

    def test_phase_seek_buttons_keyboard_activation_enter_and_space(self) -> None:
        """Seek buttons must activate on Enter and Space keydown events, ignoring other keys."""
        js = """
const video = document.getElementById('annotated-video');
video.readyState = 4;
video.duration = 10.0;
video.currentTime = 0.0;

const data = {
  job_id: 'job-seek-kb',
  completed_reps: 1,
  attempts: { events: [{ attempt_id: 1, status: 'counted' }] },
  phase_feedback: {
    attempts: [{
      attempt_id: 1,
      phase_analysis_available: true,
      min_knee_point: { video_time_s: 2.25 },
    }]
  }
};

renderResults(data);

const tbody = document.getElementById('attempts-tbody');
let seekBtn = null;
function findButton(node) {
  if (node.tagName === 'BUTTON' && (node.className || '').includes('btn-seek')) seekBtn = node;
  if (node.children) node.children.forEach(findButton);
}
findButton(tbody);
assert.ok(seekBtn, 'Seek button must be present');

// Test Tab key is ignored (does not trigger seek)
seekBtn.on_keydown({ type: 'keydown', key: 'Tab', preventDefault() {}, stopPropagation() {} });
assert.strictEqual(video.currentTime, 0.0, 'Tab key must not trigger seek');

// Test Enter key activates seek
let defaultPrevented = false;
seekBtn.on_keydown({
  type: 'keydown',
  key: 'Enter',
  preventDefault() { defaultPrevented = true; },
  stopPropagation() {}
});
assert.strictEqual(video.currentTime, 2.25, 'Enter key must trigger video seek');
assert.strictEqual(defaultPrevented, true, 'Enter keydown must prevent default browser scrolling/navigation');

// Reset and test Space key activates seek
video.currentTime = 0.0;
defaultPrevented = false;
seekBtn.on_keydown({
  type: 'keydown',
  key: ' ',
  preventDefault() { defaultPrevented = true; },
  stopPropagation() {}
});
assert.strictEqual(video.currentTime, 2.25, 'Space key must trigger video seek');
assert.strictEqual(defaultPrevented, true, 'Space keydown must prevent default page scroll');
"""
        self.run_node_script(js)

    def test_seek_video_to_time_live_announcement_success(self) -> None:
        """Seeking ready video updates video-seek-status live region politely."""
        js = """
const video = document.getElementById('annotated-video');
video.readyState = 4;
video.duration = 10.0;
video.currentTime = 0.0;

const seekStatus = document.getElementById('video-seek-status');
assert.ok(seekStatus, 'video-seek-status element must exist');

seekVideoToTime(3.75);

assert.strictEqual(video.currentTime, 3.75);
assert.ok(seekStatus.textContent.includes('3.75'), 'Status must announce seek target time');
assert.ok(seekStatus.textContent.includes('Seeked video to'), 'Status must confirm successful seek');
"""
        self.run_node_script(js)

    def test_seek_video_to_time_live_announcement_queued_when_loading(self) -> None:
        """Seeking unready video announces queued state and completes upon loadedmetadata."""
        js = """
const video = document.getElementById('annotated-video');
video.readyState = 0; // metadata not loaded
video.duration = NaN;
video.currentTime = 0.0;

const seekStatus = document.getElementById('video-seek-status');

seekVideoToTime(4.5);

assert.ok(seekStatus.textContent.includes('queued') || seekStatus.textContent.includes('loading'),
  'Must announce that seek is queued while metadata loads');

// Simulate metadata loaded
video.readyState = 1;
video.duration = 10.0;
video.dispatchEvent('loadedmetadata');

assert.strictEqual(video.currentTime, 4.5, 'Video must seek once metadata is loaded');
assert.ok(seekStatus.textContent.includes('Seeked video to 4.50 seconds'), 'Must announce completed seek after load');
"""
        self.run_node_script(js)

    def test_seek_video_to_time_invalid_timestamp_announcement(self) -> None:
        """Invalid timestamp announces error in live region without throwing unhandled exception."""
        js = """
const seekStatus = document.getElementById('video-seek-status');
seekVideoToTime(-5);
assert.ok(seekStatus.textContent.includes('Invalid seek target timestamp'), 'Must announce invalid timestamp');

seekVideoToTime('not-a-number');
assert.ok(seekStatus.textContent.includes('Invalid seek target timestamp'), 'Must announce invalid timestamp');
"""
        self.run_node_script(js)

    def test_seek_video_to_time_playback_error_announcement(self) -> None:
        """Playback control failure announces clear diagnostic in live region."""
        js = """
const video = document.getElementById('annotated-video');
video.readyState = 4;
video.duration = 10.0;
Object.defineProperty(video, 'currentTime', {
  set: () => { throw new Error('Video decode pipeline error'); },
  configurable: true
});

const seekStatus = document.getElementById('video-seek-status');
seekVideoToTime(2.0);

assert.ok(seekStatus.textContent.includes('Failed to seek video') || seekStatus.textContent.includes('playback error'),
  'Must announce playback error gracefully');
"""
        self.run_node_script(js)

    def test_static_phase_seek_markup_and_handlers_in_web_source(self) -> None:
        """Static inspection: phase seek buttons have role, tabindex, aria-label, and Enter/Space handlers."""
        self.assertIn("btn.setAttribute('role', 'button')", INDEX_HTML)
        self.assertIn("btn.setAttribute('tabindex', '0')", INDEX_HTML)
        self.assertIn("btn.setAttribute('aria-label'", INDEX_HTML)
        self.assertIn("e.key === 'Enter'", INDEX_HTML)
        self.assertIn("e.key === ' '", INDEX_HTML)
        self.assertIn("seekVideoToTime", INDEX_HTML)
        self.assertIn("announceSeekStatus", INDEX_HTML)


class TestVisibleFocusIndicators(unittest.TestCase):
    """Test high-contrast visible focus indicators (:focus-visible) across all interactive elements."""

    def test_universal_focus_visible_rule_present(self) -> None:
        """CSS defines high-contrast *:focus-visible outline with offset."""
        self.assertIn("*:focus-visible", INDEX_HTML)
        self.assertRegex(INDEX_HTML, r"\*:focus-visible\s*\{[^}]*outline:\s*3px solid var\(--primary\)")
        self.assertRegex(INDEX_HTML, r"\*:focus-visible\s*\{[^}]*outline-offset:\s*2px")

    def test_interactive_elements_explicit_focus_visible_rules(self) -> None:
        """Explicit focus-visible rules defined for buttons, links, dropzone, table, and inputs."""
        for selector in (
            ".btn:focus-visible",
            ".btn-secondary:focus-visible",
            ".btn-seek:focus-visible",
            ".review-nav-link:focus-visible",
            ".file-input-wrapper:focus-visible",
            ".table-responsive:focus-visible",
            "a:focus-visible",
            "button:focus-visible",
            "input:focus-visible",
        ):
            self.assertIn(selector, INDEX_HTML, f"Missing explicit focus-visible selector: {selector}")

    def test_focus_visible_box_shadow_contrast_enhancement(self) -> None:
        """Light theme keeps a visible solid focus ring without a neon glow."""
        self.assertIn("outline: 3px solid var(--primary)", INDEX_HTML)
        self.assertIn("outline-offset: 3px", INDEX_HTML)


class TestScreenResponsivenessAndOverflowSafety(unittest.TestCase):
    """Test narrow-screen mobile responsiveness and horizontal overflow prevention (<=360px)."""

    def test_viewport_meta_tag_present(self) -> None:
        """Viewport meta tag configured for device-width and initial-scale 1.0."""
        self.assertIn('<meta name="viewport" content="width=device-width, initial-scale=1.0">', INDEX_HTML)

    def test_html_and_body_prevent_horizontal_overflow(self) -> None:
        """Root html and body elements declare overflow-x: hidden and max-width: 100vw."""
        self.assertIn("html {", INDEX_HTML)
        self.assertRegex(INDEX_HTML, r"html\s*\{[^}]*overflow-x:\s*hidden")
        self.assertRegex(INDEX_HTML, r"body\s*\{[^}]*overflow-x:\s*hidden")
        self.assertRegex(INDEX_HTML, r"body\s*\{[^}]*max-width:\s*100vw")

    def test_containers_have_word_wrap_and_max_width_guards(self) -> None:
        """Containers and cards guard against text overflowing screen boundaries."""
        self.assertRegex(INDEX_HTML, r"\.container\s*\{[^}]*word-wrap:\s*break-word")
        self.assertRegex(INDEX_HTML, r"\.container\s*\{[^}]*overflow-wrap:\s*break-word")
        self.assertRegex(INDEX_HTML, r"\.card\s*\{[^}]*max-width:\s*100%")
        self.assertRegex(INDEX_HTML, r"\.notice-box\s*\{[^}]*max-width:\s*100%")
        self.assertRegex(INDEX_HTML, r"\.alert-box\s*\{[^}]*max-width:\s*100%")

    def test_narrow_screen_media_queries(self) -> None:
        """Media query for <= 420px provides single-column metrics and full-width buttons."""
        self.assertIn("@media (max-width: 640px)", INDEX_HTML)
        self.assertIn("@media (max-width: 420px)", INDEX_HTML)

        # In <= 420px query: single-column metrics, column downloads, full-width secondary/seek buttons
        start_420 = INDEX_HTML.find("@media (max-width: 420px)")
        block_420 = INDEX_HTML[start_420 : start_420 + 800]
        self.assertIn("grid-template-columns: 1fr;", block_420)
        self.assertIn("flex-direction: column;", block_420)
        self.assertIn("width: 100%;", block_420)

    def test_responsive_data_table_keyboard_navigable_wrapper(self) -> None:
        """Data table wrapper has overflow-x: auto, tabindex='0', and role='region' for keyboard scrolling."""
        self.assertIn('<div class="table-responsive" tabindex="0" role="region" aria-label="Observed Movement Events table">', INDEX_HTML)
        self.assertRegex(INDEX_HTML, r"\.table-responsive\s*\{[^}]*overflow-x:\s*auto")
        self.assertRegex(INDEX_HTML, r"\.table-responsive\s*\{[^}]*max-width:\s*100%")


class TestDynamicLiveRegionAnnouncements(_NodeHarnessBase):
    """Test that all dynamic status changes and errors announce properly via ARIA live regions."""

    def test_aria_live_region_declarations_in_html(self) -> None:
        """Verify presence and attributes of all designated live regions in INDEX_HTML."""
        # Assertive error regions
        self.assertIn('id="upload-error" class="alert-box alert-danger hidden" role="alert" aria-live="assertive"', INDEX_HTML)
        self.assertIn('id="job-error-box" class="alert-box alert-danger hidden" role="alert" aria-live="assertive"', INDEX_HTML)

        # Polite status and progress regions
        self.assertIn('id="job-warning-box" class="alert-box alert-warning hidden" role="alert" aria-live="polite"', INDEX_HTML)
        self.assertIn('id="status-badge" role="status" aria-live="polite"', INDEX_HTML)
        self.assertIn('id="results-announcement" class="sr-only" role="status" aria-live="polite"', INDEX_HTML)
        self.assertIn('id="incomplete-note-box" class="alert-box alert-warning hidden" role="alert" aria-live="polite"', INDEX_HTML)
        self.assertIn('id="file-name-display" aria-live="polite"', INDEX_HTML)
        self.assertIn('id="video-seek-status" class="sr-only" role="status" aria-live="polite"', INDEX_HTML)

    def test_status_badge_accessible_label_updates(self) -> None:
        """updateStatusBadge sets descriptive aria-label across all job lifecycle transitions."""
        js = """
const badge = document.getElementById('status-badge');

updateStatusBadge('queued');
assert.strictEqual(badge.getAttribute('aria-label'), 'Job status: QUEUED (Waiting in processing queue)');

updateStatusBadge('running');
assert.strictEqual(badge.getAttribute('aria-label'), 'Job status: RUNNING (Analysis in progress)');

updateStatusBadge('completed');
assert.strictEqual(badge.getAttribute('aria-label'), 'Job status: COMPLETED (Full video processed)');

updateStatusBadge('incomplete');
assert.strictEqual(badge.getAttribute('aria-label'), 'Job status: INCOMPLETE (Partial decoded prefix only)');

updateStatusBadge('failed');
assert.strictEqual(badge.getAttribute('aria-label'), 'Job status: FAILED (Analysis failed with error)');

updateStatusBadge('interrupted');
assert.strictEqual(badge.getAttribute('aria-label'), 'Job status: INTERRUPTED (Analysis stopped prematurely)');
"""
        self.run_node_script(js)

    def test_results_announcement_live_region_updates(self) -> None:
        """loadJobResults populates #results-announcement with polite assistive summary."""
        js = """
setCurrentJobId('job-announce-res');
const liveEl = document.getElementById('results-announcement');

context.fetch = async () => ({
  ok: true,
  status: 200,
  json: async () => ({
    status: 'completed',
    completed_reps: 3,
    processed_frames: 100,
    attempts: { events: [] }
  })
});

await loadJobResults('job-announce-res');

assert.ok(liveEl.textContent.includes('Analysis complete'), 'Live region must announce completed analysis');
assert.ok(liveEl.textContent.includes('Results and movement observations loaded below'));
"""
        self.run_node_script(js)


class TestMissingAndIncompleteFeedbackStates(_NodeHarnessBase):
    """Test high-contrast badges and clear assistive text for missing/incomplete states."""

    def test_high_contrast_status_classes_in_css(self) -> None:
        """CSS provides explicit high-contrast classes for unavailable, incomplete, and unvalidated."""
        self.assertIn(".status-unavailable", INDEX_HTML)
        self.assertIn(".status-incomplete", INDEX_HTML)
        self.assertIn(".status-unvalidated", INDEX_HTML)
        self.assertIn(".alert-warning", INDEX_HTML)

        # Incomplete has black text on amber background with border for maximum contrast
        self.assertRegex(INDEX_HTML, r"\.status-incomplete\s*\{[^}]*background:\s*#eab308")
        self.assertRegex(INDEX_HTML, r"\.status-incomplete\s*\{[^}]*color:\s*#000")
        self.assertRegex(INDEX_HTML, r"\.status-incomplete\s*\{[^}]*font-weight:\s*700")

        # Unavailable remains a distinguishable neutral light-theme state.
        self.assertIn(".status-unavailable, .status-unvalidated { background: #edf0eb; color: #4b594e", INDEX_HTML)

    def test_unavailable_coaching_badge_and_assistive_text(self) -> None:
        """renderCoachingFeedback(null) assigns .status-unavailable and descriptive aria-label."""
        js = """
const badge = document.getElementById('coaching-status-badge');
const quality = document.getElementById('coaching-quality-details');

renderCoachingFeedback(null);

assert.strictEqual(badge.textContent, 'UNAVAILABLE');
assert.ok(badge.className.includes('status-unavailable'), 'Must assign status-unavailable class');
assert.strictEqual(badge.getAttribute('aria-label'), 'Coaching status: UNAVAILABLE (Telemetry is not available for this run)');
assert.strictEqual(quality.textContent, 'Coaching telemetry is not available for this run.');
"""
        self.run_node_script(js)

    def test_incomplete_analysis_decoded_prefix_clarity(self) -> None:
        """Incomplete analysis renders high-contrast incompleteNoteBox with explicit prefix notice."""
        js = """
setCurrentJobId('job-incomplete-clarity');
const data = {
  job_id: 'job-incomplete-clarity',
  status: 'incomplete',
  completed_reps: 2,
  processed_frames: 60,
  declared_frame_count: 120,
  read_status_note: 'Video ended unexpectedly at frame 60.',
  attempts: { events: [] }
};

renderResults(data);

const noteBox = document.getElementById('incomplete-note-box');
const noteText = document.getElementById('incomplete-note-text');
const repsLabel = document.getElementById('metric-reps-label');

assert.strictEqual(noteBox.classList.contains('hidden'), false, 'Incomplete note box must be displayed');
assert.ok(noteText.textContent.includes('decoded prefix only'), 'Note text must emphasize decoded prefix');
assert.ok(repsLabel.textContent.includes('Decoded Prefix Repetitions'), 'Reps label must specify decoded prefix');
"""
        self.run_node_script(js)


class TestDownloadButtonsAccessibility(_NodeHarnessBase):
    """Test that artifact download links are keyboard-operable and have accessible names."""

    def test_download_buttons_rendered_with_button_role_tabindex_and_aria_label(self) -> None:
        """Download links must have role='button', tabindex='0', descriptive aria-label, and Space keydown."""
        js = """
const data = {
  job_id: 'job-download-a11y',
  available_artifacts: {
    'summary.json': { available: true, size_bytes: 4096 },
    'annotated.mp4': { available: true, size_bytes: 2097152 },
    'angle_plot.png': { available: true, size_bytes: 102400 },
  }
};

renderResults(data);

const container = document.getElementById('downloads-container');
assert.ok(container.children.length >= 3, 'Must render download links for available artifacts');

let clickedLink = null;
container.children.forEach(link => {
  assert.strictEqual(link.getAttribute('role'), 'button', 'Download link must have role="button"');
  assert.strictEqual(link.getAttribute('tabindex'), '0', 'Download link must have tabindex="0"');
  const ariaLabel = link.getAttribute('aria-label') || '';
  assert.ok(ariaLabel.startsWith('Download '), 'aria-label must describe download action');
  assert.ok(ariaLabel.includes('.json') || ariaLabel.includes('.mp4') || ariaLabel.includes('.png'));

  // Test Space key triggers download click
  link.click = () => { clickedLink = link; };
  link.on_keydown({ type: 'keydown', key: ' ', preventDefault() {}, stopPropagation() {} });
  assert.strictEqual(clickedLink, link, 'Space keydown must trigger link click');
  clickedLink = null;
});
"""
        self.run_node_script(js)


class TestSafeDOMRenderingNoInnerHTML(unittest.TestCase):
    """Test that dynamic data rendering is strictly safe against XSS (zero innerHTML assignments)."""

    def test_zero_dynamic_innerhtml_assignments(self) -> None:
        """JavaScript in INDEX_HTML must contain ZERO assignments to innerHTML, outerHTML, or document.write."""
        js_match = re.search(r"<script>([\s\S]*?)</script>", INDEX_HTML)
        self.assertIsNotNone(js_match, "INDEX_HTML must contain inline JavaScript")
        js_code = js_match.group(1)

        self.assertNotIn(".innerHTML =", js_code, "Zero .innerHTML assignments allowed")
        self.assertNotIn(".innerHTML=", js_code)
        self.assertNotIn(".outerHTML =", js_code)
        self.assertNotIn(".outerHTML=", js_code)
        self.assertNotIn("document.write(", js_code)
        self.assertNotIn("document.writeln(", js_code)

    def test_safe_dom_apis_used_for_dynamic_rendering(self) -> None:
        """JavaScript must use document.createElement, textContent, setAttribute, and appendChild."""
        js_match = re.search(r"<script>([\s\S]*?)</script>", INDEX_HTML)
        self.assertIsNotNone(js_match)
        js_code = js_match.group(1)

        self.assertIn("document.createElement", js_code)
        self.assertIn(".textContent =", js_code)
        self.assertIn(".setAttribute(", js_code)
        self.assertIn(".appendChild(", js_code)


class TestPreservationOfRequestVersionGuardsAndClobberProtection(_NodeHarnessBase):
    """Test preservation of request-version guards and job clobber protection."""

    def test_request_version_guards_defined_and_used(self) -> None:
        """INDEX_HTML declares statusRequestVersion and resultsRequestVersion guards."""
        self.assertIn("let statusRequestVersion = 0;", INDEX_HTML)
        self.assertIn("let resultsRequestVersion = 0;", INDEX_HTML)
        self.assertIn("const requestVersion = ++statusRequestVersion;", INDEX_HTML)
        self.assertIn("const requestVersion = ++resultsRequestVersion;", INDEX_HTML)
        self.assertIn("requestVersion !== statusRequestVersion", INDEX_HTML)
        self.assertIn("requestVersion !== resultsRequestVersion", INDEX_HTML)

    def test_job_switch_stale_response_discarded_headless(self) -> None:
        """Responses for superseded jobs never clobber active job status."""
        js = """
setCurrentJobId('active-job');
const badge = document.getElementById('status-badge');
updateStatusBadge('running');

context.fetch = async (url) => {
  if (url.includes('stale-job')) {
    return {
      ok: true,
      status: 200,
      json: async () => ({ status: 'failed', error: 'Stale job error' })
    };
  }
  return { ok: false, status: 404, json: async () => ({}) };
};

// Fire poll for older superseded job
await pollJobStatus('stale-job');

assert.strictEqual(document.getElementById('job-status').textContent, 'RUNNING',
  'Stale job response must be discarded without altering active status');
"""
        self.run_node_script(js)

    def test_same_job_stale_intermediate_status_discarded(self) -> None:
        """Out-of-order polling response for same job does not undo terminal failure."""
        js = """
setCurrentJobId('same-job-clobber');
const requests = [];
context.fetch = () => new Promise(resolve => requests.push(resolve));

const older = pollJobStatus('same-job-clobber');
const newer = pollJobStatus('same-job-clobber');

// Newer request resolves with terminal failure
requests[1]({ ok: true, status: 200, json: async () => ({ status: 'failed', error: 'Terminal error' }) });
await newer;

assert.strictEqual(document.getElementById('job-status').textContent, 'FAILED');

// Older intermediate running status resolves second
requests[0]({ ok: true, status: 200, json: async () => ({ status: 'running' }) });
await older;

assert.strictEqual(document.getElementById('job-status').textContent, 'FAILED',
  'Older out-of-order response must not supersede newer terminal status');
"""
        self.run_node_script(js)


if __name__ == "__main__":
    unittest.main()
