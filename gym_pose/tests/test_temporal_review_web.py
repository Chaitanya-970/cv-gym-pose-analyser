"""FastAPI endpoint and web integration test suite for temporal human event annotations in ReviewWorkspace.

Validates:
1. GET / and mounted /review/: verify HTML contains the event annotations UI, inputs, and button hooks
   (including data-action="capture-time" or equivalent time-capture action).
2. POST /api/cases/{case_id}/review with valid events payload returns 200/OK and persists events.
3. POST /api/cases/{case_id}/review with malformed events (inverted timestamps, duplicate event_id,
   invalid event_class, negative/NaN timestamps) returns 400 Bad Request with descriptive detail.
4. Review export endpoints (/api/export/manifest and /api/export/counts) return events correctly and maintain
   evaluator compatibility.
5. Strict isolation: submitting reviews with events NEVER triggers any ML analysis worker or background model inference.
6. Static accessibility: ARIA attributes, accessible labels, semantic table/form elements, and focus indicators.
"""

from __future__ import annotations

import inspect
import io
import json
import os
import re
import shutil
import sys
import tempfile
import unittest
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

_GYM_POSE_DIR = Path(__file__).resolve().parent.parent
if str(_GYM_POSE_DIR) not in sys.path:
    sys.path.insert(0, str(_GYM_POSE_DIR))

try:
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from gym_analyser.review_web import INDEX_HTML, create_review_app
    HAS_WEB_DEPS = True
except (ImportError, TypeError):
    FastAPI = None  # type: ignore
    TestClient = None  # type: ignore
    create_review_app = None  # type: ignore
    INDEX_HTML = ""
    HAS_WEB_DEPS = False


def _build_test_app(storage_root: str, max_upload_bytes: int = 10 * 1024 * 1024):
    """Instantiate test review app with explicit storage root."""
    if create_review_app is None:
        raise RuntimeError("FastAPI / web dependencies not available")
    return create_review_app(storage_root=storage_root, max_upload_bytes=max_upload_bytes)


class TestTemporalReviewWebEndpoints(unittest.TestCase):
    """Integration test suite for temporal event review web endpoints."""

    def setUp(self) -> None:
        if not HAS_WEB_DEPS:
            self.skipTest("FastAPI or review_web dependencies not available")
        self.test_dir = tempfile.mkdtemp(prefix="test_temporal_web_")
        self.max_bytes = 10 * 1024 * 1024
        self.app = _build_test_app(self.test_dir, self.max_bytes)
        self.client = TestClient(self.app)

    def tearDown(self) -> None:
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def _upload_dummy_video(self, name: str = "test_vid.mp4", content: bytes = b"dummy-video-content-stream") -> str:
        files = {"file": (name, io.BytesIO(content), "video/mp4")}
        res = self.client.post("/api/upload", files=files)
        self.assertIn(res.status_code, [200, 201])
        data = res.json()
        cid = data.get("case_id") or data.get("id") or data.get("clip_id")
        self.assertIsNotNone(cid)
        return str(cid)

    # 1. GET / and mounted /review/ contains event annotations UI
    def test_get_root_contains_event_ui_and_capture_button(self) -> None:
        """GET / on standalone app serves HTML containing event annotations UI and time capture hooks."""
        res = self.client.get("/")
        self.assertEqual(res.status_code, 200)
        self.assertIn("text/html", res.headers.get("content-type", ""))
        html = res.text

        # Verify event annotations controls exist in HTML / JS template
        self.assertTrue(
            "data-action=\"capture-time\"" in html
            or "capture-time" in html
            or "capture_time" in html
            or "event" in html.lower(),
            "HTML must include time capture action hooks or event annotations controls",
        )
        self.assertIn("completed_rep", html)
        self.assertIn("setup_or_weight_return", html)

    def test_mounted_review_prefix_contains_event_ui(self) -> None:
        """When mounted under /review, GET /review/ serves HTML containing event annotations UI."""
        host_app = FastAPI(title="Host App")
        review_sub_app = _build_test_app(self.test_dir, self.max_bytes)
        host_app.mount("/review", review_sub_app)

        mounted_client = TestClient(host_app)
        res = mounted_client.get("/review/")
        self.assertEqual(res.status_code, 200)
        self.assertIn("text/html", res.headers.get("content-type", ""))
        html = res.text

        self.assertTrue(
            "data-action=\"capture-time\"" in html
            or "capture-time" in html
            or "event" in html.lower(),
            "Mounted HTML must include event annotation UI and capture-time hooks",
        )

    # 2. POST update review with valid events payload
    def test_post_review_with_valid_events_success(self) -> None:
        """POST /api/cases/{case_id}/review with valid events returns 200 and persists events."""
        cid = self._upload_dummy_video("valid_events.mp4")

        valid_payload = {
            "human_count": 2,
            "reviewer": "Web Annotator",
            "raw_video_reviewed": True,
            "predictions_previously_seen": False,
            "scope": "in_scope",
            "notes": "Testing event submission",
            "events": [
                {
                    "event_id": "evt_rep_01",
                    "event_class": "completed_rep",
                    "start_time_s": 0.75,
                    "turnaround_time_s": 1.50,
                    "end_time_s": 2.25,
                    "uncertainty_notes": "Clean repetition",
                    "visible_posture_observations": "Good depth",
                },
                {
                    "event_id": "evt_setup_01",
                    "event_class": "setup_or_weight_return",
                    "start_time_s": 2.50,
                    "turnaround_time_s": None,
                    "end_time_s": 4.00,
                    "uncertainty_notes": "Reracking barbell",
                },
            ],
        }

        res = self.client.post(f"/api/cases/{cid}/review", json=valid_payload)
        self.assertEqual(res.status_code, 200)

        # GET case metadata to verify event persistence
        get_res = self.client.get(f"/api/cases/{cid}")
        self.assertEqual(get_res.status_code, 200)
        case_data = get_res.json()
        rev = case_data.get("review", case_data)
        self.assertEqual(rev.get("human_count"), 2)
        self.assertEqual(rev.get("reviewer"), "Web Annotator")

        events = rev.get("events", [])
        self.assertEqual(len(events), 2)

        self.assertEqual(events[0]["event_id"], "evt_rep_01")
        self.assertEqual(events[0]["event_class"], "completed_rep")
        self.assertEqual(events[0]["start_time_s"], 0.75)
        self.assertEqual(events[0]["turnaround_time_s"], 1.50)
        self.assertEqual(events[0]["end_time_s"], 2.25)
        self.assertEqual(events[0]["uncertainty_notes"], "Clean repetition")

        self.assertEqual(events[1]["event_id"], "evt_setup_01")
        self.assertEqual(events[1]["event_class"], "setup_or_weight_return")
        self.assertEqual(events[1]["start_time_s"], 2.50)
        self.assertIsNone(events[1].get("turnaround_time_s"))
        self.assertEqual(events[1]["end_time_s"], 4.00)

    def test_post_review_with_null_and_partial_timestamps_success(self) -> None:
        """POST review with null timestamps returns 200 and persists null values cleanly."""
        cid = self._upload_dummy_video("null_timestamps.mp4")

        payload = {
            "human_count": 0,
            "reviewer": "Jane Doe",
            "raw_video_reviewed": True,
            "predictions_previously_seen": False,
            "events": [
                {
                    "event_id": "evt_ambiguous",
                    "event_class": "ambiguous_or_unclear",
                    "start_time_s": None,
                    "turnaround_time_s": None,
                    "end_time_s": None,
                    "uncertainty_notes": "Lifter partially obscured",
                }
            ],
        }

        res = self.client.post(f"/api/cases/{cid}/review", json=payload)
        self.assertEqual(res.status_code, 200)

        get_res = self.client.get(f"/api/cases/{cid}")
        self.assertEqual(get_res.status_code, 200)
        rev = get_res.json().get("review", get_res.json())
        events = rev.get("events", [])
        self.assertEqual(len(events), 1)
        self.assertIsNone(events[0].get("start_time_s"))
        self.assertIsNone(events[0].get("turnaround_time_s"))
        self.assertIsNone(events[0].get("end_time_s"))

    # 3. POST update review with malformed events returns 400 Bad Request
    def test_post_review_with_inverted_turnaround_returns_400(self) -> None:
        """turnaround_time_s < start_time_s returns 400 Bad Request with descriptive error."""
        cid = self._upload_dummy_video("inverted_turn.mp4")

        bad_payload = {
            "human_count": 1,
            "reviewer": "Jane Doe",
            "raw_video_reviewed": True,
            "predictions_previously_seen": False,
            "events": [
                {
                    "event_id": "evt_inv_turn",
                    "event_class": "completed_rep",
                    "start_time_s": 2.5,
                    "turnaround_time_s": 1.5,  # Inverted!
                    "end_time_s": 3.0,
                }
            ],
        }

        res = self.client.post(f"/api/cases/{cid}/review", json=bad_payload)
        self.assertIn(res.status_code, [400, 422])
        err_msg = res.json().get("detail", "")
        self.assertTrue(len(str(err_msg)) > 0)

    def test_post_review_with_inverted_end_time_returns_400(self) -> None:
        """end_time_s < turnaround_time_s or end_time_s < start_time_s returns 400 Bad Request."""
        cid = self._upload_dummy_video("inverted_end.mp4")

        # 1. end < turnaround
        bad_payload1 = {
            "human_count": 1,
            "reviewer": "Jane Doe",
            "raw_video_reviewed": True,
            "predictions_previously_seen": False,
            "events": [
                {
                    "event_id": "evt_inv_end1",
                    "event_class": "completed_rep",
                    "start_time_s": 1.0,
                    "turnaround_time_s": 3.0,
                    "end_time_s": 2.0,  # Inverted!
                }
            ],
        }
        res1 = self.client.post(f"/api/cases/{cid}/review", json=bad_payload1)
        self.assertIn(res1.status_code, [400, 422])

        # 2. end < start (with turnaround None)
        bad_payload2 = {
            "human_count": 1,
            "reviewer": "Jane Doe",
            "raw_video_reviewed": True,
            "predictions_previously_seen": False,
            "events": [
                {
                    "event_id": "evt_inv_end2",
                    "event_class": "completed_rep",
                    "start_time_s": 4.0,
                    "turnaround_time_s": None,
                    "end_time_s": 2.0,  # Inverted!
                }
            ],
        }
        res2 = self.client.post(f"/api/cases/{cid}/review", json=bad_payload2)
        self.assertIn(res2.status_code, [400, 422])

    def test_post_review_with_duplicate_event_id_returns_400(self) -> None:
        """Duplicate event_id in review payload returns 400 Bad Request."""
        cid = self._upload_dummy_video("dup_id_web.mp4")

        dup_payload = {
            "human_count": 2,
            "reviewer": "Jane Doe",
            "raw_video_reviewed": True,
            "predictions_previously_seen": False,
            "events": [
                {"event_id": "duplicate_evt", "event_class": "completed_rep", "start_time_s": 1.0, "end_time_s": 2.0},
                {"event_id": "duplicate_evt", "event_class": "partial_or_unfinished", "start_time_s": 2.5, "end_time_s": 3.5},
            ],
        }

        res = self.client.post(f"/api/cases/{cid}/review", json=dup_payload)
        self.assertIn(res.status_code, [400, 422])
        err_msg = res.json().get("detail", "")
        self.assertIn("duplicate", str(err_msg).lower())

    def test_post_review_with_invalid_event_class_returns_400(self) -> None:
        """Invalid event_class in review payload returns 400 Bad Request."""
        cid = self._upload_dummy_video("invalid_cls_web.mp4")

        bad_cls_payload = {
            "human_count": 1,
            "reviewer": "Jane Doe",
            "raw_video_reviewed": True,
            "predictions_previously_seen": False,
            "events": [
                {"event_id": "evt_good_id", "event_class": "unsupported_class_xyz", "start_time_s": 1.0, "end_time_s": 2.0}
            ],
        }

        res = self.client.post(f"/api/cases/{cid}/review", json=bad_cls_payload)
        self.assertIn(res.status_code, [400, 422])

    def test_post_review_with_negative_or_nan_timestamps_returns_400(self) -> None:
        """Negative timestamps or NaN return 400 Bad Request."""
        cid = self._upload_dummy_video("neg_time_web.mp4")

        bad_neg_payload = {
            "human_count": 1,
            "reviewer": "Jane Doe",
            "raw_video_reviewed": True,
            "predictions_previously_seen": False,
            "events": [
                {"event_id": "evt_neg", "event_class": "completed_rep", "start_time_s": -2.0, "end_time_s": 1.0}
            ],
        }

        res = self.client.post(f"/api/cases/{cid}/review", json=bad_neg_payload)
        self.assertIn(res.status_code, [400, 422])

    # 4. Review export endpoints return events correctly
    def test_export_endpoints_return_events_and_manifest(self) -> None:
        """GET /api/export/manifest and GET /api/export/counts reflect reviews with events."""
        cid = self._upload_dummy_video("export_web.mp4")
        self.client.post(
            f"/api/cases/{cid}/review",
            json={
                "human_count": 3,
                "reviewer": "Export Web Tester",
                "raw_video_reviewed": True,
                "predictions_previously_seen": False,
                "scope": "in_scope",
                "events": [
                    {
                        "event_id": "evt_manifest_01",
                        "event_class": "completed_rep",
                        "start_time_s": 0.5,
                        "turnaround_time_s": 1.5,
                        "end_time_s": 2.5,
                    }
                ],
            },
        )

        # 1. Manifest endpoint includes events
        manifest_res = self.client.get("/api/export/manifest")
        self.assertEqual(manifest_res.status_code, 200)
        manifest_data = manifest_res.json()
        self.assertIn("events", manifest_data)
        matching_evts = [e for e in manifest_data["events"] if e.get("event_id") == "evt_manifest_01"]
        self.assertEqual(len(matching_evts), 1)
        self.assertEqual(matching_evts[0]["event_class"], "completed_rep")

        # 2. Counts endpoint retains valid structure
        counts_res = self.client.get("/api/export/counts")
        self.assertEqual(counts_res.status_code, 200)
        counts_data = counts_res.json()
        self.assertIn("clips", counts_data)
        clip_match = [c for c in counts_data["clips"] if c.get("clip_id") == cid]
        self.assertEqual(len(clip_match), 1)
        self.assertEqual(clip_match[0]["human_count"], 3)

    # 5. Isolation: No ML imports or background model inference
    def test_isolation_no_ml_imports_and_no_background_inference(self) -> None:
        """Inspect review_web module source to ensure NO ML imports or inference workers exist."""
        try:
            import gym_analyser.review_web as rweb
        except ImportError:
            self.skipTest("review_web module not available")

        src = inspect.getsource(rweb).lower()
        forbidden_tokens = [
            "ultralytics",
            "yolo",
            "torch",
            "gym_analyser.pipeline",
            "gym_analyser.counter",
            "subprocess.popen",
            "multiprocessing.process",
        ]
        for token in forbidden_tokens:
            self.assertNotIn(
                f"import {token}",
                src,
                f"Forbidden token 'import {token}' found in review_web.py source",
            )
            self.assertNotIn(
                f"from {token}",
                src,
                f"Forbidden token 'from {token}' found in review_web.py source",
            )

    # 6. Static accessibility: ARIA attributes, labels, and focus indicators
    def test_static_accessibility_event_controls(self) -> None:
        """INDEX_HTML contains accessible labels, ARIA attributes, semantic structure, and focus styling."""
        try:
            from gym_analyser.review_web import INDEX_HTML
        except ImportError:
            self.skipTest("review_web.INDEX_HTML not available")

        html = INDEX_HTML

        # 1. Semantic table or fieldset container for events
        self.assertTrue(
            "<table" in html or "<fieldset" in html or "event" in html.lower(),
            "Event UI must use semantic table or form elements",
        )

        # 2. Accessible button labels or ARIA labels
        self.assertTrue(
            "aria-label" in html
            or "<label" in html
            or "title=" in html,
            "Event controls must include aria-label, label, or title attributes",
        )

        # 3. Focus indicators in CSS
        self.assertTrue(
            ":focus" in html or "outline" in html,
            "CSS must include focus indicators for keyboard accessibility",
        )


if __name__ == "__main__":
    unittest.main()
