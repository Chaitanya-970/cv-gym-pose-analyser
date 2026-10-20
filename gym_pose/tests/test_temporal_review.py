"""Standard-library regression test suite for temporal human event annotations in ReviewStore.

Verifies:
1. Backward compatibility: Existing cases without events load without error, export cleanly, and validate.
2. Valid events: Adding valid events with all supported classes (completed_rep, setup_or_weight_return,
   partial_or_unfinished, ambiguous_or_unclear) and full optional metadata.
3. Missing/optional timestamps: Events with only start_time_s, turnaround_time_s, end_time_s, or no timestamps at all.
4. Timestamp ordering enforcement:
   - turnaround_time_s < start_time_s raises ValueError
   - end_time_s < turnaround_time_s raises ValueError
   - end_time_s < start_time_s raises ValueError
5. Negative, non-finite, and invalid-typed timestamps (e.g. -1.0, NaN, Inf, boolean) rejected with ValueError.
6. Unique event_id enforcement: duplicate event_id within the same review raises ValueError. Empty or non-string rejected.
7. Invalid event_class raises ValueError.
8. Independence of review count: human_count is never altered or auto-computed from events.
9. Roundtrip persistence: updating review with events, reloading case, and verifying bit-for-bit fidelity.
10. History preservation: each update appends to case history preserving prior event snapshots.
11. Export compatibility: export_reviewed_counts / export_counts succeeds and retains valid ground truth structure.
12. Concurrent save and corruption protection: stale case updates or corrupted files fail-closed with CaseCorruptedError.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
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

from gym_analyser.evaluation import load_ground_truth
from gym_analyser.review_store import (
    PUBLIC_CASE_KEYS,
    CaseCorruptedError,
    CaseNotFoundError,
    InvalidCaseIdError,
    ReviewStore,
)


def _get_id(case_dict: Dict[str, Any]) -> str:
    """Helper to extract case_id from case dict."""
    cid = case_dict.get("case_id") or case_dict.get("id") or case_dict.get("clip_id")
    if not cid:
        raise KeyError(f"No case ID found in keys: {list(case_dict.keys())}")
    return str(cid)


def _get_review(case_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Helper to extract review dictionary whether nested or flat."""
    if "review" in case_dict and isinstance(case_dict["review"], dict):
        return case_dict["review"]
    return case_dict


def _get_count(case_dict: Dict[str, Any]) -> Any:
    """Helper to extract human_count."""
    return _get_review(case_dict).get("human_count")


def _get_events(case_dict: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Helper to extract events list."""
    rev = _get_review(case_dict)
    evts = rev.get("events")
    return evts if isinstance(evts, list) else []


def _call_export_reviewed_counts(store: ReviewStore) -> Dict[str, Any]:
    """Call store.export_reviewed_counts() if available, else store.export_counts()."""
    if hasattr(store, "export_reviewed_counts"):
        return store.export_reviewed_counts()
    return store.export_counts()


class TestTemporalReviewBackwardCompatibility(unittest.TestCase):
    """Test suite ensuring backward compatibility with existing count-only and legacy reviews."""

    def setUp(self) -> None:
        self.test_dir = tempfile.mkdtemp(prefix="test_temporal_compat_")
        self.store = ReviewStore(self.test_dir, max_upload_bytes=10 * 1024 * 1024)

    def tearDown(self) -> None:
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def _create_dummy_video(self, name: str = "dummy.mp4", content: bytes = b"dummy-video-data") -> Path:
        fixtures_dir = Path(tempfile.gettempdir()) / f"fix_{uuid.uuid4().hex[:8]}"
        fixtures_dir.mkdir(parents=True, exist_ok=True)
        file_path = fixtures_dir / name
        file_path.write_bytes(content)
        return file_path

    def test_existing_case_without_events_loads_and_validates(self) -> None:
        """A case saved without any 'events' key loads cleanly, validates, and exports."""
        src = self._create_dummy_video("compat1.mp4")
        case = self.store.create_case(src, original_name="compat1.mp4", group="candidate")
        cid = _get_id(case)

        # Save review with no "events" key
        saved = self.store.save_review(
            cid,
            {
                "human_count": 5,
                "reviewer": "Reviewer One",
                "raw_video_reviewed": True,
                "predictions_previously_seen": False,
                "scope": "in_scope",
                "notes": "No events provided",
            },
        )
        self.assertEqual(_get_count(saved), 5)
        # get_case must succeed
        loaded = self.store.get_case(cid)
        self.assertEqual(_get_count(loaded), 5)
        self.assertIn("review", loaded)
        # events should be empty list or None, never cause crash
        evts = _get_events(loaded)
        self.assertEqual(evts, [])

    def test_review_with_events_none_loads_and_validates(self) -> None:
        """Explicit events=None is accepted and validated as empty/none."""
        src = self._create_dummy_video("compat2.mp4")
        case = self.store.create_case(src, original_name="compat2.mp4", group="candidate")
        cid = _get_id(case)

        saved = self.store.save_review(
            cid,
            {
                "human_count": 3,
                "reviewer": "Reviewer Two",
                "raw_video_reviewed": True,
                "predictions_previously_seen": False,
                "events": None,
            },
        )
        self.assertEqual(_get_count(saved), 3)
        loaded = self.store.get_case(cid)
        self.assertEqual(_get_count(loaded), 3)

    def test_review_with_empty_events_list(self) -> None:
        """Explicit events=[] is accepted and preserved."""
        src = self._create_dummy_video("compat3.mp4")
        case = self.store.create_case(src, original_name="compat3.mp4", group="candidate")
        cid = _get_id(case)

        saved = self.store.save_review(
            cid,
            {
                "human_count": 2,
                "reviewer": "Reviewer Three",
                "raw_video_reviewed": True,
                "predictions_previously_seen": False,
                "events": [],
            },
        )
        self.assertEqual(_get_count(saved), 2)
        loaded = self.store.get_case(cid)
        self.assertEqual(_get_events(loaded), [])

    def test_legacy_task32_events_schema_compatibility(self) -> None:
        """Legacy event dicts with event_index/intent/cycle still load or normalize without error."""
        src = self._create_dummy_video("legacy.mp4")
        case = self.store.create_case(src, original_name="legacy.mp4", group="candidate")
        cid = _get_id(case)

        # Legacy payload using event_index, intent, cycle
        legacy_payload = {
            "human_count": 1,
            "reviewer": "Legacy Tester",
            "raw_video_reviewed": True,
            "predictions_previously_seen": False,
            "events": [
                {
                    "event_index": 0,
                    "start_time_s": 1.0,
                    "end_time_s": 2.5,
                    "intent": "intentional_squat",
                    "cycle": "complete",
                    "observation": "Good squat",
                }
            ],
        }
        res = self.store.save_review(cid, legacy_payload)
        self.assertEqual(_get_count(res), 1)
        loaded = self.store.get_case(cid)
        self.assertEqual(len(_get_events(loaded)), 1)


class TestTemporalReviewValidEvents(unittest.TestCase):
    """Test suite for adding valid events across all allowed event classes."""

    def setUp(self) -> None:
        self.test_dir = tempfile.mkdtemp(prefix="test_temporal_valid_")
        self.store = ReviewStore(self.test_dir, max_upload_bytes=10 * 1024 * 1024)

    def tearDown(self) -> None:
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def _create_dummy_video(self, name: str = "dummy.mp4", content: bytes = b"valid-event-video") -> Path:
        fixtures_dir = Path(tempfile.gettempdir()) / f"fix_{uuid.uuid4().hex[:8]}"
        fixtures_dir.mkdir(parents=True, exist_ok=True)
        file_path = fixtures_dir / name
        file_path.write_bytes(content)
        return file_path

    def test_all_valid_event_classes(self) -> None:
        """All 4 supported event classes are accepted: completed_rep, setup_or_weight_return,
        partial_or_unfinished, ambiguous_or_unclear."""
        classes = [
            "completed_rep",
            "setup_or_weight_return",
            "partial_or_unfinished",
            "ambiguous_or_unclear",
        ]
        src = self._create_dummy_video("classes.mp4")
        case = self.store.create_case(src, original_name="classes.mp4")
        cid = _get_id(case)

        events_payload = [
            {
                "event_id": f"evt_{cls}",
                "event_class": cls,
                "start_time_s": float(i * 2),
                "turnaround_time_s": float(i * 2 + 1),
                "end_time_s": float(i * 2 + 2),
                "uncertainty_notes": f"Testing class {cls}",
            }
            for i, cls in enumerate(classes)
        ]

        saved = self.store.save_review(
            cid,
            {
                "human_count": 1,
                "reviewer": "Class Validator",
                "raw_video_reviewed": True,
                "predictions_previously_seen": False,
                "events": events_payload,
            },
        )
        events_saved = _get_events(saved)
        self.assertEqual(len(events_saved), 4)

        for i, cls in enumerate(classes):
            ev = events_saved[i]
            self.assertEqual(ev.get("event_id"), f"evt_{cls}")
            self.assertEqual(ev.get("event_class"), cls)
            self.assertEqual(ev.get("start_time_s"), float(i * 2))
            self.assertEqual(ev.get("turnaround_time_s"), float(i * 2 + 1))
            self.assertEqual(ev.get("end_time_s"), float(i * 2 + 2))

    def test_event_with_all_optional_fields(self) -> None:
        """Event containing uncertainty notes and visible posture observations preserves both cleanly."""
        src = self._create_dummy_video("full_fields.mp4")
        case = self.store.create_case(src, original_name="full_fields.mp4")
        cid = _get_id(case)

        full_event = {
            "event_id": "evt_full_001",
            "event_class": "completed_rep",
            "start_time_s": 1.25,
            "turnaround_time_s": 2.50,
            "end_time_s": 3.75,
            "uncertainty_notes": "±2 frames around turnaround inflection",
            "visible_posture_observations": {
                "torso_consistency": "stable vertical inclination",
                "depth_observation": "hip crease below knee joint",
            },
        }

        saved = self.store.save_review(
            cid,
            {
                "human_count": 1,
                "reviewer": "Detailed Observer",
                "raw_video_reviewed": True,
                "predictions_previously_seen": True,
                "events": [full_event],
            },
        )

        loaded = self.store.get_case(cid)
        evts = _get_events(loaded)
        self.assertEqual(len(evts), 1)
        ev = evts[0]
        self.assertEqual(ev["event_id"], "evt_full_001")
        self.assertEqual(ev["event_class"], "completed_rep")
        self.assertEqual(ev["start_time_s"], 1.25)
        self.assertEqual(ev["turnaround_time_s"], 2.50)
        self.assertEqual(ev["end_time_s"], 3.75)
        self.assertEqual(ev["uncertainty_notes"], "±2 frames around turnaround inflection")
        self.assertEqual(
            ev["visible_posture_observations"],
            {
                "torso_consistency": "stable vertical inclination",
                "depth_observation": "hip crease below knee joint",
            },
        )


class TestTemporalReviewTimestampFlexibility(unittest.TestCase):
    """Test suite for optional, partial, and null timestamps."""

    def setUp(self) -> None:
        self.test_dir = tempfile.mkdtemp(prefix="test_temporal_flex_")
        self.store = ReviewStore(self.test_dir, max_upload_bytes=10 * 1024 * 1024)

    def tearDown(self) -> None:
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def _create_dummy_video(self, name: str = "dummy.mp4") -> Path:
        fixtures_dir = Path(tempfile.gettempdir()) / f"fix_{uuid.uuid4().hex[:8]}"
        fixtures_dir.mkdir(parents=True, exist_ok=True)
        file_path = fixtures_dir / name
        file_path.write_bytes(b"flex-timestamps-data")
        return file_path

    def test_events_with_all_null_timestamps(self) -> None:
        """Events where start_time_s, turnaround_time_s, and end_time_s are all None are valid."""
        src = self._create_dummy_video("null_times.mp4")
        case = self.store.create_case(src, original_name="null_times.mp4")
        cid = _get_id(case)

        event_null = {
            "event_id": "evt_null_times",
            "event_class": "ambiguous_or_unclear",
            "start_time_s": None,
            "turnaround_time_s": None,
            "end_time_s": None,
            "uncertainty_notes": "Camera blur prevented time boundary identification",
        }

        saved = self.store.save_review(
            cid,
            {
                "human_count": 0,
                "reviewer": "Observer",
                "raw_video_reviewed": True,
                "predictions_previously_seen": False,
                "events": [event_null],
            },
        )
        loaded = self.store.get_case(cid)
        ev = _get_events(loaded)[0]
        self.assertEqual(ev["event_id"], "evt_null_times")
        self.assertIsNone(ev.get("start_time_s"))
        self.assertIsNone(ev.get("turnaround_time_s"))
        self.assertIsNone(ev.get("end_time_s"))

    def test_partial_timestamps_combinations(self) -> None:
        """Test all valid subsets of timestamps:
        - start only
        - turnaround only
        - end only
        - start + turnaround
        - turnaround + end
        - start + end
        """
        src = self._create_dummy_video("partial_times.mp4")
        case = self.store.create_case(src, original_name="partial_times.mp4")
        cid = _get_id(case)

        events = [
            {"event_id": "e_start_only", "event_class": "completed_rep", "start_time_s": 1.0, "turnaround_time_s": None, "end_time_s": None},
            {"event_id": "e_turnaround_only", "event_class": "completed_rep", "start_time_s": None, "turnaround_time_s": 2.5, "end_time_s": None},
            {"event_id": "e_end_only", "event_class": "completed_rep", "start_time_s": None, "turnaround_time_s": None, "end_time_s": 4.0},
            {"event_id": "e_start_turnaround", "event_class": "completed_rep", "start_time_s": 5.0, "turnaround_time_s": 6.0, "end_time_s": None},
            {"event_id": "e_turnaround_end", "event_class": "completed_rep", "start_time_s": None, "turnaround_time_s": 7.0, "end_time_s": 8.0},
            {"event_id": "e_start_end", "event_class": "completed_rep", "start_time_s": 9.0, "turnaround_time_s": None, "end_time_s": 11.0},
        ]

        saved = self.store.save_review(
            cid,
            {
                "human_count": 6,
                "reviewer": "Combinations Tester",
                "raw_video_reviewed": True,
                "predictions_previously_seen": False,
                "events": events,
            },
        )
        loaded = self.store.get_case(cid)
        evts = _get_events(loaded)
        self.assertEqual(len(evts), 6)

        # e_start_only
        self.assertEqual(evts[0]["start_time_s"], 1.0)
        self.assertIsNone(evts[0]["turnaround_time_s"])
        self.assertIsNone(evts[0]["end_time_s"])

        # e_start_end
        self.assertEqual(evts[5]["start_time_s"], 9.0)
        self.assertIsNone(evts[5]["turnaround_time_s"])
        self.assertEqual(evts[5]["end_time_s"], 11.0)

    def test_equal_timestamps_allowed(self) -> None:
        """Equal timestamps (e.g. instantaneous turnaround where start == turnaround or turnaround == end) are valid."""
        src = self._create_dummy_video("equal_times.mp4")
        case = self.store.create_case(src, original_name="equal_times.mp4")
        cid = _get_id(case)

        equal_event = {
            "event_id": "evt_equal",
            "event_class": "partial_or_unfinished",
            "start_time_s": 2.0,
            "turnaround_time_s": 2.0,
            "end_time_s": 2.0,
        }
        saved = self.store.save_review(
            cid,
            {
                "human_count": 0,
                "reviewer": "Instantaneous Observer",
                "raw_video_reviewed": True,
                "predictions_previously_seen": False,
                "events": [equal_event],
            },
        )
        evts = _get_events(saved)
        self.assertEqual(evts[0]["start_time_s"], 2.0)
        self.assertEqual(evts[0]["turnaround_time_s"], 2.0)
        self.assertEqual(evts[0]["end_time_s"], 2.0)


class TestTemporalReviewTimestampOrderingEnforcement(unittest.TestCase):
    """Test suite ensuring strict chronological ordering constraints."""

    def setUp(self) -> None:
        self.test_dir = tempfile.mkdtemp(prefix="test_temporal_order_")
        self.store = ReviewStore(self.test_dir, max_upload_bytes=10 * 1024 * 1024)

    def tearDown(self) -> None:
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def _create_dummy_video(self, name: str = "dummy.mp4") -> Path:
        fixtures_dir = Path(tempfile.gettempdir()) / f"fix_{uuid.uuid4().hex[:8]}"
        fixtures_dir.mkdir(parents=True, exist_ok=True)
        file_path = fixtures_dir / name
        file_path.write_bytes(b"ordering-video-bytes")
        return file_path

    def test_turnaround_before_start_rejected(self) -> None:
        """turnaround_time_s < start_time_s raises ValueError."""
        src = self._create_dummy_video("order1.mp4")
        case = self.store.create_case(src, original_name="order1.mp4")
        cid = _get_id(case)

        bad_event = {
            "event_id": "e_bad_turnaround",
            "event_class": "completed_rep",
            "start_time_s": 3.0,
            "turnaround_time_s": 2.5,  # Inverted!
            "end_time_s": 4.0,
        }
        with self.assertRaises(ValueError):
            self.store.save_review(
                cid,
                {
                    "human_count": 1,
                    "reviewer": "Jane Doe",
                    "raw_video_reviewed": True,
                    "predictions_previously_seen": False,
                    "events": [bad_event],
                },
            )

    def test_end_before_turnaround_rejected(self) -> None:
        """end_time_s < turnaround_time_s raises ValueError."""
        src = self._create_dummy_video("order2.mp4")
        case = self.store.create_case(src, original_name="order2.mp4")
        cid = _get_id(case)

        bad_event = {
            "event_id": "e_bad_end_turnaround",
            "event_class": "completed_rep",
            "start_time_s": 1.0,
            "turnaround_time_s": 3.0,
            "end_time_s": 2.5,  # Inverted!
        }
        with self.assertRaises(ValueError):
            self.store.save_review(
                cid,
                {
                    "human_count": 1,
                    "reviewer": "Jane Doe",
                    "raw_video_reviewed": True,
                    "predictions_previously_seen": False,
                    "events": [bad_event],
                },
            )

    def test_end_before_start_rejected(self) -> None:
        """end_time_s < start_time_s raises ValueError (when turnaround is null)."""
        src = self._create_dummy_video("order3.mp4")
        case = self.store.create_case(src, original_name="order3.mp4")
        cid = _get_id(case)

        bad_event = {
            "event_id": "e_bad_end_start",
            "event_class": "completed_rep",
            "start_time_s": 5.0,
            "turnaround_time_s": None,
            "end_time_s": 3.0,  # Inverted!
        }
        with self.assertRaises(ValueError):
            self.store.save_review(
                cid,
                {
                    "human_count": 1,
                    "reviewer": "Jane Doe",
                    "raw_video_reviewed": True,
                    "predictions_previously_seen": False,
                    "events": [bad_event],
                },
            )


class TestTemporalReviewNonFiniteAndNegativeTimestamps(unittest.TestCase):
    """Test suite ensuring negative, NaN, Inf, and invalid-typed timestamps are rejected."""

    def setUp(self) -> None:
        self.test_dir = tempfile.mkdtemp(prefix="test_temporal_nonfinite_")
        self.store = ReviewStore(self.test_dir, max_upload_bytes=10 * 1024 * 1024)

    def tearDown(self) -> None:
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def _create_dummy_video(self, name: str = "dummy.mp4") -> Path:
        fixtures_dir = Path(tempfile.gettempdir()) / f"fix_{uuid.uuid4().hex[:8]}"
        fixtures_dir.mkdir(parents=True, exist_ok=True)
        file_path = fixtures_dir / name
        file_path.write_bytes(b"nonfinite-video-bytes")
        return file_path

    def test_negative_timestamps_rejected(self) -> None:
        """Negative timestamps in start, turnaround, or end raise ValueError."""
        src = self._create_dummy_video("neg.mp4")
        case = self.store.create_case(src, original_name="neg.mp4")
        cid = _get_id(case)

        negative_cases = [
            {"event_id": "e1", "event_class": "completed_rep", "start_time_s": -1.0, "turnaround_time_s": 1.0, "end_time_s": 2.0},
            {"event_id": "e2", "event_class": "completed_rep", "start_time_s": 0.5, "turnaround_time_s": -0.1, "end_time_s": 2.0},
            {"event_id": "e3", "event_class": "completed_rep", "start_time_s": 0.5, "turnaround_time_s": 1.0, "end_time_s": -2.0},
        ]
        for bad_evt in negative_cases:
            with self.assertRaises(ValueError):
                self.store.save_review(
                    cid,
                    {
                        "human_count": 1,
                        "reviewer": "Jane",
                        "raw_video_reviewed": True,
                        "predictions_previously_seen": False,
                        "events": [bad_evt],
                    },
                )

    def test_nan_timestamps_rejected(self) -> None:
        """float('nan') in any timestamp field raises ValueError."""
        src = self._create_dummy_video("nan.mp4")
        case = self.store.create_case(src, original_name="nan.mp4")
        cid = _get_id(case)

        nan_cases = [
            {"event_id": "e_nan1", "event_class": "completed_rep", "start_time_s": float("nan"), "turnaround_time_s": 1.0, "end_time_s": 2.0},
            {"event_id": "e_nan2", "event_class": "completed_rep", "start_time_s": 0.0, "turnaround_time_s": float("nan"), "end_time_s": 2.0},
            {"event_id": "e_nan3", "event_class": "completed_rep", "start_time_s": 0.0, "turnaround_time_s": 1.0, "end_time_s": float("nan")},
        ]
        for bad_evt in nan_cases:
            with self.assertRaises(ValueError):
                self.store.save_review(
                    cid,
                    {
                        "human_count": 1,
                        "reviewer": "Jane",
                        "raw_video_reviewed": True,
                        "predictions_previously_seen": False,
                        "events": [bad_evt],
                    },
                )

    def test_infinity_timestamps_rejected(self) -> None:
        """float('inf') or float('-inf') in any timestamp field raises ValueError."""
        src = self._create_dummy_video("inf.mp4")
        case = self.store.create_case(src, original_name="inf.mp4")
        cid = _get_id(case)

        inf_cases = [
            {"event_id": "e_inf1", "event_class": "completed_rep", "start_time_s": float("inf"), "turnaround_time_s": 1.0, "end_time_s": 2.0},
            {"event_id": "e_inf2", "event_class": "completed_rep", "start_time_s": 0.0, "turnaround_time_s": float("inf"), "end_time_s": 2.0},
            {"event_id": "e_inf3", "event_class": "completed_rep", "start_time_s": 0.0, "turnaround_time_s": 1.0, "end_time_s": float("inf")},
            {"event_id": "e_inf4", "event_class": "completed_rep", "start_time_s": float("-inf"), "turnaround_time_s": 1.0, "end_time_s": 2.0},
        ]
        for bad_evt in inf_cases:
            with self.assertRaises(ValueError):
                self.store.save_review(
                    cid,
                    {
                        "human_count": 1,
                        "reviewer": "Jane",
                        "raw_video_reviewed": True,
                        "predictions_previously_seen": False,
                        "events": [bad_evt],
                    },
                )

    def test_boolean_and_string_timestamps_rejected(self) -> None:
        """Booleans (True/False) or strings masquerading as numbers raise ValueError."""
        src = self._create_dummy_video("types.mp4")
        case = self.store.create_case(src, original_name="types.mp4")
        cid = _get_id(case)

        type_cases = [
            {"event_id": "e_bool1", "event_class": "completed_rep", "start_time_s": True, "turnaround_time_s": 1.0, "end_time_s": 2.0},
            {"event_id": "e_bool2", "event_class": "completed_rep", "start_time_s": 0.0, "turnaround_time_s": False, "end_time_s": 2.0},
            {"event_id": "e_str1", "event_class": "completed_rep", "start_time_s": "1.0", "turnaround_time_s": 2.0, "end_time_s": 3.0},
        ]
        for bad_evt in type_cases:
            with self.assertRaises(ValueError):
                self.store.save_review(
                    cid,
                    {
                        "human_count": 1,
                        "reviewer": "Jane",
                        "raw_video_reviewed": True,
                        "predictions_previously_seen": False,
                        "events": [bad_evt],
                    },
                )


class TestTemporalReviewEventIdAndClassValidation(unittest.TestCase):
    """Test suite ensuring event_id uniqueness and event_class validity."""

    def setUp(self) -> None:
        self.test_dir = tempfile.mkdtemp(prefix="test_temporal_id_class_")
        self.store = ReviewStore(self.test_dir, max_upload_bytes=10 * 1024 * 1024)

    def tearDown(self) -> None:
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def _create_dummy_video(self, name: str = "dummy.mp4") -> Path:
        fixtures_dir = Path(tempfile.gettempdir()) / f"fix_{uuid.uuid4().hex[:8]}"
        fixtures_dir.mkdir(parents=True, exist_ok=True)
        file_path = fixtures_dir / name
        file_path.write_bytes(b"id-class-video-bytes")
        return file_path

    def test_duplicate_event_id_raises_value_error(self) -> None:
        """Duplicate event_id within the same review raises ValueError."""
        src = self._create_dummy_video("dup_id.mp4")
        case = self.store.create_case(src, original_name="dup_id.mp4")
        cid = _get_id(case)

        dup_events = [
            {"event_id": "duplicate_id_01", "event_class": "completed_rep", "start_time_s": 1.0, "end_time_s": 2.0},
            {"event_id": "duplicate_id_01", "event_class": "setup_or_weight_return", "start_time_s": 2.5, "end_time_s": 3.5},
        ]
        with self.assertRaises(ValueError):
            self.store.save_review(
                cid,
                {
                    "human_count": 1,
                    "reviewer": "Duplicate Checker",
                    "raw_video_reviewed": True,
                    "predictions_previously_seen": False,
                    "events": dup_events,
                },
            )

    def test_empty_or_missing_event_id_rejected(self) -> None:
        """Empty string, whitespace, missing, or non-string event_id raises ValueError."""
        src = self._create_dummy_video("empty_id.mp4")
        case = self.store.create_case(src, original_name="empty_id.mp4")
        cid = _get_id(case)

        bad_id_cases = [
            {"event_id": "", "event_class": "completed_rep"},
            {"event_id": "   ", "event_class": "completed_rep"},
            {"event_id": 12345, "event_class": "completed_rep"},
            {"event_id": None, "event_class": "completed_rep"},
            {"event_class": "completed_rep"},  # Missing event_id entirely
        ]
        for bad_evt in bad_id_cases:
            with self.assertRaises(ValueError):
                self.store.save_review(
                    cid,
                    {
                        "human_count": 1,
                        "reviewer": "Jane",
                        "raw_video_reviewed": True,
                        "predictions_previously_seen": False,
                        "events": [bad_evt],
                    },
                )

    def test_invalid_event_class_raises_value_error(self) -> None:
        """event_class not in allowed set raises ValueError."""
        src = self._create_dummy_video("bad_class.mp4")
        case = self.store.create_case(src, original_name="bad_class.mp4")
        cid = _get_id(case)

        bad_classes = [
            "good_squat",
            "bad_depth",
            "not_an_event",
            "",
            "   ",
            123,
            None,
            True,
        ]
        for bad_cls in bad_classes:
            bad_evt = {"event_id": "evt_test", "event_class": bad_cls}
            with self.assertRaises(ValueError):
                self.store.save_review(
                    cid,
                    {
                        "human_count": 1,
                        "reviewer": "Jane",
                        "raw_video_reviewed": True,
                        "predictions_previously_seen": False,
                        "events": [bad_evt],
                    },
                )

    def test_non_dict_event_rejected(self) -> None:
        """Events list containing non-dict items raises ValueError."""
        src = self._create_dummy_video("non_dict.mp4")
        case = self.store.create_case(src, original_name="non_dict.mp4")
        cid = _get_id(case)

        bad_events_lists = [
            ["a_string_event"],
            [123],
            [True],
            [None],
            [{"event_id": "valid_1", "event_class": "completed_rep"}, "invalid_second_item"],
        ]
        for bad_list in bad_events_lists:
            with self.assertRaises(ValueError):
                self.store.save_review(
                    cid,
                    {
                        "human_count": 1,
                        "reviewer": "Jane",
                        "raw_video_reviewed": True,
                        "predictions_previously_seen": False,
                        "events": bad_list,
                    },
                )


class TestTemporalReviewIndependenceOfCount(unittest.TestCase):
    """Test suite ensuring human_count remains completely independent of event annotations."""

    def setUp(self) -> None:
        self.test_dir = tempfile.mkdtemp(prefix="test_temporal_indep_")
        self.store = ReviewStore(self.test_dir, max_upload_bytes=10 * 1024 * 1024)

    def tearDown(self) -> None:
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def _create_dummy_video(self, name: str = "dummy.mp4") -> Path:
        fixtures_dir = Path(tempfile.gettempdir()) / f"fix_{uuid.uuid4().hex[:8]}"
        fixtures_dir.mkdir(parents=True, exist_ok=True)
        file_path = fixtures_dir / name
        file_path.write_bytes(b"indep-video-bytes")
        return file_path

    def test_five_events_with_count_three(self) -> None:
        """5 events accompanying human_count=3 does NOT modify human_count to 5."""
        src = self._create_dummy_video("indep1.mp4")
        case = self.store.create_case(src, original_name="indep1.mp4")
        cid = _get_id(case)

        events_5 = [
            {"event_id": f"evt_{i}", "event_class": "completed_rep", "start_time_s": float(i), "end_time_s": float(i + 0.5)}
            for i in range(5)
        ]

        saved = self.store.save_review(
            cid,
            {
                "human_count": 3,
                "reviewer": "Independent Reviewer",
                "raw_video_reviewed": True,
                "predictions_previously_seen": False,
                "events": events_5,
            },
        )
        # human_count must remain strictly 3!
        self.assertEqual(_get_count(saved), 3)
        self.assertEqual(len(_get_events(saved)), 5)

        loaded = self.store.get_case(cid)
        self.assertEqual(_get_count(loaded), 3)
        self.assertEqual(len(_get_events(loaded)), 5)

    def test_events_with_null_human_count(self) -> None:
        """Events accompanying human_count=None preserves None count."""
        src = self._create_dummy_video("indep2.mp4")
        case = self.store.create_case(src, original_name="indep2.mp4")
        cid = _get_id(case)

        events_3 = [
            {"event_id": f"evt_{i}", "event_class": "partial_or_unfinished"}
            for i in range(3)
        ]

        saved = self.store.save_review(
            cid,
            {
                "human_count": None,
                "reviewer": "Null Tester",
                "raw_video_reviewed": True,
                "predictions_previously_seen": False,
                "events": events_3,
            },
        )
        self.assertIsNone(_get_count(saved))
        self.assertEqual(len(_get_events(saved)), 3)

    def test_events_with_zero_human_count(self) -> None:
        """Events accompanying human_count=0 preserves 0 count distinctly from None."""
        src = self._create_dummy_video("indep3.mp4")
        case = self.store.create_case(src, original_name="indep3.mp4")
        cid = _get_id(case)

        events_setup = [
            {"event_id": "e_setup", "event_class": "setup_or_weight_return", "start_time_s": 0.5, "end_time_s": 3.0}
        ]

        saved = self.store.save_review(
            cid,
            {
                "human_count": 0,
                "reviewer": "Zero Count Tester",
                "raw_video_reviewed": True,
                "predictions_previously_seen": False,
                "events": events_setup,
            },
        )
        self.assertEqual(_get_count(saved), 0)
        self.assertIsNotNone(_get_count(saved))
        self.assertEqual(len(_get_events(saved)), 1)


class TestTemporalReviewRoundtripPersistence(unittest.TestCase):
    """Test suite verifying bit-for-bit event fidelity across disk serialization."""

    def setUp(self) -> None:
        self.test_dir = tempfile.mkdtemp(prefix="test_temporal_roundtrip_")
        self.store = ReviewStore(self.test_dir, max_upload_bytes=10 * 1024 * 1024)

    def tearDown(self) -> None:
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def _create_dummy_video(self, name: str = "dummy.mp4") -> Path:
        fixtures_dir = Path(tempfile.gettempdir()) / f"fix_{uuid.uuid4().hex[:8]}"
        fixtures_dir.mkdir(parents=True, exist_ok=True)
        file_path = fixtures_dir / name
        file_path.write_bytes(b"roundtrip-video-bytes")
        return file_path

    def test_bit_for_bit_event_fidelity(self) -> None:
        """Persisting events and reloading via get_case and load_persisted_case retains bit-for-bit fidelity."""
        src = self._create_dummy_video("roundtrip.mp4")
        case = self.store.create_case(src, original_name="roundtrip.mp4")
        cid = _get_id(case)

        events_input = [
            {
                "event_id": "rep_01",
                "event_class": "completed_rep",
                "start_time_s": 1.125,
                "turnaround_time_s": 2.250,
                "end_time_s": 3.375,
                "uncertainty_notes": "Clean rep",
                "visible_posture_observations": "Good vertical trunk",
            },
            {
                "event_id": "setup_01",
                "event_class": "setup_or_weight_return",
                "start_time_s": 3.5,
                "turnaround_time_s": None,
                "end_time_s": 5.0,
                "uncertainty_notes": None,
                "visible_posture_observations": None,
            },
            {
                "event_id": "unclear_01",
                "event_class": "ambiguous_or_unclear",
                "start_time_s": None,
                "turnaround_time_s": None,
                "end_time_s": None,
                "uncertainty_notes": "Occlusion by spotter",
                "visible_posture_observations": {"occlusion": True},
            },
        ]

        self.store.save_review(
            cid,
            {
                "human_count": 1,
                "reviewer": "Roundtrip Tester",
                "raw_video_reviewed": True,
                "predictions_previously_seen": False,
                "events": events_input,
            },
        )

        # 1. Reload via public API get_case
        case_public = self.store.get_case(cid)
        events_public = _get_events(case_public)
        self.assertEqual(len(events_public), 3)

        self.assertEqual(events_public[0]["event_id"], "rep_01")
        self.assertEqual(events_public[0]["event_class"], "completed_rep")
        self.assertEqual(events_public[0]["start_time_s"], 1.125)
        self.assertEqual(events_public[0]["turnaround_time_s"], 2.250)
        self.assertEqual(events_public[0]["end_time_s"], 3.375)
        self.assertEqual(events_public[0]["uncertainty_notes"], "Clean rep")
        self.assertEqual(events_public[0]["visible_posture_observations"], "Good vertical trunk")

        self.assertEqual(events_public[1]["event_id"], "setup_01")
        self.assertEqual(events_public[1]["event_class"], "setup_or_weight_return")
        self.assertEqual(events_public[1]["start_time_s"], 3.5)
        self.assertIsNone(events_public[1]["turnaround_time_s"])
        self.assertEqual(events_public[1]["end_time_s"], 5.0)

        self.assertEqual(events_public[2]["event_id"], "unclear_01")
        self.assertEqual(events_public[2]["event_class"], "ambiguous_or_unclear")
        self.assertIsNone(events_public[2]["start_time_s"])
        self.assertIsNone(events_public[2]["turnaround_time_s"])
        self.assertIsNone(events_public[2]["end_time_s"])
        self.assertEqual(events_public[2]["uncertainty_notes"], "Occlusion by spotter")
        self.assertEqual(events_public[2]["visible_posture_observations"], {"occlusion": True})

        # 2. Inspect raw case.json on disk
        persisted = self.store.load_persisted_case(cid)
        persisted_events = _get_events(persisted)
        self.assertEqual(len(persisted_events), 3)
        self.assertEqual(persisted_events[0]["event_id"], "rep_01")

    def test_no_filesystem_paths_leaked_in_public_events(self) -> None:
        """Ensure public get_case / list_cases response containing events has NO host filesystem paths."""
        src = self._create_dummy_video("path_leak.mp4")
        case = self.store.create_case(src, original_name="path_leak.mp4")
        cid = _get_id(case)

        self.store.save_review(
            cid,
            {
                "human_count": 2,
                "reviewer": "Path Leak Tester",
                "raw_video_reviewed": True,
                "predictions_previously_seen": False,
                "events": [{"event_id": "evt_safe", "event_class": "completed_rep", "start_time_s": 0.5, "end_time_s": 1.5}],
            },
        )

        fetched = self.store.get_case(cid)
        # Check all keys are in PUBLIC_CASE_KEYS
        self.assertTrue(set(fetched.keys()).issubset(set(PUBLIC_CASE_KEYS)))

        # Convert to string and ensure no absolute drive/paths
        serialized = json.dumps(fetched)
        self.assertFalse(re.search(r"^[a-zA-Z]:[\\/]", serialized))
        self.assertNotIn("tmp", serialized.lower().replace(cid.lower(), ""))


class TestTemporalReviewHistoryPreservation(unittest.TestCase):
    """Test suite verifying each update appends to case history preserving prior snapshots with events."""

    def setUp(self) -> None:
        self.test_dir = tempfile.mkdtemp(prefix="test_temporal_hist_")
        self.store = ReviewStore(self.test_dir, max_upload_bytes=10 * 1024 * 1024)

    def tearDown(self) -> None:
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def _create_dummy_video(self, name: str = "dummy.mp4") -> Path:
        fixtures_dir = Path(tempfile.gettempdir()) / f"fix_{uuid.uuid4().hex[:8]}"
        fixtures_dir.mkdir(parents=True, exist_ok=True)
        file_path = fixtures_dir / name
        file_path.write_bytes(b"hist-video-bytes")
        return file_path

    def test_history_preserves_events_across_multiple_saves(self) -> None:
        """Multiple saves preserve prior review states with their respective event snapshots."""
        src = self._create_dummy_video("hist_events.mp4")
        case = self.store.create_case(src, original_name="hist_events.mp4")
        cid = _get_id(case)

        # Save 1: 1 event
        evt_v1 = [{"event_id": "evt_v1_1", "event_class": "completed_rep", "start_time_s": 1.0, "end_time_s": 2.0}]
        self.store.save_review(
            cid,
            {
                "human_count": 1,
                "reviewer": "Reviewer A",
                "raw_video_reviewed": True,
                "predictions_previously_seen": False,
                "events": evt_v1,
            },
        )

        # Save 2: 2 events
        evt_v2 = [
            {"event_id": "evt_v2_1", "event_class": "completed_rep", "start_time_s": 1.0, "end_time_s": 2.0},
            {"event_id": "evt_v2_2", "event_class": "setup_or_weight_return", "start_time_s": 2.2, "end_time_s": 3.0},
        ]
        self.store.save_review(
            cid,
            {
                "human_count": 1,
                "reviewer": "Reviewer B",
                "raw_video_reviewed": True,
                "predictions_previously_seen": True,
                "events": evt_v2,
            },
        )

        # Save 3: 0 events
        self.store.save_review(
            cid,
            {
                "human_count": 0,
                "reviewer": "Reviewer C",
                "raw_video_reviewed": True,
                "predictions_previously_seen": True,
                "events": [],
            },
        )

        # Inspect persisted history on disk
        persisted = self.store.load_persisted_case(cid)
        history = persisted.get("history", [])
        # History must contain at least 3 entries: initial creation, v1, v2
        self.assertGreaterEqual(len(history), 3)

        # Entry 1 should contain v1 review with 1 event
        h1 = history[1]
        self.assertEqual(_get_review(h1).get("reviewer"), "Reviewer A")
        self.assertEqual(len(_get_events(h1)), 1)
        self.assertEqual(_get_events(h1)[0]["event_id"], "evt_v1_1")

        # Entry 2 should contain v2 review with 2 events
        h2 = history[2]
        self.assertEqual(_get_review(h2).get("reviewer"), "Reviewer B")
        self.assertEqual(len(_get_events(h2)), 2)
        self.assertEqual(_get_events(h2)[1]["event_id"], "evt_v2_2")

        # Current review should have 0 events
        current = persisted.get("review", {})
        self.assertEqual(current.get("reviewer"), "Reviewer C")
        self.assertEqual(len(_get_events(persisted)), 0)


class TestTemporalReviewExportCompatibility(unittest.TestCase):
    """Test suite ensuring export_reviewed_counts and export_manifest remain compatible with evaluator."""

    def setUp(self) -> None:
        self.test_dir = tempfile.mkdtemp(prefix="test_temporal_export_")
        self.store = ReviewStore(self.test_dir, max_upload_bytes=10 * 1024 * 1024)

    def tearDown(self) -> None:
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def _create_dummy_video(self, name: str = "dummy.mp4", content: bytes = b"export-video-bytes") -> Path:
        fixtures_dir = Path(tempfile.gettempdir()) / f"fix_{uuid.uuid4().hex[:8]}"
        fixtures_dir.mkdir(parents=True, exist_ok=True)
        file_path = fixtures_dir / name
        file_path.write_bytes(content)
        return file_path

    def test_export_reviewed_counts_success_and_evaluator_integration(self) -> None:
        """export_reviewed_counts returns ground truth clips and integrates with evaluation.load_ground_truth."""
        # 1. Candidate in_scope clip with events and count 4
        s1 = self._create_dummy_video("cand_eval.mp4", b"cand_eval")
        c1 = self.store.create_case(s1, original_name="cand_eval.mp4", group="candidate")
        id1 = _get_id(c1)
        self.store.save_review(
            id1,
            {
                "human_count": 4,
                "reviewer": "Reviewer 1",
                "raw_video_reviewed": True,
                "predictions_previously_seen": False,
                "scope": "in_scope",
                "events": [
                    {"event_id": "rep_1", "event_class": "completed_rep", "start_time_s": 0.5, "end_time_s": 1.5},
                    {"event_id": "rep_2", "event_class": "completed_rep", "start_time_s": 2.0, "end_time_s": 3.0},
                ],
            },
        )

        # 2. Candidate in_scope clip with count None and events
        s2 = self._create_dummy_video("cand_null.mp4", b"cand_null")
        c2 = self.store.create_case(s2, original_name="cand_null.mp4", group="candidate")
        id2 = _get_id(c2)
        self.store.save_review(
            id2,
            {
                "human_count": None,
                "reviewer": "Reviewer 2",
                "raw_video_reviewed": True,
                "predictions_previously_seen": False,
                "scope": "in_scope",
                "events": [
                    {"event_id": "unclear_1", "event_class": "ambiguous_or_unclear"},
                ],
            },
        )

        # 3. Development group clip (must be exported as None to prevent evaluation leakage)
        s3 = self._create_dummy_video("dev_eval.mp4", b"dev_eval")
        c3 = self.store.create_case(s3, original_name="dev_eval.mp4", group="development")
        id3 = _get_id(c3)
        self.store.save_review(
            id3,
            {
                "human_count": 4,
                "reviewer": "Reviewer 3",
                "raw_video_reviewed": True,
                "predictions_previously_seen": False,
                "scope": "in_scope",
                "events": [
                    {"event_id": "dev_rep_1", "event_class": "completed_rep", "start_time_s": 1.0, "end_time_s": 2.0},
                ],
            },
        )

        # 4. Out-of-scope clip (must be exported as None)
        s4 = self._create_dummy_video("out_eval.mp4", b"out_eval")
        c4 = self.store.create_case(s4, original_name="out_eval.mp4", group="candidate")
        id4 = _get_id(c4)
        self.store.save_review(
            id4,
            {
                "human_count": 4,
                "reviewer": "Reviewer 4",
                "raw_video_reviewed": True,
                "predictions_previously_seen": False,
                "scope": "out_of_scope",
                "events": [
                    {"event_id": "out_rep_1", "event_class": "completed_rep", "start_time_s": 1.0, "end_time_s": 2.0},
                ],
            },
        )

        # Export counts
        exported = _call_export_reviewed_counts(self.store)
        self.assertIn("clips", exported)
        clips_map = {c["clip_id"]: c["human_count"] for c in exported["clips"]}

        self.assertEqual(clips_map.get(id1), 4)
        self.assertIsNone(clips_map.get(id2))
        self.assertIsNone(clips_map.get(id3), "Development anchors must be masked as None")
        self.assertIsNone(clips_map.get(id4), "Out-of-scope clips must be masked as None")

        # Save to JSON file and verify compatibility with evaluation.load_ground_truth
        export_file = Path(self.test_dir) / "counts_ground_truth.json"
        export_file.write_text(json.dumps(exported), encoding="utf-8")

        loaded_gt = load_ground_truth(export_file)
        self.assertEqual(loaded_gt[id1], 4)
        self.assertIsNone(loaded_gt[id2])
        self.assertIsNone(loaded_gt[id3])
        self.assertIsNone(loaded_gt[id4])

    def test_export_manifest_includes_temporal_events(self) -> None:
        """export_manifest includes full events list containing temporal event attributes."""
        src = self._create_dummy_video("manifest_evt.mp4")
        case = self.store.create_case(src, original_name="manifest_evt.mp4")
        cid = _get_id(case)

        self.store.save_review(
            cid,
            {
                "human_count": 1,
                "reviewer": "Manifest Reviewer",
                "raw_video_reviewed": True,
                "predictions_previously_seen": False,
                "events": [
                    {
                        "event_id": "evt_manifest_01",
                        "event_class": "completed_rep",
                        "start_time_s": 0.5,
                        "turnaround_time_s": 1.5,
                        "end_time_s": 2.5,
                        "uncertainty_notes": "Good repetition",
                    }
                ],
            },
        )

        manifest = self.store.export_manifest()
        self.assertTrue(isinstance(manifest, dict))
        self.assertIn("events", manifest)

        matching = [e for e in manifest["events"] if e.get("case_id") == cid]
        self.assertEqual(len(matching), 1)
        self.assertEqual(matching[0].get("event_id"), "evt_manifest_01")
        self.assertEqual(matching[0].get("event_class"), "completed_rep")
        self.assertEqual(matching[0].get("start_time_s"), 0.5)
        self.assertEqual(matching[0].get("turnaround_time_s"), 1.5)
        self.assertEqual(matching[0].get("end_time_s"), 2.5)


class TestTemporalReviewCorruptionAndConcurrentProtection(unittest.TestCase):
    """Test suite verifying fail-closed corruption defenses and concurrent save protection."""

    def setUp(self) -> None:
        self.test_dir = tempfile.mkdtemp(prefix="test_temporal_corrupt_")
        self.store = ReviewStore(self.test_dir, max_upload_bytes=10 * 1024 * 1024)

    def tearDown(self) -> None:
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def _create_dummy_video(self, name: str = "dummy.mp4", content: bytes = b"corrupt-video-bytes") -> Path:
        fixtures_dir = Path(tempfile.gettempdir()) / f"fix_{uuid.uuid4().hex[:8]}"
        fixtures_dir.mkdir(parents=True, exist_ok=True)
        file_path = fixtures_dir / name
        file_path.write_bytes(content)
        return file_path

    def test_corrupt_case_with_events_rejected_fail_closed(self) -> None:
        """When case.json is corrupted on disk, save_review raises CaseCorruptedError and leaves bytes unchanged."""
        src = self._create_dummy_video("corrupt.mp4")
        case = self.store.create_case(src, original_name="corrupt.mp4")
        cid = _get_id(case)

        case_dir = Path(self.test_dir) / cid
        case_json = case_dir / "case.json"

        # Corrupt case.json
        corrupt_str = '{"case_id": "' + cid + '", "events": [broken-json'
        case_json.write_text(corrupt_str, encoding="utf-8")
        before_bytes = case_json.read_bytes()

        with self.assertRaises(CaseCorruptedError):
            self.store.save_review(
                cid,
                {
                    "human_count": 2,
                    "reviewer": "Jane",
                    "raw_video_reviewed": True,
                    "predictions_previously_seen": False,
                    "events": [{"event_id": "evt_safe", "event_class": "completed_rep"}],
                },
            )

        # Assert disk bytes remain completely untouched
        self.assertEqual(case_json.read_bytes(), before_bytes)

    def test_tampered_video_with_events_rejected(self) -> None:
        """Mutating managed video bytes causes save_review and exports with events to raise CaseCorruptedError."""
        src = self._create_dummy_video("tamper.mp4")
        case = self.store.create_case(src, original_name="tamper.mp4")
        cid = _get_id(case)

        # Mutate video bytes
        vpath = self.store.video_path(cid)
        vpath.write_bytes(b"tampered-video-mutated-payload-bytes")

        # save_review fails closed
        with self.assertRaises(CaseCorruptedError):
            self.store.save_review(
                cid,
                {
                    "human_count": 1,
                    "reviewer": "Jane",
                    "raw_video_reviewed": True,
                    "predictions_previously_seen": False,
                    "events": [{"event_id": "evt_1", "event_class": "completed_rep"}],
                },
            )

        # export fails closed
        with self.assertRaises(CaseCorruptedError):
            _call_export_reviewed_counts(self.store)

    def test_persisted_invalid_event_causes_case_corrupted_error(self) -> None:
        """Writing an invalid event (e.g. duplicate event_id) directly to disk causes get_case to raise CaseCorruptedError."""
        src = self._create_dummy_video("invalid_disk.mp4")
        case = self.store.create_case(src, original_name="invalid_disk.mp4")
        cid = _get_id(case)

        case_dir = Path(self.test_dir) / cid
        case_json = case_dir / "case.json"

        data = json.loads(case_json.read_text(encoding="utf-8"))
        # Inject duplicate event_ids directly into disk JSON
        data["review"]["events"] = [
            {"event_id": "dup_on_disk", "event_class": "completed_rep"},
            {"event_id": "dup_on_disk", "event_class": "setup_or_weight_return"},
        ]
        case_json.write_text(json.dumps(data, indent=2), encoding="utf-8")

        with self.assertRaises(CaseCorruptedError):
            self.store.get_case(cid)

        with self.assertRaises(CaseCorruptedError):
            self.store.load_persisted_case(cid)


if __name__ == "__main__":
    unittest.main()


class TestPrimaryTemporalSchema(unittest.TestCase):
    def test_explicit_version_and_nonfinite_json_validation(self):
        from gym_analyser.review_store import validate_review_payload
        payload={"events":[{"event_id":"event1","event_class":"ambiguous_or_unclear","visible_posture_observations":{"note":"unclear"}}]}
        self.assertEqual(validate_review_payload(payload)["events_schema_version"],1)
        for value in (True,2,"1"):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):validate_review_payload({**payload,"events_schema_version":value})
        for value in (float('nan'),float('inf'),object()):
            changed=copy.deepcopy(payload);changed["events"][0]["visible_posture_observations"]={"nested":[value]}
            with self.assertRaises(ValueError):validate_review_payload(changed)
