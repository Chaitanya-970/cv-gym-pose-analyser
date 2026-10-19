"""Comprehensive unit test suite for ReviewStore in gym_analyser.review_store.

Tests:
1. Case creation with dummy .mp4: creates case, UUID4 ID, copies file, computes SHA256,
   leaves original file untouched.
2. Invalid extensions (.txt, .avi, .exe) rejected with ValueError.
3. Oversize upload rejected with ValueError.
4. No filesystem paths in public responses: get_case and list_cases return NO raw filesystem paths.
5. video_path security: path traversal (../../) rejected, non-UUID rejected, missing file raises FileNotFoundError.
6. Review saving:
   - Valid review (human_count=3, reviewer='Jane Doe', raw_video_reviewed=True, predictions_previously_seen=False) succeeds.
   - human_count=0 saved distinctly as 0 (not None).
   - human_count=None saved as None.
   - Invalid counts rejected with ValueError: negative (-1), boolean (True, False), float (2.5), string ('3').
   - Missing reviewer rejected when count is set (human_count=3, reviewer='').
   - Missing acknowledgment rejected when count is set (human_count=3, raw_video_reviewed=False).
   - Invalid scope enum rejected.
   - Invalid event times rejected: negative, end < start, float('nan'), float('inf').
   - Invalid event enums rejected: intent, cycle.
7. History preservation: multiple updates preserve previous reviews in history with saved_at timestamps.
8. export_counts:
   - candidate in_scope with count 5 -> exported as 5.
   - candidate in_scope with count None -> exported as None.
   - development group with count 5 -> exported as None! (developer anchor excluded).
   - candidate out_of_scope with count 5 -> exported as None! (out of scope excluded).
   - Integration with evaluation.load_ground_truth: export to JSON, call evaluation.load_ground_truth(path),
     assert loaded dictionary matches expectations.
9. export_manifest: includes full details, events, history.
10. Corrupted metadata fail-closed: malformed JSON causes get_case to raise and list_cases to skip safely.
11. Regression tests:
   - test_invalid_metadata_rejected
   - test_unallowlisted_metadata_fields_omitted
   - test_failed_save_leaves_bytes_unchanged
   - test_raw_tamper_rejection
   - test_omitted_predictions_answer_rejected
   - test_duplicate_event_indices_rejected
   - test_symlink_defenses
   - test_max_upload_bytes_validation
   - test_streaming_size_abort
   - test_cwd_independent_default_storage
   - test_ground_truth_export_integration
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import re
import shutil
import tempfile
import unittest
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from gym_analyser.evaluation import load_ground_truth
from gym_analyser.review_store import (
    PUBLIC_CASE_KEYS,
    CaseCorruptedError,
    CaseNotFoundError,
    InvalidCaseIdError,
    ReviewStore,
)


def _has_raw_filesystem_path(val: Any) -> bool:
    """Recursively check if a value contains raw host filesystem paths."""
    if isinstance(val, str):
        # Check Windows drive letter like C:\ or C:/
        if re.search(r"^[a-zA-Z]:[\\/]", val):
            return True
        # Check absolute Unix path starting with /
        if val.startswith("/") and len(val) > 1 and not val.startswith("//"):
            return True
        # Check for user home or temp directory paths
        temp_dir = tempfile.gettempdir()
        if temp_dir.lower() in val.lower():
            return True
    elif isinstance(val, dict):
        for k, v in val.items():
            if _has_raw_filesystem_path(k) or _has_raw_filesystem_path(v):
                return True
    elif isinstance(val, (list, tuple, set)):
        for item in val:
            if _has_raw_filesystem_path(item):
                return True
    return False


def _get_id(case_dict: Dict[str, Any]) -> str:
    """Helper to extract case/clip ID from case dict."""
    cid = case_dict.get("case_id") or case_dict.get("id") or case_dict.get("clip_id")
    if not cid:
        raise KeyError(f"No ID found in case dict keys: {list(case_dict.keys())}")
    return str(cid)


def _get_review(case_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Helper to extract review dict whether nested or flat."""
    if "review" in case_dict and isinstance(case_dict["review"], dict):
        return case_dict["review"]
    return case_dict


def _get_count(case_dict: Dict[str, Any]) -> Any:
    """Helper to extract human_count whether nested or flat."""
    rev = _get_review(case_dict)
    return rev.get("human_count")


def _get_sha(case_dict: Dict[str, Any]) -> Any:
    """Helper to extract sha256 whether named source_sha256 or sha256."""
    return case_dict.get("source_sha256") or case_dict.get("sha256")


class TestReviewStore(unittest.TestCase):
    """Test pure standard library ReviewStore implementation."""

    def setUp(self) -> None:
        self.test_dir = tempfile.mkdtemp(prefix="test_review_store_")
        self.store = ReviewStore(self.test_dir, max_upload_bytes=10 * 1024 * 1024)

    def tearDown(self) -> None:
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def _create_dummy_video(
        self,
        name: str = "dummy.mp4",
        content: bytes = b"fake-mp4-test-content-header-data",
    ) -> Path:
        """Create a temporary video file outside the store directory."""
        fixtures_dir = Path(tempfile.gettempdir()) / f"fixtures_{uuid.uuid4().hex[:8]}"
        fixtures_dir.mkdir(parents=True, exist_ok=True)
        file_path = fixtures_dir / name
        file_path.write_bytes(content)
        return file_path

    # 1. Case creation
    def test_create_case_success(self) -> None:
        content = b"valid-mp4-content-data-12345"
        src_path = self._create_dummy_video("sample.mp4", content)
        expected_sha = hashlib.sha256(content).hexdigest()

        case = self.store.create_case(src_path, original_name="custom_sample.mp4", group="candidate")
        case_id = _get_id(case)

        # Must generate valid UUID4
        parsed_uuid = uuid.UUID(case_id)
        self.assertEqual(parsed_uuid.version, 4)

        # Check SHA256 computed
        self.assertEqual(_get_sha(case), expected_sha)

        # Original file untouched
        self.assertTrue(src_path.exists())
        self.assertEqual(src_path.read_bytes(), content)

        # Managed video file exists and matches content
        managed_video = self.store.video_path(case_id)
        self.assertTrue(managed_video.exists())
        self.assertEqual(managed_video.read_bytes(), content)

    # 2. Invalid extensions rejected
    def test_invalid_extensions_rejected(self) -> None:
        for bad_ext in [".txt", ".avi", ".exe", ".sh", ".json"]:
            src = self._create_dummy_video(f"file{bad_ext}", b"bad-extension-content")
            with self.assertRaises(ValueError):
                self.store.create_case(src, original_name=f"file{bad_ext}")

    def test_allowed_extensions_accepted(self) -> None:
        for good_ext in [".mp4", ".webm", ".mov"]:
            src = self._create_dummy_video(f"clip{good_ext}", b"good-video-content")
            case = self.store.create_case(src, original_name=f"clip{good_ext}")
            self.assertIsNotNone(_get_id(case))

    # 3. Oversize upload rejected
    def test_oversize_upload_rejected(self) -> None:
        small_store = ReviewStore(self.test_dir, max_upload_bytes=100)
        oversize_src = self._create_dummy_video("large.mp4", b"X" * 150)
        with self.assertRaises(ValueError):
            small_store.create_case(oversize_src, original_name="large.mp4")

    # 4. No filesystem paths in public responses
    def test_no_filesystem_paths_in_public_responses(self) -> None:
        src = self._create_dummy_video("safe.mp4", b"data")
        case = self.store.create_case(src, original_name="safe.mp4")
        case_id = _get_id(case)

        # Check create_case response
        self.assertFalse(_has_raw_filesystem_path(case))

        # Check get_case response
        fetched_case = self.store.get_case(case_id)
        self.assertFalse(_has_raw_filesystem_path(fetched_case))

        # Check list_cases response
        cases_list = self.store.list_cases()
        self.assertFalse(_has_raw_filesystem_path(cases_list))

    # 5. video_path security
    def test_video_path_security(self) -> None:
        # Path traversal rejected
        with self.assertRaises((ValueError, PermissionError, InvalidCaseIdError)):
            self.store.video_path("../../etc/passwd")

        with self.assertRaises((ValueError, PermissionError, InvalidCaseIdError)):
            self.store.video_path("..\\..\\Windows\\System32\\cmd.exe")

        # Non-UUID rejected
        with self.assertRaises((ValueError, InvalidCaseIdError)):
            self.store.video_path("not-a-valid-uuid")

        with self.assertRaises((ValueError, InvalidCaseIdError)):
            self.store.video_path("12345")

        # Valid UUID but non-existent case raises (FileNotFoundError, CaseNotFoundError)
        missing_uuid = str(uuid.uuid4())
        with self.assertRaises((FileNotFoundError, CaseNotFoundError)):
            self.store.video_path(missing_uuid)

    # 6. Review saving: valid and invalid inputs
    def test_review_saving_valid(self) -> None:
        src = self._create_dummy_video("rev.mp4", b"data")
        case = self.store.create_case(src, original_name="rev.mp4")
        case_id = _get_id(case)

        # Valid review
        review_data = {
            "human_count": 3,
            "reviewer": "Jane Doe",
            "raw_video_reviewed": True,
            "predictions_previously_seen": False,
            "scope": "in_scope",
            "notes": "Good clean squats",
            "events": [
                {
                    "event_index": 0,
                    "start_time_s": 1.0,
                    "end_time_s": 2.5,
                    "intent": "intentional_squat",
                    "cycle": "complete",
                    "observation": "Good depth",
                }
            ],
        }
        res = self.store.save_review(case_id, review_data)
        self.assertEqual(_get_count(res), 3)
        self.assertEqual(_get_review(res).get("reviewer"), "Jane Doe")
        self.assertTrue(_get_review(res).get("raw_video_reviewed"))
        self.assertIs(_get_review(res).get("predictions_previously_seen"), False)

    def test_human_count_zero_distinct_from_none(self) -> None:
        src = self._create_dummy_video("zero.mp4", b"data")
        case = self.store.create_case(src, original_name="zero.mp4")
        case_id = _get_id(case)

        # Count 0 is distinctly 0, not None
        self.store.save_review(
            case_id,
            {
                "human_count": 0,
                "reviewer": "Jane Doe",
                "raw_video_reviewed": True,
                "predictions_previously_seen": False,
                "scope": "in_scope",
            },
        )
        case_zero = self.store.get_case(case_id)
        self.assertIs(_get_count(case_zero), 0)

        # Count None is explicitly None
        self.store.save_review(
            case_id,
            {
                "human_count": None,
                "reviewer": "Jane Doe",
                "raw_video_reviewed": True,
                "predictions_previously_seen": False,
                "scope": "in_scope",
            },
        )
        case_none = self.store.get_case(case_id)
        self.assertIsNone(_get_count(case_none))

    def test_invalid_counts_rejected(self) -> None:
        src = self._create_dummy_video("counts.mp4", b"data")
        case = self.store.create_case(src, original_name="counts.mp4")
        case_id = _get_id(case)

        bad_counts = [-1, True, False, 2.5, "3", [3], {"count": 3}]
        for bad_count in bad_counts:
            with self.assertRaises(ValueError):
                self.store.save_review(
                    case_id,
                    {
                        "human_count": bad_count,
                        "reviewer": "Jane",
                        "raw_video_reviewed": True,
                        "predictions_previously_seen": False,
                    },
                )

    def test_missing_reviewer_rejected_when_count_set(self) -> None:
        src = self._create_dummy_video("norev.mp4", b"data")
        case = self.store.create_case(src, original_name="norev.mp4")
        case_id = _get_id(case)

        for empty_reviewer in ["", "   ", None]:
            with self.assertRaises(ValueError):
                self.store.save_review(
                    case_id,
                    {
                        "human_count": 3,
                        "reviewer": empty_reviewer,
                        "raw_video_reviewed": True,
                        "predictions_previously_seen": False,
                    },
                )

    def test_missing_acknowledgment_rejected_when_count_set(self) -> None:
        src = self._create_dummy_video("noack.mp4", b"data")
        case = self.store.create_case(src, original_name="noack.mp4")
        case_id = _get_id(case)

        with self.assertRaises(ValueError):
            self.store.save_review(
                case_id,
                {
                    "human_count": 3,
                    "reviewer": "Jane Doe",
                    "raw_video_reviewed": False,
                    "predictions_previously_seen": False,
                },
            )

        with self.assertRaises(ValueError):
            self.store.save_review(
                case_id,
                {"human_count": 3, "reviewer": "Jane Doe", "predictions_previously_seen": False},
            )

    def test_invalid_scope_enum_rejected(self) -> None:
        src = self._create_dummy_video("scope.mp4", b"data")
        case = self.store.create_case(src, original_name="scope.mp4")
        case_id = _get_id(case)

        with self.assertRaises(ValueError):
            self.store.save_review(
                case_id,
                {
                    "human_count": 3,
                    "reviewer": "Jane",
                    "raw_video_reviewed": True,
                    "predictions_previously_seen": False,
                    "scope": "invalid_scope",
                },
            )

        # Valid scopes: in_scope, out_of_scope, uncertain
        for valid_scope in ["in_scope", "out_of_scope", "uncertain"]:
            res = self.store.save_review(
                case_id,
                {
                    "human_count": 3,
                    "reviewer": "Jane",
                    "raw_video_reviewed": True,
                    "predictions_previously_seen": False,
                    "scope": valid_scope,
                },
            )
            self.assertEqual(_get_review(res).get("scope"), valid_scope)

    def test_invalid_event_times_rejected(self) -> None:
        src = self._create_dummy_video("evt.mp4", b"data")
        case = self.store.create_case(src, original_name="evt.mp4")
        case_id = _get_id(case)

        bad_times = [
            (-1.0, 2.0),  # negative start
            (1.0, -2.0),  # negative end
            (5.0, 2.0),   # end < start
            (float("nan"), 2.0),
            (1.0, float("nan")),
            (float("inf"), 2.0),
            (1.0, float("inf")),
        ]

        for s, e in bad_times:
            with self.assertRaises(ValueError):
                self.store.save_review(
                    case_id,
                    {
                        "human_count": 1,
                        "reviewer": "Jane",
                        "raw_video_reviewed": True,
                        "predictions_previously_seen": False,
                        "events": [
                            {
                                "event_index": 0,
                                "start_time_s": s,
                                "end_time_s": e,
                                "intent": "intentional_squat",
                                "cycle": "complete",
                            }
                        ],
                    },
                )

    def test_invalid_event_enums_rejected(self) -> None:
        src = self._create_dummy_video("evtenum.mp4", b"data")
        case = self.store.create_case(src, original_name="evtenum.mp4")
        case_id = _get_id(case)

        # Bad intent
        with self.assertRaises(ValueError):
            self.store.save_review(
                case_id,
                {
                    "human_count": 1,
                    "reviewer": "Jane",
                    "raw_video_reviewed": True,
                    "predictions_previously_seen": False,
                    "events": [
                        {
                            "event_index": 0,
                            "start_time_s": 0.0,
                            "end_time_s": 1.0,
                            "intent": "bad_intent",
                            "cycle": "complete",
                        }
                    ],
                },
            )

        # Bad cycle
        with self.assertRaises(ValueError):
            self.store.save_review(
                case_id,
                {
                    "human_count": 1,
                    "reviewer": "Jane",
                    "raw_video_reviewed": True,
                    "predictions_previously_seen": False,
                    "events": [
                        {
                            "event_index": 0,
                            "start_time_s": 0.0,
                            "end_time_s": 1.0,
                            "intent": "intentional_squat",
                            "cycle": "bad_cycle",
                        }
                    ],
                },
            )

    # 7. History preservation
    def test_history_preservation_on_multiple_saves(self) -> None:
        src = self._create_dummy_video("hist.mp4", b"data")
        case = self.store.create_case(src, original_name="hist.mp4")
        case_id = _get_id(case)

        # First save
        self.store.save_review(
            case_id,
            {
                "human_count": 2,
                "reviewer": "Alice",
                "raw_video_reviewed": True,
                "predictions_previously_seen": False,
                "notes": "first pass",
            },
        )
        # Second save
        self.store.save_review(
            case_id,
            {
                "human_count": 3,
                "reviewer": "Bob",
                "raw_video_reviewed": True,
                "predictions_previously_seen": False,
                "notes": "second pass",
            },
        )
        # Third save
        self.store.save_review(
            case_id,
            {
                "human_count": 4,
                "reviewer": "Carol",
                "raw_video_reviewed": True,
                "predictions_previously_seen": False,
                "notes": "third pass",
            },
        )

        case_data = self.store.get_case(case_id)
        self.assertEqual(_get_count(case_data), 4)
        self.assertEqual(_get_review(case_data).get("reviewer"), "Carol")
        self.assertEqual(case_data.get("history_count"), 3)

        # Inspect persisted history on disk
        persisted = self.store.load_persisted_case(case_id)
        history = persisted.get("history", [])
        self.assertGreaterEqual(len(history), 3)

        alice_hist = history[1]
        self.assertEqual(_get_count(alice_hist), 2)
        self.assertEqual(_get_review(alice_hist).get("reviewer"), "Alice")
        self.assertIn("saved_at", alice_hist)

        bob_hist = history[2]
        self.assertEqual(_get_count(bob_hist), 3)
        self.assertEqual(_get_review(bob_hist).get("reviewer"), "Bob")
        self.assertIn("saved_at", bob_hist)

    # 8. export_counts and evaluation.load_ground_truth integration
    def test_export_counts_and_evaluation_integration(self) -> None:
        # Case 1: candidate in_scope with count 5 -> exported as 5
        src1 = self._create_dummy_video("cand_in.mp4", b"cand_in")
        c1 = self.store.create_case(src1, original_name="cand_in.mp4", group="candidate")
        id1 = _get_id(c1)
        self.store.save_review(
            id1,
            {
                "human_count": 5,
                "reviewer": "Jane",
                "raw_video_reviewed": True,
                "predictions_previously_seen": False,
                "scope": "in_scope",
            },
        )

        # Case 2: candidate in_scope with count None -> exported as None
        src2 = self._create_dummy_video("cand_none.mp4", b"cand_none")
        c2 = self.store.create_case(src2, original_name="cand_none.mp4", group="candidate")
        id2 = _get_id(c2)
        self.store.save_review(
            id2,
            {
                "human_count": None,
                "reviewer": "Jane",
                "raw_video_reviewed": True,
                "predictions_previously_seen": False,
                "scope": "in_scope",
            },
        )

        # Case 3: development group with count 5 -> exported as None (developer anchor excluded)
        src3 = self._create_dummy_video("dev_anchor.mp4", b"dev_anchor")
        c3 = self.store.create_case(src3, original_name="dev_anchor.mp4", group="development")
        id3 = _get_id(c3)
        self.store.save_review(
            id3,
            {
                "human_count": 5,
                "reviewer": "Jane",
                "raw_video_reviewed": True,
                "predictions_previously_seen": False,
                "scope": "in_scope",
            },
        )

        # Case 4: candidate out_of_scope with count 5 -> exported as None (out of scope excluded)
        src4 = self._create_dummy_video("cand_out.mp4", b"cand_out")
        c4 = self.store.create_case(src4, original_name="cand_out.mp4", group="candidate")
        id4 = _get_id(c4)
        self.store.save_review(
            id4,
            {
                "human_count": 5,
                "reviewer": "Jane",
                "raw_video_reviewed": True,
                "predictions_previously_seen": False,
                "scope": "out_of_scope",
            },
        )

        # Run export_counts
        exported = self.store.export_counts()
        self.assertIn("clips", exported)
        clips_map = {c["clip_id"]: c["human_count"] for c in exported["clips"]}

        self.assertEqual(clips_map.get(id1), 5)
        self.assertIsNone(clips_map.get(id2))
        self.assertIsNone(clips_map.get(id3), "Development anchors must be exported as None")
        self.assertIsNone(clips_map.get(id4), "Out-of-scope clips must be exported as None")

        # Save to JSON and verify compatibility with evaluation.load_ground_truth
        export_file = Path(self.test_dir) / "exported_counts.json"
        export_file.write_text(json.dumps(exported), encoding="utf-8")

        loaded = load_ground_truth(export_file)
        self.assertEqual(loaded[id1], 5)
        self.assertIsNone(loaded[id2])
        self.assertIsNone(loaded[id3])
        self.assertIsNone(loaded[id4])

    # 9. export_manifest
    def test_export_manifest_completeness(self) -> None:
        src = self._create_dummy_video("manifest.mp4", b"manifest")
        c = self.store.create_case(src, original_name="manifest.mp4", group="candidate")
        cid = _get_id(c)
        self.store.save_review(
            cid,
            {
                "human_count": 2,
                "reviewer": "Reviewer X",
                "raw_video_reviewed": True,
                "predictions_previously_seen": False,
                "scope": "in_scope",
                "notes": "Manifest test",
                "events": [
                    {
                        "event_index": 0,
                        "start_time_s": 0.5,
                        "end_time_s": 1.5,
                        "intent": "intentional_squat",
                        "cycle": "complete",
                        "observation": "clean rep",
                    }
                ],
            },
        )

        manifest = self.store.export_manifest()
        self.assertTrue(isinstance(manifest, (dict, list)))
        items = manifest if isinstance(manifest, list) else manifest.get("cases", manifest.get("clips", []))
        found = None
        for item in items:
            if _get_id(item) == cid:
                found = item
                break
        self.assertIsNotNone(found, "Case must exist in exported manifest")
        self.assertEqual(_get_count(found), 2)
        self.assertEqual(_get_review(found).get("reviewer"), "Reviewer X")
        self.assertEqual(len(_get_review(found).get("events", [])), 1)

    # 10. Corrupted metadata fail-closed
    def test_corrupted_metadata_fail_closed(self) -> None:
        src = self._create_dummy_video("corrupt.mp4", b"corrupt")
        c = self.store.create_case(src, original_name="corrupt.mp4")
        cid = _get_id(c)

        case_dir = Path(self.test_dir) / cid
        meta_file = case_dir / "case.json"
        self.assertTrue(meta_file.exists(), f"Metadata file not found in {case_dir}")

        # Corrupt JSON with invalid syntax
        meta_file.write_text("{corrupted json content !@#$%", encoding="utf-8")

        # get_case must raise CaseCorruptedError fail-closed
        with self.assertRaises(CaseCorruptedError):
            self.store.get_case(cid)

        # list_cases must safely skip the corrupted case without crashing
        cases = self.store.list_cases()
        self.assertIsInstance(cases, list)
        case_ids = [_get_id(x) for x in cases]
        self.assertNotIn(cid, case_ids)

    # =========================================================================
    # Task 32 Rigorous Regression Tests
    # =========================================================================

    def test_invalid_metadata_rejected(self) -> None:
        """Verify invalid persisted metadata rejected with CaseCorruptedError."""
        src = self._create_dummy_video("inv_meta.mp4", b"inv-meta-data")
        c = self.store.create_case(src, original_name="inv_meta.mp4")
        cid = _get_id(c)
        case_dir = Path(self.test_dir) / cid
        case_json = case_dir / "case.json"
        self.assertTrue(case_json.exists())

        orig_text = case_json.read_text(encoding="utf-8")

        # 1. Wrong case_id (different from folder UUID)
        c_data = json.loads(orig_text)
        c_data["case_id"] = str(uuid.uuid4()).lower()
        case_json.write_text(json.dumps(c_data, indent=2), encoding="utf-8")
        with self.assertRaises(CaseCorruptedError):
            self.store.get_case(cid)

        # 2. Negative stored count (-1)
        c_data = json.loads(orig_text)
        c_data["review"]["human_count"] = -1
        case_json.write_text(json.dumps(c_data, indent=2), encoding="utf-8")
        with self.assertRaises(CaseCorruptedError):
            self.store.get_case(cid)

        # 3. Missing acknowledgment when count is set
        c_data = json.loads(orig_text)
        c_data["review"]["human_count"] = 3
        c_data["review"]["reviewer"] = "Jane Doe"
        c_data["review"]["raw_video_reviewed"] = False
        c_data["review"]["predictions_previously_seen"] = False
        case_json.write_text(json.dumps(c_data, indent=2), encoding="utf-8")
        with self.assertRaises(CaseCorruptedError):
            self.store.get_case(cid)

        # 4. Duplicate JSON keys in case.json
        dup_json_str = f'{{"case_id": "{cid}", "case_id": "{cid}", "group": "candidate"}}'
        case_json.write_text(dup_json_str, encoding="utf-8")
        with self.assertRaises(CaseCorruptedError):
            self.store.get_case(cid)

    def test_unallowlisted_metadata_fields_omitted(self) -> None:
        """Verify get_case, list_cases, create_case, save_review return ONLY allowlisted public fields."""
        src = self._create_dummy_video("unallow.mp4", b"unallow-content")
        c = self.store.create_case(src, original_name="unallow.mp4")
        cid = _get_id(c)
        case_dir = Path(self.test_dir) / cid
        case_json = case_dir / "case.json"

        # Inject extra internal/secret fields into case.json on disk
        c_data = json.loads(case_json.read_text(encoding="utf-8"))
        c_data["internal_path"] = "C:\\secret\\host\\path\\video.mp4"
        c_data["secret_key"] = "super-secret-token-12345"
        case_json.write_text(json.dumps(c_data, indent=2), encoding="utf-8")

        # 1. get_case omits extra fields and contains only public keys
        fetched = self.store.get_case(cid)
        self.assertNotIn("internal_path", fetched)
        self.assertNotIn("secret_key", fetched)
        self.assertTrue(set(fetched.keys()).issubset(set(PUBLIC_CASE_KEYS)))

        # 2. list_cases omits extra fields
        listed = self.store.list_cases()
        matched = [x for x in listed if _get_id(x) == cid]
        self.assertEqual(len(matched), 1)
        self.assertNotIn("internal_path", matched[0])
        self.assertNotIn("secret_key", matched[0])

        # 3. create_case returns only allowlisted keys
        src2 = self._create_dummy_video("unallow2.mp4", b"unallow2-content")
        c2 = self.store.create_case(src2, original_name="unallow2.mp4")
        self.assertNotIn("internal_path", c2)
        self.assertNotIn("secret_key", c2)
        self.assertTrue(set(c2.keys()).issubset(set(PUBLIC_CASE_KEYS)))

        # 4. save_review returns only allowlisted keys
        saved = self.store.save_review(
            cid,
            {
                "human_count": 2,
                "reviewer": "Jane",
                "raw_video_reviewed": True,
                "predictions_previously_seen": False,
            },
        )
        self.assertNotIn("internal_path", saved)
        self.assertNotIn("secret_key", saved)
        self.assertTrue(set(saved.keys()).issubset(set(PUBLIC_CASE_KEYS)))

    def test_failed_save_leaves_bytes_unchanged(self) -> None:
        """When case.json is corrupted or invalid, save_review raises CaseCorruptedError and does NOT overwrite."""
        src = self._create_dummy_video("corrupt_save.mp4", b"corrupt-save-data")
        c = self.store.create_case(src, original_name="corrupt_save.mp4")
        cid = _get_id(c)
        case_dir = Path(self.test_dir) / cid
        case_json = case_dir / "case.json"

        # Corrupt case.json with invalid JSON syntax
        corrupt_content = '{"case_id": "' + cid + '", "corrupted": [unclosed'
        case_json.write_text(corrupt_content, encoding="utf-8")
        before_bytes = case_json.read_bytes()

        with self.assertRaises(CaseCorruptedError):
            self.store.save_review(
                cid,
                {
                    "human_count": 3,
                    "reviewer": "Jane",
                    "raw_video_reviewed": True,
                    "predictions_previously_seen": False,
                },
            )

        # Assert disk bytes remain completely untouched
        self.assertEqual(case_json.read_bytes(), before_bytes)

    def test_raw_tamper_rejection(self) -> None:
        """Mutating bytes in managed video causes video_path, save_review, and exports to raise CaseCorruptedError."""
        src = self._create_dummy_video("tamper.mp4", b"tamper-test-bytes")
        c = self.store.create_case(src, original_name="tamper.mp4")
        cid = _get_id(c)

        # Locate managed video file and mutate its bytes
        vpath = self.store.video_path(cid)
        vpath.write_bytes(b"tampered-video-mutated-payload-bytes")

        # 1. video_path fails closed
        with self.assertRaises(CaseCorruptedError):
            self.store.video_path(cid)

        # 2. save_review raises CaseCorruptedError
        with self.assertRaises(CaseCorruptedError):
            self.store.save_review(
                cid,
                {
                    "human_count": 4,
                    "reviewer": "Jane",
                    "raw_video_reviewed": True,
                    "predictions_previously_seen": False,
                },
            )

        # 3. export_counts and export_manifest raise CaseCorruptedError fail-closed
        with self.assertRaises(CaseCorruptedError):
            self.store.export_counts()

        with self.assertRaises(CaseCorruptedError):
            self.store.export_manifest()

    def test_omitted_predictions_answer_rejected(self) -> None:
        """When human_count is set, predictions_previously_seen is strictly required (rejecting None/omitted)."""
        src = self._create_dummy_video("preds.mp4", b"preds-bytes")
        c = self.store.create_case(src, original_name="preds.mp4")
        cid = _get_id(c)

        # Omitting predictions_previously_seen when count is set raises ValueError
        with self.assertRaises(ValueError):
            self.store.save_review(
                cid,
                {"human_count": 5, "reviewer": "Jane Doe", "raw_video_reviewed": True},
            )

        # Passing predictions_previously_seen=None when count is set raises ValueError
        with self.assertRaises(ValueError):
            self.store.save_review(
                cid,
                {
                    "human_count": 5,
                    "reviewer": "Jane Doe",
                    "raw_video_reviewed": True,
                    "predictions_previously_seen": None,
                },
            )

        # Explicit True is accepted and preserved
        res_true = self.store.save_review(
            cid,
            {
                "human_count": 5,
                "reviewer": "Jane Doe",
                "raw_video_reviewed": True,
                "predictions_previously_seen": True,
            },
        )
        self.assertIs(_get_review(res_true).get("predictions_previously_seen"), True)
        case_true = self.store.get_case(cid)
        self.assertIs(_get_review(case_true).get("predictions_previously_seen"), True)

        # Explicit False is accepted and preserved
        res_false = self.store.save_review(
            cid,
            {
                "human_count": 5,
                "reviewer": "Jane Doe",
                "raw_video_reviewed": True,
                "predictions_previously_seen": False,
            },
        )
        self.assertIs(_get_review(res_false).get("predictions_previously_seen"), False)
        case_false = self.store.get_case(cid)
        self.assertIs(_get_review(case_false).get("predictions_previously_seen"), False)

    def test_duplicate_event_indices_rejected(self) -> None:
        """Passing events with duplicate event_index raises ValueError."""
        src = self._create_dummy_video("dup_evt.mp4", b"dup-event-data")
        c = self.store.create_case(src, original_name="dup_evt.mp4")
        cid = _get_id(c)

        duplicate_events = [
            {
                "event_index": 0,
                "start_time_s": 0.5,
                "end_time_s": 1.5,
                "intent": "intentional_squat",
                "cycle": "complete",
            },
            {
                "event_index": 0,  # Duplicate event_index
                "start_time_s": 2.0,
                "end_time_s": 3.0,
                "intent": "intentional_squat",
                "cycle": "complete",
            },
        ]

        with self.assertRaises(ValueError):
            self.store.save_review(
                cid,
                {
                    "human_count": 2,
                    "reviewer": "Jane",
                    "raw_video_reviewed": True,
                    "predictions_previously_seen": False,
                    "events": duplicate_events,
                },
            )

    def test_symlink_defenses(self) -> None:
        """Test rejection of symlinks for source file or contained case files (platform-skipped if unsupported)."""
        src = self._create_dummy_video("sym_target.mp4", b"sym-content")
        sym_src = Path(tempfile.gettempdir()) / f"sym_link_{uuid.uuid4().hex[:8]}.mp4"

        try:
            os.symlink(src, sym_src)
        except (OSError, NotImplementedError, AttributeError):
            self.skipTest("Symlinks not supported or insufficient privileges on this system")

        try:
            # Source file symlink rejected
            with self.assertRaises((ValueError, PermissionError, OSError)):
                self.store.create_case(sym_src, original_name="sym_link.mp4")

            # Contained file symlink defense
            valid_case = self.store.create_case(src, original_name="valid_for_sym.mp4")
            valid_id = _get_id(valid_case)
            case_dir = Path(self.test_dir) / valid_id
            case_json = case_dir / "case.json"

            ext_target = Path(tempfile.gettempdir()) / f"ext_target_{uuid.uuid4().hex[:8]}.json"
            ext_target.write_text("{}", encoding="utf-8")
            try:
                case_json.unlink()
                os.symlink(ext_target, case_json)
                with self.assertRaises((CaseCorruptedError, ValueError, PermissionError)):
                    self.store.get_case(valid_id)
            finally:
                if ext_target.exists():
                    ext_target.unlink()
        finally:
            if sym_src.exists() or sym_src.is_symlink():
                sym_src.unlink()

    def test_max_upload_bytes_validation(self) -> None:
        """Passing boolean, float, negative, or zero to ReviewStore max_upload_bytes raises ValueError."""
        invalid_limits = [True, False, 10.5, -100, 0, -1]
        for bad_val in invalid_limits:
            with self.assertRaises(ValueError):
                ReviewStore(self.test_dir, max_upload_bytes=bad_val)

    def test_streaming_size_abort(self) -> None:
        """Enforces size limit DURING copy, cleaning up temp files immediately."""
        small_store = ReviewStore(self.test_dir, max_upload_bytes=100)
        oversize_src = self._create_dummy_video("large_stream.mp4", b"S" * 300)

        with self.assertRaises(ValueError):
            small_store.create_case(oversize_src, original_name="large_stream.mp4")

        # Verify no .tmp files remain on disk
        temp_files = list(Path(self.test_dir).rglob("*.tmp"))
        self.assertEqual(temp_files, [], f"Found uncleaned temporary files: {temp_files}")

    def test_cwd_independent_default_storage(self) -> None:
        """Verify default storage root resolves to gym_pose/output/human_reviews regardless of current directory."""
        orig_cwd = os.getcwd()
        try:
            store_default = ReviewStore()
            resolved_root = store_default.root.resolve()
            normalized_path = str(resolved_root).replace("\\", "/")
            self.assertTrue(
                normalized_path.endswith("gym_pose/output/human_reviews"),
                f"Default root {resolved_root} does not end with gym_pose/output/human_reviews",
            )
            self.assertNotIn("gym_pose/gym_pose", normalized_path)

            # Change cwd to test_dir and re-instantiate
            os.chdir(self.test_dir)
            store_in_other_cwd = ReviewStore()
            self.assertEqual(
                store_in_other_cwd.root.resolve(),
                resolved_root,
                "ReviewStore default storage root must be CWD-independent",
            )
        finally:
            os.chdir(orig_cwd)

    def test_ground_truth_export_integration(self) -> None:
        """Export counts, call evaluation.load_ground_truth. Verify candidate has count, dev/out-of-scope has None."""
        s1 = self._create_dummy_video("c_in.mp4", b"c_in")
        c1 = self.store.create_case(s1, original_name="c_in.mp4", group="candidate")
        id1 = _get_id(c1)
        self.store.save_review(
            id1,
            {
                "human_count": 5,
                "reviewer": "Jane",
                "raw_video_reviewed": True,
                "predictions_previously_seen": False,
                "scope": "in_scope",
            },
        )

        s2 = self._create_dummy_video("c_none.mp4", b"c_none")
        c2 = self.store.create_case(s2, original_name="c_none.mp4", group="candidate")
        id2 = _get_id(c2)
        self.store.save_review(
            id2,
            {
                "human_count": None,
                "reviewer": "Jane",
                "raw_video_reviewed": True,
                "predictions_previously_seen": False,
                "scope": "in_scope",
            },
        )

        s3 = self._create_dummy_video("d_dev.mp4", b"d_dev")
        c3 = self.store.create_case(s3, original_name="d_dev.mp4", group="development")
        id3 = _get_id(c3)
        self.store.save_review(
            id3,
            {
                "human_count": 5,
                "reviewer": "Jane",
                "raw_video_reviewed": True,
                "predictions_previously_seen": False,
                "scope": "in_scope",
            },
        )

        s4 = self._create_dummy_video("c_out.mp4", b"c_out")
        c4 = self.store.create_case(s4, original_name="c_out.mp4", group="candidate")
        id4 = _get_id(c4)
        self.store.save_review(
            id4,
            {
                "human_count": 5,
                "reviewer": "Jane",
                "raw_video_reviewed": True,
                "predictions_previously_seen": False,
                "scope": "out_of_scope",
            },
        )

        counts = self.store.export_counts()
        export_file = Path(self.test_dir) / "gt_export.json"
        export_file.write_text(json.dumps(counts), encoding="utf-8")

        loaded = load_ground_truth(export_file)
        self.assertEqual(loaded[id1], 5)
        self.assertIsNone(loaded[id2])
        self.assertIsNone(loaded[id3], "Development group should be exported as None")
        self.assertIsNone(loaded[id4], "Out-of-scope should be exported as None")


if __name__ == "__main__":
    unittest.main()
