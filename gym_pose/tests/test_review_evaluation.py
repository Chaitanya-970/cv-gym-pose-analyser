"""Comprehensive unit test suite for review evaluation planning (Phase 1).

Tests:
1. Pure standard library: no Torch, Ultralytics, OpenCV, Matplotlib, or pipeline imports.
2. Model contract: safe names, duplicate rejection, non-existent, symlink, and empty checkpoints.
3. Output path validation: overlap with review root and model checkpoints, non-empty overwrite rejection.
4. Eligibility segregation: primary eligible (count 0 and positive), secondary seen, and all exclusions.
5. Evaluation integration: ground truth roundtrip with actual evaluation.load_ground_truth.
6. Honest status: empty and non-eligible cohorts produce status 'not_ready' with null metrics.
7. Integrity & corruption: fail-closed rejection of corrupt JSON, non-finite constants, and raw video tampering.
8. Publication seam integrity: source, annotation, and model mutation detection before plan publication.
9. Immutability: original review cases and model checkpoints untouched by planning.
10. CLI interface: successful --plan execution, unsupported --run error, argument validations.
11. Regression Suite 1 (Primary Probes):
    - Non-empty output race overwrite rejection and staging cleanup
    - Source and protected directory publish rejection
    - Annotation snapshot hash mismatch detection and fail-closed abort
12. Regression Suite 2 (Case Membership):
    - Case added during planning aborts
    - Case removed during planning aborts
13. Regression Suite 3 (Linked Ancestors):
    - Rejection of symlinks and junctions in checkpoint paths, review root, and output dir ancestors
14. Regression Suite 4 (Publication Staging & Failure Handling):
    - Staging failure leaves no partial plan or reviewed counts and cleans staging directory
    - Staging rename failure cleanup
15. Regression Suite 5 (Preservation & Ground Truth Compatibility):
    - Bit-for-bit preservation of videos, case metadata, and model weights across multiple cases
    - Strict compatibility of reviewed_counts.json with evaluation.load_ground_truth
"""

from __future__ import annotations

import io
import json
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from unittest.mock import patch

import gym_analyser.review_evaluation as rev_eval
from gym_analyser.evaluation import load_ground_truth
from gym_analyser.review_evaluation import (
    build_parser,
    build_plan,
    compute_file_sha256,
    format_plan_report,
    main,
    validate_model_name,
    validate_models_contract,
    validate_path_overlap,
)

try:
    from gym_analyser.review_evaluation import reject_linked_ancestors
except ImportError:
    reject_linked_ancestors = getattr(rev_eval, "reject_linked_ancestors", None)

from gym_analyser.review_store import (
    CaseCorruptedError,
    CaseNotFoundError,
    InvalidCaseIdError,
    ReviewStore,
    is_symlink,
)


class TestReviewEvaluationBase(unittest.TestCase):
    """Base test class providing isolated temporary environments and case helpers."""

    def setUp(self) -> None:
        self.test_root = tempfile.mkdtemp(prefix="test_rev_eval_")
        self.review_root = Path(self.test_root) / "review_storage"
        self.review_root.mkdir(parents=True, exist_ok=True)
        self.store = ReviewStore(root=self.review_root)

        self.models_dir = Path(self.test_root) / "models"
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.valid_ckpt = self.models_dir / "baseline.pt"
        self.valid_ckpt.write_bytes(b"DUMMY_YOLO_WEIGHTS_MOCK_CHECKPOINT_1234567890")

        self.output_root = Path(self.test_root) / "eval_output"

    def tearDown(self) -> None:
        shutil.rmtree(self.test_root, ignore_errors=True)

    def _create_synthetic_case(
        self,
        group: str = "candidate",
        human_count: Optional[int] = 5,
        scope: str = "in_scope",
        reviewer: str = "Reviewer One",
        raw_video_reviewed: bool = True,
        seen: Optional[bool] = False,
        video_content: bytes = b"DUMMY_VIDEO_CONTAINER_STREAM_BYTES_MP4",
        notes: str = "Synthetic test clip",
    ) -> Dict[str, Any]:
        """Create a fully authentic ReviewStore case in temporary storage."""
        temp_src = Path(self.test_root) / f"src_{os.urandom(4).hex()}.mp4"
        temp_src.write_bytes(video_content)
        try:
            case_info = self.store.create_case(
                source_path=temp_src,
                original_name="source_clip.mp4",
                group=group,
            )
            cid = case_info["case_id"]
            review_payload = {
                "human_count": human_count,
                "scope": scope,
                "reviewer": reviewer if human_count is not None else "",
                "raw_video_reviewed": raw_video_reviewed if human_count is not None else False,
                "predictions_previously_seen": seen if human_count is not None else None,
                "notes": notes,
                "events": [],
            }
            self.store.save_review(cid, review_payload)
            return case_info
        finally:
            if temp_src.exists():
                temp_src.unlink()

    def _create_symlink_or_junction(
        self, target: Path, link_path: Path, target_is_directory: bool = True
    ) -> None:
        """Safely create a directory junction or symlink, skipping if unprivileged."""
        if sys.platform == "win32" and target_is_directory:
            try:
                import _winapi

                _winapi.CreateJunction(str(target), str(link_path))
                return
            except (OSError, AttributeError):
                pass
        try:
            os.symlink(target, link_path, target_is_directory=target_is_directory)
        except (OSError, NotImplementedError) as exc:
            self.skipTest(
                f"Symlinks/junctions not supported on this platform/privilege level: {exc}"
            )


class TestPureStdlibNoMLImports(TestReviewEvaluationBase):
    """Verify pure standard library implementation with zero ML imports."""

    def test_no_ml_imports_in_source_or_namespace(self) -> None:
        source_code = Path(rev_eval.__file__).read_text(encoding="utf-8")
        disallowed = ("torch", "ultralytics", "cv2", "matplotlib", "scipy", "lap", "pipeline")
        for pkg in disallowed:
            self.assertNotIn(f"import {pkg}", source_code)
            self.assertNotIn(f"from {pkg}", source_code)
            self.assertFalse(hasattr(rev_eval, pkg))


class TestModelContractAndValidation(TestReviewEvaluationBase):
    """Verify model naming rules, duplicates, non-existent, and checkpoint validations."""

    def test_safe_model_names(self) -> None:
        valid_names = ["baseline", "yolo11n", "model_v1.0", "yolo11n-pose", "M1", "test.ckpt"]
        for name in valid_names:
            self.assertEqual(validate_model_name(name), name)

    def test_reject_unsafe_model_names(self) -> None:
        unsafe = [
            "",
            "   ",
            " baseline ",
            "../traversal",
            "..",
            "dir/model",
            "dir\\model",
            "-flag",
            ".hidden",
            "bad$char",
            "bad*char",
            "con",
            "nul",
            "prn",
            "com1",
            "lpt1",
            "a" * 65,  # Exceeds max 64 chars
        ]
        for name in unsafe:
            with self.assertRaises(ValueError, msg=f"Should reject: {name!r}"):
                validate_model_name(name)

        with self.assertRaises(TypeError):
            validate_model_name(123)

        with self.assertRaises(TypeError):
            validate_model_name(None)

    def test_reject_duplicate_model_names(self) -> None:
        ckpt2 = self.models_dir / "baseline2.pt"
        ckpt2.write_bytes(b"ANOTHER_CHECKPOINT")
        duplicate_entries = [("baseline", self.valid_ckpt), ("baseline", ckpt2)]
        with self.assertRaises(ValueError) as ctx:
            validate_models_contract(duplicate_entries)
        self.assertIn("duplicate", str(ctx.exception).lower())

    def test_reject_nonexistent_and_invalid_checkpoints(self) -> None:
        non_existent = self.models_dir / "missing.pt"
        with self.assertRaises(FileNotFoundError):
            validate_models_contract({"m1": non_existent})

        empty_file = self.models_dir / "empty.pt"
        empty_file.write_bytes(b"")
        with self.assertRaises(ValueError) as ctx:
            validate_models_contract({"m1": empty_file})
        self.assertIn("empty", str(ctx.exception).lower())

        dir_path = self.models_dir / "not_a_file"
        dir_path.mkdir(parents=True, exist_ok=True)
        with self.assertRaises(ValueError):
            validate_models_contract({"m1": dir_path})

    def test_reject_symlink_checkpoint(self) -> None:
        link_ckpt = self.models_dir / "link.pt"
        try:
            link_ckpt.symlink_to(self.valid_ckpt)
        except (OSError, NotImplementedError):
            return  # Skip if environment lacks symlink permission
        with self.assertRaises(ValueError) as ctx:
            validate_models_contract({"m1": link_ckpt})
        self.assertIn("symlink", str(ctx.exception).lower())

    def test_model_streaming_hash_correctness(self) -> None:
        sha = compute_file_sha256(self.valid_ckpt)
        self.assertEqual(len(sha), 64)
        self.assertEqual(sha, sha.lower())


class TestOutputPathOverlapAndNoOverwrite(TestReviewEvaluationBase):
    """Verify output path overlap rejection and protection against overwriting."""

    def test_reject_output_identical_to_review_root(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            validate_path_overlap(self.review_root, self.review_root, {"m1": self.valid_ckpt})
        self.assertIn("identical", str(ctx.exception).lower())

    def test_reject_output_inside_review_root(self) -> None:
        inside_out = self.review_root / "nested_output"
        with self.assertRaises(ValueError) as ctx:
            validate_path_overlap(inside_out, self.review_root, {"m1": self.valid_ckpt})
        self.assertIn("cannot be inside", str(ctx.exception).lower())

    def test_reject_review_root_inside_output(self) -> None:
        parent_out = self.review_root.parent
        with self.assertRaises(ValueError) as ctx:
            validate_path_overlap(parent_out, self.review_root, {"m1": self.valid_ckpt})
        self.assertIn("cannot be inside", str(ctx.exception).lower())

    def test_reject_output_overlapping_with_model_checkpoint(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            validate_path_overlap(self.models_dir, self.review_root, {"m1": self.valid_ckpt})
        self.assertIn("cannot be inside", str(ctx.exception).lower())

    def test_never_overwrite_existing_nonempty_output(self) -> None:
        self.output_root.mkdir(parents=True, exist_ok=True)
        stray_file = self.output_root / "existing.txt"
        stray_file.write_text("pre-existing content", encoding="utf-8")
        with self.assertRaises(FileExistsError):
            validate_path_overlap(self.output_root, self.review_root, {"m1": self.valid_ckpt})


class TestEligibilityAndExclusions(TestReviewEvaluationBase):
    """Verify cohort segregation: primary eligible, secondary seen, and all exclusions."""

    def test_cohort_segregation_and_contract(self) -> None:
        # 1. Primary eligible: count=5
        c1 = self._create_synthetic_case(
            group="candidate", human_count=5, scope="in_scope", reviewer="Alice", seen=False
        )
        # 2. Primary eligible: count=0 (valid 0 count!)
        c2 = self._create_synthetic_case(
            group="candidate", human_count=0, scope="in_scope", reviewer="Bob", seen=False
        )
        # 3. Secondary: seen=True (segregated secondary!)
        c3 = self._create_synthetic_case(
            group="candidate", human_count=4, scope="in_scope", reviewer="Carol", seen=True
        )
        # 4. Excluded: development group
        c4 = self._create_synthetic_case(
            group="development", human_count=3, scope="in_scope", reviewer="Dave", seen=False
        )
        # 5. Excluded: out_of_scope
        c5 = self._create_synthetic_case(
            group="candidate", human_count=2, scope="out_of_scope", reviewer="Eve", seen=False
        )
        # 6. Excluded: uncertain scope
        c6 = self._create_synthetic_case(
            group="candidate", human_count=1, scope="uncertain", reviewer="Frank", seen=False
        )
        # 7. Excluded: unreviewed (count is None)
        c7 = self._create_synthetic_case(
            group="candidate", human_count=None, scope="in_scope", reviewer="", seen=None
        )

        plan = build_plan(
            review_root=self.review_root,
            output_dir=self.output_root,
            models={"yolo11n": self.valid_ckpt},
        )

        self.assertEqual(plan["status"], "ready")
        self.assertIsNone(plan["metrics"])

        eligible_ids = [c["case_id"] for c in plan["eligible_cases"]]
        secondary_ids = [c["case_id"] for c in plan["secondary_cases"]]
        excluded_ids = [c["case_id"] for c in plan["excluded_cases"]]

        self.assertEqual(sorted(eligible_ids), sorted([c1["case_id"], c2["case_id"]]))
        self.assertEqual(secondary_ids, [c3["case_id"]])
        self.assertEqual(
            sorted(excluded_ids),
            sorted([c4["case_id"], c5["case_id"], c6["case_id"], c7["case_id"]]),
        )

        # Excluded cases carry case_id and reason
        for exc in plan["excluded_cases"]:
            self.assertIn("case_id", exc)
            self.assertIn("reason", exc)
            self.assertTrue(len(exc["reason"]) > 0)

        # Verify exclusion reasons are accurate
        exc_map = {e["case_id"]: e["reason"] for e in plan["excluded_cases"]}
        self.assertIn("group 'development'", exc_map[c4["case_id"]])
        self.assertIn("scope 'out_of_scope'", exc_map[c5["case_id"]])
        self.assertIn("scope 'uncertain'", exc_map[c6["case_id"]])
        self.assertIn("null/unreviewed", exc_map[c7["case_id"]])

        # Verify declared configuration
        decl = plan["declared_configuration"]
        self.assertEqual(decl["counter_config"]["down_angle"], 110.0)
        self.assertEqual(decl["counter_config"]["up_angle"], 150.0)
        self.assertEqual(decl["counter_config"]["min_joint_confidence"], 0.3)
        self.assertEqual(decl["counter_config"]["stable_frames"], 3)
        self.assertEqual(decl["counter_config"]["max_gap_seconds"], 0.5)
        self.assertEqual(decl["departure_frames"], 2)
        self.assertEqual(decl["tracker_name"], "bytetrack.yaml")
        self.assertEqual(
            decl["runtime_inference_effective_parameters"],
            "unavailable_explicit_pending_before_run",
        )

        # Verify limitations & disclaimers
        limits = plan["limitations_and_disclaimers"]
        self.assertFalse(limits["gold_standard_claim"])
        self.assertFalse(limits["source_independence_claim"])
        self.assertEqual(limits["penn_action_reserved_images_scored"], 0)
        self.assertTrue(limits["no_fake_perfect_accuracy"])


class TestGroundTruthRoundtrip(TestReviewEvaluationBase):
    """Verify integration with evaluation.load_ground_truth."""

    def test_reviewed_counts_loader_roundtrip(self) -> None:
        c1 = self._create_synthetic_case(
            group="candidate", human_count=5, scope="in_scope", reviewer="Alice", seen=False
        )
        c2 = self._create_synthetic_case(
            group="candidate", human_count=0, scope="in_scope", reviewer="Bob", seen=False
        )
        # Secondary seen case must NOT be included in primary scored counts
        c3 = self._create_synthetic_case(
            group="candidate", human_count=6, scope="in_scope", reviewer="Carol", seen=True
        )

        build_plan(
            review_root=self.review_root,
            output_dir=self.output_root,
            models={"m1": self.valid_ckpt},
        )

        counts_file = self.output_root / "reviewed_counts.json"
        self.assertTrue(counts_file.is_file())

        loaded_gt = load_ground_truth(counts_file)
        expected = {
            c1["case_id"]: 5,
            c2["case_id"]: 0,
        }
        self.assertEqual(loaded_gt, expected)
        self.assertNotIn(c3["case_id"], loaded_gt)


class TestEmptyAndNonEligibleCohorts(TestReviewEvaluationBase):
    """Verify honest 'not_ready' status and null metrics for empty or excluded cohorts."""

    def test_empty_review_root_is_not_ready(self) -> None:
        plan = build_plan(
            review_root=self.review_root,
            output_dir=self.output_root,
            models={"m1": self.valid_ckpt},
        )
        self.assertEqual(plan["status"], "not_ready")
        self.assertIsNone(plan["metrics"])
        self.assertEqual(len(plan["eligible_cases"]), 0)
        self.assertEqual(len(plan["secondary_cases"]), 0)
        self.assertEqual(len(plan["excluded_cases"]), 0)

        counts_file = self.output_root / "reviewed_counts.json"
        loaded = load_ground_truth(counts_file)
        self.assertEqual(loaded, {})

    def test_all_excluded_cohort_is_not_ready(self) -> None:
        self._create_synthetic_case(group="development", human_count=5, scope="in_scope")
        self._create_synthetic_case(group="candidate", human_count=None, scope="in_scope")
        self._create_synthetic_case(group="candidate", human_count=2, scope="out_of_scope")

        plan = build_plan(
            review_root=self.review_root,
            output_dir=self.output_root,
            models={"m1": self.valid_ckpt},
        )
        self.assertEqual(plan["status"], "not_ready")
        self.assertIsNone(plan["metrics"])
        self.assertEqual(len(plan["eligible_cases"]), 0)
        self.assertEqual(len(plan["excluded_cases"]), 3)

        counts_file = self.output_root / "reviewed_counts.json"
        loaded = load_ground_truth(counts_file)
        self.assertEqual(loaded, {})


class TestCorruptionAndIntegrityFailClosed(TestReviewEvaluationBase):
    """Verify fail-closed handling on corrupt metadata, non-finite constants, and raw tampering."""

    def test_reject_corrupt_json_in_case(self) -> None:
        case = self._create_synthetic_case()
        cid = case["case_id"]
        case_json = self.review_root / cid / "case.json"
        case_json.write_text("INVALID_NON_JSON_SYNTAX{{{", encoding="utf-8")

        with self.assertRaises(CaseCorruptedError):
            build_plan(
                review_root=self.review_root,
                output_dir=self.output_root,
                models={"m1": self.valid_ckpt},
            )

    def test_reject_nonfinite_json_constants(self) -> None:
        case = self._create_synthetic_case()
        cid = case["case_id"]
        case_json = self.review_root / cid / "case.json"
        raw_data = json.loads(case_json.read_text(encoding="utf-8"))
        raw_text = json.dumps(raw_data).replace('"human_count": 5', '"human_count": NaN')
        case_json.write_text(raw_text, encoding="utf-8")

        with self.assertRaises(CaseCorruptedError):
            build_plan(
                review_root=self.review_root,
                output_dir=self.output_root,
                models={"m1": self.valid_ckpt},
            )

    def test_reject_tampered_raw_video(self) -> None:
        case = self._create_synthetic_case()
        cid = case["case_id"]
        video_file = self.review_root / cid / "video.mp4"
        video_file.write_bytes(b"CORRUPTED_RAW_VIDEO_BYTES_TAMPERED")

        with self.assertRaises(CaseCorruptedError):
            build_plan(
                review_root=self.review_root,
                output_dir=self.output_root,
                models={"m1": self.valid_ckpt},
            )


class TestPublicationSeamMutation(TestReviewEvaluationBase):
    """Verify integrity rechecks before publishing plan at the publication seam."""

    def test_source_video_mutation_at_seam_aborts(self) -> None:
        case = self._create_synthetic_case()
        cid = case["case_id"]
        v_path = self.review_root / cid / "video.mp4"

        def seam_mutate_video() -> None:
            v_path.write_bytes(v_path.read_bytes() + b"_MUTATED_AT_SEAM")

        with self.assertRaises(RuntimeError) as ctx:
            build_plan(
                review_root=self.review_root,
                output_dir=self.output_root,
                models={"m1": self.valid_ckpt},
                _seam_hook=seam_mutate_video,
            )
        self.assertIn("source video hash verification failed", str(ctx.exception).lower())
        self.assertFalse((self.output_root / "plan.json").exists())

    def test_annotation_mutation_at_seam_aborts(self) -> None:
        case = self._create_synthetic_case()
        cid = case["case_id"]
        ann_path = self.review_root / cid / "case.json"

        def seam_mutate_annotation() -> None:
            content = ann_path.read_text(encoding="utf-8")
            ann_path.write_text(content.replace('"in_scope"', '"out_of_scope"'), encoding="utf-8")

        with self.assertRaises(RuntimeError) as ctx:
            build_plan(
                review_root=self.review_root,
                output_dir=self.output_root,
                models={"m1": self.valid_ckpt},
                _seam_hook=seam_mutate_annotation,
            )
        self.assertIn("annotation hash verification failed", str(ctx.exception).lower())
        self.assertFalse((self.output_root / "plan.json").exists())

    def test_model_mutation_at_seam_aborts(self) -> None:
        self._create_synthetic_case()

        def seam_mutate_model() -> None:
            self.valid_ckpt.write_bytes(self.valid_ckpt.read_bytes() + b"_MUTATED_MODEL")

        with self.assertRaises(RuntimeError) as ctx:
            build_plan(
                review_root=self.review_root,
                output_dir=self.output_root,
                models={"m1": self.valid_ckpt},
                _seam_hook=seam_mutate_model,
            )
        self.assertIn("model checkpoint hash verification failed", str(ctx.exception).lower())
        self.assertFalse((self.output_root / "plan.json").exists())


class TestOriginalsUnchangedAfterPlanning(TestReviewEvaluationBase):
    """Verify that build_plan never alters original review cases or checkpoints."""

    def test_originals_remain_byte_identical(self) -> None:
        c1 = self._create_synthetic_case(human_count=5)
        cid = c1["case_id"]

        video_path = self.review_root / cid / "video.mp4"
        case_json_path = self.review_root / cid / "case.json"

        v_sha_before = compute_file_sha256(video_path)
        j_sha_before = compute_file_sha256(case_json_path)
        m_sha_before = compute_file_sha256(self.valid_ckpt)

        build_plan(
            review_root=self.review_root,
            output_dir=self.output_root,
            models={"m1": self.valid_ckpt},
        )

        self.assertEqual(compute_file_sha256(video_path), v_sha_before)
        self.assertEqual(compute_file_sha256(case_json_path), j_sha_before)
        self.assertEqual(compute_file_sha256(self.valid_ckpt), m_sha_before)


class TestCLIInterface(TestReviewEvaluationBase):
    """Verify CLI interface: --plan execution, unsupported --run, and argument parsing."""

    def test_cli_plan_success(self) -> None:
        self._create_synthetic_case(human_count=4)
        argv = [
            "--review-root", str(self.review_root),
            "--output-dir", str(self.output_root),
            "--model", f"baseline={self.valid_ckpt}",
            "--plan",
        ]
        out_buf = io.StringIO()
        err_buf = io.StringIO()
        with redirect_stdout(out_buf), redirect_stderr(err_buf):
            ret = main(argv)

        self.assertEqual(ret, 0)
        self.assertTrue((self.output_root / "plan.json").is_file())
        self.assertTrue((self.output_root / "reviewed_counts.json").is_file())
        self.assertIn("GYM POSE REVIEW EVALUATION PLAN", out_buf.getvalue())

    def test_cli_unsupported_run_error(self) -> None:
        argv = [
            "--review-root", str(self.review_root),
            "--output-dir", str(self.output_root),
            "--model", f"baseline={self.valid_ckpt}",
            "--run",
        ]
        err_buf = io.StringIO()
        with redirect_stderr(err_buf):
            ret = main(argv)

        self.assertEqual(ret, 1)
        self.assertIn("--run is not supported in Phase 1", err_buf.getvalue())

    def test_cli_missing_plan_error(self) -> None:
        argv = [
            "--review-root", str(self.review_root),
            "--output-dir", str(self.output_root),
            "--model", f"baseline={self.valid_ckpt}",
        ]
        err_buf = io.StringIO()
        with redirect_stderr(err_buf):
            ret = main(argv)

        self.assertEqual(ret, 1)
        self.assertIn("Must specify --plan", err_buf.getvalue())

    def test_cli_duplicate_model_error(self) -> None:
        argv = [
            "--review-root", str(self.review_root),
            "--output-dir", str(self.output_root),
            "--model", f"baseline={self.valid_ckpt}",
            "--model", f"baseline={self.valid_ckpt}",
            "--plan",
        ]
        err_buf = io.StringIO()
        with redirect_stderr(err_buf):
            ret = main(argv)

        self.assertEqual(ret, 1)
        self.assertIn("Duplicate model name", err_buf.getvalue())

    def test_cli_invalid_model_syntax_error(self) -> None:
        argv = [
            "--review-root", str(self.review_root),
            "--output-dir", str(self.output_root),
            "--model", "invalid_no_equals_sign",
            "--plan",
        ]
        err_buf = io.StringIO()
        with redirect_stderr(err_buf):
            ret = main(argv)

        self.assertEqual(ret, 1)
        self.assertIn("Invalid --model format", err_buf.getvalue())


# ==============================================================================
# Comprehensive Regression Test Suites for 5 Review Findings
# ==============================================================================


class TestExactPrimaryProbeRegressions(TestReviewEvaluationBase):
    """Suite 1: Exact Primary Probe Regressions for findings 1, 2, and 3."""

    def test_primary_probe_nonempty_output_race_overwritten(self) -> None:
        """Verify racing non-empty output is never overwritten and staging cleans up."""
        self._create_synthetic_case(human_count=5)
        base = Path(self.test_root)
        output_dir = base / "plan"
        sentinel = "EXISTING HUMAN OUTPUT"

        def _race_seam_hook() -> None:
            output_dir.mkdir(parents=True, exist_ok=True)
            (output_dir / "reviewed_counts.json").write_text(sentinel, encoding="utf-8")
            (output_dir / "marker").write_text("preserve", encoding="utf-8")

        with self.assertRaises((FileExistsError, RuntimeError)):
            build_plan(
                review_root=self.review_root,
                output_dir=output_dir,
                models={"fixture": self.valid_ckpt},
                _seam_hook=_race_seam_hook,
            )

        # Existing content must remain untouched
        self.assertEqual((output_dir / "reviewed_counts.json").read_text(encoding="utf-8"), sentinel)
        self.assertEqual((output_dir / "marker").read_text(encoding="utf-8"), "preserve")

        # Staging directories matching .*plan-* must be cleaned up
        leftovers = [
            p.name for p in output_dir.parent.iterdir()
            if "plan-" in p.name and p.is_dir()
        ]
        self.assertEqual(leftovers, [])

    def test_primary_probe_source_directory_rejected(self) -> None:
        """Verify publication into source and protected directories is rejected."""
        probe_target = Path(rev_eval.__file__).parent / "should_never_publish_here"
        with self.assertRaises(ValueError):
            validate_path_overlap(
                probe_target,
                self.review_root,
                {"fixture": self.valid_ckpt},
            )

        package = Path(rev_eval.__file__).resolve().parent
        app_root = package.parent
        protected_dirs = [
            package,
            package / "nested_publish",
            app_root / "tests",
            app_root / "tests" / "nested_publish",
            app_root / "models",
            app_root / "models" / "nested_publish",
            app_root / "test_data",
            app_root / "test_data" / "nested_publish",
        ]
        for p in protected_dirs:
            with self.subTest(protected=str(p)):
                with self.assertRaises(ValueError):
                    validate_path_overlap(
                        p,
                        self.review_root,
                        {"fixture": self.valid_ckpt},
                    )

    def test_primary_probe_annotation_snapshot_hash_mismatch(self) -> None:
        """Verify that modifying case.json after load aborts and old count is never paired."""
        case = self._create_synthetic_case(human_count=1)
        cid = case["case_id"]
        case_json_p = self.review_root / cid / "case.json"

        orig_load = ReviewStore.load_persisted_case

        def tampering_load(store_self: ReviewStore, case_id: str, *args: Any, **kwargs: Any) -> Dict[str, Any]:
            res = orig_load(store_self, case_id, *args, **kwargs)
            if case_id == cid:
                content = case_json_p.read_text(encoding="utf-8")
                # Tamper count from 1 to 2
                tampered = content.replace('"human_count": 1', '"human_count": 2')
                case_json_p.write_text(tampered, encoding="utf-8")
            return res

        ReviewStore.load_persisted_case = tampering_load
        try:
            with self.assertRaises(RuntimeError) as ctx:
                build_plan(
                    review_root=self.review_root,
                    output_dir=self.output_root,
                    models={"fixture": self.valid_ckpt},
                )
            self.assertIn("annotation", str(ctx.exception).lower())
        finally:
            ReviewStore.load_persisted_case = orig_load

        # Plan publication must abort; old count must never be paired with new annotation hash
        self.assertFalse((self.output_root / "plan.json").exists())
        self.assertFalse((self.output_root / "reviewed_counts.json").exists())

    def test_captured_development_metadata_cannot_become_primary(self) -> None:
        """Verify captured development metadata cannot become primary via load-time mutation."""
        c = self._create_synthetic_case(group="development")
        cid = c["case_id"]
        casefile = self.review_root / cid / "case.json"
        raw = casefile.read_bytes()
        original = ReviewStore.load_persisted_case

        def metadata_swap(store: ReviewStore, *a: Any, **kw: Any) -> Dict[str, Any]:
            changed = json.loads(raw)
            changed["group"] = "candidate"
            casefile.write_text(json.dumps(changed))
            try:
                return original(store, *a, **kw)
            finally:
                casefile.write_bytes(raw)

        with patch.object(ReviewStore, "load_persisted_case", metadata_swap):
            try:
                plan = build_plan(
                    self.review_root,
                    self.output_root,
                    {"baseline": self.valid_ckpt},
                )
                self.assertEqual(len(plan.get("eligible_cases", [])), 0)
            except RuntimeError as exc:
                self.assertIn("changed while capturing case", str(exc).lower())


class TestCaseMembershipRegressions(TestReviewEvaluationBase):
    """Suite 2: Case Membership Regressions (Added and Removed Cases during planning)."""

    def test_case_added_during_planning_aborts(self) -> None:
        """Verify that adding a case during planning raises RuntimeError and aborts."""
        self._create_synthetic_case(human_count=3)
        new_cid = "12345678-1234-4234-8234-123456789abc"

        def seam_add_case() -> None:
            new_case_dir = self.review_root / new_cid
            new_case_dir.mkdir(parents=True, exist_ok=True)
            (new_case_dir / "case.json").write_text("{}", encoding="utf-8")

        with self.assertRaises(RuntimeError) as ctx:
            build_plan(
                review_root=self.review_root,
                output_dir=self.output_root,
                models={"fixture": self.valid_ckpt},
                _seam_hook=seam_add_case,
            )

        err_msg = str(ctx.exception).lower()
        self.assertIn("membership", err_msg)
        self.assertIn("added", err_msg)
        self.assertFalse((self.output_root / "plan.json").exists())

    def test_case_removed_during_planning_aborts(self) -> None:
        """Verify that removing a case during planning raises RuntimeError and aborts."""
        c1 = self._create_synthetic_case(human_count=3)
        c2 = self._create_synthetic_case(human_count=4)

        def seam_remove_case() -> None:
            shutil.rmtree(self.review_root / c2["case_id"])

        with self.assertRaises(RuntimeError) as ctx:
            build_plan(
                review_root=self.review_root,
                output_dir=self.output_root,
                models={"fixture": self.valid_ckpt},
                _seam_hook=seam_remove_case,
            )

        err_msg = str(ctx.exception).lower()
        self.assertIn("membership", err_msg)
        self.assertIn("removed", err_msg)
        self.assertFalse((self.output_root / "plan.json").exists())


class TestLinkedAncestorRegressions(TestReviewEvaluationBase):
    """Suite 3: Linked Ancestor Regressions (symlinks and junctions in ancestors)."""

    def test_reject_linked_ancestors_direct_helper(self) -> None:
        """Verify direct contract of reject_linked_ancestors helper."""
        helper = getattr(rev_eval, "reject_linked_ancestors", None)
        if helper is None:
            self.fail("reject_linked_ancestors must be exported from review_evaluation")

        # Clean regular path must not raise
        helper(self.valid_ckpt, label="Checkpoint")

        # Linked ancestor path must raise ValueError
        linked_dir = Path(self.test_root) / "linked_ancestor_test"
        self._create_symlink_or_junction(self.models_dir, linked_dir, target_is_directory=True)
        nested_path = linked_dir / "baseline.pt"
        with self.assertRaises(ValueError) as ctx:
            helper(nested_path, label="Checkpoint")
        self.assertIn("symlink or reparse ancestor", str(ctx.exception).lower())

    def test_checkpoint_linked_ancestor_rejected(self) -> None:
        """Verify checkpoint whose parent or ancestor is a symlink/junction raises ValueError."""
        linked_models = Path(self.test_root) / "linked_models"
        self._create_symlink_or_junction(self.models_dir, linked_models, target_is_directory=True)
        linked_ckpt = linked_models / "baseline.pt"

        with self.assertRaises(ValueError) as ctx:
            validate_models_contract({"fixture": linked_ckpt})
        self.assertIn("ancestor", str(ctx.exception).lower())

        with self.assertRaises(ValueError) as ctx:
            compute_file_sha256(linked_ckpt)
        self.assertIn("ancestor", str(ctx.exception).lower())

    def test_review_root_linked_ancestor_rejected(self) -> None:
        """Verify review root whose parent or ancestor is a symlink/junction raises ValueError."""
        linked_parent = Path(self.test_root) / "linked_rev_parent"
        self._create_symlink_or_junction(self.review_root.parent, linked_parent, target_is_directory=True)
        linked_root = linked_parent / self.review_root.name

        with self.assertRaises(ValueError) as ctx:
            validate_path_overlap(self.output_root, linked_root, {"fixture": self.valid_ckpt})
        self.assertIn("ancestor", str(ctx.exception).lower())

        with self.assertRaises((ValueError, CaseCorruptedError)):
            build_plan(linked_root, self.output_root, {"fixture": self.valid_ckpt})

    def test_output_dir_linked_ancestor_rejected(self) -> None:
        """Verify output dir whose parent or ancestor is a symlink/junction raises ValueError."""
        real_out_parent = Path(self.test_root) / "real_out_parent"
        real_out_parent.mkdir(parents=True, exist_ok=True)
        linked_out_parent = Path(self.test_root) / "linked_out_parent"
        self._create_symlink_or_junction(real_out_parent, linked_out_parent, target_is_directory=True)
        linked_out = linked_out_parent / "plan"

        with self.assertRaises(ValueError) as ctx:
            validate_path_overlap(linked_out, self.review_root, {"fixture": self.valid_ckpt})
        self.assertIn("ancestor", str(ctx.exception).lower())

        with self.assertRaises(ValueError):
            build_plan(self.review_root, linked_out, {"fixture": self.valid_ckpt})


class TestPublicationStagingAndFailure(TestReviewEvaluationBase):
    """Suite 4: Publication Staging and Failure Regressions."""

    def test_staging_failure_leaves_no_plan(self) -> None:
        """Verify simulated disk/staging failure cleans up staging dir and leaves no plan."""
        self._create_synthetic_case(human_count=4)

        with patch("json.dump", side_effect=IOError("Simulated disk full during staging")):
            with self.assertRaises(IOError):
                build_plan(
                    review_root=self.review_root,
                    output_dir=self.output_root,
                    models={"fixture": self.valid_ckpt},
                )

        self.assertFalse((self.output_root / "plan.json").exists())
        self.assertFalse((self.output_root / "reviewed_counts.json").exists())
        if self.output_root.parent.exists():
            leftovers = [
                p.name for p in self.output_root.parent.iterdir()
                if "plan-" in p.name and p.is_dir()
            ]
            self.assertEqual(leftovers, [])

    def test_staging_rename_failure_cleans_up(self) -> None:
        """Verify simulated failure during final rename cleans up staging directory."""
        self._create_synthetic_case(human_count=4)

        with patch.object(Path, "rename", side_effect=OSError("Simulated rename error")):
            with self.assertRaises(OSError):
                build_plan(
                    review_root=self.review_root,
                    output_dir=self.output_root,
                    models={"fixture": self.valid_ckpt},
                )

        self.assertFalse((self.output_root / "plan.json").exists())
        if self.output_root.parent.exists():
            leftovers = [
                p.name for p in self.output_root.parent.iterdir()
                if "plan-" in p.name and p.is_dir()
            ]
            self.assertEqual(leftovers, [])


class TestPreservationAndGroundTruthCompatibility(TestReviewEvaluationBase):
    """Suite 5: Bit-for-bit Preservation and Ground Truth Compatibility."""

    def test_bit_for_bit_preservation_multicase(self) -> None:
        """Verify raw video, case.json, and checkpoints remain bit-for-bit identical."""
        c1 = self._create_synthetic_case(human_count=5, seen=False)
        c2 = self._create_synthetic_case(human_count=0, seen=False)
        c3 = self._create_synthetic_case(human_count=4, seen=True)
        c4 = self._create_synthetic_case(human_count=3, group="development")

        cases = [c1, c2, c3, c4]
        video_bytes_before = {}
        json_bytes_before = {}
        for c in cases:
            cid = c["case_id"]
            video_bytes_before[cid] = (self.review_root / cid / "video.mp4").read_bytes()
            json_bytes_before[cid] = (self.review_root / cid / "case.json").read_bytes()
        ckpt_bytes_before = self.valid_ckpt.read_bytes()

        build_plan(
            review_root=self.review_root,
            output_dir=self.output_root,
            models={"fixture": self.valid_ckpt},
        )

        for c in cases:
            cid = c["case_id"]
            self.assertEqual(
                (self.review_root / cid / "video.mp4").read_bytes(),
                video_bytes_before[cid],
            )
            self.assertEqual(
                (self.review_root / cid / "case.json").read_bytes(),
                json_bytes_before[cid],
            )
        self.assertEqual(self.valid_ckpt.read_bytes(), ckpt_bytes_before)

    def test_reviewed_counts_strict_ground_truth_compatibility(self) -> None:
        """Verify reviewed_counts.json is directly readable by load_ground_truth."""
        c1 = self._create_synthetic_case(human_count=5, seen=False)
        c2 = self._create_synthetic_case(human_count=0, seen=False)
        c3 = self._create_synthetic_case(human_count=2, seen=True)  # secondary
        c4 = self._create_synthetic_case(human_count=8, group="development")  # excluded

        build_plan(
            review_root=self.review_root,
            output_dir=self.output_root,
            models={"fixture": self.valid_ckpt},
        )

        counts_file = self.output_root / "reviewed_counts.json"
        self.assertTrue(counts_file.is_file())

        loaded = load_ground_truth(counts_file)
        self.assertEqual(loaded, {c1["case_id"]: 5, c2["case_id"]: 0})
        self.assertNotIn(c3["case_id"], loaded)
        self.assertNotIn(c4["case_id"], loaded)

        # Strict JSON schema validation
        raw_json = json.loads(counts_file.read_text(encoding="utf-8"))
        self.assertEqual(raw_json["checkpoint"], "evaluation_plan")
        self.assertIn("generated_at", raw_json)
        self.assertEqual(len(raw_json["clips"]), 2)
        clip_map = {item["clip_id"]: item for item in raw_json["clips"]}
        self.assertEqual(clip_map[c1["case_id"]]["human_count"], 5)
        self.assertEqual(clip_map[c2["case_id"]]["human_count"], 0)


if __name__ == "__main__":
    unittest.main()
