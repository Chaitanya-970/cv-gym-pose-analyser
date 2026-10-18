"""Unit tests for repetition count evaluation module (gym_analyser.evaluation).

Verifies metric computations (exact match, MAE), exclusion of unreviewed clips,
input validation (booleans, floats, negatives, duplicates), missing summary.json
handling, schema flexibility, and CLI execution.
"""

from __future__ import annotations

import io
import json
import shutil
import tempfile
import unittest
from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path
from typing import Any, Dict, Optional

from gym_analyser.evaluation import (
    build_parser,
    evaluate_counts,
    format_evaluation_report,
    load_ground_truth,
    load_prediction_summary,
    load_predictions_from_dir,
    main,
    validate_count,
)


class TestCountValidation(unittest.TestCase):
    """Tests for count validation ensuring non-negative integers only."""

    def test_valid_integer_counts(self) -> None:
        self.assertEqual(validate_count(0, "test"), 0)
        self.assertEqual(validate_count(1, "test"), 1)
        self.assertEqual(validate_count(42, "test"), 42)

    def test_reject_boolean_counts(self) -> None:
        # In Python, bool is a subclass of int (isinstance(True, int) is True).
        # Must be strictly rejected.
        with self.assertRaises(ValueError) as ctx:
            validate_count(True, "human_count", "clip_01")
        self.assertIn("boolean", str(ctx.exception).lower())

        with self.assertRaises(ValueError) as ctx:
            validate_count(False, "human_count", "clip_01")
        self.assertIn("boolean", str(ctx.exception).lower())

    def test_reject_float_counts(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            validate_count(2.5, "completed_reps", "clip_01")
        self.assertIn("integer", str(ctx.exception).lower())

        with self.assertRaises(ValueError) as ctx:
            validate_count(2.0, "completed_reps", "clip_01")
        self.assertIn("integer", str(ctx.exception).lower())

    def test_reject_negative_counts(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            validate_count(-1, "human_count", "clip_01")
        self.assertIn("non-negative", str(ctx.exception).lower())

    def test_reject_string_or_other_types(self) -> None:
        with self.assertRaises(ValueError):
            validate_count("5", "completed_reps", "clip_01")
        with self.assertRaises(ValueError):
            validate_count([], "completed_reps", "clip_01")


class TestGroundTruthLoading(unittest.TestCase):
    """Tests for loading human-reviewed ground truth JSON files in various formats."""

    def setUp(self) -> None:
        self.temp_dir = tempfile.mkdtemp(prefix="test_gt_")
        self.addCleanup(lambda: shutil.rmtree(self.temp_dir, ignore_errors=True))

    def _write_json(self, name: str, data: Any) -> Path:
        p = Path(self.temp_dir) / name
        p.write_text(json.dumps(data), encoding="utf-8")
        return p

    def test_load_reviewed_counts_schema(self) -> None:
        data = {
            "checkpoint": "yolo11n-pose.pt",
            "clips": [
                {"clip_id": "clip_01", "human_count": 2},
                {"clip_id": "clip_03", "human_count": 6},
            ],
        }
        p = self._write_json("gt.json", data)
        loaded = load_ground_truth(p)
        self.assertEqual(loaded, {"clip_01": 2, "clip_03": 6})

    def test_load_direct_mapping(self) -> None:
        data = {
            "clip_01": 2,
            "clip_02": None,
            "clip_03": 6,
        }
        p = self._write_json("mapping.json", data)
        loaded = load_ground_truth(p)
        self.assertEqual(loaded, {"clip_01": 2, "clip_02": None, "clip_03": 6})

    def test_load_nested_mapping(self) -> None:
        data = {
            "clip_01": {"human_count": 2},
            "clip_02": {"human_count": None},
        }
        p = self._write_json("nested.json", data)
        loaded = load_ground_truth(p)
        self.assertEqual(loaded, {"clip_01": 2, "clip_02": None})

    def test_load_list_of_objects(self) -> None:
        data = [
            {"clip_id": "clip_01", "human_count": 2},
            {"clip_id": "clip_02", "human_count": None},
        ]
        p = self._write_json("list.json", data)
        loaded = load_ground_truth(p)
        self.assertEqual(loaded, {"clip_01": 2, "clip_02": None})

    def test_reject_duplicate_clip_ids_in_list(self) -> None:
        data = [
            {"clip_id": "clip_01", "human_count": 2},
            {"clip_id": "clip_01", "human_count": 3},
        ]
        p = self._write_json("dups.json", data)
        with self.assertRaises(ValueError) as ctx:
            load_ground_truth(p)
        self.assertIn("duplicate clip id", str(ctx.exception).lower())

    def test_reject_duplicate_keys_in_json_mapping(self) -> None:
        raw_json = '{"clip_01": 2, "clip_01": 5}'
        p = Path(self.temp_dir) / "dup_keys.json"
        p.write_text(raw_json, encoding="utf-8")
        with self.assertRaises(ValueError) as ctx:
            load_ground_truth(p)
        self.assertIn("duplicate key", str(ctx.exception).lower())

    def test_reject_invalid_counts_in_ground_truth(self) -> None:
        data = {"clip_01": 2.5}
        p = self._write_json("bad_float.json", data)
        with self.assertRaises(ValueError):
            load_ground_truth(p)

        data = {"clip_01": -2}
        p = self._write_json("bad_neg.json", data)
        with self.assertRaises(ValueError):
            load_ground_truth(p)

        data = {"clip_01": True}
        p = self._write_json("bad_bool.json", data)
        with self.assertRaises(ValueError):
            load_ground_truth(p)

    def test_missing_ground_truth_file_raises_not_found(self) -> None:
        with self.assertRaises(FileNotFoundError):
            load_ground_truth(Path(self.temp_dir) / "nonexistent.json")


class TestPredictionLoading(unittest.TestCase):
    """Tests for loading prediction summaries from clip subdirectories."""

    def setUp(self) -> None:
        self.temp_dir = tempfile.mkdtemp(prefix="test_pred_")
        self.addCleanup(lambda: shutil.rmtree(self.temp_dir, ignore_errors=True))

    def _create_clip_summary(self, dir_name: str, count: Any, clip_id_in_json: Optional[str] = None) -> Path:
        clip_dir = Path(self.temp_dir) / dir_name
        clip_dir.mkdir(parents=True, exist_ok=True)
        summary_payload: Dict[str, Any] = {"completed_reps": count}
        if clip_id_in_json:
            summary_payload["clip_id"] = clip_id_in_json
        summary_path = clip_dir / "summary.json"
        summary_path.write_text(json.dumps(summary_payload), encoding="utf-8")
        return summary_path

    def test_load_valid_predictions_directory(self) -> None:
        self._create_clip_summary("clip_01", 2)
        self._create_clip_summary("clip_02", 0)
        self._create_clip_summary("clip_03", 6)

        preds = load_predictions_from_dir(self.temp_dir)
        self.assertEqual(preds, {"clip_01": 2, "clip_02": 0, "clip_03": 6})

    def test_load_empty_predictions_directory(self) -> None:
        empty_dir = Path(self.temp_dir) / "empty"
        empty_dir.mkdir()
        preds = load_predictions_from_dir(empty_dir)
        self.assertEqual(preds, {})

    def test_load_single_clip_directory_directly(self) -> None:
        clip_dir = Path(self.temp_dir) / "clip_single"
        clip_dir.mkdir()
        (clip_dir / "summary.json").write_text(json.dumps({"completed_reps": 3}), encoding="utf-8")
        preds = load_predictions_from_dir(clip_dir)
        self.assertEqual(preds, {"clip_single": 3})

    def test_missing_summary_json_raises_by_default(self) -> None:
        clip_dir = Path(self.temp_dir) / "clip_no_summary"
        clip_dir.mkdir()
        with self.assertRaises(FileNotFoundError) as ctx:
            load_predictions_from_dir(self.temp_dir, allow_missing_summary=False)
        self.assertIn("missing summary.json", str(ctx.exception).lower())

    def test_missing_summary_json_skipped_when_allowed(self) -> None:
        self._create_clip_summary("clip_01", 2)
        (Path(self.temp_dir) / "clip_empty").mkdir()
        preds = load_predictions_from_dir(self.temp_dir, allow_missing_summary=True)
        self.assertEqual(preds, {"clip_01": 2})

    def test_mismatched_clip_id_in_summary_raises(self) -> None:
        self._create_clip_summary("folder_clip_01", 2, clip_id_in_json="other_clip_name")
        with self.assertRaises(ValueError) as ctx:
            load_predictions_from_dir(self.temp_dir)
        self.assertIn("mismatched clip id", str(ctx.exception).lower())

    def test_missing_completed_reps_field_raises(self) -> None:
        clip_dir = Path(self.temp_dir) / "clip_bad"
        clip_dir.mkdir()
        (clip_dir / "summary.json").write_text(json.dumps({"status": "completed"}), encoding="utf-8")
        with self.assertRaises(ValueError) as ctx:
            load_predictions_from_dir(self.temp_dir)
        self.assertIn("completed_reps", str(ctx.exception).lower())


class TestEvaluateCountsMetrics(unittest.TestCase):
    """Tests for repetition count metrics computation and unreviewed exclusions."""

    def test_perfect_match(self) -> None:
        predictions = {
            "clip_01_barbell_rack": 2,
            "clip_03_kettlebell_side": 6,
        }
        ground_truth = {
            "clip_01_barbell_rack": 2,
            "clip_03_kettlebell_side": 6,
        }
        results = evaluate_counts(predictions, ground_truth)

        self.assertEqual(results["evaluated_clips"], 2)
        self.assertEqual(results["unreviewed_clips"], 0)
        self.assertEqual(results["unreviewed_clip_ids"], [])
        self.assertEqual(results["exact_matches"], 2)
        self.assertEqual(results["exact_match_ratio"], 1.0)
        self.assertEqual(results["mean_absolute_error"], 0.0)

        self.assertEqual(len(results["per_clip_results"]), 2)
        self.assertTrue(results["per_clip_results"][0]["exact_match"])
        self.assertEqual(results["per_clip_results"][0]["absolute_error"], 0)
        self.assertTrue(results["per_clip_results"][1]["exact_match"])
        self.assertEqual(results["per_clip_results"][1]["absolute_error"], 0)

    def test_off_by_one_and_errors(self) -> None:
        predictions = {
            "clip_01": 2,  # human 2 -> error 0
            "clip_02": 5,  # human 4 -> error 1 (off by +1)
            "clip_03": 8,  # human 10 -> error 2 (off by -2)
        }
        ground_truth = {
            "clip_01": 2,
            "clip_02": 4,
            "clip_03": 10,
        }
        results = evaluate_counts(predictions, ground_truth)

        self.assertEqual(results["evaluated_clips"], 3)
        self.assertEqual(results["unreviewed_clips"], 0)
        self.assertEqual(results["exact_matches"], 1)
        self.assertAlmostEqual(results["exact_match_ratio"], 1 / 3, places=4)
        # sum of absolute errors = 0 + 1 + 2 = 3. MAE = 3 / 3 = 1.0
        self.assertEqual(results["mean_absolute_error"], 1.0)

        per_clip = {r["clip_id"]: r for r in results["per_clip_results"]}
        self.assertEqual(per_clip["clip_01"]["absolute_error"], 0)
        self.assertTrue(per_clip["clip_01"]["exact_match"])
        self.assertEqual(per_clip["clip_02"]["absolute_error"], 1)
        self.assertFalse(per_clip["clip_02"]["exact_match"])
        self.assertEqual(per_clip["clip_03"]["absolute_error"], 2)
        self.assertFalse(per_clip["clip_03"]["exact_match"])

    def test_unreviewed_clips_excluded_never_treated_as_zero(self) -> None:
        # Crucial test: clip_02 and clip_04 have model_count = 0.
        # If unreviewed was treated as 0, they would falsely count as exact matches!
        predictions = {
            "clip_01": 2,
            "clip_02": 0,
            "clip_03": 6,
            "clip_04": 0,
            "clip_05": 0,
        }
        # Only clip_01 and clip_03 have ground truth. clip_02 is explicitly None,
        # clips 04 and 05 are absent from ground truth.
        ground_truth = {
            "clip_01": 2,
            "clip_02": None,
            "clip_03": 6,
        }
        results = evaluate_counts(predictions, ground_truth)

        # Total predictions: 5. Evaluated: 2. Unreviewed: 3.
        self.assertEqual(results["evaluated_clips"], 2)
        self.assertEqual(results["unreviewed_clips"], 3)
        self.assertEqual(
            results["unreviewed_clip_ids"],
            ["clip_02", "clip_04", "clip_05"],
        )
        # Exactly 2 matches out of 2 evaluated clips (NOT 5 out of 5!)
        self.assertEqual(results["exact_matches"], 2)
        self.assertEqual(results["exact_match_ratio"], 1.0)
        self.assertEqual(results["mean_absolute_error"], 0.0)

    def test_empty_predictions_produces_none_metrics(self) -> None:
        results = evaluate_counts(predictions={}, ground_truth={})
        self.assertEqual(results["evaluated_clips"], 0)
        self.assertEqual(results["unreviewed_clips"], 0)
        self.assertIsNone(results["exact_match_ratio"])
        self.assertIsNone(results["mean_absolute_error"])
        self.assertEqual(results["per_clip_results"], [])

    def test_unknown_ground_truth_id_rejected_by_default(self) -> None:
        predictions = {"clip_01": 2}
        ground_truth = {"clip_01": 2, "clip_unknown": 5}
        with self.assertRaises(ValueError) as ctx:
            evaluate_counts(predictions, ground_truth, reject_unknown_ground_truth=True)
        self.assertIn("unknown clip id", str(ctx.exception).lower())

    def test_unknown_ground_truth_id_allowed_when_flag_disabled(self) -> None:
        predictions = {"clip_01": 2}
        ground_truth = {"clip_01": 2, "clip_unknown": 5}
        results = evaluate_counts(predictions, ground_truth, reject_unknown_ground_truth=False)
        self.assertEqual(results["evaluated_clips"], 1)


class TestReportFormatting(unittest.TestCase):
    """Tests for human-readable evaluation report text output."""

    def test_report_includes_denominators_and_unreviewed(self) -> None:
        predictions = {
            "clip_01": 2,
            "clip_02": 0,
            "clip_03": 6,
        }
        ground_truth = {
            "clip_01": 2,
            "clip_03": 6,
        }
        results = evaluate_counts(predictions, ground_truth)
        report = format_evaluation_report(results, "mock_pred_dir", "mock_gt.json")

        self.assertIn("Total prediction clips : 3", report)
        self.assertIn("Evaluated clips (reviewed) : 2", report)
        self.assertIn("Unreviewed clips (excluded): 1", report)
        self.assertIn("2 / 2 (100.0%)", report)
        self.assertIn("Mean Absolute Error (MAE)     : 0.00 reps", report)
        self.assertIn("clip_02", report)
        self.assertIn("Methodological Limitations", report)

    def test_report_handles_zero_evaluated_clips(self) -> None:
        results = evaluate_counts({}, {})
        report = format_evaluation_report(results)
        self.assertIn("0 evaluated clips", report)
        self.assertIn("N/A", report)


class TestCliExecution(unittest.TestCase):
    """Tests for CLI arguments, execution, receipts, and error codes."""

    def setUp(self) -> None:
        self.temp_dir = tempfile.mkdtemp(prefix="test_eval_cli_")
        self.addCleanup(lambda: shutil.rmtree(self.temp_dir, ignore_errors=True))

        # Build mock prediction directory
        self.pred_dir = Path(self.temp_dir) / "preds"
        self.pred_dir.mkdir()
        for cid, count in [("clip_01", 2), ("clip_02", 0), ("clip_03", 6)]:
            sub = self.pred_dir / cid
            sub.mkdir()
            (sub / "summary.json").write_text(json.dumps({"completed_reps": count}), encoding="utf-8")

        # Build mock ground truth JSON
        self.gt_file = Path(self.temp_dir) / "ground_truth.json"
        gt_data = {
            "clips": [
                {"clip_id": "clip_01", "human_count": 2},
                {"clip_id": "clip_03", "human_count": 6},
            ]
        }
        self.gt_file.write_text(json.dumps(gt_data), encoding="utf-8")

    def test_cli_help(self) -> None:
        with self.assertRaises(SystemExit) as ctx:
            with redirect_stdout(io.StringIO()):
                main(["--help"])
        self.assertEqual(ctx.exception.code, 0)

    def test_cli_success_with_output_receipt(self) -> None:
        receipt_path = Path(self.temp_dir) / "receipt.json"
        stdout_buf = io.StringIO()

        with redirect_stdout(stdout_buf):
            ret = main([
                "--predictions-dir", str(self.pred_dir),
                "--ground-truth", str(self.gt_file),
                "--output-json", str(receipt_path),
            ])

        self.assertEqual(ret, 0)
        output = stdout_buf.getvalue()
        self.assertIn("Evaluated clips (reviewed) : 2", output)
        self.assertIn("Unreviewed clips (excluded): 1", output)
        self.assertIn("Evaluation receipt written to", output)

        self.assertTrue(receipt_path.is_file())
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        self.assertEqual(receipt["evaluated_clips"], 2)
        self.assertEqual(receipt["unreviewed_clips"], 1)
        self.assertEqual(receipt["exact_matches"], 2)
        self.assertEqual(receipt["mean_absolute_error"], 0.0)
        self.assertEqual(receipt["unreviewed_clip_ids"], ["clip_02"])

    def test_cli_missing_ground_truth_file_fails(self) -> None:
        stderr_buf = io.StringIO()
        with redirect_stderr(stderr_buf):
            ret = main([
                "--predictions-dir", str(self.pred_dir),
                "--ground-truth", str(Path(self.temp_dir) / "nonexistent.json"),
            ])
        self.assertEqual(ret, 1)
        self.assertIn("not found", stderr_buf.getvalue().lower())

    def test_cli_missing_predictions_dir_fails(self) -> None:
        stderr_buf = io.StringIO()
        with redirect_stderr(stderr_buf):
            ret = main([
                "--predictions-dir", str(Path(self.temp_dir) / "nonexistent_dir"),
                "--ground-truth", str(self.gt_file),
            ])
        self.assertEqual(ret, 1)
        self.assertIn("not found", stderr_buf.getvalue().lower())


class TestRealDatasetBaseline(unittest.TestCase):
    """Integration check using actual workspace gym_pose/output/real_squats data."""

    def test_real_squats_evaluation(self) -> None:
        ws_root = Path(__file__).resolve().parents[1]
        real_squats_dir = ws_root / "output" / "real_squats"
        gt_path = real_squats_dir / "reviewed_counts.json"

        if not real_squats_dir.is_dir() or not gt_path.is_file():
            self.skipTest("Real squats output directory or reviewed_counts.json not found")

        predictions = load_predictions_from_dir(real_squats_dir)
        ground_truth = load_ground_truth(gt_path)
        results = evaluate_counts(predictions, ground_truth)

        # Baseline check: 5 clips total in real_squats; 2 reviewed (clip_01, clip_03); 3 unreviewed
        self.assertEqual(results["evaluated_clips"], 2)
        self.assertEqual(results["unreviewed_clips"], 3)
        self.assertEqual(results["exact_matches"], 2)
        self.assertEqual(results["exact_match_ratio"], 1.0)
        self.assertEqual(results["mean_absolute_error"], 0.0)

        clip_ids = [r["clip_id"] for r in results["per_clip_results"]]
        self.assertEqual(clip_ids, ["clip_01_barbell_rack", "clip_03_kettlebell_side"])
        self.assertEqual(
            results["unreviewed_clip_ids"],
            ["clip_02_wikipedia_compilation", "clip_04_kettlebell_goblet", "clip_05_single_leg"],
        )



class TestEvaluationReviewRegressions(unittest.TestCase):
    def test_explicit_unknown_is_not_replaced_with_an_alias_count(self):
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)/"truth.json"
            for payload in ({"clip":{"human_count":None,"count":3}}, {"clips":[{"clip_id":"clip","human_count":None,"count":3}]}):
                p.write_text(json.dumps(payload))
                self.assertEqual(load_ground_truth(p),{"clip":None})

    def test_duplicate_summary_keys_and_failed_runs_are_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)/"summary.json"
            p.write_text('{"completed_reps":2,"completed_reps":4}')
            with self.assertRaises(ValueError):load_prediction_summary(p)
            p.write_text(json.dumps({"status":"failed","completed_reps":2}))
            with self.assertRaises(ValueError):load_prediction_summary(p)

    def test_cli_cannot_overwrite_human_labels(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);clip=root/"clip";clip.mkdir()
            (clip/"summary.json").write_text(json.dumps({"completed_reps":2}))
            truth=root/"truth.json";truth.write_text(json.dumps({"clip":2}))
            original=truth.read_bytes()
            self.assertEqual(main(["--predictions-dir",str(clip),"--ground-truth",str(truth),"--output-json",str(truth)]),1)
            self.assertEqual(truth.read_bytes(),original)


if __name__ == "__main__":
    unittest.main()
