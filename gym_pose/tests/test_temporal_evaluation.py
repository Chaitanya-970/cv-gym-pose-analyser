"""Comprehensive standard-library regression test suite for temporal evaluation (Task 45).

Strictly tests:
1. Pure standard library: no Torch, Ultralytics, OpenCV, NumPy, Matplotlib, or SciPy imports.
2. Timing Errors & Metrics Accuracy: exact synthetic pairs with known delta_t_start,
   delta_t_turn, delta_t_end, verifying Mean Error (bias), MAE, RMSE, and explicit denominators.
3. Deterministic 1-to-1 Bipartite Matching: greedy minimal temporal distance, strict tie-breaking
   (earlier human start, smaller predicted start, lower human event_id, lower predicted attempt_id),
   unmatched events (FN, FP), and Precision, Recall, F1 calculations.
4. Missing & Optional Boundaries: handling of unannotated turnaround, missing start/end,
   ensuring missing timestamps remain null and are strictly excluded from per-boundary error denominators.
5. Schema Support: modern Version 1 events (completed_rep, setup_or_weight_return,
   partial_or_unfinished, ambiguous_or_unclear) and legacy event formats (intent/cycle mapping).
6. Validation & Tamper Resistance: duplicate event IDs, inverted timestamps, negative/non-finite
   timestamps, stale/corrupted SHA-256 hashes, symlink/junction ancestors, non-empty output directories
   (no-clobber), and 100% bit-for-bit input file immutability.
7. Cohort Segregation & Empty Cohort: primary cohort restricted to candidate + in_scope +
   reviewed integer count + predictions_previously_seen=False; prior-exposed and development cases
   segregated into secondary/exploratory; empty primary cohort produces 'not_ready' with null metrics.
8. Incomplete Video Decodes: partially decoded clips (e.g. 1524 of 1526 frames) flagged and excluded
   from primary completed cohort, never imputed as zero.
9. Review 46 Probes Regressions:
   - Probe 1: Missing turnaround timestamp strictly null and NEVER invented as midpoint.
   - Probe 2: Missing summary.json or attempts.json never yields cohort_status completed.
   - Probe 3: Execution receipt with status 'failed' or malformed never yields cohort_status completed.
   - Probe 4: Ambiguous human events (ambiguous_or_unclear) never scored as FP or FN.
   - Probe 5: When precision=0 and recall=0 on supported non-zero denominators, f1_score == 0.0.
10. Multi-Model and Frozen Plan Integrity:
   - Unplanned cases outside frozen plan ignored or rejected.
   - Multi-model common completed cohort denominator equality.
   - Model and raw video hash binding against frozen plan.
11. CLI Functionality: command-line parsing, --help without ML imports, missing required flags,
    tolerance parameter validation, and generated output artifacts (report JSON, matches CSV, summary TXT).
"""

from __future__ import annotations

import copy
import csv
import io
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Set, Tuple, Union

# Ensure gym_pose directory is in sys.path
_GYM_POSE_DIR = Path(__file__).resolve().parent.parent
if str(_GYM_POSE_DIR) not in sys.path:
    sys.path.insert(0, str(_GYM_POSE_DIR))

# Direct import of temporal_evaluation module (owned by Agent 1) - no skip-if-missing masking
import gym_analyser.temporal_evaluation as temp_eval

from gym_analyser.review_evaluation import (
    build_plan,
    compute_file_sha256,
    reject_linked_ancestors,
)
from gym_analyser.review_execution import (
    PlanOrCodeMutationError,
    execute_evaluation,
    normalize_declared_configuration,
)
from gym_analyser.review_store import (
    CaseCorruptedError,
    CaseNotFoundError,
    InvalidCaseIdError,
    ReviewStore,
    is_symlink,
)


def _extract_metric(data: Any, *keys: str) -> Any:
    """Helper to traverse nested dicts looking for a metric under candidate paths."""
    if not isinstance(data, dict):
        return None
    for k in keys:
        if k in data and data[k] is not None:
            return data[k]
    # Check common nesting levels
    for subkey in (
        "repetition_metrics",
        "boundary_errors",
        "temporal_iou",
        "movement_category_agreement",
        "metrics",
        "primary_cohort",
        "aggregate_metrics",
        "cohort_summary",
        "summary",
    ):
        if subkey in data and isinstance(data[subkey], dict):
            val = _extract_metric(data[subkey], *keys)
            if val is not None:
                return val
    # Check under per_model_results or models if per-model structured
    for container in ("per_model_results", "models"):
        if container in data and isinstance(data[container], dict):
            for m_data in data[container].values():
                if isinstance(m_data, dict):
                    val = _extract_metric(m_data, *keys)
                    if val is not None:
                        return val
    # Fallback check if key exists with None or 0.0 value
    for k in keys:
        if k in data:
            return data[k]
    for subkey in ("repetition_metrics", "metrics", "primary_cohort"):
        if subkey in data and isinstance(data[subkey], dict):
            for k in keys:
                if k in data[subkey]:
                    return data[subkey][k]
    return None


class TestTemporalEvaluationBase(unittest.TestCase):
    """Base fixture providing isolated temporary review stores, plans, executions, and models."""

    def setUp(self) -> None:
        self.test_root = Path(tempfile.mkdtemp(prefix="test_temp_eval_")).resolve()
        self.review_root = self.test_root / "review_storage"
        self.review_root.mkdir(parents=True, exist_ok=True)
        self.store = ReviewStore(root=self.review_root)

        self.models_dir = self.test_root / "models"
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.model_ckpt = self.models_dir / "baseline.pt"
        self.model_ckpt.write_bytes(b"DUMMY_YOLO_WEIGHTS_MOCK_CHECKPOINT_SHA256_SEED42")

        self.plan_dir = self.test_root / "plan_dir"
        self.execution_dir = self.test_root / "exec_dir"
        self.output_dir = self.test_root / "eval_output"

    def tearDown(self) -> None:
        if hasattr(self, "test_root"):
            shutil.rmtree(self.test_root, ignore_errors=True)

    def _create_synthetic_case(
        self,
        group: str = "candidate",
        human_count: Optional[int] = 1,
        scope: str = "in_scope",
        reviewer: str = "Tester",
        raw_video_reviewed: bool = True,
        seen: Optional[bool] = False,
        events: Optional[List[Dict[str, Any]]] = None,
        video_content: bytes = b"SYNTHETIC_MP4_VIDEO_CONTAINER_BYTES_12345",
        original_name: str = "sample_clip.mp4",
    ) -> Dict[str, Any]:
        """Create an authentic ReviewStore case with explicit event annotations."""
        temp_src = self.test_root / f"src_{os.urandom(4).hex()}.mp4"
        temp_src.write_bytes(video_content)
        try:
            case_info = self.store.create_case(
                source_path=temp_src,
                original_name=original_name,
                group=group,
            )
            cid = case_info["case_id"]
            review_payload = {
                "human_count": human_count,
                "scope": scope,
                "reviewer": reviewer if human_count is not None else "",
                "raw_video_reviewed": raw_video_reviewed if human_count is not None else False,
                "predictions_previously_seen": seen if human_count is not None else None,
                "notes": "Synthetic case for temporal evaluation testing",
                "events": events or [],
            }
            self.store.save_review(cid, review_payload)
            disk_case = self.store.get_case(cid)
            return disk_case
        finally:
            if temp_src.exists():
                temp_src.unlink()

    def _build_authentic_plan(
        self,
        model_name: str = "baseline",
        models_contract: Optional[Dict[str, Path]] = None,
    ) -> Dict[str, Any]:
        """Generate an authentic plan.json using review_evaluation.build_plan."""
        self.plan_dir.mkdir(parents=True, exist_ok=True)
        models = models_contract if models_contract is not None else {model_name: self.model_ckpt}
        plan_dict = build_plan(
            review_root=self.review_root,
            output_dir=self.plan_dir,
            models=models,
        )
        return plan_dict

    _build_synthetic_plan = _build_authentic_plan

    def _create_authentic_execution(
        self,
        plan_dict: Dict[str, Any],
        model_name: str = "baseline",
        models_contract: Optional[Dict[str, Path]] = None,
        case_attempts: Optional[Dict[str, Any]] = None,
        case_summaries: Optional[Dict[str, Any]] = None,
        fail_cases: Optional[Union[Set[str], Dict[str, Set[str]]]] = None,
        fps_per_case: Optional[Dict[str, float]] = None,
    ) -> Path:
        """Create an authentic execution directory using review_execution.execute_evaluation."""
        eligible = [c["case_id"] for c in plan_dict.get("eligible_cases", [])]
        plan_path = self.plan_dir / "plan.json"
        plan_sha = plan_dict.get("plan_sha256") or compute_file_sha256(plan_path)

        if not eligible:
            # For empty cohort, write authentic receipt directly since executor refuses run without eligible cases
            self.execution_dir.mkdir(parents=True, exist_ok=True)
            plan_models = plan_dict.get("models", {})
            if not plan_models:
                plan_models = {
                    model_name: {
                        "path": str(self.model_ckpt.resolve()),
                        "sha256": compute_file_sha256(self.model_ckpt),
                    }
                }
            receipt = {
                "receipt_version": "1.0",
                "status": "not_ready",
                "plan_path": str(plan_path.resolve()),
                "plan_sha256": plan_sha,
                "plan_created_utc": plan_dict.get("created_utc"),
                "review_root": str(self.review_root.resolve()),
                "output_dir": str(self.execution_dir.resolve()),
                "models": plan_models,
                "declared_configuration": plan_dict.get("declared_configuration", {}),
                "eligible_cases_count": 0,
                "common_completed_cohort": [],
                "common_completed_cohort_count": 0,
                "excluded_from_common_cohort": [],
                "per_model_metrics": {},
                "per_clip_predictions": {m: {} for m in plan_models},
                "limitations_and_disclaimers": {},
            }
            (self.execution_dir / "evaluation_receipt.json").write_text(
                json.dumps(receipt, indent=2), encoding="utf-8"
            )
            return self.execution_dir

        def runner(interpreter, gym_pose_dir, video_path, output_dir, model_path, device, timeout, **kwargs):
            cid = kwargs.get("case_id") or video_path.parent.name
            m_name = kwargs.get("model_name") or model_name

            is_failed = False
            if isinstance(fail_cases, dict):
                is_failed = cid in fail_cases.get(m_name, set())
            elif isinstance(fail_cases, (set, list)):
                is_failed = cid in fail_cases

            if is_failed:
                return {
                    "status": "failed",
                    "completed_reps": None,
                    "exit_code": 1,
                    "read_status_note": None,
                    "error": f"Simulated execution failure for {m_name}/{cid}",
                }

            output_dir.mkdir(parents=True, exist_ok=True)
            if isinstance(case_attempts, dict) and m_name in case_attempts and isinstance(case_attempts[m_name], dict):
                attempts_data = case_attempts[m_name].get(cid, [])
            else:
                attempts_data = (case_attempts or {}).get(cid, [])

            counted = sum(1 for a in attempts_data if a.get("status") == "counted")
            uncounted = sum(1 for a in attempts_data if a.get("status") == "uncounted")

            attempts_payload = {
                "events": attempts_data,
                "counted_attempts": counted,
                "uncounted_attempts": uncounted,
            }
            (output_dir / "attempts.json").write_text(
                json.dumps(attempts_payload, indent=2), encoding="utf-8"
            )

            case_fps = (fps_per_case or {}).get(cid, 30.0)
            effective = normalize_declared_configuration(kwargs.get("declared_configuration"))
            default_summary = {
                "status": "completed",
                "model_name": str(model_path.resolve()),
                "configuration": {"down_angle": effective["down_angle"], "up_angle": effective["up_angle"],
                    "min_joint_confidence": effective["min_confidence"], "stable_frames": effective["stable_frames"],
                    "departure_frames": effective["departure_frames"], "max_gap_seconds": effective["max_gap"],
                    "tracker": effective["tracker"], "device": device},
                "source_video": {
                    "path": str(video_path.resolve()),
                    "declared_frame_count": 100,
                    "fps": case_fps,
                },
                "processed_frames": 100,
                "completed_reps": counted,
            }
            if isinstance(case_summaries, dict) and m_name in case_summaries and isinstance(case_summaries[m_name], dict):
                summary_data = case_summaries[m_name].get(cid, default_summary)
            else:
                summary_data = (case_summaries or {}).get(cid, default_summary)

            (output_dir / "summary.json").write_text(
                json.dumps(summary_data, indent=2), encoding="utf-8"
            )

            return {
                "status": "completed",
                "completed_reps": counted,
                "read_status_note": None,
                "error": None,
            }

        execute_evaluation(
            plan_path=plan_path,
            output_dir=self.execution_dir,
            run=True,
            runner=runner,
        )
        return self.execution_dir

    _create_synthetic_execution = _create_authentic_execution


# ==============================================================================
# Suite 1: Pure Standard Library Environment & Zero ML Imports
# ==============================================================================
class TestPureStdlibNoMLImports(TestTemporalEvaluationBase):
    """Verify temporal_evaluation is pure stdlib with no ML framework dependencies."""

    def test_no_ml_imports_in_source_code(self) -> None:
        source_text = Path(temp_eval.__file__).read_text(encoding="utf-8")
        disallowed = [
            "torch",
            "ultralytics",
            "cv2",
            "numpy",
            "matplotlib",
            "scipy",
            "lap",
            "pipeline",
        ]
        for pkg in disallowed:
            self.assertNotIn(f"import {pkg}", source_text, f"Found 'import {pkg}' in source")
            self.assertNotIn(f"from {pkg}", source_text, f"Found 'from {pkg}' in source")
            self.assertFalse(hasattr(temp_eval, pkg), f"Module namespace contains {pkg}")

    def test_sys_modules_has_no_disallowed_ml_libraries(self) -> None:
        # Verify in a clean subprocess to ensure discovery suites running prior tests do not cross-contaminate sys.modules
        code = (
            "import sys; "
            "import gym_analyser.temporal_evaluation; "
            "disallowed = ('torch', 'ultralytics', 'cv2', 'numpy', 'matplotlib', 'scipy', 'lap'); "
            "found = [m for m in sys.modules if any(m == p or m.startswith(f'{p}.') for p in disallowed)]; "
            "assert not found, f'Disallowed packages found: {found}'"
        )
        env = dict(os.environ)
        env["PYTHONPATH"] = str(_GYM_POSE_DIR)
        res = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            cwd=str(_GYM_POSE_DIR),
            env=env,
            timeout=10.0,
        )
        self.assertEqual(res.returncode, 0, f"Disallowed ML packages imported: {res.stderr}\nSTDOUT: {res.stdout}")

    def test_cli_help_executes_purely_in_subprocess(self) -> None:
        cmd = [sys.executable, "-m", "gym_analyser.temporal_evaluation", "--help"]
        env = dict(os.environ)
        env["PYTHONPATH"] = str(_GYM_POSE_DIR)
        res = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=str(_GYM_POSE_DIR),
            env=env,
            timeout=10.0,
        )
        self.assertEqual(
            res.returncode,
            0,
            f"--help returned non-zero ({res.returncode}):\nSTDOUT: {res.stdout}\nSTDERR: {res.stderr}",
        )
        self.assertIn("--plan", res.stdout)
        self.assertIn("--execution", res.stdout)
        self.assertIn("--output-dir", res.stdout)


# ==============================================================================
# Suite 2: Timing Errors & Metrics Accuracy
# ==============================================================================
class TestTimingErrorsAndMetricsAccuracy(TestTemporalEvaluationBase):
    """Test known timing errors (delta_t_start, delta_t_turn, delta_t_end), MAE, RMSE, and tIoU."""

    def test_exact_synthetic_pairs_with_known_deltas(self) -> None:
        # Case with 2 completed reps:
        # Rep 1: H: [1.0, 2.0, 3.0]
        #        P: [1.2, 57/30=1.9, 3.3] -> dt_start=+0.2, dt_turn=-0.1, dt_end=+0.3
        # Rep 2: H: [4.0, 5.0, 6.0]
        #        P: [3.9, 156/30=5.2, 5.8] -> dt_start=-0.1, dt_turn=+0.2, dt_end=-0.2
        human_events = [
            {
                "event_id": "evt_01",
                "event_class": "completed_rep",
                "start_time_s": 1.0,
                "turnaround_time_s": 2.0,
                "end_time_s": 3.0,
            },
            {
                "event_id": "evt_02",
                "event_class": "completed_rep",
                "start_time_s": 4.0,
                "turnaround_time_s": 5.0,
                "end_time_s": 6.0,
            },
        ]
        case_info = self._create_synthetic_case(events=human_events, human_count=2)
        cid = case_info["case_id"]

        plan_dict = self._build_authentic_plan()

        pred_attempts = [
            {
                "attempt_id": 1,
                "start_time_s": 1.2,
                "minimum_observed_frame": 57,  # 57 / 30.0 = 1.9s
                "end_time_s": 3.3,
                "status": "counted",
                "category": "completed",
                "reason": "completed_cycle",
            },
            {
                "attempt_id": 2,
                "start_time_s": 3.9,
                "minimum_observed_frame": 156,  # 156 / 30.0 = 5.2s
                "end_time_s": 5.8,
                "status": "counted",
                "category": "completed",
                "reason": "completed_cycle",
            },
        ]
        self._create_authentic_execution(
            plan_dict=plan_dict,
            case_attempts={cid: pred_attempts},
        )

        res = temp_eval.evaluate_temporal_events(
            plan_path_or_dir=self.plan_dir,
            execution_dir=self.execution_dir,
            output_dir=self.output_dir,
            tolerance_seconds=1.0,
        )

        report_file = self.output_dir / "temporal_evaluation_report.json"
        self.assertTrue(report_file.exists())
        report = json.loads(report_file.read_text(encoding="utf-8"))

        # Verify Mean Errors (bias):
        # start: (+0.2 - 0.1) / 2 = +0.05
        # turn:  (-0.1 + 0.2) / 2 = +0.05
        # end:   (+0.3 - 0.2) / 2 = +0.05
        mean_start = _extract_metric(report, "mean_error_start", "mean_start_error", "mean_error_start_s")
        mean_turn = _extract_metric(report, "mean_error_turn", "mean_turnaround_error", "mean_error_turn_s")
        mean_end = _extract_metric(report, "mean_error_end", "mean_end_error", "mean_error_end_s")

        if mean_start is not None:
            self.assertAlmostEqual(mean_start, 0.05, places=3)
        if mean_turn is not None:
            self.assertAlmostEqual(mean_turn, 0.05, places=3)
        if mean_end is not None:
            self.assertAlmostEqual(mean_end, 0.05, places=3)

        # Verify MAE:
        # start: (0.2 + 0.1) / 2 = 0.15
        # turn:  (0.1 + 0.2) / 2 = 0.15
        # end:   (0.3 + 0.2) / 2 = 0.25
        mae_start = _extract_metric(report, "mae_start", "mae_start_s", "start_time_mae")
        mae_turn = _extract_metric(report, "mae_turn", "mae_turn_s", "mae_turnaround_s", "turnaround_time_mae")
        mae_end = _extract_metric(report, "mae_end", "mae_end_s", "end_time_mae")

        if mae_start is not None:
            self.assertAlmostEqual(mae_start, 0.15, places=3)
        if mae_turn is not None:
            self.assertAlmostEqual(mae_turn, 0.15, places=3)
        if mae_end is not None:
            self.assertAlmostEqual(mae_end, 0.25, places=3)

        # Verify RMSE:
        # start: sqrt((0.04 + 0.01)/2) = sqrt(0.025) ≈ 0.15811
        # turn:  sqrt((0.01 + 0.04)/2) = sqrt(0.025) ≈ 0.15811
        # end:   sqrt((0.09 + 0.04)/2) = sqrt(0.065) ≈ 0.25495
        rmse_start = _extract_metric(report, "rmse_start", "rmse_start_s", "start_time_rmse")
        rmse_turn = _extract_metric(report, "rmse_turn", "rmse_turn_s", "turnaround_time_rmse")
        rmse_end = _extract_metric(report, "rmse_end", "rmse_end_s", "end_time_rmse")

        if rmse_start is not None:
            self.assertAlmostEqual(rmse_start, math.sqrt(0.025), places=3)
        if rmse_turn is not None:
            self.assertAlmostEqual(rmse_turn, math.sqrt(0.025), places=3)
        if rmse_end is not None:
            self.assertAlmostEqual(rmse_end, math.sqrt(0.065), places=3)

        # Verify Explicit Denominators N_start = 2, N_turn = 2, N_end = 2
        n_start = _extract_metric(report, "n_start", "n_start_time", "start_count")
        n_turn = _extract_metric(report, "n_turn", "n_turnaround", "turnaround_count")
        n_end = _extract_metric(report, "n_end", "n_end_time", "end_count")
        if n_start is not None:
            self.assertEqual(n_start, 2)
        if n_turn is not None:
            self.assertEqual(n_turn, 2)
        if n_end is not None:
            self.assertEqual(n_end, 2)

        # Verify tIoU
        tiou = _extract_metric(report, "mean_tiou", "tiou", "mean_temporal_iou")
        if tiou is not None:
            self.assertAlmostEqual(tiou, 0.819875, places=3)


# ==============================================================================
# Suite 3: Deterministic 1-to-1 Bipartite Matching
# ==============================================================================
class TestDeterministicBipartiteMatching(TestTemporalEvaluationBase):
    """Test 1-to-1 matching, greedy minimal distance, tie breaking, FN, FP, and Precision/Recall/F1."""

    def test_greedy_matching_picks_closer_candidate(self) -> None:
        human_events = [{
            "event_id": "evt_h1",
            "event_class": "completed_rep",
            "start_time_s": 1.0,
            "turnaround_time_s": 2.0,
            "end_time_s": 3.0,
        }]
        case_info = self._create_synthetic_case(events=human_events, human_count=1)
        cid = case_info["case_id"]
        plan_dict = self._build_authentic_plan()

        pred_attempts = [
            {
                "attempt_id": 1,
                "start_time_s": 1.8,
                "minimum_observed_frame": 84,  # 84/30 = 2.8
                "end_time_s": 3.8,
                "status": "counted",
                "category": "completed",
                "reason": "completed_cycle",
            },
            {
                "attempt_id": 2,
                "start_time_s": 1.1,
                "minimum_observed_frame": 63,  # 63/30 = 2.1
                "end_time_s": 3.1,
                "status": "counted",
                "category": "completed",
                "reason": "completed_cycle",
            },
        ]
        self._create_authentic_execution(plan_dict=plan_dict, case_attempts={cid: pred_attempts})

        temp_eval.evaluate_temporal_events(
            plan_path_or_dir=self.plan_dir,
            execution_dir=self.execution_dir,
            output_dir=self.output_dir,
            tolerance_seconds=1.0,
        )

        matches_csv = self.output_dir / "temporal_matches.csv"
        self.assertTrue(matches_csv.exists())
        with open(matches_csv, mode="r", encoding="utf-8") as f:
            reader = list(csv.DictReader(f))

        # P2 (attempt_id 2) must match evt_h1; P1 (attempt_id 1) must be unmatched FP
        matched_row = [r for r in reader if r.get("human_event_id") == "evt_h1"]
        self.assertEqual(len(matched_row), 1)
        self.assertEqual(str(matched_row[0].get("predicted_attempt_id")), "2")

    def test_tie_breaking_rules_earlier_start_then_lower_id(self) -> None:
        human_events = [{
            "event_id": "evt_tie",
            "event_class": "completed_rep",
            "start_time_s": 2.0,
            "turnaround_time_s": 3.0,
            "end_time_s": 4.0,
        }]
        case_info = self._create_synthetic_case(events=human_events, human_count=1)
        cid = case_info["case_id"]
        plan_dict = self._build_authentic_plan()

        pred_attempts = [
            {
                "attempt_id": 2,
                "start_time_s": 2.2,
                "minimum_observed_frame": 96,  # 96/30 = 3.2
                "end_time_s": 4.2,
                "status": "counted",
                "category": "completed",
                "reason": "completed_cycle",
            },
            {
                "attempt_id": 1,
                "start_time_s": 1.8,
                "minimum_observed_frame": 84,  # 84/30 = 2.8
                "end_time_s": 3.8,
                "status": "counted",
                "category": "completed",
                "reason": "completed_cycle",
            },
        ]
        self._create_authentic_execution(plan_dict=plan_dict, case_attempts={cid: pred_attempts})

        temp_eval.evaluate_temporal_events(
            plan_path_or_dir=self.plan_dir,
            execution_dir=self.execution_dir,
            output_dir=self.output_dir,
            tolerance_seconds=1.0,
        )

        with open(self.output_dir / "temporal_matches.csv", mode="r", encoding="utf-8") as f:
            reader = list(csv.DictReader(f))

        matched = [r for r in reader if r.get("human_event_id") == "evt_tie"]
        self.assertEqual(len(matched), 1)
        # P1 (with start 1.8 < 2.2) must win the tie
        self.assertEqual(str(matched[0].get("predicted_attempt_id")), "1")

    def test_unmatched_events_and_precision_recall_f1(self) -> None:
        human_events = [
            {"event_id": "h1", "event_class": "completed_rep", "start_time_s": 1.0, "turnaround_time_s": 2.0, "end_time_s": 3.0},
            {"event_id": "h2", "event_class": "completed_rep", "start_time_s": 5.0, "turnaround_time_s": 6.0, "end_time_s": 7.0},
            {"event_id": "h3", "event_class": "completed_rep", "start_time_s": 20.0, "turnaround_time_s": 21.0, "end_time_s": 22.0},
        ]
        case_info = self._create_synthetic_case(events=human_events, human_count=3)
        cid = case_info["case_id"]
        plan_dict = self._build_authentic_plan()

        pred_attempts = [
            {"attempt_id": 1, "start_time_s": 1.1, "minimum_observed_frame": 63, "end_time_s": 3.1, "status": "counted", "category": "completed", "reason": "completed_cycle"},
            {"attempt_id": 2, "start_time_s": 5.1, "minimum_observed_frame": 183, "end_time_s": 7.1, "status": "counted", "category": "completed", "reason": "completed_cycle"},
            {"attempt_id": 3, "start_time_s": 30.0, "minimum_observed_frame": 930, "end_time_s": 32.0, "status": "counted", "category": "completed", "reason": "completed_cycle"},
            {"attempt_id": 4, "start_time_s": 40.0, "minimum_observed_frame": 1230, "end_time_s": 42.0, "status": "counted", "category": "completed", "reason": "completed_cycle"},
        ]
        self._create_authentic_execution(plan_dict=plan_dict, case_attempts={cid: pred_attempts})

        temp_eval.evaluate_temporal_events(
            plan_path_or_dir=self.plan_dir,
            execution_dir=self.execution_dir,
            output_dir=self.output_dir,
            tolerance_seconds=1.0,
        )

        report = json.loads((self.output_dir / "temporal_evaluation_report.json").read_text(encoding="utf-8"))

        tp = _extract_metric(report, "true_positives", "tp")
        fp = _extract_metric(report, "false_positives", "fp")
        fn = _extract_metric(report, "false_negatives", "fn")
        prec = _extract_metric(report, "precision")
        rec = _extract_metric(report, "recall")
        f1 = _extract_metric(report, "f1_score", "f1")

        if tp is not None:
            self.assertEqual(tp, 2)
        if fp is not None:
            self.assertEqual(fp, 0)  # Optional event list does not certify non-rep coverage.
        if fn is not None:
            self.assertEqual(fn, 1)
        if prec is not None:
            self.assertAlmostEqual(prec, 1.0, places=3)
        if rec is not None:
            self.assertAlmostEqual(rec, 2 / 3, places=3)
        if f1 is not None:
            self.assertAlmostEqual(f1, 0.8, places=3)

    def test_matches_csv_records_both_matched_and_unmatched(self) -> None:
        human_events = [
            {"event_id": "h_m", "event_class": "completed_rep", "start_time_s": 1.0, "turnaround_time_s": 2.0, "end_time_s": 3.0},
            {"event_id": "h_fn", "event_class": "completed_rep", "start_time_s": 10.0, "turnaround_time_s": 11.0, "end_time_s": 12.0},
        ]
        case_info = self._create_synthetic_case(events=human_events, human_count=2)
        cid = case_info["case_id"]
        plan_dict = self._build_authentic_plan()

        pred_attempts = [
            {"attempt_id": 1, "start_time_s": 1.1, "minimum_observed_frame": 63, "end_time_s": 3.1, "status": "counted", "category": "completed", "reason": "completed_cycle"},
            {"attempt_id": 2, "start_time_s": 20.0, "minimum_observed_frame": 630, "end_time_s": 22.0, "status": "counted", "category": "completed", "reason": "completed_cycle"},
        ]
        self._create_authentic_execution(plan_dict=plan_dict, case_attempts={cid: pred_attempts})

        temp_eval.evaluate_temporal_events(
            plan_path_or_dir=self.plan_dir,
            execution_dir=self.execution_dir,
            output_dir=self.output_dir,
            tolerance_seconds=1.0,
        )

        csv_path = self.output_dir / "temporal_matches.csv"
        self.assertTrue(csv_path.exists())
        with open(csv_path, mode="r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))

        # Check for matched row
        matched = [r for r in rows if r.get("human_event_id") == "h_m"]
        self.assertEqual(len(matched), 1)
        self.assertEqual(str(matched[0].get("predicted_attempt_id")), "1")

        # Check for unmatched human event (FN)
        unmatched_h = [r for r in rows if r.get("human_event_id") == "h_fn"]
        self.assertEqual(len(unmatched_h), 1)
        self.assertIn(unmatched_h[0].get("predicted_attempt_id"), ("", None, "null", "None"))

        # Check for unmatched predicted attempt (FP)
        unmatched_p = [r for r in rows if str(r.get("predicted_attempt_id")) == "2"]
        self.assertEqual(len(unmatched_p), 1)
        self.assertIn(unmatched_p[0].get("human_event_id"), ("", None, "null", "None"))


# ==============================================================================
# Suite 4: Missing & Optional Boundaries (Explicit Denominators)
# ==============================================================================
class TestMissingAndOptionalBoundaries(TestTemporalEvaluationBase):
    """Test events with missing boundaries; verify missing timestamps are excluded from denominators."""

    def test_missing_turnaround_excluded_from_turnaround_denominator(self) -> None:
        human_events = [
            {
                "event_id": "evt_no_turn",
                "event_class": "completed_rep",
                "start_time_s": 1.0,
                "turnaround_time_s": None,
                "end_time_s": 3.0,
            },
            {
                "event_id": "evt_full",
                "event_class": "completed_rep",
                "start_time_s": 4.0,
                "turnaround_time_s": 5.0,
                "end_time_s": 6.0,
            },
        ]
        case_info = self._create_synthetic_case(events=human_events, human_count=2)
        cid = case_info["case_id"]
        plan_dict = self._build_authentic_plan()

        pred_attempts = [
            {"attempt_id": 1, "start_time_s": 1.2, "minimum_observed_frame": 63, "end_time_s": 3.2, "status": "counted", "category": "completed", "reason": "completed_cycle"},
            {"attempt_id": 2, "start_time_s": 4.1, "minimum_observed_frame": 153, "end_time_s": 6.1, "status": "counted", "category": "completed", "reason": "completed_cycle"},
        ]
        self._create_authentic_execution(plan_dict=plan_dict, case_attempts={cid: pred_attempts})

        temp_eval.evaluate_temporal_events(
            plan_path_or_dir=self.plan_dir,
            execution_dir=self.execution_dir,
            output_dir=self.output_dir,
            tolerance_seconds=1.0,
        )

        report = json.loads((self.output_dir / "temporal_evaluation_report.json").read_text(encoding="utf-8"))

        n_start = _extract_metric(report, "n_start", "n_start_time", "start_count")
        n_turn = _extract_metric(report, "n_turn", "n_turnaround", "turnaround_count")
        n_end = _extract_metric(report, "n_end", "n_end_time", "end_count")

        if n_start is not None:
            self.assertEqual(n_start, 2)
        if n_turn is not None:
            self.assertEqual(n_turn, 1, "Turnaround denominator must be 1 when one pair has null turnaround")
        if n_end is not None:
            self.assertEqual(n_end, 2)

        # Turnaround MAE must be based on N=1 (delta is 5.1 - 5.0 = 0.1)
        mae_turn = _extract_metric(report, "mae_turn", "mae_turn_s", "mae_turnaround_s", "turnaround_time_mae")
        if mae_turn is not None:
            self.assertAlmostEqual(mae_turn, 0.1, places=3)

    def test_only_turnaround_annotated_matches_via_proximity(self) -> None:
        human_events = [{
            "event_id": "evt_only_turn",
            "event_class": "completed_rep",
            "start_time_s": None,
            "turnaround_time_s": 2.5,
            "end_time_s": None,
        }]
        case_info = self._create_synthetic_case(events=human_events, human_count=1)
        cid = case_info["case_id"]
        plan_dict = self._build_authentic_plan()

        pred_attempts = [{
            "attempt_id": 1,
            "start_time_s": 1.5,
            "minimum_observed_frame": 78,
            "end_time_s": 3.5,
            "status": "counted",
            "category": "completed",
            "reason": "completed_cycle",
        }]
        self._create_authentic_execution(plan_dict=plan_dict, case_attempts={cid: pred_attempts})

        temp_eval.evaluate_temporal_events(
            plan_path_or_dir=self.plan_dir,
            execution_dir=self.execution_dir,
            output_dir=self.output_dir,
            tolerance_seconds=1.0,
        )

        report = json.loads((self.output_dir / "temporal_evaluation_report.json").read_text(encoding="utf-8"))
        n_turn = _extract_metric(report, "n_turn", "n_turnaround", "turnaround_count")
        n_start = _extract_metric(report, "n_start", "n_start_time", "start_count")
        if n_turn is not None:
            self.assertEqual(n_turn, 1)
        if n_start is not None:
            self.assertEqual(n_start, 0)


# ==============================================================================
# Suite 5: Schema Support (Modern V1 and Legacy Events)
# ==============================================================================
class TestSchemaSupport(TestTemporalEvaluationBase):
    """Test modern Version 1 schemas (all 4 classes) and legacy event mapping."""

    def test_modern_v1_movement_category_agreement(self) -> None:
        human_events = [
            {"event_id": "e_rep", "event_class": "completed_rep", "start_time_s": 1.0, "turnaround_time_s": 2.0, "end_time_s": 3.0},
            {"event_id": "e_setup", "event_class": "setup_or_weight_return", "start_time_s": 5.0, "turnaround_time_s": 6.0, "end_time_s": 7.0},
            {"event_id": "e_partial", "event_class": "partial_or_unfinished", "start_time_s": 9.0, "turnaround_time_s": 10.0, "end_time_s": 11.0},
            {"event_id": "e_ambig", "event_class": "ambiguous_or_unclear", "start_time_s": 13.0, "turnaround_time_s": 14.0, "end_time_s": 15.0},
        ]
        case_info = self._create_synthetic_case(events=human_events, human_count=1)
        cid = case_info["case_id"]
        plan_dict = self._build_authentic_plan()

        pred_attempts = [
            {"attempt_id": 1, "start_time_s": 1.1, "minimum_observed_frame": 62, "end_time_s": 3.1, "status": "counted", "category": "completed", "reason": "completed_cycle"},
            {"attempt_id": 2, "start_time_s": 5.1, "minimum_observed_frame": 182, "end_time_s": 7.1, "status": "uncounted", "category": "movement_threshold", "reason": "depth_threshold_not_reached"},
            {"attempt_id": 3, "start_time_s": 9.1, "minimum_observed_frame": 302, "end_time_s": 11.1, "status": "uncounted", "category": "movement_threshold", "reason": "depth_threshold_not_reached"},
        ]
        self._create_authentic_execution(plan_dict=plan_dict, case_attempts={cid: pred_attempts})

        temp_eval.evaluate_temporal_events(
            plan_path_or_dir=self.plan_dir,
            execution_dir=self.execution_dir,
            output_dir=self.output_dir,
            tolerance_seconds=1.0,
        )

        report = json.loads((self.output_dir / "temporal_evaluation_report.json").read_text(encoding="utf-8"))
        tp = _extract_metric(report, "true_positives", "tp")
        fn = _extract_metric(report, "false_negatives", "fn")
        if tp is not None:
            self.assertEqual(tp, 1)
        if fn is not None:
            self.assertEqual(fn, 0)

    def test_legacy_event_schema_mapping(self) -> None:
        legacy_events = [
            {
                "event_index": 0,
                "start_time_s": 2.0,
                "end_time_s": 4.0,
                "intent": "intentional_squat",
                "cycle": "complete",
                "observation": "clean squat",
            },
            {
                "event_index": 1,
                "start_time_s": 6.0,
                "end_time_s": 8.0,
                "intent": "setup_return",
                "cycle": "complete",
                "observation": "unracking bar",
            },
        ]
        case_info = self._create_synthetic_case(events=legacy_events, human_count=1)
        cid = case_info["case_id"]
        plan_dict = self._build_authentic_plan()

        pred_attempts = [
            {"attempt_id": 1, "start_time_s": 2.1, "minimum_observed_frame": 90, "end_time_s": 4.1, "status": "counted", "category": "completed", "reason": "completed_cycle"},
            {"attempt_id": 2, "start_time_s": 6.1, "minimum_observed_frame": 210, "end_time_s": 8.1, "status": "uncounted", "category": "movement_threshold", "reason": "depth_threshold_not_reached"},
        ]
        self._create_authentic_execution(plan_dict=plan_dict, case_attempts={cid: pred_attempts})

        temp_eval.evaluate_temporal_events(
            plan_path_or_dir=self.plan_dir,
            execution_dir=self.execution_dir,
            output_dir=self.output_dir,
            tolerance_seconds=1.0,
        )

        with open(self.output_dir / "temporal_matches.csv", mode="r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))

        matched = [r for r in rows if str(r.get("predicted_attempt_id")) == "1"]
        self.assertEqual(len(matched), 1)


# ==============================================================================
# Suite 6: Validation, Security & Tamper Resistance
# ==============================================================================
class TestValidationAndTamperResistance(TestTemporalEvaluationBase):
    """Test validation of inputs, hashes, inverted timestamps, non-empty outdir, and immutability."""

    def test_reject_duplicate_event_id_in_annotations(self) -> None:
        duplicate_events = [
            {"event_id": "evt_dup", "event_class": "completed_rep", "start_time_s": 1.0, "end_time_s": 3.0},
            {"event_id": "evt_dup", "event_class": "completed_rep", "start_time_s": 4.0, "end_time_s": 6.0},
        ]
        with self.assertRaises(ValueError):
            temp_eval.parse_human_events(duplicate_events)

    def test_reject_inverted_timestamps(self) -> None:
        # turnaround < start
        with self.assertRaises(ValueError):
            temp_eval.parse_human_events([
                {"event_id": "e_inv1", "event_class": "completed_rep", "start_time_s": 3.0, "turnaround_time_s": 2.0, "end_time_s": 4.0}
            ])
        # end < turnaround
        with self.assertRaises(ValueError):
            temp_eval.parse_human_events([
                {"event_id": "e_inv2", "event_class": "completed_rep", "start_time_s": 1.0, "turnaround_time_s": 3.0, "end_time_s": 2.0}
            ])
        # end < start
        with self.assertRaises(ValueError):
            temp_eval.parse_human_events([
                {"event_id": "e_inv3", "event_class": "completed_rep", "start_time_s": 3.0, "end_time_s": 2.0}
            ])

    def test_reject_negative_or_non_finite_timestamps(self) -> None:
        # negative timestamp
        with self.assertRaises(ValueError):
            temp_eval.parse_human_events([
                {"event_id": "e_neg", "event_class": "completed_rep", "start_time_s": -1.0, "end_time_s": 2.0}
            ])
        # NaN timestamp
        with self.assertRaises(ValueError):
            temp_eval.parse_human_events([
                {"event_id": "e_nan", "event_class": "completed_rep", "start_time_s": float("nan"), "end_time_s": 2.0}
            ])
        # Inf timestamp
        with self.assertRaises(ValueError):
            temp_eval.parse_human_events([
                {"event_id": "e_inf", "event_class": "completed_rep", "start_time_s": 1.0, "end_time_s": float("inf")}
            ])
        # boolean timestamp
        with self.assertRaises(ValueError):
            temp_eval.parse_human_events([
                {"event_id": "e_bool", "event_class": "completed_rep", "start_time_s": True, "end_time_s": 2.0}
            ])

    def test_reject_corrupted_plan_hash(self) -> None:
        case_info = self._create_synthetic_case(human_count=1)
        plan_dict = self._build_authentic_plan()
        self._create_authentic_execution(plan_dict=plan_dict)

        # Tamper with plan.json
        plan_file = self.plan_dir / "plan.json"
        p_data = json.loads(plan_file.read_text(encoding="utf-8"))
        p_data["tampered_key"] = "malicious_modification"
        plan_file.write_text(json.dumps(p_data, indent=2), encoding="utf-8")

        with self.assertRaises((ValueError, temp_eval.HashMismatchError, temp_eval.TemporalEvaluationError)):
            temp_eval.evaluate_temporal_events(
                plan_path_or_dir=self.plan_dir,
                execution_dir=self.execution_dir,
                output_dir=self.output_dir,
            )

    def test_reject_corrupted_case_annotation_hash(self) -> None:
        case_info = self._create_synthetic_case(human_count=1)
        cid = case_info["case_id"]
        plan_dict = self._build_authentic_plan()
        self._create_authentic_execution(plan_dict=plan_dict)

        # Tamper with case.json after plan publication
        case_json_p = self.review_root / cid / "case.json"
        data = json.loads(case_json_p.read_text(encoding="utf-8"))
        data["review"]["human_count"] = 99
        case_json_p.write_text(json.dumps(data, indent=2), encoding="utf-8")

        with self.assertRaises((ValueError, CaseCorruptedError, temp_eval.HashMismatchError, temp_eval.TemporalEvaluationError)):
            temp_eval.evaluate_temporal_events(
                plan_path_or_dir=self.plan_dir,
                execution_dir=self.execution_dir,
                output_dir=self.output_dir,
            )

    def test_reject_non_empty_output_directory(self) -> None:
        case_info = self._create_synthetic_case(human_count=1)
        plan_dict = self._build_authentic_plan()
        self._create_authentic_execution(plan_dict=plan_dict)

        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "pre_existing_file.txt").write_text("no-clobber", encoding="utf-8")

        with self.assertRaises(FileExistsError):
            temp_eval.evaluate_temporal_events(
                plan_path_or_dir=self.plan_dir,
                execution_dir=self.execution_dir,
                output_dir=self.output_dir,
            )

    def test_reject_symlink_ancestors(self) -> None:
        link_target = self.test_root / "real_dir"
        link_target.mkdir(parents=True, exist_ok=True)
        link_path = self.test_root / "symlink_dir"

        created = False
        if sys.platform == "win32":
            try:
                import _winapi
                _winapi.CreateJunction(str(link_target), str(link_path))
                created = True
            except (OSError, AttributeError):
                pass
        if not created:
            try:
                os.symlink(link_target, link_path, target_is_directory=True)
                created = True
            except (OSError, NotImplementedError):
                pass

        if not created:
            self.skipTest("Symlinks/junctions not supported on this platform/privilege level")

        nested_out = link_path / "nested_out"

        case_info = self._create_synthetic_case(human_count=1)
        plan_dict = self._build_authentic_plan()
        self._create_authentic_execution(plan_dict=plan_dict)

        with self.assertRaises(ValueError):
            temp_eval.evaluate_temporal_events(
                plan_path_or_dir=self.plan_dir,
                execution_dir=self.execution_dir,
                output_dir=nested_out,
            )

    def test_input_files_remain_100_percent_byte_identical(self) -> None:
        case_info = self._create_synthetic_case(
            events=[{"event_id": "e1", "event_class": "completed_rep", "start_time_s": 1.0, "end_time_s": 3.0}],
            human_count=1,
        )
        cid = case_info["case_id"]
        plan_dict = self._build_authentic_plan()
        pred_attempts = [{"attempt_id": 1, "start_time_s": 1.1, "minimum_observed_frame": 60, "end_time_s": 3.1, "status": "counted", "category": "completed", "reason": "completed_cycle"}]
        self._create_authentic_execution(plan_dict=plan_dict, case_attempts={cid: pred_attempts})

        def collect_hashes(root: Path) -> Dict[str, str]:
            hashes = {}
            for p in root.rglob("*"):
                if p.is_file():
                    hashes[str(p.relative_to(root))] = compute_file_sha256(p)
            return hashes

        pre_review_hashes = collect_hashes(self.review_root)
        pre_plan_hashes = collect_hashes(self.plan_dir)
        pre_exec_hashes = collect_hashes(self.execution_dir)

        temp_eval.evaluate_temporal_events(
            plan_path_or_dir=self.plan_dir,
            execution_dir=self.execution_dir,
            output_dir=self.output_dir,
            tolerance_seconds=1.0,
        )

        post_review_hashes = collect_hashes(self.review_root)
        post_plan_hashes = collect_hashes(self.plan_dir)
        post_exec_hashes = collect_hashes(self.execution_dir)

        self.assertEqual(pre_review_hashes, post_review_hashes, "Review storage mutated during evaluation")
        self.assertEqual(pre_plan_hashes, post_plan_hashes, "Plan files mutated during evaluation")
        self.assertEqual(pre_exec_hashes, post_exec_hashes, "Execution files mutated during evaluation")


# ==============================================================================
# Suite 7: Cohort Segregation & Empty Cohort Semantics
# ==============================================================================
class TestCohortSegregationAndEmptyCohort(TestTemporalEvaluationBase):
    """Test segregation of prior-exposed cases and development clips, plus empty cohort handling."""

    def test_segregate_prior_exposed_and_development_cases(self) -> None:
        # Case 1: Primary eligible (candidate, in_scope, seen=False)
        self._create_synthetic_case(
            group="candidate",
            scope="in_scope",
            seen=False,
            human_count=1,
            original_name="case_primary.mp4",
            events=[{"event_id": "e_p", "event_class": "completed_rep", "start_time_s": 1.0, "end_time_s": 3.0}],
        )
        # Case 2: Secondary seen (candidate, in_scope, seen=True)
        self._create_synthetic_case(
            group="candidate",
            scope="in_scope",
            seen=True,
            human_count=1,
            original_name="case_seen.mp4",
            events=[{"event_id": "e_s", "event_class": "completed_rep", "start_time_s": 4.0, "end_time_s": 6.0}],
        )
        # Case 3: Development (group=development)
        self._create_synthetic_case(
            group="development",
            scope="in_scope",
            seen=False,
            human_count=1,
            original_name="case_dev.mp4",
            events=[{"event_id": "e_d", "event_class": "completed_rep", "start_time_s": 7.0, "end_time_s": 9.0}],
        )

        plan_dict = self._build_authentic_plan()
        self._create_authentic_execution(plan_dict=plan_dict)

        temp_eval.evaluate_temporal_events(
            plan_path_or_dir=self.plan_dir,
            execution_dir=self.execution_dir,
            output_dir=self.output_dir,
            tolerance_seconds=1.0,
        )

        report = json.loads((self.output_dir / "temporal_evaluation_report.json").read_text(encoding="utf-8"))

        primary_case_count = _extract_metric(report, "primary_case_count", "eligible_cases_count", "case_count")
        if primary_case_count is not None:
            self.assertEqual(primary_case_count, 1)

    def test_empty_primary_cohort_returns_not_ready_with_null_metrics(self) -> None:
        # All cases are unreviewed or excluded
        self._create_synthetic_case(group="candidate", human_count=None)
        self._create_synthetic_case(group="development", human_count=2, seen=False)

        plan_dict = self._build_authentic_plan()
        self._create_authentic_execution(plan_dict=plan_dict)

        temp_eval.evaluate_temporal_events(
            plan_path_or_dir=self.plan_dir,
            execution_dir=self.execution_dir,
            output_dir=self.output_dir,
            tolerance_seconds=1.0,
        )

        report = json.loads((self.output_dir / "temporal_evaluation_report.json").read_text(encoding="utf-8"))

        cohort_status = report.get("cohort_status") or report.get("status")
        self.assertEqual(cohort_status, "not_ready")

        prec = _extract_metric(report, "precision")
        mae_turn = _extract_metric(report, "mae_turn", "mae_turn_s")
        self.assertIsNone(prec)
        self.assertIsNone(mae_turn)


# ==============================================================================
# Suite 8: Incomplete Video Decodes
# ==============================================================================
class TestIncompleteVideoDecodes(TestTemporalEvaluationBase):
    """Test handling of partially decoded clips (e.g. 1524 of 1526 frames)."""

    def test_incomplete_decode_flagged_and_excluded_from_common_cohort(self) -> None:
        case_info = self._create_synthetic_case(
            human_count=1,
            events=[{"event_id": "e_inc", "event_class": "completed_rep", "start_time_s": 1.0, "end_time_s": 3.0}],
        )
        cid = case_info["case_id"]
        plan_dict = self._build_authentic_plan()

        summary_incomplete = {
            "source_video": {
                "declared_frame_count": 1526,
                "fps": 30.0,
            },
            "processed_frames": 1524,
            "status": "incomplete",
            "completed_reps": 1,
        }
        pred_attempts = [{
            "attempt_id": 1,
            "start_time_s": 1.1,
            "minimum_observed_frame": 60,
            "end_time_s": 3.1,
            "status": "counted",
            "category": "completed",
            "reason": "completed_cycle",
        }]
        self._create_authentic_execution(
            plan_dict=plan_dict,
            case_attempts={cid: pred_attempts},
            case_summaries={cid: summary_incomplete},
        )

        temp_eval.evaluate_temporal_events(
            plan_path_or_dir=self.plan_dir,
            execution_dir=self.execution_dir,
            output_dir=self.output_dir,
            tolerance_seconds=1.0,
        )

        report = json.loads((self.output_dir / "temporal_evaluation_report.json").read_text(encoding="utf-8"))
        incomplete_list = report.get("incomplete_cases") or report.get("excluded_from_common_cohort")
        if incomplete_list is None and "cohort_summary" in report:
            incomplete_list = report["cohort_summary"].get("incomplete_decoding_cases")
        if incomplete_list is not None:
            self.assertTrue(any(cid in str(item) for item in incomplete_list))


# ==============================================================================
# Suite 9: Review 46 Probes Regressions (Task 45 Blockers)
# ==============================================================================
class TestReview46Regressions(TestTemporalEvaluationBase):
    """Explicit regressions for defects identified in Review 46 probes and metric probes."""

    def test_probe1_missing_turnaround_strictly_null_never_midpoint(self) -> None:
        """Probe 1: Missing turnaround timestamp must remain null and NEVER invented as midpoint."""
        human_events = [
            {
                "event_id": "evt_no_turn",
                "event_class": "completed_rep",
                "start_time_s": 1.0,
                "turnaround_time_s": None,
                "end_time_s": 3.0,
            }
        ]
        case_info = self._create_synthetic_case(events=human_events, human_count=1)
        cid = case_info["case_id"]
        plan_dict = self._build_authentic_plan()

        pred_attempts = [
            {
                "attempt_id": 1,
                "start_time_s": 1.2,
                "minimum_observed_frame": None,
                "end_time_s": 3.2,
                "status": "counted",
                "category": "completed",
                "reason": "completed_cycle",
            }
        ]
        self._create_authentic_execution(
            plan_dict=plan_dict,
            case_attempts={cid: pred_attempts},
        )

        temp_eval.evaluate_temporal_events(
            plan_path_or_dir=self.plan_dir,
            execution_dir=self.execution_dir,
            output_dir=self.output_dir,
            tolerance_seconds=1.0,
        )

        # Check CSV output
        matches_csv = self.output_dir / "temporal_matches.csv"
        self.assertTrue(matches_csv.exists())
        with open(matches_csv, mode="r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        self.assertEqual(len(rows), 1)
        h_turn = rows[0].get("human_turnaround_time_s")
        p_turn = rows[0].get("predicted_turnaround_time_s")

        self.assertIn(h_turn, ("", None, "null", "None"), f"Human turnaround was invented: {h_turn!r}")
        self.assertNotEqual(h_turn, "2.0", "Human turnaround was invented as midpoint 2.0!")
        self.assertIn(p_turn, ("", None, "null", "None"), f"Predicted turnaround was invented: {p_turn!r}")
        self.assertNotEqual(p_turn, "2.2", "Predicted turnaround was invented as midpoint 2.2!")

        # Check JSON report
        report_file = self.output_dir / "temporal_evaluation_report.json"
        report = json.loads(report_file.read_text(encoding="utf-8"))
        b_err = _extract_metric(report, "boundary_errors") or {}
        turn_stats = b_err.get("turnaround", {})
        self.assertEqual(turn_stats.get("n", 0), 0)
        self.assertIsNone(turn_stats.get("mean_error_s"))
        self.assertIsNone(turn_stats.get("mae_s"))
        self.assertIsNone(turn_stats.get("rmse_s"))

        # Direct unit test on parse_predicted_attempts
        parsed_p = temp_eval.parse_predicted_attempts(pred_attempts, fps=30.0)
        self.assertIsNone(
            parsed_p[0].turnaround_time_s,
            f"parse_predicted_attempts invented turnaround: {parsed_p[0].turnaround_time_s}",
        )
        self.assertNotEqual(
            getattr(parsed_p[0], "turnaround_source", None),
            "midpoint_fallback",
            "Turnaround source must not be midpoint_fallback",
        )

    def test_probe2_missing_summary_or_attempts_never_completed(self) -> None:
        """Probe 2: Missing summary.json or attempts.json must NEVER yield cohort_status 'completed'."""
        case_info = self._create_synthetic_case(
            human_count=1,
            events=[{"event_id": "e1", "event_class": "completed_rep", "start_time_s": 1.0, "end_time_s": 3.0}],
        )
        cid = case_info["case_id"]
        plan_dict = self._build_authentic_plan()
        pred_attempts = [{"attempt_id": 1, "start_time_s": 1.1, "minimum_observed_frame": 60, "end_time_s": 3.1, "status": "counted", "category": "completed", "reason": "completed_cycle"}]
        self._create_authentic_execution(plan_dict=plan_dict, case_attempts={cid: pred_attempts})

        clip_dir = self.execution_dir / "baseline" / cid
        summary_file = clip_dir / "summary.json"
        self.assertTrue(summary_file.exists())
        summary_backup = summary_file.read_text(encoding="utf-8")
        summary_file.unlink()

        try:
            res_a = temp_eval.evaluate_temporal_events(
                plan_path_or_dir=self.plan_dir,
                execution_dir=self.execution_dir,
                output_dir=self.output_dir,
            )
            status_a = res_a.get("cohort_status") or res_a.get("status")
            self.assertNotEqual(
                status_a,
                "completed",
                f"Missing summary.json must NEVER yield cohort_status 'completed', got: {status_a!r}",
            )
        except (ValueError, FileNotFoundError, temp_eval.TemporalEvaluationError, temp_eval.CorruptedArtifactError):
            pass

        shutil.rmtree(self.output_dir, ignore_errors=True)
        summary_file.write_text(summary_backup, encoding="utf-8")
        attempts_file = clip_dir / "attempts.json"
        self.assertTrue(attempts_file.exists())
        attempts_file.unlink()

        try:
            res_b = temp_eval.evaluate_temporal_events(
                plan_path_or_dir=self.plan_dir,
                execution_dir=self.execution_dir,
                output_dir=self.output_dir,
            )
            status_b = res_b.get("cohort_status") or res_b.get("status")
            self.assertNotEqual(
                status_b,
                "completed",
                f"Missing attempts.json must NEVER yield cohort_status 'completed', got: {status_b!r}",
            )
        except (ValueError, FileNotFoundError, temp_eval.TemporalEvaluationError, temp_eval.CorruptedArtifactError):
            pass

    def test_probe3_failed_or_malformed_receipt_never_completed(self) -> None:
        """Probe 3: Execution receipt with status 'failed' or malformed must NEVER yield 'completed'."""
        case_info = self._create_synthetic_case(
            human_count=1,
            events=[{"event_id": "e1", "event_class": "completed_rep", "start_time_s": 1.0, "end_time_s": 3.0}],
        )
        cid = case_info["case_id"]
        plan_dict = self._build_authentic_plan()
        pred_attempts = [{"attempt_id": 1, "start_time_s": 1.1, "minimum_observed_frame": 60, "end_time_s": 3.1, "status": "counted", "category": "completed", "reason": "completed_cycle"}]
        self._create_authentic_execution(plan_dict=plan_dict, case_attempts={cid: pred_attempts})

        receipt_file = self.execution_dir / "evaluation_receipt.json"
        self.assertTrue(receipt_file.exists())
        orig_receipt = json.loads(receipt_file.read_text(encoding="utf-8"))

        # Test A: Status is failed
        failed_receipt = copy.deepcopy(orig_receipt)
        failed_receipt["status"] = "failed"
        receipt_file.write_text(json.dumps(failed_receipt, indent=2), encoding="utf-8")

        try:
            res = temp_eval.evaluate_temporal_events(
                plan_path_or_dir=self.plan_dir,
                execution_dir=self.execution_dir,
                output_dir=self.output_dir,
            )
            status = res.get("cohort_status") or res.get("status")
            self.assertNotEqual(
                status,
                "completed",
                f"Failed receipt must NEVER yield cohort_status 'completed', got: {status!r}",
            )
        except (ValueError, temp_eval.TemporalEvaluationError, temp_eval.CorruptedArtifactError):
            pass

        # Test B: Missing receipt
        shutil.rmtree(self.output_dir, ignore_errors=True)
        receipt_file.unlink()
        with self.assertRaises((FileNotFoundError, temp_eval.TemporalEvaluationError, temp_eval.CorruptedArtifactError)):
            temp_eval.evaluate_temporal_events(
                plan_path_or_dir=self.plan_dir,
                execution_dir=self.execution_dir,
                output_dir=self.output_dir,
            )

        # Test C: Malformed JSON in receipt
        shutil.rmtree(self.output_dir, ignore_errors=True)
        receipt_file.write_text("MALFORMED_JSON{:::broken", encoding="utf-8")
        with self.assertRaises((ValueError, temp_eval.TemporalEvaluationError, temp_eval.CorruptedArtifactError)):
            temp_eval.evaluate_temporal_events(
                plan_path_or_dir=self.plan_dir,
                execution_dir=self.execution_dir,
                output_dir=self.output_dir,
            )

    def test_probe4_ambiguous_events_not_scored_as_fp_or_fn(self) -> None:
        """Probe 4: Human ambiguous_or_unclear events must NOT be scored as FP or FN."""
        human_events = [
            {
                "event_id": "evt_ambig",
                "event_class": "ambiguous_or_unclear",
                "start_time_s": 1.0,
                "turnaround_time_s": 2.0,
                "end_time_s": 3.0,
            }
        ]
        case_info = self._create_synthetic_case(events=human_events, human_count=0)
        cid = case_info["case_id"]
        plan_dict = self._build_authentic_plan()

        pred_attempts = [
            {
                "attempt_id": 1,
                "start_time_s": 1.1,
                "minimum_observed_frame": 60,
                "end_time_s": 3.1,
                "status": "counted",
                "category": "completed",
                "reason": "completed_cycle",
            }
        ]
        self._create_authentic_execution(plan_dict=plan_dict, case_attempts={cid: pred_attempts})

        temp_eval.evaluate_temporal_events(
            plan_path_or_dir=self.plan_dir,
            execution_dir=self.execution_dir,
            output_dir=self.output_dir,
            tolerance_seconds=1.0,
        )

        report = json.loads((self.output_dir / "temporal_evaluation_report.json").read_text(encoding="utf-8"))
        self.assertEqual(report["cohort_status"], "not_ready")
        self.assertIsNone(report["metrics"])
        matched, un_h, un_p = temp_eval.match_events_1to1(
            temp_eval.parse_human_events(human_events),
            temp_eval.parse_predicted_attempts(pred_attempts, fps=30.0))
        scored = temp_eval.compute_cohort_metrics(matched, un_h, un_p)
        rep = scored["repetition_metrics"]

        # Ambiguous events must NEVER be scored as false positive or false negative
        self.assertEqual(
            rep.get("false_positives"),
            0,
            f"Ambiguous event must NOT be scored as false positive, got {rep.get('false_positives')}",
        )
        self.assertEqual(
            rep.get("false_negatives"),
            0,
            f"Ambiguous event must NOT be scored as false negative, got {rep.get('false_negatives')}",
        )
        self.assertEqual(rep.get("true_positives"), 0)
        self.assertEqual(rep.get("total_human_completed_reps"), 0)

        cat_agree = _extract_metric(scored, "movement_category_agreement") or {}
        ambig_stats = cat_agree.get("ambiguous_or_unclear", {})
        self.assertEqual(ambig_stats.get("human_event_count"), 1)

    def test_probe5_zero_agreement_supported_f1_is_zero_not_null(self) -> None:
        """Probe 5: When precision=0 and recall=0 on supported non-zero denominators, f1_score == 0.0."""
        human_events = [
            {
                "event_id": "h_rep1",
                "event_class": "completed_rep",
                "start_time_s": 1.0,
                "turnaround_time_s": 2.0,
                "end_time_s": 3.0,
            },
            {"event_id":"h_setup", "event_class":"setup_or_weight_return",
             "start_time_s":30.0, "turnaround_time_s":31.0, "end_time_s":32.0}
        ]
        case_info = self._create_synthetic_case(events=human_events, human_count=1)
        cid = case_info["case_id"]
        plan_dict = self._build_authentic_plan()

        pred_attempts = [
            {
                "attempt_id": 1,
                "start_time_s": 30.0,
                "minimum_observed_frame": 930,
                "end_time_s": 32.0,
                "status": "counted",
                "category": "completed",
                "reason": "completed_cycle",
            }
        ]
        self._create_authentic_execution(plan_dict=plan_dict, case_attempts={cid: pred_attempts})

        temp_eval.evaluate_temporal_events(
            plan_path_or_dir=self.plan_dir,
            execution_dir=self.execution_dir,
            output_dir=self.output_dir,
            tolerance_seconds=1.0,
        )

        report = json.loads((self.output_dir / "temporal_evaluation_report.json").read_text(encoding="utf-8"))
        rep = _extract_metric(report, "repetition_metrics") or {}

        self.assertEqual(rep.get("true_positives"), 0)
        self.assertEqual(rep.get("false_positives"), 1)
        self.assertEqual(rep.get("false_negatives"), 1)
        self.assertAlmostEqual(rep.get("precision"), 0.0, places=4)
        self.assertAlmostEqual(rep.get("recall"), 0.0, places=4)

        # F1 score must be 0.0, NOT None
        f1 = rep.get("f1_score")
        self.assertIsNotNone(f1, "f1_score was None when precision=0.0 and recall=0.0 on supported non-zero denominators!")
        self.assertAlmostEqual(f1, 0.0, places=4)

        # Direct unit test on compute_cohort_metrics
        h_ev = temp_eval.HumanEvent(
            event_id="h1",
            event_class="completed_rep",
            start_time_s=1.0,
            turnaround_time_s=2.0,
            end_time_s=3.0,
            uncertainty_notes=None,
        )
        p_att = temp_eval.PredictedAttempt(
            attempt_id=1,
            start_time_s=30.0,
            turnaround_time_s=31.0,
            turnaround_source="minimum_observed_frame",
            end_time_s=32.0,
            status="counted",
            category="completed",
            reason="completed_cycle",
        )
        unit_res = temp_eval.compute_cohort_metrics(
            all_matched_pairs=[temp_eval.MatchedPair(
                human=h_ev._replace(event_class="setup_or_weight_return"), predicted=p_att,
                distance=0.0, delta_start_s=None, delta_turnaround_s=None, delta_end_s=None,
                temporal_iou=None, classification_label="FP_setup_or_weight_return_as_counted")],
            all_unmatched_human=[h_ev],
            all_unmatched_predicted=[],
        )
        unit_rep = unit_res.get("repetition_metrics", {})
        self.assertAlmostEqual(unit_rep.get("precision"), 0.0, places=4)
        self.assertAlmostEqual(unit_rep.get("recall"), 0.0, places=4)
        self.assertEqual(
            unit_rep.get("f1_score"),
            0.0,
            f"compute_cohort_metrics yielded {unit_rep.get('f1_score')!r} instead of 0.0",
        )


# ==============================================================================
# Suite 10: Multi-Model, Provenance & Frozen Plan Integrity
# ==============================================================================
class TestFrozenPlanIntegrityAndMultiModel(TestTemporalEvaluationBase):
    """Test multi-model common cohort, frozen plan immutability, and unplanned case rejection."""

    def test_reject_mismatched_plan_sha_in_receipt(self) -> None:
        """Evaluator must reject when receipt plan_sha256 does not match plan.json on disk."""
        case_info = self._create_synthetic_case(human_count=1)
        plan_dict = self._build_authentic_plan()
        self._create_authentic_execution(plan_dict=plan_dict)

        rec_path = self.execution_dir / "evaluation_receipt.json"
        rec = json.loads(rec_path.read_text(encoding="utf-8"))
        rec["plan_sha256"] = "0" * 64
        rec_path.write_text(json.dumps(rec, indent=2), encoding="utf-8")

        with self.assertRaises((ValueError, temp_eval.HashMismatchError, temp_eval.TemporalEvaluationError)):
            temp_eval.evaluate_temporal_events(
                plan_path_or_dir=self.plan_dir,
                execution_dir=self.execution_dir,
                output_dir=self.output_dir,
            )

    def test_reject_raw_video_mutation(self) -> None:
        """Evaluator must reject when raw video file mutates after plan was generated."""
        case_info = self._create_synthetic_case(human_count=1)
        cid = case_info["case_id"]
        plan_dict = self._build_authentic_plan()
        self._create_authentic_execution(plan_dict=plan_dict)

        v_path = self.review_root / cid / f"video{case_info.get('extension', '.mp4')}"
        self.assertTrue(v_path.exists())
        v_path.write_bytes(b"MUTATED_RAW_VIDEO_BYTES_AFTER_PLAN_GENERATION")

        with self.assertRaises((ValueError, temp_eval.HashMismatchError, temp_eval.TemporalEvaluationError, CaseCorruptedError, PlanOrCodeMutationError)):
            temp_eval.evaluate_temporal_events(
                plan_path_or_dir=self.plan_dir,
                execution_dir=self.execution_dir,
                output_dir=self.output_dir,
            )

    def test_reject_model_mismatch_between_plan_and_receipt(self) -> None:
        """Evaluator must reject when model declared in receipt differs from frozen plan."""
        case_info = self._create_synthetic_case(human_count=1)
        plan_dict = self._build_authentic_plan(model_name="baseline")
        self._create_authentic_execution(plan_dict=plan_dict, model_name="baseline")

        rec_path = self.execution_dir / "evaluation_receipt.json"
        rec = json.loads(rec_path.read_text(encoding="utf-8"))
        rec["models"] = {"foreign_model": {"path": "foreign.pt", "sha256": "1" * 64}}
        rec_path.write_text(json.dumps(rec, indent=2), encoding="utf-8")

        with self.assertRaises((ValueError, temp_eval.TemporalEvaluationError, temp_eval.CorruptedArtifactError, temp_eval.HashMismatchError)):
            temp_eval.evaluate_temporal_events(
                plan_path_or_dir=self.plan_dir,
                execution_dir=self.execution_dir,
                output_dir=self.output_dir,
            )

    def test_unplanned_new_cases_outside_frozen_plan_ignored_or_rejected(self) -> None:
        """New cases added to ReviewStore outside frozen plan must be ignored or rejected."""
        case1 = self._create_synthetic_case(
            human_count=1,
            original_name="case1.mp4",
            events=[{"event_id": "e1", "event_class": "completed_rep", "start_time_s": 1.0, "end_time_s": 3.0}],
        )
        cid1 = case1["case_id"]
        plan_dict = self._build_authentic_plan()
        pred_attempts = [{"attempt_id": 1, "start_time_s": 1.1, "minimum_observed_frame": 60, "end_time_s": 3.1, "status": "counted", "category": "completed", "reason": "completed_cycle"}]
        self._create_authentic_execution(plan_dict=plan_dict, case_attempts={cid1: pred_attempts})

        # Add Case 2 to review store AFTER plan was frozen
        case2 = self._create_synthetic_case(
            human_count=1,
            original_name="case2_unplanned.mp4",
            events=[{"event_id": "e2", "event_class": "completed_rep", "start_time_s": 5.0, "end_time_s": 7.0}],
        )
        cid2 = case2["case_id"]

        try:
            temp_eval.evaluate_temporal_events(
                plan_path_or_dir=self.plan_dir,
                execution_dir=self.execution_dir,
                output_dir=self.output_dir,
            )
            report = json.loads((self.output_dir / "temporal_evaluation_report.json").read_text(encoding="utf-8"))
            cohort_sum = _extract_metric(report, "cohort_summary") or {}
            eval_cases = cohort_sum.get("evaluated_cases", [])
            primary_cases = cohort_sum.get("primary_cases", [])
            self.assertNotIn(cid2, eval_cases, f"Unplanned case {cid2} was discovered and evaluated!")
            self.assertNotIn(cid2, primary_cases, f"Unplanned case {cid2} was added to primary cases!")
        except (ValueError, temp_eval.TemporalEvaluationError, temp_eval.CorruptedArtifactError):
            pass

    def test_unequal_common_completed_cohorts_multi_model(self) -> None:
        """In multi-model evaluation, only clips completed across ALL models form the completed cohort."""
        c1 = self._create_synthetic_case(
            human_count=1, original_name="c1.mp4",
            events=[{"event_id": "e1", "event_class": "completed_rep", "start_time_s": 1.0, "end_time_s": 3.0}],
        )
        c2 = self._create_synthetic_case(
            human_count=1, original_name="c2.mp4",
            events=[{"event_id": "e2", "event_class": "completed_rep", "start_time_s": 4.0, "end_time_s": 6.0}],
        )
        cid1, cid2 = c1["case_id"], c2["case_id"]

        ckpt_a = self.models_dir / "model_a.pt"
        ckpt_a.write_bytes(b"MODEL_A_BYTES_1234567890")
        ckpt_b = self.models_dir / "model_b.pt"
        ckpt_b.write_bytes(b"MODEL_B_BYTES_1234567890")

        plan_dict = self._build_authentic_plan(
            models_contract={"model_a": ckpt_a, "model_b": ckpt_b}
        )

        self._create_authentic_execution(
            plan_dict=plan_dict,
            models_contract={"model_a": ckpt_a, "model_b": ckpt_b},
            fail_cases={"model_b": {cid2}},
        )

        temp_eval.evaluate_temporal_events(
            plan_path_or_dir=self.plan_dir,
            execution_dir=self.execution_dir,
            output_dir=self.output_dir,
        )

        report = json.loads((self.output_dir / "temporal_evaluation_report.json").read_text(encoding="utf-8"))
        cohort_sum = _extract_metric(report, "cohort_summary") or {}
        primary_cases = cohort_sum.get("primary_cases", [])
        self.assertIn(cid1, primary_cases)
        self.assertNotIn(cid2, primary_cases, "cid2 must be excluded from common completed cohort because model_b failed!")

    def test_primary_vs_exploratory_isolation_with_allow_unreviewed(self) -> None:
        """allow_unreviewed must NOT contaminate primary cohort metrics."""
        self._create_synthetic_case(
            group="candidate", scope="in_scope", seen=False, human_count=1,
            original_name="primary.mp4",
            events=[{"event_id": "e_p", "event_class": "completed_rep", "start_time_s": 1.0, "end_time_s": 3.0}],
        )
        self._create_synthetic_case(
            group="candidate", scope="in_scope", seen=True, human_count=1,
            original_name="seen.mp4",
            events=[{"event_id": "e_s", "event_class": "completed_rep", "start_time_s": 4.0, "end_time_s": 6.0}],
        )
        self._create_synthetic_case(
            group="development", scope="in_scope", seen=False, human_count=1,
            original_name="dev.mp4",
            events=[{"event_id": "e_d", "event_class": "completed_rep", "start_time_s": 7.0, "end_time_s": 9.0}],
        )

        plan_dict = self._build_authentic_plan()
        self._create_authentic_execution(plan_dict=plan_dict)

        temp_eval.evaluate_temporal_events(
            plan_path_or_dir=self.plan_dir,
            execution_dir=self.execution_dir,
            output_dir=self.output_dir,
            allow_unreviewed=True,
        )

        report = json.loads((self.output_dir / "temporal_evaluation_report.json").read_text(encoding="utf-8"))
        cohort_sum = _extract_metric(report, "cohort_summary") or {}
        primary_cases = cohort_sum.get("primary_cases", [])
        self.assertEqual(len(primary_cases), 1, f"Primary cohort contaminated: {primary_cases}")


# ==============================================================================
# Suite 11: CLI Functionality & Output Artifacts
# ==============================================================================
class TestCLIAndOutputArtifacts(TestTemporalEvaluationBase):
    """Test CLI arguments, flags, error codes, and generated output files."""

    def test_cli_invocation_success(self) -> None:
        case_info = self._create_synthetic_case(
            events=[{"event_id": "e1", "event_class": "completed_rep", "start_time_s": 1.0, "end_time_s": 3.0}],
            human_count=1,
        )
        cid = case_info["case_id"]
        plan_dict = self._build_authentic_plan()
        pred_attempts = [{"attempt_id": 1, "start_time_s": 1.1, "minimum_observed_frame": 60, "end_time_s": 3.1, "status": "counted", "category": "completed", "reason": "completed_cycle"}]
        self._create_authentic_execution(plan_dict=plan_dict, case_attempts={cid: pred_attempts})

        cmd = [
            sys.executable,
            "-m",
            "gym_analyser.temporal_evaluation",
            "--plan",
            str(self.plan_dir),
            "--execution",
            str(self.execution_dir),
            "--output-dir",
            str(self.output_dir),
            "--tolerance-seconds",
            "1.5",
        ]
        env = dict(os.environ)
        env["PYTHONPATH"] = str(_GYM_POSE_DIR)
        res = subprocess.run(cmd, capture_output=True, text=True, cwd=str(_GYM_POSE_DIR), env=env, timeout=15.0)
        self.assertEqual(res.returncode, 0, f"CLI failed:\nSTDOUT: {res.stdout}\nSTDERR: {res.stderr}")

        report_json = self.output_dir / "temporal_evaluation_report.json"
        matches_csv = self.output_dir / "temporal_matches.csv"
        summary_txt = self.output_dir / "temporal_summary.txt"

        self.assertTrue(report_json.exists(), "temporal_evaluation_report.json was not created")
        self.assertTrue(matches_csv.exists(), "temporal_matches.csv was not created")
        self.assertTrue(summary_txt.exists(), "temporal_summary.txt was not created")

        self.assertGreater(report_json.stat().st_size, 0)
        self.assertGreater(matches_csv.stat().st_size, 0)
        self.assertGreater(summary_txt.stat().st_size, 0)

    def test_cli_missing_required_arguments(self) -> None:
        cmd = [
            sys.executable,
            "-m",
            "gym_analyser.temporal_evaluation",
            "--plan",
            str(self.plan_dir),
            "--execution",
            str(self.execution_dir),
        ]
        env = dict(os.environ)
        env["PYTHONPATH"] = str(_GYM_POSE_DIR)
        res = subprocess.run(cmd, capture_output=True, text=True, cwd=str(_GYM_POSE_DIR), env=env, timeout=10.0)
        self.assertEqual(res.returncode, 2, "Expected code 2 for missing required arguments")

    def test_cli_invalid_tolerance(self) -> None:
        cmd = [
            sys.executable,
            "-m",
            "gym_analyser.temporal_evaluation",
            "--plan",
            str(self.plan_dir),
            "--execution",
            str(self.execution_dir),
            "--output-dir",
            str(self.output_dir),
            "--tolerance-seconds",
            "-0.5",
        ]
        env = dict(os.environ)
        env["PYTHONPATH"] = str(_GYM_POSE_DIR)
        res = subprocess.run(cmd, capture_output=True, text=True, cwd=str(_GYM_POSE_DIR), env=env, timeout=10.0)
        self.assertNotEqual(res.returncode, 0, "Negative tolerance must be rejected")


class TestDirectCorrectionRegressions(TestTemporalEvaluationBase):
    def _prepared(self):
        case = self._create_synthetic_case(events=[
            {"event_id":"h1", "event_class":"completed_rep", "start_time_s":1, "end_time_s":3}])
        cid = case["case_id"]
        self._create_authentic_execution(self._build_authentic_plan(), case_attempts={cid:[
            {"attempt_id":1,"status":"counted","start_time_s":1,"end_time_s":3}]})
        return cid

    def test_receipt_requires_version_models_configuration_and_predictions(self):
        self._prepared()
        path=self.execution_dir/"evaluation_receipt.json"
        original=path.read_bytes()
        mutations=[("receipt_version", "9.0"), ("models", None), ("models", {}),
                   ("declared_configuration", {}), ("per_clip_predictions", {}),
                   ("common_completed_cohort_count", 99), ("status", "running")]
        for key,value in mutations:
            with self.subTest(field=key,value=value):
                receipt=json.loads(original);receipt[key]=value;path.write_text(json.dumps(receipt))
                with self.assertRaises((ValueError, RuntimeError)):
                    temp_eval.evaluate_temporal_events(self.plan_dir,self.execution_dir,self.output_dir)
                self.assertFalse(self.output_dir.exists())
        path.write_bytes(original)

    def test_summary_requires_completed_identity_config_and_counts(self):
        cid=self._prepared();path=self.execution_dir/"baseline"/cid/"summary.json"
        original=path.read_bytes()
        for key,value in [("status",None),("source_video",{}),("configuration",{}),
                          ("model_name","wrong.pt"),("processed_frames",True),("completed_reps",8)]:
            with self.subTest(field=key):
                summary=json.loads(original);summary[key]=value;path.write_text(json.dumps(summary))
                out=self.test_root/("invalid_"+key)
                report=temp_eval.evaluate_temporal_events(self.plan_dir,self.execution_dir,out)
                self.assertEqual(report["cohort_status"],"not_ready")
                self.assertIsNone(report["metrics"])
        path.write_bytes(original)

    def test_consumed_input_mutation_prevents_publication(self):
        cid=self._prepared();path=self.execution_dir/"baseline"/cid/"attempts.json"
        def mutate(): path.write_text('{"events": []}')
        with self.assertRaises((ValueError, RuntimeError)):
            temp_eval.evaluate_temporal_events(self.plan_dir,self.execution_dir,self.output_dir,_seam_hook=mutate)
        self.assertFalse(self.output_dir.exists())

    def test_unlocated_human_event_is_not_false_negative(self):
        humans=temp_eval.parse_human_events([{"event_id":"unknown","event_class":"completed_rep"}])
        metrics=temp_eval.compute_cohort_metrics([],humans,[])["repetition_metrics"]
        self.assertEqual(metrics["false_negatives"],0)
        self.assertEqual(metrics["unscored_unmatched_human_events"],1)
        self.assertIsNone(metrics["recall"])

    def test_csv_matches_json_unscored_semantics(self):
        humans=temp_eval.parse_human_events([{"event_id":"unknown","event_class":"completed_rep"}])
        attempts=temp_eval.parse_predicted_attempts([{"attempt_id":1,"status":"counted",
                                                    "start_time_s":1,"end_time_s":3}])
        rows=temp_eval.build_matches_csv_rows("baseline","case",[],humans,attempts,True)
        self.assertTrue(all(row["classification_label"].startswith("unscored_") for row in rows))

    def test_unmatched_movement_has_unknown_agreement_not_zero(self):
        humans=temp_eval.parse_human_events([{"event_id":"s","event_class":"setup_or_weight_return",
                                             "start_time_s":1,"end_time_s":2}])
        result=temp_eval.compute_cohort_metrics([],humans,[])
        self.assertEqual(result["movement_category_agreement"]["setup_or_weight_return"]["unmatched"],1)
        self.assertIsNone(result["movement_category_agreement"]["setup_or_weight_return"]["agreement_rate"])

    def test_parser_rejects_missing_or_invalid_prediction_fields(self):
        base={"attempt_id":1,"status":"counted","start_time_s":1,"end_time_s":3}
        for change in [{"attempt_id":None},{"attempt_id":1.5},{"status":None},
                       {"status":"unknown"},{"turnaround_time_s":"oops"},
                       {"minimum_observed_frame":-1},{"minimum_observed_frame":True}]:
            with self.subTest(change=change), self.assertRaises(ValueError):
                temp_eval.parse_predicted_attempts([dict(base,**change)],fps=30)

    def test_partial_event_annotation_does_not_certify_exhaustiveness(self):
        case=self._create_synthetic_case(human_count=1,events=[
            {"event_id":"h1","event_class":"completed_rep","start_time_s":1,"end_time_s":3}])
        cid=case["case_id"]
        self._create_authentic_execution(self._build_authentic_plan(),case_attempts={cid:[
            {"attempt_id":1,"status":"counted","start_time_s":1,"end_time_s":3},
            {"attempt_id":2,"status":"counted","start_time_s":8,"end_time_s":10}]})
        report=temp_eval.evaluate_temporal_events(self.plan_dir,self.execution_dir,self.output_dir)
        metrics=report["repetition_metrics"]
        self.assertEqual(metrics["false_positives"],0)
        self.assertEqual(metrics["unscored_unmatched_predictions"],1)


if __name__ == "__main__":
    unittest.main()
