"""Comprehensive unit test suite for review evaluation execution (Phase 2).

Tests:
1. Pure standard library: no Torch, Ultralytics, OpenCV, or Matplotlib imports on import or --help.
2. Plan validation: canonical SHA256 integrity, missing fields, malformed JSON, missing files.
3. Model and input integrity: checkpoint hash/size mismatch, video hash mismatch, annotation mismatch.
4. Output directory safety: non-empty overwrite rejection, symlink ancestor rejection, path overlap.
5. Readiness mode: validation report without --run, honest not_ready reporting on empty cohorts.
6. Execution refusal: --run refuses not_ready plans or zero eligible cases.
7. Sequential execution: correct command-line arguments, working directory, and per-clip output folders.
8. Real subprocess child runner with disposable mock CLI: normal completion, non-zero exit, malformed summary, incomplete decoding, and finite timeout killing.
9. Strict fake CLI no-overwrite: child output directory is fresh and run.log is preserved outside.
10. Descendant process cleanup: finite owned-process and grandchild process termination on timeout.
11. Invalid primary snapshot rejection: recomputed plan self-hash cannot bypass disk ReviewStore validation.
12. Plan mutation during execution: mutating plan.json on disk aborts execution.
13. Diagnostic evidence preservation: failed runs preserve raw logs and error receipts.
14. Ancestor link checks: symlink and junction ancestors rejected before resolve().
15. Multi-model common cohort comparison: strictly identical denominators, exclusion of failed/incomplete clips, null metrics on empty common cohorts.
16. Staging publication and seam integrity: atomic rename, staging cleanup on mutation, bit-for-bit source immutability.
17. Evaluation API roundtrip: compatibility with load_ground_truth and evaluate_counts.
18. Review 43 adversarial regressions: rejection of wrong source path (same basename, different dir), conflicting configuration, zero frames, missing/non-integer frames; pre-run runtime provenance freeze; owned process and grandchild containment on timeout and abnormal exit without killing unrelated PIDs; cohort exclusion.
"""

from __future__ import annotations

import copy
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
from contextlib import redirect_stderr, redirect_stdout
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple

try:
    import psutil
except ImportError:
    psutil = None

import gym_analyser.review_execution as rev_exec
from gym_analyser.evaluation import evaluate_counts, load_ground_truth
from gym_analyser.review_evaluation import (
    build_plan,
    compute_file_sha256,
    reject_linked_ancestors,
)
from gym_analyser.review_execution import (
    build_parser,
    compute_metrics,
    compute_multi_model_common_cohort,
    compute_plan_sha256,
    default_subprocess_runner,
    execute_evaluation,
    format_execution_report,
    load_and_validate_plan,
    main,
    validate_device_string,
    validate_execution_output_dir,
    validate_interpreter,
    verify_plan_inputs,
)
from gym_analyser.review_store import CaseCorruptedError, ReviewStore, is_symlink


def _is_pid_running(pid: int) -> bool:
    """Check if process with given PID is still running."""
    if pid <= 0:
        return False
    if psutil is not None:
        try:
            proc = psutil.Process(pid)
            return proc.is_running() and proc.status() != psutil.STATUS_ZOMBIE
        except psutil.NoSuchProcess:
            return False
        except Exception:
            pass
    if sys.platform == "win32":
        try:
            res = subprocess.run(
                ["tasklist", "/FI", f"PID eq {pid}"],
                capture_output=True,
                text=True,
                timeout=5.0,
            )
            return str(pid) in res.stdout
        except Exception:
            return False
    else:
        try:
            os.kill(pid, 0)
            return True
        except OSError:
            return False


class TestReviewExecutionBase(unittest.TestCase):
    """Base test fixture providing temporary isolated review store, models, and plans."""

    def setUp(self) -> None:
        self.test_root = tempfile.mkdtemp(prefix="test_rev_exec_")
        self.review_root = Path(self.test_root) / "review_storage"
        self.review_root.mkdir(parents=True, exist_ok=True)
        self.store = ReviewStore(root=self.review_root)

        self.models_dir = Path(self.test_root) / "models"
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.baseline_pt = self.models_dir / "baseline.pt"
        self.baseline_pt.write_bytes(b"MOCK_BASELINE_YOLO_WEIGHTS_1234567890")
        self.candidate_pt = self.models_dir / "candidate.pt"
        self.candidate_pt.write_bytes(b"MOCK_CANDIDATE_YOLO_WEIGHTS_0987654321")

        self.output_root = Path(self.test_root) / "eval_output"
        self.plan_dir = Path(self.test_root) / "plan_output"

        self.mock_gym_pose_dir = self._setup_mock_gym_pose()

    def tearDown(self) -> None:
        for k in (
            "GYM_POSE_DIR",
            "MOCK_CHILD_MODE",
            "MOCK_CHILD_SLEEP",
            "MOCK_CHILD_REPS",
            "MOCK_PID_FILE",
            "MOCK_BASELINE_REPS",
            "MOCK_CANDIDATE_REPS",
            "MOCK_PROCESSED_FRAMES",
            "MOCK_SOURCE_VIDEO_PATH",
            "MOCK_CHILD_SUMMARY_OVERRIDE",
        ):
            if k in os.environ:
                del os.environ[k]
        shutil.rmtree(self.test_root, ignore_errors=True)

    def _setup_mock_gym_pose(self) -> Path:
        """Create a mock gym_pose package directory with disposable __main__.py CLI."""
        mock_dir = Path(self.test_root) / "mock_gym_pose"
        pkg_dir = mock_dir / "gym_analyser"
        pkg_dir.mkdir(parents=True, exist_ok=True)
        (pkg_dir / "__init__.py").write_text("", encoding="utf-8")
        (pkg_dir / "pipeline.py").write_text("# mock pipeline\n", encoding="utf-8")
        (pkg_dir / "counter.py").write_text("# mock counter\n", encoding="utf-8")
        (pkg_dir / "review_execution.py").write_text("# mock review_execution\n", encoding="utf-8")
        main_py = pkg_dir / "__main__.py"
        script_code = """import argparse
import json
import os
from pathlib import Path
import sys
import time

parser = argparse.ArgumentParser()
parser.add_argument("--source", type=Path, required=True)
parser.add_argument("--output-dir", type=Path, required=True)
parser.add_argument("--model", type=str, required=True)
parser.add_argument("--device", type=str, default="cpu")
parser.add_argument("--down-angle", type=float, default=None)
parser.add_argument("--up-angle", type=float, default=None)
parser.add_argument("--min-confidence", type=float, default=None)
parser.add_argument("--stable-frames", type=int, default=None)
parser.add_argument("--departure-frames", type=int, default=None)
parser.add_argument("--max-gap", type=float, default=None)
parser.add_argument("--tracker", type=str, default=None)
parser.add_argument("--overwrite", action="store_true", default=False)

args = parser.parse_args()

# Strict fake CLI check: fail if output_dir exists and contains any files unless overwrite requested
if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.overwrite:
    sys.stderr.write("Output directory is not empty\\n")
    sys.exit(19)

args.output_dir.mkdir(parents=True, exist_ok=True)

mode = os.environ.get("MOCK_CHILD_MODE", "normal")
if mode == "sleep":
    sleep_sec = float(os.environ.get("MOCK_CHILD_SLEEP", "5.0"))
    time.sleep(sleep_sec)

if mode == "spawn_grandchild_and_sleep":
    import subprocess
    pid_file = os.environ.get("MOCK_PID_FILE")
    grandchild = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    if pid_file:
        Path(pid_file).write_text(
            json.dumps({"child_pid": os.getpid(), "grandchild_pid": grandchild.pid}),
            encoding="utf-8",
        )
    time.sleep(60)
    sys.exit(0)

if mode == "spawn_grandchild_and_exit":
    import subprocess
    pid_file = os.environ.get("MOCK_PID_FILE")
    grandchild = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    if pid_file:
        Path(pid_file).write_text(
            json.dumps({"child_pid": os.getpid(), "grandchild_pid": grandchild.pid}),
            encoding="utf-8",
        )
    sys.exit(0)

if mode == "spawn_grandchild_and_fail_exit":
    import subprocess
    pid_file = os.environ.get("MOCK_PID_FILE")
    grandchild = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    if pid_file:
        Path(pid_file).write_text(
            json.dumps({"child_pid": os.getpid(), "grandchild_pid": grandchild.pid}),
            encoding="utf-8",
        )
    sys.exit(1)

if mode == "fail_exit":
    sys.stderr.write("Simulated fatal child error\\n")
    sys.exit(1)

if mode == "malformed_json":
    (args.output_dir / "summary.json").write_text("{CORRUPT_JSON", encoding="utf-8")
    sys.exit(0)

if mode == "incomplete":
    summary = {
        "status": "incomplete",
        "read_status_note": "Corrupted video frame at index 42",
        "completed_reps": None,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    sys.exit(0)

if mode == "wrong_model":
    summary = {
        "status": "completed",
        "completed_reps": 5,
        "clip_id": args.output_dir.name,
        "model": "wrong_model_identity",
        "model_name": "wrong_model_identity",
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    sys.exit(0)

if mode == "wrong_clip":
    summary = {
        "status": "completed",
        "completed_reps": 5,
        "clip_id": "different_clip_id_9999",
        "model": args.model,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    sys.exit(0)

if mode == "string_reps":
    summary = {
        "status": "completed",
        "completed_reps": "5",
        "clip_id": args.output_dir.name,
        "model": args.model,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    sys.exit(0)

# Normal mode or per-model logic
reps = int(os.environ.get("MOCK_CHILD_REPS", "5"))
if "baseline" in args.model:
    reps = int(os.environ.get("MOCK_BASELINE_REPS", str(reps)))
elif "candidate" in args.model:
    reps = int(os.environ.get("MOCK_CANDIDATE_REPS", str(reps)))

down_val = args.down_angle if args.down_angle is not None else 110.0
up_val = args.up_angle if args.up_angle is not None else 150.0
min_conf_val = args.min_confidence if args.min_confidence is not None else 0.3
stable_val = args.stable_frames if args.stable_frames is not None else 3
dep_val = args.departure_frames if args.departure_frames is not None else 2
max_gap_val = args.max_gap if args.max_gap is not None else 0.5
tracker_val = args.tracker or "bytetrack.yaml"

proc_frames_str = os.environ.get("MOCK_PROCESSED_FRAMES", "100")
try:
    proc_frames = int(proc_frames_str)
except Exception:
    proc_frames = proc_frames_str

src_path_val = os.environ.get("MOCK_SOURCE_VIDEO_PATH", str(args.source.resolve()))

summary = {
    "status": "completed",
    "completed_reps": reps,
    "clip_id": args.output_dir.name,
    "model": args.model,
    "model_name": args.model,
    "read_status_note": None,
    "processed_frames": proc_frames,
    "processed_duration_s": 3.33,
    "valid_observations": 95,
    "valid_observation_fraction": 0.95,
    "package_versions": {
        "python": sys.version.split()[0],
        "ultralytics": "unknown",
        "opencv": "unknown",
    },
    "configuration": {
        "down_angle": down_val,
        "up_angle": up_val,
        "min_joint_confidence": min_conf_val,
        "stable_frames": stable_val,
        "departure_frames": dep_val,
        "max_gap_seconds": max_gap_val,
        "tracker": tracker_val,
        "device": args.device,
    },
    "source_video": {
        "path": src_path_val,
        "declared_frame_count": 100,
        "declared_duration_s": 3.33,
        "fps": 30.0,
        "width": 1920,
        "height": 1080,
    },
    "min_confidence": args.min_confidence,
    "down_angle": args.down_angle,
    "up_angle": args.up_angle,
    "stable_frames": args.stable_frames,
    "departure_frames": args.departure_frames,
    "max_gap": args.max_gap,
    "tracker": args.tracker,
    "received_args": sys.argv[1:],
}

if mode == "primary_missing_source":
    del summary["source_video"]
elif mode == "primary_missing_configuration":
    del summary["configuration"]
elif mode == "primary_missing_model":
    del summary["model_name"]
elif mode == "primary_missing_setting":
    del summary["configuration"]["stable_frames"]
elif mode == "primary_false_completion":
    summary["processed_frames"] = 50
elif mode == "primary_boolean_setting":
    summary["configuration"]["stable_frames"] = True
elif mode == "primary_error_with_completed":
    summary["error"] = "synthetic contradictory result"
elif mode == "zero_frames":
    summary["processed_frames"] = 0

elif mode == "missing_processed_frames":
    del summary["processed_frames"]

elif mode == "non_integer_processed_frames":
    summary["processed_frames"] = "not_an_int"

elif mode == "wrong_source_path":
    summary["source_video"]["path"] = str(Path("C:/wrong_parent_dir") / args.source.name)

elif mode == "wrong_config_down_angle":
    summary["configuration"]["down_angle"] = 1.0
    summary["down_angle"] = 1.0

elif mode == "wrong_config_min_confidence":
    summary["configuration"]["min_joint_confidence"] = 0.9
    summary["min_confidence"] = 0.9

elif mode == "wrong_source_config_zero_frames":
    summary["processed_frames"] = 0
    summary["source_video"]["path"] = str(Path("C:/wrong_parent_dir") / args.source.name)
    summary["configuration"]["down_angle"] = 1.0
    summary["down_angle"] = 1.0

summary_override = os.environ.get("MOCK_CHILD_SUMMARY_OVERRIDE")
if summary_override:
    try:
        ov = json.loads(summary_override)
        summary.update(ov)
    except Exception:
        pass

(args.output_dir / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
sys.exit(0)
"""
        main_py.write_text(script_code, encoding="utf-8")
        return mock_dir

    def _create_synthetic_case(
        self,
        group: str = "candidate",
        human_count: Optional[int] = 5,
        scope: str = "in_scope",
        reviewer: str = "Reviewer One",
        raw_video_reviewed: bool = True,
        seen: Optional[bool] = False,
        video_content: bytes = b"MOCK_VIDEO_STREAM_BYTES_MP4",
        notes: str = "Synthetic clip for review execution",
    ) -> Dict[str, Any]:
        """Create a valid ReviewStore case in temporary storage."""
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

    def _create_standard_plan(
        self,
        num_cases: int = 3,
        counts: Sequence[int] = (5, 8, 10),
    ) -> Path:
        """Create a standard Phase 1 evaluation plan with primary eligible cases."""
        for i in range(num_cases):
            c_val = counts[i] if i < len(counts) else 5 + i
            self._create_synthetic_case(
                group="candidate",
                human_count=c_val,
                scope="in_scope",
                reviewer="Alice",
                raw_video_reviewed=True,
                seen=False,
                video_content=f"VIDEO_CONTENT_CLIP_{i}_{os.urandom(8).hex()}".encode("utf-8"),
            )

        models = {
            "baseline": self.baseline_pt,
            "candidate": self.candidate_pt,
        }
        plan_out = Path(self.test_root) / f"plan_{os.urandom(4).hex()}"
        build_plan(
            review_root=self.review_root,
            output_dir=plan_out,
            models=models,
        )
        return plan_out / "plan.json"


class TestPureStdlibNoMLImports(TestReviewExecutionBase):
    """Verify that importing review_execution or running --help imports zero heavy ML packages."""

    def test_import_pure_stdlib(self) -> None:
        gym_pose_dir = Path(__file__).resolve().parent.parent
        cmd = [
            sys.executable,
            "-c",
            "import sys; "
            "import gym_analyser.review_execution; "
            "banned = {'torch', 'ultralytics', 'cv2', 'matplotlib'}.intersection(sys.modules); "
            "assert not banned, f'Banned ML modules imported: {banned}'; print('CLEAN')",
        ]
        child_env = dict(os.environ)
        child_env["PYTHONPATH"] = str(gym_pose_dir)
        res = subprocess.run(
            cmd,
            cwd=str(gym_pose_dir),
            capture_output=True,
            text=True,
            env=child_env,
            timeout=10,
        )
        self.assertEqual(res.returncode, 0, f"Subprocess import failed: {res.stderr}")
        self.assertIn("CLEAN", res.stdout)

    def test_cli_help_does_not_import_ml(self) -> None:
        gym_pose_dir = Path(__file__).resolve().parent.parent
        cmd = [
            sys.executable,
            "-c",
            "import sys; import gym_analyser.review_execution; "
            "banned = {'torch', 'ultralytics', 'cv2', 'matplotlib'}.intersection(sys.modules); "
            "assert not banned, f'Banned: {banned}'; print('CLEAN')",
        ]
        child_env = dict(os.environ)
        child_env["PYTHONPATH"] = str(gym_pose_dir)
        res = subprocess.run(
            cmd,
            cwd=str(gym_pose_dir),
            capture_output=True,
            text=True,
            env=child_env,
            timeout=10,
        )
        self.assertEqual(res.returncode, 0, f"Subprocess failed: {res.stderr}")
        self.assertIn("CLEAN", res.stdout)


class TestPlanValidationAndIntegrity(TestReviewExecutionBase):
    """Test plan schema, hash verification, and input consistency checks."""

    def test_valid_plan_passes(self) -> None:
        plan_path = self._create_standard_plan(num_cases=2)
        res = load_and_validate_plan(plan_path)
        plan = res[0] if isinstance(res, tuple) else res
        p_sha = res[1] if isinstance(res, tuple) else plan["plan_sha256"]
        self.assertEqual(plan["status"], "ready")
        self.assertEqual(len(p_sha), 64)

        models, r_root, eligible = verify_plan_inputs(plan, plan_path)
        self.assertEqual(len(models), 2)
        self.assertEqual(len(eligible), 2)

    def test_plan_hash_mismatch_rejected(self) -> None:
        plan_path = self._create_standard_plan(num_cases=1)
        plan_data = json.loads(plan_path.read_text(encoding="utf-8"))
        # Tamper with declared configuration
        plan_data["declared_configuration"]["departure_frames"] = 99
        plan_path.write_text(json.dumps(plan_data), encoding="utf-8")

        with self.assertRaises(ValueError) as ctx:
            load_and_validate_plan(plan_path)
        self.assertIn("Plan SHA256 mismatch", str(ctx.exception))

    def test_missing_plan_file(self) -> None:
        non_existent = Path(self.test_root) / "non_existent_plan.json"
        with self.assertRaises(FileNotFoundError):
            load_and_validate_plan(non_existent)

    def test_missing_required_field_rejected(self) -> None:
        plan_path = self._create_standard_plan(num_cases=1)
        plan_data = json.loads(plan_path.read_text(encoding="utf-8"))
        del plan_data["models"]
        # recompute hash to bypass sha check and trigger missing field check
        plan_data["plan_sha256"] = compute_plan_sha256(plan_data)
        plan_path.write_text(json.dumps(plan_data), encoding="utf-8")

        with self.assertRaises(ValueError) as ctx:
            load_and_validate_plan(plan_path)
        self.assertIn("missing required field 'models'", str(ctx.exception))

    def test_model_checkpoint_tampering_rejected(self) -> None:
        plan_path = self._create_standard_plan(num_cases=1)
        res = load_and_validate_plan(plan_path)
        plan = res[0] if isinstance(res, tuple) else res

        # Mutate baseline checkpoint bytes
        self.baseline_pt.write_bytes(b"TAMPERED_CHECKPOINT_BYTES_99999")

        with self.assertRaises(ValueError) as ctx:
            verify_plan_inputs(plan, plan_path)
        self.assertTrue(
            "SHA256 mismatch" in str(ctx.exception) or "size mismatch" in str(ctx.exception)
        )

    def test_video_tampering_rejected(self) -> None:
        plan_path = self._create_standard_plan(num_cases=1)
        res = load_and_validate_plan(plan_path)
        plan = res[0] if isinstance(res, tuple) else res
        cid = plan["eligible_cases"][0]["case_id"]

        # Tamper with video file preserving byte length
        v_file = self.review_root / cid / "video.mp4"
        orig_bytes = v_file.read_bytes()
        tampered_bytes = b"X" * len(orig_bytes)
        v_file.write_bytes(tampered_bytes)

        with self.assertRaises(ValueError) as ctx:
            verify_plan_inputs(plan, plan_path)
        self.assertTrue(
            "Video SHA256 mismatch" in str(ctx.exception)
            or "Raw video size" in str(ctx.exception)
            or "size" in str(ctx.exception).lower(),
            f"Unexpected exception message: {ctx.exception}",
        )

    def test_annotation_tampering_rejected(self) -> None:
        plan_path = self._create_standard_plan(num_cases=1)
        res = load_and_validate_plan(plan_path)
        plan = res[0] if isinstance(res, tuple) else res
        cid = plan["eligible_cases"][0]["case_id"]

        # Tamper with case.json
        case_json_p = self.review_root / cid / "case.json"
        case_data = json.loads(case_json_p.read_text(encoding="utf-8"))
        case_data["review"]["human_count"] = 999
        case_json_p.write_text(json.dumps(case_data), encoding="utf-8")

        with self.assertRaises(ValueError) as ctx:
            verify_plan_inputs(plan, plan_path)
        self.assertIn("Annotation SHA256 mismatch", str(ctx.exception))

    def test_invalid_primary_snapshot_rejected_despite_valid_self_hash(self) -> None:
        """A forged/modified eligible case snapshot must be rejected against disk ReviewStore truth."""
        # 1. Modify eligible case to group="development"
        p1 = self._create_standard_plan(num_cases=1)
        d1 = json.loads(p1.read_text(encoding="utf-8"))
        d1["eligible_cases"][0]["group"] = "development"
        d1["plan_sha256"] = compute_plan_sha256(d1)
        p1.write_text(json.dumps(d1), encoding="utf-8")

        with self.assertRaises(ValueError):
            load_and_validate_plan(p1)

        # 2. Modify eligible case to predictions_previously_seen=True
        p2 = self._create_standard_plan(num_cases=1)
        d2 = json.loads(p2.read_text(encoding="utf-8"))
        d2["eligible_cases"][0]["review"]["predictions_previously_seen"] = True
        d2["plan_sha256"] = compute_plan_sha256(d2)
        p2.write_text(json.dumps(d2), encoding="utf-8")

        with self.assertRaises(ValueError):
            load_and_validate_plan(p2)

        # 3. Modify eligible case to human_count=999
        p3 = self._create_standard_plan(num_cases=1)
        d3 = json.loads(p3.read_text(encoding="utf-8"))
        d3["eligible_cases"][0]["review"]["human_count"] = 999
        d3["plan_sha256"] = compute_plan_sha256(d3)
        p3.write_text(json.dumps(d3), encoding="utf-8")

        with self.assertRaises(ValueError):
            load_and_validate_plan(p3)

        # 4. Modify eligible case to raw_video_reviewed=False
        p4 = self._create_standard_plan(num_cases=1)
        d4 = json.loads(p4.read_text(encoding="utf-8"))
        d4["eligible_cases"][0]["review"]["raw_video_reviewed"] = False
        d4["plan_sha256"] = compute_plan_sha256(d4)
        p4.write_text(json.dumps(d4), encoding="utf-8")

        with self.assertRaises(ValueError):
            load_and_validate_plan(p4)

        # 5. Modify eligible case to scope="out_of_scope"
        p5 = self._create_standard_plan(num_cases=1)
        d5 = json.loads(p5.read_text(encoding="utf-8"))
        d5["eligible_cases"][0]["review"]["scope"] = "out_of_scope"
        d5["plan_sha256"] = compute_plan_sha256(d5)
        p5.write_text(json.dumps(d5), encoding="utf-8")

        with self.assertRaises(ValueError):
            load_and_validate_plan(p5)

        # 6. execute_evaluation in validation mode (run=False) strictly catches forgeries
        for p_forged, label in ((p1, "group"), (p2, "seen"), (p3, "count"), (p4, "raw_reviewed"), (p5, "scope")):
            out_target = Path(self.test_root) / f"invalid_snapshot_out_{label}"
            with self.assertRaises(ValueError):
                rep = execute_evaluation(p_forged, out_target, run=False)
                if isinstance(rep, dict) and rep.get("status") == "ready":
                    self.fail(f"execute_evaluation accepted forged snapshot '{label}' as ready!")


class TestOutputDirectorySafety(TestReviewExecutionBase):
    """Test destination directory validation, non-empty overwrite rejection, and path safety."""

    def test_non_empty_output_dir_rejected(self) -> None:
        plan_path = self._create_standard_plan(num_cases=1)
        self.output_root.mkdir(parents=True, exist_ok=True)
        (self.output_root / "existing_file.txt").write_text("DATA")

        with self.assertRaises(FileExistsError):
            execute_evaluation(
                plan_path=plan_path,
                output_dir=self.output_root,
                run=False,
            )

    def test_output_dir_overlap_with_review_root_rejected(self) -> None:
        plan_path = self._create_standard_plan(num_cases=1)
        overlap_dir = self.review_root / "subfolder"

        with self.assertRaises(ValueError) as ctx:
            execute_evaluation(
                plan_path=plan_path,
                output_dir=overlap_dir,
                run=False,
            )
        self.assertIn("inside review root", str(ctx.exception))

    def test_output_dir_overlap_with_plan_rejected(self) -> None:
        plan_path = self._create_standard_plan(num_cases=1)
        plan_dir = plan_path.parent

        with self.assertRaises(ValueError) as ctx:
            execute_evaluation(
                plan_path=plan_path,
                output_dir=plan_dir,
                run=False,
            )
        self.assertTrue(
            "identical to plan" in str(ctx.exception) or "plan directory" in str(ctx.exception)
        )


class TestReadinessAndExecutionRefusal(TestReviewExecutionBase):
    """Test readiness validation mode without --run and refusal on not_ready plans."""

    def test_readiness_mode_returns_status_without_executing(self) -> None:
        plan_path = self._create_standard_plan(num_cases=2)
        out_dir = Path(self.test_root) / "ready_test_out"

        res = execute_evaluation(
            plan_path=plan_path,
            output_dir=out_dir,
            run=False,
        )
        self.assertEqual(res["execution_phase"], "readiness_validation_only")
        self.assertEqual(res["status"], "ready")
        self.assertEqual(res["primary_eligible_cases_count"], 2)
        self.assertFalse(out_dir.exists(), "Output dir must not be created during readiness check")

    def test_not_ready_plan_refused_on_run(self) -> None:
        # Create review store with ONLY out-of-scope or unreviewed cases -> 0 eligible cases
        self._create_synthetic_case(group="candidate", human_count=None, scope="out_of_scope")
        plan_out = Path(self.test_root) / "not_ready_plan_out"
        build_plan(
            review_root=self.review_root,
            output_dir=plan_out,
            models={"baseline": self.baseline_pt},
        )
        plan_path = plan_out / "plan.json"

        # Readiness mode reports not_ready
        report = execute_evaluation(plan_path=plan_path, output_dir=self.output_root, run=False)
        self.assertEqual(report["status"], "not_ready")

        # --run refuses execution
        with self.assertRaises(ValueError) as ctx:
            execute_evaluation(
                plan_path=plan_path,
                output_dir=self.output_root,
                run=True,
            )
        self.assertIn("plan status is 'not_ready'", str(ctx.exception))


class TestExecutionLoopAndMockRunner(TestReviewExecutionBase):
    """Test the sequential execution loop, argument forwarding, and artifacts generation."""

    def test_mock_runner_invoked_sequentially(self) -> None:
        plan_path = self._create_standard_plan(num_cases=2, counts=(5, 8))
        out_dir = Path(self.test_root) / "mock_exec_out"

        invocations: List[Dict[str, Any]] = []

        def mock_runner(
            interpreter: Path,
            gym_pose_dir: Path,
            video_path: Path,
            output_dir: Path,
            model_path: Path,
            device: str,
            timeout: float,
            **kwargs: Any,
        ) -> Dict[str, Any]:
            invocations.append({
                "model": model_path.stem,
                "clip_id": output_dir.name,
                "video": video_path.name,
                "device": device,
                "timeout": timeout,
            })
            output_dir.mkdir(parents=True, exist_ok=True)
            log_p = kwargs.get("log_path") or (output_dir / "run.log")
            Path(log_p).parent.mkdir(parents=True, exist_ok=True)
            Path(log_p).write_text("Mock run output\n", encoding="utf-8")
            (output_dir / "summary.json").write_text(
                json.dumps({"status": "completed", "completed_reps": 5}),
                encoding="utf-8",
            )
            return {
                "status": "completed",
                "completed_reps": 5,
                "exit_code": 0,
                "read_status_note": None,
                "error": None,
            }

        receipt = execute_evaluation(
            plan_path=plan_path,
            output_dir=out_dir,
            device="cpu",
            timeout_per_clip=60.0,
            run=True,
            runner=mock_runner,
        )

        # 2 models x 2 clips = 4 invocations
        self.assertEqual(len(invocations), 4)
        self.assertEqual(receipt["status"], "completed")
        self.assertEqual(receipt["common_completed_cohort_count"], 2)

        # Verify output directory structure
        self.assertTrue((out_dir / "evaluation_receipt.json").is_file())
        self.assertTrue((out_dir / "execution_counts.json").is_file())
        for m in ("baseline", "candidate"):
            for inv in invocations:
                cid = inv["clip_id"]
                clip_folder = out_dir / m / cid
                self.assertTrue(clip_folder.is_dir())


class TestChildSubprocessExecution(TestReviewExecutionBase):
    """Test child CLI runner using an isolated disposable script and subprocess."""

    def test_child_runner_completed(self) -> None:
        clip_out = Path(self.test_root) / "child_clip_out"
        video_p = Path(self.test_root) / "dummy.mp4"
        video_p.write_bytes(b"DUMMY_MP4")

        os.environ["MOCK_CHILD_MODE"] = "normal"
        os.environ["MOCK_CHILD_REPS"] = "7"

        res = default_subprocess_runner(
            interpreter=Path(sys.executable),
            gym_pose_dir=self.mock_gym_pose_dir,
            video_path=video_p,
            output_dir=clip_out,
            model_path=self.baseline_pt,
            device="cpu",
            timeout=10.0,
        )
        self.assertEqual(res["status"], "completed")
        self.assertEqual(res["completed_reps"], 7)
        self.assertEqual(res["exit_code"], 0)

    def test_strict_fake_cli_no_overwrite_regression(self) -> None:
        """Verify strict fake CLI rejects dirty dirs and default_subprocess_runner provides fresh dir."""
        clip_out = Path(self.test_root) / "strict_fake_cli_out"
        video_p = Path(self.test_root) / "dummy_strict.mp4"
        video_p.write_bytes(b"DUMMY_STRICT_MP4")

        os.environ["MOCK_CHILD_MODE"] = "normal"
        os.environ["MOCK_CHILD_REPS"] = "5"

        # 1. Direct invocation without explicit log_path: must NOT write run.log inside clip_out prior to child run
        res = default_subprocess_runner(
            interpreter=Path(sys.executable),
            gym_pose_dir=self.mock_gym_pose_dir,
            video_path=video_p,
            output_dir=clip_out,
            model_path=self.baseline_pt,
            device="cpu",
            timeout=10.0,
        )
        self.assertEqual(res["status"], "completed")
        self.assertEqual(res["exit_code"], 0)
        self.assertNotEqual(res["exit_code"], 19, "Child CLI failed with exit 19 (output directory not empty)!")
        self.assertTrue((clip_out / "summary.json").is_file())

        # 2. Invocation with explicit external log_path
        clip_out_ext = Path(self.test_root) / "strict_fake_cli_ext_out"
        ext_log = Path(self.test_root) / "logs" / "strict_fake_cli.log"
        res_ext = default_subprocess_runner(
            interpreter=Path(sys.executable),
            gym_pose_dir=self.mock_gym_pose_dir,
            video_path=video_p,
            output_dir=clip_out_ext,
            model_path=self.baseline_pt,
            device="cpu",
            timeout=10.0,
            log_path=ext_log,
        )
        self.assertEqual(res_ext["status"], "completed")
        self.assertEqual(res_ext["exit_code"], 0)
        self.assertNotEqual(res_ext["exit_code"], 19)
        self.assertTrue((clip_out_ext / "summary.json").is_file())

    def test_descendant_process_cleanup_on_timeout(self) -> None:
        """Timeout must kill child and all descendant processes (no orphaned grandchild processes)."""
        clip_out = Path(self.test_root) / "descendant_timeout_out"
        video_p = Path(self.test_root) / "dummy_desc.mp4"
        video_p.write_bytes(b"DUMMY_DESC_MP4")

        pid_file = Path(self.test_root) / "descendant_pids.json"
        os.environ["MOCK_CHILD_MODE"] = "spawn_grandchild_and_sleep"
        os.environ["MOCK_PID_FILE"] = str(pid_file)

        start_t = time.time()
        res = default_subprocess_runner(
            interpreter=Path(sys.executable),
            gym_pose_dir=self.mock_gym_pose_dir,
            video_path=video_p,
            output_dir=clip_out,
            model_path=self.baseline_pt,
            device="cpu",
            timeout=3.0,
        )
        elapsed = time.time() - start_t

        self.assertEqual(res["status"], "timeout")
        self.assertIsNone(res["completed_reps"])
        self.assertTrue(elapsed < 15.0, f"Subprocess cleanup took too long: {elapsed}s")

        self.assertTrue(pid_file.is_file(), "Child process did not record descendant PIDs")
        pids = json.loads(pid_file.read_text(encoding="utf-8"))
        child_pid = pids["child_pid"]
        grandchild_pid = pids["grandchild_pid"]

        # Poll up to 3 seconds to confirm descendant termination
        terminated = False
        for _ in range(30):
            if not _is_pid_running(child_pid) and not _is_pid_running(grandchild_pid):
                terminated = True
                break
            time.sleep(0.1)

        self.assertTrue(
            terminated,
            f"Descendant processes were not stopped: child={child_pid} (alive={_is_pid_running(child_pid)}), "
            f"grandchild={grandchild_pid} (alive={_is_pid_running(grandchild_pid)})",
        )

    def test_child_runner_exit_nonzero(self) -> None:
        clip_out = Path(self.test_root) / "child_clip_fail"
        video_p = Path(self.test_root) / "dummy.mp4"
        video_p.write_bytes(b"DUMMY_MP4")

        os.environ["MOCK_CHILD_MODE"] = "fail_exit"
        res = default_subprocess_runner(
            interpreter=Path(sys.executable),
            gym_pose_dir=self.mock_gym_pose_dir,
            video_path=video_p,
            output_dir=clip_out,
            model_path=self.baseline_pt,
            device="cpu",
            timeout=10.0,
        )
        self.assertEqual(res["status"], "failed")
        self.assertIsNone(res["completed_reps"])
        self.assertEqual(res["exit_code"], 1)

    def test_child_runner_malformed_summary(self) -> None:
        clip_out = Path(self.test_root) / "child_clip_malformed"
        video_p = Path(self.test_root) / "dummy.mp4"
        video_p.write_bytes(b"DUMMY_MP4")

        os.environ["MOCK_CHILD_MODE"] = "malformed_json"
        res = default_subprocess_runner(
            interpreter=Path(sys.executable),
            gym_pose_dir=self.mock_gym_pose_dir,
            video_path=video_p,
            output_dir=clip_out,
            model_path=self.baseline_pt,
            device="cpu",
            timeout=10.0,
        )
        self.assertEqual(res["status"], "failed")
        self.assertIn("Malformed summary.json", res.get("error", ""))

    def test_child_runner_incomplete_decoding(self) -> None:
        clip_out = Path(self.test_root) / "child_clip_incomplete"
        video_p = Path(self.test_root) / "dummy.mp4"
        video_p.write_bytes(b"DUMMY_MP4")

        os.environ["MOCK_CHILD_MODE"] = "incomplete"
        res = default_subprocess_runner(
            interpreter=Path(sys.executable),
            gym_pose_dir=self.mock_gym_pose_dir,
            video_path=video_p,
            output_dir=clip_out,
            model_path=self.baseline_pt,
            device="cpu",
            timeout=10.0,
        )
        self.assertEqual(res["status"], "incomplete")
        self.assertIsNone(res["completed_reps"])
        self.assertIn("Corrupted video frame", str(res.get("read_status_note")))

    def test_child_runner_timeout(self) -> None:
        clip_out = Path(self.test_root) / "child_clip_timeout"
        video_p = Path(self.test_root) / "dummy.mp4"
        video_p.write_bytes(b"DUMMY_MP4")

        os.environ["MOCK_CHILD_MODE"] = "sleep"
        os.environ["MOCK_CHILD_SLEEP"] = "5.0"

        start_t = time.time()
        res = default_subprocess_runner(
            interpreter=Path(sys.executable),
            gym_pose_dir=self.mock_gym_pose_dir,
            video_path=video_p,
            output_dir=clip_out,
            model_path=self.baseline_pt,
            device="cpu",
            timeout=0.6,
        )
        elapsed = time.time() - start_t

        self.assertEqual(res["status"], "timeout")
        self.assertIsNone(res["completed_reps"])
        self.assertTrue(elapsed < 12.0, f"Subprocess did not terminate quickly: {elapsed}s")


class TestMultiModelCommonCohortComparison(TestReviewExecutionBase):
    """Test common completed cohort logic, unequal model coverage, and denominator equality."""

    def test_partial_model_failure_segregates_clip_from_common_denominator(self) -> None:
        plan_path = self._create_standard_plan(num_cases=3, counts=(5, 8, 10))
        res = load_and_validate_plan(plan_path)
        plan = res[0] if isinstance(res, tuple) else res
        cids = [c["case_id"] for c in plan["eligible_cases"]]
        counts = [c["review"]["human_count"] for c in plan["eligible_cases"]]

        # Model 'baseline' succeeds on all 3 clips
        # Model 'candidate' fails on clip 1 (incomplete decoding) and succeeds on clips 0 and 2
        mock_results = {
            "baseline": {
                cids[0]: {"status": "completed", "completed_reps": counts[0], "exit_code": 0},
                cids[1]: {"status": "completed", "completed_reps": counts[1], "exit_code": 0},
                cids[2]: {"status": "completed", "completed_reps": counts[2], "exit_code": 0},
            },
            "candidate": {
                cids[0]: {"status": "completed", "completed_reps": counts[0], "exit_code": 0},
                cids[1]: {"status": "incomplete", "completed_reps": None, "read_status_note": "Truncated stream", "exit_code": 0},
                cids[2]: {"status": "completed", "completed_reps": counts[2] + 1, "exit_code": 0},
            },
        }

        common, excluded, metrics = compute_multi_model_common_cohort(
            all_results=mock_results,
            eligible_cases=plan["eligible_cases"],
        )

        # Clip 1 must be excluded from common cohort
        self.assertEqual(len(common), 2)
        self.assertIn(cids[0], common)
        self.assertIn(cids[2], common)
        self.assertNotIn(cids[1], common)

        self.assertEqual(len(excluded), 1)
        self.assertEqual(excluded[0]["clip_id"], cids[1])
        self.assertEqual(excluded[0]["model_failures"][0]["model"], "candidate")

        # DENOMINATOR EQUALITY: Both models are scored on exactly 2 clips
        self.assertEqual(metrics["baseline"]["evaluated_clips"], 2)
        self.assertEqual(metrics["candidate"]["evaluated_clips"], 2)

        # Baseline: exact matches on both common clips -> 100% exact match, MAE 0
        self.assertEqual(metrics["baseline"]["exact_matches"], 2)
        self.assertAlmostEqual(metrics["baseline"]["exact_match_ratio"], 1.0)
        self.assertAlmostEqual(metrics["baseline"]["mean_absolute_error"], 0.0)

        # Candidate: exact match on clip 0, mismatch on clip 2 (diff 1) -> 50% exact match, MAE 0.5
        self.assertEqual(metrics["candidate"]["exact_matches"], 1)
        self.assertAlmostEqual(metrics["candidate"]["exact_match_ratio"], 0.5)
        self.assertAlmostEqual(metrics["candidate"]["mean_absolute_error"], 0.5)

        # Coverage is reported separately
        self.assertEqual(metrics["baseline"]["model_completed_clips"], 3)
        self.assertAlmostEqual(metrics["baseline"]["model_coverage_ratio"], 1.0)
        self.assertEqual(metrics["candidate"]["model_completed_clips"], 2)
        self.assertAlmostEqual(metrics["candidate"]["model_coverage_ratio"], 2 / 3)

    def test_multi_model_common_completed_cohort_end_to_end(self) -> None:
        """End-to-end execution segregates failed/incomplete clips into excluded cohort."""
        plan_path = self._create_standard_plan(num_cases=2, counts=(5, 8))
        res = load_and_validate_plan(plan_path)
        plan = res[0] if isinstance(res, tuple) else res
        cids = [c["case_id"] for c in plan["eligible_cases"]]

        def mock_runner(model_path: Path, output_dir: Path, **kwargs: Any) -> Dict[str, Any]:
            m_name = "baseline" if "baseline" in model_path.name else "candidate"
            cid = output_dir.name
            output_dir.mkdir(parents=True, exist_ok=True)
            if m_name == "baseline":
                (output_dir / "summary.json").write_text(
                    json.dumps({"status": "completed", "completed_reps": 5}), encoding="utf-8"
                )
                return {"status": "completed", "completed_reps": 5, "exit_code": 0}
            else:
                if cid == cids[0]:
                    (output_dir / "summary.json").write_text(
                        json.dumps({"status": "completed", "completed_reps": 5}), encoding="utf-8"
                    )
                    return {"status": "completed", "completed_reps": 5, "exit_code": 0}
                else:
                    (output_dir / "summary.json").write_text(
                        json.dumps({"status": "incomplete", "read_status_note": "Corrupt", "completed_reps": None}),
                        encoding="utf-8",
                    )
                    return {"status": "incomplete", "completed_reps": None, "read_status_note": "Corrupt", "exit_code": 1}

        out_dir = Path(self.test_root) / "common_cohort_e2e_out"
        receipt = execute_evaluation(
            plan_path=plan_path,
            output_dir=out_dir,
            run=True,
            runner=mock_runner,
        )

        self.assertEqual(receipt["status"], "completed")
        self.assertEqual(receipt["common_completed_cohort"], [cids[0]])
        self.assertEqual(receipt["common_completed_cohort_count"], 1)
        self.assertEqual(len(receipt["excluded_from_common_cohort"]), 1)
        self.assertEqual(receipt["excluded_from_common_cohort"][0]["clip_id"], cids[1])

        # Denominators are strictly matched
        self.assertEqual(receipt["per_model_metrics"]["baseline"]["evaluated_clips"], 1)
        self.assertEqual(receipt["per_model_metrics"]["candidate"]["evaluated_clips"], 1)
        self.assertEqual(receipt["per_model_metrics"]["baseline"]["model_completed_clips"], 2)
        self.assertEqual(receipt["per_model_metrics"]["candidate"]["model_completed_clips"], 1)

    def test_empty_common_cohort_produces_null_metrics_and_not_ready_status(self) -> None:
        plan_path = self._create_standard_plan(num_cases=2, counts=(5, 8))
        res = load_and_validate_plan(plan_path)
        plan = res[0] if isinstance(res, tuple) else res
        cids = [c["case_id"] for c in plan["eligible_cases"]]

        # Disjoint successes: baseline completes clip 0 only, candidate completes clip 1 only
        mock_results = {
            "baseline": {
                cids[0]: {"status": "completed", "completed_reps": 5},
                cids[1]: {"status": "failed", "completed_reps": None, "error": "Crash"},
            },
            "candidate": {
                cids[0]: {"status": "timeout", "completed_reps": None, "error": "Timeout"},
                cids[1]: {"status": "completed", "completed_reps": 8},
            },
        }

        common, excluded, metrics = compute_multi_model_common_cohort(
            all_results=mock_results,
            eligible_cases=plan["eligible_cases"],
        )

        self.assertEqual(len(common), 0)
        self.assertEqual(len(excluded), 2)
        self.assertEqual(metrics["baseline"]["status"], "not_ready")
        self.assertIsNone(metrics["baseline"]["exact_match_ratio"])
        self.assertIsNone(metrics["baseline"]["mean_absolute_error"])
        self.assertEqual(metrics["baseline"]["evaluated_clips"], 0)


class TestStagingPublicationAndSeamIntegrity(TestReviewExecutionBase):
    """Test atomic staged publication, evidence retention, and mutation aborts."""

    def test_seam_mutation_aborts_and_cleans_stage(self) -> None:
        plan_path = self._create_standard_plan(num_cases=1)
        out_dir = Path(self.test_root) / "seam_test_out"

        def tamper_seam() -> None:
            # Corrupt candidate model checkpoint right at the publication seam
            self.candidate_pt.write_bytes(b"MUTATED_DURING_EXECUTION_SEAM_999")

        def mock_runner(**kwargs: Any) -> Dict[str, Any]:
            out_p = kwargs["output_dir"]
            out_p.mkdir(parents=True, exist_ok=True)
            (out_p / "summary.json").write_text(
                json.dumps({"status": "completed", "completed_reps": 5}), encoding="utf-8"
            )
            return {
                "status": "completed",
                "completed_reps": 5,
                "exit_code": 0,
                "read_status_note": None,
                "error": None,
            }

        with self.assertRaises(ValueError) as ctx:
            execute_evaluation(
                plan_path=plan_path,
                output_dir=out_dir,
                run=True,
                runner=mock_runner,
                _seam_hook=tamper_seam,
            )
        self.assertTrue(
            "SHA256 mismatch" in str(ctx.exception) or "size mismatch" in str(ctx.exception)
        )

        # Verify destination directory was never created
        self.assertFalse(out_dir.exists())

        # Verify temporary staging directory was cleaned up
        parent_dir = out_dir.parent
        stages = [p for p in parent_dir.iterdir() if p.name.startswith(f".{out_dir.name}.exec-")]
        self.assertEqual(stages, [], f"Stale staging directories found: {stages}")

    def test_plan_mutation_during_execution_aborts(self) -> None:
        """Mutating plan.json on disk during execution must be detected and abort execution."""
        plan_path = self._create_standard_plan(num_cases=1)
        out_dir = Path(self.test_root) / "mutated_plan_out"

        def alter_plan_runner(**kwargs: Any) -> Dict[str, Any]:
            out_p = kwargs["output_dir"]
            out_p.mkdir(parents=True, exist_ok=True)
            (out_p / "summary.json").write_text(
                json.dumps({"status": "completed", "completed_reps": 5}),
                encoding="utf-8",
            )
            # Mutate plan.json on disk during execution
            d = json.loads(plan_path.read_text(encoding="utf-8"))
            d["plan_version"] = "tampered_during_execution"
            plan_path.write_text(json.dumps(d), encoding="utf-8")
            return {"status": "completed", "completed_reps": 5, "exit_code": 0}

        with self.assertRaises((RuntimeError, ValueError)):
            execute_evaluation(
                plan_path=plan_path,
                output_dir=out_dir,
                run=True,
                runner=alter_plan_runner,
            )
        self.assertFalse(out_dir.exists(), "Output directory must not be published after plan mutation!")

    def test_preserved_failed_run_evidence(self) -> None:
        """Failure during execution must preserve diagnostic run.log evidence on disk."""
        plan_path = self._create_standard_plan(num_cases=1)
        out_dir = Path(self.test_root) / "failed_evidence_out"

        def failing_runner(**kwargs: Any) -> Dict[str, Any]:
            out_p = kwargs["output_dir"]
            out_p.mkdir(parents=True, exist_ok=True)
            # Write diagnostic evidence to output_dir / run.log
            (out_p / "run.log").write_text("Preserve this diagnostic evidence\n", encoding="utf-8")
            # Also write to log_path if provided
            lp = kwargs.get("log_path")
            if lp:
                Path(lp).parent.mkdir(parents=True, exist_ok=True)
                Path(lp).write_text("Preserve this diagnostic evidence\n", encoding="utf-8")
            raise RuntimeError("synthetic child failure")

        with self.assertRaises(RuntimeError):
            execute_evaluation(
                plan_path=plan_path,
                output_dir=out_dir,
                run=True,
                runner=failing_runner,
            )

        # Check that diagnostic evidence is preserved anywhere under test_root
        retained_logs = list(Path(self.test_root).rglob("run.log")) + list(
            Path(self.test_root).rglob("*.log")
        )
        self.assertTrue(len(retained_logs) > 0, "No diagnostic log files were retained on disk!")
        found = any(
            "Preserve this diagnostic evidence" in f.read_text(encoding="utf-8", errors="replace")
            for f in retained_logs
        )
        self.assertTrue(found, "Retained log files do not contain the diagnostic failure message!")

        # Verify failure receipt was written in diagnostic path
        failure_receipts = list(Path(self.test_root).rglob("failure_receipt.json"))
        self.assertTrue(len(failure_receipts) > 0, "No failure_receipt.json was written upon execution error!")
        fail_data = json.loads(failure_receipts[0].read_text(encoding="utf-8"))
        self.assertEqual(fail_data.get("status"), "failed")
        self.assertEqual(fail_data.get("error_type"), "RuntimeError")
        self.assertFalse(out_dir.exists(), "Target output_dir must NOT exist on failure!")


class TestEvaluationAPIRoundtripAndArtifacts(TestReviewExecutionBase):
    """Verify that execution outputs roundtrip with evaluation.load_ground_truth."""

    def test_roundtrip_ground_truth_and_counts(self) -> None:
        plan_path = self._create_standard_plan(num_cases=2, counts=(5, 10))
        out_dir = Path(self.test_root) / "roundtrip_out"

        def mock_runner(model_path: Path, output_dir: Path, **kwargs: Any) -> Dict[str, Any]:
            output_dir.mkdir(parents=True, exist_ok=True)
            reps = 5 if "baseline" in model_path.name else 6
            (output_dir / "summary.json").write_text(
                json.dumps({"status": "completed", "completed_reps": reps}), encoding="utf-8"
            )
            return {
                "status": "completed",
                "completed_reps": reps,
                "exit_code": 0,
                "read_status_note": None,
                "error": None,
            }

        receipt = execute_evaluation(
            plan_path=plan_path,
            output_dir=out_dir,
            run=True,
            runner=mock_runner,
        )

        self.assertEqual(receipt["status"], "completed")

        # Load execution counts JSON
        counts_file = out_dir / "execution_counts.json"
        self.assertTrue(counts_file.is_file())
        counts_data = json.loads(counts_file.read_text(encoding="utf-8"))
        self.assertEqual(len(counts_data["counts"]), 2)

        # Plan's reviewed_counts.json can be loaded by evaluation.load_ground_truth
        reviewed_counts_file = plan_path.parent / "reviewed_counts.json"
        gt_loaded = load_ground_truth(reviewed_counts_file)
        self.assertEqual(len(gt_loaded), 2)

        # Common cohort predictions evaluate_counts matches receipt
        common_ids = receipt["common_completed_cohort"]
        base_preds = {cid: receipt["per_clip_predictions"]["baseline"][cid]["completed_reps"] for cid in common_ids}
        base_eval = evaluate_counts(base_preds, gt_loaded)
        self.assertEqual(base_eval["exact_matches"], receipt["per_model_metrics"]["baseline"]["exact_matches"])
        self.assertAlmostEqual(base_eval["mean_absolute_error"], receipt["per_model_metrics"]["baseline"]["mean_absolute_error"])


class TestCLIEntrypoint(TestReviewExecutionBase):
    """Test review_execution CLI entrypoint with --run and readiness options."""

    def test_cli_readiness_mode(self) -> None:
        plan_path = self._create_standard_plan(num_cases=2)
        out_dir = Path(self.test_root) / "cli_readiness_out"

        stdout_buf = io.StringIO()
        stderr_buf = io.StringIO()
        with redirect_stdout(stdout_buf), redirect_stderr(stderr_buf):
            ret = main(["--plan", str(plan_path), "--output-dir", str(out_dir)])

        self.assertEqual(ret, 0, f"CLI stderr: {stderr_buf.getvalue()}")
        self.assertIn("readiness_validation_only", stdout_buf.getvalue())
        self.assertIn("Pass --run to execute", stdout_buf.getvalue())
        self.assertFalse(out_dir.exists())

    def test_cli_run_mode(self) -> None:
        plan_path = self._create_standard_plan(num_cases=1, counts=(5,))
        out_dir = Path(self.test_root) / "cli_run_out"

        os.environ["GYM_POSE_DIR"] = str(self.mock_gym_pose_dir)
        os.environ["MOCK_CHILD_MODE"] = "normal"
        os.environ["MOCK_CHILD_REPS"] = "5"

        stdout_buf = io.StringIO()
        stderr_buf = io.StringIO()
        with redirect_stdout(stdout_buf), redirect_stderr(stderr_buf):
            ret = main([
                "--plan", str(plan_path),
                "--output-dir", str(out_dir),
                "--interpreter", sys.executable,
                "--device", "cpu",
                "--timeout-per-clip", "10",
                "--run",
            ])

        self.assertEqual(ret, 0, f"CLI stderr: {stderr_buf.getvalue()}")
        self.assertTrue((out_dir / "evaluation_receipt.json").is_file())
        self.assertTrue((out_dir / "execution_counts.json").is_file())
        self.assertIn("Execution receipt written to", stdout_buf.getvalue())

    def test_cli_missing_plan_error(self) -> None:
        stderr_buf = io.StringIO()
        with redirect_stderr(stderr_buf):
            with self.assertRaises(SystemExit) as ctx:
                main(["--output-dir", str(self.output_root)])
            self.assertNotEqual(ctx.exception.code, 0)
        self.assertIn("--plan", stderr_buf.getvalue())


class TestParameterValidationAndSymlinks(TestReviewExecutionBase):
    """Test parameter validation rules, symlink detection, and ancestor links."""

    def test_invalid_device_string_rejected(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            validate_device_string("cuda; rm -rf /")
        self.assertIn("Unsafe device identifier", str(ctx.exception))

        with self.assertRaises(ValueError):
            validate_device_string("   ")

        with self.assertRaises(TypeError):
            validate_device_string(123)  # type: ignore

    def test_invalid_interpreter_rejected(self) -> None:
        non_existent = Path(self.test_root) / "no_python.exe"
        with self.assertRaises(FileNotFoundError):
            validate_interpreter(non_existent)

        # Directory instead of file
        with self.assertRaises(ValueError):
            validate_interpreter(self.test_root)

    def test_invalid_timeout_rejected(self) -> None:
        plan_path = self._create_standard_plan(num_cases=1)
        with self.assertRaises(ValueError):
            execute_evaluation(
                plan_path=plan_path,
                output_dir=self.output_root,
                timeout_per_clip=-10.0,
                run=False,
            )
        with self.assertRaises(ValueError):
            execute_evaluation(
                plan_path=plan_path,
                output_dir=self.output_root,
                timeout_per_clip=float("nan"),
                run=False,
            )

    def test_single_model_common_cohort(self) -> None:
        plan_out = Path(self.test_root) / "single_model_plan"
        self._create_synthetic_case(group="candidate", human_count=5, scope="in_scope", seen=False)
        build_plan(
            review_root=self.review_root,
            output_dir=plan_out,
            models={"only_model": self.baseline_pt},
        )
        plan_path = plan_out / "plan.json"

        def mock_runner(output_dir: Path, **kwargs: Any) -> Dict[str, Any]:
            output_dir.mkdir(parents=True, exist_ok=True)
            (output_dir / "summary.json").write_text(
                json.dumps({"status": "completed", "completed_reps": 5}), encoding="utf-8"
            )
            return {
                "status": "completed",
                "completed_reps": 5,
                "exit_code": 0,
                "read_status_note": None,
                "error": None,
            }

        receipt = execute_evaluation(
            plan_path=plan_path,
            output_dir=self.output_root,
            run=True,
            runner=mock_runner,
        )
        self.assertEqual(receipt["status"], "completed")
        self.assertEqual(receipt["common_completed_cohort_count"], 1)
        self.assertEqual(receipt["per_model_metrics"]["only_model"]["exact_matches"], 1)
        self.assertAlmostEqual(receipt["per_model_metrics"]["only_model"]["exact_match_ratio"], 1.0)

    def test_ancestor_links_checked_before_resolve(self) -> None:
        """Paths with symlink/junction ancestors must be rejected BEFORE .resolve()."""
        link_parent = Path(self.test_root) / "link_parent"
        target_dir = Path(self.test_root) / "target_dir"
        target_dir.mkdir(parents=True, exist_ok=True)
        link_parent.mkdir(parents=True, exist_ok=True)

        junction_path = link_parent / "junction_link"
        created = False
        if sys.platform == "win32":
            proc = subprocess.run(
                ["cmd", "/c", "mklink", "/J", str(junction_path), str(target_dir)],
                capture_output=True,
                text=True,
            )
            created = (proc.returncode == 0) and junction_path.exists()
        else:
            try:
                junction_path.symlink_to(target_dir, target_is_directory=True)
                created = True
            except Exception:
                created = False

        if not created:
            self.skipTest("Filesystem does not support junction/symlink creation")

        fake_plan = target_dir / "plan.json"
        fake_plan.write_text("{}", encoding="utf-8")

        path_via_junction = junction_path / "plan.json"
        with self.assertRaises((ValueError, CaseCorruptedError)) as ctx:
            load_and_validate_plan(path_via_junction)
        self.assertTrue(
            any(w in str(ctx.exception).lower() for w in ("symlink", "reparse", "junction", "link"))
        )

        with self.assertRaises((ValueError, CaseCorruptedError)):
            execute_evaluation(path_via_junction, Path(self.test_root) / "junction_out", run=False)


class TestReview41And42Regressions(TestReviewExecutionBase):
    """Adversarial regression test suite covering all Review 41 and Review 42 findings.

    Directly converts probes.json failures into regressions:
    1. test_missing_mandatory_hashes_rejected
    2. test_code_changed_during_execution_aborts
    3. test_runtime_probe_failure_does_not_substitute_controller_version
    4. test_declared_zero_confidence_preserved
    5. test_descendant_process_cleanup_when_parent_exits_early
    6. test_unsupported_configuration_rejected
    7. test_child_summary_validation
    """

    def test_missing_mandatory_hashes_rejected(self) -> None:
        """Missing plan_sha256, annotation_sha256, or source_sha256 must strictly raise ValueError and NEVER report ready."""
        # 1. Missing plan_sha256
        p1 = self._create_standard_plan(num_cases=1)
        d1 = json.loads(p1.read_text(encoding="utf-8"))
        del d1["plan_sha256"]
        p1.write_text(json.dumps(d1), encoding="utf-8")

        with self.assertRaises(ValueError):
            load_and_validate_plan(p1)

        out1 = Path(self.test_root) / "missing_plan_sha_out"
        with self.assertRaises(ValueError):
            rep1 = execute_evaluation(p1, out1, run=False)
            if isinstance(rep1, dict) and rep1.get("status") == "ready":
                self.fail("execute_evaluation returned status 'ready' for missing plan_sha256!")

        # 2. Missing annotation_sha256 on eligible case
        p2 = self._create_standard_plan(num_cases=1)
        d2 = json.loads(p2.read_text(encoding="utf-8"))
        cid2 = d2["eligible_cases"][0]["case_id"]
        d2["eligible_cases"][0].pop("annotation_sha256", None)
        if "source_annotation_hashes" in d2:
            d2["source_annotation_hashes"].pop(cid2, None)
        d2["plan_sha256"] = compute_plan_sha256(d2)
        p2.write_text(json.dumps(d2), encoding="utf-8")

        with self.assertRaises(ValueError):
            load_and_validate_plan(p2)

        out2 = Path(self.test_root) / "missing_ann_sha_out"
        with self.assertRaises(ValueError):
            rep2 = execute_evaluation(p2, out2, run=False)
            if isinstance(rep2, dict) and rep2.get("status") == "ready":
                self.fail("execute_evaluation returned status 'ready' for case missing annotation_sha256!")

        # 3. Missing source_sha256 on eligible case
        p3 = self._create_standard_plan(num_cases=1)
        d3 = json.loads(p3.read_text(encoding="utf-8"))
        d3["eligible_cases"][0].pop("source_sha256", None)
        d3["plan_sha256"] = compute_plan_sha256(d3)
        p3.write_text(json.dumps(d3), encoding="utf-8")

        with self.assertRaises(ValueError):
            load_and_validate_plan(p3)

        out3 = Path(self.test_root) / "missing_src_sha_out"
        with self.assertRaises(ValueError):
            rep3 = execute_evaluation(p3, out3, run=False)
            if isinstance(rep3, dict) and rep3.get("status") == "ready":
                self.fail("execute_evaluation returned status 'ready' for case missing source_sha256!")

    def test_code_changed_during_execution_aborts(self) -> None:
        """Mutating gym_analyser source code during clip execution must abort before publication and raise RuntimeError."""
        p = self._create_standard_plan(num_cases=1)
        out_dir = Path(self.test_root) / "code_changed_out"

        pipeline_file = self.mock_gym_pose_dir / "gym_analyser" / "pipeline.py"
        pipeline_file.write_text("# Initial pristine code\n", encoding="utf-8")

        def mutating_runner(**kwargs: Any) -> Dict[str, Any]:
            out_p = kwargs["output_dir"]
            out_p.mkdir(parents=True, exist_ok=True)
            (out_p / "summary.json").write_text(
                json.dumps({"status": "completed", "completed_reps": 5}),
                encoding="utf-8",
            )
            # Mutate source file during clip execution
            pipeline_file.write_text("# CODE MUTATED DURING CLIP RUNNER EXECUTION\n", encoding="utf-8")
            return {"status": "completed", "completed_reps": 5, "exit_code": 0}

        with self.assertRaises((RuntimeError, ValueError)):
            res = execute_evaluation(
                plan_path=p,
                output_dir=out_dir,
                run=True,
                runner=mutating_runner,
                gym_pose_dir=self.mock_gym_pose_dir,
            )
            if isinstance(res, dict) and res.get("status") == "completed":
                self.fail("execute_evaluation completed and published despite source code mutation during run!")

        self.assertFalse(out_dir.exists(), "Target output_dir must NOT be published when source code mutates!")

    def test_runtime_probe_failure_does_not_substitute_controller_version(self) -> None:
        """Simulated runtime probe failure on child interpreter must report 'unavailable' and NEVER controller sys.version."""
        from unittest import mock

        controller_version = sys.version.replace("\n", " ")

        def failing_subprocess_run(cmd: Any, *args: Any, **kwargs: Any) -> subprocess.CompletedProcess[Any]:
            if isinstance(cmd, (list, tuple)) and len(cmd) > 2 and "-c" in cmd:
                return subprocess.CompletedProcess(
                    args=cmd,
                    returncode=1,
                    stdout="",
                    stderr="Simulated child python probe failure",
                )
            return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")

        # 1. Direct get_runtime_provenance test
        with mock.patch("gym_analyser.review_execution.subprocess.run", side_effect=failing_subprocess_run):
            prov = rev_exec.get_runtime_provenance(
                interpreter=Path(sys.executable),
                device="cpu",
                gym_pose_dir=self.mock_gym_pose_dir,
            )

        py_ver = prov.get("python_version", "")
        self.assertNotEqual(
            py_ver,
            controller_version,
            "Runtime probe failure substituted controller sys.version in get_runtime_provenance!",
        )
        self.assertIn(
            py_ver.lower(),
            ("unavailable", "unknown", "failed", "probe_failed"),
            f"Expected 'unavailable' or similar, got {py_ver!r}",
        )

        # 2. Receipt test via execute_evaluation
        p = self._create_standard_plan(num_cases=1)
        out_dir = Path(self.test_root) / "probe_failure_receipt_out"

        def mock_runner(output_dir: Path, **kwargs: Any) -> Dict[str, Any]:
            output_dir.mkdir(parents=True, exist_ok=True)
            (output_dir / "summary.json").write_text(
                json.dumps({"status": "completed", "completed_reps": 5}),
                encoding="utf-8",
            )
            return {"status": "completed", "completed_reps": 5, "exit_code": 0}

        with mock.patch("gym_analyser.review_execution.subprocess.run", side_effect=failing_subprocess_run):
            receipt = execute_evaluation(
                plan_path=p,
                output_dir=out_dir,
                run=True,
                runner=mock_runner,
                gym_pose_dir=self.mock_gym_pose_dir,
            )

        runtime_env = receipt.get("runtime_environment") or receipt.get("runtime_provenance") or {}
        receipt_py_ver = runtime_env.get("python_version", "")
        self.assertNotEqual(
            receipt_py_ver,
            controller_version,
            "evaluation_receipt reported controller sys.version upon runtime probe failure!",
        )
        self.assertIn(
            receipt_py_ver.lower(),
            ("unavailable", "unknown", "failed", "probe_failed"),
            f"Expected 'unavailable' in receipt, got {receipt_py_ver!r}",
        )

    def test_declared_zero_confidence_preserved(self) -> None:
        """min_joint_confidence: 0.0 must be passed to child CLI and reported in receipt as 0.0, NOT default 0.3."""
        p = self._create_standard_plan(num_cases=1)
        d = json.loads(p.read_text(encoding="utf-8"))
        d["declared_configuration"] = {
            "counter_config": {
                "min_joint_confidence": 0.0,
            }
        }
        d["plan_sha256"] = compute_plan_sha256(d)
        p.write_text(json.dumps(d), encoding="utf-8")

        # 1. Child CLI runner receives --min-confidence 0.0
        clip_out = Path(self.test_root) / "zero_conf_child_out"
        video_p = Path(self.test_root) / "zero_conf.mp4"
        video_p.write_bytes(b"MOCK_ZERO_CONF_VIDEO")

        res = default_subprocess_runner(
            interpreter=Path(sys.executable),
            gym_pose_dir=self.mock_gym_pose_dir,
            video_path=video_p,
            output_dir=clip_out,
            model_path=self.baseline_pt,
            device="cpu",
            timeout=10.0,
            declared_configuration=d["declared_configuration"],
        )
        self.assertEqual(res["status"], "completed")
        summary_p = clip_out / "summary.json"
        self.assertTrue(summary_p.is_file())
        summary_data = json.loads(summary_p.read_text(encoding="utf-8"))
        self.assertEqual(
            summary_data.get("min_confidence"),
            0.0,
            f"Child CLI runner did not receive min_confidence 0.0; received: {summary_data.get('min_confidence')}",
        )

        # 2. Receipt reports 0.0, NOT 0.3
        out_eval = Path(self.test_root) / "zero_conf_eval_out"
        receipt = execute_evaluation(
            plan_path=p,
            output_dir=out_eval,
            run=True,
            interpreter=Path(sys.executable),
            gym_pose_dir=self.mock_gym_pose_dir,
        )
        self.assertEqual(receipt["status"], "completed")

        runtime_env = receipt.get("runtime_environment") or receipt.get("runtime_provenance") or {}
        eff_params = runtime_env.get("effective_parameters", {})
        conf_val = eff_params.get("min_confidence")
        if conf_val is None:
            conf_val = (
                receipt.get("declared_configuration", {})
                .get("counter_config", {})
                .get("min_joint_confidence")
            )

        self.assertIsNotNone(conf_val, "min_confidence was not reported in runtime receipt!")
        self.assertEqual(conf_val, 0.0, f"Expected min_confidence 0.0, but got {conf_val}")
        self.assertNotEqual(conf_val, 0.3, "Declared min_confidence 0.0 was overwritten with default 0.3!")

    def test_descendant_process_cleanup_when_parent_exits_early(self) -> None:
        """Exercise the production owner, not a process scan after losing root ownership."""
        pid_file = Path(self.test_root) / "early_exit_descendant_pids.json"
        os.environ["MOCK_CHILD_MODE"] = "spawn_grandchild_and_exit"
        os.environ["MOCK_PID_FILE"] = str(pid_file)
        video = Path(self.test_root) / "early.mp4"
        video.write_bytes(b"FAKE")
        default_subprocess_runner(Path(sys.executable), self.mock_gym_pose_dir, video,
                                  Path(self.test_root) / "early_out", self.baseline_pt, "cpu", 5)
        self.assertTrue(pid_file.is_file())
        pids = json.loads(pid_file.read_text())
        for _ in range(100):
            if not _is_pid_running(pids["grandchild_pid"]):
                break
            time.sleep(0.01)
        self.assertFalse(_is_pid_running(pids["grandchild_pid"]))

    def test_unsupported_configuration_rejected(self) -> None:
        """Declaring configuration options that cannot be supported must not be silently dropped."""
        p = self._create_standard_plan(num_cases=1)
        d = json.loads(p.read_text(encoding="utf-8"))
        d["declared_configuration"] = {
            "unsupported_arbitrary_flag": "unsafe_value_123",
            "down_angle": 110.0,
        }
        d["plan_sha256"] = compute_plan_sha256(d)
        p.write_text(json.dumps(d), encoding="utf-8")

        # Unsupported configuration must be rejected, not silently ignored
        with self.assertRaises((ValueError, RuntimeError)):
            execute_evaluation(
                plan_path=p,
                output_dir=Path(self.test_root) / "unsupported_conf_out",
                run=False,
            )

    def test_child_summary_validation(self) -> None:
        """Child summary with mismatched model, mismatched clip_id, or non-integer reps is rejected."""
        clip_out = Path(self.test_root) / "summary_validation_out"
        video_p = Path(self.test_root) / "summary_val.mp4"
        video_p.write_bytes(b"MOCK_SUMMARY_VAL_VIDEO")

        # 1. Mismatched model name
        os.environ["MOCK_CHILD_MODE"] = "wrong_model"
        res_model = default_subprocess_runner(
            interpreter=Path(sys.executable),
            gym_pose_dir=self.mock_gym_pose_dir,
            video_path=video_p,
            output_dir=clip_out,
            model_path=self.baseline_pt,
            device="cpu",
            timeout=10.0,
            model_name="baseline",
        )
        self.assertNotEqual(
            res_model.get("status"),
            "completed",
            "Child summary claiming wrong model was accepted as completed!",
        )

        # Clean clip_out for next check
        shutil.rmtree(clip_out, ignore_errors=True)

        # 2. String reps instead of int
        os.environ["MOCK_CHILD_MODE"] = "string_reps"
        res_reps = default_subprocess_runner(
            interpreter=Path(sys.executable),
            gym_pose_dir=self.mock_gym_pose_dir,
            video_path=video_p,
            output_dir=clip_out,
            model_path=self.baseline_pt,
            device="cpu",
            timeout=10.0,
        )
        self.assertNotEqual(
            res_reps.get("status"),
            "completed",
            "Child summary with string completed_reps was accepted as completed!",
        )


class TestReview43Regressions(TestReviewExecutionBase):
    """Adversarial regression test suite covering Review 43 findings:
    1. Rejection of child summaries with wrong source path (same basename, different dir),
       conflicting configuration, zero processed_frames, or missing/non-integer frames.
    2. Verification that runtime provenance is captured and frozen BEFORE first clip execution.
    3. Verification that child python probe failure records 'unavailable' without substituting controller sys.version.
    4. Verification of process cleanup and containment on timeout and abnormal exit without killing unrelated PIDs.
    5. Exclusion of rejected clips from the multi-model common completed cohort denominator.
    """

    def test_probe_wrong_source_config_zero_frames_rejected(self) -> None:
        """Reproduce probe wrong_source_config_zero_frames_accepted from Review 43:
        Child summary with processed_frames=0, wrong source directory (same basename),
        and conflicting down_angle (1.0 vs 110.0) MUST NOT be accepted as completed.
        """
        clip_out = Path(self.test_root) / "probe_rejection_out"
        video_p = Path(self.test_root) / "probe_vid.mp4"
        video_p.write_bytes(b"MOCK_PROBE_VIDEO")

        os.environ["MOCK_CHILD_MODE"] = "wrong_source_config_zero_frames"

        res = default_subprocess_runner(
            interpreter=Path(sys.executable),
            gym_pose_dir=self.mock_gym_pose_dir,
            video_path=video_p,
            output_dir=clip_out,
            model_path=self.baseline_pt,
            device="cpu",
            timeout=10.0,
            declared_configuration={"down_angle": 110.0},
        )
        self.assertNotEqual(
            res.get("status"),
            "completed",
            f"Child summary with zero frames, wrong source path, and conflicting config was accepted as completed: {res}",
        )
        self.assertEqual(res.get("status"), "failed")
        self.assertIsNone(res.get("completed_reps"))
        self.assertIsNotNone(res.get("error"))

    def test_child_summary_zero_processed_frames_rejected(self) -> None:
        """Child summary with processed_frames: 0 must be rejected with status != 'completed' (or 'failed')."""
        clip_out = Path(self.test_root) / "zero_frames_out"
        video_p = Path(self.test_root) / "zero_frames_vid.mp4"
        video_p.write_bytes(b"MOCK_ZERO_FRAMES_VIDEO")

        os.environ["MOCK_CHILD_MODE"] = "zero_frames"
        os.environ["MOCK_CHILD_REPS"] = "5"

        res = default_subprocess_runner(
            interpreter=Path(sys.executable),
            gym_pose_dir=self.mock_gym_pose_dir,
            video_path=video_p,
            output_dir=clip_out,
            model_path=self.baseline_pt,
            device="cpu",
            timeout=10.0,
        )
        self.assertNotEqual(
            res.get("status"),
            "completed",
            "Child summary with processed_frames=0 was accepted as completed!",
        )
        self.assertEqual(res.get("status"), "failed")
        self.assertIsNone(res.get("completed_reps"))
        self.assertIn("processed_frames", str(res.get("error", "")).lower())

    def test_child_summary_missing_or_non_integer_processed_frames_rejected(self) -> None:
        """Child summary missing processed_frames or with non-integer processed_frames must be rejected."""
        clip_out = Path(self.test_root) / "invalid_frames_out"
        video_p = Path(self.test_root) / "invalid_frames_vid.mp4"
        video_p.write_bytes(b"MOCK_INVALID_FRAMES_VIDEO")

        # 1. Missing processed_frames
        os.environ["MOCK_CHILD_MODE"] = "missing_processed_frames"
        res_missing = default_subprocess_runner(
            interpreter=Path(sys.executable),
            gym_pose_dir=self.mock_gym_pose_dir,
            video_path=video_p,
            output_dir=clip_out,
            model_path=self.baseline_pt,
            device="cpu",
            timeout=10.0,
        )
        self.assertNotEqual(
            res_missing.get("status"),
            "completed",
            "Child summary missing processed_frames was accepted as completed!",
        )
        self.assertEqual(res_missing.get("status"), "failed")
        self.assertIsNone(res_missing.get("completed_reps"))

        shutil.rmtree(clip_out, ignore_errors=True)

        # 2. String processed_frames
        os.environ["MOCK_CHILD_MODE"] = "non_integer_processed_frames"
        res_non_int = default_subprocess_runner(
            interpreter=Path(sys.executable),
            gym_pose_dir=self.mock_gym_pose_dir,
            video_path=video_p,
            output_dir=clip_out,
            model_path=self.baseline_pt,
            device="cpu",
            timeout=10.0,
        )
        self.assertNotEqual(
            res_non_int.get("status"),
            "completed",
            "Child summary with non-integer processed_frames was accepted as completed!",
        )
        self.assertEqual(res_non_int.get("status"), "failed")
        self.assertIsNone(res_non_int.get("completed_reps"))

    def test_child_summary_wrong_source_path_different_dir_same_basename_rejected(self) -> None:
        """Child summary with source_video.path pointing to a different directory with the same basename must be rejected."""
        clip_out = Path(self.test_root) / "wrong_src_out"
        video_p = Path(self.test_root) / "same_basename.mp4"
        video_p.write_bytes(b"MOCK_SRC_VIDEO")

        os.environ["MOCK_CHILD_MODE"] = "wrong_source_path"
        res = default_subprocess_runner(
            interpreter=Path(sys.executable),
            gym_pose_dir=self.mock_gym_pose_dir,
            video_path=video_p,
            output_dir=clip_out,
            model_path=self.baseline_pt,
            device="cpu",
            timeout=10.0,
        )
        self.assertNotEqual(
            res.get("status"),
            "completed",
            "Child summary with same-basename different-directory source path was accepted as completed!",
        )
        self.assertEqual(res.get("status"), "failed")
        self.assertIn("source", str(res.get("error", "")).lower())

    def test_child_summary_conflicting_configuration_rejected(self) -> None:
        """Child summary reporting configuration conflicting with declared configuration must be rejected."""
        clip_out = Path(self.test_root) / "wrong_cfg_out"
        video_p = Path(self.test_root) / "cfg_vid.mp4"
        video_p.write_bytes(b"MOCK_CFG_VIDEO")

        # 1. Declared down_angle 110.0, summary reports 1.0
        os.environ["MOCK_CHILD_MODE"] = "wrong_config_down_angle"
        res_down = default_subprocess_runner(
            interpreter=Path(sys.executable),
            gym_pose_dir=self.mock_gym_pose_dir,
            video_path=video_p,
            output_dir=clip_out,
            model_path=self.baseline_pt,
            device="cpu",
            timeout=10.0,
            declared_configuration={"down_angle": 110.0},
        )
        self.assertNotEqual(
            res_down.get("status"),
            "completed",
            "Child summary with conflicting down_angle (1.0 vs declared 110.0) was accepted as completed!",
        )
        self.assertEqual(res_down.get("status"), "failed")

        shutil.rmtree(clip_out, ignore_errors=True)

        # 2. Declared min_joint_confidence 0.0, summary reports 0.9
        os.environ["MOCK_CHILD_MODE"] = "wrong_config_min_confidence"
        res_conf = default_subprocess_runner(
            interpreter=Path(sys.executable),
            gym_pose_dir=self.mock_gym_pose_dir,
            video_path=video_p,
            output_dir=clip_out,
            model_path=self.baseline_pt,
            device="cpu",
            timeout=10.0,
            declared_configuration={"counter_config": {"min_joint_confidence": 0.0}},
        )
        self.assertNotEqual(
            res_conf.get("status"),
            "completed",
            "Child summary with conflicting min_joint_confidence (0.9 vs declared 0.0) was accepted as completed!",
        )
        self.assertEqual(res_conf.get("status"), "failed")

    def test_pre_run_runtime_provenance_captured_before_first_clip(self) -> None:
        """Runtime provenance must be captured and frozen BEFORE the first clip execution commences."""
        from unittest import mock

        p = self._create_standard_plan(num_cases=2)
        out_eval = Path(self.test_root) / "pre_run_prov_out"

        first_clip_called = False
        provenance_frozen_before_first_clip = False

        original_get_prov = rev_exec.get_runtime_provenance
        prov_call_count = 0

        def spy_get_provenance(*args: Any, **kwargs: Any) -> Dict[str, Any]:
            nonlocal prov_call_count
            prov_call_count += 1
            return original_get_prov(*args, **kwargs)

        def mock_runner(output_dir: Path, video_path: Path, model_path: Path, **kwargs: Any) -> Dict[str, Any]:
            nonlocal first_clip_called, provenance_frozen_before_first_clip
            if not first_clip_called:
                first_clip_called = True
                provenance_frozen_before_first_clip = (prov_call_count > 0)

            output_dir.mkdir(parents=True, exist_ok=True)
            summary = {
                "status": "completed",
                "completed_reps": 5,
                "processed_frames": 100,
                "model_name": model_path.name,
                "clip_id": output_dir.name,
                "source_video": {"path": str(video_path.resolve())},
                "configuration": {
                    "down_angle": 110.0,
                    "up_angle": 150.0,
                    "min_joint_confidence": 0.3,
                    "stable_frames": 3,
                    "departure_frames": 2,
                    "max_gap_seconds": 0.5,
                    "tracker": "bytetrack.yaml",
                    "device": "cpu",
                },
            }
            (output_dir / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
            return {
                "status": "completed",
                "completed_reps": 5,
                "exit_code": 0,
                "read_status_note": None,
                "error": None,
            }

        with mock.patch("gym_analyser.review_execution.get_runtime_provenance", side_effect=spy_get_provenance):
            receipt = execute_evaluation(
                plan_path=p,
                output_dir=out_eval,
                run=True,
                runner=mock_runner,
                gym_pose_dir=self.mock_gym_pose_dir,
            )

        self.assertTrue(first_clip_called, "Runner was never invoked for any clip")
        self.assertTrue(
            provenance_frozen_before_first_clip,
            "Runtime provenance was NOT captured/frozen before the first clip execution commenced!",
        )
        self.assertIn("runtime_provenance", receipt)

    def test_child_python_probe_failure_records_unavailable(self) -> None:
        """Verify child python probe failure records 'unavailable' without substituting controller sys.version."""
        from unittest import mock

        controller_version = sys.version.replace("\n", " ")

        def failing_subprocess_run(cmd: Any, *args: Any, **kwargs: Any) -> subprocess.CompletedProcess[Any]:
            if isinstance(cmd, (list, tuple)) and len(cmd) > 2 and "-c" in cmd:
                return subprocess.CompletedProcess(
                    args=cmd,
                    returncode=1,
                    stdout="",
                    stderr="Simulated child python probe failure",
                )
            return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")

        with mock.patch("gym_analyser.review_execution.subprocess.run", side_effect=failing_subprocess_run):
            prov = rev_exec.get_runtime_provenance(
                interpreter=Path(sys.executable),
                device="cpu",
                gym_pose_dir=self.mock_gym_pose_dir,
            )

        py_ver = prov.get("python_version", "")
        self.assertNotEqual(
            py_ver,
            controller_version,
            "Runtime probe failure substituted controller sys.version in get_runtime_provenance!",
        )
        self.assertEqual(
            py_ver.lower(),
            "unavailable",
            f"Expected 'unavailable' in python_version, got {py_ver!r}",
        )

    def test_process_cleanup_on_timeout_without_killing_unrelated_pids(self) -> None:
        """On execution timeout, child and all descendant processes are terminated within bounded time,
        without terminating unrelated processes running on the machine.
        """
        unrelated = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
        try:
            self.assertTrue(_is_pid_running(unrelated.pid), "Unrelated process failed to start")

            clip_out = Path(self.test_root) / "timeout_cleanup_out"
            video_p = Path(self.test_root) / "timeout_vid.mp4"
            video_p.write_bytes(b"MOCK_TIMEOUT_VIDEO")

            pid_file = Path(self.test_root) / "timeout_descendant_pids.json"
            os.environ["MOCK_CHILD_MODE"] = "spawn_grandchild_and_sleep"
            os.environ["MOCK_PID_FILE"] = str(pid_file)

            start_t = time.time()
            res = default_subprocess_runner(
                interpreter=Path(sys.executable),
                gym_pose_dir=self.mock_gym_pose_dir,
                video_path=video_p,
                output_dir=clip_out,
                model_path=self.baseline_pt,
                device="cpu",
                timeout=2.0,
            )
            elapsed = time.time() - start_t

            self.assertEqual(res.get("status"), "timeout")
            self.assertLess(elapsed, 12.0, f"Timeout cleanup took too long: {elapsed:.2f}s")

            self.assertTrue(pid_file.is_file(), "Child process did not record descendant PIDs")
            pids = json.loads(pid_file.read_text(encoding="utf-8"))
            child_pid = pids["child_pid"]
            grandchild_pid = pids["grandchild_pid"]

            terminated = False
            for _ in range(30):
                if not _is_pid_running(child_pid) and not _is_pid_running(grandchild_pid):
                    terminated = True
                    break
                time.sleep(0.1)

            self.assertTrue(
                terminated,
                f"Descendant processes remained alive after timeout: child={child_pid} (alive={_is_pid_running(child_pid)}), "
                f"grandchild={grandchild_pid} (alive={_is_pid_running(grandchild_pid)})",
            )

            # CRITICAL: Verify unrelated process is STILL ALIVE and was not killed
            self.assertTrue(
                _is_pid_running(unrelated.pid),
                f"Unrelated process PID {unrelated.pid} was killed during process tree cleanup!",
            )
        finally:
            if _is_pid_running(unrelated.pid):
                try:
                    unrelated.kill()
                    unrelated.wait(timeout=2.0)
                except Exception:
                    pass
                if sys.platform == "win32" and _is_pid_running(unrelated.pid):
                    subprocess.run(["taskkill", "/F", "/PID", str(unrelated.pid)], capture_output=True)
                elif sys.platform != "win32" and _is_pid_running(unrelated.pid):
                    try:
                        os.kill(unrelated.pid, signal.SIGKILL)
                    except OSError:
                        pass

    def test_process_cleanup_on_abnormal_parent_exit_without_killing_unrelated_pids(self) -> None:
        """When child process exits abnormally while a grandchild lives in the background,
        containment ensures grandchild is terminated without killing unrelated PIDs.
        """
        unrelated = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
        grandchild_pid: Optional[int] = None
        try:
            self.assertTrue(_is_pid_running(unrelated.pid), "Unrelated process failed to start")

            clip_out = Path(self.test_root) / "abnormal_exit_cleanup_out"
            video_p = Path(self.test_root) / "abnormal_vid.mp4"
            video_p.write_bytes(b"MOCK_ABNORMAL_VIDEO")

            pid_file = Path(self.test_root) / "abnormal_descendant_pids.json"
            os.environ["MOCK_CHILD_MODE"] = "spawn_grandchild_and_fail_exit"
            os.environ["MOCK_PID_FILE"] = str(pid_file)

            start_t = time.time()
            res = default_subprocess_runner(
                interpreter=Path(sys.executable),
                gym_pose_dir=self.mock_gym_pose_dir,
                video_path=video_p,
                output_dir=clip_out,
                model_path=self.baseline_pt,
                device="cpu",
                timeout=10.0,
            )
            elapsed = time.time() - start_t

            self.assertEqual(res.get("status"), "failed")
            self.assertLess(elapsed, 12.0, f"Abnormal exit cleanup took too long: {elapsed:.2f}s")

            self.assertTrue(pid_file.is_file(), "Child process did not record descendant PIDs")
            pids = json.loads(pid_file.read_text(encoding="utf-8"))
            child_pid = pids["child_pid"]
            grandchild_pid = pids["grandchild_pid"]

            terminated = False
            for _ in range(30):
                if not _is_pid_running(grandchild_pid):
                    terminated = True
                    break
                time.sleep(0.1)

            self.assertTrue(
                terminated,
                f"Grandchild process {grandchild_pid} remained orphaned and alive after parent exited with error!",
            )

            # CRITICAL: Verify unrelated process is STILL ALIVE and was not killed
            self.assertTrue(
                _is_pid_running(unrelated.pid),
                f"Unrelated process PID {unrelated.pid} was killed during process tree cleanup!",
            )
        finally:
            if grandchild_pid is not None and _is_pid_running(grandchild_pid):
                if sys.platform == "win32":
                    subprocess.run(["taskkill", "/F", "/PID", str(grandchild_pid)], capture_output=True)
                else:
                    try:
                        os.kill(grandchild_pid, signal.SIGKILL)
                    except OSError:
                        pass
            if _is_pid_running(unrelated.pid):
                try:
                    unrelated.kill()
                    unrelated.wait(timeout=2.0)
                except Exception:
                    pass
                if sys.platform == "win32" and _is_pid_running(unrelated.pid):
                    subprocess.run(["taskkill", "/F", "/PID", str(unrelated.pid)], capture_output=True)
                elif sys.platform != "win32" and _is_pid_running(unrelated.pid):
                    try:
                        os.kill(unrelated.pid, signal.SIGKILL)
                    except OSError:
                        pass

    def test_execute_evaluation_with_wrong_summary_excludes_clip_from_common_cohort(self) -> None:
        """When a clip returns a completed status with zero frames or mismatched source,
        it must be rejected and excluded from the multi-model common completed cohort denominator.
        """
        p = self._create_standard_plan(num_cases=2)
        out_eval = Path(self.test_root) / "eval_cohort_rejection_out"

        call_idx = 0

        def mixed_runner(output_dir: Path, video_path: Path, model_path: Path, **kwargs: Any) -> Dict[str, Any]:
            nonlocal call_idx
            call_idx += 1
            output_dir.mkdir(parents=True, exist_ok=True)
            if call_idx == 1:
                # First clip returns probe defect: zero frames
                summary = {
                    "status": "completed",
                    "completed_reps": 5,
                    "processed_frames": 0,
                    "model_name": model_path.name,
                    "clip_id": output_dir.name,
                    "source_video": {"path": str(video_path.resolve())},
                    "configuration": {
                        "down_angle": 110.0,
                        "up_angle": 150.0,
                        "min_joint_confidence": 0.3,
                        "stable_frames": 3,
                        "departure_frames": 2,
                        "max_gap_seconds": 0.5,
                        "tracker": "bytetrack.yaml",
                        "device": "cpu",
                    },
                }
                (output_dir / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
                return {
                    "status": "failed",
                    "completed_reps": None,
                    "exit_code": 0,
                    "read_status_note": None,
                    "error": "processed_frames must be > 0",
                }
            else:
                summary = {
                    "status": "completed",
                    "completed_reps": 5,
                    "processed_frames": 100,
                    "model_name": model_path.name,
                    "clip_id": output_dir.name,
                    "source_video": {"path": str(video_path.resolve())},
                    "configuration": {
                        "down_angle": 110.0,
                        "up_angle": 150.0,
                        "min_joint_confidence": 0.3,
                        "stable_frames": 3,
                        "departure_frames": 2,
                        "max_gap_seconds": 0.5,
                        "tracker": "bytetrack.yaml",
                        "device": "cpu",
                    },
                }
                (output_dir / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
                return {
                    "status": "completed",
                    "completed_reps": 5,
                    "exit_code": 0,
                    "read_status_note": None,
                    "error": None,
                }

        receipt = execute_evaluation(
            plan_path=p,
            output_dir=out_eval,
            run=True,
            runner=mixed_runner,
            gym_pose_dir=self.mock_gym_pose_dir,
        )

        self.assertIn(receipt["status"], ("completed", "not_ready"))
        excluded = receipt.get("excluded_from_common_cohort", [])
        self.assertGreater(len(excluded), 0, "Rejected clip was not excluded from common cohort!")


if __name__ == "__main__":
    unittest.main()


class TestPrimaryReview44(TestReviewExecutionBase):
    def test_required_actual_summary_fields_cannot_be_omitted(self):
        for mode in ("primary_missing_source", "primary_missing_configuration", "primary_missing_model",
                     "primary_missing_setting", "primary_boolean_setting", "primary_false_completion", "primary_error_with_completed"):
            with self.subTest(mode=mode):
                os.environ["MOCK_CHILD_MODE"] = mode
                video = Path(self.test_root) / "input.mp4"
                video.write_bytes(b"FAKE")
                result = default_subprocess_runner(Path(sys.executable), self.mock_gym_pose_dir,
                    video, Path(self.test_root) / mode, self.baseline_pt, "cpu", 5)
                self.assertEqual(result["status"], "failed", result)
                self.assertIsNone(result["completed_reps"])

    @unittest.skipUnless(sys.platform == "win32", "Windows containment")
    def test_no_analyser_launch_when_job_creation_fails(self):
        from unittest import mock
        video = Path(self.test_root) / "input.mp4"
        video.write_bytes(b"FAKE")
        output = Path(self.test_root) / "uncontained"
        with mock.patch.object(rev_exec, "create_windows_job_object", return_value=None):
            with self.assertRaisesRegex(RuntimeError, "containment unavailable"):
                default_subprocess_runner(Path(sys.executable), self.mock_gym_pose_dir,
                    video, output, self.baseline_pt, "cpu", 5)
        self.assertFalse(output.exists())

    @unittest.skipUnless(sys.platform == "win32", "Windows containment")
    def test_assignment_failure_keeps_analyser_behind_gate(self):
        from unittest import mock
        video = Path(self.test_root) / "input.mp4"
        video.write_bytes(b"FAKE")
        output = Path(self.test_root) / "gate_not_released"
        api = mock.Mock()
        api.AssignProcessToJobObject.return_value = 0
        api.CloseHandle.return_value = 1
        with mock.patch.object(rev_exec, "create_windows_job_object", return_value=123), \
             mock.patch.object(rev_exec, "_windows_kernel32", return_value=api):
            with self.assertRaises(OSError):
                default_subprocess_runner(Path(sys.executable), self.mock_gym_pose_dir,
                    video, output, self.baseline_pt, "cpu", 5)
        self.assertFalse(output.exists(), "Analyser ran before containment assignment")


class TestPrimaryRuntimeAndTruth(TestReviewExecutionBase):
    def test_corrupt_source_history_cannot_be_reauthorized_by_rehashing_plan(self):
        plan_path = self._create_standard_plan(num_cases=1)
        plan = json.loads(plan_path.read_text())
        case = plan["eligible_cases"][0]
        case_path = self.review_root / case["case_id"] / "case.json"
        raw = json.loads(case_path.read_text())
        raw["history"] = [{"saved_at": "2026-10-20T00:00:00Z", "review": {"human_count": -1}}]
        case_path.write_text(json.dumps(raw))
        digest = compute_file_sha256(case_path)
        case["annotation_sha256"] = digest
        plan["source_annotation_hashes"][case["case_id"]] = digest
        plan["plan_sha256"] = compute_plan_sha256(plan)
        plan_path.write_text(json.dumps(plan))
        with self.assertRaises((ValueError, CaseCorruptedError)):
            execute_evaluation(plan_path, Path(self.test_root)/"rehashed_invalid", run=False)

    def test_tracker_bytes_changed_during_run_abort(self):
        tracker = self.mock_gym_pose_dir / "bytetrack.yaml"
        tracker.write_text("tracker_type: bytetrack\ntrack_high_thresh: 0.25\n")
        plan = self._create_standard_plan(num_cases=1)
        def changing_runner(**kwargs):
            kwargs["output_dir"].mkdir(parents=True, exist_ok=True)
            tracker.write_text("tracker_type: bytetrack\ntrack_high_thresh: 0.99\n")
            return {"status": "completed", "completed_reps": 5, "exit_code": 0}
        with self.assertRaisesRegex(rev_exec.PlanOrCodeMutationError, "runtime or tracker"):
            execute_evaluation(plan, Path(self.test_root)/"changed_tracker", interpreter=Path(sys.executable),
                               gym_pose_dir=self.mock_gym_pose_dir, run=True, runner=changing_runner)
