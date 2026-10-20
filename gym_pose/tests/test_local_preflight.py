"""Unit tests for local launch preflight validator.

Verifies:
- Standard library metadata-only discovery without ML imports or GPU initialization.
- Non-ML dummy module isolation proving marker file is NOT created.
- Timeout input validation rejecting non-finite, non-positive, or unbounded values.
- Strict remaining subprocess budget enforcement skipping expired probes.
- Unresolved path and symlink ancestor detection before resolve().
- Output report containment preventing writes to protected source/models/data/review/job paths.
- Atomic exclusive creation (O_CREAT | O_EXCL) and prohibition of --overwrite.
- Disjoint storage roots, model checkpoint verification, and FFmpeg degradation warning.
"""

from __future__ import annotations

import io
import json
import math
import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import MagicMock, patch

_GYM_POSE_DIR = Path(__file__).resolve().parent.parent
if str(_GYM_POSE_DIR) not in sys.path:
    sys.path.insert(0, str(_GYM_POSE_DIR))

from gym_analyser.local_preflight import (
    CANONICAL_MODEL_SHA256,
    DEFAULT_PROBE_TIMEOUT,
    DEFAULT_TOTAL_TIMEOUT,
    DISCLAIMER_RUNTIME_READINESS,
    STATUS_NOT_READY,
    STATUS_READY,
    STATUS_READY_WITH_WARNINGS,
    build_parser,
    check_model_checkpoint,
    check_storage_separation,
    check_symlinks,
    discover_environments,
    find_venv_python,
    format_text_report,
    is_subpath,
    is_symlink,
    main,
    probe_ffmpeg,
    probe_interpreter,
    resolve_inference_interpreter,
    resolve_web_interpreter,
    run_preflight,
    validate_output_path,
    validate_timeout,
    write_fresh_report,
)


class TestSymlinkAndHelpers(unittest.TestCase):
    """Test symlink and unresolved path safety helper functions."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def test_is_symlink_regular_file_and_dir(self):
        f = self.root / "test.txt"
        f.write_text("hello", encoding="utf-8")
        d = self.root / "subdir"
        d.mkdir()
        self.assertFalse(is_symlink(f))
        self.assertFalse(is_symlink(d))
        self.assertIsNone(check_symlinks(f, "Test file"))
        self.assertIsNone(check_symlinks(d, "Test dir"))

    def test_check_symlinks_unresolved_ancestors(self):
        regular_child = self.root / "deep" / "nested" / "file.txt"
        self.assertIsNone(check_symlinks(regular_child, "Child"))


class TestTimeoutAndBudgetValidation(unittest.TestCase):
    """Test rejection of invalid timeout inputs and strict subprocess budget enforcement."""

    def test_validate_timeout_rejects_non_finite_and_non_positive(self):
        with self.assertRaises(ValueError) as cm:
            validate_timeout(float("nan"), "timeout")
        self.assertIn("NaN", str(cm.exception))

        with self.assertRaises(ValueError) as cm:
            validate_timeout(float("inf"), "timeout")
        self.assertIn("infinity", str(cm.exception))

        with self.assertRaises(ValueError) as cm:
            validate_timeout(-1.0, "timeout")
        self.assertIn("positive", str(cm.exception))

        with self.assertRaises(ValueError) as cm:
            validate_timeout(0, "timeout")
        self.assertIn("positive", str(cm.exception))

        with self.assertRaises(ValueError) as cm:
            validate_timeout("invalid_string", "timeout")
        self.assertIn("valid float", str(cm.exception))

    def test_validate_timeout_rejects_unbounded(self):
        with self.assertRaises(ValueError) as cm:
            validate_timeout(9999.0, "timeout", max_val=300.0)
        self.assertIn("exceeds maximum", str(cm.exception))

    def test_validate_timeout_accepts_valid(self):
        self.assertEqual(validate_timeout(5.0, "timeout"), 5.0)
        self.assertEqual(validate_timeout("10.5", "timeout"), 10.5)

    def test_run_preflight_rejects_invalid_timeouts(self):
        with self.assertRaises(ValueError):
            run_preflight(probe_timeout=-5.0)
        with self.assertRaises(ValueError):
            run_preflight(total_timeout=float("nan"))
        with self.assertRaises(ValueError):
            run_preflight(total_timeout=99999.0)

    def test_remaining_budget_expiry_skips_probes(self):
        """Simulate monotonic clock advancing past total_timeout to verify strict skip with diagnostic."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_root = Path(tmp_dir)
            # Create minimal structure
            (tmp_root / "models").mkdir()
            (tmp_root / "models" / "yolo11n-pose.pt").write_bytes(b"dummy")

            # Mock monotonic time advancing past total_timeout (5.0s) immediately
            with patch("time.monotonic", side_effect=[0.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0]), \
                 patch("gym_analyser.local_preflight.probe_interpreter") as mock_probe, \
                 patch("gym_analyser.local_preflight.probe_ffmpeg") as mock_ff:

                report = run_preflight(gym_pose_dir=tmp_root, total_timeout=5.0)

                # Neither probe should be invoked because budget expired
                mock_probe.assert_not_called()
                mock_ff.assert_not_called()

                self.assertEqual(report["status"], STATUS_NOT_READY)
                self.assertTrue(report["environments"]["selected_web"].get("skipped"))
                self.assertTrue(report["environments"]["selected_inference"].get("skipped"))
                self.assertTrue(report["ffmpeg"].get("skipped"))
                errors_str = " ".join(report["summary"]["errors"])
                self.assertIn("total timeout budget (5.0s) expired", errors_str)


class TestEnvironmentDiscoveryAndPrecedence(unittest.TestCase):
    """Test environment discovery and interpreter selection precedence."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def _create_mock_venv(self, env_name: str) -> Path:
        env_dir = self.root / env_name
        scripts_dir = env_dir / ("Scripts" if sys.platform == "win32" else "bin")
        scripts_dir.mkdir(parents=True, exist_ok=True)
        py_name = "python.exe" if sys.platform == "win32" else "python"
        py_file = scripts_dir / py_name
        py_file.write_text("#!/usr/bin/env python\n", encoding="utf-8")
        return py_file

    def test_find_venv_python(self):
        env_dir = self.root / ".venv_test"
        self.assertIsNone(find_venv_python(env_dir))
        py_file = self._create_mock_venv(".venv_test")
        found = find_venv_python(env_dir)
        self.assertIsNotNone(found)
        self.assertEqual(found, py_file.resolve())

    def test_discover_environments(self):
        py_venv = self._create_mock_venv(".venv")
        py_cuda = self._create_mock_venv(".venv_cuda")
        disc = discover_environments(self.root)
        self.assertEqual(disc[".venv"], str(py_venv.resolve()))
        self.assertEqual(disc[".venv_cuda"], str(py_cuda.resolve()))
        self.assertIsNone(disc[".venv_web"])

    def test_web_interpreter_precedence(self):
        py, src = resolve_web_interpreter(self.root)
        self.assertEqual(src, "sys.executable")
        self.assertEqual(py, Path(sys.executable).resolve())

        web_py = self._create_mock_venv(".venv_web")
        py, src = resolve_web_interpreter(self.root)
        self.assertEqual(src, ".venv_web")
        self.assertEqual(py, web_py.resolve())

        custom_py = self.root / "custom_python.exe"
        custom_py.write_text("", encoding="utf-8")
        py, src = resolve_web_interpreter(self.root, override=str(custom_py))
        self.assertEqual(src, "override")
        self.assertEqual(py, custom_py.resolve())

    def test_inference_interpreter_precedence(self):
        py, src = resolve_inference_interpreter(self.root)
        self.assertEqual(src, "sys.executable")

        venv_py = self._create_mock_venv(".venv")
        py, src = resolve_inference_interpreter(self.root)
        self.assertEqual(src, ".venv")
        self.assertEqual(py, venv_py.resolve())

        cuda_py = self._create_mock_venv(".venv_cuda")
        py, src = resolve_inference_interpreter(self.root)
        self.assertEqual(src, ".venv_cuda")
        self.assertEqual(py, cuda_py.resolve())

        override_py = self.root / "my_py.exe"
        override_py.write_text("", encoding="utf-8")
        py, src = resolve_inference_interpreter(self.root, override=str(override_py))
        self.assertEqual(src, "override")
        self.assertEqual(py, override_py.resolve())


class TestSubprocessProbes(unittest.TestCase):
    """Test interpreter and module subprocess probe behaviors and failure modes."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.fake_py = self.root / "python.exe"
        self.fake_py.write_text("", encoding="utf-8")

    def test_probe_missing_interpreter(self):
        missing = self.root / "nonexistent" / "python.exe"
        res = probe_interpreter(missing, ["torch", "fastapi"])
        self.assertFalse(res["available"])
        self.assertIn("not found", res["error"])
        for m in ("torch", "fastapi"):
            self.assertFalse(res["modules"][m]["available"])

    def test_probe_interpreter_success(self):
        mock_output = json.dumps({
            "python_version": "3.10.20",
            "platform": "win32",
            "modules": {
                "fastapi": {"available": True, "version": "0.146.0", "distribution": "fastapi", "error": None},
                "uvicorn": {"available": True, "version": "0.54.0", "distribution": "uvicorn", "error": None},
            },
            "runtime_readiness_disclaimer": DISCLAIMER_RUNTIME_READINESS,
        })
        with patch("gym_analyser.local_preflight.subprocess.run") as mock_run:
            mock_run.return_value = subprocess.CompletedProcess(
                args=[], returncode=0, stdout=mock_output, stderr=""
            )
            res = probe_interpreter(self.fake_py, ["fastapi", "uvicorn"])
            self.assertTrue(res["available"])
            self.assertEqual(res["python_version"], "3.10.20")
            self.assertTrue(res["modules"]["fastapi"]["available"])
            self.assertEqual(res["modules"]["fastapi"]["version"], "0.146.0")
            self.assertEqual(res["runtime_readiness_disclaimer"], DISCLAIMER_RUNTIME_READINESS)

    def test_probe_interpreter_timeout(self):
        with patch("gym_analyser.local_preflight.subprocess.run") as mock_run:
            mock_run.side_effect = subprocess.TimeoutExpired(cmd=["python"], timeout=5.0)
            res = probe_interpreter(self.fake_py, ["torch"], timeout=5.0)
            self.assertFalse(res["available"])
            self.assertIn("timed out", res["error"])
            self.assertFalse(res["modules"]["torch"]["available"])

    def test_probe_interpreter_nonzero_exit(self):
        with patch("gym_analyser.local_preflight.subprocess.run") as mock_run:
            mock_run.return_value = subprocess.CompletedProcess(
                args=[], returncode=1, stdout="", stderr="Fatal error loading DLL"
            )
            res = probe_interpreter(self.fake_py, ["torch"])
            self.assertFalse(res["available"])
            self.assertIn("Fatal error loading DLL", res["error"])

    def test_probe_interpreter_malformed_json(self):
        with patch("gym_analyser.local_preflight.subprocess.run") as mock_run:
            mock_run.return_value = subprocess.CompletedProcess(
                args=[], returncode=0, stdout="Not a json output", stderr=""
            )
            res = probe_interpreter(self.fake_py, ["torch"])
            self.assertFalse(res["available"])
            self.assertIn("parse probe JSON", res["error"])

    def test_dummy_module_discovery_without_import(self):
        """CRITICAL: Test that a dummy ML module is discovered with metadata WITHOUT importing it.

        The dummy module writes a disk marker and raises an exception if imported.
        We verify that the marker is NOT created and no exception is raised, proving
        that __import__ is eliminated and discovery is pure metadata.
        """
        marker_file = self.root / "imported_marker.txt"
        mod_name = "test_dummy_ml_module"
        mod_file = self.root / f"{mod_name}.py"

        # Code that MUST NOT execute during preflight
        mod_code = (
            "from pathlib import Path\n"
            f"Path({repr(str(marker_file))}).write_text('FAILED_IMPORTED', encoding='utf-8')\n"
            "raise RuntimeError('FATAL: Dummy ML module was imported during preflight!')\n"
        )
        mod_file.write_text(mod_code, encoding="utf-8")

        # Create valid standard package distribution metadata
        dist_info = self.root / f"{mod_name}-3.14.15.dist-info"
        dist_info.mkdir()
        (dist_info / "METADATA").write_text(
            f"Metadata-Version: 2.1\nName: {mod_name}\nVersion: 3.14.15\n", encoding="utf-8"
        )
        (dist_info / "top_level.txt").write_text(f"{mod_name}\n", encoding="utf-8")

        # Run probe against real python interpreter with the dummy module on sys.path
        res = probe_interpreter(
            Path(sys.executable),
            [mod_name],
            timeout=5.0,
            extra_paths=[self.root],
        )

        self.assertTrue(res["available"], f"Probe failed: {res.get('error')}")
        self.assertIn(mod_name, res["modules"])
        m_info = res["modules"][mod_name]
        self.assertTrue(m_info["available"])
        self.assertEqual(m_info["version"], "3.14.15")
        self.assertEqual(m_info["distribution"], mod_name)
        self.assertIsNone(m_info["error"])

        # PROOF: Disk marker must NOT exist!
        self.assertFalse(
            marker_file.exists(),
            "VIOLATION: Dummy ML module was imported! Marker file exists on disk.",
        )
        # Disclaimer must be present
        self.assertIn("Discoverability and distribution metadata", res["runtime_readiness_disclaimer"])

    def test_dummy_module_without_metadata_handled_honestly(self):
        """A module file without distribution metadata is discovered via spec but has version=None."""
        marker_file = self.root / "no_meta_marker.txt"
        mod_name = "test_no_meta_module"
        mod_file = self.root / f"{mod_name}.py"
        mod_code = (
            "from pathlib import Path\n"
            f"Path({repr(str(marker_file))}).write_text('FAILED_IMPORTED', encoding='utf-8')\n"
            "raise RuntimeError('FATAL: Module without metadata was imported!')\n"
        )
        mod_file.write_text(mod_code, encoding="utf-8")

        res = probe_interpreter(
            Path(sys.executable),
            [mod_name],
            timeout=5.0,
            extra_paths=[self.root],
        )
        self.assertTrue(res["available"])
        m_info = res["modules"][mod_name]
        self.assertTrue(m_info["available"])
        self.assertIsNone(m_info["version"])
        self.assertIsNone(m_info["distribution"])
        self.assertFalse(marker_file.exists())

    def test_honest_distribution_mapping_cv2(self):
        """Verify that cv2 maps to opencv-python distribution without importing cv2."""
        res = probe_interpreter(
            Path(sys.executable),
            ["cv2"],
            timeout=5.0,
        )
        self.assertTrue(res["available"])
        m_info = res["modules"]["cv2"]
        # If cv2 is installed in test interpreter, verify distribution name is recorded honestly
        if m_info["available"]:
            self.assertIn("opencv", (m_info.get("distribution") or "").lower())


class TestModelCheckpointCheck(unittest.TestCase):
    """Test model checkpoint existence, size, and hash validation."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.model_file = self.root / "yolo11n-pose.pt"

    def test_missing_model_checkpoint(self):
        res = check_model_checkpoint(self.model_file)
        self.assertFalse(res["exists"])
        self.assertFalse(res["matches"])
        self.assertTrue(len(res["errors"]) > 0)
        self.assertIn("not found", res["errors"][0])

    def test_empty_model_checkpoint(self):
        self.model_file.write_bytes(b"")
        res = check_model_checkpoint(self.model_file)
        self.assertTrue(res["exists"])
        self.assertFalse(res["matches"])
        self.assertIn("empty", res["errors"][0])

    def test_hash_mismatch_model_checkpoint(self):
        self.model_file.write_bytes(b"corrupted model weights")
        res = check_model_checkpoint(self.model_file, expected_sha256=CANONICAL_MODEL_SHA256)
        self.assertTrue(res["exists"])
        self.assertFalse(res["matches"])
        self.assertIn("mismatch", res["errors"][0])
        self.assertNotEqual(res["computed_sha256"], CANONICAL_MODEL_SHA256)

    def test_valid_model_checkpoint(self):
        test_data = b"canonical_yolo11n_pose_bytes"
        import hashlib
        expected_hash = hashlib.sha256(test_data).hexdigest()
        self.model_file.write_bytes(test_data)
        res = check_model_checkpoint(self.model_file, expected_sha256=expected_hash)
        self.assertTrue(res["exists"])
        self.assertTrue(res["matches"])
        self.assertEqual(res["computed_sha256"], expected_hash)
        self.assertEqual(res["size_bytes"], len(test_data))
        self.assertEqual(res["errors"], [])


class TestFFmpegProbe(unittest.TestCase):
    """Test FFmpeg detection and graceful non-fatal warning on absence."""

    def test_ffmpeg_missing_is_warning(self):
        with patch("gym_analyser.local_preflight.shutil.which", return_value=None):
            res = probe_ffmpeg()
            self.assertFalse(res["available"])
            self.assertIsNone(res["version"])
            self.assertIsNotNone(res["warning"])
            self.assertIn("Browser video preview generation will be disabled", res["warning"])

    def test_ffmpeg_found_and_version_parsed(self):
        mock_output = (
            "ffmpeg version 9.0.2-full_build-www.gyan.dev Copyright (c) 2000-2026\n"
            "built with gcc 16.2.0\n"
        )
        with patch("gym_analyser.local_preflight.shutil.which", return_value="C:\\ffmpeg\\ffmpeg.exe"), \
             patch("gym_analyser.local_preflight.subprocess.run") as mock_run:
            mock_run.return_value = subprocess.CompletedProcess(
                args=[], returncode=0, stdout=mock_output, stderr=""
            )
            res = probe_ffmpeg()
            self.assertTrue(res["available"])
            self.assertEqual(res["version"], "9.0.2-full_build-www.gyan.dev")
            self.assertIsNone(res["warning"])

    def test_ffmpeg_timeout(self):
        with patch("gym_analyser.local_preflight.shutil.which", return_value="ffmpeg"), \
             patch("gym_analyser.local_preflight.subprocess.run") as mock_run:
            mock_run.side_effect = subprocess.TimeoutExpired(cmd=["ffmpeg"], timeout=5.0)
            res = probe_ffmpeg(timeout=5.0)
            self.assertFalse(res["available"])
            self.assertIn("timed out", res["warning"])


class TestStorageRootSeparation(unittest.TestCase):
    """Test storage root separation, nested directory detection, and protected path checks."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.gym_pose = self.root / "gym_pose"
        self.gym_pose.mkdir()
        (self.gym_pose / "gym_analyser").mkdir()
        (self.gym_pose / "models").mkdir()
        (self.gym_pose / "tests").mkdir()
        (self.gym_pose / "test_data").mkdir()

    def test_storage_separation_clean(self):
        web_jobs = self.gym_pose / "output" / "web_jobs"
        human_reviews = self.gym_pose / "output" / "human_reviews"
        res = check_storage_separation(self.gym_pose, web_jobs, human_reviews)
        self.assertTrue(res["disjoint"])
        self.assertEqual(res["errors"], [])

    def test_storage_separation_identical_collision(self):
        same_dir = self.gym_pose / "output" / "storage"
        res = check_storage_separation(self.gym_pose, same_dir, same_dir)
        self.assertFalse(res["disjoint"])
        self.assertTrue(any("same directory" in e for e in res["errors"]))

    def test_storage_separation_nested_collision(self):
        parent_dir = self.gym_pose / "output" / "shared"
        child_dir = parent_dir / "reviews"
        res = check_storage_separation(self.gym_pose, parent_dir, child_dir)
        self.assertFalse(res["disjoint"])
        self.assertTrue(any("nested inside" in e for e in res["errors"]))

    def test_storage_separation_collides_with_protected(self):
        web_jobs = self.gym_pose / "gym_analyser"
        human_reviews = self.gym_pose / "output" / "human_reviews"
        res = check_storage_separation(self.gym_pose, web_jobs, human_reviews)
        self.assertFalse(res["disjoint"])
        self.assertTrue(any("collides directly with protected directory" in e for e in res["errors"]))


class TestOutputReportSecurity(unittest.TestCase):
    """Test output report path containment, exclusive creation, and overwrite prevention."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.gym_pose = self.root / "gym_pose"
        self.gym_pose.mkdir()
        (self.gym_pose / "gym_analyser").mkdir()
        (self.gym_pose / "models").mkdir()
        (self.gym_pose / "tests").mkdir()
        (self.gym_pose / "test_data").mkdir()
        (self.gym_pose / "output" / "web_jobs").mkdir(parents=True)
        (self.gym_pose / "output" / "human_reviews").mkdir(parents=True)

    def test_validate_output_path_rejects_protected_models(self):
        bad_path = self.gym_pose / "models" / "report.json"
        p, err = validate_output_path(bad_path, gym_pose_dir=self.gym_pose)
        self.assertIsNone(p)
        self.assertIn("Models", err)

    def test_validate_output_path_rejects_protected_source(self):
        bad_path = self.gym_pose / "gym_analyser" / "report.json"
        p, err = validate_output_path(bad_path, gym_pose_dir=self.gym_pose)
        self.assertIsNone(p)
        self.assertIn("Source code", err)

    def test_validate_output_path_rejects_web_jobs_storage(self):
        bad_path = self.gym_pose / "output" / "web_jobs" / "report.json"
        p, err = validate_output_path(bad_path, gym_pose_dir=self.gym_pose)
        self.assertIsNone(p)
        self.assertIn("Web jobs storage", err)

    def test_validate_output_path_rejects_human_reviews_storage(self):
        bad_path = self.gym_pose / "output" / "human_reviews" / "report.json"
        p, err = validate_output_path(bad_path, gym_pose_dir=self.gym_pose)
        self.assertIsNone(p)
        self.assertIn("Human reviews storage", err)

    def test_validate_output_path_rejects_directory(self):
        bad_dir = self.root / "my_dir"
        bad_dir.mkdir()
        p, err = validate_output_path(bad_dir, gym_pose_dir=self.gym_pose)
        self.assertIsNone(p)
        self.assertIn("existing directory", err)

    def test_validate_output_path_accepts_clean_outside_destination(self):
        safe_path = self.root / "reports" / "fresh_report.json"
        p, err = validate_output_path(safe_path, gym_pose_dir=self.gym_pose)
        self.assertIsNone(err)
        self.assertEqual(p, safe_path.resolve())

    def test_write_fresh_report_exclusive_creation_and_no_clobber(self):
        report_file = self.root / "reports" / "first_run.json"
        write_fresh_report(report_file, '{"initial": 1}')
        self.assertTrue(report_file.exists())

        # Attempt to write again to same destination MUST raise FileExistsError
        with self.assertRaises(FileExistsError) as cm:
            write_fresh_report(report_file, '{"overwrite": 2}')
        self.assertIn("already exists", str(cm.exception))
        # Content unchanged
        self.assertEqual(report_file.read_text(encoding="utf-8"), '{"initial": 1}')


class TestPreflightOrchestrationAndCLI(unittest.TestCase):
    """Test end-to-end preflight orchestration, status outcomes, and CLI arguments."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

        # Build complete valid mock project layout
        self.models_dir = self.root / "models"
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.model_path = self.models_dir / "yolo11n-pose.pt"
        self.model_content = b"valid_model_data_for_test"
        import hashlib
        self.model_hash = hashlib.sha256(self.model_content).hexdigest()
        self.model_path.write_bytes(self.model_content)

        (self.root / "gym_analyser").mkdir(parents=True, exist_ok=True)
        (self.root / "tests").mkdir(parents=True, exist_ok=True)
        (self.root / "test_data").mkdir(parents=True, exist_ok=True)

        scripts = "Scripts" if sys.platform == "win32" else "bin"
        py_exe = "python.exe" if sys.platform == "win32" else "python"

        for venv in (".venv", ".venv_cuda", ".venv_web"):
            d = self.root / venv / scripts
            d.mkdir(parents=True, exist_ok=True)
            (d / py_exe).write_text("#!/usr/bin/env python\n", encoding="utf-8")

        self.fake_python = self.root / ".venv_web" / scripts / py_exe

    def _mock_success_probe(self, *args, **kwargs):
        modules_arg = args[1] if len(args) > 1 else kwargs.get("required_modules", [])
        return {
            "path": str(args[0]),
            "available": True,
            "python_version": "3.10.20",
            "platform": "win32",
            "modules": {
                m: {"available": True, "version": "1.0", "distribution": m, "error": None}
                for m in modules_arg
            },
            "error": None,
            "runtime_readiness_disclaimer": DISCLAIMER_RUNTIME_READINESS,
        }

    def test_run_preflight_all_ready(self):
        with patch("gym_analyser.local_preflight.probe_interpreter", side_effect=self._mock_success_probe), \
             patch("gym_analyser.local_preflight.CANONICAL_MODEL_SHA256", self.model_hash), \
             patch("gym_analyser.local_preflight.probe_ffmpeg", return_value={"available": True, "path": "ffmpeg", "version": "9.0", "warning": None}):

            report = run_preflight(gym_pose_dir=self.root)
            self.assertEqual(report["status"], STATUS_READY)
            self.assertTrue(report["summary"]["all_required_checks_passed"])
            self.assertEqual(report["summary"]["errors"], [])
            self.assertEqual(report["summary"]["warnings"], [])
            self.assertIn("runtime_readiness_disclaimer", report["summary"])

    def test_run_preflight_ready_with_warnings_ffmpeg_absent(self):
        with patch("gym_analyser.local_preflight.probe_interpreter", side_effect=self._mock_success_probe), \
             patch("gym_analyser.local_preflight.CANONICAL_MODEL_SHA256", self.model_hash), \
             patch("gym_analyser.local_preflight.probe_ffmpeg", return_value={"available": False, "path": None, "version": None, "warning": "No FFmpeg"}):

            report = run_preflight(gym_pose_dir=self.root)
            self.assertEqual(report["status"], STATUS_READY_WITH_WARNINGS)
            self.assertTrue(report["summary"]["all_required_checks_passed"])
            self.assertEqual(report["summary"]["errors"], [])
            self.assertEqual(report["summary"]["warnings"], ["No FFmpeg"])

    def test_run_preflight_not_ready_on_missing_module(self):
        def _mock_failing_inference(*args, **kwargs):
            return {
                "path": str(args[0]),
                "available": True,
                "python_version": "3.10.20",
                "platform": "win32",
                "modules": {
                    "torch": {"available": False, "version": None, "distribution": None, "error": "No module named torch"},
                    "ultralytics": {"available": True, "version": "8.0", "distribution": "ultralytics", "error": None},
                    "cv2": {"available": True, "version": "4.0", "distribution": "opencv-python", "error": None},
                },
                "error": None,
                "runtime_readiness_disclaimer": DISCLAIMER_RUNTIME_READINESS,
            }

        with patch("gym_analyser.local_preflight.probe_interpreter", side_effect=_mock_failing_inference), \
             patch("gym_analyser.local_preflight.CANONICAL_MODEL_SHA256", self.model_hash):

            report = run_preflight(gym_pose_dir=self.root)
            self.assertEqual(report["status"], STATUS_NOT_READY)
            self.assertFalse(report["summary"]["all_required_checks_passed"])
            self.assertTrue(any("torch" in e for e in report["summary"]["errors"]))

    def test_run_preflight_whitespace_paths(self):
        spaced_root = self.root / "gym pose workspace with spaces"
        spaced_root.mkdir()
        (spaced_root / "models").mkdir()
        spaced_model = spaced_root / "models" / "yolo11n-pose.pt"
        spaced_model.write_bytes(self.model_content)

        scripts = "Scripts" if sys.platform == "win32" else "bin"
        py_exe = "python.exe" if sys.platform == "win32" else "python"
        for venv in (".venv", ".venv_cuda", ".venv_web"):
            d = spaced_root / venv / scripts
            d.mkdir(parents=True, exist_ok=True)
            (d / py_exe).write_text("#!/usr/bin/env python\n", encoding="utf-8")

        with patch("gym_analyser.local_preflight.probe_interpreter", side_effect=self._mock_success_probe), \
             patch("gym_analyser.local_preflight.CANONICAL_MODEL_SHA256", self.model_hash):

            report = run_preflight(gym_pose_dir=spaced_root)
            self.assertIn("with spaces", report["gym_pose_dir"])
            self.assertEqual(report["status"], STATUS_READY)

    def test_cli_json_and_output_file(self):
        out_report = self.root / "fresh_report.json"
        cli_args = [
            "local_preflight.py",
            "--root", str(self.root),
            "--output", str(out_report),
            "--json",
            "--quiet",
        ]

        with patch.object(sys, "argv", cli_args), \
             patch("gym_analyser.local_preflight.probe_interpreter", side_effect=self._mock_success_probe), \
             patch("gym_analyser.local_preflight.CANONICAL_MODEL_SHA256", self.model_hash), \
             patch("sys.exit") as mock_exit:

            main()
            mock_exit.assert_called_with(0)
            self.assertTrue(out_report.exists())
            saved = json.loads(out_report.read_text(encoding="utf-8"))
            self.assertEqual(saved["status"], STATUS_READY)

    def test_cli_output_no_clobber_exits_2(self):
        out_report = self.root / "existing_report.json"
        out_report.write_text("{}", encoding="utf-8")

        cli_args = [
            "local_preflight.py",
            "--root", str(self.root),
            "--output", str(out_report),
            "--quiet",
        ]

        with patch.object(sys, "argv", cli_args), \
             patch("gym_analyser.local_preflight.probe_interpreter", side_effect=self._mock_success_probe), \
             patch("gym_analyser.local_preflight.CANONICAL_MODEL_SHA256", self.model_hash), \
             patch("sys.stderr", new_callable=io.StringIO) as mock_stderr, \
             patch("sys.exit") as mock_exit:

            main()
            mock_exit.assert_called_with(2)
            self.assertIn("Refusing to overwrite", mock_stderr.getvalue())

    def test_cli_rejects_overwrite_flag(self):
        cli_args = [
            "local_preflight.py",
            "--root", str(self.root),
            "--overwrite",
        ]
        with patch.object(sys, "argv", cli_args), \
             patch("sys.stderr", new_callable=io.StringIO) as mock_stderr, \
             patch("sys.exit") as mock_exit:

            main()
            mock_exit.assert_called_with(2)
            self.assertIn("--overwrite is prohibited", mock_stderr.getvalue())

    def test_cli_rejects_invalid_timeout(self):
        cli_args = [
            "local_preflight.py",
            "--root", str(self.root),
            "--probe-timeout", "-1.0",
        ]
        with patch.object(sys, "argv", cli_args), \
             patch("sys.stderr", new_callable=io.StringIO) as mock_stderr, \
             patch("sys.exit") as mock_exit:

            main()
            mock_exit.assert_called_with(2)
            self.assertIn("Invalid timeout parameter", mock_stderr.getvalue())

    def test_format_text_report(self):
        report = {
            "status": "ready",
            "timestamp_utc": "2026-10-20T12:00:00Z",
            "elapsed_seconds": 2.345,
            "gym_pose_dir": "C:/fake/gym_pose",
            "environments": {
                "discovered": {".venv": "C:/fake/.venv", ".venv_cuda": None, ".venv_web": "C:/fake/.venv_web"},
                "selected_web": {
                    "source": ".venv_web", "path": "C:/fake/py_web", "available": True,
                    "python_version": "3.10", "modules": {"fastapi": {"available": True, "version": "0.146.0"}},
                },
                "selected_inference": {
                    "source": ".venv", "path": "C:/fake/py_inf", "available": True,
                    "python_version": "3.10", "modules": {"torch": {"available": True, "version": "2.14"}},
                },
            },
            "checkpoint": {
                "path": "C:/fake/model.pt", "matches": True, "size_bytes": 1000,
                "canonical_sha256": "abc", "computed_sha256": "abc", "errors": [],
            },
            "ffmpeg": {"available": True, "path": "C:/ffmpeg/ffmpeg.exe", "version": "9.0", "warning": None},
            "storage": {
                "web_jobs_dir": "C:/fake/out/web", "human_reviews_dir": "C:/fake/out/rev",
                "disjoint": True, "errors": [],
            },
            "summary": {
                "all_required_checks_passed": True,
                "human_readiness": "Pending raw video reviews/labels; not a software launch blocker.",
                "errors": [], "warnings": [],
            },
        }
        text = format_text_report(report)
        self.assertIn("Overall Status: READY", text)
        self.assertIn("Canonical SHA-256: abc", text)
        self.assertIn("VERIFIED DISJOINT", text)
        self.assertIn("fastapi: OK (v0.146.0)", text)
        self.assertIn("Notice: Discoverability and version metadata indicate package presence only", text)
        self.assertIn("they do NOT prove runtime, ABI, or CUDA/GPU execution readiness", text)


class TestRealPreflightIntegration(unittest.TestCase):
    """Integration test running real preflight against repository structure."""

    def test_real_preflight_schema(self):
        report = run_preflight(total_timeout=25.0)
        self.assertIn("status", report)
        self.assertIn(report["status"], (STATUS_READY, STATUS_READY_WITH_WARNINGS, STATUS_NOT_READY))
        self.assertIn("environments", report)
        self.assertIn("checkpoint", report)
        self.assertIn("storage", report)
        self.assertIn("summary", report)
        self.assertIn("disclaimer", report)
        self.assertIn("runtime_readiness_disclaimer", report["summary"])
        self.assertTrue(report["checkpoint"]["exists"])
        self.assertEqual(report["checkpoint"]["canonical_sha256"], CANONICAL_MODEL_SHA256)
        self.assertTrue(report["storage"]["disjoint"])


class TestReview49MetadataRegression(unittest.TestCase):
    def test_stale_distribution_without_module_is_not_available(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            name="review49_ghost_package"
            dist=root/(name+"-1.0.dist-info");dist.mkdir()
            (dist/"METADATA").write_text("Metadata-Version: 2.1\nName: "+name+"\nVersion: 1.0\n")
            with patch.dict(os.environ,{"PYTHONPATH":str(root)}):
                report=probe_interpreter(Path(sys.executable),[name],timeout=5)
            item=report["modules"][name]
            self.assertFalse(item["available"])
            self.assertFalse(item["module_discoverable"])
            self.assertTrue(item["distribution_metadata_found"])
            self.assertEqual(item["version"],"1.0")


if __name__ == "__main__":
    unittest.main()
