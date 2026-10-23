"""Whole-cycle correspondence and abstention; no models/downloads in these tests."""
import io
import importlib.util
import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from gym_analyser.pose_crosscheck import choose_run, run
from gym_analyser.web_jobs import JobManager


def record(intervals, frames=100, valid=100):
    return {"frames": frames, "valid_frames": valid,
            "cycles": [{"start_time_s": a, "end_time_s": b} for a, b in intervals]}


class CrosscheckTests(unittest.TestCase):
    def test_corroborates_cycle_timings_not_just_matching_counts(self):
        answer = choose_run({"baseline": record([(0, 1)]), "candidate": record([(3, 4)]),
                             "mediapipe": record([(6, 7)])})
        self.assertEqual(answer["status"], "needs_review")
        self.assertEqual(answer["selected"], "baseline")

    def test_other_estimator_can_corroborate_complete_candidate_run(self):
        full = [(1, 2), (3, 4), (5, 6)]
        answer = choose_run({"baseline": record(full[:1]), "candidate": record(full),
                             "mediapipe": record([(1.1, 2.1), (3.1, 4.1), (5.1, 6.1)])})
        self.assertEqual(answer["selected"], "candidate")
        self.assertEqual(answer["selected_count"], 3)
        self.assertEqual(answer["support"], ["candidate", "mediapipe"])
        self.assertFalse(answer["accuracy_validated"])

    def test_does_not_combine_unmatched_cycles_into_invented_count(self):
        answer = choose_run({"baseline": record([(1, 2)]), "candidate": record([(3, 4)]),
                             "mediapipe": record([(1, 2), (3, 4)])})
        self.assertEqual(answer["selected_count"], 1)
        self.assertEqual(answer["status"], "needs_review")

    def test_retains_original_when_runs_agree(self):
        answer = choose_run({"baseline": record([(1, 2)]), "candidate": record([(1.1, 2.1)])})
        self.assertEqual(answer["selected"], "baseline")
        self.assertEqual(answer["status"], "corroborated")

    def test_missing_pose_is_not_evidence_of_zero_repetitions(self):
        answer = choose_run({"baseline": record([], valid=0), "candidate": record([], valid=0)})
        self.assertEqual(answer["status"], "needs_review")

    def test_insufficient_coverage_cannot_cast_decisive_vote(self):
        answer = choose_run({"baseline": record([(1, 2)]), "candidate": record([(1, 2)], valid=20)})
        self.assertEqual(answer["status"], "needs_review")

    def test_mismatched_video_lengths_are_rejected(self):
        with self.assertRaises(ValueError):
            choose_run({"baseline": record([]), "candidate": record([], frames=99, valid=99)})

    def test_malformed_timing_and_accounting_are_rejected(self):
        for bad in (record([(float("nan"), 2)]), record([(2, 1)]),
                    record([(1, 3), (2, 4)]), record([], valid=101), record([], valid=True)):
            with self.subTest(run=bad), self.assertRaises(ValueError):
                choose_run({"baseline": bad, "candidate": record([])})

    def test_changed_asset_is_rejected_before_model_loading(self):
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory) / "asset.pt"
            p.write_bytes(b"not the reviewed model")
            with patch("gym_analyser.pose_crosscheck.collect") as collect:
                with self.assertRaises(ValueError):
                    run(p, p, p, p, Path(directory) / "out.json", "cpu", 10)
                collect.assert_not_called()


class CountingModeStorageTests(unittest.TestCase):
    def test_explicit_mode_is_stored_and_standard_remains_api_default(self):
        with tempfile.TemporaryDirectory() as directory:
            manager = JobManager(Path(directory))
            first, _, _ = manager.save_upload_file(io.BytesIO(b"video"), "first.mp4")
            second, _, _ = manager.save_upload_file(io.BytesIO(b"video"), "second.mp4", "cross_check")
            self.assertEqual(manager.get_metadata(first)["counting_mode"], "standard")
            self.assertEqual(manager.get_metadata(second)["counting_mode"], "cross_check")

    def test_unknown_mode_is_rejected_before_creating_job(self):
        with tempfile.TemporaryDirectory() as directory:
            manager = JobManager(Path(directory))
            with self.assertRaises(ValueError):
                manager.save_upload_file(io.BytesIO(b"video"), "first.mp4", "../../model.pt")
            self.assertEqual(list(Path(directory).iterdir()), [])


@unittest.skipUnless(importlib.util.find_spec("fastapi") is not None, "Optional web dependencies unavailable")
class CountingModeWebTests(unittest.TestCase):
    def test_missing_optional_assets_disable_the_ui_option(self):
        from fastapi.testclient import TestClient
        from gym_analyser.web import create_app
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manager = JobManager(root / "jobs", gym_pose_dir=root)
            client = TestClient(create_app(job_manager=manager, enable_review=False))
            html = client.get("/").text
            self.assertIn('id="counting-cross-check" type="checkbox" disabled', html)
            self.assertIn("Standard counting remains available", html)

    def test_upload_mode_reaches_metadata_and_unknown_modes_are_rejected(self):
        from fastapi.testclient import TestClient
        from gym_analyser.web import create_app
        with tempfile.TemporaryDirectory() as directory:
            manager = JobManager(Path(directory), runner=lambda **kw: {"status": "completed", "exit_code": 0})
            client = TestClient(create_app(job_manager=manager, enable_review=False))
            response = client.post("/api/jobs", files={"file": ("clip.mp4", b"video", "video/mp4")}, data={"counting_mode": "cross_check"})
            self.assertEqual(response.status_code, 202)
            self.assertEqual(manager.get_metadata(response.json()["job_id"])["counting_mode"], "cross_check")
            bad = client.post("/api/jobs", files={"file": ("clip.mp4", b"video", "video/mp4")}, data={"counting_mode": "unknown"})
            self.assertEqual(bad.status_code, 400)
