"""Unit tests for pipeline export utilities and adapter behavior."""

import csv
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import torch

from gym_analyser.counter import CounterConfig, FrameResult
from gym_analyser.pipeline import (
    draw_knee_overlay,
    export_csv,
    export_summary,
    run_pipeline,
)


def create_test_video(
    path: Path,
    num_frames: int = 5,
    fps: float = 10.0,
    width: int = 64,
    height: int = 64,
) -> None:
    """Create a real minimal MP4 video file for pipeline testing."""
    import cv2

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (width, height))
    for _ in range(num_frames):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        writer.write(frame)
    writer.release()


def create_mock_pose_model(
    task: str = "pose",
    num_instances: int = 1,
    num_kpts: int = 17,
    num_channels: int = 3,
    match_ids: bool = True,
    exception_on_track: bool = False,
) -> MagicMock:
    """Create a mock YOLO model conforming to the Ultralytics API."""
    mock_model = MagicMock()
    mock_model.task = task

    if exception_on_track:
        mock_model.track.side_effect = RuntimeError("Simulated inference engine error")
        return mock_model

    mock_res = MagicMock()
    if num_instances == 0:
        mock_res.boxes = MagicMock()
        mock_res.boxes.id = None
        mock_res.keypoints = MagicMock()
        mock_res.keypoints.data = torch.zeros((0, num_kpts, num_channels))
    else:
        mock_res.boxes = MagicMock()
        if match_ids:
            mock_res.boxes.id = torch.tensor(list(range(1, num_instances + 1)))
        else:
            mock_res.boxes.id = torch.tensor([1])  # Mismatch if num_instances != 1

        kpt_array = np.zeros((num_instances, num_kpts, num_channels), dtype=np.float32)
        for i in range(num_instances):
            if num_kpts >= 17 and num_channels >= 3:
                # Hip, Knee, Ankle for left leg forming a standing angle
                kpt_array[i, 11] = [32.0, 16.0, 0.95]
                kpt_array[i, 13] = [32.0, 32.0, 0.95]
                kpt_array[i, 15] = [32.0, 48.0, 0.95]
        mock_res.keypoints = MagicMock()
        mock_res.keypoints.data = torch.from_numpy(kpt_array)

    mock_res.plot.return_value = np.zeros((64, 64, 3), dtype=np.uint8)
    mock_model.track.return_value = [mock_res]
    return mock_model


class TestPipeline(unittest.TestCase):
    """Test output formatting, CSV blanks, and adapter error handling."""

    def test_csv_formatting_and_missing_values_blank(self):
        """CSV must contain exactly one row per frame with blanks for missing numbers."""
        results = [
            FrameResult(
                frame_index=0,
                video_time_s=0.0,
                track_id=1,
                selected_side="left",
                knee_angle_deg=165.25,
                confidence=0.92,
                valid=True,
                reason="ok",
                stage="standing",
                completed_reps=0,
            ),
            FrameResult(
                frame_index=1,
                video_time_s=0.0333,
                track_id=1,
                selected_side="left",
                knee_angle_deg=None,
                confidence=None,
                valid=False,
                reason="low_joint_confidence",
                stage="standing",
                completed_reps=0,
            ),
            FrameResult(
                frame_index=2,
                video_time_s=0.0667,
                track_id=None,
                selected_side=None,
                knee_angle_deg=float("nan"),
                confidence=float("nan"),
                valid=False,
                reason="no_subject_detected",
                stage="standing",
                completed_reps=0,
            ),
        ]

        with tempfile.TemporaryDirectory() as tmp_dir:
            csv_path = Path(tmp_dir) / "angles.csv"
            export_csv(results, csv_path)

            self.assertTrue(csv_path.exists())
            with open(csv_path, mode="r", newline="", encoding="utf-8") as f:
                reader = list(csv.DictReader(f))

            self.assertEqual(len(reader), 3)

            # Row 0: valid numbers
            self.assertEqual(reader[0]["frame_index"], "0")
            self.assertEqual(reader[0]["video_time_s"], "0.0000")
            self.assertEqual(reader[0]["knee_angle_deg"], "165.25")
            self.assertEqual(reader[0]["confidence"], "0.9200")
            self.assertEqual(reader[0]["valid"], "True")

            # Row 1: missing values must be blank strings, NOT 0 or 0.0
            self.assertEqual(reader[1]["frame_index"], "1")
            self.assertEqual(reader[1]["video_time_s"], "0.0333")
            self.assertEqual(reader[1]["knee_angle_deg"], "")
            self.assertEqual(reader[1]["confidence"], "")
            self.assertEqual(reader[1]["valid"], "False")
            self.assertEqual(reader[1]["reason"], "low_joint_confidence")

            # Row 2: NaN coordinates must also serialize as blank
            self.assertEqual(reader[2]["knee_angle_deg"], "")
            self.assertEqual(reader[2]["confidence"], "")
            self.assertEqual(reader[2]["track_id"], "")

    def test_summary_json_structure(self):
        """Summary JSON must export proper metadata and configuration without fabricated metrics."""
        summary_data = {
            "status": "completed",
            "model_name": "yolo11n-pose.pt",
            "processed_frames": 100,
            "completed_reps": 2,
            "valid_observation_fraction": 0.85,
        }
        with tempfile.TemporaryDirectory() as tmp_dir:
            summary_path = Path(tmp_dir) / "summary.json"
            export_summary(summary_data, summary_path)

            with open(summary_path, mode="r", encoding="utf-8") as f:
                loaded = json.load(f)

            self.assertEqual(loaded["status"], "completed")
            self.assertEqual(loaded["completed_reps"], 2)
            self.assertEqual(loaded["processed_frames"], 100)

    def test_pipeline_validation_errors(self):
        """Pipeline must fail clearly on nonexistent files, same paths, or non-empty output dirs."""
        config = CounterConfig()

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            non_existent = tmp / "missing.mp4"
            out_dir = tmp / "out"

            # Nonexistent source file
            with self.assertRaises(FileNotFoundError):
                run_pipeline(non_existent, out_dir, config)

            # Create a real dummy file
            real_file = tmp / "video.mp4"
            real_file.write_bytes(b"dummy")

            # Output path inside source or equal
            with self.assertRaises(ValueError):
                run_pipeline(real_file, real_file, config)

            # Non-empty output dir without overwrite
            out_dir.mkdir()
            (out_dir / "unrelated.txt").write_text("content")
            with self.assertRaises(FileExistsError):
                run_pipeline(real_file, out_dir, config, overwrite=False)

    def test_invalid_fps_and_dimensions_fail(self):
        """Zero or negative FPS and invalid dimensions must raise ValueError."""
        config = CounterConfig()
        mock_cv2 = MagicMock()
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.side_effect = lambda prop: {
            3: 640,  # width
            4: 480,  # height
            5: 0.0,  # FPS <= 0
            7: 100,  # frame count
        }.get(prop, 0.0)
        mock_cv2.VideoCapture.return_value = mock_cap
        mock_cv2.CAP_PROP_FRAME_WIDTH = 3
        mock_cv2.CAP_PROP_FRAME_HEIGHT = 4
        mock_cv2.CAP_PROP_FPS = 5
        mock_cv2.CAP_PROP_FRAME_COUNT = 7

        with patch.dict(sys.modules, {"cv2": mock_cv2}):
            with tempfile.TemporaryDirectory() as tmp_dir:
                tmp = Path(tmp_dir)
                source = tmp / "source.mp4"
                source.write_bytes(b"data")
                out_dir = tmp / "out"

                with self.assertRaises(ValueError) as ctx:
                    run_pipeline(source, out_dir, config)
                self.assertIn("Invalid video FPS", str(ctx.exception))

    def test_normal_processing_loop_with_fake_video(self):
        """Pipeline completes end-to-end with correct frame counts, timestamps, CSV, and summary."""
        config = CounterConfig()
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            video_path = tmp / "input.mp4"
            out_dir = tmp / "output"
            create_test_video(video_path, num_frames=5, fps=10.0)

            mock_model = create_mock_pose_model(task="pose", num_instances=1)
            summary = run_pipeline(video_path, out_dir, config, model_name=mock_model)

            self.assertEqual(summary["status"], "completed")
            self.assertEqual(summary["processed_frames"], 5)
            self.assertEqual(summary["valid_observations"], 5)
            self.assertEqual(summary["valid_observation_fraction"], 1.0)
            self.assertEqual(summary["processed_duration_s"], 0.5)

            # Verify CSV output
            csv_path = out_dir / "angles.csv"
            self.assertTrue(csv_path.exists())
            with open(csv_path, mode="r", newline="", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(len(rows), 5)
            self.assertTrue(all(row["valid"] == "True" for row in rows))
            self.assertTrue(all(float(row["knee_angle_deg"]) == 180.0 for row in rows))
            self.assertEqual(rows[0]["frame_index"], "0")
            self.assertEqual(rows[0]["video_time_s"], "0.0000")
            self.assertEqual(rows[1]["frame_index"], "1")
            self.assertEqual(rows[1]["video_time_s"], "0.1000")
            self.assertEqual(rows[4]["video_time_s"], "0.4000")

            # Verify generated files
            self.assertTrue((out_dir / "annotated.mp4").exists())
            self.assertTrue((out_dir / "summary.json").exists())
            self.assertTrue((out_dir / "angle_plot.png").exists())

    def test_pipeline_rejects_non_pose_model(self):
        """Passing a non-pose model (e.g. task='detect') raises ValueError and records failed summary."""
        config = CounterConfig()
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            video_path = tmp / "input.mp4"
            out_dir = tmp / "output"
            create_test_video(video_path, num_frames=3)

            mock_model = create_mock_pose_model(task="detect")
            with self.assertRaises(ValueError) as ctx:
                run_pipeline(video_path, out_dir, config, model_name=mock_model)
            self.assertIn("task='pose'", str(ctx.exception))

            # Verify failed summary was recorded
            summary_path = out_dir / "summary.json"
            self.assertTrue(summary_path.exists())
            with open(summary_path, mode="r", encoding="utf-8") as f:
                saved = json.load(f)
            self.assertEqual(saved["status"], "failed")
            self.assertIn("Unsupported model task 'detect'", saved["error"])

    def test_pipeline_rejects_incompatible_keypoint_layout(self):
        """Keypoint tensor not matching 17 keypoints raises ValueError and marks run failed."""
        config = CounterConfig()
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            video_path = tmp / "input.mp4"
            out_dir = tmp / "output"
            create_test_video(video_path, num_frames=3)

            mock_model = create_mock_pose_model(num_kpts=14)
            with self.assertRaises(ValueError) as ctx:
                run_pipeline(video_path, out_dir, config, model_name=mock_model)
            self.assertIn("COCO-17", str(ctx.exception))

            with open(out_dir / "summary.json", mode="r", encoding="utf-8") as f:
                saved = json.load(f)
            self.assertEqual(saved["status"], "failed")

    def test_pipeline_rejects_missing_confidence_dimension(self):
        """Keypoint tensor with fewer than 3 channels raises ValueError."""
        config = CounterConfig()
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            video_path = tmp / "input.mp4"
            out_dir = tmp / "output"
            create_test_video(video_path, num_frames=3)

            mock_model = create_mock_pose_model(num_channels=2)
            with self.assertRaises(ValueError) as ctx:
                run_pipeline(video_path, out_dir, config, model_name=mock_model)
            self.assertIn("channels", str(ctx.exception))

            with open(out_dir / "summary.json", mode="r", encoding="utf-8") as f:
                saved = json.load(f)
            self.assertEqual(saved["status"], "failed")

    def test_pipeline_rejects_mismatched_boxes_and_keypoints(self):
        """Mismatch between keypoint row count and box ID count raises ValueError."""
        config = CounterConfig()
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            video_path = tmp / "input.mp4"
            out_dir = tmp / "output"
            create_test_video(video_path, num_frames=3)

            mock_model = create_mock_pose_model(num_instances=2, match_ids=False)
            with self.assertRaises(ValueError) as ctx:
                run_pipeline(video_path, out_dir, config, model_name=mock_model)
            self.assertIn("Mismatch between detected keypoint count", str(ctx.exception))

    def test_pipeline_handles_empty_detections_cleanly(self):
        """Empty detections do not cause errors; counter updates with empty list."""
        config = CounterConfig()
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            video_path = tmp / "input.mp4"
            out_dir = tmp / "output"
            create_test_video(video_path, num_frames=4)

            mock_model = create_mock_pose_model(num_instances=0)
            summary = run_pipeline(video_path, out_dir, config, model_name=mock_model)

            self.assertEqual(summary["status"], "completed")
            self.assertEqual(summary["processed_frames"], 4)
            self.assertEqual(summary["valid_observations"], 0)

    def test_pipeline_writer_opening_failure(self):
        """Failure to open VideoWriter raises IOError and writes failed summary."""
        config = CounterConfig()
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            video_path = tmp / "input.mp4"
            out_dir = tmp / "output"
            create_test_video(video_path, num_frames=3)

            import cv2

            orig_writer = cv2.VideoWriter

            class MockFailingWriter:
                def __init__(self, *args, **kwargs):
                    pass

                def isOpened(self):
                    return False

                def release(self):
                    pass

            with patch.object(cv2, "VideoWriter", MockFailingWriter):
                with self.assertRaises(IOError) as ctx:
                    run_pipeline(video_path, out_dir, config)
                self.assertIn("Failed to open VideoWriter", str(ctx.exception))

            with open(out_dir / "summary.json", mode="r", encoding="utf-8") as f:
                saved = json.load(f)
            self.assertEqual(saved["status"], "failed")

    def test_pipeline_inference_exception_cleanup(self):
        """Inference exceptions cleanly release resources, preserve partial CSV, and write failed summary."""
        config = CounterConfig()
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            video_path = tmp / "input.mp4"
            out_dir = tmp / "output"
            create_test_video(video_path, num_frames=5)

            mock_model = create_mock_pose_model(exception_on_track=True)
            with self.assertRaises(RuntimeError) as ctx:
                run_pipeline(video_path, out_dir, config, model_name=mock_model)
            self.assertIn("Simulated inference engine error", str(ctx.exception))

            with open(out_dir / "summary.json", mode="r", encoding="utf-8") as f:
                saved = json.load(f)
            self.assertEqual(saved["status"], "failed")
            self.assertIn("Simulated inference engine error", saved["error"])

    def test_pipeline_empty_video_fails(self):
        """A video with zero readable frames raises RuntimeError and marks run failed."""
        config = CounterConfig()
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            video_path = tmp / "input.mp4"
            out_dir = tmp / "output"
            create_test_video(video_path, num_frames=2)

            mock_model = create_mock_pose_model()

            import cv2
            orig_video_capture = cv2.VideoCapture

            class MockZeroFrameCap:
                def __init__(self, path):
                    self.real_cap = orig_video_capture(path)

                def isOpened(self):
                    return True

                def get(self, prop):
                    if prop == cv2.CAP_PROP_FRAME_COUNT:
                        return 0.0
                    return self.real_cap.get(prop)

                def read(self):
                    return False, None

                def release(self):
                    self.real_cap.release()

            with patch.object(cv2, "VideoCapture", MockZeroFrameCap):
                with self.assertRaises(RuntimeError) as ctx:
                    run_pipeline(video_path, out_dir, config, model_name=mock_model)
                self.assertIn("No readable video frames", str(ctx.exception))

            with open(out_dir / "summary.json", mode="r", encoding="utf-8") as f:
                saved = json.load(f)
            self.assertEqual(saved["status"], "failed")
            self.assertEqual(saved["processed_frames"], 0)

            # Also verify unopenable / empty file
            unreadable = tmp / "unreadable.mp4"
            unreadable.write_bytes(b"")
            with self.assertRaises(IOError):
                run_pipeline(unreadable, out_dir, config, model_name=mock_model, overwrite=True)

    def test_pipeline_early_read_termination_reports_incomplete(self):
        """When container declares more frames than decoded, status is incomplete and error raised."""
        config = CounterConfig()
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            video_path = tmp / "input.mp4"
            out_dir = tmp / "output"
            # Real video has 3 frames
            create_test_video(video_path, num_frames=3, fps=10.0)

            mock_model = create_mock_pose_model()

            import cv2

            orig_video_capture = cv2.VideoCapture

            class MockCapWithInflatedCount:
                def __init__(self, path):
                    self.real_cap = orig_video_capture(path)

                def isOpened(self):
                    return self.real_cap.isOpened()

                def get(self, prop):
                    if prop == cv2.CAP_PROP_FRAME_COUNT:
                        return 10.0  # Container declared 10 frames
                    return self.real_cap.get(prop)

                def read(self):
                    return self.real_cap.read()

                def release(self):
                    self.real_cap.release()

            with patch.object(cv2, "VideoCapture", MockCapWithInflatedCount):
                with self.assertRaises(RuntimeError) as ctx:
                    run_pipeline(video_path, out_dir, config, model_name=mock_model)
                self.assertIn("Processing ended early", str(ctx.exception))

            # Verify incomplete summary
            with open(out_dir / "summary.json", mode="r", encoding="utf-8") as f:
                saved = json.load(f)
            self.assertEqual(saved["status"], "incomplete")
            self.assertEqual(saved["processed_frames"], 3)
            self.assertEqual(saved["source_video"]["declared_frame_count"], 10)
            self.assertEqual(saved["processed_duration_s"], 0.3)
            self.assertEqual(saved["source_video"]["declared_duration_s"], 1.0)
            self.assertIn("Container declared 10 frames, but only 3 frames were successfully decoded", saved["read_status_note"])

            # Verify partial CSV has exactly 3 rows
            with open(out_dir / "angles.csv", mode="r", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(len(rows), 3)

    def test_pipeline_overwrite_clears_stale_artifacts(self):
        """Running with overwrite clears stale successful summary if subsequent run fails."""
        config = CounterConfig()
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            video_path = tmp / "input.mp4"
            out_dir = tmp / "output"
            out_dir.mkdir()
            create_test_video(video_path, num_frames=3)

            # Write a stale successful summary
            stale_summary = out_dir / "summary.json"
            stale_summary.write_text(json.dumps({"status": "completed", "completed_reps": 99}))

            # Run with a failing model and overwrite=True
            mock_model = create_mock_pose_model(task="detect")
            with self.assertRaises(ValueError):
                run_pipeline(video_path, out_dir, config, model_name=mock_model, overwrite=True)

            with open(stale_summary, mode="r", encoding="utf-8") as f:
                saved = json.load(f)
            self.assertEqual(saved["status"], "failed")
            self.assertNotEqual(saved.get("completed_reps"), 99)

    def test_draw_knee_overlay_rendering(self):
        """draw_knee_overlay modifies frame when given valid points and ignores invalid inputs."""
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        kpts = [(0.0, 0.0, 0.0)] * 17
        kpts[11] = (30.0, 20.0, 0.9)  # Hip
        kpts[13] = (30.0, 50.0, 0.9)  # Knee
        kpts[15] = (60.0, 50.0, 0.9)  # Ankle

        # Valid left knee
        draw_knee_overlay(frame, kpts, "left", 90.0)
        self.assertTrue(np.any(frame > 0))  # Overlay was drawn

        # Invalid side / missing angle / NaN
        empty_frame = np.zeros((100, 100, 3), dtype=np.uint8)
        draw_knee_overlay(empty_frame, kpts, "invalid_side", 90.0)
        self.assertTrue(np.all(empty_frame == 0))

        draw_knee_overlay(empty_frame, kpts, "left", None)
        self.assertTrue(np.all(empty_frame == 0))

        draw_knee_overlay(empty_frame, kpts, "left", float("nan"))
        self.assertTrue(np.all(empty_frame == 0))


    def test_float32_pose_sequence_counts_a_complete_rep(self):
        """Real tensor dtypes must survive the adapter and produce measurements/reps."""
        from tests.test_counter import make_pose
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            source = tmp / "input.mp4"
            create_test_video(source, num_frames=9, fps=30.0)
            model = create_mock_pose_model()
            sequence = []
            for angle in [170.0] * 3 + [90.0] * 3 + [170.0] * 3:
                result = create_mock_pose_model().track.return_value[0]
                result.keypoints.data = torch.tensor([make_pose(left_angle=angle)], dtype=torch.float32)
                sequence.append([result])
            model.track.side_effect = sequence
            summary = run_pipeline(source, tmp / "out", CounterConfig(), model_name=model)
            self.assertEqual(summary["valid_observations"], 9)
            self.assertEqual(summary["completed_reps"], 1)
            with (tmp / "out" / "angles.csv").open(encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual([row["stage"] for row in rows],
                             ["unknown", "unknown", "standing", "standing", "standing",
                              "lowered", "lowered", "lowered", "standing"])
            self.assertTrue(all(row["knee_angle_deg"] for row in rows))

    def test_plot_failure_during_incomplete_run_still_writes_failure_summary(self):
        """An export error must not suppress the only summary for an incomplete input."""
        import cv2
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            source = tmp / "input.mp4"
            create_test_video(source, num_frames=3)
            real_capture = cv2.VideoCapture(str(source))
            capture = MagicMock(wraps=real_capture)
            capture.get.side_effect = lambda prop: 10.0 if prop == cv2.CAP_PROP_FRAME_COUNT else real_capture.get(prop)
            with patch.object(cv2, "VideoCapture", return_value=capture), patch(
                "gym_analyser.pipeline.export_angle_plot", side_effect=RuntimeError("plot export failed")
            ):
                with self.assertRaisesRegex(RuntimeError, "plot export failed"):
                    run_pipeline(source, tmp / "out", CounterConfig(), model_name=create_mock_pose_model())
            saved = json.loads((tmp / "out" / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(saved["status"], "failed")
            self.assertIn("plot export failed", saved["error"])
            self.assertEqual(saved["processed_frames"], 3)
            capture.release.assert_called_once()


if __name__ == "__main__":
    unittest.main()
