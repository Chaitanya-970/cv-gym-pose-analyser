"""Unit tests for pipeline export utilities and adapter behavior."""

import csv
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from gym_analyser.counter import CounterConfig, FrameResult
from gym_analyser.pipeline import export_csv, export_summary, run_pipeline


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


if __name__ == "__main__":
    unittest.main()
