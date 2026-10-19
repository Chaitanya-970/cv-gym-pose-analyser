"""Integration tests for phase-localized movement feedback in pipeline execution.

Validates end-to-end integration of feedback_phases.json export within the
video processing pipeline:
- Normal execution exporting feedback_phases.json and summary.json integration
- Repetition count invariance (phase feedback does not mutate counter metrics)
- Incomplete and partial video execution handling and fail_summary output paths
- Overwrite behavior purging stale feedback_phases.json artifacts
- Path coherence ensuring all output_paths resolve strictly inside output_dir
- Resilience when one phase is unavailable without failing whole video
- Uncounted attempt suppression with transparent reason logging

No GPU, external model downloads, or network calls required.
Synthetic frames and mock YOLO pose model conform to Ultralytics API.
"""

import csv
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import MagicMock, patch

import numpy as np
import torch

from gym_analyser.counter import CounterConfig, FrameResult
from gym_analyser.pipeline import run_pipeline
from gym_analyser.phase_feedback import PhaseFeedbackConfig, build_phase_feedback


def create_test_video(
    path: Path,
    num_frames: int = 15,
    fps: float = 30.0,
    width: int = 64,
    height: int = 64,
) -> None:
    """Create a minimal real MP4 video file for pipeline execution testing."""
    import cv2

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (width, height))
    for _ in range(num_frames):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        writer.write(frame)
    writer.release()


def make_pose_with_posture(
    knee_angle: float = 170.0,
    torso_angle: float = 10.0,
    side: str = "left",
    confidence: float = 0.95,
    torso_missing: bool = False,
) -> List[Tuple[float, float, float]]:
    """Construct a mock COCO-17 keypoint list with given knee and torso angles."""
    kpts = [(0.0, 0.0, 0.0)] * 17
    hip_y = 50.0
    hip_x = 100.0 if side == "left" else 200.0
    knee_x = hip_x
    knee_y = 100.0

    k_rad = math.radians(knee_angle)
    ankle_x = knee_x + 50.0 * math.sin(k_rad)
    ankle_y = knee_y - 50.0 * math.cos(k_rad)

    t_rad = math.radians(torso_angle)
    shoulder_x = hip_x + 30.0 * math.sin(t_rad)
    shoulder_y = hip_y - 30.0 * math.cos(t_rad)

    t_conf = 0.0 if torso_missing else confidence

    if side == "left":
        kpts[5] = (shoulder_x, shoulder_y, t_conf)   # LEFT_SHOULDER
        kpts[11] = (hip_x, hip_y, confidence)        # LEFT_HIP
        kpts[13] = (knee_x, knee_y, confidence)      # LEFT_KNEE
        kpts[15] = (ankle_x, ankle_y, confidence)    # LEFT_ANKLE
    else:
        kpts[6] = (shoulder_x, shoulder_y, t_conf)   # RIGHT_SHOULDER
        kpts[12] = (hip_x, hip_y, confidence)        # RIGHT_HIP
        kpts[14] = (knee_x, knee_y, confidence)      # RIGHT_KNEE
        kpts[16] = (ankle_x, ankle_y, confidence)    # RIGHT_ANKLE

    return kpts


def create_mock_pose_model(
    task: str = "pose",
    frames_keypoints: Optional[List[List[Tuple[float, float, float]]]] = None,
    exception_on_track: bool = False,
) -> MagicMock:
    """Create a mock YOLO model conforming to the Ultralytics pose tracking API."""
    mock_model = MagicMock()
    mock_model.task = task

    if exception_on_track:
        mock_model.track.side_effect = RuntimeError("Simulated inference engine error")
        return mock_model

    if frames_keypoints is not None:
        sequence = []
        for kpts in frames_keypoints:
            res = MagicMock()
            res.boxes = MagicMock()
            res.boxes.id = torch.tensor([1])
            res.keypoints = MagicMock()
            res.keypoints.data = torch.tensor([kpts], dtype=torch.float32)
            res.plot.return_value = np.zeros((64, 64, 3), dtype=np.uint8)
            sequence.append([res])
        mock_model.track.side_effect = sequence
    else:
        mock_res = MagicMock()
        mock_res.boxes = MagicMock()
        mock_res.boxes.id = torch.tensor([1])
        default_kpts = make_pose_with_posture(knee_angle=170.0, torso_angle=10.0)
        mock_res.keypoints = MagicMock()
        mock_res.keypoints.data = torch.tensor([default_kpts], dtype=torch.float32)
        mock_res.plot.return_value = np.zeros((64, 64, 3), dtype=np.uint8)
        mock_model.track.return_value = [mock_res]

    return mock_model


def build_squat_sequence(
    num_frames: int = 15,
    min_knee: float = 90.0,
    side: str = "left",
    torso_missing_return: bool = False,
) -> List[List[Tuple[float, float, float]]]:
    """Generate a 15-frame squat keypoint sequence: standing -> descent -> turnaround -> return -> standing."""
    angles = [
        170.0, 170.0, 170.0,
        135.0, 110.0,
        min_knee, min_knee, 95.0,
        110.0, 135.0,
        170.0, 170.0, 170.0, 170.0, 170.0,
    ]
    torso_angles = [
        10.0, 10.0, 10.0,
        15.0, 20.0,
        25.0, 25.0, 23.0,
        20.0, 15.0,
        10.0, 10.0, 10.0, 10.0, 10.0,
    ]
    frames = []
    for i, (k_ang, t_ang) in enumerate(zip(angles[:num_frames], torso_angles[:num_frames])):
        is_missing_torso = torso_missing_return and (i >= 6)
        frames.append(
            make_pose_with_posture(
                knee_angle=k_ang,
                torso_angle=t_ang,
                side=side,
                torso_missing=is_missing_torso,
            )
        )
    return frames


def build_uncounted_shallow_sequence(
    num_frames: int = 15,
    shallow_knee: float = 125.0,
    side: str = "left",
) -> List[List[Tuple[float, float, float]]]:
    """Generate a 15-frame movement that never reaches the 110° depth threshold."""
    angles = [
        170.0, 170.0, 170.0,
        155.0, 140.0,
        shallow_knee, shallow_knee, shallow_knee,
        140.0, 155.0,
        170.0, 170.0, 170.0, 170.0, 170.0,
    ]
    torso_angles = [10.0] * 15
    return [
        make_pose_with_posture(knee_angle=k, torso_angle=t, side=side)
        for k, t in zip(angles[:num_frames], torso_angles[:num_frames])
    ]


class TestPhasePipelineIntegration(unittest.TestCase):
    """Integration test suite for phase feedback export within pipeline execution."""

    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.tmp = Path(self.temp_dir.name)
        self.video_path = self.tmp / "sample_squat.mp4"
        self.output_dir = self.tmp / "pipeline_output"
        self.config = CounterConfig(
            down_angle=110.0,
            up_angle=150.0,
            min_joint_confidence=0.3,
            stable_frames=3,
            max_gap_seconds=0.5,
        )

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_normal_run_exports_feedback_phases(self) -> None:
        """Normal execution exports feedback_phases.json, summary phase_feedback key, and preserves rep count."""
        create_test_video(self.video_path, num_frames=15, fps=30.0)
        frames = build_squat_sequence(num_frames=15, min_knee=90.0, side="left")
        mock_model = create_mock_pose_model(frames_keypoints=frames)

        summary = run_pipeline(self.video_path, self.output_dir, self.config, model_name=mock_model)

        # 1. Verify summary status and repetition count integrity
        self.assertEqual(summary["status"], "completed")
        self.assertEqual(summary["completed_reps"], 1)
        self.assertEqual(summary["processed_frames"], 15)
        self.assertEqual(summary["attempt_observations"]["counted_attempts"], 1)

        # 2. Verify feedback_phases.json exists and is valid JSON
        feedback_phases_path = self.output_dir / "feedback_phases.json"
        self.assertTrue(feedback_phases_path.is_file(), "feedback_phases.json must be exported in output_dir")

        with open(feedback_phases_path, "r", encoding="utf-8") as f:
            phase_data = json.load(f)

        self.assertIn("summary", phase_data)
        self.assertIn("attempts", phase_data)
        self.assertIn("configuration", phase_data)
        self.assertIn("biomechanical_limitations", phase_data)

        # 3. Verify phase feedback summary in summary.json matches exported payload
        self.assertIn("phase_feedback", summary, "summary.json must contain 'phase_feedback' summary")
        self.assertEqual(summary["phase_feedback"], phase_data["summary"])
        self.assertEqual(phase_data["summary"]["total_attempt_events"], 1)
        self.assertEqual(phase_data["summary"]["counted_events"], 1)
        self.assertEqual(phase_data["summary"]["phase_analyzed_events"], 1)

        # 4. Verify output_paths contract
        output_paths = summary.get("output_paths", {})
        self.assertIn("phase_feedback_json", output_paths, "output_paths must contain 'phase_feedback_json'")
        self.assertEqual(
            Path(output_paths["phase_feedback_json"]).resolve(),
            feedback_phases_path.resolve(),
        )

        # 5. Verify attempt phase localization details
        att = phase_data["attempts"][0]
        self.assertTrue(att["phase_analysis_available"])
        self.assertIsNone(att["suppression_reason"])
        self.assertIsNotNone(att["min_knee_point"])
        self.assertAlmostEqual(att["min_knee_point"]["knee_angle_deg"], 90.0, places=1)
        self.assertIsNotNone(att["descent_phase"])
        self.assertIsNotNone(att["return_phase"])
        self.assertGreaterEqual(att["descent_phase"]["sample_count"], 3)
        self.assertGreaterEqual(att["return_phase"]["sample_count"], 3)

    def test_repetition_counts_remain_unchanged_and_accurate(self) -> None:
        """Adding phase feedback must not alter repetition count or drop valid reps."""
        create_test_video(self.video_path, num_frames=15, fps=30.0)
        frames = build_squat_sequence(num_frames=15, min_knee=90.0, side="left")
        mock_model = create_mock_pose_model(frames_keypoints=frames)

        summary = run_pipeline(self.video_path, self.output_dir, self.config, model_name=mock_model)

        # Exactly 1 repetition was performed and counted
        self.assertEqual(summary["completed_reps"], 1)
        self.assertEqual(summary["attempt_observations"]["counted_attempts"], 1)

        # Verify feedback_phases.json also sees exactly 1 counted event
        phase_path = self.output_dir / "feedback_phases.json"
        with open(phase_path, "r", encoding="utf-8") as f:
            phase_data = json.load(f)
        self.assertEqual(phase_data["summary"]["counted_events"], 1)
        self.assertEqual(phase_data["summary"]["total_attempt_events"], 1)

    def test_incomplete_or_partial_video_exports_feedback_phases(self) -> None:
        """Incomplete or interrupted video run exports feedback_phases.json and includes path in fail_summary."""
        create_test_video(self.video_path, num_frames=8, fps=10.0)
        frames = build_squat_sequence(num_frames=8, min_knee=90.0, side="left")
        mock_model = create_mock_pose_model(frames_keypoints=frames)

        import cv2
        real_cap_cls = cv2.VideoCapture

        class MockEarlyEndCap:
            def __init__(self, path: Any):
                self.real = real_cap_cls(str(path))

            def isOpened(self) -> bool:
                return self.real.isOpened()

            def get(self, prop: int) -> float:
                if prop == cv2.CAP_PROP_FRAME_COUNT:
                    return 20.0  # Declared 20 frames, but only 8 exist
                return self.real.get(prop)

            def read(self) -> Tuple[bool, Any]:
                return self.real.read()

            def release(self) -> None:
                self.real.release()

        with patch.object(cv2, "VideoCapture", MockEarlyEndCap):
            with self.assertRaises(RuntimeError) as ctx:
                run_pipeline(self.video_path, self.output_dir, self.config, model_name=mock_model)
            self.assertIn("Processing ended early", str(ctx.exception))

        # Check summary.json recorded incomplete status and phase_feedback_json path
        summary_path = self.output_dir / "summary.json"
        self.assertTrue(summary_path.is_file())
        with open(summary_path, "r", encoding="utf-8") as sf:
            saved_summary = json.load(sf)

        self.assertEqual(saved_summary["status"], "incomplete")
        self.assertIn("phase_feedback_json", saved_summary["output_paths"])
        expected_phase_path = self.output_dir / "feedback_phases.json"
        self.assertEqual(
            Path(saved_summary["output_paths"]["phase_feedback_json"]).resolve(),
            expected_phase_path.resolve(),
        )

        # Check partial feedback_phases.json was exported
        self.assertTrue(expected_phase_path.is_file())
        with open(expected_phase_path, "r", encoding="utf-8") as pf:
            phase_payload = json.load(pf)
        self.assertIn("summary", phase_payload)
        self.assertIn("attempts", phase_payload)

    def test_overwrite_purges_stale_feedback_phases(self) -> None:
        """When overwrite=True is requested, pre-existing feedback_phases.json is purged and replaced."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        stale_phase_file = self.output_dir / "feedback_phases.json"
        stale_phase_file.write_text(json.dumps({"stale_run_marker": "legacy_data_123"}), encoding="utf-8")

        create_test_video(self.video_path, num_frames=15, fps=30.0)
        frames = build_squat_sequence(num_frames=15, min_knee=90.0, side="left")
        mock_model = create_mock_pose_model(frames_keypoints=frames)

        # Overwrite=False must fail because directory is not empty
        with self.assertRaises(FileExistsError):
            run_pipeline(self.video_path, self.output_dir, self.config, model_name=mock_model, overwrite=False)

        # Overwrite=True succeeds and purges stale file
        summary = run_pipeline(self.video_path, self.output_dir, self.config, model_name=mock_model, overwrite=True)
        self.assertEqual(summary["status"], "completed")

        with open(stale_phase_file, "r", encoding="utf-8") as pf:
            fresh_content = json.load(pf)

        self.assertNotIn("stale_run_marker", fresh_content)
        self.assertIn("summary", fresh_content)
        self.assertEqual(fresh_content["summary"]["counted_events"], 1)

    def test_coherent_output_paths_strictly_inside_output_dir(self) -> None:
        """All output_paths keys are valid, non-empty, and resolve strictly inside output_dir without traversal."""
        create_test_video(self.video_path, num_frames=15, fps=30.0)
        frames = build_squat_sequence(num_frames=15, min_knee=90.0, side="left")
        mock_model = create_mock_pose_model(frames_keypoints=frames)

        summary = run_pipeline(self.video_path, self.output_dir, self.config, model_name=mock_model)
        output_paths = summary["output_paths"]

        resolved_output_dir = self.output_dir.resolve()

        # Required artifact keys
        required_keys = [
            "annotated_video",
            "angles_csv",
            "summary_json",
            "angle_plot",
            "attempts_json",
            "posture_summary_json",
            "feedback_observations_json",
            "phase_feedback_json",
        ]
        for key in required_keys:
            self.assertIn(key, output_paths, f"Missing output path key: {key}")
            raw_path_str = output_paths[key]
            self.assertIsNotNone(raw_path_str, f"Output path for '{key}' must not be None")
            resolved_file = Path(raw_path_str).resolve()
            self.assertEqual(
                resolved_file.parent,
                resolved_output_dir,
                f"Artifact path for '{key}' ({resolved_file}) must reside directly inside output_dir ({resolved_output_dir})",
            )
            self.assertNotIn("..", raw_path_str, f"Traversal sequence detected in path for '{key}'")
            self.assertTrue(resolved_file.exists(), f"Exported artifact for '{key}' must exist on disk")

    def test_unavailable_phase_does_not_fail_video(self) -> None:
        """When return phase is unavailable (e.g. obscured torso), pipeline succeeds and suppresses phase analysis."""
        create_test_video(self.video_path, num_frames=15, fps=30.0)
        # Obscure torso during return phase (frames 6..14)
        frames = build_squat_sequence(num_frames=15, min_knee=90.0, side="left", torso_missing_return=True)
        mock_model = create_mock_pose_model(frames_keypoints=frames)

        summary = run_pipeline(self.video_path, self.output_dir, self.config, model_name=mock_model)

        # Video run completes normally without exception
        self.assertEqual(summary["status"], "completed")
        self.assertEqual(summary["completed_reps"], 1)

        # feedback_phases.json is exported with suppressed phase analysis
        phase_path = self.output_dir / "feedback_phases.json"
        self.assertTrue(phase_path.is_file())
        with open(phase_path, "r", encoding="utf-8") as f:
            phase_data = json.load(f)

        att = phase_data["attempts"][0]
        self.assertFalse(att["phase_analysis_available"])
        self.assertIn(att["suppression_reason"], ("insufficient_phase_samples", "insufficient_coverage"))
        self.assertEqual(phase_data["summary"]["phase_analyzed_events"], 0)
        self.assertEqual(phase_data["summary"]["counted_events"], 1)

    def test_uncounted_attempts_suppressed_with_reason(self) -> None:
        """Uncounted attempts (e.g. shallow knee bend) are suppressed with uncounted_attempt reason."""
        create_test_video(self.video_path, num_frames=15, fps=30.0)
        # Knee only reaches 125°, failing the 110° depth threshold
        frames = build_uncounted_shallow_sequence(num_frames=15, shallow_knee=125.0, side="left")
        mock_model = create_mock_pose_model(frames_keypoints=frames)

        summary = run_pipeline(self.video_path, self.output_dir, self.config, model_name=mock_model)

        # Video run completes normally with 0 completed repetitions
        self.assertEqual(summary["status"], "completed")
        self.assertEqual(summary["completed_reps"], 0)

        phase_path = self.output_dir / "feedback_phases.json"
        self.assertTrue(phase_path.is_file())
        with open(phase_path, "r", encoding="utf-8") as f:
            phase_data = json.load(f)

        self.assertGreaterEqual(phase_data["summary"]["total_attempt_events"], 1)
        self.assertEqual(phase_data["summary"]["counted_events"], 0)
        self.assertGreaterEqual(phase_data["summary"]["uncounted_events"], 1)

        uncounted_att = phase_data["attempts"][0]
        self.assertEqual(uncounted_att["status"], "uncounted")
        self.assertFalse(uncounted_att["phase_analysis_available"])
        self.assertEqual(uncounted_att["suppression_reason"], "uncounted_attempt")
        self.assertIsNotNone(uncounted_att["context_note"])
        self.assertIn("intent", uncounted_att["context_note"].lower())


if __name__ == "__main__":
    unittest.main()
