"""Meaningful pipeline checks for independent knee and posture availability."""
import csv
import json
import tempfile
import unittest
from pathlib import Path
import torch
from gym_analyser.counter import CounterConfig, FrameResult
from gym_analyser.pipeline import attach_posture, run_pipeline
from tests.test_pipeline import create_test_video, create_mock_pose_model

class TestPostureIntegration(unittest.TestCase):
    def test_real_video_exports_zero_torso_as_valid_and_hip_angle(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);video=root/"input.mp4";create_test_video(video)
            model=create_mock_pose_model()
            model.track.return_value[0].keypoints.data[0,5]=torch.tensor([32.,0.,.95])
            summary=run_pipeline(video,root/"out",CounterConfig(),model_name=model)
            self.assertEqual(summary["completed_reps"],0)
            feedback = json.loads((root/"out/feedback_observations.json").read_text())
            self.assertEqual(feedback["summary"]["counted_events"], summary["completed_reps"])
            self.assertIn("scope_limitations", feedback)
            self.assertEqual(summary["output_paths"]["feedback_observations_json"], str(root/"out/feedback_observations.json"))
            self.assertEqual(summary["posture_observations"]["torso_valid_frames"],5)
            with (root/"out/angles.csv").open() as f:rows=list(csv.DictReader(f))
            self.assertEqual(rows[0]["posture_side"],"left")
            self.assertEqual(rows[0]["torso_inclination_deg"],"0.00")
            self.assertEqual(rows[0]["hip_angle_deg"],"180.00")
            self.assertEqual(rows[0]["torso_reason"],"ok")

    def test_missing_knee_does_not_hide_visible_torso(self):
        keypoints=[[0.,0.,0.] for _ in range(17)]
        keypoints[5]=[5.,0.,.9];keypoints[11]=[5.,10.,.9]
        frame=FrameResult(0,0.,1,None,None,None,False,"both_sides_invalid", "unknown",0)
        result=attach_posture(frame,[{"track_id":1,"keypoints":keypoints}],CounterConfig())
        self.assertEqual(result.torso_inclination_deg,0.)
        self.assertIsNone(result.hip_angle_deg)
        self.assertEqual(result.completed_reps,0)

    def test_ambiguous_subjects_do_not_produce_posture(self):
        frame=FrameResult(0,0.,None,None,None,None,False,"multiple_subjects_ambiguous","unknown",0)
        result=attach_posture(frame,[{},{}],CounterConfig())
        self.assertIsNone(result.torso_inclination_deg)
        self.assertEqual(result.torso_reason,"multiple_subjects_ambiguous")
