"""Landmark topology, unavailable-foot and lifecycle regressions without optional ML."""
import copy
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
from gym_analyser.auxiliary_pose import serialize_landmarks, visible_foot_pairs, MediaPipeLandmarker

class TestAuxiliaryPose(unittest.TestCase):
    def points(self):
        return [SimpleNamespace(x=.25,y=.5,visibility=.9,presence=.8) for _ in range(33)]
    def test_original_pixel_coordinates_not_square_model_coordinates(self):
        p=serialize_landmarks(self.points(),478,850)
        self.assertEqual(p[29]['x_px'],119.5);self.assertEqual(p[29]['y_px'],425)
        self.assertEqual(visible_foot_pairs(p,478,850),['left','right'])
    def test_toe_and_heel_not_ankle_visibility_controls_foot(self):
        p=serialize_landmarks(self.points(),478,850);p[27]['visibility']=0
        self.assertIn('left',visible_foot_pairs(p,478,850));p[31]['visibility']=0
        self.assertEqual(visible_foot_pairs(p,478,850),['right'])
    def test_unknown_presence_does_not_become_measured(self):
        p=serialize_landmarks(self.points(),478,850);p[29]['presence']=0
        self.assertNotIn('left',visible_foot_pairs(p,478,850))
    def test_out_of_frame_not_clamped_and_not_available(self):
        pts=self.points();pts[30].x=1.2;p=serialize_landmarks(pts,478,850)
        self.assertGreater(p[30]['x_px'],478);self.assertNotIn('right',visible_foot_pairs(p,478,850))
    def test_invalid_schema_numbers_and_thresholds_rejected(self):
        with self.assertRaises(ValueError):serialize_landmarks(self.points()[:17],478,850)
        for value in [float('nan'),float('inf'),True]:
            pts=self.points();pts[0].x=value
            with self.assertRaises(ValueError):serialize_landmarks(pts,478,850)
        for dim in [0,-1,True,478.5]:
            with self.assertRaises(ValueError):serialize_landmarks(self.points(),dim,850)
        with self.assertRaises(ValueError):visible_foot_pairs(serialize_landmarks(self.points(),478,850),478,850,min_visibility=float('nan'))
    def test_missing_model_and_bad_mode_fail_without_download(self):
        with self.assertRaises(FileNotFoundError):MediaPipeLandmarker('definitely_missing_auxiliary_model.task')
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'model.task';p.touch()
            with self.assertRaises(ValueError):MediaPipeLandmarker(p,mode='live')
    def test_context_closes_on_error_and_resets(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'model.task';p.touch();backend=MediaPipeLandmarker(p);fake=Mock();backend._detector=fake
            backend.__exit__(ValueError,ValueError('test'),None)
            fake.close.assert_called_once();self.assertIsNone(backend._detector)
            with self.assertRaises(RuntimeError):backend.detect_rgb(None)
    def test_video_time_validation_precedes_model_call(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'model.task';p.touch();backend=MediaPipeLandmarker(p,mode='video');backend._detector=Mock();backend._last_timestamp=100
            frame=SimpleNamespace(shape=(850,478,3),dtype='uint8')
            for timestamp in [None,True,-1,99,100,100.5]:
                with self.assertRaises(ValueError):backend.detect_rgb(frame,timestamp)
            backend._detector.detect_for_video.assert_not_called()

if __name__=='__main__':unittest.main()
