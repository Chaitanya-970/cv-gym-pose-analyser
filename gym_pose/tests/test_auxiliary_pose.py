"""Landmark topology, unavailable-foot and lifecycle regressions without optional ML."""
import copy
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch
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

    def test_guarded_smoothing_requires_video_and_boolean_option(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'model.task';p.touch()
            for mode,value in [('image',True),('video',1),('video','yes')]:
                with self.assertRaises(ValueError):MediaPipeLandmarker(p,mode,smooth_video=value)

    def test_independent_guard_blocks_multiple_people_and_absence(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'model.task';p.touch()
            for count in [0,1,2]:
                backend=MediaPipeLandmarker(p,'video',smooth_video=True)
                backend._detector=Mock();backend._guard=Mock()
                backend._detector.detect_for_video.return_value=SimpleNamespace(pose_landmarks=[self.points()])
                backend._guard.detect.return_value=SimpleNamespace(pose_landmarks=[self.points() for _ in range(count)])
                backend._mp=SimpleNamespace(Image=lambda **kw:object(),ImageFormat=SimpleNamespace(SRGB='RGB'))
                result=backend.detect_rgb(SimpleNamespace(shape=(850,478,3),dtype='uint8'),100)
                self.assertEqual(result['guard_person_count'],count)
                self.assertEqual(result['person_count'],count)
                self.assertEqual(bool(result['foot_pair_available']),count==1)
                self.assertEqual(result['landmark_processing'],'guarded_single_person_video')
                backend._guard.detect.assert_called_once()
                backend.__exit__(None,None,None)

    def test_both_resources_close_even_if_tracker_close_raises(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'model.task';p.touch();backend=MediaPipeLandmarker(p,'video',smooth_video=True)
            tracker=Mock();guard=Mock();tracker.close.side_effect=RuntimeError('close failed')
            backend._detector=tracker;backend._guard=guard
            with self.assertRaises(RuntimeError):backend.__exit__(None,None,None)
            guard.close.assert_called_once();self.assertIsNone(backend._guard);self.assertIsNone(backend._detector)

    def test_guard_construction_failure_closes_tracker(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'model.task';p.touch();backend=MediaPipeLandmarker(p,'video',smooth_video=True)
            tracker=Mock();factory=Mock(side_effect=[tracker,RuntimeError('guard unavailable')])
            fake=SimpleNamespace(tasks=SimpleNamespace(BaseOptions=lambda **kw:kw,vision=SimpleNamespace(
                RunningMode=SimpleNamespace(IMAGE='image',VIDEO='video'),
                PoseLandmarkerOptions=lambda **kw:kw,
                PoseLandmarker=SimpleNamespace(create_from_options=factory))))
            with patch.dict('sys.modules',{'mediapipe':fake}):
                with self.assertRaises(RuntimeError):backend.__enter__()
            tracker.close.assert_called_once();self.assertIsNone(backend._detector)
            self.assertEqual(factory.call_args_list[0].args[0]['num_poses'],1)
            self.assertEqual(factory.call_args_list[1].args[0]['num_poses'],2)
            self.assertEqual(factory.call_args_list[1].args[0]['running_mode'],'image')

if __name__=='__main__':unittest.main()
