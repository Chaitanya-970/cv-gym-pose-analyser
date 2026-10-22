"""Adapter contracts, not anatomical accuracy tests."""
import unittest
from unittest.mock import Mock
from tempfile import TemporaryDirectory
from pathlib import Path
from gym_analyser.spine_pose import serialize_spine,body_proposal,SpineLandmarker


def guard():
    return dict(landmark_schema='mediapipe33',person_count=1,width=480,height=640,poses=[[dict(x_px=100+i*3,y_px=50+i*12,visibility=.9,presence=.9) for i in range(33)]])

class TestSpinePose(unittest.TestCase):
    def test_raw_scores_above_one_preserved(self):
        out=serialize_spine([[100.,100.]]*37,[2.4]*37,480,640)
        self.assertEqual(out[0]['raw_score'],2.4);self.assertTrue(out[0]['usable'])
    def test_offscreen_and_nonpositive_not_usable(self):
        pts=[[100.,100.] for _ in range(37)];pts[0]=[-3.,100.]
        scores=[1.]*37;scores[1]=0
        out=serialize_spine(pts,scores,480,640)
        self.assertEqual(out[0]['x_px'],-3.);self.assertFalse(out[0]['usable']);self.assertFalse(out[1]['usable'])
    def test_invalid_outputs_rejected(self):
        for pts,scores,w in [([[1,2]]*36,[1]*37,480),([[1,float('nan')]]*37,[1]*37,480),([[1,2]]*37,[True]*37,480),([[1,2]]*37,[1]*37,True)]:
            with self.assertRaises(ValueError):serialize_spine(pts,scores,w,640)
    def test_body_guard_rejections(self):
        self.assertIsNotNone(body_proposal(guard(),480,640))
        for field,value in [('person_count',2),('landmark_schema','unknown'),('poses',[])]:
            g=guard();g[field]=value;self.assertIsNone(body_proposal(g,480,640))
        g=guard()
        for p in g['poses'][0]:p['presence']=.1
        self.assertIsNone(body_proposal(g,480,640))
        with self.assertRaises(ValueError):body_proposal(guard(),240,640)
    def test_missing_model_fails_without_import(self):
        with TemporaryDirectory() as d:
            with self.assertRaises(FileNotFoundError):SpineLandmarker(Path(d)/'missing.onnx')
    def test_ambiguous_guard_does_not_infer_and_context_closes(self):
        with TemporaryDirectory() as d:
            p=Path(d)/'model.onnx';p.touch();adapter=SpineLandmarker(p);model=Mock();adapter._model=model
            image=Mock(shape=(640,480,3),dtype='uint8')
            out=adapter.predict_bgr(image,dict(person_count=2))
            self.assertEqual(out['reason'],'single_person_body_proposal_unavailable');model.assert_not_called()
            adapter.__exit__(None,None,None);model.close.assert_called_once();self.assertIsNone(adapter._model)
            with self.assertRaises(RuntimeError):adapter.predict_bgr(image,guard())
    def test_result_contract_and_original_coordinates(self):
        with TemporaryDirectory() as d:
            p=Path(d)/'m.onnx';p.touch();adapter=SpineLandmarker(p)
            adapter._model=Mock(return_value=([[[100.,120.]]*37],[[1.7]*37]))
            out=adapter.predict_bgr(Mock(shape=(640,480,3),dtype='uint8'),guard())
            self.assertEqual(out['accuracy'],'not_validated');self.assertEqual(out['score_basis'],'uncalibrated_simcc_peak')
            self.assertEqual(out['landmarks'][0]['y_px'],120.);self.assertEqual(len(out['landmarks']),37)

if __name__=='__main__':unittest.main()
