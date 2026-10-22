"""Projected knee geometry tests; synthetic fixtures do not validate human faults."""
import copy
import unittest
from gym_analyser.knee_tracking import assess_knee_tracking,KneeTrackingSettings
SHA='b'*64

def frame(t,left=0,right=0,mirror=False):
    points=[dict(x_px=100.,y_px=100.,visibility=.95,presence=.95) for _ in range(33)]
    for i,x,y in [(23,180,100),(24,300,100),(25,180+left,250),(26,300-right,250),(27,180,400),(28,300,400)]:
        points[i].update(x_px=480-x if mirror else x,y_px=y)
    return dict(timestamp_ms=t,width=480,height=640,landmark_schema='mediapipe33',person_count=1,poses=[points])

def calibration():return dict(source_sha256=SHA,front_view=True,stationary_camera=True,standing_reference_confirmed=True,human_evidence_id='synthetic_fixture_only',reference_timestamps_ms=[0,200,400])

class TestKneeTracking(unittest.TestCase):
    def run_case(self,frames=None,cal=None):
        return assess_knee_tracking(frames if frames is not None else [frame(t) for t in range(0,1600,200)],source_sha256=SHA,calibration=cal if cal is not None else calibration())
    def test_bilateral_inward_change_emits_separate_replay_flags(self):
        x=self.run_case([frame(t,40 if t>=600 else 0,40 if t>=600 else 0) for t in range(0,1600,200)])
        self.assertEqual(x['status'],'review_flags');self.assertEqual({e['side'] for e in x['events']},{'left','right'})
        self.assertEqual(x['events'][0]['start_s'],.6);self.assertEqual(x['accuracy'],'not_validated')
    def test_mirror_keeps_inward_sign(self):
        x=self.run_case([frame(t,40 if t>=600 else 0,0,True) for t in range(0,1600,200)])
        self.assertEqual([e['side'] for e in x['events']],['left'])
    def test_outward_motion_does_not_become_inward_fault(self):
        x=self.run_case([frame(t,-40 if t>=600 else 0) for t in range(0,1600,200)])
        self.assertEqual(x['events'],[]);self.assertEqual(x['status'],'no_review_flag')
    def test_side_view_or_unconfirmed_reference_fails_closed(self):
        for key in ['front_view','stationary_camera','standing_reference_confirmed']:
            for value in [None,False,1]:
                c=calibration();c[key]=value;self.assertEqual(self.run_case(cal=c)['status'],'unable_to_assess')
    def test_ankle_movement_is_unavailable_not_good_form(self):
        fs=[frame(t) for t in range(0,1600,200)]
        for f in fs[3:]:f['poses'][0][27]['x_px']+=50
        x=self.run_case(fs);self.assertEqual(x['measurable_samples'],3);self.assertEqual(x['samples'][-1]['reason'],'stance_or_camera_changed')
    def test_tracking_loss_and_time_gaps_do_not_bridge(self):
        fs=[frame(t,40 if t>=600 else 0) for t in [0,200,400,600,1000]]
        self.assertEqual(self.run_case(fs)['events'],[])
        fs=[frame(t,40 if t>=600 else 0) for t in [0,200,400,600,800,1000]];fs[4]['poses'][0][25]['presence']=0
        self.assertEqual(self.run_case(fs)['events'],[])
    def test_ambiguous_people_or_invalid_reference_do_not_produce_flags(self):
        fs=[frame(t) for t in range(0,1600,200)];fs[0]['person_count']=2;fs[0]['poses']*=2
        self.assertEqual(self.run_case(fs)['reason'],'reference_tracking_unavailable')
        c=calibration();c['source_sha256']='c'*64;self.assertEqual(self.run_case(cal=c)['reason'],'reference_source_mismatch')
    def test_unstable_reference_rejected(self):
        fs=[frame(t) for t in range(0,1600,200)];fs[1]['poses'][0][25]['x_px']+=30
        self.assertEqual(self.run_case(fs)['reason'],'reference_unstable')
    def test_turning_body_rejected(self):
        fs=[frame(t) for t in range(0,1600,200)]
        for f in fs[3:]:f['poses'][0][24]['x_px']=200
        x=self.run_case(fs);self.assertEqual(x['samples'][-1]['reason'],'view_or_body_orientation_changed')
    def test_malformed_numbers_dimensions_and_timestamps(self):
        with self.assertRaises(ValueError):self.run_case([frame(0),frame(0)])
        with self.assertRaises(ValueError):KneeTrackingSettings(inward_fraction=float('nan'))
        fs=[frame(t) for t in range(0,1600,200)];fs[0]['poses'][0][23]['x_px']=float('nan')
        self.assertEqual(self.run_case(fs)['reason'],'reference_tracking_unavailable')
        fs=[frame(t) for t in range(0,1600,200)];fs[-1]['width']=240
        self.assertEqual(self.run_case(fs)['reason'],'dimensions_changed')

if __name__=='__main__':unittest.main()
