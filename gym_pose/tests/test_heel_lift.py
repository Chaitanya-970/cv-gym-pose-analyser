"""Synthetic geometry and failure-state tests; not human fault accuracy."""
import copy
import unittest
from gym_analyser.heel_lift import assess_heel_lift, HeelLiftSettings

SHA='a'*64

def frame(t, rise=0, mirror=False):
    pts=[dict(x_px=100.,y_px=100.,visibility=.95,presence=.95) for _ in range(33)]
    pts[29].update(x_px=200. if mirror else 100.,y_px=200.-rise)
    pts[31].update(x_px=100. if mirror else 200.,y_px=200.)
    return dict(timestamp_ms=t,width=640,height=480,landmark_schema='mediapipe33',person_count=1,poses=[pts])

def calibration():
    return dict(source_sha256=SHA,side='left',reference_timestamps_ms=[0,200,400],suitable_side_view=True,stationary_camera=True,planted_reference_confirmed=True,human_evidence_id='synthetic_fixture_only')

class TestHeelLift(unittest.TestCase):
    def assess(self,fs=None,c=None):
        return assess_heel_lift(fs if fs is not None else [frame(t) for t in range(0,1800,200)],source_sha256=SHA,calibration=c if c is not None else calibration())
    def test_sustained_rise_has_replay_evidence(self):
        fs=[frame(t,20 if 600<=t<=1200 else 0) for t in range(0,1800,200)]
        x=self.assess(fs);self.assertEqual(x['status'],'review_flags');e=x['events'][0]
        self.assertEqual((e['start_s'],e['end_s']),(.6,1.2));self.assertAlmostEqual(e['peak_rise_fraction'],.2);self.assertEqual(x['accuracy'],'not_validated')
    def test_flat_is_no_flag_not_proven_good_form(self):
        x=self.assess();self.assertEqual(x['status'],'no_review_flag');self.assertIn('does not prove',x['interpretation'])
    def test_single_spike_not_sustained_event(self):
        fs=[frame(t,20 if t==800 else 0) for t in range(0,1800,200)]
        self.assertEqual(self.assess(fs)['events'],[])
    def test_mirrored_side_view_still_detects_upward_rise(self):
        fs=[frame(t,20 if t>=600 else 0,True) for t in range(0,1400,200)]
        self.assertEqual(self.assess(fs)['status'],'review_flags')
    def test_all_required_human_confirmations_enforced(self):
        for key in ['suitable_side_view','stationary_camera','planted_reference_confirmed']:
            for value in [False,None,1,'yes']:
                c=calibration();c[key]=value;self.assertEqual(self.assess(c=c)['status'],'unable_to_assess')
        c=calibration();c['human_evidence_id']='';self.assertEqual(self.assess(c=c)['reason'],'missing_human_reference_evidence')
    def test_wrong_source_or_missing_reference_fails_closed(self):
        c=calibration();c['source_sha256']='b'*64;self.assertEqual(self.assess(c=c)['reason'],'reference_source_mismatch')
        c=calibration();c['reference_timestamps_ms']=[0,200,999];self.assertEqual(self.assess(c=c)['reason'],'reference_sample_missing')
        c=calibration();c['reference_timestamps_ms']=[0,0,200];self.assertEqual(self.assess(c=c)['reason'],'need_three_distinct_reference_samples')
    def test_no_person_multiple_people_low_confidence_break_events(self):
        for mode in ['no_person','multiple','confidence']:
            fs=[frame(t,20 if t>=600 else 0) for t in range(0,1200,200)]
            if mode=='no_person':fs[4]['person_count']=0;fs[4]['poses']=[]
            elif mode=='multiple':fs[4]['person_count']=2;fs[4]['poses']*=2
            else:fs[4]['poses'][0][29]['presence']=0
            self.assertEqual(self.assess(fs)['events'],[])
    def test_missing_time_gap_does_not_bridge(self):
        fs=[frame(t,20 if t>=600 else 0) for t in [0,200,400,600,1000]]
        self.assertEqual(self.assess(fs)['events'],[])
    def test_toe_translation_is_unavailable(self):
        fs=[frame(t) for t in range(0,1200,200)]
        for f in fs[3:]:
            f['poses'][0][29]['x_px']+=50;f['poses'][0][31]['x_px']+=50
        x=self.assess(fs);self.assertEqual(x['measurable_samples'],3);self.assertTrue(all(s['reason']=='toe_moved_or_camera_changed' for s in x['samples'][3:]))
    def test_turnaround_is_unavailable(self):
        fs=[frame(t,20 if t>=600 else 0, t>=600) for t in range(0,1200,200)]
        x=self.assess(fs);self.assertEqual(x['events'],[]);self.assertEqual(x['measurable_samples'],3)
    def test_uneven_reference_is_not_trusted(self):
        fs=[frame(t) for t in range(0,1200,200)];fs[1]['poses'][0][29]['y_px']-=15
        self.assertEqual(self.assess(fs)['reason'],'reference_unstable')
    def test_dimension_change_invalidates_calibration(self):
        fs=[frame(t) for t in range(0,1200,200)];fs[-1]['width']=320
        self.assertEqual(self.assess(fs)['reason'],'dimensions_changed')
    def test_time_order_and_settings_invalid_values(self):
        for ts in [[0,0],[200,0],[True,200],[-1,200]]:
            with self.assertRaises(ValueError):self.assess([frame(t) for t in ts])
        for v in [0,-1,True,float('nan'),float('inf')]:
            with self.assertRaises(ValueError):HeelLiftSettings(rise_fraction=v)
    def test_ground_slope_reference_uses_normal_not_raw_vertical_gap(self):
        fs=[frame(t) for t in range(0,1400,200)]
        for f in fs:
            f['poses'][0][29]['y_px']=210
            if f['timestamp_ms']>=600:f['poses'][0][29]['y_px']-=20
        self.assertEqual(self.assess(fs)['status'],'review_flags')

if __name__=='__main__':unittest.main()
