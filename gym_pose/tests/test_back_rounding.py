"""Synthetic geometric invariants; no human fault accuracy implied."""
import math,unittest
from gym_analyser.back_rounding import assess_back_rounding,BackRoundingSettings
SHA='a'*64

def frame(t,bend=0,rotation=0,mirror=False,scale=1):
    points=[dict(id=i,x_px=240.,y_px=320.,raw_score=1.7) for i in range(37)]
    angle=math.radians(rotation)
    for i,x,y in [(26,0,50),(27,bend,0),(28,0,-50)]:
        xx=(x*math.cos(angle)-y*math.sin(angle))*scale
        yy=(x*math.sin(angle)+y*math.cos(angle))*scale
        points[i].update(x_px=240+(-xx if mirror else xx),y_px=320+yy)
    return dict(timestamp_ms=t,width=480,height=640,landmark_schema='spinepose37',person_count=1,landmarks=points)

def calibration(mirror=False):
    return dict(source_sha256=SHA,side_view=True,stationary_camera=True,standing_reference_confirmed=True,same_facing_direction=True,facing='right' if mirror else 'left',human_evidence_id='synthetic_only',reference_timestamps_ms=[0,200,400])

class TestBackRounding(unittest.TestCase):
    def run_case(self,fs=None,c=None):
        return assess_back_rounding(fs if fs is not None else [frame(t) for t in range(0,1600,200)],source_sha256=SHA,calibration=c or calibration())
    def test_sustained_posterior_change_flag_and_replay(self):
        out=self.run_case([frame(t,20 if t>=600 else 0) for t in range(0,1600,200)])
        self.assertEqual(out['status'],'review_flags');self.assertEqual(out['events'][0]['start_s'],.6)
        self.assertEqual(out['accuracy'],'not_validated');self.assertAlmostEqual(out['events'][0]['peak_posterior_change_fraction'],.2)
    def test_rigid_lean_does_not_become_rounding(self):
        out=self.run_case([frame(t,rotation=50 if t>=600 else 0) for t in range(0,1600,200)])
        self.assertEqual(out['events'],[]);self.assertEqual(out['measurable_samples'],8)
    def test_curvature_survives_rigid_lean(self):
        out=self.run_case([frame(t,20 if t>=600 else 0,rotation=40 if t>=600 else 0) for t in range(0,1600,200)])
        self.assertEqual(len(out['events']),1)
    def test_mirrored_view_retains_posterior_sign(self):
        out=self.run_case([frame(t,20 if t>=600 else 0,mirror=True) for t in range(0,1600,200)],calibration(True))
        self.assertEqual(len(out['events']),1)
    def test_scale_and_translation_not_curvature(self):
        out=self.run_case([frame(t,scale=1.2 if t>=600 else 1) for t in range(0,1600,200)])
        self.assertEqual(out['events'],[])
    def test_anterior_change_not_rounding(self):
        self.assertEqual(self.run_case([frame(t,-20 if t>=600 else 0) for t in range(0,1600,200)])['events'],[])
    def test_short_spike_and_gaps_do_not_join(self):
        self.assertEqual(self.run_case([frame(t,20 if t==600 else 0) for t in range(0,1600,200)])['events'],[])
        fs=[frame(t,20 if t>=600 else 0) for t in [0,200,400,600,1000,1400]]
        self.assertEqual(self.run_case(fs)['events'],[])
    def test_human_confirmations_and_source_required(self):
        for key in ['side_view','stationary_camera','standing_reference_confirmed','same_facing_direction']:
            c=calibration();c[key]=1;self.assertEqual(self.run_case(c=c)['status'],'unable_to_assess')
        c=calibration();c['source_sha256']='b'*64;self.assertEqual(self.run_case(c=c)['reason'],'reference_source_mismatch')
    def test_unstable_reference_rejected(self):
        fs=[frame(t,20 if t==200 else 0) for t in range(0,1600,200)]
        self.assertEqual(self.run_case(fs)['reason'],'reference_unstable')
    def test_missing_offscreen_or_nonfinite_spine_rejected(self):
        for key,value in [('x_px',-10),('y_px',float('nan')),('raw_score',0),('id',0)]:
            fs=[frame(t) for t in range(0,1600,200)];fs[0]['landmarks'][27][key]=value
            self.assertEqual(self.run_case(fs)['reason'],'reference_tracking_unavailable')
    def test_dimension_change_and_multi_person_fail_closed(self):
        fs=[frame(t) for t in range(0,1600,200)];fs[-1]['width']=960
        self.assertEqual(self.run_case(fs)['reason'],'dimensions_changed')
        fs=[frame(t) for t in range(0,1600,200)];fs[0]['person_count']=2
        self.assertEqual(self.run_case(fs)['reason'],'reference_tracking_unavailable')
    def test_invalid_time_and_setting_rejected(self):
        with self.assertRaises(ValueError):self.run_case([frame(0),frame(0)])
        with self.assertRaises(ValueError):BackRoundingSettings(posterior_change_fraction=float('nan'))

if __name__=='__main__':unittest.main()
