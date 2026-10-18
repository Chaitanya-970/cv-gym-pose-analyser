"""Independent temporal regression tests for the attempt observer and real counter."""
import math
import unittest
from gym_analyser.counter import CounterConfig, SquatCounter
from gym_analyser.attempts import AttemptTracker


def points(angle, side='left', confidence=.95):
    k=[[0.,0.,0.] for _ in range(17)]
    hip,knee,ankle=(11,13,15) if side=='left' else (12,14,16)
    k[hip]=[0.,-50.,confidence];k[knee]=[0.,0.,confidence]
    k[ankle]=[50*math.sin(math.radians(angle)),-50*math.cos(math.radians(angle)),confidence]
    return k


def replay(angles, config=None):
    config=config or CounterConfig()
    counter=SquatCounter(config);observer=AttemptTracker(config)
    for i,angle in enumerate(angles):
        det=[] if angle is None else [{'track_id':1,'keypoints':points(angle)}]
        r=counter.update(i,round(i*.1,4),det)
        observer.observe(r,counter.armed_side)
    return counter,observer,observer.finish()


class TestAttemptTracker(unittest.TestCase):
    def test_counted_cycle_and_neutral_explanation(self):
        c,o,r=replay([165]*3+[145,130,100,95,90,120]+[165]*3)
        self.assertEqual((c.completed_reps,r['counted_attempts'],r['uncounted_attempts']),(1,1,0))
        event=r['events'][0]
        self.assertEqual(event['start_frame'],3)
        self.assertEqual(event['confirmed_at_frame'],4)
        self.assertEqual(event['reason'],'completed_cycle')
        self.assertTrue(event['depth_confirmed'])
        self.assertEqual(o.finish(),r)

    def test_shallow_return_requires_stability(self):
        c,o,r=replay([165]*3+[145,130,125]+[165]*3)
        self.assertEqual(c.completed_reps,0)
        self.assertEqual(r['events'][0]['reason'],'depth_threshold_not_reached')
        self.assertEqual(r['events'][0]['end_frame'],8)

    def test_depth_crossing_without_stability_is_distinct(self):
        _,_,r=replay([165]*3+[145,130,100,95]+[165]*3)
        self.assertEqual(r['events'][0]['reason'],'depth_not_sustained')

    def test_noise_and_invalid_frames_do_not_accumulate_departure(self):
        _,_,r=replay([165]*3+[140,165,140,None,140,165,165,165])
        self.assertEqual(r['events'],[])

    def test_opening_crouch_is_telemetry_not_attempt(self):
        _,_,r=replay([90]*3+[165]*3)
        self.assertEqual(r['events'],[])
        self.assertEqual(r['pre_arming']['opening_lowered_frames'],3)

    def test_eof_preserves_unresolved_return(self):
        _,_,r=replay([165]*3+[145,120,100,95,90,130,165,165])
        event=r['events'][0]
        self.assertEqual(event['reason'],'video_ended_mid_attempt')
        self.assertTrue(event['depth_confirmed'])
        self.assertFalse(event['return_confirmed'])

    def test_long_measurement_gap_abandons_one_attempt(self):
        _,_,r=replay([165]*3+[145,120,100,95,90]+[None]*7+[165]*3)
        self.assertEqual(len(r['events']),1)
        self.assertEqual(r['events'][0]['reason'],'tracking_gap')
        self.assertIn('no_subject_detected',r['events'][0]['measurement_reasons'])

    def test_short_measurement_gap_can_recover_without_invented_motion(self):
        c,_,r=replay([165]*3+[145,120,100,95,90,None,130]+[165]*3)
        self.assertEqual(c.completed_reps,1)
        self.assertEqual(r['events'][0]['status'],'counted')
        self.assertEqual(r['events'][0]['measurement_reasons'],{'no_subject_detected':1})

    def test_stable_one_counter_is_still_authoritative(self):
        c,_,r=replay([165,90,165],CounterConfig(stable_frames=1))
        self.assertEqual((c.completed_reps,r['counted_attempts']),(1,1))

    def test_repeated_cycles_are_not_merged(self):
        cycle=[145,120,100,95,90]+[165]*3
        c,_,r=replay([165]*3+cycle+cycle)
        self.assertEqual(c.completed_reps,2)
        self.assertEqual([e['attempt_id'] for e in r['events']],[1,2])

    def test_end_after_failure_is_neutral(self):
        cfg=CounterConfig();counter=SquatCounter(cfg);observer=AttemptTracker(cfg)
        for i,a in enumerate([165]*3+[145,130]):
            r=counter.update(i,i*.1,[{'track_id':1,'keypoints':points(a)}]);observer.observe(r,counter.armed_side)
        self.assertEqual(observer.finish('processing_failed')['events'][0]['reason'],'processing_failed')

    def test_invalid_config_and_observation_order_rejected(self):
        with self.assertRaises(ValueError):AttemptTracker(CounterConfig(),True)
        cfg=CounterConfig();counter=SquatCounter(cfg);observer=AttemptTracker(cfg)
        r=counter.update(0,0,[]);observer.observe(r,counter.armed_side)
        with self.assertRaises(ValueError):observer.observe(r,counter.armed_side)

    def test_long_low_confidence_gap_has_measurement_explanation(self):
        cfg=CounterConfig();counter=SquatCounter(cfg);observer=AttemptTracker(cfg)
        angles=[165]*3+[145,120,100,95,90]+[90]*7+[165]*3
        for i,angle in enumerate(angles):
            confidence=.1 if 8<=i<15 else .95
            frame=counter.update(i,round(i*.1,4),[{'track_id':1,'keypoints':points(angle,confidence=confidence)}])
            observer.observe(frame,counter.armed_side)
        event=observer.finish()['events'][0]
        self.assertEqual(event['reason'],'confidence_loss')
        self.assertEqual(event['category'],'measurement')
        self.assertFalse(event['return_confirmed'])

    def test_long_ambiguous_subject_gap_is_not_bad_form(self):
        cfg=CounterConfig();counter=SquatCounter(cfg);observer=AttemptTracker(cfg)
        for i,angle in enumerate([165]*3+[145,120,100,95,90]+[90]*7):
            detections=[{'track_id':1,'keypoints':points(angle)}]
            if i>=8:detections.append({'track_id':2,'keypoints':points(angle)})
            frame=counter.update(i,round(i*.1,4),detections)
            observer.observe(frame,counter.armed_side)
        event=observer.finish()['events'][0]
        self.assertEqual(event['reason'],'ambiguous_subjects')
        self.assertEqual(event['category'],'measurement')

    def test_two_frame_dip_is_distinct_from_single_frame_noise(self):
        _,_,report=replay([165]*3+[130,104,None,130]+[165]*3)
        self.assertEqual(len(report['events']),1)
        self.assertEqual(report['events'][0]['reason'],'depth_not_sustained')
        self.assertEqual(report['events'][0]['start_frame'],3)
