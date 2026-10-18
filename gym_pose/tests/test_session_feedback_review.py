"""Independent regressions for failure modes found during integration review."""
import unittest
from gym_analyser.session_feedback import summarise_attempt_posture


def row(**changes):
    result=dict(frame_index=0,video_time_s='0.0',track_id='1',posture_side='left',
                selected_side='left',valid='False',reason='low_joint_confidence',
                torso_inclination_deg='0.0',torso_reason='ok',hip_angle_deg='',hip_reason='low_joint_confidence')
    result.update(changes)
    return result


def event(**changes):
    result=dict(attempt_id=1,start_frame=0,end_frame=0,track_id=1,side='left',status='counted')
    result.update(changes)
    return result


class TestFeedbackReview(unittest.TestCase):
    def metric(self, r, e=None):
        return summarise_attempt_posture([r],[e or event()])['events'][0]['torso']

    def test_numeric_value_cannot_override_measurement_failure(self):
        result=self.metric(row(torso_reason='low_joint_confidence',torso_inclination_deg='30'))
        self.assertEqual(result['measured_count'],0)
        self.assertEqual(result['unavailable_reasons'],{'low_joint_confidence':1})

    def test_counter_identity_failure_suppresses_stale_numeric_values(self):
        result=self.metric(row(reason='track_id_mismatch_2_vs_1'))
        self.assertEqual(result['measured_count'],0)
        self.assertEqual(result['unavailable_reasons'],{'track_mismatch':1})

    def test_missing_measurement_side_does_not_fall_back_to_knee_side(self):
        result=self.metric(row(posture_side=''))
        self.assertEqual(result['unavailable_reasons'],{'missing_side':1})

    def test_missing_event_identity_does_not_mix_people_or_sides(self):
        result=self.metric(row(),event(track_id=None))
        self.assertEqual(result['unavailable_reasons'],{'missing_event_identity':1})

    def test_valid_torso_with_invalid_knee_preserves_real_zero(self):
        result=self.metric(row())
        self.assertEqual(result['median_deg'],0.0)
        self.assertEqual(result['coverage_fraction'],1.0)

    def test_fractional_boolean_and_container_track_ids_fail_clearly(self):
        for value in (True,1.9,[],{},'9007199254740993.1'):
            with self.subTest(value=value),self.assertRaises(ValueError):
                self.metric(row(track_id=value))

    def test_unknown_event_status_and_duplicate_ids_rejected(self):
        for status in ('passed', [], None):
            with self.subTest(status=status), self.assertRaises(ValueError):
                self.metric(row(),event(status=status))
        with self.assertRaises(ValueError):
            summarise_attempt_posture([row()],[event(),event(start_frame=2,end_frame=2)])

    def test_boolean_csv_format_is_validated_without_gating_torso(self):
        self.assertEqual(self.metric(row(valid=False))['measured_count'],1)
        self.assertEqual(self.metric(row(valid='False'))['measured_count'],1)
        with self.assertRaises(ValueError):self.metric(row(valid='definitely'))

    def test_enormous_numeric_angle_becomes_unavailable(self):
        result=self.metric(row(torso_inclination_deg=10**10000))
        self.assertEqual(result['measured_count'],0)

    def test_timestamp_order_must_match_frame_order(self):
        with self.assertRaisesRegex(ValueError,'Descending video_time_s'):
            summarise_attempt_posture([row(video_time_s=1),row(frame_index=1,video_time_s=0)],[])
