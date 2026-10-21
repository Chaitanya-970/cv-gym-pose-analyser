"""Review52 regressions for real feedback failures, not inference quality."""
import unittest
from tests import test_posture_critique as fixtures
from gym_analyser.coaching_feedback import generate_coaching_feedback


class TestReview52(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.TestChangingVsStableMeasurements()
        self.fixture.setUp()

    def generate(self):
        f = self.fixture
        return generate_coaching_feedback(f.base_summary, f.base_attempts,
            f.base_posture, f.base_obs, f.base_phases)

    def test_zero_or_rounded_zero_duration_does_not_crash_or_advise_tempo(self):
        for inflection in (0.5, 0.501):
            with self.subTest(inflection=inflection):
                f = self.fixture
                f.base_phases['attempts'][0]['min_knee_point']['video_time_s'] = inflection
                result = self.generate()
                self.assertEqual(result['overall_summary']['total_counted_reps'], 2)
                self.assertFalse(any(s['rule_id'] == 'SUG-PACING-01'
                    for s in result['per_repetition_cues'][0]['suggestions']))

    def test_missing_track_identity_cannot_establish_session_median(self):
        for attempt in self.fixture.base_attempts['events']:
            attempt.pop('track_id', None)
        for cue in self.generate()['per_repetition_cues']:
            obs = next(o for o in cue['observations'] if o['rule_id'] == 'OBS-TORSO-VAR-01')
            self.assertEqual(obs['confidence_level'], 'unavailable')
            self.assertNotIn('session group median', obs['message'])

    def test_sparse_peer_cannot_establish_torso_comparison(self):
        self.fixture.base_posture['events'][1]['torso_inclination']['coverage_fraction'] = .01
        result = self.generate()
        for cue in result['per_repetition_cues']:
            obs = next(o for o in cue['observations'] if o['rule_id'] == 'OBS-TORSO-VAR-01')
            self.assertEqual(obs['confidence_level'], 'unavailable')
            review = next(s for s in cue['suggestions'] if s['rule_id'] == 'SUG-REVIEW-01')
            self.assertNotIn('torso inclination deviation', review['message'])

    def test_pacing_filter_is_not_prescriptive(self):
        result = self.generate()
        suggestions = [s for c in result['per_repetition_cues'] for s in c['suggestions']
                       if s['rule_id'] == 'SUG-PACING-01']
        self.assertTrue(suggestions)
        for s in suggestions:
            self.assertIn('provisional', (s['message'] + s['disclaimer']).lower())
            self.assertNotIn('Consider maintaining a controlled', s['message'])

if __name__ == '__main__':
    unittest.main()
