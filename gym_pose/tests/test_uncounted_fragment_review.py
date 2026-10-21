"""Review54: detector fragments are not automatically separate squat attempts."""
import copy
import unittest
from gym_analyser.coaching_feedback import generate_coaching_feedback

class TestFragmentInterpretation(unittest.TestCase):
    def test_between_counted_fragment_is_ambiguous_and_counts_unchanged(self):
        attempts = {'counted_attempts': 2, 'uncounted_attempts': 1, 'events': [
            {'attempt_id': 1, 'status': 'counted'},
            {'attempt_id': 2, 'status': 'uncounted', 'reason': 'depth_threshold_not_reached',
             'min_knee_angle_deg': 118.0},
            {'attempt_id': 3, 'status': 'counted'}]}
        before = copy.deepcopy(attempts)
        result = generate_coaching_feedback({'completed_reps': 2}, attempts)
        item = result['uncounted_movement_explanations'][0]
        self.assertEqual(item['context_hypothesis'], 'ambiguous')
        self.assertIn('part of a nearby counted squat', item['explanation'])
        self.assertEqual(result['overall_summary']['total_counted_reps'], 2)
        self.assertEqual(result['overall_summary']['total_uncounted_attempts'], 1)
        self.assertEqual(item['reason_code'], 'depth_threshold_not_reached')
        self.assertEqual(attempts, before)

if __name__ == '__main__': unittest.main()
