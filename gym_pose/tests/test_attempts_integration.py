"""Verify attempt exports through real video IO and synthetic model detections."""
import csv
import json
import tempfile
import unittest
from pathlib import Path
import torch
from tests.test_attempts import points
from tests.test_pipeline import create_mock_pose_model, create_test_video
from gym_analyser.pipeline import run_pipeline
from gym_analyser.counter import CounterConfig


class TestAttemptExports(unittest.TestCase):
    def run_case(self, angles, failure=False):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=root/'source.mp4';out=root/'out'
            create_test_video(source, num_frames=len(angles)+(1 if failure else 0))
            model=create_mock_pose_model()
            template=model.track.return_value[0]
            import copy
            outputs=[]
            for angle in angles:
                result=copy.deepcopy(template)
                result.keypoints.data=torch.tensor([points(angle)])
                outputs.append([result])
            if failure:outputs.append(RuntimeError('Inference interrupted'))
            model.track.side_effect=outputs
            if failure:
                with self.assertRaisesRegex(RuntimeError,'Inference interrupted'):
                    run_pipeline(source,out,CounterConfig(),model_name=model)
            else:run_pipeline(source,out,CounterConfig(),model_name=model)
            with (out/'angles.csv').open(newline='', encoding='utf-8') as stream:
                rows=list(csv.DictReader(stream))
            return (json.loads((out/'summary.json').read_text()),
                    json.loads((out/'attempts.json').read_text()), rows)

    def test_shallow_attempt_reason_reaches_all_exports(self):
        summary,report,rows=self.run_case([165]*3+[145,130,125]+[165]*3)
        self.assertEqual(summary['completed_reps'],0)
        self.assertEqual(summary['attempt_observations']['uncounted_attempts'],1)
        self.assertEqual(report['events'][0]['reason'],'depth_threshold_not_reached')
        self.assertEqual(rows[-1]['last_attempt_reason'],'depth_threshold_not_reached')
        self.assertEqual(rows[4]['attempt_id'],'1')
        self.assertEqual(len(rows),9)

    def test_counted_cycle_matches_counter_and_summary(self):
        summary,report,rows=self.run_case([165]*3+[145,130,100,95,90]+[165]*3)
        self.assertEqual(summary['completed_reps'],1)
        self.assertEqual(report['counted_attempts'],1)
        self.assertEqual(report['uncounted_attempts'],0)
        self.assertEqual(rows[-1]['last_attempt_reason'],'completed_cycle')

    def test_failure_preserves_unresolved_attempt_and_partial_csv(self):
        summary,report,rows=self.run_case([165]*3+[145,130],failure=True)
        self.assertEqual(summary['status'],'failed')
        self.assertEqual(report['events'][0]['reason'],'processing_failed')
        self.assertEqual(len(rows),5)
