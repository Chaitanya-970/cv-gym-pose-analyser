"""Review53: conflicting saved counts must not be silently reconciled."""
import json
import tempfile
import unittest
import uuid
from pathlib import Path
from gym_analyser.web_jobs import JobManager
from tests import test_final_web_acceptance as fixtures

TestClient = fixtures.TestClient
create_app = fixtures.create_app

@unittest.skipUnless(fixtures.HAS_WEB_DEPS, "Web dependencies not installed")
class TestCountConsistency(unittest.TestCase):
    def test_summary_zero_preserved_and_conflict_reported(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); job_id = str(uuid.uuid4()); job = root / job_id
            results = job / 'results'; results.mkdir(parents=True)
            files = {'summary.json': {'status': 'completed', 'completed_reps': 0},
                     'attempts.json': {'events': [{'status': 'counted'}]},
                     'coaching_feedback.json': {'overall_summary': {'total_counted_reps': 1}}}
            for name, data in files.items():
                (results/name).write_text(json.dumps(data))
            (job/'job.json').write_text(json.dumps({'job_id': job_id, 'status': 'completed'}))
            app = create_app(job_manager=JobManager(storage_dir=root), enable_review=False)
            response = TestClient(app).get(f'/api/jobs/{job_id}/results')
            self.assertEqual(response.status_code, 200)
            data = response.json()
            self.assertEqual(data['completed_reps'], 0)
            self.assertEqual(data['count_consistency']['status'], 'conflict')
            self.assertEqual(data['count_consistency']['counts'], {'summary': 0, 'attempt_events': 1, 'coaching': 1})
            self.assertEqual(json.loads((results/'summary.json').read_text())['completed_reps'], 0)

class TestCountDisplay(fixtures._NodeAcceptanceHarness):
    def test_zero_stays_zero_and_warning_visible(self):
        self.run_node_script("""
renderResults({status:'completed',completed_reps:0,processed_frames:10,
  attempts:{events:[{status:'counted'}]},
  count_consistency:{status:'conflict',counts:{summary:0,attempt_events:1}}});
assert.strictEqual(document.getElementById('metric-reps').textContent, '0');
assert.ok(document.getElementById('incomplete-note-text').textContent.includes('disagree'));
""")

if __name__ == '__main__': unittest.main()
