"""Knee review API wiring and genuine-reference rejection behavior."""
import io,json,tempfile,unittest
from pathlib import Path
from unittest.mock import Mock
from fastapi.testclient import TestClient
from gym_analyser.web import create_app,INDEX_HTML
from gym_analyser.web_jobs import JobManager
from gym_analyser.heel_web import validate_knee_request
BODY=dict(reference_start_s=3.,reference_end_s=3.4,front_view=True,stationary_camera=True,standing_still=True)

def fixture(**kw):
    request=json.loads(Path(kw['calibration_path']).read_text())
    return dict(status='review_flags',accuracy='not_validated',review_id=request['review_id'],
        events=[dict(type='possible_inward_knee_tracking',side='left',start_s=4.,end_s=5.,peak_s=4.4)],measurable_samples=12,total_samples=20)

class TestKneeWeb(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.manager=JobManager(Path(self.tmp.name)/'jobs',runner=lambda **kw:None)
        self.job,_,_=self.manager.save_upload_file(io.BytesIO(b'video'),'front.mp4')
        meta=self.manager.get_metadata(self.job);meta['status']='incomplete';self.manager._save_metadata(self.job,meta)
        self.called=Mock(side_effect=fixture)
        self.client=TestClient(create_app(self.manager,enable_review=False,knee_review_runner=self.called))
    def test_knee_route_export_has_distinct_artifact_and_uncertainty(self):
        self.assertEqual(self.client.post(f'/api/jobs/{self.job}/knee-review',json=BODY).status_code,202)
        state=self.client.get(f'/api/jobs/{self.job}/knee-review').json();self.assertEqual(state['status'],'completed')
        self.assertEqual(state['result']['accuracy'],'not_validated');self.assertEqual(self.called.call_count,1)
        self.assertEqual(self.client.get(f'/api/jobs/{self.job}/download/knee_review.json').json(),state['result'])
        self.assertEqual(self.client.get(f'/api/jobs/{self.job}/heel-review').json()['status'],'not_started')
        self.assertEqual(self.manager.get_metadata(self.job)['status'],'incomplete')
    def test_actual_rejected_moving_reference_cannot_start_worker(self):
        response=self.client.post(f'/api/jobs/{self.job}/knee-review',json=dict(BODY,standing_still=False))
        self.assertEqual(response.status_code,400);self.called.assert_not_called();self.assertFalse(self.manager.is_busy())
    def test_side_view_or_heel_request_cannot_be_misread_as_front_reference(self):
        wrong=dict(reference_start_s=3.,reference_end_s=3.4,side_view=True,stationary_camera=True,heels_flat=True)
        self.assertEqual(self.client.post(f'/api/jobs/{self.job}/knee-review',json=wrong).status_code,400)
        self.called.assert_not_called()
    def test_shared_worker_lock_blocks_knee_and_heel_overlap(self):
        self.manager._active_job_id=self.job
        self.assertEqual(self.client.post(f'/api/jobs/{self.job}/knee-review',json=BODY).status_code,429)
        self.called.assert_not_called()
    def test_refuse_arbitrary_worker_or_server_paths(self):
        self.assertEqual(self.client.post(f'/api/jobs/{self.job}/knee-review',json=dict(BODY,model='../secret')).status_code,400)
        self.assertEqual(self.client.get(f'/api/jobs/{self.job}/download/calibration.json').status_code,404)
    def test_reference_frame_quantization_does_not_lose_boundary_sample(self):
        from gym_analyser.heel_runner import reference_timestamps
        frames=[dict(timestamp_ms=t) for t in [2798,2998,3197,3397,3597]]
        self.assertEqual(reference_timestamps(frames,3.,3.4,30.024),[2998,3197,3397])
        self.assertNotIn(2798,reference_timestamps(frames,3.,3.4,30.024))
    def test_original_confirmed_heel_reference_is_preserved(self):
        from gym_analyser.heel_runner import reference_timestamps
        frames=[dict(timestamp_ms=t) for t in [1600,1800,2000,2200,2400]]
        self.assertEqual(reference_timestamps(frames,1.8,2.2,29.996),[1800,2000,2200])
    def test_ui_front_view_reference_and_replay_labels(self):
        for text in ['knee-review-form','knee-front-view','knee-standing','front_view:','standing_still:',
                     'refreshKneeReview','possible inward-knee','knee_review.json']:
            self.assertIn(text,INDEX_HTML)
        self.assertNotIn('knees_flat:',INDEX_HTML)

if __name__=='__main__':unittest.main()
