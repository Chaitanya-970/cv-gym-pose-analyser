"""Experimental back review API isolation and user-confirmation contracts."""
import io,json,tempfile,unittest
from pathlib import Path
from unittest.mock import Mock
from fastapi.testclient import TestClient
from gym_analyser.web import create_app,INDEX_HTML
from gym_analyser.web_jobs import JobManager
BODY=dict(reference_start_s=1.8,reference_end_s=2.2,side_view=True,stationary_camera=True,standing_still=True,facing='left',same_facing_direction=True)


def fixture(**kw):
    request=json.loads(Path(kw['calibration_path']).read_text())
    return dict(status='review_flags',accuracy='not_validated',review_id=request['review_id'],events=[dict(type='possible_back_rounding',start_s=4.,end_s=5.,peak_s=4.4)],measurable_samples=12,total_samples=20)

class TestBackWeb(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.manager=JobManager(Path(self.tmp.name)/'jobs',runner=lambda **kw:None)
        self.job,_,_=self.manager.save_upload_file(io.BytesIO(b'video'),'side.mp4')
        meta=self.manager.get_metadata(self.job);meta['status']='completed';self.manager._save_metadata(self.job,meta)
        self.called=Mock(side_effect=fixture)
        self.client=TestClient(create_app(self.manager,enable_review=False,back_review_runner=self.called))
    def test_export_and_review_isolation(self):
        self.assertEqual(self.client.post(f'/api/jobs/{self.job}/back-review',json=BODY).status_code,202)
        state=self.client.get(f'/api/jobs/{self.job}/back-review').json();self.assertEqual(state['status'],'completed')
        self.assertEqual(self.client.get(f'/api/jobs/{self.job}/download/back_review.json').json(),state['result'])
        for kind in ['heel','knee']:self.assertEqual(self.client.get(f'/api/jobs/{self.job}/{kind}-review').json()['status'],'not_started')
        self.assertFalse(self.manager.is_busy());self.assertEqual(self.manager.get_metadata(self.job)['status'],'completed')
    def test_movement_turning_or_unknown_facing_cannot_launch(self):
        for changed in [dict(standing_still=False),dict(same_facing_direction=False),dict(facing=''),dict(facing=True),dict(side_view=False)]:
            self.assertEqual(self.client.post(f'/api/jobs/{self.job}/back-review',json=dict(BODY,**changed)).status_code,400)
        self.called.assert_not_called();self.assertFalse(self.manager.is_busy())
    def test_shared_lease_no_overlap(self):
        self.manager._active_job_id=self.job
        self.assertEqual(self.client.post(f'/api/jobs/{self.job}/back-review',json=BODY).status_code,429);self.called.assert_not_called()
    def test_worker_failure_releases_lease_without_main_mutation(self):
        self.called.side_effect=RuntimeError('optional missing dependency')
        self.assertEqual(self.client.post(f'/api/jobs/{self.job}/back-review',json=BODY).status_code,202)
        self.assertEqual(self.client.get(f'/api/jobs/{self.job}/back-review').json()['status'],'failed')
        self.assertFalse(self.manager.is_busy());self.assertEqual(self.manager.get_metadata(self.job)['status'],'completed')
        self.assertEqual(self.client.get(f'/api/jobs/{self.job}/download/back_review.json').status_code,404)
    def test_no_client_paths_and_exact_schema(self):
        self.assertEqual(self.client.post(f'/api/jobs/{self.job}/back-review',json=dict(BODY,spine_model='../model')).status_code,400)
        self.assertEqual(self.client.get(f'/api/jobs/{self.job}/download/calibration.json').status_code,404)
        self.called.assert_not_called()
    def test_result_identity_checked(self):
        self.called.side_effect=lambda **kw:dict(status='no_review_flag',accuracy='not_validated',review_id='wrong',events=[])
        self.client.post(f'/api/jobs/{self.job}/back-review',json=BODY)
        self.assertEqual(self.client.get(f'/api/jobs/{self.job}/back-review').json()['status'],'failed')
        self.assertFalse(self.manager.is_busy())
    def test_ui_distinguishes_reference_from_neutral_spine(self):
        for text in ['back-review-form','back-no-turning','back-facing','same_facing_direction:','standing_still:', 'refreshBackReview','back_review.json','does not certify a neutral spine','possible lumbar-curvature review flags']:
            self.assertIn(text,INDEX_HTML)
        self.assertNotIn('backs_flat:',INDEX_HTML)

if __name__=='__main__':unittest.main()
