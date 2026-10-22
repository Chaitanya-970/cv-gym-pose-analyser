"""API/lease/export/failure regressions for optional heel review."""
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch
from fastapi.testclient import TestClient
from gym_analyser.web import create_app,INDEX_HTML
from gym_analyser.web_jobs import JobManager,JobBusyError
from gym_analyser.heel_web import HeelReviewService,validate_request,run_optional_worker

BODY=dict(reference_start_s=1.8,reference_end_s=2.2,side_view=True,stationary_camera=True,heels_flat=True)

def success(**kw):
    request=json.loads(Path(kw['calibration_path']).read_text())
    return dict(status='review_flags',accuracy='not_validated',review_id=request['review_id'],
                events=[dict(start_s=4.,end_s=5.,peak_s=4.4)],measurable_samples=100,total_samples=116)

class TestHeelWeb(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.manager=JobManager(Path(self.tmp.name)/'jobs',runner=lambda **kw:None)
        self.job,_,_=self.manager.save_upload_file(io.BytesIO(b'video'),'clip.mp4')
        meta=self.manager.get_metadata(self.job);meta['status']='completed';self.manager._save_metadata(self.job,meta)
        (self.manager._get_results_dir(self.job)/'summary.json').write_text(json.dumps({'completed_reps':6}))
        self.client=TestClient(create_app(self.manager,enable_review=False,heel_review_runner=success))
    def test_post_status_export_preserves_count_and_unvalidated_label(self):
        response=self.client.post(f'/api/jobs/{self.job}/heel-review',json=BODY);self.assertEqual(response.status_code,202)
        state=self.client.get(f'/api/jobs/{self.job}/heel-review').json();self.assertEqual(state['status'],'completed')
        self.assertEqual(state['result']['accuracy'],'not_validated');self.assertFalse(self.manager.is_busy())
        exported=self.client.get(f'/api/jobs/{self.job}/download/heel_review.json');self.assertEqual(exported.status_code,200)
        self.assertEqual(exported.json(),state['result'])
        result=self.client.get(f'/api/jobs/{self.job}/results').json();self.assertEqual(result['summary']['completed_reps'],6)
    def test_new_reference_preserves_revision_evidence(self):
        for body in [BODY,dict(BODY,reference_start_s=2.,reference_end_s=2.6)]:
            self.assertEqual(self.client.post(f'/api/jobs/{self.job}/heel-review',json=body).status_code,202)
        runs=list((self.manager._get_job_dir(self.job)/'heel_reviews').iterdir());self.assertEqual(len(runs),2)
        self.assertTrue(all((p/'accepted_result.json').is_file() for p in runs))
    def test_main_or_auxiliary_busy_rejects_without_second_worker(self):
        self.manager._active_job_id=self.job
        self.assertEqual(self.client.post(f'/api/jobs/{self.job}/heel-review',json=BODY).status_code,429)
        self.assertFalse((self.manager._get_job_dir(self.job)/'heel_reviews').exists())
    def test_worker_failure_releases_lease_and_keeps_main_completed(self):
        def fail(**kw):raise TimeoutError('worker timeout')
        c=TestClient(create_app(self.manager,enable_review=False,heel_review_runner=fail))
        self.assertEqual(c.post(f'/api/jobs/{self.job}/heel-review',json=BODY).status_code,202)
        self.assertEqual(c.get(f'/api/jobs/{self.job}/heel-review').json()['status'],'failed')
        self.assertFalse(self.manager.is_busy());self.assertEqual(self.manager.get_metadata(self.job)['status'],'completed')
    def test_false_missing_extra_or_invalid_reference_rejected(self):
        values=[dict(BODY,side_view=False),dict(BODY,heels_flat=1),dict(BODY,reference_start_s=True),
                dict(BODY,reference_end_s=1.9),dict(BODY,reference_end_s=9),dict(BODY,source_path='../private')]
        for body in values:self.assertEqual(self.client.post(f'/api/jobs/{self.job}/heel-review',json=body).status_code,400)
        self.assertFalse(self.manager.is_busy())
    def test_unknown_invalid_and_noncompleted_job(self):
        self.assertEqual(self.client.get('/api/jobs/not-a-job/heel-review').status_code,400)
        meta=self.manager.get_metadata(self.job);meta['status']='failed';self.manager._save_metadata(self.job,meta)
        self.assertEqual(self.client.post(f'/api/jobs/{self.job}/heel-review',json=BODY).status_code,400)
    def test_download_allowlist_still_blocks_private_calibration_log(self):
        self.client.post(f'/api/jobs/{self.job}/heel-review',json=BODY)
        self.assertEqual(self.client.get(f'/api/jobs/{self.job}/download/calibration.json').status_code,404)
        self.assertEqual(self.client.get(f'/api/jobs/{self.job}/download/worker.log').status_code,404)
    def test_bad_result_identity_cannot_replace_public_output(self):
        def bad(**kw):return dict(status='review_flags',accuracy='validated',review_id='wrong')
        c=TestClient(create_app(self.manager,enable_review=False,heel_review_runner=bad))
        c.post(f'/api/jobs/{self.job}/heel-review',json=BODY)
        self.assertEqual(c.get(f'/api/jobs/{self.job}/heel-review').json()['status'],'failed')
        self.assertFalse((self.manager._get_results_dir(self.job)/'heel_review.json').exists())
    def test_scheduling_failure_releases_lease(self):
        background=Mock();background.add_task.side_effect=RuntimeError('scheduler failed')
        with self.assertRaises(RuntimeError):HeelReviewService(self.manager,success).submit(self.job,BODY,background)
        self.assertFalse(self.manager.is_busy())
        self.assertEqual(HeelReviewService(self.manager).status(self.job)['status'],'failed')
    def test_stale_running_status_is_explicitly_interrupted(self):
        path=self.manager._get_job_dir(self.job)/'heel_review_status.json'
        path.write_text(json.dumps({'status':'running','review_id':'old'}))
        state=HeelReviewService(self.manager).status(self.job)
        self.assertEqual(state['status'],'failed');self.assertIn('interrupted',state['error'])
    def test_timeout_cleans_owned_process(self):
        root=Path(self.tmp.name);model=root/'model.task';model.touch();proc=Mock();proc.wait.side_effect=TimeoutError('timeout')
        with patch('gym_analyser.review_execution._start_owned_child',return_value=(proc,123)),patch('gym_analyser.review_execution.kill_process_tree') as kill:
            with self.assertRaises(TimeoutError):run_optional_worker(input_path=root/'video',model_path=model,
              calibration_path=root/'cal',output_path=root/'out',log_path=root/'run.log',gym_pose_dir=root,timeout=20)
        kill.assert_called_once_with(proc,job_handle=123)
    def test_ui_has_confirmation_uncertainty_replay_and_bounded_fetch(self):
        for value in ['heel-review-form','heel-flat','heel-stationary','heel-side-view','Accuracy unvalidated',
                      'seekVideoToTime(peak)','heel_review.json','controller.abort(), 15000']:
            self.assertIn(value,INDEX_HTML)

if __name__=='__main__':unittest.main()
