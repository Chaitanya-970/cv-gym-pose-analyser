"""Teng review contracts, model metadata rejection and frozen feature parity."""
import io,json,tempfile,unittest,math
from pathlib import Path
from unittest.mock import Mock
from fastapi.testclient import TestClient
from gym_analyser.web import create_app,INDEX_HTML
from gym_analyser.web_jobs import JobManager
from gym_analyser.teng_runner import load_candidate,probability
BODY={'side_view':True,'stationary_camera':True}
def fixture(**kw):
 q=json.loads(Path(kw['calibration_path']).read_text())
 return dict(status='review_flags',accuracy='not_validated',review_id=q['review_id'],events=[dict(type='teng_back_image_match',start_s=4.,end_s=4.,peak_s=4.)],measurable_samples=12,total_samples=20)
class TestTengWeb(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.manager=JobManager(Path(self.tmp.name)/'jobs',runner=lambda **kw:None);self.job,_,_=self.manager.save_upload_file(io.BytesIO(b'video'),'side.mp4');meta=self.manager.get_metadata(self.job);meta['status']='completed';self.manager._save_metadata(self.job,meta);self.called=Mock(side_effect=fixture);self.client=TestClient(create_app(self.manager,enable_review=False,teng_review_runner=self.called))
 def test_run_download_isolation(self):
  self.assertEqual(self.client.post(f'/api/jobs/{self.job}/teng-review',json=BODY).status_code,202);state=self.client.get(f'/api/jobs/{self.job}/teng-review').json();self.assertEqual(state['status'],'completed');self.assertEqual(self.client.get(f'/api/jobs/{self.job}/download/teng_review.json').json(),state['result']);self.assertFalse(self.manager.is_busy());self.assertEqual(self.manager.get_metadata(self.job)['status'],'completed')
  for kind in ['heel','back','knee']:self.assertEqual(self.client.get(f'/api/jobs/{self.job}/{kind}-review').json()['status'],'not_started')
 def test_reject_paths_or_unconfirmed_view(self):
  for body in [{},dict(BODY,side_view=False),dict(BODY,side_view=1),dict(BODY,model='private'),dict(BODY,reference_start_s=0)]:self.assertEqual(self.client.post(f'/api/jobs/{self.job}/teng-review',json=body).status_code,400)
  self.called.assert_not_called()
 def test_shared_lease(self):
  self.manager._active_job_id=self.job;self.assertEqual(self.client.post(f'/api/jobs/{self.job}/teng-review',json=BODY).status_code,429);self.called.assert_not_called()
 def test_failure_preserves_count_job(self):
  self.called.side_effect=RuntimeError('missing model');self.client.post(f'/api/jobs/{self.job}/teng-review',json=BODY);self.assertEqual(self.client.get(f'/api/jobs/{self.job}/teng-review').json()['status'],'failed');self.assertFalse(self.manager.is_busy());self.assertEqual(self.manager.get_metadata(self.job)['status'],'completed')
 def test_no_claim_of_fault_probability_or_reference(self):
  self.assertIn('Teng-trained back and heel review',INDEX_HTML);self.assertIn('not fault probabilities',INDEX_HTML);self.assertIn('teng_review.json',INDEX_HTML);self.assertNotIn("getElementById('teng-start')",INDEX_HTML);self.assertNotIn("getElementById('teng-flat')",INDEX_HTML);self.assertIn('Back suggestions',INDEX_HTML)
class TestCandidate(unittest.TestCase):
 def test_logit_and_dimension(self):
  c=dict(weights=[1.,-1.],mean=[0.,0.],std=[1.,1.],bias=0.,threshold=.5);self.assertEqual(probability([1.,1.],c),.5)
  for x in [[1.],[math.nan,0.]]:
   with self.assertRaises(ValueError):probability(x,c)
 def test_bad_model_metadata(self):
  with tempfile.TemporaryDirectory() as folder:
   p=Path(folder)/'model.json';base=dict(weights=[1.]*4,mean=[0.]*4,std=[1.]*4,bias=0.,threshold=.5)
   for patch in [dict(std=[0.]*4),dict(weights=[math.inf]*4),dict(bias=True),dict(threshold=.4),dict(mean=[0.])]:
    p.write_text(json.dumps(dict(base,**patch)))
    with self.assertRaises(ValueError):load_candidate(p,4)
if __name__=='__main__':unittest.main()
