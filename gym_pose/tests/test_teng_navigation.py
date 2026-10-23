"""Review grouping must preserve contexts and never invent fault intervals."""
import json,shutil,subprocess,unittest
from gym_analyser.teng_review_ui import GROUP_JS
class TengNavigationTests(unittest.TestCase):
 def evaluate(self,expression):
  node=shutil.which('node')
  if not node:self.skipTest('Node unavailable')
  code=GROUP_JS+'\nconsole.log(JSON.stringify('+expression+'));';res=subprocess.run([node,'-e',code],capture_output=True,text=True,timeout=10);self.assertEqual(res.returncode,0,res.stderr);return json.loads(res.stdout)
 def test_near_samples_grouped_by_kind_but_gaps_retained(self):
  rows=self.evaluate("groupTengSamples([{type:'teng_heel_image_match',peak_s:2.2},{type:'teng_heel_image_match',peak_s:2},{type:'teng_heel_image_match',peak_s:3},{type:'teng_back_image_match',peak_s:2}])")
  self.assertEqual([(g['kind'],g['start'],g['end'],len(g['times'])) for g in rows],[('heel',2,2.2,2),('heel',3,3,1),('back',2,2,1)])
 def test_counted_context_not_mixed_with_setup_or_partial_movement(self):
  rows=self.evaluate("(()=>{setTengCountedWindows({events:[{status:'counted',start_time_s:2.1,end_time_s:2.5},{status:'uncounted',start_time_s:2.7,end_time_s:3}]});return groupTengSamples([{type:'teng_heel_image_match',peak_s:2},{type:'teng_heel_image_match',peak_s:2.2},{type:'teng_heel_image_match',peak_s:2.4},{type:'teng_heel_image_match',peak_s:2.6}])})()")
  self.assertEqual([(g['context'],len(g['times'])) for g in rows],[('outside',1),('counted',2),('outside',1)])
 def test_unknown_context_invalid_times_and_duplicates(self):
  rows=self.evaluate("groupTengSamples([{type:'teng_back_image_match',peak_s:null},{type:'teng_back_image_match',peak_s:-1},{type:'other',peak_s:2},{type:'teng_back_image_match',peak_s:2},{type:'teng_back_image_match',peak_s:2}])")
  self.assertEqual(len(rows),1);self.assertEqual(rows[0]['context'],'unknown');self.assertEqual(rows[0]['times'],[2])
if __name__=='__main__':unittest.main()
