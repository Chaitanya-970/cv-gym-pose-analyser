"""Execute actual temporal-review page handlers; no browser or real annotations."""
import json,re,shutil,subprocess,tempfile,unittest
from pathlib import Path
from gym_analyser.review_web import INDEX_HTML

NODE = r"""
const fs=require('fs'),vm=require('vm'),assert=require('assert');
const html=JSON.parse(fs.readFileSync(process.argv[2],'utf8'));
class E {
 constructor(tag='div'){this.tagName=tag;this.children=[];this.value='';this.handlers={};this.style={};this.checked=false;this.disabled=false;this.className='';this.files=[];this.classList={add(){},remove(){},toggle(){}};}
 set textContent(v){this.text=String(v);this.children=[];} get textContent(){return this.text||'';}
 appendChild(c){this.children.push(c);c.parent=this;if(this.tagName==='select' && (c.selected || this.children.length===1))this.value=c.value;return c;}
 removeChild(c){this.children=this.children.filter(x=>x!==c);} remove(){if(this.parent)this.parent.removeChild(this);}
 setAttribute(k,v){this[k]=v;} addEventListener(k,f){this.handlers[k]=f;} load(){} pause(){}
 querySelectorAll(sel){const options=sel.split(',').map(x=>x.trim()),out=[];const walk=n=>{for(const c of n.children){if(options.some(s=>s[0]==='.'?c.className.split(' ').includes(s.slice(1)):c.tagName===s))out.push(c);walk(c);}};walk(this);return out;}
 querySelector(s){return this.querySelectorAll(s)[0]||null;}
}
const elements={},el=id=>elements[id]||(elements[id]=new E());
const posts=[];
const context=vm.createContext({console,Number,Promise,Date,NaN,JSON,encodeURIComponent,module:{exports:{}},confirm:()=>true,
 window:{location:{pathname:'/review/'}},document:{getElementById:el,createElement:tag=>new E(tag)},
 fetch:async(url,opts)=>{if(opts&&opts.method==='POST'){const payload=JSON.parse(opts.body);posts.push(payload);return {ok:true,json:async()=>({case_id:'case-1',review:payload})};}return {ok:true,json:async()=>({cases:[]})};}});
vm.runInContext(html.match(/<script>([\s\S]*?)<\/script>/)[1],context);
vm.runInContext("currentCaseId='case-1';activeRequestGen=1;",context);
const submit=async()=>{el('review-count').value='';el('review-scope').value='uncertain';await el('review-form').handlers.submit({preventDefault(){}});};
const populate=review=>{context.testReview=review;vm.runInContext("populateCaseView({case_id:'case-1',review:testReview});",context);};
(async()=>{
 vm.runInContext('addEventRow()',context);
 let row=el('events-tbody').querySelectorAll('tr')[0];
 assert.strictEqual(row.querySelector('.event-class').value,'','new human event must not be classified automatically');
 await submit();assert.strictEqual(posts.length,0,'unclassified event must not be saved as completed');
 row.querySelector('.event-class').value='ambiguous_or_unclear';await submit();assert.strictEqual(posts.length,1);
 assert.strictEqual(posts[0].events[0].event_class,'ambiguous_or_unclear');assert.strictEqual(posts[0].human_count,null);
 const legacy={event_index:7,start_time_s:1,end_time_s:2,intent:'uncertain',cycle:'complete',observation:'original human note'};
 populate({human_count:null,events:[legacy]});el('review-notes').value='unrelated note edit';await submit();
 assert.deepStrictEqual(posts.at(-1).events,[legacy],'saving unrelated notes must preserve legacy annotation exactly');
 const modern={event_id:'evt_1',event_class:'ambiguous_or_unclear',start_time_s:null,turnaround_time_s:null,end_time_s:null,visible_posture_observations:{visibility:'unclear'},uncertainty_notes:null};
 populate({human_count:null,events:[modern]});await submit();assert.deepStrictEqual(posts.at(-1).events[0].visible_posture_observations,modern.visible_posture_observations,'structured annotation must not become text');
 populate({human_count:null,events:[modern,{...modern,event_id:'evt_3'}]});vm.runInContext('addEventRow()',context);
 const ids=el('events-tbody').querySelectorAll('.event-id').map(x=>x.value);assert.strictEqual(new Set(ids).size,ids.length,'generated IDs must remain unique after deletion');
 console.log('ACTUAL_TEMPORAL_CLIENT_OK');
})().catch(e=>{console.error(e);process.exitCode=1;});
"""

class TestActualTemporalClient(unittest.TestCase):
    @unittest.skipUnless(shutil.which('node'), 'Node unavailable')
    def test_explicit_truth_and_lossless_roundtrip(self):
        with tempfile.TemporaryDirectory() as td:
            base=Path(td);(base/'client.cjs').write_text(NODE,encoding='utf-8');(base/'page.json').write_text(json.dumps(INDEX_HTML),encoding='utf-8')
            run=subprocess.run(['node',str(base/'client.cjs'),str(base/'page.json')],capture_output=True,text=True,timeout=15)
            self.assertEqual(run.returncode,0,run.stdout+run.stderr)
            self.assertIn('ACTUAL_TEMPORAL_CLIENT_OK',run.stdout)
