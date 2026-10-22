"""Optional local heel-review API, bounded worker and input validation."""
import json
import math
import os
import subprocess
import sys
import threading
import uuid
from functools import partial
from datetime import datetime, timezone
from pathlib import Path
from .web_jobs import JobBusyError, is_valid_job_id


def _validate_reference(body,view,reference):
    expected={'reference_start_s','reference_end_s',view,'stationary_camera',reference}
    if not isinstance(body,dict) or set(body)!=expected:raise ValueError('Supply a reference interval and all three confirmations')
    for field in [view,'stationary_camera',reference]:
        if body[field] is not True:raise ValueError('Confirm the required view, stationary camera and reference stance')
    start,end=body['reference_start_s'],body['reference_end_s']
    if any(isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) for v in [start,end]):
        raise ValueError('Reference times must be finite seconds')
    if not 0<=start<end<=1200 or not .4-1e-9<=end-start<=5:
        raise ValueError('Choose a 0.4 to 5 second reference interval within the video')
    return dict(body)


def validate_request(body):
    return _validate_reference(body,'side_view','heels_flat')


def validate_knee_request(body):
    return _validate_reference(body,'front_view','standing_still')



def validate_back_request(body):
    extra={'facing','same_facing_direction'}
    if not isinstance(body,dict) or not extra.issubset(body):raise ValueError('Supply facing direction and confirm no turning')
    if body['facing'] not in ('left','right') or body['same_facing_direction'] is not True:raise ValueError('Choose left or right facing and confirm no turning')
    _validate_reference({k:v for k,v in body.items() if k not in extra},'side_view','standing_still')
    return dict(body)


def _child(parent,name):
    path=parent/name
    if path.resolve().parent!=parent.resolve():raise ValueError('Review storage cannot redirect outside its job')
    return path


def _save(path,data):
    tmp=_child(path.parent,path.name+'.tmp')
    with tmp.open('w',encoding='utf-8') as stream:json.dump(data,stream,indent=2,allow_nan=False)
    tmp.replace(path)


def run_optional_worker(*, input_path, model_path, calibration_path, output_path, log_path, gym_pose_dir, timeout, module="gym_analyser.heel_runner"):
    if module not in ['gym_analyser.heel_runner','gym_analyser.knee_runner','gym_analyser.back_runner']:raise ValueError('Unsupported review worker')
    from .review_execution import _start_owned_child,kill_process_tree
    root=Path(gym_pose_dir);is_back=module=='gym_analyser.back_runner'
    special=root/('.venv_spine/Scripts/python.exe' if is_back else '.venv_fault/Scripts/python.exe')
    interpreter=special if special.is_file() else Path(sys.executable)
    if not Path(model_path).is_file():raise RuntimeError('Optional local auxiliary model is unavailable')
    env=os.environ.copy();paths=[str(root)]
    if interpreter==special:
        paths.extend(str(p) for p in ([root/'.venv_spine/Lib/site-packages'] if is_back else [])+[root/'.venv_fault/Lib/site-packages',root/'.venv_cuda/Lib/site-packages'] if p.is_dir())
    env['PYTHONPATH']=os.pathsep.join(paths)
    command=[str(interpreter),'-m',module,'--input',str(input_path),
             '--model',str(model_path),'--calibration',str(calibration_path),'--output',str(output_path),
             '--time-limit',str(max(1.,timeout-5))]
    if is_back:
        spine_model=root/'models/spinepose-s-v2.onnx'
        if not spine_model.is_file():raise RuntimeError('Optional local spine model is unavailable')
        command.extend(['--spine-model',str(spine_model)])
    with Path(log_path).open('x',encoding='utf-8') as log:
        proc,job=_start_owned_child(command,log,{'cwd':str(root),'env':env})
        try:code=proc.wait(timeout=timeout)
        finally:kill_process_tree(proc,job_handle=job)
    if code!=0:raise RuntimeError('Optional technique worker failed; check the local server review log')
    return json.loads(Path(output_path).read_text(encoding='utf-8'))


class HeelReviewService:
    def __init__(self,manager,runner=None,*,kind='heel'):
        if kind not in ['heel','knee','back']:raise ValueError('Unsupported review kind')
        self.manager=manager;self.kind=kind
        self.validator={'heel':validate_request,'knee':validate_knee_request,'back':validate_back_request}[kind]
        self.runner=runner or partial(run_optional_worker,module='gym_analyser.'+kind+'_runner')
        self.status_filename=kind+'_review_status.json'
        self.artifact_filename=kind+'_review.json'
        self.runs_name=kind+'_reviews'

    def status(self,job_id):
        if not is_valid_job_id(job_id):raise ValueError('Invalid job ID')
        if not self.manager.get_metadata(job_id):raise KeyError('Unknown job')
        root=self.manager._get_job_dir(job_id);path=_child(root,self.status_filename)
        state=json.loads(path.read_text(encoding='utf-8')) if path.is_file() else {'status':'not_started'}
        if state['status']=='running' and self.manager.get_active_job_id()!=job_id:
            state={'status':'failed','error':'Technique review was interrupted. Submit a new review to retry.'}
            _save(path,state)
        if state['status']=='completed':
            output=self.manager.get_artifact_path(job_id,self.artifact_filename)
            state['result']=json.loads(output.read_text(encoding='utf-8'))
        return state

    def submit(self,job_id,body,background=None):
        request=self.validator(body)
        if not is_valid_job_id(job_id):raise ValueError('Invalid job ID')
        metadata=self.manager.get_metadata(job_id)
        if not metadata:raise KeyError('Unknown job')
        if metadata.get('status') not in ['completed','incomplete']:raise ValueError('Finish the main video analysis before technique review')
        root=self.manager._get_job_dir(job_id);sources=list(_child(root,'input').glob('source.*'))
        if len(sources)!=1 or sources[0].resolve().parent!=root/'input':raise ValueError('Uploaded source is unavailable')
        with self.manager._lock:
            if self.manager._active_job_id is not None:raise JobBusyError('One video worker at a time')
            self.manager._active_job_id=job_id
        try:
            runs=_child(root,self.runs_name);runs.mkdir(exist_ok=True);request['review_id']=str(uuid.uuid4())
            run=_child(runs,request['review_id']);run.mkdir();_save(_child(run,'calibration.json'),request)
            _save(_child(root,self.status_filename),{'status':'running','review_id':request['review_id']})
            if background is not None:background.add_task(self._execute,job_id,run,sources[0],request)
            else:threading.Thread(target=self._execute,args=(job_id,run,sources[0],request),daemon=True).start()
        except BaseException:
            try:_save(_child(root,self.status_filename),{'status':'failed','error':'Review could not be scheduled. Main results unchanged.'})
            finally:
                with self.manager._lock:self.manager._active_job_id=None
            raise
        return {'status':'running','review_id':request['review_id']}

    def _execute(self,job_id,run,source,request):
        root=self.manager._get_job_dir(job_id)
        state={'status':'failed','review_id':request['review_id'],'error':'Technique review interrupted; main results unchanged.'}
        try:
            report=self.runner(input_path=source,model_path=self.manager.gym_pose_dir/'models/mediapipe_pose_full.task',
                calibration_path=run/'calibration.json',output_path=run/'result.json',log_path=run/'worker.log',
                gym_pose_dir=self.manager.gym_pose_dir,timeout=min(self.manager.timeout_seconds,180.))
            if not isinstance(report,dict) or report.get('status') not in ['unable_to_assess','review_flags','no_review_flag']:
                raise ValueError('Invalid optional review result')
            if report.get('review_id')!=request['review_id'] or report.get('accuracy')!='not_validated':
                raise ValueError('Optional review identity/status mismatch')
            _save(_child(run,'accepted_result.json'),report)
            _save(_child(self.manager._get_results_dir(job_id),self.artifact_filename),report)
            state={'status':'completed','review_id':request['review_id'],'finished_at':datetime.now(timezone.utc).isoformat()}
        except Exception as exc:
            state={'status':'failed','review_id':request['review_id'],
                   'error':'Technique review unavailable or failed. Main repetition results are unchanged.'}
            _save(_child(run,'failure.json'),{'type':type(exc).__name__,'message':str(exc)})
        finally:
            try:_save(_child(root,self.status_filename),state)
            finally:
                with self.manager._lock:
                    if self.manager._active_job_id==job_id:self.manager._active_job_id=None


def mount_heel_review(app,manager,runner=None,*,kind="heel"):
    from fastapi import BackgroundTasks,HTTPException
    service=HeelReviewService(manager,runner,kind=kind);setattr(app.state,kind+"_review_service",service)
    @app.post('/api/jobs/{job_id}/'+kind+'-review',status_code=202)
    def submit(job_id:str,body:dict,background_tasks:BackgroundTasks):
        try:return service.submit(job_id,body,background_tasks)
        except JobBusyError as exc:raise HTTPException(429,str(exc))
        except KeyError:raise HTTPException(404,'Unknown job')
        except ValueError as exc:raise HTTPException(400,str(exc))
    @app.get('/api/jobs/{job_id}/'+kind+'-review')
    def get_status(job_id:str):
        try:return service.status(job_id)
        except KeyError:raise HTTPException(404,'Unknown job')
        except ValueError as exc:raise HTTPException(400,str(exc))
    return service
