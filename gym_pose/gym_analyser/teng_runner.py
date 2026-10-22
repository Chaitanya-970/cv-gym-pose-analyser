"""Bounded Teng-trained image-label review; no counter or calibration changes."""
import argparse,hashlib,json,math,time
from pathlib import Path
import numpy as np
from .teng_features import derive,outline
from .auxiliary_pose import MediaPipeLandmarker
from .spine_pose import SpineLandmarker


EXPECTED_CANDIDATE_SHA256 = {'heel': '8731cd6daa38ab43d797cbc63d39075275d6f79a10885ab372c414aa69b09353', 'back': '59138957edaff9cfd92a7ae68c78808f96edd2545f9cacf497ad233106939abf'}


def load_candidate(path,dimension):
    if Path(path).stat().st_size>32000:raise ValueError('Candidate metadata too large')
    c=json.loads(Path(path).read_text(encoding='utf-8'))
    for key in ['weights','mean','std']:
        a=np.asarray(c.get(key),dtype=float)
        if a.shape!=(dimension,) or not np.isfinite(a).all():raise ValueError('Invalid candidate '+key)
    if np.any(np.asarray(c['std'])<=0) or isinstance(c.get('bias'),bool) or not isinstance(c.get('bias'),(int,float)) or not math.isfinite(c['bias']) or c.get('threshold')!=.5:raise ValueError('Invalid candidate scale, bias or threshold')
    return c


def probability(x,c):
    x=np.asarray(x,float)
    if x.shape!=np.asarray(c['mean']).shape or not np.isfinite(x).all():raise ValueError('Invalid feature vector')
    value=float(((x-c['mean'])/c['std'])@c['weights']+c['bias'])
    return float(1/(1+math.exp(-max(-80,min(80,value)))))


def analyse_video(input_path,model_path,spine_model,request,*,time_limit=170.):
    import cv2,mediapipe as mp
    if not math.isfinite(time_limit) or time_limit<=0:raise ValueError('Positive finite time limit required')
    if request.get('side_view') is not True or request.get('stationary_camera') is not True:raise ValueError('Confirm stationary side-view recording')
    start=time.monotonic();root=Path(model_path).parent;paths={k:root/('teng_'+k+'_candidate.json') for k in ['heel','back']};candidates={k:load_candidate(p,4 if k=='heel' else 32) for k,p in paths.items()};digest=hashlib.sha256()
    for kind,path in paths.items():
        if hashlib.sha256(path.read_bytes()).hexdigest()!=EXPECTED_CANDIDATE_SHA256[kind]:raise ValueError('Teng candidate differs from evaluated version')
    with Path(input_path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):
            digest.update(chunk)
            if time.monotonic()-start>time_limit:raise TimeoutError('Teng review deadline')
    cap=cv2.VideoCapture(str(input_path));decoded=0;samples=[];events=[]
    opts=mp.tasks.vision.PoseLandmarkerOptions(base_options=mp.tasks.BaseOptions(model_asset_path=str(model_path)),running_mode=mp.tasks.vision.RunningMode.IMAGE,num_poses=2,min_pose_detection_confidence=.5,min_pose_presence_confidence=.5,output_segmentation_masks=True)
    try:
        if not cap.isOpened():raise ValueError('Video could not be opened')
        fps=cap.get(cv2.CAP_PROP_FPS)
        if not math.isfinite(fps) or not 1<=fps<=240:raise ValueError('Unsupported video frame rate')
        step=max(1,round(fps/5))
        with MediaPipeLandmarker(model_path,'image') as guard,SpineLandmarker(spine_model) as spine,mp.tasks.vision.PoseLandmarker.create_from_options(opts) as segmenter:
            while True:
                if time.monotonic()-start>time_limit or decoded>=30000:raise TimeoutError('Teng review frame/time limit')
                ok,image=cap.read()
                if not ok:break
                index=decoded;decoded+=1
                if index%step:continue
                h,w=image.shape[:2]
                if max(h,w)>1280:image=cv2.resize(image,(round(w*1280/max(h,w)),round(h*1280/max(h,w))))
                rgb=cv2.cvtColor(image,cv2.COLOR_BGR2RGB);gd=guard.detect_rgb(rgb);sp=spine.predict_bgr(image,gd);sample=dict(timestamp_ms=round(index/fps*1000),frame_index=index,heel_score=None,back_score=None,status='unavailable')
                try:
                    features,selection=derive(gd,sp);sample.update(heel_score=probability(features[2:6],candidates['heel']),status='heel_only');result=segmenter.detect(mp.Image(image_format=mp.ImageFormat.SRGB,data=rgb))
                    if len(result.pose_landmarks)!=1 or not result.segmentation_masks or len(result.segmentation_masks)!=1:raise ValueError('single_person_mask_unavailable')
                    mask=np.asarray(result.segmentation_masks[0].numpy_view());mask=mask[:,:,0] if mask.ndim==3 and mask.shape[2]==1 else mask
                    if mask.shape!=image.shape[:2] or not np.isfinite(mask).all():raise ValueError('invalid_person_mask')
                    low,high=[tuple(sp['landmarks'][i][key] for key in ['x_px','y_px']) for i in [19,18]];profile,_=outline(mask>=.5,low,high,1 if selection['facing_estimate']=='left' else -1);sample.update(back_score=probability(profile,candidates['back']),status='measured')
                except ValueError as exc:sample['reason']=str(exc)
                samples.append(sample)
                for kind in ['heel','back']:
                    value=sample[kind+'_score']
                    if value is not None and value>=.5:
                        t=index/fps;events.append(dict(type='teng_'+kind+'_image_match',start_s=t,end_s=t,peak_s=t,score=value,message='Sample resembles Teng '+('Bad Heel' if kind=='heel' else 'Bad Back')+' images. Replay to inspect; not a confirmed fault or an event boundary.'))
    finally:cap.release()
    if not samples:raise ValueError('No decoded samples')
    measured=sum(s['status']=='measured' for s in samples)
    return dict(status='review_flags' if events else 'no_review_flag' if measured else 'unable_to_assess',reason=None if measured or events else 'no_measurable_samples',accuracy='not_validated',review_id=request['review_id'],measurement_basis='teng_native_image_label_match',source_sha256=digest.hexdigest(),candidate_sha256={k:hashlib.sha256(p.read_bytes()).hexdigest() for k,p in paths.items()},events=events,samples=samples,measurable_samples=measured,total_samples=len(samples),tracking=dict(decoded_frames=decoded,sampled_frames=len(samples),fps=fps,sampling_step=step),elapsed_s=time.monotonic()-start,native_test=dict(samples=969,unknown=50,heel_F1_measurable=.8402366863905325,back_F1_measurable=.9032258064516129),limitations=['Scores are uncalibrated Teng author-folder matches, not probabilities of a technique fault.','Native test participant overlap is unknown; video transfer and event timing are unvalidated.','Silhouette follows clothing, not the anatomical spine. Missing measurements are unknown.','Frames can match both labels; author folders do not label all coexisting faults.','Teng has no inward-knee category. Standing, turning and other movements can produce misleading matches.','Individual sampled matches have no verified onset, offset or duration; no match does not prove good form.'])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['input','model','spine-model','calibration','output']:parser.add_argument('--'+name,required=True)
    parser.add_argument('--time-limit',type=float,default=170.);args=parser.parse_args()
    from .heel_web import validate_teng_request
    request=json.loads(Path(args.calibration).read_text(encoding='utf-8'));checked=validate_teng_request({k:v for k,v in request.items() if k!='review_id'});checked['review_id']=request['review_id'];output=Path(args.output)
    if output.exists():raise FileExistsError('Fresh output required')
    report=analyse_video(args.input,args.model,args.spine_model,checked,time_limit=args.time_limit)
    with output.open('x',encoding='utf-8') as stream:json.dump(report,stream,indent=2,allow_nan=False)
if __name__=='__main__':main()
