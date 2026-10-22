"""Bounded optional heel-review worker; no YOLO counter or automatic downloads."""
import argparse
import hashlib
import json
import math
import time
from pathlib import Path
from .auxiliary_pose import MediaPipeLandmarker
from .heel_lift import assess_heel_lift


def analyse_video(input_path, model_path, request, *, time_limit=170.):
    import cv2
    start=time.monotonic()
    digest=hashlib.sha256()
    with Path(input_path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):
            digest.update(chunk)
            if time.monotonic()-start>time_limit:raise TimeoutError('Heel review deadline reached')
    cap=cv2.VideoCapture(str(input_path));frames=[];decoded=0
    try:
        if not cap.isOpened():raise ValueError('Video could not be opened')
        fps=cap.get(cv2.CAP_PROP_FPS)
        if not math.isfinite(fps) or not 1<=fps<=240:raise ValueError('Unsupported video frame rate')
        step=max(1,round(fps/5))
        with MediaPipeLandmarker(model_path,'video',smooth_video=True) as detector:
            while True:
                if time.monotonic()-start>time_limit or decoded>=30000:
                    raise TimeoutError('Heel review frame/time limit reached')
                ok,img=cap.read()
                if not ok:break
                index=decoded;decoded+=1;t=round(index/fps*1000)
                result=detector.detect_rgb(cv2.cvtColor(img,cv2.COLOR_BGR2RGB),t)
                if index%step==0:
                    result['frame_index']=index;frames.append(result)
    finally:cap.release()
    if not frames:raise ValueError('No decoded video frames')
    refs=[f['timestamp_ms'] for f in frames
          if request['reference_start_s']*1000-.5<=f['timestamp_ms']<=request['reference_end_s']*1000+.5]
    calibration=dict(source_sha256=digest.hexdigest(),reference_timestamps_ms=refs,
        suitable_side_view=request['side_view'],stationary_camera=request['stationary_camera'],
        planted_reference_confirmed=request['heels_flat'],human_evidence_id=request['review_id'])
    reports=[]
    for side in ['left','right']:
        reports.append(assess_heel_lift(frames,source_sha256=digest.hexdigest(),calibration=dict(calibration,side=side)))
    eligible=[q for q in reports if q['status']!='unable_to_assess']
    selected=min(eligible or reports,key=lambda q:q.get('reference',{}).get('spread_fraction',float('inf')))
    selected.update(review_id=request['review_id'],selected_side=selected['calibration']['side'],
        selection_basis='Lowest reference spread among assessable feet; no selection by flag count.',
        tracking=dict(decoded_frames=decoded,sampled_frames=len(frames),sampling_step=step,fps=fps,
                      processing='guarded_single_person_video'),
        other_side_reason=next(q['reason'] for q in reports if q is not selected),
        elapsed_s=time.monotonic()-start,
        limitations=['Experimental projected heel-rise review flags, not clinical or safe/unsafe grading.',
            'Reference confirmations are user declarations, not a validated fault benchmark.',
            'Sampled output can miss short events; temporal filtering can delay or suppress movement.',
            'Back rounding and inward-knee tracking are not assessed by this heel review.'])
    return selected


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['input','model','calibration','output']:parser.add_argument('--'+name,required=True)
    parser.add_argument('--time-limit',type=float,default=170.)
    args=parser.parse_args()
    if not math.isfinite(args.time_limit) or args.time_limit<=0:parser.error('Positive finite time limit required')
    from .heel_web import validate_request
    request=json.loads(Path(args.calibration).read_text(encoding='utf-8'))
    checked=validate_request({k:v for k,v in request.items() if k!='review_id'})
    checked['review_id']=request['review_id']
    output=Path(args.output)
    if output.exists():raise FileExistsError('Fresh output required')
    result=analyse_video(args.input,args.model,checked,time_limit=args.time_limit)
    with output.open('x',encoding='utf-8') as stream:json.dump(result,stream,indent=2,allow_nan=False)

if __name__=='__main__':main()
