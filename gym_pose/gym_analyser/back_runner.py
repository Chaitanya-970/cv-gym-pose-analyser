"""Bounded optional spine review using explicit local research models."""
import argparse,hashlib,json,math,time
from pathlib import Path
from .auxiliary_pose import MediaPipeLandmarker
from .spine_pose import SpineLandmarker
from .back_rounding import assess_back_rounding
from .heel_runner import reference_timestamps
from .heel_web import validate_back_request


def analyse_video(input_path,model_path,spine_model,request,*,time_limit=170.):
    import cv2
    if not math.isfinite(time_limit) or time_limit<=0:raise ValueError('Positive finite time limit required')
    started=time.monotonic();digest=hashlib.sha256()
    with Path(input_path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):
            digest.update(chunk)
            if time.monotonic()-started>time_limit:raise TimeoutError('Spine review deadline reached')
    cap=cv2.VideoCapture(str(input_path));frames=[];decoded=0
    try:
        if not cap.isOpened():raise ValueError('Video could not be opened')
        fps=cap.get(cv2.CAP_PROP_FPS)
        if not math.isfinite(fps) or not 1<=fps<=240:raise ValueError('Unsupported video frame rate')
        step=max(1,round(fps/5))
        with MediaPipeLandmarker(model_path,'video',smooth_video=True) as detector, SpineLandmarker(spine_model) as spine:
            while True:
                if time.monotonic()-started>time_limit or decoded>=30000:raise TimeoutError('Spine review frame/time limit reached')
                ok,img=cap.read()
                if not ok:break
                index=decoded;decoded+=1;t=round(index/fps*1000)
                guard=detector.detect_rgb(cv2.cvtColor(img,cv2.COLOR_BGR2RGB),t)
                if index%step==0:
                    result=spine.predict_bgr(img,guard);result['frame_index']=index;frames.append(result)
    finally:cap.release()
    if not frames:raise ValueError('No decoded video frames')
    calibration=dict(source_sha256=digest.hexdigest(),reference_timestamps_ms=reference_timestamps(frames,request['reference_start_s'],request['reference_end_s'],fps),
        side_view=request['side_view'],stationary_camera=request['stationary_camera'],standing_reference_confirmed=request['standing_still'],
        same_facing_direction=request['same_facing_direction'],facing=request['facing'],human_evidence_id=request['review_id'])
    report=assess_back_rounding(frames,source_sha256=digest.hexdigest(),calibration=calibration)
    report.update(review_id=request['review_id'],tracking=dict(decoded_frames=decoded,sampled_frames=len(frames),sampling_step=step,fps=fps),
        elapsed_s=time.monotonic()-started,spine_model_sha256=hashlib.sha256(Path(spine_model).read_bytes()).hexdigest(),
        scope='Relative estimated lumbar curvature, not confirmed back-rounding accuracy. Turning, occlusion and model error can invalidate measurements.')
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['input','model','spine-model','calibration','output']:parser.add_argument('--'+name,required=True)
    parser.add_argument('--time-limit',type=float,default=170.)
    args=parser.parse_args()
    request=json.loads(Path(args.calibration).read_text(encoding='utf-8'))
    checked=validate_back_request({k:v for k,v in request.items() if k!='review_id'});checked['review_id']=request['review_id']
    output=Path(args.output)
    if output.exists():raise FileExistsError('Fresh output required')
    report=analyse_video(args.input,args.model,args.spine_model,checked,time_limit=args.time_limit)
    with output.open('x',encoding='utf-8') as stream:json.dump(report,stream,indent=2,allow_nan=False)

if __name__=='__main__':main()
