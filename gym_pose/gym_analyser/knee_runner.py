"""Bounded optional front-view knee review CLI."""
import argparse,json,math
from pathlib import Path
from .heel_runner import analyse_video
from .heel_web import validate_knee_request

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['input','model','calibration','output']:parser.add_argument('--'+name,required=True)
    parser.add_argument('--time-limit',type=float,default=170.)
    args=parser.parse_args()
    if not math.isfinite(args.time_limit) or args.time_limit<=0:parser.error('Positive finite time limit required')
    request=json.loads(Path(args.calibration).read_text(encoding='utf-8'))
    checked=validate_knee_request({k:v for k,v in request.items() if k!='review_id'});checked['review_id']=request['review_id']
    output=Path(args.output)
    if output.exists():raise FileExistsError('Fresh output required')
    result=analyse_video(args.input,args.model,checked,time_limit=args.time_limit,assessment_kind='knee')
    with output.open('x',encoding='utf-8') as stream:json.dump(result,stream,indent=2,allow_nan=False)

if __name__=='__main__':main()
