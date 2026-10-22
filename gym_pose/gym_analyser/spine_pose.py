"""Optional source-local SpinePose37 adapter for non-commercial research.

Uses the author's optional CC-BY-NC inference package and explicit local weights.
No author code/weights are vendored here. Scores are raw SimCC peaks, NOT
probabilities or clinical confidence. Anatomical/fault accuracy is unvalidated.
"""
from pathlib import Path
from math import isfinite
from numbers import Real

SPINE_CHAIN=(19,26,27,28,29,30,18)  # sacral hip,L5,L3,T12/L1,T8,T3,neck


def _finite(value):return isinstance(value,Real) and not isinstance(value,bool) and isfinite(value)


def serialize_spine(points,scores,width,height):
    if any(isinstance(v,bool) or not isinstance(v,int) or v<=0 for v in [width,height]):raise ValueError('Positive integer dimensions required')
    if len(points)!=37 or len(scores)!=37 or any(len(p)!=2 for p in points):raise ValueError('Expected SpinePose37 layout')
    result=[]
    for index,(point,score) in enumerate(zip(points,scores)):
        x,y=point
        if not all(_finite(v) for v in [x,y,score]):raise ValueError('Nonfinite spine output')
        inside=0<=x<width and 0<=y<height
        result.append(dict(id=index,x_px=float(x),y_px=float(y),raw_score=float(score),in_frame=bool(inside),usable=bool(inside and score>0)))
    return result


def body_proposal(guard,width,height):
    if not isinstance(guard,dict) or guard.get('landmark_schema')!='mediapipe33':return None
    if guard.get('person_count')!=1 or len(guard.get('poses',[]))!=1:return None
    if (guard.get('width'),guard.get('height'))!=(width,height):raise ValueError('Guard/image dimensions differ')
    pose=guard['poses'][0]
    if len(pose)!=33:return None
    points=[]
    for p in pose:
        values=[p.get(k) for k in ['x_px','y_px','visibility','presence']]
        if not all(_finite(v) for v in values):continue
        x,y,visibility,presence=values
        if 0<=x<width and 0<=y<height and .5<=visibility<=1 and .5<=presence<=1:points.append((x,y))
    if len(points)<10:return None
    bbox=[min(p[0] for p in points),min(p[1] for p in points),max(p[0] for p in points),max(p[1] for p in points)]
    return bbox if bbox[2]-bbox[0]>=20 and bbox[3]-bbox[1]>=50 else None


class SpineLandmarker:
    """CPU top-down model with mandatory independent person/body proposal."""
    def __init__(self,model_path):
        self.model_path=Path(model_path).resolve();self._model=None
        if not self.model_path.is_file():raise FileNotFoundError('Local spine model missing; automatic download disabled')
    def __enter__(self):
        if self._model is not None:raise RuntimeError('Spine context already entered')
        try:from spinepose.tools.pose_estimation import RTMPose
        except ImportError as exc:raise RuntimeError('Optional SpinePose runtime unavailable') from exc
        self._model=RTMPose(onnx_model=str(self.model_path),model_input_size=(192,256),hardware_acceleration=False)
        return self
    def predict_bgr(self,image,guard):
        if self._model is None:raise RuntimeError('Use spine model inside its context')
        shape=getattr(image,'shape',())
        if len(shape)!=3 or shape[2]!=3 or str(getattr(image,'dtype',''))!='uint8':raise ValueError('Expected original BGR uint8 image')
        height,width=int(shape[0]),int(shape[1])
        if width<=0 or height<=0:raise ValueError('Image dimensions must be positive')
        bbox=body_proposal(guard,width,height)
        result=dict(landmark_schema='spinepose37',measurement_basis='model_estimate',accuracy='not_validated',
                    score_basis='uncalibrated_simcc_peak',width=width,height=height,
                    timestamp_ms=guard.get('timestamp_ms') if isinstance(guard,dict) else None,
                    person_count=guard.get('person_count') if isinstance(guard,dict) else None,landmarks=[],reason=None)
        if bbox is None:result['reason']='single_person_body_proposal_unavailable';return result
        points,scores=self._model(image,bboxes=[bbox])
        if len(points)!=1 or len(scores)!=1:raise ValueError('Unexpected top-down model batch')
        result.update(landmarks=serialize_spine(points[0],scores[0],width,height),bbox=bbox,
                      bbox_basis='independent_model_body_proposal',reason='single_person')
        return result
    def __exit__(self,exc_type,exc,traceback):
        model=self._model;self._model=None
        if model is not None:model.close()
