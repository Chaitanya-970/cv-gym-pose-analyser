"""Experimental front-view inward-knee tracking, not a 3D valgus diagnosis."""
from dataclasses import dataclass,asdict
from math import hypot,isfinite
from numbers import Real
from statistics import median


def _finite(v):return isinstance(v,Real) and not isinstance(v,bool) and isfinite(v)


@dataclass(frozen=True)
class KneeTrackingSettings:
    inward_fraction:float=.06
    min_duration_s:float=.3
    max_gap_s:float=.25
    max_reference_spread_fraction:float=.03
    max_ankle_drift_fraction:float=.25
    def __post_init__(self):
        if any(not _finite(v) or v<=0 for v in asdict(self).values()):raise ValueError('Finite positive settings required')


def _geometry(frame):
    if frame.get('landmark_schema')!='mediapipe33' or frame.get('person_count')!=1 or len(frame.get('poses',[]))!=1:return None
    width,height=frame.get('width'),frame.get('height')
    if any(isinstance(v,bool) or not isinstance(v,int) or v<=0 for v in [width,height]):return None
    pose=frame['poses'][0]
    if len(pose)!=33:return None
    pts={}
    for i in [23,24,25,26,27,28]:
        p=pose[i];x,y,v,c=(p.get(k) for k in ['x_px','y_px','visibility','presence'])
        if not all(_finite(z) for z in [x,y,v,c]) or not (0<=x<width and 0<=y<height and .5<=v<=1 and .5<=c<=1):return None
        pts[i]=(x,y)
    hip_left,hip_right=pts[23],pts[24];hip_width=hypot(hip_right[0]-hip_left[0],hip_right[1]-hip_left[1])
    if hip_width<15 or abs(hip_right[1]-hip_left[1])>.4*hip_width:return None
    center=((hip_left[0]+hip_right[0])/2,(hip_left[1]+hip_right[1])/2)
    result=dict(hip_width_px=hip_width,hip_order=hip_right[0]-hip_left[0],ankles=[pts[27],pts[28]])
    for side,h,k,a in [('left',23,25,27),('right',24,26,28)]:
        hip,knee,ankle=pts[h],pts[k],pts[a];dx,dy=ankle[0]-hip[0],ankle[1]-hip[1];length=hypot(dx,dy)
        if length<30:return None
        normal=(-dy/length,dx/length)
        if normal[0]*(center[0]-hip[0])+normal[1]*(center[1]-hip[1])<0:normal=(-normal[0],-normal[1])
        result[side]=dict(offset_px=(knee[0]-hip[0])*normal[0]+(knee[1]-hip[1])*normal[1],length_px=length)
    return result


def assess_knee_tracking(frames,*,source_sha256,calibration,settings=None):
    settings=settings or KneeTrackingSettings();frames=list(frames)
    if not isinstance(source_sha256,str) or len(source_sha256)!=64 or any(c not in '0123456789abcdef' for c in source_sha256):raise ValueError('Expected source SHA256')
    times=[f.get('timestamp_ms') for f in frames]
    if any(isinstance(t,bool) or not isinstance(t,int) or t<0 for t in times) or any(a>=b for a,b in zip(times,times[1:])):raise ValueError('Strictly increasing integer milliseconds required')
    out=dict(status='unable_to_assess',reason=None,accuracy='not_validated',source_sha256=source_sha256,
             settings=asdict(settings),calibration=calibration,events=[],samples=[],total_samples=len(frames),measurable_samples=0)
    def fail(reason):out['reason']=reason;return out
    if not frames:return fail('no_frames')
    if not isinstance(calibration,dict) or calibration.get('source_sha256')!=source_sha256:return fail('reference_source_mismatch')
    for field in ['front_view','stationary_camera','standing_reference_confirmed']:
        if calibration.get(field) is not True:return fail(field+'_unconfirmed')
    if not isinstance(calibration.get('human_evidence_id'),str) or not calibration['human_evidence_id'].strip():return fail('missing_human_reference_evidence')
    if len({(f.get('width'),f.get('height')) for f in frames})!=1:return fail('dimensions_changed')
    refs=calibration.get('reference_timestamps_ms')
    if not isinstance(refs,list) or len(refs)<3 or any(isinstance(t,bool) or not isinstance(t,int) for t in refs) or len(set(refs))!=len(refs):return fail('need_three_reference_samples')
    by_time=dict(zip(times,frames))
    if any(t not in by_time for t in refs):return fail('reference_sample_missing')
    reference=[_geometry(by_time[t]) for t in refs]
    if any(p is None for p in reference):return fail('reference_tracking_unavailable')
    hip_width=median(p['hip_width_px'] for p in reference)
    ankles=[tuple(median(p['ankles'][i][j] for p in reference) for j in range(2)) for i in range(2)]
    baseline={side:dict(offset_px=median(p[side]['offset_px'] for p in reference),length_px=median(p[side]['length_px'] for p in reference)) for side in ['left','right']}
    spread=max(abs(p[side]['offset_px']-baseline[side]['offset_px'])/baseline[side]['length_px'] for p in reference for side in baseline)
    anchor_spread=max(hypot(p['ankles'][i][0]-ankles[i][0],p['ankles'][i][1]-ankles[i][1])/hip_width for p in reference for i in range(2))
    if spread>settings.max_reference_spread_fraction or anchor_spread>settings.max_reference_spread_fraction:return fail('reference_unstable')
    out['reference']=dict(hip_width_px=hip_width,ankles_px=ankles,legs=baseline,spread_fraction=spread,anchor_spread_fraction=anchor_spread)
    active={'left':[],'right':[]}
    def flush(side):
        group=active[side]
        if len(group)>=2 and (group[-1]['timestamp_ms']-group[0]['timestamp_ms'])/1000>=settings.min_duration_s:
            peak=max(group,key=lambda q:q[side+'_inward_fraction'])
            out['events'].append(dict(type='possible_inward_knee_tracking',side=side,status='review_flag',
                start_s=group[0]['timestamp_ms']/1000,end_s=group[-1]['timestamp_ms']/1000,peak_s=peak['timestamp_ms']/1000,
                peak_inward_fraction=peak[side+'_inward_fraction'],accuracy='not_validated',
                message='Projected knee moved inward relative to the reviewed standing reference; replay to check. Not a 3D valgus diagnosis.'))
        group.clear()
    for frame in frames:
        t=frame['timestamp_ms'];geometry=_geometry(frame);sample=dict(timestamp_ms=t,status='unavailable',reason='joint_tracking_unavailable')
        if geometry is not None:
            if not .65<=geometry['hip_width_px']/hip_width<=1.5 or geometry['hip_order']*reference[0]['hip_order']<=0:sample['reason']='view_or_body_orientation_changed'
            elif any(hypot(p[0]-q[0],p[1]-q[1])/hip_width>settings.max_ankle_drift_fraction for p,q in zip(geometry['ankles'],ankles)):sample['reason']='stance_or_camera_changed'
            else:
                sample.update(status='measured',reason=None);out['measurable_samples']+=1
                for side in baseline:sample[side+'_inward_fraction']=(geometry[side]['offset_px']-baseline[side]['offset_px'])/baseline[side]['length_px']
        out['samples'].append(sample)
        for side in active:
            if sample['status']=='measured' and sample[side+'_inward_fraction']>=settings.inward_fraction:
                if active[side] and (t-active[side][-1]['timestamp_ms'])/1000>settings.max_gap_s:flush(side)
                active[side].append(sample)
            else:flush(side)
    for side in active:flush(side)
    if not out['measurable_samples']:return fail('no_measurable_samples')
    out.update(status='review_flags' if out['events'] else 'no_review_flag',reason=None)
    out['events'].sort(key=lambda e:(e['start_s'],e['side']))
    out['interpretation']='Experimental 2D review flags, not anatomical valgus or safe/unsafe grading. No flag does not establish absence; gaps and out-of-plane motion are not assessed.'
    return out
