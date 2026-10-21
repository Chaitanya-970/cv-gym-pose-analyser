"""Experimental heel-rise review flags; requires verified planted-foot reference.

Settings are engineering heuristics, not universal technique/safety thresholds.
No foot-contact, clinical, or validated fault-accuracy claim is made.
"""
from dataclasses import dataclass, asdict
from math import hypot, isfinite
from numbers import Real
from statistics import median
from .auxiliary_pose import FOOT_INDICES, visible_foot_pairs


def _number(x):
    return isinstance(x, Real) and not isinstance(x, bool) and isfinite(x)


@dataclass(frozen=True)
class HeelLiftSettings:
    rise_fraction: float = .12
    min_duration_s: float = .3
    max_gap_s: float = .25
    max_toe_drift_fraction: float = .3
    max_reference_spread_fraction: float = .04
    min_foot_pixels: float = 15.

    def __post_init__(self):
        if any(not _number(v) or v <= 0 for v in asdict(self).values()):
            raise ValueError("Settings must be finite positive numbers")


def assess_heel_lift(frames, *, source_sha256, calibration, settings=None):
    """Assess one selected side, without changing repetition counts.

    Frames are MediaPipeLandmarker VIDEO outputs. Calibration contains side,
    reference_timestamps_ms (>=3), source_sha256, suitable_side_view,
    stationary_camera, planted_reference_confirmed and human_evidence_id.
    Suitability/contact confirmations must come from actual review, never inference.
    All outputs remain experimental, including when no review flag was raised.
    """
    settings = settings or HeelLiftSettings()
    if not isinstance(source_sha256, str) or len(source_sha256) != 64 or any(c not in '0123456789abcdef' for c in source_sha256):
        raise ValueError("Expected source SHA256")
    frames = list(frames)
    timestamps = [f.get('timestamp_ms') for f in frames]
    if any(isinstance(t, bool) or not isinstance(t, int) or t < 0 for t in timestamps) or any(a >= b for a,b in zip(timestamps,timestamps[1:])):
        raise ValueError("Strictly increasing nonnegative integer timestamps required")
    output = dict(status='unable_to_assess', reason=None, source_sha256=source_sha256,
                  accuracy='not_validated', settings=asdict(settings), events=[], samples=[],
                  total_samples=len(frames), measurable_samples=0, calibration=calibration)
    def fail(reason):
        output['reason'] = reason
        return output
    if not frames:
        return fail('no_frames')
    if not isinstance(calibration, dict):
        return fail('missing_calibration')
    if calibration.get('source_sha256') != source_sha256:
        return fail('reference_source_mismatch')
    side = calibration.get('side')
    if side not in FOOT_INDICES:
        return fail('missing_selected_side')
    dimensions = [(f.get('width'),f.get('height')) for f in frames]
    if len(set(dimensions)) != 1:
        return fail('dimensions_changed')
    width,height = dimensions[0]
    if any(isinstance(v,bool) or not isinstance(v,int) or v <= 0 for v in [width,height]):
        return fail('invalid_dimensions')
    def points(frame):
        poses = frame.get('poses',[])
        if frame.get('landmark_schema') != 'mediapipe33' or frame.get('person_count') != 1 or len(poses) != 1:
            return None
        try:
            if side not in visible_foot_pairs(poses[0],width,height):
                return None
            heel,toe = (poses[0][i] for i in FOOT_INDICES[side])
            return (heel['x_px'],heel['y_px']), (toe['x_px'],toe['y_px'])
        except (ValueError,TypeError,KeyError):
            return None
    refs = calibration.get('reference_timestamps_ms')
    if not isinstance(refs,list) or len(refs) < 3 or any(isinstance(t,bool) or not isinstance(t,int) for t in refs) or len(set(refs)) != len(refs):
        return fail('need_three_distinct_reference_samples')
    by_time = dict(zip(timestamps,frames))
    if any(t not in by_time for t in refs):
        return fail('reference_sample_missing')
    reference = [points(by_time[t]) for t in refs]
    if any(p is None for p in reference):
        return fail('reference_foot_unavailable')
    heels = [p[0] for p in reference]; toes = [p[1] for p in reference]
    heel = tuple(median(p[i] for p in heels) for i in range(2))
    toe = tuple(median(p[i] for p in toes) for i in range(2))
    dx,dy = toe[0]-heel[0],toe[1]-heel[1]
    length = hypot(dx,dy)
    if length < settings.min_foot_pixels or abs(dx)/length < .5:
        return fail('reference_foot_too_small_or_foreshortened')
    spread = max(hypot(p[i][0]-q[0],p[i][1]-q[1])/length for p in reference for i,q in [(0,heel),(1,toe)])
    output['reference'] = dict(heel_px=heel,toe_px=toe,foot_length_px=length,spread_fraction=spread)
    if spread > settings.max_reference_spread_fraction:
        return fail('reference_unstable')
    for field, reason in [('suitable_side_view','side_view_unconfirmed'),
                          ('stationary_camera','stationary_camera_unconfirmed'),
                          ('planted_reference_confirmed','planted_reference_unconfirmed')]:
        if calibration.get(field) is not True:
            return fail(reason)
    if not isinstance(calibration.get('human_evidence_id'),str) or not calibration['human_evidence_id'].strip():
        return fail('missing_human_reference_evidence')
    normal = (dy/length,-dx/length)
    if normal[1] > 0:
        normal = tuple(-v for v in normal)
    offset = (heel[0]-toe[0],heel[1]-toe[1])
    active=[]
    def flush():
        if len(active) >= 2 and (active[-1]['timestamp_ms']-active[0]['timestamp_ms'])/1000 >= settings.min_duration_s:
            peak=max(active,key=lambda x:x['rise_fraction'])
            output['events'].append(dict(type='possible_heel_lift',status='review_flag',side=side,
                start_s=active[0]['timestamp_ms']/1000,end_s=active[-1]['timestamp_ms']/1000,
                peak_s=peak['timestamp_ms']/1000,peak_rise_fraction=peak['rise_fraction'],
                message='Heel landmark rose relative to the reviewed planted-foot reference; replay to confirm.',
                accuracy='not_validated'))
        active.clear()
    for frame in frames:
        t=frame['timestamp_ms']; p=points(frame)
        sample=dict(timestamp_ms=t,status='unavailable',reason='foot_tracking_unavailable')
        if p is not None:
            h,q=p; current_length=hypot(h[0]-q[0],h[1]-q[1])
            drift=hypot(q[0]-toe[0],q[1]-toe[1])/length
            if drift > settings.max_toe_drift_fraction:
                sample['reason']='toe_moved_or_camera_changed'
            elif (q[0]-h[0])*dx <= 0 or not .65 <= current_length/length <= 1.4 or abs(q[0]-h[0])/length < .4:
                sample['reason']='foot_geometry_changed'
            else:
                rise=((h[0]-q[0]-offset[0])*normal[0]+(h[1]-q[1]-offset[1])*normal[1])/length
                sample.update(status='measured',reason=None,rise_fraction=rise,toe_drift_fraction=drift)
                output['measurable_samples']+=1
        output['samples'].append(sample)
        if sample['status']=='measured' and sample['rise_fraction'] >= settings.rise_fraction:
            if active and (t-active[-1]['timestamp_ms'])/1000 > settings.max_gap_s:
                flush()
            active.append(sample)
        else:
            flush()
    flush()
    if output['measurable_samples'] == 0:
        return fail('no_measurable_samples')
    output.update(status='review_flags' if output['events'] else 'no_review_flag',reason=None)
    output['interpretation']='Experimental sampled evidence only; no flag does not prove no heel lift. Unavailable intervals are not assessed.'
    return output
