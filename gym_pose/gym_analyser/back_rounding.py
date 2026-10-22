"""Experimental relative lumbar curvature review; not a clinical form grade.

Uses estimated L5/L3/T12 points, not shoulder-hip tilt. Thresholds are
engineering settings with no established diagnostic sensitivity/specificity.
"""
from dataclasses import dataclass, asdict
from math import hypot, isfinite
from numbers import Real
from statistics import median


def _finite(v):
    return isinstance(v, Real) and not isinstance(v, bool) and isfinite(v)


@dataclass(frozen=True)
class BackRoundingSettings:
    posterior_change_fraction: float = .10
    min_duration_s: float = .30
    max_gap_s: float = .25
    max_reference_spread_fraction: float = .03

    def __post_init__(self):
        if any(not _finite(v) or v <= 0 for v in asdict(self).values()):
            raise ValueError('Finite positive settings required')


def _geometry(frame):
    if not isinstance(frame, dict) or frame.get('landmark_schema') != 'spinepose37' or frame.get('person_count') != 1:
        return None
    width, height = frame.get('width'), frame.get('height')
    if any(isinstance(v, bool) or not isinstance(v, int) or v <= 0 for v in (width, height)):
        return None
    points = frame.get('landmarks')
    if not isinstance(points, list) or len(points) != 37:
        return None
    coords = []
    for i in (26, 27, 28):  # estimated L5, L3 and T12/L1
        p = points[i]
        if not isinstance(p, dict) or p.get('id') != i:
            return None
        x, y, score = (p.get(k) for k in ('x_px', 'y_px', 'raw_score'))
        if not all(_finite(v) for v in (x, y, score)) or not (0 <= x < width and 0 <= y < height and score > 0):
            return None
        # Raw SimCC score is deliberately not interpreted as probability.
        coords.append((x, y))
    low, middle, high = coords
    dx, dy = high[0]-low[0], high[1]-low[1]
    length = hypot(dx, dy)
    if length < 15:
        return None
    axis = (dx/length, dy/length)
    projection = ((middle[0]-low[0])*axis[0]+(middle[1]-low[1])*axis[1])/length
    if not .1 <= projection <= .9:
        return None
    normal = (-axis[1], axis[0])
    offset = ((middle[0]-low[0])*normal[0]+(middle[1]-low[1])*normal[1])/length
    if abs(offset) > .6:
        return None
    return dict(length_px=length, axis=axis, normal=normal, offset_fraction=offset)


def assess_back_rounding(frames, *, source_sha256, calibration, settings=None):
    settings = settings or BackRoundingSettings()
    frames = list(frames)
    if not isinstance(source_sha256, str) or len(source_sha256) != 64 or any(c not in '0123456789abcdef' for c in source_sha256):
        raise ValueError('Expected source SHA256')
    times = [f.get('timestamp_ms') for f in frames]
    if any(isinstance(t, bool) or not isinstance(t, int) or t < 0 for t in times) or any(a >= b for a, b in zip(times, times[1:])):
        raise ValueError('Strictly increasing integer milliseconds required')
    out = dict(status='unable_to_assess', reason=None, accuracy='not_validated',
               measurement_basis='estimated_relative_lumbar_curvature', source_sha256=source_sha256,
               settings=asdict(settings), calibration=calibration, events=[], samples=[],
               total_samples=len(frames), measurable_samples=0,
               limitations=['Inferred spinal points under clothing are not anatomical ground truth.',
                            'A standing reference does not establish a neutral or ideal spine.',
                            'No flag does not establish good form or exclude rounding.',
                            'Facing direction and stable side view require human confirmation; turning can invalidate results.'])
    def fail(reason):
        out['reason'] = reason
        return out
    if not frames:
        return fail('no_frames')
    if not isinstance(calibration, dict) or calibration.get('source_sha256') != source_sha256:
        return fail('reference_source_mismatch')
    for field in ('side_view', 'stationary_camera', 'standing_reference_confirmed', 'same_facing_direction'):
        if calibration.get(field) is not True:
            return fail(field+'_unconfirmed')
    facing = calibration.get('facing')
    if facing not in ('left', 'right'):
        return fail('facing_direction_required')
    if not isinstance(calibration.get('human_evidence_id'), str) or not calibration['human_evidence_id'].strip():
        return fail('missing_human_reference_evidence')
    if len({(f.get('width'), f.get('height')) for f in frames}) != 1:
        return fail('dimensions_changed')
    refs = calibration.get('reference_timestamps_ms')
    if not isinstance(refs, list) or len(refs) < 3 or any(isinstance(t, bool) or not isinstance(t, int) for t in refs) or len(set(refs)) != len(refs):
        return fail('need_three_reference_samples')
    by_time = dict(zip(times, frames))
    if any(t not in by_time for t in refs):
        return fail('reference_sample_missing')
    reference = [_geometry(by_time[t]) for t in refs]
    if any(g is None for g in reference):
        return fail('reference_tracking_unavailable')
    anchor = reference[0]
    if abs(anchor['normal'][0]) < .5:
        return fail('reference_not_upright')
    posterior_x = 1 if facing == 'left' else -1
    sign = 1 if anchor['normal'][0]*posterior_x > 0 else -1
    baseline = median(g['offset_fraction']*sign for g in reference)
    length = median(g['length_px'] for g in reference)
    if any(abs(g['offset_fraction']*sign-baseline) > settings.max_reference_spread_fraction or
           abs(g['length_px']/length-1) > .1 or
           sum(a*b for a,b in zip(g['axis'], anchor['axis'])) < .98 for g in reference):
        return fail('reference_unstable')
    out['reference'] = dict(offset_fraction=baseline, lumbar_chord_px=length,
                            standing_is_not_neutral_truth=True)
    active = []
    def flush():
        if len(active) >= 2 and (active[-1]['timestamp_ms']-active[0]['timestamp_ms'])/1000 >= settings.min_duration_s:
            peak = max(active, key=lambda p: p['posterior_change_fraction'])
            out['events'].append(dict(type='possible_back_rounding', status='review_flag',
                start_s=active[0]['timestamp_ms']/1000, end_s=active[-1]['timestamp_ms']/1000,
                peak_s=peak['timestamp_ms']/1000, peak_posterior_change_fraction=peak['posterior_change_fraction'],
                accuracy='not_validated', message='Estimated lumbar curve moved posteriorly relative to your standing reference. Replay to review; this is not a diagnosis or confirmed technique fault.'))
        active.clear()
    for f in frames:
        t = f['timestamp_ms']; g = _geometry(f)
        sample = dict(timestamp_ms=t, status='unavailable', reason='spine_tracking_unavailable')
        if g is not None:
            alignment = sum(a*b for a,b in zip(g['axis'], anchor['axis']))
            if not .6 <= g['length_px']/length <= 1.5 or alignment < .34:
                sample['reason'] = 'geometry_or_orientation_changed'
            else:
                sample.update(status='measured', reason=None,
                              posterior_change_fraction=g['offset_fraction']*sign-baseline)
                out['measurable_samples'] += 1
        out['samples'].append(sample)
        if sample['status'] == 'measured' and sample['posterior_change_fraction'] >= settings.posterior_change_fraction:
            if active and (t-active[-1]['timestamp_ms'])/1000 > settings.max_gap_s:
                flush()
            active.append(sample)
        else:
            flush()
    flush()
    if not out['measurable_samples']:
        return fail('no_measurable_samples')
    out.update(status='review_flags' if out['events'] else 'no_review_flag', reason=None)
    return out
