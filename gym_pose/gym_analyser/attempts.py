"""Observed squat attempts, kept separate from the authoritative repetition counter."""
from collections import Counter
from typing import Optional
from .counter import CounterConfig, FrameResult


class AttemptTracker:
    """Observe every counter result in order; never infer motion across missing frames.

    Departure requires consecutive valid frames below the configured standing
    angle. This is an operational attempt definition, not a squat-form standard.
    Confirmed counter transitions always take precedence over departure debounce.
    """
    def __init__(self, config: CounterConfig, departure_frames: int = 2):
        if isinstance(departure_frames,bool) or not isinstance(departure_frames,int) or departure_frames < 1:
            raise ValueError('departure_frames must be a positive integer')
        self.config=config
        self.departure_frames=departure_frames
        self.events=[]
        self.active=None
        self.candidate=[]
        self.last=None
        self.last_valid_time=None
        self.last_count=0
        self.unarmed_frames=0
        self.opening_lowered_frames=0
        self.ever_armed=False
        self.finished=False

    @property
    def active_id(self) -> Optional[int]:
        return self.active['attempt_id'] if self.active else None

    def _include(self, frame):
        a=self.active
        a['observed_frames']+=1
        if frame.valid and frame.selected_side==a['side'] and frame.track_id==a['track_id']:
            a['valid_frames']+=1
            if frame.knee_angle_deg is not None:
                if frame.knee_angle_deg < a['min_knee_angle_deg']:
                    a['min_knee_angle_deg']=frame.knee_angle_deg
                    a['minimum_observed_frame']=frame.frame_index
            if frame.stage=='lowered':a['depth_confirmed']=True
        else:
            a['measurement_reasons'][frame.reason]+=1

    def _open(self, frames):
        start=frames[0]
        self.active={
            'attempt_id':len(self.events)+1,'start_frame':start.frame_index,
            'start_time_s':start.video_time_s,'confirmed_at_frame':frames[-1].frame_index,
            'track_id':start.track_id,'side':start.selected_side,
            'min_knee_angle_deg':start.knee_angle_deg,'minimum_observed_frame':start.frame_index,
            'depth_confirmed':False,'observed_frames':0,'valid_frames':0,
            'measurement_reasons':Counter(),
        }
        self.return_streak=0
        for frame in frames:self._include(frame)
        self.candidate=[]

    def _close(self, frame, reason, status='uncounted', termination=None):
        a=self.active
        if a is None:return
        if reason=='tracking_gap' and a['measurement_reasons']:
            dominant=max(a['measurement_reasons'],key=a['measurement_reasons'].get)
            if dominant=='multiple_subjects_ambiguous':reason='ambiguous_subjects'
            elif 'confidence' in dominant:reason='confidence_loss'
        messages={
            'completed_cycle':'A complete cycle met the configured counting thresholds.',
            'depth_threshold_not_reached':f'The observed knee angle did not reach the configured {self.config.down_angle:g}-degree depth threshold before returning to standing.',
            'depth_not_sustained':'The depth threshold was crossed briefly, but not for enough consecutive valid frames.',
            'confidence_loss':'Joint confidence remained too low to confirm one continuous cycle.',
            'ambiguous_subjects':'Multiple people made the subject ambiguous too long to confirm this cycle.',
            'tracking_gap':'Tracking or joint measurements were interrupted too long to confirm one continuous cycle.',
            'side_changed':'The observed body side changed before this cycle could be confirmed.',
            'subject_changed':'The observed subject changed before this cycle could be confirmed.',
            'video_ended_mid_attempt':'The video ended before a complete return to standing was confirmed.',
            'processing_failed':'Processing stopped before this attempt could be confirmed.',
        }
        a.update(status=status,reason=reason,explanation=messages[reason],
                 end_frame=frame.frame_index,end_time_s=frame.video_time_s,
                 valid_frame_fraction=a['valid_frames']/a['observed_frames'] if a['observed_frames'] else None)
        a['measurement_reasons']=dict(a['measurement_reasons'])
        a['secondary_reasons']=list(a['measurement_reasons'])
        if a['measurement_reasons']:
            readable=[]
            for code in a['measurement_reasons']:
                readable.append('ambiguous subjects' if code=='multiple_subjects_ambiguous' else 'low joint confidence' if 'confidence' in code else 'subject not detected' if code=='no_subject_detected' else code.replace('_',' '))
            a['explanation']+=' Measurement interruptions included: '+', '.join(readable)+'.'
        a['category']=('completed' if status=='counted' else 'movement_threshold' if reason.startswith('depth_') else 'stream_end' if reason=='video_ended_mid_attempt' else 'measurement')
        a['return_confirmed']=status=='counted' or reason in {'depth_threshold_not_reached','depth_not_sustained'}
        if termination:a['termination']=termination
        a['interpretation']='An uncounted attempt is not a judgement of incorrect or unsafe form.'
        self.events.append(a)
        self.active=None
        self.candidate=[]
        self.return_streak=0

    def observe(self, frame: FrameResult, armed_side: Optional[str]):
        if self.finished:raise RuntimeError('Cannot observe after finalization')
        if self.last and (frame.frame_index<=self.last.frame_index or frame.video_time_s<self.last.video_time_s):
            raise ValueError('Attempt observations must be ordered, with unique increasing frame indices')
        delta=frame.completed_reps-self.last_count
        if delta not in {0,1}:raise ValueError('Observe every counter frame; counts must advance by at most one')
        if self.last_valid_time is not None and frame.video_time_s-self.last_valid_time>self.config.max_gap_seconds:
            self._close(frame,'tracking_gap')
            self.candidate=[]
        self.last=frame
        self.last_count=frame.completed_reps
        if frame.valid:self.last_valid_time=frame.video_time_s
        if frame.stage=='unknown':
            self.unarmed_frames+=1
            if not self.ever_armed and frame.valid and frame.knee_angle_deg is not None and frame.knee_angle_deg<=self.config.down_angle:
                self.opening_lowered_frames+=1
        eligible=(frame.valid and frame.stage!='unknown' and frame.selected_side==armed_side and armed_side is not None)
        if eligible:self.ever_armed=True
        if self.active:
            if frame.valid and frame.track_id!=self.active['track_id']:
                self._close(frame,'subject_changed')
            elif frame.valid and frame.selected_side!=self.active['side']:
                self._close(frame,'side_changed')
        if delta:
            included_current=False
            if not self.active:
                # Stable-frame=1 configurations can count faster than departure debounce.
                # The counter is authoritative; do not discard an actual counted cycle.
                initial=self.candidate or [frame]
                included_current=initial[-1].frame_index==frame.frame_index
                self._open(initial)
            if not included_current:self._include(frame)
            self._close(frame,'completed_cycle',status='counted')
            return
        if self.active:
            self._include(frame)
            if not eligible:
                self.return_streak=0
                return
            if not self.active['depth_confirmed']:
                self.return_streak=self.return_streak+1 if frame.knee_angle_deg>=self.config.up_angle else 0
                if self.return_streak>=self.config.stable_frames:
                    reason='depth_threshold_not_reached' if self.active['min_knee_angle_deg']>self.config.down_angle else 'depth_not_sustained'
                    self._close(frame,reason)
            return
        if not eligible or frame.knee_angle_deg is None or frame.knee_angle_deg>=self.config.up_angle:
            self.candidate=[]
            return
        if self.candidate and (self.candidate[-1].selected_side!=frame.selected_side or self.candidate[-1].track_id!=frame.track_id):
            self.candidate=[]
        self.candidate.append(frame)
        if len(self.candidate)>=self.departure_frames or frame.stage=='lowered':
            self._open(self.candidate)

    def finish(self, termination='video_ended'):
        if not self.finished and self.last:
            self._close(self.last,'processing_failed' if termination=='processing_failed' else 'video_ended_mid_attempt',termination=termination)
        self.finished=True
        self.candidate=[]
        return self.report()

    def report(self):
        return {
            'configuration':{'departure_frames':self.departure_frames,'up_angle':self.config.up_angle,
                             'down_angle':self.config.down_angle,'stable_frames':self.config.stable_frames,
                             'max_gap_seconds':self.config.max_gap_seconds},
            'counter_completed_reps':self.last_count,
            'counted_attempts':sum(e['status']=='counted' for e in self.events),
            'uncounted_attempts':sum(e['status']=='uncounted' for e in self.events),
            'pre_arming':{'unarmed_frames':self.unarmed_frames,'opening_lowered_frames':self.opening_lowered_frames,
                          'explanation':'No attempt is assigned to an opening crouch before standing is confirmed.'},
            'events':self.events,
            'limitations':['Attempts are software-defined observed movements, not expert-annotated repetitions or form grades.',
                           'A brief invalid observation resets departure debounce; unseen motion is never filled in.',
                           'Frame-based debounce depends on video frame rate; event timing needs independent human review.'],
        }
