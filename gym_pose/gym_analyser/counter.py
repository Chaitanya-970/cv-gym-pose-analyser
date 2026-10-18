"""Repetition counting state machine for squat exercises."""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .geometry import evaluate_leg


@dataclass(frozen=True)
class CounterConfig:
    """Configuration settings for squat analysis."""

    down_angle: float = 110.0
    up_angle: float = 150.0
    min_joint_confidence: float = 0.3
    stable_frames: int = 3
    max_gap_seconds: float = 0.5

    def __post_init__(self) -> None:
        if not (0.0 < self.down_angle < self.up_angle < 180.0):
            raise ValueError(
                f"Invalid angle thresholds: 0 < down ({self.down_angle}) < "
                f"up ({self.up_angle}) < 180 required"
            )
        if self.min_joint_confidence < 0.0 or self.min_joint_confidence > 1.0:
            raise ValueError(
                f"Invalid min_joint_confidence: {self.min_joint_confidence} (must be in [0, 1])"
            )
        if self.stable_frames < 1:
            raise ValueError(
                f"Invalid stable_frames: {self.stable_frames} (must be >= 1)"
            )
        if self.max_gap_seconds <= 0.0:
            raise ValueError(
                f"Invalid max_gap_seconds: {self.max_gap_seconds} (must be > 0)"
            )


@dataclass(frozen=True)
class FrameResult:
    """Output metrics for a single video frame."""

    frame_index: int
    video_time_s: float
    track_id: Optional[int]
    selected_side: Optional[str]
    knee_angle_deg: Optional[float]
    confidence: Optional[float]
    valid: bool
    reason: str
    stage: str
    completed_reps: int


class SquatCounter:
    """State machine tracking squat repetitions from 2D pose keypoints."""

    STAGE_UNKNOWN = "unknown"
    STAGE_STANDING = "standing"
    STAGE_LOWERED = "lowered"

    def __init__(self, config: Optional[CounterConfig] = None) -> None:
        self.config = config or CounterConfig()
        self.stage = self.STAGE_UNKNOWN
        self.completed_reps = 0
        self.locked_side: Optional[str] = None
        self.locked_track_id: Optional[int] = None

        self.standing_streak = 0
        self.lowered_streak = 0

        self.last_timestamp_s: Optional[float] = None
        self.last_valid_time_s: Optional[float] = None

    def _abandon_cycle(self) -> None:
        """Reset partial repetition state while preserving completed reps."""
        self.stage = self.STAGE_UNKNOWN
        self.locked_side = None
        self.standing_streak = 0
        self.lowered_streak = 0

    def update(
        self,
        frame_index: int,
        video_time_s: float,
        detections: Sequence[Dict[str, Any]],
    ) -> FrameResult:
        """Process a frame with a list of detections.

        Each detection dictionary must have:
        - 'keypoints': Sequence of (x, y) or (x, y, conf) keypoints
        - 'track_id': Optional[int] tracking identifier
        """
        if self.last_timestamp_s is not None and video_time_s < self.last_timestamp_s:
            raise ValueError(
                f"Non-monotonic timestamp: previous {self.last_timestamp_s}, current {video_time_s}"
            )

        # Enforce gap timeout before evaluating new observation
        if (
            self.last_valid_time_s is not None
            and (video_time_s - self.last_valid_time_s) > self.config.max_gap_seconds
        ):
            if (
                self.stage != self.STAGE_UNKNOWN
                or self.standing_streak > 0
                or self.lowered_streak > 0
            ):
                self._abandon_cycle()

        self.last_timestamp_s = video_time_s

        # Ambiguity checks on subject detection
        num_det = len(detections)
        if num_det == 0:
            self.standing_streak = 0
            self.lowered_streak = 0
            return FrameResult(
                frame_index=frame_index,
                video_time_s=video_time_s,
                track_id=self.locked_track_id,
                selected_side=self.locked_side,
                knee_angle_deg=None,
                confidence=None,
                valid=False,
                reason="no_subject_detected",
                stage=self.stage,
                completed_reps=self.completed_reps,
            )

        if num_det > 1:
            self.standing_streak = 0
            self.lowered_streak = 0
            return FrameResult(
                frame_index=frame_index,
                video_time_s=video_time_s,
                track_id=self.locked_track_id,
                selected_side=self.locked_side,
                knee_angle_deg=None,
                confidence=None,
                valid=False,
                reason="multiple_subjects_ambiguous",
                stage=self.stage,
                completed_reps=self.completed_reps,
            )

        det = detections[0]
        track_id = det.get("track_id")
        keypoints = det.get("keypoints", [])

        if track_id is None:
            self.standing_streak = 0
            self.lowered_streak = 0
            return FrameResult(
                frame_index=frame_index,
                video_time_s=video_time_s,
                track_id=None,
                selected_side=self.locked_side,
                knee_angle_deg=None,
                confidence=None,
                valid=False,
                reason="missing_track_id",
                stage=self.stage,
                completed_reps=self.completed_reps,
            )

        if self.locked_track_id is None:
            self.locked_track_id = track_id
        elif track_id != self.locked_track_id:
            self.standing_streak = 0
            self.lowered_streak = 0
            return FrameResult(
                frame_index=frame_index,
                video_time_s=video_time_s,
                track_id=track_id,
                selected_side=self.locked_side,
                knee_angle_deg=None,
                confidence=None,
                valid=False,
                reason=f"track_id_mismatch_{track_id}_vs_{self.locked_track_id}",
                stage=self.stage,
                completed_reps=self.completed_reps,
            )

        # Evaluate keypoints and side selection
        valid_side, angle, conf, reason, side = self._select_and_evaluate(keypoints)
        if not valid_side:
            self.standing_streak = 0
            self.lowered_streak = 0
            return FrameResult(
                frame_index=frame_index,
                video_time_s=video_time_s,
                track_id=self.locked_track_id,
                selected_side=self.locked_side or side,
                knee_angle_deg=None,
                confidence=conf,
                valid=False,
                reason=reason,
                stage=self.stage,
                completed_reps=self.completed_reps,
            )

        # Observation is valid
        self.last_valid_time_s = video_time_s

        # Apply state transitions
        self._step_state(angle, side)

        return FrameResult(
            frame_index=frame_index,
            video_time_s=video_time_s,
            track_id=self.locked_track_id,
            selected_side=self.locked_side,
            knee_angle_deg=angle,
            confidence=conf,
            valid=True,
            reason="ok",
            stage=self.stage,
            completed_reps=self.completed_reps,
        )

    def _select_and_evaluate(
        self, keypoints: Sequence[Sequence[float]]
    ) -> Tuple[bool, Optional[float], Optional[float], str, Optional[str]]:
        """Select appropriate side and compute angle."""
        if self.locked_side is not None:
            is_valid, angle, conf, reason = evaluate_leg(
                keypoints, self.locked_side, self.config.min_joint_confidence
            )
            return is_valid, angle, conf, reason, self.locked_side

        # Evaluate both sides to find best candidate
        v_l, a_l, c_l, r_l = evaluate_leg(
            keypoints, "left", self.config.min_joint_confidence
        )
        v_r, a_r, c_r, r_r = evaluate_leg(
            keypoints, "right", self.config.min_joint_confidence
        )

        if v_l and v_r:
            if (c_l or 0.0) >= (c_r or 0.0):
                return True, a_l, c_l, "ok", "left"
            return True, a_r, c_r, "ok", "right"
        if v_l:
            return True, a_l, c_l, "ok", "left"
        if v_r:
            return True, a_r, c_r, "ok", "right"

        best_conf = max(c_l or 0.0, c_r or 0.0)
        return False, None, best_conf, f"both_sides_invalid_left_{r_l}_right_{r_r}", None

    def _step_state(self, angle: float, observed_side: Optional[str]) -> None:
        """Advance repetition state machine using a validated angle."""
        if self.stage == self.STAGE_UNKNOWN:
            if angle >= self.config.up_angle:
                self.standing_streak += 1
                if self.standing_streak >= self.config.stable_frames:
                    self.stage = self.STAGE_STANDING
                    self.locked_side = observed_side
                    self.standing_streak = 0
            else:
                self.standing_streak = 0

        elif self.stage == self.STAGE_STANDING:
            if self.locked_side is None:
                self.locked_side = observed_side

            if angle <= self.config.down_angle:
                self.lowered_streak += 1
                if self.lowered_streak >= self.config.stable_frames:
                    self.stage = self.STAGE_LOWERED
                    self.lowered_streak = 0
            else:
                self.lowered_streak = 0

        elif self.stage == self.STAGE_LOWERED:
            if angle >= self.config.up_angle:
                self.standing_streak += 1
                if self.standing_streak >= self.config.stable_frames:
                    self.stage = self.STAGE_STANDING
                    self.completed_reps += 1
                    self.standing_streak = 0
                    # Permit side reselection after fresh standing observation
                    self.locked_side = None
            else:
                self.standing_streak = 0
