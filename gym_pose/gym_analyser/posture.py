"""2D human posture analysis module for exercise tracking.

Provides pure per-frame geometric calculations for torso inclination angle
and hip joint angle from 2D pose keypoints (COCO-17 format).

Does not depend on deep learning frameworks, video processing libraries,
or training loops. All calculations are pure standard-library Python and math.
"""

import math
from dataclasses import asdict, dataclass
from numbers import Real
from typing import Any, Dict, List, Optional, Sequence, Tuple

try:
    from .geometry import calculate_angle, extract_confidence
except (ImportError, ValueError):
    from gym_analyser.geometry import calculate_angle, extract_confidence

# COCO-17 keypoint indices for posture evaluation
LEFT_SHOULDER = 5
RIGHT_SHOULDER = 6
LEFT_HIP = 11
RIGHT_HIP = 12
LEFT_KNEE = 13
RIGHT_KNEE = 14
LEFT_ANKLE = 15
RIGHT_ANKLE = 16

COCO_POSTURE_PAIRS: Dict[str, Dict[str, int]] = {
    "left": {
        "shoulder": LEFT_SHOULDER,
        "hip": LEFT_HIP,
        "knee": LEFT_KNEE,
        "ankle": LEFT_ANKLE,
    },
    "right": {
        "shoulder": RIGHT_SHOULDER,
        "hip": RIGHT_HIP,
        "knee": RIGHT_KNEE,
        "ankle": RIGHT_ANKLE,
    },
}


@dataclass(frozen=True)
class PostureFrameResult:
    """Structured per-frame posture measurement results and validity reasons.

    Attributes:
        torso_inclination_deg: Torso vector angle relative to upward vertical in degrees [0, 180].
        hip_angle_deg: 2D angle at hip vertex between shoulder and knee in degrees [0, 180].
        torso_valid: True if torso inclination was successfully computed with sufficient confidence.
        hip_valid: True if hip angle was successfully computed with sufficient confidence.
        torso_reason: Status or failure reason for torso measurement.
        hip_reason: Status or failure reason for hip measurement.
        side: Body side evaluated ('left' or 'right').
        min_confidence: Minimum observed keypoint confidence across evaluated joints, or None.
    """

    torso_inclination_deg: Optional[float]
    hip_angle_deg: Optional[float]
    torso_valid: bool
    hip_valid: bool
    torso_reason: str
    hip_reason: str
    side: str
    min_confidence: Optional[float]

    @property
    def is_valid(self) -> bool:
        """True if both torso and hip measurements are valid."""
        return self.torso_valid and self.hip_valid

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to a serializable dictionary."""
        return asdict(self)


def _validate_coordinates(*points: Sequence[float]) -> bool:
    """Check that points contain at least 2 real, finite coordinates."""
    for pt in points:
        try:
            if len(pt) < 2:
                return False
        except TypeError:
            return False
        for val in (pt[0], pt[1]):
            if (
                isinstance(val, bool)
                or not isinstance(val, Real)
                or not math.isfinite(val)
            ):
                return False
    return True


def calculate_torso_inclination(
    shoulder: Sequence[float],
    hip: Sequence[float],
) -> Optional[float]:
    """Calculate 2D torso inclination angle relative to upward vertical.

    In 2D image coordinates (y increases downward, shoulder is above hip
    so shoulder_y < hip_y in standard upright posture), the upward vertical
    reference vector is (0.0, -1.0).

    The torso vector points from hip to shoulder:
        v_torso = (shoulder_x - hip_x, shoulder_y - hip_y)

    The returned angle is in degrees [0.0, 180.0]:
        0.0°   : vertical upright torso (shoulder directly above hip)
        45.0°  : 45° forward or backward lean
        90.0°  : horizontal torso (shoulder level with hip)
        180.0° : completely inverted torso (shoulder directly below hip)

    Returns:
        float angle in [0.0, 180.0], or None if inputs are non-finite,
        invalid types, or form a zero-length vector (coincident points).
    """
    if not _validate_coordinates(shoulder, hip):
        return None

    sx, sy = shoulder[0], shoulder[1]
    hx, hy = hip[0], hip[1]

    vx = sx - hx
    vy = sy - hy

    len_torso = math.hypot(vx, vy)
    if not math.isfinite(len_torso) or len_torso < 1e-6:
        return None

    # Reference vector upward in image coords: rx = 0.0, ry = -1.0 (len = 1.0)
    # Dot product: vx * 0.0 + vy * (-1.0) = -vy = hy - sy
    dot = hy - sy
    cos_val = dot / len_torso
    if not math.isfinite(cos_val):
        return None
    cos_clamped = max(-1.0, min(1.0, cos_val))
    return math.degrees(math.acos(cos_clamped))


def calculate_hip_angle(
    shoulder: Sequence[float],
    hip: Sequence[float],
    knee: Sequence[float],
) -> Optional[float]:
    """Calculate the 2D projected joint angle at the hip vertex between shoulder and knee.

    Vertex is hip (point b in angle a-b-c).
    Vector 1: hip -> shoulder
    Vector 2: hip -> knee

    The returned angle is in degrees [0.0, 180.0]:
        ~180.0° : upright standing leg/torso alignment
        90.0°   : right angle at hip
        < 90.0° : acute deep squat hip flexion

    Returns:
        float angle in [0.0, 180.0], or None if inputs are non-finite,
        invalid types, or form zero-length limbs.
    """
    if not _validate_coordinates(shoulder,hip,knee):
        return None
    return calculate_angle(shoulder, hip, knee)


def _validate_threshold(value: float) -> None:
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError("min_confidence must be finite in [0,1]")


def _has_indices(keypoints: Any, maximum: int) -> bool:
    try:return len(keypoints) > maximum
    except TypeError:return False


def evaluate_torso(
    keypoints: Sequence[Sequence[float]],
    side: str,
    min_confidence: float = 0.3,
) -> Tuple[bool, Optional[float], Optional[float], str]:
    """Evaluate 2D torso inclination angle for a specified side.

    Returns a tuple of:
        (is_valid, angle_deg, min_confidence, reason)
    """
    _validate_threshold(min_confidence)
    if side not in COCO_POSTURE_PAIRS:
        return False, None, None, f"invalid_side_{side}"

    indices = COCO_POSTURE_PAIRS[side]
    sh_idx = indices["shoulder"]
    hip_idx = indices["hip"]

    max_idx = max(sh_idx, hip_idx)
    if not _has_indices(keypoints, max_idx):
        return False, None, None, "missing_keypoint_indices"

    sh_pt = keypoints[sh_idx]
    hip_pt = keypoints[hip_idx]

    if not _validate_coordinates(sh_pt, hip_pt):
        return False, None, None, "invalid_coordinates"
    c_sh = extract_confidence(sh_pt)
    c_hip = extract_confidence(hip_pt)

    if c_sh is None or c_hip is None:
        return False, None, None, "missing_or_invalid_confidence"

    min_conf = min(c_sh, c_hip)
    if min_conf < min_confidence:
        return False, None, min_conf, "low_joint_confidence"

    if not _validate_coordinates(sh_pt, hip_pt):
        return False, None, min_conf, "invalid_coordinates"

    angle = calculate_torso_inclination(sh_pt, hip_pt)
    if angle is None:
        return False, None, min_conf, "degenerate_geometry"

    return True, angle, min_conf, "ok"


def evaluate_hip(
    keypoints: Sequence[Sequence[float]],
    side: str,
    min_confidence: float = 0.3,
) -> Tuple[bool, Optional[float], Optional[float], str]:
    """Evaluate 2D hip joint angle for a specified side.

    Returns a tuple of:
        (is_valid, angle_deg, min_confidence, reason)
    """
    _validate_threshold(min_confidence)
    if side not in COCO_POSTURE_PAIRS:
        return False, None, None, f"invalid_side_{side}"

    indices = COCO_POSTURE_PAIRS[side]
    sh_idx = indices["shoulder"]
    hip_idx = indices["hip"]
    knee_idx = indices["knee"]

    max_idx = max(sh_idx, hip_idx, knee_idx)
    if not _has_indices(keypoints, max_idx):
        return False, None, None, "missing_keypoint_indices"

    sh_pt = keypoints[sh_idx]
    hip_pt = keypoints[hip_idx]
    knee_pt = keypoints[knee_idx]

    if not _validate_coordinates(sh_pt, hip_pt):
        return False, None, None, "invalid_coordinates"
    c_sh = extract_confidence(sh_pt)
    c_hip = extract_confidence(hip_pt)
    if not _validate_coordinates(knee_pt):
        return False, None, None, "invalid_coordinates"
    c_knee = extract_confidence(knee_pt)

    if c_sh is None or c_hip is None or c_knee is None:
        return False, None, None, "missing_or_invalid_confidence"

    min_conf = min(c_sh, c_hip, c_knee)
    if min_conf < min_confidence:
        return False, None, min_conf, "low_joint_confidence"

    if not _validate_coordinates(sh_pt, hip_pt, knee_pt):
        return False, None, min_conf, "invalid_coordinates"

    angle = calculate_hip_angle(sh_pt, hip_pt, knee_pt)
    if angle is None:
        return False, None, min_conf, "degenerate_geometry"

    return True, angle, min_conf, "ok"


def evaluate_posture(
    keypoints: Sequence[Sequence[float]],
    side: str,
    min_confidence: float = 0.3,
) -> PostureFrameResult:
    """Evaluate 2D torso inclination and hip angle for a specified side.

    Calculates:
    - Torso inclination relative to upward vertical (shoulder and hip)
    - Hip angle at hip vertex between shoulder and knee

    Returns:
        PostureFrameResult with angles, validation flags, reason codes,
        and observed minimum joint confidence.
    """
    _validate_threshold(min_confidence)
    if side not in COCO_POSTURE_PAIRS:
        return PostureFrameResult(
            torso_inclination_deg=None,
            hip_angle_deg=None,
            torso_valid=False,
            hip_valid=False,
            torso_reason=f"invalid_side_{side}",
            hip_reason=f"invalid_side_{side}",
            side=side,
            min_confidence=None,
        )

    t_valid, t_angle, t_conf, t_reason = evaluate_torso(
        keypoints, side, min_confidence=min_confidence
    )
    h_valid, h_angle, h_conf, h_reason = evaluate_hip(
        keypoints, side, min_confidence=min_confidence
    )

    confs = [c for c in (t_conf, h_conf) if c is not None]
    overall_conf = min(confs) if confs else None

    return PostureFrameResult(
        torso_inclination_deg=t_angle,
        hip_angle_deg=h_angle,
        torso_valid=t_valid,
        hip_valid=h_valid,
        torso_reason=t_reason,
        hip_reason=h_reason,
        side=side,
        min_confidence=overall_conf,
    )


def select_posture_side(
    keypoints: Sequence[Sequence[float]],
    min_confidence: float = 0.3,
) -> Tuple[Optional[str], Optional[float], str]:
    """Select the best body side ('left' or 'right') for posture analysis.

    Evaluates both left and right posture keypoints:
    1. If both sides are fully valid (both torso and hip), selects the side
       with the higher minimum keypoint confidence (defaulting to 'left' on tie).
    2. If only one side is fully valid, selects that side.
    3. If neither side is fully valid, checks if one side has a valid torso or hip.
    4. If neither side has any valid measurement, returns None and the failure reason.

    Returns:
        (selected_side, best_confidence, reason)
    """
    res_l = evaluate_posture(keypoints, "left", min_confidence=min_confidence)
    res_r = evaluate_posture(keypoints, "right", min_confidence=min_confidence)

    c_l = res_l.min_confidence or 0.0
    c_r = res_r.min_confidence or 0.0

    # Both sides fully valid
    if res_l.is_valid and res_r.is_valid:
        if c_l >= c_r:
            return "left", res_l.min_confidence, "ok"
        return "right", res_r.min_confidence, "ok"

    # Only one side fully valid
    if res_l.is_valid:
        return "left", res_l.min_confidence, "ok"
    if res_r.is_valid:
        return "right", res_r.min_confidence, "ok"

    # Partial validity (e.g. valid torso on one side)
    l_valid_count = int(res_l.torso_valid) + int(res_l.hip_valid)
    r_valid_count = int(res_r.torso_valid) + int(res_r.hip_valid)
    if l_valid_count > r_valid_count:
        return "left", res_l.min_confidence, "partial_left_valid"
    if r_valid_count > l_valid_count:
        return "right", res_r.min_confidence, "partial_right_valid"

    if l_valid_count > 0:
        left_conf = evaluate_torso(keypoints,"left",min_confidence)[2] or 0.0
        right_conf = evaluate_torso(keypoints,"right",min_confidence)[2] or 0.0
        chosen = res_l if left_conf >= right_conf else res_r
        return chosen.side, max(left_conf,right_conf), "partial_valid"

    # Neither valid
    best_conf = max(res_l.min_confidence or 0.0, res_r.min_confidence or 0.0)
    if res_l.min_confidence is None and res_r.min_confidence is None:
        best_conf = None
    reason = f"both_sides_invalid_left_{res_l.torso_reason}_right_{res_r.torso_reason}"
    return None, best_conf, reason


def evaluate_posture_auto_side(
    keypoints: Sequence[Sequence[float]],
    min_confidence: float = 0.3,
) -> PostureFrameResult:
    """Evaluate posture by automatically selecting the side with best visibility."""
    side, conf, reason = select_posture_side(keypoints, min_confidence=min_confidence)
    if side is None:
        return PostureFrameResult(
            torso_inclination_deg=None,
            hip_angle_deg=None,
            torso_valid=False,
            hip_valid=False,
            torso_reason=reason,
            hip_reason=reason,
            side="unselected",
            min_confidence=conf,
        )
    return evaluate_posture(keypoints, side=side, min_confidence=min_confidence)
