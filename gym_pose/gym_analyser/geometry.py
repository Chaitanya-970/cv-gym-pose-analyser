"""Geometric calculations for 2D human pose keypoints."""

import math
from numbers import Real
from typing import Any, Dict, List, Optional, Sequence, Tuple

LEFT_HIP = 11
LEFT_KNEE = 13
LEFT_ANKLE = 15

RIGHT_HIP = 12
RIGHT_KNEE = 14
RIGHT_ANKLE = 16

COCO_PAIRS: Dict[str, Tuple[int, int, int]] = {
    "left": (LEFT_HIP, LEFT_KNEE, LEFT_ANKLE),
    "right": (RIGHT_HIP, RIGHT_KNEE, RIGHT_ANKLE),
}


def calculate_angle(
    a: Sequence[float],
    b: Sequence[float],
    c: Sequence[float],
) -> Optional[float]:
    """Calculate the 2D projected joint angle at vertex b between points a and c.

    Points must contain at least x and y coordinates.
    Returns angle in degrees [0.0, 180.0], or None if inputs are non-finite,
    invalid types, or form zero-length vectors.
    """
    if len(a) < 2 or len(b) < 2 or len(c) < 2:
        return None

    ax, ay = a[0], a[1]
    bx, by = b[0], b[1]
    cx, cy = c[0], c[1]

    for val in (ax, ay, bx, by, cx, cy):
        if (
            isinstance(val, bool)
            or not isinstance(val, Real)
            or not math.isfinite(val)
        ):
            return None

    v1x = ax - bx
    v1y = ay - by
    v2x = cx - bx
    v2y = cy - by

    len1 = math.hypot(v1x, v1y)
    len2 = math.hypot(v2x, v2y)

    if len1 < 1e-6 or len2 < 1e-6:
        return None

    cos_val = (v1x * v2x + v1y * v2y) / (len1 * len2)
    cos_clamped = max(-1.0, min(1.0, cos_val))
    return math.degrees(math.acos(cos_clamped))


def extract_confidence(point: Sequence[float]) -> Optional[float]:
    """Extract and validate the confidence score from a point tuple.

    Points without an explicit confidence score are rejected.
    """
    if len(point) < 3:
        return None
    conf = point[2]
    if (
        isinstance(conf, bool)
        or not isinstance(conf, Real)
        or not math.isfinite(conf)
        or conf < 0.0
        or conf > 1.0
    ):
        return None
    return float(conf)


def evaluate_leg(
    keypoints: Sequence[Sequence[float]],
    side: str,
    min_confidence: float = 0.3,
) -> Tuple[bool, Optional[float], Optional[float], str]:
    """Evaluate knee angle for a given side ('left' or 'right').

    Returns a tuple of:
    (is_valid, angle_deg, min_confidence, reason)
    """
    if side not in COCO_PAIRS:
        return False, None, None, f"invalid_side_{side}"

    hip_idx, knee_idx, ankle_idx = COCO_PAIRS[side]
    max_idx = max(hip_idx, knee_idx, ankle_idx)
    if len(keypoints) <= max_idx:
        return False, None, None, "missing_keypoint_indices"

    hip = keypoints[hip_idx]
    knee = keypoints[knee_idx]
    ankle = keypoints[ankle_idx]

    c_hip = extract_confidence(hip)
    c_knee = extract_confidence(knee)
    c_ankle = extract_confidence(ankle)

    if c_hip is None or c_knee is None or c_ankle is None:
        return False, None, None, "missing_or_invalid_confidence"

    min_conf = min(c_hip, c_knee, c_ankle)
    if min_conf < min_confidence:
        return False, None, min_conf, "low_joint_confidence"

    angle = calculate_angle(hip, knee, ankle)
    if angle is None:
        return False, None, min_conf, "degenerate_geometry"

    return True, angle, min_conf, "ok"
