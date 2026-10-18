"""Unit tests for 2D posture analysis module."""

import math
from numbers import Real
from typing import List, Tuple
import unittest

from gym_analyser.posture import (
    COCO_POSTURE_PAIRS,
    LEFT_ANKLE,
    LEFT_HIP,
    LEFT_KNEE,
    LEFT_SHOULDER,
    RIGHT_ANKLE,
    RIGHT_HIP,
    RIGHT_KNEE,
    RIGHT_SHOULDER,
    PostureFrameResult,
    calculate_hip_angle,
    calculate_torso_inclination,
    evaluate_hip,
    evaluate_posture,
    evaluate_posture_auto_side,
    evaluate_torso,
    select_posture_side,
)


def _make_coco17_keypoints(
    shoulder_pos: Tuple[float, float] = (100.0, 50.0),
    hip_pos: Tuple[float, float] = (100.0, 100.0),
    knee_pos: Tuple[float, float] = (100.0, 150.0),
    ankle_pos: Tuple[float, float] = (100.0, 200.0),
    conf: float = 0.9,
    side: str = "left",
) -> List[List[float]]:
    """Helper to create a standard 17-keypoint COCO list with realistic defaults."""
    # Initialize 17 points with benign non-zero dummy values
    kpts: List[List[float]] = [[50.0, 50.0, conf] for _ in range(17)]

    sh_idx = LEFT_SHOULDER if side == "left" else RIGHT_SHOULDER
    h_idx = LEFT_HIP if side == "left" else RIGHT_HIP
    k_idx = LEFT_KNEE if side == "left" else RIGHT_KNEE
    a_idx = LEFT_ANKLE if side == "left" else RIGHT_ANKLE

    kpts[sh_idx] = [shoulder_pos[0], shoulder_pos[1], conf]
    kpts[h_idx] = [hip_pos[0], hip_pos[1], conf]
    kpts[k_idx] = [knee_pos[0], knee_pos[1], conf]
    kpts[a_idx] = [ankle_pos[0], ankle_pos[1], conf]

    return kpts


class TestTorsoInclinationGeometry(unittest.TestCase):
    """Test geometric calculations for torso inclination."""

    def test_vertical_torso_zero_deg(self):
        """Shoulder directly above hip in image space (sy < hy) must equal 0 degrees."""
        hip = (100.0, 100.0)
        shoulder = (100.0, 50.0)
        angle = calculate_torso_inclination(shoulder, hip)
        self.assertIsNotNone(angle)
        self.assertAlmostEqual(angle, 0.0, places=4)

    def test_forward_lean_45_deg(self):
        """Shoulder at 45 degree forward lean must equal 45 degrees."""
        hip = (100.0, 100.0)
        shoulder = (150.0, 50.0)
        angle = calculate_torso_inclination(shoulder, hip)
        self.assertIsNotNone(angle)
        self.assertAlmostEqual(angle, 45.0, places=4)

    def test_backward_lean_45_deg(self):
        """Shoulder at 45 degree backward lean must equal 45 degrees (symmetric unsigned)."""
        hip = (100.0, 100.0)
        shoulder = (50.0, 50.0)
        angle = calculate_torso_inclination(shoulder, hip)
        self.assertIsNotNone(angle)
        self.assertAlmostEqual(angle, 45.0, places=4)

    def test_horizontal_torso_90_deg(self):
        """Torso perfectly horizontal (sy == hy) must equal 90 degrees."""
        hip = (100.0, 100.0)
        shoulder = (200.0, 100.0)
        angle = calculate_torso_inclination(shoulder, hip)
        self.assertIsNotNone(angle)
        self.assertAlmostEqual(angle, 90.0, places=4)

    def test_inverted_torso_180_deg(self):
        """Shoulder directly below hip (sy > hy) must equal 180 degrees."""
        hip = (100.0, 100.0)
        shoulder = (100.0, 150.0)
        angle = calculate_torso_inclination(shoulder, hip)
        self.assertIsNotNone(angle)
        self.assertAlmostEqual(angle, 180.0, places=4)

    def test_coincident_points_zero_length(self):
        """Coincident shoulder and hip must return None (degenerate vector)."""
        hip = (100.0, 100.0)
        shoulder = (100.0, 100.0)
        angle = calculate_torso_inclination(shoulder, hip)
        self.assertIsNone(angle)

    def test_non_finite_coordinates(self):
        """NaN or infinite coordinates must return None, never a fabricated angle."""
        hip = (100.0, 100.0)
        self.assertIsNone(calculate_torso_inclination((float("nan"), 50.0), hip))
        self.assertIsNone(calculate_torso_inclination((100.0, float("inf")), hip))
        self.assertIsNone(calculate_torso_inclination((100.0, 50.0), (100.0, float("-inf"))))

    def test_boolean_and_type_rejection(self):
        """Booleans and invalid types must return None."""
        hip = (100.0, 100.0)
        self.assertIsNone(calculate_torso_inclination((True, 50.0), hip))
        self.assertIsNone(calculate_torso_inclination((100.0, "50.0"), hip))
        self.assertIsNone(calculate_torso_inclination((100.0,), hip))


class TestHipAngleGeometry(unittest.TestCase):
    """Test geometric calculations for hip joint angle."""

    def test_standing_hip_angle_180_deg(self):
        """Standing upright straight line (shoulder-hip-knee) must equal 180 degrees."""
        shoulder = (100.0, 50.0)
        hip = (100.0, 100.0)
        knee = (100.0, 150.0)
        angle = calculate_hip_angle(shoulder, hip, knee)
        self.assertIsNotNone(angle)
        self.assertAlmostEqual(angle, 180.0, places=4)

    def test_right_angle_hip_90_deg(self):
        """Torso vertical and femur horizontal must equal 90 degrees."""
        shoulder = (100.0, 50.0)
        hip = (100.0, 100.0)
        knee = (150.0, 100.0)
        angle = calculate_hip_angle(shoulder, hip, knee)
        self.assertIsNotNone(angle)
        self.assertAlmostEqual(angle, 90.0, places=4)

    def test_acute_deep_squat_hip_angle(self):
        """Deep squat hip flexion (torso lean + femur below/at parallel) must be acute (< 90 deg)."""
        hip = (100.0, 100.0)
        shoulder = (140.0, 70.0)  # Lean forward
        knee = (140.0, 100.0)     # Femur horizontal forward
        angle = calculate_hip_angle(shoulder, hip, knee)
        self.assertIsNotNone(angle)
        self.assertLess(angle, 90.0)
        # Vector 1: (40, -30), len=50. Vector 2: (40, 0), len=40.
        # cos = 1600 / 2000 = 0.8 -> acos(0.8) approx 36.8699 degrees
        expected_angle = math.degrees(math.acos(0.8))
        self.assertAlmostEqual(angle, expected_angle, places=4)

    def test_coincident_points_zero_length(self):
        """Limb with zero length must return None."""
        shoulder = (100.0, 100.0)
        hip = (100.0, 100.0)
        knee = (100.0, 150.0)
        self.assertIsNone(calculate_hip_angle(shoulder, hip, knee))

        shoulder2 = (100.0, 50.0)
        knee2 = (100.0, 100.0)
        self.assertIsNone(calculate_hip_angle(shoulder2, hip, knee2))

    def test_non_finite_and_type_rejection(self):
        """Non-finite coordinates and invalid types must return None."""
        shoulder = (100.0, 50.0)
        hip = (100.0, 100.0)
        knee = (100.0, 150.0)
        self.assertIsNone(calculate_hip_angle((float("nan"), 50.0), hip, knee))
        self.assertIsNone(calculate_hip_angle(shoulder, (100.0, float("inf")), knee))
        self.assertIsNone(calculate_hip_angle(shoulder, hip, (True, 150.0)))


class TestGeometricInvariance(unittest.TestCase):
    """Test translation and scaling invariance for all posture angles."""

    def test_translation_invariance(self):
        """Offsetting all coordinates by (dx, dy) must leave angles unchanged."""
        shoulder = (120.0, 60.0)
        hip = (100.0, 100.0)
        knee = (130.0, 140.0)

        orig_torso = calculate_torso_inclination(shoulder, hip)
        orig_hip = calculate_hip_angle(shoulder, hip, knee)

        for dx, dy in [(250.0, -180.0), (-500.0, 320.0), (12.345, 67.89)]:
            s_trans = (shoulder[0] + dx, shoulder[1] + dy)
            h_trans = (hip[0] + dx, hip[1] + dy)
            k_trans = (knee[0] + dx, knee[1] + dy)

            t_angle = calculate_torso_inclination(s_trans, h_trans)
            h_angle = calculate_hip_angle(s_trans, h_trans, k_trans)

            self.assertIsNotNone(t_angle)
            self.assertIsNotNone(h_angle)
            self.assertAlmostEqual(t_angle, orig_torso, places=4)
            self.assertAlmostEqual(h_angle, orig_hip, places=4)

    def test_scaling_invariance(self):
        """Multiplying all coordinates by a positive scale factor must leave angles unchanged."""
        shoulder = (130.0, 70.0)
        hip = (100.0, 100.0)
        knee = (140.0, 110.0)

        orig_torso = calculate_torso_inclination(shoulder, hip)
        orig_hip = calculate_hip_angle(shoulder, hip, knee)

        for scale in [0.25, 0.5, 2.0, 10.0]:
            s_scaled = (shoulder[0] * scale, shoulder[1] * scale)
            h_scaled = (hip[0] * scale, hip[1] * scale)
            k_scaled = (knee[0] * scale, knee[1] * scale)

            t_angle = calculate_torso_inclination(s_scaled, h_scaled)
            h_angle = calculate_hip_angle(s_scaled, h_scaled, k_scaled)

            self.assertIsNotNone(t_angle)
            self.assertIsNotNone(h_angle)
            self.assertAlmostEqual(t_angle, orig_torso, places=4)
            self.assertAlmostEqual(h_angle, orig_hip, places=4)


class TestConfidenceAndKeypointValidation(unittest.TestCase):
    """Test confidence thresholds, missing indices, and invalid inputs."""

    def test_low_joint_confidence_rejection(self):
        """Any required joint with confidence below min_confidence yields None, valid=False."""
        # Baseline valid left-side keypoints
        kpts = _make_coco17_keypoints(conf=0.9, side="left")

        # 1. Low confidence shoulder (idx 5)
        kpts[LEFT_SHOULDER][2] = 0.25
        t_valid, t_ang, t_conf, t_reason = evaluate_torso(kpts, "left", min_confidence=0.3)
        self.assertFalse(t_valid)
        self.assertIsNone(t_ang)
        self.assertEqual(t_conf, 0.25)
        self.assertEqual(t_reason, "low_joint_confidence")

        # 2. Low confidence knee (idx 13) affects hip angle, but torso remains valid
        kpts[LEFT_SHOULDER][2] = 0.9  # Restore shoulder
        kpts[LEFT_KNEE][2] = 0.20     # Low knee
        res = evaluate_posture(kpts, "left", min_confidence=0.3)
        self.assertTrue(res.torso_valid)
        self.assertIsNotNone(res.torso_inclination_deg)
        self.assertEqual(res.torso_reason, "ok")
        self.assertFalse(res.hip_valid)
        self.assertIsNone(res.hip_angle_deg)
        self.assertEqual(res.hip_reason, "low_joint_confidence")
        self.assertEqual(res.min_confidence, 0.20)

    def test_missing_or_invalid_confidence(self):
        """Points lacking confidence or having non-float/nan confidence are rejected."""
        kpts = _make_coco17_keypoints(conf=0.9, side="left")
        # Point with only (x, y)
        kpts[LEFT_HIP] = [100.0, 100.0]
        res = evaluate_posture(kpts, "left")
        self.assertFalse(res.torso_valid)
        self.assertFalse(res.hip_valid)
        self.assertIsNone(res.torso_inclination_deg)
        self.assertIsNone(res.hip_angle_deg)
        self.assertEqual(res.torso_reason, "missing_or_invalid_confidence")
        self.assertEqual(res.hip_reason, "missing_or_invalid_confidence")

    def test_non_finite_and_type_coordinates_in_frame(self):
        """NaN, inf, and boolean coordinates yield unavailable, never zero degrees."""
        kpts = _make_coco17_keypoints(conf=0.9, side="left")
        kpts[LEFT_SHOULDER] = [float("nan"), 50.0, 0.9]

        res = evaluate_posture(kpts, "left")
        self.assertFalse(res.torso_valid)
        self.assertFalse(res.hip_valid)
        self.assertIsNone(res.torso_inclination_deg)
        self.assertIsNone(res.hip_angle_deg)
        self.assertIn(res.torso_reason, ("invalid_coordinates", "degenerate_geometry"))
        self.assertIn(res.hip_reason, ("invalid_coordinates", "degenerate_geometry"))

    def test_zero_length_coincident_joints(self):
        """Coincident joints in keypoints array yield unavailable with degenerate_geometry."""
        kpts = _make_coco17_keypoints(conf=0.9, side="left")
        # Make shoulder coincident with hip
        kpts[LEFT_SHOULDER] = [100.0, 100.0, 0.9]
        kpts[LEFT_HIP] = [100.0, 100.0, 0.9]

        res = evaluate_posture(kpts, "left")
        self.assertFalse(res.torso_valid)
        self.assertIsNone(res.torso_inclination_deg)
        self.assertEqual(res.torso_reason, "degenerate_geometry")

    def test_truncated_keypoints_list(self):
        """Keypoint lists with fewer than 17 keypoints must be cleanly rejected."""
        short_kpts = [[100.0, 100.0, 0.9]] * 10
        res = evaluate_posture(short_kpts, "left")
        self.assertFalse(res.torso_valid)
        self.assertFalse(res.hip_valid)
        self.assertIsNone(res.torso_inclination_deg)
        self.assertIsNone(res.hip_angle_deg)
        self.assertEqual(res.torso_reason, "missing_keypoint_indices")
        self.assertEqual(res.hip_reason, "missing_keypoint_indices")

        empty_kpts = []
        res_empty = evaluate_posture(empty_kpts, "left")
        self.assertFalse(res_empty.torso_valid)
        self.assertEqual(res_empty.torso_reason, "missing_keypoint_indices")

    def test_invalid_side_identifier(self):
        """Unsupported side identifiers must be rejected."""
        kpts = _make_coco17_keypoints(conf=0.9, side="left")
        for bad_side in ["center", "front", "both", ""]:
            res = evaluate_posture(kpts, bad_side)
            self.assertFalse(res.torso_valid)
            self.assertFalse(res.hip_valid)
            self.assertEqual(res.torso_reason, f"invalid_side_{bad_side}")


class TestBothSidesAndSideSelection(unittest.TestCase):
    """Test both left and right sides and the side-selection helper."""

    def test_right_side_evaluation(self):
        """Right-side keypoints (indices 6, 12, 14) are evaluated accurately."""
        kpts = _make_coco17_keypoints(
            shoulder_pos=(120.0, 60.0),
            hip_pos=(100.0, 100.0),
            knee_pos=(130.0, 130.0),
            conf=0.95,
            side="right",
        )
        res = evaluate_posture(kpts, "right", min_confidence=0.3)
        self.assertTrue(res.torso_valid)
        self.assertTrue(res.hip_valid)
        self.assertEqual(res.side, "right")
        self.assertEqual(res.torso_reason, "ok")
        self.assertEqual(res.hip_reason, "ok")
        self.assertAlmostEqual(res.min_confidence, 0.95, places=4)

    def test_side_selection_higher_confidence(self):
        """Selects side with higher minimum joint confidence when both are valid."""
        kpts = [[0.0, 0.0, 0.0] for _ in range(17)]
        # Left side points with confidence 0.92
        kpts[LEFT_SHOULDER] = [100.0, 50.0, 0.92]
        kpts[LEFT_HIP] = [100.0, 100.0, 0.95]
        kpts[LEFT_KNEE] = [100.0, 150.0, 0.93]

        # Right side points with lower confidence 0.65
        kpts[RIGHT_SHOULDER] = [100.0, 50.0, 0.65]
        kpts[RIGHT_HIP] = [100.0, 100.0, 0.70]
        kpts[RIGHT_KNEE] = [100.0, 150.0, 0.68]

        side, conf, reason = select_posture_side(kpts, min_confidence=0.3)
        self.assertEqual(side, "left")
        self.assertAlmostEqual(conf, 0.92, places=4)
        self.assertEqual(reason, "ok")

        # Now make right side have higher confidence
        kpts[RIGHT_SHOULDER][2] = 0.98
        kpts[RIGHT_HIP][2] = 0.98
        kpts[RIGHT_KNEE][2] = 0.98
        side2, conf2, reason2 = select_posture_side(kpts, min_confidence=0.3)
        self.assertEqual(side2, "right")
        self.assertAlmostEqual(conf2, 0.98, places=4)
        self.assertEqual(reason2, "ok")

    def test_side_selection_one_side_valid(self):
        """Selects the only valid side when the other side has low confidence."""
        kpts = [[0.0, 0.0, 0.0] for _ in range(17)]
        # Left valid
        kpts[LEFT_SHOULDER] = [100.0, 50.0, 0.85]
        kpts[LEFT_HIP] = [100.0, 100.0, 0.85]
        kpts[LEFT_KNEE] = [100.0, 150.0, 0.85]
        # Right low confidence
        kpts[RIGHT_SHOULDER] = [100.0, 50.0, 0.15]
        kpts[RIGHT_HIP] = [100.0, 100.0, 0.15]
        kpts[RIGHT_KNEE] = [100.0, 150.0, 0.15]

        side, conf, reason = select_posture_side(kpts, min_confidence=0.3)
        self.assertEqual(side, "left")
        self.assertAlmostEqual(conf, 0.85, places=4)

    def test_side_selection_neither_valid(self):
        """Returns None with failure reason when neither side meets criteria."""
        kpts = [[100.0, 100.0, 0.1] for _ in range(17)]
        side, conf, reason = select_posture_side(kpts, min_confidence=0.3)
        self.assertIsNone(side)
        self.assertIn("both_sides_invalid", reason)

    def test_evaluate_posture_auto_side(self):
        """evaluate_posture_auto_side seamlessly selects side and computes posture."""
        kpts = [[0.0, 0.0, 0.0] for _ in range(17)]
        # Left valid vertical torso and standing hip
        kpts[LEFT_SHOULDER] = [100.0, 50.0, 0.9]
        kpts[LEFT_HIP] = [100.0, 100.0, 0.9]
        kpts[LEFT_KNEE] = [100.0, 150.0, 0.9]

        res = evaluate_posture_auto_side(kpts, min_confidence=0.3)
        self.assertEqual(res.side, "left")
        self.assertTrue(res.is_valid)
        self.assertAlmostEqual(res.torso_inclination_deg, 0.0, places=4)
        self.assertAlmostEqual(res.hip_angle_deg, 180.0, places=4)

    def test_posture_frame_result_structure_and_dict(self):
        """Verify PostureFrameResult dataclass serialization and field integrity."""
        res = PostureFrameResult(
            torso_inclination_deg=12.5,
            hip_angle_deg=165.0,
            torso_valid=True,
            hip_valid=True,
            torso_reason="ok",
            hip_reason="ok",
            side="left",
            min_confidence=0.88,
        )
        self.assertTrue(res.is_valid)
        d = res.to_dict()
        self.assertEqual(d["torso_inclination_deg"], 12.5)
        self.assertEqual(d["hip_angle_deg"], 165.0)
        self.assertEqual(d["side"], "left")
        self.assertEqual(d["min_confidence"], 0.88)
        self.assertTrue(d["torso_valid"])
        self.assertTrue(d["hip_valid"])



class TestPostureReviewRegressions(unittest.TestCase):
    def test_partial_torso_on_both_sides_is_retained(self):
        kpts = [[0.,0.,0.] for _ in range(17)]
        for sh,hip,knee,x,confidence in ((5,11,13,0.,.8),(6,12,14,5.,.95)):
            kpts[sh]=[x,0.,confidence]; kpts[hip]=[x,1.,confidence]; kpts[knee]=[x,2.,.1]
        result=evaluate_posture_auto_side(kpts)
        self.assertEqual(result.side,"right")
        self.assertTrue(result.torso_valid)
        self.assertEqual(result.torso_inclination_deg,0.)
        self.assertIsNone(result.hip_angle_deg)

    def test_malformed_points_and_thresholds(self):
        self.assertIsNone(calculate_torso_inclination(None,[0,1]))
        points=[[0,0,.9] for _ in range(17)]
        points[5]=None
        self.assertFalse(evaluate_posture(points,"left").is_valid)
        self.assertFalse(evaluate_posture(None,"left").is_valid)
        for threshold in (float("nan"),True,-.1,1.1):
            with self.assertRaises(ValueError): evaluate_posture([],"left",threshold)

    def test_overflow_is_not_a_fabricated_zero_angle(self):
        self.assertIsNone(calculate_torso_inclination([1e308,1e308],[-1e308,-1e308]))
        self.assertIsNone(calculate_hip_angle([1e308,1e308],[-1e308,-1e308],[0,0]))


if __name__ == "__main__":
    unittest.main()
