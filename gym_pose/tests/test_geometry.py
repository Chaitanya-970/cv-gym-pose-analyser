"""Unit tests for geometry module."""

import math
import unittest

from gym_analyser.geometry import (
    calculate_angle,
    evaluate_leg,
    extract_confidence,
)


class TestGeometry(unittest.TestCase):
    """Test geometric vector math and confidence evaluation."""

    def test_straight_line_angle(self):
        """Straight leg must equal 180 degrees."""
        hip = (100.0, 50.0)
        knee = (100.0, 100.0)
        ankle = (100.0, 150.0)
        angle = calculate_angle(hip, knee, ankle)
        self.assertIsNotNone(angle)
        self.assertAlmostEqual(angle, 180.0, places=4)

    def test_right_angle(self):
        """Right angle joint must equal 90 degrees."""
        hip = (100.0, 50.0)
        knee = (100.0, 100.0)
        ankle = (150.0, 100.0)
        angle = calculate_angle(hip, knee, ankle)
        self.assertIsNotNone(angle)
        self.assertAlmostEqual(angle, 90.0, places=4)

    def test_acute_angle(self):
        """Acute 45 degree angle verification."""
        hip = (100.0, 50.0)
        knee = (100.0, 100.0)
        # 45 degrees relative to (0, -50)
        rad = math.radians(45.0)
        ankle = (100.0 + 50.0 * math.sin(rad), 100.0 - 50.0 * math.cos(rad))
        angle = calculate_angle(hip, knee, ankle)
        self.assertIsNotNone(angle)
        self.assertAlmostEqual(angle, 45.0, places=4)

    def test_zero_angle_folded_limb(self):
        """Completely folded limb must yield 0 degrees."""
        hip = (100.0, 50.0)
        knee = (100.0, 100.0)
        ankle = (100.0, 50.0)
        angle = calculate_angle(hip, knee, ankle)
        self.assertIsNotNone(angle)
        self.assertAlmostEqual(angle, 0.0, places=4)

    def test_zero_length_limb_vector(self):
        """Coincident points must be rejected as degenerate."""
        hip = (100.0, 100.0)
        knee = (100.0, 100.0)
        ankle = (100.0, 150.0)
        angle = calculate_angle(hip, knee, ankle)
        self.assertIsNone(angle)

    def test_nan_and_infinite_coordinates(self):
        """Non-finite coordinates must return None, never a fabricated angle."""
        knee = (100.0, 100.0)
        ankle = (100.0, 150.0)

        self.assertIsNone(calculate_angle((float("nan"), 50.0), knee, ankle))
        self.assertIsNone(calculate_angle((100.0, float("inf")), knee, ankle))
        self.assertIsNone(calculate_angle((100.0, 50.0), (float("-inf"), 100.0), ankle))

    def test_confidence_validation(self):
        """Confidence score extraction boundaries and rejections."""
        # Points without explicit confidence must be rejected (return None)
        self.assertIsNone(extract_confidence((10.0, 20.0)))
        self.assertEqual(extract_confidence((10.0, 20.0, 0.85)), 0.85)
        self.assertIsNone(extract_confidence((10.0, 20.0, -0.1)))
        self.assertIsNone(extract_confidence((10.0, 20.0, 1.2)))
        self.assertIsNone(extract_confidence((10.0, 20.0, float("nan"))))
        self.assertIsNone(extract_confidence((10.0, 20.0, float("inf"))))
        self.assertIsNone(extract_confidence((10.0, 20.0, True)))
        self.assertIsNone(extract_confidence((10.0, 20.0, "0.85")))

    def test_calculate_angle_types_and_finiteness(self):
        """Invalid types, booleans, and non-finite inputs return None."""
        knee = (100.0, 100.0)
        ankle = (100.0, 150.0)
        self.assertIsNone(calculate_angle((True, 50.0), knee, ankle))
        self.assertIsNone(calculate_angle(("100.0", 50.0), knee, ankle))
        self.assertIsNone(calculate_angle((100.0, 50.0), knee, (100.0, None)))

    def test_evaluate_leg_missing_confidence(self):
        """Keypoints missing confidence dimension are rejected."""
        # 17 keypoints with only (x, y) coordinates
        kpts_2d = [(100.0, 100.0)] * 17
        valid, angle, conf, reason = evaluate_leg(kpts_2d, "left")
        self.assertFalse(valid)
        self.assertIsNone(angle)
        self.assertEqual(reason, "missing_or_invalid_confidence")

    def test_evaluate_leg_missing_indices(self):
        """Incomplete keypoint lists must be rejected."""
        short_kpts = [(0.0, 0.0, 0.9)] * 10
        valid, angle, conf, reason = evaluate_leg(short_kpts, "left")
        self.assertFalse(valid)
        self.assertIsNone(angle)
        self.assertEqual(reason, "missing_keypoint_indices")

    def test_evaluate_leg_low_confidence(self):
        """Keypoints below minimum confidence threshold must be rejected."""
        kpts = [(0.0, 0.0, 0.9)] * 17
        # Make left knee low confidence (idx 13)
        kpts[13] = (100.0, 100.0, 0.25)
        kpts[11] = (100.0, 50.0, 0.9)
        kpts[15] = (100.0, 150.0, 0.9)

        valid, angle, conf, reason = evaluate_leg(kpts, "left", min_confidence=0.3)
        self.assertFalse(valid)
        self.assertIsNone(angle)
        self.assertEqual(conf, 0.25)
        self.assertEqual(reason, "low_joint_confidence")


if __name__ == "__main__":
    unittest.main()
