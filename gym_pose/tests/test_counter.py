"""Unit tests for squat repetition state machine."""

import math
import unittest

from gym_analyser.counter import CounterConfig, SquatCounter


def make_pose(
    left_angle=None,
    right_angle=None,
    left_conf=0.9,
    right_conf=0.9,
):
    """Construct a mock COCO-17 keypoint list with given knee angles."""
    kpts = [(0.0, 0.0, 0.0)] * 17
    if left_angle is not None:
        hip = (100.0, 50.0, left_conf)
        knee = (100.0, 100.0, left_conf)
        rad = math.radians(left_angle)
        ankle = (
            100.0 + 50.0 * math.sin(rad),
            100.0 - 50.0 * math.cos(rad),
            left_conf,
        )
        kpts[11] = hip
        kpts[13] = knee
        kpts[15] = ankle
    if right_angle is not None:
        hip = (200.0, 50.0, right_conf)
        knee = (200.0, 100.0, right_conf)
        rad = math.radians(right_angle)
        ankle = (
            200.0 + 50.0 * math.sin(rad),
            200.0 - 50.0 * math.cos(rad),
            right_conf,
        )
        kpts[12] = hip
        kpts[14] = knee
        kpts[16] = ankle
    return kpts


class TestSquatCounter(unittest.TestCase):
    """Deterministic tests for SquatCounter contract."""

    def setUp(self):
        self.config = CounterConfig(
            down_angle=110.0,
            up_angle=150.0,
            min_joint_confidence=0.3,
            stable_frames=3,
            max_gap_seconds=0.5,
        )
        self.counter = SquatCounter(self.config)

    def test_01_valid_standing_down_standing_cycle(self):
        """A full standing-down-standing sequence increments completed_reps by one."""
        # 3 standing frames to arm
        for i in range(3):
            t = i * 0.1
            res = self.counter.update(
                frame_index=i,
                video_time_s=t,
                detections=[{"track_id": 1, "keypoints": make_pose(left_angle=170.0)}],
            )
            self.assertTrue(res.valid)
            self.assertEqual(res.completed_reps, 0)
        self.assertEqual(self.counter.stage, SquatCounter.STAGE_STANDING)

        # 3 lowered frames
        for i in range(3, 6):
            t = i * 0.1
            res = self.counter.update(
                frame_index=i,
                video_time_s=t,
                detections=[{"track_id": 1, "keypoints": make_pose(left_angle=95.0)}],
            )
            self.assertTrue(res.valid)
            self.assertEqual(res.completed_reps, 0)
        self.assertEqual(self.counter.stage, SquatCounter.STAGE_LOWERED)

        # 3 standing frames to complete return
        for i in range(6, 9):
            t = i * 0.1
            res = self.counter.update(
                frame_index=i,
                video_time_s=t,
                detections=[{"track_id": 1, "keypoints": make_pose(left_angle=170.0)}],
            )
            self.assertTrue(res.valid)

        self.assertEqual(res.completed_reps, 1)
        self.assertEqual(self.counter.stage, SquatCounter.STAGE_STANDING)

    def test_02_crouched_start_rise_counts_zero_subsequent_counts_one(self):
        """A clip starting crouched counts zero on rising; next full cycle counts once."""
        # Starts crouched
        for i in range(4):
            t = i * 0.1
            res = self.counter.update(
                frame_index=i,
                video_time_s=t,
                detections=[{"track_id": 1, "keypoints": make_pose(left_angle=90.0)}],
            )
            self.assertEqual(res.completed_reps, 0)
            self.assertEqual(res.stage, SquatCounter.STAGE_UNKNOWN)

        # Rises to standing
        for i in range(4, 7):
            t = i * 0.1
            res = self.counter.update(
                frame_index=i,
                video_time_s=t,
                detections=[{"track_id": 1, "keypoints": make_pose(left_angle=165.0)}],
            )
            self.assertEqual(res.completed_reps, 0)
        self.assertEqual(res.stage, SquatCounter.STAGE_STANDING)

        # Now perform a full cycle
        for i in range(7, 10):
            t = i * 0.1
            res = self.counter.update(
                frame_index=i,
                video_time_s=t,
                detections=[{"track_id": 1, "keypoints": make_pose(left_angle=95.0)}],
            )
            self.assertEqual(res.completed_reps, 0)
        self.assertEqual(res.stage, SquatCounter.STAGE_LOWERED)

        for i in range(10, 13):
            t = i * 0.1
            res = self.counter.update(
                frame_index=i,
                video_time_s=t,
                detections=[{"track_id": 1, "keypoints": make_pose(left_angle=165.0)}],
            )

        self.assertEqual(res.completed_reps, 1)

    def test_03_partial_dip_and_standing_only_and_ending_down(self):
        """Partial dips, standing-only, and ending-down sequences count zero."""
        # Arm counter
        for i in range(3):
            self.counter.update(
                i, i * 0.1, [{"track_id": 1, "keypoints": make_pose(left_angle=160.0)}]
            )
        self.assertEqual(self.counter.stage, SquatCounter.STAGE_STANDING)

        # Partial dip to 125 degrees (above down_angle 110)
        for i in range(3, 8):
            res = self.counter.update(
                i, i * 0.1, [{"track_id": 1, "keypoints": make_pose(left_angle=125.0)}]
            )
            self.assertEqual(res.stage, SquatCounter.STAGE_STANDING)
            self.assertEqual(res.completed_reps, 0)

        # Return to standing
        for i in range(8, 12):
            res = self.counter.update(
                i, i * 0.1, [{"track_id": 1, "keypoints": make_pose(left_angle=160.0)}]
            )
            self.assertEqual(res.completed_reps, 0)

        # Lower into crouch and end there
        for i in range(12, 16):
            res = self.counter.update(
                i, i * 0.1, [{"track_id": 1, "keypoints": make_pose(left_angle=95.0)}]
            )
        self.assertEqual(res.stage, SquatCounter.STAGE_LOWERED)
        self.assertEqual(res.completed_reps, 0)

    def test_04_threshold_jitter_and_isolated_bad_frame(self):
        """Jitter around threshold and isolated bad frames reset streaks."""
        # Arm counter
        for i in range(3):
            self.counter.update(
                i, i * 0.1, [{"track_id": 1, "keypoints": make_pose(left_angle=160.0)}]
            )

        # Oscillate across down_angle (110)
        angles = [105.0, 115.0, 105.0, 115.0]
        for idx, ang in enumerate(angles, start=3):
            res = self.counter.update(
                idx, idx * 0.1, [{"track_id": 1, "keypoints": make_pose(left_angle=ang)}]
            )
            self.assertEqual(res.stage, SquatCounter.STAGE_STANDING)

        # Two good lowered frames, then an invalid frame
        res1 = self.counter.update(
            8, 0.8, [{"track_id": 1, "keypoints": make_pose(left_angle=95.0)}]
        )
        res2 = self.counter.update(
            9, 0.9, [{"track_id": 1, "keypoints": make_pose(left_angle=95.0)}]
        )
        self.assertEqual(res2.stage, SquatCounter.STAGE_STANDING)

        # Invalid frame (low confidence) clears lowered streak
        res3 = self.counter.update(
            10, 1.0, [{"track_id": 1, "keypoints": make_pose(left_angle=95.0, left_conf=0.1)}]
        )
        self.assertFalse(res3.valid)
        self.assertEqual(self.counter.lowered_streak, 0)

        # Third lowered frame cannot trigger transition because streak was reset
        res4 = self.counter.update(
            11, 1.1, [{"track_id": 1, "keypoints": make_pose(left_angle=95.0)}]
        )
        self.assertEqual(res4.stage, SquatCounter.STAGE_STANDING)

    def test_05_missing_low_conf_nan_degenerate_never_trigger_rep(self):
        """Corrupt or missing observations do not trigger transitions or reps."""
        # Arm counter
        for i in range(3):
            self.counter.update(
                i, i * 0.1, [{"track_id": 1, "keypoints": make_pose(left_angle=160.0)}]
            )

        bad_kpts = [(float("nan"), 0.0, 0.9)] * 17
        res = self.counter.update(3, 0.3, [{"track_id": 1, "keypoints": bad_kpts}])
        self.assertFalse(res.valid)
        self.assertEqual(res.completed_reps, 0)

        # Empty detections
        res_empty = self.counter.update(4, 0.4, [])
        self.assertFalse(res_empty.valid)
        self.assertEqual(res_empty.completed_reps, 0)

    def test_06_gap_abandons_partial_cycle_preserves_completed(self):
        """Time gap over max_gap_seconds abandons partial cycle and preserves completed reps."""
        # Complete rep 1
        t = 0.0
        for _ in range(3):
            self.counter.update(0, t, [{"track_id": 1, "keypoints": make_pose(left_angle=160.0)}])
            t += 0.1
        for _ in range(3):
            self.counter.update(0, t, [{"track_id": 1, "keypoints": make_pose(left_angle=90.0)}])
            t += 0.1
        for _ in range(3):
            res = self.counter.update(0, t, [{"track_id": 1, "keypoints": make_pose(left_angle=160.0)}])
            t += 0.1
        self.assertEqual(res.completed_reps, 1)

        # Start rep 2 into LOWERED
        for _ in range(3):
            res = self.counter.update(0, t, [{"track_id": 1, "keypoints": make_pose(left_angle=90.0)}])
            t += 0.1
        self.assertEqual(res.stage, SquatCounter.STAGE_LOWERED)

        # Introduce a 0.8s gap (> max_gap_seconds 0.5)
        t += 0.8
        res_gap = self.counter.update(
            0, t, [{"track_id": 1, "keypoints": make_pose(left_angle=160.0)}]
        )
        # Timeout abandons partial cycle and resets to UNKNOWN before processing new frame
        self.assertEqual(res_gap.completed_reps, 1)
        self.assertEqual(self.counter.stage, SquatCounter.STAGE_UNKNOWN)

    def test_07_timeout_applied_before_consuming_next_valid(self):
        """Timeout triggers before consuming valid observation, preventing instant completion."""
        # Arm counter and lower
        t = 0.0
        for _ in range(3):
            self.counter.update(0, t, [{"track_id": 1, "keypoints": make_pose(left_angle=160.0)}])
            t += 0.1
        for _ in range(3):
            self.counter.update(0, t, [{"track_id": 1, "keypoints": make_pose(left_angle=90.0)}])
            t += 0.1
        self.assertEqual(self.counter.stage, SquatCounter.STAGE_LOWERED)

        # Jump forward by 1.0 second with valid standing observation
        t += 1.0
        res = self.counter.update(0, t, [{"track_id": 1, "keypoints": make_pose(left_angle=165.0)}])
        # Timeout was enforced, resetting to UNKNOWN. Standing streak is now 1 in UNKNOWN.
        self.assertEqual(res.completed_reps, 0)
        self.assertEqual(self.counter.stage, SquatCounter.STAGE_UNKNOWN)

    def test_08_alternating_visible_sides_mid_cycle(self):
        """Locked side prevents mid-cycle side swapping from fabricating reps."""
        # Arm on left side
        t = 0.0
        for _ in range(3):
            self.counter.update(
                0, t, [{"track_id": 1, "keypoints": make_pose(left_angle=165.0, left_conf=0.9)}]
            )
            t += 0.1
        self.assertEqual(self.counter.locked_side, "left")

        # Lower on left side
        for _ in range(3):
            self.counter.update(
                0, t, [{"track_id": 1, "keypoints": make_pose(left_angle=95.0, left_conf=0.9)}]
            )
            t += 0.1
        self.assertEqual(self.counter.stage, SquatCounter.STAGE_LOWERED)

        # Next frames have left side occluded, right side standing
        for _ in range(3):
            res = self.counter.update(
                0,
                t,
                [
                    {
                        "track_id": 1,
                        "keypoints": make_pose(left_angle=None, right_angle=165.0, right_conf=0.9),
                    }
                ],
            )
            t += 0.1
            self.assertFalse(res.valid)
            self.assertEqual(res.completed_reps, 0)

        # Cycle remains uncompleted
        self.assertEqual(self.counter.completed_reps, 0)

    def test_09_subject_id_change_cannot_complete_cycle(self):
        """A different track ID cannot complete the locked subject's cycle."""
        # Arm subject 1 and lower
        t = 0.0
        for _ in range(3):
            self.counter.update(0, t, [{"track_id": 1, "keypoints": make_pose(left_angle=160.0)}])
            t += 0.1
        for _ in range(3):
            self.counter.update(0, t, [{"track_id": 1, "keypoints": make_pose(left_angle=90.0)}])
            t += 0.1
        self.assertEqual(self.counter.stage, SquatCounter.STAGE_LOWERED)

        # Subject 2 appears standing
        res = self.counter.update(
            0, t, [{"track_id": 2, "keypoints": make_pose(left_angle=160.0)}]
        )
        self.assertFalse(res.valid)
        self.assertIn("track_id_mismatch", res.reason)
        self.assertEqual(res.completed_reps, 0)

    def test_10_bad_thresholds_and_nonmonotonic_timestamps(self):
        """Configuration validation and non-monotonic timestamps raise ValueError."""
        with self.assertRaises(ValueError):
            CounterConfig(down_angle=160.0, up_angle=120.0)

        with self.assertRaises(ValueError):
            CounterConfig(down_angle=-10.0, up_angle=120.0)

        with self.assertRaises(ValueError):
            CounterConfig(down_angle=110.0, up_angle=190.0)

        with self.assertRaises(ValueError):
            CounterConfig(stable_frames=0)

        with self.assertRaises(ValueError):
            CounterConfig(max_gap_seconds=-0.5)

        # Non-monotonic timestamp in update
        self.counter.update(0, 1.0, [{"track_id": 1, "keypoints": make_pose(left_angle=160.0)}])
        with self.assertRaises(ValueError):
            self.counter.update(1, 0.8, [{"track_id": 1, "keypoints": make_pose(left_angle=160.0)}])


if __name__ == "__main__":
    unittest.main()
