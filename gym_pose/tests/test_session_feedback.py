"""Unit tests for standalone per-attempt posture summary and session feedback module.

All fixtures are purely synthetic and do not depend on local external files,
models, or network calls.
"""

import math
import unittest
from typing import Any, Dict, List

from gym_analyser.session_feedback import (
    summarise_attempt_posture,
    summarize_attempt_posture,
)


class TestSessionFeedback(unittest.TestCase):
    """Test suite verifying per-attempt posture summarisation and validation."""

    def _make_row(
        self,
        frame_index: int,
        track_id: int = 1,
        side: str = "right",
        torso_deg: Any = "15.00",
        hip_deg: Any = "140.00",
        torso_reason: str = "ok",
        hip_reason: str = "ok",
        valid: bool = True,
        reason: str = "ok",
        attempt_id: Any = "",
    ) -> Dict[str, Any]:
        """Create a synthetic angles.csv frame row dictionary."""
        return {
            "frame_index": frame_index,
            "video_time_s": f"{frame_index / 30.0:.4f}",
            "track_id": str(track_id) if track_id is not None else "",
            "selected_side": side,
            "knee_angle_deg": "145.00" if valid else "",
            "confidence": "0.9000",
            "valid": "True" if valid else "False",
            "reason": reason,
            "stage": "standing",
            "completed_reps": "0",
            "posture_side": side,
            "torso_inclination_deg": str(torso_deg) if torso_deg is not None else "",
            "hip_angle_deg": str(hip_deg) if hip_deg is not None else "",
            "torso_reason": torso_reason,
            "hip_reason": hip_reason,
            "attempt_id": str(attempt_id) if attempt_id != "" else "",
            "last_attempt_reason": "",
        }

    def test_clean_single_counted_event_full_coverage(self):
        """Single counted event with full valid frame coverage matches statistics."""
        rows = [
            self._make_row(i, torso_deg=f"{10.0 + i:.2f}", hip_deg=f"{130.0 + i:.2f}", attempt_id=1)
            for i in range(10, 20)
        ]
        events = [
            {
                "attempt_id": 1,
                "start_frame": 10,
                "end_frame": 19,
                "track_id": 1,
                "side": "right",
                "status": "counted",
                "reason": "completed_cycle",
            }
        ]

        result = summarise_attempt_posture(rows, events)

        self.assertEqual(result["summary"]["total_attempt_events"], 1)
        self.assertEqual(result["summary"]["counted_events"], 1)
        self.assertEqual(result["summary"]["uncounted_events"], 0)
        self.assertEqual(result["summary"]["events_with_full_torso_coverage"], 1)
        self.assertEqual(result["summary"]["events_with_full_hip_coverage"], 1)

        ev = result["events"][0]
        self.assertEqual(ev["attempt_id"], 1)
        self.assertEqual(ev["status"], "counted")
        self.assertEqual(ev["reason"], "completed_cycle")
        self.assertEqual(ev["total_frames"], 10)

        # Torso checks
        torso = ev["torso"]
        self.assertEqual(torso["measured_count"], 10)
        self.assertEqual(torso["total_frames"], 10)
        self.assertEqual(torso["coverage_fraction"], 1.0)
        self.assertEqual(torso["min_deg"], 20.00)  # 10.0 + 10 = 20.0
        self.assertEqual(torso["max_deg"], 29.00)  # 10.0 + 19 = 29.0
        self.assertEqual(torso["median_deg"], 24.50)  # average of 24 and 25
        self.assertEqual(torso["unavailable_reasons"], {})

        # Hip checks
        hip = ev["hip"]
        self.assertEqual(hip["measured_count"], 10)
        self.assertEqual(hip["total_frames"], 10)
        self.assertEqual(hip["coverage_fraction"], 1.0)
        self.assertEqual(hip["min_deg"], 140.00)
        self.assertEqual(hip["max_deg"], 149.00)
        self.assertEqual(hip["median_deg"], 144.50)
        self.assertEqual(hip["unavailable_reasons"], {})

    def test_clean_single_uncounted_event_neutral_observations(self):
        """Uncounted attempt generates neutral descriptive observations without form grading."""
        rows = [
            self._make_row(i, torso_deg="18.25", hip_deg="92.10", attempt_id=1)
            for i in range(5, 15)
        ]
        events = [
            {
                "attempt_id": 1,
                "start_frame": 5,
                "end_frame": 14,
                "track_id": 1,
                "side": "right",
                "status": "uncounted",
                "reason": "depth_threshold_not_reached",
            }
        ]

        result = summarise_attempt_posture(rows, events)

        self.assertEqual(result["summary"]["counted_events"], 0)
        self.assertEqual(result["summary"]["uncounted_events"], 1)

        ev = result["events"][0]
        self.assertEqual(ev["status"], "uncounted")
        self.assertEqual(ev["reason"], "depth_threshold_not_reached")
        self.assertEqual(ev["torso"]["measured_count"], 10)
        self.assertEqual(ev["torso"]["median_deg"], 18.25)
        self.assertEqual(ev["hip"]["median_deg"], 92.10)

        # Verify neutral descriptive observations
        obs = " ".join(ev["observations"])
        self.assertIn("Torso inclination relative to upward image vertical", obs)
        self.assertIn("Hip joint angle", obs)
        # Verify absence of value judgements
        for forbidden in ["bad", "unsafe", "poor", "incorrect", "fail", "score", "wrong form"]:
            self.assertNotIn(forbidden, obs.lower())

    def test_genuinely_available_torso_when_knee_and_hip_invalid(self):
        """Torso is measured when knee and hip are invalid; knee-validity is not required."""
        # 5 frames where knee is invalid and hip is invalid, but torso is confident and valid
        rows = []
        for i in range(10, 15):
            r = self._make_row(
                i,
                valid=False,  # knee invalid in counter
                reason="knee_confidence_low",
                torso_deg="22.50",
                torso_reason="ok",
                hip_deg="",  # hip invalid
                hip_reason="low_joint_confidence",
                attempt_id=1,
            )
            rows.append(r)

        events = [
            {
                "attempt_id": 1,
                "start_frame": 10,
                "end_frame": 14,
                "track_id": 1,
                "side": "right",
                "status": "uncounted",
                "reason": "tracking_gap",
            }
        ]

        result = summarise_attempt_posture(rows, events)
        ev = result["events"][0]

        # Torso MUST be fully measured (independent of knee/hip)
        self.assertEqual(ev["torso"]["measured_count"], 5)
        self.assertEqual(ev["torso"]["coverage_fraction"], 1.0)
        self.assertEqual(ev["torso"]["median_deg"], 22.50)
        self.assertEqual(ev["torso"]["unavailable_reasons"], {})

        # Hip MUST be unavailable with low_joint_confidence
        self.assertEqual(ev["hip"]["measured_count"], 0)
        self.assertEqual(ev["hip"]["coverage_fraction"], 0.0)
        self.assertIsNone(ev["hip"]["median_deg"])
        self.assertEqual(ev["hip"]["unavailable_reasons"], {"low_joint_confidence": 5})

    def test_missing_blank_csv_values_remain_none_not_zero(self):
        """Missing or blank CSV angle entries remain None and are never converted to zero."""
        rows = [
            self._make_row(i, torso_deg="", hip_deg=None, torso_reason="low_joint_confidence", hip_reason="ok", attempt_id=1)
            for i in range(1, 6)
        ]
        events = [{"attempt_id": 1, "start_frame": 1, "end_frame": 5, "track_id": 1, "side": "right"}]

        result = summarise_attempt_posture(rows, events)
        ev = result["events"][0]

        # Values must remain None, never 0.0
        self.assertEqual(ev["torso"]["measured_count"], 0)
        self.assertEqual(ev["torso"]["coverage_fraction"], 0.0)
        self.assertIsNone(ev["torso"]["median_deg"])
        self.assertIsNone(ev["torso"]["min_deg"])
        self.assertIsNone(ev["torso"]["max_deg"])
        self.assertNotEqual(ev["torso"]["median_deg"], 0.0)
        self.assertEqual(ev["torso"]["unavailable_reasons"], {"low_joint_confidence": 5})

        self.assertEqual(ev["hip"]["measured_count"], 0)
        self.assertEqual(ev["hip"]["coverage_fraction"], 0.0)
        self.assertIsNone(ev["hip"]["median_deg"])
        self.assertNotEqual(ev["hip"]["median_deg"], 0.0)
        self.assertEqual(ev["hip"]["unavailable_reasons"], {"missing_measurement": 5})

    def test_measured_zero_degree_angle_is_valid_not_missing(self):
        """A measured 0.0 degree angle (perfectly vertical torso) is valid and preserved."""
        # Frame with string "0.00" and float 0.0
        rows = [
            self._make_row(10, torso_deg="0.00", hip_deg="180.00", attempt_id=1),
            self._make_row(11, torso_deg=0.0, hip_deg=180.0, attempt_id=1),
            self._make_row(12, torso_deg="0", hip_deg="180", attempt_id=1),
        ]
        events = [{"attempt_id": 1, "start_frame": 10, "end_frame": 12, "track_id": 1, "side": "right"}]

        result = summarise_attempt_posture(rows, events)
        ev = result["events"][0]

        self.assertEqual(ev["torso"]["measured_count"], 3)
        self.assertEqual(ev["torso"]["coverage_fraction"], 1.0)
        self.assertEqual(ev["torso"]["min_deg"], 0.0)
        self.assertEqual(ev["torso"]["max_deg"], 0.0)
        self.assertEqual(ev["torso"]["median_deg"], 0.0)
        self.assertEqual(ev["torso"]["unavailable_reasons"], {})

    def test_track_filtering_suppresses_mismatched_track(self):
        """Rows from a mismatched track ID are suppressed from posture statistics."""
        # Event is on track 1, but rows are on track 2
        rows = [self._make_row(i, track_id=2, torso_deg="15.00", hip_deg="120.00") for i in range(10, 15)]
        events = [{"attempt_id": 1, "start_frame": 10, "end_frame": 14, "track_id": 1, "side": "right"}]

        result = summarise_attempt_posture(rows, events)
        ev = result["events"][0]

        self.assertEqual(ev["torso"]["measured_count"], 0)
        self.assertEqual(ev["hip"]["measured_count"], 0)
        self.assertEqual(ev["torso"]["unavailable_reasons"], {"track_mismatch": 5})
        self.assertEqual(ev["hip"]["unavailable_reasons"], {"track_mismatch": 5})

    def test_side_filtering_suppresses_mismatched_side(self):
        """Rows from the opposite side are suppressed from posture statistics."""
        # Event is on right side, but rows are on left side
        rows = [self._make_row(i, side="left", torso_deg="15.00", hip_deg="120.00") for i in range(10, 15)]
        events = [{"attempt_id": 1, "start_frame": 10, "end_frame": 14, "track_id": 1, "side": "right"}]

        result = summarise_attempt_posture(rows, events)
        ev = result["events"][0]

        self.assertEqual(ev["torso"]["measured_count"], 0)
        self.assertEqual(ev["hip"]["measured_count"], 0)
        self.assertEqual(ev["torso"]["unavailable_reasons"], {"side_mismatch": 5})
        self.assertEqual(ev["hip"]["unavailable_reasons"], {"side_mismatch": 5})

    def test_ambiguous_subjects_suppressed(self):
        """Frames marked multiple_subjects_ambiguous are suppressed."""
        rows = [
            self._make_row(10, reason="ok", torso_deg="15.00", hip_deg="120.00", attempt_id=1),
            self._make_row(11, reason="multiple_subjects_ambiguous", torso_deg="15.00", hip_deg="120.00", attempt_id=1),
            self._make_row(12, reason="ok", torso_deg="17.00", hip_deg="122.00", attempt_id=1),
        ]
        events = [{"attempt_id": 1, "start_frame": 10, "end_frame": 12, "track_id": 1, "side": "right"}]

        result = summarise_attempt_posture(rows, events)
        ev = result["events"][0]

        self.assertEqual(ev["torso"]["measured_count"], 2)
        self.assertEqual(ev["torso"]["total_frames"], 3)
        self.assertEqual(ev["torso"]["unavailable_reasons"], {"multiple_subjects_ambiguous": 1})
        self.assertEqual(ev["hip"]["measured_count"], 2)
        self.assertEqual(ev["hip"]["unavailable_reasons"], {"multiple_subjects_ambiguous": 1})

    def test_backdated_opening_row_with_blank_attempt_id(self):
        """Opening departure frame with blank attempt_id in CSV is mapped by [start_frame, end_frame]."""
        # Frame 42 was the first departure frame (before 2-frame debounce confirmed at frame 43)
        rows = [
            self._make_row(42, torso_deg="11.71", hip_deg="138.21", attempt_id=""),  # blank attempt_id!
            self._make_row(43, torso_deg="12.34", hip_deg="133.98", attempt_id=1),
            self._make_row(44, torso_deg="12.32", hip_deg="134.10", attempt_id=1),
        ]
        events = [{"attempt_id": 1, "start_frame": 42, "end_frame": 44, "track_id": 1, "side": "right"}]

        result = summarise_attempt_posture(rows, events)
        ev = result["events"][0]

        # All 3 frames must be successfully included!
        self.assertEqual(ev["torso"]["measured_count"], 3)
        self.assertEqual(ev["torso"]["total_frames"], 3)
        self.assertEqual(ev["torso"]["coverage_fraction"], 1.0)
        self.assertEqual(ev["torso"]["min_deg"], 11.71)

    def test_conflicting_attempt_id_raises_value_error(self):
        """Explicit attempt_id in CSV conflicting with event attempt_id raises ValueError."""
        rows = [
            self._make_row(42, attempt_id=1),
            self._make_row(43, attempt_id=2),  # Conflict! Event is attempt 1
        ]
        events = [{"attempt_id": 1, "start_frame": 42, "end_frame": 43, "track_id": 1, "side": "right"}]

        with self.assertRaises(ValueError) as ctx:
            summarise_attempt_posture(rows, events)
        self.assertIn("incompatible with event attempt_id", str(ctx.exception))

    def test_validation_descending_frame_indices(self):
        """Descending frame indices raise ValueError."""
        rows = [self._make_row(0), self._make_row(2), self._make_row(1)]
        events = [{"attempt_id": 1, "start_frame": 0, "end_frame": 2}]

        with self.assertRaises(ValueError) as ctx:
            summarise_attempt_posture(rows, events)
        self.assertIn("Descending frame_index", str(ctx.exception))

    def test_validation_duplicate_frame_indices(self):
        """Duplicate frame indices raise ValueError."""
        rows = [self._make_row(0), self._make_row(1), self._make_row(1)]
        events = [{"attempt_id": 1, "start_frame": 0, "end_frame": 1}]

        with self.assertRaises(ValueError) as ctx:
            summarise_attempt_posture(rows, events)
        self.assertIn("Duplicate frame_index", str(ctx.exception))

    def test_validation_negative_frame_index(self):
        """Negative frame index raises ValueError."""
        rows = [self._make_row(-1), self._make_row(0)]
        events = [{"attempt_id": 1, "start_frame": 0, "end_frame": 0}]

        with self.assertRaises(ValueError) as ctx:
            summarise_attempt_posture(rows, events)
        self.assertIn("non-negative", str(ctx.exception))

    def test_validation_boolean_frame_index(self):
        """Boolean frame index raises ValueError."""
        rows = [self._make_row(True)]
        events = [{"attempt_id": 1, "start_frame": 0, "end_frame": 0}]

        with self.assertRaises(ValueError) as ctx:
            summarise_attempt_posture(rows, events)
        self.assertIn("boolean", str(ctx.exception))

    def test_validation_overlapping_events(self):
        """Overlapping event boundaries raise ValueError."""
        rows = [self._make_row(i) for i in range(10, 30)]
        events = [
            {"attempt_id": 1, "start_frame": 10, "end_frame": 20},
            {"attempt_id": 2, "start_frame": 18, "end_frame": 28},  # Overlaps [10, 20]
        ]

        with self.assertRaises(ValueError) as ctx:
            summarise_attempt_posture(rows, events)
        self.assertIn("Overlapping event intervals", str(ctx.exception))

    def test_validation_touching_inclusive_events(self):
        """Events sharing an inclusive boundary frame raise ValueError."""
        rows = [self._make_row(i) for i in range(10, 31)]
        events = [
            {"attempt_id": 1, "start_frame": 10, "end_frame": 20},
            {"attempt_id": 2, "start_frame": 20, "end_frame": 30},  # Shares frame 20!
        ]

        with self.assertRaises(ValueError) as ctx:
            summarise_attempt_posture(rows, events)
        self.assertIn("Overlapping event intervals", str(ctx.exception))

    def test_validation_end_frame_less_than_start_frame(self):
        """Event with end_frame < start_frame raises ValueError."""
        rows = [self._make_row(i) for i in range(10, 20)]
        events = [{"attempt_id": 1, "start_frame": 15, "end_frame": 10}]

        with self.assertRaises(ValueError) as ctx:
            summarise_attempt_posture(rows, events)
        self.assertIn("end_frame (10) < start_frame (15)", str(ctx.exception))

    def test_validation_negative_event_boundaries(self):
        """Negative event start_frame raises ValueError."""
        rows = [self._make_row(0)]
        events = [{"attempt_id": 1, "start_frame": -5, "end_frame": 0}]

        with self.assertRaises(ValueError) as ctx:
            summarise_attempt_posture(rows, events)
        self.assertIn("non-negative", str(ctx.exception))

    def test_validation_non_finite_video_time(self):
        """Non-finite video_time_s raises ValueError."""
        r = self._make_row(0)
        r["video_time_s"] = "NaN"
        events = [{"attempt_id": 1, "start_frame": 0, "end_frame": 0}]

        with self.assertRaises(ValueError) as ctx:
            summarise_attempt_posture([r], events)
        self.assertIn("must be finite", str(ctx.exception))

    def test_unobserved_frames_in_event_interval(self):
        """Dropped/unobserved frames in event interval are tracked under missing_frame."""
        # Stream has frames 10, 11, 14 (frames 12 and 13 are missing from the stream)
        rows = [
            self._make_row(10, torso_deg="15.00", hip_deg="130.00", attempt_id=1),
            self._make_row(11, torso_deg="16.00", hip_deg="131.00", attempt_id=1),
            self._make_row(14, torso_deg="19.00", hip_deg="134.00", attempt_id=1),
        ]
        events = [{"attempt_id": 1, "start_frame": 10, "end_frame": 14, "track_id": 1, "side": "right"}]

        result = summarise_attempt_posture(rows, events)
        ev = result["events"][0]

        self.assertEqual(ev["total_frames"], 5)
        self.assertEqual(ev["torso"]["measured_count"], 3)
        self.assertEqual(ev["torso"]["total_frames"], 5)
        self.assertEqual(ev["torso"]["coverage_fraction"], 0.6)
        self.assertEqual(ev["torso"]["unavailable_reasons"], {"missing_frame": 2})

    def test_mathematical_invariant_measured_plus_unavailable_equals_total(self):
        """Invariant: measured_count + sum(unavailable_reasons.values()) == total_frames."""
        rows = [
            self._make_row(10, torso_deg="15.00", hip_deg="130.00"),  # measured
            self._make_row(11, torso_deg="", hip_deg="", torso_reason="low_joint_confidence", hip_reason="low_joint_confidence"),  # low conf
            self._make_row(12, track_id=99),  # track mismatch
            # Frame 13 missing from stream
            self._make_row(14, reason="multiple_subjects_ambiguous"),  # ambiguous
        ]
        events = [{"attempt_id": 1, "start_frame": 10, "end_frame": 14, "track_id": 1, "side": "right"}]

        result = summarise_attempt_posture(rows, events)
        ev = result["events"][0]

        total = ev["total_frames"]  # 5
        self.assertEqual(total, 5)

        t_measured = ev["torso"]["measured_count"]
        t_unavail_sum = sum(ev["torso"]["unavailable_reasons"].values())
        self.assertEqual(t_measured + t_unavail_sum, total)

        h_measured = ev["hip"]["measured_count"]
        h_unavail_sum = sum(ev["hip"]["unavailable_reasons"].values())
        self.assertEqual(h_measured + h_unavail_sum, total)

    def test_csv_precision_limits_documented(self):
        """Output metadata documents two-decimal-place precision limitation."""
        rows = [self._make_row(0, torso_deg="15.25", hip_deg="130.50")]
        events = [{"attempt_id": 1, "start_frame": 0, "end_frame": 0}]

        result = summarise_attempt_posture(rows, events)
        self.assertIn("telemetry_limitations", result)
        self.assertEqual(result["telemetry_limitations"]["angle_precision_deg"], 0.01)
        self.assertIn("two decimal places", result["telemetry_limitations"]["precision_note"])

    def test_biomechanical_limitations_included(self):
        """Output contains explicit 2D monocular lateral-view biomechanical limitations."""
        rows = [self._make_row(0)]
        events = [{"attempt_id": 1, "start_frame": 0, "end_frame": 0}]

        result = summarise_attempt_posture(rows, events)
        self.assertIn("biomechanical_limitations", result)
        limits_text = " ".join(result["biomechanical_limitations"])
        self.assertIn("lateral view", limits_text)
        self.assertIn("bilateral symmetry", limits_text)
        self.assertIn("Frontal and transverse plane", limits_text)

    def test_accepts_attempts_json_dict_wrapper(self):
        """Function accepts both a raw event list and an attempts.json dict containing 'events'."""
        rows = [self._make_row(0)]
        events_dict = {
            "configuration": {},
            "events": [{"attempt_id": 1, "start_frame": 0, "end_frame": 0}],
        }

        result = summarise_attempt_posture(rows, events_dict)
        self.assertEqual(len(result["events"]), 1)

    def test_american_english_alias(self):
        """American English summarize_attempt_posture is identical to summarise_attempt_posture."""
        self.assertIs(summarize_attempt_posture, summarise_attempt_posture)


if __name__ == "__main__":
    unittest.main()
