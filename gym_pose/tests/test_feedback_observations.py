"""Unit tests for standalone posture observations and session feedback aggregation.

Tests target observable behavior and edge cases:
- Single attempt honest limitations (no fictional consistency or form critique)
- Measured torso inclination differences within the SAME track and side across COUNTED cycles
- Exclusion of uncounted (setup/return/interrupted) movements from cross-attempt comparisons
- Separation of different tracks and different body sides (no cross-comparisons)
- Unknown/unselected side never becoming a comparison group
- Minimum knee angle reliability requiring verified valid_frames coverage (low valid frame extreme)
- Supplementary event validation (rejecting duplicates, mismatches, and malformed shapes)
- Strict integer validation (rejecting booleans, fractions, and negatives)
- Metrics bounds and ordering validation (min <= median <= max, coverage in [0, 1])
- Treatment of 0.0Ã‚° as a valid measured angle (not missing or None)
- Missing and NaN data handling with clear unavailability reasons
- Context notes using neutral intent language ("may/cannot determine intent")
- Preservation of upstream explanations, counts, and reason codes
- Immutability of input structures
- Real summarise_attempt_posture and AttemptTracker-shaped schema fixtures
- Honest scope limitations without superficial test scans

All fixtures are synthetic and standard-library only.
"""

import copy
import math
import unittest
from typing import Any, Dict, List

from gym_analyser.feedback_observations import (
    ObservationConfig,
    build_feedback_observations,
    build_feedback_observations_from_raw,
    summarize_feedback_observations,
)
from gym_analyser.session_feedback import summarise_attempt_posture


class TestFeedbackObservations(unittest.TestCase):
    """Test suite verifying feedback observations and aggregation behavior."""

    def _make_posture_event(
        self,
        attempt_id: int = 1,
        status: str = "counted",
        reason: str = "completed_cycle",
        track_id: int = 1,
        side: str = "right",
        start_frame: int = 10,
        end_frame: int = 29,
        torso_med: Any = 15.0,
        torso_min: Any = 10.0,
        torso_max: Any = 20.0,
        torso_cov: float = 1.0,
        torso_count: int = None,
        torso_unavail: Dict[str, int] = None,
        hip_med: Any = 120.0,
        hip_min: Any = 110.0,
        hip_max: Any = 130.0,
        hip_cov: float = 1.0,
        hip_count: int = None,
        hip_unavail: Dict[str, int] = None,
        min_knee_angle_deg: Any = 92.5,
    ) -> Dict[str, Any]:
        """Create a synthetic posture summary event matching session_feedback schema."""
        total_frames = end_frame - start_frame + 1
        torso_count = total_frames if torso_count is None else torso_count
        hip_count = total_frames if hip_count is None else hip_count
        t_dict = {
            "measured_count": torso_count,
            "total_frames": total_frames,
            "coverage_fraction": torso_cov,
            "median_deg": torso_med,
            "min_deg": torso_min,
            "max_deg": torso_max,
            "unavailable_reasons": torso_unavail or {},
        }
        h_dict = {
            "measured_count": hip_count,
            "total_frames": total_frames,
            "coverage_fraction": hip_cov,
            "median_deg": hip_med,
            "min_deg": hip_min,
            "max_deg": hip_max,
            "unavailable_reasons": hip_unavail or {},
        }
        return {
            "attempt_id": attempt_id,
            "status": status,
            "reason": reason,
            "start_frame": start_frame,
            "end_frame": end_frame,
            "total_frames": total_frames,
            "track_id": track_id,
            "side": side,
            "torso_inclination": t_dict,
            "hip_angle": h_dict,
            "torso": t_dict,
            "hip": h_dict,
            "min_knee_angle_deg": min_knee_angle_deg,
            "observations": ["Descriptive summary."],
        }

    def _make_attempt_supplement(
        self,
        attempt_id: int = 1,
        status: str = "counted",
        reason: str = "completed_cycle",
        track_id: int = 1,
        side: str = "right",
        start_frame: int = 10,
        end_frame: int = 29,
        min_knee_angle_deg: float = 92.5,
        depth_confirmed: bool = True,
        category: str = "completed",
        valid_frames: int = 20,
        observed_frames: int = 20,
        explanation: str = "A complete cycle met the configured counting thresholds.",
    ) -> Dict[str, Any]:
        """Create supplementary attempt metadata matching attempts.py schema."""
        return {
            "attempt_id": attempt_id,
            "status": status,
            "reason": reason,
            "track_id": track_id,
            "side": side,
            "start_frame": start_frame,
            "end_frame": end_frame,
            "min_knee_angle_deg": min_knee_angle_deg,
            "depth_confirmed": depth_confirmed,
            "category": category,
            "valid_frames": valid_frames,
            "observed_frames": observed_frames,
            "explanation": explanation,
        }

    def test_single_attempt_honest_limitation(self):
        """Single attempt reports valid telemetry but honestly declines cross-attempt consistency."""
        ev = self._make_posture_event(attempt_id=1, torso_med=16.5, min_knee_angle_deg=94.0)
        posture_summary = {"summary": {}, "events": [ev]}

        result = build_feedback_observations(posture_summary)

        summary = result["summary"]
        self.assertEqual(summary["total_attempt_events"], 1)
        self.assertEqual(summary["counted_events"], 1)
        self.assertEqual(summary["uncounted_events"], 0)

        # Single-attempt session observation
        session_text = " ".join(summary["session_observations"])
        self.assertIn("only 1 observed movement attempt", session_text)
        self.assertIn("cannot be evaluated from a single attempt", session_text)

        # Group comparison should decline comparison
        cmp_group = summary["comparisons"][0]
        self.assertFalse(cmp_group["torso_inclination"]["available"])
        self.assertFalse(cmp_group["movement_depth"]["available"])
        self.assertIn("requires at least 2 counted cycles", " ".join(cmp_group["limitations"]))

        # Attempt level observations must be informative and non-normative
        att = result["attempts"][0]
        self.assertTrue(att["availability"]["torso"]["available"])
        self.assertEqual(att["evidence"]["torso_median_deg"], 16.50)

    def test_counted_only_cross_attempt_comparison_excludes_uncounted(self):
        """Cross-attempt comparison compares ONLY counted cycles, excluding uncounted movements."""
        ev1 = self._make_posture_event(attempt_id=1, status="counted", start_frame=10, end_frame=29, torso_med=15.00)
        # Attempt 2 is uncounted setup/movement
        ev2 = self._make_posture_event(
            attempt_id=2, status="uncounted", reason="depth_threshold_not_reached",
            start_frame=35, end_frame=55, torso_med=30.00, torso_max=40.0, min_knee_angle_deg=125.0
        )
        ev3 = self._make_posture_event(attempt_id=3, status="counted", start_frame=60, end_frame=80, torso_med=20.00)

        posture_summary = {"summary": {}, "events": [ev1, ev2, ev3]}
        supps = [
            self._make_attempt_supplement(attempt_id=1, status="counted", start_frame=10, end_frame=29, min_knee_angle_deg=92.0),
            self._make_attempt_supplement(
                attempt_id=2, status="uncounted", reason="depth_threshold_not_reached",
                start_frame=35, end_frame=55, min_knee_angle_deg=125.0, category="movement_threshold"
            ),
            self._make_attempt_supplement(attempt_id=3, status="counted", start_frame=60, end_frame=80, min_knee_angle_deg=96.0),
        ]

        result = build_feedback_observations(posture_summary, attempt_events=supps)

        cmp_group = result["summary"]["comparisons"][0]
        self.assertEqual(cmp_group["counted_attempts_in_group"], 2)
        self.assertEqual(cmp_group["uncounted_attempts_in_group"], 1)

        # Torso comparison must evaluate ONLY Attempt 1 (15.0) and Attempt 3 (20.0), NOT Attempt 2 (30.0)
        torso_cmp = cmp_group["torso_inclination"]
        self.assertTrue(torso_cmp["available"])
        self.assertEqual(torso_cmp["attempt_ids"], [1, 3])
        self.assertEqual(torso_cmp["min_median_deg"], 15.00)
        self.assertEqual(torso_cmp["max_median_deg"], 20.00)
        self.assertEqual(torso_cmp["spread_deg"], 5.00)
        self.assertEqual(torso_cmp["group_median_deg"], 17.50)

        # Depth comparison must evaluate ONLY Attempt 1 (92.0) and Attempt 3 (96.0)
        depth_cmp = cmp_group["movement_depth"]
        self.assertTrue(depth_cmp["available"])
        self.assertEqual(depth_cmp["attempt_ids"], [1, 3])
        self.assertEqual(depth_cmp["min_knee_angle_deg"], 92.00)
        self.assertEqual(depth_cmp["max_knee_angle_deg"], 96.00)
        self.assertEqual(depth_cmp["spread_deg"], 4.00)

        # Limitation clearly notes the excluded uncounted attempt
        self.assertIn("1 unconfirmed segment(s) on track 1 (right side) excluded", " ".join(cmp_group["limitations"]))

    def test_unknown_side_never_becomes_comparison_group(self):
        """Attempts with unknown or unselected side are never grouped for cross-attempt comparisons."""
        ev1 = self._make_posture_event(attempt_id=1, side="unknown", torso_med=15.00)
        ev2 = self._make_posture_event(attempt_id=2, side="unknown", torso_med=18.00)
        posture_summary = {"summary": {}, "events": [ev1, ev2]}

        result = build_feedback_observations(posture_summary)

        # Unknown side must never form a comparison group
        self.assertEqual(len(result["summary"]["comparisons"]), 0)
        self.assertEqual(len(result["attempts"]), 2)
        self.assertEqual(result["attempts"][0]["side"], "unknown")

    def test_low_valid_frame_knee_extreme_excluded_from_depth_comparison(self):
        """Knee depth with low valid-frame coverage is marked unverified and excluded from comparisons."""
        ev1 = self._make_posture_event(attempt_id=1, start_frame=0, end_frame=99, min_knee_angle_deg=90.0)
        ev2 = self._make_posture_event(attempt_id=2, start_frame=110, end_frame=209, min_knee_angle_deg=95.0)
        posture_summary = {"summary": {}, "events": [ev1, ev2]}

        # Attempt 1 has only 1 valid frame out of 100 (1% coverage, below 50% threshold)
        supps = [
            self._make_attempt_supplement(attempt_id=1, start_frame=0, end_frame=99, valid_frames=1, observed_frames=100, min_knee_angle_deg=90.0),
            self._make_attempt_supplement(attempt_id=2, start_frame=110, end_frame=209, valid_frames=90, observed_frames=100, min_knee_angle_deg=95.0),
        ]

        result = build_feedback_observations(posture_summary, attempt_events=supps)

        # Attempt 1 depth should be unverified
        att1 = result["attempts"][0]
        self.assertFalse(att1["availability"]["movement_depth"]["available"])
        self.assertEqual(att1["evidence"]["min_knee_angle_deg"], 90.00)
        self.assertIn("below operational threshold", att1["availability"]["movement_depth"]["reason"])

        # Attempt 2 depth is available
        att2 = result["attempts"][1]
        self.assertTrue(att2["availability"]["movement_depth"]["available"])

        # Group comparison depth should be unavailable because only 1 reliable counted depth exists
        cmp_group = result["summary"]["comparisons"][0]
        self.assertFalse(cmp_group["movement_depth"]["available"])
        self.assertIn("Only 1 counted cycle on track 1 (right side) met depth reliability criteria", " ".join(cmp_group["limitations"]))

    def test_context_note_neutral_intent_and_upstream_explanation_preserved(self):
        """Context notes use neutral intent language, and upstream explanations are preserved."""
        # Custom upstream explanation in attempts metadata
        custom_expl = "Cycle completed with verified knee extension return."
        supp = self._make_attempt_supplement(attempt_id=1, explanation=custom_expl)
        ev1 = self._make_posture_event(attempt_id=1, status="counted")

        # Uncounted movement threshold
        ev2 = self._make_posture_event(attempt_id=2, status="uncounted", reason="depth_threshold_not_reached", start_frame=35, end_frame=55)

        # Processing failed
        ev3 = self._make_posture_event(attempt_id=3, status="uncounted", reason="processing_failed", start_frame=60, end_frame=75)

        # Video ended mid attempt
        ev4 = self._make_posture_event(attempt_id=4, status="uncounted", reason="video_ended_mid_attempt", start_frame=80, end_frame=95)

        # Unknown reason
        ev5 = self._make_posture_event(attempt_id=5, status="uncounted", reason="custom_unrecognized_code", start_frame=100, end_frame=115)

        posture_summary = {"summary": {}, "events": [ev1, ev2, ev3, ev4, ev5]}
        result = build_feedback_observations(posture_summary, attempt_events=[supp])

        # 1. Preserved upstream explanation
        self.assertEqual(result["attempts"][0]["movement_explanation"]["explanation"], custom_expl)

        # 2. Movement threshold uses 'cannot determine exercise intent'
        note2 = result["attempts"][1]["movement_explanation"]["context_note"]
        self.assertIn("cannot determine exercise intent", note2)
        self.assertIn("may reflect setup, bar adjustment, weight pickup/return", note2)
        self.assertNotIn("frequently are setup", note2)

        # 3. Processing failed does NOT falsely claim video ended
        note3 = result["attempts"][2]["movement_explanation"]["context_note"]
        self.assertIn("Processing halted before this attempt could be confirmed", note3)
        self.assertNotIn("video recording ended", note3)

        # 4. Video ended mid attempt
        note4 = result["attempts"][3]["movement_explanation"]["context_note"]
        self.assertIn("Video recording ended before return to standing was confirmed", note4)
        self.assertIn("cannot determine exercise intent", note4)

        # 5. Unknown reason does NOT become certain tracking failure
        note5 = result["attempts"][4]["movement_explanation"]["context_note"]
        self.assertIn("custom unrecognized code", note5)
        self.assertNotIn("tracking or visibility interruption", note5)

    def test_supplement_matching_and_duplicate_rejection(self):
        """Supplement validation catches mismatches and duplicate IDs."""
        ev = self._make_posture_event(attempt_id=1, track_id=1, side="right", start_frame=10, end_frame=20)
        posture_summary = {"summary": {}, "events": [ev]}

        # Duplicate ID in supplement
        supp_dups = [
            self._make_attempt_supplement(attempt_id=1),
            self._make_attempt_supplement(attempt_id=1),
        ]
        with self.assertRaises(ValueError) as ctx:
            build_feedback_observations(posture_summary, attempt_events=supp_dups)
        self.assertIn("Duplicate attempt_id", str(ctx.exception))

        # Track mismatch
        supp_track_mismatch = [self._make_attempt_supplement(attempt_id=1, track_id=2, start_frame=10, end_frame=20)]
        with self.assertRaises(ValueError) as ctx:
            build_feedback_observations(posture_summary, attempt_events=supp_track_mismatch)
        self.assertIn("track_id mismatch", str(ctx.exception))

        # Side mismatch
        supp_side_mismatch = [self._make_attempt_supplement(attempt_id=1, side="left", start_frame=10, end_frame=20)]
        with self.assertRaises(ValueError) as ctx:
            build_feedback_observations(posture_summary, attempt_events=supp_side_mismatch)
        self.assertIn("side mismatch", str(ctx.exception))

        # Start frame mismatch
        supp_frame_mismatch = [self._make_attempt_supplement(attempt_id=1, start_frame=15, end_frame=20)]
        with self.assertRaises(ValueError) as ctx:
            build_feedback_observations(posture_summary, attempt_events=supp_frame_mismatch)
        self.assertIn("start_frame mismatch", str(ctx.exception))

    def test_integer_parsing_and_bounds_validation(self):
        """Strict integer parsing rejects booleans, fractions, and invalid frame bounds."""
        # Boolean attempt_id rejected
        ev_bool = self._make_posture_event(attempt_id=1)
        ev_bool["attempt_id"] = True
        with self.assertRaises(ValueError) as ctx:
            build_feedback_observations({"events": [ev_bool]})
        self.assertIn("boolean", str(ctx.exception).lower())

        # Fractional string frame rejected
        ev_frac = self._make_posture_event(attempt_id=1)
        ev_frac["start_frame"] = "10.5"
        with self.assertRaises(ValueError):
            build_feedback_observations({"events": [ev_frac]})

        # Negative frame rejected
        ev_neg = self._make_posture_event(attempt_id=1)
        ev_neg["start_frame"] = -1
        with self.assertRaises(ValueError):
            build_feedback_observations({"events": [ev_neg]})

        # Inconsistent total_frames
        ev_bounds = self._make_posture_event(attempt_id=1, start_frame=10, end_frame=20)
        ev_bounds["total_frames"] = 50  # 20 - 10 + 1 = 11 != 50
        with self.assertRaises(ValueError) as ctx:
            build_feedback_observations({"events": [ev_bounds]})
        self.assertIn("inconsistent with bounds", str(ctx.exception))

        # End frame < start frame
        ev_inverted = self._make_posture_event(attempt_id=1, start_frame=20, end_frame=10)
        with self.assertRaises(ValueError):
            build_feedback_observations({"events": [ev_inverted]})

    def test_metrics_range_and_ordering_validation(self):
        """Metrics validate finite coverage in [0, 1] and ordered angles in [0, 180]."""
        # Out-of-range angle (> 180)
        ev_angle = self._make_posture_event(attempt_id=1)
        ev_angle["torso_inclination"]["median_deg"] = 195.0
        with self.assertRaises(ValueError):
            build_feedback_observations({"events": [ev_angle]})

        # Unordered angles (min > median)
        ev_order = self._make_posture_event(attempt_id=1)
        ev_order["torso_inclination"]["min_deg"] = 25.0
        ev_order["torso_inclination"]["median_deg"] = 15.0
        ev_order["torso_inclination"]["max_deg"] = 30.0
        with self.assertRaises(ValueError) as ctx:
            build_feedback_observations({"events": [ev_order]})
        self.assertIn("min <= median <= max", str(ctx.exception))

        # Coverage fraction > 1.0 or non-finite
        ev_cov = self._make_posture_event(attempt_id=1)
        ev_cov["torso_inclination"]["coverage_fraction"] = 1.5
        with self.assertRaises(ValueError):
            build_feedback_observations({"events": [ev_cov]})

        ev_cov_inf = self._make_posture_event(attempt_id=1)
        ev_cov_inf["torso_inclination"]["coverage_fraction"] = float("inf")
        with self.assertRaises(ValueError):
            build_feedback_observations({"events": [ev_cov_inf]})

        # Measured count > total frames
        ev_count = self._make_posture_event(attempt_id=1, start_frame=10, end_frame=19)  # total 10
        ev_count["torso_inclination"]["measured_count"] = 15
        with self.assertRaises(ValueError):
            build_feedback_observations({"events": [ev_count]})

    def test_no_fabricated_fallback_attempt_id(self):
        """Missing attempt_id in posture summary raises ValueError without inventing IDs."""
        ev = self._make_posture_event()
        del ev["attempt_id"]
        with self.assertRaises(ValueError) as ctx:
            build_feedback_observations({"events": [ev]})
        self.assertIn("missing required 'attempt_id'", str(ctx.exception))

    def test_zero_degree_valid_angle(self):
        """A measured 0.00Ã‚° torso inclination (vertical) is valid and treated numerically, not missing."""
        ev1 = self._make_posture_event(attempt_id=1, torso_med=0.00, torso_min=0.00, torso_max=0.00)
        ev2 = self._make_posture_event(attempt_id=2, start_frame=35, end_frame=55, torso_med=8.00, torso_min=4.00, torso_max=10.00)
        posture_summary = {"summary": {}, "events": [ev1, ev2]}

        result = build_feedback_observations(posture_summary)

        att1 = result["attempts"][0]
        self.assertTrue(att1["availability"]["torso"]["available"])
        self.assertEqual(att1["evidence"]["torso_median_deg"], 0.00)

        torso_cmp = result["summary"]["comparisons"][0]["torso_inclination"]
        self.assertTrue(torso_cmp["available"])
        self.assertEqual(torso_cmp["min_median_deg"], 0.00)
        self.assertEqual(torso_cmp["max_median_deg"], 8.00)
        self.assertEqual(torso_cmp["spread_deg"], 8.00)

    def test_scope_limitations_and_non_normative_assertions(self):
        """Assertions verify honest limitations while checking observations never grade form."""
        ev1 = self._make_posture_event(attempt_id=1, status="counted", torso_med=15.0)
        ev2 = self._make_posture_event(attempt_id=2, status="uncounted", reason="depth_threshold_not_reached", torso_med=35.0, torso_max=40.0, start_frame=35, end_frame=55)
        posture_summary = {"summary": {}, "events": [ev1, ev2]}

        result = build_feedback_observations(posture_summary)

        # 1. Assert honest scope limitations are present and explicit
        limits = result["scope_limitations"]
        self.assertTrue(any("dynamic knee valgus" in limit for limit in limits))
        self.assertTrue(any("unsigned 2D angle" in limit for limit in limits))
        self.assertTrue(any("exercise intent is not classified" in limit for limit in limits))
        self.assertTrue(any("Technique quality, clinical safety" in limit for limit in limits))

        # 2. Assert generated observations do NOT judge or grade form
        forbidden_in_obs = ["safe", "unsafe", "correct", "incorrect", "bad form", "good form", "proper", "poor"]
        for att in result["attempts"]:
            all_text = " ".join(att["observations"]) + " " + att["movement_explanation"]["context_note"]
            for word in forbidden_in_obs:
                self.assertNotIn(word, all_text.lower(), f"Observation contained normative term '{word}'")

    def test_input_immutability(self):
        """Input posture_summary and attempt_events are not mutated by observation aggregation."""
        ev = self._make_posture_event(attempt_id=1, torso_med=15.0)
        posture_summary = {"summary": {"total": 1}, "events": [ev]}
        supp = self._make_attempt_supplement(attempt_id=1)
        supps = [supp]

        summary_copy = copy.deepcopy(posture_summary)
        supps_copy = copy.deepcopy(supps)

        build_feedback_observations(posture_summary, attempt_events=supps)

        self.assertEqual(posture_summary, summary_copy)
        self.assertEqual(supps, supps_copy)

    def test_min_attempts_for_comparison_config(self):
        """ObservationConfig enforces min_attempts_for_comparison >= 2."""
        with self.assertRaises(ValueError) as ctx:
            ObservationConfig(min_attempts_for_comparison=1)
        self.assertIn("must be an integer >= 2", str(ctx.exception))

        with self.assertRaises(ValueError):
            ObservationConfig(min_attempts_for_comparison=True)  # booleans rejected

    def test_real_summarise_attempt_posture_integration(self):
        """Integration test using actual summarise_attempt_posture output."""
        frame_rows = [
            {
                "frame_index": i,
                "video_time_s": f"{i / 30.0:.4f}",
                "track_id": "1",
                "posture_side": "right",
                "torso_inclination_deg": f"{14.0 + (i % 3):.2f}",
                "hip_angle_deg": f"{120.0 + (i % 4):.2f}",
                "torso_reason": "ok",
                "hip_reason": "ok",
                "valid": "True",
                "reason": "ok",
            }
            for i in range(10, 30)
        ]
        attempt_events = [
            {
                "attempt_id": 1,
                "start_frame": 10,
                "end_frame": 29,
                "track_id": 1,
                "side": "right",
                "status": "counted",
                "reason": "completed_cycle",
                "min_knee_angle_deg": 91.5,
                "depth_confirmed": True,
                "valid_frames": 20,
                "observed_frames": 20,
                "explanation": "A complete cycle met the configured counting thresholds.",
            }
        ]

        # 1. Run genuine summarise_attempt_posture
        real_summary = summarise_attempt_posture(frame_rows, attempt_events)

        # 2. Pass directly to build_feedback_observations
        feedback = build_feedback_observations(real_summary, attempt_events=attempt_events)

        self.assertEqual(feedback["summary"]["total_attempt_events"], 1)
        self.assertEqual(feedback["summary"]["counted_events"], 1)
        att = feedback["attempts"][0]
        self.assertTrue(att["availability"]["torso"]["available"])
        self.assertTrue(att["availability"]["movement_depth"]["available"])
        self.assertEqual(att["evidence"]["min_knee_angle_deg"], 91.50)

    def test_convenience_raw_function(self):
        """build_feedback_observations_from_raw operates cleanly on CSV frame rows."""
        frame_rows = [
            {
                "frame_index": i,
                "video_time_s": f"{i / 30.0:.4f}",
                "track_id": "1",
                "posture_side": "right",
                "torso_inclination_deg": "15.00",
                "hip_angle_deg": "130.00",
                "torso_reason": "ok",
                "hip_reason": "ok",
                "valid": "True",
                "reason": "ok",
            }
            for i in range(10, 20)
        ]
        attempt_events = [
            {
                "attempt_id": 1,
                "start_frame": 10,
                "end_frame": 19,
                "track_id": 1,
                "side": "right",
                "status": "counted",
                "reason": "completed_cycle",
                "min_knee_angle_deg": 90.5,
                "valid_frames": 10,
                "observed_frames": 10,
            }
        ]

        result = build_feedback_observations_from_raw(frame_rows, attempt_events)
        self.assertEqual(result["summary"]["total_attempt_events"], 1)
        self.assertEqual(result["summary"]["counted_events"], 1)
        self.assertTrue(result["attempts"][0]["availability"]["torso"]["available"])
        self.assertEqual(result["attempts"][0]["evidence"]["torso_median_deg"], 15.00)

    def test_empty_events_list(self):
        """Zero events handled gracefully without division errors."""
        posture_summary = {"summary": {}, "events": []}
        result = build_feedback_observations(posture_summary)
        self.assertEqual(result["summary"]["total_attempt_events"], 0)
        self.assertEqual(len(result["attempts"]), 0)
        self.assertIn("No discrete movement attempts", " ".join(result["summary"]["session_observations"]))


if __name__ == "__main__":
    unittest.main()


class TestFeedbackReviewRegressions(unittest.TestCase):
    def setUp(self):
        self.fixtures = TestFeedbackObservations()

    def test_coverage_cannot_inflate_single_frame(self):
        event = self.fixtures._make_posture_event(torso_count=1, torso_cov=1.0)
        with self.assertRaisesRegex(ValueError, "coverage"):
            build_feedback_observations({"events": [event]})

    def test_sparse_observed_frames_do_not_establish_depth(self):
        event = self.fixtures._make_posture_event(start_frame=0, end_frame=99)
        supplement = self.fixtures._make_attempt_supplement(start_frame=0, end_frame=99,
            valid_frames=1, observed_frames=1)
        result = build_feedback_observations({"events": [event]}, [supplement])
        self.assertFalse(result["attempts"][0]["availability"]["movement_depth"]["available"])

    def test_conflicting_supplement_status_rejected(self):
        event = self.fixtures._make_posture_event()
        supplement = self.fixtures._make_attempt_supplement(status="uncounted")
        with self.assertRaisesRegex(ValueError, "status"):
            build_feedback_observations({"events": [event]}, [supplement])

    def test_different_tracks_and_sides_cannot_form_comparison(self):
        events = [self.fixtures._make_posture_event(attempt_id=1, track_id=1, side="left"),
                  self.fixtures._make_posture_event(attempt_id=2, track_id=2, side="left"),
                  self.fixtures._make_posture_event(attempt_id=3, track_id=1, side="right")]
        result = build_feedback_observations({"events": events})
        self.assertEqual(len(result["summary"]["comparisons"]), 3)
        self.assertTrue(all(not group["torso_inclination"]["available"]
                            for group in result["summary"]["comparisons"]))
