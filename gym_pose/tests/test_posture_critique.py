# -*- coding: utf-8 -*-
"""Independent regression test suite for posture critique and coaching feedback.

Authoritatively coordinates with POSTURE_CRITIQUE_CONTRACT_59.md, REVIEW_50_ACCEPTED.md,
and FEEDBACK_CONTRACT_38.md to verify actual generated output structures, observations,
suggestions, and evidence links without coupling to private internal helpers.

Test Suites:
    A. Changing vs Stable Measurements: Depth progression, drift suppression, pacing drift, torso variation.
    B. Comparability & Side/Track Switching: Prerequisite enforcement, side/track switch suppression.
    C. Missing, Non-finite, and Low-Coverage Data: Thresholds (<50.0%), non-chronological timestamps, missing phases.
    D. Uncounted Movements & Epistemological Guardrails: Setup/return vs shallow depth, zero clinical diagnoses.
    E. Evidence-Linked Suggestions: Exact timestamp citations, pacing triggers, depth consistency (>5.0 deg).
"""

import copy
import json
import math
import sys
import unittest
from typing import Any, Dict

from gym_analyser.coaching_feedback import generate_coaching_feedback


class TestChangingVsStableMeasurements(unittest.TestCase):
    """Test suite A: Changing vs Stable Measurements across comparable reps."""

    def setUp(self) -> None:
        self.base_summary: Dict[str, Any] = {
            "status": "completed",
            "completed_reps": 2,
            "processed_frames": 100,
            "valid_observations": 90,
            "valid_observation_fraction": 0.90,
            "configuration": {"down_angle": 110.0, "up_angle": 150.0},
            "posture_observations": {"torso_valid_frames": 85, "processed_frames": 100},
        }
        self.base_attempts: Dict[str, Any] = {
            "counted_attempts": 2,
            "uncounted_attempts": 0,
            "events": [
                {
                    "attempt_id": 1,
                    "status": "counted",
                    "track_id": 1,
                    "side": "right",
                    "start_time_s": 0.50,
                    "end_time_s": 1.50,
                    "min_knee_angle_deg": 88.0,
                    "valid_frames": 30,
                    "observed_frames": 30,
                },
                {
                    "attempt_id": 2,
                    "status": "counted",
                    "track_id": 1,
                    "side": "right",
                    "start_time_s": 2.00,
                    "end_time_s": 3.00,
                    "min_knee_angle_deg": 88.0,
                    "valid_frames": 30,
                    "observed_frames": 30,
                },
            ],
        }
        self.base_posture: Dict[str, Any] = {
            "events": [
                {
                    "attempt_id": 1,
                    "status": "counted",
                    "track_id": 1,
                    "side": "right",
                    "torso_inclination": {"median_deg": 20.0, "coverage_fraction": 0.90},
                },
                {
                    "attempt_id": 2,
                    "status": "counted",
                    "track_id": 1,
                    "side": "right",
                    "torso_inclination": {"median_deg": 21.0, "coverage_fraction": 0.90},
                },
            ]
        }
        self.base_phases: Dict[str, Any] = {
            "attempts": [
                {
                    "attempt_id": 1,
                    "status": "counted",
                    "phase_analysis_available": True,
                    "min_knee_point": {"knee_angle_deg": 88.0, "video_time_s": 1.00},
                    "descent_phase": {"sample_count": 10, "phase_start_time_s": 0.50, "phase_end_time_s": 1.00},
                    "return_phase": {"sample_count": 10, "phase_start_time_s": 1.00, "phase_end_time_s": 1.50},
                },
                {
                    "attempt_id": 2,
                    "status": "counted",
                    "phase_analysis_available": True,
                    "min_knee_point": {"knee_angle_deg": 88.0, "video_time_s": 2.50},
                    "descent_phase": {"sample_count": 10, "phase_start_time_s": 2.00, "phase_end_time_s": 2.50},
                    "return_phase": {"sample_count": 10, "phase_start_time_s": 2.50, "phase_end_time_s": 3.00},
                },
            ]
        }
        self.base_obs: Dict[str, Any] = {
            "summary": {
                "comparisons": [
                    {
                        "track_id": 1,
                        "side": "right",
                        "torso_inclination": {"group_median_deg": 20.5},
                    }
                ]
            }
        }

    def test_within_session_depth_variation_reports_delta_and_reference_rep(self) -> None:
        """Depth variation across comparable reps reports delta, reference rep ID, and non-normative cue."""
        phases = copy.deepcopy(self.base_phases)
        phases["attempts"][1]["min_knee_point"]["knee_angle_deg"] = 96.0
        phases["attempts"][1]["relative_depth"] = {
            "is_relatively_shallower": True,
            "difference_deg": 8.0,
            "compared_to_attempt_id": 1,
            "reference_min_knee_deg": 88.0,
        }
        res = generate_coaching_feedback(self.base_summary, self.base_attempts, self.base_posture, self.base_obs, phases)
        rep2 = res["per_repetition_cues"][1]
        depth_obs = next(o for o in rep2["observations"] if o["rule_id"] == "OBS-DEPTH-01")
        self.assertEqual(depth_obs["confidence_level"], "observed")
        self.assertIn("+8.00°", depth_obs["message"])
        self.assertIn("cycle 1", depth_obs["message"])
        self.assertIn("temporal review cue, not form critique", depth_obs["message"])

    def test_stable_consistent_depth_suppresses_spurious_depth_drift_warnings(self) -> None:
        """Stable consistent depth produces standard turnaround observation without spurious drift/shallowing warnings."""
        res = generate_coaching_feedback(self.base_summary, self.base_attempts, self.base_posture, self.base_obs, self.base_phases)
        rep2 = res["per_repetition_cues"][1]
        depth_obs = next(o for o in rep2["observations"] if o["rule_id"] == "OBS-DEPTH-01")
        self.assertEqual(depth_obs["confidence_level"], "observed")
        self.assertNotIn("shallower", depth_obs["message"].lower())
        self.assertNotIn("drift", depth_obs["message"].lower())
        self.assertNotIn("deeper", depth_obs["message"].lower())

    def test_rapid_pacing_triggers_sug_pacing_01(self) -> None:
        """Rapid descent duration (< 0.6s) triggers SUG-PACING-01 movement pacing recommendation."""
        phases = copy.deepcopy(self.base_phases)
        phases["attempts"][0]["descent_phase"]["phase_start_time_s"] = 0.60
        phases["attempts"][0]["descent_phase"]["phase_end_time_s"] = 1.00
        res = generate_coaching_feedback(self.base_summary, self.base_attempts, self.base_posture, self.base_obs, phases)
        rep1 = res["per_repetition_cues"][0]
        sug_ids = [s["rule_id"] for s in rep1["suggestions"]]
        self.assertIn("SUG-PACING-01", sug_ids)

    def test_asymmetric_pacing_drift_triggers_sug_pacing_01(self) -> None:
        """Asymmetric pacing tempo ratio > 2.5 triggers SUG-PACING-01 movement pacing recommendation."""
        phases = copy.deepcopy(self.base_phases)
        phases["attempts"][0]["descent_phase"]["phase_start_time_s"] = 0.00
        phases["attempts"][0]["descent_phase"]["phase_end_time_s"] = 1.80
        phases["attempts"][0]["min_knee_point"]["video_time_s"] = 1.80
        phases["attempts"][0]["return_phase"]["phase_start_time_s"] = 1.80
        phases["attempts"][0]["return_phase"]["phase_end_time_s"] = 2.40
        res = generate_coaching_feedback(self.base_summary, self.base_attempts, self.base_posture, self.base_obs, phases)
        rep1 = res["per_repetition_cues"][0]
        sug_ids = [s["rule_id"] for s in rep1["suggestions"]]
        self.assertIn("SUG-PACING-01", sug_ids)

    def test_controlled_consistent_pacing_suppresses_sug_pacing_01(self) -> None:
        """Controlled pacing (descent >= 0.6s and balanced ratio) does not trigger SUG-PACING-01."""
        phases = copy.deepcopy(self.base_phases)
        phases["attempts"][0]["descent_phase"]["phase_start_time_s"] = 0.00
        phases["attempts"][0]["descent_phase"]["phase_end_time_s"] = 1.20
        phases["attempts"][0]["min_knee_point"]["video_time_s"] = 1.20
        phases["attempts"][0]["return_phase"]["phase_start_time_s"] = 1.20
        phases["attempts"][0]["return_phase"]["phase_end_time_s"] = 2.20
        res = generate_coaching_feedback(self.base_summary, self.base_attempts, self.base_posture, self.base_obs, phases)
        rep1 = res["per_repetition_cues"][0]
        sug_ids = [s["rule_id"] for s in rep1["suggestions"]]
        self.assertNotIn("SUG-PACING-01", sug_ids)

    def test_torso_inclination_variation_across_comparable_reps_relative_to_session_median(self) -> None:
        """Torso inclination reports deviation relative to session group median with non-normative framing."""
        posture = copy.deepcopy(self.base_posture)
        posture["events"][0]["torso_inclination"]["median_deg"] = 20.0
        posture["events"][1]["torso_inclination"]["median_deg"] = 25.0
        obs = {
            "summary": {
                "comparisons": [
                    {
                        "track_id": 1,
                        "side": "right",
                        "torso_inclination": {"group_median_deg": 22.5},
                    }
                ]
            }
        }
        res = generate_coaching_feedback(self.base_summary, self.base_attempts, posture, obs, self.base_phases)
        rep2 = res["per_repetition_cues"][1]
        torso_obs = next(o for o in rep2["observations"] if o["rule_id"] == "OBS-TORSO-VAR-01")
        self.assertEqual(torso_obs["confidence_level"], "observed")
        self.assertIn("25.00°", torso_obs["message"])
        self.assertIn("session group median 22.50°", torso_obs["message"])
        self.assertIn("deviation +2.50°", torso_obs["message"])
        self.assertIn("Descriptive session comparison, not a technique grade or clinical assessment.", torso_obs["message"])


class TestComparabilityAndSwitching(unittest.TestCase):
    """Test suite B: Comparability prerequisites, body side, and track ID switching."""

    def setUp(self) -> None:
        self.base_summary: Dict[str, Any] = {
            "status": "completed",
            "completed_reps": 2,
            "processed_frames": 100,
            "valid_observations": 90,
            "valid_observation_fraction": 0.90,
            "configuration": {"down_angle": 110.0, "up_angle": 150.0},
        }
        self.base_attempts: Dict[str, Any] = {
            "counted_attempts": 2,
            "events": [
                {"attempt_id": 1, "status": "counted", "track_id": 1, "side": "right", "start_time_s": 0.5, "end_time_s": 1.5, "min_knee_angle_deg": 88.0, "valid_frames": 30, "observed_frames": 30},
                {"attempt_id": 2, "status": "counted", "track_id": 1, "side": "right", "start_time_s": 2.0, "end_time_s": 3.0, "min_knee_angle_deg": 90.0, "valid_frames": 30, "observed_frames": 30},
            ],
        }
        self.base_posture: Dict[str, Any] = {
            "events": [
                {"attempt_id": 1, "status": "counted", "track_id": 1, "side": "right", "torso_inclination": {"median_deg": 20.0, "coverage_fraction": 0.90}},
                {"attempt_id": 2, "status": "counted", "track_id": 1, "side": "right", "torso_inclination": {"median_deg": 22.0, "coverage_fraction": 0.90}},
            ]
        }
        self.base_phases: Dict[str, Any] = {
            "attempts": [
                {
                    "attempt_id": 1, "status": "counted", "phase_analysis_available": True,
                    "min_knee_point": {"knee_angle_deg": 88.0, "video_time_s": 1.0},
                    "descent_phase": {"sample_count": 10, "phase_start_time_s": 0.5, "phase_end_time_s": 1.0},
                    "return_phase": {"sample_count": 10, "phase_start_time_s": 1.0, "phase_end_time_s": 1.5},
                },
                {
                    "attempt_id": 2, "status": "counted", "phase_analysis_available": True,
                    "min_knee_point": {"knee_angle_deg": 90.0, "video_time_s": 2.5},
                    "descent_phase": {"sample_count": 10, "phase_start_time_s": 2.0, "phase_end_time_s": 2.5},
                    "return_phase": {"sample_count": 10, "phase_start_time_s": 2.5, "phase_end_time_s": 3.0},
                },
            ]
        }

    def test_reps_on_different_sides_suppresses_cross_attempt_progression(self) -> None:
        """Side switch between Rep 1 (right) and Rep 2 (left) suppresses progression critique."""
        attempts = copy.deepcopy(self.base_attempts)
        attempts["events"][1]["side"] = "left"
        posture = copy.deepcopy(self.base_posture)
        posture["events"][1]["side"] = "left"
        res = generate_coaching_feedback(self.base_summary, attempts, posture, None, self.base_phases)
        for rep in res["per_repetition_cues"]:
            torso_obs = next(o for o in rep["observations"] if o["rule_id"] == "OBS-TORSO-VAR-01")
            self.assertEqual(torso_obs["confidence_level"], "unavailable")
            self.assertIn("Cross-attempt consistency comparisons are suppressed across differing tracks or body sides.", torso_obs["message"])
        notes = res["overall_summary"]["recording_quality"]["notes"]
        self.assertTrue(any("cross-attempt consistency is not evaluated across different tracks or sides" in n for n in notes))

    def test_reps_on_different_track_ids_suppresses_cross_attempt_progression(self) -> None:
        """Track ID switch between Rep 1 (track 1) and Rep 2 (track 2) suppresses progression."""
        attempts = copy.deepcopy(self.base_attempts)
        attempts["events"][1]["track_id"] = 2
        posture = copy.deepcopy(self.base_posture)
        posture["events"][1]["track_id"] = 2
        res = generate_coaching_feedback(self.base_summary, attempts, posture, None, self.base_phases)
        for rep in res["per_repetition_cues"]:
            torso_obs = next(o for o in rep["observations"] if o["rule_id"] == "OBS-TORSO-VAR-01")
            self.assertEqual(torso_obs["confidence_level"], "unavailable")
            self.assertIn("Cross-attempt consistency comparisons are suppressed across differing tracks or body sides.", torso_obs["message"])

    def test_single_repetition_session_reports_progression_unavailable(self) -> None:
        """Single-rep session explicitly reports that progression requires at least 2 counted reps."""
        summary = copy.deepcopy(self.base_summary)
        summary["completed_reps"] = 1
        attempts = {"counted_attempts": 1, "events": [self.base_attempts["events"][0]]}
        posture = {"events": [self.base_posture["events"][0]]}
        phases = {"attempts": [self.base_phases["attempts"][0]]}
        res = generate_coaching_feedback(summary, attempts, posture, None, phases)
        rep1 = res["per_repetition_cues"][0]
        torso_obs = next(o for o in rep1["observations"] if o["rule_id"] == "OBS-TORSO-VAR-01")
        self.assertEqual(torso_obs["confidence_level"], "unavailable")
        self.assertIn("Within-session consistency comparisons require at least 2 counted repetitions on the same track and side.", torso_obs["message"])

    def test_multiple_switching_contexts_marks_camera_view_false(self) -> None:
        """When multiple tracks or sides occur, camera_view_adequate is marked False."""
        attempts = copy.deepcopy(self.base_attempts)
        attempts["events"][1]["side"] = "left"
        res = generate_coaching_feedback(self.base_summary, attempts, self.base_posture, None, self.base_phases)
        self.assertIs(res["overall_summary"]["recording_quality"]["camera_view_adequate"], False)


class TestMissingNonfiniteAndLowCoverageData(unittest.TestCase):
    """Test suite C: Missing, non-finite, out-of-order, and low-coverage data handling."""

    def setUp(self) -> None:
        self.base_summary: Dict[str, Any] = {
            "status": "completed",
            "completed_reps": 1,
            "processed_frames": 100,
            "valid_observations": 90,
            "valid_observation_fraction": 0.90,
            "configuration": {"down_angle": 110.0, "up_angle": 150.0},
        }

    def test_low_coverage_under_50_pct_marks_telemetry_unavailable(self) -> None:
        """Coverage < 50.0% marks depth and torso telemetry explicitly unavailable."""
        summary = copy.deepcopy(self.base_summary)
        summary["valid_observation_fraction"] = 0.40
        attempts = {
            "counted_attempts": 1,
            "events": [
                {
                    "attempt_id": 1, "status": "counted", "track_id": 1, "side": "right",
                    "start_time_s": 0.50, "end_time_s": 1.50, "min_knee_angle_deg": 88.0,
                    "valid_frames": 12, "observed_frames": 30,
                }
            ],
        }
        posture = {
            "events": [
                {
                    "attempt_id": 1, "status": "counted", "track_id": 1, "side": "right",
                    "torso_inclination": {"median_deg": 20.0, "coverage_fraction": 0.40},
                }
            ]
        }
        phases = {
            "attempts": [
                {
                    "attempt_id": 1, "status": "counted", "phase_analysis_available": True,
                    "min_knee_point": {"knee_angle_deg": 88.0, "video_time_s": 1.00},
                    "descent_phase": {"sample_count": 10, "phase_start_time_s": 0.50, "phase_end_time_s": 1.00},
                    "return_phase": {"sample_count": 10, "phase_start_time_s": 1.00, "phase_end_time_s": 1.50},
                }
            ]
        }
        res = generate_coaching_feedback(summary, attempts, posture, None, phases)
        rep1 = res["per_repetition_cues"][0]
        depth_obs = next(o for o in rep1["observations"] if o["rule_id"] == "OBS-DEPTH-01")
        self.assertEqual(depth_obs["confidence_level"], "unavailable")
        self.assertIn("insufficient valid frame coverage (40.0% < 50.0%)", depth_obs["message"])

        torso_obs = next(o for o in rep1["observations"] if o["rule_id"] == "OBS-TORSO-VAR-01")
        self.assertEqual(torso_obs["confidence_level"], "unavailable")
        self.assertIn("frame coverage 40.0% below operational 50.0% threshold", torso_obs["message"])

    def test_low_coverage_under_50_pct_triggers_sug_rec_01(self) -> None:
        """Coverage < 50.0% triggers SUG-REC-01 recording setup guidance."""
        attempts = {
            "counted_attempts": 1,
            "events": [
                {
                    "attempt_id": 1, "status": "counted", "track_id": 1, "side": "right",
                    "start_time_s": 0.50, "end_time_s": 1.50, "min_knee_angle_deg": 88.0,
                    "valid_frames": 10, "observed_frames": 30,
                }
            ],
        }
        posture = {
            "events": [
                {
                    "attempt_id": 1, "status": "counted", "track_id": 1, "side": "right",
                    "torso_inclination": {"median_deg": 20.0, "coverage_fraction": 0.35},
                }
            ]
        }
        res = generate_coaching_feedback(self.base_summary, attempts, posture)
        rep1 = res["per_repetition_cues"][0]
        sug_ids = [s["rule_id"] for s in rep1["suggestions"]]
        self.assertIn("SUG-REC-01", sug_ids)

    def test_non_chronological_timestamps_marks_timing_observation_unavailable(self) -> None:
        """Non-chronological timestamps mark timing telemetry unavailable."""
        attempts = {
            "counted_attempts": 1,
            "events": [
                {
                    "attempt_id": 1, "status": "counted", "track_id": 1, "side": "right",
                    "start_time_s": 2.50, "end_time_s": 1.50, "min_knee_angle_deg": 88.0,
                    "valid_frames": 30, "observed_frames": 30,
                }
            ],
        }
        phases = {
            "attempts": [
                {
                    "attempt_id": 1, "status": "counted", "phase_analysis_available": True,
                    "min_knee_point": {"knee_angle_deg": 88.0, "video_time_s": 2.00},
                    "descent_phase": {"sample_count": 10, "phase_start_time_s": 2.50, "phase_end_time_s": 2.00},
                    "return_phase": {"sample_count": 10, "phase_start_time_s": 2.00, "phase_end_time_s": 1.50},
                }
            ]
        }
        res = generate_coaching_feedback(self.base_summary, attempts, None, None, phases)
        rep1 = res["per_repetition_cues"][0]
        timing_obs = next(o for o in rep1["observations"] if o["rule_id"] == "OBS-PACING-01")
        self.assertEqual(timing_obs["confidence_level"], "unavailable")
        self.assertIn("timing observation unavailable", timing_obs["message"].lower())

    def test_missing_phase_inflection_marks_depth_turnaround_unavailable(self) -> None:
        """Missing phase inflection evidence marks turnaround depth unavailable."""
        attempts = {
            "counted_attempts": 1,
            "events": [
                {
                    "attempt_id": 1, "status": "counted", "track_id": 1, "side": "right",
                    "start_time_s": 0.50, "end_time_s": 1.50, "min_knee_angle_deg": 88.0,
                    "valid_frames": 30, "observed_frames": 30,
                }
            ],
        }
        res = generate_coaching_feedback(self.base_summary, attempts, None, None, None)
        rep1 = res["per_repetition_cues"][0]
        depth_obs = next(o for o in rep1["observations"] if o["rule_id"] == "OBS-DEPTH-01")
        self.assertEqual(depth_obs["confidence_level"], "unavailable")
        self.assertIn("verified phase inflection evidence is missing from upstream telemetry", depth_obs["message"])

    def test_missing_phase_inflection_marks_timing_turnaround_unavailable(self) -> None:
        """Missing phase inflection evidence marks phase timing turnaround unavailable."""
        attempts = {
            "counted_attempts": 1,
            "events": [
                {
                    "attempt_id": 1, "status": "counted", "track_id": 1, "side": "right",
                    "start_time_s": 0.50, "end_time_s": 1.50, "min_knee_angle_deg": 88.0,
                    "valid_frames": 30, "observed_frames": 30,
                }
            ],
        }
        res = generate_coaching_feedback(self.base_summary, attempts, None, None, None)
        rep1 = res["per_repetition_cues"][0]
        timing_obs = next(o for o in rep1["observations"] if o["rule_id"] == "OBS-PACING-01")
        self.assertEqual(timing_obs["confidence_level"], "unavailable")
        self.assertIn("turnaround inflection point was not reliably resolved", timing_obs["message"])

    def test_non_finite_angle_measurements_handled_defensively(self) -> None:
        """Non-finite angle values (NaN, inf, None) are validated and coerced to None."""
        attempts = {
            "counted_attempts": 1,
            "events": [
                {
                    "attempt_id": 1, "status": "counted", "track_id": 1, "side": "right",
                    "start_time_s": 0.50, "end_time_s": 1.50, "min_knee_angle_deg": float("nan"),
                    "valid_frames": 30, "observed_frames": 30,
                }
            ],
        }
        res = generate_coaching_feedback(self.base_summary, attempts)
        rep1 = res["per_repetition_cues"][0]
        self.assertIsNone(rep1["measurements"]["min_knee_angle_deg"])


class TestUncountedMovementsAndGuardrails(unittest.TestCase):
    """Test suite D: Uncounted movement hypotheses and strict epistemological guardrails."""

    def test_uncounted_setup_and_return_classification(self) -> None:
        """Uncounted movements before first rep or after last rep are classified as setup/return."""
        summary = {"status": "completed", "completed_reps": 1, "processed_frames": 100}
        attempts = {
            "counted_attempts": 1,
            "uncounted_attempts": 2,
            "events": [
                {"attempt_id": 1, "status": "uncounted", "reason": "depth_threshold_not_reached", "track_id": 1, "side": "right"},
                {"attempt_id": 2, "status": "counted", "reason": "completed_cycle", "track_id": 1, "side": "right"},
                {"attempt_id": 3, "status": "uncounted", "reason": "depth_threshold_not_reached", "track_id": 1, "side": "right"},
            ],
        }
        res = generate_coaching_feedback(summary, attempts)
        uncounted = res["uncounted_movement_explanations"]
        self.assertEqual(len(uncounted), 2)
        self.assertEqual(uncounted[0]["context_hypothesis"], "possible_setup_or_weight_return")
        self.assertEqual(uncounted[1]["context_hypothesis"], "possible_setup_or_weight_return")

    def test_uncounted_fragment_between_counted_reps(self) -> None:
        """An uncounted fragment between counted repetitions remains ambiguous."""
        summary = {"status": "completed", "completed_reps": 2, "processed_frames": 150}
        attempts = {
            "counted_attempts": 2,
            "uncounted_attempts": 1,
            "events": [
                {"attempt_id": 1, "status": "counted", "reason": "completed_cycle", "track_id": 1, "side": "right"},
                {"attempt_id": 2, "status": "uncounted", "reason": "depth_threshold_not_reached", "track_id": 1, "side": "right"},
                {"attempt_id": 3, "status": "counted", "reason": "completed_cycle", "track_id": 1, "side": "right"},
            ],
        }
        res = generate_coaching_feedback(summary, attempts)
        uncounted = res["uncounted_movement_explanations"]
        self.assertEqual(len(uncounted), 1)
        self.assertEqual(uncounted[0]["context_hypothesis"], "ambiguous")
        self.assertIn("did not reach the configured depth threshold", uncounted[0]["explanation"])

    def test_uncounted_explanations_have_no_form_critique(self) -> None:
        """Uncounted attempt explanations provide factual cause without derogatory critique."""
        summary = {"status": "completed", "completed_reps": 1, "processed_frames": 100}
        attempts = {
            "counted_attempts": 1,
            "uncounted_attempts": 1,
            "events": [
                {"attempt_id": 1, "status": "uncounted", "reason": "depth_threshold_not_reached", "track_id": 1, "side": "right"},
                {"attempt_id": 2, "status": "counted", "reason": "completed_cycle", "track_id": 1, "side": "right"},
            ],
        }
        res = generate_coaching_feedback(summary, attempts)
        uncounted = res["uncounted_movement_explanations"]
        for u in uncounted:
            self.assertNotIn("bad", u["explanation"].lower())
            self.assertNotIn("cheat", u["explanation"].lower())
            self.assertNotIn("poor", u["explanation"].lower())
            self.assertNotIn("fail", u["explanation"].lower())

    def test_strict_epistemological_guardrails_no_clinical_diagnoses(self) -> None:
        """Output JSON contains zero clinical diagnostic terms (spinal rounding, valgus/varus)."""
        summary = {"status": "completed", "completed_reps": 1, "processed_frames": 100}
        attempts = {"counted_attempts": 1, "events": [{"attempt_id": 1, "status": "counted", "track_id": 1, "side": "right"}]}
        res = generate_coaching_feedback(summary, attempts)
        dumped = json.dumps(res).lower()
        clinical_terms = ["spinal rounding", "spinal flexion", "lumbar rounding", "lumbar flexion", "knee valgus", "knee varus"]
        for term in clinical_terms:
            self.assertNotIn(term, dumped)

    def test_strict_epistemological_guardrails_no_technique_letter_grades(self) -> None:
        """Output JSON contains zero subjective technique letter grades."""
        summary = {"status": "completed", "completed_reps": 1, "processed_frames": 100}
        attempts = {"counted_attempts": 1, "events": [{"attempt_id": 1, "status": "counted", "track_id": 1, "side": "right"}]}
        res = generate_coaching_feedback(summary, attempts)
        dumped = json.dumps(res).lower()
        grading_terms = ["grade a", "grade b", "grade c", "grade f", "technique grade:"]
        for term in grading_terms:
            self.assertNotIn(term, dumped)

    def test_strict_epistemological_guardrails_no_universal_safe_angles(self) -> None:
        """Output JSON contains zero assertions of universal absolute 'safe' angle thresholds."""
        summary = {"status": "completed", "completed_reps": 1, "processed_frames": 100}
        attempts = {"counted_attempts": 1, "events": [{"attempt_id": 1, "status": "counted", "track_id": 1, "side": "right"}]}
        res = generate_coaching_feedback(summary, attempts)
        dumped = json.dumps(res).lower()
        prescriptive_terms = ["safe angle", "unsafe angle", "injury risk", "injury prevention", "universally correct"]
        for term in prescriptive_terms:
            self.assertNotIn(term, dumped)

    def test_mandatory_clinical_and_validation_disclaimers(self) -> None:
        """Mandatory clinical disclaimer and unvalidated human status strings are present."""
        summary = {"status": "completed", "completed_reps": 1, "processed_frames": 100}
        res = generate_coaching_feedback(summary)
        self.assertEqual(
            res["disclaimers"]["clinical_disclaimer"],
            "These 2D kinematic measurements are descriptive planar observations, not clinical diagnostic assessments or technique grades.",
        )
        self.assertEqual(
            res["disclaimers"]["human_validation_status"],
            "UNVALIDATED by human expert review on diverse unseen subjects.",
        )

    def test_monocular_camera_adequacy_is_never_true_in_any_context(self) -> None:
        """Monocular 2D tracking never asserts camera_view_adequate as True."""
        summary = {
            "status": "completed",
            "completed_reps": 2,
            "processed_frames": 100,
            "valid_observations": 95,
            "valid_observation_fraction": 0.95,
            "posture_observations": {"torso_valid_frames": 95, "processed_frames": 100},
        }
        attempts = {
            "counted_attempts": 2,
            "events": [
                {"attempt_id": 1, "status": "counted", "track_id": 1, "side": "right"},
                {"attempt_id": 2, "status": "counted", "track_id": 1, "side": "right"},
            ],
        }
        res = generate_coaching_feedback(summary, attempts)
        camera_view = res["overall_summary"]["recording_quality"]["camera_view_adequate"]
        self.assertIsNot(camera_view, True)
        self.assertIsNone(camera_view)


class TestEvidenceLinkedSuggestions(unittest.TestCase):
    """Test suite E: Evidence-linked suggestion rules and timestamp citations."""

    def setUp(self) -> None:
        self.base_summary: Dict[str, Any] = {
            "status": "completed",
            "completed_reps": 2,
            "processed_frames": 100,
            "valid_observations": 90,
            "valid_observation_fraction": 0.90,
            "configuration": {"down_angle": 110.0, "up_angle": 150.0},
        }
        self.base_attempts: Dict[str, Any] = {
            "counted_attempts": 2,
            "events": [
                {"attempt_id": 1, "status": "counted", "track_id": 1, "side": "right", "start_time_s": 0.50, "end_time_s": 1.50, "min_knee_angle_deg": 88.0, "valid_frames": 30, "observed_frames": 30},
                {"attempt_id": 2, "status": "counted", "track_id": 1, "side": "right", "start_time_s": 2.00, "end_time_s": 3.00, "min_knee_angle_deg": 96.0, "valid_frames": 30, "observed_frames": 30},
            ],
        }
        self.base_posture: Dict[str, Any] = {
            "events": [
                {"attempt_id": 1, "status": "counted", "track_id": 1, "side": "right", "torso_inclination": {"median_deg": 20.0, "coverage_fraction": 0.90}},
                {"attempt_id": 2, "status": "counted", "track_id": 1, "side": "right", "torso_inclination": {"median_deg": 22.0, "coverage_fraction": 0.90}},
            ]
        }
        self.base_phases: Dict[str, Any] = {
            "attempts": [
                {
                    "attempt_id": 1, "status": "counted", "phase_analysis_available": True,
                    "min_knee_point": {"knee_angle_deg": 88.0, "video_time_s": 1.00},
                    "descent_phase": {"sample_count": 10, "phase_start_time_s": 0.50, "phase_end_time_s": 1.00},
                    "return_phase": {"sample_count": 10, "phase_start_time_s": 1.00, "phase_end_time_s": 1.50},
                },
                {
                    "attempt_id": 2, "status": "counted", "phase_analysis_available": True,
                    "min_knee_point": {"knee_angle_deg": 96.0, "video_time_s": 2.50},
                    "descent_phase": {"sample_count": 10, "phase_start_time_s": 2.00, "phase_end_time_s": 2.50},
                    "return_phase": {"sample_count": 10, "phase_start_time_s": 2.50, "phase_end_time_s": 3.00},
                    "relative_depth": {
                        "is_relatively_shallower": True,
                        "difference_deg": 8.0,
                        "compared_to_attempt_id": 1,
                        "reference_min_knee_deg": 88.0,
                    },
                },
            ]
        }

    def test_sug_review_01_cites_specific_timestamps_when_turnaround_verified(self) -> None:
        """SUG-REVIEW-01 cites specific timestamp intervals [start_time_s, end_time_s] and inflection time."""
        res = generate_coaching_feedback(self.base_summary, self.base_attempts, self.base_posture, None, self.base_phases)
        rep1 = res["per_repetition_cues"][0]
        sug_rev = next(s for s in rep1["suggestions"] if s["rule_id"] == "SUG-REVIEW-01")
        # Must cite timestamp interval [0.50s, 1.50s] and turnaround inflection time (1.00s)
        self.assertIn("from 0.50s to 1.50s", sug_rev["message"])
        self.assertIn("turnaround at 1.00s", sug_rev["message"])
        self.assertNotIn("generic", sug_rev["message"].lower())

    def test_sug_review_01_interval_reference_when_turnaround_unresolved(self) -> None:
        """SUG-REVIEW-01 cites repetition timestamp interval when turnaround inflection is unresolved."""
        res = generate_coaching_feedback(self.base_summary, self.base_attempts, self.base_posture, None, None)
        rep1 = res["per_repetition_cues"][0]
        sug_rev = next(s for s in rep1["suggestions"] if s["rule_id"] == "SUG-REVIEW-01")
        # Has start_time_s and end_time_s from base_attempts
        self.assertIn("from 0.50s to 1.50s", sug_rev["message"])
        self.assertNotIn("turnaround at", sug_rev["message"])

    def test_sug_review_01_fallback_when_timestamps_missing(self) -> None:
        """SUG-REVIEW-01 falls back to 'across repetition interval' when timestamps are None."""
        attempts = {
            "counted_attempts": 1,
            "events": [
                {"attempt_id": 1, "status": "counted", "track_id": 1, "side": "right"}
            ]
        }
        res = generate_coaching_feedback(self.base_summary, attempts, None, None, None)
        rep1 = res["per_repetition_cues"][0]
        sug_rev = next(s for s in rep1["suggestions"] if s["rule_id"] == "SUG-REVIEW-01")
        self.assertIn("across repetition interval", sug_rev["message"])

    def test_sug_depth_consistency_01_not_triggered_when_depth_variation_under_5_deg(self) -> None:
        """SUG-DEPTH-CONSISTENCY-01 is not triggered when depth variation between comparable reps <= 5.0 deg."""
        phases = copy.deepcopy(self.base_phases)
        phases["attempts"][1]["min_knee_point"]["knee_angle_deg"] = 91.5
        phases["attempts"][1]["relative_depth"] = {
            "is_relatively_shallower": True,
            "difference_deg": 3.5,
            "compared_to_attempt_id": 1,
            "reference_min_knee_deg": 88.0,
        }
        res = generate_coaching_feedback(self.base_summary, self.base_attempts, self.base_posture, None, phases)
        rep2 = res["per_repetition_cues"][1]
        sug_ids = [s["rule_id"] for s in rep2["suggestions"]]
        self.assertNotIn("SUG-DEPTH-CONSISTENCY-01", sug_ids)

    def test_sug_depth_consistency_01_triggered_when_depth_varies_over_5_deg(self) -> None:
        """SUG-DEPTH-CONSISTENCY-01 is triggered when comparable depth varies > 5.0 deg."""
        res = generate_coaching_feedback(self.base_summary, self.base_attempts, self.base_posture, None, self.base_phases)
        rep2 = res["per_repetition_cues"][1]
        sug_ids = [s["rule_id"] for s in rep2["suggestions"]]
        self.assertIn(
            "SUG-DEPTH-CONSISTENCY-01",
            sug_ids,
            msg=(
                "Contract Defect in coaching_feedback.py: SUG-DEPTH-CONSISTENCY-01 was not triggered "
                "for comparable Rep 2 where depth variation exceeded 5.0 deg (difference: 8.0 deg relative to Rep 1)."
            ),
        )


if __name__ == "__main__":
    unittest.main()
