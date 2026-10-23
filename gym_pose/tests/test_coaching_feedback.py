# -*- coding: utf-8 -*-
"""Unit tests for the coaching feedback generation module.

Verifies strict schema conformity with FEEDBACK_CONTRACT_38.md, defensive
handling of sparse/interrupted telemetry, zero external ML dependencies,
and non-normative reporting guardrails.
"""

import copy
import sys
import unittest
from typing import Any, Dict

from gym_analyser.coaching_feedback import generate_coaching_feedback


class TestCoachingFeedback(unittest.TestCase):
    """Test suite for generate_coaching_feedback."""

    def _build_synthetic_fixture(self) -> Dict[str, Any]:
        """Build a coherent multi-rep synthetic session fixture."""
        summary = {
            "status": "completed",
            "completed_reps": 2,
            "processed_frames": 100,
            "valid_observations": 90,
            "valid_observation_fraction": 0.90,
            "configuration": {
                "down_angle": 110.0,
                "up_angle": 150.0,
                "min_joint_confidence": 0.3,
            },
            "posture_observations": {
                "torso_valid_frames": 85,
                "processed_frames": 100,
            },
        }

        attempts = {
            "counted_attempts": 2,
            "uncounted_attempts": 1,
            "pre_arming": {
                "unarmed_frames": 5,
                "opening_lowered_frames": 0,
                "explanation": "No attempt is assigned to an opening crouch before standing is confirmed.",
            },
            "events": [
                {
                    "attempt_id": 1,
                    "status": "counted",
                    "reason": "completed_cycle",
                    "explanation": "A complete cycle met the configured counting thresholds.",
                    "track_id": 1,
                    "side": "right",
                    "start_frame": 10,
                    "end_frame": 40,
                    "start_time_s": 0.33,
                    "end_time_s": 1.33,
                    "min_knee_angle_deg": 88.5,
                    "valid_frames": 30,
                    "observed_frames": 31,
                },
                {
                    "attempt_id": 2,
                    "status": "counted",
                    "reason": "completed_cycle",
                    "explanation": "A complete cycle met the configured counting thresholds.",
                    "track_id": 1,
                    "side": "right",
                    "start_frame": 45,
                    "end_frame": 75,
                    "start_time_s": 1.50,
                    "end_time_s": 2.50,
                    "min_knee_angle_deg": 92.0,
                    "valid_frames": 30,
                    "observed_frames": 31,
                },
                {
                    "attempt_id": 3,
                    "status": "uncounted",
                    "reason": "depth_threshold_not_reached",
                    "explanation": "The observed knee angle did not reach the configured depth threshold.",
                    "track_id": 1,
                    "side": "right",
                    "start_frame": 80,
                    "end_frame": 95,
                    "start_time_s": 2.67,
                    "end_time_s": 3.17,
                    "min_knee_angle_deg": 138.0,
                    "valid_frames": 15,
                    "observed_frames": 16,
                },
            ],
        }

        posture_summary = {
            "summary": {"total_events": 3, "counted_events": 2, "uncounted_events": 1},
            "events": [
                {
                    "attempt_id": 1,
                    "status": "counted",
                    "reason": "completed_cycle",
                    "track_id": 1,
                    "side": "right",
                    "start_frame": 10,
                    "end_frame": 40,
                    "total_frames": 31,
                    "min_knee_angle_deg": 88.5,
                    "torso_inclination": {
                        "measured_count": 28,
                        "total_frames": 31,
                        "coverage_fraction": 0.9032,
                        "median_deg": 22.0,
                        "min_deg": 18.0,
                        "max_deg": 26.0,
                    },
                    "hip_angle": {
                        "measured_count": 28,
                        "total_frames": 31,
                        "coverage_fraction": 0.9032,
                        "median_deg": 115.0,
                        "min_deg": 95.0,
                        "max_deg": 140.0,
                    },
                },
                {
                    "attempt_id": 2,
                    "status": "counted",
                    "reason": "completed_cycle",
                    "track_id": 1,
                    "side": "right",
                    "start_frame": 45,
                    "end_frame": 75,
                    "total_frames": 31,
                    "min_knee_angle_deg": 92.0,
                    "torso_inclination": {
                        "measured_count": 28,
                        "total_frames": 31,
                        "coverage_fraction": 0.9032,
                        "median_deg": 24.0,
                        "min_deg": 19.0,
                        "max_deg": 28.0,
                    },
                    "hip_angle": {
                        "measured_count": 28,
                        "total_frames": 31,
                        "coverage_fraction": 0.9032,
                        "median_deg": 118.0,
                        "min_deg": 98.0,
                        "max_deg": 142.0,
                    },
                },
                {
                    "attempt_id": 3,
                    "status": "uncounted",
                    "reason": "depth_threshold_not_reached",
                    "track_id": 1,
                    "side": "right",
                    "start_frame": 80,
                    "end_frame": 95,
                    "total_frames": 16,
                    "min_knee_angle_deg": 138.0,
                    "torso_inclination": {
                        "measured_count": 14,
                        "total_frames": 16,
                        "coverage_fraction": 0.875,
                        "median_deg": 12.0,
                        "min_deg": 8.0,
                        "max_deg": 15.0,
                    },
                    "hip_angle": {
                        "measured_count": 14,
                        "total_frames": 16,
                        "coverage_fraction": 0.875,
                        "median_deg": 150.0,
                        "min_deg": 140.0,
                        "max_deg": 160.0,
                    },
                },
            ],
        }

        feedback_phases = {
            "summary": {"total_attempt_events": 3, "counted_events": 2, "uncounted_events": 1},
            "attempts": [
                {
                    "attempt_id": 1,
                    "status": "counted",
                    "track_id": 1,
                    "side": "right",
                    "phase_analysis_available": True,
                    "min_knee_point": {
                        "frame_index": 25,
                        "video_time_s": 0.83,
                        "knee_angle_deg": 88.5,
                    },
                    "descent_phase": {
                        "sample_count": 15,
                        "coverage_fraction": 0.94,
                        "phase_start_time_s": 0.33,
                        "phase_end_time_s": 0.83,
                    },
                    "return_phase": {
                        "sample_count": 15,
                        "coverage_fraction": 0.94,
                        "phase_start_time_s": 0.83,
                        "phase_end_time_s": 1.33,
                    },
                },
                {
                    "attempt_id": 2,
                    "status": "counted",
                    "track_id": 1,
                    "side": "right",
                    "phase_analysis_available": True,
                    "min_knee_point": {
                        "frame_index": 60,
                        "video_time_s": 2.00,
                        "knee_angle_deg": 92.0,
                    },
                    "descent_phase": {
                        "sample_count": 15,
                        "coverage_fraction": 0.94,
                        "phase_start_time_s": 1.50,
                        "phase_end_time_s": 2.00,
                    },
                    "return_phase": {
                        "sample_count": 15,
                        "coverage_fraction": 0.94,
                        "phase_start_time_s": 2.00,
                        "phase_end_time_s": 2.50,
                    },
                    "relative_depth": {
                        "is_relatively_shallower": True,
                        "difference_deg": 3.5,
                        "compared_to_attempt_id": 1,
                        "reference_min_knee_deg": 88.5,
                    },
                },
            ],
        }

        feedback_observations = {
            "summary": {
                "total_attempts": 3,
                "counted_attempts": 2,
                "uncounted_attempts": 1,
                "comparisons": [
                    {
                        "track_id": 1,
                        "side": "right",
                        "torso_inclination": {
                            "available": True,
                            "group_median_deg": 23.0,
                            "min_median_deg": 22.0,
                            "max_median_deg": 24.0,
                            "spread_deg": 2.0,
                        },
                    }
                ],
            },
            "attempts": [
                {
                    "attempt_id": 1,
                    "evidence": {
                        "torso_median_deg": 22.0,
                        "hip_median_deg": 115.0,
                        "min_knee_angle_deg": 88.5,
                    },
                    "availability": {
                        "torso": {"coverage_fraction": 0.9032},
                    },
                },
                {
                    "attempt_id": 2,
                    "evidence": {
                        "torso_median_deg": 24.0,
                        "hip_median_deg": 118.0,
                        "min_knee_angle_deg": 92.0,
                    },
                    "availability": {
                        "torso": {"coverage_fraction": 0.9032},
                    },
                },
            ],
        }

        return {
            "summary": summary,
            "attempts": attempts,
            "posture_summary": posture_summary,
            "feedback_phases": feedback_phases,
            "feedback_observations": feedback_observations,
        }

    def test_schema_keys_and_contract_version(self):
        """Verify top-level keys and contract version exactly conform to FEEDBACK_CONTRACT_38.md."""
        data = self._build_synthetic_fixture()
        result = generate_coaching_feedback(
            summary=data["summary"],
            attempts=data["attempts"],
            posture_summary=data["posture_summary"],
            feedback_observations=data["feedback_observations"],
            feedback_phases=data["feedback_phases"],
        )

        expected_top_keys = {
            "contract_version",
            "status",
            "overall_summary",
            "per_repetition_cues",
            "uncounted_movement_explanations",
            "disclaimers",
        }
        self.assertEqual(set(result.keys()), expected_top_keys)
        self.assertEqual(result["contract_version"], "1.0")
        self.assertIn(result["status"], {"ready", "partial", "insufficient_data"})

        # overall_summary keys
        os_keys = set(result["overall_summary"].keys())
        expected_os_keys = {"total_counted_reps", "total_uncounted_attempts", "recording_quality"}
        self.assertEqual(os_keys, expected_os_keys)

        rq_keys = set(result["overall_summary"]["recording_quality"].keys())
        expected_rq_keys = {"camera_view_adequate", "tracking_continuity", "posture_coverage_pct", "notes"}
        self.assertEqual(rq_keys, expected_rq_keys)
        self.assertIn(result["overall_summary"]["recording_quality"]["tracking_continuity"], {"adequate", "interrupted", "insufficient"})

        # disclaimers keys
        d_keys = set(result["disclaimers"].keys())
        expected_d_keys = {"clinical_disclaimer", "human_validation_status"}
        self.assertEqual(d_keys, expected_d_keys)
        self.assertIn("UNVALIDATED", result["disclaimers"]["human_validation_status"])

    def test_normal_multi_rep_session(self):
        """Verify normal multi-rep session produces ready status and complete cues."""
        data = self._build_synthetic_fixture()
        result = generate_coaching_feedback(
            summary=data["summary"],
            attempts=data["attempts"],
            posture_summary=data["posture_summary"],
            feedback_observations=data["feedback_observations"],
            feedback_phases=data["feedback_phases"],
        )

        self.assertEqual(result["status"], "ready")
        self.assertEqual(result["overall_summary"]["total_counted_reps"], 2)
        self.assertEqual(result["overall_summary"]["total_uncounted_attempts"], 1)
        self.assertIsNone(result["overall_summary"]["recording_quality"]["camera_view_adequate"])
        self.assertEqual(result["overall_summary"]["recording_quality"]["tracking_continuity"], "adequate")

        # per_repetition_cues verification
        cues = result["per_repetition_cues"]
        self.assertEqual(len(cues), 2)

        rep1 = cues[0]
        self.assertEqual(rep1["rep_index"], 1)
        self.assertEqual(rep1["attempt_id"], 1)
        self.assertTrue(rep1["is_counted"])
        self.assertEqual(rep1["timestamps"]["start_time_s"], 0.33)
        self.assertEqual(rep1["timestamps"]["inflection_time_s"], 0.83)
        self.assertEqual(rep1["timestamps"]["end_time_s"], 1.33)
        self.assertEqual(rep1["measurements"]["min_knee_angle_deg"], 88.5)
        self.assertEqual(rep1["measurements"]["median_torso_inclination_deg"], 22.0)
        self.assertEqual(rep1["measurements"]["median_hip_angle_deg"], 115.0)

        # Observation rule IDs
        rule_ids = [obs["rule_id"] for obs in rep1["observations"]]
        self.assertIn("OBS-DEPTH-01", rule_ids)
        self.assertIn("OBS-PACING-01", rule_ids)
        self.assertIn("OBS-TORSO-VAR-01", rule_ids)
        self.assertIn("OBS-TRACKING-01", rule_ids)

        for obs in rep1["observations"]:
            self.assertIn(obs["category"], {"depth", "timing", "posture_consistency", "tracking", "context"})
            self.assertIn(obs["confidence_level"], {"observed", "hypothesis", "unavailable"})

        # Rep 2 relative depth observation
        rep2 = cues[1]
        self.assertEqual(rep2["rep_index"], 2)
        depth_obs = next(o for o in rep2["observations"] if o["rule_id"] == "OBS-DEPTH-01")
        self.assertIn("relatively shallower", depth_obs["message"])

        # Suggestions
        sug_types = [s["type"] for s in rep1["suggestions"]]
        self.assertIn("review_recommendation", sug_types)

    def test_missing_or_none_optional_inputs(self):
        """Verify behavior when optional inputs are None or missing."""
        summary = {
            "status": "completed",
            "completed_reps": 2,
            "processed_frames": 60,
            "valid_observations": 55,
            "valid_observation_fraction": 0.9167,
        }

        result = generate_coaching_feedback(
            summary=summary,
            attempts=None,
            posture_summary=None,
            feedback_observations=None,
            feedback_phases=None,
        )

        self.assertEqual(result["status"], "partial")
        self.assertEqual(result["overall_summary"]["total_counted_reps"], 2)
        self.assertEqual(result["overall_summary"]["total_uncounted_attempts"], 0)
        self.assertEqual(len(result["per_repetition_cues"]), 2)
        for cue in result["per_repetition_cues"]:
            self.assertTrue(cue["is_counted"])
            self.assertIsNone(cue["attempt_id"])
            self.assertIsNone(cue["measurements"]["min_knee_angle_deg"])

    def test_empty_summary_and_all_none(self):
        """Verify completely empty input returns insufficient_data without crashing."""
        result = generate_coaching_feedback(summary={})
        self.assertEqual(result["status"], "insufficient_data")
        self.assertEqual(result["overall_summary"]["total_counted_reps"], 0)
        self.assertEqual(result["overall_summary"]["total_uncounted_attempts"], 0)
        self.assertEqual(result["per_repetition_cues"], [])
        self.assertEqual(result["uncounted_movement_explanations"], [])
        self.assertEqual(result["overall_summary"]["recording_quality"]["tracking_continuity"], "insufficient")

    def test_low_posture_measurement_coverage(self):
        """Verify low frame coverage (<50%) marks posture metrics unavailable and offers setup cue."""
        data = self._build_synthetic_fixture()
        # Set coverage to 15%
        data["posture_summary"]["events"][0]["torso_inclination"]["coverage_fraction"] = 0.15
        data["posture_summary"]["events"][0]["torso_inclination"]["measured_count"] = 5
        data["feedback_observations"]["attempts"][0]["availability"]["torso"]["coverage_fraction"] = 0.15

        result = generate_coaching_feedback(
            summary=data["summary"],
            attempts=data["attempts"],
            posture_summary=data["posture_summary"],
            feedback_observations=data["feedback_observations"],
            feedback_phases=data["feedback_phases"],
        )

        rep1 = result["per_repetition_cues"][0]
        torso_obs = next(o for o in rep1["observations"] if o["rule_id"] == "OBS-TORSO-VAR-01")
        self.assertEqual(torso_obs["confidence_level"], "unavailable")
        self.assertIn("below operational 50.0% threshold", torso_obs["message"])

        # Recording setup suggestion should be triggered
        sug_rules = [s["rule_id"] for s in rep1["suggestions"]]
        self.assertIn("SUG-REC-01", sug_rules)

    def test_zero_posture_measurement_coverage(self):
        """Verify 0% coverage returns partial status and unavailable telemetry."""
        summary = {
            "status": "completed",
            "completed_reps": 1,
            "processed_frames": 50,
            "valid_observations": 0,
            "valid_observation_fraction": 0.0,
            "posture_observations": {"torso_valid_frames": 0, "processed_frames": 50},
        }
        attempts = {
            "counted_attempts": 1,
            "uncounted_attempts": 0,
            "events": [
                {
                    "attempt_id": 1,
                    "status": "counted",
                    "reason": "completed_cycle",
                    "track_id": 1,
                    "side": "right",
                    "valid_frames": 0,
                    "observed_frames": 50,
                    "valid_frame_fraction": 0.0,
                }
            ],
        }

        result = generate_coaching_feedback(summary=summary, attempts=attempts)
        self.assertEqual(result["status"], "partial")
        self.assertEqual(result["overall_summary"]["recording_quality"]["tracking_continuity"], "insufficient")
        self.assertEqual(result["overall_summary"]["recording_quality"]["posture_coverage_pct"], 0.0)

    def test_inconsistent_track_or_body_side(self):
        """Verify attempts across multiple tracks/sides suppress consistency and flag view."""
        data = self._build_synthetic_fixture()
        # Rep 1 is track 1, right side; make rep 2 track 2, left side
        data["attempts"]["events"][1]["track_id"] = 2
        data["attempts"]["events"][1]["side"] = "left"
        data["posture_summary"]["events"][1]["track_id"] = 2
        data["posture_summary"]["events"][1]["side"] = "left"

        result = generate_coaching_feedback(
            summary=data["summary"],
            attempts=data["attempts"],
            posture_summary=data["posture_summary"],
            feedback_observations=data["feedback_observations"],
            feedback_phases=data["feedback_phases"],
        )

        self.assertFalse(result["overall_summary"]["recording_quality"]["camera_view_adequate"])
        rep1 = result["per_repetition_cues"][0]
        torso_obs = next(o for o in rep1["observations"] if o["rule_id"] == "OBS-TORSO-VAR-01")
        self.assertEqual(torso_obs["confidence_level"], "unavailable")
        self.assertIn("suppressed across differing tracks or body sides", torso_obs["message"])

        # Notes describe multi-context tracking
        notes_text = " ".join(result["overall_summary"]["recording_quality"]["notes"])
        self.assertIn("multiple tracking contexts or body sides", notes_text)

    def test_uncounted_setup_and_weight_return_movements(self):
        """Verify uncounted movements classify context hypotheses without form critique."""
        data = self._build_synthetic_fixture()
        result = generate_coaching_feedback(
            summary=data["summary"],
            attempts=data["attempts"],
            posture_summary=data["posture_summary"],
            feedback_observations=data["feedback_observations"],
            feedback_phases=data["feedback_phases"],
        )

        uncounted = result["uncounted_movement_explanations"]
        self.assertEqual(len(uncounted), 1)
        item = uncounted[0]
        self.assertEqual(item["attempt_id"], 3)
        self.assertEqual(item["reason_code"], "depth_threshold_not_reached")
        # Attempt 3 occurred after counted reps with high knee angle (138°)
        self.assertEqual(item["context_hypothesis"], "possible_setup_or_weight_return")
        self.assertIn("depth threshold", item["explanation"])

    def test_uncounted_hypotheses_comprehensive(self):
        """Verify all four contract hypotheses are properly produced across reason codes."""
        attempts = {
            "counted_attempts": 1,
            "uncounted_attempts": 4,
            "events": [
                {
                    "attempt_id": 1,
                    "status": "uncounted",
                    "reason": "depth_threshold_not_reached",
                    "min_knee_angle_deg": 140.0,
                },
                {
                    "attempt_id": 2,
                    "status": "counted",
                    "reason": "completed_cycle",
                },
                {
                    "attempt_id": 3,
                    "status": "uncounted",
                    "reason": "depth_not_sustained",
                    "min_knee_angle_deg": 105.0,
                },
                {
                    "attempt_id": 4,
                    "status": "uncounted",
                    "reason": "tracking_gap",
                },
                {
                    "attempt_id": 5,
                    "status": "uncounted",
                    "reason": "video_ended_mid_attempt",
                },
            ],
        }

        result = generate_coaching_feedback(summary={"completed_reps": 1}, attempts=attempts)
        exps = {e["attempt_id"]: e["context_hypothesis"] for e in result["uncounted_movement_explanations"]}

        self.assertEqual(exps[1], "possible_setup_or_weight_return")
        self.assertEqual(exps[3], "ambiguous")
        self.assertEqual(exps[4], "tracking_lost")
        self.assertEqual(exps[5], "unfinished_cycle")

    def test_pre_arming_unconfirmed_crouch(self):
        """Verify pre-arming opening crouch frames are noted and excluded from count."""
        summary = {"completed_reps": 1, "processed_frames": 50}
        attempts = {
            "counted_attempts": 1,
            "uncounted_attempts": 0,
            "pre_arming": {
                "unarmed_frames": 15,
                "opening_lowered_frames": 10,
                "explanation": "No attempt assigned.",
            },
            "events": [{"attempt_id": 1, "status": "counted", "reason": "completed_cycle"}],
        }

        result = generate_coaching_feedback(summary=summary, attempts=attempts)
        notes = " ".join(result["overall_summary"]["recording_quality"]["notes"])
        self.assertIn("opening crouch was detected across 10 frames", notes)

    def test_valid_zero_degree_torso_inclination(self):
        """Verify 0.0 degree torso inclination (perfect vertical) is preserved and not treated as None."""
        data = self._build_synthetic_fixture()
        data["posture_summary"]["events"][0]["torso_inclination"]["median_deg"] = 0.0
        data["feedback_observations"]["attempts"][0]["evidence"]["torso_median_deg"] = 0.0

        result = generate_coaching_feedback(
            summary=data["summary"],
            attempts=data["attempts"],
            posture_summary=data["posture_summary"],
            feedback_observations=data["feedback_observations"],
            feedback_phases=data["feedback_phases"],
        )

        rep1 = result["per_repetition_cues"][0]
        self.assertEqual(rep1["measurements"]["median_torso_inclination_deg"], 0.0)
        self.assertIsNotNone(rep1["measurements"]["median_torso_inclination_deg"])
        torso_obs = next(o for o in rep1["observations"] if o["rule_id"] == "OBS-TORSO-VAR-01")
        self.assertIn("0.00° from vertical", torso_obs["message"])

    def test_non_normative_guardrails(self):
        """Verify feedback contains no diagnostic, valgus/varus, or medical injury claims."""
        data = self._build_synthetic_fixture()
        result = generate_coaching_feedback(
            summary=data["summary"],
            attempts=data["attempts"],
            posture_summary=data["posture_summary"],
            feedback_observations=data["feedback_observations"],
            feedback_phases=data["feedback_phases"],
        )

        prohibited_terms = [
            "spinal rounding",
            "lumbar flexion",
            "valgus",
            "varus",
            "injury risk",
            "bad form",
            "incorrect technique",
            "unsafe",
            "pathology",
        ]

        # Scan all generated messages
        cue = result["per_repetition_cues"][0]
        for obs in cue["observations"]:
            for term in prohibited_terms:
                self.assertNotIn(term, obs["message"].lower())

        for sug in cue["suggestions"]:
            for term in prohibited_terms:
                self.assertNotIn(term, sug["message"].lower())

        for unc in result["uncounted_movement_explanations"]:
            for term in prohibited_terms:
                self.assertNotIn(term, unc["explanation"].lower())

    def test_zero_ml_imports(self):
        """Verify coaching_feedback relies strictly on pure Python stdlib and imports zero ML libs."""
        forbidden_modules = {"torch", "torchvision", "ultralytics", "cv2", "numpy", "scipy", "sklearn"}
        imported_modules = set(sys.modules.keys())

        # Inspect module imports of coaching_feedback
        import gym_analyser.coaching_feedback as cf
        cf_globals = dir(cf)

        for mod in forbidden_modules:
            self.assertNotIn(mod, cf_globals, f"Forbidden module {mod} found in coaching_feedback namespace")

    def test_input_immutability(self):
        """Verify generate_coaching_feedback does not mutate input dictionaries."""
        data = self._build_synthetic_fixture()
        data_copy = copy.deepcopy(data)

        _ = generate_coaching_feedback(
            summary=data["summary"],
            attempts=data["attempts"],
            posture_summary=data["posture_summary"],
            feedback_observations=data["feedback_observations"],
            feedback_phases=data["feedback_phases"],
        )

        self.assertEqual(data, data_copy)

    def test_only_uncounted_attempts(self):
        """Verify session with 0 counted reps and only uncounted attempts produces partial status."""
        summary = {"completed_reps": 0, "processed_frames": 40}
        attempts = {
            "counted_attempts": 0,
            "uncounted_attempts": 2,
            "events": [
                {
                    "attempt_id": 1,
                    "status": "uncounted",
                    "reason": "depth_threshold_not_reached",
                    "min_knee_angle_deg": 140.0,
                },
                {
                    "attempt_id": 2,
                    "status": "uncounted",
                    "reason": "video_ended_mid_attempt",
                },
            ],
        }

        result = generate_coaching_feedback(summary=summary, attempts=attempts)
        self.assertEqual(result["status"], "partial")
        self.assertEqual(result["overall_summary"]["total_counted_reps"], 0)
        self.assertEqual(result["overall_summary"]["total_uncounted_attempts"], 2)
        self.assertEqual(result["per_repetition_cues"], [])
        self.assertEqual(len(result["uncounted_movement_explanations"]), 2)

    def test_rapid_pacing_triggers_pacing_suggestion(self):
        """Verify very fast descent duration triggers movement_pacing suggestion."""
        data = self._build_synthetic_fixture()
        # Set descent phase to 0.4s (rapid) and return to 1.5s
        data["feedback_phases"]["attempts"][0]["descent_phase"]["phase_start_time_s"] = 0.50
        data["feedback_phases"]["attempts"][0]["descent_phase"]["phase_end_time_s"] = 0.90
        data["feedback_phases"]["attempts"][0]["min_knee_point"]["video_time_s"] = 0.90
        data["feedback_phases"]["attempts"][0]["return_phase"]["phase_start_time_s"] = 0.90
        data["feedback_phases"]["attempts"][0]["return_phase"]["phase_end_time_s"] = 2.40

        result = generate_coaching_feedback(
            summary=data["summary"],
            attempts=data["attempts"],
            posture_summary=data["posture_summary"],
            feedback_observations=data["feedback_observations"],
            feedback_phases=data["feedback_phases"],
        )

        rep1 = result["per_repetition_cues"][0]
        sug_types = [s["type"] for s in rep1["suggestions"]]
        self.assertIn("movement_pacing", sug_types)
        pacing_sug = next(s for s in rep1["suggestions"] if s["type"] == "movement_pacing")
        self.assertEqual(pacing_sug["rule_id"], "SUG-PACING-01")

    def test_timestamps_out_of_order(self):
        """Verify chronological sequence mismatch marks timing unavailable without crashing."""
        data = self._build_synthetic_fixture()
        # Reverse timestamps
        data["feedback_phases"]["attempts"][0]["descent_phase"]["phase_start_time_s"] = 2.00
        data["feedback_phases"]["attempts"][0]["min_knee_point"]["video_time_s"] = 1.00
        data["feedback_phases"]["attempts"][0]["return_phase"]["phase_end_time_s"] = 0.50

        result = generate_coaching_feedback(
            summary=data["summary"],
            attempts=data["attempts"],
            posture_summary=data["posture_summary"],
            feedback_observations=data["feedback_observations"],
            feedback_phases=data["feedback_phases"],
        )

        rep1 = result["per_repetition_cues"][0]
        timing_obs = next(o for o in rep1["observations"] if o["rule_id"] == "OBS-PACING-01")
        self.assertEqual(timing_obs["confidence_level"], "unavailable")

    def test_exact_disclaimer_strings(self):
        """Verify disclaimers exactly match FEEDBACK_CONTRACT_38.md specification."""
        result = generate_coaching_feedback(summary={"completed_reps": 1})
        self.assertEqual(
            result["disclaimers"]["clinical_disclaimer"],
            "These 2D kinematic measurements are descriptive planar observations, "
            "not clinical diagnostic assessments or technique grades.",
        )
        self.assertEqual(
            result["disclaimers"]["human_validation_status"],
            "UNVALIDATED by human expert review on diverse unseen subjects.",
        )

    # -------------------------------------------------------------------------
    # Review 43 Findings: Camera View Adequacy (Item 1)
    # -------------------------------------------------------------------------

    def test_camera_view_adequate_none_for_normal_adequate_session(self):
        """Verify 2D monocular tracking defaults camera_view_adequate to None (not True)."""
        data = self._build_synthetic_fixture()
        result = generate_coaching_feedback(
            summary=data["summary"],
            attempts=data["attempts"],
            posture_summary=data["posture_summary"],
            feedback_observations=data["feedback_observations"],
            feedback_phases=data["feedback_phases"],
        )
        rq = result["overall_summary"]["recording_quality"]
        self.assertEqual(rq["tracking_continuity"], "adequate")
        self.assertIsNone(rq["camera_view_adequate"])
        notes_text = " ".join(rq["notes"])
        self.assertIn("Camera view adequacy and 3D orthogonality cannot be verified", notes_text)

    def test_camera_view_adequate_false_for_interrupted_tracking(self):
        """Verify camera_view_adequate is False when tracking continuity is interrupted."""
        summary = {
            "status": "completed",
            "completed_reps": 1,
            "processed_frames": 100,
            "valid_observations": 50,
            "valid_observation_fraction": 0.50,  # 50% -> interrupted
        }
        attempts = {
            "counted_attempts": 1,
            "uncounted_attempts": 0,
            "events": [
                {
                    "attempt_id": 1,
                    "status": "counted",
                    "reason": "completed_cycle",
                    "track_id": 1,
                    "side": "right",
                    "valid_frames": 50,
                    "observed_frames": 100,
                }
            ],
        }
        result = generate_coaching_feedback(summary=summary, attempts=attempts)
        rq = result["overall_summary"]["recording_quality"]
        self.assertEqual(rq["tracking_continuity"], "interrupted")
        self.assertFalse(rq["camera_view_adequate"])

    def test_camera_view_adequate_false_for_mid_attempt_side_switch(self):
        """Verify camera_view_adequate is False when mid-attempt side_changed event occurs."""
        summary = {"status": "completed", "completed_reps": 1, "processed_frames": 50}
        attempts = {
            "counted_attempts": 1,
            "uncounted_attempts": 1,
            "events": [
                {
                    "attempt_id": 1,
                    "status": "counted",
                    "reason": "completed_cycle",
                    "track_id": 1,
                    "side": "right",
                },
                {
                    "attempt_id": 2,
                    "status": "uncounted",
                    "reason": "side_changed",
                    "track_id": 1,
                    "side": "left",
                },
            ],
        }
        result = generate_coaching_feedback(summary=summary, attempts=attempts)
        rq = result["overall_summary"]["recording_quality"]
        self.assertFalse(rq["camera_view_adequate"])
        notes_text = " ".join(rq["notes"])
        self.assertIn("multiple tracking contexts or body sides", notes_text)

    def test_camera_view_adequate_never_reports_true_from_tracking_alone(self):
        """Verify camera_view_adequate is never True across any 2D tracking configuration."""
        for vof in (1.0, 0.95, 0.75, 0.50, 0.20, 0.0):
            summary = {
                "status": "completed",
                "completed_reps": 1,
                "processed_frames": 100,
                "valid_observations": int(100 * vof),
                "valid_observation_fraction": vof,
            }
            res = generate_coaching_feedback(summary=summary)
            adequacy = res["overall_summary"]["recording_quality"]["camera_view_adequate"]
            self.assertNotEqual(adequacy, True, f"camera_view_adequate was True for vof={vof}")
            self.assertIn(adequacy, (None, False))

    # -------------------------------------------------------------------------
    # Review 43 Findings: Turnaround Depth Observation (Item 3 & Probes)
    # -------------------------------------------------------------------------

    def test_probe_failed_run_unknown_coverage_does_not_emit_observed_turnaround(self):
        """Reproduction of probe 'failed_run_unknown_coverage_observed_depth'.

        When summary has status="failed" (or unknown coverage), do NOT emit OBS-DEPTH-01
        claiming a turnaround point or cycle crossing. Mark unavailable with explicit reason.
        """
        summary = {
            "status": "failed",
            "completed_reps": 1,
            "processed_frames": 50,
            # No valid_observation_fraction, no posture_observations
        }
        attempts = {
            "counted_attempts": 1,
            "uncounted_attempts": 0,
            "events": [
                {
                    "attempt_id": 1,
                    "status": "counted",
                    "reason": "completed_cycle",
                    "track_id": 1,
                    "side": "right",
                    "min_knee_angle_deg": 90.0,
                    # No frame coverage counts
                }
            ],
        }
        # Note: feedback_phases is completely missing (as in probe)
        result = generate_coaching_feedback(summary=summary, attempts=attempts)

        self.assertEqual(result["status"], "partial")
        cues = result["per_repetition_cues"]
        self.assertEqual(len(cues), 1)

        rep1 = cues[0]
        # Raw measurement is preserved (no blanket erasure)
        self.assertEqual(rep1["measurements"]["min_knee_angle_deg"], 90.0)

        # But OBS-DEPTH-01 must NOT claim turnaround point or full cycle crossing
        depth_obs = next(o for o in rep1["observations"] if o["rule_id"] == "OBS-DEPTH-01")
        self.assertEqual(depth_obs["confidence_level"], "unavailable")
        self.assertNotIn("turnaround point", depth_obs["message"].lower())
        self.assertNotIn("crossing the configured", depth_obs["message"].lower())
        self.assertIn("unavailable", depth_obs["message"].lower())

    def test_missing_feedback_phases_marks_turnaround_depth_unavailable(self):
        """Verify OBS-DEPTH-01 requires feedback_phases evidence to claim turnaround depth."""
        summary = {
            "status": "completed",
            "completed_reps": 1,
            "processed_frames": 100,
            "valid_observations": 90,
            "valid_observation_fraction": 0.90,
        }
        attempts = {
            "counted_attempts": 1,
            "uncounted_attempts": 0,
            "events": [
                {
                    "attempt_id": 1,
                    "status": "counted",
                    "track_id": 1,
                    "side": "right",
                    "min_knee_angle_deg": 85.0,
                    "valid_frames": 90,
                    "observed_frames": 100,
                }
            ],
        }
        # feedback_phases is omitted
        result = generate_coaching_feedback(summary=summary, attempts=attempts, feedback_phases=None)
        rep1 = result["per_repetition_cues"][0]
        depth_obs = next(o for o in rep1["observations"] if o["rule_id"] == "OBS-DEPTH-01")
        self.assertEqual(depth_obs["confidence_level"], "unavailable")
        self.assertIn("verified phase inflection evidence is missing", depth_obs["message"])
        # Measurement angle preserved
        self.assertEqual(rep1["measurements"]["min_knee_angle_deg"], 85.0)

    def test_insufficient_phase_samples_suppresses_turnaround_observation(self):
        """Verify phase sample count < 3 marks turnaround observation unavailable."""
        data = self._build_synthetic_fixture()
        # Set descent sample count to 1 and return to 1 (total 2 < 3)
        data["feedback_phases"]["attempts"][0]["descent_phase"]["sample_count"] = 1
        data["feedback_phases"]["attempts"][0]["return_phase"]["sample_count"] = 1
        data["feedback_phases"]["attempts"][0]["sample_count"] = 2

        result = generate_coaching_feedback(
            summary=data["summary"],
            attempts=data["attempts"],
            posture_summary=data["posture_summary"],
            feedback_observations=data["feedback_observations"],
            feedback_phases=data["feedback_phases"],
        )
        rep1 = result["per_repetition_cues"][0]
        depth_obs = next(o for o in rep1["observations"] if o["rule_id"] == "OBS-DEPTH-01")
        self.assertEqual(depth_obs["confidence_level"], "unavailable")
        self.assertIn("sample count < 3", depth_obs["message"])

    def test_incomplete_run_suppresses_turnaround_observation(self):
        """Verify summary status="incomplete" marks turnaround observation unavailable."""
        data = self._build_synthetic_fixture()
        data["summary"]["status"] = "incomplete"

        result = generate_coaching_feedback(
            summary=data["summary"],
            attempts=data["attempts"],
            posture_summary=data["posture_summary"],
            feedback_observations=data["feedback_observations"],
            feedback_phases=data["feedback_phases"],
        )
        self.assertEqual(result["status"], "partial")
        rep1 = result["per_repetition_cues"][0]
        depth_obs = next(o for o in rep1["observations"] if o["rule_id"] == "OBS-DEPTH-01")
        self.assertEqual(depth_obs["confidence_level"], "unavailable")
        self.assertIn("incomplete", depth_obs["message"])

    # -------------------------------------------------------------------------
    # Review 43 Findings: Hypothesis Classifier & No 135° Heuristic (Item 2)
    # -------------------------------------------------------------------------

    def test_uncounted_depth_not_sustained_is_ambiguous_not_shallow(self):
        """Verify depth_not_sustained produces ambiguous (not shallow_depth) without justification."""
        attempts = {
            "counted_attempts": 1,
            "uncounted_attempts": 1,
            "events": [
                {"attempt_id": 1, "status": "counted", "reason": "completed_cycle"},
                {
                    "attempt_id": 2,
                    "status": "uncounted",
                    "reason": "depth_not_sustained",
                    "min_knee_angle_deg": 105.0,
                },
            ],
        }
        result = generate_coaching_feedback(summary={"completed_reps": 1}, attempts=attempts)
        unc = result["uncounted_movement_explanations"]
        self.assertEqual(len(unc), 1)
        self.assertEqual(unc[0]["context_hypothesis"], "ambiguous")
        self.assertEqual(unc[0]["detector_reason"], "depth_not_sustained")
        self.assertEqual(unc[0]["reason_code"], "depth_not_sustained")

    def test_uncounted_no_135_degree_threshold_for_setup(self):
        """Verify uncounted setup before counted reps produces setup without 135° angle cutoff."""
        # Lifter did an opening squat crouch reaching 115° (well below old 135° cutoff)
        attempts = {
            "counted_attempts": 1,
            "uncounted_attempts": 1,
            "events": [
                {
                    "attempt_id": 1,
                    "status": "uncounted",
                    "reason": "depth_threshold_not_reached",
                    "min_knee_angle_deg": 115.0,  # Below 135.0°
                },
                {"attempt_id": 2, "status": "counted", "reason": "completed_cycle"},
            ],
        }
        result = generate_coaching_feedback(summary={"completed_reps": 1}, attempts=attempts)
        unc = result["uncounted_movement_explanations"]
        self.assertEqual(len(unc), 1)
        # Occurred before the first counted rep -> possible_setup_or_weight_return
        self.assertEqual(unc[0]["context_hypothesis"], "possible_setup_or_weight_return")

    def test_uncounted_between_counted_reps_remains_ambiguous(self):
        """Temporal placement between counted reps does not establish a separate shallow attempt."""
        attempts = {
            "counted_attempts": 2,
            "uncounted_attempts": 1,
            "events": [
                {"attempt_id": 1, "status": "counted", "reason": "completed_cycle"},
                {
                    "attempt_id": 2,
                    "status": "uncounted",
                    "reason": "depth_threshold_not_reached",
                    "min_knee_angle_deg": 118.0,
                },
                {"attempt_id": 3, "status": "counted", "reason": "completed_cycle"},
            ],
        }
        result = generate_coaching_feedback(summary={"completed_reps": 2}, attempts=attempts)
        unc = result["uncounted_movement_explanations"]
        self.assertEqual(len(unc), 1)
        self.assertEqual(unc[0]["context_hypothesis"], "ambiguous")

    def test_uncounted_no_counted_reps_in_session_is_ambiguous(self):
        """Verify depth_threshold_not_reached with no counted reps produces ambiguous."""
        attempts = {
            "counted_attempts": 0,
            "uncounted_attempts": 1,
            "events": [
                {
                    "attempt_id": 1,
                    "status": "uncounted",
                    "reason": "depth_threshold_not_reached",
                    "min_knee_angle_deg": 125.0,
                }
            ],
        }
        result = generate_coaching_feedback(summary={"completed_reps": 0}, attempts=attempts)
        unc = result["uncounted_movement_explanations"]
        self.assertEqual(len(unc), 1)
        self.assertEqual(unc[0]["context_hypothesis"], "ambiguous")

    def test_uncounted_unrecognized_reason_produces_unknown_no_substring_guess(self):
        """Verify arbitrary reasons produce unknown without substring matching."""
        attempts = {
            "counted_attempts": 1,
            "uncounted_attempts": 1,
            "events": [
                {"attempt_id": 1, "status": "counted", "reason": "completed_cycle"},
                {
                    "attempt_id": 2,
                    "status": "uncounted",
                    "reason": "unusual_knee_motion_irregularity",
                    "detector_reason": "unusual_knee_motion_irregularity",
                },
            ],
        }
        result = generate_coaching_feedback(summary={"completed_reps": 1}, attempts=attempts)
        unc = result["uncounted_movement_explanations"]
        self.assertEqual(len(unc), 1)
        self.assertEqual(unc[0]["context_hypothesis"], "unknown")
        self.assertEqual(unc[0]["detector_reason"], "unusual_knee_motion_irregularity")

    def test_uncounted_preserves_explicit_upstream_hypothesis(self):
        """Verify explicit upstream hypothesis in attempt dictionary is honored."""
        attempts = {
            "counted_attempts": 1,
            "uncounted_attempts": 1,
            "events": [
                {"attempt_id": 1, "status": "counted", "reason": "completed_cycle"},
                {
                    "attempt_id": 2,
                    "status": "uncounted",
                    "reason": "depth_not_sustained",
                    "context_hypothesis": "shallow_depth",  # Upstream justification
                },
            ],
        }
        result = generate_coaching_feedback(summary={"completed_reps": 1}, attempts=attempts)
        unc = result["uncounted_movement_explanations"]
        self.assertEqual(unc[0]["context_hypothesis"], "shallow_depth")

    # -------------------------------------------------------------------------
    # Review 43 Findings: Count Authority, Conflicts & Duplicates (Item 4)
    # -------------------------------------------------------------------------

    def test_explicit_zero_reps_preserved_without_recounting(self):
        """Verify count-authoritative explicit 0 is preserved and not silently recounted."""
        summary = {"completed_reps": 0, "processed_frames": 50}
        attempts = {
            "counted_attempts": 0,
            "uncounted_attempts": 0,
            "events": [],
        }
        result = generate_coaching_feedback(summary=summary, attempts=attempts)
        self.assertEqual(result["overall_summary"]["total_counted_reps"], 0)
        self.assertIsInstance(result["overall_summary"]["total_counted_reps"], int)
        self.assertEqual(result["overall_summary"]["total_uncounted_attempts"], 0)

    def test_count_conflict_flagged_in_notes_and_status(self):
        """Verify conflict between summary completed_reps and attempts counted_attempts is flagged."""
        summary = {"completed_reps": 5, "processed_frames": 100}
        attempts = {
            "counted_attempts": 2,
            "uncounted_attempts": 1,
            "events": [
                {"attempt_id": 1, "status": "counted", "reason": "completed_cycle"},
                {"attempt_id": 2, "status": "counted", "reason": "completed_cycle"},
            ],
        }
        result = generate_coaching_feedback(summary=summary, attempts=attempts)
        self.assertEqual(result["status"], "partial")
        self.assertEqual(result["overall_summary"]["total_counted_reps"], 2)
        notes_text = " ".join(result["overall_summary"]["recording_quality"]["notes"])
        self.assertIn("Count conflict detected", notes_text)
        self.assertIn("summary completed_reps (5) differs from attempts counted_attempts (2)", notes_text)

    def test_duplicate_attempt_ids_filtered_and_flagged(self):
        """Verify duplicate attempt IDs are filtered and flagged as malformed metadata."""
        summary = {"completed_reps": 1, "processed_frames": 100}
        attempts = {
            "counted_attempts": 1,
            "uncounted_attempts": 0,
            "events": [
                {"attempt_id": 1, "status": "counted", "reason": "completed_cycle", "min_knee_angle_deg": 90.0},
                {"attempt_id": 1, "status": "counted", "reason": "completed_cycle", "min_knee_angle_deg": 92.0},  # Duplicate
            ],
        }
        result = generate_coaching_feedback(summary=summary, attempts=attempts)
        self.assertEqual(result["status"], "partial")
        # Only 1 unique rep cue
        self.assertEqual(len(result["per_repetition_cues"]), 1)
        notes_text = " ".join(result["overall_summary"]["recording_quality"]["notes"])
        self.assertIn("duplicate attempt IDs were identified and filtered", notes_text)

    def test_out_of_range_angle_measurements_rejected(self):
        """Verify invalid/out-of-range angles (>180° or <0°) are set to None."""
        summary = {"completed_reps": 1, "processed_frames": 50}
        attempts = {
            "counted_attempts": 1,
            "uncounted_attempts": 0,
            "events": [
                {
                    "attempt_id": 1,
                    "status": "counted",
                    "reason": "completed_cycle",
                    "min_knee_angle_deg": 250.0,  # Invalid: > 180°
                    "valid_frames": 30,
                    "observed_frames": 30,
                }
            ],
        }
        posture_summary = {
            "events": [
                {
                    "attempt_id": 1,
                    "status": "counted",
                    "torso_inclination": {"median_deg": -15.0, "coverage_fraction": 1.0},  # Invalid: < 0°
                    "hip_angle": {"median_deg": 210.0},  # Invalid: > 180°
                }
            ]
        }
        result = generate_coaching_feedback(summary=summary, attempts=attempts, posture_summary=posture_summary)
        rep1 = result["per_repetition_cues"][0]
        self.assertIsNone(rep1["measurements"]["min_knee_angle_deg"])
        self.assertIsNone(rep1["measurements"]["median_torso_inclination_deg"])
        self.assertIsNone(rep1["measurements"]["median_hip_angle_deg"])


    def test_missing_or_invalid_depth_configuration_is_not_invented(self):
        for value in (None, {}, {"down_angle": None}, {"down_angle": True},
                      {"down_angle": float("nan")}, {"down_angle": -1}, {"down_angle": 190}):
            with self.subTest(configuration=value):
                fixture = self._build_synthetic_fixture()
                fixture["summary"]["configuration"] = value
                result = generate_coaching_feedback(**fixture)
                messages = [ob["message"] for rep in result["per_repetition_cues"]
                            for ob in rep["observations"] if ob["rule_id"] == "OBS-DEPTH-01"]
                self.assertTrue(messages)
                self.assertTrue(all("threshold is unavailable" in message for message in messages))
                self.assertTrue(all("110.0" not in message for message in messages))

if __name__ == "__main__":
    unittest.main()
