# -*- coding: utf-8 -*-
"""Behavioral tests and regressions for phase-localized movement feedback module.

Tests target:
- Valid continuous counted cycle with known turnaround and return-phase torso change
- Honest missing timestamp handling without fabrication or format crashes
- Rejection of negative, non-finite (NaN), and descending timestamps
- Suppression of cycles with duplicate timestamps during active motion
- Unknown side rejection and prevention of unknown-side comparison groups
- Knee selected_side and torso posture_side conflict handling
- Invalid torso_reason gating finite numeric angle values
- Separate actual delta endpoint evidence vs phase interval bounds
- Suppression of delta assertion when fewer than 2 valid temporal measurements exist
- Duplicate attempt IDs and malformed status raising ValueError
- 0.0 degree angle validity
- Frame gap suppression regardless of missing time
- Exclusion of uncounted setup/return attempts from consistency
- Relative depth comparison across counted cycles in matching context
- Strict immutability of inputs
- Real CSV-schema fixture derived from pipeline definitions
"""

import copy
import unittest
from typing import Any, Dict, List

from gym_analyser.phase_feedback import PhaseFeedbackConfig, build_phase_feedback


class TestPhaseFeedback(unittest.TestCase):
    """Test suite verifying phase-localized feedback behavioral contract and regressions."""

    def _make_row(
        self,
        frame_index: int,
        video_time_s: Any = None,
        knee_deg: Any = 140.0,
        torso_deg: Any = 15.0,
        hip_deg: Any = 135.0,
        track_id: Any = 1,
        selected_side: str = "right",
        posture_side: str = "right",
        valid: bool = True,
        reason: str = "ok",
        torso_reason: str = "ok",
        hip_reason: str = "ok",
        stage: str = "standing",
        attempt_id: Any = 1,
    ) -> Dict[str, Any]:
        """Create a synthetic frame row matching complete 17-column angles.csv schema."""
        vt_str = ""
        if video_time_s is not None:
            vt_str = str(video_time_s) if isinstance(video_time_s, str) else f"{float(video_time_s):.4f}"

        return {
            "frame_index": frame_index,
            "video_time_s": vt_str,
            "track_id": str(track_id) if track_id is not None else "",
            "selected_side": selected_side,
            "knee_angle_deg": str(knee_deg) if knee_deg is not None else "",
            "confidence": "0.9500",
            "valid": "True" if valid else "False",
            "reason": reason,
            "stage": stage,
            "completed_reps": "0",
            "posture_side": posture_side,
            "torso_inclination_deg": str(torso_deg) if torso_deg is not None else "",
            "hip_angle_deg": str(hip_deg) if hip_deg is not None else "",
            "torso_reason": torso_reason,
            "hip_reason": hip_reason,
            "attempt_id": str(attempt_id) if attempt_id is not None else "",
            "last_attempt_reason": "",
        }

    def test_valid_continuous_cycle_with_phase_localization(self):
        """Continuous counted cycle with known turnaround candidate and return torso change."""
        knee_profile = [140.0, 130.0, 120.0, 110.0, 100.0, 90.0, 100.0, 110.0, 120.0, 130.0, 140.0]
        torso_profile = [10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 13.0, 16.0, 19.0, 22.0, 25.0]
        rows = [
            self._make_row(
                frame_index=10 + i,
                video_time_s=(10 + i) / 30.0,
                knee_deg=knee_profile[i],
                torso_deg=torso_profile[i],
                attempt_id=1,
            )
            for i in range(11)
        ]
        events = [{
            "attempt_id": 1,
            "start_frame": 10,
            "end_frame": 20,
            "track_id": 1,
            "side": "right",
            "status": "counted",
            "reason": "completed_cycle",
            "explanation": "A complete cycle met configured thresholds.",
        }]

        result = build_phase_feedback(rows, events)

        self.assertEqual(result["summary"]["total_attempt_events"], 1)
        self.assertEqual(result["summary"]["counted_events"], 1)
        self.assertEqual(result["summary"]["phase_analyzed_events"], 1)

        att = result["attempts"][0]
        self.assertTrue(att["phase_analysis_available"])
        self.assertIsNone(att["suppression_reason"])

        # Turnaround candidate check (not mathematical inflection)
        min_pt = att["min_knee_point"]
        self.assertEqual(min_pt["frame_index"], 15)
        self.assertAlmostEqual(min_pt["knee_angle_deg"], 90.0, places=2)
        self.assertIn("turnaround candidate", min_pt["interpretation"])
        self.assertNotIn("mathematical inflection", min_pt["interpretation"].replace("not claimed true anatomical bottom or mathematical inflection", ""))

        # Descent & Return phases
        desc = att["descent_phase"]
        ret = att["return_phase"]
        self.assertEqual(desc["phase_start_frame"], 10)
        self.assertEqual(desc["phase_end_frame"], 15)
        self.assertEqual(desc["sample_count"], 6)

        self.assertEqual(ret["phase_start_frame"], 15)
        self.assertEqual(ret["phase_end_frame"], 20)
        self.assertEqual(ret["delta_start_frame"], 15)
        self.assertEqual(ret["delta_end_frame"], 20)
        self.assertAlmostEqual(ret["delta_deg"], 15.0, places=2)

        # Review cues
        cues = att["review_cues"]
        self.assertTrue(any("Observed minimum/turnaround candidate of 90.00° located at frame 15" in c for c in cues))
        self.assertTrue(any("Torso inclination changed by 15.00°" in c and "review measured frames 15–20" in c for c in cues))

    def test_missing_time_honest_frame_evidence(self):
        """Missing timestamps are handled honestly without fabrication or string format crashes."""
        knee_profile = [140.0, 120.0, 100.0, 85.0, 100.0, 120.0, 140.0]
        torso_profile = [10.0, 10.0, 10.0, 10.0, 15.0, 20.0, 25.0]
        rows = [
            self._make_row(
                frame_index=10 + i,
                video_time_s=None,  # No video time provided
                knee_deg=knee_profile[i],
                torso_deg=torso_profile[i],
            )
            for i in range(7)
        ]
        events = [{
            "attempt_id": 1,
            "start_frame": 10,
            "end_frame": 16,
            "track_id": 1,
            "side": "right",
            "status": "counted",
            "reason": "completed_cycle",
        }]

        # Must not raise TypeError: unsupported format string passed to NoneType.__format__
        result = build_phase_feedback(rows, events)
        att = result["attempts"][0]
        self.assertTrue(att["phase_analysis_available"])
        self.assertIsNone(att["min_knee_point"]["video_time_s"])
        self.assertIsNone(att["return_phase"]["delta_start_time_s"])

        # Cue must provide frame evidence without fabricated time
        cues = att["review_cues"]
        self.assertTrue(any("located at frame 13;" in c for c in cues))
        self.assertTrue(any("review measured frames 13–16 across 4 valid frames" in c for c in cues))

    def test_nan_time_rejected(self):
        """NaN video_time_s raises ValueError via upstream validation."""
        rows = [
            self._make_row(10, video_time_s="NaN", knee_deg=140.0),
            self._make_row(11, video_time_s=0.36, knee_deg=120.0),
        ]
        events = [{"attempt_id": 1, "start_frame": 10, "end_frame": 11, "track_id": 1, "side": "right", "status": "counted"}]
        with self.assertRaises(ValueError):
            build_phase_feedback(rows, events)

    def test_negative_time_rejected(self):
        """Negative video_time_s raises ValueError via upstream validation."""
        rows = [
            self._make_row(10, video_time_s="-0.1000", knee_deg=140.0),
            self._make_row(11, video_time_s=0.36, knee_deg=120.0),
        ]
        events = [{"attempt_id": 1, "start_frame": 10, "end_frame": 11, "track_id": 1, "side": "right", "status": "counted"}]
        with self.assertRaises(ValueError):
            build_phase_feedback(rows, events)

    def test_descending_time_rejected(self):
        """Descending video_time_s raises ValueError via upstream validation."""
        rows = [
            self._make_row(10, video_time_s=1.0000, knee_deg=140.0),
            self._make_row(11, video_time_s=0.5000, knee_deg=120.0),
        ]
        events = [{"attempt_id": 1, "start_frame": 10, "end_frame": 11, "track_id": 1, "side": "right", "status": "counted"}]
        with self.assertRaises(ValueError):
            build_phase_feedback(rows, events)

    def test_duplicate_time_during_cycle_suppressed(self):
        """Duplicate timestamp during active attempt suppresses phase analysis."""
        rows = [
            self._make_row(10, video_time_s=0.3333, knee_deg=140.0),
            self._make_row(11, video_time_s=0.3333, knee_deg=120.0),  # duplicate timestamp
            self._make_row(12, video_time_s=0.4000, knee_deg=90.0),
            self._make_row(13, video_time_s=0.4333, knee_deg=140.0),
        ]
        events = [{"attempt_id": 1, "start_frame": 10, "end_frame": 13, "track_id": 1, "side": "right", "status": "counted"}]
        res = build_phase_feedback(rows, events)
        att = res["attempts"][0]
        self.assertFalse(att["phase_analysis_available"])
        self.assertEqual(att["suppression_reason"], "tracking_gap_exceeded")

    def test_unknown_side_cannot_form_comparison_group(self):
        """Non-left/right side suppresses analysis and prevents cross-cycle comparison."""
        # Event 1 and Event 2 with side 'unknown'
        rows = [
            self._make_row(i, round(i / 30.0, 4), knee_deg=140.0 if i not in (12, 22) else 90.0, selected_side="unknown", posture_side="unknown")
            for i in range(10, 25)
        ]
        events = [
            {"attempt_id": 1, "start_frame": 10, "end_frame": 14, "track_id": 1, "side": "unknown", "status": "counted"},
            {"attempt_id": 2, "start_frame": 20, "end_frame": 24, "track_id": 1, "side": "unknown", "status": "counted"},
        ]
        # _validate_events enforces left/right for explicit side
        with self.assertRaises(ValueError):
            build_phase_feedback(rows, events)

    def test_missing_event_identity_suppressed(self):
        """Event with missing side or track is suppressed without fabricating IDs."""
        rows = [self._make_row(i, i / 30.0, knee_deg=140.0 if i != 12 else 90.0) for i in range(10, 15)]
        events = [{
            "start_frame": 10,
            "end_frame": 14,
            "track_id": None,  # Missing track
            "side": None,      # Missing side
            "status": "counted",
        }]
        res = build_phase_feedback(rows, events)
        att = res["attempts"][0]
        self.assertFalse(att["phase_analysis_available"])
        self.assertEqual(att["suppression_reason"], "missing_event_identity")
        self.assertIsNone(att["attempt_id"])  # No fabricated default ID

    def test_selected_side_posture_side_conflict(self):
        """Knee selected_side and torso posture_side conflict suppresses phase analysis."""
        rows = [
            self._make_row(10, 0.33, knee_deg=140.0, selected_side="left", posture_side="right"),  # conflict
            self._make_row(11, 0.36, knee_deg=120.0, selected_side="right", posture_side="right"),
            self._make_row(12, 0.40, knee_deg=90.0, selected_side="right", posture_side="right"),
            self._make_row(13, 0.43, knee_deg=140.0, selected_side="right", posture_side="right"),
        ]
        events = [{"attempt_id": 1, "start_frame": 10, "end_frame": 13, "track_id": 1, "side": "right", "status": "counted"}]
        res = build_phase_feedback(rows, events)
        att = res["attempts"][0]
        self.assertFalse(att["phase_analysis_available"])
        self.assertEqual(att["suppression_reason"], "side_mismatch")

    def test_torso_reason_invalid_with_finite_angle_not_evidence(self):
        """Finite torso angle with non-ok torso_reason is rejected as valid evidence."""
        rows = [
            self._make_row(10, 0.33, knee_deg=140.0, torso_deg=15.0, torso_reason="confidence_too_low"),  # finite but non-ok
            self._make_row(11, 0.36, knee_deg=120.0, torso_deg=15.0, torso_reason="confidence_too_low"),
            self._make_row(12, 0.40, knee_deg=90.0, torso_deg=15.0, torso_reason="ok"),
            self._make_row(13, 0.43, knee_deg=120.0, torso_deg=20.0, torso_reason="ok"),
            self._make_row(14, 0.46, knee_deg=140.0, torso_deg=25.0, torso_reason="ok"),
        ]
        events = [{"attempt_id": 1, "start_frame": 10, "end_frame": 14, "track_id": 1, "side": "right", "status": "counted"}]
        res = build_phase_feedback(rows, events, PhaseFeedbackConfig(min_phase_samples=3))
        att = res["attempts"][0]
        # Descent phase (frames 10..12) only has 1 valid torso (frame 12), failing min_phase_samples 3
        self.assertFalse(att["phase_analysis_available"])
        self.assertEqual(att["suppression_reason"], "insufficient_phase_samples")

    def test_first_last_torso_invalid_correct_endpoint_time(self):
        """Delta endpoints correctly reflect actual measured torso rows rather than phase bounds."""
        # Return phase is frames 12..16
        # Frame 12 has missing_joint torso; frames 13..15 have valid torso (10°, 15°, 20°); frame 16 has missing_joint
        rows = [
            self._make_row(10, 0.33, knee_deg=140.0, torso_deg=10.0),
            self._make_row(11, 0.36, knee_deg=110.0, torso_deg=10.0),
            self._make_row(12, 0.40, knee_deg=85.0, torso_deg=None, torso_reason="missing_joint"),  # min knee
            self._make_row(13, 0.43, knee_deg=100.0, torso_deg=10.0, torso_reason="ok"),
            self._make_row(14, 0.46, knee_deg=120.0, torso_deg=15.0, torso_reason="ok"),
            self._make_row(15, 0.50, knee_deg=130.0, torso_deg=20.0, torso_reason="ok"),
            self._make_row(16, 0.53, knee_deg=140.0, torso_deg=None, torso_reason="missing_joint"),
        ]
        events = [{"attempt_id": 1, "start_frame": 10, "end_frame": 16, "track_id": 1, "side": "right", "status": "counted"}]
        res = build_phase_feedback(rows, events, PhaseFeedbackConfig(min_phase_samples=2, min_coverage_fraction=0.4))
        att = res["attempts"][0]
        self.assertTrue(att["phase_analysis_available"])

        ret = att["return_phase"]
        self.assertEqual(ret["phase_start_frame"], 12)
        self.assertEqual(ret["phase_end_frame"], 16)
        # Measured delta bounds must strictly be 13 to 15!
        self.assertEqual(ret["delta_start_frame"], 13)
        self.assertEqual(ret["delta_end_frame"], 15)
        self.assertAlmostEqual(ret["delta_start_time_s"], 0.43, places=2)
        self.assertAlmostEqual(ret["delta_end_time_s"], 0.50, places=2)
        self.assertAlmostEqual(ret["delta_deg"], 10.0, places=2)  # 20.0 - 10.0

        # Cue text references measured frames 13–15 and phase interval 12–16
        cue = [c for c in att["review_cues"] if "Torso inclination changed" in c][0]
        self.assertIn("review measured frames 13–15 (0.43s–0.50s)", cue)
        self.assertIn("phase interval: frames 12–16", cue)

    def test_single_valid_torso_sample_no_delta(self):
        """Phase with only 1 valid torso sample cannot compute delta (delta_deg is None)."""
        rows = [
            self._make_row(10, 0.33, knee_deg=140.0, torso_deg=10.0),
            self._make_row(11, 0.36, knee_deg=110.0, torso_deg=10.0),
            self._make_row(12, 0.40, knee_deg=85.0, torso_deg=10.0),  # min knee
            self._make_row(13, 0.43, knee_deg=110.0, torso_deg=None, torso_reason="missing_joint"),
            self._make_row(14, 0.46, knee_deg=140.0, torso_deg=None, torso_reason="missing_joint"),
        ]
        events = [{"attempt_id": 1, "start_frame": 10, "end_frame": 14, "track_id": 1, "side": "right", "status": "counted"}]
        # Return phase only has frame 12 with valid torso
        res = build_phase_feedback(rows, events, PhaseFeedbackConfig(min_phase_samples=1, min_coverage_fraction=0.2))
        att = res["attempts"][0]
        ret = att["return_phase"]
        self.assertIsNone(ret["delta_deg"])
        self.assertIsNone(ret["delta_start_frame"])
        # No delta cue asserted in return phase
        self.assertFalse(any("Torso inclination changed" in c for c in att["review_cues"]))

    def test_duplicate_attempt_ids_raise_value_error(self):
        """Duplicate attempt IDs in attempt_events raises ValueError."""
        rows = [self._make_row(i, i / 30.0) for i in range(10, 30)]
        events = [
            {"attempt_id": 1, "start_frame": 10, "end_frame": 15, "track_id": 1, "side": "right", "status": "counted"},
            {"attempt_id": 1, "start_frame": 20, "end_frame": 25, "track_id": 1, "side": "right", "status": "counted"},
        ]
        with self.assertRaises(ValueError):
            build_phase_feedback(rows, events)

    def test_malformed_status_raises_value_error(self):
        """Malformed status string raises ValueError."""
        rows = [self._make_row(i, i / 30.0) for i in range(10, 20)]
        events = [{"attempt_id": 1, "start_frame": 10, "end_frame": 15, "track_id": 1, "side": "right", "status": "partial"}]
        with self.assertRaises(ValueError):
            build_phase_feedback(rows, events)

    def test_zero_degrees_is_valid_not_missing(self):
        """0.0 degree angle measurement is valid and produces correct stats."""
        knee_profile = [140.0, 120.0, 100.0, 80.0, 100.0, 120.0, 140.0]
        rows = [self._make_row(10 + i, (10 + i) / 30.0, knee_deg=knee_profile[i], torso_deg=0.0) for i in range(7)]
        events = [{"attempt_id": 1, "start_frame": 10, "end_frame": 16, "track_id": 1, "side": "right", "status": "counted"}]
        result = build_phase_feedback(rows, events)
        att = result["attempts"][0]
        self.assertTrue(att["phase_analysis_available"])
        self.assertEqual(att["descent_phase"]["median_deg"], 0.0)
        self.assertEqual(att["return_phase"]["delta_deg"], 0.0)

    def test_frame_gap_suppresses_regardless_of_missing_time(self):
        """Missing frame index in sequence suppresses analysis even when video_time_s is absent."""
        rows = [
            self._make_row(10, knee_deg=140.0),
            self._make_row(11, knee_deg=120.0),
            # frame 12 missing
            self._make_row(13, knee_deg=90.0),
            self._make_row(14, knee_deg=140.0),
        ]
        events = [{"attempt_id": 1, "start_frame": 10, "end_frame": 14, "track_id": 1, "side": "right", "status": "counted"}]
        res = build_phase_feedback(rows, events)
        self.assertFalse(res["attempts"][0]["phase_analysis_available"])
        self.assertEqual(res["attempts"][0]["suppression_reason"], "frame_gap_detected")

    def test_uncounted_setup_and_excluded_comparisons(self):
        """Uncounted attempts are excluded from consistency and retain intent uncertainty."""
        rows = [
            self._make_row(10, 0.33, knee_deg=145.0, attempt_id=1),
            self._make_row(11, 0.36, knee_deg=135.0, attempt_id=1),
            self._make_row(12, 0.40, knee_deg=145.0, attempt_id=1),
            self._make_row(20, 0.66, knee_deg=140.0, attempt_id=2),
            self._make_row(21, 0.70, knee_deg=100.0, attempt_id=2),
            self._make_row(22, 0.73, knee_deg=85.0, attempt_id=2),
            self._make_row(23, 0.76, knee_deg=100.0, attempt_id=2),
            self._make_row(24, 0.80, knee_deg=140.0, attempt_id=2),
        ]
        events = [
            {"attempt_id": 1, "start_frame": 10, "end_frame": 12, "track_id": 1, "side": "right", "status": "uncounted", "reason": "depth_threshold_not_reached"},
            {"attempt_id": 2, "start_frame": 20, "end_frame": 24, "track_id": 1, "side": "right", "status": "counted", "reason": "completed_cycle"},
        ]
        res = build_phase_feedback(rows, events, PhaseFeedbackConfig(min_phase_samples=2))
        att1 = res["attempts"][0]
        self.assertFalse(att1["phase_analysis_available"])
        self.assertEqual(att1["suppression_reason"], "uncounted_attempt")
        self.assertIn("cannot determine exercise intent", att1["context_note"])
        # Only 1 counted cycle; cross-cycle comparison is not performed
        self.assertEqual(len(res["summary"]["comparisons"]), 0)

    def test_immutability_of_inputs(self):
        """Input frame rows and attempt events are not modified."""
        rows = [self._make_row(10 + i, (10 + i) / 30.0, knee_deg=140.0 if i != 2 else 90.0) for i in range(5)]
        events = [{"attempt_id": 1, "start_frame": 10, "end_frame": 14, "track_id": 1, "side": "right", "status": "counted"}]
        rows_copy = copy.deepcopy(rows)
        events_copy = copy.deepcopy(events)
        _ = build_phase_feedback(rows, events)
        self.assertEqual(rows, rows_copy)
        self.assertEqual(events, events_copy)

    def test_real_csv_schema_fixture_derived_from_pipeline(self):
        """Process realistic frame data matching pipeline.py output schema without hardcoded paths."""
        rows = []
        for f in range(20, 31):
            t = f / 30.0
            k_deg = 145.0 - (10 - abs(f - 25)) * 5.0
            torso_deg = 12.0 + (f - 20) * 1.5
            rows.append({
                "frame_index": f,
                "video_time_s": f"{t:.4f}",
                "track_id": "1",
                "selected_side": "left",
                "knee_angle_deg": f"{k_deg:.2f}",
                "confidence": "0.9123",
                "valid": "True",
                "reason": "ok",
                "stage": "lowered" if k_deg <= 110.0 else "standing",
                "completed_reps": "1",
                "posture_side": "left",
                "torso_inclination_deg": f"{torso_deg:.2f}",
                "hip_angle_deg": "115.50",
                "torso_reason": "ok",
                "hip_reason": "ok",
                "attempt_id": "1",
                "last_attempt_reason": "",
            })
        events = [{"attempt_id": 1, "start_frame": 20, "end_frame": 30, "track_id": 1, "side": "left", "status": "counted", "reason": "completed_cycle"}]
        res = build_phase_feedback(rows, events)
        self.assertEqual(res["summary"]["phase_analyzed_events"], 1)
        att = res["attempts"][0]
        self.assertEqual(att["min_knee_point"]["frame_index"], 25)
        self.assertAlmostEqual(att["min_knee_point"]["knee_angle_deg"], 95.0, places=2)


if __name__ == "__main__":
    unittest.main()


class TestPhaseReviewRegressions(unittest.TestCase):
    def fixture(self):
        make = TestPhaseFeedback()._make_row
        angles = [160, 130, 100, 90, 100, 120, 140, 155, 160]
        rows = [make(i, video_time_s=i/30, knee_deg=angle) for i, angle in enumerate(angles)]
        event = {"attempt_id": 1, "track_id": 1, "side": "right", "start_frame": 0,
                 "end_frame": 8, "status": "counted", "reason": "completed_cycle"}
        return rows, [event]

    def test_missing_torso_availability_reason_suppresses_evidence(self):
        rows, events = self.fixture()
        for row in rows:row["torso_reason"] = ""
        result = build_phase_feedback(rows, events)
        self.assertFalse(result["attempts"][0]["phase_analysis_available"])

    def test_out_of_range_angles_never_produce_phase_cues(self):
        for field, value in [("knee_angle_deg", -1), ("torso_inclination_deg", 181)]:
            with self.subTest(field=field):
                rows, events = self.fixture()
                for row in rows:row[field] = value
                result = build_phase_feedback(rows, events)
                self.assertFalse(result["attempts"][0]["phase_analysis_available"])
                self.assertEqual(result["attempts"][0]["review_cues"], [])
