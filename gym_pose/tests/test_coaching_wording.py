# -*- coding: utf-8 -*-
"""Unit tests for coaching feedback wording and camera-view adequacy honesty (Task 56).

Verifies:
1. Replacement of unsupported camera orientation assertions ("Camera maintained a
   consistent {single_side} lateral view") with accurate selected-leg tracking
   descriptions ("Analysis consistently tracked the {single_side} leg throughout
   the monitored movements.").
2. Replacement of repetition tracking message ("on {side_label} lateral view")
   with honest tracking-side wording ("tracking the {side_label} leg").
3. Correct behavior across right, left, switching, and missing side conditions.
4. Correct behavior across low-coverage, sparse, and interrupted conditions.
5. Strict preservation of measurement unknowns: camera_view_adequate is None (null)
   when tracking cannot measure 3D camera geometry, and NEVER asserts True.
6. Absolute absence of unsupported "lateral view" assertions in generated notes
   and per-rep tracking observations.
"""

import copy
import os
import sys
import unittest
from typing import Any, Dict, List, Optional

# Ensure gym_pose directory is in sys.path
_current_dir = os.path.dirname(os.path.abspath(__file__))
_gym_pose_dir = os.path.dirname(_current_dir)
if _gym_pose_dir not in sys.path:
    sys.path.insert(0, _gym_pose_dir)

from gym_analyser.coaching_feedback import generate_coaching_feedback


class TestCoachingWording(unittest.TestCase):
    """Test suite for verified non-inferred wording in coaching_feedback.py."""

    def _build_base_fixture(
        self,
        side: Optional[str] = "right",
        track_id: int = 1,
        rep_coverage: float = 0.90,
        completed_reps: int = 1,
        status: str = "completed",
    ) -> Dict[str, Any]:
        """Construct a minimal synthetic pipeline summary and attempt fixture."""
        valid_frames = int(30 * rep_coverage)
        observed_frames = 30
        summary = {
            "status": status,
            "completed_reps": completed_reps,
            "processed_frames": 100,
            "valid_observations": 90,
            "valid_observation_fraction": 0.90,
            "configuration": {
                "down_angle": 110.0,
                "up_angle": 150.0,
                "min_joint_confidence": 0.3,
            },
            "posture_observations": {
                "torso_valid_frames": 90,
                "processed_frames": 100,
            },
        }

        events: List[Dict[str, Any]] = []
        for i in range(1, completed_reps + 1):
            event: Dict[str, Any] = {
                "attempt_id": i,
                "status": "counted",
                "reason": "completed_cycle",
                "explanation": "A complete cycle met the configured counting thresholds.",
                "track_id": track_id,
                "start_frame": (i - 1) * 35 + 5,
                "end_frame": (i - 1) * 35 + 35,
                "start_time_s": round((i - 1) * 1.2 + 0.2, 2),
                "end_time_s": round((i - 1) * 1.2 + 1.2, 2),
                "min_knee_angle_deg": 90.0,
                "valid_frames": valid_frames,
                "observed_frames": observed_frames,
                "valid_frame_fraction": rep_coverage,
            }
            if side is not None:
                event["side"] = side
            events.append(event)

        attempts = {
            "counted_attempts": completed_reps,
            "uncounted_attempts": 0,
            "events": events,
        }

        return {"summary": summary, "attempts": attempts}

    def test_right_side_tracked_wording(self) -> None:
        """Right-side tracking generates honest tracked-leg wording and preserves null camera adequacy."""
        fixture = self._build_base_fixture(side="right", rep_coverage=0.90)
        feedback = generate_coaching_feedback(
            summary=fixture["summary"],
            attempts=fixture["attempts"],
        )

        recording_quality = feedback["overall_summary"]["recording_quality"]
        notes = recording_quality["notes"]

        # 1. Assert replacement wording is generated for single right side
        self.assertIn(
            "Analysis consistently tracked the right leg throughout the monitored movements.",
            notes,
        )

        # 2. Assert old unsupported camera orientation claim is absent
        for note in notes:
            self.assertNotIn("Camera maintained a consistent", note)
            self.assertNotIn("lateral view of the exerciser", note)

        # 3. Assert camera_view_adequate is None (unverified 2D monocular tracking)
        self.assertIsNone(
            recording_quality["camera_view_adequate"],
            "camera_view_adequate must be None (null) when 3D geometry is unverified.",
        )
        self.assertIn(
            "Camera view adequacy and 3D orthogonality cannot be verified from 2D monocular tracking alone.",
            notes,
        )

        # 4. Assert rep cue OBS-TRACKING-01 uses tracked-leg wording without 'lateral view'
        cues = feedback["per_repetition_cues"]
        self.assertEqual(len(cues), 1)
        tracking_obs = [o for o in cues[0]["observations"] if o["rule_id"] == "OBS-TRACKING-01"]
        self.assertEqual(len(tracking_obs), 1)
        msg = tracking_obs[0]["message"]
        self.assertEqual(
            msg,
            "Tracking continuity maintained across 90.0% of frames tracking the right leg.",
        )
        self.assertNotIn("lateral view", msg)

    def test_left_side_tracked_wording(self) -> None:
        """Left-side tracking generates honest tracked-leg wording and preserves null camera adequacy."""
        fixture = self._build_base_fixture(side="left", rep_coverage=0.85)
        feedback = generate_coaching_feedback(
            summary=fixture["summary"],
            attempts=fixture["attempts"],
        )

        recording_quality = feedback["overall_summary"]["recording_quality"]
        notes = recording_quality["notes"]

        # 1. Assert replacement wording is generated for single left side
        self.assertIn(
            "Analysis consistently tracked the left leg throughout the monitored movements.",
            notes,
        )

        # 2. Assert old unsupported camera orientation claim is absent
        for note in notes:
            self.assertNotIn("Camera maintained a consistent", note)
            self.assertNotIn("lateral view of the exerciser", note)

        # 3. Assert camera_view_adequate is None
        self.assertIsNone(recording_quality["camera_view_adequate"])
        self.assertIn(
            "Camera view adequacy and 3D orthogonality cannot be verified from 2D monocular tracking alone.",
            notes,
        )

        # 4. Assert rep cue OBS-TRACKING-01 uses tracked-leg wording
        cues = feedback["per_repetition_cues"]
        tracking_obs = [o for o in cues[0]["observations"] if o["rule_id"] == "OBS-TRACKING-01"]
        self.assertEqual(len(tracking_obs), 1)
        msg = tracking_obs[0]["message"]
        self.assertEqual(
            msg,
            "Tracking continuity maintained across 85.0% of frames tracking the left leg.",
        )
        self.assertNotIn("lateral view", msg)

    def test_switching_side_condition(self) -> None:
        """Side switching between attempts flags inconsistent view and sets camera_view_adequate=False."""
        fixture = self._build_base_fixture(completed_reps=2, rep_coverage=0.90)
        # Rep 1 is right, Rep 2 is left
        fixture["attempts"]["events"][0]["side"] = "right"
        fixture["attempts"]["events"][1]["side"] = "left"

        feedback = generate_coaching_feedback(
            summary=fixture["summary"],
            attempts=fixture["attempts"],
        )

        recording_quality = feedback["overall_summary"]["recording_quality"]
        notes = recording_quality["notes"]

        # 1. Neither single-side tracking note nor single-side camera view note should be emitted
        self.assertNotIn(
            "Analysis consistently tracked the right leg throughout the monitored movements.",
            notes,
        )
        self.assertNotIn(
            "Analysis consistently tracked the left leg throughout the monitored movements.",
            notes,
        )
        for note in notes:
            self.assertNotIn("Camera maintained a consistent", note)
            self.assertNotIn("lateral view", note)

        # 2. Multi-context warning emitted
        self.assertIn(
            "Observed movements occurred across multiple tracking contexts or body sides; "
            "cross-attempt consistency is not evaluated across different tracks or sides.",
            notes,
        )

        # 3. camera_view_adequate MUST be False due to view inconsistency
        self.assertIs(recording_quality["camera_view_adequate"], False)
        self.assertIn(
            "Camera view was suboptimal due to multiple tracking contexts or body side changes.",
            notes,
        )

        # 4. Per-rep cues reflect individual tracked legs without 'lateral view'
        cues = feedback["per_repetition_cues"]
        self.assertEqual(len(cues), 2)
        obs_r1 = [o for o in cues[0]["observations"] if o["rule_id"] == "OBS-TRACKING-01"][0]
        obs_r2 = [o for o in cues[1]["observations"] if o["rule_id"] == "OBS-TRACKING-01"][0]

        self.assertEqual(
            obs_r1["message"],
            "Tracking continuity maintained across 90.0% of frames tracking the right leg.",
        )
        self.assertEqual(
            obs_r2["message"],
            "Tracking continuity maintained across 90.0% of frames tracking the left leg.",
        )
        self.assertNotIn("lateral view", obs_r1["message"])
        self.assertNotIn("lateral view", obs_r2["message"])

    def test_side_changed_reason_in_attempt(self) -> None:
        """Mid-attempt side change event flags view inconsistency and suppresses single-side wording."""
        fixture = self._build_base_fixture(completed_reps=1, rep_coverage=0.80)
        # Add an uncounted attempt with reason='side_changed'
        fixture["attempts"]["events"].append({
            "attempt_id": 2,
            "status": "uncounted",
            "reason": "side_changed",
            "explanation": "The observed body side changed before this cycle could be confirmed.",
            "track_id": 1,
            "side": "left",
            "start_time_s": 1.5,
            "end_time_s": 2.0,
            "min_knee_angle_deg": 120.0,
            "valid_frames": 10,
            "observed_frames": 15,
        })
        fixture["attempts"]["uncounted_attempts"] = 1

        feedback = generate_coaching_feedback(
            summary=fixture["summary"],
            attempts=fixture["attempts"],
        )

        recording_quality = feedback["overall_summary"]["recording_quality"]
        notes = recording_quality["notes"]

        self.assertIs(recording_quality["camera_view_adequate"], False)
        self.assertIn(
            "Camera view was suboptimal due to multiple tracking contexts or body side changes.",
            notes,
        )
        for note in notes:
            self.assertNotIn("Camera maintained a consistent", note)
            self.assertNotIn("lateral view", note)

    def test_missing_side_condition(self) -> None:
        """Missing or unknown side label uses 'monitored leg' fallback without lateral view claim."""
        fixture = self._build_base_fixture(side=None, rep_coverage=0.92)
        feedback = generate_coaching_feedback(
            summary=fixture["summary"],
            attempts=fixture["attempts"],
        )

        recording_quality = feedback["overall_summary"]["recording_quality"]
        notes = recording_quality["notes"]

        # No single-side tracked note emitted when side is unknown
        for note in notes:
            self.assertNotIn("consistently tracked", note)
            self.assertNotIn("Camera maintained a consistent", note)
            self.assertNotIn("lateral view", note)

        # Rep cue uses 'monitored leg' fallback
        cues = feedback["per_repetition_cues"]
        obs = [o for o in cues[0]["observations"] if o["rule_id"] == "OBS-TRACKING-01"][0]
        self.assertEqual(
            obs["message"],
            "Tracking continuity maintained across 92.0% of frames tracking the monitored leg.",
        )
        self.assertNotIn("lateral view", obs["message"])

        # camera_view_adequate remains None because tracking continuity is adequate
        self.assertIsNone(recording_quality["camera_view_adequate"])

    def test_low_coverage_and_interrupted_tracking(self) -> None:
        """Low coverage (< 70%) generates tracking interruption message and setup suggestion."""
        fixture = self._build_base_fixture(side="right", rep_coverage=0.45)
        # Also simulate low session observation fraction
        fixture["summary"]["valid_observation_fraction"] = 0.45
        feedback = generate_coaching_feedback(
            summary=fixture["summary"],
            attempts=fixture["attempts"],
        )

        recording_quality = feedback["overall_summary"]["recording_quality"]
        notes = recording_quality["notes"]

        # camera_view_adequate is False because continuity is interrupted
        self.assertIs(recording_quality["camera_view_adequate"], False)
        self.assertIn(
            "Camera view was suboptimal due to interrupted tracking continuity.",
            notes,
        )

        # Cues report interruption, line-of-sight limits, and setup suggestion
        cues = feedback["per_repetition_cues"]
        obs = [o for o in cues[0]["observations"] if o["rule_id"] == "OBS-TRACKING-01"][0]
        self.assertEqual(
            obs["message"],
            "Tracking interruptions observed during repetition (45.0% valid frames); 2D camera line-of-sight limits.",
        )
        self.assertNotIn("lateral view", obs["message"])

        sugs = cues[0]["suggestions"]
        setup_sugs = [s for s in sugs if s["rule_id"] == "SUG-REC-01"]
        self.assertEqual(len(setup_sugs), 1)
        self.assertIn("Mount camera stationary at hip height orthogonal", setup_sugs[0]["message"])

    def test_sparse_coverage_condition(self) -> None:
        """Missing valid frame counts produces 'sparse valid frames' message."""
        fixture = self._build_base_fixture(side="left", rep_coverage=0.0)
        # Clear frame counts
        event = fixture["attempts"]["events"][0]
        event.pop("valid_frames", None)
        event.pop("valid_frame_fraction", None)
        event["observed_frames"] = 0

        feedback = generate_coaching_feedback(
            summary=fixture["summary"],
            attempts=fixture["attempts"],
        )

        cues = feedback["per_repetition_cues"]
        obs = [o for o in cues[0]["observations"] if o["rule_id"] == "OBS-TRACKING-01"][0]
        self.assertEqual(
            obs["message"],
            "Tracking interruptions observed during repetition (sparse valid frames); 2D camera line-of-sight limits.",
        )
        self.assertNotIn("lateral view", obs["message"])

    def test_camera_view_adequate_never_true(self) -> None:
        """camera_view_adequate is NEVER True regardless of perfect 2D telemetry."""
        # 100% coverage, pristine single track
        fixture = self._build_base_fixture(side="right", rep_coverage=1.0)
        feedback = generate_coaching_feedback(
            summary=fixture["summary"],
            attempts=fixture["attempts"],
        )
        adequacy = feedback["overall_summary"]["recording_quality"]["camera_view_adequate"]
        self.assertIsNot(
            adequacy,
            True,
            "2D monocular tracking cannot verify 3D geometry; camera_view_adequate must never be True.",
        )
        self.assertIsNone(adequacy)

    def test_fallback_counted_reps_without_attempts(self) -> None:
        """Summary reps without attempts payload fallback cleanly without lateral view wording."""
        summary = {
            "status": "completed",
            "completed_reps": 2,
            "processed_frames": 60,
            "valid_observations": 50,
            "valid_observation_fraction": 0.83,
        }
        feedback = generate_coaching_feedback(summary=summary, attempts=None)

        self.assertEqual(feedback["overall_summary"]["total_counted_reps"], 2)
        cues = feedback["per_repetition_cues"]
        self.assertEqual(len(cues), 2)
        for cue in cues:
            obs = [o for o in cue["observations"] if o["rule_id"] == "OBS-TRACKING-01"][0]
            self.assertEqual(
                obs["message"],
                "Repetition counted by pipeline threshold transitions; detailed kinematic telemetry was not supplied.",
            )
            self.assertNotIn("lateral view", obs["message"])

    def test_absence_of_unsupported_lateral_view_across_all_conditions(self) -> None:
        """Iterative check across diverse conditions to confirm zero unsupported lateral claims."""
        test_cases = [
            ("right", 0.95, 2),
            ("left", 0.80, 1),
            ("unknown", 0.75, 1),
            (None, 0.40, 1),
        ]

        for side, cov, reps in test_cases:
            with self.subTest(side=side, coverage=cov, reps=reps):
                fixture = self._build_base_fixture(
                    side=side if side != "unknown" else "unknown",
                    rep_coverage=cov,
                    completed_reps=reps,
                )
                if side == "unknown":
                    fixture["attempts"]["events"][0]["side"] = "unknown"

                feedback = generate_coaching_feedback(
                    summary=fixture["summary"],
                    attempts=fixture["attempts"],
                )

                # Check notes
                for note in feedback["overall_summary"]["recording_quality"]["notes"]:
                    self.assertNotIn(
                        "camera maintained a consistent",
                        note.lower(),
                        f"Found unsupported camera claim in note: {note}",
                    )
                    self.assertNotIn(
                        "lateral view of the exerciser",
                        note.lower(),
                        f"Found unsupported lateral view claim in note: {note}",
                    )

                # Check per-rep cues
                for cue in feedback["per_repetition_cues"]:
                    for obs in cue["observations"]:
                        if obs["rule_id"] == "OBS-TRACKING-01":
                            self.assertNotIn(
                                "lateral view",
                                obs["message"].lower(),
                                f"Found 'lateral view' in OBS-TRACKING-01: {obs['message']}",
                            )


if __name__ == "__main__":
    unittest.main()
