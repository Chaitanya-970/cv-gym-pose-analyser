"""Regression checks for angle convention and honest source scope."""
import unittest
from gym_analyser.angle_policy import internal_to_flexion, describe_angle_policy
from gym_analyser.__main__ import build_parser
from gym_analyser.counter import CounterConfig

class AnglePolicyTests(unittest.TestCase):
    def test_clinical_and_internal_conventions_are_not_confused(self):
        self.assertEqual(internal_to_flexion(180), 0)
        self.assertEqual(internal_to_flexion(110), 70)
        self.assertEqual(internal_to_flexion(60), 120)
        for bad in (True, None, float("nan"), float("inf"), -1, 181):
            with self.subTest(value=bad), self.assertRaises(ValueError):
                internal_to_flexion(bad)

    def test_citation_does_not_certify_full_depth_or_custom_gates(self):
        legacy = describe_angle_policy(110, 150)
        self.assertTrue(legacy["depth_gate"]["related_work"])
        self.assertFalse(legacy["depth_gate"]["full_squat_definition_validated"])
        self.assertFalse(legacy["standing_gate"]["clinical_extension_test"])
        self.assertEqual(legacy["full_depth_accuracy"], "not_established")
        custom = describe_angle_policy(120, 160)
        self.assertEqual(custom["depth_gate"]["related_work"], [])
        self.assertEqual(custom["depth_gate"]["equivalent_flexion_degrees"], 60)

    def test_cli_compatibility_gates_and_custom_angles(self):
        args = build_parser().parse_args(["--source", "x.mp4", "--output-dir", "out"])
        cfg = CounterConfig()
        self.assertEqual((args.down_angle, args.up_angle), (cfg.down_angle, cfg.up_angle))
        custom = build_parser().parse_args(["--source", "x.mp4", "--output-dir", "out", "--down-angle", "90", "--up-angle", "170"])
        self.assertEqual((custom.down_angle, custom.up_angle), (90, 170))
