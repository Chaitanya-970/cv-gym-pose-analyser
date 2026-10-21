# -*- coding: utf-8 -*-
"""Evidence-linked posture critique engine for exercise repetition analysis.

Re-exports core critique components from gym_analyser.coaching_feedback conforming
to POSTURE_CRITIQUE_CONTRACT_59.md and FEEDBACK_CONTRACT_38.md.
"""

try:
    from gym_analyser.coaching_feedback import (
        RepetitionEvidence,
        check_repetition_comparability,
        get_comparable_repetitions,
        build_depth_observation,
        build_pacing_observation,
        build_torso_observation,
        build_evidence_linked_suggestions,
        MIN_COMPARABLE_COVERAGE_PCT,
        DEPTH_CONSISTENCY_VARIANCE_DEG,
        RAPID_DESCENT_THRESHOLD_S,
        ASYMMETRIC_PACING_RATIO,
        MARKED_TEMPO_DRIFT_THRESHOLD_S,
        NOTICEABLE_TORSO_DEVIATION_DEG,
        determine_unavailability_reason,
        parse_float,
        parse_int,
        CLINICAL_DISCLAIMER,
        HUMAN_VALIDATION_STATUS,
    )
except (ImportError, ValueError):
    from coaching_feedback import (
        RepetitionEvidence,
        check_repetition_comparability,
        get_comparable_repetitions,
        build_depth_observation,
        build_pacing_observation,
        build_torso_observation,
        build_evidence_linked_suggestions,
        MIN_COMPARABLE_COVERAGE_PCT,
        DEPTH_CONSISTENCY_VARIANCE_DEG,
        RAPID_DESCENT_THRESHOLD_S,
        ASYMMETRIC_PACING_RATIO,
        MARKED_TEMPO_DRIFT_THRESHOLD_S,
        NOTICEABLE_TORSO_DEVIATION_DEG,
        determine_unavailability_reason,
        parse_float,
        parse_int,
        CLINICAL_DISCLAIMER,
        HUMAN_VALIDATION_STATUS,
    )

__all__ = [
    "RepetitionEvidence",
    "check_repetition_comparability",
    "get_comparable_repetitions",
    "build_depth_observation",
    "build_pacing_observation",
    "build_torso_observation",
    "build_evidence_linked_suggestions",
    "MIN_COMPARABLE_COVERAGE_PCT",
    "DEPTH_CONSISTENCY_VARIANCE_DEG",
    "RAPID_DESCENT_THRESHOLD_S",
    "ASYMMETRIC_PACING_RATIO",
    "MARKED_TEMPO_DRIFT_THRESHOLD_S",
    "NOTICEABLE_TORSO_DEVIATION_DEG",
    "determine_unavailability_reason",
    "parse_float",
    "parse_int",
    "CLINICAL_DISCLAIMER",
    "HUMAN_VALIDATION_STATUS",
]
