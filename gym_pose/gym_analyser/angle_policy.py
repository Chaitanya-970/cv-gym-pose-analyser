"""Angle provenance: research targets and software gates are different concepts.

See README angle definitions. No listed research protocol validates our 2D estimator.
"""
import math
from numbers import Real

# Compatibility settings, not research definitions of a full squat.
LEGACY_DOWN_ANGLE_DEG = 110.0
STANDING_GATE_DEG = 150.0
DEPTH_REVIEW_DIFFERENCE_DEG = 5.0
TORSO_REVIEW_DIFFERENCE_DEG = 3.0
TURNAROUND_REVIEW_DIFFERENCE_DEG = 2.0

RESEARCH_SOURCES = {
    "mois2024": "https://essay.utwente.nl/100988/",
    "healthmate2026": "https://doi.org/10.22214/ijraset.2026.80172",
    "bryanton2013": "https://pmc.ncbi.nlm.nih.gov/articles/PMC4064719/",
    "bloomquist2013": "https://doi.org/10.1007/s00421-013-2642-7",
    "kubo2019": "https://doi.org/10.1007/s00421-019-04181-y",
    "lorenzetti2018": "https://doi.org/10.1186/s13102-018-0103-7",
    "gwynne2014": "https://pmc.ncbi.nlm.nih.gov/articles/PMC4275194/",
}


def internal_to_flexion(angle):
    """Convert conventions only; does not correct 2D projection/model errors."""
    if isinstance(angle, bool) or not isinstance(angle, Real) or not math.isfinite(angle) or not 0 <= angle <= 180:
        raise ValueError("Expected finite internal angle from 0 to 180 degrees")
    return 180.0 - float(angle)


def describe_angle_policy(down_angle, up_angle):
    """Describe the actual run settings without upgrading them to scientific truth."""
    return {
        "measurement": "projected_2d_hip_knee_ankle_internal_angle",
        "straight_leg_degrees": 180.0,
        "clinical_flexion_conversion": "180 - internal angle; convention conversion only",
        "depth_gate": {
            "internal_degrees": down_angle,
            "equivalent_flexion_degrees": internal_to_flexion(down_angle),
            "provenance": "legacy_engineering_gate" if down_angle == LEGACY_DOWN_ANGLE_DEG else "configured_engineering_gate",
            "full_squat_definition_validated": False,
            "related_work": [RESEARCH_SOURCES["mois2024"], RESEARCH_SOURCES["healthmate2026"]] if down_angle == LEGACY_DOWN_ANGLE_DEG else [],
            "citation_scope": "similar software thresholds; not validation of this model or full-depth definition",
        },
        "standing_gate": {
            "internal_degrees": up_angle,
            "purpose": "return-to-standing state hysteresis",
            "clinical_extension_test": False,
        },
        "research_baseline": {
            "target": "parallel_or_below_reference_only",
            "adoption": "not_adopted; user retains current engineering counter",
            "criterion": "hip/inguinal-crease depth relative to knee; not a universal knee-angle cutoff",
            "source": RESEARCH_SOURCES["bryanton2013"],
            "automatic_implementation_validated": False,
            "pose_centres_equal_anatomical_crease": False,
        },
        "published_deep_protocols": [
            {"source": RESEARCH_SOURCES["bloomquist2013"], "knee_flexion_degrees": 120.0, "internal_degrees": 60.0},
            {"source": RESEARCH_SOURCES["kubo2019"], "knee_flexion_degrees": 140.0, "internal_degrees": 40.0},
        ],
        "review_cues": {
            "depth_difference_degrees": DEPTH_REVIEW_DIFFERENCE_DEG,
            "torso_difference_degrees": TORSO_REVIEW_DIFFERENCE_DEG,
            "turnaround_difference_degrees": TURNAROUND_REVIEW_DIFFERENCE_DEG,
            "provenance": "engineering_attention_filters_not_validated_fault_or_error_bounds",
        },
        "full_depth_accuracy": "not_established",
    }
