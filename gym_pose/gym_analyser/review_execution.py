"""Human-reviewed video evaluation execution engine for Gym Pose Analyser (Phase 2).

Pure Python standard-library module and CLI for executing matched video evaluations
from pre-generated Phase 1 evaluation plans, enforcing strict process isolation,
finite per-clip timeouts with process-tree cleanup, multi-model common completed
cohort denominator equality, fail-closed hash and symlink verification against
source truth, immutable frozen plan verification, and atomic staged receipt publication.
"""

from __future__ import annotations

import argparse
import copy
import datetime
import hashlib
import inspect
import json
import logging
import math
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Set, Tuple, Union

from .counter import CounterConfig
from .evaluation import evaluate_counts, load_ground_truth, validate_count
from .review_evaluation import (
    CHUNK_SIZE,
    SAFE_MODEL_NAME_PATTERN,
    compute_file_sha256,
    reject_linked_ancestors,
    validate_model_name,
    validate_models_contract,
    validate_path_overlap,
)
from .review_store import (
    CaseCorruptedError,
    ReviewStore,
    SHA256_HEX_PATTERN,
    _reject_json_constant,
    detect_duplicate_pairs,
    is_symlink,
    is_valid_uuid4,
    load_persisted_case,
)

logger = logging.getLogger(__name__)

SAFE_DEVICE_PATTERN: re.Pattern[str] = re.compile(r"^[a-zA-Z0-9_\:\,\.\-]+$")
DEFAULT_TIMEOUT_PER_CLIP: float = 180.0


# ==============================================================================
# Custom Exceptions
# ==============================================================================


class PlanOrCodeMutationError(RuntimeError, ValueError):
    """Raised when evaluation plan, model, dataset, or source code mutates during execution."""


# ==============================================================================
# Pure Stdlib Provenance and Environment Helpers
# ==============================================================================


def capture_package_code_hashes(gym_pose_dir: Path) -> Dict[str, str]:
    """Capture SHA256 hashes of all Python source files in gym_analyser package."""
    pkg_dir = gym_pose_dir / "gym_analyser"
    if not pkg_dir.is_dir():
        return {}
    code_hashes: Dict[str, str] = {}
    for p in sorted(pkg_dir.glob("*.py")):
        if p.is_file() and not is_symlink(p):
            code_hashes[p.name] = compute_file_sha256(p)
    return code_hashes


def normalize_declared_configuration(
    declared_configuration: Optional[Mapping[str, Any]],
) -> Dict[str, Any]:
    """Normalize and strictly validate declared configuration settings.

    Zero values are valid configurations (e.g. min_joint_confidence = 0.0), NOT absence.
    Rejects unsupported parameters or invalid values.
    """
    if not declared_configuration:
        return {
            "down_angle": 110.0,
            "up_angle": 150.0,
            "min_confidence": 0.3,
            "stable_frames": 3,
            "departure_frames": 2,
            "max_gap": 0.5,
            "tracker": "bytetrack.yaml",
        }

    if not isinstance(declared_configuration, (dict, Mapping)):
        raise TypeError(
            f"declared_configuration must be a dictionary, got {type(declared_configuration).__name__}"
        )

    allowed_top_keys = {
        "counter_config",
        "down_angle",
        "up_angle",
        "min_joint_confidence",
        "min_confidence",
        "stable_frames",
        "departure_frames",
        "max_gap_seconds",
        "max_gap",
        "tracker",
        "tracker_name",
        "runtime_inference_effective_parameters",
    }
    for k in declared_configuration:
        if k not in allowed_top_keys:
            raise ValueError(f"Unsupported configuration key in declared_configuration: {k!r}")

    c_cfg = declared_configuration.get("counter_config")
    if c_cfg is not None:
        if not isinstance(c_cfg, (dict, Mapping)):
            raise TypeError(f"counter_config must be a dictionary, got {type(c_cfg).__name__}")
        allowed_counter_keys = {
            "down_angle",
            "up_angle",
            "min_joint_confidence",
            "min_confidence",
            "stable_frames",
            "max_gap_seconds",
            "max_gap",
        }
        for k in c_cfg:
            if k not in allowed_counter_keys:
                raise ValueError(f"Unsupported configuration key in counter_config: {k!r}")
    else:
        c_cfg = {}

    down = c_cfg.get("down_angle")
    if down is None:
        down = declared_configuration.get("down_angle")
    if down is None:
        down = 110.0

    up = c_cfg.get("up_angle")
    if up is None:
        up = declared_configuration.get("up_angle")
    if up is None:
        up = 150.0

    conf = c_cfg.get("min_joint_confidence")
    if conf is None:
        conf = c_cfg.get("min_confidence")
    if conf is None:
        conf = declared_configuration.get("min_joint_confidence")
    if conf is None:
        conf = declared_configuration.get("min_confidence")
    if conf is None:
        conf = 0.3

    stable = c_cfg.get("stable_frames")
    if stable is None:
        stable = declared_configuration.get("stable_frames")
    if stable is None:
        stable = 3

    dep = declared_configuration.get("departure_frames")
    if dep is None:
        dep = 2

    gap = c_cfg.get("max_gap_seconds")
    if gap is None:
        gap = c_cfg.get("max_gap")
    if gap is None:
        gap = declared_configuration.get("max_gap_seconds")
    if gap is None:
        gap = declared_configuration.get("max_gap")
    if gap is None:
        gap = 0.5

    trk = declared_configuration.get("tracker_name")
    if trk is None:
        trk = declared_configuration.get("tracker")
    if trk is None:
        trk = "bytetrack.yaml"

    try:
        cc = CounterConfig(
            down_angle=float(down),
            up_angle=float(up),
            min_joint_confidence=float(conf),
            stable_frames=int(stable) if not isinstance(stable, bool) else -1,
            max_gap_seconds=float(gap),
        )
    except Exception as exc:
        raise ValueError(f"Invalid counter configuration: {exc}") from exc

    if isinstance(dep, bool) or not isinstance(dep, int) or dep < 1:
        raise ValueError(f"departure_frames must be an integer >= 1, got {dep!r}")

    if not isinstance(trk, str) or not trk.strip():
        raise ValueError(f"tracker identifier must be a non-empty string, got {trk!r}")

    return {
        "down_angle": cc.down_angle,
        "up_angle": cc.up_angle,
        "min_confidence": cc.min_joint_confidence,
        "stable_frames": cc.stable_frames,
        "departure_frames": dep,
        "max_gap": cc.max_gap_seconds,
        "tracker": trk.strip(),
    }


def _capture_runtime_fingerprint(interpreter: Path, gym_pose_dir: Path, tracker: str) -> Dict[str, Any]:
    """Read selected environment metadata/config without importing ML packages."""
    code = """import sys,json,hashlib,importlib.metadata as m
from pathlib import Path
def version(name):
    try: return m.version(name)
    except m.PackageNotFoundError: return 'unavailable'
tracker=Path(sys.argv[1])
if not tracker.is_file():
    try: tracker=Path(m.distribution('ultralytics').locate_file('ultralytics/cfg/trackers/'+sys.argv[1]))
    except m.PackageNotFoundError: pass
print(json.dumps({'python_version':sys.version.replace('\\n',' '),'platform':sys.platform,
 'package_versions':{n:version(n) for n in ('torch','ultralytics','numpy','opencv-python')},
 'tracker_path':str(tracker.resolve()) if tracker.is_file() else None,
 'tracker_sha256':hashlib.sha256(tracker.read_bytes()).hexdigest() if tracker.is_file() else None}))
"""
    try:
        result = subprocess.run([str(interpreter), "-c", code, tracker], cwd=str(gym_pose_dir),
                                capture_output=True, text=True, timeout=5, shell=False)
        fingerprint = json.loads(result.stdout) if result.returncode == 0 else None
        if isinstance(fingerprint, dict) and isinstance(fingerprint.get("python_version"), str):
            return fingerprint
    except Exception:
        pass
    return {"python_version": "unavailable", "platform": "unavailable",
            "package_versions": {}, "tracker_path": None, "tracker_sha256": None}


def get_runtime_provenance(
    interpreter: Path,
    device: str,
    gym_pose_dir: Optional[Path] = None,
    declared_configuration: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Capture execution environment provenance safely without shell dependencies."""
    reject_linked_ancestors(Path(interpreter), label="Python interpreter")
    search_dir = gym_pose_dir or Path(__file__).parent.parent
    reject_linked_ancestors(Path(search_dir), label="Gym pose directory")

    effective = normalize_declared_configuration(declared_configuration)
    fingerprint = _capture_runtime_fingerprint(interpreter, search_dir, effective["tracker"])
    provenance: Dict[str, Any] = {
        "interpreter": str(interpreter.resolve()),
        **fingerprint,
        "device": device,
        "git_commit": "unavailable",
        "git_dirty": "unavailable",
        "code_hashes": capture_package_code_hashes(search_dir),
        "effective_parameters": effective,
        "environment_scope": "Interpreter/project bytes and observed dependency versions/tracker digest; not a complete environment image or hardware determinism guarantee",
    }

    try:
        proc = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(search_dir),
            capture_output=True,
            text=True,
            timeout=5.0,
            shell=False,
        )
        if proc.returncode == 0 and proc.stdout.strip():
            provenance["git_commit"] = proc.stdout.strip()
            status_proc = subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=str(search_dir),
                capture_output=True,
                text=True,
                timeout=5.0,
                shell=False,
            )
            if status_proc.returncode == 0:
                provenance["git_dirty"] = bool(status_proc.stdout.strip())
    except Exception:
        pass

    return provenance


def resolve_default_interpreter(gym_pose_dir: Optional[Path] = None) -> Path:
    """Discover default Python interpreter favoring virtual environments over sys.executable."""
    search_base = gym_pose_dir or Path(__file__).parent.parent
    reject_linked_ancestors(Path(search_base), label="Gym pose directory")
    base = Path(search_base).resolve()
    candidates = [
        base / env / script
        for env in (".venv_cuda", ".venv", ".venv_web")
        for script in ("Scripts/python.exe", "bin/python", "Scripts/python")
    ]
    for c in candidates:
        if c.is_file():
            try:
                reject_linked_ancestors(c, label="Interpreter candidate")
                if not is_symlink(c):
                    return c.resolve()
            except Exception:
                pass
    sys_exec = Path(sys.executable)
    reject_linked_ancestors(sys_exec, label="System executable")
    return sys_exec.resolve()


def validate_interpreter(interpreter_path: Union[str, Path]) -> Path:
    """Validate that the given interpreter is an existing regular executable file."""
    p = Path(interpreter_path)
    reject_linked_ancestors(p, label="Python interpreter")
    if is_symlink(p):
        raise ValueError(f"Python interpreter cannot be a symlink or reparse point: {p}")
    if not p.exists():
        raise FileNotFoundError(f"Python interpreter not found: {p}")
    if not p.is_file():
        raise ValueError(f"Python interpreter is not a regular file: {p}")
    return p.resolve()


def validate_device_string(device: str) -> str:
    """Validate safe compute device specification."""
    if not isinstance(device, str):
        raise TypeError(f"Device must be a string, got {type(device).__name__}")
    dev = device.strip()
    if not dev:
        raise ValueError("Device cannot be empty")
    if not SAFE_DEVICE_PATTERN.match(dev):
        raise ValueError(f"Unsafe device identifier {device!r}; must match {SAFE_DEVICE_PATTERN.pattern}")
    return dev


# ==============================================================================
# Plan Verification & Hashing
# ==============================================================================


def compute_plan_sha256(plan: Dict[str, Any]) -> str:
    """Compute the canonical SHA256 digest of an evaluation plan payload."""
    payload = {k: v for k, v in plan.items() if k != "plan_sha256"}
    canonical_bytes = json.dumps(
        payload, sort_keys=True, indent=2, allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(canonical_bytes).hexdigest().lower()


def load_and_validate_plan(
    plan_path: Union[str, Path],
) -> Tuple[Dict[str, Any], str]:
    """Load plan.json and verify its schema, syntax, internal hash, and source truth."""
    p = Path(plan_path)
    reject_linked_ancestors(p, label="Plan path")
    if is_symlink(p):
        raise ValueError(f"Plan path cannot be a symlink or reparse point: {p}")
    if not p.exists():
        raise FileNotFoundError(f"Plan file not found: {p}")
    if not p.is_file():
        raise ValueError(f"Plan path is not a regular file: {p}")

    raw_bytes = p.read_bytes()
    try:
        plan_dict = json.loads(
            raw_bytes.decode("utf-8"),
            object_pairs_hook=detect_duplicate_pairs,
            parse_constant=_reject_json_constant,
        )
    except Exception as exc:
        raise ValueError(f"Failed to parse plan JSON: {exc}") from exc

    if not isinstance(plan_dict, dict):
        raise ValueError(f"Top-level plan must be a JSON object, got {type(plan_dict).__name__}")

    for req in ("plan_version", "status", "created_utc", "review_root", "models", "eligible_cases", "source_annotation_hashes", "plan_sha256"):
        if req not in plan_dict:
            raise ValueError(f"Plan is missing required field '{req}'")

    if plan_dict["plan_version"] != "1.0":
        raise ValueError(f"Unsupported plan version {plan_dict['plan_version']!r}; expected '1.0'")

    recorded_hash = plan_dict.get("plan_sha256")
    if not recorded_hash or not isinstance(recorded_hash, str) or not SHA256_HEX_PATTERN.match(recorded_hash.strip().lower()):
        raise ValueError("Plan is missing mandatory valid 64-hex plan_sha256")
    recorded_hash = recorded_hash.strip().lower()

    computed_hash = compute_plan_sha256(plan_dict)
    if computed_hash != recorded_hash:
        raise ValueError(
            f"Plan SHA256 mismatch: recorded {recorded_hash}, computed {computed_hash}"
        )

    if "declared_configuration" in plan_dict and plan_dict["declared_configuration"] is not None:
        normalize_declared_configuration(plan_dict["declared_configuration"])

    # Validate against source truth on disk (models, review_root, cases)
    verify_plan_inputs(plan_dict, p)

    return plan_dict, computed_hash


def verify_plan_inputs(
    plan: Dict[str, Any],
    plan_path: Path,
) -> Tuple[Dict[str, Path], Path, List[Dict[str, Any]]]:
    """Verify that all models, source videos, and annotation files exist and match hashes and source truth."""
    rev_root_raw = plan.get("review_root")
    if not rev_root_raw or not isinstance(rev_root_raw, str):
        raise ValueError("Plan is missing or has invalid 'review_root'")
    review_root = Path(rev_root_raw)
    if not review_root.is_absolute():
        p_plan_parent = plan_path.parent
        reject_linked_ancestors(p_plan_parent, label="Plan parent directory")
        review_root = p_plan_parent / review_root

    reject_linked_ancestors(review_root, label="Review root")
    if is_symlink(review_root):
        raise CaseCorruptedError(f"Review root cannot be a symlink or reparse point: {review_root}")
    if not review_root.exists():
        raise FileNotFoundError(f"Review root directory not found: {review_root}")
    if not review_root.is_dir():
        raise ValueError(f"Review root is not a directory: {review_root}")
    review_root = review_root.resolve()

    # Validate models contract
    raw_models = plan.get("models")
    if not isinstance(raw_models, dict) or not raw_models:
        raise ValueError("Plan contains no models to evaluate or models is not a dict")

    validated_models: Dict[str, Path] = {}
    for m_name, m_info in raw_models.items():
        if not isinstance(m_name, str) or not SAFE_MODEL_NAME_PATTERN.match(m_name):
            raise ValueError(f"Invalid model name {m_name!r}; must match {SAFE_MODEL_NAME_PATTERN.pattern}")
        if not isinstance(m_info, dict) or "path" not in m_info or "sha256" not in m_info:
            raise ValueError(f"Invalid model record in plan for '{m_name}'")
        m_sha = m_info.get("sha256")
        if not m_sha or not isinstance(m_sha, str) or not SHA256_HEX_PATTERN.match(m_sha.strip().lower()):
            raise ValueError(f"Model '{m_name}' is missing mandatory valid sha256 in plan")
        m_sha = m_sha.strip().lower()

        m_path = Path(m_info["path"])
        if not m_path.is_absolute():
            m_path = plan_path.parent / m_path

        reject_linked_ancestors(m_path, label=f"Model checkpoint '{m_name}'")
        if is_symlink(m_path):
            raise ValueError(f"Model checkpoint cannot be a symlink: {m_path}")
        if not m_path.exists():
            raise FileNotFoundError(f"Model checkpoint not found for '{m_name}': {m_path}")
        if not m_path.is_file():
            raise ValueError(f"Model checkpoint is not a regular file: {m_path}")
        if m_path.stat().st_size <= 0:
            raise ValueError(f"Model checkpoint is empty for '{m_name}': {m_path}")

        exp_size = m_info.get("file_size_bytes")
        if exp_size is not None and m_path.stat().st_size != exp_size:
            raise ValueError(
                f"Model checkpoint size mismatch for '{m_name}': expected {exp_size}, got {m_path.stat().st_size}"
            )

        actual_sha = compute_file_sha256(m_path)
        if actual_sha != m_sha:
            raise ValueError(
                f"Model checkpoint SHA256 mismatch for '{m_name}': expected {m_sha}, got {actual_sha}"
            )
        validated_models[m_name] = m_path.resolve()

    # Validate eligible cases
    eligible_cases: List[Dict[str, Any]] = plan.get("eligible_cases", [])
    if not isinstance(eligible_cases, list):
        raise ValueError("Plan 'eligible_cases' must be a list")

    seen_case_ids: Set[str] = set()
    source_annotation_hashes = plan.get("source_annotation_hashes", {})
    if not isinstance(source_annotation_hashes, dict):
        raise ValueError("Plan 'source_annotation_hashes' must be a dictionary")

    for c in eligible_cases:
        if not isinstance(c, dict):
            raise ValueError("Item in eligible_cases is not a dict")
        cid = c.get("case_id")
        if not cid or not isinstance(cid, str) or not is_valid_uuid4(cid):
            raise ValueError(f"Invalid case ID in eligible_cases: {cid!r}")
        if cid in seen_case_ids:
            raise ValueError(f"Duplicate case ID in eligible_cases: {cid}")
        seen_case_ids.add(cid)

        # Mandatory annotation hash in plan
        plan_ann_sha = c.get("annotation_sha256")
        map_ann_sha = source_annotation_hashes.get(cid)
        if not plan_ann_sha and map_ann_sha:
            plan_ann_sha = map_ann_sha
        if not plan_ann_sha or not isinstance(plan_ann_sha, str) or not SHA256_HEX_PATTERN.match(plan_ann_sha.strip().lower()):
            raise ValueError(f"Case '{cid}' is missing mandatory valid annotation_sha256 in plan")
        plan_ann_sha = plan_ann_sha.strip().lower()
        if map_ann_sha and map_ann_sha.strip().lower() != plan_ann_sha:
            raise ValueError(
                f"Annotation SHA256 mismatch between case and source_annotation_hashes for case '{cid}'"
            )

        # Mandatory source video hash in plan
        plan_src_sha = c.get("source_sha256")
        if not plan_src_sha or not isinstance(plan_src_sha, str) or not SHA256_HEX_PATTERN.match(plan_src_sha.strip().lower()):
            raise ValueError(f"Case '{cid}' is missing mandatory valid source_sha256 in plan")
        plan_src_sha = plan_src_sha.strip().lower()

        # Unresolved ancestor & symlink checks BEFORE resolve()
        case_dir = review_root / cid
        reject_linked_ancestors(case_dir, label=f"Case directory '{cid}'")
        if is_symlink(case_dir):
            raise CaseCorruptedError(f"Case directory cannot be a symlink: {case_dir}")
        if not case_dir.exists():
            raise FileNotFoundError(f"Case directory not found: {case_dir}")
        if not case_dir.is_dir():
            raise ValueError(f"Case directory is not a directory: {case_dir}")

        case_json_p = case_dir / "case.json"
        reject_linked_ancestors(case_json_p, label=f"case.json for '{cid}'")
        if is_symlink(case_json_p):
            raise CaseCorruptedError(f"case.json cannot be a symlink: {case_json_p}")
        if not case_json_p.exists():
            raise FileNotFoundError(f"case.json not found for case '{cid}': {case_json_p}")
        if not case_json_p.is_file():
            raise ValueError(f"case.json is not a regular file: {case_json_p}")

        # Validate disk file SHA256 matches plan annotation_sha256
        actual_ann_sha = compute_file_sha256(case_json_p)
        if actual_ann_sha != plan_ann_sha:
            raise ValueError(
                f"Annotation SHA256 mismatch for case '{cid}': expected {plan_ann_sha}, got {actual_ann_sha}"
            )

        # Validate raw video file and SHA256 matches plan source_sha256
        ext = c.get("extension", ".mp4")
        video_p = case_dir / f"video{ext}"
        reject_linked_ancestors(video_p, label=f"Video file for '{cid}'")
        if is_symlink(video_p):
            raise CaseCorruptedError(f"Video file cannot be a symlink: {video_p}")
        if not video_p.exists():
            raise FileNotFoundError(f"Video file not found for case '{cid}': {video_p}")
        if not video_p.is_file():
            raise ValueError(f"Video file is not a regular file: {video_p}")

        actual_v_sha = compute_file_sha256(video_p)
        if actual_v_sha != plan_src_sha:
            raise ValueError(
                f"Video SHA256 mismatch for case '{cid}': expected {plan_src_sha}, got {actual_v_sha}"
            )

        # Load and validate disk case using ReviewStore
        try:
            disk_case = load_persisted_case(review_root, cid, verify_raw_sha=True)
        except Exception as exc:
            raise ValueError(f"Case '{cid}' failed ReviewStore validation: {exc}") from exc

        # Strictly validate against ReviewStore rules:
        # group == "candidate" (reject if "development" or anything else!)
        disk_group = disk_case.get("group")
        if disk_group != "candidate":
            raise ValueError(
                f"Eligible case '{cid}' has group {disk_group!r} on disk; only 'candidate' group is primary-eligible."
            )
        plan_group = c.get("group")
        if plan_group != "candidate":
            raise ValueError(
                f"Eligible case '{cid}' has plan group {plan_group!r}; only 'candidate' group is primary-eligible."
            )

        # review object
        validated_case = ReviewStore(root=review_root).load_persisted_case(cid, verify_raw_sha=True)
        if c.get("review") != validated_case["review"]:
            raise ValueError(f"Plan review snapshot differs from validated source truth for '{cid}'")
        if compute_file_sha256(case_json_p) != plan_ann_sha:
            raise PlanOrCodeMutationError(f"Case metadata mutated while validating '{cid}'")

        disk_review = disk_case.get("review")
        if not isinstance(disk_review, dict):
            raise ValueError(f"Case '{cid}' has missing or non-dict review object on disk.")
        plan_review = c.get("review")
        if not isinstance(plan_review, dict):
            raise ValueError(f"Case '{cid}' has missing or non-dict review object in plan.")

        # review["scope"] == "in_scope"
        if disk_review.get("scope") != "in_scope":
            raise ValueError(
                f"Eligible case '{cid}' scope is {disk_review.get('scope')!r}; expected 'in_scope'."
            )
        if plan_review.get("scope") != "in_scope":
            raise ValueError(
                f"Plan eligible case '{cid}' scope is {plan_review.get('scope')!r}; expected 'in_scope'."
            )

        # review["predictions_previously_seen"] is False (reject True or None!)
        if disk_review.get("predictions_previously_seen") is not False:
            raise ValueError(
                f"Eligible case '{cid}' predictions_previously_seen is {disk_review.get('predictions_previously_seen')!r}; expected False."
            )
        if plan_review.get("predictions_previously_seen") is not False:
            raise ValueError(
                f"Plan eligible case '{cid}' predictions_previously_seen is {plan_review.get('predictions_previously_seen')!r}; expected False."
            )

        # review["raw_video_reviewed"] is True
        if disk_review.get("raw_video_reviewed") is not True:
            raise ValueError(
                f"Eligible case '{cid}' raw_video_reviewed is {disk_review.get('raw_video_reviewed')!r}; expected True."
            )
        if plan_review.get("raw_video_reviewed") is not True:
            raise ValueError(
                f"Plan eligible case '{cid}' raw_video_reviewed is {plan_review.get('raw_video_reviewed')!r}; expected True."
            )

        # review["human_count"] is non-negative int
        disk_human_count = disk_review.get("human_count")
        if disk_human_count is None or isinstance(disk_human_count, bool) or not isinstance(disk_human_count, int) or disk_human_count < 0:
            raise ValueError(f"Case '{cid}' has invalid disk human_count: {disk_human_count!r}")
        plan_human_count = plan_review.get("human_count")
        if plan_human_count != disk_human_count:
            raise ValueError(
                f"Plan human_count ({plan_human_count}) mismatch with disk ({disk_human_count}) for case '{cid}'."
            )

        # reviewer identity
        disk_reviewer = disk_review.get("reviewer")
        if not disk_reviewer or not isinstance(disk_reviewer, str) or not disk_reviewer.strip():
            raise ValueError(f"Case '{cid}' has empty reviewer identity on disk")
        plan_reviewer = plan_review.get("reviewer")
        if plan_reviewer != disk_reviewer:
            raise ValueError(
                f"Plan reviewer ({plan_reviewer!r}) mismatch with disk ({disk_reviewer!r}) for case '{cid}'."
            )

        # Validate disk case source_sha256 matches actual video
        if disk_case.get("source_sha256") != actual_v_sha:
            raise ValueError(
                f"Video SHA256 mismatch with disk case.json for case '{cid}': expected {disk_case.get('source_sha256')}, got {actual_v_sha}"
            )

        # Validate snapshot fields match disk
        for field in ("original_filename", "extension", "file_size_bytes", "created_at", "updated_at", "source_sha256"):
            if field in c and c[field] != disk_case.get(field):
                raise ValueError(
                    f"Plan case '{cid}' field '{field}' ({c[field]!r}) mismatch with disk ({disk_case.get(field)!r})"
                )
        if "history" in c and c["history"] != disk_case.get("history"):
            raise ValueError(
                f"Plan case '{cid}' history mismatch with disk"
            )

    return validated_models, review_root, eligible_cases


def validate_execution_output_dir(
    output_dir: Path,
    plan_path: Path,
    review_root: Path,
    models: Dict[str, Path],
) -> None:
    """Validate that execution output_dir is fresh, empty, and does not overlap protected paths."""
    reject_linked_ancestors(output_dir, label="Output directory")
    reject_linked_ancestors(output_dir.parent, label="Output parent directory")

    if is_symlink(output_dir):
        raise ValueError(f"Output directory cannot be a symlink or reparse point: {output_dir}")

    out_res = output_dir.resolve()
    plan_res = plan_path.resolve()
    plan_parent_res = plan_path.parent.resolve()

    if out_res == plan_res:
        raise ValueError(f"Output directory cannot be identical to plan path: {output_dir}")
    if out_res == plan_parent_res:
        raise ValueError(f"Output directory cannot be identical to plan parent directory: {output_dir}")

    try:
        out_res.relative_to(plan_parent_res)
        raise ValueError(f"Output directory cannot be inside plan directory: {output_dir}")
    except ValueError as exc:
        if "Output directory cannot" in str(exc):
            raise

    try:
        plan_parent_res.relative_to(out_res)
        raise ValueError(f"Plan directory cannot be inside output directory: {plan_path.parent}")
    except ValueError as exc:
        if "Plan directory cannot" in str(exc):
            raise

    # Reject non-empty destination if it exists
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory exists and is not empty: {output_dir}")

    # Standard package and review root overlap checks
    validate_path_overlap(output_dir, review_root, models)


# ==============================================================================
# Child Subprocess Analysis Runner & Process Tree Cleanup
# ==============================================================================


def _windows_kernel32():
    """Declare pointer-sized Win32 signatures; ctypes defaults truncate x64 handles."""
    import ctypes
    from ctypes import wintypes
    api = ctypes.WinDLL("kernel32", use_last_error=True)
    signatures = {
        "CreateJobObjectW": ([ctypes.c_void_p, wintypes.LPCWSTR], wintypes.HANDLE),
        "SetInformationJobObject": ([wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD], wintypes.BOOL),
        "AssignProcessToJobObject": ([wintypes.HANDLE, wintypes.HANDLE], wintypes.BOOL),
        "TerminateJobObject": ([wintypes.HANDLE, wintypes.UINT], wintypes.BOOL),
        "QueryInformationJobObject": ([wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD, ctypes.c_void_p], wintypes.BOOL),
        "CloseHandle": ([wintypes.HANDLE], wintypes.BOOL),
    }
    for name, (args, result) in signatures.items():
        function = getattr(api, name)
        function.argtypes, function.restype = args, result
    return api


def create_windows_job_object() -> Optional[Any]:
    """Create a Windows Job Object configured to kill all processes on close."""
    if sys.platform != "win32":
        return None
    try:
        import ctypes
        from ctypes import wintypes
        kernel32 = _windows_kernel32()
        job = kernel32.CreateJobObjectW(None, None)
        if not job:
            return None

        class JOBOBJECT_BASIC_LIMIT_INFORMATION(ctypes.Structure):
            _fields_ = [
                ('PerProcessUserTimeLimit', ctypes.c_int64),
                ('PerJobUserTimeLimit', ctypes.c_int64),
                ('LimitFlags', wintypes.DWORD),
                ('MinimumWorkingSetSize', ctypes.c_size_t),
                ('MaximumWorkingSetSize', ctypes.c_size_t),
                ('ActiveProcessLimit', wintypes.DWORD),
                ('Affinity', ctypes.c_size_t),
                ('PriorityClass', wintypes.DWORD),
                ('SchedulingClass', wintypes.DWORD),
            ]

        class IO_COUNTERS(ctypes.Structure):
            _fields_ = [(f'Field{i}', ctypes.c_uint64) for i in range(6)]

        class JOBOBJECT_EXTENDED_LIMIT_INFORMATION(ctypes.Structure):
            _fields_ = [
                ('BasicLimitInformation', JOBOBJECT_BASIC_LIMIT_INFORMATION),
                ('IoInfo', IO_COUNTERS),
                ('ProcessMemoryLimit', ctypes.c_size_t),
                ('JobMemoryLimit', ctypes.c_size_t),
                ('PeakProcessMemoryLimit', ctypes.c_size_t),
                ('PeakJobMemoryLimit', ctypes.c_size_t),
            ]

        info = JOBOBJECT_EXTENDED_LIMIT_INFORMATION()
        info.BasicLimitInformation.LimitFlags = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        if not kernel32.SetInformationJobObject(job, 9, ctypes.byref(info), ctypes.sizeof(info)):
            error = ctypes.WinError(ctypes.get_last_error())
            kernel32.CloseHandle(job)
            raise error
        return job
    except Exception as exc:
        logger.debug("Failed to create Windows Job Object: %s", exc)
        return None


def kill_process_tree(proc, timeout_sec=5.0, *, job_handle=None, extra_pids=None):
    """Stop only an owned job/session; never discover or kill processes by recycled PID."""
    if sys.platform == "win32":
        if job_handle is None:
            raise RuntimeError("Cannot establish process ownership without the Windows job")
        import ctypes
        from ctypes import wintypes
        class Accounting(ctypes.Structure):
            _fields_ = [("times", ctypes.c_int64 * 4), ("faults", wintypes.DWORD),
                        ("total", wintypes.DWORD), ("active", wintypes.DWORD),
                        ("terminated", wintypes.DWORD)]
        api = _windows_kernel32()
        try:
            if not api.TerminateJobObject(job_handle, 1):
                raise ctypes.WinError(ctypes.get_last_error())
            deadline = time.monotonic() + timeout_sec
            while True:
                accounting = Accounting()
                if not api.QueryInformationJobObject(job_handle, 1, ctypes.byref(accounting), ctypes.sizeof(accounting), None):
                    raise ctypes.WinError(ctypes.get_last_error())
                if accounting.active == 0:
                    break
                if time.monotonic() >= deadline:
                    raise TimeoutError("Owned Windows job still has live processes after cleanup deadline")
                time.sleep(0.01)
        finally:
            if not api.CloseHandle(job_handle):
                raise ctypes.WinError(ctypes.get_last_error())
    else:
        # Every child below starts a new session, whose process-group ID is its PID.
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    proc.wait(timeout=timeout_sec)
    return True


def _start_owned_child(cmd, log_file, popen_kwargs):
    """Assign a gated native-Python wrapper before it can spawn the requested venv."""
    if sys.platform != "win32":
        return subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT, **popen_kwargs), None
    import ctypes
    job = create_windows_job_object()
    if job is None:
        raise RuntimeError("Windows process containment unavailable; no analyser was launched")
    proc = None
    try:
        # sys._base_executable avoids the venv launcher's spawn-before-assignment race.
        wrapper = ("import sys,subprocess; "
                   "gate=sys.stdin.buffer.read(1); "
                   "sys.exit(subprocess.call(sys.argv[1:],stdin=subprocess.DEVNULL) if gate==b'G' else 125)")
        options = dict(popen_kwargs)
        options["stdin"] = subprocess.PIPE
        proc = subprocess.Popen([sys._base_executable, "-c", wrapper, *cmd],
                                stdout=log_file, stderr=subprocess.STDOUT, **options)
        api = _windows_kernel32()
        if not api.AssignProcessToJobObject(job, int(proc._handle)):
            raise ctypes.WinError(ctypes.get_last_error())
        proc.stdin.write(b"G")
        proc.stdin.flush()
        proc.stdin.close()
        return proc, job
    except BaseException:
        if proc is not None:
            if proc.stdin is not None and not proc.stdin.closed:
                proc.stdin.close()  # gate EOF prevents starting the analyser
            proc.kill()  # owned Popen handle, never a discovered PID
            proc.wait(timeout=5)
        _windows_kernel32().CloseHandle(job)
        raise


def default_subprocess_runner(
    interpreter: Path,
    gym_pose_dir: Path,
    video_path: Path,
    output_dir: Path,
    model_path: Path,
    device: str,
    timeout: float,
    *,
    log_path: Optional[Path] = None,
    declared_configuration: Optional[Mapping[str, Any]] = None,
    case_id: Optional[str] = None,
    model_name: Optional[str] = None,
    extra_args: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """Execute gym_analyser CLI in a dedicated child process with explicit finite deadline."""
    if not model_path.is_file():
        raise FileNotFoundError(f"Model checkpoint is missing: {model_path}")

    # Ensure output_dir is fresh and empty; DO NOT write run.log inside it before running!
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output_dir}")

    cmd = [
        str(interpreter),
        "-m",
        "gym_analyser",
        "--source",
        str(video_path),
        "--output-dir",
        str(output_dir),
        "--model",
        str(model_path),
        "--device",
        str(device),
    ]

    # Inspect CLI capability to forward declared configuration safely
    main_py = gym_pose_dir / "gym_analyser" / "__main__.py"
    supports_flags = True
    if main_py.is_file():
        try:
            cli_code = main_py.read_text(encoding="utf-8")
            if "--down-angle" not in cli_code and "parse_known_args" not in cli_code:
                supports_flags = False
        except Exception:
            pass

    if declared_configuration:
        normalized_cfg = normalize_declared_configuration(declared_configuration)
        if not supports_flags:
            raise ValueError(
                f"Child analyser at {main_py} does not support forwarding declared configuration flags"
            )
        cmd.extend(["--down-angle", str(normalized_cfg["down_angle"])])
        cmd.extend(["--up-angle", str(normalized_cfg["up_angle"])])
        cmd.extend(["--min-confidence", str(normalized_cfg["min_confidence"])])
        cmd.extend(["--stable-frames", str(normalized_cfg["stable_frames"])])
        cmd.extend(["--departure-frames", str(normalized_cfg["departure_frames"])])
        cmd.extend(["--max-gap", str(normalized_cfg["max_gap"])])
        cmd.extend(["--tracker", str(normalized_cfg["tracker"])])

    if extra_args:
        if "--overwrite" in extra_args:
            raise ValueError("--overwrite is strictly forbidden in review execution")
        cmd.extend(extra_args)

    # Place live log OUTSIDE output_dir so output_dir remains completely fresh
    if log_path is not None:
        target_log = Path(log_path)
    else:
        target_log = output_dir.parent / "logs" / f"{output_dir.name}.log"
    target_log.parent.mkdir(parents=True, exist_ok=True)

    exit_code: Optional[int] = None
    timed_out = False

    child_env = dict(os.environ)
    existing_pp = child_env.get("PYTHONPATH", "")
    child_env["PYTHONPATH"] = (
        str(gym_pose_dir.resolve()) + (os.pathsep + existing_pp if existing_pp else "")
    )

    popen_kwargs: Dict[str, Any] = {
        "stdin": subprocess.DEVNULL,
        "cwd": str(gym_pose_dir),
        "env": child_env,
        "shell": False,
    }
    if sys.platform == "win32":
        popen_kwargs["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    else:
        popen_kwargs["start_new_session"] = True

    with open(target_log, "w", encoding="utf-8") as log_file:
        proc, job_handle = _start_owned_child(cmd, log_file, popen_kwargs)
        try:
            try:
                exit_code = proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                timed_out = True
                log_file.write(f"\n[EXECUTION TIMEOUT]: Child exceeded {timeout}s deadline.\n")
                log_file.flush()
        finally:
            # Also closes contained descendants after normal/early root exit and controller errors.
            kill_process_tree(proc, timeout_sec=5.0, job_handle=job_handle)

    # After process completes, copy log to output_dir / "run.log" if output_dir exists
    if output_dir.exists():
        out_run_log = output_dir / "run.log"
        try:
            shutil.copy2(target_log, out_run_log)
        except Exception:
            pass

    if timed_out:
        return {
            "status": "timeout",
            "completed_reps": None,
            "exit_code": None,
            "read_status_note": None,
            "error": f"Process exceeded timeout of {timeout}s",
        }

    summary_path = output_dir / "summary.json"
    if not summary_path.is_file():
        return {
            "status": "failed",
            "completed_reps": None,
            "exit_code": exit_code,
            "read_status_note": None,
            "error": f"summary.json not produced by analyser (exit code: {exit_code})",
        }

    try:
        raw_text = summary_path.read_text(encoding="utf-8")
        summary = json.loads(
            raw_text,
            object_pairs_hook=detect_duplicate_pairs,
            parse_constant=_reject_json_constant,
        )
    except Exception as exc:
        return {
            "status": "failed",
            "completed_reps": None,
            "exit_code": exit_code,
            "read_status_note": None,
            "error": f"Malformed summary.json output: {exc}",
        }

    if not isinstance(summary, dict):
        return {
            "status": "failed",
            "completed_reps": None,
            "exit_code": exit_code,
            "read_status_note": None,
            "error": "Malformed summary.json output: root is not a JSON object",
        }

    child_status = summary.get("status")
    note = summary.get("read_status_note")
    err = summary.get("error")

    if child_status == "completed" and exit_code == 0:
        required = {"source_video": dict, "configuration": dict, "model_name": str}
        for key, kind in required.items():
            if not isinstance(summary.get(key), kind) or not summary[key]:
                return {"status": "failed", "completed_reps": None, "exit_code": exit_code,
                        "read_status_note": note, "error": f"Missing/invalid required {key} in completed summary"}
        if Path(summary["model_name"]).resolve() != model_path.resolve():
            return {"status": "failed", "completed_reps": None, "exit_code": exit_code,
                    "read_status_note": note, "error": "Model checkpoint path mismatch"}
        declared = summary["source_video"].get("declared_frame_count")
        processed = summary.get("processed_frames")
        if (isinstance(declared, bool) or
            (declared is not None and (not isinstance(declared, int) or declared <= 0)) or
            (isinstance(processed, int) and declared is not None and processed < declared)):
            return {"status": "failed", "completed_reps": None, "exit_code": exit_code,
                    "read_status_note": note, "error": "Completed summary contradicts declared_frame_count/processed_frames completeness"}
        required_cfg = {"down_angle", "up_angle", "min_joint_confidence", "stable_frames",
                        "departure_frames", "max_gap_seconds", "tracker", "device"}
        missing = required_cfg - summary["configuration"].keys()
        if missing:
            return {"status": "failed", "completed_reps": None, "exit_code": exit_code,
                    "read_status_note": note, "error": f"Missing configuration fields: {sorted(missing)}"}
        expected = normalize_declared_configuration(declared_configuration)
        fields = {"down_angle": "down_angle", "up_angle": "up_angle", "min_joint_confidence": "min_confidence",
                  "stable_frames": "stable_frames", "departure_frames": "departure_frames", "max_gap_seconds": "max_gap"}
        for actual_key, expected_key in fields.items():
            value, wanted = summary["configuration"][actual_key], expected[expected_key]
            integral = actual_key in ("stable_frames", "departure_frames")
            if (isinstance(value, bool) or not isinstance(value, (int, float)) or
                not math.isfinite(value) or (integral and not isinstance(value, int)) or
                not math.isclose(value, wanted, rel_tol=1e-12, abs_tol=1e-12)):
                return {"status": "failed", "completed_reps": None, "exit_code": exit_code,
                        "read_status_note": note, "error": f"Configuration mismatch for {actual_key}: expected {wanted}, got {value!r}"}
        if summary["configuration"]["device"] != device or summary["configuration"]["tracker"] != expected["tracker"]:
            return {"status": "failed", "completed_reps": None, "exit_code": exit_code,
                    "read_status_note": note, "error": "Configuration device/tracker mismatch"}
        if summary.get("error"):
            return {"status": "failed", "completed_reps": None, "exit_code": exit_code,
                    "read_status_note": note, "error": "Completed summary contains an error"}

    # 1. Match source video if present in summary against actual video_path
    for src_key in ("source_video", "source"):
        if src_key in summary and summary[src_key] is not None:
            sv = summary[src_key]
            sv_path: Optional[str] = None
            if isinstance(sv, dict):
                sv_path = sv.get("path")
                if not sv_path:
                    return {
                        "status": "failed",
                        "completed_reps": None,
                        "exit_code": exit_code,
                        "read_status_note": note,
                        "error": f"Missing or empty 'path' in summary '{src_key}'",
                    }
            elif isinstance(sv, (str, Path)):
                sv_path = str(sv)

            if sv_path is not None:
                try:
                    sv_p = Path(sv_path)
                    sv_resolved = sv_p.resolve()
                    vid_resolved = video_path.resolve()
                    sv_norm = os.path.normcase(os.path.abspath(str(sv_path)))
                    vid_norm = os.path.normcase(os.path.abspath(str(video_path)))
                    if sv_resolved != vid_resolved and sv_norm != vid_norm:
                        return {
                            "status": "failed",
                            "completed_reps": None,
                            "exit_code": exit_code,
                            "read_status_note": note,
                            "error": f"Source video mismatch in summary ('{src_key}'): expected {vid_resolved}, got {sv_path}",
                        }
                except Exception as exc:
                    return {
                        "status": "failed",
                        "completed_reps": None,
                        "exit_code": exit_code,
                        "read_status_note": note,
                        "error": f"Invalid source video path in summary ('{src_key}') {sv_path!r}: {exc}",
                    }

    # 2. Match configuration if present in summary and declared_configuration provided
    if declared_configuration:
        norm_decl = normalize_declared_configuration(declared_configuration)
        summary_cfg = summary.get("configuration")
        if "configuration" in summary:
            if not isinstance(summary_cfg, dict):
                return {
                    "status": "failed",
                    "completed_reps": None,
                    "exit_code": exit_code,
                    "read_status_note": note,
                    "error": "Invalid configuration object in summary: expected JSON object",
                }

        cfg_source = summary_cfg if summary_cfg is not None else summary

        # down_angle
        if "down_angle" in cfg_source and cfg_source["down_angle"] is not None:
            try:
                val = float(cfg_source["down_angle"])
                if abs(val - float(norm_decl["down_angle"])) > 1e-4:
                    return {
                        "status": "failed",
                        "completed_reps": None,
                        "exit_code": exit_code,
                        "read_status_note": note,
                        "error": f"Configuration mismatch in summary for 'down_angle': expected {norm_decl['down_angle']}, got {cfg_source['down_angle']}",
                    }
            except (ValueError, TypeError):
                return {
                    "status": "failed",
                    "completed_reps": None,
                    "exit_code": exit_code,
                    "read_status_note": note,
                    "error": f"Invalid 'down_angle' value in summary: {cfg_source['down_angle']!r}",
                }

        # up_angle
        if "up_angle" in cfg_source and cfg_source["up_angle"] is not None:
            try:
                val = float(cfg_source["up_angle"])
                if abs(val - float(norm_decl["up_angle"])) > 1e-4:
                    return {
                        "status": "failed",
                        "completed_reps": None,
                        "exit_code": exit_code,
                        "read_status_note": note,
                        "error": f"Configuration mismatch in summary for 'up_angle': expected {norm_decl['up_angle']}, got {cfg_source['up_angle']}",
                    }
            except (ValueError, TypeError):
                return {
                    "status": "failed",
                    "completed_reps": None,
                    "exit_code": exit_code,
                    "read_status_note": note,
                    "error": f"Invalid 'up_angle' value in summary: {cfg_source['up_angle']!r}",
                }

        # min_joint_confidence / min_confidence
        min_conf_val = None
        if "min_joint_confidence" in cfg_source and cfg_source["min_joint_confidence"] is not None:
            min_conf_val = cfg_source["min_joint_confidence"]
        elif "min_confidence" in cfg_source and cfg_source["min_confidence"] is not None:
            min_conf_val = cfg_source["min_confidence"]

        if min_conf_val is not None:
            try:
                val = float(min_conf_val)
                if abs(val - float(norm_decl["min_confidence"])) > 1e-4:
                    return {
                        "status": "failed",
                        "completed_reps": None,
                        "exit_code": exit_code,
                        "read_status_note": note,
                        "error": f"Configuration mismatch in summary for 'min_joint_confidence': expected {norm_decl['min_confidence']}, got {min_conf_val}",
                    }
            except (ValueError, TypeError):
                return {
                    "status": "failed",
                    "completed_reps": None,
                    "exit_code": exit_code,
                    "read_status_note": note,
                    "error": f"Invalid confidence value in summary: {min_conf_val!r}",
                }

        # stable_frames
        if "stable_frames" in cfg_source and cfg_source["stable_frames"] is not None:
            try:
                if isinstance(cfg_source["stable_frames"], bool) or not isinstance(cfg_source["stable_frames"], int) or int(cfg_source["stable_frames"]) != int(norm_decl["stable_frames"]):
                    return {
                        "status": "failed",
                        "completed_reps": None,
                        "exit_code": exit_code,
                        "read_status_note": note,
                        "error": f"Configuration mismatch in summary for 'stable_frames': expected {norm_decl['stable_frames']}, got {cfg_source['stable_frames']}",
                    }
            except (ValueError, TypeError):
                return {
                    "status": "failed",
                    "completed_reps": None,
                    "exit_code": exit_code,
                    "read_status_note": note,
                    "error": f"Invalid 'stable_frames' value in summary: {cfg_source['stable_frames']!r}",
                }

        # departure_frames
        if "departure_frames" in cfg_source and cfg_source["departure_frames"] is not None:
            try:
                if isinstance(cfg_source["departure_frames"], bool) or not isinstance(cfg_source["departure_frames"], int) or int(cfg_source["departure_frames"]) != int(norm_decl["departure_frames"]):
                    return {
                        "status": "failed",
                        "completed_reps": None,
                        "exit_code": exit_code,
                        "read_status_note": note,
                        "error": f"Configuration mismatch in summary for 'departure_frames': expected {norm_decl['departure_frames']}, got {cfg_source['departure_frames']}",
                    }
            except (ValueError, TypeError):
                return {
                    "status": "failed",
                    "completed_reps": None,
                    "exit_code": exit_code,
                    "read_status_note": note,
                    "error": f"Invalid 'departure_frames' value in summary: {cfg_source['departure_frames']!r}",
                }

        # max_gap / max_gap_seconds
        max_gap_val = None
        if "max_gap_seconds" in cfg_source and cfg_source["max_gap_seconds"] is not None:
            max_gap_val = cfg_source["max_gap_seconds"]
        elif "max_gap" in cfg_source and cfg_source["max_gap"] is not None:
            max_gap_val = cfg_source["max_gap"]

        if max_gap_val is not None:
            try:
                val = float(max_gap_val)
                if abs(val - float(norm_decl["max_gap"])) > 1e-4:
                    return {
                        "status": "failed",
                        "completed_reps": None,
                        "exit_code": exit_code,
                        "read_status_note": note,
                        "error": f"Configuration mismatch in summary for 'max_gap': expected {norm_decl['max_gap']}, got {max_gap_val}",
                    }
            except (ValueError, TypeError):
                return {
                    "status": "failed",
                    "completed_reps": None,
                    "exit_code": exit_code,
                    "read_status_note": note,
                    "error": f"Invalid 'max_gap' value in summary: {max_gap_val!r}",
                }

        # tracker / tracker_name
        tracker_val = None
        if "tracker" in cfg_source and cfg_source["tracker"] is not None:
            tracker_val = cfg_source["tracker"]
        elif "tracker_name" in cfg_source and cfg_source["tracker_name"] is not None:
            tracker_val = cfg_source["tracker_name"]

        if tracker_val is not None:
            if str(tracker_val).strip() != str(norm_decl["tracker"]).strip():
                return {
                    "status": "failed",
                    "completed_reps": None,
                    "exit_code": exit_code,
                    "read_status_note": note,
                    "error": f"Configuration mismatch in summary for 'tracker': expected {norm_decl['tracker']!r}, got {tracker_val!r}",
                }

    # 3. Match model / model_name if present in summary
    for model_key in ("model_name", "model"):
        if model_key in summary and summary[model_key] is not None:
            sm_val = str(summary[model_key]).strip()
            if ("/" in sm_val or "\\" in sm_val) or (Path(sm_val).parent != Path("") and str(Path(sm_val).parent) != "."):
                sm_p = Path(sm_val)
                if sm_p.resolve() != model_path.resolve() and os.path.normcase(os.path.abspath(sm_val)) != os.path.normcase(os.path.abspath(str(model_path))):
                    return {
                        "status": "failed",
                        "completed_reps": None,
                        "exit_code": exit_code,
                        "read_status_note": note,
                        "error": f"Model mismatch in summary ('{model_key}'): expected {model_path.resolve()}, got {sm_val}",
                    }
            else:
                expected_identifiers = {model_path.name, model_path.stem}
                if model_name:
                    expected_identifiers.add(model_name)
                if sm_val not in expected_identifiers:
                    return {
                        "status": "failed",
                        "completed_reps": None,
                        "exit_code": exit_code,
                        "read_status_note": note,
                        "error": f"Model mismatch in summary ('{model_key}'): expected {model_name or model_path.name}, got {sm_val}",
                    }

    # 4. Match clip_id if present in summary
    if "clip_id" in summary and summary["clip_id"] is not None:
        sm_clip = str(summary["clip_id"]).strip()
        expected_clips = set()
        if case_id:
            expected_clips.add(case_id)
        expected_clips.add(output_dir.name)
        if sm_clip not in expected_clips:
            return {
                "status": "failed",
                "completed_reps": None,
                "exit_code": exit_code,
                "read_status_note": note,
                "error": f"Clip ID mismatch in summary: expected {case_id or output_dir.name}, got {sm_clip}",
            }

    # 5. Validate status & repetition counts & frame completeness
    if child_status == "completed" and exit_code == 0:
        # Validate processed_frames strictly > 0 integer
        if "processed_frames" not in summary:
            return {
                "status": "failed",
                "completed_reps": None,
                "exit_code": exit_code,
                "read_status_note": note,
                "error": "Missing 'processed_frames' in completed summary.json",
            }
        raw_pf = summary["processed_frames"]
        if raw_pf is None or isinstance(raw_pf, bool) or not isinstance(raw_pf, int) or raw_pf <= 0:
            return {
                "status": "failed",
                "completed_reps": None,
                "exit_code": exit_code,
                "read_status_note": note,
                "error": f"Invalid processed_frames in completed summary: {raw_pf!r} (must be integer > 0)",
            }

        # Validate completed_reps non-negative integer
        if "completed_reps" not in summary:
            return {
                "status": "failed",
                "completed_reps": None,
                "exit_code": exit_code,
                "read_status_note": note,
                "error": "Missing 'completed_reps' in completed summary.json",
            }
        raw_reps = summary["completed_reps"]
        if raw_reps is None or isinstance(raw_reps, bool) or not isinstance(raw_reps, int) or raw_reps < 0:
            return {
                "status": "failed",
                "completed_reps": None,
                "exit_code": exit_code,
                "read_status_note": note,
                "error": f"Invalid completed_reps in summary: {raw_reps!r}",
            }
        try:
            reps = validate_count(raw_reps, "completed_reps", output_dir.name)
            return {
                "status": "completed",
                "completed_reps": reps,
                "exit_code": exit_code,
                "read_status_note": note,
                "error": None,
            }
        except ValueError as val_err:
            return {
                "status": "failed",
                "completed_reps": None,
                "exit_code": exit_code,
                "read_status_note": note,
                "error": f"Invalid completed_reps in summary: {val_err}",
            }
    elif child_status == "incomplete":
        return {
            "status": "incomplete",
            "completed_reps": None,
            "exit_code": exit_code,
            "read_status_note": note or "Incomplete video decoding",
            "error": err,
        }
    else:
        return {
            "status": "failed",
            "completed_reps": None,
            "exit_code": exit_code,
            "read_status_note": note,
            "error": err or f"Analysis ended with status: {child_status} (exit code: {exit_code})",
        }


# ==============================================================================
# Multi-Model Common Completed Cohort Metrics Evaluation
# ==============================================================================


def compute_metrics(
    predictions: Dict[str, int],
    ground_truth: Dict[str, Optional[int]],
    reject_unknown_ground_truth: bool = True,
) -> Dict[str, Any]:
    """Compute exact-match and MAE metrics on repetition counts using evaluation.evaluate_counts."""
    return evaluate_counts(
        predictions=predictions,
        ground_truth=ground_truth,
        reject_unknown_ground_truth=reject_unknown_ground_truth,
    )


def compute_multi_model_common_cohort(
    all_results: Dict[str, Dict[str, Dict[str, Any]]],
    eligible_cases: List[Dict[str, Any]],
) -> Tuple[List[str], List[Dict[str, Any]], Dict[str, Any]]:
    """Determine the intersection cohort of clips fully and completely decoded by ALL models.

    Args:
        all_results: Mapping of model_name -> clip_id -> execution result record.
        eligible_cases: List of primary eligible case dictionaries.

    Returns:
        Tuple of (common_cohort_clip_ids, excluded_clips_records, per_model_metrics).
    """
    model_names = sorted(all_results.keys())
    eligible_ids = [c["case_id"] for c in eligible_cases]
    gt_map: Dict[str, int] = {
        c["case_id"]: c["review"]["human_count"] for c in eligible_cases
    }

    common_cohort: List[str] = []
    excluded_clips: List[Dict[str, Any]] = []

    for c in eligible_cases:
        cid = c["case_id"]
        all_completed = True
        model_failures: List[Dict[str, Any]] = []

        for m in model_names:
            res = all_results[m].get(cid, {})
            status = res.get("status")
            reps = res.get("completed_reps")
            exit_code = res.get("exit_code")

            if (
                status != "completed"
                or reps is None
                or isinstance(reps, bool)
                or not isinstance(reps, int)
                or reps < 0
                or (exit_code is not None and exit_code != 0)
            ):
                all_completed = False
                model_failures.append({
                    "model": m,
                    "status": status,
                    "error": res.get("error"),
                    "read_status_note": res.get("read_status_note"),
                    "exit_code": exit_code,
                })

        if all_completed:
            common_cohort.append(cid)
        else:
            excluded_clips.append({
                "clip_id": cid,
                "original_filename": c.get("original_filename", ""),
                "human_count": gt_map.get(cid),
                "model_failures": model_failures,
            })

    common_cohort.sort()
    n_common = len(common_cohort)
    total_eligible = len(eligible_ids)

    per_model_metrics: Dict[str, Any] = {}

    for m in model_names:
        m_completed_count = sum(
            1 for cid in eligible_ids
            if all_results[m].get(cid, {}).get("status") == "completed"
            and (all_results[m].get(cid, {}).get("exit_code") is None or all_results[m].get(cid, {}).get("exit_code") == 0)
            and all_results[m].get(cid, {}).get("completed_reps") is not None
            and not isinstance(all_results[m].get(cid, {}).get("completed_reps"), bool)
            and isinstance(all_results[m].get(cid, {}).get("completed_reps"), int)
            and all_results[m].get(cid, {}).get("completed_reps") >= 0
        )
        m_coverage_ratio = (m_completed_count / total_eligible) if total_eligible > 0 else 0.0

        if n_common == 0:
            per_model_metrics[m] = {
                "status": "not_ready",
                "evaluated_clips": 0,
                "unreviewed_clips": 0,
                "unreviewed_clip_ids": [],
                "exact_matches": 0,
                "exact_match_ratio": None,
                "mean_absolute_error": None,
                "model_completed_clips": m_completed_count,
                "model_coverage_ratio": m_coverage_ratio,
                "common_cohort_size": 0,
                "common_cohort_coverage_ratio": 0.0,
                "per_clip_results": [],
                "note": "No valid common completed cohort denominator across all compared models; metrics are null.",
            }
        else:
            preds_common = {
                cid: all_results[m][cid]["completed_reps"]
                for cid in common_cohort
            }
            gt_common = {cid: gt_map[cid] for cid in common_cohort}
            eval_dict = compute_metrics(
                predictions=preds_common,
                ground_truth=gt_common,
                reject_unknown_ground_truth=True,
            )
            eval_dict["status"] = "completed"
            eval_dict["model_completed_clips"] = m_completed_count
            eval_dict["model_coverage_ratio"] = m_coverage_ratio
            eval_dict["common_cohort_size"] = n_common
            eval_dict["common_cohort_coverage_ratio"] = (n_common / total_eligible) if total_eligible > 0 else 0.0
            per_model_metrics[m] = eval_dict

    return common_cohort, excluded_clips, per_model_metrics


# ==============================================================================
# Core Execution Engine & Staged Publication
# ==============================================================================


def execute_evaluation(
    plan_path: Union[str, Path],
    output_dir: Union[str, Path],
    *,
    interpreter: Optional[Union[str, Path]] = None,
    device: str = "cpu",
    timeout_per_clip: float = DEFAULT_TIMEOUT_PER_CLIP,
    run: bool = False,
    runner: Optional[Callable[..., Dict[str, Any]]] = None,
    gym_pose_dir: Optional[Union[str, Path]] = None,
    _seam_hook: Optional[Callable[[], None]] = None,
) -> Dict[str, Any]:
    """Execute matched video evaluation adhering to the frozen evaluation plan.

    Args:
        plan_path: Path to phase 1 plan.json.
        output_dir: Target fresh output directory for execution outputs and receipts.
        interpreter: Optional path to specific Python interpreter executable.
        device: Target inference compute device (e.g. 'cpu', '0').
        timeout_per_clip: Timeout in seconds per individual clip execution.
        run: If True, execute full inference. If False, validate configuration only.
        runner: Optional dependency-injected runner for testing.
        gym_pose_dir: Optional root directory of gym_pose package.
        _seam_hook: Optional test hook invoked immediately before publication hash recheck.

    Returns:
        Execution receipt or readiness report dictionary.
    """
    p_plan = Path(plan_path)
    p_out = Path(output_dir)

    # 1. Validate device and timeout
    dev = validate_device_string(device)
    if not math.isfinite(timeout_per_clip) or timeout_per_clip <= 0:
        raise ValueError(f"timeout_per_clip must be a positive finite number, got {timeout_per_clip}")

    # 2. Discover / validate interpreter & gym_pose root
    gp_env = os.environ.get("GYM_POSE_DIR")
    if gym_pose_dir is not None:
        target_gym_pose_dir = Path(gym_pose_dir)
    elif gp_env:
        target_gym_pose_dir = Path(gp_env)
    else:
        target_gym_pose_dir = Path(__file__).parent.parent

    reject_linked_ancestors(target_gym_pose_dir, label="Gym pose directory")
    target_gym_pose_dir = target_gym_pose_dir.resolve()

    if interpreter is None:
        int_path = resolve_default_interpreter(target_gym_pose_dir)
    else:
        int_path = validate_interpreter(interpreter)

    # 3. Freeze exact plan bytes and load/validate plan integrity and hashes
    reject_linked_ancestors(p_plan, label="Plan path")
    if not p_plan.is_file():
        raise FileNotFoundError(f"Plan file not found: {p_plan}")
    frozen_plan_bytes = p_plan.read_bytes()

    plan, plan_sha256 = load_and_validate_plan(p_plan)
    validated_models, review_root, eligible_cases = verify_plan_inputs(plan, p_plan)

    # 4. Check readiness and primary eligibility
    plan_status = plan.get("status", "not_ready")
    n_eligible = len(eligible_cases)

    # 5. Validate output directory freshness and path safety
    validate_execution_output_dir(p_out, p_plan, review_root, validated_models)

    # Readiness report (returned when --run is not specified)
    if not run:
        readiness_status = "ready" if (plan_status == "ready" and n_eligible > 0) else "not_ready"
        return {
            "execution_phase": "readiness_validation_only",
            "status": readiness_status,
            "plan_status": plan_status,
            "plan_path": str(p_plan.resolve()),
            "plan_sha256": plan_sha256,
            "review_root": str(review_root),
            "output_dir": str(p_out.resolve()),
            "interpreter": str(int_path),
            "device": dev,
            "timeout_per_clip_seconds": timeout_per_clip,
            "models_count": len(validated_models),
            "models": {k: str(v) for k, v in validated_models.items()},
            "primary_eligible_cases_count": n_eligible,
            "secondary_cases_count": len(plan.get("secondary_cases", [])),
            "excluded_cases_count": len(plan.get("excluded_cases", [])),
            "message": (
                "Configuration and plan validated successfully. Pass --run to execute inference."
                if readiness_status == "ready"
                else "Plan status is not_ready or eligible cohort is empty; execution cannot proceed."
            ),
        }

    # If run requested but plan is not ready, refuse execution honestly
    if plan_status != "ready" or n_eligible == 0:
        raise ValueError(
            f"Cannot execute evaluation: plan status is {plan_status!r} with {n_eligible} eligible cases."
        )

    # Collect initial hashes for continuous immutability checks
    frozen_hashes: Dict[Path, str] = {}
    for m_name, m_p in validated_models.items():
        exp_m_sha = plan["models"][m_name]["sha256"].strip().lower()
        curr_m_sha = compute_file_sha256(m_p)
        if curr_m_sha != exp_m_sha:
            raise PlanOrCodeMutationError(
                f"Model checkpoint hash mismatch for '{m_name}': expected {exp_m_sha}, got {curr_m_sha}"
            )
        frozen_hashes[m_p] = exp_m_sha

    for c in eligible_cases:
        cid = c["case_id"]
        c_json = review_root / cid / "case.json"
        v_file = review_root / cid / f"video{c['extension']}"
        exp_ann_sha = c["annotation_sha256"].strip().lower()
        exp_src_sha = c["source_sha256"].strip().lower()
        curr_ann_sha = compute_file_sha256(c_json)
        if curr_ann_sha != exp_ann_sha:
            raise PlanOrCodeMutationError(
                f"Annotation hash mismatch for case '{cid}': expected {exp_ann_sha}, got {curr_ann_sha}"
            )
        curr_src_sha = compute_file_sha256(v_file)
        if curr_src_sha != exp_src_sha:
            raise PlanOrCodeMutationError(
                f"Video hash mismatch for case '{cid}': expected {exp_src_sha}, got {curr_src_sha}"
            )
        frozen_hashes[c_json] = exp_ann_sha
        frozen_hashes[v_file] = exp_src_sha

    frozen_hashes[int_path] = compute_file_sha256(int_path)
    frozen_code_hashes = capture_package_code_hashes(target_gym_pose_dir)

    # Freeze selected runtime provenance BEFORE executing the first child process
    frozen_provenance = get_runtime_provenance(
        int_path, dev, target_gym_pose_dir, plan.get("declared_configuration")
    )

    runtime_keys = ("python_version", "platform", "package_versions", "tracker_path", "tracker_sha256")
    frozen_runtime = {k: frozen_provenance.get(k) for k in runtime_keys}

    def verify_immutability(stage_label: str) -> None:
        """Verify that plan bytes, models, annotations, videos, and source code have not mutated."""
        current_runtime = _capture_runtime_fingerprint(
            int_path, target_gym_pose_dir, normalize_declared_configuration(plan.get("declared_configuration"))["tracker"])
        if {k: current_runtime.get(k) for k in runtime_keys} != frozen_runtime:
            raise PlanOrCodeMutationError(f"Selected runtime or tracker configuration changed {stage_label}")
        if p_plan.read_bytes() != frozen_plan_bytes:
            raise PlanOrCodeMutationError(f"Plan file mutated {stage_label}: {p_plan} (SHA256 mismatch)")
        for f_path, exp_h in frozen_hashes.items():
            if not f_path.is_file():
                raise PlanOrCodeMutationError(f"Required file disappeared {stage_label}: {f_path}")
            curr_h = compute_file_sha256(f_path)
            if curr_h != exp_h:
                raise PlanOrCodeMutationError(
                    f"File mutated {stage_label} ({f_path}): SHA256 mismatch: expected {exp_h}, got {curr_h}"
                )
        current_code_hashes = capture_package_code_hashes(target_gym_pose_dir)
        if current_code_hashes != frozen_code_hashes:
            raise PlanOrCodeMutationError(
                f"Source code mutated {stage_label} in {target_gym_pose_dir / 'gym_analyser'}!"
            )

    # 6. Begin Staged Execution
    started_utc = datetime.datetime.now(datetime.timezone.utc).isoformat()
    p_out.parent.mkdir(parents=True, exist_ok=True)
    reject_linked_ancestors(p_out.parent, label="Output parent directory")

    stage_dir: Optional[Path] = None
    exec_runner = runner if runner is not None else default_subprocess_runner
    exception_occurred = False

    try:
        stage_dir = Path(
            tempfile.mkdtemp(prefix=f".{p_out.name}.exec-", dir=p_out.parent.resolve())
        ).resolve()

        all_results: Dict[str, Dict[str, Dict[str, Any]]] = {m: {} for m in validated_models}

        # Check immutability before any execution
        verify_immutability("before execution")

        # Sequential execution across all models and eligible cases
        for m_name in sorted(validated_models.keys()):
            m_path = validated_models[m_name]
            for c in eligible_cases:
                cid = c["case_id"]
                clip_out = stage_dir / m_name / cid
                v_path = review_root / cid / f"video{c['extension']}"
                log_path = stage_dir / "logs" / f"{m_name}_{cid}.log"

                # Check immutability before each clip run
                verify_immutability(f"before clip '{cid}' model '{m_name}'")

                runner_kwargs: Dict[str, Any] = {
                    "interpreter": int_path,
                    "gym_pose_dir": target_gym_pose_dir,
                    "video_path": v_path,
                    "output_dir": clip_out,
                    "model_path": m_path,
                    "device": dev,
                    "timeout": timeout_per_clip,
                }
                # Pass optional parameters only if accepted by custom test runners
                try:
                    sig = inspect.signature(exec_runner)
                    params = sig.parameters
                    has_varkw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())
                    if has_varkw or "log_path" in params:
                        runner_kwargs["log_path"] = log_path
                    if has_varkw or "declared_configuration" in params:
                        runner_kwargs["declared_configuration"] = plan.get("declared_configuration", {})
                    if has_varkw or "case_id" in params:
                        runner_kwargs["case_id"] = cid
                    if has_varkw or "model_name" in params:
                        runner_kwargs["model_name"] = m_name
                except (ValueError, TypeError):
                    pass

                run_res = exec_runner(**runner_kwargs)
                all_results[m_name][cid] = run_res

                # Check immutability after each clip run
                verify_immutability(f"after clip '{cid}' model '{m_name}'")

        # Check immutability before aggregation
        verify_immutability("before aggregation")

        # 7. Multi-Model Common Completed Cohort Computation
        common_cohort, excluded_clips, per_model_metrics = compute_multi_model_common_cohort(
            all_results=all_results,
            eligible_cases=eligible_cases,
        )

        overall_status = "completed" if len(common_cohort) > 0 else "not_ready"
        finished_utc = datetime.datetime.now(datetime.timezone.utc).isoformat()

        # 8. Build Receipt and Counts Artifacts
        receipt: Dict[str, Any] = {
            "receipt_version": "1.0",
            "status": overall_status,
            "started_utc": started_utc,
            "finished_utc": finished_utc,
            "plan_path": str(p_plan.resolve()),
            "plan_sha256": plan_sha256,
            "plan_created_utc": plan.get("created_utc"),
            "review_root": str(review_root),
            "output_dir": str(p_out.resolve()),
            "runtime_provenance": frozen_provenance,
            "runtime_environment": frozen_provenance,
            "models": plan.get("models", {}),
            "declared_configuration": plan.get("declared_configuration", {}),
            "eligible_cases_count": n_eligible,
            "common_completed_cohort": common_cohort,
            "common_completed_cohort_count": len(common_cohort),
            "excluded_from_common_cohort": excluded_clips,
            "per_model_metrics": per_model_metrics,
            "per_clip_predictions": {
                m: {
                    cid: {
                        "status": all_results[m][cid]["status"],
                        "completed_reps": all_results[m][cid]["completed_reps"],
                        "read_status_note": all_results[m][cid].get("read_status_note"),
                        "error": all_results[m][cid].get("error"),
                    }
                    for cid in all_results[m]
                }
                for m in sorted(all_results.keys())
            },
            "limitations_and_disclaimers": {
                "gold_standard_claim": False,
                "gold_standard_note": (
                    "Human-reviewed video labels reflect single-reviewer observations "
                    "and are not claimed as gold-standard ground truth."
                ),
                "source_independence_claim": False,
                "source_independence_note": (
                    "Source candidate videos are not claimed to be an independent test set; "
                    "unknown person and source independence."
                ),
                "penn_action_reserved_images_scored": 0,
                "penn_action_note": "None of the 100 reserved Penn Action images are scored or included.",
                "model_promotion_claim": False,
                "model_promotion_note": (
                    "Evaluation on candidate clips does not constitute automated model promotion "
                    "or clinical certification."
                ),
                "common_cohort_fairness_note": (
                    "All comparative metrics use strictly identical common fully completed clip "
                    "cohorts across models."
                ),
                "incomplete_decoding_note": (
                    "Incompletely decoded or failed clips are explicitly excluded from common denominator; "
                    "never imputed as zero."
                ),
            },
        }

        execution_counts: Dict[str, Any] = {
            "generated_utc": finished_utc,
            "plan_sha256": plan_sha256,
            "common_cohort_count": len(common_cohort),
            "eligible_cases_count": n_eligible,
            "counts": [
                {
                    "clip_id": c["case_id"],
                    "original_filename": c.get("original_filename", ""),
                    "human_count": c["review"]["human_count"],
                    "in_common_cohort": c["case_id"] in common_cohort,
                    "model_predictions": {
                        m: {
                            "predicted_count": all_results[m][c["case_id"]]["completed_reps"],
                            "status": all_results[m][c["case_id"]]["status"],
                            "read_status_note": all_results[m][c["case_id"]].get("read_status_note"),
                        }
                        for m in sorted(all_results.keys())
                    },
                }
                for c in eligible_cases
            ],
        }

        # Write artifacts into staging directory
        receipt_file = stage_dir / "evaluation_receipt.json"
        with open(receipt_file, "w", encoding="utf-8") as f:
            json.dump(receipt, f, indent=2, allow_nan=False)
            f.flush()
            os.fsync(f.fileno())

        counts_file = stage_dir / "execution_counts.json"
        with open(counts_file, "w", encoding="utf-8") as f:
            json.dump(execution_counts, f, indent=2, allow_nan=False)
            f.flush()
            os.fsync(f.fileno())

        # Publication seam hook
        if _seam_hook is not None:
            _seam_hook()

        # Re-check models, source videos, case.json, code, and plan immutability before publication
        verify_immutability("before publication")
        verify_plan_inputs(plan, p_plan)
        validate_execution_output_dir(p_out, p_plan, review_root, validated_models)

        # Atomic publication: remove empty destination if present, rename stage
        if p_out.exists():
            if any(p_out.iterdir()):
                raise FileExistsError(
                    f"Refusing to overwrite existing non-empty output directory: {p_out}"
                )
            p_out.rmdir()

        stage_dir.rename(p_out)
        stage_dir = None
        return receipt

    except Exception as exc:
        exception_occurred = True
        # Preserve diagnostic evidence and failure receipt on any exception
        if stage_dir is not None and stage_dir.exists():
            diag_base = p_out.parent / f".{p_out.name}.failed-diagnostics"
            diagnostic_dir = diag_base
            diag_counter = 1
            while diagnostic_dir.exists():
                diagnostic_dir = p_out.parent / f".{p_out.name}.failed-diagnostics-{diag_counter}"
                diag_counter += 1
            preserved_dir = None
            try:
                stage_dir.rename(diagnostic_dir)
                preserved_dir = diagnostic_dir
                stage_dir = None
            except Exception:
                try:
                    shutil.copytree(stage_dir, diagnostic_dir, dirs_exist_ok=True)
                    preserved_dir = diagnostic_dir
                except Exception:
                    preserved_dir = stage_dir

            try:
                fail_receipt = {
                    "receipt_version": "1.0",
                    "status": "failed",
                    "error": str(exc),
                    "error_type": type(exc).__name__,
                    "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    "plan_path": str(p_plan.resolve()),
                    "plan_sha256": plan_sha256,
                }
                (preserved_dir / "failure_receipt.json").write_text(
                    json.dumps(fail_receipt, indent=2), encoding="utf-8"
                )
            except Exception:
                pass
        raise
    finally:
        # Never delete stage_dir if an exception occurred!
        if not exception_occurred and stage_dir is not None and stage_dir.exists():
            shutil.rmtree(stage_dir, ignore_errors=True)


# ==============================================================================
# CLI Implementation & Report Formatter
# ==============================================================================


def format_execution_report(result: Dict[str, Any]) -> str:
    """Format evaluation execution receipt or readiness report into human-readable summary."""
    lines: List[str] = [
        "=" * 64,
        "       GYM POSE REVIEW EVALUATION EXECUTION REPORT (PHASE 2)",
        "=" * 64,
        f"Status                  : {result.get('status')}",
        f"Plan SHA256             : {result.get('plan_sha256')}",
        f"Output directory        : {result.get('output_dir')}",
    ]

    if "runtime_provenance" in result:
        prov = result["runtime_provenance"]
        lines.append(f"Interpreter             : {prov.get('interpreter')}")
        lines.append(f"Device                  : {prov.get('device')}")
        lines.append(f"Git commit              : {prov.get('git_commit')[:12] if prov.get('git_commit') != 'unavailable' else 'unavailable'}")

    if result.get("execution_phase") == "readiness_validation_only":
        lines.extend([
            f"Execution phase         : {result.get('execution_phase')}",
            "",
            f"Primary eligible cases  : {result.get('primary_eligible_cases_count')}",
            f"Models declared         : {result.get('models_count')}",
            f"Message                 : {result.get('message')}",
            "=" * 64,
        ])
        return "\n".join(lines)

    lines.extend([
        f"Started (UTC)           : {result.get('started_utc')}",
        f"Finished (UTC)          : {result.get('finished_utc')}",
        "",
        f"Primary eligible cases  : {result.get('eligible_cases_count')}",
        f"Common completed cohort : {result.get('common_completed_cohort_count')} clips",
        f"Excluded from comparison: {len(result.get('excluded_from_common_cohort', []))} clips",
        "",
        "Model Comparison (Common Completed Cohort):",
    ])

    per_model = result.get("per_model_metrics", {})
    for m_name, m_metrics in per_model.items():
        em_ratio = m_metrics.get("exact_match_ratio")
        em_pct = f"{em_ratio * 100:.1f}%" if em_ratio is not None else "N/A"
        mae = m_metrics.get("mean_absolute_error")
        mae_str = f"{mae:.2f} reps" if mae is not None else "N/A"
        cov = m_metrics.get("model_coverage_ratio", 0.0)
        lines.extend([
            f"  - Model '{m_name}':",
            f"      Exact match ratio : {em_pct} ({m_metrics.get('exact_matches', 0)}/{m_metrics.get('evaluated_clips', 0)})",
            f"      MAE               : {mae_str}",
            f"      Model coverage    : {cov * 100:.1f}% ({m_metrics.get('model_completed_clips', 0)}/{result.get('eligible_cases_count')})",
        ])

    if result.get("excluded_from_common_cohort"):
        lines.extend(["", "Excluded Clips from Common Cohort:"])
        for exc_clip in result["excluded_from_common_cohort"]:
            fail_notes = [
                f"{f['model']}: {f['status']} ({f.get('error') or f.get('read_status_note') or 'failed'})"
                for f in exc_clip.get("model_failures", [])
            ]
            lines.append(f"  - {exc_clip['clip_id']}: {'; '.join(fail_notes)}")

    lines.extend([
        "",
        "Methodological Disclaimers:",
        f"  - {result['limitations_and_disclaimers']['gold_standard_note']}",
        f"  - {result['limitations_and_disclaimers']['source_independence_note']}",
        f"  - {result['limitations_and_disclaimers']['common_cohort_fairness_note']}",
        f"  - {result['limitations_and_disclaimers']['incomplete_decoding_note']}",
        "=" * 64,
    ])
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    """Build CLI argument parser for review evaluation execution."""
    parser = argparse.ArgumentParser(
        prog="python -m gym_analyser.review_execution",
        description="Phase 2: Human-reviewed video evaluation execution engine.",
    )
    parser.add_argument(
        "--plan",
        type=Path,
        required=True,
        help="Path to pre-generated plan.json file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Fresh output directory for evaluation outputs and evaluation_receipt.json.",
    )
    parser.add_argument(
        "--interpreter",
        type=Path,
        default=None,
        help="Path to Python executable to run analyser child processes.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Inference compute device string (default: cpu).",
    )
    parser.add_argument(
        "--timeout-per-clip",
        type=float,
        default=DEFAULT_TIMEOUT_PER_CLIP,
        help=f"Execution timeout in seconds per clip (default: {DEFAULT_TIMEOUT_PER_CLIP}).",
    )
    parser.add_argument(
        "--run",
        action="store_true",
        help="Trigger actual analysis execution. If omitted, validate configuration and plan readiness.",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI execution entrypoint."""
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        result = execute_evaluation(
            plan_path=args.plan,
            output_dir=args.output_dir,
            interpreter=args.interpreter,
            device=args.device,
            timeout_per_clip=args.timeout_per_clip,
            run=args.run,
        )
        report = format_execution_report(result)
        print(report)

        if args.run:
            print(f"\nExecution receipt written to: {args.output_dir / 'evaluation_receipt.json'}")
            print(f"Execution counts written to:  {args.output_dir / 'execution_counts.json'}")

        if result.get("status") == "not_ready":
            return 1
        return 0

    except Exception as exc:
        sys.stderr.write(f"Execution error: {exc}\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
