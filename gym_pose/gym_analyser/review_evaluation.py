"""Human-reviewed video evaluation plan generator for Gym Pose Analyser.

Pure Python standard-library module and CLI for preparing evaluation plans,
validating review-store integrity, segregating primary vs secondary cohorts,
declaring fixed counting configurations, and generating evaluation receipts
without running ML inference or modifying review cases.
"""

from __future__ import annotations

import argparse
import copy
import datetime
import hashlib
import json
import logging
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Set, Tuple, Union

from .counter import CounterConfig
from .evaluation import validate_count
from .review_store import (
    CaseCorruptedError,
    CaseNotFoundError,
    InvalidCaseIdError,
    ReviewStore,
    _reject_json_constant,
    detect_duplicate_pairs,
    is_symlink,
    is_valid_uuid4,
    validate_review_payload,
)

logger = logging.getLogger(__name__)

# ==============================================================================
# Constants & Model Contract Regex
# ==============================================================================

SAFE_MODEL_NAME_PATTERN: re.Pattern[str] = re.compile(r"^[a-zA-Z0-9_\-\.]{1,64}$")
RESERVED_DEVICE_NAMES: Set[str] = {
    "con", "prn", "aux", "nul",
    "com1", "com2", "com3", "com4", "com5", "com6", "com7", "com8", "com9",
    "lpt1", "lpt2", "lpt3", "lpt4", "lpt5", "lpt6", "lpt7", "lpt8", "lpt9",
}

CHUNK_SIZE: int = 64 * 1024


# ==============================================================================
# Ancestor & Hashing & Model Validation Helpers
# ==============================================================================


def reject_linked_ancestors(path: Union[str, Path], label: str = "Path") -> None:
    """Inspect path ancestors and reject if any parent is a symlink or reparse point.

    Args:
        path: Path to inspect.
        label: Descriptive label for error messages.

    Raises:
        ValueError: If any ancestor of path is a symlink or reparse point.
    """
    abs_p = Path(path).absolute()
    for part in abs_p.parents:
        if is_symlink(part):
            raise ValueError(f"{label} has a symlink or reparse ancestor: {part}")


def compute_file_sha256(path: Union[str, Path]) -> str:
    """Stream-hash a file with SHA256 in 64KB chunks to avoid high memory usage.

    Args:
        path: Path to the target file.

    Returns:
        Hex-encoded lowercase 64-character SHA256 digest string.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If path points to a symlink, directory, or has symlinked ancestors.
    """
    p = Path(path)
    reject_linked_ancestors(p, label="Path")
    if not p.exists():
        raise FileNotFoundError(f"File not found for hashing: {p}")
    if is_symlink(p):
        raise ValueError(f"Cannot hash symlink or reparse point: {p}")
    if not p.is_file():
        raise ValueError(f"Path is not a regular file: {p}")

    hasher = hashlib.sha256()
    with open(p, "rb") as f:
        while True:
            chunk = f.read(CHUNK_SIZE)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest().lower()


def validate_model_name(name: Any) -> str:
    """Validate a safe, non-empty model identifier.

    Rules:
    - Must be a string.
    - Stripped string must equal original (no leading/trailing whitespace).
    - Length between 1 and 64 characters.
    - Allowed characters: alphanumeric, underscore, hyphen, dot.
    - Must not start with '.' or '-'.
    - Must not contain '..' or path separators ('/' or '\\').
    - Must not match Windows reserved device names (CON, NUL, COM1, etc.).

    Args:
        name: Name to validate.

    Returns:
        The validated model name string.

    Raises:
        TypeError: If name is not a string.
        ValueError: If name violates naming safety rules.
    """
    if not isinstance(name, str):
        raise TypeError(f"Model name must be a string, got {type(name).__name__}")
    name_clean = name.strip()
    if not name_clean:
        raise ValueError("Model name cannot be empty")
    if name != name_clean:
        raise ValueError(f"Model name cannot contain leading or trailing whitespace: {name!r}")
    if not SAFE_MODEL_NAME_PATTERN.match(name):
        raise ValueError(
            f"Unsafe model name {name!r}: must match ^[a-zA-Z0-9_\\-\\.]+$, max 64 chars"
        )
    if name.startswith(".") or name.startswith("-"):
        raise ValueError(f"Model name cannot start with '.' or '-': {name!r}")
    if ".." in name:
        raise ValueError(f"Model name cannot contain '..': {name!r}")
    if "/" in name or "\\" in name:
        raise ValueError(f"Model name cannot contain path separators: {name!r}")
    if name.lower() in RESERVED_DEVICE_NAMES:
        raise ValueError(f"Model name is a reserved device name: {name!r}")
    return name


def validate_models_contract(
    models: Union[Mapping[str, Union[str, Path]], Sequence[Tuple[str, Union[str, Path]]]],
) -> Dict[str, Path]:
    """Validate and normalize the models input mapping.

    Documented Models Contract:
    models must be a dictionary (Mapping) or a sequence of (name, path) pairs
    mapping safe model names to existing regular filesystem checkpoint Paths.
    Each model name must:
    - Pass validate_model_name (1-64 chars, safe characters, no traversal).
    - Be unique across the entire models mapping (no duplicates).
    Each checkpoint path must:
    - Pass reject_linked_ancestors.
    - Exist as a regular file.
    - Not be a symbolic link or reparse point.
    - Not be empty (file size > 0).

    Args:
        models: Mapping or sequence defining model name -> checkpoint path.

    Returns:
        Dict mapping validated safe model names to verified Path objects.

    Raises:
        TypeError: If models or keys have invalid types.
        ValueError: If models is empty, has duplicate names, or checkpoints are invalid.
        FileNotFoundError: If any model checkpoint file does not exist.
    """
    validated: Dict[str, Path] = {}

    if isinstance(models, Mapping):
        entries = list(models.items())
    elif isinstance(models, Sequence) and not isinstance(models, (str, bytes)):
        entries = list(models)
    else:
        raise TypeError(
            f"models contract requires a dict mapping safe names to existing Paths, "
            f"got {type(models).__name__}"
        )

    if not entries:
        raise ValueError("At least one model must be provided in models mapping")

    for entry in entries:
        if not isinstance(entry, (tuple, list)) or len(entry) != 2:
            raise ValueError(f"Each model entry must be a (name, path) pair, got {entry!r}")
        m_name, m_path = entry
        validate_model_name(m_name)
        if m_name in validated:
            raise ValueError(f"Duplicate model name detected: {m_name!r}")

        if m_path is None or (isinstance(m_path, str) and not m_path.strip()):
            raise ValueError(f"Model checkpoint path cannot be empty for model {m_name!r}")

        p = Path(m_path)
        reject_linked_ancestors(p, label=f"Model checkpoint '{m_name}'")
        if not p.exists():
            raise FileNotFoundError(f"Model checkpoint does not exist for '{m_name}': {p}")
        if is_symlink(p):
            raise ValueError(f"Model checkpoint is a symlink or reparse point for '{m_name}': {p}")
        if not p.is_file():
            raise ValueError(f"Model checkpoint is not a regular file for '{m_name}': {p}")
        if p.stat().st_size <= 0:
            raise ValueError(f"Model checkpoint file is empty for '{m_name}': {p}")

        validated[m_name] = p

    return validated


def validate_path_overlap(
    output_dir: Path,
    review_root: Path,
    models: Dict[str, Path],
) -> None:
    """Validate that output_dir does not overlap with review_root, models, or protected dirs.

    Enforces that output_dir has no symlink ancestors, is not a symlink itself,
    is strictly disjoint from package roots, test roots, model checkpoints, and review storage,
    and never overwrites existing non-empty directories.

    Args:
        output_dir: Target output directory Path.
        review_root: Human review storage root Path.
        models: Dict mapping model names to checkpoint Paths.

    Raises:
        ValueError: If output_dir overlaps with review_root, models, or protected dirs.
        NotADirectoryError: If output_dir exists and is not a directory.
        FileExistsError: If output_dir exists and is non-empty.
    """
    reject_linked_ancestors(output_dir, label="Output directory")
    reject_linked_ancestors(review_root, label="Review root")

    if is_symlink(output_dir):
        raise ValueError(f"Output directory cannot be a symlink or reparse point: {output_dir}")

    out_res = output_dir.resolve()
    rev_res = review_root.resolve()

    package = Path(__file__).resolve().parent
    app_root = package.parent
    protected_paths = [
        package,
        (app_root / "tests").resolve(),
        (app_root / "models").resolve(),
        (app_root / "test_data").resolve(),
    ]

    # Check if out_res == app_root or app_root is inside out_res
    if out_res == app_root.resolve():
        raise ValueError(f"Output overlaps protected directory: {app_root}")
    try:
        app_root.resolve().relative_to(out_res)
        raise ValueError(f"Output overlaps protected directory: {app_root}")
    except ValueError as exc:
        if "Output overlaps" in str(exc):
            raise

    # For each protected directory, if out_res == protected or out_res inside protected or protected inside out_res
    for protected in protected_paths:
        prot_res = protected.resolve()
        if out_res == prot_res:
            raise ValueError(f"Output overlaps protected directory: {protected}")
        try:
            out_res.relative_to(prot_res)
            raise ValueError(f"Output overlaps protected directory: {protected}")
        except ValueError as exc:
            if "Output overlaps" in str(exc):
                raise
        try:
            prot_res.relative_to(out_res)
            raise ValueError(f"Output overlaps protected directory: {protected}")
        except ValueError as exc:
            if "Output overlaps" in str(exc):
                raise

    if out_res == rev_res:
        raise ValueError(f"Output directory cannot be identical to review root: {output_dir}")

    try:
        out_res.relative_to(rev_res)
        raise ValueError(f"Output directory cannot be inside review root: {output_dir}")
    except ValueError as exc:
        if "cannot be inside" in str(exc):
            raise

    try:
        rev_res.relative_to(out_res)
        raise ValueError(f"Review root cannot be inside output directory: {review_root}")
    except ValueError as exc:
        if "cannot be inside" in str(exc):
            raise

    for m_name, m_path in models.items():
        m_res = m_path.resolve()
        if out_res == m_res:
            raise ValueError(
                f"Output directory cannot be identical to model checkpoint '{m_name}': {m_path}"
            )
        if out_res == m_res.parent:
            raise ValueError(
                f"Model checkpoint '{m_name}' cannot be inside output directory: {m_path}"
            )
        try:
            m_res.relative_to(out_res)
            raise ValueError(
                f"Model checkpoint '{m_name}' cannot be inside output directory: {m_path}"
            )
        except ValueError as exc:
            if "cannot be inside" in str(exc):
                raise

        try:
            out_res.relative_to(m_res)
            raise ValueError(
                f"Output directory cannot be inside model checkpoint path '{m_name}': {m_path}"
            )
        except ValueError as exc:
            if "cannot be inside" in str(exc):
                raise

    if output_dir.exists():
        if not output_dir.is_dir():
            raise NotADirectoryError(f"Output path exists and is not a directory: {output_dir}")
        if any(output_dir.iterdir()):
            raise FileExistsError(
                f"Refusing to overwrite existing non-empty output directory: {output_dir}"
            )


# ==============================================================================
# Plan Builder (Phase 1 Engine)
# ==============================================================================


def build_plan(
    review_root: Union[str, Path],
    output_dir: Union[str, Path],
    models: Union[Mapping[str, Union[str, Path]], Sequence[Tuple[str, Union[str, Path]]]],
    *,
    _seam_hook: Optional[Callable[[], None]] = None,
) -> Dict[str, Any]:
    """Build a deterministic evaluation plan and ground truth export from review storage.

    Phase 1 Only:
    - Reads review cases and computes stream SHA256 hashes of raw videos, case annotations,
      and model checkpoints.
    - Strictly rejects corrupted metadata, invalid labels, non-finite JSON constants,
      symlinks, duplicate keys, duplicate model names, and path overlap.
    - Evaluates eligibility:
      * Primary eligible: candidate + in_scope + reviewed integer count + reviewer/rawack
        + predictions_previously_seen is False.
      * Secondary: candidate + in_scope + reviewed integer count + reviewer/rawack
        + predictions_previously_seen is True (segregated for bias isolation).
      * Excluded: development group, out_of_scope, uncertain, unreviewed count,
        missing reviewer, unacknowledged raw review.
    - Saves plan.json (with status 'ready' or 'not_ready', primary metrics null) and
      reviewed_counts.json (compatible with evaluation.load_ground_truth, containing
      only primary eligible counts).
    - If primary cohort has 0 eligible cases, status is honest 'not_ready'.
    - Re-verifies case membership, all source video, annotation, and model hashes before
      publishing; mutation at the publication seam raises RuntimeError and aborts write.
    - Staged publication with no-clobber destination handling and atomic directory rename.

    Args:
        review_root: Path to the ReviewStore directory containing review cases.
        output_dir: Fresh target directory for plan.json and reviewed_counts.json.
        models: Mapping of safe model names to existing regular checkpoint Paths.
        _seam_hook: Optional test hook invoked immediately before publication hash recheck.

    Returns:
        The generated evaluation plan dictionary.

    Raises:
        FileNotFoundError: If review_root or any model checkpoint is missing.
        NotADirectoryError: If review_root or output_dir is not a directory.
        FileExistsError: If output_dir exists and is non-empty.
        ValueError: If model names, paths, or configurations fail validation.
        CaseCorruptedError: If any review case is corrupt, tampered, or invalid.
        RuntimeError: If source or model files mutate at the publication seam.
    """
    rev_path = Path(review_root)
    out_path = Path(output_dir)

    reject_linked_ancestors(rev_path, label="Review root")
    reject_linked_ancestors(out_path, label="Output directory")
    reject_linked_ancestors(out_path.parent, label="Output parent directory")

    if not rev_path.exists():
        raise FileNotFoundError(f"Review root directory does not exist: {rev_path}")
    if is_symlink(rev_path):
        raise CaseCorruptedError(f"Review root cannot be a symlink or reparse point: {rev_path}")
    if not rev_path.is_dir():
        raise NotADirectoryError(f"Review root is not a directory: {rev_path}")

    # 1. Validate models contract & path overlap
    validated_models = validate_models_contract(models)
    validate_path_overlap(out_path, rev_path, validated_models)

    # 2. Hash model checkpoints in streams
    model_metadata: Dict[str, Dict[str, Any]] = {}
    for m_name, m_ckpt in validated_models.items():
        m_sha = compute_file_sha256(m_ckpt)
        m_size = m_ckpt.stat().st_size
        model_metadata[m_name] = {
            "name": m_name,
            "path": str(m_ckpt.resolve()),
            "sha256": m_sha,
            "file_size_bytes": m_size,
        }

    # Record initial case IDs (case membership freeze)
    initial_case_ids: Set[str] = set()
    for item in rev_path.iterdir():
        if is_symlink(item):
            raise CaseCorruptedError(f"Symlink found in review root: {item.name}")
        if item.is_dir():
            initial_case_ids.add(item.name)

    # 3. Load review cases using ReviewStore with fail-closed integrity checks
    store = ReviewStore(root=rev_path)
    loaded_cases: List[Dict[str, Any]] = []
    source_annotation_hashes: Dict[str, str] = {}

    for item in sorted(rev_path.iterdir()):
        if is_symlink(item):
            raise CaseCorruptedError(f"Symlink found in review root: {item.name}")
        if not item.is_dir():
            continue
        case_id = item.name
        case_json_file = item / "case.json"
        has_case_json = case_json_file.exists()
        if not is_valid_uuid4(case_id):
            if has_case_json:
                raise CaseCorruptedError(
                    f"Directory {case_id!r} contains case.json but is not a valid UUID4 case ID"
                )
            continue

        if not has_case_json:
            raise CaseCorruptedError(f"case.json missing in case '{case_id}'")

        if is_symlink(case_json_file):
            raise CaseCorruptedError(f"case.json is a symlink or reparse point in case '{case_id}'")

        try:
            captured_bytes = case_json_file.read_bytes()
        except OSError as exc:
            raise CaseCorruptedError(f"Failed to read case.json for '{case_id}': {exc}") from exc

        ann_sha = hashlib.sha256(captured_bytes).hexdigest().lower()

        try:
            captured_dict = json.loads(
                captured_bytes.decode("utf-8"),
                object_pairs_hook=detect_duplicate_pairs,
                parse_constant=_reject_json_constant,
            )
            if not isinstance(captured_dict, dict):
                raise ValueError(
                    f"case.json in '{case_id}' must contain a JSON object, got {type(captured_dict).__name__}"
                )
        except (json.JSONDecodeError, ValueError, UnicodeDecodeError) as exc:
            raise CaseCorruptedError(
                f"Validation failed for captured case.json in case '{case_id}': {exc}"
            ) from exc

        case_data = store.load_persisted_case(case_id, verify_raw_sha=True)

        try:
            captured_dict["original_filename"] = str(captured_dict.get("original_filename", ""))
            captured_dict["review"] = validate_review_payload(captured_dict.get("review", {}))
            history_raw = captured_dict.get("history", [])
            if not isinstance(history_raw, list):
                raise ValueError(f"Stored history must be a list in case '{case_id}'")
            captured_dict["history"] = [
                {
                    "saved_at": e["saved_at"],
                    "review": validate_review_payload(e.get("review", {})),
                }
                for e in history_raw
            ]
        except (ValueError, KeyError, TypeError) as exc:
            raise CaseCorruptedError(
                f"Validation failed for captured case.json in case '{case_id}': {exc}"
            ) from exc

        for field in (
            "case_id",
            "group",
            "original_filename",
            "extension",
            "source_sha256",
            "file_size_bytes",
            "created_at",
            "updated_at",
            "review",
            "history",
        ):
            if captured_dict.get(field) != case_data.get(field):
                raise RuntimeError(
                    f"Metadata field '{field}' changed while capturing case '{case_id}'"
                )

        if compute_file_sha256(case_json_file) != ann_sha:
            raise RuntimeError(f"Annotation changed while capturing case '{case_id}'")

        captured_case = {
            "case_id": captured_dict["case_id"],
            "group": captured_dict["group"],
            "original_filename": captured_dict["original_filename"],
            "extension": captured_dict["extension"],
            "source_sha256": captured_dict["source_sha256"],
            "file_size_bytes": captured_dict["file_size_bytes"],
            "created_at": captured_dict["created_at"],
            "updated_at": captured_dict["updated_at"],
            "review": captured_dict["review"],
            "history": captured_dict["history"],
        }
        loaded_cases.append(captured_case)
        source_annotation_hashes[case_id] = ann_sha

    # 4. Cohort segregation (Primary Eligible vs Secondary Seen vs Excluded)
    eligible_cases: List[Dict[str, Any]] = []
    secondary_cases: List[Dict[str, Any]] = []
    excluded_cases: List[Dict[str, Any]] = []

    for case in loaded_cases:
        cid = case["case_id"]
        group = case["group"]
        review = case["review"]
        scope = review.get("scope", "")
        human_count = review.get("human_count")
        raw_video_reviewed = review.get("raw_video_reviewed", False)
        reviewer = review.get("reviewer", "")
        seen = review.get("predictions_previously_seen")
        ann_sha = source_annotation_hashes[cid]

        snapshot = {
            "case_id": cid,
            "original_filename": case["original_filename"],
            "extension": case["extension"],
            "source_sha256": case["source_sha256"],
            "file_size_bytes": case["file_size_bytes"],
            "group": group,
            "created_at": case["created_at"],
            "updated_at": case["updated_at"],
            "review": copy.deepcopy(review),
            "history": copy.deepcopy(case["history"]),
            "annotation_sha256": ann_sha,
            "source_independence_note": (
                "Unknown person/source independence; candidate videos are not "
                "claimed to be an independent benchmark."
            ),
        }

        reasons: List[str] = []
        if group != "candidate":
            reasons.append(f"group '{group}' is excluded (must be 'candidate')")
        if scope != "in_scope":
            reasons.append(f"scope '{scope}' is excluded (must be 'in_scope')")
        if human_count is None:
            reasons.append("human_count is null/unreviewed")
        elif isinstance(human_count, bool) or not isinstance(human_count, int) or human_count < 0:
            reasons.append(f"human_count {human_count!r} is not a valid non-negative integer")
        if raw_video_reviewed is not True:
            reasons.append("raw_video_reviewed is False (must be acknowledged)")
        if not isinstance(reviewer, str) or not reviewer.strip():
            reasons.append("reviewer identity is empty or missing")

        if reasons:
            excluded_cases.append({
                "case_id": cid,
                "reason": "; ".join(reasons),
                "group": group,
                "scope": scope,
                "human_count": human_count,
                "reviewer": reviewer,
                "raw_video_reviewed": raw_video_reviewed,
                "predictions_previously_seen": seen,
                "original_filename": case["original_filename"],
                "source_sha256": case["source_sha256"],
            })
        elif seen is True:
            # Segregated secondary: reviewer previously exposed to model predictions
            secondary_cases.append(snapshot)
        elif seen is False:
            # Primary eligible case: no prior prediction exposure
            eligible_cases.append(snapshot)
        else:
            excluded_cases.append({
                "case_id": cid,
                "reason": f"predictions_previously_seen must be an explicit boolean, got {seen!r}",
                "group": group,
                "scope": scope,
                "human_count": human_count,
                "reviewer": reviewer,
                "raw_video_reviewed": raw_video_reviewed,
                "predictions_previously_seen": seen,
                "original_filename": case["original_filename"],
                "source_sha256": case["source_sha256"],
            })

    # 5. Cohort status & declared configuration
    cohort_status = "ready" if len(eligible_cases) > 0 else "not_ready"
    created_utc = datetime.datetime.now(datetime.timezone.utc).isoformat()

    default_counter = CounterConfig()
    declared_config = {
        "counter_config": {
            "down_angle": default_counter.down_angle,
            "up_angle": default_counter.up_angle,
            "min_joint_confidence": default_counter.min_joint_confidence,
            "stable_frames": default_counter.stable_frames,
            "max_gap_seconds": default_counter.max_gap_seconds,
        },
        "departure_frames": 2,
        "tracker_name": "bytetrack.yaml",
        "runtime_inference_effective_parameters": "unavailable_explicit_pending_before_run",
    }

    limitations = {
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
        "phase": "phase1_plan_only",
        "phase2_status": "pending (full matched inference and common complete cohort metrics)",
        "no_fake_perfect_accuracy": True,
    }

    # 6. Build pre-hash plan payload
    plan_payload: Dict[str, Any] = {
        "plan_version": "1.0",
        "status": cohort_status,
        "created_utc": created_utc,
        "review_root": str(rev_path.resolve()),
        "output_dir": str(out_path.resolve()),
        "models": model_metadata,
        "declared_configuration": declared_config,
        "source_annotation_hashes": source_annotation_hashes,
        "eligible_cases": eligible_cases,
        "secondary_cases": secondary_cases,
        "excluded_cases": excluded_cases,
        "metrics": None,
        "limitations_and_disclaimers": limitations,
    }

    canonical_bytes = json.dumps(
        plan_payload, sort_keys=True, indent=2, allow_nan=False
    ).encode("utf-8")
    plan_sha256 = hashlib.sha256(canonical_bytes).hexdigest().lower()

    final_plan = dict(plan_payload)
    final_plan["plan_sha256"] = plan_sha256

    reviewed_counts_dict: Dict[str, Any] = {
        "checkpoint": "evaluation_plan",
        "generated_at": created_utc,
        "description": "Human-reviewed ground truth repetition counts for primary eligible cohort.",
        "clips": [
            {
                "clip_id": c["case_id"],
                "human_count": c["review"]["human_count"],
                "scope": c["review"]["scope"],
                "group": c["group"],
                "original_filename": c.get("original_filename", ""),
                "source_sha256": c.get("source_sha256", ""),
            }
            for c in eligible_cases
        ],
    }

    # 7. Staged Publication with No-Clobber Destination Handling
    stage_dir: Optional[Path] = None
    try:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        reject_linked_ancestors(out_path.parent, label="Output parent directory")
        parent_res = out_path.parent.resolve()
        stage_dir = Path(
            tempfile.mkdtemp(prefix=f".{out_path.name}.plan-", dir=parent_res)
        ).resolve()

        counts_file = stage_dir / "reviewed_counts.json"
        with open(counts_file, "w", encoding="utf-8") as f:
            json.dump(reviewed_counts_dict, f, indent=2, allow_nan=False)
            f.flush()
            os.fsync(f.fileno())

        plan_file = stage_dir / "plan.json"
        with open(plan_file, "w", encoding="utf-8") as f:
            json.dump(final_plan, f, indent=2, allow_nan=False)
            f.flush()
            os.fsync(f.fileno())

        # Publication seam hook
        if _seam_hook is not None:
            _seam_hook()

        # Recheck case membership
        for item in rev_path.iterdir():
            if is_symlink(item):
                raise CaseCorruptedError(f"Symlink found in review root: {item.name}")
        current_case_ids = {item.name for item in rev_path.iterdir() if item.is_dir()}
        if current_case_ids != initial_case_ids:
            raise RuntimeError(
                f"Review store case membership changed before publication: "
                f"added={sorted(current_case_ids - initial_case_ids)}, "
                f"removed={sorted(initial_case_ids - current_case_ids)}"
            )

        # Recheck source raw video hashes
        for case in loaded_cases:
            cid = case["case_id"]
            v_path = rev_path / cid / f"video{case['extension']}"
            actual_v_sha = compute_file_sha256(v_path)
            if actual_v_sha != case["source_sha256"]:
                raise RuntimeError(
                    f"Source video hash verification failed before publishing plan for case '{cid}': "
                    f"expected {case['source_sha256']}, got {actual_v_sha}"
                )

        # Recheck case.json annotation hashes
        for cid, exp_ann_sha in source_annotation_hashes.items():
            case_json_p = rev_path / cid / "case.json"
            actual_ann_sha = compute_file_sha256(case_json_p)
            if actual_ann_sha != exp_ann_sha:
                raise RuntimeError(
                    f"Annotation hash verification failed before publishing plan for case '{cid}': "
                    f"expected {exp_ann_sha}, got {actual_ann_sha}"
                )

        # Recheck model checkpoint hashes and sizes
        for m_name, m_path in validated_models.items():
            exp_m_sha = model_metadata[m_name]["sha256"]
            actual_m_sha = compute_file_sha256(m_path)
            if actual_m_sha != exp_m_sha:
                raise RuntimeError(
                    f"Model checkpoint hash verification failed before publishing plan for model '{m_name}': "
                    f"expected {exp_m_sha}, got {actual_m_sha}"
                )
            exp_m_size = model_metadata[m_name]["file_size_bytes"]
            actual_m_size = m_path.stat().st_size
            if actual_m_size != exp_m_size:
                raise RuntimeError(
                    f"Model checkpoint size verification failed before publishing plan for model '{m_name}': "
                    f"expected {exp_m_size}, got {actual_m_size}"
                )

        # Re-check path overlap
        validate_path_overlap(out_path, rev_path, validated_models)

        # If destination exists: refuse if non-empty, rmdir if empty
        if out_path.exists():
            if any(out_path.iterdir()):
                raise FileExistsError(
                    f"Refusing to overwrite existing non-empty output directory: {out_path}"
                )
            out_path.rmdir()

        stage_dir.rename(out_path)
        stage_dir = None

    finally:
        if stage_dir is not None and stage_dir.exists():
            shutil.rmtree(stage_dir, ignore_errors=True)

    return final_plan


# ==============================================================================
# CLI Implementation
# ==============================================================================


def format_plan_report(plan: Dict[str, Any]) -> str:
    """Format evaluation plan metadata into a human-readable CLI summary."""
    lines: List[str] = [
        "=" * 64,
        "          GYM POSE REVIEW EVALUATION PLAN (PHASE 1)",
        "=" * 64,
        f"Status                  : {plan['status']}",
        f"Created (UTC)           : {plan['created_utc']}",
        f"Plan SHA256             : {plan['plan_sha256']}",
        f"Review root             : {plan['review_root']}",
        f"Output directory        : {plan['output_dir']}",
        f"Models declared ({len(plan['models'])}):",
    ]
    for m_name, m_info in plan["models"].items():
        lines.append(f"  - {m_name}: {m_info['path']} (sha256: {m_info['sha256'][:16]}...)")

    lines.extend([
        "",
        f"Eligible cases (primary): {len(plan['eligible_cases'])}",
        f"Secondary cases (seen)  : {len(plan['secondary_cases'])}",
        f"Excluded cases          : {len(plan['excluded_cases'])}",
        "",
        "Declared configuration  :",
        f"  - Knee flexion down   : {plan['declared_configuration']['counter_config']['down_angle']} deg",
        f"  - Knee extension up   : {plan['declared_configuration']['counter_config']['up_angle']} deg",
        f"  - Min joint confidence: {plan['declared_configuration']['counter_config']['min_joint_confidence']}",
        f"  - Stable frames       : {plan['declared_configuration']['counter_config']['stable_frames']}",
        f"  - Max gap seconds     : {plan['declared_configuration']['counter_config']['max_gap_seconds']} s",
        f"  - Departure frames    : {plan['declared_configuration']['departure_frames']}",
        f"  - Tracker name        : {plan['declared_configuration']['tracker_name']}",
        f"  - Runtime parameters  : {plan['declared_configuration']['runtime_inference_effective_parameters']}",
        "",
        f"Metrics                 : {plan['metrics']} (Phase 1 plan only; no inference run)",
        "",
        "Methodological Disclaimers:",
        f"  - {plan['limitations_and_disclaimers']['gold_standard_note']}",
        f"  - {plan['limitations_and_disclaimers']['source_independence_note']}",
        f"  - {plan['limitations_and_disclaimers']['penn_action_note']}",
        f"  - Phase 2 status: {plan['limitations_and_disclaimers']['phase2_status']}",
        "=" * 64,
    ])
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    """Build CLI argument parser for review evaluation planning."""
    parser = argparse.ArgumentParser(
        prog="python -m gym_analyser.review_evaluation",
        description="Phase 1: Human-reviewed video evaluation plan generator.",
    )
    parser.add_argument(
        "--review-root",
        type=Path,
        required=True,
        help="Path to ReviewStore directory containing review cases.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Fresh output directory for evaluation plan and reviewed counts.",
    )
    parser.add_argument(
        "--model",
        action="append",
        dest="models",
        required=True,
        help="Model checkpoint mapping in NAME=EXISTING_PT format (repeatable).",
    )
    parser.add_argument(
        "--plan",
        action="store_true",
        help="Generate phase 1 evaluation plan.",
    )
    parser.add_argument(
        "--run",
        action="store_true",
        help="Execute full evaluation (unsupported in phase 1).",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI execution entrypoint."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.run:
        sys.stderr.write(
            "Evaluation error: --run is not supported in Phase 1. "
            "Full matched inference and common complete cohort metrics are pending in Phase 2. "
            "Use --plan to generate an evaluation plan.\n"
        )
        return 1

    if not args.plan:
        sys.stderr.write(
            "Evaluation error: Must specify --plan to generate an evaluation plan. "
            "(Phase 2 --run is pending implementation.)\n"
        )
        return 1

    try:
        models_dict: Dict[str, Path] = {}
        for m_arg in args.models:
            name, sep, path_str = m_arg.partition("=")
            if not sep or not name.strip() or not path_str.strip():
                sys.stderr.write(
                    f"Configuration error: Invalid --model format {m_arg!r}; expected NAME=EXISTING_PT\n"
                )
                return 1
            name_clean = name.strip()
            if name_clean in models_dict:
                sys.stderr.write(
                    f"Configuration error: Duplicate model name detected: {name_clean!r}\n"
                )
                return 1
            models_dict[name_clean] = Path(path_str.strip())

        plan = build_plan(
            review_root=args.review_root,
            output_dir=args.output_dir,
            models=models_dict,
        )
        report = format_plan_report(plan)
        print(report)
        print(f"\nPlan written to: {args.output_dir / 'plan.json'}")
        print(f"Reviewed counts written to: {args.output_dir / 'reviewed_counts.json'}")
        return 0

    except Exception as exc:
        sys.stderr.write(f"Evaluation error: {exc}\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
