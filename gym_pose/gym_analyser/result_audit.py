"""Pure standard library cross-artifact consistency auditor for gym_analyser outputs.

Validates mutual consistency, chronological monotonicities, attempt identifiers,
and telemetry availability across saved analysis artifacts (summary.json, attempts.json,
posture_summary.json, feedback_observations.json, feedback_phases.json,
coaching_feedback.json, angles.csv, and media files).

Enforces:
- 100% read-only input immutability with pre/post SHA-256 byte verification on consumed
  JSON/CSV artifacts and stat immutability on media files.
- Rejection of symlinks, directory junctions, reparse points, and path traversal,
  checking unresolved ancestors before path resolution.
- Honest telemetry availability: absent optional artifacts or null coaching feedback
  are reported honestly as UNAVAILABLE (never assumed healthy, zero, or corrupt).
- Wrong type or malformed supplied optional payloads are reported as ERRORS,
  never silently ignored.
- Decoded prefix clarity: incomplete runs (status: 'incomplete' or processed_frames < declared_frames)
  are strictly distinguished from whole-video completion.
- Rejection of duplicate JSON keys, duplicate attempt IDs, and non-standard constants (NaN/Inf).
- True no-clobber policy on optional audit report destinations using exclusive creation
  (os.O_CREAT | os.O_EXCL | os.O_WRONLY) without final replace over intervening files.
- Explicit non-inferential safety boundary: internal consistency does NOT establish
  human ground truth, clinical safety, movement intent, or technique correctness.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import stat
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple, Union

AUDIT_VERSION = "1.1.0"

# Explicit non-inferential clinical and safety disclaimer
DISCLAIMER_TEXT = (
    "Internal consistency verification confirms mutual artifact agreement, schema validity, "
    "and chronological monotonicity only. It does NOT establish human ground truth, clinical safety, "
    "exercise intent, or technique correctness."
)

# Standard expected artifacts in gym_analyser outputs
REQUIRED_ARTIFACTS: Tuple[str, ...] = (
    "summary.json",
    "attempts.json",
)

OPTIONAL_ARTIFACTS: Tuple[str, ...] = (
    "posture_summary.json",
    "feedback_observations.json",
    "feedback_phases.json",
    "coaching_feedback.json",
    "angles.csv",
    "angle_plot.png",
    "annotated.mp4",
    "preview.mp4",
)

CONSUMED_EXTENSIONS: Set[str] = {".json", ".csv"}

STATUS_PASSED = "passed"
STATUS_WARNING = "warning"
STATUS_FAILED = "failed"

CHECK_PASSED = "PASSED"
CHECK_WARNING = "WARNING"
CHECK_FAILED = "FAILED"
CHECK_UNAVAILABLE = "UNAVAILABLE"

SEVERITY_ERROR = "ERROR"
SEVERITY_WARNING = "WARNING"


# ---------------------------------------------------------------------------
# Path and Security Helpers
# ---------------------------------------------------------------------------

def is_symlink_or_reparse(path: Union[str, Path]) -> bool:
    """Check if path is a symlink or Windows directory junction / reparse point."""
    try:
        p = Path(path)
        if p.is_symlink():
            return True
        st = os.lstat(p)
        if hasattr(st, "st_file_attributes"):
            # FILE_ATTRIBUTE_REPARSE_POINT = 0x0400
            if bool(st.st_file_attributes & 0x0400):
                return True
    except (OSError, ValueError):
        pass
    return False


def check_unresolved_ancestors_safety(path: Union[str, Path]) -> Tuple[bool, Optional[str]]:
    """Walk unresolved path ancestors from root down to target, checking for symlinks/reparse points.

    Must be executed BEFORE calling path.resolve() to prevent following linked ancestors.
    """
    try:
        p = Path(path)
        parts = p.parts
        if not parts:
            return True, None
        current = Path(parts[0])
        for part in parts[1:]:
            current = current / part
            if current.exists() and is_symlink_or_reparse(current):
                return False, f"Path component is a symlink or reparse point: {current}"
        return True, None
    except Exception as exc:
        return False, f"Failed checking path safety: {exc}"


def paths_overlap(p1: Path, p2: Path) -> bool:
    """Check if p1 is inside p2 or p2 is inside p1."""
    try:
        p1.relative_to(p2)
        return True
    except ValueError:
        pass
    try:
        p2.relative_to(p1)
        return True
    except ValueError:
        pass
    return False


def compute_sha256(filepath: Path) -> str:
    """Compute SHA-256 hash of a file reading in 64KB blocks."""
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while True:
            chunk = f.read(65536)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


# ---------------------------------------------------------------------------
# Strict JSON and Primitive Validation
# ---------------------------------------------------------------------------

def load_json_strict(filepath: Path) -> Tuple[Optional[Any], Optional[str]]:
    """Parse JSON with rejection of duplicate keys and non-standard constants (NaN/Inf).

    Returns:
        (data, None) on success, or (None, error_message) on failure.
    """
    def _pairs_hook(pairs: List[Tuple[str, Any]]) -> Dict[str, Any]:
        d: Dict[str, Any] = {}
        for k, v in pairs:
            if k in d:
                raise ValueError(f"Duplicate JSON key detected: {k!r}")
            d[k] = v
        return d

    def _reject_constant(c: str) -> None:
        raise ValueError(f"Non-standard JSON constant rejected: {c!r}")

    try:
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
        parsed = json.loads(
            content,
            object_pairs_hook=_pairs_hook,
            parse_constant=_reject_constant,
        )
        return parsed, None
    except Exception as exc:
        return None, str(exc)


def is_strict_int(val: Any) -> bool:
    """Check if value is an integer and strictly not a boolean."""
    return type(val) is int and not isinstance(val, bool)


def is_finite_number(val: Any) -> bool:
    """Check if value is a real number (int or float, not bool) and finite."""
    if isinstance(val, bool):
        return False
    if not isinstance(val, (int, float)):
        return False
    return math.isfinite(val)


# ---------------------------------------------------------------------------
# Result Auditor Class
# ---------------------------------------------------------------------------

class ResultAuditor:
    """Read-only auditor for a single saved gym_analyser results directory."""

    def __init__(
        self,
        results_dir: Union[str, Path],
        strict: bool = False,
        quiet: bool = False,
    ) -> None:
        self.raw_results_dir = Path(results_dir)
        self.strict = bool(strict)
        self.quiet = bool(quiet)
        self.checks: List[Dict[str, Any]] = []
        self.issues: List[Dict[str, Any]] = []
        self.artifacts_inventory: Dict[str, Dict[str, Any]] = {}
        self.initial_consumed_hashes: Dict[str, str] = {}
        self.initial_other_stats: Dict[str, Tuple[int, int]] = {}
        self.is_incomplete_prefix = False
        self.prefix_details: Dict[str, Any] = {}

    def _add_check(
        self,
        check_id: str,
        category: str,
        status: str,
        message: str,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.checks.append({
            "check_id": check_id,
            "category": category,
            "status": status,
            "message": message,
            "details": details or {},
        })

    def _add_issue(self, check_id: str, severity: str, message: str) -> None:
        self.issues.append({
            "check_id": check_id,
            "severity": severity,
            "message": message,
        })

    def audit(self) -> Dict[str, Any]:
        """Execute full read-only audit of the target results directory."""
        # 1. Path Safety & Directory Resolution (checking unresolved ancestors first)
        if not self._check_directory_safety():
            return self._build_report(status=STATUS_FAILED)

        # 2. Record initial hashes/stats for byte-for-byte read-only verification
        self._record_input_baselines()

        # 3. Artifact Inventory & Strict Ingestion
        parsed_data = self._ingest_artifacts()

        # If required anchor artifacts are missing/corrupted or invalid shapes, stop further cross-checks
        has_fatal_anchor_error = any(
            iss["severity"] == SEVERITY_ERROR
            and iss["check_id"] in ("CHK_REQUIRED_ARTIFACTS", "CHK_JSON_SYNTAX")
            for iss in self.issues
        )
        if has_fatal_anchor_error:
            self._verify_input_immutability()
            return self._build_report(status=STATUS_FAILED)

        summary_json = parsed_data.get("summary.json")
        attempts_json = parsed_data.get("attempts.json")
        posture_summary_json = parsed_data.get("posture_summary.json")
        feedback_obs_json = parsed_data.get("feedback_observations.json")
        feedback_phases_json = parsed_data.get("feedback_phases.json")
        coaching_json = parsed_data.get("coaching_feedback.json")
        csv_rows = parsed_data.get("angles.csv")

        # 4. Run Status & Decoded Prefix Clarity
        self._check_run_status_and_prefix(summary_json)

        # 5. Validate Optional Artifact Schemas (malformed shapes must be ERRORS)
        self._validate_optional_schemas(
            posture_summary_json=posture_summary_json,
            feedback_obs_json=feedback_obs_json,
            feedback_phases_json=feedback_phases_json,
            coaching_json=coaching_json,
        )

        # 6. Cross-Artifact Repetition & Attempt Count Agreement
        self._check_repetition_counts(
            summary_json=summary_json,
            attempts_json=attempts_json,
            feedback_obs_json=feedback_obs_json,
            feedback_phases_json=feedback_phases_json,
            coaching_json=coaching_json,
            csv_rows=csv_rows,
        )

        # 7. Attempt / Repetition ID References and Sequence Continuity
        self._check_attempt_id_integrity(
            attempts_json=attempts_json,
            feedback_obs_json=feedback_obs_json,
            feedback_phases_json=feedback_phases_json,
            coaching_json=coaching_json,
            csv_rows=csv_rows,
        )

        # 8. Valid Chronological Timestamp Monotonicity and Ordering
        self._check_chronological_ordering(
            attempts_json=attempts_json,
            feedback_phases_json=feedback_phases_json,
            coaching_json=coaching_json,
            csv_rows=csv_rows,
        )

        # 9. Configuration & Source Metadata Consistency
        self._check_configuration_identity(
            summary_json=summary_json,
            attempts_json=attempts_json,
        )

        # 10. Biomechanical and Scope Limitation Disclaimers
        self._check_limitation_disclaimers(
            summary_json=summary_json,
            attempts_json=attempts_json,
            coaching_json=coaching_json,
        )

        # 11. Verify 100% byte-for-byte read-only input immutability
        self._verify_input_immutability()

        # 12. Determine overall status
        has_errors = any(iss["severity"] == SEVERITY_ERROR for iss in self.issues)
        has_warnings = any(iss["severity"] == SEVERITY_WARNING for iss in self.issues)

        if has_errors:
            overall_status = STATUS_FAILED
        elif has_warnings and self.strict:
            overall_status = STATUS_FAILED
        elif has_warnings:
            overall_status = STATUS_WARNING
        else:
            overall_status = STATUS_PASSED

        return self._build_report(status=overall_status)

    # -----------------------------------------------------------------------
    # Step 1: Directory Safety
    # -----------------------------------------------------------------------

    def _check_directory_safety(self) -> bool:
        check_id = "CHK_PATH_SAFETY"

        # Check unresolved ancestors BEFORE resolving path
        safe_ancestors, err_ancestor = check_unresolved_ancestors_safety(self.raw_results_dir)
        if not safe_ancestors:
            self._add_check(
                check_id, "safety", CHECK_FAILED,
                f"Unresolved ancestor safety violation: {err_ancestor}",
            )
            self._add_issue(check_id, SEVERITY_ERROR, f"Ancestor symlink/reparse point detected: {err_ancestor}")
            return False

        if not self.raw_results_dir.exists():
            self._add_check(
                check_id, "safety", CHECK_FAILED,
                f"Results directory does not exist: {self.raw_results_dir}",
            )
            self._add_issue(check_id, SEVERITY_ERROR, f"Directory does not exist: {self.raw_results_dir}")
            return False

        if not self.raw_results_dir.is_dir():
            self._add_check(
                check_id, "safety", CHECK_FAILED,
                f"Target path is not a directory: {self.raw_results_dir}",
            )
            self._add_issue(check_id, SEVERITY_ERROR, f"Target path is not a directory: {self.raw_results_dir}")
            return False

        if is_symlink_or_reparse(self.raw_results_dir):
            self._add_check(
                check_id, "safety", CHECK_FAILED,
                f"Results directory is a symlink or directory junction: {self.raw_results_dir}",
            )
            self._add_issue(check_id, SEVERITY_ERROR, "Results directory is a symlink/reparse point (rejected)")
            return False

        # Scan all items for symlinks / junctions
        resolved_root = self.raw_results_dir.resolve()
        for root, dirs, files in os.walk(self.raw_results_dir):
            root_path = Path(root)
            if is_symlink_or_reparse(root_path):
                self._add_check(check_id, "safety", CHECK_FAILED, f"Subdirectory is a symlink: {root_path}")
                self._add_issue(check_id, SEVERITY_ERROR, f"Subdirectory is a symlink: {root_path}")
                return False
            for d in dirs:
                dp = root_path / d
                if is_symlink_or_reparse(dp):
                    self._add_check(check_id, "safety", CHECK_FAILED, f"Directory entry is a symlink: {dp}")
                    self._add_issue(check_id, SEVERITY_ERROR, f"Directory entry is a symlink: {dp}")
                    return False
            for f in files:
                fp = root_path / f
                if is_symlink_or_reparse(fp):
                    self._add_check(check_id, "safety", CHECK_FAILED, f"File entry is a symlink: {fp}")
                    self._add_issue(check_id, SEVERITY_ERROR, f"File entry is a symlink: {fp}")
                    return False
                try:
                    if not fp.resolve().is_relative_to(resolved_root):
                        self._add_check(check_id, "safety", CHECK_FAILED, f"File resolves outside root: {fp}")
                        self._add_issue(check_id, SEVERITY_ERROR, f"Out-of-root path traversal detected: {fp}")
                        return False
                except AttributeError:
                    try:
                        fp.resolve().relative_to(resolved_root)
                    except ValueError:
                        self._add_check(check_id, "safety", CHECK_FAILED, f"File resolves outside root: {fp}")
                        self._add_issue(check_id, SEVERITY_ERROR, f"Out-of-root path traversal detected: {fp}")
                        return False

        self._add_check(check_id, "safety", CHECK_PASSED, "Directory is safe; no symlinks or path traversal detected")
        return True

    # -----------------------------------------------------------------------
    # Step 2: Input Immutability Baselines (Precise Input Hashing)
    # -----------------------------------------------------------------------

    def _record_input_baselines(self) -> None:
        """Hash precisely consumed inputs (JSONs, CSV) and stat-check media files."""
        self.initial_consumed_hashes.clear()
        self.initial_other_stats.clear()
        for root, _, files in os.walk(self.raw_results_dir):
            for f in sorted(files):
                fp = Path(root) / f
                rel_path = str(fp.relative_to(self.raw_results_dir)).replace("\\", "/")
                suffix = fp.suffix.lower()
                st = fp.stat()
                if suffix in CONSUMED_EXTENSIONS:
                    self.initial_consumed_hashes[rel_path] = compute_sha256(fp)
                else:
                    self.initial_other_stats[rel_path] = (st.st_size, st.st_mtime_ns)

    def _verify_input_immutability(self) -> None:
        check_id = "CHK_INPUT_IMMUTABILITY"
        current_consumed: Dict[str, str] = {}
        current_other: Dict[str, Tuple[int, int]] = {}

        for root, _, files in os.walk(self.raw_results_dir):
            for f in sorted(files):
                fp = Path(root) / f
                rel_path = str(fp.relative_to(self.raw_results_dir)).replace("\\", "/")
                suffix = fp.suffix.lower()
                st = fp.stat()
                if suffix in CONSUMED_EXTENSIONS:
                    current_consumed[rel_path] = compute_sha256(fp)
                else:
                    current_other[rel_path] = (st.st_size, st.st_mtime_ns)

        all_initial = set(self.initial_consumed_hashes.keys()) | set(self.initial_other_stats.keys())
        all_current = set(current_consumed.keys()) | set(current_other.keys())

        added = all_current - all_initial
        removed = all_initial - all_current

        modified_consumed = [
            k for k in self.initial_consumed_hashes
            if k in current_consumed and self.initial_consumed_hashes[k] != current_consumed[k]
        ]
        modified_other = [
            k for k in self.initial_other_stats
            if k in current_other and self.initial_other_stats[k] != current_other[k]
        ]

        if added or removed or modified_consumed or modified_other:
            msg = (
                f"Audit mutated target directory! Added: {sorted(added)}, Removed: {sorted(removed)}, "
                f"Modified consumed: {modified_consumed}, Modified other: {modified_other}"
            )
            self._add_check(check_id, "safety", CHECK_FAILED, msg)
            self._add_issue(check_id, SEVERITY_ERROR, msg)
        else:
            self._add_check(
                check_id,
                "safety",
                CHECK_PASSED,
                f"Read-only immutability verified: {len(self.initial_consumed_hashes)} consumed JSON/CSV files "
                f"SHA-256 byte-verified, {len(self.initial_other_stats)} media/other files stat-verified",
                details={
                    "consumed_files_hashed": len(self.initial_consumed_hashes),
                    "other_files_stat_checked": len(self.initial_other_stats),
                },
            )

    # -----------------------------------------------------------------------
    # Step 3: Artifact Inventory & Strict Ingestion
    # -----------------------------------------------------------------------

    def _ingest_artifacts(self) -> Dict[str, Any]:
        parsed_data: Dict[str, Any] = {}
        all_expected = list(REQUIRED_ARTIFACTS) + list(OPTIONAL_ARTIFACTS)

        for artifact_name in all_expected:
            file_path = self.raw_results_dir / artifact_name
            if not file_path.is_file():
                self.artifacts_inventory[artifact_name] = {
                    "present": False,
                    "status": CHECK_UNAVAILABLE,
                    "size_bytes": 0,
                    "sha256": None,
                }
                continue

            size_bytes = file_path.stat().st_size
            sha256_val = self.initial_consumed_hashes.get(artifact_name)

            # Ingest JSON files strictly
            if artifact_name.endswith(".json"):
                data, err = load_json_strict(file_path)
                if err is not None:
                    self.artifacts_inventory[artifact_name] = {
                        "present": True,
                        "status": "CORRUPT",
                        "size_bytes": size_bytes,
                        "sha256": sha256_val,
                    }
                    self._add_check(
                        "CHK_JSON_SYNTAX",
                        "syntax",
                        CHECK_FAILED,
                        f"Corrupt JSON or syntax error in {artifact_name}: {err}",
                    )
                    self._add_issue(
                        "CHK_JSON_SYNTAX",
                        SEVERITY_ERROR,
                        f"Syntax error in {artifact_name}: {err}",
                    )
                else:
                    # Check for null telemetry vs valid payload
                    if data is None and artifact_name == "coaching_feedback.json":
                        # Optional absent/null telemetry: coaching_feedback.json permitted to be null/unavailable
                        self.artifacts_inventory[artifact_name] = {
                            "present": True,
                            "status": CHECK_UNAVAILABLE,
                            "size_bytes": size_bytes,
                            "sha256": sha256_val,
                            "note": "null payload (unavailable telemetry)",
                        }
                    else:
                        self.artifacts_inventory[artifact_name] = {
                            "present": True,
                            "status": "AVAILABLE",
                            "size_bytes": size_bytes,
                            "sha256": sha256_val,
                        }
                    parsed_data[artifact_name] = data

            # Ingest CSV file strictly
            elif artifact_name.endswith(".csv"):
                self.artifacts_inventory[artifact_name] = {
                    "present": True,
                    "status": "AVAILABLE",
                    "size_bytes": size_bytes,
                    "sha256": sha256_val,
                }
                try:
                    rows: List[Dict[str, str]] = []
                    with open(file_path, "r", encoding="utf-8", newline="") as cf:
                        reader = csv.DictReader(cf)
                        for r in reader:
                            rows.append(r)
                    parsed_data[artifact_name] = rows
                except Exception as exc:
                    self._add_check(
                        "CHK_CSV_SYNTAX",
                        "syntax",
                        CHECK_FAILED,
                        f"Corrupt CSV in {artifact_name}: {exc}",
                    )
                    self._add_issue(
                        "CHK_CSV_SYNTAX",
                        SEVERITY_ERROR,
                        f"CSV parsing error in {artifact_name}: {exc}",
                    )
            else:
                # Media or non-JSON files
                self.artifacts_inventory[artifact_name] = {
                    "present": True,
                    "status": "AVAILABLE",
                    "size_bytes": size_bytes,
                    "sha256": sha256_val,
                }

        # Check required anchor artifacts existence and valid dict type
        check_id = "CHK_REQUIRED_ARTIFACTS"
        missing_req = [
            req for req in REQUIRED_ARTIFACTS
            if not self.artifacts_inventory.get(req, {}).get("present", False)
        ]
        if missing_req:
            msg = f"Missing required analysis artifact(s): {', '.join(missing_req)}"
            self._add_check(check_id, "presence", CHECK_FAILED, msg)
            self._add_issue(check_id, SEVERITY_ERROR, msg)
            return parsed_data

        # Required anchors must be non-null dicts (JSON objects)
        anchor_shape_errors: List[str] = []
        for req in REQUIRED_ARTIFACTS:
            val = parsed_data.get(req)
            if not isinstance(val, dict):
                anchor_shape_errors.append(
                    f"{req} must be a valid JSON object (dict), got {type(val).__name__} (value: {val!r})"
                )

        if anchor_shape_errors:
            msg = f"Invalid required anchor artifact shape(s): {'; '.join(anchor_shape_errors)}"
            self._add_check(check_id, "presence", CHECK_FAILED, msg)
            self._add_issue(check_id, SEVERITY_ERROR, msg)
        else:
            self._add_check(
                check_id,
                "presence",
                CHECK_PASSED,
                "All required anchor artifacts are present and valid JSON objects",
            )

        return parsed_data

    # -----------------------------------------------------------------------
    # Step 4: Run Status & Decoded Prefix Clarity
    # -----------------------------------------------------------------------

    def _check_run_status_and_prefix(self, summary_json: Optional[Dict[str, Any]]) -> None:
        check_id = "CHK_RUN_STATUS"
        if not isinstance(summary_json, dict):
            self._add_check(check_id, "execution", CHECK_UNAVAILABLE, "summary.json not available")
            return

        status = summary_json.get("status")
        processed_frames = summary_json.get("processed_frames")
        source_video = summary_json.get("source_video") or {}
        declared_frames = (
            source_video.get("declared_frame_count")
            if isinstance(source_video, dict)
            else None
        )
        read_status_note = summary_json.get("read_status_note", "")

        # 1. Run status validity check
        if status is None or not isinstance(status, str):
            msg = f"summary.json missing valid 'status' string, got: {status!r}"
            self._add_check(check_id, "execution", CHECK_FAILED, msg)
            self._add_issue(check_id, SEVERITY_ERROR, msg)
            return

        status_clean = status.strip().lower()

        # Explicitly failed or interrupted run status must FAIL
        if status_clean in ("failed", "interrupted"):
            msg = f"Pipeline execution declared failure or interruption in summary.json: status={status!r}"
            self._add_check(check_id, "execution", CHECK_FAILED, msg, details={"status": status})
            self._add_issue(check_id, SEVERITY_ERROR, msg)
            return

        # Unknown status must FAIL
        if status_clean not in ("completed", "incomplete"):
            msg = f"summary.json declared unrecognized or unsupported run status: {status!r}"
            self._add_check(check_id, "execution", CHECK_FAILED, msg, details={"status": status})
            self._add_issue(check_id, SEVERITY_ERROR, msg)
            return

        # 2. Decoded Prefix Clarity
        prefix_check_id = "CHK_DECODED_PREFIX"
        is_incomplete = False
        notes: List[str] = []

        if status_clean == "incomplete":
            is_incomplete = True
            notes.append("summary.json explicitly declares status: 'incomplete'")

        if (
            is_strict_int(declared_frames)
            and is_strict_int(processed_frames)
            and declared_frames > 0
            and processed_frames < declared_frames
        ):
            is_incomplete = True
            notes.append(
                f"Decoded prefix truncation: processed {processed_frames} of {declared_frames} declared container frames"
            )

        self.is_incomplete_prefix = is_incomplete
        self.prefix_details = {
            "status": status,
            "is_incomplete_prefix": is_incomplete,
            "declared_frame_count": declared_frames,
            "processed_frames": processed_frames,
            "read_status_note": read_status_note,
        }

        if not is_incomplete and (not is_strict_int(processed_frames) or processed_frames <= 0
                                  or not is_strict_int(declared_frames) or declared_frames <= 0):
            self._add_check(prefix_check_id, "prefix", CHECK_UNAVAILABLE,
                            "Run reports completion, but valid processed/declared frame counts are unavailable",
                            details=self.prefix_details)
            return

        if is_incomplete:
            msg = f"Decoded prefix run detected: {'; '.join(notes)}. Status note: {read_status_note}"
            self._add_check(
                prefix_check_id,
                "prefix",
                CHECK_WARNING,
                msg,
                details=self.prefix_details,
            )
            # Incomplete runs are recorded as warnings to maintain clear distinction from full completion
            self._add_issue(prefix_check_id, SEVERITY_WARNING, msg)
        else:
            msg = f"Full completion verified: processed {processed_frames} frames (declared: {declared_frames})"
            self._add_check(
                prefix_check_id,
                "prefix",
                CHECK_PASSED,
                msg,
                details=self.prefix_details,
            )

        self._add_check(
            check_id,
            "execution",
            CHECK_PASSED,
            f"Run status valid: {status!r}",
            details={"status": status},
        )

    # -----------------------------------------------------------------------
    # Step 5: Optional Artifact Schemas Validation
    # -----------------------------------------------------------------------

    def _validate_optional_schemas(
        self,
        posture_summary_json: Any,
        feedback_obs_json: Any,
        feedback_phases_json: Any,
        coaching_json: Any,
    ) -> None:
        """Validate shapes of supplied optional artifacts.

        When an optional artifact IS supplied, wrong types (e.g. list instead of dict,
        or malformed schema) must be reported as ERRORS, never silently ignored.
        """
        check_id = "CHK_OPTIONAL_SCHEMA"
        schema_errors: List[str] = []

        # 1. posture_summary.json
        if self.artifacts_inventory.get("posture_summary.json", {}).get("present", False):
            if posture_summary_json is None:
                schema_errors.append("posture_summary.json cannot be null; expected a JSON object (dict)")
            elif not isinstance(posture_summary_json, dict):
                schema_errors.append(
                    f"posture_summary.json has invalid type: expected JSON object (dict), got {type(posture_summary_json).__name__}"
                )
            else:
                summary_block = posture_summary_json.get("summary")
                events_block = posture_summary_json.get("events")
                if not isinstance(summary_block, dict):
                    schema_errors.append("posture_summary.json missing required 'summary' dict")
                if not isinstance(events_block, list):
                    schema_errors.append("posture_summary.json missing required 'events' list")
                elif events_block:
                    for idx, ev in enumerate(events_block):
                        if not isinstance(ev, dict):
                            schema_errors.append(f"posture_summary.json event[{idx}] is not a dict")
                            break
                        aid = ev.get("attempt_id")
                        if aid is not None and not is_strict_int(aid):
                            schema_errors.append(f"posture_summary.json event[{idx}] invalid attempt_id: {aid!r}")

        # 2. feedback_observations.json
        if self.artifacts_inventory.get("feedback_observations.json", {}).get("present", False):
            if feedback_obs_json is None:
                schema_errors.append("feedback_observations.json cannot be null; expected a JSON object (dict)")
            elif not isinstance(feedback_obs_json, dict):
                schema_errors.append(
                    f"feedback_observations.json has invalid type: expected JSON object (dict), got {type(feedback_obs_json).__name__}"
                )
            else:
                summary_block = feedback_obs_json.get("summary")
                attempts_block = feedback_obs_json.get("attempts")
                if not isinstance(summary_block, dict):
                    schema_errors.append("feedback_observations.json missing required 'summary' dict")
                if not isinstance(attempts_block, list):
                    schema_errors.append("feedback_observations.json missing required 'attempts' list")
                elif attempts_block:
                    for idx, att in enumerate(attempts_block):
                        if not isinstance(att, dict):
                            schema_errors.append(f"feedback_observations.json attempt[{idx}] is not a dict")
                            break
                        aid = att.get("attempt_id")
                        if aid is not None and not is_strict_int(aid):
                            schema_errors.append(f"feedback_observations.json attempt[{idx}] invalid attempt_id: {aid!r}")

        # 3. feedback_phases.json
        if self.artifacts_inventory.get("feedback_phases.json", {}).get("present", False):
            if feedback_phases_json is None:
                schema_errors.append("feedback_phases.json cannot be null; expected a JSON object (dict)")
            elif not isinstance(feedback_phases_json, dict):
                schema_errors.append(
                    f"feedback_phases.json has invalid type: expected JSON object (dict), got {type(feedback_phases_json).__name__}"
                )
            else:
                summary_block = feedback_phases_json.get("summary")
                phases_block = None
                if "attempts" in feedback_phases_json:
                    phases_block = feedback_phases_json["attempts"]
                elif "repetition_phases" in feedback_phases_json:
                    phases_block = feedback_phases_json["repetition_phases"]

                if not isinstance(summary_block, dict):
                    schema_errors.append("feedback_phases.json missing required 'summary' dict")
                if not isinstance(phases_block, list):
                    schema_errors.append("feedback_phases.json missing required 'attempts' or 'repetition_phases' list")
                elif phases_block:
                    for idx, ph in enumerate(phases_block):
                        if not isinstance(ph, dict):
                            schema_errors.append(f"feedback_phases.json attempt[{idx}] is not a dict")
                            break
                        aid = ph.get("attempt_id")
                        if aid is not None and not is_strict_int(aid):
                            schema_errors.append(f"feedback_phases.json attempt[{idx}] invalid attempt_id: {aid!r}")

        # 4. coaching_feedback.json
        if self.artifacts_inventory.get("coaching_feedback.json", {}).get("present", False):
            if coaching_json is None:
                # Permitted null telemetry where exporter allows it
                pass
            elif not isinstance(coaching_json, dict):
                schema_errors.append(
                    f"coaching_feedback.json has invalid type: expected JSON object (dict), got {type(coaching_json).__name__}"
                )
            else:
                c_status = coaching_json.get("status")
                if c_status is not None and not isinstance(c_status, str):
                    schema_errors.append(f"coaching_feedback.json status must be a string, got {type(c_status).__name__}")
                overall = coaching_json.get("overall_summary")
                if overall is not None and not isinstance(overall, dict):
                    schema_errors.append("coaching_feedback.json 'overall_summary' must be a dict")
                cues = coaching_json.get("per_repetition_cues")
                if cues is not None and not isinstance(cues, list):
                    schema_errors.append("coaching_feedback.json 'per_repetition_cues' must be a list")
                elif isinstance(cues, list):
                    for idx, cue in enumerate(cues):
                        if not isinstance(cue, dict):
                            schema_errors.append(f"coaching_feedback.json cue[{idx}] is not a dict")
                            break
                        caid = cue.get("attempt_id")
                        if caid is not None and not is_strict_int(caid):
                            schema_errors.append(f"coaching_feedback.json cue[{idx}] invalid attempt_id: {caid!r}")
                        # Strict check on is_counted (never default to True)
                        if "is_counted" in cue and not isinstance(cue["is_counted"], bool):
                            schema_errors.append(f"coaching_feedback.json cue[{idx}] is_counted must be a boolean")

        if schema_errors:
            msg = f"Optional artifact schema/type validation errors: {'; '.join(schema_errors)}"
            self._add_check(check_id, "schema", CHECK_FAILED, msg, details={"errors": schema_errors})
            self._add_issue(check_id, SEVERITY_ERROR, msg)
        else:
            self._add_check(
                check_id,
                "schema",
                CHECK_PASSED,
                "Supplied optional artifact structures adhere strictly to expected schemas",
            )

    # -----------------------------------------------------------------------
    # Step 6: Repetition Count Agreement
    # -----------------------------------------------------------------------

    def _check_repetition_counts(
        self,
        summary_json: Optional[Dict[str, Any]],
        attempts_json: Optional[Dict[str, Any]],
        feedback_obs_json: Optional[Dict[str, Any]],
        feedback_phases_json: Optional[Dict[str, Any]],
        coaching_json: Optional[Dict[str, Any]],
        csv_rows: Optional[List[Dict[str, str]]],
    ) -> None:
        check_id = "CHK_REP_COUNT_AGREEMENT"

        if not isinstance(summary_json, dict) or not isinstance(attempts_json, dict):
            self._add_check(check_id, "consistency", CHECK_UNAVAILABLE, "Required artifacts missing for rep count check")
            return

        # Summary completed_reps
        summary_reps = summary_json.get("completed_reps")
        if not is_strict_int(summary_reps) or summary_reps < 0:
            msg = f"summary.json completed_reps must be a non-negative integer, got: {summary_reps!r}"
            self._add_check(check_id, "consistency", CHECK_FAILED, msg)
            self._add_issue(check_id, SEVERITY_ERROR, msg)
            return

        discrepancies: List[str] = []

        # 1. Summary attempt_observations.counted_attempts
        attempt_obs = summary_json.get("attempt_observations")
        if isinstance(attempt_obs, dict):
            s_counted = attempt_obs.get("counted_attempts")
            if is_strict_int(s_counted) and s_counted != summary_reps:
                discrepancies.append(
                    f"summary.json attempt_observations.counted_attempts ({s_counted}) != completed_reps ({summary_reps})"
                )

        # 2. attempts.json counter_completed_reps and counted_attempts
        att_counter_reps = attempts_json.get("counter_completed_reps")
        if is_strict_int(att_counter_reps) and att_counter_reps != summary_reps:
            discrepancies.append(
                f"attempts.json counter_completed_reps ({att_counter_reps}) != summary completed_reps ({summary_reps})"
            )

        att_counted = attempts_json.get("counted_attempts")
        if is_strict_int(att_counted) and att_counted != summary_reps:
            discrepancies.append(
                f"attempts.json counted_attempts ({att_counted}) != summary completed_reps ({summary_reps})"
            )

        # 3. attempts.json events status == "counted"
        events = attempts_json.get("events")
        counted_events: List[Dict[str, Any]] = []
        if isinstance(events, list):
            for idx, ev in enumerate(events):
                if not isinstance(ev, dict):
                    discrepancies.append(f"attempts.json event[{idx}] is not an object: {ev!r}")
                    continue
                ev_status = ev.get("status")
                if not isinstance(ev_status, str):
                    discrepancies.append(f"attempts.json event[{idx}] status is not a string: {ev_status!r}")
                    continue
                if ev_status == "counted":
                    counted_events.append(ev)

            if len(counted_events) != summary_reps:
                discrepancies.append(
                    f"attempts.json events count with status=='counted' ({len(counted_events)}) != summary completed_reps ({summary_reps})"
                )
        else:
            discrepancies.append("attempts.json 'events' is not a list")

        # 4. feedback_observations.json (optional)
        if isinstance(feedback_obs_json, dict):
            fb_summary = feedback_obs_json.get("summary")
            if isinstance(fb_summary, dict):
                fb_counted = fb_summary.get("counted_events")
                if is_strict_int(fb_counted) and fb_counted != summary_reps:
                    discrepancies.append(
                        f"feedback_observations.json summary.counted_events ({fb_counted}) != summary completed_reps ({summary_reps})"
                    )

            fb_attempts = feedback_obs_json.get("attempts")
            if isinstance(fb_attempts, list):
                fb_counted_list = [
                    a for a in fb_attempts
                    if isinstance(a, dict) and (a.get("status") == "counted" or a.get("is_counted") is True)
                ]
                if len(fb_counted_list) != summary_reps:
                    discrepancies.append(
                        f"feedback_observations.json attempts counted count ({len(fb_counted_list)}) != summary completed_reps ({summary_reps})"
                    )

        # 5. feedback_phases.json (optional)
        if isinstance(feedback_phases_json, dict):
            ph_summary = feedback_phases_json.get("summary")
            if isinstance(ph_summary, dict):
                ph_counted = ph_summary.get("counted_events")
                if is_strict_int(ph_counted) and ph_counted != summary_reps:
                    discrepancies.append(
                        f"feedback_phases.json summary.counted_events ({ph_counted}) != summary completed_reps ({summary_reps})"
                    )

            if "repetition_phases" in feedback_phases_json and isinstance(feedback_phases_json["repetition_phases"], list):
                if len(feedback_phases_json["repetition_phases"]) != summary_reps:
                    discrepancies.append(
                        f"feedback_phases.json repetition_phases count ({len(feedback_phases_json['repetition_phases'])}) != summary completed_reps ({summary_reps})"
                    )
            elif "attempts" in feedback_phases_json and isinstance(feedback_phases_json["attempts"], list):
                ph_counted_list = [
                    a for a in feedback_phases_json["attempts"]
                    if isinstance(a, dict) and a.get("status") == "counted"
                ]
                if len(ph_counted_list) != summary_reps:
                    discrepancies.append(
                        f"feedback_phases.json attempts counted count ({len(ph_counted_list)}) != summary completed_reps ({summary_reps})"
                    )

        # 6. coaching_feedback.json (optional)
        if isinstance(coaching_json, dict):
            c_status = coaching_json.get("status")
            if c_status != "unavailable":
                c_overall = coaching_json.get("overall_summary")
                if isinstance(c_overall, dict):
                    c_reps = c_overall.get("total_counted_reps")
                    if is_strict_int(c_reps) and c_reps != summary_reps:
                        discrepancies.append(
                            f"coaching_feedback.json overall_summary.total_counted_reps ({c_reps}) != summary completed_reps ({summary_reps})"
                        )

                c_cues = coaching_json.get("per_repetition_cues")
                if isinstance(c_cues, list):
                    # Never default is_counted to True; check strict is True
                    counted_cues = [
                        c for c in c_cues
                        if isinstance(c, dict) and c.get("is_counted") is True
                    ]
                    if len(counted_cues) != summary_reps:
                        discrepancies.append(
                            f"coaching_feedback.json counted per_repetition_cues ({len(counted_cues)}) != summary completed_reps ({summary_reps})"
                        )

        # 7. angles.csv (optional)
        if isinstance(csv_rows, list) and csv_rows:
            max_csv_rep = 0
            has_rep_col = "completed_reps" in csv_rows[0]
            if has_rep_col:
                prev_rep = 0
                for idx, r in enumerate(csv_rows):
                    val_str = r.get("completed_reps", "").strip()
                    if val_str.isdigit():
                        cur_rep = int(val_str)
                        if cur_rep < prev_rep:
                            discrepancies.append(
                                f"angles.csv row {idx} completed_reps decreased: {cur_rep} < {prev_rep}"
                            )
                            break
                        prev_rep = cur_rep
                        max_csv_rep = max(max_csv_rep, cur_rep)
                if max_csv_rep != summary_reps:
                    discrepancies.append(
                        f"angles.csv final completed_reps ({max_csv_rep}) != summary completed_reps ({summary_reps})"
                    )

        if discrepancies:
            msg = f"Repetition count contradictions detected across artifacts: {'; '.join(discrepancies)}"
            self._add_check(check_id, "consistency", CHECK_FAILED, msg, details={"discrepancies": discrepancies})
            self._add_issue(check_id, SEVERITY_ERROR, msg)
        else:
            self._add_check(
                check_id,
                "consistency",
                CHECK_PASSED,
                f"Mutual agreement on completed repetition count ({summary_reps}) verified across all present artifacts",
            )

    # -----------------------------------------------------------------------
    # Step 7: Attempt ID References & Sequence Continuity
    # -----------------------------------------------------------------------

    def _check_attempt_id_integrity(
        self,
        attempts_json: Optional[Dict[str, Any]],
        feedback_obs_json: Optional[Dict[str, Any]],
        feedback_phases_json: Optional[Dict[str, Any]],
        coaching_json: Optional[Dict[str, Any]],
        csv_rows: Optional[List[Dict[str, str]]],
    ) -> None:
        check_id = "CHK_ATTEMPT_ID_INTEGRITY"

        if not isinstance(attempts_json, dict):
            self._add_check(check_id, "references", CHECK_UNAVAILABLE, "attempts.json not available")
            return

        events = attempts_json.get("events")
        if not isinstance(events, list):
            self._add_check(check_id, "references", CHECK_FAILED, "attempts.json events is not a list")
            self._add_issue(check_id, SEVERITY_ERROR, "attempts.json events is not a list")
            return

        attempt_ids: List[int] = []
        seen_ids: Set[int] = set()
        duplicates: List[int] = []

        for idx, ev in enumerate(events):
            if not isinstance(ev, dict):
                msg = f"attempts.json event[{idx}] is not an object: {ev!r}"
                self._add_check(check_id, "references", CHECK_FAILED, msg)
                self._add_issue(check_id, SEVERITY_ERROR, msg)
                return
            aid = ev.get("attempt_id")
            if not is_strict_int(aid) or aid <= 0:
                msg = f"Invalid non-integer or non-positive attempt_id in attempts.json event[{idx}]: {aid!r}"
                self._add_check(check_id, "references", CHECK_FAILED, msg)
                self._add_issue(check_id, SEVERITY_ERROR, msg)
                return
            if aid in seen_ids:
                duplicates.append(aid)
            seen_ids.add(aid)
            attempt_ids.append(aid)

        if duplicates:
            msg = f"Duplicate attempt_id values detected in attempts.json: {duplicates}"
            self._add_check(check_id, "references", CHECK_FAILED, msg)
            self._add_issue(check_id, SEVERITY_ERROR, msg)
            return

        # Check sequence monotonicity
        if attempt_ids != sorted(attempt_ids):
            msg = f"attempt_id values are not monotonically increasing: {attempt_ids}"
            self._add_check(check_id, "references", CHECK_FAILED, msg)
            self._add_issue(check_id, SEVERITY_ERROR, msg)
            return

        dangling_refs: List[str] = []

        # Check feedback_observations attempt_ids (and duplicates within feedback_obs)
        if isinstance(feedback_obs_json, dict):
            fb_attempts = feedback_obs_json.get("attempts")
            if isinstance(fb_attempts, list):
                fb_seen_ids: Set[int] = set()
                for idx, fba in enumerate(fb_attempts):
                    if not isinstance(fba, dict):
                        continue
                    faid = fba.get("attempt_id")
                    if faid is not None:
                        if not is_strict_int(faid):
                            dangling_refs.append(f"feedback_observations attempt[{idx}] invalid attempt_id: {faid!r}")
                        else:
                            if faid in fb_seen_ids:
                                dangling_refs.append(f"duplicate attempt_id {faid} in feedback_observations")
                            fb_seen_ids.add(faid)
                            if faid not in seen_ids:
                                dangling_refs.append(f"feedback_observations attempt_id {faid}")

        # Check feedback_phases attempt_ids (and duplicates within feedback_phases)
        if isinstance(feedback_phases_json, dict):
            phases = None
            if "attempts" in feedback_phases_json:
                phases = feedback_phases_json["attempts"]
            elif "repetition_phases" in feedback_phases_json:
                phases = feedback_phases_json["repetition_phases"]

            if isinstance(phases, list):
                ph_seen_ids: Set[int] = set()
                for idx, p in enumerate(phases):
                    if not isinstance(p, dict):
                        continue
                    paid = p.get("attempt_id")
                    if paid is not None:
                        if not is_strict_int(paid):
                            dangling_refs.append(f"feedback_phases attempt[{idx}] invalid attempt_id: {paid!r}")
                        else:
                            if paid in ph_seen_ids:
                                dangling_refs.append(f"duplicate attempt_id {paid} in feedback_phases")
                            ph_seen_ids.add(paid)
                            if paid not in seen_ids:
                                dangling_refs.append(f"feedback_phases attempt_id {paid}")

        # Check coaching_feedback attempt_ids (and duplicates within coaching cues)
        if isinstance(coaching_json, dict):
            cues = coaching_json.get("per_repetition_cues")
            if isinstance(cues, list):
                cue_seen_aids: Set[int] = set()
                cue_seen_reps: Set[int] = set()
                for idx, c in enumerate(cues):
                    if not isinstance(c, dict):
                        continue
                    rep_idx = c.get("rep_index")
                    if rep_idx is not None and is_strict_int(rep_idx):
                        if rep_idx in cue_seen_reps:
                            dangling_refs.append(f"duplicate rep_index {rep_idx} in coaching_feedback cues")
                        cue_seen_reps.add(rep_idx)
                    caid = c.get("attempt_id")
                    if caid is not None:
                        if not is_strict_int(caid):
                            dangling_refs.append(f"coaching_feedback cue[{idx}] invalid attempt_id: {caid!r}")
                        else:
                            if caid in cue_seen_aids:
                                dangling_refs.append(f"duplicate attempt_id {caid} in coaching_feedback cues")
                            cue_seen_aids.add(caid)
                            if caid not in seen_ids:
                                dangling_refs.append(f"coaching_feedback cue attempt_id {caid}")

            uncounted_expl = coaching_json.get("uncounted_movement_explanations")
            if isinstance(uncounted_expl, list):
                for idx, u in enumerate(uncounted_expl):
                    if not isinstance(u, dict):
                        continue
                    uaid = u.get("attempt_id")
                    if uaid is not None:
                        if not is_strict_int(uaid):
                            dangling_refs.append(f"coaching_feedback uncounted[{idx}] invalid attempt_id: {uaid!r}")
                        elif uaid not in seen_ids:
                            dangling_refs.append(f"coaching_feedback uncounted attempt_id {uaid}")

        # Check CSV attempt_id values
        if isinstance(csv_rows, list) and csv_rows:
            if "attempt_id" in csv_rows[0]:
                for r in csv_rows:
                    val_str = r.get("attempt_id", "").strip()
                    if val_str.isdigit():
                        vaid = int(val_str)
                        if vaid not in seen_ids:
                            dangling_refs.append(f"angles.csv row attempt_id {vaid}")
                            break

        if dangling_refs:
            msg = f"Dangling attempt_id references found (not in attempts.json): {', '.join(dangling_refs)}"
            self._add_check(check_id, "references", CHECK_FAILED, msg)
            self._add_issue(check_id, SEVERITY_ERROR, msg)
        else:
            self._add_check(
                check_id,
                "references",
                CHECK_PASSED,
                f"Attempt ID sequence continuity and cross-artifact referential integrity verified ({len(attempt_ids)} attempts)",
            )

    # -----------------------------------------------------------------------
    # Step 8: Chronological Monotonicity & Valid Timestamps
    # -----------------------------------------------------------------------

    def _check_chronological_ordering(
        self,
        attempts_json: Optional[Dict[str, Any]],
        feedback_phases_json: Optional[Dict[str, Any]],
        coaching_json: Optional[Dict[str, Any]],
        csv_rows: Optional[List[Dict[str, str]]],
    ) -> None:
        check_id = "CHK_CHRONOLOGICAL_TIMESTAMPS"

        if not isinstance(attempts_json, dict):
            self._add_check(check_id, "timestamps", CHECK_UNAVAILABLE, "attempts.json not available")
            return

        events = attempts_json.get("events")
        if not isinstance(events, list):
            return

        timestamp_errors: List[str] = []
        prev_start_time: Optional[float] = None
        prev_start_frame: Optional[int] = None

        for idx, ev in enumerate(events):
            if not isinstance(ev, dict):
                continue
            aid = ev.get("attempt_id", idx + 1)
            st = ev.get("start_time_s")
            et = ev.get("end_time_s")
            sf = ev.get("start_frame")
            ef = ev.get("end_frame")

            # Check start_time_s: must be finite number >= 0
            if not is_finite_number(st) or st < 0:
                timestamp_errors.append(f"Attempt {aid} start_time_s invalid or negative: {st!r}")

            # Check end_time_s: must be finite number >= start_time_s
            if not is_finite_number(et) or et < 0:
                timestamp_errors.append(f"Attempt {aid} end_time_s invalid or negative: {et!r}")
            elif is_finite_number(st) and et < st:
                timestamp_errors.append(f"Attempt {aid} end_time_s ({et}) < start_time_s ({st})")

            # Check frames
            if not is_strict_int(sf) or sf < 0:
                timestamp_errors.append(f"Attempt {aid} start_frame invalid or negative: {sf!r}")
            if not is_strict_int(ef) or ef < 0:
                timestamp_errors.append(f"Attempt {aid} end_frame invalid or negative: {ef!r}")
            elif is_strict_int(sf) and ef < sf:
                timestamp_errors.append(f"Attempt {aid} end_frame ({ef}) < start_frame ({sf})")

            # Check turnaround frame if present
            turnaround_f = ev.get("minimum_observed_frame")
            if turnaround_f is not None:
                if not is_strict_int(turnaround_f):
                    timestamp_errors.append(f"Attempt {aid} minimum_observed_frame invalid: {turnaround_f!r}")
                elif is_strict_int(sf) and is_strict_int(ef) and not (sf <= turnaround_f <= ef):
                    timestamp_errors.append(
                        f"Attempt {aid} minimum_observed_frame ({turnaround_f}) outside [{sf}, {ef}]"
                    )

            # Monotonic start ordering
            if is_finite_number(st):
                if prev_start_time is not None and st < prev_start_time:
                    timestamp_errors.append(
                        f"Attempt {aid} start_time_s ({st}) is before previous attempt start ({prev_start_time})"
                    )
                prev_start_time = float(st)

            if is_strict_int(sf):
                if prev_start_frame is not None and sf < prev_start_frame:
                    timestamp_errors.append(
                        f"Attempt {aid} start_frame ({sf}) is before previous attempt start ({prev_start_frame})"
                    )
                prev_start_frame = sf

        # Check feedback_phases nested bounds and turnaround inflection
        if isinstance(feedback_phases_json, dict):
            phases = None
            if "attempts" in feedback_phases_json:
                phases = feedback_phases_json["attempts"]
            elif "repetition_phases" in feedback_phases_json:
                phases = feedback_phases_json["repetition_phases"]

            if isinstance(phases, list):
                for p in phases:
                    if not isinstance(p, dict):
                        continue
                    paid = p.get("attempt_id", "?")
                    psf = p.get("start_frame")
                    pef = p.get("end_frame")
                    pst = p.get("start_time_s")
                    pet = p.get("end_time_s")

                    # min_knee_point
                    mkp = p.get("min_knee_point")
                    if isinstance(mkp, dict):
                        kf = mkp.get("frame_index")
                        kt = mkp.get("video_time_s")
                        if is_strict_int(kf) and is_strict_int(psf) and is_strict_int(pef):
                            if not (psf <= kf <= pef):
                                timestamp_errors.append(
                                    f"feedback_phases attempt {paid} min_knee_point frame ({kf}) outside [{psf}, {pef}]"
                                )
                        if is_finite_number(kt) and is_finite_number(pst) and is_finite_number(pet):
                            if not (pst <= kt <= pet):
                                timestamp_errors.append(
                                    f"feedback_phases attempt {paid} min_knee_point time ({kt}) outside [{pst}, {pet}]"
                                )

                    # descent_phase bounds
                    dp = p.get("descent_phase")
                    if isinstance(dp, dict):
                        dsf = dp.get("phase_start_frame")
                        def_ = dp.get("phase_end_frame")
                        dst = dp.get("phase_start_time_s")
                        det = dp.get("phase_end_time_s")
                        if is_strict_int(dsf) and is_strict_int(def_) and def_ < dsf:
                            timestamp_errors.append(
                                f"feedback_phases attempt {paid} descent_phase end_frame ({def_}) < start ({dsf})"
                            )
                        if is_finite_number(dst) and is_finite_number(det) and det < dst:
                            timestamp_errors.append(
                                f"feedback_phases attempt {paid} descent_phase end_time ({det}) < start ({dst})"
                            )

                    # return_phase bounds
                    rp = p.get("return_phase")
                    if isinstance(rp, dict):
                        rsf = rp.get("phase_start_frame")
                        ref = rp.get("phase_end_frame")
                        rst = rp.get("phase_start_time_s")
                        ret = rp.get("phase_end_time_s")
                        if is_strict_int(rsf) and is_strict_int(ref) and ref < rsf:
                            timestamp_errors.append(
                                f"feedback_phases attempt {paid} return_phase end_frame ({ref}) < start ({rsf})"
                            )
                        if is_finite_number(rst) and is_finite_number(ret) and ret < rst:
                            timestamp_errors.append(
                                f"feedback_phases attempt {paid} return_phase end_time ({ret}) < start ({rst})"
                            )

        # Check coaching_feedback cue timestamps
        if isinstance(coaching_json, dict):
            cues = coaching_json.get("per_repetition_cues")
            if isinstance(cues, list):
                for c in cues:
                    if not isinstance(c, dict):
                        continue
                    caid = c.get("attempt_id", "?")
                    ts = c.get("timestamps")
                    if isinstance(ts, dict):
                        cst = ts.get("start_time_s")
                        cinf = ts.get("inflection_time_s")
                        cet = ts.get("end_time_s")

                        if is_finite_number(cst) and is_finite_number(cet):
                            if cet < cst:
                                timestamp_errors.append(
                                    f"coaching cue {caid} inverted timestamps: start={cst}, end={cet}"
                                )
                        if is_finite_number(cinf) and is_finite_number(cst) and is_finite_number(cet):
                            if not (cst <= cinf <= cet):
                                timestamp_errors.append(
                                    f"coaching cue {caid} inflection ({cinf}) outside [{cst}, {cet}]"
                                )

        # Check angles.csv monotonic timestamps
        if isinstance(csv_rows, list) and csv_rows and len(csv_rows) > 1:
            prev_csv_time = -1.0
            for idx, r in enumerate(csv_rows):
                if not isinstance(r, dict):
                    continue
                vt_str = r.get("video_time_s", "")
                if vt_str is not None and str(vt_str).strip() != "":
                    try:
                        vt = float(vt_str)
                        if math.isfinite(vt):
                            if vt < prev_csv_time:
                                timestamp_errors.append(
                                    f"angles.csv row {idx} video_time_s ({vt}) inverted before previous ({prev_csv_time})"
                                )
                                break
                            prev_csv_time = vt
                    except (ValueError, TypeError):
                        timestamp_errors.append(f"angles.csv row {idx} video_time_s invalid: {vt_str!r}")
                        break

        if timestamp_errors:
            msg = f"Chronological timestamp / ordering violations detected: {'; '.join(timestamp_errors)}"
            self._add_check(check_id, "timestamps", CHECK_FAILED, msg, details={"errors": timestamp_errors})
            self._add_issue(check_id, SEVERITY_ERROR, msg)
        else:
            self._add_check(
                check_id,
                "timestamps",
                CHECK_PASSED,
                "Chronological monotonicity, interval validity, and turnaround bounds verified",
            )

    # -----------------------------------------------------------------------
    # Step 9: Configuration Identity
    # -----------------------------------------------------------------------

    def _check_configuration_identity(
        self,
        summary_json: Optional[Dict[str, Any]],
        attempts_json: Optional[Dict[str, Any]],
    ) -> None:
        check_id = "CHK_CONFIGURATION_IDENTITY"

        if not isinstance(summary_json, dict) or not isinstance(attempts_json, dict):
            self._add_check(check_id, "config", CHECK_UNAVAILABLE, "Summary or attempts artifact missing")
            return

        cfg_summary = summary_json.get("configuration")
        cfg_attempts = attempts_json.get("configuration")

        if not isinstance(cfg_summary, dict) or not isinstance(cfg_attempts, dict):
            self._add_check(check_id, "config", CHECK_UNAVAILABLE, "Configuration blocks not present in both artifacts")
            return

        mismatches: List[str] = []
        keys_to_compare = ("down_angle", "up_angle", "stable_frames", "departure_frames", "max_gap_seconds")
        checked_keys: List[str] = []

        for k in keys_to_compare:
            if k in cfg_summary and k in cfg_attempts:
                checked_keys.append(k)
                val_s = cfg_summary[k]
                val_a = cfg_attempts[k]
                if is_finite_number(val_s) and is_finite_number(val_a):
                    if not math.isclose(float(val_s), float(val_a), abs_tol=1e-5):
                        mismatches.append(f"{k}: summary={val_s} != attempts={val_a}")
                elif val_s != val_a:
                    mismatches.append(f"{k}: summary={val_s!r} != attempts={val_a!r}")

        details = {
            "checked_keys": checked_keys,
            "summary_keys": sorted(list(cfg_summary.keys())),
            "attempts_keys": sorted(list(cfg_attempts.keys())),
        }

        if mismatches:
            msg = f"Configuration parameter mismatches between summary and attempts: {', '.join(mismatches)}"
            self._add_check(check_id, "config", CHECK_FAILED, msg, details=details)
            self._add_issue(check_id, SEVERITY_ERROR, msg)
        else:
            self._add_check(
                check_id,
                "config",
                CHECK_PASSED,
                f"Shared configuration parameters ({', '.join(checked_keys)}) identical across summary and attempts",
                details=details,
            )

    # -----------------------------------------------------------------------
    # Step 10: Biomechanical and Scope Limitation Disclaimers
    # -----------------------------------------------------------------------

    def _check_limitation_disclaimers(
        self,
        summary_json: Optional[Dict[str, Any]],
        attempts_json: Optional[Dict[str, Any]],
        coaching_json: Optional[Dict[str, Any]],
    ) -> None:
        check_id = "CHK_LIMITATIONS_DISCLAIMERS"

        missing_disclaimers: List[str] = []

        if isinstance(summary_json, dict):
            limits = summary_json.get("limitations")
            if not isinstance(limits, list) or not limits:
                missing_disclaimers.append("summary.json lacks limitations list")

        if isinstance(attempts_json, dict):
            limits = attempts_json.get("limitations")
            if not isinstance(limits, list) or not limits:
                missing_disclaimers.append("attempts.json lacks limitations list")

        if isinstance(coaching_json, dict) and coaching_json.get("status") != "unavailable":
            disc = coaching_json.get("disclaimers")
            if not isinstance(disc, dict) or not disc:
                missing_disclaimers.append("coaching_feedback.json lacks disclaimers block")

        if missing_disclaimers:
            msg = f"Limitation or clinical disclaimers missing: {'; '.join(missing_disclaimers)}"
            self._add_check(check_id, "disclaimers", CHECK_WARNING, msg)
            self._add_issue(check_id, SEVERITY_WARNING, msg)
        else:
            self._add_check(
                check_id,
                "disclaimers",
                CHECK_PASSED,
                "Standard 2D kinematic limitations and unvalidated human status disclaimers verified",
            )

    # -----------------------------------------------------------------------
    # Step 11: Build Final Report
    # -----------------------------------------------------------------------

    def _build_report(self, status: str) -> Dict[str, Any]:
        passed_count = sum(1 for c in self.checks if c["status"] == CHECK_PASSED)
        failed_count = sum(1 for c in self.checks if c["status"] == CHECK_FAILED)
        warning_count = sum(1 for c in self.checks if c["status"] == CHECK_WARNING)
        unavail_count = sum(1 for c in self.checks if c["status"] == CHECK_UNAVAILABLE)

        total_files = len(self.initial_consumed_hashes) + len(self.initial_other_stats)
        byte_for_byte = not any(
            i["check_id"] == "CHK_INPUT_IMMUTABILITY" and i["severity"] == SEVERITY_ERROR
            for i in self.issues
        )

        return {
            "audit_version": AUDIT_VERSION,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "results_dir": str(self.raw_results_dir.resolve()),
            "status": status,
            "strict_mode": self.strict,
            "is_incomplete_prefix": self.is_incomplete_prefix,
            "prefix_details": self.prefix_details,
            "summary": {
                "total_checks": len(self.checks),
                "passed": passed_count,
                "failed": failed_count,
                "warning": warning_count,
                "unavailable": unavail_count,
            },
            "artifacts": self.artifacts_inventory,
            "checks": self.checks,
            "issues": self.issues,
            "input_integrity": {
                "consumed_files_hashed": len(self.initial_consumed_hashes),
                "other_files_stat_checked": len(self.initial_other_stats),
                "total_files_audited": total_files,
                "byte_for_byte_verified": byte_for_byte,
                "scope_note": "SHA-256 byte-verified for consumed JSON/CSV artifacts; media files verified via stat immutability.",
            },
            "disclaimer": DISCLAIMER_TEXT,
        }


# ---------------------------------------------------------------------------
# Public Functional API
# ---------------------------------------------------------------------------

def audit_results_directory(
    results_dir: Union[str, Path],
    strict: bool = False,
    quiet: bool = False,
    output_report: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """Audit a single saved results directory and optionally emit a fresh JSON report.

    Enforces true no-clobber publication using exclusive creation (os.O_CREAT | os.O_EXCL).
    """
    results_path = Path(results_dir)

    # 1. Validate report destination if provided
    report_path: Optional[Path] = None
    if output_report is not None:
        report_path = Path(output_report)

        # Unresolved ancestor safety check on destination
        dest_safe, dest_err = check_unresolved_ancestors_safety(report_path)
        if not dest_safe:
            raise ValueError(f"Report destination path safety violation: {dest_err}")

        # Protect input / output overlap (both unresolved and resolved)
        if paths_overlap(results_path, report_path):
            raise ValueError(
                f"Report destination cannot be placed inside or overlap with audited directory: {report_path}"
            )

        if results_path.exists():
            if paths_overlap(results_path.resolve(), report_path.resolve()):
                raise ValueError(
                    f"Resolved report destination cannot overlap with audited directory: {report_path}"
                )

        if report_path.exists():
            raise FileExistsError(
                f"Report destination already exists (no-clobber policy enforced): {report_path}"
            )

    # 2. Execute audit (auditor handles directory safety check internally and returns failed report if unsafe)
    auditor = ResultAuditor(results_path, strict=strict, quiet=quiet)
    report = auditor.audit()

    # 3. True no-clobber atomic report publication
    if report_path is not None:
        report_bytes = json.dumps(report, indent=2).encode("utf-8")
        report_path.parent.mkdir(parents=True, exist_ok=True)

        flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
        if hasattr(os, "O_BINARY"):
            flags |= os.O_BINARY

        try:
            fd = os.open(str(report_path), flags)
        except FileExistsError:
            raise FileExistsError(
                f"Report destination already exists (no-clobber policy enforced): {report_path}"
            )

        try:
            with open(fd, "wb", closefd=True) as f:
                f.write(report_bytes)
                f.flush()
                os.fsync(f.fileno())
        except Exception:
            try:
                report_path.unlink()
            except Exception:
                pass
            raise

    return report


# ---------------------------------------------------------------------------
# CLI Entrypoint
# ---------------------------------------------------------------------------

def format_console_report(report: Dict[str, Any]) -> str:
    """Format audit results for clean, readable console output."""
    lines: List[str] = []
    status = report["status"].upper()
    dir_path = report["results_dir"]
    s = report["summary"]

    lines.append("=" * 72)
    lines.append(f"GYM ANALYSER RESULT AUDIT: {status}")
    lines.append(f"Target: {dir_path}")
    lines.append(f"Checks: {s['total_checks']} total | {s['passed']} passed | {s['failed']} failed | {s['warning']} warnings | {s['unavailable']} unavailable")

    if report.get("is_incomplete_prefix"):
        pd = report.get("prefix_details", {})
        proc = pd.get("processed_frames")
        decl = pd.get("declared_frame_count")
        lines.append(f"PREFIX STATUS: INCOMPLETE RUN ({proc} / {decl} declared frames)")
    else:
        verified = any(c.get("check_id") == "CHK_DECODED_PREFIX" and c.get("status") == CHECK_PASSED
                       for c in report.get("checks", []))
        lines.append("PREFIX STATUS: FULL COMPLETION" if verified else "PREFIX STATUS: NOT VERIFIED")

    lines.append("-" * 72)
    lines.append("ARTIFACTS INVENTORY:")
    for art, info in sorted(report["artifacts"].items()):
        presence = info.get("status", "AVAILABLE") if info["present"] else "UNAVAILABLE"
        sz = f"{info['size_bytes']} bytes" if info["present"] else "-"
        lines.append(f"  - {art:<28}: {presence:<12} ({sz})")

    if report["issues"]:
        lines.append("-" * 72)
        lines.append("ISSUES FOUND:")
        for iss in report["issues"]:
            lines.append(f"  [{iss['severity']}] {iss['check_id']}: {iss['message']}")

    lines.append("-" * 72)
    lines.append(f"NOTICE: {report['disclaimer']}")
    lines.append("=" * 72)
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Command-line interface entry point.

    Exit codes:
      0 = Audit PASSED (or WARNING in non-strict mode)
      1 = Audit FAILED (or WARNING in strict mode)
      2 = CLI argument or invocation error (e.g. invalid args, file exists)
    """
    parser = argparse.ArgumentParser(
        prog="python -m gym_analyser.result_audit",
        description="Read-only cross-artifact consistency auditor for gym_analyser outputs.",
    )
    parser.add_argument(
        "--results-dir", "-r",
        required=True,
        help="Path to directory containing analysis artifacts (e.g. summary.json, attempts.json).",
    )
    parser.add_argument(
        "--output-report", "-o",
        default=None,
        help="Path to fresh JSON destination for audit report (enforces no-clobber).",
    )
    parser.add_argument(
        "--strict", "-s",
        action="store_true",
        help="Strict mode: treat any warning as an audit failure (exit code 1).",
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Quiet mode: suppress detailed console output.",
    )

    args = parser.parse_args(argv)

    try:
        report = audit_results_directory(
            results_dir=args.results_dir,
            strict=args.strict,
            quiet=args.quiet,
            output_report=args.output_report,
        )
    except FileExistsError as fee:
        sys.stderr.write(f"Error: {fee}\n")
        return 2
    except ValueError as ve:
        sys.stderr.write(f"Error: {ve}\n")
        return 2
    except Exception as exc:
        sys.stderr.write(f"Unexpected error: {exc}\n")
        return 2

    if not args.quiet:
        print(format_console_report(report))
    else:
        status = report["status"].upper()
        s = report["summary"]
        print(f"AUDIT {status}: {args.results_dir} ({s['passed']} passed, {s['failed']} failed, {s['warning']} warnings)")

    if report["status"] == STATUS_FAILED:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
