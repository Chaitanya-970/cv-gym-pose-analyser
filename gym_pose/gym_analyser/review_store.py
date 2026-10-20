"""Review Store for Gym Analyser.

Pure standard-library module managing raw-video human-review cases,
metadata persistence, atomic JSON updates, and evaluator-compatible
exports without third-party dependencies.
"""

from __future__ import annotations

import copy
import datetime
import hashlib
import json
import logging
import math
import os
from pathlib import Path
import re
import shutil
import stat
import threading
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union
import uuid

logger = logging.getLogger(__name__)

# ==============================================================================
# Constants & Enums
# ==============================================================================

ALLOWED_EXTENSIONS: Set[str] = {".mp4", ".webm", ".mov"}
DEFAULT_MAX_UPLOAD_BYTES: int = 200 * 1024 * 1024  # 200MB
ALLOWED_GROUPS: Set[str] = {"candidate", "development"}
ALLOWED_SCOPES: Set[str] = {"in_scope", "out_of_scope", "uncertain"}
ALLOWED_INTENTS: Set[str] = {"intentional_squat", "setup_return", "uncertain"}
ALLOWED_CYCLES: Set[str] = {"complete", "incomplete", "uncertain"}
ALLOWED_EVENT_CLASSES: Set[str] = {
    "completed_rep",
    "setup_or_weight_return",
    "partial_or_unfinished",
    "ambiguous_or_unclear",
}

DEFAULT_STORAGE_ROOT: Path = (
    Path(__file__).resolve().parent.parent / "output" / "human_reviews"
)

PUBLIC_CASE_KEYS: Tuple[str, ...] = (
    "case_id",
    "original_filename",
    "extension",
    "source_sha256",
    "file_size_bytes",
    "group",
    "created_at",
    "updated_at",
    "review",
    "history_count",
)

SHA256_HEX_PATTERN: re.Pattern[str] = re.compile(r"^[0-9a-f]{64}$")


# ==============================================================================
# Custom Exception Classes
# ==============================================================================


class CaseNotFoundError(KeyError, FileNotFoundError):
    """Raised when a review case cannot be found by ID."""


class CaseCorruptedError(KeyError, FileNotFoundError, ValueError):
    """Raised when a review case is corrupted, tampered with, or fails validation."""


class InvalidCaseIdError(ValueError, KeyError, FileNotFoundError):
    """Raised when a case_id fails UUID4 validation."""


# ==============================================================================
# Validation & Helper Functions
# ==============================================================================


def is_valid_uuid4(val: Any) -> bool:
    """Validate that val is a 36-character lowercase UUID4 string with dashes.

    Args:
        val: Value to validate.

    Returns:
        True if val is a valid 36-char lowercase UUID4 string with dashes, else False.
    """
    if not isinstance(val, str) or len(val) != 36:
        return False
    if val != val.lower():
        return False
    try:
        parsed = uuid.UUID(val)
        return str(parsed) == val and parsed.version == 4
    except (ValueError, AttributeError, TypeError):
        return False


def is_valid_iso8601(val: Any) -> bool:
    """Validate that val is a valid ISO8601 datetime string.

    Args:
        val: Value to validate.

    Returns:
        True if val is a valid ISO8601 datetime string, else False.
    """
    if not isinstance(val, str) or not val.strip():
        return False
    try:
        dt_str = val.strip().replace("Z", "+00:00")
        datetime.datetime.fromisoformat(dt_str)
        return True
    except (ValueError, TypeError, AttributeError):
        return False


def is_symlink(path: Union[str, Path]) -> bool:
    """Check if path is a symbolic link or Windows reparse point.

    Args:
        path: Path to inspect.

    Returns:
        True if the path is a symlink or reparse point, else False.
    """
    p = Path(path)
    try:
        if p.is_symlink():
            return True
    except (OSError, ValueError):
        pass
    try:
        st = p.lstat()
        if stat.S_ISLNK(st.st_mode):
            return True
        attrs = getattr(st, "st_file_attributes", 0)
        # 0x400 is FILE_ATTRIBUTE_REPARSE_POINT (NTFS junction, symlink, mount point)
        if attrs and (attrs & 0x400):
            return True
    except (OSError, AttributeError, ValueError):
        pass
    return False


def detect_duplicate_pairs(pairs: Sequence[Tuple[str, Any]]) -> Dict[str, Any]:
    """Object pairs hook for json.loads that raises ValueError on duplicate keys.

    Args:
        pairs: Sequence of (key, value) pairs decoded from JSON object.

    Returns:
        Dictionary mapping keys to values.

    Raises:
        ValueError: If any key appears more than once.
    """
    res: Dict[str, Any] = {}
    for k, v in pairs:
        if k in res:
            raise ValueError(f"Duplicate key detected in JSON: {k!r}")
        res[k] = v
    return res


def _reject_json_constant(val: str) -> Any:
    """Reject non-finite JSON constants like NaN, Infinity, -Infinity.

    Args:
        val: Decoded JSON constant string.

    Raises:
        ValueError: Always raised to reject non-finite values.
    """
    raise ValueError(f"Disallowed non-finite JSON constant: {val!r}")


def to_public_case(data: Dict[str, Any]) -> Dict[str, Any]:
    """Return dictionary with ONLY allowlisted public keys.

    Keys returned:
    ['case_id', 'original_filename', 'extension', 'source_sha256',
     'file_size_bytes', 'group', 'created_at', 'updated_at', 'review',
     'history_count']

    NEVER returns raw filesystem paths, internal paths, or unallowlisted disk keys.

    Args:
        data: Internal or loaded case dictionary.

    Returns:
        Sanitized public case dictionary.

    Raises:
        ValueError: If required public fields are missing or invalid.
    """
    if not isinstance(data, dict):
        raise ValueError(f"Case data must be a dictionary, got {type(data).__name__}")

    if "case_id" not in data:
        raise ValueError("Missing 'case_id' in case data")

    history = data.get("history")
    if isinstance(history, list):
        history_count = len(history)
    else:
        hc = data.get("history_count")
        history_count = int(hc) if (hc is not None and not isinstance(hc, bool)) else 0

    return {
        "case_id": str(data["case_id"]),
        "original_filename": str(data.get("original_filename", "")),
        "extension": str(data["extension"]),
        "source_sha256": str(data["source_sha256"]),
        "file_size_bytes": int(data["file_size_bytes"]),
        "group": str(data["group"]),
        "created_at": str(data["created_at"]),
        "updated_at": str(data["updated_at"]),
        "review": copy.deepcopy(data["review"]),
        "history_count": history_count,
    }


def validate_review_payload(review: Dict[str, Any]) -> Dict[str, Any]:
    """Validate and normalize a human review payload dictionary.

    Rules:
    - human_count: Optional[int]. Rejects bool, float, negative, str. Distinct unset (None) vs 0.
    - When human_count is not None:
      * reviewer must be non-empty string after strip.
      * raw_video_reviewed must be strictly True.
      * predictions_previously_seen MUST BE EXPLICIT BOOLEAN (True or False), NOT None.
    - When human_count is None:
      * predictions_previously_seen can be None, True, or False.
    - scope: in ALLOWED_SCOPES ('in_scope', 'out_of_scope', 'uncertain').
    - notes: str, max 5000 chars.
    - events: list of event dicts (max 200). Reject duplicate event_index.
      * start_time_s: finite float/int >= 0.0.
      * end_time_s: finite float/int >= start_time_s.
      * intent: in ALLOWED_INTENTS.
      * cycle: in ALLOWED_CYCLES.
      * observation: str, max 500 chars.

    Args:
        review: Dictionary containing review fields.

    Returns:
        Normalized dictionary with validated review fields.

    Raises:
        ValueError: If any field fails type, bounds, or semantic constraints.
    """
    if not isinstance(review, dict):
        raise ValueError(f"Review payload must be a dictionary, got {type(review).__name__}")

    # human_count validation
    human_count = review.get("human_count")
    if human_count is not None:
        if isinstance(human_count, bool) or not isinstance(human_count, int) or human_count < 0:
            raise ValueError(
                f"human_count must be a non-negative integer or None, got {human_count!r}"
            )

    # scope validation
    scope = review.get("scope", "in_scope")
    if scope is None:
        scope = "in_scope"
    if not isinstance(scope, str) or scope not in ALLOWED_SCOPES:
        raise ValueError(f"scope must be one of {sorted(ALLOWED_SCOPES)}, got {scope!r}")

    # reviewer validation
    reviewer_raw = review.get("reviewer", "")
    if reviewer_raw is None:
        reviewer_raw = ""
    if not isinstance(reviewer_raw, str):
        raise ValueError(f"reviewer must be a string, got {type(reviewer_raw).__name__}")
    reviewer = reviewer_raw.strip()
    if len(reviewer) > 100:
        raise ValueError(f"reviewer must be at most 100 characters, got {len(reviewer)}")

    # raw_video_reviewed validation
    raw_video_reviewed = review.get("raw_video_reviewed", False)
    if not isinstance(raw_video_reviewed, bool):
        raise ValueError(
            f"raw_video_reviewed must be a boolean, got {type(raw_video_reviewed).__name__}"
        )

    # predictions_previously_seen validation
    if "predictions_previously_seen" in review:
        predictions_previously_seen = review["predictions_previously_seen"]
    else:
        predictions_previously_seen = None

    if human_count is not None:
        # Strict co-dependency rules when count is set
        if not reviewer:
            raise ValueError(
                "reviewer must be a non-empty string after stripping when human_count is set"
            )
        if raw_video_reviewed is not True:
            raise ValueError(
                "raw_video_reviewed must be strictly True when human_count is set"
            )
        if predictions_previously_seen is None or not isinstance(predictions_previously_seen, bool):
            raise ValueError(
                "predictions_previously_seen must be an explicit boolean (True or False) when human_count is set"
            )
    else:
        # When human_count is None: predictions_previously_seen can be None, True, or False
        if predictions_previously_seen is not None and not isinstance(predictions_previously_seen, bool):
            raise ValueError(
                f"predictions_previously_seen must be None or boolean when human_count is None, got {predictions_previously_seen!r}"
            )

    # notes validation
    notes = review.get("notes", "")
    if notes is None:
        notes = ""
    if not isinstance(notes, str):
        raise ValueError(f"notes must be a string, got {type(notes).__name__}")
    if len(notes) > 5000:
        raise ValueError(f"notes must be at most 5000 characters, got {len(notes)}")

    schema_version = review.get("events_schema_version")
    if schema_version is not None and (type(schema_version) is not int or schema_version != 1):
        raise ValueError("events_schema_version must be 1 when supplied")

    # events validation
    events_raw = review.get("events")
    if events_raw is None:
        validated_events = None
    elif not isinstance(events_raw, list):
        raise ValueError(f"events must be a list or None, got {type(events_raw).__name__}")
    elif len(events_raw) > 200:
        raise ValueError(f"events list exceeds maximum length of 200, got {len(events_raw)}")
    else:
        validated_events_list: List[Dict[str, Any]] = []
        seen_event_ids: Set[str] = set()
        seen_indices: Set[int] = set()

        for i, evt in enumerate(events_raw):
            if not isinstance(evt, dict):
                raise ValueError(f"Event at index {i} must be a dictionary")

            is_new_event = ("event_id" in evt) or ("event_class" in evt)

            if is_new_event:
                # event_id: non-empty string, unique within case review
                event_id = evt.get("event_id")
                if not isinstance(event_id, str) or not event_id.strip():
                    raise ValueError(
                        f"Event {i} 'event_id' must be a non-empty string, got {event_id!r}"
                    )
                event_id = event_id.strip()
                if event_id in seen_event_ids:
                    raise ValueError(
                        f"Duplicate event_id {event_id!r} detected in events at position {i}"
                    )
                seen_event_ids.add(event_id)

                # event_class: in ALLOWED_EVENT_CLASSES
                event_class = evt.get("event_class")
                if not isinstance(event_class, str) or event_class not in ALLOWED_EVENT_CLASSES:
                    raise ValueError(
                        f"Event {i} 'event_class' must be one of {sorted(ALLOWED_EVENT_CLASSES)}, got {event_class!r}"
                    )

                # timestamps: start_time_s, turnaround_time_s, end_time_s
                parsed_times: Dict[str, Optional[float]] = {}
                for ts_name in ("start_time_s", "turnaround_time_s", "end_time_s"):
                    ts_val = evt.get(ts_name)
                    if ts_val is not None:
                        if isinstance(ts_val, bool) or not isinstance(ts_val, (int, float)):
                            raise ValueError(
                                f"Event {i} '{ts_name}' must be a finite number >= 0.0 or None, got {ts_val!r}"
                            )
                        if not math.isfinite(ts_val) or ts_val < 0.0:
                            raise ValueError(
                                f"Event {i} '{ts_name}' must be finite and >= 0.0, got {ts_val}"
                            )
                        parsed_times[ts_name] = float(ts_val)
                    else:
                        parsed_times[ts_name] = None

                st = parsed_times["start_time_s"]
                tt = parsed_times["turnaround_time_s"]
                et = parsed_times["end_time_s"]

                if st is not None and tt is not None and tt < st:
                    raise ValueError(
                        f"Event {i} 'turnaround_time_s' ({tt}) cannot be earlier than 'start_time_s' ({st})"
                    )
                if tt is not None and et is not None and et < tt:
                    raise ValueError(
                        f"Event {i} 'end_time_s' ({et}) cannot be earlier than 'turnaround_time_s' ({tt})"
                    )
                if st is not None and et is not None and et < st:
                    raise ValueError(
                        f"Event {i} 'end_time_s' ({et}) cannot be earlier than 'start_time_s' ({st})"
                    )

                # uncertainty_notes: optional string or None
                uncertainty_notes = evt.get("uncertainty_notes")
                if uncertainty_notes is not None:
                    if not isinstance(uncertainty_notes, str):
                        raise ValueError(
                            f"Event {i} 'uncertainty_notes' must be a string or None, got {type(uncertainty_notes).__name__}"
                        )
                    if len(uncertainty_notes) > 5000:
                        raise ValueError(
                            f"Event {i} 'uncertainty_notes' must be at most 5000 characters, got {len(uncertainty_notes)}"
                        )

                # visible_posture_observations: optional dict/string or None
                visible_posture_obs = evt.get("visible_posture_observations")
                if visible_posture_obs is not None:
                    if not isinstance(visible_posture_obs, (str, dict)):
                        raise ValueError(
                            f"Event {i} 'visible_posture_observations' must be a dict, string, or None, got {type(visible_posture_obs).__name__}"
                        )
                    if isinstance(visible_posture_obs, str) and len(visible_posture_obs) > 5000:
                        raise ValueError(
                            f"Event {i} 'visible_posture_observations' string must be at most 5000 characters, got {len(visible_posture_obs)}"
                        )
                    if isinstance(visible_posture_obs, dict):
                        try:
                            encoded = json.dumps(visible_posture_obs, allow_nan=False)
                            if len(encoded) > 5000:
                                raise ValueError("Posture observation object exceeds 5000 characters")
                            visible_posture_obs = json.loads(encoded)
                        except (ValueError, TypeError, RecursionError) as exc:
                            raise ValueError(f"Event {i} posture observations must be finite JSON: {exc}") from exc

                validated_events_list.append({
                    "event_id": event_id,
                    "event_class": event_class,
                    "start_time_s": st,
                    "turnaround_time_s": tt,
                    "end_time_s": et,
                    "uncertainty_notes": uncertainty_notes,
                    "visible_posture_observations": visible_posture_obs,
                })

            elif "event_index" in evt:
                # Legacy event format
                event_index = evt.get("event_index")
                if isinstance(event_index, bool) or not isinstance(event_index, int) or event_index < 0:
                    raise ValueError(
                        f"Event {i} 'event_index' must be a non-negative integer, got {event_index!r}"
                    )
                if event_index in seen_indices:
                    raise ValueError(
                        f"Duplicate event_index {event_index} detected in events at position {i}"
                    )
                seen_indices.add(event_index)

                st_leg = evt.get("start_time_s")
                if isinstance(st_leg, bool) or not isinstance(st_leg, (int, float)):
                    raise ValueError(
                        f"Event {i} 'start_time_s' must be a finite number >= 0.0, got {st_leg!r}"
                    )
                if not math.isfinite(st_leg) or st_leg < 0.0:
                    raise ValueError(
                        f"Event {i} 'start_time_s' must be finite and >= 0.0, got {st_leg}"
                    )

                et_leg = evt.get("end_time_s")
                if isinstance(et_leg, bool) or not isinstance(et_leg, (int, float)):
                    raise ValueError(
                        f"Event {i} 'end_time_s' must be a finite number >= start_time_s, got {et_leg!r}"
                    )
                if not math.isfinite(et_leg) or et_leg < st_leg:
                    raise ValueError(
                        f"Event {i} 'end_time_s' ({et_leg}) must be finite and >= start_time_s ({st_leg})"
                    )

                intent = evt.get("intent")
                if not isinstance(intent, str) or intent not in ALLOWED_INTENTS:
                    raise ValueError(
                        f"Event {i} 'intent' must be one of {sorted(ALLOWED_INTENTS)}, got {intent!r}"
                    )

                cycle = evt.get("cycle")
                if not isinstance(cycle, str) or cycle not in ALLOWED_CYCLES:
                    raise ValueError(
                        f"Event {i} 'cycle' must be one of {sorted(ALLOWED_CYCLES)}, got {cycle!r}"
                    )

                observation = evt.get("observation", "")
                if observation is None:
                    observation = ""
                if not isinstance(observation, str):
                    raise ValueError(
                        f"Event {i} 'observation' must be a string, got {type(observation).__name__}"
                    )
                if len(observation) > 500:
                    raise ValueError(
                        f"Event {i} 'observation' must be at most 500 characters, got {len(observation)}"
                    )

                validated_events_list.append({
                    "event_index": event_index,
                    "start_time_s": float(st_leg),
                    "end_time_s": float(et_leg),
                    "intent": intent,
                    "cycle": cycle,
                    "observation": observation,
                })
            else:
                raise ValueError(
                    f"Event at index {i} missing required event fields ('event_id' and 'event_class' required)"
                )

        validated_events = validated_events_list

    validated = {
        "human_count": human_count,
        "scope": scope,
        "reviewer": reviewer,
        "raw_video_reviewed": raw_video_reviewed,
        "predictions_previously_seen": predictions_previously_seen,
        "notes": notes,
        "events": validated_events,
    }
    if schema_version is not None or any("event_id" in event for event in (validated_events or [])):
        validated["events_schema_version"] = 1
    return validated



def load_persisted_case(
    root_or_store: Union[ReviewStore, str, Path],
    case_id: str,
    verify_raw_sha: bool = True,
) -> Dict[str, Any]:
    """Shared persisted-case loader.

    Validates:
    - case_dir must be directory, NOT symlink. Must resolve inside root.resolve().
    - case.json must exist, be regular file, NOT symlink.
    - JSON parsed with detect_duplicate_pairs and non-finite rejection.
    - Stored case_id MUST EXACTLY match the directory name / expected case_id.
    - extension must be in ALLOWED_EXTENSIONS.
    - source_sha256 must be 64-char lowercase hex string.
    - group must be in ALLOWED_GROUPS.
    - created_at and updated_at must be valid ISO8601 strings.
    - file_size_bytes must be positive int.
    - Stored review must pass validate_review_payload.
    - Each item in stored history must have valid ISO saved_at and valid review.
    - Raw video file must exist, be regular file, NOT symlink, size must match file_size_bytes.
    - If verify_raw_sha is True (default True): stream-hash the raw video file and verify
      SHA256 matches stored source_sha256. If mismatch, raises CaseCorruptedError.

    Args:
        root_or_store: ReviewStore instance or Path/str root storage directory.
        case_id: 36-character lowercase UUID4 string.
        verify_raw_sha: If True, stream-hashes raw video and verifies against source_sha256.

    Returns:
        Internal persisted case dictionary with validated fields.

    Raises:
        InvalidCaseIdError: If case_id is not a valid UUID4.
        CaseNotFoundError: If case directory does not exist.
        CaseCorruptedError: If any validation check fails or video is tampered.
    """
    if isinstance(root_or_store, ReviewStore):
        return root_or_store.load_persisted_case(case_id, verify_raw_sha=verify_raw_sha)
    store = ReviewStore(root=root_or_store)
    return store.load_persisted_case(case_id, verify_raw_sha=verify_raw_sha)


# ==============================================================================
# ReviewStore Class
# ==============================================================================


class ReviewStore:
    """Store for managing raw-video human-review cases on disk.

    Provides atomic case creation, retrieval, validation, review history
    tracking, and evaluator-compatible counts and manifest exports.
    """

    def __init__(
        self,
        root: Optional[Union[str, Path]] = None,
        max_upload_bytes: int = DEFAULT_MAX_UPLOAD_BYTES,
    ) -> None:
        """Initialize the ReviewStore with an optional root directory and size limit.

        Args:
            root: Root filesystem directory for storing review cases.
                Defaults to DEFAULT_STORAGE_ROOT (gym_pose/output/human_reviews).
            max_upload_bytes: Maximum allowed video upload size in bytes.
                Must be a positive integer (rejects bool, float, negative, zero).

        Raises:
            ValueError: If max_upload_bytes is not a positive integer or root is a symlink.
        """
        if (
            isinstance(max_upload_bytes, bool)
            or not isinstance(max_upload_bytes, int)
            or max_upload_bytes <= 0
        ):
            raise ValueError(
                f"max_upload_bytes must be a positive integer, got {max_upload_bytes!r}"
            )
        self.max_upload_bytes: int = max_upload_bytes

        if root is None:
            self.root: Path = DEFAULT_STORAGE_ROOT
        else:
            self.root = Path(root)

        if self.root.exists() and is_symlink(self.root):
            raise ValueError(f"Storage root cannot be a symlink or reparse point: {self.root}")

        self.root.mkdir(parents=True, exist_ok=True)

        if is_symlink(self.root):
            raise ValueError(f"Storage root cannot be a symlink or reparse point: {self.root}")

        self._lock: threading.RLock = threading.RLock()

    def create_case(
        self,
        source_path: Union[str, Path],
        original_name: Optional[str] = None,
        group: str = "candidate",
    ) -> Dict[str, Any]:
        """Create a new review case by safely copying a video into managed storage.

        Args:
            source_path: Path to the raw video file. Must exist, be a regular file, and not a symlink.
            original_name: Optional original filename uploaded by the user.
            group: Review group assignment ('candidate' or 'development').

        Returns:
            Public case metadata dictionary with allowlisted keys only.

        Raises:
            FileNotFoundError: If source_path does not exist.
            ValueError: If source_path is a symlink or not a regular file, extension is disallowed,
                file size exceeds max_upload_bytes during copy, or group is invalid.
        """
        source = Path(source_path)
        if not source.exists():
            raise FileNotFoundError(f"Source file does not exist: {source}")
        if is_symlink(source):
            raise ValueError(f"Source path cannot be a symbolic link or reparse point: {source}")
        if not source.is_file():
            raise ValueError(f"Source path is not a regular file: {source}")

        # Extension check
        source_ext = source.suffix.lower()
        orig_ext = Path(original_name).suffix.lower() if original_name else ""

        if original_name and orig_ext and orig_ext not in ALLOWED_EXTENSIONS:
            raise ValueError(
                f"File extension {orig_ext!r} is not allowed. Must be one of {sorted(ALLOWED_EXTENSIONS)}"
            )

        if source_ext in ALLOWED_EXTENSIONS:
            ext = source_ext
        elif orig_ext in ALLOWED_EXTENSIONS:
            ext = orig_ext
        else:
            raise ValueError(
                f"File extension {source_ext!r} is not allowed. Must be one of {sorted(ALLOWED_EXTENSIONS)}"
            )

        # Group check
        if group not in ALLOWED_GROUPS:
            raise ValueError(
                f"Group {group!r} is invalid. Must be one of {sorted(ALLOWED_GROUPS)}"
            )

        # Initial stat check
        initial_size = source.stat().st_size
        if initial_size <= 0:
            raise ValueError("Raw video file must not be empty")
        if initial_size > self.max_upload_bytes:
            raise ValueError(
                f"File size {initial_size} bytes exceeds maximum limit of {self.max_upload_bytes} bytes"
            )

        with self._lock:
            case_id = str(uuid.uuid4()).lower()
            case_dir = self.root / case_id

            # Exclusive directory creation
            case_dir.mkdir(parents=False, exist_ok=False)

            created_files: List[Path] = []
            created_dir: Optional[Path] = case_dir

            video_dest = case_dir / f"video{ext}"
            temp_video = case_dir / f"video{ext}.tmp"
            created_files.append(temp_video)

            hasher = hashlib.sha256()
            bytes_written = 0
            chunk_size = 64 * 1024

            try:
                with open(source, "rb") as src_f, open(temp_video, "wb") as dst_f:
                    while True:
                        chunk = src_f.read(chunk_size)
                        if not chunk:
                            break
                        bytes_written += len(chunk)
                        if bytes_written > self.max_upload_bytes:
                            raise ValueError(
                                f"File size exceeded maximum limit of {self.max_upload_bytes} bytes during copy"
                            )
                        hasher.update(chunk)
                        dst_f.write(chunk)
                    dst_f.flush()
                    os.fsync(dst_f.fileno())

                temp_video.replace(video_dest)
                created_files.remove(temp_video)
                created_files.append(video_dest)

                sha256_hex = hasher.hexdigest().lower()
                iso_now = datetime.datetime.now(datetime.timezone.utc).isoformat()
                orig_filename = original_name.strip() if original_name else source.name

                case_data: Dict[str, Any] = {
                    "case_id": case_id,
                    "original_filename": orig_filename,
                    "extension": ext,
                    "source_sha256": sha256_hex,
                    "file_size_bytes": bytes_written,
                    "group": group,
                    "created_at": iso_now,
                    "updated_at": iso_now,
                    "review": {
                        "human_count": None,
                        "scope": "in_scope",
                        "reviewer": "",
                        "raw_video_reviewed": False,
                        "predictions_previously_seen": None,
                        "notes": "",
                        "events": [],
                    },
                    "history": [],
                }

                case_json = case_dir / "case.json"
                temp_json = case_dir / "case.json.tmp"
                created_files.append(temp_json)

                with open(temp_json, "w", encoding="utf-8") as f:
                    json.dump(case_data, f, indent=2, allow_nan=False)
                    f.flush()
                    os.fsync(f.fileno())

                temp_json.replace(case_json)
                created_files.remove(temp_json)
                created_files.append(case_json)

                return to_public_case(case_data)

            except Exception:
                # Clean ONLY newly created files and directory from this request
                for cf in reversed(created_files):
                    try:
                        if cf.exists():
                            cf.unlink()
                    except OSError:
                        pass
                if created_dir and created_dir.exists():
                    try:
                        shutil.rmtree(created_dir)
                    except OSError:
                        pass
                raise

    def load_persisted_case(
        self,
        case_id: str,
        verify_raw_sha: bool = True,
    ) -> Dict[str, Any]:
        """Load and strictly validate a persisted review case from storage.

        Validates directory containment, symlink absence, JSON integrity,
        stored case ID match, timestamps, schema, and raw video SHA256 integrity.

        Args:
            case_id: 36-character lowercase UUID4 string.
            verify_raw_sha: If True (default), stream-hashes raw video and verifies SHA256.

        Returns:
            Validated internal case dictionary.

        Raises:
            InvalidCaseIdError: If case_id is not a valid UUID4.
            CaseNotFoundError: If case directory does not exist.
            CaseCorruptedError: If any validation or integrity check fails.
        """
        if not is_valid_uuid4(case_id):
            raise InvalidCaseIdError(f"Invalid UUID4 case_id: {case_id!r}")

        if is_symlink(self.root):
            raise CaseCorruptedError("Storage root is a symlink or reparse point")
        root_resolved = self.root.resolve()
        case_dir = self.root / case_id

        if not case_dir.exists():
            raise CaseNotFoundError(f"Case directory not found: {case_id}")

        if is_symlink(case_dir):
            raise CaseCorruptedError(
                f"Case directory is a symlink or reparse point: {case_dir}"
            )

        if not case_dir.is_dir():
            raise CaseCorruptedError(f"Case path is not a directory: {case_dir}")

        case_dir_resolved = case_dir.resolve()
        try:
            rel = case_dir_resolved.relative_to(root_resolved)
            if str(rel) == "." or str(rel).startswith(".."):
                raise CaseCorruptedError(
                    f"Case directory {case_dir_resolved} escapes storage root {root_resolved}"
                )
        except ValueError as exc:
            raise CaseCorruptedError(
                f"Case directory {case_dir_resolved} is not inside storage root {root_resolved}"
            ) from exc

        if case_dir_resolved.name != case_id or case_dir.name != case_id:
            raise CaseCorruptedError(
                f"Case directory name '{case_dir.name}' does not match expected '{case_id}'"
            )

        case_file = case_dir / "case.json"
        if not case_file.exists():
            raise CaseCorruptedError(f"case.json missing in case '{case_id}'")

        if is_symlink(case_file):
            raise CaseCorruptedError(f"case.json is a symlink or reparse point in '{case_id}'")

        if not case_file.is_file():
            raise CaseCorruptedError(f"case.json is not a regular file in '{case_id}'")

        case_file_resolved = case_file.resolve()
        if case_file_resolved.parent != case_dir_resolved:
            raise CaseCorruptedError(f"case.json escapes case directory in '{case_id}'")

        try:
            content = case_file.read_text(encoding="utf-8")
            data = json.loads(
                content,
                object_pairs_hook=detect_duplicate_pairs,
                parse_constant=_reject_json_constant,
            )
        except (ValueError, json.JSONDecodeError, OSError) as exc:
            raise CaseCorruptedError(
                f"Invalid or corrupted JSON in case.json for '{case_id}': {exc}"
            ) from exc

        if not isinstance(data, dict):
            raise CaseCorruptedError(
                f"case.json in '{case_id}' must contain a JSON object, got {type(data).__name__}"
            )

        # Stored case_id MUST EXACTLY match the directory name / expected case_id
        stored_case_id = data.get("case_id")
        if stored_case_id != case_id:
            raise CaseCorruptedError(
                f"Stored case_id {stored_case_id!r} does not match expected {case_id!r}"
            )

        # extension must be in ALLOWED_EXTENSIONS
        extension = data.get("extension")
        if not isinstance(extension, str) or extension not in ALLOWED_EXTENSIONS:
            raise CaseCorruptedError(
                f"Stored extension {extension!r} is not allowed for case '{case_id}'"
            )

        # source_sha256 must be 64-char lowercase hex string
        source_sha256 = data.get("source_sha256")
        if not isinstance(source_sha256, str) or not SHA256_HEX_PATTERN.match(source_sha256):
            raise CaseCorruptedError(
                f"Invalid source_sha256 format {source_sha256!r} for case '{case_id}'"
            )

        # group must be in ALLOWED_GROUPS
        group = data.get("group")
        if not isinstance(group, str) or group not in ALLOWED_GROUPS:
            raise CaseCorruptedError(f"Invalid group {group!r} for case '{case_id}'")

        # created_at and updated_at must be valid ISO8601 strings
        created_at = data.get("created_at")
        if not is_valid_iso8601(created_at):
            raise CaseCorruptedError(
                f"Invalid created_at ISO8601 timestamp {created_at!r} in case '{case_id}'"
            )
        updated_at = data.get("updated_at")
        if not is_valid_iso8601(updated_at):
            raise CaseCorruptedError(
                f"Invalid updated_at ISO8601 timestamp {updated_at!r} in case '{case_id}'"
            )

        # file_size_bytes must be positive int
        file_size_bytes = data.get("file_size_bytes")
        if (
            isinstance(file_size_bytes, bool)
            or not isinstance(file_size_bytes, int)
            or file_size_bytes <= 0
        ):
            raise CaseCorruptedError(
                f"Invalid file_size_bytes {file_size_bytes!r} in case '{case_id}'"
            )

        # Stored review must pass validate_review_payload
        stored_review = data.get("review")
        if not isinstance(stored_review, dict):
            raise CaseCorruptedError(f"Missing or invalid review dict in case '{case_id}'")
        try:
            validated_review = validate_review_payload(stored_review)
        except ValueError as exc:
            raise CaseCorruptedError(
                f"Stored review failed validation in case '{case_id}': {exc}"
            ) from exc

        # Each item in stored history must have valid ISO saved_at and valid review
        stored_history = data.get("history", [])
        if not isinstance(stored_history, list):
            raise CaseCorruptedError(f"Stored history must be a list in case '{case_id}'")

        validated_history: List[Dict[str, Any]] = []
        for idx, item in enumerate(stored_history):
            if not isinstance(item, dict):
                raise CaseCorruptedError(
                    f"History item {idx} in case '{case_id}' is not a dict"
                )
            saved_at = item.get("saved_at")
            if not is_valid_iso8601(saved_at):
                raise CaseCorruptedError(
                    f"History item {idx} has invalid saved_at timestamp {saved_at!r} in case '{case_id}'"
                )
            h_rev = item.get("review")
            if not isinstance(h_rev, dict):
                raise CaseCorruptedError(
                    f"History item {idx} missing review dict in case '{case_id}'"
                )
            try:
                val_h_rev = validate_review_payload(h_rev)
            except ValueError as exc:
                raise CaseCorruptedError(
                    f"History item {idx} review failed validation in case '{case_id}': {exc}"
                ) from exc
            validated_history.append({"saved_at": saved_at, "review": val_h_rev})

        # Raw video file check
        video_file = case_dir / f"video{extension}"
        if not video_file.exists():
            raise CaseCorruptedError(f"Raw video file missing: {video_file}")
        if is_symlink(video_file):
            raise CaseCorruptedError(
                f"Raw video file is a symlink or reparse point: {video_file}"
            )
        if not video_file.is_file():
            raise CaseCorruptedError(f"Raw video file is not a regular file: {video_file}")

        video_resolved = video_file.resolve()
        if video_resolved.parent != case_dir_resolved:
            raise CaseCorruptedError(f"Raw video file escapes case directory: {video_file}")

        actual_size = video_file.stat().st_size
        if actual_size != file_size_bytes:
            raise CaseCorruptedError(
                f"Raw video size ({actual_size} bytes) does not match recorded size ({file_size_bytes} bytes)"
            )

        # Stream-hash raw video file and verify SHA256 if requested
        if verify_raw_sha:
            hasher = hashlib.sha256()
            try:
                with open(video_file, "rb") as vf:
                    while True:
                        chunk = vf.read(64 * 1024)
                        if not chunk:
                            break
                        hasher.update(chunk)
            except OSError as exc:
                raise CaseCorruptedError(
                    f"Failed to read raw video file for hashing in case '{case_id}': {exc}"
                ) from exc
            actual_sha = hasher.hexdigest().lower()
            if actual_sha != source_sha256:
                raise CaseCorruptedError(
                    f"Video integrity check failed for case '{case_id}': "
                    f"computed SHA256 {actual_sha} does not match stored {source_sha256}"
                )

        return {
            "case_id": case_id,
            "original_filename": str(data.get("original_filename", "")),
            "extension": extension,
            "source_sha256": source_sha256,
            "file_size_bytes": file_size_bytes,
            "group": group,
            "created_at": created_at,
            "updated_at": updated_at,
            "review": validated_review,
            "history": validated_history,
        }

    def get_case(self, case_id: str) -> Dict[str, Any]:
        """Retrieve a single review case by ID, validated and sanitized.

        Args:
            case_id: 36-character lowercase UUID4 string.

        Returns:
            Public case dictionary with allowlisted keys only.

        Raises:
            InvalidCaseIdError: If case_id is not a valid UUID4.
            CaseNotFoundError: If case directory is not found.
            CaseCorruptedError: If metadata or raw video fails validation.
        """
        persisted = self.load_persisted_case(case_id, verify_raw_sha=True)
        return to_public_case(persisted)

    def video_path(self, case_id: str) -> Path:
        """Locate and verify the managed raw video file for a review case.

        Verifies that the raw video SHA256 matches stored metadata.
        If tampered or corrupted, raises CaseCorruptedError.

        Args:
            case_id: 36-character lowercase UUID4 string.

        Returns:
            Resolved Path pointing to the verified video file.

        Raises:
            InvalidCaseIdError: If case_id is not a valid UUID4.
            CaseNotFoundError: If case directory is not found.
            CaseCorruptedError: If metadata or video integrity check fails.
        """
        persisted = self.load_persisted_case(case_id, verify_raw_sha=True)
        v_path = self.root / case_id / f"video{persisted['extension']}"
        return v_path.resolve()

    def save_review(self, case_id: str, review_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Save a validated review update for a case, appending previous review to history.

        Verifies existing case integrity (including raw video SHA256) BEFORE touching disk.
        If corrupt or tampered, raises CaseCorruptedError and leaves existing bytes untouched.

        Args:
            case_id: 36-character lowercase UUID4 string.
            review_dict: Review dictionary to validate and save.

        Returns:
            Updated public case dictionary with allowlisted keys only.

        Raises:
            InvalidCaseIdError: If case_id is not a valid UUID4.
            CaseNotFoundError: If case is not found.
            CaseCorruptedError: If existing metadata or video is corrupted or tampered.
            ValueError: If review_dict fails validation.
        """
        with self._lock:
            # Validate input review dictionary first
            validated_review = validate_review_payload(review_dict)

            # Verify existing case BEFORE touching any disk file
            persisted = self.load_persisted_case(case_id, verify_raw_sha=True)

            case_dir = self.root / case_id
            case_file = case_dir / "case.json"

            old_updated_at = persisted.get("updated_at") or persisted.get("created_at")
            old_review = copy.deepcopy(persisted.get("review", {}))

            history = list(persisted.get("history", []))
            history.append({
                "saved_at": old_updated_at,
                "review": old_review,
            })

            now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()
            updated_data = dict(persisted)
            updated_data["review"] = validated_review
            updated_data["updated_at"] = now_iso
            updated_data["history"] = history

            temp_file = case_dir / "case.json.tmp"
            try:
                with open(temp_file, "w", encoding="utf-8") as f:
                    json.dump(updated_data, f, indent=2, allow_nan=False)
                    f.flush()
                    os.fsync(f.fileno())
                temp_file.replace(case_file)
            except Exception:
                if temp_file.exists():
                    try:
                        temp_file.unlink()
                    except OSError:
                        pass
                raise

            return to_public_case(updated_data)

    def list_cases(self) -> List[Dict[str, Any]]:
        """List all valid review cases in storage, sorted by creation date descending.

        Safely skips corrupted cases (fail-closed) and logs diagnostic.

        Returns:
            List of public case dictionaries with allowlisted keys only.
        """
        cases: List[Dict[str, Any]] = []
        if not self.root.exists() or not self.root.is_dir() or is_symlink(self.root):
            return cases

        for item in self.root.iterdir():
            if is_symlink(item):
                logger.warning(
                    "list_cases: skipping symlink item in storage root: %s", item.name
                )
                continue
            if not item.is_dir():
                continue
            case_id = item.name
            if not is_valid_uuid4(case_id):
                logger.warning(
                    "list_cases: skipping non-UUID4 directory in storage root: %s", case_id
                )
                continue
            try:
                case_data = self.load_persisted_case(case_id, verify_raw_sha=False)
                cases.append(to_public_case(case_data))
            except Exception as exc:
                logger.warning(
                    "list_cases: skipping corrupted case '%s': %s", case_id, exc
                )
                continue

        cases.sort(key=lambda c: str(c.get("created_at", "")), reverse=True)
        return cases

    def export_counts(self) -> Dict[str, Any]:
        """Export repetition counts formatted for evaluation.load_ground_truth.

        Verifies each case fail-closed. If any case is corrupt or tampered,
        raises CaseCorruptedError.

        Clips belonging to 'development' group or marked 'out_of_scope' have
        human_count forced to None to prevent evaluation leakage.

        Returns:
            Standard dictionary {'clips': [{...}]}.
        """
        with self._lock:
            if not self.root.exists() or not self.root.is_dir():
                return {"clips": []}
            if is_symlink(self.root):
                raise CaseCorruptedError(f"Storage root is a symlink: {self.root}")

            clips: List[Dict[str, Any]] = []
            for item in sorted(self.root.iterdir()):
                if is_symlink(item):
                    raise CaseCorruptedError(f"Found symlink in storage root: {item.name}")
                if not item.is_dir():
                    continue
                has_case_json = (item / "case.json").exists()
                if not is_valid_uuid4(item.name):
                    if has_case_json:
                        raise CaseCorruptedError(
                            f"Directory {item.name!r} has case.json but is not a valid UUID4 case ID"
                        )
                    continue

                # Strict fail-closed verification of metadata and raw video SHA
                case_data = self.load_persisted_case(item.name, verify_raw_sha=True)

                group = case_data.get("group", "candidate")
                scope = case_data.get("review", {}).get("scope", "in_scope")
                raw_count = case_data.get("review", {}).get("human_count")

                if group == "development" or scope == "out_of_scope":
                    human_count = None
                else:
                    human_count = raw_count

                clips.append({
                    "clip_id": case_data["case_id"],
                    "human_count": human_count,
                    "scope": scope,
                    "group": group,
                    "original_filename": case_data.get("original_filename", ""),
                    "source_sha256": case_data.get("source_sha256", ""),
                })

            return {"clips": clips}

    def export_reviewed_counts(self) -> Dict[str, Any]:
        """Export reviewed counts formatted for evaluation.load_ground_truth.

        Alias to export_counts for Task 43 compatibility.

        Returns:
            Standard dictionary {'clips': [{...}]}.
        """
        return self.export_counts()

    def export_manifest(self) -> Dict[str, Any]:
        """Export a comprehensive review manifest including cases, counts, clips, histories, events, exclusions.

        Verifies each case fail-closed. If any case is corrupt or tampered,
        raises CaseCorruptedError.

        Returns:
            Standard dictionary containing full audit manifest.
        """
        with self._lock:
            counts_export = self.export_counts()
            clips_list = counts_export["clips"]

            case_dirs = sorted([
                p for p in self.root.iterdir()
                if p.is_dir() and not is_symlink(p) and (is_valid_uuid4(p.name) or (p / "case.json").exists())
            ])
            for p in case_dirs:
                if not is_valid_uuid4(p.name):
                    raise CaseCorruptedError(
                        f"Directory {p.name!r} has case.json but is not a valid UUID4 case ID"
                    )

            full_cases = [
                self.load_persisted_case(cdir.name, verify_raw_sha=False)
                for cdir in case_dirs
            ]

            public_cases = [to_public_case(c) for c in full_cases]

            histories: Dict[str, List[Dict[str, Any]]] = {
                c["case_id"]: c.get("history", []) for c in full_cases
            }

            events: List[Dict[str, Any]] = [
                {"case_id": c["case_id"], **evt}
                for c in full_cases
                for evt in (c.get("review", {}).get("events") or [])
            ]

            exclusions: List[Dict[str, Any]] = [
                {
                    "case_id": c["case_id"],
                    "group": c.get("group", "candidate"),
                    "scope": c.get("review", {}).get("scope", "in_scope"),
                    "reason": (
                        "development"
                        if c.get("group") == "development"
                        else "out_of_scope"
                    ),
                    "original_filename": c.get("original_filename", ""),
                    "source_sha256": c.get("source_sha256", ""),
                }
                for c in full_cases
                if c.get("group") == "development"
                or c.get("review", {}).get("scope") == "out_of_scope"
            ]

            now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()
            return {
                "manifest_version": "1.0",
                "generated_at": now_iso,
                "total_cases": len(public_cases),
                "cases": public_cases,
                "counts": counts_export,
                "clips": clips_list,
                "histories": histories,
                "events": events,
                "exclusions": exclusions,
            }


__all__ = [
    "ALLOWED_EXTENSIONS",
    "DEFAULT_MAX_UPLOAD_BYTES",
    "DEFAULT_STORAGE_ROOT",
    "ALLOWED_GROUPS",
    "ALLOWED_SCOPES",
    "ALLOWED_INTENTS",
    "ALLOWED_CYCLES",
    "ALLOWED_EVENT_CLASSES",
    "PUBLIC_CASE_KEYS",
    "CaseNotFoundError",
    "CaseCorruptedError",
    "InvalidCaseIdError",
    "is_valid_uuid4",
    "is_valid_iso8601",
    "is_symlink",
    "detect_duplicate_pairs",
    "to_public_case",
    "validate_review_payload",
    "load_persisted_case",
    "ReviewStore",
]
