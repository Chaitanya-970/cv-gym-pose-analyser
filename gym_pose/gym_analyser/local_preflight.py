"""Local launch preflight validator for 2D Gym Pose Analyser.

Pure standard-library module and CLI inspecting concrete launch prerequisites
without initializing GPU, importing ML frameworks, loading model weights,
or mutating project source or environments.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import stat
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Union

# ==============================================================================
# Canonical Constants
# ==============================================================================

CANONICAL_MODEL_SHA256: str = (
    "869e83fcdffdc7371fa4e34cd8e51c838cc729571d1635e5141e3075e9319dc0"
)
CANONICAL_MODEL_RELPATH: Path = Path("models") / "yolo11n-pose.pt"
DEFAULT_WEB_STORAGE_RELPATH: Path = Path("output") / "web_jobs"
DEFAULT_REVIEW_STORAGE_RELPATH: Path = Path("output") / "human_reviews"

REQUIRED_WEB_MODULES: Sequence[str] = ("fastapi", "uvicorn")
REQUIRED_INFERENCE_MODULES: Sequence[str] = ("torch", "ultralytics", "cv2")

DEFAULT_PROBE_TIMEOUT: float = 7.0
DEFAULT_TOTAL_TIMEOUT: float = 30.0
MAX_PROBE_TIMEOUT: float = 60.0
MAX_TOTAL_TIMEOUT: float = 300.0

STATUS_READY: str = "ready"
STATUS_READY_WITH_WARNINGS: str = "ready_with_warnings"
STATUS_NOT_READY: str = "not_ready"

DISCLAIMER_RUNTIME_READINESS: str = (
    "Discoverability and distribution metadata indicate package presence only; "
    "they do NOT prove runtime, ABI, or CUDA/GPU execution readiness."
)

KNOWN_MODULE_DISTRIBUTIONS: Dict[str, Sequence[str]] = {
    "cv2": (
        "opencv-python",
        "opencv-python-headless",
        "opencv-contrib-python",
        "opencv-contrib-python-headless",
    ),
}

# Subprocess probe script executed via `python -c` in the target interpreter.
# Crucially: uses importlib.util.find_spec and importlib.metadata ONLY.
# ZERO __import__, import_module, or module code execution of any inspected module.
PROBE_SCRIPT: str = r"""
import sys
import json
import importlib.util
import importlib.metadata

extra = sys.argv[1] if len(sys.argv) > 1 else ""
if extra:
    for p in extra.split(";"):
        if p and p not in sys.path:
            sys.path.insert(0, p)

modules_to_check = sys.argv[2:] if len(sys.argv) > 2 else []

KNOWN_MODULE_DISTRIBUTIONS = {
    "cv2": [
        "opencv-python",
        "opencv-python-headless",
        "opencv-contrib-python",
        "opencv-contrib-python-headless",
    ],
}

pkg_dists = {}
try:
    pkg_dists = importlib.metadata.packages_distributions()
except Exception:
    pass

results = {}
for name in modules_to_check:
    spec_found = False
    spec_origin = None
    spec_err = None
    try:
        spec = importlib.util.find_spec(name)
        if spec is not None:
            spec_found = True
            spec_origin = getattr(spec, "origin", None)
    except Exception as exc:
        spec_err = f"{type(exc).__name__}: {exc}"

    candidate_dists = []
    if name in pkg_dists:
        for d in pkg_dists[name]:
            if d not in candidate_dists:
                candidate_dists.append(d)
    if name in KNOWN_MODULE_DISTRIBUTIONS:
        for d in KNOWN_MODULE_DISTRIBUTIONS[name]:
            if d not in candidate_dists:
                candidate_dists.append(d)
    if name not in candidate_dists:
        candidate_dists.append(name)

    dist_found = None
    ver_found = None
    dist_err = None
    for d in candidate_dists:
        try:
            ver_found = importlib.metadata.version(d)
            dist_found = d
            break
        except importlib.metadata.PackageNotFoundError:
            continue
        except Exception as exc:
            dist_err = f"{type(exc).__name__}: {exc}"

    # Metadata can survive removal of the module; it is not module discoverability.
    is_available = spec_found
    if is_available:
        err = None
    else:
        err = spec_err or dist_err or f"Module '{name}' not found via find_spec or distribution metadata"

    results[name] = {
        "available": is_available,
        "module_discoverable": spec_found,
        "distribution_metadata_found": dist_found is not None,
        "version": ver_found,
        "distribution": dist_found,
        "spec_origin": str(spec_origin) if spec_origin else None,
        "error": err,
    }

output = {
    "python_version": sys.version.replace("\n", " "),
    "platform": sys.platform,
    "modules": results,
    "runtime_readiness_disclaimer": (
        "Discoverability and distribution metadata indicate package presence only; "
        "they do NOT prove runtime, ABI, or CUDA/GPU execution readiness."
    ),
}
print(json.dumps(output))
"""


# ==============================================================================
# Pure Stdlib Symlink & Reparse Point Helpers
# ==============================================================================


def is_symlink(path: Union[str, Path]) -> bool:
    """Check if path is a symbolic link or Windows NTFS reparse point."""
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


def check_symlinks(path: Union[str, Path], label: str) -> Optional[str]:
    """Inspect unresolved path and its ancestors before resolve(), returning error if any is symlink/reparse."""
    try:
        raw_p = Path(path)
        # Check raw_p itself
        if is_symlink(raw_p):
            return f"{label} cannot be a symlink or reparse point: {raw_p}"
        # Check parents on raw_p before resolve
        for parent in raw_p.parents:
            if is_symlink(parent):
                return f"{label} has a symlink or reparse ancestor: {parent}"
        # Check parents on absolute path without resolving symlinks
        abs_p = raw_p.absolute()
        if is_symlink(abs_p):
            return f"{label} cannot be a symlink or reparse point: {abs_p}"
        for parent in abs_p.parents:
            if is_symlink(parent):
                return f"{label} has a symlink or reparse ancestor: {parent}"
    except Exception as exc:
        return f"{label} symlink check encountered error: {exc}"
    return None


# ==============================================================================
# Input Validation & Budget Enforcement
# ==============================================================================


def validate_timeout(
    val: Any,
    label: str,
    min_val: float = 0.1,
    max_val: float = 300.0,
) -> float:
    """Validate that timeout is a finite, positive float within [min_val, max_val]."""
    try:
        f_val = float(val)
    except (TypeError, ValueError):
        raise ValueError(f"{label} must be a valid float number, got {val!r}")
    if math.isnan(f_val):
        raise ValueError(f"{label} must be a finite number, got NaN")
    if math.isinf(f_val):
        raise ValueError(f"{label} must be a finite number, got infinity")
    if f_val <= 0:
        raise ValueError(f"{label} must be positive (> 0), got {f_val}")
    if f_val < min_val:
        raise ValueError(f"{label} must be at least {min_val}s, got {f_val}")
    if f_val > max_val:
        raise ValueError(f"{label} exceeds maximum allowed limit ({max_val}s), got {f_val}")
    return f_val


def is_subpath(target: Path, parent: Path) -> bool:
    """Check if target path is equal to or located inside parent directory."""
    try:
        target.relative_to(parent)
        return True
    except ValueError:
        return False


def validate_output_path(
    output_path: Union[str, Path],
    gym_pose_dir: Path,
    web_jobs_dir: Optional[Path] = None,
    human_reviews_dir: Optional[Path] = None,
) -> tuple[Optional[Path], Optional[str]]:
    """Validate report output destination path:
    - Checks unresolved path and ancestors for symlinks/reparse points before resolve().
    - Ensures destination is not inside protected source, models, tests, data, review, or job storage.
    - Ensures destination is not an existing directory.
    """
    raw_p = Path(output_path)
    sym_err = check_symlinks(raw_p, "Output report destination")
    if sym_err:
        return None, sym_err

    if raw_p.exists() and raw_p.is_dir():
        return None, f"Output report destination cannot be an existing directory: {raw_p}"

    base_res = gym_pose_dir.resolve()
    w_res = (web_jobs_dir or (base_res / DEFAULT_WEB_STORAGE_RELPATH)).resolve()
    r_res = (human_reviews_dir or (base_res / DEFAULT_REVIEW_STORAGE_RELPATH)).resolve()

    protected = [
        ("Source code", (base_res / "gym_analyser").resolve()),
        ("Tests", (base_res / "tests").resolve()),
        ("Models", (base_res / "models").resolve()),
        ("Test data", (base_res / "test_data").resolve()),
        ("Web jobs storage", w_res),
        ("Human reviews storage", r_res),
    ]
    data_dir = (base_res / "data").resolve()
    if data_dir.exists():
        protected.append(("Data", data_dir))

    abs_p = raw_p.absolute()
    try:
        resolved_p = abs_p.resolve()
    except Exception:
        resolved_p = abs_p

    for check_candidate in (abs_p, resolved_p):
        for prot_name, prot_dir in protected:
            if check_candidate == prot_dir or is_subpath(check_candidate, prot_dir):
                return None, f"Output report destination is inside protected {prot_name} directory ({prot_dir})"

    return resolved_p, None


def write_fresh_report(report_path: Path, content: str) -> None:
    """Safely write report content using atomic exclusive creation (O_CREAT | O_EXCL).

    Refuses to overwrite prior evidence, avoiding race conditions and clobbering.
    """
    if report_path.exists():
        raise FileExistsError(
            f"Report destination already exists: {report_path}. Overwriting prior evidence is prohibited."
        )

    parent = report_path.parent
    parent.mkdir(parents=True, exist_ok=True)

    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_BINARY"):
        flags |= os.O_BINARY

    fd = os.open(str(report_path), flags, 0o644)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(content)


# ==============================================================================
# Environment Discovery & Resolution
# ==============================================================================


def find_venv_python(venv_dir: Path) -> Optional[Path]:
    """Locate python executable in a virtual environment directory without resolving symlinks."""
    candidates = [
        venv_dir / "Scripts" / "python.exe",
        venv_dir / "bin" / "python",
        venv_dir / "Scripts" / "python",
    ]
    for c in candidates:
        if c.is_file() and not is_symlink(c):
            if check_symlinks(c, "Virtualenv python") is None:
                return c.resolve()
    return None


def discover_environments(gym_pose_dir: Path) -> Dict[str, Optional[str]]:
    """Discover standard virtual environments in gym_pose_dir."""
    envs = {
        ".venv": gym_pose_dir / ".venv",
        ".venv_cuda": gym_pose_dir / ".venv_cuda",
        ".venv_web": gym_pose_dir / ".venv_web",
    }
    discovered: Dict[str, Optional[str]] = {}
    for name, path in envs.items():
        py = find_venv_python(path)
        discovered[name] = str(py) if py else None
    return discovered


def resolve_web_interpreter(
    gym_pose_dir: Path,
    override: Optional[Union[str, Path]] = None,
) -> tuple[Path, str]:
    """Resolve web server interpreter following precedence: override -> .venv_web -> sys.executable."""
    if override:
        p = Path(override)
        return p.resolve(), "override"
    web_venv_py = find_venv_python(gym_pose_dir / ".venv_web")
    if web_venv_py:
        return web_venv_py, ".venv_web"
    return Path(sys.executable).resolve(), "sys.executable"


def resolve_inference_interpreter(
    gym_pose_dir: Path,
    override: Optional[Union[str, Path]] = None,
) -> tuple[Path, str]:
    """Resolve analysis worker interpreter following precedence: override -> .venv_cuda -> .venv -> sys.executable."""
    if override:
        p = Path(override)
        return p.resolve(), "override"
    for env_name in (".venv_cuda", ".venv"):
        env_py = find_venv_python(gym_pose_dir / env_name)
        if env_py:
            return env_py, env_name
    return Path(sys.executable).resolve(), "sys.executable"


# ==============================================================================
# Subprocess Probes (Bounded, Isolated, Safe - ZERO ML Imports)
# ==============================================================================


def probe_interpreter(
    interpreter_path: Path,
    required_modules: Sequence[str],
    timeout: float = DEFAULT_PROBE_TIMEOUT,
    extra_paths: Optional[Sequence[Union[str, Path]]] = None,
    env: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """Run a bounded child subprocess to check Python metadata and module discoverability without importing ML libraries."""
    symlink_err = check_symlinks(interpreter_path, "Interpreter")
    if symlink_err:
        return {
            "path": str(interpreter_path),
            "available": False,
            "python_version": None,
            "platform": None,
            "modules": {
                m: {"available": False, "version": None, "distribution": None, "error": symlink_err}
                for m in required_modules
            },
            "error": symlink_err,
            "runtime_readiness_disclaimer": DISCLAIMER_RUNTIME_READINESS,
        }

    if not interpreter_path.exists():
        return {
            "path": str(interpreter_path),
            "available": False,
            "python_version": None,
            "platform": None,
            "modules": {
                m: {"available": False, "version": None, "distribution": None, "error": "Interpreter file does not exist"}
                for m in required_modules
            },
            "error": f"Interpreter executable not found: {interpreter_path}",
            "runtime_readiness_disclaimer": DISCLAIMER_RUNTIME_READINESS,
        }
    if not interpreter_path.is_file():
        return {
            "path": str(interpreter_path),
            "available": False,
            "python_version": None,
            "platform": None,
            "modules": {
                m: {"available": False, "version": None, "distribution": None, "error": "Interpreter is not a regular file"}
                for m in required_modules
            },
            "error": f"Interpreter path is not a file: {interpreter_path}",
            "runtime_readiness_disclaimer": DISCLAIMER_RUNTIME_READINESS,
        }

    extra_str = ";".join(str(p) for p in extra_paths) if extra_paths else ""

    try:
        proc = subprocess.run(
            [str(interpreter_path), "-c", PROBE_SCRIPT, extra_str, *required_modules],
            capture_output=True,
            text=True,
            timeout=timeout,
            shell=False,
            env=env,
        )
    except subprocess.TimeoutExpired:
        return {
            "path": str(interpreter_path),
            "available": False,
            "python_version": None,
            "platform": None,
            "modules": {
                m: {"available": False, "version": None, "distribution": None, "error": f"Probe timed out after {timeout:.1f}s"}
                for m in required_modules
            },
            "error": f"Subprocess probe timed out after {timeout:.1f}s",
            "runtime_readiness_disclaimer": DISCLAIMER_RUNTIME_READINESS,
        }
    except Exception as exc:
        return {
            "path": str(interpreter_path),
            "available": False,
            "python_version": None,
            "platform": None,
            "modules": {
                m: {"available": False, "version": None, "distribution": None, "error": f"Probe error: {exc}"}
                for m in required_modules
            },
            "error": f"Failed to execute interpreter probe: {exc}",
            "runtime_readiness_disclaimer": DISCLAIMER_RUNTIME_READINESS,
        }

    if proc.returncode != 0:
        err_msg = proc.stderr.strip() or f"Process exited with code {proc.returncode}"
        return {
            "path": str(interpreter_path),
            "available": False,
            "python_version": None,
            "platform": None,
            "modules": {
                m: {"available": False, "version": None, "distribution": None, "error": err_msg}
                for m in required_modules
            },
            "error": f"Probe exited with non-zero code {proc.returncode}: {err_msg}",
            "runtime_readiness_disclaimer": DISCLAIMER_RUNTIME_READINESS,
        }

    try:
        data = json.loads(proc.stdout)
        return {
            "path": str(interpreter_path),
            "available": True,
            "python_version": data.get("python_version"),
            "platform": data.get("platform"),
            "modules": data.get("modules", {}),
            "error": None,
            "runtime_readiness_disclaimer": data.get(
                "runtime_readiness_disclaimer", DISCLAIMER_RUNTIME_READINESS
            ),
        }
    except Exception as exc:
        return {
            "path": str(interpreter_path),
            "available": False,
            "python_version": None,
            "platform": None,
            "modules": {
                m: {"available": False, "version": None, "distribution": None, "error": f"Malformed probe JSON: {exc}"}
                for m in required_modules
            },
            "error": f"Failed to parse probe JSON output: {exc}. Raw output: {proc.stdout.strip()[:200]}",
            "runtime_readiness_disclaimer": DISCLAIMER_RUNTIME_READINESS,
        }


# ==============================================================================
# Model Checkpoint Validation
# ==============================================================================


def check_model_checkpoint(
    model_path: Path,
    expected_sha256: str = CANONICAL_MODEL_SHA256,
) -> Dict[str, Any]:
    """Verify presence and canonical SHA-256 of the YOLO pose model checkpoint."""
    symlink_err = check_symlinks(model_path, "Model checkpoint")
    if symlink_err:
        return {
            "path": str(model_path),
            "exists": False,
            "is_file": False,
            "size_bytes": 0,
            "canonical_sha256": expected_sha256,
            "computed_sha256": None,
            "matches": False,
            "errors": [symlink_err],
        }

    if not model_path.exists():
        return {
            "path": str(model_path),
            "exists": False,
            "is_file": False,
            "size_bytes": 0,
            "canonical_sha256": expected_sha256,
            "computed_sha256": None,
            "matches": False,
            "errors": [f"Model checkpoint file not found: {model_path}"],
        }

    if not model_path.is_file():
        return {
            "path": str(model_path),
            "exists": True,
            "is_file": False,
            "size_bytes": 0,
            "canonical_sha256": expected_sha256,
            "computed_sha256": None,
            "matches": False,
            "errors": [f"Model checkpoint path is not a regular file: {model_path}"],
        }

    try:
        size = model_path.stat().st_size
        if size <= 0:
            return {
                "path": str(model_path),
                "exists": True,
                "is_file": True,
                "size_bytes": size,
                "canonical_sha256": expected_sha256,
                "computed_sha256": None,
                "matches": False,
                "errors": [f"Model checkpoint file is empty (0 bytes): {model_path}"],
            }

        hasher = hashlib.sha256()
        with open(model_path, "rb") as f:
            while chunk := f.read(64 * 1024):
                hasher.update(chunk)
        computed_sha = hasher.hexdigest().lower()
        matches = computed_sha == expected_sha256.lower()
        errors = [] if matches else [
            f"Model checkpoint SHA-256 mismatch: expected {expected_sha256}, got {computed_sha}"
        ]
        return {
            "path": str(model_path),
            "exists": True,
            "is_file": True,
            "size_bytes": size,
            "canonical_sha256": expected_sha256,
            "computed_sha256": computed_sha,
            "matches": matches,
            "errors": errors,
        }
    except Exception as exc:
        return {
            "path": str(model_path),
            "exists": True,
            "is_file": True,
            "size_bytes": 0,
            "canonical_sha256": expected_sha256,
            "computed_sha256": None,
            "matches": False,
            "errors": [f"Failed to read/hash model checkpoint: {exc}"],
        }


# ==============================================================================
# Optional FFmpeg Probe
# ==============================================================================


def probe_ffmpeg(
    ffmpeg_bin: Optional[Union[str, Path]] = None,
    timeout: float = 5.0,
) -> Dict[str, Any]:
    """Probe FFmpeg binary in PATH or specified location; missing FFmpeg is non-fatal warning."""
    bin_name = str(ffmpeg_bin) if ffmpeg_bin else shutil.which("ffmpeg")
    if not bin_name:
        return {
            "available": False,
            "path": None,
            "version": None,
            "warning": (
                "FFmpeg executable not detected in PATH. Browser video preview generation "
                "will be disabled; original video analysis and downloads are unaffected."
            ),
        }

    bin_path = Path(bin_name)
    sym_err = check_symlinks(bin_path, "FFmpeg binary")
    if sym_err:
        return {
            "available": False,
            "path": str(bin_path),
            "version": None,
            "warning": f"FFmpeg path rejected: {sym_err}",
        }

    try:
        proc = subprocess.run(
            [str(bin_path), "-nostdin", "-version"],
            capture_output=True,
            text=True,
            timeout=timeout,
            shell=False,
        )
        if proc.returncode != 0:
            return {
                "available": False,
                "path": str(bin_path),
                "version": None,
                "warning": f"FFmpeg version check returned non-zero code {proc.returncode}",
            }
        match = re.search(r"ffmpeg version\s+([^\s]+)", proc.stdout)
        ver = match.group(1) if match else "unknown"
        return {
            "available": True,
            "path": str(bin_path.resolve()),
            "version": ver,
            "warning": None,
        }
    except subprocess.TimeoutExpired:
        return {
            "available": False,
            "path": str(bin_path),
            "version": None,
            "warning": f"FFmpeg version check timed out after {timeout:.1f}s",
        }
    except Exception as exc:
        return {
            "available": False,
            "path": str(bin_path),
            "version": None,
            "warning": f"FFmpeg probe failed: {exc}",
        }


# ==============================================================================
# Storage Root Separation Check
# ==============================================================================


def check_storage_separation(
    gym_pose_dir: Path,
    web_jobs_dir: Path,
    human_reviews_dir: Path,
) -> Dict[str, Any]:
    """Verify that web job storage and human review storage are strictly separate directories."""
    errors: List[str] = []

    for p, label in [
        (web_jobs_dir, "Web jobs storage"),
        (human_reviews_dir, "Human reviews storage"),
    ]:
        err = check_symlinks(p, label)
        if err:
            errors.append(err)

    web_res = web_jobs_dir.resolve()
    rev_res = human_reviews_dir.resolve()
    app_res = gym_pose_dir.resolve()

    # Direct identity check
    if web_res == rev_res:
        errors.append(
            f"Storage path collision: web jobs storage and human reviews storage point to the same directory ({web_res})"
        )

    # Relative ancestor / descendant containment check
    try:
        web_res.relative_to(rev_res)
        errors.append(
            f"Storage path collision: web jobs storage ({web_res}) is nested inside human reviews storage ({rev_res})"
        )
    except ValueError:
        pass

    try:
        rev_res.relative_to(web_res)
        errors.append(
            f"Storage path collision: human reviews storage ({rev_res}) is nested inside web jobs storage ({web_res})"
        )
    except ValueError:
        pass

    # Protected paths overlap check
    protected_dirs = [
        app_res / "gym_analyser",
        app_res / "tests",
        app_res / "models",
        app_res / "test_data",
    ]
    for p_dir in protected_dirs:
        p_res = p_dir.resolve()
        for s_dir, s_label in [
            (web_res, "Web jobs storage"),
            (rev_res, "Human reviews storage"),
        ]:
            if s_dir == p_res:
                errors.append(
                    f"{s_label} ({s_dir}) collides directly with protected directory ({p_res})"
                )
            else:
                try:
                    s_dir.relative_to(p_res)
                    errors.append(
                        f"{s_label} ({s_dir}) is located inside protected directory ({p_res})"
                    )
                except ValueError:
                    pass
                try:
                    p_res.relative_to(s_dir)
                    errors.append(
                        f"Protected directory ({p_res}) is inside {s_label.lower()} ({s_dir})"
                    )
                except ValueError:
                    pass

    return {
        "web_jobs_dir": str(web_jobs_dir),
        "human_reviews_dir": str(human_reviews_dir),
        "disjoint": len(errors) == 0,
        "web_jobs_exists": web_jobs_dir.exists(),
        "human_reviews_exists": human_reviews_dir.exists(),
        "errors": errors,
    }


# ==============================================================================
# Main Preflight Orchestration
# ==============================================================================


def run_preflight(
    gym_pose_dir: Optional[Union[str, Path]] = None,
    web_interpreter: Optional[Union[str, Path]] = None,
    inference_interpreter: Optional[Union[str, Path]] = None,
    model_path: Optional[Union[str, Path]] = None,
    web_jobs_dir: Optional[Union[str, Path]] = None,
    human_reviews_dir: Optional[Union[str, Path]] = None,
    ffmpeg_bin: Optional[Union[str, Path]] = None,
    probe_timeout: float = DEFAULT_PROBE_TIMEOUT,
    total_timeout: float = DEFAULT_TOTAL_TIMEOUT,
) -> Dict[str, Any]:
    """Execute complete read-only local preflight check and return structured report."""
    # 1. Reject non-finite, non-positive, or unbounded timeout inputs
    probe_t_val = validate_timeout(
        probe_timeout, "probe_timeout", min_val=0.1, max_val=MAX_PROBE_TIMEOUT
    )
    total_t_val = validate_timeout(
        total_timeout, "total_timeout", min_val=0.5, max_val=MAX_TOTAL_TIMEOUT
    )

    start_time = time.monotonic()
    timestamp_utc = datetime.datetime.now(datetime.timezone.utc).isoformat()

    errors: List[str] = []
    warnings: List[str] = []

    # Check unresolved gym_pose_dir before resolve
    if gym_pose_dir:
        root_symlink_err = check_symlinks(gym_pose_dir, "Gym pose root directory")
        if root_symlink_err:
            errors.append(root_symlink_err)
        base_dir = Path(gym_pose_dir).resolve()
    else:
        base_dir = Path(__file__).resolve().parent.parent

    if not base_dir.exists():
        errors.append(f"Gym pose root directory does not exist: {base_dir}")

    # Discover environments
    discovered = discover_environments(base_dir)

    # Check unresolved interpreter overrides before resolve
    if web_interpreter:
        web_sym_err = check_symlinks(web_interpreter, "Web interpreter override")
        if web_sym_err:
            errors.append(web_sym_err)
    if inference_interpreter:
        inf_sym_err = check_symlinks(inference_interpreter, "Inference interpreter override")
        if inf_sym_err:
            errors.append(inf_sym_err)

    # Resolve interpreters following actual system precedence
    web_py, web_src = resolve_web_interpreter(base_dir, web_interpreter)
    inf_py, inf_src = resolve_inference_interpreter(base_dir, inference_interpreter)

    # Bounded probe: Web interpreter (budget-aware)
    rem_web = total_t_val - (time.monotonic() - start_time)
    if rem_web <= 0:
        web_probe = {
            "path": str(web_py),
            "available": False,
            "python_version": None,
            "platform": None,
            "modules": {
                m: {
                    "available": False,
                    "version": None,
                    "distribution": None,
                    "error": f"Skipped: total timeout budget ({total_t_val}s) expired",
                }
                for m in REQUIRED_WEB_MODULES
            },
            "error": f"Web interpreter probe skipped: total timeout budget ({total_t_val}s) expired",
            "skipped": True,
            "runtime_readiness_disclaimer": DISCLAIMER_RUNTIME_READINESS,
        }
        errors.append(f"Web interpreter probe skipped: total timeout budget ({total_t_val}s) expired")
    else:
        probe_t = min(probe_t_val, rem_web)
        web_probe = probe_interpreter(web_py, REQUIRED_WEB_MODULES, timeout=probe_t)
        if not web_probe.get("available"):
            errors.append(
                f"Web interpreter probe failed ({web_src}: {web_py}): {web_probe.get('error')}"
            )
        else:
            for m in REQUIRED_WEB_MODULES:
                m_res = web_probe.get("modules", {}).get(m, {})
                if not m_res.get("available"):
                    errors.append(
                        f"Missing required web module '{m}' in web interpreter ({web_py}): {m_res.get('error')}"
                    )

    # Bounded probe: Inference interpreter (budget-aware)
    rem_inf = total_t_val - (time.monotonic() - start_time)
    if rem_inf <= 0:
        inf_probe = {
            "path": str(inf_py),
            "available": False,
            "python_version": None,
            "platform": None,
            "modules": {
                m: {
                    "available": False,
                    "version": None,
                    "distribution": None,
                    "error": f"Skipped: total timeout budget ({total_t_val}s) expired",
                }
                for m in REQUIRED_INFERENCE_MODULES
            },
            "error": f"Inference interpreter probe skipped: total timeout budget ({total_t_val}s) expired",
            "skipped": True,
            "runtime_readiness_disclaimer": DISCLAIMER_RUNTIME_READINESS,
        }
        errors.append(f"Inference interpreter probe skipped: total timeout budget ({total_t_val}s) expired")
    else:
        probe_t = min(probe_t_val, rem_inf)
        inf_probe = probe_interpreter(inf_py, REQUIRED_INFERENCE_MODULES, timeout=probe_t)
        if not inf_probe.get("available"):
            errors.append(
                f"Inference interpreter probe failed ({inf_src}: {inf_py}): {inf_probe.get('error')}"
            )
        else:
            for m in REQUIRED_INFERENCE_MODULES:
                m_res = inf_probe.get("modules", {}).get(m, {})
                if not m_res.get("available"):
                    errors.append(
                        f"Missing required inference module '{m}' in inference interpreter ({inf_py}): {m_res.get('error')}"
                    )

    # Base model checkpoint check
    if model_path:
        model_sym_err = check_symlinks(model_path, "Model checkpoint path")
        if model_sym_err:
            errors.append(model_sym_err)
        target_model = Path(model_path).resolve()
    else:
        target_model = (base_dir / CANONICAL_MODEL_RELPATH).resolve()

    ckpt_res = check_model_checkpoint(target_model, expected_sha256=CANONICAL_MODEL_SHA256)
    if ckpt_res.get("errors"):
        errors.extend(ckpt_res["errors"])

    # Optional FFmpeg probe (budget-aware)
    rem_ff = total_t_val - (time.monotonic() - start_time)
    if rem_ff <= 0:
        ffmpeg_res = {
            "available": False,
            "path": None,
            "version": None,
            "warning": f"FFmpeg probe skipped: total timeout budget ({total_t_val}s) expired",
            "skipped": True,
        }
        warnings.append(f"FFmpeg probe skipped: total timeout budget ({total_t_val}s) expired")
    else:
        ffmpeg_t = min(min(probe_t_val, 5.0), rem_ff)
        ffmpeg_res = probe_ffmpeg(ffmpeg_bin, timeout=ffmpeg_t)
        if not ffmpeg_res.get("available") and ffmpeg_res.get("warning"):
            warnings.append(ffmpeg_res["warning"])

    # Storage root separation check
    if web_jobs_dir:
        wj_sym_err = check_symlinks(web_jobs_dir, "Web jobs storage")
        if wj_sym_err:
            errors.append(wj_sym_err)
    if human_reviews_dir:
        hr_sym_err = check_symlinks(human_reviews_dir, "Human reviews storage")
        if hr_sym_err:
            errors.append(hr_sym_err)

    w_dir = (
        Path(web_jobs_dir).resolve()
        if web_jobs_dir
        else (base_dir / DEFAULT_WEB_STORAGE_RELPATH).resolve()
    )
    r_dir = (
        Path(human_reviews_dir).resolve()
        if human_reviews_dir
        else (base_dir / DEFAULT_REVIEW_STORAGE_RELPATH).resolve()
    )
    storage_res = check_storage_separation(base_dir, w_dir, r_dir)
    if storage_res.get("errors"):
        errors.extend(storage_res["errors"])

    # Determine overall status
    if errors:
        status = STATUS_NOT_READY
    elif warnings:
        status = STATUS_READY_WITH_WARNINGS
    else:
        status = STATUS_READY

    elapsed_seconds = round(time.monotonic() - start_time, 3)

    return {
        "status": status,
        "timestamp_utc": timestamp_utc,
        "elapsed_seconds": elapsed_seconds,
        "gym_pose_dir": str(base_dir),
        "disclaimer": DISCLAIMER_RUNTIME_READINESS,
        "environments": {
            "discovered": discovered,
            "selected_web": {
                "source": web_src,
                **web_probe,
            },
            "selected_inference": {
                "source": inf_src,
                **inf_probe,
            },
        },
        "checkpoint": ckpt_res,
        "ffmpeg": ffmpeg_res,
        "storage": storage_res,
        "summary": {
            "status": status,
            "all_required_checks_passed": len(errors) == 0,
            "human_readiness": "Pending raw video reviews/labels; not a software launch blocker.",
            "runtime_readiness_disclaimer": DISCLAIMER_RUNTIME_READINESS,
            "errors": errors,
            "warnings": warnings,
        },
    }


# ==============================================================================
# Text Formatting
# ==============================================================================


def format_text_report(report: Dict[str, Any]) -> str:
    """Format structured preflight report into clear, human-readable terminal output."""
    status = report.get("status", "unknown").upper()
    lines = [
        "=" * 78,
        "2D Gym Pose Analyser: Local Launch Preflight Check",
        "=" * 78,
        f"Overall Status: {status}",
        f"Timestamp (UTC): {report.get('timestamp_utc')}",
        f"Elapsed Time:   {report.get('elapsed_seconds')}s",
        f"Project Root:   {report.get('gym_pose_dir')}",
        "",
        "Notice: Discoverability and version metadata indicate package presence only;",
        "        they do NOT prove runtime, ABI, or CUDA/GPU execution readiness.",
        "",
        "--- Discovered Python Environments ---",
    ]

    discovered = report.get("environments", {}).get("discovered", {})
    for env_name in (".venv", ".venv_cuda", ".venv_web"):
        p = discovered.get(env_name)
        status_label = f"Found: {p}" if p else "Not found"
        lines.append(f"  {env_name:12}: {status_label}")

    lines.append("")
    lines.append("--- Selected Interpreters & Modules ---")

    web_sel = report.get("environments", {}).get("selected_web", {})
    lines.append(f"  Web Server ({web_sel.get('source')}): {web_sel.get('path')}")
    if web_sel.get("available"):
        lines.append(f"    Python: {web_sel.get('python_version')}")
        for m, m_info in web_sel.get("modules", {}).items():
            st = "OK" if m_info.get("available") else f"FAIL ({m_info.get('error')})"
            dist = (
                f" [{m_info.get('distribution')}]"
                if m_info.get("distribution") and m_info.get("distribution") != m
                else ""
            )
            ver = f" (v{m_info.get('version')})" if m_info.get("version") else ""
            lines.append(f"    - {m}: {st}{dist}{ver}")
    else:
        lines.append(f"    Status: FAILED - {web_sel.get('error')}")

    inf_sel = report.get("environments", {}).get("selected_inference", {})
    lines.append(f"  Inference Worker ({inf_sel.get('source')}): {inf_sel.get('path')}")
    if inf_sel.get("available"):
        lines.append(f"    Python: {inf_sel.get('python_version')}")
        for m, m_info in inf_sel.get("modules", {}).items():
            st = "OK" if m_info.get("available") else f"FAIL ({m_info.get('error')})"
            dist = (
                f" [{m_info.get('distribution')}]"
                if m_info.get("distribution") and m_info.get("distribution") != m
                else ""
            )
            ver = f" (v{m_info.get('version')})" if m_info.get("version") else ""
            lines.append(f"    - {m}: {st}{dist}{ver}")
    else:
        lines.append(f"    Status: FAILED - {inf_sel.get('error')}")

    lines.append("")
    lines.append("--- Model Checkpoint ---")
    ckpt = report.get("checkpoint", {})
    lines.append(f"  Path:             {ckpt.get('path')}")
    if ckpt.get("matches"):
        lines.append(
            f"  Status:           VALID (canonical SHA-256 matched, {ckpt.get('size_bytes'):,} bytes)"
        )
    else:
        lines.append("  Status:           FAILED")
        for err in ckpt.get("errors", []):
            lines.append(f"    Error: {err}")
    lines.append(f"  Canonical SHA-256: {ckpt.get('canonical_sha256')}")
    if ckpt.get("computed_sha256"):
        lines.append(f"  Computed SHA-256:  {ckpt.get('computed_sha256')}")

    lines.append("")
    lines.append("--- FFmpeg Video Preview ---")
    ff = report.get("ffmpeg", {})
    if ff.get("available"):
        lines.append(f"  Status:   AVAILABLE (version: {ff.get('version')})")
        lines.append(f"  Path:     {ff.get('path')}")
    else:
        lines.append("  Status:   OPTIONAL PREVIEW DEGRADED (non-fatal)")
        lines.append(f"  Notice:   {ff.get('warning')}")

    lines.append("")
    lines.append("--- Storage Root Separation ---")
    storage = report.get("storage", {})
    lines.append(f"  Web Jobs Root:      {storage.get('web_jobs_dir')}")
    lines.append(f"  Human Reviews Root:  {storage.get('human_reviews_dir')}")
    if storage.get("disjoint"):
        lines.append("  Separation:         VERIFIED DISJOINT (no path collisions or nesting)")
    else:
        lines.append("  Separation:         FAILED (collision detected)")
        for err in storage.get("errors", []):
            lines.append(f"    Error: {err}")

    lines.append("")
    lines.append("--- Preflight Summary ---")
    summary = report.get("summary", {})
    lines.append(f"  All Required Checks Passed: {summary.get('all_required_checks_passed')}")
    lines.append(f"  Human Readiness:            {summary.get('human_readiness')}")

    errors = summary.get("errors", [])
    if errors:
        lines.append(f"  Errors ({len(errors)}):")
        for err in errors:
            lines.append(f"    [X] {err}")
    else:
        lines.append("  Errors: None")

    warnings = summary.get("warnings", [])
    if warnings:
        lines.append(f"  Warnings ({len(warnings)}):")
        for warn in warnings:
            lines.append(f"    [!] {warn}")
    else:
        lines.append("  Warnings: None")

    lines.append("=" * 78)
    return "\n".join(lines)


# ==============================================================================
# CLI Entry Point
# ==============================================================================


def build_parser() -> argparse.ArgumentParser:
    """Build preflight command-line argument parser."""
    parser = argparse.ArgumentParser(
        prog="python -m gym_analyser.local_preflight",
        description="2D Gym Pose Analyser: reproducible local launch preflight validator.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output structured preflight report as JSON to stdout.",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        metavar="PATH",
        help="Save structured JSON preflight report to specified path (exclusive creation; never overwrites).",
    )
    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Suppress human-readable text output on stdout.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        default=False,
        help="Prohibited: preflight reports require a fresh destination to preserve prior evidence.",
    )
    parser.add_argument(
        "--root",
        "--gym-pose-dir",
        dest="gym_pose_dir",
        type=str,
        default=None,
        help="Explicit project root directory override (defaults to package parent).",
    )
    parser.add_argument(
        "--web-interpreter",
        type=str,
        default=None,
        help="Explicit web interpreter path override.",
    )
    parser.add_argument(
        "--inference-interpreter",
        type=str,
        default=None,
        help="Explicit inference interpreter path override.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Explicit model checkpoint path override.",
    )
    parser.add_argument(
        "--web-jobs-dir",
        type=str,
        default=None,
        help="Explicit web jobs storage directory override.",
    )
    parser.add_argument(
        "--human-reviews-dir",
        type=str,
        default=None,
        help="Explicit human reviews storage directory override.",
    )
    parser.add_argument(
        "--probe-timeout",
        type=float,
        default=DEFAULT_PROBE_TIMEOUT,
        help=f"Per-probe subprocess timeout in seconds (default: {DEFAULT_PROBE_TIMEOUT}).",
    )
    parser.add_argument(
        "--total-timeout",
        type=float,
        default=DEFAULT_TOTAL_TIMEOUT,
        help=f"Total preflight execution timeout in seconds (default: {DEFAULT_TOTAL_TIMEOUT}).",
    )
    return parser


def main() -> None:
    """CLI entry point for local launch preflight."""
    parser = build_parser()
    args = parser.parse_args()

    # Reject --overwrite explicitly: overwriting evidence is strictly prohibited
    if args.overwrite:
        sys.stderr.write(
            "Error: --overwrite is prohibited. Preflight report destinations must be fresh "
            "to prevent overwriting prior evidence.\n"
        )
        sys.exit(2)
        return

    try:
        probe_t = validate_timeout(
            args.probe_timeout, "probe-timeout", min_val=0.1, max_val=MAX_PROBE_TIMEOUT
        )
        total_t = validate_timeout(
            args.total_timeout, "total-timeout", min_val=0.5, max_val=MAX_TOTAL_TIMEOUT
        )
    except ValueError as exc:
        sys.stderr.write(f"Error: Invalid timeout parameter: {exc}\n")
        sys.exit(2)
        return

    # Base dir for validation
    if args.gym_pose_dir:
        sym_err = check_symlinks(args.gym_pose_dir, "Project root directory")
        if sym_err:
            sys.stderr.write(f"Error: {sym_err}\n")
            sys.exit(2)
            return
        base_dir = Path(args.gym_pose_dir).resolve()
    else:
        base_dir = Path(__file__).resolve().parent.parent

    w_dir = (
        Path(args.web_jobs_dir).resolve()
        if args.web_jobs_dir
        else (base_dir / DEFAULT_WEB_STORAGE_RELPATH).resolve()
    )
    r_dir = (
        Path(args.human_reviews_dir).resolve()
        if args.human_reviews_dir
        else (base_dir / DEFAULT_REVIEW_STORAGE_RELPATH).resolve()
    )

    # If --output is provided, validate BEFORE running preflight
    valid_out_path = None
    if args.output:
        valid_out_path, val_err = validate_output_path(
            args.output,
            gym_pose_dir=base_dir,
            web_jobs_dir=w_dir,
            human_reviews_dir=r_dir,
        )
        if val_err:
            sys.stderr.write(f"Error: Invalid output path: {val_err}\n")
            sys.exit(2)
            return
        if valid_out_path.exists():
            sys.stderr.write(
                f"Error: Refusing to overwrite existing preflight report file: {valid_out_path}\n"
                "Report destination must be fresh; overwriting prior evidence is prohibited.\n"
            )
            sys.exit(2)
            return

    try:
        report = run_preflight(
            gym_pose_dir=args.gym_pose_dir,
            web_interpreter=args.web_interpreter,
            inference_interpreter=args.inference_interpreter,
            model_path=args.model,
            web_jobs_dir=args.web_jobs_dir,
            human_reviews_dir=args.human_reviews_dir,
            probe_timeout=probe_t,
            total_timeout=total_t,
        )
    except ValueError as exc:
        sys.stderr.write(f"Error: Preflight configuration error: {exc}\n")
        sys.exit(2)
        return

    if valid_out_path:
        try:
            write_fresh_report(valid_out_path, json.dumps(report, indent=2))
        except FileExistsError:
            sys.stderr.write(
                f"Error: Refusing to overwrite existing preflight report file: {valid_out_path}\n"
                "Report destination must be fresh; overwriting prior evidence is prohibited.\n"
            )
            sys.exit(2)
            return
        except OSError as exc:
            sys.stderr.write(f"Error: Failed to write report file ({valid_out_path}): {exc}\n")
            sys.exit(2)
            return

    if args.json:
        if not args.quiet:
            print(json.dumps(report, indent=2))
    elif not args.quiet:
        print(format_text_report(report))

    # Exit code: 0 if ready or ready_with_warnings, 1 if not_ready
    if report.get("status") == STATUS_NOT_READY:
        sys.exit(1)
        return
    sys.exit(0)
    return


if __name__ == "__main__":
    main()
