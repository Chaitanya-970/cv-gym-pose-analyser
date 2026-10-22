"""Job storage and single-worker execution orchestration for gym_analyser web interface.

Provides testable, isolated management of video analysis jobs:
- UUID4 job identifiers and dedicated storage under output/web_jobs/<id>/
- Enforced upload limits, chunked copy, non-empty validation, and cleanup
- Bounded single-worker concurrency with guaranteed busy-state release
- Subprocess CLI execution with timeout, private logging, and status inspection
- Artifact download security via strict allowlist and path-traversal prevention
- Server restart recovery for interrupted jobs
"""

import json
import math
import os
import shutil
import subprocess
import sys
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

# Supported video extensions for uploaded files
ALLOWED_EXTENSIONS: Set[str] = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}

# Default upload limit: 200 MB
DEFAULT_MAX_UPLOAD_BYTES: int = 200 * 1024 * 1024

# Exact allowlist of downloadable job artifacts and their media types
ALLOWED_ARTIFACTS: Dict[str, str] = {
    "heel_review.json": "application/json",
    "knee_review.json": "application/json",
    "back_review.json": "application/json",
    "summary.json": "application/json",
    "attempts.json": "application/json",
    "posture_summary.json": "application/json",
    "feedback_observations.json": "application/json",
    "feedback_phases.json": "application/json",
    "coaching_feedback.json": "application/json",
    "angles.csv": "text/csv; charset=utf-8",
    "angle_plot.png": "image/png",
    "annotated.mp4": "video/mp4",
    "preview.mp4": "video/mp4",
}

# Standard job lifecycle statuses
STATUS_QUEUED = "queued"
STATUS_RUNNING = "running"
STATUS_COMPLETED = "completed"
STATUS_INCOMPLETE = "incomplete"
STATUS_FAILED = "failed"
STATUS_INTERRUPTED = "interrupted"

TERMINAL_STATUSES: Set[str] = {
    STATUS_COMPLETED,
    STATUS_INCOMPLETE,
    STATUS_FAILED,
    STATUS_INTERRUPTED,
}


class JobBusyError(Exception):
    """Raised when an analysis job is submitted while another job is active."""
    pass


def is_valid_job_id(job_id: str) -> bool:
    """Validate that a string is a canonical UUID4 identifier."""
    if not isinstance(job_id, str) or len(job_id) != 36:
        return False
    try:
        parsed = uuid.UUID(job_id, version=4)
        return str(parsed).lower() == job_id.lower()
    except (ValueError, AttributeError, TypeError):
        return False


def _current_iso_time() -> str:
    """Return current UTC timestamp in ISO 8601 format."""
    return datetime.now(timezone.utc).isoformat()


class JobManager:
    """Manages file storage, concurrency, and subprocess execution for analysis jobs."""

    def __init__(
        self,
        storage_dir: Path,
        device: str = "cpu",
        model_path: Optional[Path] = None,
        gym_pose_dir: Optional[Path] = None,
        max_upload_bytes: int = DEFAULT_MAX_UPLOAD_BYTES,
        timeout_seconds: float = 600.0,
        runner: Optional[Callable[..., Dict[str, Any]]] = None,
    ) -> None:
        self.storage_dir = Path(storage_dir).resolve()
        self.device = str(device)
        self.max_upload_bytes = int(max_upload_bytes)
        self.timeout_seconds = float(timeout_seconds)
        if self.max_upload_bytes < 1 or not math.isfinite(self.timeout_seconds) or self.timeout_seconds <= 0:
            raise ValueError("Upload limit and timeout must be positive and finite")

        # Resolve gym_pose root directory and fixed model path
        if gym_pose_dir is not None:
            self.gym_pose_dir = Path(gym_pose_dir).resolve()
        else:
            self.gym_pose_dir = Path(__file__).resolve().parent.parent

        candidates = [self.gym_pose_dir / env / executable
                      for env in (".venv_cuda", ".venv")
                      for executable in ("Scripts/python.exe", "bin/python")]
        self.analysis_python = next((path for path in candidates if path.is_file()), Path(sys.executable))

        if model_path is not None:
            self.model_path = Path(model_path).resolve()
        else:
            self.model_path = (self.gym_pose_dir / "models" / "yolo11n-pose.pt").resolve()

        # Injected or default CLI runner
        self.runner = runner if runner is not None else self._default_cli_runner

        # Single-worker concurrency state
        self._lock = threading.Lock()
        self._active_job_id: Optional[str] = None

        # Ensure root storage exists and recover any interrupted jobs from previous run
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.recover_interrupted_jobs()

    def is_busy(self) -> bool:
        """Check whether a job is currently queued or executing."""
        with self._lock:
            return self._active_job_id is not None

    def get_active_job_id(self) -> Optional[str]:
        """Return the ID of the currently active job, if any."""
        with self._lock:
            return self._active_job_id

    def recover_interrupted_jobs(self) -> None:
        """Mark any dangling queued or running jobs as interrupted on server start."""
        if not self.storage_dir.exists():
            return
        for entry in self.storage_dir.iterdir():
            if entry.is_dir() and is_valid_job_id(entry.name):
                meta = self.get_metadata(entry.name)
                if meta and meta.get("status") in (STATUS_QUEUED, STATUS_RUNNING):
                    meta["status"] = STATUS_INTERRUPTED
                    meta["finished_at"] = _current_iso_time()
                    meta["error"] = "Job was interrupted by server shutdown or restart."
                    self._save_metadata(entry.name, meta)

    def _get_job_dir(self, job_id: str) -> Path:
        """Get verified job directory path preventing path traversal."""
        if not is_valid_job_id(job_id):
            raise ValueError(f"Invalid job ID: '{job_id}'")
        job_dir = (self.storage_dir / job_id).resolve()
        if job_dir.parent != self.storage_dir:
            raise ValueError(f"Security error: path traversal detected for job ID '{job_id}'")
        return job_dir

    def get_metadata(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Load job metadata from job.json, or None if not found."""
        try:
            job_dir = self._get_job_dir(job_id)
            meta_path = job_dir / "job.json"
            if not meta_path.is_file():
                return None
            with open(meta_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None

    def _save_metadata(self, job_id: str, data: Dict[str, Any]) -> None:
        """Persist metadata dictionary to job.json."""
        job_dir = self._get_job_dir(job_id)
        job_dir.mkdir(parents=True, exist_ok=True)
        meta_path = job_dir / "job.json"
        tmp_path = job_dir / "job.json.tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        for attempt in range(6):
            try:
                tmp_path.replace(meta_path)
                break
            except PermissionError:
                if attempt == 5:
                    raise
                time.sleep(0.05 * (2 ** attempt))

    def save_upload_file(
        self,
        file_obj: Any,
        raw_filename: str,
    ) -> Tuple[str, Path, int]:
        """Validate, chunk-copy, and initialize a new video upload job.

        Returns (job_id, input_video_path, total_bytes).
        Cleans up partially written files and directories on any validation error.
        """
        # Validate filename and extract clean suffix
        display_name = Path(raw_filename).name if raw_filename else "video.mp4"
        suffix = Path(display_name).suffix.lower()
        if suffix not in ALLOWED_EXTENSIONS:
            raise ValueError(
                f"Unsupported video format '{suffix}'. Allowed formats: {', '.join(sorted(ALLOWED_EXTENSIONS))}"
            )

        job_id = str(uuid.uuid4())
        job_dir = self._get_job_dir(job_id)
        input_dir = job_dir / "input"
        results_dir = job_dir / "results"
        input_dir.mkdir(parents=True, exist_ok=True)
        results_dir.mkdir(parents=True, exist_ok=True)

        target_file = input_dir / f"source{suffix}"
        total_bytes = 0
        chunk_size = 1024 * 1024  # 1MB chunks

        try:
            with open(target_file, "wb") as dest:
                while True:
                    chunk = file_obj.read(chunk_size)
                    if not chunk:
                        break
                    total_bytes += len(chunk)
                    if total_bytes > self.max_upload_bytes:
                        raise ValueError(
                            f"File size exceeds maximum upload limit of {self.max_upload_bytes // (1024 * 1024)} MB."
                        )
                    dest.write(chunk)

            if total_bytes == 0:
                raise ValueError("Uploaded file is empty (0 bytes).")

            # Persist initial queued metadata
            metadata: Dict[str, Any] = {
                "job_id": job_id,
                "status": STATUS_QUEUED,
                "created_at": _current_iso_time(),
                "started_at": None,
                "finished_at": None,
                "original_filename": display_name,
                "file_size_bytes": total_bytes,
                "device": self.device,
                "error": None,
                "read_status_note": None,
                "exit_code": None,
            }
            self._save_metadata(job_id, metadata)
            return job_id, target_file, total_bytes

        except Exception:
            # Clean up on failure to prevent stale debris
            if target_file.exists():
                try:
                    target_file.unlink()
                except OSError:
                    pass
            if job_dir.exists():
                try:
                    shutil.rmtree(job_dir, ignore_errors=True)
                except OSError:
                    pass
            raise

    def enqueue_job(
        self,
        job_id: str,
        background_tasks: Optional[Any] = None,
    ) -> None:
        """Enqueue a validated job for background execution.

        Checks single-worker busy state under lock. Releases lock and schedules worker.
        """
        with self._lock:
            if self._active_job_id is not None:
                raise JobBusyError(
                    f"Server is busy: an analysis job is currently active ({self._active_job_id})."
                )
            self._active_job_id = job_id

        # Scheduling itself can fail: release the reservation as well as worker failures.
        try:
            if background_tasks is not None and hasattr(background_tasks, "add_task"):
                background_tasks.add_task(self._execute_job, job_id)
            else:
                thread = threading.Thread(target=self._execute_job, args=(job_id,), daemon=True,
                                          name=f"worker-{job_id[:8]}")
                thread.start()
        except Exception:
            with self._lock:
                if self._active_job_id == job_id:
                    self._active_job_id = None
            raise

    def _execute_job(self, job_id: str) -> None:
        """Run one job; even metadata failures must release the busy state."""
        meta = None
        try:
            meta = self.get_metadata(job_id)
            if not meta:
                return
            job_dir = self._get_job_dir(job_id)
            inputs = list((job_dir / "input").glob("source.*"))
            if len(inputs) != 1 or inputs[0].resolve().parent != job_dir / "input":
                raise ValueError("Input video missing or invalid")
            meta.update(status=STATUS_RUNNING, started_at=_current_iso_time())
            self._save_metadata(job_id, meta)
            outcome = self.runner(
                job_id=job_id, input_path=inputs[0], results_dir=self._get_results_dir(job_id),
                log_path=job_dir / "run.log", device=self.device, model_path=self.model_path,
                gym_pose_dir=self.gym_pose_dir, timeout=self.timeout_seconds,
            )
            if not isinstance(outcome, dict) or outcome.get("status") not in TERMINAL_STATUSES:
                raise ValueError("Worker did not return a terminal outcome")
            meta.update(outcome)
            if meta.get("status") in (STATUS_COMPLETED, STATUS_INCOMPLETE) and (self._get_results_dir(job_id) / "annotated.mp4").is_file():
                from .web_media import create_preview
                preview = create_preview(self._get_results_dir(job_id) / "annotated.mp4", self._get_results_dir(job_id) / "preview.mp4")
                meta["preview_warning"] = preview["warning"]
        except Exception as exc:
            if meta is not None:
                meta.update(status=STATUS_FAILED, error=f"Execution error: {type(exc).__name__}: {exc}")
        finally:
            try:
                if meta is not None:
                    meta["finished_at"] = _current_iso_time()
                    self._save_metadata(job_id, meta)
            finally:
                with self._lock:
                    if self._active_job_id == job_id:
                        self._active_job_id = None

    def _default_cli_runner(
        self,
        job_id: str,
        input_path: Path,
        results_dir: Path,
        log_path: Path,
        device: str,
        model_path: Path,
        gym_pose_dir: Path,
        timeout: float,
    ) -> Dict[str, Any]:
        """Execute gym_analyser CLI in a child process with explicit arguments."""
        if not model_path.is_file():
            raise FileNotFoundError("Configured pretrained checkpoint is missing; automatic download disabled")
        cmd = [
            str(self.analysis_python),
            "-m",
            "gym_analyser",
            "--source",
            str(input_path),
            "--output-dir",
            str(results_dir),
            "--model",
            str(model_path),
            "--device",
            str(device),
        ]

        exit_code = -1
        with open(log_path, "w", encoding="utf-8") as log_file:
            proc = subprocess.Popen(
                cmd,
                stdin=subprocess.DEVNULL,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                stdout=log_file,
                stderr=subprocess.STDOUT,
                cwd=str(gym_pose_dir),
                shell=False,
            )
            try:
                exit_code = proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
                raise TimeoutError(f"Subprocess exceeded timeout of {timeout}s")

        # Inspect generated summary.json
        summary_path = results_dir / "summary.json"
        outcome: Dict[str, Any] = {"exit_code": exit_code}

        if summary_path.is_file():
            try:
                with open(summary_path, "r", encoding="utf-8") as sf:
                    summary = json.load(sf)
                reported_status = summary.get("status")
                if reported_status == "completed" and exit_code == 0:
                    outcome["status"] = STATUS_COMPLETED
                elif reported_status == "incomplete":
                    outcome["status"] = STATUS_INCOMPLETE
                    outcome["read_status_note"] = summary.get("read_status_note")
                else:
                    outcome["status"] = STATUS_FAILED
                    outcome["error"] = summary.get("error", f"Processing ended with status: {reported_status}")
            except Exception as e:
                outcome["status"] = STATUS_FAILED
                outcome["error"] = f"Failed to parse summary output: {e}"
        else:
            outcome["status"] = STATUS_FAILED
            # Read tail of log file for diagnostic error message
            log_tail = ""
            if log_path.is_file():
                try:
                    with open(log_path, "r", encoding="utf-8", errors="replace") as lf:
                        lines = lf.readlines()
                        log_tail = "".join(lines[-10:]).strip()
                except Exception:
                    pass
            outcome["error"] = (
                f"Subprocess exited with code {exit_code} without generating summary.json. {log_tail}".strip()
            )

        return outcome

    def _get_results_dir(self, job_id: str) -> Path:
        job_dir = self._get_job_dir(job_id)
        result = (job_dir / "results").resolve()
        if result != job_dir / "results":
            raise ValueError("Results directory must not redirect outside its job")
        return result

    def get_artifact_path(self, job_id: str, artifact_name: str) -> Path:
        """Retrieve path to a verified output artifact with strict allowlist and traversal protection."""
        if not is_valid_job_id(job_id):
            raise ValueError(f"Invalid job ID: '{job_id}'")

        if (
            artifact_name not in ALLOWED_ARTIFACTS
            or "/" in artifact_name
            or "\\" in artifact_name
            or ".." in artifact_name
        ):
            raise FileNotFoundError(f"Requested artifact '{artifact_name}' is not in the allowed export list.")

        job_dir = self._get_job_dir(job_id)
        results_dir = self._get_results_dir(job_id)
        artifact_path = (results_dir / artifact_name).resolve()

        # Strict path traversal checks
        if artifact_path.parent != results_dir:
            raise FileNotFoundError(f"Security error: path traversal detected for artifact '{artifact_name}'")
        if not artifact_path.is_file():
            raise FileNotFoundError(f"Artifact '{artifact_name}' does not exist for job '{job_id}'")

        return artifact_path

    def get_job_results(self, job_id: str) -> Dict[str, Any]:
        """Aggregate and serialize reviewed output schemas for the web client."""
        meta = self.get_metadata(job_id)
        if not meta:
            raise KeyError(f"Job '{job_id}' not found.")

        job_dir = self._get_job_dir(job_id)
        results_dir = self._get_results_dir(job_id)

        # Check available artifacts on disk
        available_artifacts: Dict[str, Dict[str, Any]] = {}
        for name, media_type in ALLOWED_ARTIFACTS.items():
            try:
                path = self.get_artifact_path(job_id, name)
                exists = True
            except FileNotFoundError:
                path = results_dir / name
                exists = False
            available_artifacts[name] = {
                "name": name,
                "available": exists,
                "size_bytes": path.stat().st_size if exists else 0,
                "media_type": media_type,
                "download_url": f"/api/jobs/{job_id}/download/{name}",
            }

        # Load parsed JSON files safely
        summary_data: Optional[Dict[str, Any]] = None
        summary_path = results_dir / "summary.json"
        if summary_path.is_file() and summary_path.resolve().parent == results_dir:
            try:
                with open(summary_path, "r", encoding="utf-8") as f:
                    summary_data = json.load(f)
            except Exception:
                pass

        attempts_data: Optional[Dict[str, Any]] = None
        attempts_path = results_dir / "attempts.json"
        if attempts_path.is_file() and attempts_path.resolve().parent == results_dir:
            try:
                with open(attempts_path, "r", encoding="utf-8") as f:
                    attempts_data = json.load(f)
            except Exception:
                pass

        posture_data: Optional[Dict[str, Any]] = None
        posture_path = results_dir / "posture_summary.json"
        if posture_path.is_file() and posture_path.resolve().parent == results_dir:
            try:
                with open(posture_path, "r", encoding="utf-8") as f:
                    posture_data = json.load(f)
            except Exception:
                pass

        feedback_data = None
        feedback_path = results_dir / "feedback_observations.json"
        if feedback_path.is_file() and feedback_path.resolve().parent == results_dir:
            try:
                feedback_data = json.loads(feedback_path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                pass

        phase_feedback_data = None
        phase_feedback_path = results_dir / "feedback_phases.json"
        if phase_feedback_path.is_file() and phase_feedback_path.resolve().parent == results_dir:
            try:
                phase_feedback_data = json.loads(phase_feedback_path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                pass

        coaching_feedback_data = None
        coaching_feedback_path = results_dir / "coaching_feedback.json"
        if coaching_feedback_path.is_file() and coaching_feedback_path.resolve().parent == results_dir:
            try:
                coaching_feedback_data = json.loads(coaching_feedback_path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                pass

        completed_reps = 0
        processed_frames = 0
        read_status_note = meta.get("read_status_note")

        if summary_data:
            completed_reps = summary_data.get("completed_reps", 0)
            processed_frames = summary_data.get("processed_frames", 0)
            if not read_status_note:
                read_status_note = summary_data.get("read_status_note")

        return {
            "job_id": job_id,
            "status": meta.get("status", STATUS_QUEUED),
            "created_at": meta.get("created_at"),
            "started_at": meta.get("started_at"),
            "finished_at": meta.get("finished_at"),
            "original_filename": meta.get("original_filename"),
            "file_size_bytes": meta.get("file_size_bytes"),
            "error": meta.get("error"),
            "preview_warning": meta.get("preview_warning"),
            "read_status_note": read_status_note,
            "completed_reps": completed_reps,
            "processed_frames": processed_frames,
            "summary": summary_data,
            "attempts": attempts_data,
            "posture_summary": posture_data,
            "feedback_observations": feedback_data,
            "phase_feedback": phase_feedback_data,
            "coaching_feedback": coaching_feedback_data,
            "available_artifacts": available_artifacts,
        }
