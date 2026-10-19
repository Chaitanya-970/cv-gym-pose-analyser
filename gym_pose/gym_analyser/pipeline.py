"""Pipeline orchestrating video ingestion, YOLO pose tracking, and output export."""

import csv
import json
import math
import os
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from .counter import CounterConfig, FrameResult, SquatCounter
from .attempts import AttemptTracker
from .feedback_observations import build_feedback_observations
from .phase_feedback import build_phase_feedback
from .session_feedback import summarise_attempt_posture, load_csv_frame_rows
from .geometry import COCO_PAIRS
from .posture import evaluate_posture, evaluate_posture_auto_side


def export_csv(results: List[FrameResult], csv_path: Path) -> None:
    """Export frame results to CSV. Missing numeric values are written as blanks."""
    fieldnames = [
        "frame_index",
        "video_time_s",
        "track_id",
        "selected_side",
        "knee_angle_deg",
        "confidence",
        "valid",
        "reason",
        "stage",
        "completed_reps",
        "posture_side", "torso_inclination_deg", "hip_angle_deg", "torso_reason", "hip_reason",
        "attempt_id", "last_attempt_reason",
    ]
    with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow(
                {
                    "frame_index": r.frame_index,
                    "video_time_s": f"{r.video_time_s:.4f}",
                    "track_id": "" if r.track_id is None else r.track_id,
                    "selected_side": "" if r.selected_side is None else r.selected_side,
                    "knee_angle_deg": (
                        ""
                        if r.knee_angle_deg is None or not math.isfinite(r.knee_angle_deg)
                        else f"{r.knee_angle_deg:.2f}"
                    ),
                    "confidence": (
                        ""
                        if r.confidence is None or not math.isfinite(r.confidence)
                        else f"{r.confidence:.4f}"
                    ),
                    "valid": "True" if r.valid else "False",
                    "reason": r.reason,
                    "stage": r.stage,
                    "completed_reps": r.completed_reps,
                    "posture_side": r.posture_side or "",
                    "torso_inclination_deg": "" if r.torso_inclination_deg is None else f"{r.torso_inclination_deg:.2f}",
                    "hip_angle_deg": "" if r.hip_angle_deg is None else f"{r.hip_angle_deg:.2f}",
                    "torso_reason": r.torso_reason,
                    "hip_reason": r.hip_reason,
                    "attempt_id": r.attempt_id if r.attempt_id is not None else "",
                    "last_attempt_reason": r.last_attempt_reason,
                }
            )


def export_summary(
    summary_data: Dict[str, Any], summary_path: Path
) -> None:
    """Save execution metadata and counts to summary.json."""
    with open(summary_path, mode="w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)


def export_angle_plot(
    results: List[FrameResult],
    config: CounterConfig,
    plot_path: Path,
) -> None:
    """Generate a knee angle time-series plot preserving missing data gaps."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    times: List[float] = []
    angles: List[float] = []

    for r in results:
        times.append(r.video_time_s)
        if r.valid and r.knee_angle_deg is not None and math.isfinite(r.knee_angle_deg):
            angles.append(r.knee_angle_deg)
        else:
            angles.append(math.nan)

    fig, ax = plt.subplots(figsize=(10, 4.5), dpi=150)
    ax.plot(times, angles, label="Knee Angle", color="#1f77b4", linewidth=1.5)
    ax.axhline(
        config.up_angle,
        color="#2ca02c",
        linestyle="--",
        linewidth=1.2,
        label=f"Up Threshold ({config.up_angle}Â°)",
    )
    ax.axhline(
        config.down_angle,
        color="#d62728",
        linestyle="--",
        linewidth=1.2,
        label=f"Down Threshold ({config.down_angle}Â°)",
    )

    ax.set_xlabel("Video Time (seconds)")
    ax.set_ylabel("Projected Knee Angle (degrees)")
    ax.set_title("Squat Repetition Knee Angle Time Series")
    ax.set_ylim(0, 190)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="upper right")

    fig.tight_layout()
    fig.savefig(plot_path)
    plt.close(fig)


def draw_knee_overlay(
    frame: Any,
    keypoints: Sequence[Sequence[float]],
    selected_side: Optional[str],
    knee_angle_deg: Optional[float],
) -> None:
    """Draw knee angle arc, highlighted limb segments, and angle label on the frame."""
    if selected_side not in COCO_PAIRS or knee_angle_deg is None:
        return
    if not math.isfinite(knee_angle_deg):
        return

    import cv2

    hip_idx, knee_idx, ankle_idx = COCO_PAIRS[selected_side]
    if len(keypoints) <= max(hip_idx, knee_idx, ankle_idx):
        return

    hip = keypoints[hip_idx]
    knee = keypoints[knee_idx]
    ankle = keypoints[ankle_idx]

    kx, ky = int(round(knee[0])), int(round(knee[1]))
    hx, hy = int(round(hip[0])), int(round(hip[1]))
    ax, ay = int(round(ankle[0])), int(round(ankle[1]))

    # Highlight limb segments on the selected side
    limb_color = (0, 255, 255)  # Yellow
    cv2.line(frame, (hx, hy), (kx, ky), limb_color, 3, cv2.LINE_AA)
    cv2.line(frame, (kx, ky), (ax, ay), limb_color, 3, cv2.LINE_AA)
    cv2.circle(frame, (kx, ky), 5, (0, 165, 255), -1, cv2.LINE_AA)

    # Calculate angles from knee vertex
    v1x, v1y = hx - kx, hy - ky
    v2x, v2y = ax - kx, ay - ky

    len1 = math.hypot(v1x, v1y)
    len2 = math.hypot(v2x, v2y)
    if len1 < 5.0 or len2 < 5.0:
        return

    ang1 = math.degrees(math.atan2(v1y, v1x)) % 360
    ang2 = math.degrees(math.atan2(v2y, v2x)) % 360

    diff = (ang2 - ang1) % 360
    if diff <= 180:
        start_angle, end_angle = ang1, ang1 + diff
    else:
        start_angle, end_angle = ang2, ang2 + (360 - diff)

    arc_radius = min(40, max(15, int(min(len1, len2) * 0.4)))
    cv2.ellipse(
        frame,
        (kx, ky),
        (arc_radius, arc_radius),
        0,
        start_angle,
        end_angle,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )

    # Draw angle text label near the knee
    label_text = f"{knee_angle_deg:.1f}Â°"
    cv2.putText(
        frame,
        label_text,
        (kx + 10, ky - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )


def draw_hud(
    frame: Any,
    result: FrameResult,
    config: CounterConfig,
) -> None:
    """Draw exercise HUD banner on output video frame using OpenCV."""
    import cv2

    h, w = frame.shape[:2]
    # Draw top semi-transparent banner
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 110), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

    angle_str = (
        f"{result.knee_angle_deg:.1f}Â°"
        if result.knee_angle_deg is not None
        else f"N/A ({result.reason})"
    )
    side_str = result.selected_side.upper() if result.selected_side else "UNLOCKED"
    track_str = str(result.track_id) if result.track_id is not None else "NONE"
    stage_str = result.stage.upper()

    col1 = f"TRACK: {track_str} | SIDE: {side_str}"
    col2 = f"ANGLE: {angle_str} | STAGE: {stage_str}"
    col3 = f"REPS: {result.completed_reps}"

    cv2.putText(
        frame, col1, (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 2, cv2.LINE_AA
    )
    cv2.putText(
        frame, col2, (15, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 2, cv2.LINE_AA
    )
    cv2.putText(
        frame, col3, (w - 180, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (50, 230, 50), 2, cv2.LINE_AA
    )

    torso = "N/A" if result.torso_inclination_deg is None else f"{result.torso_inclination_deg:.1f}"
    hip = "N/A" if result.hip_angle_deg is None else f"{result.hip_angle_deg:.1f}"
    cv2.putText(frame, f"TORSO FROM VERTICAL: {torso} | HIP: {hip}",
                (15,77),cv2.FONT_HERSHEY_SIMPLEX,0.48,(220,220,220),1,cv2.LINE_AA)

    if result.last_attempt_reason:
        cv2.putText(frame,"LAST ATTEMPT: "+result.last_attempt_reason.replace("_"," "),
                    (15,99),cv2.FONT_HERSHEY_SIMPLEX,0.45,(220,220,220),1,cv2.LINE_AA)


def attach_posture(result: FrameResult, detections: List[Dict[str, Any]], config: CounterConfig) -> FrameResult:
    """Measure only an unambiguous compatible subject; knee and torso availability are independent."""
    if len(detections) != 1 or result.reason.startswith(("track_id_mismatch", "missing_track_id")):
        return replace(result,torso_reason=result.reason,hip_reason=result.reason)
    points = detections[0].get("keypoints")
    measurement = (evaluate_posture(points,result.selected_side,config.min_joint_confidence)
                   if result.selected_side else evaluate_posture_auto_side(points,config.min_joint_confidence))
    return replace(result,posture_side=measurement.side if measurement.side in {"left","right"} else None,
                   torso_inclination_deg=measurement.torso_inclination_deg,
                   hip_angle_deg=measurement.hip_angle_deg,
                   torso_reason=measurement.torso_reason,hip_reason=measurement.hip_reason)


def run_pipeline(
    source_path: Path,
    output_dir: Path,
    config: CounterConfig,
    model_name: Any = "yolo11n-pose.pt",
    device: str = "cpu",
    tracker: str = "bytetrack.yaml",
    overwrite: bool = False,
    departure_frames: int = 2,
) -> Dict[str, Any]:
    """Execute complete video analysis pipeline with explicit lifecycle and error reporting."""
    # 1. Input and output path validation
    source_path = source_path.resolve()
    output_dir = output_dir.resolve()

    if not source_path.exists() or not source_path.is_file():
        raise FileNotFoundError(f"Input source video not found: {source_path}")

    if source_path == output_dir or output_dir in source_path.parents:
        raise ValueError("Source video path cannot be inside or equal to output directory")

    if output_dir.exists() and not overwrite:
        existing_files = list(output_dir.glob("*"))
        if existing_files:
            raise FileExistsError(
                f"Output directory {output_dir} is not empty. Use --overwrite to permit overwriting."
            )

    output_dir.mkdir(parents=True, exist_ok=True)

    # Setup artifact paths
    out_video_path = output_dir / "annotated.mp4"
    out_csv_path = output_dir / "angles.csv"
    out_summary_path = output_dir / "summary.json"
    out_plot_path = output_dir / "angle_plot.png"
    out_attempts_path = output_dir / "attempts.json"
    out_posture_summary_path = output_dir / "posture_summary.json"
    out_feedback_path = output_dir / "feedback_observations.json"
    out_feedback_phases_path = output_dir / "feedback_phases.json"

    # Purge stale files from earlier runs when overwrite is permitted
    if overwrite:
        for stale_path in (out_video_path, out_csv_path, out_summary_path, out_plot_path, out_attempts_path, out_posture_summary_path, out_feedback_path, out_feedback_phases_path):
            if stale_path.exists():
                try:
                    stale_path.unlink()
                except OSError:
                    pass

    # Lifecycle resources
    cap = None
    writer = None
    results: List[FrameResult] = []
    frame_idx = 0
    valid_count = 0
    counter = SquatCounter(config)
    attempts = AttemptTracker(config, departure_frames)

    fps = 0.0
    width = 0
    height = 0
    declared_frame_count = 0
    is_incomplete = False

    try:
        import cv2

        # 2. Open input video capture
        cap = cv2.VideoCapture(str(source_path))
        if not cap.isOpened():
            raise IOError(f"Failed to open video source: {source_path}")

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        declared_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if fps <= 0.0 or not math.isfinite(fps):
            raise ValueError(f"Invalid video FPS: {fps}")

        if width <= 0 or height <= 0:
            raise ValueError(f"Invalid video dimensions: {width}x{height}")

        # 3. Open video writer
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(out_video_path), fourcc, fps, (width, height))
        if not writer.isOpened():
            raise IOError(f"Failed to open VideoWriter for path: {out_video_path}")

        # 4. Load and validate pose model
        if isinstance(model_name, (str, Path)):
            from ultralytics import YOLO

            try:
                model = YOLO(str(model_name))
            except Exception as e:
                raise RuntimeError(f"Failed to load YOLO pose model '{model_name}': {e}") from e
            actual_model_name = str(model_name)
        else:
            model = model_name
            actual_model_name = type(model_name).__name__

        # Enforce model task validation
        model_task = getattr(model, "task", None)
        if model_task != "pose":
            raise ValueError(
                f"Unsupported model task '{model_task}'. A pose estimation model with task='pose' is required."
            )

        # 5. Process video frames
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                break

            video_time_s = frame_idx / fps

            # Model tracking inference
            track_results = model.track(
                source=frame,
                persist=True,
                tracker=tracker,
                device=device,
                verbose=False,
            )

            detections: List[Dict[str, Any]] = []
            if track_results and len(track_results) > 0:
                res0 = track_results[0]
                if res0.boxes is not None and res0.keypoints is not None:
                    kpt_data = res0.keypoints.data.cpu().numpy()
                    box_ids = res0.boxes.id

                    # If instances are detected, validate tensor shape and channels
                    if len(kpt_data) > 0:
                        if kpt_data.ndim != 3 or kpt_data.shape[1] != 17:
                            raise ValueError(
                                f"Unsupported keypoint layout: expected 17 keypoints (COCO-17), but received shape {kpt_data.shape}"
                            )
                        if kpt_data.shape[2] < 3:
                            raise ValueError(
                                f"Keypoint tensor has {kpt_data.shape[2]} channels; expected at least 3 channels (x, y, confidence)"
                            )
                        if box_ids is not None and len(box_ids) != len(kpt_data):
                            raise ValueError(
                                f"Mismatch between detected keypoint count ({len(kpt_data)}) and bounding box track IDs ({len(box_ids)})"
                            )

                    ids = (
                        box_ids.int().cpu().tolist()
                        if box_ids is not None
                        else [None] * len(kpt_data)
                    )
                    for kpts, tid in zip(kpt_data, ids):
                        detections.append({"track_id": tid, "keypoints": kpts})

                elif res0.boxes is not None and len(res0.boxes) > 0 and res0.keypoints is None:
                    raise ValueError(
                        "Model detected bounding boxes without keypoints. A pose estimation model is required."
                    )

            # Update repetition counter
            result = counter.update(
                frame_index=frame_idx,
                video_time_s=video_time_s,
                detections=detections,
            )
            attempts.observe(result,counter.armed_side)
            latest=attempts.events[-1] if attempts.events else None
            event_id=attempts.active_id
            if latest and latest['end_frame']==result.frame_index:event_id=latest['attempt_id']
            result=replace(result,attempt_id=event_id,last_attempt_reason=latest['reason'] if latest else '')
            result = attach_posture(result, detections, config)
            results.append(result)
            if result.valid:
                valid_count += 1

            # Render frame annotations
            if track_results and len(track_results) > 0:
                plot_frame = track_results[0].plot(boxes=False)
            else:
                plot_frame = frame.copy()

            # Draw knee angle arc and limb highlight if valid
            if result.valid and result.selected_side and detections:
                active_det = next(
                    (d for d in detections if d.get("track_id") == result.track_id),
                    detections[0],
                )
                draw_knee_overlay(
                    plot_frame,
                    active_det["keypoints"],
                    result.selected_side,
                    result.knee_angle_deg,
                )

            draw_hud(plot_frame, result, config)
            writer.write(plot_frame)
            frame_idx += 1

        # 6. Validate read results
        if frame_idx == 0:
            raise RuntimeError(f"No readable video frames could be decoded from {source_path}")

        declared_frames = declared_frame_count if declared_frame_count > 0 else None
        declared_duration_s = round(declared_frames / fps, 2) if declared_frames is not None else None
        processed_duration_s = round(frame_idx / fps, 2)

        if declared_frames is not None and frame_idx < declared_frames:
            is_incomplete = True
            status = "incomplete"
            read_note = (
                f"Container declared {declared_frames} frames, but only {frame_idx} frames were successfully decoded. "
                "OpenCV VideoCapture cannot determine whether early termination was caused by a codec/decode failure or inaccurate container metadata."
            )
        else:
            is_incomplete = False
            status = "completed"
            read_note = (
                f"Successfully processed all {frame_idx} decoded frames. "
                + (
                    f"Container metadata declared {declared_frames} frames."
                    if declared_frames is not None
                    else "Container provided no declared frame count metadata."
                )
            )

        attempt_report=attempts.finish('decode_incomplete' if is_incomplete else 'video_ended')
        export_summary(attempt_report,out_attempts_path)
        # Export CSV and Plot
        export_csv(results, out_csv_path)
        posture_summary = summarise_attempt_posture(load_csv_frame_rows(out_csv_path), attempt_report)
        export_summary(posture_summary, out_posture_summary_path)
        feedback = build_feedback_observations(posture_summary, attempt_report)
        export_summary(feedback, out_feedback_path)
        phase_feedback = build_phase_feedback(load_csv_frame_rows(out_csv_path), attempt_report)
        export_summary(phase_feedback, out_feedback_phases_path)
        export_angle_plot(results, config, out_plot_path)

        valid_fraction = valid_count / frame_idx if frame_idx > 0 else 0.0

        import ultralytics
        summary = {
            "status": status,
            "model_name": actual_model_name,
            "read_status_note": read_note,
            "package_versions": {
                "python": sys.version.split()[0],
                "ultralytics": ultralytics.__version__ if "ultralytics" in sys.modules else "unknown",
                "opencv": cv2.__version__,
            },
            "configuration": {
                "down_angle": config.down_angle,
                "up_angle": config.up_angle,
                "min_joint_confidence": config.min_joint_confidence,
                "stable_frames": config.stable_frames,
                "max_gap_seconds": config.max_gap_seconds,
                "tracker": tracker,
                "device": device,
            },
            "source_video": {
                "path": str(source_path),
                "declared_frame_count": declared_frames,
                "declared_duration_s": declared_duration_s,
                "fps": fps,
                "width": width,
                "height": height,
            },
            "processed_frames": frame_idx,
            "processed_duration_s": processed_duration_s,
            "valid_observations": valid_count,
            "valid_observation_fraction": round(valid_fraction, 4),
            "completed_reps": counter.completed_reps,
            "per_attempt_posture": posture_summary["summary"],
            "feedback_observations": feedback["summary"],
            "phase_feedback": phase_feedback["summary"],
            "attempt_observations": {
                "counted_attempts": attempt_report['counted_attempts'],
                "uncounted_attempts": attempt_report['uncounted_attempts'],
                "pre_arming": attempt_report['pre_arming'],
                "departure_frames": departure_frames,
                "interpretation": "Software-defined observed attempts, not posture judgements",
            },
            "posture_observations": {
                "torso_valid_frames": sum(r.torso_inclination_deg is not None for r in results),
                "hip_valid_frames": sum(r.hip_angle_deg is not None for r in results),
                "processed_frames": frame_idx,
                "interpretation": "Projected 2D measurements; unsigned torso angle from upward image vertical, not a form grade",
            },
            "output_paths": {
                "annotated_video": str(out_video_path),
                "angles_csv": str(out_csv_path),
                "summary_json": str(out_summary_path),
                "angle_plot": str(out_plot_path),
                "attempts_json": str(out_attempts_path),
                "posture_summary_json": str(out_posture_summary_path),
                "feedback_observations_json": str(out_feedback_path),
                "phase_feedback_json": str(out_feedback_phases_path),
            },
            "limitations": [
                "Projected 2D joint angles, not calibrated 3D anatomical angles.",
                "Repetition counts reflect threshold transitions, not validated clinical accuracy.",
                "Assumes a stationary camera with a clear side profile of a single person.",
                "Torso inclination is unsigned and depends on camera roll; posture observations do not establish safe or correct form.",
            ],
        }

        export_summary(summary, out_summary_path)

        if is_incomplete:
            raise RuntimeError(f"Processing ended early: {read_note}")

        return summary

    except Exception as exc:
        try:
            export_summary(attempts.finish('processing_failed'),out_attempts_path)
        except Exception:
            pass
        # Preserve partial CSV if any frames were processed
        if results:
            try:
                export_csv(results, out_csv_path)
                partial_posture = summarise_attempt_posture(load_csv_frame_rows(out_csv_path), attempts.report())
                export_summary(partial_posture, out_posture_summary_path)
                export_summary(build_feedback_observations(partial_posture, attempts.report()), out_feedback_path)
            except Exception:
                pass
            try:
                export_summary(build_phase_feedback(load_csv_frame_rows(out_csv_path), attempts.report()), out_feedback_phases_path)
            except Exception:
                pass

        # Write failed summary if not already written as incomplete
        if not is_incomplete or not out_summary_path.exists():
            fail_summary = {
                "status": "failed",
                "error": f"{type(exc).__name__}: {str(exc)}",
                "model_name": actual_model_name if "actual_model_name" in locals() else str(model_name),
                "processed_frames": frame_idx,
                "processed_duration_s": round(frame_idx / fps, 2) if fps > 0 else 0.0,
                "completed_reps": counter.completed_reps,
                "source_video": {
                    "path": str(source_path),
                    "declared_frame_count": declared_frame_count if declared_frame_count > 0 else None,
                    "fps": fps if fps > 0 else None,
                    "width": width if width > 0 else None,
                    "height": height if height > 0 else None,
                },
                "output_paths": {
                    "annotated_video": str(out_video_path) if out_video_path.exists() else None,
                    "angles_csv": str(out_csv_path) if out_csv_path.exists() else None,
                    "summary_json": str(out_summary_path),
                    "angle_plot": str(out_plot_path) if out_plot_path.exists() else None,
                    "attempts_json": str(out_attempts_path) if out_attempts_path.exists() else None,
                    "posture_summary_json": str(out_posture_summary_path) if out_posture_summary_path.exists() else None,
                    "feedback_observations_json": str(out_feedback_path) if out_feedback_path.exists() else None,
                    "phase_feedback_json": str(out_feedback_phases_path) if out_feedback_phases_path.exists() else None,
                },
                "limitations": [
                    "Projected 2D joint angles, not calibrated 3D anatomical angles.",
                    "Repetition counts reflect threshold transitions, not validated clinical accuracy.",
                    "Assumes a stationary camera with a clear side profile of a single person.",
                "Torso inclination is unsigned and depends on camera roll; posture observations do not establish safe or correct form.",
                ],
            }
            try:
                export_summary(fail_summary, out_summary_path)
            except Exception:
                pass

        raise

    finally:
        if cap is not None:
            cap.release()
        if writer is not None:
            writer.release()
