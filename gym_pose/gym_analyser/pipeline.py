"""Pipeline orchestrating video ingestion, YOLO pose tracking, and output export."""

import csv
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from .counter import CounterConfig, FrameResult, SquatCounter


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
    # Delayed import of matplotlib
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
        label=f"Up Threshold ({config.up_angle}°)",
    )
    ax.axhline(
        config.down_angle,
        color="#d62728",
        linestyle="--",
        linewidth=1.2,
        label=f"Down Threshold ({config.down_angle}°)",
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
    cv2.rectangle(overlay, (0, 0), (w, 65), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

    angle_str = (
        f"{result.knee_angle_deg:.1f}°"
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


def run_pipeline(
    source_path: Path,
    output_dir: Path,
    config: CounterConfig,
    model_name: str = "yolo11n-pose.pt",
    device: str = "cpu",
    tracker: str = "bytetrack.yaml",
    overwrite: bool = False,
) -> Dict[str, Any]:
    """Execute complete video analysis pipeline."""
    # 1. Input and output path validation (pure standard library)
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

    # 2. Open input video capture
    import cv2

    cap = cv2.VideoCapture(str(source_path))
    if not cap.isOpened():
        raise IOError(f"Failed to open video source: {source_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if fps <= 0.0 or not math.isfinite(fps):
        cap.release()
        raise ValueError(f"Invalid video FPS: {fps}")

    if width <= 0 or height <= 0:
        cap.release()
        raise ValueError(f"Invalid video dimensions: {width}x{height}")

    duration_s = total_frames / fps if total_frames > 0 else 0.0

    # 3. Setup output paths
    out_video_path = output_dir / "annotated.mp4"
    out_csv_path = output_dir / "angles.csv"
    out_summary_path = output_dir / "summary.json"
    out_plot_path = output_dir / "angle_plot.png"

    # 4. Open video writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_video_path), fourcc, fps, (width, height))
    if not writer.isOpened():
        cap.release()
        raise IOError(f"Failed to open VideoWriter for path: {out_video_path}")

    # 5. Load model
    from ultralytics import YOLO

    try:
        model = YOLO(model_name)
    except Exception as e:
        cap.release()
        writer.release()
        raise RuntimeError(f"Failed to load YOLO pose model '{model_name}': {e}") from e

    counter = SquatCounter(config)
    results: List[FrameResult] = []
    frame_idx = 0
    valid_count = 0

    try:
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
                if res0.keypoints is not None and res0.boxes is not None:
                    kpt_data = res0.keypoints.data.cpu().numpy()
                    box_ids = res0.boxes.id
                    ids = (
                        box_ids.int().cpu().tolist()
                        if box_ids is not None
                        else [None] * len(kpt_data)
                    )
                    for kpts, tid in zip(kpt_data, ids):
                        detections.append({"track_id": tid, "keypoints": kpts})

            # Update counter
            result = counter.update(
                frame_index=frame_idx,
                video_time_s=video_time_s,
                detections=detections,
            )
            results.append(result)
            if result.valid:
                valid_count += 1

            # Render annotations and HUD
            if track_results and len(track_results) > 0:
                plot_frame = track_results[0].plot(boxes=False)
            else:
                plot_frame = frame.copy()

            draw_hud(plot_frame, result, config)
            writer.write(plot_frame)
            frame_idx += 1

    finally:
        cap.release()
        writer.release()

    # 6. Save data artifacts
    export_csv(results, out_csv_path)
    export_angle_plot(results, config, out_plot_path)

    valid_fraction = valid_count / frame_idx if frame_idx > 0 else 0.0

    import ultralytics
    summary = {
        "status": "completed",
        "model_name": model_name,
        "package_versions": {
            "python": sys.version.split()[0],
            "ultralytics": ultralytics.__version__,
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
            "frame_count": frame_idx,
            "fps": fps,
            "duration_s": round(duration_s, 2),
            "width": width,
            "height": height,
        },
        "processed_frames": frame_idx,
        "valid_observations": valid_count,
        "valid_observation_fraction": round(valid_fraction, 4),
        "completed_reps": counter.completed_reps,
        "output_paths": {
            "annotated_video": str(out_video_path),
            "angles_csv": str(out_csv_path),
            "summary_json": str(out_summary_path),
            "angle_plot": str(out_plot_path),
        },
        "limitations": [
            "Projected 2D joint angles, not calibrated 3D anatomical angles.",
            "Repetition counts reflect threshold transitions, not validated clinical accuracy.",
            "Assumes a stationary camera with a clear side profile of a single person.",
        ],
    }

    export_summary(summary, out_summary_path)
    return summary
