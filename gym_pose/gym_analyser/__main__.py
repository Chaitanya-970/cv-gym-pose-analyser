"""Command-line interface for 2D Gym Pose Analyser."""

import argparse
import sys
from pathlib import Path

from .counter import CounterConfig


def build_parser() -> argparse.ArgumentParser:
    """Build command-line argument parser."""
    parser = argparse.ArgumentParser(
        prog="python -m gym_analyser",
        description="2D Gym Pose Analyser: evaluate squat repetitions from video footage.",
    )
    parser.add_argument(
        "--source",
        type=Path,
        required=True,
        help="Path to input video file (e.g. mp4).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to store annotated video, CSV, plot, and summary JSON.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolo11n-pose.pt",
        help="YOLO pose model path or name (default: yolo11n-pose.pt).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Inference compute device (e.g. 'cpu' or 'cuda').",
    )
    parser.add_argument(
        "--down-angle",
        type=float,
        default=CounterConfig().down_angle,
        help="Legacy internal knee-angle gate (straight=180; default: 110). Not a validated full-squat definition.",
    )
    parser.add_argument(
        "--up-angle",
        type=float,
        default=CounterConfig().up_angle,
        help="Internal knee-angle standing gate (default: 150). Software hysteresis, not a clinical extension standard.",
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.3,
        help="Minimum confidence required per joint (default: 0.3).",
    )
    parser.add_argument(
        "--stable-frames",
        type=int,
        default=3,
        help="Consecutive frames required for stage transitions (default: 3).",
    )
    parser.add_argument("--departure-frames",type=int,default=2,help="Valid consecutive departure frames needed to observe an attempt (default:2)")
    parser.add_argument(
        "--max-gap",
        type=float,
        default=0.5,
        help="Maximum gap in seconds before abandoning partial repetition (default: 0.5).",
    )
    parser.add_argument(
        "--tracker",
        type=str,
        default="bytetrack.yaml",
        help="Ultralytics tracking configuration (default: bytetrack.yaml).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting existing files in the output directory.",
    )
    return parser


def main() -> None:
    """Entry point for CLI execution."""
    parser = build_parser()
    args = parser.parse_args()

    try:
        if args.departure_frames < 1:
            raise ValueError("departure-frames must be positive")
        config = CounterConfig(
            down_angle=args.down_angle,
            up_angle=args.up_angle,
            min_joint_confidence=args.min_confidence,
            stable_frames=args.stable_frames,
            max_gap_seconds=args.max_gap,
        )
    except (ValueError, TypeError) as e:
        sys.stderr.write(f"Configuration error: {e}\n")
        sys.exit(1)

    # Delay pipeline import until arguments are verified
    from .pipeline import run_pipeline

    try:
        summary = run_pipeline(
            source_path=args.source,
            output_dir=args.output_dir,
            config=config,
            model_name=args.model,
            device=args.device,
            tracker=args.tracker,
            overwrite=args.overwrite,
            departure_frames=args.departure_frames,
        )
        if summary.get("status") != "completed":
            sys.stderr.write(f"Processing ended with non-completed status '{summary.get('status')}': {summary.get('read_status_note', '')}\n")
            sys.exit(1)

        print("Processing completed successfully.")
        print(f"Processed frames: {summary['processed_frames']}")
        print(f"Completed repetitions: {summary['completed_reps']}")
        print(f"Uncounted observed attempts: {summary.get('attempt_observations',{}).get('uncounted_attempts',0)}")
        print(f"Outputs written to: {args.output_dir}")
    except Exception as e:
        sys.stderr.write(f"Execution error: {e}\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
