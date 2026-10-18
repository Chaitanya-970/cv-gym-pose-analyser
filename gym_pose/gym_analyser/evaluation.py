"""Repetition count evaluation tools for gym pose analysis.

Pure Python standard-library module and CLI for matching model prediction
summaries to human-reviewed ground truth clip counts, computing exact-match
ratios, mean absolute error (MAE), and tracking unreviewed clips.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union


def validate_count(val: Any, field_name: str = "count", clip_id: str = "") -> int:
    """Validate that a repetition count is a non-negative integer.

    Rejects booleans (which are subclasses of int in Python), floats,
    negative numbers, strings, and other non-integer types.

    Args:
        val: Value to validate.
        field_name: Name of the field for descriptive error messages.
        clip_id: Identifier of the clip for error context.

    Returns:
        The validated integer count.

    Raises:
        ValueError: If val is a boolean, non-integer, or negative.
    """
    prefix = f"Clip '{clip_id}': " if clip_id else ""
    if isinstance(val, bool):
        raise ValueError(
            f"{prefix}{field_name} must be an integer, got boolean ({val})"
        )
    if not isinstance(val, int):
        raise ValueError(
            f"{prefix}{field_name} must be an integer, got {type(val).__name__} ({val!r})"
        )
    if val < 0:
        raise ValueError(
            f"{prefix}{field_name} must be non-negative (>= 0), got {val}"
        )
    return val


def _detect_duplicate_pairs(pairs: Sequence[Tuple[str, Any]]) -> Dict[str, Any]:
    """JSON object_pairs_hook that detects duplicate dictionary keys."""
    res: Dict[str, Any] = {}
    for k, v in pairs:
        if k in res:
            raise ValueError(f"Duplicate key found in JSON: '{k}'")
        res[k] = v
    return res


def load_ground_truth(path: Union[str, Path]) -> Dict[str, Optional[int]]:
    """Load human-reviewed ground truth repetition counts from a JSON file.

    Supports:
    1. Schema with 'clips' list (e.g. reviewed_counts.json):
       {"clips": [{"clip_id": "clip_01", "human_count": 2}, ...]}
    2. Direct mapping: {"clip_01": 2, "clip_02": null, ...}
    3. Nested mapping: {"clip_01": {"human_count": 2}, ...}
    4. List of clip objects: [{"clip_id": "clip_01", "human_count": 2}, ...]

    Counts must be valid non-negative integers or None (for unreviewed).
    Booleans, floats, negative values, and duplicate clip IDs are rejected.

    Args:
        path: Path to the ground truth JSON file.

    Returns:
        Mapping of clip_id -> human_count (or None if unreviewed).

    Raises:
        FileNotFoundError: If the ground truth file does not exist.
        ValueError: If JSON contains duplicate keys, duplicate clip IDs,
            or invalid counts.
    """
    gt_path = Path(path)
    if not gt_path.is_file():
        raise FileNotFoundError(f"Ground truth file not found: {gt_path}")

    raw_text = gt_path.read_text(encoding="utf-8")
    try:
        data = json.loads(raw_text, object_pairs_hook=_detect_duplicate_pairs)
    except json.JSONDecodeError as e:
        raise ValueError(f"Malformed JSON in ground truth file {gt_path}: {e}") from e

    seen_ids: Set[str] = set()
    ground_truth: Dict[str, Optional[int]] = {}

    def _process_clip_item(item: Any, index_or_key: Any) -> None:
        if not isinstance(item, dict):
            raise ValueError(
                f"Expected object for clip entry {index_or_key}, got {type(item).__name__}"
            )
        cid = item.get("clip_id") or item.get("id") or item.get("clip")
        if not cid or not isinstance(cid, str):
            raise ValueError(f"Missing or invalid 'clip_id' for entry {index_or_key}")
        cid = cid.strip()
        if not cid:
            raise ValueError(f"Empty 'clip_id' for entry {index_or_key}")
        if cid in seen_ids:
            raise ValueError(f"Duplicate clip ID in ground truth: '{cid}'")
        seen_ids.add(cid)

        raw_val = item.get("human_count")
        if "human_count" not in item and "count" in item:
            raw_val = item["count"]

        if raw_val is None:
            ground_truth[cid] = None
        else:
            ground_truth[cid] = validate_count(raw_val, "human_count", cid)

    if isinstance(data, list):
        for idx, elem in enumerate(data):
            _process_clip_item(elem, idx)
    elif isinstance(data, dict):
        if "clips" in data and isinstance(data["clips"], list):
            for idx, elem in enumerate(data["clips"]):
                _process_clip_item(elem, idx)
        else:
            for key, val in data.items():
                cid = str(key).strip()
                if not cid:
                    raise ValueError("Empty clip ID in ground truth mapping")
                if cid in seen_ids:
                    raise ValueError(f"Duplicate clip ID in ground truth: '{cid}'")
                seen_ids.add(cid)

                if isinstance(val, dict):
                    raw_val = val.get("human_count")
                    if "human_count" not in val and "count" in val:
                        raw_val = val["count"]
                else:
                    raw_val = val

                if raw_val is None:
                    ground_truth[cid] = None
                else:
                    ground_truth[cid] = validate_count(raw_val, "human_count", cid)
    else:
        raise ValueError(f"Top-level ground truth JSON must be an object or list, got {type(data).__name__}")

    return ground_truth


def load_prediction_summary(
    summary_path: Union[str, Path],
    expected_clip_id: Optional[str] = None,
) -> int:
    """Load model repetition count from a single summary.json file.

    Args:
        summary_path: Path to summary.json.
        expected_clip_id: Optional expected clip ID to verify against summary contents.

    Returns:
        The validated completed_reps count.

    Raises:
        FileNotFoundError: If summary_path does not exist.
        ValueError: If summary JSON is invalid, missing completed_reps, or has
            conflicting clip IDs.
    """
    path = Path(summary_path)
    if not path.is_file():
        raise FileNotFoundError(f"Summary file not found: {path}")

    raw_text = path.read_text(encoding="utf-8")
    try:
        data = json.loads(raw_text, object_pairs_hook=_detect_duplicate_pairs)
    except json.JSONDecodeError as e:
        raise ValueError(f"Malformed summary JSON in {path}: {e}") from e

    if not isinstance(data, dict):
        raise ValueError(f"Summary JSON in {path} must be a dictionary")

    if data.get("status") == "failed":
        raise ValueError(f"Cannot evaluate failed processing output: {path}")
    clip_id = expected_clip_id or path.parent.name
    if "clip_id" in data and data["clip_id"] and data["clip_id"] != clip_id:
        raise ValueError(
            f"Mismatched clip ID: summary declares '{data['clip_id']}' but directory is '{clip_id}'"
        )

    if "completed_reps" not in data:
        raise ValueError(f"Summary JSON in {path} is missing required 'completed_reps' field")

    return validate_count(data["completed_reps"], "completed_reps", clip_id)


def load_predictions_from_dir(
    predictions_dir: Union[str, Path],
    allow_missing_summary: bool = False,
) -> Dict[str, int]:
    """Scan a directory containing clip subdirectories with summary.json.

    Args:
        predictions_dir: Directory containing clip subfolders.
        allow_missing_summary: If True, subdirectories lacking summary.json are skipped.
            If False, missing summary.json raises FileNotFoundError.

    Returns:
        Mapping of clip_id -> model predicted completed_reps.

    Raises:
        FileNotFoundError: If predictions_dir doesn't exist or a clip lacks summary.json.
        NotADirectoryError: If predictions_dir is not a directory.
        ValueError: If duplicate clip IDs or invalid counts are found.
    """
    root = Path(predictions_dir)
    if not root.exists():
        raise FileNotFoundError(f"Predictions directory not found: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"Predictions path is not a directory: {root}")

    # Case: root itself contains summary.json directly
    direct_summary = root / "summary.json"
    if direct_summary.is_file():
        clip_id = root.name
        count = load_prediction_summary(direct_summary, expected_clip_id=clip_id)
        return {clip_id: count}

    predictions: Dict[str, int] = {}
    subdirs = sorted([d for d in root.iterdir() if d.is_dir() and not d.name.startswith(".")])

    for sub in subdirs:
        summary_path = sub / "summary.json"
        if not summary_path.is_file():
            if allow_missing_summary:
                continue
            raise FileNotFoundError(
                f"Missing summary.json in clip directory '{sub.name}' ({sub})"
            )
        clip_id = sub.name
        if clip_id in predictions:
            raise ValueError(f"Duplicate clip ID in predictions: '{clip_id}'")
        count = load_prediction_summary(summary_path, expected_clip_id=clip_id)
        predictions[clip_id] = count

    return predictions


def evaluate_counts(
    predictions: Dict[str, int],
    ground_truth: Dict[str, Optional[int]],
    reject_unknown_ground_truth: bool = True,
) -> Dict[str, Any]:
    """Evaluate predicted repetition counts against human ground truth.

    Args:
        predictions: Mapping of clip_id -> model count.
        ground_truth: Mapping of clip_id -> human count (or None if unreviewed).
        reject_unknown_ground_truth: If True, raises ValueError if ground_truth contains
            reviewed clip IDs that do not exist in predictions.

    Returns:
        Dictionary with evaluation metrics:
        - evaluated_clips: count of clips with both valid prediction and human ground truth
        - unreviewed_clips: count of prediction clips missing human ground truth
        - unreviewed_clip_ids: list of unreviewed clip IDs
        - exact_matches: count where model_count == human_count
        - exact_match_ratio: exact_matches / evaluated_clips (or None if evaluated_clips == 0)
        - mean_absolute_error: sum(|model - human|) / evaluated_clips (or None if 0)
        - per_clip_results: list of dicts for evaluated clips:
            clip_id, human_count, model_count, absolute_error, exact_match
        - limitations: list of methodological limitations
    """
    if not isinstance(predictions, dict):
        raise ValueError(f"predictions must be a dict, got {type(predictions).__name__}")
    if not isinstance(ground_truth, dict):
        raise ValueError(f"ground_truth must be a dict, got {type(ground_truth).__name__}")

    for mapping in (predictions, ground_truth):
        if any(not isinstance(cid,str) or not cid.strip() or cid != cid.strip() for cid in mapping):
            raise ValueError("Clip IDs must be nonempty, unpadded strings")
    for cid, count in ground_truth.items():
        if count is not None:validate_count(count, "human_count", cid)

    # Validate ground truth clip IDs against predictions if requested
    if reject_unknown_ground_truth:
        unknown_gt = [
            cid for cid, val in ground_truth.items()
            if val is not None and cid not in predictions
        ]
        if unknown_gt:
            raise ValueError(
                f"Ground truth contains unknown clip ID(s) not found in predictions: {unknown_gt}"
            )

    evaluated_results: List[Dict[str, Any]] = []
    unreviewed_ids: List[str] = []

    for cid in sorted(predictions.keys()):
        model_count = validate_count(predictions[cid], "model_count", cid)
        human_count = ground_truth.get(cid)

        # UNREVIEWED: Unknown human counts must remain missing/unreviewed and excluded from accuracy metrics!
        # NEVER treated as zero!
        if human_count is None:
            unreviewed_ids.append(cid)
            continue

        human_count = validate_count(human_count, "human_count", cid)
        abs_error = abs(model_count - human_count)
        exact = (model_count == human_count)

        evaluated_results.append({
            "clip_id": cid,
            "human_count": human_count,
            "model_count": model_count,
            "absolute_error": abs_error,
            "exact_match": exact,
        })

    n_evaluated = len(evaluated_results)
    n_unreviewed = len(unreviewed_ids)
    n_exact = sum(1 for r in evaluated_results if r["exact_match"])
    total_abs_error = sum(r["absolute_error"] for r in evaluated_results)

    exact_match_ratio: Optional[float] = (
        (n_exact / n_evaluated) if n_evaluated > 0 else None
    )
    mae: Optional[float] = (
        (total_abs_error / n_evaluated) if n_evaluated > 0 else None
    )

    limitations = [
        "Small sample size; exact matches on exploratory clips do not establish general accuracy.",
        "Unreviewed prediction clips are strictly excluded from error metrics and never assumed zero.",
        "Repetition count evaluation does not validate temporal repetition boundaries (start/inflection/end timestamps) or cycle duration.",
        "Keypoint localization accuracy (e.g. OKS) and exercise posture validity require independent expert annotations."
    ]

    return {
        "evaluated_clips": n_evaluated,
        "unreviewed_clips": n_unreviewed,
        "unreviewed_clip_ids": unreviewed_ids,
        "exact_matches": n_exact,
        "exact_match_ratio": exact_match_ratio,
        "mean_absolute_error": mae,
        "per_clip_results": evaluated_results,
        "limitations": limitations,
    }


def format_evaluation_report(
    results: Dict[str, Any],
    predictions_source: Optional[str] = None,
    ground_truth_source: Optional[str] = None,
) -> str:
    """Format evaluation metrics into a human-readable report stating denominators clearly."""
    lines: List[str] = []
    lines.append("=" * 64)
    lines.append("          GYM POSE REPETITION COUNT EVALUATION REPORT")
    lines.append("=" * 64)
    if predictions_source:
        lines.append(f"Predictions Source : {predictions_source}")
    if ground_truth_source:
        lines.append(f"Ground Truth Source: {ground_truth_source}")
    lines.append("")
    lines.append("Evaluation Summary:")
    total_predictions = results["evaluated_clips"] + results["unreviewed_clips"]
    lines.append(f"  Total prediction clips : {total_predictions}")
    lines.append(f"  Evaluated clips (reviewed) : {results['evaluated_clips']}")
    lines.append(f"  Unreviewed clips (excluded): {results['unreviewed_clips']}")

    if results["evaluated_clips"] > 0:
        ratio = results["exact_match_ratio"]
        ratio_pct = ratio * 100.0 if ratio is not None else 0.0
        mae_val = results["mean_absolute_error"]
        lines.append(
            f"  Exact matches (model == human): {results['exact_matches']} / {results['evaluated_clips']} ({ratio_pct:.1f}%)"
        )
        lines.append(f"  Mean Absolute Error (MAE)     : {mae_val:.2f} reps")
    else:
        lines.append("  Exact matches (model == human): N/A (0 evaluated clips)")
        lines.append("  Mean Absolute Error (MAE)     : N/A (0 evaluated clips)")

    lines.append("")
    lines.append("Per-Clip Results:")
    if results["per_clip_results"]:
        for r in results["per_clip_results"]:
            status_tag = "MATCH" if r["exact_match"] else "MISMATCH"
            lines.append(
                f"  [{status_tag}] {r['clip_id']}: human={r['human_count']}, model={r['model_count']}, abs_error={r['absolute_error']}"
            )
    else:
        lines.append("  (No reviewed clips evaluated)")

    if results["unreviewed_clip_ids"]:
        lines.append("")
        lines.append(f"Unreviewed Clips ({len(results['unreviewed_clip_ids'])}) - EXCLUDED from metrics:")
        for cid in results["unreviewed_clip_ids"]:
            lines.append(f"  - {cid}")

    lines.append("")
    lines.append("Methodological Limitations:")
    for lim in results.get("limitations", []):
        lines.append(f"  - {lim}")
    lines.append("=" * 64)
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    """Build CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="python -m gym_analyser.evaluation",
        description="Evaluate model predicted repetition counts against human ground truth.",
    )
    parser.add_argument(
        "--predictions-dir",
        type=Path,
        required=True,
        help="Directory containing clip subdirectories with summary.json.",
    )
    parser.add_argument(
        "--ground-truth",
        type=Path,
        required=True,
        help="Path to JSON file mapping clip_id to human_count (or matching reviewed_counts.json schema).",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional path to write evaluation receipt JSON.",
    )
    parser.add_argument(
        "--allow-missing-summary",
        action="store_true",
        help="Skip clip subdirectories that lack summary.json instead of raising an error.",
    )
    parser.add_argument(
        "--allow-unmatched-ground-truth",
        action="store_true",
        help="Allow ground truth to contain clip IDs that do not exist in predictions.",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI execution entrypoint."""
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        predictions = load_predictions_from_dir(
            args.predictions_dir,
            allow_missing_summary=args.allow_missing_summary,
        )
        ground_truth = load_ground_truth(args.ground_truth)
        results = evaluate_counts(
            predictions=predictions,
            ground_truth=ground_truth,
            reject_unknown_ground_truth=not args.allow_unmatched_ground_truth,
        )
        report = format_evaluation_report(
            results,
            predictions_source=str(args.predictions_dir),
            ground_truth_source=str(args.ground_truth),
        )
        print(report)

        if args.output_json:
            out_path = Path(args.output_json)
            if out_path.exists() or out_path.is_symlink():
                raise FileExistsError(f"Refusing to overwrite evaluation output: {out_path}")
            out_path.parent.mkdir(parents=True, exist_ok=True)
            receipt = {
                "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "predictions_dir": str(args.predictions_dir),
                "ground_truth_path": str(args.ground_truth),
                **results,
            }
            with out_path.open("x", encoding="utf-8") as f:
                json.dump(receipt, f, indent=2)
            print(f"\nEvaluation receipt written to: {out_path}")

        return 0
    except Exception as e:
        sys.stderr.write(f"Evaluation failed: {e}\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
