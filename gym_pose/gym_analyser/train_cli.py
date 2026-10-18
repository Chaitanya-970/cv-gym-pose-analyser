"""Training and validation CLI for pose estimation models.

Provides a reproducible interface for baseline validation and fine-tuning
using official Ultralytics YOLO pose APIs.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import re
import stat
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple


@dataclass(frozen=True)
class ExperimentConfig:
    """Validated experiment configuration."""

    mode: str
    data_yaml: Path
    model: str
    device: str
    imgsz: int
    batch: int
    epochs: int
    workers: int
    seed: int
    amp: bool
    output_dir: Path
    overwrite: bool
    dry_run: bool
    split_manifest: Optional[Path] = None
    optimizer: str = "AdamW"
    lr0: float = 0.0001
    freeze: int = 0


def compute_file_sha256(file_path: Path) -> str:
    """Compute hex SHA-256 digest of a local file."""
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def parse_data_yaml(yaml_path: Path) -> Dict[str, Any]:
    """Parse YOLO data.yaml configuration."""
    if not yaml_path.is_file():
        raise FileNotFoundError(f"Data configuration file not found: {yaml_path}")

    try:
        import yaml
        with open(yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        if not isinstance(data, dict):
            raise ValueError(f"Invalid YAML content in {yaml_path}: expected mapping")
        return data
    except ImportError as e:
        raise ImportError("PyYAML is required to parse pose dataset configuration") from e


def resolve_split_dir(base_path: Path, rel_or_abs: str) -> Path:
    """Resolve a split directory relative to dataset path or YAML location."""
    p = Path(rel_or_abs)
    if p.is_absolute():
        return p
    return (base_path / p).resolve()


def collect_image_files(split_dir: Path) -> List[Path]:
    """Collect image paths from a split directory."""
    if not split_dir.is_dir():
        return []
    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    images = [
        p for p in split_dir.iterdir()
        if p.is_file() and p.suffix.lower() in valid_extensions
    ]
    images.sort()
    return images


def validate_dataset_splits(
    data_yaml_path: Path,
    split_manifest_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Validate dataset split existence, non-emptiness, and disjointness.

    Rejects empty train or val sets, overlapping images, and inclusion
    of native test sequences in training or validation splits.
    """
    data_cfg = parse_data_yaml(data_yaml_path)

    # Resolve base path
    if "path" in data_cfg and data_cfg["path"]:
        base_path = Path(str(data_cfg["path"]))
    else:
        base_path = data_yaml_path.parent

    if not base_path.is_absolute():
        base_path = (data_yaml_path.parent / base_path).resolve()
    if data_cfg.get("kpt_shape") != [17, 3]:
        raise ValueError("This runner requires COCO-compatible kpt_shape [17, 3]")
    if data_cfg.get("names") not in ({0: "person"}, ["person"]):
        raise ValueError("Expected a single class: person")

    # Check train and val paths
    if "train" not in data_cfg:
        raise ValueError(f"Missing required 'train' key in {data_yaml_path}")
    if "val" not in data_cfg:
        raise ValueError(f"Missing required 'val' key in {data_yaml_path}")

    train_dir = resolve_split_dir(base_path, str(data_cfg["train"]))
    val_dir = resolve_split_dir(base_path, str(data_cfg["val"]))

    if not train_dir.is_dir():
        raise FileNotFoundError(f"Train image directory not found: {train_dir}")
    if not val_dir.is_dir():
        raise FileNotFoundError(f"Validation image directory not found: {val_dir}")

    train_images = collect_image_files(train_dir)
    val_images = collect_image_files(val_dir)

    if not train_images:
        raise ValueError(f"Train split is empty: {train_dir}")
    if not val_images:
        raise ValueError(f"Validation split is empty: {val_dir}")

    # Verify disjointness by filename
    train_filenames = {p.name for p in train_images}
    val_filenames = {p.name for p in val_images}
    overlap_filenames = train_filenames.intersection(val_filenames)
    if overlap_filenames:
        sample = sorted(list(overlap_filenames))[:5]
        raise ValueError(
            f"Train and validation splits share {len(overlap_filenames)} identical filenames: {sample}"
        )

    # Optional test split check
    test_dir: Optional[Path] = None
    test_images: List[Path] = []
    if "test" in data_cfg and data_cfg["test"]:
        resolved_test = resolve_split_dir(base_path, str(data_cfg["test"]))
        if not resolved_test.is_dir():
            raise FileNotFoundError(f"Configured test directory not found: {resolved_test}")
        if resolved_test.is_dir():
            test_dir = resolved_test
            test_images = collect_image_files(test_dir)
            test_filenames = {p.name for p in test_images}
            train_test_overlap = train_filenames.intersection(test_filenames)
            if train_test_overlap:
                sample = sorted(list(train_test_overlap))[:5]
                raise ValueError(
                    f"Train and test splits share {len(train_test_overlap)} filenames: {sample}"
                )
            val_test_overlap = val_filenames.intersection(test_filenames)
            if val_test_overlap:
                sample = sorted(list(val_test_overlap))[:5]
                raise ValueError(
                    f"Validation and test splits share {len(val_test_overlap)} filenames: {sample}"
                )

    manifest_to_check = split_manifest_path or base_path / "split_manifest.json"
    if not manifest_to_check.is_file():
        raise FileNotFoundError(f"Required sequence manifest not found: {manifest_to_check}")
    manifest_content = json.loads(manifest_to_check.read_text(encoding="utf-8"))
    if not isinstance(manifest_content, dict) or set(manifest_content) != {"train", "val", "test"}:
        raise ValueError("Manifest must contain train, val and test lists")
    sets = {}
    for split, ids in manifest_content.items():
        if not isinstance(ids, list) or any(not isinstance(i, str) or re.fullmatch(r"[0-9]{4}", i) is None for i in ids):
            raise ValueError(f"Invalid sequence list for {split}")
        if len(set(ids)) != len(ids):
            raise ValueError(f"Duplicate sequence IDs in {split}")
        sets[split] = set(ids)
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        if sets[a] & sets[b]:
            raise ValueError(f"Manifest {a} and {b} share sequences: {sets[a] & sets[b]}")
    split_images = {"train": train_images, "val": val_images, "test": test_images}
    for split, files in split_images.items():
        actual = set()
        for file in files:
            match = re.fullmatch(r"([0-9]{4})_[0-9]{6}\.[A-Za-z]+", file.name)
            if match is None:
                raise ValueError(f"Unexpected Penn image filename: {file.name}")
            seq = match.group(1)
            if split in {"train", "val"} and seq in sets["test"]:
                raise ValueError(f"Native test sequence {seq} found in {split}")
            actual.add(seq)
        if actual != sets[split]:
            raise ValueError(f"Actual {split} sequence membership differs from manifest: {actual ^ sets[split]}")
    inventory_path = base_path / "source_inventory.json"
    if not inventory_path.is_file():
        raise FileNotFoundError("Missing original source_inventory.json for native split verification")
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    if not isinstance(inventory, list):
        raise ValueError("Source inventory must be a list of original annotation metadata")
    native = {}
    for row in inventory:
        seq, flag = row.get("seq"), row.get("train")
        if seq in native or not isinstance(seq, str) or flag not in {-1, 0, 1} or isinstance(flag, bool):
            raise ValueError("Invalid or duplicate source inventory row")
        native[seq] = flag
    for split, ids in sets.items():
        for seq in ids:
            if seq not in native or (split in {"train", "val"} and native[seq] != 1):
                raise ValueError(f"Original native test or unknown sequence {seq} in {split}")
    fingerprints = []
    seen_bytes = {}
    for split, files in split_images.items():
        for file in files:
            digest = compute_file_sha256(file)
            if digest in seen_bytes and seen_bytes[digest] != split:
                raise ValueError("Identical image content occurs across splits")
            seen_bytes[digest] = split
            label = base_path / "labels" / split / (file.stem + ".txt")
            if not label.is_file():
                raise FileNotFoundError(f"Missing pose label: {label}")
            lines = [line.split() for line in label.read_text(encoding="utf-8").splitlines() if line.strip()]
            if len(lines) != 1 or len(lines[0]) != 56:
                raise ValueError(f"Expected exactly one 56-token person pose label: {label}")
            values = [float(token) for token in lines[0]]
            if not all(math.isfinite(value) for value in values) or values[0] != 0:
                raise ValueError(f"Nonfinite coordinates or invalid class in {label}")
            if not all(0 <= value <= 1 for value in values[1:5]) or values[3] <= 0 or values[4] <= 0:
                raise ValueError(f"Invalid normalized box in {label}")
            for k in range(5, 56, 3):
                if not 0 <= values[k] <= 1 or not 0 <= values[k+1] <= 1 or values[k+2] not in {0,1,2}:
                    raise ValueError(f"Invalid normalized joint or visibility in {label}")
            fingerprints.append([split, file.name, digest, compute_file_sha256(label)])
    manifest_info = {
        "path": str(manifest_to_check.resolve()), "sha256": compute_file_sha256(manifest_to_check),
        "source_inventory_sha256": compute_file_sha256(inventory_path),
        "train_sequences": sorted(sets["train"]), "val_sequences": sorted(sets["val"]),
        "test_sequences": sorted(sets["test"]),
    }
    content_sha256 = hashlib.sha256(json.dumps(fingerprints, sort_keys=True).encode()).hexdigest()

    return {
        "base_path": str(base_path.resolve()),
        "train_dir": str(train_dir.resolve()),
        "train_count": len(train_images),
        "val_dir": str(val_dir.resolve()),
        "val_count": len(val_images),
        "test_dir": str(test_dir.resolve()) if test_dir else None,
        "test_count": len(test_images),
        "manifest": manifest_info,
        "content_sha256": content_sha256,
    }


def get_environment_info() -> Dict[str, Any]:
    """Collect platform and runtime version information."""
    info: Dict[str, Any] = {
        "platform": platform.platform(),
        "python_version": sys.version.split()[0],
        "torch_version": "unavailable",
        "cuda_available": False,
        "cuda_device_name": None,
        "ultralytics_version": "unavailable",
    }
    for package, key in (("torch", "torch_version"), ("ultralytics", "ultralytics_version")):
        try:info[key] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:pass
    # Configuration-only runs inspect versions without importing GPU frameworks.
    torch = sys.modules.get("torch")
    info["cuda_available"] = None
    if torch is not None:
        info["cuda_available"] = bool(torch.cuda.is_available())
        if info["cuda_available"]:info["cuda_device_name"] = torch.cuda.get_device_name(0)

    return info


def generate_experiment_receipt(
    config: ExperimentConfig,
    dataset_summary: Dict[str, Any],
    status: str,
    extra_metrics: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Generate a run receipt recording full reproducibility metadata."""
    receipt = {
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "status": status,
        "mode": config.mode,
        "seed": config.seed,
        "environment": get_environment_info(),
        "dataset": {
            "data_yaml_path": str(config.data_yaml.resolve()),
            "data_yaml_sha256": compute_file_sha256(config.data_yaml),
            "train_count": dataset_summary["train_count"],
            "val_count": dataset_summary["val_count"],
            "test_count": dataset_summary["test_count"],
            "manifest": dataset_summary.get("manifest"),
            "content_sha256": dataset_summary.get("content_sha256"),
        },
        "effective_config": {
            "model": config.model,
            "device": config.device,
            "imgsz": config.imgsz,
            "batch": config.batch,
            "epochs": config.epochs,
            "workers": config.workers,
            "seed": config.seed,
            "amp": config.amp,
            "optimizer": config.optimizer,
            "lr0": config.lr0,
            "freeze": config.freeze,
            "overwrite": config.overwrite,
            "dry_run": config.dry_run,
        },
        "validation_checks": {
            "data_yaml_valid": True,
            "train_nonempty": dataset_summary["train_count"] > 0,
            "val_nonempty": dataset_summary["val_count"] > 0,
            "splits_disjoint": True,
            "native_test_protected": True,
        },
        "notes": (
            "Metrics measure single-frame 2D keypoint localization across 12 shared body joints. "
            "They do not evaluate multi-frame repetition counting or hysteresis state transitions. "
            "Test split remains reserved for frozen evaluation."
        ),
    }
    if extra_metrics:
        receipt["results"] = extra_metrics
    return receipt


def build_parser() -> argparse.ArgumentParser:
    """Construct command-line argument parser."""
    parser = argparse.ArgumentParser(
        prog="gym_analyser.train_cli",
        description="Reproducible training and validation CLI for pose models.",
    )
    parser.add_argument(
        "--mode",
        choices=["val", "train"],
        default="val",
        help="Execution mode: 'val' for baseline evaluation, 'train' for fine-tuning (default: val).",
    )
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to YOLO data.yaml configuration file.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolo11n-pose.pt",
        help="Local pose checkpoint; default also searches gym_pose/models/. No implicit downloads.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Compute device for execution, e.g. 'cpu', 'cuda', 'cuda:0' (default: cpu).",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=320,
        help="Input image dimension in pixels (default: 320).",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=16,
        help="Batch size for training or validation (default: 16).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of fine-tuning epochs (default: 10).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="Number of dataloader worker processes; use 0 on Windows (default: 0).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Deterministic random seed (default: 42).",
    )
    parser.add_argument("--optimizer", choices=["AdamW", "SGD", "auto"], default="AdamW")
    parser.add_argument("--lr0", type=float, default=0.0001, help="Initial learning rate; optimizer=auto may override it")
    parser.add_argument("--freeze", type=int, default=0, help="Number of initial model layers to freeze")
    parser.add_argument(
        "--no-amp",
        action="store_true",
        help="Disable automatic mixed precision (AMP is enabled by default).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory to store run receipts and output artifacts.",
    )
    parser.add_argument(
        "--split-manifest",
        type=str,
        default=None,
        help="Optional path to sequence split manifest JSON for leak checks.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Deprecated compatibility flag; nonempty output directories are always refused.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate configuration, dataset splits, and manifests without launching training or inference.",
    )
    return parser


def run_cli(args: Optional[Sequence[str]] = None) -> int:
    """Main CLI entrypoint."""
    parser = build_parser()
    parsed = parser.parse_args(args)

    data_path = Path(parsed.data).resolve()
    if not data_path.is_file():
        sys.stderr.write(f"Error: Data configuration file not found: {data_path}\n")
        return 1

    raw_out = Path(os.path.abspath(parsed.output_dir))
    for component in (raw_out, *raw_out.parents):
        if component.exists() or component.is_symlink():
            attrs = component.lstat()
            if stat.S_ISLNK(attrs.st_mode) or getattr(attrs, "st_file_attributes", 0) & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 1024):
                sys.stderr.write("Error: output path contains a link/reparse point\n")
                return 1
    out_dir = raw_out.resolve()
    if out_dir.is_dir() and any(out_dir.iterdir()):
        sys.stderr.write(
            f"Error: Output directory {out_dir} is non-empty. Choose a fresh output directory.\n"
        )
        return 1

    manifest_path = Path(parsed.split_manifest).resolve() if parsed.split_manifest else None

    # Validate numeric bounds
    if parsed.imgsz <= 0 or parsed.imgsz % 32 != 0:
        sys.stderr.write(f"Error: --imgsz must be a positive multiple of 32, got {parsed.imgsz}\n")
        return 1
    if parsed.batch <= 0:
        sys.stderr.write(f"Error: --batch must be positive, got {parsed.batch}\n")
        return 1
    if parsed.epochs <= 0:
        sys.stderr.write(f"Error: --epochs must be positive, got {parsed.epochs}\n")
        return 1
    if parsed.workers < 0:
        sys.stderr.write(f"Error: --workers must be non-negative, got {parsed.workers}\n")
        return 1

    if not math.isfinite(parsed.lr0) or not 0 < parsed.lr0 <= 1 or parsed.freeze < 0:
        sys.stderr.write("Error: --lr0 must be finite in (0,1] and --freeze nonnegative\n")
        return 1

    config = ExperimentConfig(
        mode=parsed.mode,
        data_yaml=data_path,
        model=parsed.model,
        device=parsed.device,
        imgsz=parsed.imgsz,
        batch=parsed.batch,
        epochs=parsed.epochs,
        workers=parsed.workers,
        seed=parsed.seed,
        amp=not parsed.no_amp,
        output_dir=out_dir,
        overwrite=parsed.overwrite,
        dry_run=parsed.dry_run,
        split_manifest=manifest_path,
        optimizer=parsed.optimizer,
        lr0=parsed.lr0,
        freeze=parsed.freeze,
    )

    # Validate dataset
    try:
        dataset_summary = validate_dataset_splits(data_path, manifest_path)
    except Exception as e:
        sys.stderr.write(f"Dataset validation failed: {e}\n")
        return 1

    base_path = Path(dataset_summary["base_path"])
    if out_dir == base_path or out_dir in base_path.parents or base_path in out_dir.parents:
        sys.stderr.write("Error: experiment output must be separate from source dataset\n")
        return 1
    out_dir.mkdir(parents=True, exist_ok=True)

    if config.dry_run:
        receipt = generate_experiment_receipt(
            config=config,
            dataset_summary=dataset_summary,
            status="dry_run_passed",
        )
        receipt_path = out_dir / "experiment_receipt.json"
        receipt_path.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
        print(f"Dry run succeeded. Experiment receipt written to {receipt_path}")
        print(f"  Mode: {config.mode}")
        print(f"  Train samples: {dataset_summary['train_count']}")
        print(f"  Val samples: {dataset_summary['val_count']}")
        print(f"  Effective model: {config.model}")
        print(f"  Device: {config.device}")
        return 0

    # Non-dry-run execution
    # Delayed import of Ultralytics to keep CLI startup and dry-run lightweight
    try:
        from ultralytics import YOLO
    except ImportError as e:
        sys.stderr.write(f"Error: Ultralytics is required for execution: {e}\n")
        return 1

    try:
        # Resolve local checkpoints first; never download a missing model implicitly.
        checkpoint = Path(config.model)
        if not checkpoint.is_file():
            candidate = Path(__file__).resolve().parents[1] / "models" / checkpoint.name
            if candidate.is_file():checkpoint = candidate
            else:raise FileNotFoundError(f"Local pose checkpoint not found: {config.model}")
        model = YOLO(str(checkpoint))
        if model.task != "pose":raise ValueError("A YOLO pose checkpoint is required")
        import torch
        gpu_run = torch.cuda.is_available() and config.device not in {"cpu", "mps"}
        if gpu_run:torch.cuda.reset_peak_memory_stats()
        started = datetime.datetime.now(datetime.timezone.utc)
        import yaml
        resolved_data = parse_data_yaml(config.data_yaml)
        resolved_data["path"] = dataset_summary["base_path"]
        for split in ("train", "val", "test"):
            if dataset_summary.get(split + "_dir"):
                resolved_data[split] = dataset_summary[split + "_dir"]
        resolved_yaml = out_dir / "resolved_data.yaml"
        resolved_yaml.write_text(yaml.safe_dump(resolved_data, sort_keys=False), encoding="utf-8")
        running = generate_experiment_receipt(config, dataset_summary, "running")
        running["model_sha256"] = compute_file_sha256(checkpoint)
        (out_dir / "experiment_receipt.json").write_text(json.dumps(running, indent=2))
        if config.mode == "val":
            results = model.val(
                data=str(resolved_yaml),
                imgsz=config.imgsz,
                batch=config.batch,
                device=config.device,
                workers=config.workers,
                split="val",
                half=bool(config.amp and gpu_run),
                project=str(config.output_dir),
                name="val_run",
                exist_ok=False,
                seed=config.seed,
                deterministic=True,
            )
            extra = {"metrics": results if isinstance(results, dict) else results.results_dict}
        else:
            results = model.train(
                data=str(resolved_yaml),
                epochs=config.epochs,
                optimizer=config.optimizer,
                lr0=config.lr0,
                freeze=config.freeze,
                batch=config.batch,
                imgsz=config.imgsz,
                device=config.device,
                workers=config.workers,
                seed=config.seed,
                amp=config.amp,
                project=str(config.output_dir),
                name="train_run",
                exist_ok=False,
                deterministic=True,
            )
            extra = {"metrics": results if isinstance(results, dict) else results.results_dict}
            if hasattr(model, "trainer") and hasattr(model.trainer, "args"):
                extra["ultralytics_train_args"] = dict(vars(model.trainer.args))

        extra["elapsed_seconds"] = (datetime.datetime.now(datetime.timezone.utc) - started).total_seconds()
        extra["peak_gpu_allocated_bytes"] = torch.cuda.max_memory_allocated() if gpu_run else None
        extra["peak_gpu_reserved_bytes"] = torch.cuda.max_memory_reserved() if gpu_run else None
        extra["model_sha256"] = compute_file_sha256(checkpoint)
        extra["resolved_data_sha256"] = compute_file_sha256(resolved_yaml)
        receipt = generate_experiment_receipt(
            config=config,
            dataset_summary=dataset_summary,
            status="completed",
            extra_metrics=extra,
        )
        receipt_path = out_dir / "experiment_receipt.json"
        receipt_path.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
        return 0
    except Exception as e:
        failed = generate_experiment_receipt(config, dataset_summary, "failed", {"error": str(e)})
        (out_dir / "experiment_receipt.json").write_text(json.dumps(failed, indent=2))
        sys.stderr.write(f"Execution failed: {e}\n")
        return 1


if __name__ == "__main__":
    sys.exit(run_cli())
