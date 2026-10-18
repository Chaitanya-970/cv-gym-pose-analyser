"""Unit and adapter tests for the training and validation CLI.

Tests argument parsing, configuration validation, dataset split verification,
leakage protection, receipt generation, and mock Ultralytics API interactions.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Tuple
from unittest.mock import MagicMock, patch

from gym_analyser.train_cli import (
    build_parser,
    compute_file_sha256,
    generate_experiment_receipt,
    parse_data_yaml,
    run_cli,
    validate_dataset_splits,
    ExperimentConfig,
)


class TestTrainCliUnit(unittest.TestCase):
    """Deterministic unit tests for CLI argument parsing and dataset validation."""

    def setUp(self) -> None:
        self.temp_dir = tempfile.mkdtemp(prefix="test_train_cli_")
        self.addCleanup(lambda: shutil.rmtree(self.temp_dir, ignore_errors=True))

    def _create_synthetic_dataset(
        self,
        train_count: int = 2,
        val_count: int = 1,
        test_count: int = 1,
        train_overlap: bool = False,
        leak_native_test: bool = False,
        manifest_overlap: bool = False,
        name: str = "dataset",
    ) -> Tuple[Path, Path]:
        """Create a minimal synthetic YOLO pose dataset for testing."""
        root = Path(self.temp_dir) / name
        images_dir = root / "images"
        (images_dir / "train").mkdir(parents=True, exist_ok=True)
        (images_dir / "val").mkdir(parents=True, exist_ok=True)
        (images_dir / "test").mkdir(parents=True, exist_ok=True)

        # Write dummy train images
        for i in range(train_count):
            seq = "1001" if not leak_native_test or i > 0 else "9999"
            (images_dir / "train" / f"{seq}_{i+1:06d}.jpg").write_bytes((str(root) + str(i) + "train").encode())

        # Write dummy val images
        for i in range(val_count):
            filename = "1001_000001.jpg" if train_overlap else f"1002_{i+1:06d}.jpg"
            (images_dir / "val" / filename).write_bytes((str(root) + str(i) + "val").encode())

        # Write dummy test images (use offset to avoid identical filenames if testing sequence leakage)
        test_offset = 100 if leak_native_test else 0
        for i in range(test_count):
            (images_dir / "test" / f"9999_{test_offset+i+1:06d}.jpg").write_bytes((str(root) + str(i) + "test").encode())

        # Write data.yaml
        yaml_content = f"""path: {root.as_posix()}
train: images/train
val: images/val
test: images/test
kpt_shape: [17, 3]
names:
  0: person
"""
        yaml_path = root / "data.yaml"
        yaml_path.write_text(yaml_content, encoding="utf-8")

        # Write split manifest
        val_seqs = ["1001"] if manifest_overlap else ["1002"]
        manifest = {
            "train": ["1001"],
            "val": val_seqs,
            "test": ["9999"],
        }
        manifest_path = root / "split_manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

        (root / "source_inventory.json").write_text(json.dumps([
            {"seq": "1001", "train": 1}, {"seq": "1002", "train": 1},
            {"seq": "9999", "train": -1}
        ]))
        for split in ("train", "val", "test"):
            label_dir = root / "labels" / split
            label_dir.mkdir(parents=True)
            for img in (images_dir / split).glob("*.jpg"):
                (label_dir / (img.stem + ".txt")).write_text("0 .5 .5 .5 .5 " + "0 0 0 "*17)

        return yaml_path, manifest_path

    def test_help_message(self) -> None:
        """Verify --help raises SystemExit(0) without importing heavy dependencies."""
        parser = build_parser()
        with self.assertRaises(SystemExit) as ctx:
            parser.parse_args(["--help"])
        self.assertEqual(ctx.exception.code, 0)

    def test_missing_data_yaml(self) -> None:
        """Verify run_cli returns 1 when data YAML is missing."""
        missing = Path(self.temp_dir) / "nonexistent.yaml"
        out = Path(self.temp_dir) / "out"
        code = run_cli(["--data", str(missing), "--output-dir", str(out), "--dry-run"])
        self.assertEqual(code, 1)

    def test_invalid_numeric_bounds(self) -> None:
        """Verify bounds checking on image size, batch, epochs, and workers."""
        yaml_path, _ = self._create_synthetic_dataset()
        out = Path(self.temp_dir) / "out"

        # Invalid imgsz (not multiple of 32)
        code = run_cli([
            "--data", str(yaml_path), "--output-dir", str(out), "--dry-run",
            "--imgsz", "300",
        ])
        self.assertEqual(code, 1)

        # Invalid batch
        code = run_cli([
            "--data", str(yaml_path), "--output-dir", str(out), "--dry-run",
            "--batch", "0",
        ])
        self.assertEqual(code, 1)

        # Invalid epochs
        code = run_cli([
            "--data", str(yaml_path), "--output-dir", str(out), "--dry-run",
            "--epochs", "-5",
        ])
        self.assertEqual(code, 1)

        # Invalid workers
        code = run_cli([
            "--data", str(yaml_path), "--output-dir", str(out), "--dry-run",
            "--workers", "-1",
        ])
        self.assertEqual(code, 1)

    def test_empty_train_or_val_rejected(self) -> None:
        """Verify empty train or val directories are rejected."""
        # Empty val
        yaml_path, manifest_path = self._create_synthetic_dataset(train_count=2, val_count=0)
        with self.assertRaises(ValueError) as ctx:
            validate_dataset_splits(yaml_path, manifest_path)
        self.assertIn("Validation split is empty", str(ctx.exception))

        # Empty train
        yaml_path_2, manifest_path_2 = self._create_synthetic_dataset(
            train_count=0, val_count=2, name="dataset_empty_train"
        )
        with self.assertRaises(ValueError) as ctx:
            validate_dataset_splits(yaml_path_2, manifest_path_2)
        self.assertIn("Train split is empty", str(ctx.exception))

    def test_split_filename_overlap_rejected(self) -> None:
        """Verify identical image filenames across train and val splits are rejected."""
        yaml_path, manifest_path = self._create_synthetic_dataset(train_overlap=True)
        with self.assertRaises(ValueError) as ctx:
            validate_dataset_splits(yaml_path, manifest_path)
        self.assertIn("identical filenames", str(ctx.exception))

    def test_native_test_leakage_rejected(self) -> None:
        """Verify inclusion of native test sequence in train images is rejected."""
        yaml_path, manifest_path = self._create_synthetic_dataset(leak_native_test=True)
        with self.assertRaises(ValueError) as ctx:
            validate_dataset_splits(yaml_path, manifest_path)
        self.assertIn("Native test sequence", str(ctx.exception))

    def test_manifest_overlap_rejected(self) -> None:
        """Verify shared sequences across manifest splits are rejected."""
        yaml_path, manifest_path = self._create_synthetic_dataset(manifest_overlap=True)
        with self.assertRaises(ValueError) as ctx:
            validate_dataset_splits(yaml_path, manifest_path)
        self.assertIn("Manifest train and val share sequences", str(ctx.exception))

    def test_overwrite_protection(self) -> None:
        """Verify run_cli refuses to overwrite an existing non-empty directory."""
        yaml_path, _ = self._create_synthetic_dataset()
        out = Path(self.temp_dir) / "output_test"

        # First run succeeds
        code = run_cli([
            "--data", str(yaml_path),
            "--output-dir", str(out),
            "--dry-run",
        ])
        self.assertEqual(code, 0)
        self.assertTrue((out / "experiment_receipt.json").is_file())

        # Second run without overwrite fails
        code_fail = run_cli([
            "--data", str(yaml_path),
            "--output-dir", str(out),
            "--dry-run",
        ])
        self.assertEqual(code_fail, 1)

        # Third run still refuses overwrite; receipts are immutable
        code_overwrite = run_cli([
            "--data", str(yaml_path),
            "--output-dir", str(out),
            "--dry-run",
            "--overwrite",
        ])
        self.assertEqual(code_overwrite, 1)

    def test_dry_run_receipt_content(self) -> None:
        """Verify dry-run receipt structure, effective config, and hashes."""
        yaml_path, manifest_path = self._create_synthetic_dataset()
        out = Path(self.temp_dir) / "receipt_out"

        code = run_cli([
            "--data", str(yaml_path),
            "--output-dir", str(out),
            "--mode", "val",
            "--model", "yolo11n-pose.pt",
            "--imgsz", "320",
            "--batch", "16",
            "--seed", "42",
            "--dry-run",
        ])
        self.assertEqual(code, 0)

        receipt_file = out / "experiment_receipt.json"
        self.assertTrue(receipt_file.is_file())
        receipt = json.loads(receipt_file.read_text(encoding="utf-8"))

        self.assertEqual(receipt["status"], "dry_run_passed")
        self.assertEqual(receipt["mode"], "val")
        self.assertEqual(receipt["seed"], 42)
        self.assertEqual(receipt["effective_config"]["imgsz"], 320)
        self.assertEqual(receipt["effective_config"]["batch"], 16)
        self.assertEqual(receipt["dataset"]["train_count"], 2)
        self.assertEqual(receipt["dataset"]["val_count"], 1)
        self.assertEqual(receipt["dataset"]["test_count"], 1)
        self.assertTrue(receipt["validation_checks"]["data_yaml_valid"])
        self.assertTrue(receipt["validation_checks"]["train_nonempty"])
        self.assertTrue(receipt["validation_checks"]["val_nonempty"])
        self.assertTrue(receipt["validation_checks"]["splits_disjoint"])
        self.assertTrue(receipt["validation_checks"]["native_test_protected"])

    @patch("gym_analyser.train_cli.get_environment_info")
    def test_mock_val_execution(self, mock_env: MagicMock) -> None:
        """Verify model.val is called with exact expected arguments when dry-run is False."""
        mock_env.return_value = {"platform": "mock_platform"}
        yaml_path, _ = self._create_synthetic_dataset()
        out = Path(self.temp_dir) / "val_exec_out"

        mock_yolo_cls = MagicMock()
        mock_model_instance = MagicMock()
        mock_yolo_cls.return_value = mock_model_instance
        mock_model_instance.task = "pose"
        mock_model_instance.trainer.args = SimpleNamespace(optimizer="AdamW", lr0=0.0001)
        mock_model_instance.val.return_value = {"pose_map50": 0.85}

        checkpoint = Path(self.temp_dir) / "mock_pose.pt"
        checkpoint.write_bytes(b"synthetic checkpoint; never loaded")
        with patch.dict("sys.modules", {"ultralytics": MagicMock(YOLO=mock_yolo_cls)}):
            code = run_cli([
                "--data", str(yaml_path),
                "--output-dir", str(out),
                "--mode", "val",
                "--model", str(checkpoint),
                "--device", "cpu",
                "--imgsz", "320",
                "--batch", "8",
                "--workers", "0",
            ])
            self.assertEqual(code, 0)
            mock_yolo_cls.assert_called_once_with(str(checkpoint))
            mock_model_instance.val.assert_called_once()
            val_call_kwargs = mock_model_instance.val.call_args[1]
            self.assertEqual(val_call_kwargs["imgsz"], 320)
            self.assertEqual(val_call_kwargs["batch"], 8)
            self.assertEqual(val_call_kwargs["device"], "cpu")
            self.assertEqual(val_call_kwargs["split"], "val")

    @patch("gym_analyser.train_cli.get_environment_info")
    def test_mock_train_execution(self, mock_env: MagicMock) -> None:
        """Verify model.train is called with exact expected arguments when dry-run is False."""
        mock_env.return_value = {"platform": "mock_platform"}
        yaml_path, _ = self._create_synthetic_dataset()
        out = Path(self.temp_dir) / "train_exec_out"

        mock_yolo_cls = MagicMock()
        mock_model_instance = MagicMock()
        mock_yolo_cls.return_value = mock_model_instance
        mock_model_instance.task = "pose"
        mock_model_instance.trainer.args = SimpleNamespace(optimizer="AdamW", lr0=0.0001)
        mock_model_instance.train.return_value = {"final_loss": 0.12}

        checkpoint = Path(self.temp_dir) / "mock_pose.pt"
        checkpoint.write_bytes(b"synthetic checkpoint; never loaded")
        with patch.dict("sys.modules", {"ultralytics": MagicMock(YOLO=mock_yolo_cls)}):
            code = run_cli([
                "--data", str(yaml_path),
                "--output-dir", str(out),
                "--mode", "train",
                "--model", str(checkpoint),
                "--device", "cpu",
                "--imgsz", "320",
                "--batch", "16",
                "--epochs", "5",
                "--workers", "0",
                "--seed", "42",
            ])
            self.assertEqual(code, 0)
            mock_yolo_cls.assert_called_once_with(str(checkpoint))
            mock_model_instance.train.assert_called_once()
            train_call_kwargs = mock_model_instance.train.call_args[1]
            self.assertEqual(train_call_kwargs["epochs"], 5)
            self.assertEqual(train_call_kwargs["batch"], 16)
            self.assertEqual(train_call_kwargs["imgsz"], 320)
            self.assertEqual(train_call_kwargs["seed"], 42)
            self.assertTrue(train_call_kwargs["amp"])
            self.assertEqual(train_call_kwargs["optimizer"], "AdamW")
            self.assertEqual(train_call_kwargs["lr0"], 0.0001)


    def test_missing_requested_manifest_is_rejected(self):
        yaml_path, _ = self._create_synthetic_dataset()
        with self.assertRaises(FileNotFoundError):
            validate_dataset_splits(yaml_path, Path(self.temp_dir)/"missing.json")

    def test_different_frames_from_same_sequence_cannot_cross_splits(self):
        yaml_path, manifest = self._create_synthetic_dataset()
        old = yaml_path.parent/"images/val/1002_000001.jpg"
        old.rename(old.with_name("1001_000099.jpg"))
        with self.assertRaisesRegex(ValueError, "membership"):
            validate_dataset_splits(yaml_path, manifest)

    def test_manifest_cannot_reclassify_original_native_test(self):
        yaml_path, manifest = self._create_synthetic_dataset()
        inventory = yaml_path.parent/"source_inventory.json"
        rows = json.loads(inventory.read_text())
        rows[0]["train"] = -1
        inventory.write_text(json.dumps(rows))
        with self.assertRaisesRegex(ValueError, "Original native test"):
            validate_dataset_splits(yaml_path, manifest)

    def test_renamed_duplicate_content_cannot_cross_splits(self):
        yaml_path, manifest = self._create_synthetic_dataset()
        train = yaml_path.parent/"images/train/1001_000001.jpg"
        val = yaml_path.parent/"images/val/1002_000001.jpg"
        val.write_bytes(train.read_bytes())
        with self.assertRaisesRegex(ValueError, "Identical image content"):
            validate_dataset_splits(yaml_path, manifest)

    def test_pose_shape_required_and_content_hash_tracks_labels(self):
        yaml_path, manifest = self._create_synthetic_dataset()
        first = validate_dataset_splits(yaml_path, manifest)["content_sha256"]
        label = yaml_path.parent/"labels/train/1001_000001.txt"
        label.write_text(label.read_text() + "\n")
        second = validate_dataset_splits(yaml_path, manifest)["content_sha256"]
        self.assertNotEqual(first, second)
        yaml_path.write_text(yaml_path.read_text().replace("[17, 3]", "[12, 3]"))
        with self.assertRaisesRegex(ValueError, "kpt_shape"):
            validate_dataset_splits(yaml_path, manifest)

    def test_relative_dataset_root_uses_yaml_location(self):
        yaml_path, manifest = self._create_synthetic_dataset()
        yaml_path.write_text(yaml_path.read_text().replace(yaml_path.parent.as_posix(), "."))
        summary = validate_dataset_splits(yaml_path, manifest)
        self.assertEqual(summary["base_path"], str(yaml_path.parent.resolve()))


    def test_finetuning_controls_validate_without_training(self):
        yaml_path, _ = self._create_synthetic_dataset()
        for args in (("--lr0", "nan"), ("--lr0", "0"), ("--freeze", "-1")):
            out = Path(self.temp_dir)/"invalid_controls"
            self.assertEqual(run_cli(["--data", str(yaml_path), "--output-dir", str(out), "--dry-run", *args]), 1)
            self.assertFalse(out.exists())
        out = Path(self.temp_dir)/"controls"
        self.assertEqual(run_cli(["--data",str(yaml_path),"--output-dir",str(out),"--dry-run","--optimizer","AdamW","--lr0","0.00005","--freeze","10"]),0)
        config = json.loads((out/"experiment_receipt.json").read_text())["effective_config"]
        self.assertEqual((config["optimizer"],config["lr0"],config["freeze"]),("AdamW",0.00005,10))

    @patch("gym_analyser.train_cli.get_environment_info")
    def test_failed_run_preserves_failure_receipt(self, env):
        env.return_value = {}
        yaml_path, _ = self._create_synthetic_dataset()
        out = Path(self.temp_dir)/"failed_run"
        model = MagicMock()
        model.task = "detect"
        checkpoint = Path(self.temp_dir)/"detect.pt"
        checkpoint.write_bytes(b"mock checkpoint")
        with patch.dict("sys.modules", {"ultralytics":MagicMock(YOLO=MagicMock(return_value=model))}):
            self.assertEqual(run_cli(["--data",str(yaml_path),"--output-dir",str(out),"--model",str(checkpoint)]),1)
        receipt = json.loads((out/"experiment_receipt.json").read_text())
        self.assertEqual(receipt["status"],"failed")
        model.train.assert_not_called()


    def test_invalid_label_rejected_before_experiment(self):
        yaml_path, _ = self._create_synthetic_dataset()
        label = yaml_path.parent/"labels/train/1001_000001.txt"
        label.write_text("0 nan .5 .5 .5")
        output = Path(self.temp_dir)/"invalid_label_run"
        self.assertEqual(run_cli(["--data",str(yaml_path),"--output-dir",str(output),"--dry-run"]),1)
        self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
