"""Unit tests for Penn Action dataset converter, semantic mapping, and YOLO pose export.

All fixtures used in this module are explicitly synthetic and used solely for parser,
serialization, and error-handling unit testing. They are never represented as real dataset evidence.
All synthetic coordinates and bounding boxes conform strictly to legal image boundaries.
"""

import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np

from gym_analyser.penn_converter import (
    COCO_FLIP_INDEXES,
    COCO_JOINT_NAMES,
    COCO_TO_PENN_BODY_MAP,
    PENN_JOINT_NAMES,
    PENN_TO_COCO_BODY_MAP,
    PennActionAnnotation,
    convert_penn_frame_to_yolo,
    generate_sequence_splits,
    load_penn_mat,
    render_penn_overlay,
    run_conversion,
    validate_split_manifest,
    write_yolo_data_yaml,
)


def _make_legal_synthetic_mat(
    action: str = "squats",
    nframes: int = 5,
    is_train: int = 1,
    pose_view: str = "side",
    bbox_shape: str = "normal",
    corrupt_key: str = "",
    bad_vis_val: int = 0,
    img_w: int = 320,
    img_h: int = 240,
) -> Dict[str, object]:
    """Generate explicitly synthetic Penn Action .mat dictionary with strictly legal coordinates."""
    rng = np.random.RandomState(42)

    # 13 joints x in [50, 250], y in [50, 180] strictly inside [0, img_w] and [0, img_h]
    x = rng.uniform(60.0, 240.0, size=(nframes, 13))
    y = rng.uniform(60.0, 180.0, size=(nframes, 13))

    if bad_vis_val != 0:
        vis = np.full((nframes, 13), bad_vis_val, dtype=np.int32)
    else:
        vis = rng.choice([0, 1], size=(nframes, 13)).astype(np.int64)

    if bbox_shape == "normal":
        bbox = np.zeros((nframes, 4), dtype=np.float64)
        for i in range(nframes):
            bbox[i] = [
                np.min(x[i]) - 10.0,
                np.min(y[i]) - 10.0,
                np.max(x[i]) + 10.0,
                np.max(y[i]) + 10.0,
            ]
    elif bbox_shape == "short_by_one":
        bbox = np.zeros((nframes - 1, 4), dtype=np.float64)
        for i in range(nframes - 1):
            bbox[i] = [
                np.min(x[i]) - 10.0,
                np.min(y[i]) - 10.0,
                np.max(x[i]) + 10.0,
                np.max(y[i]) + 10.0,
            ]
    else:
        bbox = np.zeros((nframes, 2), dtype=np.float64)

    mat_dict = {
        "action": np.array([action], dtype=object),
        "nframes": np.array([[nframes]], dtype=np.int32),
        "train": np.array([[is_train]], dtype=np.int32),
        "pose": np.array([pose_view], dtype=object),
        "dimensions": np.array([[img_h, img_w, nframes]], dtype=np.int32),
        "bbox": bbox,
        "x": x,
        "y": y,
        "visibility": vis,
    }

    if corrupt_key and corrupt_key in mat_dict:
        del mat_dict[corrupt_key]

    return mat_dict


class TestPennConverter(unittest.TestCase):
    """Test suite for Penn Action semantic conversion and YOLO export."""

    def test_01_semantic_mapping_and_left_right_identity(self) -> None:
        """Verify 12 body joints map 1-to-1 to COCO body joints and head is excluded."""
        # Head (index 0) must NOT be mapped to COCO nose or any other joint
        self.assertNotIn(0, PENN_TO_COCO_BODY_MAP)

        # Verified 12 body joints
        expected_mappings = {
            1: 5,   # left_shoulder
            2: 6,   # right_shoulder
            3: 7,   # left_elbow
            4: 8,   # right_elbow
            5: 9,   # left_wrist
            6: 10,  # right_wrist
            7: 11,  # left_hip
            8: 12,  # right_hip
            9: 13,  # left_knee
            10: 14, # right_knee
            11: 15, # left_ankle
            12: 16, # right_ankle
        }
        self.assertEqual(PENN_TO_COCO_BODY_MAP, expected_mappings)

        # Check names match anatomically
        for p_idx, c_idx in PENN_TO_COCO_BODY_MAP.items():
            self.assertEqual(PENN_JOINT_NAMES[p_idx], COCO_JOINT_NAMES[c_idx])

        # Verify COCO flip indexes swap left/right pairs
        self.assertEqual(COCO_FLIP_INDEXES[5], 6)
        self.assertEqual(COCO_FLIP_INDEXES[6], 5)
        self.assertEqual(COCO_FLIP_INDEXES[13], 14)
        self.assertEqual(COCO_FLIP_INDEXES[14], 13)

    def test_02_yolo_56_token_layout_and_normalization(self) -> None:
        """Verify YOLO pose label line formatting on non-square images with legal coordinates."""
        img_w, img_h = 640, 480
        bbox = [100.0, 50.0, 300.0, 450.0]  # xc=200, yc=250, bw=200, bh=400
        x = [200.0] * 13
        y = [250.0] * 13
        vis = [1] * 13

        line = convert_penn_frame_to_yolo(bbox, x, y, vis, img_w, img_h)
        tokens = line.split()

        # Token count must be exactly 56 (1 class + 4 bbox + 17*3 keypoints)
        self.assertEqual(len(tokens), 56)

        # Class ID must be 0
        self.assertEqual(tokens[0], "0")

        # Check normalized box coordinates
        xc = float(tokens[1])
        yc = float(tokens[2])
        bw = float(tokens[3])
        bh = float(tokens[4])

        self.assertAlmostEqual(xc, 200.0 / 640.0, places=5)
        self.assertAlmostEqual(yc, 250.0 / 480.0, places=5)
        self.assertAlmostEqual(bw, 200.0 / 640.0, places=5)
        self.assertAlmostEqual(bh, 400.0 / 480.0, places=5)

        # Ensure all normalized tokens are strictly within [0.0, 1.0]
        for i in range(1, 5):
            val = float(tokens[i])
            self.assertTrue(0.0 <= val <= 1.0)

    def test_03_missing_face_masks_and_occlusion_semantics(self) -> None:
        """Verify facial joints are unannotated (v=0) and unavailable body joints are masked despite finite coordinates."""
        img_w, img_h = 500, 500
        bbox = [50.0, 50.0, 450.0, 450.0]

        x = [250.0] * 13
        y = [250.0] * 13
        # Left joints visible (1), right joints occluded (0)
        vis = [1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]

        line = convert_penn_frame_to_yolo(bbox, x, y, vis, img_w, img_h)
        tokens = line.split()

        # Check facial keypoints (COCO indices 0, 1, 2, 3, 4)
        for c_idx in range(5):
            k_start = 5 + c_idx * 3
            kx = float(tokens[k_start])
            ky = float(tokens[k_start + 1])
            kv = int(tokens[k_start + 2])
            self.assertEqual(kx, 0.0)
            self.assertEqual(ky, 0.0)
            self.assertEqual(kv, 0, f"COCO facial joint {c_idx} must have visibility 0")

        # Check left shoulder (COCO 5, Penn 1): visible -> v=2
        k_start = 5 + 5 * 3
        self.assertEqual(int(tokens[k_start + 2]), 2)

        # Check right shoulder (COCO 6, Penn 2): unavailable -> all zero
        k_start = 5 + 6 * 3
        self.assertEqual(tokens[k_start:k_start + 3], ["0.000000", "0.000000", "0"])

    def test_04_bounds_and_out_of_frame_points(self) -> None:
        """Verify out-of-frame points are masked with (0,0,0) and invalid boxes are rejected."""
        img_w, img_h = 640, 480
        valid_bbox = [10.0, 10.0, 100.0, 100.0]
        valid_coords = [50.0] * 13
        valid_vis = [1] * 13

        # 1. Direct call with x=700 on 640-wide image:
        # Penn joint 1 (left_shoulder, COCO 5) placed at x=700.0 (outside 640 boundary)
        out_of_bounds_x = list(valid_coords)
        out_of_bounds_x[1] = 700.0

        line = convert_penn_frame_to_yolo(valid_bbox, out_of_bounds_x, valid_coords, valid_vis, img_w, img_h)
        tokens = line.split()

        # Left shoulder is at COCO index 5 -> token offset: 5 + 5*3 = 20
        c5_x = float(tokens[20])
        c5_y = float(tokens[21])
        c5_v = int(tokens[22])

        # Must NOT export 700/640=1.093750, must NOT clamp to 1.0; must explicitly mask with 0.0, 0.0, 0
        self.assertEqual(c5_x, 0.0)
        self.assertEqual(c5_y, 0.0)
        self.assertEqual(c5_v, 0)

        # 2. Bounding box exceeding image bounds: must raise ValueError
        with self.assertRaises(ValueError):
            convert_penn_frame_to_yolo([0.0, 0.0, 700.0, 400.0], valid_coords, valid_coords, valid_vis, img_w, img_h)

        # 3. Inverted bounding box: must raise ValueError
        with self.assertRaises(ValueError):
            convert_penn_frame_to_yolo([100.0, 10.0, 50.0, 100.0], valid_coords, valid_coords, valid_vis, img_w, img_h)

        # 4. Non-finite keypoint: must raise ValueError
        bad_x = list(valid_coords)
        bad_x[5] = float("nan")
        with self.assertRaises(ValueError):
            convert_penn_frame_to_yolo(valid_bbox, bad_x, valid_coords, valid_vis, img_w, img_h)

    def test_05_visibility_strict_validation(self) -> None:
        """Verify visibility values outside {0, 1} are strictly rejected without blind casting."""
        import scipy.io as scio

        with tempfile.TemporaryDirectory() as tmpdir:
            # 1. Invalid visibility value -1
            mat_bad1 = _make_legal_synthetic_mat(bad_vis_val=-1)
            p1 = os.path.join(tmpdir, "bad1.mat")
            scio.savemat(p1, mat_bad1)
            with self.assertRaises(ValueError):
                load_penn_mat(p1)

            # 2. Invalid visibility value 255
            mat_bad2 = _make_legal_synthetic_mat(bad_vis_val=255)
            p2 = os.path.join(tmpdir, "bad2.mat")
            scio.savemat(p2, mat_bad2)
            with self.assertRaises(ValueError):
                load_penn_mat(p2)

    def test_06_load_penn_mat_and_no_manufactured_bboxes(self) -> None:
        """Verify load_penn_mat rejects mismatched bbox shapes and does not manufacture rows."""
        import scipy.io as scio

        with tempfile.TemporaryDirectory() as tmpdir:
            # 1. Normal valid synthetic mat
            mat_dict = _make_legal_synthetic_mat(nframes=5, is_train=1)
            normal_path = os.path.join(tmpdir, "0001.mat")
            scio.savemat(normal_path, mat_dict)

            annot = load_penn_mat(normal_path)
            self.assertEqual(annot.seq_id, "0001")
            self.assertEqual(annot.action, "squats")
            self.assertEqual(annot.nframes, 5)
            self.assertEqual(annot.is_train, 1)
            self.assertEqual(annot.bboxes.shape, (5, 4))

            # 2. Bbox short by one row: must be REJECTED, not extended by duplicating rows
            mat_quirk = _make_legal_synthetic_mat(nframes=5, bbox_shape="short_by_one")
            quirk_path = os.path.join(tmpdir, "1865.mat")
            scio.savemat(quirk_path, mat_quirk)

            with self.assertRaises(ValueError) as cm:
                load_penn_mat(quirk_path)
            self.assertIn("duplicating or manufacturing boxes is prohibited", str(cm.exception))

            # 3. Missing key
            mat_bad = _make_legal_synthetic_mat(corrupt_key="visibility")
            bad_path = os.path.join(tmpdir, "bad.mat")
            scio.savemat(bad_path, mat_bad)
            with self.assertRaises(KeyError):
                load_penn_mat(bad_path)

            # 4. Nonexistent file
            with self.assertRaises(FileNotFoundError):
                load_penn_mat(os.path.join(tmpdir, "nonexistent.mat"))

    def test_07_sequence_splits_and_single_training_sequence(self) -> None:
        """Verify sequence-level isolation and ensure train is never empty even with 1 sequence."""
        # 1. 10 sequences: 7 train, 3 test
        annotations: List[PennActionAnnotation] = []
        for i in range(10):
            seq_id = f"{i+1:04d}"
            is_train = 1 if i < 7 else 0
            annotations.append(
                PennActionAnnotation(
                    seq_id=seq_id,
                    action="squats",
                    nframes=5,
                    is_train=is_train,
                    pose_view="side",
                    bboxes=np.zeros((5, 4)),
                    x=np.zeros((5, 13)),
                    y=np.zeros((5, 13)),
                    visibility=np.zeros((5, 13), dtype=np.int64),
                )
            )

        splits = generate_sequence_splits(annotations, val_fraction=0.3, seed=42)
        self.assertEqual(splits["test"], ["0008", "0009", "0010"])
        self.assertTrue(len(splits["train"]) > 0)
        self.assertTrue(len(splits["val"]) > 0)

        # Disjointness
        s_tr = set(splits["train"])
        s_va = set(splits["val"])
        s_te = set(splits["test"])
        self.assertTrue(s_tr.isdisjoint(s_va))
        self.assertTrue(s_tr.isdisjoint(s_te))
        self.assertTrue(s_va.isdisjoint(s_te))

        # 2. Corner case: exactly 1 native training sequence must yield non-empty train
        single_train = [annotations[0]]  # sequence 0001 (is_train=1)
        single_splits = generate_sequence_splits(single_train, val_fraction=0.2)
        self.assertEqual(single_splits["train"], ["0001"])
        self.assertEqual(single_splits["val"], [])

        # 3. Corner case: 0 native training sequences must raise ValueError
        zero_train = [annotations[7]]  # sequence 0008 (is_train=0)
        with self.assertRaises(ValueError):
            generate_sequence_splits(zero_train)

    def test_08_split_manifest_validation(self) -> None:
        """Verify explicit split manifest validation, native test protection, and duplicate checks."""
        # Setup mock annotations
        annots = {
            "0001": PennActionAnnotation("0001", "squats", 5, 1, "side", np.zeros((5, 4)), np.zeros((5, 13)), np.zeros((5, 13)), np.zeros((5, 13), dtype=np.int64)),
            "0002": PennActionAnnotation("0002", "squats", 5, 1, "side", np.zeros((5, 4)), np.zeros((5, 13)), np.zeros((5, 13)), np.zeros((5, 13), dtype=np.int64)),
            "0003": PennActionAnnotation("0003", "squats", 5, 0, "side", np.zeros((5, 4)), np.zeros((5, 13)), np.zeros((5, 13)), np.zeros((5, 13), dtype=np.int64)),  # native test
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            # 1. Missing manifest file
            with self.assertRaises(FileNotFoundError):
                validate_split_manifest(os.path.join(tmpdir, "missing.json"), annots)

            # 2. Invalid split keys (e.g. 'extra')
            p_bad_key = os.path.join(tmpdir, "bad_key.json")
            with open(p_bad_key, "w") as f:
                json.dump({"train": ["0001"], "extra": ["0002"]}, f)
            with self.assertRaises(ValueError):
                validate_split_manifest(p_bad_key, annots)

            # 3. Unknown sequence ID
            p_unk = os.path.join(tmpdir, "unknown.json")
            with open(p_unk, "w") as f:
                json.dump({"train": ["0001"], "val": ["9999"]}, f)
            with self.assertRaises(ValueError):
                validate_split_manifest(p_unk, annots)

            # 4. Duplicate sequence across splits
            p_dup = os.path.join(tmpdir, "dup.json")
            with open(p_dup, "w") as f:
                json.dump({"train": ["0001"], "val": ["0001"]}, f)
            with self.assertRaises(ValueError):
                validate_split_manifest(p_dup, annots)

            # 5. Native test sequence (0003) in train: MUST be rejected
            p_leak = os.path.join(tmpdir, "leak.json")
            with open(p_leak, "w") as f:
                json.dump({"train": ["0003"], "test": ["0001"]}, f)
            with self.assertRaises(ValueError):
                validate_split_manifest(p_leak, annots)

            # 6. Empty train partition: MUST be rejected
            p_empty_tr = os.path.join(tmpdir, "empty_tr.json")
            with open(p_empty_tr, "w") as f:
                json.dump({"train": [], "test": ["0003"]}, f)
            with self.assertRaises(ValueError):
                validate_split_manifest(p_empty_tr, annots)

            # 7. Valid manifest
            p_valid = os.path.join(tmpdir, "valid.json")
            with open(p_valid, "w") as f:
                json.dump({"train": ["0001"], "val": ["0002"], "test": ["0003"]}, f)
            val_splits = validate_split_manifest(p_valid, annots)
            self.assertEqual(val_splits["train"], ["0001"])
            self.assertEqual(val_splits["val"], ["0002"])
            self.assertEqual(val_splits["test"], ["0003"])

    def test_09_unsafe_output_overlap_rejection(self) -> None:
        """Verify safety guards against identical/overlapping directories and nonempty outputs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            src_root = os.path.join(tmpdir, "source")
            os.makedirs(os.path.join(src_root, "labels"))
            os.makedirs(os.path.join(src_root, "frames"))

            # 1. Output identical to source
            with self.assertRaises(ValueError):
                run_conversion(source_penn_dir=src_root, output_dir=src_root)

            # 2. Output inside source
            nested_out = os.path.join(src_root, "output")
            with self.assertRaises(ValueError):
                run_conversion(source_penn_dir=src_root, output_dir=nested_out)

            # 3. Source inside output
            with self.assertRaises(ValueError):
                run_conversion(source_penn_dir=src_root, output_dir=tmpdir)

            # 4. Non-empty output directory refusal (non-destructive)
            nonempty_out = os.path.join(tmpdir, "nonempty_out")
            os.makedirs(nonempty_out)
            canary_file = os.path.join(nonempty_out, "user_data.txt")
            with open(canary_file, "w") as f:
                f.write("precious user data")

            with self.assertRaises(FileExistsError):
                run_conversion(source_penn_dir=src_root, output_dir=nonempty_out, overwrite=True)

            # Canary file must NOT have been deleted
            self.assertTrue(os.path.isfile(canary_file))

    def test_10_stride_and_preflight_validation(self) -> None:
        """Verify positive stride validation and preflight checks for missing frames/images."""
        import scipy.io as scio

        with tempfile.TemporaryDirectory() as tmpdir:
            src_root = os.path.join(tmpdir, "source")
            out_root = os.path.join(tmpdir, "fresh_out")
            labels_dir = os.path.join(src_root, "labels")
            frames_dir = os.path.join(src_root, "frames")
            os.makedirs(labels_dir)
            os.makedirs(frames_dir)

            mat_dict = _make_legal_synthetic_mat(action="squats", nframes=3, is_train=1)
            scio.savemat(os.path.join(labels_dir, "0001.mat"), mat_dict)

            # 1. Invalid stride
            with self.assertRaises(ValueError):
                run_conversion(source_penn_dir=src_root, output_dir=out_root, frame_stride=0)
            with self.assertRaises(ValueError):
                run_conversion(source_penn_dir=src_root, output_dir=out_root, frame_stride=-2)

            # 2. Missing frames directory for sequence 0001
            with self.assertRaises(FileNotFoundError):
                run_conversion(source_penn_dir=src_root, output_dir=out_root, frame_stride=1)

            # Create frames directory but omit frame image files
            os.makedirs(os.path.join(frames_dir, "0001"))
            with self.assertRaises(FileNotFoundError):
                run_conversion(source_penn_dir=src_root, output_dir=out_root, frame_stride=1)

    def test_11_legal_synthetic_fixtures_end_to_end(self) -> None:
        """Verify end-to-end conversion on strictly legal synthetic fixtures."""
        import scipy.io as scio

        with tempfile.TemporaryDirectory() as tmpdir:
            src_root = os.path.join(tmpdir, "penn_synthetic")
            out_root = os.path.join(tmpdir, "yolo_output")

            labels_dir = os.path.join(src_root, "labels")
            frames_dir = os.path.join(src_root, "frames")
            os.makedirs(labels_dir)
            os.makedirs(frames_dir)

            # Create 3 synthetic sequences: 2 squats (1 train, 1 test), 1 tennis
            for seq_int, action, is_tr in [(1, "squats", 1), (2, "squats", 0), (3, "tennis", 1)]:
                seq_id = f"{seq_int:04d}"
                mat_dict = _make_legal_synthetic_mat(
                    action=action, nframes=4, is_train=is_tr, img_w=320, img_h=240
                )
                scio.savemat(os.path.join(labels_dir, f"{seq_id}.mat"), mat_dict)

                seq_frames = os.path.join(frames_dir, seq_id)
                os.makedirs(seq_frames)
                for f in range(4):
                    img = np.zeros((240, 320, 3), dtype=np.uint8)
                    cv2.imwrite(os.path.join(seq_frames, f"{f+1:06d}.jpg"), img)

            stats = run_conversion(
                source_penn_dir=src_root,
                output_dir=out_root,
                action_filter="squats",
                frame_stride=2,
            )

            self.assertEqual(stats["total_matching_sequences"], 2)
            self.assertTrue(os.path.isfile(os.path.join(out_root, "data.yaml")))
            self.assertTrue(os.path.isfile(os.path.join(out_root, "split_manifest.json")))

            # Check that train has sequence 0001 and test has sequence 0002
            self.assertEqual(stats["split_counts"]["train"], 1)
            self.assertEqual(stats["split_counts"]["test"], 1)

            # Check generated label files: all tokens must be valid
            lbl_dir = os.path.join(out_root, "labels", "train")
            for lf in os.listdir(lbl_dir):
                with open(os.path.join(lbl_dir, lf), "r") as f:
                    content = f.read().strip()
                tokens = content.split()
                self.assertEqual(len(tokens), 56)
                # Check bounding box in [0, 1]
                for t in tokens[1:5]:
                    val = float(t)
                    self.assertTrue(0.0 <= val <= 1.0)

    def test_12_render_overlay(self) -> None:
        """Verify render_penn_overlay runs without crashing on legal coordinates."""
        img = np.zeros((300, 400, 3), dtype=np.uint8)
        bbox = [50.0, 50.0, 250.0, 250.0]
        x = [150.0] * 13
        y = [150.0] * 13
        vis = [1] * 13

        rendered = render_penn_overlay(img, bbox, x, y, vis, title="Synthetic Test")
        self.assertEqual(rendered.shape, (300, 400, 3))
        self.assertEqual(rendered.dtype, np.uint8)


    def test_source_origin_and_exact_sequence_ids(self):
        from scipy.io import savemat
        with tempfile.TemporaryDirectory() as tmp:
            data = _make_legal_synthetic_mat()
            source = Path(tmp) / "0001.mat"
            savemat(source, data)
            parsed = load_penn_mat(str(source))
            np.testing.assert_allclose(parsed.x, data["x"] - 1)
            np.testing.assert_allclose(parsed.y, data["y"] - 1)
            np.testing.assert_allclose(parsed.bboxes, data["bbox"] - 1)
            wrong = Path(tmp) / "0001_extra.mat"
            savemat(wrong, data)
            with self.assertRaises(ValueError): load_penn_mat(str(wrong))

    def test_invalid_direct_visibility_and_fractional_flags(self):
        from scipy.io import savemat
        vis = [1] * 13
        vis[1] = 0.5
        with self.assertRaises(ValueError):
            convert_penn_frame_to_yolo([10, 10, 100, 100], [50]*13, [50]*13, vis, 320, 240)
        with tempfile.TemporaryDirectory() as tmp:
            for flag in (2, 0.5):
                data = _make_legal_synthetic_mat()
                data["train"] = np.array([[flag]])
                source = Path(tmp) / "0001.mat"
                savemat(source, data)
                with self.assertRaises(ValueError):load_penn_mat(str(source))

    def test_real_action_alias_preflight_and_lossless_copy(self):
        from scipy.io import savemat
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "source"
            (source / "labels").mkdir(parents=True)
            (source / "frames" / "0001").mkdir(parents=True)
            data = _make_legal_synthetic_mat(action="squat", nframes=1)
            mat = source / "labels" / "0001.mat"
            frame = source / "frames" / "0001" / "000001.jpg"
            cv2.imwrite(str(frame), np.zeros((240, 320, 3), dtype=np.uint8))
            # Illegal label must fail before creating output.
            bad = dict(data)
            bad["bbox"] = np.array([[1., 1., 400., 240.]])
            savemat(mat, bad)
            output = Path(tmp) / "output"
            with self.assertRaises(ValueError):run_conversion(str(source), str(output))
            self.assertFalse(output.exists())
            savemat(mat, data)
            stats = run_conversion(str(source), str(output))
            exported = output / "images" / "train" / "0001_000001.jpg"
            self.assertEqual(frame.read_bytes(), exported.read_bytes())
            self.assertFalse(stats["trainable_split"])
            self.assertTrue((output / "conversion_receipt.json").exists())

    def test_output_link_rejected_before_resolution(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "source"
            (source / "labels").mkdir(parents=True)
            (source / "frames").mkdir()
            target = Path(tmp) / "target"
            target.mkdir()
            link = Path(tmp) / "link"
            try:link.symlink_to(target, target_is_directory=True)
            except OSError:self.skipTest("Symlink creation unavailable")
            with self.assertRaisesRegex(ValueError, "link or reparse"):
                run_conversion(str(source), str(link / "output"))
            self.assertFalse((target / "output").exists())


if __name__ == "__main__":
    unittest.main()
