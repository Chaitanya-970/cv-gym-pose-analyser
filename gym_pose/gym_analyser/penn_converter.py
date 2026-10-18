"""Penn Action dataset loader, semantic keypoint converter, and YOLO pose exporter.

Converts 13-joint Penn Action annotations into COCO-17 compatible YOLO pose labels,
mapping the 12 corresponding body joints and setting facial landmarks as unannotated.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
import shutil
import stat
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np


# Official 13 Penn Action joint names in exact array order
PENN_JOINT_NAMES = [
    "head",            # 0
    "left_shoulder",   # 1
    "right_shoulder",  # 2
    "left_elbow",      # 3
    "right_elbow",     # 4
    "left_wrist",      # 5
    "right_wrist",     # 6
    "left_hip",        # 7
    "right_hip",       # 8
    "left_knee",       # 9
    "right_knee",      # 10
    "left_ankle",      # 11
    "right_ankle",     # 12
]

# Official 17 COCO keypoint names in exact array order
COCO_JOINT_NAMES = [
    "nose",            # 0
    "left_eye",        # 1
    "right_eye",       # 2
    "left_ear",        # 3
    "right_ear",       # 4
    "left_shoulder",   # 5
    "right_shoulder",  # 6
    "left_elbow",      # 7
    "right_elbow",     # 8
    "left_wrist",      # 9
    "right_wrist",     # 10
    "left_hip",        # 11
    "right_hip",       # 12
    "left_knee",       # 13
    "right_knee",      # 14
    "left_ankle",      # 15
    "right_ankle",     # 16
]

# Verified semantic mapping: 12 shared body joints from Penn (0-indexed) to COCO (0-indexed).
# Penn joint 0 ('head') is intentionally excluded to avoid corrupting facial landmarks.
PENN_TO_COCO_BODY_MAP: Dict[int, int] = {
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

COCO_TO_PENN_BODY_MAP: Dict[int, int] = {
    coco_idx: penn_idx for penn_idx, coco_idx in PENN_TO_COCO_BODY_MAP.items()
}

# Standard COCO-17 horizontal flip mapping for data augmentation
COCO_FLIP_INDEXES: List[int] = [
    0,   # nose -> nose
    2,   # left_eye -> right_eye
    1,   # right_eye -> left_eye
    4,   # left_ear -> right_ear
    3,   # right_ear -> left_ear
    6,   # left_shoulder -> right_shoulder
    5,   # right_shoulder -> left_shoulder
    8,   # left_elbow -> right_elbow
    7,   # right_elbow -> left_elbow
    10,  # left_wrist -> right_wrist
    9,   # right_wrist -> left_wrist
    12,  # left_hip -> right_hip
    11,  # right_hip -> left_hip
    14,  # left_knee -> right_knee
    13,  # right_knee -> left_knee
    16,  # left_ankle -> right_ankle
    15,  # right_ankle -> left_ankle
]


@dataclass(frozen=True)
class PennActionAnnotation:
    """Parsed annotations for a single Penn Action sequence."""

    seq_id: str
    action: str
    nframes: int
    is_train: int  # 1: native train, -1 or 0: native test
    pose_view: str
    bboxes: np.ndarray       # shape: (nframes, 4) in [xmin, ymin, xmax, ymax]
    x: np.ndarray            # shape: (nframes, 13)
    y: np.ndarray            # shape: (nframes, 13)
    visibility: np.ndarray   # shape: (nframes, 13); 0 unavailable, 1 visible


def load_penn_mat(mat_path: str) -> PennActionAnnotation:
    """Load and validate a Penn Action MATLAB (.mat) annotation file.

    scipy.io is imported locally inside this function to ensure core modules
    and dependency-free tests run without requiring scipy.
    """
    if not os.path.isfile(mat_path):
        raise FileNotFoundError(f"Annotation file not found: {mat_path}")

    try:
        import scipy.io as scio
    except ImportError as e:
        raise ImportError(
            "scipy is required for parsing Penn Action .mat files. "
            "Install with 'pip install scipy'."
        ) from e

    try:
        data = scio.loadmat(mat_path)
    except Exception as e:
        raise ValueError(f"Malformed or unreadable .mat file {mat_path}: {e}") from e

    required_keys = ["action", "nframes", "x", "y", "visibility", "bbox", "train"]
    for k in required_keys:
        if k not in data:
            raise KeyError(f"Missing required key '{k}' in {mat_path}")

    # Extract sequence ID from filename
    seq_id = Path(mat_path).stem
    if re.fullmatch(r"[0-9]{4}", seq_id) is None:
        raise ValueError(f"Invalid Penn sequence filename: {mat_path}")

    # Parse scalar metadata
    def _parse_str(val: Any) -> str:
        if isinstance(val, np.ndarray):
            if val.size == 0:
                return ""
            item = val.flat[0]
            while isinstance(item, np.ndarray) and item.size > 0:
                item = item.flat[0]
            return str(item).strip()
        return str(val).strip()

    action = _parse_str(data["action"])
    def _integer_scalar(key: str) -> int:
        values = np.asarray(data[key])
        if values.size != 1:
            raise ValueError(f"{key} must be a scalar in {mat_path}")
        value = float(values.item())
        if not math.isfinite(value) or value != int(value):
            raise ValueError(f"{key} must be a finite integer in {mat_path}")
        return int(value)
    nframes = _integer_scalar("nframes")
    is_train = _integer_scalar("train")
    if is_train not in {-1, 0, 1}:
        raise ValueError(f"Unknown native split flag {is_train} in {mat_path}")
    pose_view = _parse_str(data["pose"]) if "pose" in data else "unknown"

    if nframes < 1:
        raise ValueError(f"Invalid frame count {nframes} in {mat_path}")

    # Extract arrays
    x = np.asarray(data["x"], dtype=np.float64)
    y = np.asarray(data["y"], dtype=np.float64)
    raw_vis = np.asarray(data["visibility"])
    bbox = np.asarray(data["bbox"], dtype=np.float64)

    # Validate coordinate array shapes
    if x.shape != (nframes, 13):
        raise ValueError(f"Unexpected x shape {x.shape}, expected ({nframes}, 13) in {mat_path}")
    if y.shape != (nframes, 13):
        raise ValueError(f"Unexpected y shape {y.shape}, expected ({nframes}, 13) in {mat_path}")
    if raw_vis.shape != (nframes, 13):
        raise ValueError(f"Unexpected visibility shape {raw_vis.shape}, expected ({nframes}, 13) in {mat_path}")

    # Validate visibility values: must strictly be in {0, 1}
    # Avoid blind casting to uint8 which can hide negative or unsupported values
    if not np.all(np.isin(raw_vis, [0, 1])):
        raise ValueError(
            f"Invalid visibility values in {mat_path}; all values must be 0 or 1, "
            f"got unsupported values"
        )
    vis = raw_vis.astype(np.int64)

    # Validate coordinate finiteness
    if not (np.all(np.isfinite(x)) and np.all(np.isfinite(y))):
        raise ValueError(f"Non-finite keypoint coordinates in {mat_path}")

    # Validate bbox shape strictly against nframes
    # Do NOT manufacture or duplicate last-row bounding boxes
    if bbox.shape != (nframes, 4):
        raise ValueError(
            f"Mismatched bbox shape {bbox.shape}, expected ({nframes}, 4) in {mat_path}. "
            f"Row count must match nframes; duplicating or manufacturing boxes is prohibited."
        )

    # Validate finiteness of bounding boxes
    if not np.all(np.isfinite(bbox)):
        raise ValueError(f"Non-finite bounding box values in {mat_path}")

    # Author CreatePointLightDisplay.m indexes pixels with sub2ind(dims,Y,X,T).
    # Translate those MATLAB one-based coordinates to OpenCV zero-based pixels.
    # Continuous box endpoint differences are preserved; do not invent inclusive +1 extents.
    x = x - 1.0
    y = y - 1.0
    bbox = bbox - 1.0
    return PennActionAnnotation(
        seq_id=seq_id,
        action=action,
        nframes=nframes,
        is_train=is_train,
        pose_view=pose_view,
        bboxes=bbox,
        x=x,
        y=y,
        visibility=vis,
    )


def convert_penn_frame_to_yolo(
    bbox_xyxy: Sequence[float],
    x_coords: Sequence[float],
    y_coords: Sequence[float],
    visibility: Sequence[int],
    img_width: int,
    img_height: int,
) -> str:
    """Convert a single Penn Action frame annotation to a 56-token YOLO pose text line.

    Output format:
    class_id x_center y_center width height kpt1_x kpt1_y kpt1_v ... kpt17_x kpt17_y kpt17_v
    Inputs use zero-based image coordinates; loader converts MATLAB origins.
    All spatial coordinates and box dimensions are strictly normalized to [0.0, 1.0].
    Visibility=0 is unavailable and always masked: source coordinates can be placeholders.

    Policy for out-of-frame / truncated keypoints:
    Points outside [0, img_width] or [0, img_height] are explicitly masked with
    coordinates (0.000000, 0.000000) and visibility v=0 to avoid emitting invalid
    normalized coordinates (> 1.0 or < 0.0) or fabricating border points.
    """
    if any(isinstance(v, bool) or not isinstance(v, (int, np.integer)) or v <= 0 for v in (img_width, img_height)):
        raise ValueError(f"Invalid image dimensions: {img_width}x{img_height}")

    if len(bbox_xyxy) != 4:
        raise ValueError(f"Expected 4 bbox elements, got {len(bbox_xyxy)}")
    if len(x_coords) != 13 or len(y_coords) != 13 or len(visibility) != 13:
        raise ValueError(
            f"Expected 13 joints, got x={len(x_coords)}, y={len(y_coords)}, vis={len(visibility)}"
        )

    if not np.all(np.isin(np.asarray(visibility), [0, 1])):
        raise ValueError("Penn visibility must contain only 0 or 1")

    xmin, ymin, xmax, ymax = (float(v) for v in bbox_xyxy)
    if not (math.isfinite(xmin) and math.isfinite(ymin) and math.isfinite(xmax) and math.isfinite(ymax)):
        raise ValueError(f"Non-finite bbox coordinates: {bbox_xyxy}")
    if xmax <= xmin or ymax <= ymin:
        raise ValueError(f"Degenerate or inverted bbox: [{xmin}, {ymin}, {xmax}, {ymax}]")

    # Bounding boxes must lie within image extent
    if xmin < 0.0 or ymin < 0.0 or xmax > float(img_width) or ymax > float(img_height):
        raise ValueError(
            f"Bounding box [{xmin}, {ymin}, {xmax}, {ymax}] exceeds image boundaries "
            f"{img_width}x{img_height}"
        )

    # Compute normalized bounding box center and size (guaranteed in [0.0, 1.0])
    xc = ((xmin + xmax) / 2.0) / img_width
    yc = ((ymin + ymax) / 2.0) / img_height
    bw = (xmax - xmin) / img_width
    bh = (ymax - ymin) / img_height

    # Construct the 17 COCO keypoints
    tokens: List[str] = [
        "0",  # class 0: person
        f"{xc:.6f}",
        f"{yc:.6f}",
        f"{bw:.6f}",
        f"{bh:.6f}",
    ]

    for coco_idx in range(17):
        if coco_idx < 5:
            # Indices 0-4 (nose, left_eye, right_eye, left_ear, right_ear) are unannotated in Penn Action.
            # Explicitly set to (0.000000, 0.000000, 0) to mask them from spatial loss.
            tokens.extend(["0.000000", "0.000000", "0"])
        else:
            penn_idx = COCO_TO_PENN_BODY_MAP[coco_idx]
            px = float(x_coords[penn_idx])
            py = float(y_coords[penn_idx])
            pvis = int(visibility[penn_idx])

            if not (math.isfinite(px) and math.isfinite(py)):
                raise ValueError(
                    f"Non-finite keypoint coordinate at Penn joint {penn_idx} "
                    f"({PENN_JOINT_NAMES[penn_idx]}): ({px}, {py})"
                )

            # Check if keypoint lies within visible image frame boundaries
            in_bounds = (0.0 <= px <= float(img_width)) and (0.0 <= py <= float(img_height))

            if in_bounds and pvis == 1:
                norm_x = px / img_width
                norm_y = py / img_height
                # Original author tools discard visibility=0; those coordinates can be placeholders.
                v_flag = 2
                tokens.extend([f"{norm_x:.6f}", f"{norm_y:.6f}", str(v_flag)])
            else:
                # Unavailable (visibility=0) or out-of-frame coordinates:
                # Do NOT export coordinates outside [0, 1] (e.g. > 1.0) and do NOT clamp to borders.
                # Explicitly mask out-of-frame keypoints with (0.000000, 0.000000) and v=0.
                tokens.extend(["0.000000", "0.000000", "0"])

    line = " ".join(tokens)
    token_count = len(tokens)
    if token_count != 56:
        raise ValueError(f"Generated line has {token_count} tokens, expected 56")

    return line


def render_penn_overlay(
    image: np.ndarray,
    bbox_xyxy: Sequence[float],
    x_coords: Sequence[float],
    y_coords: Sequence[float],
    visibility: Sequence[int],
    title: str = "",
) -> np.ndarray:
    """Render Penn Action ground-truth annotations and mapped COCO skeleton onto an image."""
    canvas = image.copy()
    h, w = canvas.shape[:2]

    # Draw bounding box
    xmin, ymin, xmax, ymax = (int(round(v)) for v in bbox_xyxy)
    cv2.rectangle(canvas, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)

    # Skeleton connectivity among Penn body joints
    penn_skeleton = [
        (1, 3),   # left_shoulder -> left_elbow
        (3, 5),   # left_elbow -> left_wrist
        (2, 4),   # right_shoulder -> right_elbow
        (4, 6),   # right_elbow -> right_wrist
        (1, 2),   # left_shoulder -> right_shoulder
        (1, 7),   # left_shoulder -> left_hip
        (2, 8),   # right_shoulder -> right_hip
        (7, 8),   # left_hip -> right_hip
        (7, 9),   # left_hip -> left_knee
        (9, 11),  # left_knee -> left_ankle
        (8, 10),  # right_hip -> right_knee
        (10, 12), # right_knee -> right_ankle
    ]

    for p1, p2 in penn_skeleton:
        if visibility[p1] == 1 and visibility[p2] == 1:
            pt1 = (int(round(x_coords[p1])), int(round(y_coords[p1])))
            pt2 = (int(round(x_coords[p2])), int(round(y_coords[p2])))
            cv2.line(canvas, pt1, pt2, (255, 200, 0), 2, cv2.LINE_AA)

    # Draw individual joints
    for p_idx in range(13):
        if visibility[p_idx] != 1:
            continue
        px = int(round(x_coords[p_idx]))
        py = int(round(y_coords[p_idx]))
        vis = visibility[p_idx]

        # Green for visible, orange for occluded
        color = (0, 255, 0) if vis == 1 else (0, 140, 255)
        cv2.circle(canvas, (px, py), 4, color, -1, cv2.LINE_AA)
        cv2.circle(canvas, (px, py), 5, (0, 0, 0), 1, cv2.LINE_AA)

        # Joint text label
        name = PENN_JOINT_NAMES[p_idx]
        coco_idx = PENN_TO_COCO_BODY_MAP.get(p_idx, -1)
        tag = f"{name}(C:{coco_idx})" if coco_idx != -1 else f"{name}(head)"
        cv2.putText(
            canvas,
            tag,
            (px + 6, py - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    if title:
        cv2.putText(
            canvas,
            title,
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )

    return canvas


def generate_sequence_splits(
    squat_sequences: List[PennActionAnnotation],
    val_fraction: float = 0.2,
    seed: int = 42,
) -> Dict[str, List[str]]:
    """Partition squat sequences into sequence-disjoint train, validation, and test splits.

    Native test sequences (is_train != 1) are preserved strictly in 'test'.
    Validation is derived exclusively by partitioning native training sequences.
    Guarantees that 'train' is never empty when at least one native training sequence exists.
    """
    if not (0.0 < val_fraction < 1.0):
        raise ValueError(f"val_fraction must be in range (0.0, 1.0), got {val_fraction}")

    native_train = [s.seq_id for s in squat_sequences if s.is_train == 1]
    native_test = [s.seq_id for s in squat_sequences if s.is_train != 1]

    if not native_train:
        raise ValueError("Cannot generate splits: zero native training sequences available")

    rng = random.Random(seed)
    train_shuffled = list(native_train)
    rng.shuffle(train_shuffled)

    # If only 1 native training sequence is available, allocate to train so train is not empty
    if len(train_shuffled) == 1:
        val_seqs: List[str] = []
        train_seqs: List[str] = list(train_shuffled)
    else:
        val_count = max(1, int(round(len(train_shuffled) * val_fraction)))
        # Guard: val_count must not consume all training sequences
        if val_count >= len(train_shuffled):
            val_count = len(train_shuffled) - 1
        val_seqs = sorted(train_shuffled[:val_count])
        train_seqs = sorted(train_shuffled[val_count:])

    test_seqs = sorted(native_test)

    # Verify strict disjointness
    s_tr, s_va, s_te = set(train_seqs), set(val_seqs), set(test_seqs)
    if s_tr & s_va or s_tr & s_te or s_va & s_te:
        raise RuntimeError("Sequence overlap detected across dataset splits")

    return {
        "train": train_seqs,
        "val": val_seqs,
        "test": test_seqs,
    }


def validate_split_manifest(
    manifest_path: str,
    available_annotations: Dict[str, PennActionAnnotation],
) -> Dict[str, List[str]]:
    """Load and validate an explicit sequence split JSON manifest.

    Enforces:
    1. File existence (fails explicitly if missing; does not silently fallback).
    2. Allowed keys strictly subset of {'train', 'val', 'test'}.
    3. Every sequence ID must exist in available annotations.
    4. No duplicate sequence IDs within or across splits.
    5. Native test protection: sequences with is_train != 1 cannot enter 'train' or 'val'.
    6. Non-empty 'train' partition.
    """
    if not os.path.isfile(manifest_path):
        raise FileNotFoundError(f"Requested split manifest file not found: {manifest_path}")

    try:
        with open(manifest_path, "r", encoding="utf-8") as f:
            splits = json.load(f)
    except Exception as e:
        raise ValueError(f"Failed to parse split manifest JSON at {manifest_path}: {e}") from e

    if not isinstance(splits, dict):
        raise ValueError(f"Split manifest root must be a JSON object, got {type(splits)}")

    allowed_keys = {"train", "val", "test"}
    extra_keys = set(splits.keys()) - allowed_keys
    if extra_keys:
        raise ValueError(
            f"Invalid split keys {extra_keys} in manifest {manifest_path}; "
            f"allowed keys are 'train', 'val', 'test'"
        )

    seen_ids: Dict[str, str] = {}
    validated_splits: Dict[str, List[str]] = {k: [] for k in ["train", "val", "test"]}

    for split_name, seq_list in splits.items():
        if not isinstance(seq_list, list):
            raise ValueError(f"Split '{split_name}' must be a list of sequence IDs")

        for seq_id in seq_list:
            if not isinstance(seq_id, str) or not seq_id.strip():
                raise ValueError(f"Invalid sequence ID '{seq_id}' in split '{split_name}'")
            seq_id_clean = seq_id.strip()

            # Check for unknown sequence ID
            if seq_id_clean not in available_annotations:
                raise ValueError(
                    f"Unknown sequence ID '{seq_id_clean}' in split manifest '{manifest_path}'; "
                    f"sequence not found among matching dataset annotations"
                )

            # Check for duplicate sequence ID
            if seq_id_clean in seen_ids:
                prior_split = seen_ids[seq_id_clean]
                raise ValueError(
                    f"Duplicate sequence ID '{seq_id_clean}' in split manifest; "
                    f"already assigned to '{prior_split}', cannot re-add to '{split_name}'"
                )
            seen_ids[seq_id_clean] = split_name

            # Check native test protection
            annot = available_annotations[seq_id_clean]
            if split_name in ("train", "val") and annot.is_train != 1:
                raise ValueError(
                    f"Native test sequence '{seq_id_clean}' (is_train={annot.is_train}) "
                    f"cannot enter '{split_name}' partition in {manifest_path}. "
                    f"Native test sequences must be placed exclusively in 'test'."
                )

            validated_splits[split_name].append(seq_id_clean)

    if not validated_splits["train"]:
        raise ValueError(
            f"Split manifest {manifest_path} defines an empty 'train' partition. "
            f"A trainable pilot requires at least one sequence in 'train'."
        )

    return validated_splits


def write_yolo_data_yaml(output_dir: str, dataset_name: str = "penn_action_squats") -> str:
    """Generate official YOLO pose data.yaml configuration."""
    yaml_path = os.path.join(output_dir, "data.yaml")
    abs_out = Path(output_dir).resolve().as_posix()

    content = f"""# YOLO Pose configuration for {dataset_name}
# Automatically generated by gym_pose.gym_analyser.penn_converter

path: {abs_out}
train: images/train
val: images/val
test: images/test

# Keypoint layout: 17 joints, 3 dimensions [x, y, visibility]
kpt_shape: [17, 3]

# Official COCO-17 horizontal flip mapping for keypoint augmentation
flip_idx: [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]

# Class definition
names:
  0: person
"""
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(content)

    return yaml_path


def run_conversion(
    source_penn_dir: str,
    output_dir: str,
    action_filter: str = "squats",
    split_manifest_path: Optional[str] = None,
    frame_stride: int = 5,
    overwrite: bool = False,
) -> Dict[str, Any]:
    """Execute end-to-end dataset conversion from Penn Action to YOLO Pose format.

    Safety controls:
    - Verifies frame_stride is a positive integer >= 1.
    - Prevents output directory from overlapping with, being inside, or being ancestor of source.
    - Blocks symbolic link or junction output directories.
    - Non-empty output directories are strictly refused to avoid recursive data deletion.
    - Performs complete preflight validation of all selected sequences, frames, and images
      before writing data to disk.
    """
    if isinstance(frame_stride, bool) or not isinstance(frame_stride, int) or frame_stride < 1:
        raise ValueError(f"frame_stride must be a positive integer >= 1, got {frame_stride}")

    source_path = Path(source_penn_dir).resolve()
    raw_output = Path(os.path.abspath(output_dir))
    for component in (raw_output, *raw_output.parents):
        if component.exists() or component.is_symlink():
            attributes = component.lstat()
            if stat.S_ISLNK(attributes.st_mode) or getattr(attributes, "st_file_attributes", 0) & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 1024):
                raise ValueError(f"Output path contains a link or reparse point: {component}")
    out_path = raw_output.resolve()
    labels_dir = source_path / "labels"
    frames_dir = source_path / "frames"

    if not labels_dir.is_dir():
        raise FileNotFoundError(f"Penn Action labels directory not found: {labels_dir}")
    if not frames_dir.is_dir():
        raise FileNotFoundError(f"Penn Action frames directory not found: {frames_dir}")

    # Guard against output overlapping with source
    if out_path == source_path:
        raise ValueError(f"Output directory cannot be identical to source directory: {out_path}")
    if out_path in source_path.parents:
        raise ValueError(f"Output directory cannot be an ancestor of source directory: {out_path}")
    if source_path in out_path.parents:
        raise ValueError(f"Output directory cannot be inside source directory: {out_path}")

    # Guard against symlinks/junctions
    if os.path.islink(str(out_path)):
        raise ValueError(f"Output directory cannot be a symbolic link or junction: {out_path}")

    # Refuse all non-empty output directories; do not recursively delete user directories
    if out_path.exists():
        has_items = any(out_path.iterdir())
        if has_items:
            raise FileExistsError(
                f"Output directory '{out_path}' exists and is not empty. "
                f"Refusing to write or delete existing data; please specify a fresh empty directory."
            )

    # Find and parse all relevant .mat annotation files
    mat_files = sorted(labels_dir.glob("*.mat"))
    print(f"Scanning {len(mat_files)} .mat files in {labels_dir}...")

    matching_annotations: List[PennActionAnnotation] = []
    skipped_receipts: List[Dict[str, str]] = []

    for mf in mat_files:
        try:
            annot = load_penn_mat(str(mf))
            requested_action = action_filter.lower() if action_filter is not None else None
            aliases = {"squats": "squat"}
            if requested_action is None or aliases.get(annot.action.lower(), annot.action.lower()) == aliases.get(requested_action, requested_action):
                matching_annotations.append(annot)
        except Exception as e:
            skipped_receipts.append({"file": mf.name, "reason": str(e)})

    if skipped_receipts:
        print(f"Skipped {len(skipped_receipts)} non-matching or unparseable .mat files.")

    print(f"Found {len(matching_annotations)} sequences matching action='{action_filter}'.")
    if not matching_annotations:
        raise ValueError(f"No sequences found for action '{action_filter}'")

    annot_by_seq = {a.seq_id: a for a in matching_annotations}
    if len(annot_by_seq) != len(matching_annotations):
        raise ValueError("Duplicate source sequence IDs")

    # Determine sequence splits
    if split_manifest_path is not None:
        splits = validate_split_manifest(split_manifest_path, annot_by_seq)
        print(f"Loaded and validated explicit split manifest from {split_manifest_path}")
    else:
        splits = generate_sequence_splits(matching_annotations, val_fraction=0.2, seed=42)

    # Build sequence to split lookup
    split_lookup: Dict[str, str] = {}
    for s_name, seq_list in splits.items():
        for seq_id in seq_list:
            split_lookup[seq_id] = s_name

    prepared_labels: Dict[Tuple[str, int], str] = {}

    # Preflight verification: validate all selected sequence frame directories and frame images
    # Fail preflight before writing any files if requested frames are missing or corrupt
    selected_frames_to_convert: List[Tuple[str, str, int, Path, PennActionAnnotation]] = []
    for seq_id, split_name in split_lookup.items():
        if seq_id not in annot_by_seq:
            raise ValueError(f"Selected sequence '{seq_id}' has no matching annotation")
        annot = annot_by_seq[seq_id]
        seq_frames_dir = frames_dir / seq_id

        if not seq_frames_dir.is_dir():
            raise FileNotFoundError(
                f"Frames directory missing for selected sequence '{seq_id}': {seq_frames_dir}"
            )

        for f_idx in range(0, annot.nframes, frame_stride):
            frame_num = f_idx + 1
            img_fname = f"{frame_num:06d}.jpg"
            src_img_path = seq_frames_dir / img_fname

            if not src_img_path.is_file():
                raise FileNotFoundError(
                    f"Frame image missing for sequence '{seq_id}' frame {frame_num}: {src_img_path}"
                )

            # Test-read image to verify decode and dimensions
            img = cv2.imread(str(src_img_path))
            if img is None:
                raise ValueError(
                    f"Failed to decode frame image for sequence '{seq_id}' frame {frame_num}: {src_img_path}"
                )
            ih, iw = img.shape[:2]
            if ih <= 0 or iw <= 0:
                raise ValueError(f"Invalid frame dimensions ({iw}x{ih}) in {src_img_path}")

            # Validate every serialized label before creating any output files.
            prepared_labels[(seq_id, f_idx)] = convert_penn_frame_to_yolo(
                annot.bboxes[f_idx], annot.x[f_idx], annot.y[f_idx], annot.visibility[f_idx], iw, ih
            )
            selected_frames_to_convert.append((seq_id, split_name, f_idx, src_img_path, annot))

    # All preflights passed: create output directories
    out_path.mkdir(parents=True, exist_ok=True)
    manifest_save = out_path / "split_manifest.json"
    with open(manifest_save, "w", encoding="utf-8") as f:
        json.dump(splits, f, indent=2)

    for split_name in ["train", "val", "test"]:
        (out_path / "images" / split_name).mkdir(parents=True, exist_ok=True)
        (out_path / "labels" / split_name).mkdir(parents=True, exist_ok=True)

    stats: Dict[str, Any] = {
        "action": action_filter,
        "total_matching_sequences": len(matching_annotations),
        "split_counts": {k: len(v) for k, v in splits.items()},
        "converted_frames": {"train": 0, "val": 0, "test": 0},
        "skipped_files": skipped_receipts,
    }

    # Execute conversion
    for seq_id, split_name, f_idx, src_img_path, annot in selected_frames_to_convert:
        frame_num = f_idx + 1
        yolo_line = prepared_labels[(seq_id, f_idx)]

        sample_stem = f"{seq_id}_{frame_num:06d}"
        dst_img = out_path / "images" / split_name / f"{sample_stem}.jpg"
        dst_lbl = out_path / "labels" / split_name / f"{sample_stem}.txt"

        # Copy image and write label file
        shutil.copyfile(src_img_path, dst_img)
        with open(dst_lbl, "w", encoding="utf-8") as lf:
            lf.write(yolo_line + "\n")

        stats["converted_frames"][split_name] += 1

    # Native split evidence travels with the export for independent runner checks.
    source_inventory = []
    for seq_id in sorted(split_lookup):
        annotation = annot_by_seq[seq_id]
        original_file = labels_dir / (seq_id + ".mat")
        source_inventory.append({
            "seq": seq_id, "train": annotation.is_train, "action": annotation.action,
            "nframes": annotation.nframes,
            "sha256": hashlib.sha256(original_file.read_bytes()).hexdigest(),
        })
    with (out_path / "source_inventory.json").open("w", encoding="utf-8") as inventory_file:
        json.dump(source_inventory, inventory_file, indent=2)

    # Write data.yaml
    yaml_path = write_yolo_data_yaml(str(out_path), dataset_name=f"penn_{action_filter}")
    stats["data_yaml"] = yaml_path
    stats["trainable_split"] = bool(splits["train"] and splits["val"])
    stats["visibility_policy"] = "visible_only; visibility=0 or out-of-frame => (0,0,0); face unavailable"
    with (out_path / "conversion_receipt.json").open("w", encoding="utf-8") as receipt:
        json.dump(stats, receipt, indent=2)

    print("\nConversion Summary:")
    print(f"  Total sequences: {stats['total_matching_sequences']}")
    print(f"  Split counts: {stats['split_counts']}")
    print(f"  Converted frames: {stats['converted_frames']}")
    print(f"  Configuration: {stats['data_yaml']}")

    return stats


def main() -> None:
    """CLI entry point for Penn Action dataset conversion."""
    parser = argparse.ArgumentParser(
        description="Convert Penn Action annotations into COCO-17 YOLO pose dataset."
    )
    parser.add_argument(
        "--source-dir",
        type=str,
        required=True,
        help="Path to Penn Action root containing frames/ and labels/ subdirectories",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory to save converted images, labels, and data.yaml",
    )
    parser.add_argument(
        "--action",
        type=str,
        default="squats",
        help="Action class to filter (default: squats)",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=5,
        help="Frame sampling stride (default: 5 frames)",
    )
    parser.add_argument(
        "--split-manifest",
        type=str,
        default=None,
        help="Optional path to explicit sequence split JSON",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Deprecated compatibility flag; nonempty outputs are always refused",
    )

    args = parser.parse_args()

    try:
        run_conversion(
            source_penn_dir=args.source_dir,
            output_dir=args.output_dir,
            action_filter=args.action,
            split_manifest_path=args.split_manifest,
            frame_stride=args.stride,
            overwrite=args.overwrite,
        )
    except Exception as e:
        print(f"Conversion failed: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
