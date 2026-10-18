# 2D gym pose analyser

A Python prototype for analyzing single-person, side-view squat exercise videos into projected knee angle time series, repetition counts, and annotated video overlays using pretrained YOLO pose estimation.

## Architecture and installation

The package is structured as a standalone module inside `gym_pose/`:

```
gym_pose/
├── .gitignore
├── README.md
├── requirements.txt
├── gym_analyser/
│   ├── __init__.py
│   ├── __main__.py      (CLI entry point)
│   ├── geometry.py      (Pure vector math and keypoint evaluation)
│   ├── counter.py       (Hysteresis repetition state machine)
│   └── pipeline.py      (Video I/O, YOLO tracking, plotting, CSV/JSON export)
└── tests/
    ├── __init__.py
    ├── test_geometry.py (Deterministic geometric validation)
    ├── test_counter.py  (Deterministic state machine contract tests)
    └── test_pipeline.py (Formatting, blank outputs, and adapter validation)
```

The core geometry and counting logic use Python standard library components only. Inference and visualization dependencies (`ultralytics`, `opencv-python`, `matplotlib`, `torch`, `lap`) are imported only during CLI execution.

### Environment setup on Windows

From the workspace root, change directory into `gym_pose/`:

```powershell
cd gym_pose
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

Installing PyTorch with the CPU index avoids accidentally downloading large CUDA runtime binaries on machines without GPU requirements. Pinned dependencies include `lap==0.5.13` for reproducible ByteTrack association.

### Running unit tests

Deterministic unit and adapter tests run locally without downloading models:

```powershell
cd gym_pose
.\.venv\Scripts\python.exe -m unittest discover -s tests -p "test_*.py" -v
```

## Repetition definition and state transitions

A valid squat repetition requires a completed, continuous cycle:

1. Initial state: The counter begins in `unknown`. An opening rise from a crouch is not credited.
2. Arming phase: The counter requires `stable_frames` (default 3) consecutive observations of standing posture (knee angle >= 150.0 degrees) on a single side to arm into `standing`.
3. Descent phase: From `standing`, the subject must reach flexion depth, maintaining knee angle <= 110.0 degrees on the armed side for at least `stable_frames` consecutive frames to transition to `lowered`.
4. Ascent phase: From `lowered`, the subject must return to extension, maintaining knee angle >= 150.0 degrees for at least `stable_frames` consecutive frames to transition back to `standing`.
5. Repetition increment: Repetition count increments by 1 only upon returning to `standing`.
6. Side locking and reselection: During descent and while lowered, the counter locks onto the active side. After completing a cycle, the returning side remains armed for consecutive reps. If the subject turns or switches sides, the newly selected side must provide its own `stable_frames` consecutive standing observations before its descent can arm a new repetition.
7. Timeout and gaps: If tracking is lost or ambiguous for longer than `max_gap_seconds` (default 0.5s), the partial cycle is abandoned and resets to `unknown`. Completed repetitions are preserved.

Partial descents, oscillations that do not meet stability thresholds, and ending in a crouch count zero.

## Configuration parameters

| Parameter | Default | Validation rule | Description |
| :--- | :--- | :--- | :--- |
| `--down-angle` | 110.0 | `0 < down < up < 180` | Knee flexion threshold in degrees. Angle must drop to or below this value to register descent. |
| `--up-angle` | 150.0 | `down < up < 180` | Knee extension threshold in degrees. Angle must reach or exceed this value to register standing. |
| `--min-confidence` | 0.3 | `0.0 <= conf <= 1.0` | Minimum confidence score required across hip, knee, and ankle keypoints. |
| `--stable-frames` | 3 | Integer `>= 1` | Number of consecutive frames required to confirm stage transitions. |
| `--max-gap` | 0.5 | `> 0.0` | Maximum time gap in seconds before partial repetition state is abandoned. |
| `--tracker` | `bytetrack.yaml` | Valid YAML tracker file | Tracking configuration file passed to Ultralytics. |
| `--device` | `cpu` | Valid device string | Compute device for inference (`cpu` or `cuda`). |
| `--overwrite` | False | Boolean flag | Clear and overwrite existing output directory artifacts. |

## Usage and exact Windows commands

All commands assume the current working directory is `gym_pose`:

### Display help

```powershell
cd gym_pose
.\.venv\Scripts\python.exe -m gym_analyser --help
```

### Process video footage

```powershell
cd gym_pose
.\.venv\Scripts\python.exe -m gym_analyser `
    --source test_data\synthetic_test.mp4 `
    --output-dir output\smoke_test_03 `
    --model models\yolo11n-pose.pt `
    --device cpu `
    --overwrite
```

## Output artifacts

The pipeline produces four artifacts in `--output-dir`:

1. `annotated.mp4`: Source video retaining original dimensions and framerate, overlaid with COCO skeleton lines, highlighted active limb segments, knee joint vertex circle, joint angle arc drawn via OpenCV ellipse, and the repetition counter HUD banner.
2. `angles.csv`: Per-frame tabular data with columns:
   `frame_index, video_time_s, track_id, selected_side, knee_angle_deg, confidence, valid, reason, stage, completed_reps`.
   Missing numeric values are blank strings, never zeros. Valid frames report the actual observed side even during arming.
3. `summary.json`: Run metadata including Python and package versions, configuration parameters, source declared frame count and duration, processed frame count and duration, valid observation count and fraction, completed repetition count, read status note, and explicit limitations.
4. `angle_plot.png`: Matplotlib time series plot of knee angles over video time. Missing and invalid frames appear as gaps in the curve rather than connected lines.

## Third-party credits

- Ultralytics YOLO Pose: `https://github.com/ultralytics/ultralytics` (AGPL-3.0). Informed model loading, COCO keypoint conventions, and tracker integration.
- Shalbulov exercise_counter: `https://github.com/Shalbulov/exercise_counter` (MIT intent). Informed confidence gating, independent leg evaluation, and return-to-extension counter trigger logic.

## Known limitations and remaining manual checks

1. 2D projected angles: Angles are computed from 2D pixel coordinates and represent planar projections, not calibrated 3D anatomical joint angles.
2. Subject orientation: Designed for side-view single-person recordings. Oblique or frontal angles foreshorten limb segments and distort angle measurements.
3. No automatic form correction: The system detects threshold crossings. It does not validate squat depth safety, back alignment, or medical correctness.
4. Manual validation requirement: Synthetic test videos verify output mechanics and file generation, not human pose squat counting accuracy. Verification against human-reviewed ground truth remains pending for untested clips.
