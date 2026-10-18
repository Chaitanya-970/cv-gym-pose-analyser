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

The core geometry and counting logic use Python standard library components only. Inference and visualization dependencies (`ultralytics`, `opencv-python`, `matplotlib`, `torch`) are imported only during CLI execution.

### Environment setup on Windows

Create a local virtual environment and install dependencies:

```powershell
python -m venv gym_pose\.venv
.\gym_pose\.venv\Scripts\Activate.ps1
pip install -r gym_pose\requirements.txt
```

### Running deterministic unit tests

Unit tests run without neural network models, GPUs, or third-party dependencies:

```powershell
.\gym_pose\.venv\Scripts\python.exe -m unittest discover -s gym_pose\tests -p "test_*.py" -v
```

## Full repetition definition

A valid squat repetition requires a completed, continuous cycle:

1. **Initial state:** The state machine starts in `UNKNOWN`. It does not credit an opening upward motion if the video begins at the bottom of a crouch.
2. **Arming phase:** The counter requires 3 consecutive observations of standing posture (knee angle >= 150.0 degrees) to arm into `STANDING`.
3. **Descent phase:** From `STANDING`, the subject must reach or exceed flexion depth, maintaining knee angle <= 110.0 degrees for at least 3 consecutive frames to transition to `LOWERED`.
4. **Ascent phase:** From `LOWERED`, the subject must return to extension, maintaining knee angle >= 150.0 degrees for at least 3 consecutive frames to transition back to `STANDING`.
5. **Repetition increment:** The repetition count increments by exactly 1 only upon returning to `STANDING`.
6. **Side locking:** When first arming, the counter selects the side with higher detection confidence (left or right) and locks that side for the duration of the repetition cycle. If the locked side becomes occluded mid-cycle, the cycle cannot be completed by the opposing side. Reselection is permitted only between completed cycles after a fresh standing observation.
7. **Timeout and gaps:** If tracking is lost, occluded, or ambiguous for longer than `max_gap_seconds` (default 0.5s), the partial cycle is abandoned and resets to `UNKNOWN`. Completed repetitions are preserved.

Partial descents, oscillations that do not satisfy stability requirements, and ending crouched count zero.

## Configuration parameters

| Parameter | Default | Validation rule | Description |
| :--- | :--- | :--- | :--- |
| `--down-angle` | 110.0 | `0 < down < up < 180` | Knee flexion threshold in degrees. Angle must drop to or below this value to register descent. |
| `--up-angle` | 150.0 | `down < up < 180` | Knee extension threshold in degrees. Angle must reach or exceed this value to register standing. |
| `--min-confidence` | 0.3 | `0.0 <= conf <= 1.0` | Minimum confidence score required across hip, knee, and ankle keypoints. |
| `--stable-frames` | 3 | `>= 1` | Number of consecutive frames required to confirm stage transitions. |
| `--max-gap` | 0.5 | `> 0.0` | Maximum time gap in seconds before partial repetition state is abandoned. |
| `--tracker` | `bytetrack.yaml` | Valid YAML tracker file | Tracking configuration file passed to Ultralytics. |
| `--device` | `cpu` | Valid device string | Compute device for inference (`cpu` or `cuda`). |

## Usage and exact Windows commands

### Display help

```powershell
.\gym_pose\.venv\Scripts\python.exe -m gym_analyser --help
```

### Process a prerecorded video

```powershell
.\gym_pose\.venv\Scripts\python.exe -m gym_analyser `
    --source path\to\input.mp4 `
    --output-dir path\to\output_dir `
    --model yolo11n-pose.pt `
    --device cpu `
    --down-angle 110.0 `
    --up-angle 150.0 `
    --min-confidence 0.3 `
    --stable-frames 3 `
    --max-gap 0.5 `
    --overwrite
```

## Output artifacts

The pipeline produces four artifacts in `--output-dir`:

1. `annotated.mp4`: Source video retaining original dimensions and framerate, overlaid with COCO skeleton lines, locked side knee angle arc, tracking ID, stage status, and repetition counter HUD banner.
2. `angles.csv`: Per-frame tabular data with columns:
   `frame_index, video_time_s, track_id, selected_side, knee_angle_deg, confidence, valid, reason, stage, completed_reps`.
   Missing numeric values are blank strings, never zeros.
3. `summary.json`: Run metadata including Python/package versions, configuration settings, source dimensions/FPS, processed frame count, valid observation fraction, completed repetition count, and explicit limitation notices.
4. `angle_plot.png`: Matplotlib time series plot of knee angles over video time. Missing and invalid frames appear as gaps in the curve rather than connected lines.

## Third-party credits

- **Ultralytics YOLO Pose:** `https://github.com/ultralytics/ultralytics` (AGPL-3.0). Informed model loading, COCO keypoint conventions, and tracker integration.
- **Shalbulov exercise_counter:** `https://github.com/Shalbulov/exercise_counter` (MIT intent). Informed confidence gating, independent leg evaluation, and return-to-extension counter trigger logic.

## Known limitations and remaining manual checks

1. **2D projected angles:** Angles are computed from 2D pixel coordinates and represent planar projections, not calibrated 3D anatomical joint angles.
2. **Subject orientation:** Designed for side-view single-person recordings. Oblique or frontal angles foreshorten limb segments and distort angle measurements.
3. **No automatic form correction:** The system detects threshold crossings. It does not validate squat depth safety, back alignment, or medical correctness.
4. **Manual validation requirement:** Reported repetition counts reflect algorithmic threshold crossings. Verification against human-reviewed ground truth remains pending for untested clips.
