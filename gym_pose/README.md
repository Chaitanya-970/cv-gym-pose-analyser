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

### Optional dependencies for dataset conversion

Parsing original Penn Action MATLAB (`.mat`) annotations requires `scipy`:

```powershell
pip install scipy==1.15.3
```

`scipy` is imported only by `gym_analyser.penn_converter.load_penn_mat`. The core geometry, hysteresis counter, and standard tests continue to run without `scipy` using standard library Python (`python -S`).

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

### Convert Penn Action dataset to YOLO pose format

```powershell
cd gym_pose
.\.venv\Scripts\python.exe -m gym_analyser.penn_converter `
    --source-dir <penn_action_root> `
    --output-dir <fresh_empty_yolo_dir> `
    --action squats `
    --stride 5
```

The converter verifies bounding box and keypoint bounds, masks unannotated facial landmarks (COCO 0-4) and out-of-frame coordinates as `(0.000000, 0.000000, 0)`, maps 12 body joints to COCO-17 indices 5-16, protects native test sequences from entering training or validation, and strictly refuses non-empty output directories to prevent accidental deletion of user data.

The archive uses action `squat`; the CLI also accepts `squats`. The loader translates MATLAB one-based coordinates to zero-based image coordinates, following the author's `CreatePointLightDisplay.m` indexing. Continuous box endpoint differences are retained. Penn visibility 0 is treated as unavailable and masked, because many such coordinates are placeholders near the image corner; these are not valid occluded-joint targets. Original JPEG bytes are copied without recompression. Generated receipts record skipped annotations, split counts, and whether both train and validation are nonempty.

A local exploratory pilot was checked using 14 original sequences: 8 train, 2 validation, and 4 native test. Sampling every fifth frame produced 148/41/100 images. Ultralytics accepted all 289 labels without corrupt-label reports, and representative source/annotation overlays were inspected. Sequence 1865 has 73 coordinate rows but 72 boxes and is excluded rather than padded. This establishes converter operation, not model accuracy or training improvement. Identities and source overlap remain to be audited; some footage includes unannotated background people. Raw data, split manifests, model weights, environments and generated images are excluded from Git.

For fine-tuning, supply a reviewed explicit sequence manifest and require nonempty training and validation. Tiny parser fixtures may have empty validation; their receipt marks `trainable_split: false`. Default Ultralytics training masks unavailable coordinate targets but still trains their keypoint confidence toward zero, including the five unavailable facial points. Body-joint-only results are not the standard full COCO pose benchmark.


## Output artifacts

The pipeline produces four artifacts in `--output-dir`:

1. `annotated.mp4`: Source video retaining original dimensions and framerate, overlaid with COCO skeleton lines, highlighted active limb segments, knee joint vertex circle, joint angle arc drawn via OpenCV ellipse, and the repetition counter HUD banner.
2. `angles.csv`: Per-frame tabular data with columns:
   `frame_index, video_time_s, track_id, selected_side, knee_angle_deg, confidence, valid, reason, stage, completed_reps, posture_side, torso_inclination_deg, hip_angle_deg, torso_reason, hip_reason`.
   Missing numeric values are blank strings, never zeros. Valid frames report the actual observed side even during arming.
3. `summary.json`: Run metadata including Python and package versions, configuration parameters, source declared frame count and duration, processed frame count and duration, valid observation count and fraction, completed repetition count, read status note, and explicit limitations.
4. `angle_plot.png`: Matplotlib time series plot of knee angles over video time. Missing and invalid frames appear as gaps in the curve rather than connected lines.

## Third-party credits

- Ultralytics YOLO Pose: `https://github.com/ultralytics/ultralytics` (AGPL-3.0). Informed model loading, COCO keypoint conventions, and tracker integration.
- Shalbulov exercise_counter: `https://github.com/Shalbulov/exercise_counter` (MIT intent). Informed confidence gating, independent leg evaluation, and return-to-extension counter trigger logic.
- Penn Action Dataset: Weiyu Zhang, Menglong Zhu, and Konstantinos Derpanis, "From Actemes to Action: A Strongly-supervised Representation for Detailed Action Understanding", IEEE International Conference on Computer Vision (ICCV), 2013. URL: `https://www.cis.upenn.edu/~kostas/Penn_Action.tar.gz`, project page: `http://dreamdragon.github.io/PennAction/`. Informs keypoint coordinate parsing, official 13-joint order, and sequence-level train/test splits for YOLO pose converter.

## Known limitations and remaining manual checks

1. 2D projected angles: Angles are computed from 2D pixel coordinates and represent planar projections, not calibrated 3D anatomical joint angles.
2. Subject orientation: Designed for side-view single-person recordings. Oblique or frontal angles foreshorten limb segments and distort angle measurements.
3. No automatic form correction: The system detects threshold crossings. It does not validate squat depth safety, back alignment, or medical correctness.
4. Manual validation requirement: Synthetic test videos verify output mechanics and file generation, not human pose squat counting accuracy. Verification against human-reviewed ground truth remains pending for untested clips.

## Training and validation runner

Use `python -m gym_analyser.train_cli --help` from `gym_pose/`. Real runs require a local pose checkpoint (default searches `models/`), a COCO-17 person data YAML, nonempty train/validation, an exact sequence split manifest, original `source_inventory.json` metadata, and matching pose labels. The converter now emits both split and source metadata. Split membership and original native flags are checked independently; exact duplicate image content across splits is rejected. Near-duplicates and shared identities still require a separate audit.

For configuration validation only, add `--dry-run`. Run outputs must use a fresh directory outside the dataset; `--overwrite` does not bypass this. Receipts include original/frozen YAML hashes, image/label content fingerprints, source/split hashes, model hash, numeric metrics, runtime and PyTorch GPU memory when executed. Training accepts `--optimizer`, `--lr0`, `--freeze`, `--batch`, `--imgsz`, `--epochs` and `--no-amp`; automatic optimizer selection can override its requested learning rate. Current explicit defaults are AdamW and learning rate 0.0001, pending experiment validation. GPU memory figures exclude other apps and driver allocations.

Example baseline validation (Windows, separate CUDA environment):

```powershell
.\.venv_cuda\Scripts\python.exe -m gym_analyser.train_cli `
  --mode val --data test_data/training_audit/penn_pilot_yolo_09/data.yaml `
  --model models/yolo11n-pose.pt --device 0 --imgsz 320 --batch 4 `
  --output-dir output/experiments/my_baseline
```

A local pretrained baseline and bounded one-epoch training smoke tests have run. The smoke tests establish that training/checkpoint saving work; they do not establish an improved exercise model. The validation set is only two correlated videos containing unlabelled spotters, so its scores have substantial limitations. The original inference checkpoint stays in use; experimental weights and all run artifacts remain ignored. The held-out test split has not been scored. Missing faces are masked in OKS and do not automatically reduce its denominator; reported pose results concern the annotated visible body-joint subset rather than full COCO equivalence.

## Posture observations and count evaluation

The video pipeline now exports projected torso inclination and hip angle, their measurement side and availability reasons. Torso inclination is unsigned relative to upward image vertical (upright 0 degrees), so it cannot distinguish forward/backward lean without orientation information and is affected by camera roll. Hip angle uses shoulder-hip-knee. Confidence/geometry failures stay blank rather than zero; a genuinely measured zero-degree torso remains valid. Visible torso measurements may remain available when the knee is hidden. Ambiguous subjects and identity mismatch suppress observations. The video HUD shows both angles, and `summary.json` includes torso/hip valid-frame counts. These are descriptive measurements, not validated posture grades or safety advice. Observed attempts now produce neutral explanations in `attempts.json` (details below).

Two previously human-reviewed clips were rerun after integration: counts remained 2 and 6, with 213 and 1524 decoded frames respectively. The latter retains its pre-existing container metadata discrepancy (1526 declared versus 1524 decoded); a correct count does not erase that warning. These two examples do not establish general counting or posture accuracy.

Evaluate saved counts without loading a model:

```powershell
.\.venv\Scripts\python.exe -m gym_analyser.evaluation `
  --predictions-dir output/real_squats `
  --ground-truth output/real_squats/reviewed_counts.json
```

Only explicit human-reviewed counts enter MAE/exact-match denominators. Unknown counts remain excluded, including explicit `human_count: null` even if an alias field exists. Duplicate JSON keys and failed-processing summaries are rejected. Optional `--output-json` must use a new path and cannot overwrite human labels or existing receipts. Count comparison does not verify processing completeness or temporal event boundaries; inspect each run's status as well. Native data/label/source identities still require independent review. Corrected visual review supports a broader exploratory 22/6/4-sequence export (401/111/100 sampled images). Native splits and exact duplicates are checked; shared identities and remaining scene uncertainty prevent treating this as a definitive independent benchmark.

## Observed attempts and uncounted explanations

Every run also writes `attempts.json`, containing per-attempt frame/time boundaries, minimum observed knee angle, measurement interruptions and a neutral explanation. Reasons distinguish a shallow return, an insufficiently sustained depth crossing, measurement gaps, and a video ending mid-attempt. Opening crouches before a confirmed standing position stay separate telemetry. The repetition counter remains authoritative; observations do not change its counts.

Departure confirmation defaults to two consecutive valid frames below the standing threshold (`--departure-frames 2`); invalid observations reset that confirmation. This frame-based policy depends on frame rate. A confirmed attempt can be caused by pose noise, so these are software-defined observations pending human event review. They do not diagnose incorrect or unsafe form. CSV adds active `attempt_id` and the latest resolved `last_attempt_reason`; summaries include attempt totals, and the video displays the latest reason. End-of-video resolutions appear in JSON because no later video frame exists to display them.

## Per-movement posture summaries

The pipeline writes `posture_summary.json` alongside `attempts.json`. For each observed movement it reports torso and hip angle minimum/maximum/median, measured-frame coverage and reasons measurements were unavailable. Summaries use the exported CSV precision (0.01 degrees); this is formatting resolution, not anatomical measurement accuracy. The main summary links this artifact and includes coverage totals. Available torso measurements can be included when knee measurements are unavailable; incompatible identities, sides or failed measurement flags suppress the relevant values.

The observer detects software-defined movements and does not classify exercise intent. In the reviewed kettlebell example, weight pickup and weight return also create uncounted events. Treat these as observed movements requiring context, not confirmed failed squats. No automatic torso-angle cutoff removes them. Posture summaries remain descriptive; validated technique critique is still pending.

A five-epoch fine-tuning experiment on the expanded exploratory dataset improved validation pose localization scores but regressed the kettlebell example from six counted repetitions to seven and reduced posture measurement coverage. The extra count occurred during weight return. The pretrained application checkpoint remains in use; a better pose metric alone is insufficient for model promotion. Training artifacts and experimental weights remain local and ignored.

## Local web interface

The optional local interface uploads a video, runs the existing analyser, shows movement counts and per-event posture observations, and offers result downloads. It binds to localhost and allows one analysis at a time. These observations are projected 2D measurements; validated form grading and broader accuracy evaluation remain future work.

The web server also hosts the raw-video human review workspace directly under `/review/` on the same port, allowing you to record ground truth before running automated model analysis without running a second server process.

The downloadable `feedback_observations.json` adds explanations and within-video comparisons of measured counted cycles on the same track and body side. It reports torso inclination and projected knee-angle variation, excludes uncounted setup/return movements from comparisons, and explains missing measurements. Coverage settings are operational filters, not validated technique standards. The web results show these observations alongside each movement. Older result folders without this artifact remain readable.

The additional `feedback_phases.json` locates the minimum observed knee angle and return-phase measurements within counted cycles. Web results provide seek buttons for reviewing those moments. These are attention guides based on projected measurements, not validated technique grades; missing timing or measurement evidence is reported explicitly.

Keep web dependencies separate from the inference environment on Windows:

```powershell
python -m venv .venv_web
.\.venv_web\Scripts\python.exe -m pip install -r requirements-web.txt
.\.venv_web\Scripts\python.exe -m gym_analyser.web --port 8000 --device cpu
```

Open http://127.0.0.1:8000 for the video analyser, or navigate to http://127.0.0.1:8000/review/ for the raw-video human review workspace. A top navigation bar links directly between the analyser and the review workspace. With the configured CUDA environment, use `--device 0`. Analysis uses the existing `.venv_cuda` interpreter when present, otherwise `.venv`, then the current interpreter. Prepare the inference dependencies separately using the installation instructions above. The pretrained checkpoint must already exist at `models/yolo11n-pose.pt`; the web worker does not automatically download a missing model. It does not select experimental fine-tuned checkpoints automatically.

Uploads are limited to 200 MB and stored privately under ignored `output/web_jobs/`. Each job gets separate input/results folders. A result URL includes its job ID so reloading can restore results. Incomplete decoding is shown explicitly; counts from an incomplete run are partial observations. Uncounted events can include setup or weight-return movements and do not establish poor form.

If FFmpeg with the `libx264` encoder is available on PATH, a separate H.264 `preview.mp4` is generated for browser playback; the original annotated video is preserved. Without FFmpeg or if encoding fails, the original remains downloadable. Preview encoding has a finite timeout. No FFmpeg binary is bundled in this repository.

Web checks (no inference or downloads):

```powershell
.\.venv_web\Scripts\python.exe -m unittest tests.test_web tests.test_web_media tests.test_review_web tests.test_review_mount
```

The server is a local development application. Do not expose it to a network or run multiple instances against the same output storage. A server restart marks unfinished metadata interrupted; it cannot guarantee recovery from a killed machine or an externally surviving analysis process. Preserve old jobs and inspect running processes before restarting interrupted work.

The training CLI also accepts `--warmup-bias-lr` in train mode for controlled experiments. Omitting it preserves the installed Ultralytics default; an explicit finite value from 0 to 1 is forwarded and recorded in the experiment receipt. Training settings are experimental and do not automatically replace the application model.

For loss-ablation experiments, train mode accepts optional `--kobj` (finite nonnegative keypoint-presence loss gain). An explicit zero disables that loss term; omission preserves the installed default. Shared network updates can still change keypoint confidence. The setting and resolved training arguments are saved in receipts.


## Raw-video human review workspace

Record human observations before inspecting model predictions. When running the main web interface (`gym_analyser.web`), the review workspace is automatically available at http://127.0.0.1:8000/review/.

Alternatively, run the dedicated standalone review server without analyser or ML routes:

    .\.venv_web\Scripts\python.exe -m gym_analyser.review_web --port 8001

Run from this gym_pose directory, then open http://127.0.0.1:8001. Upload raw MP4, WebM or MOV video, watch it, enter a count (blank means unreviewed, zero means no repetitions), and save. A count requires a reviewer identifier, raw-video acknowledgment and an explicit answer about prior exposure to predictions. Optional movement events record human intent and uncertainty. The page does not run inference or infer technique quality.

Downloads include evaluator-compatible counts and a full review manifest with source hashes and edit history. Development cases and out-of-scope clips are excluded from count scoring. Candidate status alone does not establish independent final-test eligibility. Prepare any final evaluation protocol separately.

Files stay in ignored output/human_reviews/ regardless of the launch directory. Modified raw files or corrupted metadata block serving/saving/export rather than silently changing labels. Keep original recordings and use one server process. Unsupported browser codecs require an external player through the raw download; this tool does not transcode videos.
