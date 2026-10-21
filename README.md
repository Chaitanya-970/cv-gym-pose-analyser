# CV Gym Pose Analyser

An early academic prototype for analysing single-person, side-view squat videos using pretrained YOLO pose estimation.

## Current capabilities

- Projected 2D knee-angle measurements and threshold-based repetition counting.
- Annotated video, per-frame CSV, summary JSON, and angle graph exports.
- Geometry/counter unit tests plus adapter tests using local generated videos and mocked pose outputs.

## Status

Work in progress. In addition to two initial real clips, the original model matched raw-first human totals on four personal recordings (9,6,7,5; the frontal recording is a separate challenge). General accuracy and posture feedback have not been validated. A once-only frozen-baseline 100-image Penn test yielded pose mAP50 0.2966 and mAP50–95 0.0687; these are pose metrics, not form-correction accuracy. The initial synthetic-video smoke test contained no person detections; it verifies output generation, not squat-counting accuracy. Reviewed regression tests cover side reselection, missing-confidence rejection, native float32 pose data, tracking gaps, and incomplete-video/error reporting. This prototype does not assess medical safety or provide validated form correction.

## Initial real-video count checks

The pretrained baseline was compared with counts supplied by the human project owner:

| Clip | Human count | Model count |
| --- | ---: | ---: |
| [Barbell squat demonstration](https://commons.wikimedia.org/wiki/File:Squat_-_exercise_demonstration_video.webm) (FitnessScape) | 2 | 2 |
| [Side-view kettlebell squats](https://commons.wikimedia.org/wiki/File:Kettlebell_Racked_Squats_(side_view).webm) (Taco fleur) | 6 | 6 |

Both clip totals match. These are two exploratory examples, not an estimate of general accuracy, rep-event precision, keypoint accuracy, or posture correctness. No per-repetition timestamps were labeled. The kettlebell run retains an `incomplete` integrity status because container-reported frame totals differ; independent FFprobe and OpenCV decodes agree on the available frame count. Footage and generated results remain local and excluded from Git. These clips should remain outside training for future baseline comparisons.

## Training evidence and remaining validation

Actual YOLO/PyTorch fine-tuning comparisons have completed on a 724-image Penn Action subset and an expanded 1,224-image Penn/COCO rehearsal set. The original pretrained checkpoint remains active because alternatives regressed on personal-video counting or measurement coverage. These limited experiments do not establish general posture accuracy.

1. Verify dataset access, usage terms, body-joint definitions, and sample annotations.
2. Convert compatible labels to YOLO pose format without inventing missing joints; split by subject/source to reduce leakage.
3. Prepare a CUDA-enabled training environment, targeting a 6 GB GPU initially, and run a tiny training sanity check.
4. Fine-tune and compare with the original checkpoint on the same held-out labeled images and separate squat videos.
5. Extend posture feedback and explanations for uncounted attempts; these are separate from joint-localization training.

Dataset choice, conversion, and training settings are still under review. Reuse public code and data where terms permit, retaining attribution and evaluating actual outcomes.

## Run the tests

From the repository root:

```powershell
cd gym_pose
python -m unittest tests.test_geometry tests.test_counter -v
python -m gym_analyser --help
```

The geometry/counter tests and command-line help do not require inference dependencies. The full adapter suite requires the packages in requirements.txt; run `.\.venv\Scripts\python.exe -m unittest discover -s tests -p "test_*.py" -v` after setup. For inference setup and output descriptions, see [the application README](gym_pose/README.md). Run module commands from `gym_pose/`.

## Repository contents

Only application source, tests, dependency requirements, README files, and Git ignore rules are included. Local recordings, transcripts, unrelated helper scripts, planning documents, virtual environments, model weights, input media, and generated results are excluded.

## Credits and licensing

The inference engine uses [Ultralytics YOLO](https://github.com/ultralytics/ultralytics), which has its own licensing terms. The counting design references [Shalbulov/exercise_counter](https://github.com/Shalbulov/exercise_counter), with adaptations for observed standing, confidence checks, and tracking gaps. See the application README for credits and limitations. No blanket permissive license is asserted for the integrated project.


## Local website and validation status

From `gym_pose`, launch the existing local environments with:

```powershell
.\.venv_web\Scripts\python.exe -m gym_analyser.web --port 8000 --device 0
```

Open http://127.0.0.1:8000. The local server provides upload, job status, annotated-video playback/seek, graphs, JSON/CSV/video downloads and a separate raw human-review workspace at `/review/`. It uses the local original checkpoint and a compatible inference environment; use `--device cpu` when CUDA is unavailable. See the application README for setup.

Current engineering checks: 866 tests passed with 3 skips in the full existing-dependency run; 167 relevant tests passed after the final fragment-wording changes. Real local upload/download and desktop/mobile rendering/video-seek checks passed. These results do not establish clinical or general posture accuracy. Feedback observations are model estimates, and unconfirmed fragments are not automatically separate failed repetitions. Higher inference resolution and alternative checkpoints remain experimental rather than default replacements.
