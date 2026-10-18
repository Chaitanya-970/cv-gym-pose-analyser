# CV Gym Pose Analyser

An early academic prototype for analysing single-person, side-view squat videos using pretrained YOLO pose estimation.

## Current capabilities

- Projected 2D knee-angle measurements and threshold-based repetition counting.
- Annotated video, per-frame CSV, summary JSON, and angle graph exports.
- Geometry/counter unit tests plus adapter tests using local generated videos and mocked pose outputs.

## Status

Work in progress. Total repetition counts match user-reported counts on two initial real clips; general accuracy and posture feedback have not been validated. The initial synthetic-video smoke test contained no person detections; it verifies output generation, not squat-counting accuracy. Reviewed regression tests cover side reselection, missing-confidence rejection, native float32 pose data, tracking gaps, and incomplete-video/error reporting. This prototype does not assess medical safety or provide validated form correction.

## Initial real-video count checks

The pretrained baseline was compared with counts supplied by the human project owner:

| Clip | Human count | Model count |
| --- | ---: | ---: |
| [Barbell squat demonstration](https://commons.wikimedia.org/wiki/File:Squat_-_exercise_demonstration_video.webm) (FitnessScape) | 2 | 2 |
| [Side-view kettlebell squats](https://commons.wikimedia.org/wiki/File:Kettlebell_Racked_Squats_(side_view).webm) (Taco fleur) | 6 | 6 |

Both clip totals match. These are two exploratory examples, not an estimate of general accuracy, rep-event precision, keypoint accuracy, or posture correctness. No per-repetition timestamps were labeled. The kettlebell run retains an `incomplete` integrity status because container-reported frame totals differ; independent FFprobe and OpenCV decodes agree on the available frame count. Footage and generated results remain local and excluded from Git. These clips should remain outside training for future baseline comparisons.

## Training roadmap

The planned training stage is **fine-tuning YOLO pose on public exercise data**, starting from pretrained weights. The current implementation is the pretrained baseline; no exercise-specific fine-tuning has been completed yet.

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
