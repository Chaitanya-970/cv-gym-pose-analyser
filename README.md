# CV Gym Pose Analyser

An early academic prototype for analysing single-person, side-view squat videos using pretrained YOLO pose estimation.

## Current capabilities

- Projected 2D knee-angle measurements and threshold-based repetition counting.
- Annotated video, per-frame CSV, summary JSON, and angle graph exports.
- Standard-library unit tests for geometry, counting, and output handling.

## Status

Work in progress. Real exercise-video accuracy has not been validated. The initial synthetic-video smoke test contained no person detections; it verifies output generation, not squat-counting accuracy. Review has identified remaining issues in side reselection, confidence validation, and incomplete-video/error reporting. This prototype does not assess medical safety or provide validated form correction.

## Run the tests

From the repository root:

```powershell
cd gym_pose
python -m unittest discover -s tests -p "test_*.py" -v
python -m gym_analyser --help
```

The core tests and command-line help do not require model weights or inference dependencies. For inference setup and output descriptions, see [the application README](gym_pose/README.md). Run module commands from `gym_pose/`.

## Repository contents

Only application source, tests, dependency requirements, README files, and Git ignore rules are included. Local recordings, transcripts, unrelated helper scripts, planning documents, virtual environments, model weights, input media, and generated results are excluded.

## Credits and licensing

The inference engine uses [Ultralytics YOLO](https://github.com/ultralytics/ultralytics), which has its own licensing terms. The counting design references [Shalbulov/exercise_counter](https://github.com/Shalbulov/exercise_counter), with adaptations for observed standing, confidence checks, and tracking gaps. See the application README for credits and limitations. No blanket permissive license is asserted for the integrated project.
