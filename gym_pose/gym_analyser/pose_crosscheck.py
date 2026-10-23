"""Whole-cycle corroboration, not a calibrated accuracy or form assessment.

Select one complete YOLO run; never splice joints, change angles or use target counts.
"""
import argparse
import hashlib
import json
import math
import time
from dataclasses import asdict
from pathlib import Path
from .counter import CounterConfig, SquatCounter
from .attempts import AttemptTracker

BASELINE_SHA = "869e83fcdffdc7371fa4e34cd8e51c838cc729571d1635e5141e3075e9319dc0"
CANDIDATE_SHA = "1060bda4a27012060eca246f9b2adeea22eabb045a1e58f8d229be29b7ebc2ba"
TEMPORAL_IOU = .5  # Operational correspondence, not a biological threshold.
MIN_COVERAGE = .5  # Data-availability gate, not an accuracy probability.


def cycles_agree(first, second):
    if len(first) != len(second):
        return False
    for a, b in zip(first, second):
        lo = max(a["start_time_s"], b["start_time_s"])
        hi = min(a["end_time_s"], b["end_time_s"])
        union = max(a["end_time_s"], b["end_time_s"]) - min(a["start_time_s"], b["start_time_s"])
        if union <= 0 or max(0, hi - lo) / union < TEMPORAL_IOU:
            return False
    return True


def _validate(run):
    for key in ("frames", "valid_frames"):
        if type(run.get(key)) is not int or run[key] < 0:
            raise ValueError("Invalid frame accounting")
    if run["valid_frames"] > run["frames"]:
        raise ValueError("Valid frames exceed decoded frames")
    previous = -1.
    for event in run["cycles"]:
        start, end = event["start_time_s"], event["end_time_s"]
        if any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) for x in (start, end)):
            raise ValueError("Invalid cycle timing")
        if not 0 <= start < end or start < previous:
            raise ValueError("Cycles must be ordered and non-overlapping")
        previous = end


def usable(run):
    return run["valid_frames"] >= CounterConfig().stable_frames and run["valid_frames"] / max(1, run["frames"]) >= MIN_COVERAGE


def choose_run(runs):
    if "baseline" not in runs or "candidate" not in runs:
        raise ValueError("Baseline and candidate runs are required")
    for run in runs.values():
        _validate(run)
    if len({run["frames"] for run in runs.values()}) != 1:
        raise ValueError("Models did not decode the same complete video")
    selected, support = "baseline", []
    for first, second in (("baseline", "candidate"), ("baseline", "mediapipe"), ("candidate", "mediapipe")):
        if first in runs and second in runs and usable(runs[first]) and usable(runs[second]) and cycles_agree(runs[first]["cycles"], runs[second]["cycles"]):
            selected, support = first, [first, second]
            break
    return {
        "status": "corroborated" if support else "needs_review",
        "selected": selected,
        "selected_count": len(runs[selected]["cycles"]),
        "support": support,
        "temporal_iou": TEMPORAL_IOU,
        "minimum_valid_coverage": MIN_COVERAGE,
        "accuracy_validated": False,
        "note": ("Two pose estimators agree on complete cycle timings; this remains an estimate, not a form grade."
                 if support else "Pose estimators do not fully corroborate this count. Review the video; the original model estimate is retained."),
    }


def _hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def collect(source, deadline, *, model_path=None, device="cpu", mediapipe_path=None):
    import cv2
    from contextlib import nullcontext
    if model_path is not None:
        from ultralytics import YOLO
        model = YOLO(str(model_path))
        context = nullcontext(None)
    else:
        from .auxiliary_pose import MediaPipeLandmarker
        context = MediaPipeLandmarker(mediapipe_path, "video", smooth_video=True)
    cap = cv2.VideoCapture(str(source))
    counter = SquatCounter(CounterConfig())
    attempts = AttemptTracker(counter.config)
    frames = valid = 0
    began = time.monotonic()
    mapping = [0, 2, 5, 7, 8, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]
    try:
        if not cap.isOpened():
            raise ValueError("Video could not be opened")
        fps = cap.get(cv2.CAP_PROP_FPS)
        if not math.isfinite(fps) or not 1 <= fps <= 240:
            raise ValueError("Unsupported video frame rate")
        with context as detector:
            while True:
                if time.monotonic() >= deadline:
                    raise TimeoutError("Counting cross-check time budget exceeded")
                ok, image = cap.read()
                if not ok:
                    break
                detections = []
                if model_path is not None:
                    results = model.track(image, persist=True, tracker="bytetrack.yaml", imgsz=640, device=device, verbose=False)
                    if results and results[0].boxes is not None and results[0].keypoints is not None:
                        points = results[0].keypoints.data.cpu().tolist()
                        ids = results[0].boxes.id
                        ids = ids.int().cpu().tolist() if ids is not None else [None] * len(points)
                        detections = [{"track_id": tid, "keypoints": p} for tid, p in zip(ids, points)]
                else:
                    height, width = image.shape[:2]
                    result = detector.detect_rgb(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), round(frames / fps * 1000))
                    for pose in result["poses"]:
                        points = []
                        for index in mapping:
                            p = pose[index]
                            inside = 0 <= p["x_px"] < width and 0 <= p["y_px"] < height
                            points.append([p["x_px"], p["y_px"], min(p["visibility"], p["presence"]) if inside else 0.])
                        detections.append({"track_id": 0, "keypoints": points})
                result = counter.update(frames, frames / fps, detections)
                attempts.observe(result, counter.armed_side)
                valid += int(result.valid)
                frames += 1
    finally:
        cap.release()
    return {"frames": frames, "valid_frames": valid,
            "cycles": [e for e in attempts.finish()["events"] if e["status"] == "counted"],
            "elapsed_s": time.monotonic() - began}


def run(source, baseline, candidate, mediapipe, output, device, time_limit):
    if not math.isfinite(time_limit) or time_limit <= 0:
        raise ValueError("Positive finite time limit required")
    if Path(output).exists():
        raise FileExistsError("Fresh counting cross-check output required")
    if _hash(baseline) != BASELINE_SHA or _hash(candidate) != CANDIDATE_SHA:
        raise ValueError("Cross-check requires the reviewed original YOLO11n and YOLO11s assets")
    deadline = time.monotonic() + time_limit
    runs = {"baseline": collect(source, deadline, model_path=baseline, device=device),
            "candidate": collect(source, deadline, model_path=candidate, device=device)}
    selection = choose_run(runs)
    if selection["status"] != "corroborated" and any(usable(x) for x in runs.values()):
        runs["mediapipe"] = collect(source, deadline, mediapipe_path=mediapipe)
        selection = choose_run(runs)
    selection.update(schema_version=1, source_sha256=_hash(source), configuration=asdict(CounterConfig()),
                     selected_model_path=str(Path(baseline if selection["selected"] == "baseline" else candidate).resolve()),
                     models=runs, model_sha256={"baseline": BASELINE_SHA, "candidate": CANDIDATE_SHA},
                     limitations=["Model errors are correlated; agreement is not independent ground truth.",
                                  "The retained 110-degree rule is not a universal definition of full squat depth.",
                                  "This check does not validate anatomical angles or posture faults."])
    if "mediapipe" in runs:
        selection["model_sha256"]["mediapipe"] = _hash(mediapipe)
    with Path(output).open("x", encoding="utf-8") as stream:
        json.dump(selection, stream, indent=2, allow_nan=False)
    return selection


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "baseline", "candidate", "mediapipe", "output"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--time-limit", type=float, default=390.)
    args = parser.parse_args()
    run(args.source, args.baseline, args.candidate, args.mediapipe, args.output, args.device, args.time_limit)


if __name__ == "__main__":
    main()
