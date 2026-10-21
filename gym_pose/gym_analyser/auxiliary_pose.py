"""Optional MediaPipe33 landmark provider; never labels technique or changes counts.

Requires an explicit local model asset and optional mediapipe runtime. No downloads
or imports of the optional inference package occur until the context is entered.
"""
import math
from numbers import Real
from pathlib import Path

FOOT_INDICES = {"left": (29, 31), "right": (30, 32)}

def _dimension(value):
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError("Image dimensions must be positive integers")
    return value

def _finite(value):
    return isinstance(value, Real) and not isinstance(value, bool) and math.isfinite(value)

def serialize_landmarks(points, width, height):
    """Preserve estimated original-image pixels, including off-screen predictions."""
    _dimension(width); _dimension(height)
    if len(points) != 33:
        raise ValueError("Expected MediaPipe33 landmark layout")
    records = []
    for point in points:
        values = [point.x, point.y, point.visibility, point.presence]
        if not all(_finite(v) for v in values):
            raise ValueError("Non-finite or invalid landmark output")
        if not 0 <= point.visibility <= 1 or not 0 <= point.presence <= 1:
            raise ValueError("Landmark confidence outside [0,1]")
        records.append({"x_px": float(point.x * width), "y_px": float(point.y * height),
                        "visibility": float(point.visibility), "presence": float(point.presence)})
    return records

def visible_foot_pairs(landmarks, width, height, min_visibility=.5, min_presence=.5):
    """Return sides with measurable foot points, not sides known to have good form."""
    _dimension(width); _dimension(height)
    if not all(_finite(v) and 0 <= v <= 1 for v in [min_visibility, min_presence]):
        raise ValueError("Confidence settings must be finite in [0,1]")
    if len(landmarks) != 33:
        raise ValueError("Expected MediaPipe33 landmark layout")
    available = []
    for side, indices in FOOT_INDICES.items():
        valid = True
        for idx in indices:
            p = landmarks[idx]
            values = [p.get(k) for k in ["x_px", "y_px", "visibility", "presence"]]
            if not all(_finite(v) for v in values):
                valid = False; break
            x, y, visibility, presence = values
            if not (0 <= x < width and 0 <= y < height and
                    min_visibility <= visibility <= 1 and min_presence <= presence <= 1):
                valid = False; break
        if valid:
            available.append(side)
    return available

class MediaPipeLandmarker:
    """Explicit image/video provider with strict time ordering and owned cleanup."""
    def __init__(self, model_path, mode="image"):
        self.model_path = Path(model_path).resolve()
        if not self.model_path.is_file():
            raise FileNotFoundError("Local auxiliary pose model is missing; automatic download disabled")
        if mode not in ("image", "video"):
            raise ValueError("mode must be image or video")
        self.mode = mode; self._detector = None; self._mp = None; self._last_timestamp = None

    def __enter__(self):
        if self._detector is not None:
            raise RuntimeError("Landmarker context already entered")
        try:
            import mediapipe as mp
        except ImportError as exc:
            raise RuntimeError("Optional MediaPipe runtime is unavailable") from exc
        running = mp.tasks.vision.RunningMode.IMAGE if self.mode == "image" else mp.tasks.vision.RunningMode.VIDEO
        options = mp.tasks.vision.PoseLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(self.model_path)),
            running_mode=running, num_poses=2,
            min_pose_detection_confidence=.5, min_pose_presence_confidence=.5)
        self._detector = mp.tasks.vision.PoseLandmarker.create_from_options(options)
        self._mp = mp; self._last_timestamp = None
        return self

    def detect_rgb(self, frame, timestamp_ms=None):
        if self._detector is None:
            raise RuntimeError("Use landmarker inside its context")
        shape = getattr(frame, "shape", ())
        if len(shape) != 3 or shape[2] != 3 or str(getattr(frame, "dtype", "")) != "uint8":
            raise ValueError("Expected original RGB uint8 HxWx3 frame")
        height, width = int(shape[0]), int(shape[1]); _dimension(width); _dimension(height)
        if self.mode == "video":
            if (isinstance(timestamp_ms, bool) or not isinstance(timestamp_ms, int) or timestamp_ms < 0 or
                (self._last_timestamp is not None and timestamp_ms <= self._last_timestamp)):
                raise ValueError("Video timestamps must be nonnegative strictly increasing integer milliseconds")
        elif timestamp_ms is not None:
            raise ValueError("Image mode does not consume video timestamps")
        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=frame)
        if self.mode == "video":
            result = self._detector.detect_for_video(image, timestamp_ms)
            self._last_timestamp = timestamp_ms
        else:
            result = self._detector.detect(image)
        poses = [serialize_landmarks(points, width, height) for points in result.pose_landmarks]
        reason = "single_person" if len(poses) == 1 else ("no_person" if not poses else "multiple_people")
        feet = visible_foot_pairs(poses[0], width, height) if len(poses) == 1 else []
        return {"landmark_schema": "mediapipe33", "measurement_basis": "model_estimate",
                "width": width, "height": height, "timestamp_ms": timestamp_ms,
                "person_count": len(poses), "reason": reason, "poses": poses,
                "foot_pair_available": feet, "technique_accuracy": "not_validated"}

    def __exit__(self, exc_type, exc, traceback):
        if self._detector is not None:
            try:
                self._detector.close()
            finally:
                self._detector = None; self._mp = None; self._last_timestamp = None
