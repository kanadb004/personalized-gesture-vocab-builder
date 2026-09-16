"""Wraps MediaPipe Tasks HandLandmarker for both video streams and single images."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core.base_options import BaseOptions

_DEFAULT_MODEL_PATH = Path(__file__).resolve().parents[2] / "models" / "hand_landmarker.task"


@dataclass
class HandFrame:
    landmarks: np.ndarray  # (21, 3) float32, image-normalized (x, y, z) from MediaPipe
    handedness: str  # "Left" or "Right"
    score: float
    ts_ms: float


class HandTracker:
    """Wraps HandLandmarker. Use track() for a VIDEO-mode stream with monotonic timestamps,
    or track_image() for one-off IMAGE-mode detections (used by Phase 2 dataset extraction)."""

    def __init__(
        self,
        model_path: str | Path | None = None,
        num_hands: int = 1,
        min_hand_score: float = 0.5,
        min_hand_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ) -> None:
        self.model_path = Path(model_path) if model_path is not None else _DEFAULT_MODEL_PATH
        self.num_hands = num_hands
        self.min_hand_score = min_hand_score
        self.min_hand_detection_confidence = min_hand_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence

        self._video_landmarker = self._build(vision.RunningMode.VIDEO)
        self._image_landmarker = self._build(vision.RunningMode.IMAGE)

    def _build(self, running_mode: vision.RunningMode) -> vision.HandLandmarker:
        options = vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(self.model_path)),
            running_mode=running_mode,
            num_hands=self.num_hands,
            min_hand_detection_confidence=self.min_hand_detection_confidence,
            min_tracking_confidence=self.min_tracking_confidence,
        )
        return vision.HandLandmarker.create_from_options(options)

    def _best_hand(self, result: vision.HandLandmarkerResult) -> tuple[int, float] | None:
        if not result.hand_landmarks:
            return None
        best_index = -1
        best_score = -1.0
        for i, categories in enumerate(result.handedness):
            score = categories[0].score
            if score > best_score:
                best_score = score
                best_index = i
        if best_index < 0 or best_score < self.min_hand_score:
            return None
        return best_index, best_score

    def track(self, frame_bgr: np.ndarray, ts_ms: float) -> HandFrame | None:
        """Runs VIDEO-mode detection on one BGR frame. ts_ms must be monotonically
        increasing across calls. Returns None if no hand scores above min_hand_score."""
        rgb = frame_bgr[:, :, ::-1]
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
        result = self._video_landmarker.detect_for_video(mp_image, int(ts_ms))
        return self._to_hand_frame(result, ts_ms)

    def track_image(self, image_rgb: np.ndarray) -> HandFrame | None:
        """Runs IMAGE-mode detection on one RGB image. Used for offline dataset extraction."""
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(image_rgb))
        result = self._image_landmarker.detect(mp_image)
        return self._to_hand_frame(result, ts_ms=0.0)

    def _to_hand_frame(self, result: vision.HandLandmarkerResult, ts_ms: float) -> HandFrame | None:
        picked = self._best_hand(result)
        if picked is None:
            return None
        index, score = picked
        landmarks = np.array(
            [[lm.x, lm.y, lm.z] for lm in result.hand_landmarks[index]],
            dtype=np.float32,
        )
        handedness = result.handedness[index][0].category_name
        return HandFrame(landmarks=landmarks, handedness=handedness, score=score, ts_ms=ts_ms)
