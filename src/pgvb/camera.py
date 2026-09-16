"""Webcam capture as a context manager, plus a rolling fps counter."""

from __future__ import annotations

import time
from types import TracebackType

import cv2
import numpy as np


class Camera:
    """Wraps cv2.VideoCapture. Yields (frame_bgr, timestamp_ms) pairs via read()."""

    def __init__(
        self,
        index: int = 0,
        width: int = 1280,
        height: int = 720,
        mirror: bool = True,
    ) -> None:
        self.index = index
        self.width = width
        self.height = height
        self.mirror = mirror
        self._cap: cv2.VideoCapture | None = None
        self._start_time: float | None = None

    def __enter__(self) -> "Camera":
        self._cap = cv2.VideoCapture(self.index)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        if not self._cap.isOpened():
            raise RuntimeError(f"could not open camera index {self.index}")
        self._start_time = time.monotonic()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def read(self) -> tuple[np.ndarray, float] | None:
        """Reads one frame. Returns (frame_bgr, timestamp_ms) or None if the read failed."""
        if self._cap is None or self._start_time is None:
            raise RuntimeError("Camera.read() called outside a with block")
        ok, frame = self._cap.read()
        if not ok:
            return None
        if self.mirror:
            frame = cv2.flip(frame, 1)
        ts_ms = (time.monotonic() - self._start_time) * 1000.0
        return frame, ts_ms


class FpsMeter:
    """Tracks a rolling average fps over the last `window` frame intervals."""

    def __init__(self, window: int = 30) -> None:
        self.window = window
        self._timestamps: list[float] = []

    def tick(self) -> float:
        """Records one frame arriving now and returns the current rolling fps."""
        now = time.monotonic()
        self._timestamps.append(now)
        if len(self._timestamps) > self.window:
            self._timestamps.pop(0)
        if len(self._timestamps) < 2:
            return 0.0
        elapsed = self._timestamps[-1] - self._timestamps[0]
        if elapsed <= 0:
            return 0.0
        return (len(self._timestamps) - 1) / elapsed
