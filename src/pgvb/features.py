"""Normalizes raw MediaPipe hand landmarks into a canonical 63-d feature vector.

Normalization (see docs/PLAN.md Section 2):
  1. Translate so the wrist (landmark 0) is the origin.
  2. Scale so the distance from the wrist to the middle-finger MCP (landmark 9) is 1.
  3. Mirror x for left hands so both hands share one canonical space.
  4. Optionally rotate about the wrist so the wrist-to-middle-MCP direction is fixed
     (config flag `features.rotate`, default off).
Output is the flattened (21, 3) array as a (63,) float32 vector.
"""

from __future__ import annotations

import numpy as np

_WRIST = 0
_MIDDLE_MCP = 9
_NUM_LANDMARKS = 21
_DIM = _NUM_LANDMARKS * 3


def feature_dim() -> int:
    return _DIM


def normalize_landmarks(
    lm: np.ndarray, handedness: str, rotate: bool = False
) -> np.ndarray:
    """lm: (21, 3) float array of raw landmarks. handedness: "Left" or "Right".
    Returns a (63,) float32 vector."""
    points = np.asarray(lm, dtype=np.float64).reshape(_NUM_LANDMARKS, 3)

    translated = points - points[_WRIST]

    scale = np.linalg.norm(translated[_MIDDLE_MCP])
    if scale < 1e-8:
        scale = 1e-8
    scaled = translated / scale

    if handedness.lower().startswith("left"):
        scaled = scaled.copy()
        scaled[:, 0] *= -1.0

    if rotate:
        direction = scaled[_MIDDLE_MCP, :2]
        angle = np.arctan2(direction[1], direction[0])
        target_angle = -np.pi / 2.0
        theta = target_angle - angle
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        rot = np.array([[cos_t, -sin_t], [sin_t, cos_t]])
        scaled = scaled.copy()
        scaled[:, :2] = scaled[:, :2] @ rot.T

    return scaled.reshape(_DIM).astype(np.float32)
