"""Loads extracted ASL landmark features and splits them into a training pool (classes used
for episodic training) and a held-out pool (classes never seen during training, used for the
few-shot generalization check in scripts/eval_backbone.py). Training-only module (Keras)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

_NUM_LANDMARKS = 21


@dataclass
class SplitData:
    x: np.ndarray  # (N, 63) float32
    y: np.ndarray  # (N,) class name strings
    source: np.ndarray  # (N,) source name strings, e.g. "asl_train" or "asl_test"


def load_split(path: str | Path, heldout: list[str]) -> tuple[SplitData, SplitData]:
    """Loads the .npz written by scripts/extract_landmarks.py and splits it by class name
    into (train, heldout) SplitData, where heldout contains only the given class names."""
    data = np.load(path, allow_pickle=False)
    x, y, source = data["x"], data["y"], data["source"]
    is_heldout = np.isin(y, list(heldout))
    train = SplitData(x=x[~is_heldout], y=y[~is_heldout], source=source[~is_heldout])
    held = SplitData(x=x[is_heldout], y=y[is_heldout], source=source[is_heldout])
    return train, held


def augment(
    x: np.ndarray,
    rng: np.random.Generator,
    rotation_deg: float = 15.0,
    scale_min: float = 0.9,
    scale_max: float = 1.1,
    jitter_sigma: float = 0.01,
) -> np.ndarray:
    """Applies random rotation about the wrist (origin, x-y plane), random uniform scale, and
    Gaussian coordinate jitter to a batch of normalized feature vectors. x: (N, 63) float32,
    already normalized so the wrist is at the origin. Returns a new (N, 63) float32 array."""
    n = x.shape[0]
    points = x.reshape(n, _NUM_LANDMARKS, 3).astype(np.float64).copy()

    angles = np.deg2rad(rng.uniform(-rotation_deg, rotation_deg, size=n))
    cos_t, sin_t = np.cos(angles), np.sin(angles)
    xy = points[:, :, :2]
    rotated_x = cos_t[:, None] * xy[:, :, 0] - sin_t[:, None] * xy[:, :, 1]
    rotated_y = sin_t[:, None] * xy[:, :, 0] + cos_t[:, None] * xy[:, :, 1]
    points[:, :, 0] = rotated_x
    points[:, :, 1] = rotated_y

    scales = rng.uniform(scale_min, scale_max, size=n)
    points *= scales[:, None, None]

    points += rng.normal(0.0, jitter_sigma, size=points.shape)

    return points.reshape(n, _NUM_LANDMARKS * 3).astype(np.float32)
