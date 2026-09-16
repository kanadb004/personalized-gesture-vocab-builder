"""Numpy forward pass over the frozen, exported backbone weights. This is the inference-time
path: no TensorFlow import, no Keras. Weights come from scripts/export_backbone.py, which folds
each BatchNorm into the preceding Dense layer."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def _relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0.0)


class Backbone:
    """Loads folded Dense(+ReLU) layer weights from an .npz file and runs the forward pass in
    numpy. Layers are named `dense_0`, `dense_1`, ..., and a final `embedding` layer; every
    layer but the last is followed by ReLU, and the output is L2-normalized."""

    def __init__(self, weights: list[tuple[np.ndarray, np.ndarray]]) -> None:
        self._weights = weights

    @classmethod
    def load(cls, path: str | Path) -> "Backbone":
        data = np.load(path)
        n_layers = int(data["n_layers"])
        weights = [
            (data[f"W{i}"].astype(np.float32), data[f"b{i}"].astype(np.float32))
            for i in range(n_layers)
        ]
        return cls(weights)

    def embed(self, x: np.ndarray) -> np.ndarray:
        """x: (input_dim,) float32. Returns (embed_dim,) unit-norm float32 vector."""
        return self.embed_batch(x[None, :])[0]

    def embed_batch(self, x: np.ndarray) -> np.ndarray:
        """x: (N, input_dim) float32. Returns (N, embed_dim) unit-norm float32 array."""
        h = np.asarray(x, dtype=np.float32)
        n_layers = len(self._weights)
        for i, (w, b) in enumerate(self._weights):
            h = h @ w + b
            if i < n_layers - 1:
                h = _relu(h)
        norm = np.linalg.norm(h, axis=-1, keepdims=True)
        norm = np.where(norm < 1e-8, 1e-8, norm)
        return (h / norm).astype(np.float32)
