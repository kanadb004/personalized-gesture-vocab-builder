"""Prototype math for the nearest-prototype recognizer: unit-normalized mean embeddings,
cosine distances to a set of prototypes, and the nearest-two lookup used for margin checks."""

from __future__ import annotations

import numpy as np


def make_prototype(examples: np.ndarray) -> np.ndarray:
    """examples: (K, D) unit-norm embeddings. Returns the unit-normalized mean, (D,)."""
    examples = np.asarray(examples, dtype=np.float32)
    mean = examples.mean(axis=0)
    norm = np.linalg.norm(mean)
    if norm < 1e-8:
        norm = 1e-8
    return (mean / norm).astype(np.float32)


def cosine_distances(e: np.ndarray, protos: np.ndarray) -> np.ndarray:
    """e: (D,) unit vector. protos: (P, D) unit vectors. Returns (P,) cosine distances."""
    e = np.asarray(e, dtype=np.float32)
    protos = np.asarray(protos, dtype=np.float32)
    return 1.0 - protos @ e


def nearest(e: np.ndarray, protos: np.ndarray) -> tuple[int, float, float]:
    """Returns (index of the nearest prototype, its distance d1, second-nearest distance d2).
    d2 is inf when fewer than two prototypes are given."""
    distances = cosine_distances(e, protos)
    order = np.argsort(distances)
    idx = int(order[0])
    d1 = float(distances[idx])
    d2 = float(distances[order[1]]) if len(order) > 1 else float("inf")
    return idx, d1, d2
