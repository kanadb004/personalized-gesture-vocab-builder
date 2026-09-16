"""Numpy forward pass parity with the Keras model it was folded from. Skips without
TensorFlow/Keras (not required at runtime) and without a trained, exported backbone."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pgvb.embedding import Backbone

_NPZ_PATH = Path("models/backbone_v1.npz")
_KERAS_PATH = Path("models/backbone_v1.keras")

pytest.importorskip("tensorflow")
pytest.importorskip("keras")

if not _NPZ_PATH.exists() or not _KERAS_PATH.exists():
    pytest.skip("trained backbone not exported yet", allow_module_level=True)

import keras  # noqa: E402

from pgvb.train.model import L2Normalize  # noqa: E402, F401  registers the custom layer


def test_numpy_forward_matches_keras():
    keras_model = keras.models.load_model(_KERAS_PATH)
    numpy_backbone = Backbone.load(_NPZ_PATH)

    rng = np.random.default_rng(42)
    x = rng.normal(size=(100, 63)).astype(np.float32)

    keras_out = keras_model(x, training=False).numpy()
    numpy_out = numpy_backbone.embed_batch(x)

    np.testing.assert_allclose(numpy_out, keras_out, atol=1e-5)


def test_output_is_unit_norm():
    numpy_backbone = Backbone.load(_NPZ_PATH)
    rng = np.random.default_rng(7)
    x = rng.normal(size=(20, 63)).astype(np.float32)
    out = numpy_backbone.embed_batch(x)
    norms = np.linalg.norm(out, axis=-1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-5)
