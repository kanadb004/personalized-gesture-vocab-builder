"""Keras backbone architecture: 63 -> [128, 128] -> 64, L2-normalized output. Training-time
module only; frozen inference uses the numpy forward pass in pgvb.embedding."""

from __future__ import annotations

import keras
from keras import layers, ops


@keras.saving.register_keras_serializable(package="pgvb")
class L2Normalize(layers.Layer):
    def call(self, x):
        norm = ops.sqrt(ops.sum(ops.square(x), axis=-1, keepdims=True))
        return x / (norm + 1e-8)


def build_backbone(
    input_dim: int = 63,
    hidden_dims: list[int] | tuple[int, ...] = (128, 128),
    embed_dim: int = 64,
    dropout: float = 0.2,
) -> keras.Model:
    inputs = layers.Input(shape=(input_dim,), name="landmarks")
    x = inputs
    for i, h in enumerate(hidden_dims):
        x = layers.Dense(h, name=f"dense_{i}")(x)
        x = layers.BatchNormalization(name=f"bn_{i}")(x)
        x = layers.Activation("relu", name=f"relu_{i}")(x)
        x = layers.Dropout(dropout, name=f"dropout_{i}")(x)
    x = layers.Dense(embed_dim, name="embedding")(x)
    outputs = L2Normalize(name="l2_normalize")(x)
    return keras.Model(inputs, outputs, name="backbone_v1")
