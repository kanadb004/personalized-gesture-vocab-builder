"""Prototypical-network loss and episode accuracy (Snell et al. 2017). Training-only module."""

from __future__ import annotations

import tensorflow as tf


def prototypical_loss(
    support_emb: tf.Tensor,
    support_y: tf.Tensor,
    query_emb: tf.Tensor,
    query_y: tf.Tensor,
    n_way: int,
) -> tuple[tf.Tensor, tf.Tensor]:
    """support_emb: (n_way*k_shot, D) unit-normalized embeddings, support_y: matching
    episode-local class ints in [0, n_way). query_emb/query_y: same shape convention for the
    query set. Returns (scalar cross-entropy loss over cosine-distance logits, mean accuracy)."""
    protos = []
    for c in range(n_way):
        mask = tf.equal(support_y, c)
        class_emb = tf.boolean_mask(support_emb, mask)
        protos.append(tf.reduce_mean(class_emb, axis=0))
    protos = tf.stack(protos, axis=0)
    protos = protos / (tf.norm(protos, axis=-1, keepdims=True) + 1e-8)

    similarities = tf.matmul(query_emb, protos, transpose_b=True)  # (n_query_total, n_way)
    logits = similarities  # cosine similarity as logits; higher = closer

    loss = tf.reduce_mean(
        tf.keras.losses.sparse_categorical_crossentropy(query_y, logits, from_logits=True)
    )
    preds = tf.argmax(logits, axis=-1)
    acc = tf.reduce_mean(tf.cast(tf.equal(preds, tf.cast(query_y, preds.dtype)), tf.float32))
    return loss, acc
