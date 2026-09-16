"""Folds each Dense+BatchNorm pair of the trained Keras backbone into a single affine layer
and exports numpy-only inference weights, plus a metadata JSON. This is what pgvb.embedding
loads at runtime; TensorFlow is not needed after this step.

Usage: python scripts/export_backbone.py
"""

from __future__ import annotations

import argparse
import datetime
import json
import subprocess
import time
from pathlib import Path

import keras
import numpy as np

from pgvb.config import load as load_pgvb_config
from pgvb.embedding import Backbone
from pgvb.train.model import L2Normalize  # noqa: F401  registers the custom layer for loading

_MODEL_IN = Path("models/backbone_v1.keras")
_NPZ_OUT = Path("models/backbone_v1.npz")
_JSON_OUT = Path("models/backbone_v1.json")
_EVAL_JSON = Path("reports/backbone_v1_eval.json")
_N_TIMING_CALLS = 1000


def _fold_bn(dense_layer, bn_layer) -> tuple[np.ndarray, np.ndarray]:
    w, b = dense_layer.get_weights()
    gamma, beta, moving_mean, moving_var = bn_layer.get_weights()
    scale = gamma / np.sqrt(moving_var + bn_layer.epsilon)
    new_w = w * scale[None, :]
    new_b = (b - moving_mean) * scale + beta
    return new_w.astype(np.float32), new_b.astype(np.float32)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=_MODEL_IN, type=Path)
    args = parser.parse_args()

    model = keras.models.load_model(args.model)
    cfg = load_pgvb_config().backbone

    weights: list[tuple[np.ndarray, np.ndarray]] = []
    for i in range(len(cfg.hidden_dims)):
        dense = model.get_layer(f"dense_{i}")
        bn = model.get_layer(f"bn_{i}")
        weights.append(_fold_bn(dense, bn))
    embedding_layer = model.get_layer("embedding")
    w, b = embedding_layer.get_weights()
    weights.append((w.astype(np.float32), b.astype(np.float32)))

    npz_kwargs: dict[str, np.ndarray | int] = {"n_layers": len(weights)}
    for i, (w, b) in enumerate(weights):
        npz_kwargs[f"W{i}"] = w
        npz_kwargs[f"b{i}"] = b
    _NPZ_OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez(_NPZ_OUT, **npz_kwargs)

    backbone = Backbone.load(_NPZ_OUT)
    rng = np.random.default_rng(0)
    x = rng.normal(size=(cfg.input_dim,)).astype(np.float32)
    backbone.embed(x)  # warmup
    start = time.perf_counter()
    for _ in range(_N_TIMING_CALLS):
        backbone.embed(x)
    mean_ms = (time.perf_counter() - start) / _N_TIMING_CALLS * 1000

    commit = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()

    eval_summary = {}
    if _EVAL_JSON.exists():
        eval_summary = json.loads(_EVAL_JSON.read_text(encoding="utf-8"))

    meta = {
        "id": cfg.id,
        "input_dim": cfg.input_dim,
        "embed_dim": cfg.embed_dim,
        "hidden_dims": list(cfg.hidden_dims),
        "train_date": datetime.date.today().isoformat(),
        "git_commit": commit,
        "mean_embed_ms": mean_ms,
        "heldout_five_shot_acc": eval_summary.get("five_shot_acc"),
        "heldout_one_shot_acc": eval_summary.get("one_shot_acc"),
        "heldout_letters": eval_summary.get("heldout_letters"),
    }
    _JSON_OUT.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")

    print(f"wrote {_NPZ_OUT} and {_JSON_OUT}")
    print(f"mean embed time: {mean_ms:.4f} ms over {_N_TIMING_CALLS} calls")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
