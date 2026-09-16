"""Trains the frozen embedding backbone with episodic prototypical-network training.

Usage: python scripts/train_backbone.py --config configs/train_v1.yaml
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np
import tensorflow as tf
import yaml

from pgvb.config import load as load_pgvb_config
from pgvb.train.dataset import load_split
from pgvb.train.episodes import EpisodeSampler
from pgvb.train.losses import prototypical_loss
from pgvb.train.model import build_backbone

_MODEL_OUT = Path("models/backbone_v1.keras")
_HISTORY_OUT = Path("reports/train_v1_history.csv")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    seed = cfg["seed"]
    np.random.seed(seed)
    tf.random.set_seed(seed)
    rng = np.random.default_rng(seed)

    train_split, heldout_split = load_split(cfg["data"]["landmarks_path"], cfg["data"]["heldout"])
    print(f"train classes: {len(set(train_split.y.tolist()))}, examples: {train_split.x.shape[0]}")
    print(f"heldout classes: {sorted(set(heldout_split.y.tolist()))}, examples: {heldout_split.x.shape[0]}")

    ep_cfg = cfg["episode"]
    aug_cfg = cfg["augment"]

    train_sampler = EpisodeSampler(
        train_split, ep_cfg["n_way"], ep_cfg["k_shot"], ep_cfg["n_query"], aug_cfg, rng
    )
    val_sampler = EpisodeSampler(
        heldout_split, ep_cfg["n_way"], ep_cfg["k_shot"], ep_cfg["n_query"], None, rng
    )

    backbone_cfg = load_pgvb_config().backbone
    model = build_backbone(
        backbone_cfg.input_dim, backbone_cfg.hidden_dims, backbone_cfg.embed_dim, backbone_cfg.dropout
    )
    optimizer = tf.keras.optimizers.Adam(learning_rate=cfg["optimizer"]["learning_rate"])

    best_val_acc = -1.0
    best_epoch = -1
    patience_counter = 0
    history = []

    _MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)
    _HISTORY_OUT.parent.mkdir(parents=True, exist_ok=True)

    start = time.time()
    for epoch in range(ep_cfg["max_epochs"]):
        epoch_losses, epoch_accs = [], []
        for _ in range(ep_cfg["episodes_per_epoch"]):
            support_x, support_y, query_x, query_y = train_sampler.sample_episode(apply_augment=True)
            with tf.GradientTape() as tape:
                support_emb = model(support_x, training=True)
                query_emb = model(query_x, training=True)
                loss, acc = prototypical_loss(support_emb, support_y, query_emb, query_y, ep_cfg["n_way"])
            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            epoch_losses.append(float(loss))
            epoch_accs.append(float(acc))

        val_accs = []
        for _ in range(50):
            support_x, support_y, query_x, query_y = val_sampler.sample_episode(apply_augment=False)
            support_emb = model(support_x, training=False)
            query_emb = model(query_x, training=False)
            _, acc = prototypical_loss(support_emb, support_y, query_emb, query_y, ep_cfg["n_way"])
            val_accs.append(float(acc))

        mean_loss = float(np.mean(epoch_losses))
        mean_train_acc = float(np.mean(epoch_accs))
        mean_val_acc = float(np.mean(val_accs))
        history.append(
            {"epoch": epoch, "loss": mean_loss, "train_acc": mean_train_acc, "val_acc": mean_val_acc}
        )
        print(
            f"epoch {epoch}: loss={mean_loss:.4f} train_acc={mean_train_acc:.3f} "
            f"val_acc={mean_val_acc:.3f}"
        )

        if mean_val_acc > best_val_acc:
            best_val_acc = mean_val_acc
            best_epoch = epoch
            patience_counter = 0
            model.save(_MODEL_OUT)
        else:
            patience_counter += 1
            if patience_counter >= ep_cfg["patience"]:
                print(f"early stopping at epoch {epoch}, best epoch {best_epoch}, val_acc={best_val_acc:.3f}")
                break

    elapsed = time.time() - start
    print(f"training took {elapsed:.1f}s, best val_acc={best_val_acc:.3f} at epoch {best_epoch}")

    with open(_HISTORY_OUT, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "loss", "train_acc", "val_acc"])
        writer.writeheader()
        writer.writerows(history)
    print(f"wrote {_HISTORY_OUT} and {_MODEL_OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
