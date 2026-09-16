"""Evaluates the trained backbone on the five held-out ASL letters: cross-signer few-shot
accuracy (support from signer A, query from signer B) and the intra-pose vs inter-pose cosine
distance gap. Writes reports/backbone_v1_eval.md, reports/backbone_v1_eval.json (consumed by
scripts/export_backbone.py), and reports/backbone_v1_distances.png.

Usage: python scripts/eval_backbone.py --config configs/train_v1.yaml
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import keras
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

from pgvb.train.dataset import load_split
from pgvb.train.episodes import EpisodeSampler
from pgvb.train.model import L2Normalize  # noqa: F401  registers the custom layer for loading

_MODEL_PATH = Path("models/backbone_v1.keras")
_MD_OUT = Path("reports/backbone_v1_eval.md")
_JSON_OUT = Path("reports/backbone_v1_eval.json")
_PNG_OUT = Path("reports/backbone_v1_distances.png")
_N_EPISODES = 500


def _run_episodes(sampler: EpisodeSampler, model, k_shot: int, n_episodes: int) -> float:
    correct, total = 0, 0
    for _ in range(n_episodes):
        support_x, support_y, query_x, query_y = sampler.sample_episode(apply_augment=False)
        support_emb = model(support_x, training=False).numpy()
        query_emb = model(query_x, training=False).numpy()

        protos = []
        for c in range(sampler.n_way):
            class_emb = support_emb[support_y == c]
            proto = class_emb.mean(axis=0)
            proto = proto / (np.linalg.norm(proto) + 1e-8)
            protos.append(proto)
        protos = np.stack(protos)

        dists = 1.0 - query_emb @ protos.T
        preds = dists.argmin(axis=1)
        correct += int((preds == query_y).sum())
        total += len(query_y)
    return correct / total


def _distance_stats(embeddings: np.ndarray, labels: np.ndarray) -> dict:
    sims = embeddings @ embeddings.T
    dists = 1.0 - sims
    iu = np.triu_indices(len(labels), k=1)
    pair_dists = dists[iu]
    same = labels[iu[0]] == labels[iu[1]]
    intra = pair_dists[same]
    inter = pair_dists[~same]
    return {
        "intra": intra,
        "inter": inter,
        "intra_mean": float(intra.mean()),
        "intra_p95": float(np.percentile(intra, 95)),
        "inter_mean": float(inter.mean()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--model", default=_MODEL_PATH, type=Path)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    _, heldout_split = load_split(cfg["data"]["landmarks_path"], cfg["data"]["heldout"])
    model = keras.models.load_model(args.model)

    rng = np.random.default_rng(cfg["seed"] + 1000)
    n_query = cfg["episode"]["n_query"]

    sampler_5shot = EpisodeSampler(heldout_split, n_way=5, k_shot=5, n_query=n_query, aug_cfg=None, rng=rng)
    sampler_1shot = EpisodeSampler(heldout_split, n_way=5, k_shot=1, n_query=n_query, aug_cfg=None, rng=rng)

    five_shot_acc = _run_episodes(sampler_5shot, model, k_shot=5, n_episodes=_N_EPISODES)
    one_shot_acc = _run_episodes(sampler_1shot, model, k_shot=1, n_episodes=_N_EPISODES)

    all_emb = model(heldout_split.x, training=False).numpy()
    stats = _distance_stats(all_emb, heldout_split.y)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(stats["intra"], bins=40, alpha=0.6, label="intra-pose", density=True)
    ax.hist(stats["inter"], bins=40, alpha=0.6, label="inter-pose", density=True)
    ax.set_xlabel("cosine distance")
    ax.set_ylabel("density")
    ax.set_title("Held-out letter embedding distances")
    ax.legend()
    fig.tight_layout()
    _PNG_OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(_PNG_OUT, dpi=150)
    plt.close(fig)

    summary = {
        "heldout_letters": cfg["data"]["heldout"],
        "n_episodes": _N_EPISODES,
        "five_shot_acc": five_shot_acc,
        "one_shot_acc": one_shot_acc,
        "intra_mean": stats["intra_mean"],
        "intra_p95": stats["intra_p95"],
        "inter_mean": stats["inter_mean"],
    }
    with open(_JSON_OUT, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
        f.write("\n")

    md = f"""# Backbone v1 held-out evaluation

Held-out letters (never seen during training): {", ".join(cfg["data"]["heldout"])}.
Support from signer A (`asl_train`), query from signer B (`asl_test`), {_N_EPISODES} episodes.

| Metric | Value | Target |
|---|---|---|
| 5-way 5-shot accuracy | {five_shot_acc:.3f} | >= 0.85 (target 0.90) |
| 5-way 1-shot accuracy | {one_shot_acc:.3f} | >= 0.70 |
| Mean intra-pose cosine distance | {stats["intra_mean"]:.4f} | lower than inter-pose |
| 95th percentile intra-pose distance | {stats["intra_p95"]:.4f} | |
| Mean inter-pose cosine distance | {stats["inter_mean"]:.4f} | higher than intra-pose |

![distance distributions](backbone_v1_distances.png)
"""
    _MD_OUT.write_text(md, encoding="utf-8")
    print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
