#!/usr/bin/env python
"""Renders the confusion matrix, latency histogram, and stability heatmap for one participant's
eval clips as PNGs under reports/img/. Recomputes from the same recorded clips eval_recognition.py
and eval_stability.py use, so the images stay consistent with the tables in reports/results.md.
Droppable per the plan's over-budget fallback (tables alone still satisfy the DoD).

Usage:
  python scripts/make_figures.py --profile profiles/aadit.json --dir data/sessions/eval/aadit \\
      --nongesture "nongesture*.npz" --out-prefix reports/img/aadit
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from _eval_common import discover_gesture_clips, discover_pattern_clips, replay_clip

from pgvb.config import load
from pgvb.embedding import Backbone
from pgvb.profile import Profile, load as load_profile


def _confusion_matrix(cfg, backbone, profile, args) -> tuple[list[str], np.ndarray]:
    names = [g.name for g in profile.gestures] + ["none"]
    matrix = np.zeros((len(names) - 1, len(names)), dtype=int)
    for i, gesture in enumerate(profile.gestures):
        for path in discover_gesture_clips(args.dir, gesture.name):
            result = replay_clip(cfg, backbone, profile, path, gesture.name)
            predicted = result.trigger_labels[0] if result.trigger_labels else "none"
            matrix[i, names.index(predicted)] += 1
    return names, matrix


def _latencies(cfg, backbone, profile, args) -> list[float]:
    latencies = []
    for gesture in profile.gestures:
        for path in discover_gesture_clips(args.dir, gesture.name):
            result = replay_clip(cfg, backbone, profile, path, gesture.name)
            if result.algorithmic_latency_ms is not None:
                latencies.append(result.algorithmic_latency_ms)
    return latencies


def _stability_matrix(cfg, backbone, profile, directory: Path) -> tuple[list[str], np.ndarray]:
    ordered = sorted(profile.gestures, key=lambda g: g.created_at)
    names = [g.name for g in ordered]
    matrix = np.full((len(ordered), len(ordered)), np.nan)
    for k in range(1, len(ordered) + 1):
        partial = Profile(user_id=profile.user_id, backbone_id=profile.backbone_id, gestures=ordered[:k])
        for j, gesture in enumerate(ordered[:k]):
            clips = discover_gesture_clips(directory, gesture.name)
            if not clips:
                continue
            results = [replay_clip(cfg, backbone, partial, path, gesture.name) for path in clips]
            matrix[k - 1, j] = sum(r.correct for r in results) / len(results)
    return names, matrix


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--profile", required=True, type=Path)
    parser.add_argument("--dir", required=True, type=Path)
    parser.add_argument("--nongesture", nargs="+", default=[])
    parser.add_argument("--out-prefix", required=True)
    args = parser.parse_args()

    cfg = load()
    profile = load_profile(args.profile)
    backbone = Backbone.load(Path("models") / f"{cfg.backbone.id}.npz")
    out_prefix = Path(args.out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    names, matrix = _confusion_matrix(cfg, backbone, profile, args)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha="right")
    ax.set_yticks(range(len(names) - 1))
    ax.set_yticklabels(names[:-1])
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, str(matrix[i, j]), ha="center", va="center")
    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(f"{out_prefix}_confusion.png", dpi=150)
    plt.close(fig)

    latencies = _latencies(cfg, backbone, profile, args)
    fig, ax = plt.subplots(figsize=(5, 4))
    if latencies:
        ax.hist(latencies, bins=10)
    ax.set_xlabel("algorithmic latency (ms)")
    ax.set_ylabel("count")
    fig.tight_layout()
    fig.savefig(f"{out_prefix}_latency.png", dpi=150)
    plt.close(fig)

    stability_names, stability_matrix = _stability_matrix(cfg, backbone, profile, args.dir)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(stability_matrix, cmap="Greens", vmin=0, vmax=1)
    ax.set_xticks(range(len(stability_names)))
    ax.set_xticklabels(stability_names, rotation=45, ha="right")
    ax.set_yticks(range(len(stability_names)))
    ax.set_yticklabels(range(1, len(stability_names) + 1))
    ax.set_xlabel("gesture")
    ax.set_ylabel("gestures enrolled")
    for i in range(stability_matrix.shape[0]):
        for j in range(stability_matrix.shape[1]):
            value = stability_matrix[i, j]
            if not np.isnan(value):
                ax.text(j, i, f"{value:.2f}", ha="center", va="center")
    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(f"{out_prefix}_stability.png", dpi=150)
    plt.close(fig)

    print(f"wrote {out_prefix}_confusion.png, {out_prefix}_latency.png, {out_prefix}_stability.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
