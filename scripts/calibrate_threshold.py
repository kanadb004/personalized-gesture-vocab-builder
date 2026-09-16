"""Calibrates `recognize.max_distance` on the five held-out ASL letters (the never-seen poses
from Phase 2), simulating a few-shot enrollment: over many random trials, enroll 3 of the 5
held-out letters from 8 signer A samples each, then query with signer B samples of the enrolled
letters (should accept, correct label) and of the 2 unenrolled letters (should reject, an
accept there is a false accept). Sweeps `max_distance` and reports enrolled accuracy and false
accept rate at each value; picks the largest threshold with false accept rate at most 5 percent.

Writes reports/threshold_sweep.csv, reports/threshold_sweep.png, reports/threshold_sweep.md.
Does not modify configs/default.yaml; the chosen value is written there by hand in the same PR
(see docs/phases/phase-3.md) so the numbers used in the config are the ones this script reports.

Usage: python scripts/calibrate_threshold.py --config configs/train_v1.yaml
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

from pgvb.embedding import Backbone

_LANDMARKS_PATH = Path("data/landmarks/asl_v1.npz")
_BACKBONE_PATH = Path("models/backbone_v1.npz")
_CSV_OUT = Path("reports/threshold_sweep.csv")
_PNG_OUT = Path("reports/threshold_sweep.png")
_MD_OUT = Path("reports/threshold_sweep.md")

_N_TRIALS = 200
_N_ENROLL_SAMPLES = 8
_N_ENROLLED_LETTERS = 3
_THRESHOLDS = np.round(np.arange(0.02, 0.40 + 1e-9, 0.01), 2)
_MAX_FALSE_ACCEPT_RATE = 0.05


def _run_trial(
    rng: np.random.Generator,
    heldout_letters: list[str],
    by_letter_a: dict[str, np.ndarray],
    by_letter_b: dict[str, np.ndarray],
    thresholds: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Returns (accepted_correct, false_accepts) each shape (len(thresholds),), summed over
    this trial's enrolled-letter and unenrolled-letter queries respectively."""
    enrolled = list(rng.choice(heldout_letters, size=_N_ENROLLED_LETTERS, replace=False))
    unenrolled = [letter for letter in heldout_letters if letter not in enrolled]

    protos = []
    for letter in enrolled:
        pool = by_letter_a[letter]
        idx = rng.choice(len(pool), size=_N_ENROLL_SAMPLES, replace=False)
        support = pool[idx]
        proto = support.mean(axis=0)
        proto = proto / (np.linalg.norm(proto) + 1e-8)
        protos.append(proto)
    protos = np.stack(protos)

    enrolled_query = np.concatenate([by_letter_b[letter] for letter in enrolled], axis=0)
    enrolled_query_labels = np.concatenate(
        [np.full(len(by_letter_b[letter]), letter) for letter in enrolled]
    )
    unenrolled_query = np.concatenate([by_letter_b[letter] for letter in unenrolled], axis=0)

    def _nearest(query: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        distances = 1.0 - query @ protos.T
        return distances.min(axis=1), distances.argmin(axis=1)

    enrolled_min_dist, enrolled_argmin = _nearest(enrolled_query)
    accepted = enrolled_min_dist[None, :] <= thresholds[:, None]
    predicted = np.array(enrolled)[enrolled_argmin]
    correct = predicted[None, :] == enrolled_query_labels[None, :]
    accepted_correct = (accepted & correct).sum(axis=1)

    unenrolled_min_dist, _ = _nearest(unenrolled_query)
    false_accepts = (unenrolled_min_dist[None, :] <= thresholds[:, None]).sum(axis=1)

    return (
        accepted_correct,
        false_accepts,
        len(enrolled_query),
        len(unenrolled_query),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    heldout_letters = cfg["data"]["heldout"]

    backbone = Backbone.load(_BACKBONE_PATH)
    data = np.load(_LANDMARKS_PATH)
    x, y, source = data["x"], data["y"], data["source"]

    by_letter_a: dict[str, np.ndarray] = {}
    by_letter_b: dict[str, np.ndarray] = {}
    for letter in heldout_letters:
        mask_a = (y == letter) & (source == "asl_train")
        mask_b = (y == letter) & (source == "asl_test")
        by_letter_a[letter] = backbone.embed_batch(x[mask_a])
        by_letter_b[letter] = backbone.embed_batch(x[mask_b])

    rng = np.random.default_rng(args.seed)
    total_correct = np.zeros(len(_THRESHOLDS))
    total_false_accepts = np.zeros(len(_THRESHOLDS))
    total_enrolled_queries = 0
    total_unenrolled_queries = 0

    for _ in range(_N_TRIALS):
        correct, false_accepts, n_enrolled_q, n_unenrolled_q = _run_trial(
            rng, heldout_letters, by_letter_a, by_letter_b, _THRESHOLDS
        )
        total_correct += correct
        total_false_accepts += false_accepts
        total_enrolled_queries += n_enrolled_q
        total_unenrolled_queries += n_unenrolled_q

    enrolled_accuracy = total_correct / total_enrolled_queries
    false_accept_rate = total_false_accepts / total_unenrolled_queries

    _CSV_OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(_CSV_OUT, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["max_distance", "enrolled_accuracy", "false_accept_rate"])
        for threshold, acc, far in zip(_THRESHOLDS, enrolled_accuracy, false_accept_rate):
            writer.writerow([f"{threshold:.2f}", f"{acc:.4f}", f"{far:.4f}"])

    valid = np.where(false_accept_rate <= _MAX_FALSE_ACCEPT_RATE)[0]
    if len(valid) == 0:
        chosen_idx = int(np.argmin(false_accept_rate))
    else:
        chosen_idx = int(valid[np.argmax(_THRESHOLDS[valid])])
    chosen_threshold = float(_THRESHOLDS[chosen_idx])
    chosen_accuracy = float(enrolled_accuracy[chosen_idx])
    chosen_far = float(false_accept_rate[chosen_idx])

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(_THRESHOLDS, enrolled_accuracy, label="enrolled accuracy", marker=".")
    ax.plot(_THRESHOLDS, false_accept_rate, label="false accept rate", marker=".")
    ax.axvline(chosen_threshold, color="gray", linestyle="--", label=f"chosen = {chosen_threshold:.2f}")
    ax.set_xlabel("max_distance")
    ax.set_ylabel("rate")
    ax.set_title("Threshold sweep on held-out letters")
    ax.legend()
    fig.tight_layout()
    fig.savefig(_PNG_OUT, dpi=150)
    plt.close(fig)

    md = f"""# Threshold sweep (Phase 3)

Held-out letters: {", ".join(heldout_letters)}. {_N_TRIALS} trials, each enrolling
{_N_ENROLLED_LETTERS} of the 5 letters from {_N_ENROLL_SAMPLES} signer A samples, querying with
signer B samples of the enrolled letters (correct-accept target) and of the remaining
{len(heldout_letters) - _N_ENROLLED_LETTERS} unenrolled letters (false-accept probes).

Chosen `recognize.max_distance`: **{chosen_threshold:.2f}**
(enrolled accuracy {chosen_accuracy:.3f}, false accept rate {chosen_far:.3f},
target false accept rate <= {_MAX_FALSE_ACCEPT_RATE:.2f}).

Full sweep in `reports/threshold_sweep.csv` and `reports/threshold_sweep.png`.
"""
    _MD_OUT.write_text(md, encoding="utf-8")
    print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
