#!/usr/bin/env python
"""Simulates incremental enrollment (O4) on a participant's already-enrolled profile: gestures
are added back one at a time in the order they were originally enrolled (`created_at`), and
after each addition every already-enrolled gesture is re-evaluated on its own eval clips. Writes
an accuracy matrix (rows: number of gestures enrolled so far, columns: gesture) plus an
assertion that each earlier gesture's prototype is bit-identical to the one in the full profile
no matter how many later gestures were added, since a later `add_gesture` never touches it.

Usage:
  python scripts/eval_stability.py --profile profiles/aadit.json --dir data/sessions/eval/aadit \\
      --out reports/eval_stability_aadit
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from _eval_common import discover_gesture_clips, replay_clip

from pgvb.config import load
from pgvb.embedding import Backbone
from pgvb.profile import Profile, load as load_profile


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--profile", required=True, type=Path)
    parser.add_argument("--dir", required=True, type=Path)
    parser.add_argument("--out", default="reports/eval_stability")
    args = parser.parse_args()

    cfg = load()
    full_profile = load_profile(args.profile)
    backbone = Backbone.load(Path("models") / f"{cfg.backbone.id}.npz")

    ordered = sorted(full_profile.gestures, key=lambda g: g.created_at)
    if len(ordered) < 2:
        raise SystemExit(f"{args.profile} has fewer than 2 gestures; nothing to test stability across")

    reference_prototypes = {g.id: g.prototype.copy() for g in ordered}

    rows: list[dict[str, float | None]] = []
    mismatches: list[str] = []
    for k in range(1, len(ordered) + 1):
        enrolled_so_far = ordered[:k]
        partial_profile = Profile(
            user_id=full_profile.user_id, backbone_id=full_profile.backbone_id, gestures=enrolled_so_far
        )
        row: dict[str, float | None] = {"n_enrolled": k}
        for gesture in enrolled_so_far:
            if not np.array_equal(gesture.prototype, reference_prototypes[gesture.id]):
                mismatches.append(f"n_enrolled={k}: {gesture.name} prototype changed")
            clips = discover_gesture_clips(args.dir, gesture.name)
            if not clips:
                row[gesture.name] = None
                continue
            results = [replay_clip(cfg, backbone, partial_profile, path, gesture.name) for path in clips]
            row[gesture.name] = sum(r.correct for r in results) / len(results)
        rows.append(row)

    names = [g.name for g in ordered]
    lines = [
        f"# Stability evaluation: {args.profile}",
        "",
        f"Enrollment order (by `created_at`): {', '.join(names)}.",
        "",
        "## Accuracy matrix (rows: gestures enrolled so far, blank: not yet enrolled)",
        "",
        "| enrolled | " + " | ".join(names) + " |",
        "|---|" + "---|" * len(names),
    ]
    for row in rows:
        cells = []
        for name in names:
            value = row.get(name)
            cells.append(f"{value:.3f}" if isinstance(value, float) else "")
        lines.append(f"| {row['n_enrolled']} | " + " | ".join(cells) + " |")

    lines += ["", "## Prototype stability (O4)", ""]
    if mismatches:
        lines.append("FAILED: " + "; ".join(mismatches))
    else:
        lines.append(
            f"Passed: all {len(ordered)} gestures' prototypes were bit-identical to the fully "
            "enrolled profile at every earlier enrollment step."
        )

    md_path = Path(f"{args.out}.md")
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    if mismatches:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
