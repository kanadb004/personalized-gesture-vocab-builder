#!/usr/bin/env python
"""Replays every recorded eval clip for one participant's profile through the full pipeline
(document 5.6): per-gesture clip accuracy (the first trigger in the clip is the correct label
and no wrong-label trigger occurs), a confusion matrix, false triggers per minute on the
non-gesture clips, and algorithmic latency (first correct raw per-frame decision to the
confirming trigger).

Gesture clips are discovered as `<dir>/<gesture_name>_*.npz` for each gesture in the profile;
non-gesture clips are discovered by glob pattern(s) passed with --nongesture.

Usage:
  python scripts/eval_recognition.py --profile profiles/aadit.json \\
      --dir data/sessions/eval/aadit --nongesture "nongesture*.npz" \\
      --out reports/eval_recognition_aadit

  python scripts/eval_recognition.py --profile profiles/example.json \\
      --dir data/sessions/phase3 --nongesture "idle*.npz" "unrelated*.npz" \\
      --out reports/eval_recognition_phase3_smoke
"""

from __future__ import annotations

import argparse
import csv
import statistics
from pathlib import Path

from _eval_common import ClipResult, discover_gesture_clips, discover_pattern_clips, replay_clip

from pgvb.config import load
from pgvb.embedding import Backbone
from pgvb.profile import load as load_profile


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--profile", required=True, type=Path)
    parser.add_argument("--dir", required=True, type=Path)
    parser.add_argument("--nongesture", required=True, nargs="+", help="glob pattern(s) for non-gesture clips")
    parser.add_argument("--out", default="reports/eval_recognition", help="output path prefix")
    args = parser.parse_args()

    cfg = load()
    profile = load_profile(args.profile)
    backbone = Backbone.load(Path("models") / f"{cfg.backbone.id}.npz")

    gesture_results: list[ClipResult] = []
    for gesture in profile.gestures:
        for path in discover_gesture_clips(args.dir, gesture.name):
            gesture_results.append(replay_clip(cfg, backbone, profile, path, gesture.name))

    nongesture_paths = discover_pattern_clips(args.dir, args.nongesture)
    nongesture_results = [replay_clip(cfg, backbone, profile, path, None) for path in nongesture_paths]

    if not gesture_results:
        raise SystemExit(f"no gesture clips found under {args.dir} for profile {args.profile}")

    csv_path = Path(f"{args.out}.csv")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["clip", "expected", "n_frames", "n_triggers", "first_trigger", "correct", "algorithmic_latency_ms"]
        )
        for r in gesture_results + nongesture_results:
            writer.writerow(
                [
                    r.path.name,
                    r.expected_label or "",
                    r.n_frames,
                    r.n_triggers,
                    r.trigger_labels[0] if r.trigger_labels else "",
                    r.correct,
                    f"{r.algorithmic_latency_ms:.1f}" if r.algorithmic_latency_ms is not None else "",
                ]
            )

    gesture_names = [g.name for g in profile.gestures]
    per_gesture_acc = {}
    for name in gesture_names:
        clips = [r for r in gesture_results if r.expected_label == name]
        if clips:
            per_gesture_acc[name] = sum(r.correct for r in clips) / len(clips)
    overall_acc = sum(r.correct for r in gesture_results) / len(gesture_results)

    confusion: dict[tuple[str, str], int] = {}
    for r in gesture_results:
        predicted = r.trigger_labels[0] if r.trigger_labels else "none"
        confusion[(r.expected_label, predicted)] = confusion.get((r.expected_label, predicted), 0) + 1

    total_nongesture_ms = sum(r.duration_ms for r in nongesture_results)
    total_false_triggers = sum(r.n_triggers for r in nongesture_results)
    false_triggers_per_min = (
        total_false_triggers / (total_nongesture_ms / 60000.0) if total_nongesture_ms > 0 else float("nan")
    )

    latencies = [r.algorithmic_latency_ms for r in gesture_results if r.algorithmic_latency_ms is not None]
    latency_median = statistics.median(latencies) if latencies else float("nan")

    lines = [
        f"# Recognition evaluation: {args.profile}",
        "",
        f"Profile: `{args.profile}`. Sessions: `{args.dir}`. {len(gesture_results)} gesture clips, "
        f"{len(nongesture_results)} non-gesture clips.",
        "",
        "## Per-gesture clip accuracy",
        "",
        "| gesture | clips | accuracy |",
        "|---|---|---|",
    ]
    for name in gesture_names:
        clips = [r for r in gesture_results if r.expected_label == name]
        acc = per_gesture_acc.get(name)
        lines.append(f"| {name} | {len(clips)} | {acc:.3f} |" if acc is not None else f"| {name} | 0 | n/a |")
    lines.append(f"| **overall** | {len(gesture_results)} | {overall_acc:.3f} |")

    lines += [
        "",
        "## Confusion matrix (rows: true gesture, columns: predicted label of first trigger)",
        "",
        "| true \\ predicted | " + " | ".join(gesture_names + ["none"]) + " |",
        "|---|" + "---|" * (len(gesture_names) + 1),
    ]
    for name in gesture_names:
        row = [str(confusion.get((name, pred), 0)) for pred in gesture_names + ["none"]]
        lines.append(f"| {name} | " + " | ".join(row) + " |")

    lines += [
        "",
        "## Non-gesture clips",
        "",
        f"Total duration: {total_nongesture_ms / 1000.0:.1f} s. Total false triggers: "
        f"{total_false_triggers}. False triggers per minute: {false_triggers_per_min:.3f}.",
        "",
        "## Algorithmic latency",
        "",
        f"Median over {len(latencies)} correctly triggered clips: {latency_median:.1f} ms.",
        "",
        f"Full per-clip results in `{csv_path}`.",
    ]
    md_path = Path(f"{args.out}.md")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
