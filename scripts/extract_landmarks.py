"""Extracts normalized 63-d landmark features from an ASL Alphabet image folder (one
subfolder per class) and appends them to a landmarks .npz file.

Output arrays: x (N, 63) float32, y (N,) class name strings, source (N,) strings.
Skips the "nothing" class (no hand) and any subfolder with no images directly inside it
(the Kaggle test set ships a duplicate nested copy under a folder of its own). Images with
no detected hand are skipped and counted for the per-class detection rate. Running again
with the same --out and --source skips classes already present for that source, so a run
can be resumed after an interruption.

Usage:
  python scripts/extract_landmarks.py --images data/raw/asl_alphabet/asl_alphabet_train/asl_alphabet_train \\
      --out data/landmarks/asl_v1.npz --limit-per-class 500 --source asl_train
  python scripts/extract_landmarks.py --images data/raw/asl_alphabet_test \\
      --out data/landmarks/asl_v1.npz --source asl_test
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from pgvb.features import normalize_landmarks
from pgvb.landmarks import HandTracker

_IMAGE_EXTS = {".jpg", ".jpeg", ".png"}
_SKIP_CLASSES = {"nothing"}


def _list_classes(images_dir: Path) -> list[Path]:
    classes = []
    for entry in sorted(images_dir.iterdir()):
        if not entry.is_dir():
            continue
        if entry.name in _SKIP_CLASSES:
            continue
        has_direct_images = any(
            f.suffix.lower() in _IMAGE_EXTS for f in entry.iterdir() if f.is_file()
        )
        if not has_direct_images:
            print(f"skipping {entry.name}: no images directly inside")
            continue
        classes.append(entry)
    return classes


def _load_existing(out_path: Path) -> dict[str, np.ndarray]:
    if not out_path.exists():
        return {"x": np.zeros((0, 63), dtype=np.float32), "y": np.array([]), "source": np.array([])}
    data = np.load(out_path, allow_pickle=False)
    return {"x": data["x"], "y": data["y"], "source": data["source"]}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--limit-per-class", type=int, default=None)
    parser.add_argument("--source", required=True)
    args = parser.parse_args()

    tracker = HandTracker()
    existing = _load_existing(args.out)
    done_classes = set(
        existing["y"][existing["source"] == args.source].tolist()
    )

    new_x: list[np.ndarray] = []
    new_y: list[str] = []
    new_source: list[str] = []

    for class_dir in _list_classes(args.images):
        class_name = class_dir.name
        if class_name in done_classes:
            print(f"{class_name}: already extracted for source={args.source}, skipping")
            continue

        files = sorted(
            f for f in class_dir.iterdir() if f.is_file() and f.suffix.lower() in _IMAGE_EXTS
        )
        if args.limit_per_class is not None:
            files = files[: args.limit_per_class]

        detected = 0
        for f in files:
            bgr = cv2.imread(str(f))
            if bgr is None:
                continue
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            hand_frame = tracker.track_image(rgb)
            if hand_frame is None:
                continue
            feature = normalize_landmarks(hand_frame.landmarks, hand_frame.handedness)
            new_x.append(feature)
            new_y.append(class_name)
            new_source.append(args.source)
            detected += 1

        rate = detected / len(files) if files else 0.0
        print(f"{class_name}: {detected}/{len(files)} detected ({rate:.1%})")

    if new_x:
        combined_x = np.concatenate([existing["x"], np.stack(new_x)], axis=0)
        combined_y = np.concatenate([existing["y"], np.array(new_y)], axis=0)
        combined_source = np.concatenate([existing["source"], np.array(new_source)], axis=0)
    else:
        combined_x, combined_y, combined_source = existing["x"], existing["y"], existing["source"]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, x=combined_x, y=combined_y, source=combined_source)
    print(f"wrote {args.out}: {combined_x.shape[0]} total examples")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
