#!/usr/bin/env python
"""Ensures models/hand_landmarker.task is present. The file is already on this machine
(see docs/OFFLINE.md); this script never re-downloads an existing model of the right size
and must not touch the network when the file is already there."""

from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)
EXPECTED_SIZE = 7_819_105
MODELS_DIR = Path(__file__).resolve().parents[1] / "models"
MODEL_PATH = MODELS_DIR / "hand_landmarker.task"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args(argv)

    if MODEL_PATH.exists():
        size = MODEL_PATH.stat().st_size
        if size == EXPECTED_SIZE:
            print(f"{MODEL_PATH} already present ({size} bytes), skipping download.")
            return 0
        print(
            f"{MODEL_PATH} exists but is {size} bytes, expected {EXPECTED_SIZE}. "
            "Refusing to overwrite; remove it manually if it is corrupt.",
            file=sys.stderr,
        )
        return 1

    print(f"{MODEL_PATH} not found, downloading from {MODEL_URL} ...")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    try:
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    except OSError as exc:
        print(
            f"Could not download {MODEL_URL}: {exc}. "
            "This project must not require a download after 2026-09-13; "
            "see docs/OFFLINE.md.",
            file=sys.stderr,
        )
        return 1

    size = MODEL_PATH.stat().st_size
    if size != EXPECTED_SIZE:
        print(
            f"Downloaded file is {size} bytes, expected {EXPECTED_SIZE}.",
            file=sys.stderr,
        )
        return 1
    print(f"Downloaded {MODEL_PATH} ({size} bytes).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
