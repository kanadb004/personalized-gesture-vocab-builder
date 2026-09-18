#!/usr/bin/env python
"""Headless session recorder: counts down 3 seconds, then records --seconds of landmark
frames from the live camera and saves them to a .npz for offline replay."""

from __future__ import annotations

import argparse
import sys
import time

from pgvb.camera import Camera
from pgvb.config import load
from pgvb.landmarks import HandTracker
from pgvb.replay import SessionRecorder


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", required=True, help="gesture or session label")
    parser.add_argument("--seconds", type=float, required=True, help="recording duration")
    parser.add_argument("--out", default=None, help="output .npz path")
    parser.add_argument("--participant", default=None, help="participant name, for Phase 5")
    parser.add_argument("--gesture", default=None, help="gesture name, for Phase 5")
    parser.add_argument(
        "--clip", type=int, default=None, help="clip index, appended as _<clip>, for Phase 5 (e.g. 1, 2, 3)"
    )
    args = parser.parse_args()

    cfg = load()
    tracker = HandTracker(
        num_hands=cfg.landmarks.num_hands,
        min_hand_score=cfg.landmarks.min_hand_score,
        min_hand_detection_confidence=cfg.landmarks.min_hand_detection_confidence,
        min_tracking_confidence=cfg.landmarks.min_tracking_confidence,
    )

    if args.out:
        out_path = args.out
    elif args.participant and args.gesture:
        suffix = f"_{args.clip}" if args.clip is not None else ""
        out_path = f"data/sessions/eval/{args.participant}/{args.gesture}{suffix}.npz"
    else:
        out_path = f"data/sessions/sample_{args.label}.npz"

    for remaining in (3, 2, 1):
        print(f"recording in {remaining}...")
        time.sleep(1)
    print("recording")

    recorder = SessionRecorder(label=args.label)
    with Camera(
        index=cfg.camera.index,
        width=cfg.camera.width,
        height=cfg.camera.height,
        mirror=cfg.camera.mirror,
    ) as cam:
        start = time.monotonic()
        while time.monotonic() - start < args.seconds:
            result = cam.read()
            if result is None:
                print("camera read failed", file=sys.stderr)
                break
            frame, ts_ms = result
            hand_frame = tracker.track(frame, ts_ms)
            recorder.append(hand_frame, ts_ms)

    recorder.save(out_path)
    print(f"saved {out_path} ({len(recorder)} frames)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
