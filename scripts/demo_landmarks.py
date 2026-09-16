#!/usr/bin/env python
"""Live webcam demo: overlays the 21 hand landmarks and connections, handedness, and fps.

Keys: r toggles recording to data/sessions/<name>.npz, q quits.
"""

from __future__ import annotations

import argparse
import sys

import cv2

from pgvb.camera import Camera, FpsMeter
from pgvb.config import load
from pgvb.landmarks import HandTracker
from pgvb.replay import SessionRecorder

# MediaPipe's 21-landmark hand skeleton connections (index pairs).
_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]


def _draw_overlay(frame, hand_frame, fps: float) -> None:
    h, w = frame.shape[:2]
    if hand_frame is not None:
        pts = [(int(x * w), int(y * h)) for x, y, _ in hand_frame.landmarks]
        for a, b in _CONNECTIONS:
            cv2.line(frame, pts[a], pts[b], (0, 255, 0), 2)
        for x, y in pts:
            cv2.circle(frame, (x, y), 4, (0, 0, 255), -1)
        label = f"{hand_frame.handedness} ({hand_frame.score:.2f})"
    else:
        label = "no hand"
    cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(
        frame, f"fps: {fps:.1f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", default="live", help="base name for recordings started with r")
    args = parser.parse_args()

    cfg = load()
    tracker = HandTracker(
        num_hands=cfg.landmarks.num_hands,
        min_hand_score=cfg.landmarks.min_hand_score,
        min_hand_detection_confidence=cfg.landmarks.min_hand_detection_confidence,
        min_tracking_confidence=cfg.landmarks.min_tracking_confidence,
    )
    fps_meter = FpsMeter()
    recorder: SessionRecorder | None = None
    record_index = 0

    with Camera(
        index=cfg.camera.index,
        width=cfg.camera.width,
        height=cfg.camera.height,
        mirror=cfg.camera.mirror,
    ) as cam:
        print("press r to toggle recording, q to quit")
        while True:
            result = cam.read()
            if result is None:
                print("camera read failed", file=sys.stderr)
                break
            frame, ts_ms = result

            hand_frame = tracker.track(frame, ts_ms)
            fps = fps_meter.tick()

            if recorder is not None:
                recorder.append(hand_frame, ts_ms)

            _draw_overlay(frame, hand_frame, fps)
            if recorder is not None:
                cv2.putText(
                    frame, f"REC ({len(recorder)})", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2,
                )
            cv2.imshow("pgvb landmarks", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("r"):
                if recorder is None:
                    recorder = SessionRecorder(label=args.name)
                    print("recording started")
                else:
                    out_path = f"data/sessions/{args.name}_{record_index}.npz"
                    recorder.save(out_path)
                    print(f"saved {out_path} ({len(recorder)} frames)")
                    record_index += 1
                    recorder = None

    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
