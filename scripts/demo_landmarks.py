#!/usr/bin/env python
"""Live webcam demo: overlays the 21 hand landmarks and connections, handedness, and fps.

Hand tracking runs in a worker thread (Section 6 fallback for frame rate under 20 fps: the
HandLandmarker call itself takes about 25-30 ms, well under the camera's own rate, so the
capture/display loop reads and draws every frame while the tracker keeps up as fast as it can
on the most recent frame, instead of blocking the display loop on every call).

Keys: r toggles recording to data/sessions/<name>.npz, q quits.
"""

from __future__ import annotations

import argparse
import sys
import threading
import time

import cv2

from pgvb.camera import Camera, FpsMeter
from pgvb.config import load
from pgvb.landmarks import HandFrame, HandTracker
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


class _AsyncTracker:
    """Runs HandTracker.track() on a background thread against the most recently submitted
    frame, so a slow tracker call never blocks the capture/display loop."""

    def __init__(self, tracker: HandTracker) -> None:
        self._tracker = tracker
        self._lock = threading.Lock()
        self._pending: tuple[object, float] | None = None
        self._latest: HandFrame | None = None
        self._stop = False
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def submit(self, frame, ts_ms: float) -> None:
        with self._lock:
            self._pending = (frame.copy(), ts_ms)

    def latest(self) -> HandFrame | None:
        with self._lock:
            return self._latest

    def _run(self) -> None:
        last_ts = None
        while not self._stop:
            with self._lock:
                item = self._pending
            if item is None or item[1] == last_ts:
                time.sleep(0.001)
                continue
            frame, ts_ms = item
            last_ts = ts_ms
            result = self._tracker.track(frame, ts_ms)
            with self._lock:
                self._latest = result

    def stop(self) -> None:
        self._stop = True
        self._thread.join(timeout=1.0)


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
    async_tracker = _AsyncTracker(tracker)
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
        try:
            while True:
                result = cam.read()
                if result is None:
                    print("camera read failed", file=sys.stderr)
                    break
                frame, ts_ms = result

                async_tracker.submit(frame, ts_ms)
                hand_frame = async_tracker.latest()
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
        finally:
            async_tracker.stop()

    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
