"""Records and replays streams of HandFrame (or no-hand) per-frame results to/from .npz.

File format (arrays in the .npz, one row per recorded frame):
  ts_ms       (N,) float64      timestamp of the frame, milliseconds
  has_hand    (N,) bool         whether a hand was detected on this frame
  landmarks   (N, 21, 3) float32  raw landmarks; zero-filled where has_hand is False
  handedness  (N,) unicode str  "Left" or "Right"; empty string where has_hand is False
  score       (N,) float32      handedness score; 0.0 where has_hand is False
  label       ()   unicode str  0-d array holding the session label
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from pgvb.landmarks import HandFrame


class SessionRecorder:
    """Appends one HandFrame (or None for no hand) per frame, then saves to .npz."""

    def __init__(self, label: str) -> None:
        self.label = label
        self._ts_ms: list[float] = []
        self._has_hand: list[bool] = []
        self._landmarks: list[np.ndarray] = []
        self._handedness: list[str] = []
        self._score: list[float] = []

    def append(self, hand_frame: HandFrame | None, ts_ms: float) -> None:
        self._ts_ms.append(ts_ms)
        if hand_frame is None:
            self._has_hand.append(False)
            self._landmarks.append(np.zeros((21, 3), dtype=np.float32))
            self._handedness.append("")
            self._score.append(0.0)
        else:
            self._has_hand.append(True)
            self._landmarks.append(hand_frame.landmarks.astype(np.float32))
            self._handedness.append(hand_frame.handedness)
            self._score.append(float(hand_frame.score))

    def __len__(self) -> int:
        return len(self._ts_ms)

    def save(self, path: str | Path) -> None:
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            out_path,
            ts_ms=np.array(self._ts_ms, dtype=np.float64),
            has_hand=np.array(self._has_hand, dtype=bool),
            landmarks=np.stack(self._landmarks).astype(np.float32)
            if self._landmarks
            else np.zeros((0, 21, 3), dtype=np.float32),
            handedness=np.array(self._handedness, dtype=str),
            score=np.array(self._score, dtype=np.float32),
            label=np.array(self.label),
        )


class SessionPlayer:
    """Loads a session saved by SessionRecorder and iterates it back as (HandFrame|None, ts_ms)."""

    def __init__(self, path: str | Path) -> None:
        data = np.load(Path(path), allow_pickle=False)
        self.label = str(data["label"])
        self._ts_ms = data["ts_ms"]
        self._has_hand = data["has_hand"]
        self._landmarks = data["landmarks"]
        self._handedness = data["handedness"]
        self._score = data["score"]

    def __len__(self) -> int:
        return len(self._ts_ms)

    def __iter__(self):
        for i in range(len(self)):
            ts_ms = float(self._ts_ms[i])
            if not self._has_hand[i]:
                yield None, ts_ms
                continue
            yield (
                HandFrame(
                    landmarks=self._landmarks[i],
                    handedness=str(self._handedness[i]),
                    score=float(self._score[i]),
                    ts_ms=ts_ms,
                ),
                ts_ms,
            )
