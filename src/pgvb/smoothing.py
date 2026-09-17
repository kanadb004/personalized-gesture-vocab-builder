"""Sliding-window majority vote and the trigger state machine that turns per-frame recognition
decisions into rising-edge speech/display triggers."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass


class VoteWindow:
    """Keeps the last `window` per-frame labels (None allowed) and reports the confirmed
    label: the label with at least `min_votes` votes in the window, or None."""

    def __init__(self, window: int, min_votes: int) -> None:
        self.window = window
        self.min_votes = min_votes
        self._labels: deque[str | None] = deque(maxlen=window)

    def push(self, label: str | None) -> str | None:
        self._labels.append(label)
        counts: dict[str, int] = {}
        for lab in self._labels:
            if lab is None:
                continue
            counts[lab] = counts.get(lab, 0) + 1
        if not counts:
            return None
        best_label = max(counts, key=counts.get)
        if counts[best_label] >= self.min_votes:
            return best_label
        return None

    def clear(self) -> None:
        self._labels.clear()


@dataclass
class Trigger:
    label: str
    ts_ms: float


class TriggerStateMachine:
    """Fires a Trigger on the rising edge of a confirmed label only: a held gesture fires once.
    The same label can fire again once it has been absent (confirmed label differs) for at
    least `release_frames` consecutive frames, and no more often than every `cooldown_ms` even
    across separate holds."""

    def __init__(self, release_frames: int, cooldown_ms: float) -> None:
        self.release_frames = release_frames
        self.cooldown_ms = cooldown_ms
        self._held_label: str | None = None
        self._absence_streak = 0
        self._last_trigger_ts: dict[str, float] = {}

    def update(self, confirmed_label: str | None, ts_ms: float) -> Trigger | None:
        if confirmed_label is None:
            if self._held_label is not None:
                self._absence_streak += 1
                if self._absence_streak >= self.release_frames:
                    self._held_label = None
                    self._absence_streak = 0
            return None

        if confirmed_label == self._held_label:
            self._absence_streak = 0
            return None

        last_ts = self._last_trigger_ts.get(confirmed_label)
        if last_ts is not None and (ts_ms - last_ts) < self.cooldown_ms:
            return None

        self._held_label = confirmed_label
        self._absence_streak = 0
        self._last_trigger_ts[confirmed_label] = ts_ms
        return Trigger(label=confirmed_label, ts_ms=ts_ms)
