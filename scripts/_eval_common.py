"""Shared clip-replay logic for eval_recognition.py and eval_stability.py. Not a pgvb package
module since it only exists to avoid duplicating this between the two eval scripts."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from pgvb.config import Config
from pgvb.embedding import Backbone
from pgvb.pipeline import Pipeline
from pgvb.profile import Profile
from pgvb.replay import SessionPlayer


@dataclass
class ClipResult:
    path: Path
    expected_label: str | None  # None for a non-gesture clip
    n_frames: int
    duration_ms: float
    trigger_labels: list[str] = field(default_factory=list)
    trigger_ts_ms: list[float] = field(default_factory=list)
    first_correct_raw_ts_ms: float | None = None

    @property
    def n_triggers(self) -> int:
        return len(self.trigger_labels)

    @property
    def wrong_trigger(self) -> bool:
        return any(label != self.expected_label for label in self.trigger_labels)

    @property
    def correct(self) -> bool:
        """Only meaningful for a gesture clip: the first trigger is the right label and no
        trigger in the clip carries the wrong label."""
        if self.expected_label is None:
            return not self.trigger_labels
        return (
            bool(self.trigger_labels)
            and self.trigger_labels[0] == self.expected_label
            and not self.wrong_trigger
        )

    @property
    def algorithmic_latency_ms(self) -> float | None:
        """First correct raw per-frame decision to the first (correct) trigger."""
        if self.expected_label is None or not self.correct or self.first_correct_raw_ts_ms is None:
            return None
        return self.trigger_ts_ms[0] - self.first_correct_raw_ts_ms


def replay_clip(cfg: Config, backbone: Backbone, profile: Profile, path: Path, expected_label: str | None) -> ClipResult:
    pipeline = Pipeline(cfg, profile, backbone=backbone)
    player = SessionPlayer(path)
    result = ClipResult(path=path, expected_label=expected_label, n_frames=len(player), duration_ms=0.0)
    first_ts = None
    last_ts = None
    for hand_frame, ts_ms in player:
        if first_ts is None:
            first_ts = ts_ms
        last_ts = ts_ms
        frame_result = pipeline.process_hand_frame(hand_frame, ts_ms)
        if (
            expected_label is not None
            and result.first_correct_raw_ts_ms is None
            and frame_result.decision.label == expected_label
        ):
            result.first_correct_raw_ts_ms = ts_ms
        if frame_result.trigger is not None:
            result.trigger_labels.append(frame_result.trigger.label)
            result.trigger_ts_ms.append(frame_result.trigger.ts_ms)
    if first_ts is not None and last_ts is not None:
        result.duration_ms = last_ts - first_ts
    return result


def discover_gesture_clips(directory: Path, gesture_name: str) -> list[Path]:
    return sorted(directory.glob(f"{gesture_name}_*.npz"))


def discover_pattern_clips(directory: Path, patterns: list[str]) -> list[Path]:
    paths: list[Path] = []
    for pattern in patterns:
        paths.extend(sorted(directory.glob(pattern)))
    return paths
