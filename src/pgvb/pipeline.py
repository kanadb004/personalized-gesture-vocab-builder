"""Glue: webcam frame -> landmarks -> features -> embedding -> decision -> smoothing -> trigger.
`process` takes a raw BGR frame (live camera, runs the tracker); `process_hand_frame` takes an
already-tracked HandFrame or None (session replay), skipping the tracking stage."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from pgvb.config import Config
from pgvb.embedding import Backbone
from pgvb.landmarks import HandFrame, HandTracker
from pgvb.profile import Profile
from pgvb.recognize import Decision, Recognizer
from pgvb.smoothing import Trigger, TriggerStateMachine, VoteWindow

_MODELS_DIR = Path(__file__).resolve().parents[2] / "models"


@dataclass
class FrameResult:
    hand_frame: HandFrame | None
    decision: Decision
    confirmed_label: str | None
    trigger: Trigger | None
    timings_ms: dict[str, float] = field(default_factory=dict)


class Pipeline:
    def __init__(self, cfg: Config, profile: Profile, backbone: Backbone | None = None) -> None:
        self.cfg = cfg
        self.profile = profile
        self.backbone = backbone or Backbone.load(_MODELS_DIR / f"{cfg.backbone.id}.npz")
        self.recognizer = Recognizer(self.backbone, profile, cfg.recognize)
        self.vote_window = VoteWindow(cfg.smoothing.window, cfg.smoothing.min_votes)
        self.trigger_machine = TriggerStateMachine(
            cfg.smoothing.release_frames, cfg.smoothing.cooldown_ms
        )
        self._tracker: HandTracker | None = None

    def reload_profile(self) -> None:
        self.recognizer.reload_profile()

    def process(self, frame_bgr: np.ndarray, ts_ms: float) -> FrameResult:
        """Runs the full pipeline, including landmark tracking, on a raw camera frame."""
        if self._tracker is None:
            self._tracker = HandTracker(
                num_hands=self.cfg.landmarks.num_hands,
                min_hand_score=self.cfg.landmarks.min_hand_score,
                min_hand_detection_confidence=self.cfg.landmarks.min_hand_detection_confidence,
                min_tracking_confidence=self.cfg.landmarks.min_tracking_confidence,
            )
        t0 = time.perf_counter()
        hand_frame = self._tracker.track(frame_bgr, ts_ms)
        t1 = time.perf_counter()
        result = self._process_common(hand_frame, ts_ms)
        result.timings_ms["track_ms"] = (t1 - t0) * 1000.0
        return result

    def process_hand_frame(self, hand_frame: HandFrame | None, ts_ms: float) -> FrameResult:
        """Runs recognition, smoothing, and triggering on an already-tracked frame, for
        replaying recorded sessions without a camera."""
        return self._process_common(hand_frame, ts_ms)

    def _process_common(self, hand_frame: HandFrame | None, ts_ms: float) -> FrameResult:
        t0 = time.perf_counter()
        decision = self.recognizer.decide(hand_frame)
        t1 = time.perf_counter()
        confirmed_label = self.vote_window.push(decision.label)
        t2 = time.perf_counter()
        trigger = self.trigger_machine.update(confirmed_label, ts_ms)
        t3 = time.perf_counter()
        timings = {
            "recognize_ms": (t1 - t0) * 1000.0,
            "smooth_ms": (t2 - t1) * 1000.0,
            "trigger_ms": (t3 - t2) * 1000.0,
        }
        return FrameResult(
            hand_frame=hand_frame,
            decision=decision,
            confirmed_label=confirmed_label,
            trigger=trigger,
            timings_ms=timings,
        )
