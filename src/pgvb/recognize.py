"""Per-frame recognition: nearest-prototype decision with an open-set distance threshold and
an optional margin requirement (document 5.1, objectives O2/O3)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pgvb.config import RecognizeConfig
from pgvb.embedding import Backbone
from pgvb.features import normalize_landmarks
from pgvb.landmarks import HandFrame
from pgvb.profile import Profile
from pgvb.prototypes import nearest


@dataclass
class Decision:
    label: str | None
    distance: float | None
    margin: float | None
    embedding: np.ndarray | None


class Recognizer:
    def __init__(self, backbone: Backbone, profile: Profile, cfg: RecognizeConfig) -> None:
        self.backbone = backbone
        self.profile = profile
        self.cfg = cfg
        self._prototypes: np.ndarray | None = None
        self._labels: list[str] = []
        self.reload_profile()

    def reload_profile(self) -> None:
        """Re-reads prototypes from the profile. Call after enrollment or refinement changes
        the set or content of gestures."""
        gestures = self.profile.gestures
        self._labels = [g.name for g in gestures]
        self._prototypes = np.stack([g.prototype for g in gestures]) if gestures else None

    def decide(self, hand_frame: HandFrame | None) -> Decision:
        if hand_frame is None or self._prototypes is None:
            return Decision(label=None, distance=None, margin=None, embedding=None)

        feature = normalize_landmarks(hand_frame.landmarks, hand_frame.handedness)
        embedding = self.backbone.embed(feature)
        idx, d1, d2 = nearest(embedding, self._prototypes)
        margin = d2 - d1

        if d1 > self.cfg.max_distance:
            return Decision(label=None, distance=d1, margin=margin, embedding=embedding)
        if self.cfg.min_margin > 0.0 and margin < self.cfg.min_margin:
            return Decision(label=None, distance=d1, margin=margin, embedding=embedding)

        return Decision(label=self._labels[idx], distance=d1, margin=margin, embedding=embedding)
