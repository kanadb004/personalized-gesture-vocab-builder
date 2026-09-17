"""Enrollment: collect stable frames for a new gesture, then commit it as a Gesture in the
profile (O2, few-shot, no retraining). Refinement: append one more example to an existing
gesture's prototype only, for the caregiver miss-flagging loop (document 5.5)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pgvb.config import EnrollConfig
from pgvb.embedding import Backbone
from pgvb.features import normalize_landmarks
from pgvb.landmarks import HandFrame
from pgvb.profile import Gesture, Profile
from pgvb.prototypes import cosine_distances, make_prototype


@dataclass
class SampleStatus:
    ready: bool
    reason: str | None = None


@dataclass
class Quality:
    intra_sample_distance: float
    closest_gesture: str | None
    closest_distance: float | None
    collision: bool


class Enroller:
    """One enrollment session: start(), feed frames with add_frame() until ready, accept_sample()
    to keep the current stable pose, repeat for min_samples to max_samples, then commit()."""

    def __init__(self, backbone: Backbone, profile: Profile, cfg: EnrollConfig) -> None:
        self.backbone = backbone
        self.profile = profile
        self.cfg = cfg
        self._name: str | None = None
        self._message: str | None = None
        self._examples: list[np.ndarray] = []
        self._recent_features: list[np.ndarray] = []

    def start(self, name: str, message: str) -> None:
        self._name = name
        self._message = message
        self._examples = []
        self._recent_features = []

    def add_frame(self, hand_frame: HandFrame | None) -> SampleStatus:
        """Feeds one frame. Rejects no-hand frames and resets the stability buffer. Ready
        becomes true once the last `stability_frames` frames' landmarks have settled below
        `stability_max_motion`."""
        if hand_frame is None:
            self._recent_features = []
            return SampleStatus(ready=False, reason="no_hand")

        feature = normalize_landmarks(hand_frame.landmarks, hand_frame.handedness)
        self._recent_features.append(feature)
        if len(self._recent_features) > self.cfg.stability_frames:
            self._recent_features.pop(0)

        if len(self._recent_features) < self.cfg.stability_frames:
            return SampleStatus(ready=False, reason="collecting")

        window = np.stack(self._recent_features)
        motion = float(np.linalg.norm(window - window.mean(axis=0), axis=1).max())
        if motion > self.cfg.stability_max_motion:
            return SampleStatus(ready=False, reason="moving")
        return SampleStatus(ready=True, reason=None)

    def accept_sample(self) -> None:
        """Embeds the current stable pose and keeps it as one collected example."""
        if len(self._recent_features) < self.cfg.stability_frames:
            raise RuntimeError("no stable sample to accept, call add_frame() until ready")
        feature = self._recent_features[-1]
        embedding = self.backbone.embed(feature)
        self._examples.append(embedding)
        self._recent_features = []

    @property
    def progress(self) -> tuple[int, int]:
        return len(self._examples), self.cfg.max_samples

    def quality(self) -> Quality:
        """Intra-sample consistency of the examples collected so far, plus a collision warning
        against the closest existing gesture in the profile."""
        if len(self._examples) < 2:
            intra = 0.0
        else:
            examples = np.stack(self._examples)
            proto = make_prototype(examples)
            intra = float(cosine_distances(proto, examples).mean())

        closest_gesture = None
        closest_distance = None
        if self._examples and self.profile.gestures:
            proto = make_prototype(np.stack(self._examples))
            protos = np.stack([g.prototype for g in self.profile.gestures])
            distances = cosine_distances(proto, protos)
            idx = int(np.argmin(distances))
            closest_gesture = self.profile.gestures[idx].name
            closest_distance = float(distances[idx])

        collision = closest_distance is not None and closest_distance < self.cfg.collision_distance
        return Quality(
            intra_sample_distance=intra,
            closest_gesture=closest_gesture,
            closest_distance=closest_distance,
            collision=collision,
        )

    def commit(self) -> Gesture:
        if self._name is None:
            raise RuntimeError("start() was not called")
        if len(self._examples) < self.cfg.min_samples:
            raise RuntimeError(
                f"only {len(self._examples)} samples collected, need at least "
                f"{self.cfg.min_samples}"
            )
        gesture = self.profile.add_gesture(self._name, self._message, self._examples)
        self.cancel()
        return gesture

    def cancel(self) -> None:
        self._name = None
        self._message = None
        self._examples = []
        self._recent_features = []


class Refiner:
    """Appends one example to an existing gesture and recomputes only that gesture's
    prototype, leaving every other gesture untouched (O4)."""

    def __init__(self, backbone: Backbone, profile: Profile) -> None:
        self.backbone = backbone
        self.profile = profile
        self._last_refined: dict[str, np.ndarray] = {}

    def refine(self, gesture_id: str, hand_frame: HandFrame) -> None:
        gesture = self.profile.find(gesture_id)
        if gesture is None:
            raise KeyError(gesture_id)
        feature = normalize_landmarks(hand_frame.landmarks, hand_frame.handedness)
        embedding = self.backbone.embed(feature)
        gesture.examples.append(embedding)
        gesture.prototype = make_prototype(np.stack(gesture.examples))
        gesture.n_refinements += 1
        self._last_refined[gesture_id] = embedding

    def undo_last(self, gesture_id: str) -> None:
        gesture = self.profile.find(gesture_id)
        if gesture is None:
            raise KeyError(gesture_id)
        if gesture_id not in self._last_refined or not gesture.examples:
            raise RuntimeError("nothing to undo")
        last_embedding = self._last_refined.pop(gesture_id)
        for i in range(len(gesture.examples) - 1, -1, -1):
            if np.array_equal(gesture.examples[i], last_embedding):
                del gesture.examples[i]
                break
        if gesture.examples:
            gesture.prototype = make_prototype(np.stack(gesture.examples))
        gesture.n_refinements = max(0, gesture.n_refinements - 1)
