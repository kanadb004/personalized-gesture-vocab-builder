"""Enrolling gesture N never changes recognition of gestures 1..N-1 (O4). Uses a small
deterministic fake backbone instead of the trained one so this test needs neither TensorFlow
nor a camera."""

from __future__ import annotations

import numpy as np

from pgvb.config import EnrollConfig, RecognizeConfig
from pgvb.enroll import Enroller
from pgvb.landmarks import HandFrame
from pgvb.profile import Profile
from pgvb.recognize import Recognizer


class _FakeBackbone:
    """Deterministic, unit-normalized linear projection standing in for the real backbone."""

    def __init__(self, input_dim: int, embed_dim: int, seed: int) -> None:
        rng = np.random.default_rng(seed)
        self._w = rng.normal(size=(input_dim, embed_dim)).astype(np.float32)

    def embed(self, x: np.ndarray) -> np.ndarray:
        h = x @ self._w
        return (h / np.linalg.norm(h)).astype(np.float32)

    def embed_batch(self, x: np.ndarray) -> np.ndarray:
        return np.stack([self.embed(row) for row in x])


def _pose_landmarks(rng: np.random.Generator, center: np.ndarray) -> np.ndarray:
    """A 21x3 landmark array with a fixed wrist/middle-MCP frame (so normalize_landmarks has a
    well-defined scale) and every other joint perturbed near `center`."""
    lm = np.zeros((21, 3), dtype=np.float64)
    lm[9] = [0.0, -0.1, 0.0]
    for i in range(1, 21):
        if i == 9:
            continue
        lm[i] = center + rng.normal(scale=0.01, size=3)
    return lm


def _enroll_from_cluster(
    enroller: Enroller, name: str, message: str, center: np.ndarray, seed: int, n: int
) -> None:
    rng = np.random.default_rng(seed)
    enroller.start(name, message)
    while enroller.progress[0] < n:
        status = None
        for _ in range(enroller.cfg.stability_frames):
            hand_frame = HandFrame(
                landmarks=_pose_landmarks(rng, center), handedness="Right", score=0.9, ts_ms=0.0
            )
            status = enroller.add_frame(hand_frame)
        assert status.ready
        enroller.accept_sample()
    enroller.commit()


def test_enrolling_new_gesture_does_not_change_earlier_prototypes_or_decisions():
    profile = Profile(user_id="test", backbone_id="fake")
    backbone = _FakeBackbone(input_dim=63, embed_dim=16, seed=0)
    cfg = EnrollConfig(
        min_samples=5, max_samples=10, stability_frames=3, stability_max_motion=1.0
    )

    centers = {
        "A": np.array([3.0, 0.0, 0.0]),
        "B": np.array([0.0, 3.0, 0.0]),
        "C": np.array([0.0, 0.0, 3.0]),
        "D": np.array([3.0, 3.0, 3.0]),
    }

    enroller = Enroller(backbone, profile, cfg)
    _enroll_from_cluster(enroller, "A", "message a", centers["A"], seed=1, n=5)
    _enroll_from_cluster(enroller, "B", "message b", centers["B"], seed=2, n=5)
    _enroll_from_cluster(enroller, "C", "message c", centers["C"], seed=3, n=5)

    snapshot = {
        g.name: (g.prototype.copy(), [e.copy() for e in g.examples]) for g in profile.gestures
    }

    recognizer = Recognizer(backbone, profile, RecognizeConfig(max_distance=1.0, min_margin=0.0))

    def _decide(name: str, seed: int):
        rng = np.random.default_rng(seed)
        hand_frame = HandFrame(
            landmarks=_pose_landmarks(rng, centers[name]), handedness="Right", score=0.9, ts_ms=0.0
        )
        return recognizer.decide(hand_frame)

    decisions_before = {name: _decide(name, seed=100 + i) for i, name in enumerate("ABC")}

    _enroll_from_cluster(enroller, "D", "message d", centers["D"], seed=4, n=5)
    recognizer.reload_profile()

    for gesture in profile.gestures:
        if gesture.name == "D":
            continue
        proto_before, examples_before = snapshot[gesture.name]
        np.testing.assert_array_equal(gesture.prototype, proto_before)
        assert len(gesture.examples) == len(examples_before)
        for before, after in zip(examples_before, gesture.examples):
            np.testing.assert_array_equal(before, after)

    for i, name in enumerate("ABC"):
        decision_after = _decide(name, seed=100 + i)
        assert decision_after.label == decisions_before[name].label == name
