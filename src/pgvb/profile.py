"""User profile storage: gestures with their examples and prototypes, persisted as JSON.

Schema (profiles/<user>.json):
  schema_version  int
  user_id         str
  backbone_id     str        must match the loaded Backbone; validate() rejects a mismatch
  gestures        list of Gesture, each:
    id              str
    name            str
    message         str
    examples        list of list[float], each embed_dim long
    prototype       list[float], embed_dim long
    created_at      str, ISO 8601
    n_refinements   int
"""

from __future__ import annotations

import json
import os
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from pgvb.prototypes import make_prototype

SCHEMA_VERSION = 1


@dataclass
class Gesture:
    id: str
    name: str
    message: str
    examples: list[np.ndarray]
    prototype: np.ndarray
    created_at: str
    n_refinements: int = 0

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "message": self.message,
            "examples": [e.tolist() for e in self.examples],
            "prototype": self.prototype.tolist(),
            "created_at": self.created_at,
            "n_refinements": self.n_refinements,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Gesture":
        return cls(
            id=d["id"],
            name=d["name"],
            message=d["message"],
            examples=[np.asarray(e, dtype=np.float32) for e in d["examples"]],
            prototype=np.asarray(d["prototype"], dtype=np.float32),
            created_at=d["created_at"],
            n_refinements=int(d.get("n_refinements", 0)),
        )


@dataclass
class Profile:
    user_id: str
    backbone_id: str
    gestures: list[Gesture] = field(default_factory=list)
    schema_version: int = SCHEMA_VERSION

    def find(self, gesture_id: str) -> Gesture | None:
        for gesture in self.gestures:
            if gesture.id == gesture_id:
                return gesture
        return None

    def add_gesture(self, name: str, message: str, examples: list[np.ndarray]) -> Gesture:
        gesture = Gesture(
            id=uuid.uuid4().hex[:12],
            name=name,
            message=message,
            examples=[np.asarray(e, dtype=np.float32) for e in examples],
            prototype=make_prototype(np.stack(examples)),
            created_at=datetime.now(timezone.utc).isoformat(),
        )
        self.gestures.append(gesture)
        return gesture

    def remove_gesture(self, gesture_id: str) -> None:
        self.gestures = [g for g in self.gestures if g.id != gesture_id]

    def rename(self, gesture_id: str, name: str) -> None:
        gesture = self.find(gesture_id)
        if gesture is None:
            raise KeyError(gesture_id)
        gesture.name = name

    def set_message(self, gesture_id: str, message: str) -> None:
        gesture = self.find(gesture_id)
        if gesture is None:
            raise KeyError(gesture_id)
        gesture.message = message

    def to_dict(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "user_id": self.user_id,
            "backbone_id": self.backbone_id,
            "gestures": [g.to_dict() for g in self.gestures],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Profile":
        return cls(
            user_id=d["user_id"],
            backbone_id=d["backbone_id"],
            gestures=[Gesture.from_dict(g) for g in d.get("gestures", [])],
            schema_version=int(d.get("schema_version", SCHEMA_VERSION)),
        )


def validate(profile: Profile, backbone_id: str, embed_dim: int) -> None:
    """Raises ValueError if the profile does not match the running backbone id or embedding
    dimension."""
    if profile.backbone_id != backbone_id:
        raise ValueError(
            f"profile backbone_id {profile.backbone_id!r} does not match loaded backbone "
            f"{backbone_id!r}"
        )
    for gesture in profile.gestures:
        if gesture.prototype.shape != (embed_dim,):
            raise ValueError(
                f"gesture {gesture.name!r} prototype has shape {gesture.prototype.shape}, "
                f"expected ({embed_dim},)"
            )
        for example in gesture.examples:
            if example.shape != (embed_dim,):
                raise ValueError(
                    f"gesture {gesture.name!r} has an example with shape {example.shape}, "
                    f"expected ({embed_dim},)"
                )


def load(path: str | Path) -> Profile:
    with open(path, "r", encoding="utf-8") as f:
        return Profile.from_dict(json.load(f))


def save(profile: Profile, path: str | Path) -> None:
    """Writes to a temp file in the same directory, then renames, so a crash mid-write never
    corrupts the existing profile."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(profile.to_dict(), f, indent=2)
    os.replace(tmp_path, out_path)
