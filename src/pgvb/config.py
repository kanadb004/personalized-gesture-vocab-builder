"""Loads configs/default.yaml into a dataclass tree. Every tunable lives in the YAML file;
modules should read values from the loaded Config rather than hardcoding numbers."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

_DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "default.yaml"


@dataclass
class CameraConfig:
    index: int = 0
    width: int = 640
    height: int = 480
    mirror: bool = True


@dataclass
class LandmarksConfig:
    num_hands: int = 1
    min_hand_score: float = 0.5
    min_hand_detection_confidence: float = 0.5
    min_tracking_confidence: float = 0.5


@dataclass
class FeaturesConfig:
    rotate: bool = False
    dim: int = 63


@dataclass
class BackboneConfig:
    input_dim: int = 63
    embed_dim: int = 64
    hidden_dims: list[int] = field(default_factory=lambda: [128, 128])
    dropout: float = 0.2
    id: str = "backbone_v1"


@dataclass
class RecognizeConfig:
    max_distance: float = 0.15
    min_margin: float = 0.0


@dataclass
class EnrollConfig:
    min_samples: int = 5
    max_samples: int = 10
    stability_frames: int = 5
    stability_max_motion: float = 0.02
    collision_distance: float = 0.2


@dataclass
class SmoothingConfig:
    window: int = 12
    min_votes: int = 8
    release_frames: int = 6
    cooldown_ms: int = 800


@dataclass
class OutputConfig:
    tts_backend: str = "pyttsx3"
    tts_rate: int = 175
    tts_voice: str = "default"
    board_size: int = 5
    muted: bool = False


@dataclass
class ProfileConfig:
    schema_version: int = 1


@dataclass
class Config:
    camera: CameraConfig = field(default_factory=CameraConfig)
    landmarks: LandmarksConfig = field(default_factory=LandmarksConfig)
    features: FeaturesConfig = field(default_factory=FeaturesConfig)
    backbone: BackboneConfig = field(default_factory=BackboneConfig)
    recognize: RecognizeConfig = field(default_factory=RecognizeConfig)
    enroll: EnrollConfig = field(default_factory=EnrollConfig)
    smoothing: SmoothingConfig = field(default_factory=SmoothingConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    profile: ProfileConfig = field(default_factory=ProfileConfig)


_SECTION_TYPES = {
    "camera": CameraConfig,
    "landmarks": LandmarksConfig,
    "features": FeaturesConfig,
    "backbone": BackboneConfig,
    "recognize": RecognizeConfig,
    "enroll": EnrollConfig,
    "smoothing": SmoothingConfig,
    "output": OutputConfig,
    "profile": ProfileConfig,
}


def load(path: str | Path | None = None) -> Config:
    """Loads a YAML config file into a Config dataclass tree. Defaults to
    configs/default.yaml when path is None."""
    config_path = Path(path) if path is not None else _DEFAULT_CONFIG_PATH
    with open(config_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    sections = {}
    for name, section_type in _SECTION_TYPES.items():
        section_raw = raw.get(name, {}) or {}
        sections[name] = section_type(**section_raw)
    return Config(**sections)
