"""Typed configuration loaded from YAML with dataclass defaults."""
from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ModelsConfig:
    detector: str = "models/player_detector.pt"
    ball_detector: str | None = None
    court_keypoints: str | None = "models/court_keypoints.pt"
    device: str | None = None


@dataclass
class DetectionConfig:
    imgsz: int = 1280
    conf: float = 0.10
    ball_conf: float = 0.05
    batch_size: int = 16
    track_buffer: int = 60  # frames a lost player track is kept alive (occlusions in the paint)
    class_names: dict[str, list[str]] = field(default_factory=lambda: {
        "player": ["Player"],
        "referee": ["Ref"],
        "ball": ["Ball"],
        "hoop": ["Hoop"],
    })


@dataclass
class TeamsConfig:
    method: str = "color"
    team_names: list[str] = field(default_factory=lambda: ["Team A", "Team B"])
    clip_prompts: list[str] | None = None
    samples_per_track: int = 12


@dataclass
class BallConfig:
    max_speed_px_per_frame: float = 90.0
    max_gap_frames: int = 20


@dataclass
class PossessionConfig:
    max_distance_ratio: float = 0.35
    min_frames: int = 5


@dataclass
class EventsConfig:
    interception_max_gap_frames: int = 45
    shot_window_frames: int = 45


@dataclass
class CourtConfig:
    standard: str = "nba"
    keypoint_conf: float = 0.5
    homography_hold_frames: int = 15
    homography_smoothing: float = 0.6


@dataclass
class MetricsConfig:
    speed_window_frames: int = 5
    max_speed_mps: float = 11.0


@dataclass
class OutputConfig:
    dir: str = "output"
    cache_dir: str | None = "stubs"
    write_video: bool = True
    write_3d_html: bool = True


@dataclass
class Config:
    models: ModelsConfig = field(default_factory=ModelsConfig)
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    teams: TeamsConfig = field(default_factory=TeamsConfig)
    ball: BallConfig = field(default_factory=BallConfig)
    possession: PossessionConfig = field(default_factory=PossessionConfig)
    events: EventsConfig = field(default_factory=EventsConfig)
    court: CourtConfig = field(default_factory=CourtConfig)
    metrics: MetricsConfig = field(default_factory=MetricsConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    @classmethod
    def load(cls, path: str | Path | None = None, **overrides: Any) -> "Config":
        cfg = cls()
        if path is not None:
            with open(path) as f:
                _merge(cfg, yaml.safe_load(f) or {})
        _merge(cfg, overrides)
        return cfg


def _merge(obj: Any, data: dict[str, Any]) -> None:
    names = {f.name for f in fields(obj)}
    for key, value in data.items():
        if key not in names:
            raise KeyError(f"Unknown config key '{key}' for {type(obj).__name__}")
        current = getattr(obj, key)
        if is_dataclass(current) and isinstance(value, dict):
            _merge(current, value)
        else:
            setattr(obj, key, value)
