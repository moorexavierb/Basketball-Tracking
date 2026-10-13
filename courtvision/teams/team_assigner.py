"""Assign each player track to one of two teams.

Two methods:

* ``color`` (default, no extra model): for each track, sample torso crops,
  separate jersey pixels from background with a 2-cluster KMeans (the cluster
  that dominates the crop corners is background), then cluster every track's
  jersey colour into two teams in LAB space. One decision per *track*, so a
  player never flickers between teams.
* ``clip``: zero-shot CLIP with two text prompts (e.g. "white jersey",
  "dark blue jersey"), majority-voted per track. Better when one team's
  jersey matches the floor.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from typing import Mapping

import cv2
import numpy as np
from sklearn.cluster import KMeans

from ..config import TeamsConfig
from ..types import PlayerTracks


def torso_crop(frame: np.ndarray, bbox: list[float]) -> np.ndarray | None:
    x1, y1, x2, y2 = map(int, bbox)
    h, w = y2 - y1, x2 - x1
    if h < 20 or w < 8:
        return None
    crop = frame[max(y1 + int(0.15 * h), 0): max(y1 + int(0.50 * h), 0),
                 max(x1 + int(0.20 * w), 0): max(x2 - int(0.20 * w), 0)]
    return crop if crop.size else None


def jersey_color(crop: np.ndarray) -> np.ndarray:
    """Dominant non-background colour of a torso crop, in LAB."""
    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(np.float32)
    if len(lab) < 4 or lab.std(axis=0).max() < 1.0:
        return lab.mean(axis=0)
    km = KMeans(n_clusters=2, n_init=3, random_state=0).fit(lab)
    labels = km.labels_.reshape(crop.shape[:2])
    corners = [labels[0, 0], labels[0, -1], labels[-1, 0], labels[-1, -1]]
    background = Counter(corners).most_common(1)[0][0]
    return km.cluster_centers_[1 - background]


class TeamAssigner:
    def __init__(self, cfg: TeamsConfig):
        self.cfg = cfg
        self.team_colors: np.ndarray | None = None  # (2, 3) LAB
        self._clip = None

    def sample_frames(self, tracks: PlayerTracks) -> dict[int, list[int]]:
        seen: dict[int, list[int]] = defaultdict(list)
        for f, frame_tracks in enumerate(tracks):
            for tid in frame_tracks:
                seen[tid].append(f)
        n = self.cfg.samples_per_track
        return {tid: [fs[i] for i in np.linspace(0, len(fs) - 1, min(n, len(fs))).astype(int)]
                for tid, fs in seen.items()}

    def assign(self, frames: Mapping[int, np.ndarray] | list[np.ndarray], tracks: PlayerTracks) -> dict[int, int]:
        """Return {track_id: team (0 or 1)}."""
        samples = self.sample_frames(tracks)
        if self.cfg.method == "clip":
            return self._assign_clip(frames, tracks, samples)

        track_colors: dict[int, np.ndarray] = {}
        for tid, fs in samples.items():
            colors = []
            for f in fs:
                crop = torso_crop(frames[f], tracks[f][tid]["bbox"])
                if crop is not None:
                    colors.append(jersey_color(crop))
            if colors:
                track_colors[tid] = np.median(np.stack(colors), axis=0)
        if len(track_colors) < 2:
            return {tid: 0 for tid in samples}

        ids = list(track_colors)
        X = np.stack([track_colors[t] for t in ids])
        km = KMeans(n_clusters=2, n_init=10, random_state=0).fit(X)
        self.team_colors = km.cluster_centers_
        return {tid: int(l) for tid, l in zip(ids, km.labels_)}

    def _assign_clip(self, frames, tracks, samples) -> dict[int, int]:
        import torch
        from PIL import Image
        from transformers import CLIPModel, CLIPProcessor

        if not self.cfg.clip_prompts or len(self.cfg.clip_prompts) != 2:
            raise ValueError("teams.clip_prompts must list two prompts, e.g. ['white jersey', 'dark jersey']")
        if self._clip is None:
            name = "patrickjohncyh/fashion-clip"
            self._clip = (CLIPModel.from_pretrained(name), CLIPProcessor.from_pretrained(name))
        model, proc = self._clip
        out = {}
        for tid, fs in samples.items():
            votes = []
            for f in fs:
                crop = torso_crop(frames[f], tracks[f][tid]["bbox"])
                if crop is None:
                    continue
                img = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
                inputs = proc(text=self.cfg.clip_prompts, images=img, return_tensors="pt", padding=True)
                with torch.no_grad():
                    votes.append(int(model(**inputs).logits_per_image.argmax()))
            out[tid] = Counter(votes).most_common(1)[0][0] if votes else 0
        return out

    @staticmethod
    def team_bgr(team_colors: np.ndarray | None) -> list[tuple[int, int, int]]:
        """Display colours for the two teams (their mean jersey colour, else defaults)."""
        if team_colors is None:
            return [(235, 235, 235), (200, 60, 30)]  # white, blue (BGR)
        lab = np.clip(team_colors, 0, 255).astype(np.uint8).reshape(2, 1, 3)
        return [tuple(int(c) for c in px[0]) for px in cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)]
