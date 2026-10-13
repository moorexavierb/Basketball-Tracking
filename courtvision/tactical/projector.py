"""Project tracked players onto the court plane (metres)."""
from __future__ import annotations

from typing import Optional

import numpy as np

from ..court.geometry import CourtSpec
from ..court.homography import apply_h
from ..types import PlayerTracks, bbox_foot

CourtPositions = dict[int, list[Optional[tuple[float, float]]]]


def project_tracks(tracks: PlayerTracks, homographies: list[Optional[np.ndarray]],
                   court: CourtSpec) -> CourtPositions:
    """{track_id: per-frame (x, y) in metres, or None when unseen/unmappable}."""
    n = len(tracks)
    out: CourtPositions = {}
    for f, (frame_tracks, H) in enumerate(zip(tracks, homographies)):
        if H is None or not frame_tracks:
            continue
        ids = list(frame_tracks)
        feet = np.array([bbox_foot(frame_tracks[t]["bbox"]) for t in ids])
        for tid, (x, y) in zip(ids, apply_h(H, feet)):
            if court.in_bounds(x, y):
                out.setdefault(tid, [None] * n)[f] = (float(x), float(y))
    return out


def project_point(H: Optional[np.ndarray], pt: tuple[float, float]) -> Optional[tuple[float, float]]:
    if H is None:
        return None
    x, y = apply_h(H, np.array([pt]))[0]
    return float(x), float(y)
