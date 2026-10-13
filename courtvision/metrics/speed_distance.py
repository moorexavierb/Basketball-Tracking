"""Per-player speed and cumulative distance in real units.

Foot positions jitter by tens of centimetres frame to frame, which would
inflate distance badly if summed raw. Positions are median-smoothed first,
then speed is measured over a window and capped at a human limit.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


def smooth_positions(track: list[Optional[tuple[float, float]]], window: int = 5) -> np.ndarray:
    arr = np.array([p if p is not None else (np.nan, np.nan) for p in track], dtype=float)
    # Full windows only; edges (and frames next to gaps) keep the raw position,
    # otherwise the median pulls track ends inward and shortens the distance.
    smooth = pd.DataFrame(arr).rolling(window, center=True).median().to_numpy()
    return np.where(np.isnan(smooth), arr, smooth)


def speed_and_distance(positions: dict[int, list[Optional[tuple[float, float]]]], fps: float,
                       window: int = 5, max_speed_mps: float = 11.0
                       ) -> tuple[dict[int, list[Optional[float]]], dict[int, list[Optional[float]]]]:
    """Return ({id: per-frame speed km/h}, {id: per-frame cumulative distance m})."""
    speeds, dists = {}, {}
    for tid, track in positions.items():
        p = smooth_positions(track, window)
        n = len(p)
        speed: list[Optional[float]] = [None] * n
        dist: list[Optional[float]] = [None] * n
        total = 0.0
        last_valid = None
        for f in range(n):
            if np.isnan(p[f, 0]):
                continue
            if last_valid is not None:
                dt = (f - last_valid) / fps
                step = float(np.hypot(*(p[f] - p[last_valid])))
                if dt > 0 and step / dt <= max_speed_mps:
                    total += step
            back = f - window
            if back >= 0 and not np.isnan(p[back, 0]):
                v = float(np.hypot(*(p[f] - p[back]))) / (window / fps)
                speed[f] = min(v, max_speed_mps) * 3.6
            dist[f] = total
            last_valid = f
        speeds[tid], dists[tid] = speed, dist
    return speeds, dists
