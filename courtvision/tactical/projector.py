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


def smooth_court_positions(positions: CourtPositions, fps: float, max_fill_s: float = 0.5,
                           window_s: float = 0.3) -> CourtPositions:
    """Remove foot-point spikes, fill short gaps and take out frame-to-frame jitter.

    Foot points jump when a box is cut off by another player or the detector
    briefly loses someone, which shows as twitching and flickering bodies in
    3D. Per track: a 5-frame median drops single-frame spikes, gaps up to
    ``max_fill_s`` inside the track are interpolated, then a centred moving
    average over ``window_s`` smooths what is left. Track ends keep their own
    values so the smoothing never drags a player inward, and long gaps stay
    empty.
    """
    import pandas as pd

    limit = max(1, int(round(max_fill_s * fps)))
    win = max(3, int(round(window_s * fps)) | 1)
    out: CourtPositions = {}
    for tid, track in positions.items():
        arr = pd.DataFrame([p if p is not None else (np.nan, np.nan) for p in track], dtype=float)
        valid = arr[0].notna().to_numpy()
        if valid.sum() < 3:
            out[tid] = list(track)
            continue
        med = arr.rolling(5, center=True).median().where(arr.notna()).fillna(arr)
        filled = med.interpolate(limit_area="inside")
        # Undo fills inside gaps longer than the limit.
        gap_run = (~valid).astype(int)
        run_id = np.cumsum(np.r_[0, np.diff(gap_run) != 0])
        run_len = pd.Series(gap_run).groupby(run_id).transform("sum").to_numpy()
        filled.loc[(~valid) & (run_len > limit)] = np.nan
        smooth = filled.rolling(win, center=True).mean()
        res = np.where(smooth.isna(), filled, smooth)
        out[tid] = [None if np.isnan(x) else (float(x), float(y)) for x, y in res]
    return out
