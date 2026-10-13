"""Turn noisy per-frame ball candidates into one clean ball track.

1. ``select_ball``: per frame, pick the candidate that best continues the
   track (confidence weighted by distance to the last accepted position)
   instead of blindly taking the top-confidence box, which is often a head or
   a light.
2. ``remove_outliers``: drop detections that imply an impossible jump.
3. ``interpolate_ball``: fill short gaps (passes behind players) and leave long
   gaps empty rather than inventing a path.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..types import BallTrack, bbox_center


def select_ball(candidates: list[list[dict]], max_jump_px: float = 150.0) -> BallTrack:
    track: BallTrack = []
    last: tuple[float, float] | None = None
    since_last = 0
    for cands in candidates:
        if not cands:
            track.append(None)
            since_last += 1
            continue
        if last is None:
            best = max(cands, key=lambda c: c["conf"])
        else:
            reach = max_jump_px * (since_last + 1)

            def score(c):
                d = np.hypot(*(np.subtract(bbox_center(c["bbox"]), last)))
                return c["conf"] * max(0.0, 1.0 - d / reach)
            best = max(cands, key=score)
            if score(best) <= 0:
                best = max(cands, key=lambda c: c["conf"]) if since_last > 30 else None
        if best is None:
            track.append(None)
            since_last += 1
        else:
            track.append(list(best["bbox"]))
            last = bbox_center(best["bbox"])
            since_last = 0
    return track


def remove_outliers(ball: BallTrack, max_speed_px_per_frame: float) -> BallTrack:
    out = list(ball)
    last_idx = None
    for i, b in enumerate(out):
        if b is None:
            continue
        if last_idx is not None:
            gap = i - last_idx
            d = np.hypot(*np.subtract(bbox_center(b), bbox_center(out[last_idx])))
            if d > max_speed_px_per_frame * gap:
                out[i] = None
                continue
        last_idx = i
    return out


def interpolate_ball(ball: BallTrack, max_gap_frames: int = 20) -> BallTrack:
    rows = [b if b is not None else [np.nan] * 4 for b in ball]
    df = pd.DataFrame(rows, columns=["x1", "y1", "x2", "y2"], dtype=float)
    # limit_area="inside" never extrapolates past the first/last sighting.
    filled = df.interpolate(method="linear", limit=max_gap_frames, limit_area="inside")
    # pandas fills the first `limit` frames of a longer gap; undo that so long
    # gaps stay empty rather than half-filled.
    isnan = df["x1"].isna().to_numpy()
    run_start = None
    for i in range(len(isnan) + 1):
        if i < len(isnan) and isnan[i]:
            run_start = i if run_start is None else run_start
        elif run_start is not None:
            if i - run_start > max_gap_frames:
                filled.iloc[run_start:i] = np.nan
            run_start = None
    return [None if np.isnan(r[0]) else [float(v) for v in r] for r in filled.to_numpy()]


def clean_ball_track(candidates: list[list[dict]], max_speed_px_per_frame: float,
                     max_gap_frames: int) -> BallTrack:
    track = select_ball(candidates, max_jump_px=max_speed_px_per_frame * 1.5)
    track = remove_outliers(track, max_speed_px_per_frame)
    return interpolate_ball(track, max_gap_frames)
