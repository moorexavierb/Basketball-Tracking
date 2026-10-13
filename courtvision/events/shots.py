"""Shot attempts and makes from the ball track and the detected hoop.

* attempt: the free ball (nobody holding it) enters the zone above the rim.
* make: within ``window`` frames the ball centre crosses the rim line from
  above to below while horizontally inside the rim.
* shooter: holder of the last possession that ended before the attempt, if it
  ended recently enough to be a release rather than a tip or a loose ball.

Shot location and 2PT/3PT are added later from court coordinates
(``courtvision.events.timeline.annotate_shots``).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..possession.acquisition import NO_POSSESSION
from ..types import BallTrack, Event, Possession, bbox_center


def fill_hoops(hoops: list[list[float] | None], max_gap: int = 60) -> list[list[float] | None]:
    rows = [h if h is not None else [np.nan] * 4 for h in hoops]
    df = pd.DataFrame(rows, dtype=float).interpolate(limit=max_gap, limit_direction="both")
    return [None if np.isnan(r[0]) else list(map(float, r)) for r in df.to_numpy()]


def _rim(h: list[float]) -> tuple[float, float, float, float]:
    """(rim_left, rim_right, rim_y, box_height). The rim sits near the top of a hoop box."""
    x1, y1, x2, y2 = h
    return x1, x2, y1 + 0.15 * (y2 - y1), (y2 - y1)


def detect_shots(ball: BallTrack, hoops: list[list[float] | None], possessor: list[int],
                 segments: list[Possession], window: int = 45, max_release_gap: int = 60) -> list[Event]:
    hoops = fill_hoops(hoops)
    events: list[Event] = []
    f, n = 0, len(ball)
    while f < n:
        b, h = ball[f], hoops[f]
        if b is None or h is None or possessor[f] != NO_POSSESSION:
            f += 1
            continue
        rl, rr, ry, hh = _rim(h)
        w = rr - rl
        bx, by = bbox_center(b)
        in_upper_zone = (rl - w) <= bx <= (rr + w) and (ry - 4 * hh) <= by < ry
        if not in_upper_zone:
            f += 1
            continue

        attempt = f
        made_frame = None
        prev_above = True
        for g in range(attempt + 1, min(n, attempt + window)):
            bg, hg = ball[g], hoops[g]
            if bg is None or hg is None:
                continue
            rl, rr, ry, hh = _rim(hg)
            gx, gy = bbox_center(bg)
            margin = 0.1 * (rr - rl)
            if prev_above and gy >= ry and rl + margin <= gx <= rr - margin and gy <= ry + 2.5 * hh:
                made_frame = g
                break
            prev_above = gy < ry
            if possessor[g] != NO_POSSESSION:
                break

        shooter = next((s for s in reversed(segments) if s.end_frame < attempt), None)
        if shooter is not None and attempt - shooter.end_frame > max_release_gap:
            shooter = None
        extra = {"release_frame": shooter.end_frame if shooter else None, "made": made_frame is not None}
        pid = shooter.player_id if shooter else None
        team = shooter.team if shooter else None
        events.append(Event(attempt, "shot", pid, team, extra=dict(extra)))
        if made_frame is not None:
            events.append(Event(made_frame, "made_shot", pid, team, extra=dict(extra)))
        f = (made_frame or attempt) + window
    return events
