"""Event, tracking and box-score tables. The touches table lives in ``events.touches``."""
from __future__ import annotations

from typing import Optional

import pandas as pd

from ..court.geometry import CourtSpec
from ..types import Event, PipelineResult
from .touches import touches_table  # noqa: F401  (re-exported)


def annotate_shots(events: list[Event], court_positions: dict[int, list[Optional[tuple[float, float]]]],
                   court: CourtSpec) -> None:
    """Add shot location, distance and 2PT/3PT to shot events, in place."""
    for e in events:
        if e.type not in ("shot", "made_shot") or e.player_id is None:
            continue
        f = e.extra.get("release_frame") or e.frame
        pos = court_positions.get(e.player_id, [None] * (f + 1))[f] if f is not None else None
        if pos is None:
            continue
        e.extra.update(x_m=pos[0], y_m=pos[1], distance_m=round(court.shot_distance(*pos), 2),
                       points=3 if court.is_three(*pos) else 2)


def events_table(events: list[Event], fps: float, team_names: list[str]) -> pd.DataFrame:
    rows = []
    for e in sorted(events, key=lambda e: e.frame):
        rows.append({"frame": e.frame, "time_s": round(e.frame / fps, 2), "type": e.type,
                     "player_id": e.player_id,
                     "team": team_names[e.team] if e.team in (0, 1) else None,
                     "target_id": e.target_id, **e.extra})
    return pd.DataFrame(rows)


def tracking_table(result: PipelineResult) -> pd.DataFrame:
    """Long-format per-frame tracking data: every player and the ball, in metres."""
    rows = []
    for tid, track in result.court_positions.items():
        for f, p in enumerate(track):
            if p is None:
                continue
            rows.append({"frame": f, "time_s": round(f / result.fps, 3), "object": "player", "id": tid,
                         "team": result.team_of.get(tid), "x_m": p[0], "y_m": p[1], "z_m": 0.0,
                         "speed_kmh": result.speed_kmh.get(tid, [None] * len(track))[f],
                         "distance_m": result.distance_m.get(tid, [None] * len(track))[f],
                         "has_ball": result.possessor[f] == tid})
    for f, b in enumerate(result.ball_3d):
        if b is not None:
            rows.append({"frame": f, "time_s": round(f / result.fps, 3), "object": "ball", "id": -1,
                         "team": None, "x_m": b[0], "y_m": b[1], "z_m": b[2]})
    return pd.DataFrame(rows).sort_values(["frame", "object", "id"]).reset_index(drop=True)


def box_score(events: pd.DataFrame, touches: pd.DataFrame) -> pd.DataFrame:
    """Per-player counts: touches, passes, interceptions, shots (FGA/FGM, 3PA/3PM)."""
    if touches.empty:
        return pd.DataFrame()
    df = touches.groupby(["player_id", "team"], dropna=False).agg(touches=("touch_id", "count"),
                                                    time_of_possession_s=("duration_s", "sum")).reset_index()

    def count(kind, extra=None):
        if events.empty:
            return pd.Series(dtype=int)
        sel = events[events["type"] == kind]
        if extra is not None:
            sel = sel[extra(sel)]
        return sel.groupby("player_id").size()

    three = (lambda s: s.get("points", pd.Series(index=s.index, dtype=float)) == 3)
    for col, series in {
        "passes": count("pass"), "interceptions": count("interception"), "turnovers": count("turnover"),
        "fga": count("shot"), "fgm": count("made_shot"),
        "3pa": count("shot", three), "3pm": count("made_shot", three),
    }.items():
        df[col] = df["player_id"].map(series).fillna(0).astype(int)
    return df
