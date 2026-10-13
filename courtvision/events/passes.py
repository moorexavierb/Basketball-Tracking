"""Passes, interceptions and turnovers from the possession timeline.

Between two consecutive possession segments held by different players:

* same team                         -> pass (passer -> receiver)
* other team, ball free for a short
  gap and no shot in between        -> interception (or steal)
* other team after a shot           -> rebound, not a turnover (handled by shots)
* other team after a long gap       -> turnover (loose ball recovered)
"""
from __future__ import annotations

from ..types import Event, Possession


def detect_passes(segments: list[Possession], interception_max_gap_frames: int = 45,
                  shot_frames: set[int] | None = None) -> list[Event]:
    shot_frames = shot_frames or set()
    events: list[Event] = []
    for prev, nxt in zip(segments, segments[1:]):
        if prev.player_id == nxt.player_id:
            continue
        gap = nxt.start_frame - prev.end_frame
        info = {"release_frame": prev.end_frame, "catch_frame": nxt.start_frame, "air_frames": gap}
        shot_between = any(prev.end_frame <= f <= nxt.start_frame for f in shot_frames)
        if shot_between:
            if prev.team != nxt.team:
                events.append(Event(nxt.start_frame, "defensive_rebound", nxt.player_id, nxt.team, extra=info))
            else:
                events.append(Event(nxt.start_frame, "offensive_rebound", nxt.player_id, nxt.team, extra=info))
        elif prev.team == nxt.team and prev.team != -1:
            events.append(Event(prev.end_frame, "pass", prev.player_id, prev.team, nxt.player_id, info))
        elif gap <= interception_max_gap_frames:
            events.append(Event(nxt.start_frame, "interception", nxt.player_id, nxt.team, prev.player_id, info))
        else:
            events.append(Event(nxt.start_frame, "turnover", prev.player_id, prev.team, nxt.player_id, info))
    return events
