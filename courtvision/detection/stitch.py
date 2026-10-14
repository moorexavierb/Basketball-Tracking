"""Re-link player tracks that the tracker split in two.

ByteTrack matches boxes in image space, so a camera pan, an occlusion in the
paint or a missed detection often ends a player's track and starts a new ID a
few frames later. Downstream that reads as a player vanishing and a new one
appearing (a pop in the 3D view, a broken touch). Here a track that ends is
joined to a track that starts shortly after, when both are on the same team,
never on screen together, and the court distance between the end and the
start is reachable at a running speed. Candidates are taken cheapest first,
so each track gets at most one predecessor and one successor.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from ..types import PlayerTracks

CourtPositions = dict[int, list[Optional[tuple[float, float]]]]


def _ends(track: list[Optional[tuple[float, float]]], k: int = 5):
    idx = [f for f, p in enumerate(track) if p is not None]
    if not idx:
        return None
    first, last = idx[0], idx[-1]
    start = np.median([track[f] for f in idx[:k]], axis=0)
    end = np.median([track[f] for f in idx[-k:]], axis=0)
    return first, last, start, end


def stitch_tracks(tracks: PlayerTracks, court_pos: CourtPositions, team_of: dict[int, int], fps: float,
                  max_gap_s: float = 1.0, max_speed_mps: float = 6.0, slack_m: float = 1.0) -> dict[int, int]:
    """Return {track_id: id it should be renamed to}; ids not in the map keep their own."""
    seen: dict[int, list[int]] = {}
    for f, ft in enumerate(tracks):
        for tid in ft:
            seen.setdefault(tid, []).append(f)
    spans = {tid: (fs[0], fs[-1]) for tid, fs in seen.items()}
    ends = {tid: e for tid in spans if tid in court_pos and (e := _ends(court_pos[tid])) is not None}

    max_gap = max_gap_s * fps
    cands = []
    for a, (_, a_last, _, a_end) in ends.items():
        for b, (b_first, _, b_start, _) in ends.items():
            if a == b or team_of.get(a, -1) != team_of.get(b, -1):
                continue
            if spans[b][0] <= spans[a][1]:  # on screen together at some point (or b starts first)
                continue
            gap = b_first - a_last
            if gap <= 0 or gap > max_gap:
                continue
            reach = slack_m + max_speed_mps * gap / fps
            dist = float(np.hypot(*(b_start - a_end)))
            if dist <= reach:
                cands.append((dist / reach + gap / max_gap, a, b))

    nxt: dict[int, int] = {}
    prv: dict[int, int] = {}
    for _, a, b in sorted(cands):
        if a not in nxt and b not in prv:
            nxt[a], prv[b] = b, a

    remap = {}
    for tid in prv:
        root = tid
        while root in prv:
            root = prv[root]
        remap[tid] = root
    return remap


def apply_remap(tracks: PlayerTracks, remap: dict[int, int]) -> PlayerTracks:
    if not remap:
        return tracks
    return [{remap.get(tid, tid): d for tid, d in ft.items()} for ft in tracks]
