"""Who has the ball, frame by frame.

A player is a candidate holder when the ball centre lies inside their box, or
close to one of their "hand points" (sides of the box at chest and waist
height, plus the top for over-the-head holds), measured relative to the
player's box height so it works at any zoom. A change of holder is accepted
only after ``min_frames`` consecutive frames (hysteresis), which removes the
one-frame flips you get when the ball passes in front of another player.
"""
from __future__ import annotations

import numpy as np

from ..types import BallTrack, PlayerTracks, Possession, bbox_center, bbox_height

NO_POSSESSION = -1


def _hand_points(b: list[float]) -> list[tuple[float, float]]:
    x1, y1, x2, y2 = b
    h = y2 - y1
    cx = (x1 + x2) / 2
    pts = []
    for frac in (0.3, 0.55):
        y = y1 + frac * h
        pts += [(x1, y), (x2, y)]
    pts += [(cx, y1), (cx, y1 + 0.55 * h)]
    return pts


def frame_holder(players: dict[int, dict], ball: list[float] | None, max_distance_ratio: float) -> int:
    if ball is None or not players:
        return NO_POSSESSION
    bx, by = bbox_center(ball)
    best, best_score = NO_POSSESSION, np.inf
    for tid, p in players.items():
        b = p["bbox"]
        h = max(bbox_height(b), 1.0)
        inside = b[0] <= bx <= b[2] and b[1] <= by <= b[3]
        d = min(np.hypot(bx - x, by - y) for x, y in _hand_points(b)) / h
        if inside:
            d *= 0.5  # containment is strong evidence; still prefer the closer hands
        if d <= max_distance_ratio and d < best_score:
            best, best_score = tid, d
    return best


def detect_possession(players: PlayerTracks, ball: BallTrack, max_distance_ratio: float = 0.35,
                      min_frames: int = 5) -> list[int]:
    raw = [frame_holder(p, b, max_distance_ratio) for p, b in zip(players, ball)]
    out = []
    current, pending, count = NO_POSSESSION, None, 0
    for r in raw:
        if r == current:
            pending, count = None, 0
        elif r == NO_POSSESSION:
            # Ball left the hands (pass/shot/dribble). Keep the holder for a few
            # frames to bridge dribbles, then release.
            if pending == NO_POSSESSION:
                count += 1
            else:
                pending, count = NO_POSSESSION, 1
            if count >= min_frames:
                current, pending, count = NO_POSSESSION, None, 0
        else:
            if r == pending:
                count += 1
            else:
                pending, count = r, 1
            if count >= min_frames or current == NO_POSSESSION and count >= max(1, min_frames // 2):
                current, pending, count = r, None, 0
        out.append(current)
    return out


def possession_segments(possessor: list[int], team_of: dict[int, int]) -> list[Possession]:
    segs: list[Possession] = []
    for f, pid in enumerate(possessor):
        if pid == NO_POSSESSION:
            continue
        if segs and segs[-1].player_id == pid and segs[-1].end_frame == f - 1:
            segs[-1].end_frame = f
        else:
            segs.append(Possession(pid, team_of.get(pid, -1), f, f))
    return segs


def team_control(possessor: list[int], team_of: dict[int, int]) -> list[int]:
    """Per-frame team in control; carries the last team through loose-ball frames."""
    out, last = [], -1
    for pid in possessor:
        if pid != NO_POSSESSION:
            last = team_of.get(pid, -1)
        out.append(last)
    return out
