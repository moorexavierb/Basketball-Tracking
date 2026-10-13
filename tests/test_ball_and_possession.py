import numpy as np

from courtvision.ball.interpolation import clean_ball_track, interpolate_ball, remove_outliers, select_ball
from courtvision.events.passes import detect_passes
from courtvision.possession.acquisition import (NO_POSSESSION, detect_possession, possession_segments,
                                                team_control)
from courtvision.types import Possession


def box(cx, cy, s=10):
    return [cx - s, cy - s, cx + s, cy + s]


def test_select_prefers_continuity_over_confidence():
    cands = [[{"bbox": box(100, 100), "conf": 0.9}],
             [{"bbox": box(110, 100), "conf": 0.4}, {"bbox": box(900, 50), "conf": 0.8}]]
    track = select_ball(cands)
    assert track[1] == box(110, 100)


def test_outlier_removed_and_gap_interpolated():
    track = [box(100 + 10 * i, 100) for i in range(10)]
    track[4] = box(800, 800)       # impossible jump
    track[6] = None
    cleaned = interpolate_ball(remove_outliers(track, 50), max_gap_frames=5)
    assert cleaned[4] is not None and abs(cleaned[4][0] - box(140, 100)[0]) < 1e-6
    assert abs(cleaned[6][0] - box(160, 100)[0]) < 1e-6


def test_long_gap_not_invented():
    track = [box(100, 100)] + [None] * 30 + [box(400, 100)]
    out = interpolate_ball(track, max_gap_frames=10)
    assert all(b is None for b in out[1:31])


def player(cx, top=100, h=200, w=60):
    return {"bbox": [cx - w / 2, top, cx + w / 2, top + h]}


def test_possession_hysteresis_and_pass():
    n = 60
    players = [{1: player(100), 2: player(600)} for _ in range(n)]
    ball = []
    for f in range(n):
        if f < 20:
            ball.append(box(130, 210))              # in player 1's hands
        elif f < 35:
            x = 130 + (f - 20) * (440 / 15)          # in the air
            ball.append(box(x, 150))
        else:
            ball.append(box(570, 210))              # caught by player 2
    ball[10] = box(570, 210)                         # one-frame glitch must not flip possession
    possessor = detect_possession(players, ball, min_frames=4)
    assert possessor[10] == 1
    assert possessor[19] == 1
    assert possessor[-1] == 2
    segs = possession_segments(possessor, {1: 0, 2: 0})
    assert [s.player_id for s in segs] == [1, 2]
    events = detect_passes(segs)
    assert len(events) == 1 and events[0].type == "pass" and events[0].target_id == 2
    assert team_control(possessor, {1: 0, 2: 0})[-1] == 0


def test_interception_turnover_and_rebound():
    segs = [Possession(1, 0, 0, 10), Possession(5, 1, 20, 30), Possession(2, 0, 200, 210),
            Possession(6, 1, 230, 240)]
    events = detect_passes(segs, interception_max_gap_frames=45, shot_frames={215})
    assert [e.type for e in events] == ["interception", "turnover", "defensive_rebound"]
