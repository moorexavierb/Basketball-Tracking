"""Touch rows: how each touch starts and ends, dribbles, and the off-ball player columns."""
import numpy as np

from courtvision.court.geometry import NBA
from courtvision.events.passes import detect_passes
from courtvision.events.touches import touches_table
from courtvision.possession.acquisition import possession_segments
from courtvision.types import Event, PipelineResult

N = 80
BOX = {1: [100, 100, 140, 200], 2: [300, 100, 340, 200], 3: [500, 100, 540, 200]}
COURT = {1: (10.0, 7.0), 2: (11.0, 7.0), 3: (20.0, 5.0)}
TEAM = {1: 0, 2: 1, 3: 0}


def make_result():
    # 1 dribbles twice and throws a pass that 2 intercepts; 2 hits a three; 3 rebounds the next miss.
    possessor = [1] * 20 + [-1] * 5 + [2] * 15 + [-1] * 20 + [3] * 11 + [-1] * 9
    ball = []
    for f, pid in enumerate(possessor):
        if pid == 1:
            x1, y1, x2, _ = BOX[1]
            y = 190 if f in (5, 6, 12, 13) else 140  # two bounces near the feet
            ball.append([x2 - 4, y - 4, x2 + 4, y + 4])
        else:
            ball.append([400, 50, 408, 58])
    segs = possession_segments(possessor, TEAM)
    shot = Event(45, "shot", 2, 1, extra={"release_frame": 39, "made": False, "points": 3, "distance_m": 7.9})
    events = detect_passes(segs, 45, shot_frames={45}) + [shot]
    return PipelineResult(
        fps=20.0, frame_size=(1280, 720),
        players=[{t: {"bbox": b, "conf": 0.9} for t, b in BOX.items()} for _ in range(N)],
        referees=[{} for _ in range(N)], ball=ball, hoops=[None] * N, team_of=TEAM, possessor=possessor,
        possessions=segs, events=events, homographies=[None] * N,
        court_positions={t: [xy] * N for t, xy in COURT.items()},
        speed_kmh={t: [10.0] * N for t in COURT}, distance_m={t: list(np.arange(N) * 0.1) for t in COURT},
        ball_3d=[None] * N)


def test_touch_actions_and_context():
    df = touches_table(make_result().possessions, make_result(), ["Home", "Away"], NBA,
                       {2: {"jersey": "0", "name": "Jayson Tatum"}})
    assert list(df.player_id) == [1, 2, 3]
    a, b, c = df.itertuples()
    assert (a.start_action, a.action, a.outcome, a.target_id, a.dribbles) == ("start_of_clip", "pass", "intercepted", 2, 2)
    assert (b.start_action, b.action, b.outcome, b.shot_points) == ("steal", "shot", "missed", 3)
    assert b.name == "Jayson Tatum" and a.target_name == "Jayson Tatum"
    assert c.start_action == "defensive_rebound" and c.action == "end_of_clip"
    assert [a.possession_id, b.possession_id, c.possession_id] == [1, 2, 3]
    # Off-ball players at the action frame.
    assert a.tm1_id == 3 and a.op1_id == 2 and a.nearest_defender_m == 1.0 and a.defenders_within_2m == 1
    assert a.tm1_dist_m == round(np.hypot(10, 2), 2) and a.op1_speed_kmh == 10.0
    assert a.distance_covered_m == 1.9 and a.dist_to_basket_m is not None
    assert all(0 < x <= 1 for x in df.confidence)


def test_no_calibration_keeps_ids_and_actions():
    r = make_result()
    r.court_positions, r.speed_kmh, r.distance_m = {}, {}, {}
    df = touches_table(r.possessions, r, ["Home", "Away"], NBA)
    assert list(df.action) == ["pass", "shot", "end_of_clip"]
    assert df.x_m.isna().all() and df.op1_id.notna().all() and df.nearest_defender_id.isna().all()
