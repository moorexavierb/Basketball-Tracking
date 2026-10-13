import numpy as np

from courtvision.court.geometry import NBA
from courtvision.court.homography import project
from courtvision.events.shots import detect_shots
from courtvision.metrics.speed_distance import speed_and_distance
from courtvision.possession.acquisition import NO_POSSESSION
from courtvision.recon3d.ball3d import cameras_for_frames, reconstruct_ball_3d
from courtvision.types import Possession
from conftest import floor_homography


def bx(cx, cy, s=6):
    return [cx - s, cy - s, cx + s, cy + s]


def test_made_and_missed_shot():
    hoop = [500, 200, 540, 240]  # rim near y = 206
    n = 120
    ball, possessor = [None] * n, [NO_POSSESSION] * n
    for f in range(0, 10):
        possessor[f] = 7
        ball[f] = bx(300, 400)
    for f in range(10, 30):                     # arc up and into the rim
        s = (f - 10) / 19
        ball[f] = bx(300 + 220 * s, 400 - 600 * s + 420 * s * s)
    for f in range(30, 40):
        ball[f] = bx(520, 230 + (f - 30) * 3)   # through the net
    segs = [Possession(7, 0, 0, 9)]
    events = detect_shots(ball, [hoop] * n, possessor, segs, window=40)
    types = [e.type for e in events]
    assert types[:2] == ["shot", "made_shot"]
    assert events[0].player_id == 7

    # Miss: ball comes down outside the rim.
    ball2 = list(ball)
    for f in range(30, 40):
        ball2[f] = bx(600, 230 + (f - 30) * 3)
    for f in range(20, 30):
        s = (f - 10) / 19
        ball2[f] = bx(300 + 330 * s, 400 - 600 * s + 420 * s * s)
    ev2 = detect_shots(ball2, [hoop] * n, possessor, segs, window=40)
    assert [e.type for e in ev2][:1] == ["shot"] and "made_shot" not in [e.type for e in ev2]


def test_speed_and_distance():
    fps = 30
    track = [(i * 0.2, 5.0) for i in range(31)]   # 6 m/s for 1 s
    speeds, dists = speed_and_distance({1: track}, fps, window=5)
    assert abs(dists[1][-1] - 6.0) < 0.05
    assert abs(speeds[1][-1] - 6 * 3.6) < 0.5


def test_ball_flight_recovered_in_3d(broadcast_cam):
    P, K, R, t, size = broadcast_cam
    H = floor_homography(P)
    fps = 30.0
    p0, v0 = np.array([8.0, 6.0, 2.2]), np.array([-3.5, 0.6, 5.0])
    n = 40
    true = [p0 + v0 * (f / fps) + 0.5 * np.array([0, 0, -9.81]) * (f / fps) ** 2 for f in range(n)]
    ball = [bx(*project(P, X)[0], s=4) for X in true]
    cams = cameras_for_frames([H] * n, size)
    out = reconstruct_ball_3d(ball, [NO_POSSESSION] * n, cams, {}, fps)
    err = np.array([np.linalg.norm(np.array(o) - X) for o, X in zip(out, true)])
    assert np.median(err) < 0.15, err
    assert abs(max(o[2] for o in out) - max(X[2] for X in true)) < 0.25
