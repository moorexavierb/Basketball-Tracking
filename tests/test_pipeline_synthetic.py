"""End-to-end run of everything after the models, on a synthetic broadcast clip.

Four players (two teams, white and dark-blue jerseys) are placed on a real NBA
court, imaged through a synthetic broadcast camera, and drawn into a video.
Player 1 passes to player 2. The test checks team assignment, the detected
pass, court-space tracking, the 3D ball and every output file.
"""
import numpy as np
import cv2
import pandas as pd

from courtvision.config import Config
from courtvision.court.geometry import NBA
from courtvision.court.homography import project
from courtvision.io.video import save_video
from courtvision.pipeline import Pipeline
from courtvision.teams.team_assigner import TeamAssigner
from courtvision.types import DetectionResult
from conftest import floor_homography

JERSEYS = {0: (240, 240, 240), 1: (140, 40, 20)}  # BGR: white, dark blue


def player_box(P, x, y, height=2.0, width=0.6):
    foot = project(P, np.array([x, y, 0.0]))[0]
    head = project(P, np.array([x, y, height]))[0]
    h = foot[1] - head[1]
    w = h * width / height
    return [foot[0] - w / 2, head[1], foot[0] + w / 2, foot[1]]


def test_synthetic_play(tmp_path, broadcast_cam):
    P, K, R, t, size = broadcast_cam
    H = floor_homography(P)
    fps, n = 30.0, 90
    starts = {1: ((8.0, 6.0), 0), 2: ((12.0, 9.0), 0), 3: ((9.0, 4.0), 1), 4: ((13.0, 10.5), 1)}

    players, ball_c, frames = [], [], []
    for f in range(n):
        img = np.full((size[1], size[0], 3), (120, 170, 214), np.uint8)
        fp = {}
        for tid, ((x, y), team) in starts.items():
            x += 0.05 * f  # everyone drifts at 1.5 m/s
            b = player_box(P, x, y)
            fp[tid] = {"bbox": b, "conf": 0.9}
            x1, y1, x2, y2 = map(int, b)
            cv2.rectangle(img, (x1, y1), (x2, y2), JERSEYS[team], -1)
        players.append(fp)
        frames.append(img)

        p1 = np.array([8.0 + 0.05 * f, 6.0, 1.0])
        p2 = np.array([12.0 + 0.05 * f, 9.0, 1.0])
        if f < 25:
            pos = p1
        elif f < 55:  # one-second lob pass under real gravity: peaks ~1.23 m above release
            T, tt = 1.0, (f - 25) / fps
            pos = p1 + (p2 - p1) * tt / T + np.array([0, 0, 0.5 * 9.81 * tt * (T - tt)])
        else:
            pos = p2
        u, v = project(P, pos)[0]
        ball_c.append([{"bbox": [u - 4, v - 4, u + 4, v + 4], "conf": 0.6}])

    video = tmp_path / "synthetic.mp4"
    save_video(frames, video, fps)
    det = DetectionResult(players=players, referees=[{} for _ in range(n)], ball_candidates=ball_c,
                          hoops=[None] * n)

    cfg = Config.load(None, output={"dir": str(tmp_path / "out"), "cache_dir": None})
    pipe = Pipeline(cfg)
    assigner = TeamAssigner(cfg.teams)
    team_of = assigner.assign(frames, players)
    assert team_of[1] == team_of[2] != team_of[3] == team_of[4]

    result = pipe.analyze(det, [H] * n, team_of, fps, size)
    passes = [e for e in result.events if e.type == "pass"]
    assert len(passes) == 1 and passes[0].player_id == 1 and passes[0].target_id == 2

    # Feet land where we put them on the court.
    assert np.allclose(result.court_positions[3][0], (9.0, 4.0), atol=0.05)
    assert abs(result.speed_kmh[1][-1] - 1.5 * 3.6) < 0.3

    # The lob peaks well above the 1 m hold height.
    peak = max(b[2] for b in result.ball_3d[25:55] if b is not None)
    assert abs(peak - 2.23) < 0.3, peak

    out_dir = tmp_path / "out" / "synthetic"
    out_dir.mkdir(parents=True)
    pipe.write_outputs(result, video, out_dir)
    for name in ["touches.csv", "events.csv", "box_score.csv", "tracking.csv", "scene.json",
                 "play_3d.html", "synthetic_annotated.mp4"]:
        assert (out_dir / name).exists(), name
    box = pd.read_csv(out_dir / "box_score.csv")
    assert box.set_index("player_id").loc[1, "passes"] == 1
