"""Court mapping through a broadcast pan, on rendered frames.

A synthetic camera pans across the court while court keypoints are only seen
now and then (and sometimes mislabelled). The camera-motion tracker must keep
the mapping right in between; the old hold-last-fit behaviour drifts.
"""
import cv2
import numpy as np

from courtvision.court.camera_motion import CameraMotionEstimator
from courtvision.court.geometry import NBA
from courtvision.court.homography import apply_h, estimate_homographies, track_homographies
from courtvision.detection.stitch import apply_remap, stitch_tracks
from courtvision.tactical.projector import smooth_court_positions
from conftest import floor_homography, look_at_camera

SIZE = (960, 540)
PX_PER_M, MARGIN = 20, 30.0


def floor_texture(seed=0):
    rng = np.random.default_rng(seed)
    w = int((NBA.length + 2 * MARGIN) * PX_PER_M)
    h = int((NBA.width + 2 * MARGIN) * PX_PER_M)
    noise = rng.integers(0, 255, (h // 8, w // 8, 3), dtype=np.uint8)
    tex = cv2.resize(noise, (w, h), interpolation=cv2.INTER_NEAREST)
    tex = cv2.GaussianBlur(tex, (5, 5), 0)
    for (x0, y0), (x1, y1) in [((0, 0), (NBA.length, 0)), ((0, NBA.width), (NBA.length, NBA.width)),
                               ((0, 0), (0, NBA.width)), ((NBA.length, 0), (NBA.length, NBA.width)),
                               ((NBA.length / 2, 0), (NBA.length / 2, NBA.width))]:
        p = lambda x, y: (int((x + MARGIN) * PX_PER_M), int((y + MARGIN) * PX_PER_M))
        cv2.line(tex, p(x0, y0), p(x1, y1), (255, 255, 255), 4)
    return tex


def tex_to_court():
    return np.array([[1 / PX_PER_M, 0, -MARGIN], [0, 1 / PX_PER_M, -MARGIN], [0, 0, 1.0]])


def pan_sequence(n=120):
    """Camera on the near sideline panning from the left half to the right half."""
    cams = []
    for f in range(n):
        tx = 7.0 + 14.0 * f / (n - 1)
        P, *_ = look_at_camera((NBA.length / 2, -12.0, 8.0), (tx, NBA.width / 2, 0.0), f=800.0, size=SIZE)
        cams.append(P)
    return cams


def render(P, tex):
    H_court_to_img = np.linalg.inv(floor_homography(P))
    img = cv2.warpPerspective(tex, H_court_to_img @ tex_to_court(), SIZE)
    cv2.rectangle(img, (20, 470), (330, 520), (30, 30, 30), -1)  # static scoreboard overlay
    cv2.putText(img, "LAL 88  SAS 91  Q4 2:31", (28, 503), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return img


def keypoints_for(P, conf=0.9, noise=0.0, rng=None):
    img = apply_h(np.linalg.inv(floor_homography(P)), NBA.keypoints)
    inside = (img[:, 0] > 0) & (img[:, 0] < SIZE[0]) & (img[:, 1] > 0) & (img[:, 1] < SIZE[1])
    if noise and rng is not None:
        img = img + rng.normal(0, noise, img.shape)
    return np.column_stack([img, np.where(inside, conf, 0.0)])


def mapping_error(H, P):
    """Median court-metre error over a grid of visible floor pixels."""
    us, vs = np.meshgrid(np.linspace(50, SIZE[0] - 50, 12), np.linspace(SIZE[1] * 0.45, SIZE[1] - 30, 6))
    pix = np.column_stack([us.ravel(), vs.ravel()])
    return float(np.median(np.linalg.norm(apply_h(H, pix) - apply_h(floor_homography(P), pix), axis=1)))


def test_camera_motion_matches_true_pan():
    cams = pan_sequence(30)
    tex = floor_texture()
    frames = [render(P, tex) for P in cams]
    motion, ratios = CameraMotionEstimator().run(frames)
    assert motion[0] is None and all(m is not None for m in motion[1:])
    for t in (5, 15, 29):
        true = np.linalg.inv(floor_homography(cams[t])) @ floor_homography(cams[t - 1])
        pts = np.array([[100, 300], [480, 400], [850, 500]], float)
        assert np.abs(apply_h(motion[t], pts) - apply_h(true, pts)).max() < 1.0


def test_cut_breaks_motion():
    cams = pan_sequence(10)
    frames = [render(P, floor_texture()) for P in cams[:5]] + [render(P, floor_texture(7)) * 0 + 40
                                                                for P in cams[5:]]
    motion, _ = CameraMotionEstimator().run(frames)
    assert motion[5] is None


def test_tracking_through_pan_beats_hold():
    rng = np.random.default_rng(1)
    n = 120
    cams = pan_sequence(n)
    tex = floor_texture()
    frames = [render(P, tex) for P in cams]
    motion, _ = CameraMotionEstimator().run(frames)

    # Keypoints only every 25 frames, noisy; frame 60 has left and right swapped (a mislabel).
    kps = [np.zeros((18, 3)) for _ in range(n)]
    for f in range(0, n, 25):
        kps[f] = keypoints_for(cams[f], noise=1.0, rng=rng)
    bad = keypoints_for(cams[60])
    kps[60] = bad[[15, 14, 13, 12, 11, 10, 6, 7, 16, 17, 5, 4, 3, 2, 1, 0, 8, 9]]

    Hs, info = track_homographies(kps, motion, NBA)
    assert all(H is not None for H in Hs)
    errs = [mapping_error(H, P) for H, P in zip(Hs, cams)]
    assert np.median(errs) < 0.25 and max(errs) < 0.8, (np.median(errs), max(errs))
    assert info[60]["kp_rejected"] and info[10]["source"] == "flow"

    old = estimate_homographies(kps, NBA, hold_frames=15)
    old_errs = [mapping_error(H, P) for H, P in zip(old, cams) if H is not None]
    assert sum(H is None for H in old) > 30          # the old way loses the court between fits
    assert np.median(old_errs) > 2 * np.median(errs)  # and drifts while holding


def test_relink_and_smoothing():
    fps, n = 30.0, 60
    # Track 1 is lost at frame 29 and comes back as track 7 at frame 33, 0.3 m further on.
    pos = {1: [(5 + 0.05 * f, 5.0) if f < 30 else None for f in range(n)],
           7: [(5 + 0.05 * f, 5.0) if f >= 33 else None for f in range(n)],
           2: [(5 + 0.05 * f, 12.0) if f >= 33 else None for f in range(n)]}  # far away: not a match
    tracks = [{t: {"bbox": [0, 0, 1, 1], "conf": 0.9} for t, p in pos.items() if p[f] is not None} for f in range(n)]
    remap = stitch_tracks(tracks, pos, {1: 0, 7: 0, 2: 0}, fps)
    assert remap == {7: 1}
    assert set().union(*apply_remap(tracks, remap)) == {1, 2}

    noise = np.random.default_rng(0).normal(0, 0.15, n)
    jitter = {1: [(5 + 0.05 * f + noise[f], 5.0) if f not in (20, 21) else None for f in range(n)]}
    jitter[1][40] = (15.0, 5.0)  # one-frame spike
    sm = smooth_court_positions(jitter, fps)[1]
    assert sm[20] is not None and abs(sm[20][0] - 6.0) < 0.15     # gap filled
    assert abs(sm[40][0] - 7.0) < 0.2                               # spike removed
    raw_err = np.std([noise[f] for f in range(10, 50)])
    assert np.std([sm[f][0] - (5 + 0.05 * f) for f in range(10, 50)]) < 0.7 * raw_err


def test_run_writes_diagnostics(tmp_path, monkeypatch):
    """pipe.run on a rendered pan with stubbed models: camera tracking on, diagnostics written."""
    import json
    from courtvision.config import Config
    from courtvision.io.video import save_video
    from courtvision.pipeline import Pipeline
    from courtvision.types import DetectionResult

    n = 60
    cams = pan_sequence(n)
    tex = floor_texture()
    frames = [render(P, tex) for P in cams]
    video = tmp_path / "pan.mp4"
    save_video(frames, video, 30.0)
    kp_model = tmp_path / "court_keypoints.pt"
    kp_model.write_bytes(b"stub")

    players = []
    for P in cams:
        G = np.linalg.inv(floor_homography(P))
        fp = {}
        for tid, (x, y) in {1: (12.0, 6.0), 2: (15.0, 9.0), 3: (13.0, 4.0)}.items():
            u, v = apply_h(G, np.array([[x, y]]))[0]
            fp[tid] = {"bbox": [u - 15, v - 90, u + 15, v], "conf": 0.9}
        players.append(fp)
    det = DetectionResult(players=players, referees=[{}] * n, ball_candidates=[[]] * n, hoops=[None] * n)
    kps = [keypoints_for(P) if f % 20 == 0 else np.zeros((18, 3)) for f, P in enumerate(cams)]

    cfg = Config.load(None, output={"dir": str(tmp_path / "out"), "cache_dir": None, "write_video": False,
                                    "write_3d_html": False},
                      models={"court_keypoints": str(kp_model)})
    pipe = Pipeline(cfg)
    monkeypatch.setattr(pipe, "detect", lambda *a: det)
    monkeypatch.setattr(pipe, "court_keypoints", lambda *a: kps)
    result = pipe.run(video)

    d = json.loads((tmp_path / "out" / "pan" / "diagnostics.json").read_text())
    assert d["court"]["mapped_share"] == 1.0 and d["court"]["flow_share"] > 0.8
    assert d["ball"]["raw_detection_share"] == 0.0
    assert any(f["stage"] == "ball detection" for f in d["findings"])
    assert (tmp_path / "out" / "pan" / "diagnostics_frames.csv").exists()
    # Player 1 stands still on the court while the camera pans: its court position must stay put.
    xs = np.array([p for p in result.court_positions[1] if p is not None])
    assert np.abs(xs - (12.0, 6.0)).max() < 0.5, np.abs(xs - (12.0, 6.0)).max()
