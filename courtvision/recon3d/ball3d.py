"""Recover the ball's 3D path from one broadcast camera.

A homography only maps the floor, so a ball in the air projects to the wrong
place on the court. We instead recover the full camera per frame
(``camera_from_homography``) and use physics:

* **Held** (someone has possession): the ball is near the holder, so its
  height is where the camera ray through the ball pixel passes closest to the
  holder's vertical axis.
* **In flight** (passes, shots, loose balls): the ball follows a parabola under
  gravity. We fit its initial position and velocity (6 numbers) to minimise
  reprojection error over every frame of the flight, which gives height for
  free. Flights that do not fit one parabola (bounce passes, deflections) are
  split and refit.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
from scipy.optimize import least_squares
from scipy.ndimage import median_filter

from ..court.homography import camera_from_homography, project
from ..possession.acquisition import NO_POSSESSION
from ..types import BallTrack, bbox_center

G = np.array([0.0, 0.0, -9.81])
MAX_BALL_HEIGHT_M = 8.0  # higher than any real arc: the fit has latched onto noise
Point3 = tuple[float, float, float]


def cameras_for_frames(homographies: list[Optional[np.ndarray]], image_size: tuple[int, int],
                       smooth_window: int = 15) -> list[Optional[dict]]:
    raw = [camera_from_homography(H, image_size) if H is not None else None for H in homographies]
    focals = np.array([c["focal"] if c else np.nan for c in raw])
    valid = ~np.isnan(focals)
    if valid.sum() == 0:
        return raw
    filled = np.interp(np.arange(len(focals)), np.flatnonzero(valid), focals[valid])
    smooth = median_filter(filled, size=smooth_window, mode="nearest")
    return [camera_from_homography(H, image_size, focal=float(f)) if H is not None else None
            for H, f in zip(homographies, smooth)]


def pixel_ray(cam: dict, uv: tuple[float, float]) -> tuple[np.ndarray, np.ndarray]:
    d = np.linalg.solve(cam["R"], np.linalg.solve(cam["K"], np.array([uv[0], uv[1], 1.0])))
    return cam["center"], d / np.linalg.norm(d)


def ray_plane_z(cam: dict, uv: tuple[float, float], z: float) -> np.ndarray:
    c, d = pixel_ray(cam, uv)
    if abs(d[2]) < 1e-9:
        return c
    return c + d * ((z - c[2]) / d[2])


def ray_nearest_to_vertical(cam: dict, uv: tuple[float, float], xy: tuple[float, float]) -> np.ndarray:
    """Point on the camera ray closest to the vertical line through court point xy."""
    c, d = pixel_ray(cam, uv)
    dh = d[:2]
    denom = dh @ dh
    s = 0.0 if denom < 1e-12 else float((np.asarray(xy) - c[:2]) @ dh / denom)
    p = c + s * d
    p[2] = np.clip(p[2], 0.0, 3.6)
    return p


def fit_parabola(frames: list[int], uvs: np.ndarray, cams: list[dict], fps: float,
                 start_guess: np.ndarray, end_guess: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    t = (np.array(frames) - frames[0]) / fps
    T = max(t[-1], 1e-3)
    v_guess = (end_guess - start_guess) / T - 0.5 * G * T

    def traj(params, tt):
        return params[:3] + np.outer(tt, params[3:]) + 0.5 * np.outer(tt ** 2, G)

    def residuals(params):
        X = traj(params, t)
        reproj = np.concatenate([project(cam["P"], X[i])[0] - uvs[i] for i, cam in enumerate(cams)])
        floor = np.minimum(X[:, 2], 0.0) * 200.0  # the ball cannot go through the floor
        return np.concatenate([reproj, floor])

    sol = least_squares(residuals, np.concatenate([start_guess, v_guess]), loss="soft_l1", f_scale=3.0)
    X = traj(sol.x, t)
    err = float(np.median(np.abs(residuals(sol.x)[: 2 * len(t)])))
    return X, sol.x, err


def reconstruct_ball_3d(ball: BallTrack, possessor: list[int], cams: list[Optional[dict]],
                        player_court: dict[int, list[Optional[tuple[float, float]]]], fps: float,
                        min_flight_frames: int = 5, max_px_error: float = 6.0) -> list[Optional[Point3]]:
    n = len(ball)
    out: list[Optional[Point3]] = [None] * n

    # Held ball.
    for f in range(n):
        pid, b, cam = possessor[f], ball[f], cams[f]
        if b is None or cam is None or pid == NO_POSSESSION:
            continue
        pos = player_court.get(pid, [None] * n)[f]
        if pos is not None:
            out[f] = tuple(map(float, ray_nearest_to_vertical(cam, bbox_center(b), pos)))

    # Free ball: contiguous runs, each fitted as one or more parabolas.
    f = 0
    while f < n:
        if possessor[f] != NO_POSSESSION or ball[f] is None or cams[f] is None:
            f += 1
            continue
        start = f
        while f < n and possessor[f] == NO_POSSESSION and ball[f] is not None and cams[f] is not None:
            f += 1
        _fit_run(list(range(start, f)), ball, cams, out, fps, min_flight_frames, max_px_error)

    # Anything left (short runs): assume chest height.
    for f in range(n):
        if out[f] is None and ball[f] is not None and cams[f] is not None:
            out[f] = tuple(map(float, ray_plane_z(cams[f], bbox_center(ball[f]), 1.2)))
    return out


def _fit_run(frames, ball, cams, out, fps, min_len, max_err, depth=0):
    if len(frames) < min_len:
        return
    uvs = np.array([bbox_center(ball[f]) for f in frames])
    fcams = [cams[f] for f in frames]
    prev = out[frames[0] - 1] if frames[0] > 0 else None
    nxt = out[frames[-1] + 1] if frames[-1] + 1 < len(out) else None
    start = np.array(prev) if prev is not None else ray_plane_z(fcams[0], uvs[0], 2.0)
    end = np.array(nxt) if nxt is not None else ray_plane_z(fcams[-1], uvs[-1], 2.0)
    X, _, err = fit_parabola(frames, uvs, fcams, fps, start, end)
    plausible = err <= max_err and X[:, 2].max() < MAX_BALL_HEIGHT_M
    if plausible or depth >= 2 or len(frames) < 2 * min_len:
        if plausible:
            for f, p in zip(frames, X):
                out[f] = tuple(map(float, p))
        return
    mid = len(frames) // 2
    _fit_run(frames[:mid], ball, cams, out, fps, min_len, max_err, depth + 1)
    _fit_run(frames[mid:], ball, cams, out, fps, min_len, max_err, depth + 1)
