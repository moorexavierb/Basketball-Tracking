"""Lift 2D body keypoints to 3D court coordinates with the recovered camera.

Each keypoint's 3D position lies on the camera ray through its pixel, so the
only unknown per joint is its depth along that ray (17 numbers per player per
frame). Depths are solved by least squares so that:

* bone lengths match anthropometric priors scaled to the player's height,
* the body stays within arm's reach of the player's tracked court position,
* no joint goes through the floor,
* depths change smoothly from the previous frame (resolves front/back flips).

Reprojection error is zero by construction, which is why this works from a
single broadcast camera without a learned lifting network. A learned lifter
(MotionBERT, VideoPose3D) or SMPL fitting can replace ``lift_pose`` later and
keep the same input and output.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
from scipy.optimize import least_squares

from .estimator import Poses2D
from .skeleton import ANKLES, BONES, HIPS

Pose3D = np.ndarray  # (17, 3) court metres (x, y, z), NaN where a joint was not seen


def _rays(cam: dict, uv: np.ndarray) -> np.ndarray:
    pix = np.column_stack([uv, np.ones(len(uv))])
    d = np.linalg.solve(cam["R"], np.linalg.solve(cam["K"], pix.T)).T
    return d / np.linalg.norm(d, axis=1, keepdims=True)


def lift_pose(kps: np.ndarray, cam: dict, root_xy: tuple[float, float], height_m: float = 2.0,
              min_conf: float = 0.3, prev_depth: Optional[np.ndarray] = None,
              reach_m: float = 0.6) -> tuple[Optional[Pose3D], Optional[np.ndarray]]:
    """Lift one player's (17, 3) keypoints. Returns (pose, depths) or (None, None)."""
    kps = np.asarray(kps, float)
    valid = kps[:, 2] >= min_conf
    if valid.sum() < 8 or not valid[list(HIPS)].all():
        return None, None
    c = np.asarray(cam["center"], float)
    d = _rays(cam, kps[:, :2])
    root = np.asarray(root_xy, float)

    # Initial depth: where each ray passes closest to the player's vertical axis.
    dh = d[:, :2]
    s0 = ((root - c[:2]) @ dh.T) / np.maximum((dh ** 2).sum(1), 1e-12)
    if prev_depth is not None:
        s0 = np.where(np.isnan(prev_depth), s0, prev_depth)

    idx = np.flatnonzero(valid)
    bones = [(a, b, L * height_m, min(kps[a, 2], kps[b, 2])) for a, b, L in BONES if valid[a] and valid[b]]
    ankles = [j for j in ANKLES if valid[j]]
    axis_s0 = ((root - c[:2]) @ dh[idx].T) / np.maximum((dh[idx] ** 2).sum(1), 1e-12)

    def points(s):
        X = np.full((17, 3), np.nan)
        X[idx] = c + d[idx] * s[:, None]
        return X

    def residuals(s):
        X = points(s)
        r = [(np.linalg.norm(X[a] - X[b]) - L) / 0.03 * np.sqrt(w) for a, b, L, w in bones]
        horiz = np.linalg.norm(X[idx, :2] - root, axis=1)
        r.extend(np.maximum(horiz - reach_m, 0.0) / 0.05)
        r.extend(np.minimum(X[idx, 2], 0.0) / 0.02)
        r.extend((X[ankles, 2] - 0.08) / 0.25)            # feet are usually near the floor
        r.extend((s - axis_s0) / 0.6)                      # weak pull toward the player's axis
        if prev_depth is not None:
            p = prev_depth[idx]
            ok = ~np.isnan(p)
            r.extend((s[ok] - p[ok]) / 0.08)
        return np.asarray(r)

    sol = least_squares(residuals, s0[idx], loss="soft_l1", f_scale=2.0, max_nfev=200)
    depth = np.full(17, np.nan)
    depth[idx] = sol.x
    return points(sol.x), depth


def smooth_track(poses: list[Optional[Pose3D]], window: int = 5) -> list[Optional[Pose3D]]:
    """Centred moving average per joint, ignoring frames and joints that are missing."""
    if window <= 1:
        return poses
    have = [i for i, p in enumerate(poses) if p is not None]
    if not have:
        return poses
    stack = np.full((len(poses), 17, 3), np.nan)
    for i in have:
        stack[i] = poses[i]
    h = window // 2
    out: list[Optional[Pose3D]] = [None] * len(poses)
    for i in have:
        win = stack[max(0, i - h): i + h + 1]
        with np.errstate(invalid="ignore"):
            cnt = (~np.isnan(win)).sum(0)
            m = np.nansum(win, 0) / np.maximum(cnt, 1)
        m[cnt == 0] = np.nan
        m[np.isnan(stack[i])] = np.nan
        out[i] = m
    return out


def lift_poses(poses2d: Poses2D, cams: list[Optional[dict]],
               court_positions: dict[int, list[Optional[tuple[float, float]]]], height_m: float = 2.0,
               min_conf: float = 0.3, smooth_window: int = 5) -> dict[int, list[Optional[Pose3D]]]:
    """Per track, per frame 3D skeletons (None where the pose could not be lifted)."""
    n = len(poses2d)
    out: dict[int, list[Optional[Pose3D]]] = {}
    prev: dict[int, tuple[int, np.ndarray]] = {}
    for f in range(n):
        cam = cams[f] if f < len(cams) else None
        if cam is None:
            prev.clear()  # camera cut or lost calibration: don't carry depths across it
            continue
        for tid, kps in poses2d[f].items():
            pos = court_positions.get(tid, [None] * n)
            root = pos[f] if f < len(pos) else None
            if root is None:
                continue
            last = prev.get(tid)
            prev_depth = last[1] if last is not None and f - last[0] <= 2 else None
            X, depth = lift_pose(kps, cam, root, height_m, min_conf, prev_depth)
            if X is None:
                continue
            out.setdefault(tid, [None] * n)[f] = X
            prev[tid] = (f, depth)
    return {tid: smooth_track(track, smooth_window) for tid, track in out.items()}
