"""Image <-> court mapping and camera recovery.

``estimate_homographies`` turns per-frame keypoints into a per-frame image->court
homography (metres) with RANSAC, rejects fits with a large reprojection error,
holds the last good fit through short dropouts, and smooths jitter while
resetting on camera cuts.

``camera_from_homography`` recovers a full 3x4 projection matrix from the
floor homography (focal length from the orthogonality constraint, principal
point at the image centre). That is what lets us lift the ball off the floor
in 3D: a homography only knows about z = 0.
"""
from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from .geometry import CourtSpec


def fit_homography(kps: np.ndarray, court: CourtSpec, min_conf: float = 0.5,
                   max_error_m: float = 1.0) -> Optional[np.ndarray]:
    mask = kps[:, 2] >= min_conf
    if mask.sum() < 4:
        return None
    img = kps[mask, :2].astype(np.float64)
    world = court.keypoints[mask]
    # Four or more points on one line give a degenerate fit.
    if np.linalg.matrix_rank(np.column_stack([world, np.ones(len(world))])) < 3:
        return None
    H, inliers = cv2.findHomography(img, world, cv2.RANSAC, ransacReprojThreshold=0.75)
    if H is None or inliers is None or inliers.sum() < 4:
        return None
    err = np.linalg.norm(apply_h(H, img) - world, axis=1)[inliers.ravel() == 1]
    if np.median(err) > max_error_m:
        return None
    return H / H[2, 2]


def apply_h(H: np.ndarray, pts: np.ndarray) -> np.ndarray:
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
    return cv2.perspectiveTransform(pts[None], H)[0]


def estimate_homographies(keypoints: list[np.ndarray], court: CourtSpec, min_conf: float = 0.5,
                          hold_frames: int = 15, smoothing: float = 0.6,
                          cut_threshold_m: float = 2.0) -> list[Optional[np.ndarray]]:
    out: list[Optional[np.ndarray]] = []
    prev, since = None, 0
    probe = court.keypoints
    for kps in keypoints:
        H = fit_homography(kps, court, min_conf)
        if H is None:
            since += 1
            out.append(prev if prev is not None and since <= hold_frames else None)
            continue
        if prev is not None and since <= hold_frames:
            # Compare where both fits send the same image points; a big jump is a cut.
            img_probe = apply_h(np.linalg.inv(H), probe)
            jump = np.median(np.linalg.norm(apply_h(prev, img_probe) - probe, axis=1))
            if jump < cut_threshold_m:
                H = smoothing * prev + (1 - smoothing) * H
                H /= H[2, 2]
        prev, since = H, 0
        out.append(H)
    return out


def camera_from_homography(H_img_to_court: np.ndarray, image_size: tuple[int, int],
                           focal: float | None = None) -> dict:
    """Recover K and a 3x4 projection P (court metres -> pixels) from a floor homography.

    The court frame has z pointing up (camera above the floor).
    """
    w, h = image_size
    cx, cy = w / 2.0, h / 2.0
    G = np.linalg.inv(H_img_to_court)  # court -> image
    G = G / G[2, 2]
    if focal is None:
        focal = _focal_from_homography(G, cx, cy) or 1.2 * w
    K = np.array([[focal, 0, cx], [0, focal, cy], [0, 0, 1.0]])
    B = np.linalg.inv(K) @ G
    lam = 1.0 / np.linalg.norm(B[:, 0])
    r1, r2, t = lam * B[:, 0], lam * B[:, 1], lam * B[:, 2]
    # Court centre must be in front of the camera.
    if t[2] < 0:
        r1, r2, t = -r1, -r2, -t
    # Orthonormalise [r1 r2] (noise makes them slightly non-orthogonal).
    U, _, Vt = np.linalg.svd(np.column_stack([r1, r2]), full_matrices=False)
    r1, r2 = (U @ Vt).T
    r3 = np.cross(r1, r2)
    R = np.column_stack([r1, r2, r3])
    center = -R.T @ t
    if center[2] < 0:
        # Court axes are left-handed relative to "up": flip z so the camera sits above the floor.
        r3 = -r3
        R = np.column_stack([r1, r2, r3])
        center = -np.linalg.solve(R, t)
    P = K @ np.column_stack([R, t])
    return {"K": K, "R": R, "t": t, "P": P, "center": center, "focal": focal}


def _focal_from_homography(G: np.ndarray, cx: float, cy: float) -> float | None:
    T = np.array([[1, 0, -cx], [0, 1, -cy], [0, 0, 1.0]])
    A = T @ G
    h1, h2 = A[:, 0], A[:, 1]
    cands = []
    den = h1[2] * h2[2]
    if abs(den) > 1e-12:
        f2 = -(h1[0] * h2[0] + h1[1] * h2[1]) / den
        if f2 > 0:
            cands.append(f2)
    den = h1[2] ** 2 - h2[2] ** 2
    if abs(den) > 1e-12:
        f2 = -(h1[0] ** 2 + h1[1] ** 2 - h2[0] ** 2 - h2[1] ** 2) / den
        if f2 > 0:
            cands.append(f2)
    if not cands:
        return None
    return float(np.sqrt(np.median(cands)))


def project(P: np.ndarray, X: np.ndarray) -> np.ndarray:
    X = np.atleast_2d(X)
    x = (P @ np.column_stack([X, np.ones(len(X))]).T).T
    return x[:, :2] / x[:, 2:3]
