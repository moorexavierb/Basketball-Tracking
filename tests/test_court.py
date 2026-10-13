import numpy as np

from courtvision.court.geometry import FIBA, FLIP_IDX, NBA
from courtvision.court.homography import (apply_h, camera_from_homography, estimate_homographies,
                                          fit_homography, project)
from conftest import floor_homography


def test_keypoints_shape_and_symmetry():
    kp = NBA.keypoints
    assert kp.shape == (18, 2)
    # Left and right free-throw lines mirror each other.
    assert np.allclose(kp[8, 0], NBA.length - kp[16, 0])
    assert FIBA.keypoints.shape == (18, 2)


def test_three_point_classification():
    hx, hy, _ = NBA.hoops[0]
    assert not NBA.is_three(hx + 3.0, hy)          # top of the key, inside the arc
    assert NBA.is_three(hx + 7.5, hy)              # beyond the 7.24 m arc
    assert NBA.is_three(0.5, 0.5)                  # left corner three (0.5 m from sideline)
    assert not NBA.is_three(0.5, 1.2)              # just inside the corner line
    right = NBA.hoops[1]
    assert NBA.is_three(right[0] - 7.5, right[1])


def test_homography_from_keypoints_roundtrip(broadcast_cam):
    P, *_ = broadcast_cam
    H_true = floor_homography(P)
    img = apply_h(np.linalg.inv(H_true), NBA.keypoints)
    kps = np.column_stack([img, np.ones(18)])
    kps[[0, 5, 10], 2] = 0.0  # some keypoints undetected
    H = fit_homography(kps, NBA)
    assert H is not None
    assert np.allclose(apply_h(H, img), NBA.keypoints, atol=1e-3)


def test_homography_rejects_collinear():
    kps = np.zeros((18, 3))
    kps[:6] = np.column_stack([np.linspace(100, 600, 6), np.linspace(100, 600, 6), np.ones(6)])
    assert fit_homography(kps, NBA) is None


def test_hold_and_smoothing(broadcast_cam):
    P, *_ = broadcast_cam
    img = apply_h(np.linalg.inv(floor_homography(P)), NBA.keypoints)
    good = np.column_stack([img, np.ones(18)])
    seq = [good, np.zeros((18, 3)), np.zeros((18, 3)), good]
    Hs = estimate_homographies(seq, NBA, hold_frames=1)
    assert Hs[0] is not None and Hs[1] is not None and Hs[2] is None and Hs[3] is not None


def test_camera_recovery_projects_3d_points(broadcast_cam):
    P, K, R, t, size = broadcast_cam
    cam = camera_from_homography(floor_homography(P), size)
    assert abs(cam["focal"] - K[0, 0]) / K[0, 0] < 0.01
    rim = NBA.hoops[0]
    assert np.allclose(project(cam["P"], rim), project(P, rim), atol=1.0)
    assert cam["center"][2] > 0  # camera above the floor


def test_flip_idx_mirrors_court():
    mirrored = NBA.keypoints.copy()
    mirrored[:, 0] = NBA.length - mirrored[:, 0]
    assert np.allclose(mirrored[FLIP_IDX], NBA.keypoints)
