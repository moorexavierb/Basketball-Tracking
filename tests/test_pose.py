import numpy as np

from courtvision.court.homography import project
from courtvision.pose.estimator import match_poses_to_tracks
from courtvision.pose.lift import lift_poses
from courtvision.pose.skeleton import BONES, J
from courtvision.recon3d.ball3d import cameras_for_frames
from conftest import floor_homography


def standing_skeleton(x, y, height=2.0, yaw=0.0, arm_raise=0.0):
    """A 17-joint body at court point (x, y), facing direction yaw (radians from +x)."""
    H = height
    fwd = np.array([np.cos(yaw), np.sin(yaw), 0.0])
    left = np.array([-np.sin(yaw), np.cos(yaw), 0.0])
    up = np.array([0, 0, 1.0])
    base = np.array([x, y, 0.0])
    P = {}
    for side, s in (("left", 1), ("right", -1)):
        P[f"{side}_ankle"] = base + left * s * 0.065 * H + up * 0.04 * H
        P[f"{side}_knee"] = P[f"{side}_ankle"] + up * 0.246 * H
        P[f"{side}_hip"] = P[f"{side}_knee"] + up * 0.245 * H
        P[f"{side}_shoulder"] = base + left * s * 0.10 * H + up * (0.531 + 0.29 * 0.995) * H - fwd * 0.0
        elbow_dir = -up * np.cos(arm_raise) + fwd * np.sin(arm_raise)
        P[f"{side}_elbow"] = P[f"{side}_shoulder"] + elbow_dir * 0.180 * H
        P[f"{side}_wrist"] = P[f"{side}_elbow"] + elbow_dir * 0.146 * H
    mid_sh = (P["left_shoulder"] + P["right_shoulder"]) / 2
    P["nose"] = mid_sh + up * 0.11 * H + fwd * 0.05 * H
    for k, o in (("left_eye", 1), ("right_eye", -1)):
        P[k] = P["nose"] + up * 0.02 + left * o * 0.03
    for k, o in (("left_ear", 1), ("right_ear", -1)):
        P[k] = P["nose"] + left * o * 0.07 - fwd * 0.05
    names = sorted(J, key=J.get)
    return np.array([P[k] for k in names])


def test_bone_priors_match_synthetic_skeleton():
    X = standing_skeleton(10, 6)
    for a, b, L in BONES[:8]:
        assert abs(np.linalg.norm(X[a] - X[b]) - L * 2.0) < 0.02


def test_match_poses_to_tracks():
    tracks = {7: {"bbox": [100, 100, 150, 250]}, 9: {"bbox": [400, 120, 450, 260]}}
    boxes = np.array([[402, 118, 452, 262], [98, 102, 151, 249], [800, 100, 850, 200]])
    kps = np.arange(3 * 17 * 3, dtype=float).reshape(3, 17, 3)
    out = match_poses_to_tracks(boxes, kps, tracks)
    assert set(out) == {7, 9}
    assert np.array_equal(out[9], kps[0]) and np.array_equal(out[7], kps[1])


def test_lift_recovers_3d_body(broadcast_cam):
    P, K, R, t, size = broadcast_cam
    cams = cameras_for_frames([floor_homography(P)] * 20, size)
    rng = np.random.default_rng(0)
    truth, poses2d, court = [], [], {3: []}
    for f in range(20):
        X = standing_skeleton(10 + 0.05 * f, 6, yaw=0.4, arm_raise=1.2 * f / 19)
        uv = np.array([project(P, x)[0] for x in X]) + rng.normal(0, 0.7, (17, 2))
        poses2d.append({3: np.column_stack([uv, np.full(17, 0.9)])})
        court[3].append((10 + 0.05 * f, 6.0))
        truth.append(X)
    lifted = lift_poses(poses2d, cams, court, height_m=2.0)[3]
    err = np.array([np.linalg.norm(lifted[f] - truth[f], axis=1) for f in range(20)])
    assert np.median(err) < 0.12, np.median(err)
    # the moving arm is recovered too, not just the trunk
    wrists = err[:, [J["left_wrist"], J["right_wrist"]]]
    assert np.median(wrists) < 0.2, wrists
    assert abs(lifted[-1][J["left_ankle"], 2] - truth[-1][J["left_ankle"], 2]) < 0.15


def test_low_confidence_pose_is_skipped(broadcast_cam):
    P, K, R, t, size = broadcast_cam
    cams = cameras_for_frames([floor_homography(P)], size)
    X = standing_skeleton(10, 6)
    uv = np.array([project(P, x)[0] for x in X])
    kps = np.column_stack([uv, np.full(17, 0.1)])
    assert lift_poses([{1: kps}], cams, {1: [(10.0, 6.0)]}) == {}
