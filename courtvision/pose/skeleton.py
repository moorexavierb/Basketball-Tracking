"""COCO-17 body keypoints and the bone-length priors used to lift them to 3D.

Lengths are fractions of standing height (anthropometric averages), so one
height prior scales the whole skeleton.
"""
from __future__ import annotations

JOINTS = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist",
    "left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle",
]
J = {name: i for i, name in enumerate(JOINTS)}

# (joint a, joint b, length as a fraction of height)
BONES: list[tuple[int, int, float]] = [
    (J["left_shoulder"], J["left_elbow"], 0.180), (J["left_elbow"], J["left_wrist"], 0.146),
    (J["right_shoulder"], J["right_elbow"], 0.180), (J["right_elbow"], J["right_wrist"], 0.146),
    (J["left_hip"], J["left_knee"], 0.245), (J["left_knee"], J["left_ankle"], 0.246),
    (J["right_hip"], J["right_knee"], 0.245), (J["right_knee"], J["right_ankle"], 0.246),
    (J["left_shoulder"], J["right_shoulder"], 0.200), (J["left_hip"], J["right_hip"], 0.130),
    (J["left_shoulder"], J["left_hip"], 0.290), (J["right_shoulder"], J["right_hip"], 0.290),
    # torso diagonals keep the trunk rigid
    (J["left_shoulder"], J["right_hip"], 0.334), (J["right_shoulder"], J["left_hip"], 0.334),
    (J["nose"], J["left_shoulder"], 0.135), (J["nose"], J["right_shoulder"], 0.135),
]

ANKLES = (J["left_ankle"], J["right_ankle"])
HIPS = (J["left_hip"], J["right_hip"])
