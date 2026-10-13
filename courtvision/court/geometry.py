"""Real-world court model, in metres.

Court frame: origin at one corner, x along the length (baseline to baseline),
y along the width (sideline to sideline), z up. The 18 keypoints follow the
ordering of the public Roboflow court-keypoint dataset used by the reference
repo, so the same labelled data can train our pose model, but positions use
true NBA (or FIBA) measurements instead of a rounded 28 x 15 m court.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property

import numpy as np

RIM_HEIGHT_M = 3.05

# Keypoint index each keypoint becomes under a horizontal image flip (left and
# right halves of the court swap). Goes into the pose dataset's data.yaml.
FLIP_IDX = [15, 14, 13, 12, 11, 10, 6, 7, 16, 17, 5, 4, 3, 2, 1, 0, 8, 9]


@dataclass(frozen=True)
class CourtSpec:
    name: str
    length: float
    width: float
    lane_width: float
    free_throw_dist: float       # baseline to free-throw line
    corner_three_offset: float   # sideline to corner-three line
    three_radius: float
    hoop_offset: float           # baseline to centre of rim
    center_circle_radius: float = 1.83
    restricted_radius: float = 1.22

    @cached_property
    def keypoints(self) -> np.ndarray:
        """(18, 2) keypoint positions in metres, in dataset order."""
        L, W = self.length, self.width
        lane_lo = (W - self.lane_width) / 2
        lane_hi = W - lane_lo
        c = self.corner_three_offset
        ft = self.free_throw_dist
        return np.array([
            (0, 0), (0, c), (0, lane_lo), (0, lane_hi), (0, W - c), (0, W),   # left baseline
            (L / 2, W), (L / 2, 0),                                            # half-court line
            (ft, lane_lo), (ft, lane_hi),                                      # left free-throw line
            (L, W), (L, W - c), (L, lane_hi), (L, lane_lo), (L, c), (L, 0),   # right baseline
            (L - ft, lane_lo), (L - ft, lane_hi),                              # right free-throw line
        ], dtype=np.float64)

    @cached_property
    def hoops(self) -> np.ndarray:
        """(2, 3) rim centres (x, y, z) for the left and right baskets."""
        return np.array([
            (self.hoop_offset, self.width / 2, RIM_HEIGHT_M),
            (self.length - self.hoop_offset, self.width / 2, RIM_HEIGHT_M),
        ])

    def nearest_hoop(self, x: float, y: float) -> int:
        return int(np.argmin(np.hypot(self.hoops[:, 0] - x, self.hoops[:, 1] - y)))

    def shot_distance(self, x: float, y: float) -> float:
        h = self.hoops[self.nearest_hoop(x, y)]
        return float(np.hypot(x - h[0], y - h[1]))

    def is_three(self, x: float, y: float) -> bool:
        hoop = self.hoops[self.nearest_hoop(x, y)]
        from_baseline = x if hoop[0] < self.length / 2 else self.length - x
        half_w = self.width / 2
        corner_y = half_w - self.corner_three_offset
        corner_len = self.hoop_offset + np.sqrt(max(self.three_radius ** 2 - corner_y ** 2, 0.0))
        if from_baseline <= corner_len:
            return abs(y - half_w) >= corner_y
        return np.hypot(x - hoop[0], y - hoop[1]) >= self.three_radius

    def in_bounds(self, x: float, y: float, margin: float = 1.5) -> bool:
        return -margin <= x <= self.length + margin and -margin <= y <= self.width + margin


NBA = CourtSpec("nba", length=28.65, width=15.24, lane_width=4.88, free_throw_dist=5.79,
                corner_three_offset=0.91, three_radius=7.24, hoop_offset=1.575)
FIBA = CourtSpec("fiba", length=28.0, width=15.0, lane_width=4.9, free_throw_dist=5.8,
                 corner_three_offset=0.9, three_radius=6.75, hoop_offset=1.575, center_circle_radius=1.8,
                 restricted_radius=1.25)


def get_court(standard: str) -> CourtSpec:
    return {"nba": NBA, "fiba": FIBA}[standard.lower()]
