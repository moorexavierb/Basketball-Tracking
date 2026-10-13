"""Draw a top-down court from its real dimensions and place players on it."""
from __future__ import annotations

import cv2
import numpy as np

from ..court.geometry import CourtSpec

FLOOR = (120, 170, 214)   # BGR hardwood
LINE = (255, 255, 255)


class TacticalRenderer:
    def __init__(self, court: CourtSpec, px_per_m: float = 12.0, pad_m: float = 1.0):
        self.court = court
        self.s = px_per_m
        self.pad = pad_m
        self.size = (int((court.length + 2 * pad_m) * px_per_m), int((court.width + 2 * pad_m) * px_per_m))
        self._base = self._draw_court()

    def to_px(self, x: float, y: float) -> tuple[int, int]:
        return int(round((x + self.pad) * self.s)), int(round((y + self.pad) * self.s))

    def _draw_court(self) -> np.ndarray:
        c, s = self.court, self.s
        w, h = self.size
        img = np.full((h, w, 3), FLOOR, np.uint8)
        t = max(1, int(s / 10))
        L, W = c.length, c.width
        cv2.rectangle(img, self.to_px(0, 0), self.to_px(L, W), LINE, t)
        cv2.line(img, self.to_px(L / 2, 0), self.to_px(L / 2, W), LINE, t)
        cv2.circle(img, self.to_px(L / 2, W / 2), int(c.center_circle_radius * s), LINE, t)
        lane_lo = (W - c.lane_width) / 2
        for side in (0, 1):
            bx = 0 if side == 0 else L
            sign = 1 if side == 0 else -1
            ft_x = bx + sign * c.free_throw_dist
            cv2.rectangle(img, self.to_px(min(bx, ft_x), lane_lo), self.to_px(max(bx, ft_x), W - lane_lo), LINE, t)
            cv2.circle(img, self.to_px(ft_x, W / 2), int(1.8 * s), LINE, t)
            hx, hy, _ = c.hoops[side]
            cv2.circle(img, self.to_px(hx, hy), max(2, int(0.23 * s)), (0, 90, 255), t)
            # Three-point line: corner straights plus the arc between them.
            corner_y = W / 2 - c.corner_three_offset
            arc_x = np.sqrt(max(c.three_radius ** 2 - corner_y ** 2, 0))
            for y in (c.corner_three_offset, W - c.corner_three_offset):
                cv2.line(img, self.to_px(bx, y), self.to_px(hx + sign * arc_x, y), LINE, t)
            ang = np.degrees(np.arctan2(corner_y, arc_x))
            start, end = (-ang, ang) if side == 0 else (180 - ang, 180 + ang)
            cv2.ellipse(img, self.to_px(hx, hy), (int(c.three_radius * s),) * 2, 0, start, end, LINE, t)
        return img

    def render(self, players: dict[int, tuple[float, float]], team_of: dict[int, int],
               team_colors: list[tuple[int, int, int]], ball: tuple[float, float] | None = None,
               possessor: int | None = None) -> np.ndarray:
        img = self._base.copy()
        r = max(4, int(0.4 * self.s))
        for tid, (x, y) in players.items():
            color = team_colors[team_of.get(tid, 0)] if team_of.get(tid, -1) in (0, 1) else (128, 128, 128)
            p = self.to_px(x, y)
            cv2.circle(img, p, r, color, -1)
            cv2.circle(img, p, r, (0, 0, 0) if tid != possessor else (0, 0, 255), 2)
            cv2.putText(img, str(tid), (p[0] - r, p[1] - r - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 1)
        if ball is not None:
            cv2.circle(img, self.to_px(*ball), max(3, r // 2), (0, 140, 255), -1)
        return img
