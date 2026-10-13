"""Broadcast-style overlays for the output video."""
from __future__ import annotations

import cv2
import numpy as np

from ..types import BBox, bbox_center


def draw_player(frame: np.ndarray, bbox: BBox, color, track_id: int, has_ball: bool = False,
                label: str | None = None) -> None:
    x1, _, x2, y2 = map(int, bbox)
    cx = (x1 + x2) // 2
    w = x2 - x1
    cv2.ellipse(frame, (cx, y2), (max(w // 2, 6), max(w // 6, 3)), 0, -45, 235, color, 2, cv2.LINE_AA)
    text = label or str(track_id)
    cv2.rectangle(frame, (cx - 16, y2 + 6), (cx + 16, y2 + 22), color, -1)
    cv2.putText(frame, text, (cx - 12, y2 + 19), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
    if has_ball:
        _triangle(frame, (cx, int(bbox[1]) - 4), (0, 0, 255))


def draw_ball(frame: np.ndarray, bbox: BBox) -> None:
    x, y = map(int, bbox_center(bbox))
    _triangle(frame, (x, int(bbox[1]) - 2), (0, 255, 0))


def _triangle(frame, tip, color):
    x, y = tip
    pts = np.array([[x, y], [x - 9, y - 18], [x + 9, y - 18]], np.int32)
    cv2.drawContours(frame, [pts], 0, color, -1)
    cv2.drawContours(frame, [pts], 0, (0, 0, 0), 1)


def draw_panel(frame: np.ndarray, lines: list[str], origin: tuple[int, int] | None = None) -> None:
    h, w = frame.shape[:2]
    pw, ph = 330, 24 + 22 * len(lines)
    x, y = origin or (w - pw - 20, h - ph - 20)
    overlay = frame.copy()
    cv2.rectangle(overlay, (x, y), (x + pw, y + ph), (255, 255, 255), -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
    for i, line in enumerate(lines):
        cv2.putText(frame, line, (x + 10, y + 26 + 22 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)


def paste_minimap(frame: np.ndarray, minimap: np.ndarray, width_frac: float = 0.3, alpha: float = 0.85) -> None:
    h, w = frame.shape[:2]
    mw = int(w * width_frac)
    mh = int(minimap.shape[0] * mw / minimap.shape[1])
    small = cv2.resize(minimap, (mw, mh))
    roi = frame[16:16 + mh, 16:16 + mw]
    cv2.addWeighted(small, alpha, roi, 1 - alpha, 0, roi)
