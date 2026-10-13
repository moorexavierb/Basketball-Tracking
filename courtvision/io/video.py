"""Video reading and writing."""
from __future__ import annotations

from pathlib import Path
from typing import Iterator

import cv2
import numpy as np


def video_info(path: str | Path) -> tuple[float, int, tuple[int, int]]:
    """Return (fps, frame_count, (width, height))."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    cap.release()
    return fps, count, size


def iter_frames(path: str | Path, start: int = 0, end: int | None = None) -> Iterator[np.ndarray]:
    """Stream BGR frames without holding the whole video in memory."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {path}")
    if start:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    idx = start
    try:
        while end is None or idx < end:
            ok, frame = cap.read()
            if not ok:
                break
            yield frame
            idx += 1
    finally:
        cap.release()


def read_video(path: str | Path, start: int = 0, end: int | None = None) -> list[np.ndarray]:
    return list(iter_frames(path, start, end))


def save_video(frames: list[np.ndarray], path: str | Path, fps: float = 30.0) -> Path:
    """Write frames to .mp4 (mp4v) or .avi (XVID), creating parent folders."""
    if not frames:
        raise ValueError("No frames to write")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*("XVID" if path.suffix.lower() == ".avi" else "mp4v"))
    h, w = frames[0].shape[:2]
    writer = cv2.VideoWriter(str(path), fourcc, fps, (w, h))
    for frame in frames:
        writer.write(frame)
    writer.release()
    return path


def read_frames_at(path: str | Path, indices: list[int]) -> dict[int, np.ndarray]:
    """Fetch specific frames (sorted sequential read; seeking is unreliable on many codecs)."""
    wanted = set(indices)
    out: dict[int, np.ndarray] = {}
    if not wanted:
        return out
    last = max(wanted)
    for i, frame in enumerate(iter_frames(path, end=last + 1)):
        if i in wanted:
            out[i] = frame
    return out
