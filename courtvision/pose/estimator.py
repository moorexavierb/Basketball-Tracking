"""2D body keypoints for every tracked player.

A YOLO pose model (COCO-17 keypoints, downloaded by ultralytics on first use)
runs on full frames; each detected person is matched to a player track by box
overlap, so poses inherit the tracker's IDs and team assignment.
"""
from __future__ import annotations

from typing import Iterable

import numpy as np
from scipy.optimize import linear_sum_assignment
from tqdm.auto import tqdm

from ..types import PlayerTracks

Poses2D = list[dict[int, np.ndarray]]  # frame -> {track_id: (17, 3) array of u, v, confidence}


def box_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise IoU between (N, 4) and (M, 4) xyxy boxes."""
    a, b = np.asarray(a, float).reshape(-1, 4), np.asarray(b, float).reshape(-1, 4)
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area = lambda r: (r[:, 2] - r[:, 0]) * (r[:, 3] - r[:, 1])
    return inter / np.maximum(area(a)[:, None] + area(b)[None, :] - inter, 1e-9)


def match_poses_to_tracks(person_boxes: np.ndarray, person_kps: np.ndarray, tracks: dict[int, dict],
                          min_iou: float = 0.4) -> dict[int, np.ndarray]:
    """One-to-one assignment of detected people to player tracks by IoU."""
    if len(person_boxes) == 0 or not tracks:
        return {}
    ids = list(tracks)
    iou = box_iou(np.array([tracks[t]["bbox"] for t in ids]), person_boxes)
    rows, cols = linear_sum_assignment(-iou)
    return {int(ids[r]): np.asarray(person_kps[c], dtype=np.float32)
            for r, c in zip(rows, cols) if iou[r, c] >= min_iou}


class PoseEstimator:
    def __init__(self, model: str = "yolo11m-pose.pt", imgsz: int = 1280, conf: float = 0.25,
                 batch_size: int = 16, min_iou: float = 0.4, device: str | None = None):
        from ultralytics import YOLO

        self.model = YOLO(model)
        self.imgsz, self.conf, self.batch_size, self.min_iou, self.device = imgsz, conf, batch_size, min_iou, device

    def run(self, frames: Iterable[np.ndarray], players: PlayerTracks, total: int | None = None) -> Poses2D:
        out: Poses2D = []
        batch: list[np.ndarray] = []

        def flush():
            results = self.model.predict(batch, conf=self.conf, imgsz=self.imgsz, device=self.device, verbose=False)
            for res in results:
                f = len(out)
                if res.keypoints is None or len(res.boxes) == 0 or f >= len(players):
                    out.append({})
                    continue
                boxes = res.boxes.xyxy.cpu().numpy()
                kps = res.keypoints.data.cpu().numpy()
                out.append(match_poses_to_tracks(boxes, kps, players[f], self.min_iou))
            batch.clear()

        for frame in tqdm(frames, total=total, desc="pose"):
            batch.append(frame)
            if len(batch) == self.batch_size:
                flush()
        if batch:
            flush()
        return out
