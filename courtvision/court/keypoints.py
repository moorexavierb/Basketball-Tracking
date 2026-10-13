"""Court keypoint detection with a YOLO pose model (18 keypoints, see geometry.py)."""
from __future__ import annotations

from typing import Iterable

import numpy as np
from tqdm.auto import tqdm


class CourtKeypointDetector:
    def __init__(self, model_path: str, device: str | None = None, imgsz: int = 640, batch_size: int = 16):
        from ultralytics import YOLO
        self.model = YOLO(model_path)
        self.device = device
        self.imgsz = imgsz
        self.batch_size = batch_size

    def run(self, frames: Iterable[np.ndarray], total: int | None = None) -> list[np.ndarray]:
        """Return per-frame (18, 3) arrays of (x, y, conf); undetected points have conf 0."""
        out: list[np.ndarray] = []
        batch: list[np.ndarray] = []

        def flush():
            for res in self.model.predict(batch, imgsz=self.imgsz, device=self.device, verbose=False, conf=0.25):
                out.append(_best_instance(res))
            batch.clear()

        for frame in tqdm(frames, total=total, desc="court keypoints"):
            batch.append(frame)
            if len(batch) == self.batch_size:
                flush()
        if batch:
            flush()
        return out


def _best_instance(res) -> np.ndarray:
    kp = res.keypoints
    if kp is None or kp.xy is None or len(kp.xy) == 0:
        return np.zeros((18, 3))
    i = int(res.boxes.conf.argmax()) if res.boxes is not None and len(res.boxes) else 0
    xy = kp.xy[i].cpu().numpy()
    conf = kp.conf[i].cpu().numpy() if kp.conf is not None else np.ones(len(xy))
    conf = np.where((xy[:, 0] > 0) & (xy[:, 1] > 0), conf, 0.0)
    return np.column_stack([xy, conf])
