"""Pickle caches ("stubs") so expensive model passes run once per video."""
from __future__ import annotations

import hashlib
import pickle
from pathlib import Path
from typing import Any, Callable


def cache_key(video_path: str | Path, stage: str, *parts: Any) -> str:
    p = Path(video_path)
    stat = p.stat() if p.exists() else None
    raw = f"{p.name}|{stat.st_size if stat else 0}|{stage}|{parts}"
    return f"{p.stem}_{stage}_{hashlib.md5(raw.encode()).hexdigest()[:8]}.pkl"


def cached(cache_dir: str | Path | None, key: str, compute: Callable[[], Any]) -> Any:
    """Return the cached value for key, computing and storing it on a miss."""
    if cache_dir is None:
        return compute()
    path = Path(cache_dir) / key
    if path.exists():
        with open(path, "rb") as f:
            return pickle.load(f)
    value = compute()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(value, f)
    return value
