"""CourtVision: basketball video to tracking data, touch-by-touch events and 3D plays."""
from .config import Config

__all__ = ["Config", "Pipeline"]
__version__ = "0.1.0"


def __getattr__(name):
    # Lazy so `import courtvision` does not pull in torch/ultralytics.
    if name == "Pipeline":
        from .pipeline import Pipeline
        return Pipeline
    raise AttributeError(name)
