"""aria_devices — laptop / tablet / phone / hand detection for Project Aria Gen 2."""

from __future__ import annotations

__version__ = "0.1.0"

from .config import (
    CANONICAL_DEVICES,
    CANONICAL_LABELS,
    HAND,
    LAPTOP,
    PHONE,
    TABLET,
    DetectorConfig,
    PipelineConfig,
    load_config,
    save_config,
)
from .detect.base import Detection
from .frames import Frame, GazeSample, HandSample

__all__ = [
    "CANONICAL_DEVICES",
    "CANONICAL_LABELS",
    "Detection",
    "DetectorConfig",
    "Frame",
    "GazeSample",
    "HAND",
    "HandSample",
    "LAPTOP",
    "PHONE",
    "PipelineConfig",
    "TABLET",
    "__version__",
    "load_config",
    "save_config",
]
