"""Detector backends. The backend is a config choice, never a hardcode."""

from __future__ import annotations

from ..config import DetectorConfig
from .base import Detection, Detector, containment, intersection_area, iou

__all__ = [
    "Detection",
    "Detector",
    "build_detector",
    "containment",
    "intersection_area",
    "iou",
]


def build_detector(cfg: DetectorConfig) -> Detector:
    """Instantiate the configured backend.

    Imports are deferred so that installing only one backend's dependencies is
    enough, and so the unit tests import nothing heavy.
    """
    backend = cfg.backend.lower()
    if backend in ("yoloworld", "yolo-world", "yoloe"):
        from .yoloworld import YoloWorldDetector

        return YoloWorldDetector(cfg)
    if backend in ("owlv2", "owl-v2", "owl"):
        from .owlv2 import Owlv2Detector

        return Owlv2Detector(cfg)
    raise ValueError(
        f"unknown detector backend {cfg.backend!r}; expected one of: yoloworld, yoloe, owlv2"
    )
