"""Detector ABC and the Detection record every backend must produce."""

from __future__ import annotations

import abc
from dataclasses import dataclass, field

import numpy as np


@dataclass
class Detection:
    """One detection, from raw prompt hit through to final canonical label.

    ``raw_label`` is whatever prompt the open-vocab model matched; ``label`` is
    the canonical class after disambiguation. Keeping both means a bad collapse
    rule is debuggable after the fact.
    """

    bbox_xyxy: tuple[float, float, float, float]
    score: float
    raw_label: str
    label: str = ""

    track_id: int | None = None
    gazed_at: bool = False
    gaze_dwell_ms: float = 0.0

    # Per-signal breakdown, populated by the disambiguator. Logged at debug
    # level and optionally drawn on the frame so the weights can be tuned.
    signals: dict[str, float] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    #: True when this box was carried over from an earlier frame rather than
    #: seen in this one — see track.DevicePersistence. Consumers that must not
    #: act on remembered geometry can filter on it.
    persisted: bool = False

    @property
    def width(self) -> float:
        return max(0.0, self.bbox_xyxy[2] - self.bbox_xyxy[0])

    @property
    def height(self) -> float:
        return max(0.0, self.bbox_xyxy[3] - self.bbox_xyxy[1])

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def center(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.bbox_xyxy
        return (0.5 * (x1 + x2), 0.5 * (y1 + y2))

    @property
    def aspect_portrait(self) -> float:
        """Short edge / long edge, so always in (0, 1] regardless of rotation."""
        w, h = self.width, self.height
        if w <= 0 or h <= 0:
            return 0.0
        return min(w, h) / max(w, h)

    @property
    def is_landscape(self) -> bool:
        return self.width >= self.height


def iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    """Intersection over union of two xyxy boxes."""
    inter = intersection_area(a, b)
    if inter <= 0.0:
        return 0.0
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def intersection_area(
    a: tuple[float, float, float, float], b: tuple[float, float, float, float]
) -> float:
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    return (x2 - x1) * (y2 - y1)


def containment(inner: tuple[float, float, float, float], outer: tuple[float, float, float, float]) -> float:
    """Fraction of `inner` that lies inside `outer`."""
    area_inner = max(0.0, inner[2] - inner[0]) * max(0.0, inner[3] - inner[1])
    if area_inner <= 0.0:
        return 0.0
    return intersection_area(inner, outer) / area_inner


class Detector(abc.ABC):
    """An open-vocabulary detector behind one interface.

    Implementations must accept an arbitrary prompt list at construction and be
    swappable purely by config, so nothing downstream is allowed to depend on
    Ultralytics-specific behaviour.
    """

    #: Prompts this instance was configured with, in model class-index order.
    prompts: list[str]

    @abc.abstractmethod
    def detect(self, image_rgb: np.ndarray) -> list[Detection]:
        """Run detection on one uint8 HxWx3 RGB image."""

    def warmup(self) -> None:
        """Optional: run a dummy inference so the first real frame isn't slow."""
        return None

    def close(self) -> None:
        return None

    def __enter__(self) -> "Detector":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
