"""The Frame: one rectified RGB image plus whatever Aria sensors had to say."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class GazeSample:
    """Eye gaze, already reduced to something the pipeline can use.

    ``point_px`` is the gaze projected into the *rectified* image, or None when
    the ray falls outside the frame or no gaze data exists for this timestamp.
    """

    yaw_rad: float
    pitch_rad: float
    depth_m: float | None = None
    point_px: tuple[float, float] | None = None
    valid: bool = True


@dataclass
class HandSample:
    """One hand, however it was found.

    ``landmarks_px`` is optional: Aria's on-device tracker and MediaPipe both
    give joints, the open-vocab fallback gives only a box.
    """

    bbox_xyxy: tuple[float, float, float, float]
    side: str = "unknown"  # "left" | "right" | "unknown"
    score: float = 1.0
    landmarks_px: np.ndarray | None = None  # (N, 2) float32
    source: str = "unknown"  # "aria" | "mediapipe" | "openvocab"


@dataclass
class Frame:
    """One processed frame handed to the detector.

    ``rgb`` is always uint8 HxWx3 in RGB order and already rectified, rotated
    upright and scaled — sources are responsible for that so the detector never
    has to care where the pixels came from.
    """

    rgb: np.ndarray
    timestamp_ns: int
    frame_idx: int

    calib: Any | None = None  # projectaria CameraCalibration for `rgb`, if any
    gaze: GazeSample | None = None
    hands: list[HandSample] = field(default_factory=list)

    # Scale applied from the full-res rectified image to `rgb`, so boxes can be
    # mapped back to full resolution if needed.
    scale: float = 1.0
    source: str = "unknown"
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def size(self) -> tuple[int, int]:
        """(width, height) of `rgb`."""
        h, w = self.rgb.shape[:2]
        return w, h
