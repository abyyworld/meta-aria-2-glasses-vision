"""A dashboard of everything the Aria Gen 2 glasses are sensing, in one window.

Gen 2 is not one camera. It is an RGB camera, four SLAM cameras, two eye
tracking cameras, on-device gaze and hand tracking, VIO and IMU — and when
something looks wrong in the detector, the usual cause is upstream of the
detector. Seeing every stream side by side, with its actual rate, is how you
tell "the model is confused" from "the SLAM feed is dropping frames".

``render_dashboard`` is a pure function from a snapshot to a canvas, so the
layout is unit-testable with no hardware and no glasses attached.
"""

from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass, field

import cv2
import numpy as np

log = logging.getLogger(__name__)

# Dark UI so the camera feeds carry the brightness.
BG = (18, 18, 20)
PANEL_BG = (30, 30, 34)
BORDER = (70, 70, 78)
TEXT = (225, 225, 230)
DIM_TEXT = (150, 150, 158)
ACCENT = (80, 220, 100)
WARN = (40, 165, 255)
DEAD = (70, 70, 200)


@dataclass
class SensorPanel:
    """One sensor's latest frame, plus how it is doing."""

    name: str
    image: np.ndarray | None = None  # gray or BGR; None = no data yet
    rate_hz: float = 0.0
    subtitle: str = ""

    @property
    def is_live(self) -> bool:
        return self.image is not None


@dataclass
class AriaSnapshot:
    """Everything to draw for one instant."""

    timestamp_ns: int = 0
    main: np.ndarray | None = None  # annotated RGB view, BGR order
    main_title: str = "camera-rgb"
    panels: list[SensorPanel] = field(default_factory=list)
    stats: list[tuple[str, str]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


class RateMeter:
    """Rolling frequency of a stream, in Hz.

    Each sensor runs at its own rate — RGB around 30 Hz, IMU up to 1 kHz, VIO
    high-frequency far above that — so a single global FPS number tells you
    nothing about which stream stalled.
    """

    def __init__(self, window: int = 30) -> None:
        self._times: deque[float] = deque(maxlen=window)

    def tick(self) -> None:
        self._times.append(time.perf_counter())

    @property
    def hz(self) -> float:
        if len(self._times) < 2:
            return 0.0
        span = self._times[-1] - self._times[0]
        return (len(self._times) - 1) / span if span > 0 else 0.0

    @property
    def stale_for(self) -> float:
        """Seconds since the last sample; large means the stream died."""
        if not self._times:
            return float("inf")
        return time.perf_counter() - self._times[-1]


def _to_bgr(image: np.ndarray) -> np.ndarray:
    """SLAM and eye-tracking cameras are monochrome; RGB is not."""
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
    return image


def fit_into(image: np.ndarray, width: int, height: int) -> np.ndarray:
    """Letterbox an image into a box without distorting its aspect ratio.

    Distorting would be actively misleading here: these panels are how you
    judge whether the fisheye rectification is behaving.
    """
    canvas = np.full((height, width, 3), PANEL_BG, np.uint8)
    if image is None or image.size == 0:
        return canvas
    bgr = _to_bgr(image)
    h, w = bgr.shape[:2]
    if h == 0 or w == 0:
        return canvas
    scale = min(width / w, height / h)
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    resized = cv2.resize(bgr, (new_w, new_h), interpolation=interp)
    x0 = (width - new_w) // 2
    y0 = (height - new_h) // 2
    canvas[y0 : y0 + new_h, x0 : x0 + new_w] = resized
    return canvas


def _label(canvas: np.ndarray, text: str, org: tuple[int, int], color=TEXT, scale: float = 0.45) -> None:
    cv2.putText(canvas, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def _panel_status_color(panel: SensorPanel) -> tuple[int, int, int]:
    if not panel.is_live:
        return DEAD
    if panel.rate_hz > 0 and panel.rate_hz < 5.0:
        return WARN
    return ACCENT


def render_dashboard(
    snapshot: AriaSnapshot,
    width: int = 1600,
    height: int = 900,
    panel_columns: int = 2,
) -> np.ndarray:
    """Compose one dashboard frame. Returns a BGR uint8 canvas.

    Layout is a big main view on the left, a grid of sensor thumbnails on the
    right, and a stats strip along the bottom. Panels that have never produced
    data are still drawn, greyed out — a missing stream is information, and
    silently omitting it would hide exactly the failure you are looking for.
    """
    width = max(640, int(width))
    height = max(360, int(height))
    canvas = np.full((height, width, 3), BG, np.uint8)

    stats_h = 26 * max(1, (len(snapshot.stats) + 2) // 3) + 16 if snapshot.stats else 0
    body_h = height - stats_h
    header_h = 26

    main_w = int(width * 0.62) if snapshot.panels else width
    side_w = width - main_w

    # -- main view ---------------------------------------------------------
    _label(canvas, snapshot.main_title, (10, 18), TEXT, 0.5)
    main_box_h = body_h - header_h
    main_img = fit_into(snapshot.main, main_w - 12, main_box_h - 8) if main_box_h > 8 else None
    if main_img is not None:
        canvas[header_h : header_h + main_img.shape[0], 6 : 6 + main_img.shape[1]] = main_img
        cv2.rectangle(
            canvas, (6, header_h),
            (6 + main_img.shape[1], header_h + main_img.shape[0]), BORDER, 1,
        )

    # -- sensor grid -------------------------------------------------------
    if snapshot.panels and side_w > 80:
        cols = max(1, panel_columns)
        rows = (len(snapshot.panels) + cols - 1) // cols
        cell_w = (side_w - 8) // cols
        cell_h = max(48, (body_h - 8) // max(1, rows))

        for i, panel in enumerate(snapshot.panels):
            r, c = divmod(i, cols)
            x0 = main_w + 4 + c * cell_w
            y0 = 4 + r * cell_h
            if y0 + cell_h > body_h:
                break
            img_h = cell_h - 20
            thumb = fit_into(panel.image, cell_w - 6, img_h)
            canvas[y0 + 16 : y0 + 16 + img_h, x0 : x0 + thumb.shape[1]] = thumb

            status = _panel_status_color(panel)
            cv2.rectangle(canvas, (x0, y0 + 16), (x0 + thumb.shape[1], y0 + 16 + img_h), BORDER, 1)
            cv2.circle(canvas, (x0 + 6, y0 + 8), 3, status, -1, cv2.LINE_AA)

            rate = f"{panel.rate_hz:5.1f} Hz" if panel.is_live else "no data"
            _label(canvas, f"{panel.name}", (x0 + 14, y0 + 12), TEXT, 0.4)
            _label(canvas, rate, (x0 + thumb.shape[1] - 62, y0 + 12), DIM_TEXT, 0.38)
            if panel.subtitle:
                _label(canvas, panel.subtitle, (x0 + 3, y0 + 14 + img_h + 1), DIM_TEXT, 0.35)

    # -- stats strip -------------------------------------------------------
    if stats_h:
        y_base = body_h
        cv2.rectangle(canvas, (0, y_base), (width, height), PANEL_BG, -1)
        cv2.line(canvas, (0, y_base), (width, y_base), BORDER, 1)
        col_w = width // 3
        for i, (key, value) in enumerate(snapshot.stats):
            r, c = divmod(i, 3)
            x = 12 + c * col_w
            y = y_base + 22 + r * 22
            if y > height - 4:
                break
            _label(canvas, f"{key}", (x, y), DIM_TEXT, 0.42)
            _label(canvas, f"{value}", (x + 132, y), TEXT, 0.42)

    for i, note in enumerate(snapshot.notes[:3]):
        _label(canvas, note, (10, body_h - 10 - 16 * i), WARN, 0.42)

    return canvas
