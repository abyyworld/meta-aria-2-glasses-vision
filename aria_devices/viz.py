"""Drawing: boxes, labels, hand skeletons, gaze cursor, and MP4 writing.

Everything here works in BGR because that is what OpenCV wants; callers hand in
RGB frames and get BGR back, converted once at the top of ``draw_frame``.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path

import cv2
import numpy as np

from .config import HAND, LAPTOP, PHONE, TABLET, VizConfig
from .detect.base import Detection
from .frames import GazeSample, HandSample
from .hands import HAND_CONNECTIONS

log = logging.getLogger(__name__)

# BGR. Chosen to stay distinguishable in the desaturated, often blown-out
# footage that egocentric cameras produce indoors.
LABEL_COLORS: dict[str, tuple[int, int, int]] = {
    LAPTOP: (80, 220, 100),    # green
    TABLET: (40, 165, 255),    # amber
    PHONE: (255, 170, 60),     # blue
    HAND: (220, 100, 240),     # magenta
}
DEFAULT_COLOR = (200, 200, 200)
GAZE_COLOR = (60, 60, 255)     # red


def color_for(label: str) -> tuple[int, int, int]:
    return LABEL_COLORS.get(label, DEFAULT_COLOR)


def draw_frame(
    image_rgb: np.ndarray,
    detections: Sequence[Detection],
    cfg: VizConfig,
    hands: Sequence[HandSample] = (),
    gaze: GazeSample | None = None,
    hud: Sequence[str] = (),
    interactions: Sequence = (),
) -> np.ndarray:
    """Render detections onto a copy of the frame. Returns BGR uint8."""
    canvas = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

    if cfg.draw_hands:
        for i, hand in enumerate(hands):
            pose = interactions[i] if i < len(interactions) else None
            _draw_hand(canvas, hand, cfg, pose)

    # Crosshair where a hand sits on a device, as in the v1 study. Matched by
    # index: track_id is None with tracking off, and None == None would draw a
    # crosshair on every device at once.
    for inter in interactions:
        idx = getattr(inter, "device_index", None)
        if inter.relative_xy is None or idx is None or idx >= len(detections):
            continue
        det = detections[idx]
        _draw_hand_on_device(canvas, det.bbox_xyxy, inter.relative_xy, color_for(det.label))

    for det in detections:
        if det.label == HAND and cfg.draw_hands and hands:
            continue  # already drawn with skeleton
        _draw_detection(canvas, det, cfg)

    if cfg.draw_gaze and gaze is not None and gaze.point_px is not None:
        _draw_gaze(canvas, gaze)

    if hud:
        _draw_hud(canvas, hud)
    return canvas


def _draw_detection(canvas: np.ndarray, det: Detection, cfg: VizConfig) -> None:
    x1, y1, x2, y2 = (int(round(v)) for v in det.bbox_xyxy)
    color = color_for(det.label)
    # A gazed-at device gets a heavier border — the whole point of doing this
    # on Aria is knowing which screen the wearer is actually attending to.
    thickness = cfg.box_thickness + 2 if det.gazed_at else cfg.box_thickness
    cv2.rectangle(canvas, (x1, y1), (x2, y2), color, thickness)

    parts = [det.label]
    if det.track_id is not None:
        parts.append(f"#{det.track_id}")
    parts.append(f"{det.score:.2f}")
    if det.gazed_at:
        parts.append(f"GAZE {det.gaze_dwell_ms:.0f}ms")
    label = " ".join(parts)

    _draw_label(canvas, label, (x1, y1), color, cfg.font_scale)

    if cfg.draw_signals and det.signals:
        keys = ("text", "shape", "size", "diag_cm")
        line = " ".join(f"{k}={det.signals[k]:.2f}" for k in keys if k in det.signals)
        if line:
            cv2.putText(
                canvas, line, (x1, min(canvas.shape[0] - 4, y2 + 14)),
                cv2.FONT_HERSHEY_SIMPLEX, cfg.font_scale * 0.8, color, 1, cv2.LINE_AA,
            )


def _draw_label(
    canvas: np.ndarray,
    text: str,
    origin: tuple[int, int],
    color: tuple[int, int, int],
    font_scale: float,
) -> None:
    x, y = origin
    (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
    top = max(0, y - th - baseline - 4)
    cv2.rectangle(canvas, (x, top), (x + tw + 6, top + th + baseline + 4), color, -1)
    cv2.putText(
        canvas, text, (x + 3, top + th + 2),
        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (20, 20, 20), 1, cv2.LINE_AA,
    )


def _draw_hand(canvas: np.ndarray, hand: HandSample, cfg: VizConfig, interaction=None) -> None:
    color = color_for(HAND)
    pts = hand.landmarks_px
    # The skeleton edge table is MediaPipe's numbering; Aria orders its 21
    # landmarks differently, so only draw joints for Aria rather than a
    # scrambled skeleton.
    if pts is not None and len(pts) >= 21:
        if hand.source != "aria":
            for a, b in HAND_CONNECTIONS:
                pa = (int(pts[a][0]), int(pts[a][1]))
                pb = (int(pts[b][0]), int(pts[b][1]))
                cv2.line(canvas, pa, pb, color, 2, cv2.LINE_AA)
        for p in pts:
            cv2.circle(canvas, (int(p[0]), int(p[1])), 3, (255, 255, 255), -1, cv2.LINE_AA)
    x1, y1, x2, y2 = (int(round(v)) for v in hand.bbox_xyxy)
    cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 1)

    text = f"hand {hand.side}"
    if interaction is not None and getattr(interaction, "pose", None) is not None:
        pose = interaction.pose.value if hasattr(interaction.pose, "value") else interaction.pose
        if pose != "unknown":
            text += f" [{pose.upper()}]"
        if interaction.device_label:
            text += f" on {interaction.device_label}"
    _draw_label(canvas, text, (x1, y1), color, cfg.font_scale)


def _draw_hand_on_device(
    canvas: np.ndarray,
    box: tuple[float, float, float, float],
    relative_xy: tuple[float, float],
    color: tuple[int, int, int],
) -> None:
    """Crosshair marking where on a device the hand is — the v1 behaviour."""
    x1, y1, x2, y2 = (int(round(v)) for v in box)
    rx, ry = relative_xy
    col = int(x1 + rx * (x2 - x1))
    row = int(y1 + ry * (y2 - y1))
    cv2.line(canvas, (x1, row), (x2, row), color, 1, cv2.LINE_AA)
    cv2.line(canvas, (col, y1), (col, y2), color, 1, cv2.LINE_AA)
    cv2.circle(canvas, (col, row), 5, (255, 255, 255), 1, cv2.LINE_AA)


def _draw_gaze(canvas: np.ndarray, gaze: GazeSample) -> None:
    assert gaze.point_px is not None
    x, y = (int(round(v)) for v in gaze.point_px)
    cv2.circle(canvas, (x, y), 16, GAZE_COLOR, 2, cv2.LINE_AA)
    cv2.circle(canvas, (x, y), 3, GAZE_COLOR, -1, cv2.LINE_AA)
    cv2.line(canvas, (x - 24, y), (x - 8, y), GAZE_COLOR, 1, cv2.LINE_AA)
    cv2.line(canvas, (x + 8, y), (x + 24, y), GAZE_COLOR, 1, cv2.LINE_AA)
    cv2.line(canvas, (x, y - 24), (x, y - 8), GAZE_COLOR, 1, cv2.LINE_AA)
    cv2.line(canvas, (x, y + 8), (x, y + 24), GAZE_COLOR, 1, cv2.LINE_AA)


def _draw_hud(canvas: np.ndarray, lines: Sequence[str]) -> None:
    pad = 6
    scale = 0.5
    sizes = [cv2.getTextSize(t, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)[0] for t in lines]
    w = max(s[0] for s in sizes) + 2 * pad
    h = sum(s[1] for s in sizes) + pad * (len(lines) + 1)
    overlay = canvas.copy()
    cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.45, canvas, 0.55, 0, canvas)
    y = pad
    for text, (_tw, th) in zip(lines, sizes):
        y += th
        cv2.putText(
            canvas, text, (pad, y), cv2.FONT_HERSHEY_SIMPLEX, scale,
            (240, 240, 240), 1, cv2.LINE_AA,
        )
        y += pad


class Mp4Writer:
    """Lazy VideoWriter that sizes itself from the first frame it is given."""

    def __init__(self, path: str | Path, fps: float = 30.0) -> None:
        self.path = Path(path)
        self.fps = max(1.0, float(fps))
        self._writer: cv2.VideoWriter | None = None
        self._size: tuple[int, int] | None = None
        self.frames_written = 0

    def write(self, frame_bgr: np.ndarray) -> None:
        h, w = frame_bgr.shape[:2]
        if self._writer is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._size = (w, h)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self._writer = cv2.VideoWriter(str(self.path), fourcc, self.fps, self._size)
            if not self._writer.isOpened():  # pragma: no cover
                raise RuntimeError(f"could not open video writer for {self.path}")
        elif self._size != (w, h):
            frame_bgr = cv2.resize(frame_bgr, self._size)
        self._writer.write(frame_bgr)
        self.frames_written += 1

    def close(self) -> None:
        if self._writer is not None:
            self._writer.release()
            self._writer = None

    def __enter__(self) -> "Mp4Writer":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
