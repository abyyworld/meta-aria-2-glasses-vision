"""Plain MP4 / webcam source, so the whole pipeline is testable without glasses.

No Aria dependency at all: importing this module must work on a machine that
has never heard of projectaria_tools.
"""

from __future__ import annotations

import logging
import math
import time
from collections.abc import Iterator, Sequence
from pathlib import Path

import cv2

from ..frames import Frame
from .base import FrameSource

log = logging.getLogger(__name__)

#: Rough horizontal FOV of a typical laptop/webcam, used to guess a focal
#: length when we have no calibration. A Mac built-in camera is around 54-60
#: degrees. This is a guess and the size prior it feeds is correspondingly
#: approximate — see README.
DEFAULT_WEBCAM_HFOV_DEG = 58.0


def focal_from_hfov(width_px: int, hfov_deg: float = DEFAULT_WEBCAM_HFOV_DEG) -> float:
    """Pinhole focal length in pixels implied by a horizontal FOV."""
    return (width_px / 2.0) / math.tan(math.radians(hfov_deg) / 2.0)


class ImageFrameSource(FrameSource):
    """One or more still images as a frame source.

    The fastest way to sanity-check detection quality: photograph the desk and
    run it, with no camera permissions, no glasses and no video encoding in the
    way. Tracking is meaningless here (every image is an unrelated scene), so
    callers should turn it off.
    """

    def __init__(self, paths: Sequence[str | Path], hfov_deg: float = DEFAULT_WEBCAM_HFOV_DEG) -> None:
        self.paths = [Path(p) for p in paths]
        missing = [p for p in self.paths if not p.exists()]
        if missing:
            raise RuntimeError(f"image not found: {missing[0]}")
        self.fps = 1.0
        self._hfov_deg = hfov_deg
        first = cv2.imread(str(self.paths[0]))
        if first is None:
            raise RuntimeError(f"could not decode image {self.paths[0]}")
        self.width, self.height = first.shape[1], first.shape[0]
        self.focal_px = focal_from_hfov(self.width, hfov_deg)

    def __len__(self) -> int:
        return len(self.paths)

    def __iter__(self) -> Iterator[Frame]:
        for idx, path in enumerate(self.paths):
            bgr = cv2.imread(str(path))
            if bgr is None:
                log.warning("skipping undecodable image %s", path)
                continue
            yield Frame(
                rgb=cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB),
                timestamp_ns=idx * 1_000_000_000,
                frame_idx=idx,
                source="image",
                meta={"path": str(path), "focal_px": focal_from_hfov(bgr.shape[1], self._hfov_deg)},
            )


class VideoFrameSource(FrameSource):
    """Reads an MP4 file or a live camera index via OpenCV.

    ``source`` is a path for a file or an int for a camera. Camera capture is
    opened at ``request_size`` and the driver's actual size is used, since
    webcams silently substitute the nearest supported mode.
    """

    def __init__(
        self,
        source: str | int,
        stride: int = 1,
        max_frames: int = 0,
        request_size: tuple[int, int] | None = None,
        request_fps: float | None = None,
        hfov_deg: float = DEFAULT_WEBCAM_HFOV_DEG,
        mirror: bool | None = None,
    ) -> None:
        self.source = source
        self.stride = max(1, stride)
        self.max_frames = max_frames
        self.is_camera = isinstance(source, int)
        # Mirroring a webcam makes hand movement feel right to the person in
        # front of it; a recorded file must never be flipped.
        self.mirror = self.is_camera if mirror is None else mirror

        backend = cv2.CAP_AVFOUNDATION if self.is_camera else cv2.CAP_ANY
        self.cap = cv2.VideoCapture(source, backend)
        if not self.cap.isOpened() and self.is_camera:
            self.cap = cv2.VideoCapture(source)  # retry with the default backend
        if not self.cap.isOpened():
            if self.is_camera:
                raise RuntimeError(
                    f"could not open camera {source}.\n"
                    "On macOS this is almost always a permissions problem, not a bug: the\n"
                    "app running this process needs camera access. Grant it under\n"
                    "System Settings > Privacy & Security > Camera (tick your terminal, or\n"
                    "the IDE you launched from), then restart that app and retry.\n"
                    "If the camera is simply a different index, try --index 1."
                )
            raise RuntimeError(f"could not open video source {source!r}")

        if self.is_camera:
            if request_size:
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, request_size[0])
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, request_size[1])
            if request_fps:
                self.cap.set(cv2.CAP_PROP_FPS, request_fps)
            # A 1-frame buffer keeps the preview from lagging behind reality.
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
        reported_fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.fps = float(reported_fps) if reported_fps and reported_fps > 1 else 30.0
        self.focal_px = focal_from_hfov(self.width, hfov_deg)

        log.info(
            "video source %r: %dx%d @ %.1f fps (focal~%.0f px)",
            source, self.width, self.height, self.fps, self.focal_px,
        )

    def __len__(self) -> int:
        if self.is_camera:
            raise TypeError("camera sources have no length")
        total = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0:
            raise TypeError("frame count unavailable for this file")
        n = total // self.stride
        return min(n, self.max_frames) if self.max_frames else n

    def __iter__(self) -> Iterator[Frame]:
        raw_idx = 0
        emitted = 0
        t0 = time.time()
        while True:
            ok, frame_bgr = self.cap.read()
            if not ok:
                break
            if raw_idx % self.stride != 0:
                raw_idx += 1
                continue
            raw_idx += 1

            if self.mirror:
                frame_bgr = cv2.flip(frame_bgr, 1)
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

            if self.is_camera:
                timestamp_ns = int((time.time() - t0) * 1e9)
            else:
                pos_ms = self.cap.get(cv2.CAP_PROP_POS_MSEC)
                timestamp_ns = int(pos_ms * 1e6) if pos_ms > 0 else int(emitted / self.fps * 1e9)

            yield Frame(
                rgb=rgb,
                timestamp_ns=timestamp_ns,
                frame_idx=emitted,
                source="camera" if self.is_camera else "video",
                meta={"focal_px": self.focal_px},
            )
            emitted += 1
            if self.max_frames and emitted >= self.max_frames:
                break

    def close(self) -> None:
        if self.cap is not None:
            self.cap.release()
