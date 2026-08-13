"""Eye gaze: project the gaze ray into the rectified image and attribute it.

This is the reason to run the study on Aria rather than a webcam. Knowing that
a laptop is in frame is cheap; knowing the wearer was *looking* at it, and for
how long, is the actual measurement.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Sequence

import numpy as np

from .config import HAND
from .detect.base import Detection
from .frames import GazeSample

log = logging.getLogger(__name__)


class GazeProjector:
    """Projects Aria eye gaze into a rectified camera image.

    The transform chain is CPF (Central Pupil Frame, where gaze is expressed)
    -> Device -> Camera -> pixels. ``camera_calib`` must be the calibration of
    the image you are projecting into — i.e. the *rectified, rotated* linear
    calib, not the raw fisheye one. Getting that wrong puts the cursor in
    plausible-looking but consistently wrong places.
    """

    def __init__(self, device_calib, camera_calib, default_depth_m: float = 1.0) -> None:
        self.device_calib = device_calib
        self.camera_calib = camera_calib
        self.default_depth_m = default_depth_m
        self._T_device_cpf = self._resolve_device_cpf(device_calib)

    @staticmethod
    def _resolve_device_cpf(device_calib):
        """Fetch T_Device_CPF, tolerating naming differences across versions."""
        if device_calib is None:
            return None
        for name in ("get_transform_device_cpf", "get_transform_device_sensor"):
            fn = getattr(device_calib, name, None)
            if fn is None:
                continue
            try:
                return fn() if name == "get_transform_device_cpf" else fn("camera-rgb")
            except Exception:  # pragma: no cover - version dependent
                continue
        log.warning("device calibration exposes no CPF transform; gaze will be approximate")
        return None

    def project(self, eye_gaze, depth_m: float | None = None) -> GazeSample:
        """Project one EyeGaze record into pixels.

        Returns a GazeSample whose ``point_px`` is None when the ray leaves the
        image, which is common — the wearer looks well outside the RGB camera's
        rectified FOV all the time.
        """
        yaw = float(getattr(eye_gaze, "yaw", 0.0))
        pitch = float(getattr(eye_gaze, "pitch", 0.0))
        depth = depth_m
        if depth is None:
            d = getattr(eye_gaze, "depth", None)
            depth = float(d) if d else self.default_depth_m
        if not depth or depth <= 0:
            depth = self.default_depth_m

        sample = GazeSample(yaw_rad=yaw, pitch_rad=pitch, depth_m=depth)

        point_cpf = self._gaze_point_cpf(yaw, pitch, depth)
        if point_cpf is None or self.camera_calib is None:
            sample.valid = False
            return sample

        point_cam = point_cpf
        if self._T_device_cpf is not None:
            try:
                T_device_cam = self.camera_calib.get_transform_device_camera()
                point_device = self._T_device_cpf @ point_cpf
                point_cam = T_device_cam.inverse() @ point_device
            except Exception as exc:  # pragma: no cover - version dependent
                log.debug("gaze transform failed (%s); using CPF ray directly", exc)
                point_cam = point_cpf

        point_cam = np.asarray(point_cam, dtype=np.float64).reshape(-1)[:3]
        if point_cam[2] <= 0:  # behind the camera
            sample.valid = False
            return sample

        try:
            pixel = self.camera_calib.project(point_cam)
        except Exception as exc:  # pragma: no cover
            log.debug("gaze projection failed: %s", exc)
            pixel = None
        if pixel is None:
            sample.valid = False
            return sample

        px = np.asarray(pixel, dtype=np.float64).reshape(-1)
        w, h = (int(v) for v in self.camera_calib.get_image_size())
        if not (0 <= px[0] < w and 0 <= px[1] < h):
            sample.valid = False
            return sample

        sample.point_px = (float(px[0]), float(px[1]))
        return sample

    @staticmethod
    def _gaze_point_cpf(yaw: float, pitch: float, depth: float):
        """3D gaze point in CPF at a given depth."""
        try:
            from projectaria_tools.core.mps import get_eyegaze_point_at_depth

            return np.asarray(get_eyegaze_point_at_depth(yaw, pitch, depth), dtype=np.float64)
        except Exception:
            # Aria's convention: x = tan(yaw) * z, y = tan(pitch) * z.
            return np.array(
                [depth * math.tan(yaw), depth * math.tan(pitch), depth], dtype=np.float64
            )


def rotate_point_cw90(
    point: tuple[float, float], src_size: tuple[int, int]
) -> tuple[float, float]:
    """Rotate a pixel coordinate 90 degrees clockwise.

    Must be applied to the gaze point whenever the image was rotated upright,
    or the cursor ends up mirrored into the wrong quadrant. ``src_size`` is the
    (width, height) *before* rotation; after rotation the image is (height,
    width).
    """
    x, y = point
    src_w, _src_h = src_size
    return (src_w - 1 - x, y)[::-1]  # (y, src_w - 1 - x)


class GazeAttributor:
    """Decides which detection the wearer is looking at, and for how long.

    A gaze point inside a box wins outright. Otherwise the nearest box within
    ``hit_radius_px`` wins — eye tracking has real angular error, and demanding
    a strict containment hit throws away most of the signal on small targets
    like a phone.
    """

    def __init__(self, hit_radius_px: float = 90.0) -> None:
        self.hit_radius_px = hit_radius_px
        self._dwell_ms: dict[int, float] = {}
        self._last_ts_ns: int | None = None
        self._last_target: int | None = None

    def attribute(
        self,
        detections: Sequence[Detection],
        gaze: GazeSample | None,
        timestamp_ns: int,
    ) -> Detection | None:
        """Mark the gazed-at detection in place and update its dwell time."""
        dt_ms = 0.0
        if self._last_ts_ns is not None:
            dt_ms = max(0.0, (timestamp_ns - self._last_ts_ns) / 1e6)
            if dt_ms > 1000.0:  # a gap this big means a seek, not a dwell
                dt_ms = 0.0
        self._last_ts_ns = timestamp_ns

        if gaze is None or gaze.point_px is None:
            self._last_target = None
            return None

        gx, gy = gaze.point_px
        # Hands are not gaze targets for this study; devices are.
        targets = [d for d in detections if d.label != HAND]

        inside = [d for d in targets if _contains(d.bbox_xyxy, gx, gy)]
        if inside:
            # Smallest containing box: a phone lying on a laptop should win.
            target = min(inside, key=lambda d: d.area)
        else:
            scored = [(_distance_to_box(d.bbox_xyxy, gx, gy), d) for d in targets]
            scored = [(dist, d) for dist, d in scored if dist <= self.hit_radius_px]
            if not scored:
                self._last_target = None
                return None
            target = min(scored, key=lambda t: t[0])[1]

        key = target.track_id if target.track_id is not None else id(target)
        if self._last_target == key:
            self._dwell_ms[key] = self._dwell_ms.get(key, 0.0) + dt_ms
        else:
            self._dwell_ms[key] = dt_ms
        self._last_target = key

        target.gazed_at = True
        target.gaze_dwell_ms = self._dwell_ms[key]
        return target

    def dwell_for(self, track_id: int) -> float:
        return self._dwell_ms.get(track_id, 0.0)


def _contains(box: tuple[float, float, float, float], x: float, y: float) -> bool:
    return box[0] <= x <= box[2] and box[1] <= y <= box[3]


def _distance_to_box(box: tuple[float, float, float, float], x: float, y: float) -> float:
    """Euclidean distance from a point to a box (0 if inside)."""
    dx = max(box[0] - x, 0.0, x - box[2])
    dy = max(box[1] - y, 0.0, y - box[3])
    return math.hypot(dx, dy)
