"""Offline Aria Gen 2 VRS source: rectify, rotate upright, gaze, native hands.

Image handling order matters and getting it wrong quietly destroys detection
quality:

1. read the raw RGB frame — fisheye, and on Gen 2 far larger than any detector
   wants
2. rectify fisheye -> pinhole against a linear destination calibration
3. rotate 90 degrees clockwise, pixels *and* calibration together, because Aria
   stores RGB rotated and every prior in this package assumes an upright,
   gravity-aligned view
4. downscale to the detector's input size, keeping the scale factor

Steps 2 and 3 use cached remap tables. ``distort_by_calibration`` rebuilds its
warp on every call, which at 12 MP makes the pipeline unusable; we build the
map once, cache it on disk, and hand it to ``cv2.remap``.
"""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Iterator
from pathlib import Path

import cv2
import numpy as np

from ..config import RectifyConfig
from ..frames import Frame, HandSample
from ..gaze import GazeProjector
from .base import FrameSource

log = logging.getLogger(__name__)

RGB_LABEL = "camera-rgb"
CACHE_DIR = Path.home() / ".cache" / "aria_devices" / "rectify_maps"


def build_rectify_maps(
    src_calib, dst_calib, cache: bool = True
) -> tuple[np.ndarray, np.ndarray]:
    """Build cv2.remap tables that warp a fisheye image to a pinhole one.

    Destination pixels are unprojected through the (linear) destination model
    and reprojected through the source fisheye model. The destination
    unprojection is done in numpy — it is just an inverse intrinsics matrix —
    but the fisheye projection has to go through the calibration object one
    point at a time, so at 704x704 this takes a few seconds. Hence the disk
    cache: the cost is paid once per (calibration, size) pair.
    """
    dst_w, dst_h = (int(v) for v in dst_calib.get_image_size())
    key = _cache_key(src_calib, dst_calib)
    cache_path = CACHE_DIR / f"{key}.npz"
    if cache and cache_path.exists():
        try:
            data = np.load(cache_path)
            log.info("loaded cached rectification map %s", cache_path.name)
            return data["map_x"], data["map_y"]
        except Exception as exc:  # pragma: no cover - corrupt cache
            log.warning("ignoring corrupt rectification cache: %s", exc)

    log.info("building rectification map %dx%d (one-off, then cached)", dst_w, dst_h)
    fx, fy = (float(v) for v in dst_calib.get_focal_lengths())
    cx, cy = (float(v) for v in dst_calib.get_principal_point())

    us, vs = np.meshgrid(np.arange(dst_w, dtype=np.float64), np.arange(dst_h, dtype=np.float64))
    rays = np.stack(
        [(us - cx) / fx, (vs - cy) / fy, np.ones_like(us)], axis=-1
    ).reshape(-1, 3)

    map_x = np.full(rays.shape[0], -1.0, dtype=np.float32)
    map_y = np.full(rays.shape[0], -1.0, dtype=np.float32)
    project = src_calib.project  # bound once; this is the hot loop
    for i, ray in enumerate(rays):
        px = project(ray)
        if px is not None:
            map_x[i] = px[0]
            map_y[i] = px[1]

    map_x = map_x.reshape(dst_h, dst_w)
    map_y = map_y.reshape(dst_h, dst_w)

    if cache:
        try:
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(cache_path, map_x=map_x, map_y=map_y)
        except Exception as exc:  # pragma: no cover
            log.warning("could not write rectification cache: %s", exc)
    return map_x, map_y


def _cache_key(src_calib, dst_calib) -> str:
    parts = [
        src_calib.get_label(),
        str(src_calib.get_model_name()),
        np.asarray(src_calib.get_projection_params()).tobytes().hex()[:64],
        str(tuple(int(v) for v in src_calib.get_image_size())),
        str(tuple(int(v) for v in dst_calib.get_image_size())),
        str(tuple(round(float(v), 4) for v in dst_calib.get_focal_lengths())),
    ]
    return hashlib.sha1("|".join(parts).encode()).hexdigest()[:20]


def rotate_image_cw90(image: np.ndarray) -> np.ndarray:
    """Rotate pixels 90 degrees clockwise, matching rotate_camera_calib_cw90deg."""
    return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)


#: How many clockwise quarter-turns each rotation mode is worth. Anticlockwise
#: is expressed as three clockwise turns because projectaria_tools only ships a
#: clockwise calibration rotation — applying it three times is the only way to
#: keep the calibration in step with an anticlockwise pixel rotation.
_ROTATION_QUARTER_TURNS = {"none": 0, "cw90": 1, "ccw90": 3}


def rotate_image(image: np.ndarray, rotation: str) -> np.ndarray:
    """Rotate pixels by a named rotation."""
    if rotation == "none":
        return image
    if rotation == "cw90":
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    if rotation == "ccw90":
        return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    raise ValueError(f"rotation must be one of {sorted(_ROTATION_QUARTER_TURNS)}, got {rotation!r}")


def rotate_calibration(calib, rotation: str):
    """Rotate a linear calibration to match ``rotate_image``.

    Pixels and calibration must always move together; rotating one without the
    other puts gaze projection and every physical prior in the wrong place.
    """
    turns = _ROTATION_QUARTER_TURNS.get(rotation)
    if turns is None:
        raise ValueError(f"rotation must be one of {sorted(_ROTATION_QUARTER_TURNS)}, got {rotation!r}")
    if turns == 0:
        return calib
    from projectaria_tools.core import calibration as cal

    out = calib
    for _ in range(turns):
        out = cal.rotate_camera_calib_cw90deg(out)
    return out


class VrsFrameSource(FrameSource):
    """Iterates RGB frames from a Gen 2 .vrs, with gaze and native hand poses."""

    def __init__(
        self,
        path: str | Path,
        rectify: RectifyConfig | None = None,
        stride: int = 1,
        max_frames: int = 0,
        mps_folder: str | None = None,
    ) -> None:
        try:
            from projectaria_tools.core import data_provider
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise ImportError(
                "projectaria-tools is required for VRS input: pip install 'aria-devices[vrs]'"
            ) from exc

        self.path = str(path)
        self.cfg = rectify or RectifyConfig()
        self.stride = max(1, stride)
        self.max_frames = max_frames

        self.provider = data_provider.create_vrs_data_provider(self.path)
        if self.provider is None:
            raise RuntimeError(f"could not open VRS file {self.path}")

        # Image quality flags. Devignetting needs a mask folder shipped
        # separately; enabling it without one raises, so we guard.
        if self.cfg.color_correct:
            try:
                self.provider.set_color_correction(True)
            except Exception as exc:
                log.warning("color correction unavailable: %s", exc)
        if self.cfg.devignette:
            try:
                if self.cfg.devignetting_mask_path:
                    self.provider.set_devignetting_mask_folder_path(
                        self.cfg.devignetting_mask_path
                    )
                self.provider.set_devignetting(True)
            except Exception as exc:
                log.warning("devignetting unavailable (needs mask folder): %s", exc)

        self.rgb_stream_id = self.provider.get_stream_id_from_label(RGB_LABEL)
        if self.rgb_stream_id is None:
            raise RuntimeError(f"{self.path} has no {RGB_LABEL} stream")
        self.num_frames = self.provider.get_num_data(self.rgb_stream_id)
        log.info("%s: %d RGB frames", Path(self.path).name, self.num_frames)

        cfg_img = self.provider.get_image_configuration(self.rgb_stream_id)
        nominal = getattr(cfg_img, "nominal_rate_hz", None)
        self.fps = float(nominal) if nominal else 30.0

        # -- optional streams ---------------------------------------------
        self.gaze_stream_id = self._maybe_stream("eyegaze")
        self.hand_stream_id = self._maybe_stream("handtracking")
        self.has_hand_tracking = self.hand_stream_id is not None

        # -- calibration ---------------------------------------------------
        self.device_calib = self.provider.get_device_calibration()
        self.src_calib = self.provider.get_sensor_calibration(
            self.rgb_stream_id
        ).camera_calibration()

        self._map_x: np.ndarray | None = None
        self._map_y: np.ndarray | None = None
        self.dst_calib = None
        self.final_calib = None

        if self.cfg.enabled:
            self._setup_rectification()
        else:
            self.final_calib = self.src_calib
            w, h = (int(v) for v in self.src_calib.get_image_size())
            self.focal_px = float(self.src_calib.get_focal_lengths()[0])

        self.gaze_projector = (
            GazeProjector(self.device_calib, self.final_calib)
            if self.gaze_stream_id is not None
            else None
        )
        self._mps_provider = self._open_mps(mps_folder) if mps_folder else None

    # -- setup helpers -----------------------------------------------------
    def _maybe_stream(self, label: str):
        try:
            sid = self.provider.get_stream_id_from_label(label)
        except Exception:
            return None
        if sid is None:
            return None
        try:
            if self.provider.get_num_data(sid) == 0:
                return None
        except Exception:
            return None
        log.info("found %s stream", label)
        return sid

    def _setup_rectification(self) -> None:
        from projectaria_tools.core import calibration as cal

        w, h = self.cfg.size
        self.dst_calib = cal.get_linear_camera_calibration(
            w, h, self.cfg.focal, RGB_LABEL, self.src_calib.get_transform_device_camera()
        )
        self._map_x, self._map_y = build_rectify_maps(self.src_calib, self.dst_calib)

        # Pixels and calibration must rotate together, or gaze projection
        # and every geometric prior lands in the wrong place.
        self.final_calib = rotate_calibration(self.dst_calib, self.cfg.rotation)

        self.focal_px = float(self.final_calib.get_focal_lengths()[0])
        log.info(
            "rectified to %s, focal %.1f px%s",
            tuple(int(v) for v in self.final_calib.get_image_size()),
            self.focal_px,
            f" (rotated {self.cfg.rotation})" if self.cfg.rotation != "none" else "",
        )

    def _open_mps(self, folder: str):
        try:
            from projectaria_tools.core import mps

            paths = mps.MpsDataPathsProvider(folder).get_data_paths()
            return mps.MpsDataProvider(paths)
        except Exception as exc:
            log.warning("could not open MPS folder %s: %s", folder, exc)
            return None

    # -- per frame ---------------------------------------------------------
    def _rectify(self, image: np.ndarray) -> np.ndarray:
        if self._map_x is None or self._map_y is None:
            return image
        out = cv2.remap(
            image, self._map_x, self._map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT
        )
        return rotate_image(out, self.cfg.rotation)

    def _gaze_for(self, timestamp_ns: int):
        if self.gaze_stream_id is None or self.gaze_projector is None:
            return None
        try:
            eye_gaze = self.provider.get_eye_gaze_data_by_time_ns(
                self.gaze_stream_id, timestamp_ns
            )
        except Exception:
            return None
        if eye_gaze is None:
            return None
        return self.gaze_projector.project(eye_gaze)

    def _hands_for(self, timestamp_ns: int) -> list[HandSample]:
        """Project Aria's on-device 3D hand landmarks into the rectified image.

        These are metric positions in device frame, which is strictly better
        than anything we can infer from pixels: it gives real depth, so the
        physical size prior stops being an approximation.
        """
        if self.hand_stream_id is None or self.final_calib is None:
            return []
        try:
            result = self.provider.get_hand_pose_data_by_time_ns(
                self.hand_stream_id, timestamp_ns
            )
        except Exception:
            return []
        if result is None:
            return []

        T_device_cam = self.final_calib.get_transform_device_camera()
        T_cam_device = T_device_cam.inverse()
        w_img, h_img = (int(v) for v in self.final_calib.get_image_size())

        out: list[HandSample] = []
        for side in ("left", "right"):
            hand = getattr(result, f"{side}_hand", None)
            if hand is None:
                continue
            landmarks = getattr(hand, "landmark_positions_device", None)
            if landmarks is None:
                continue
            pts_px: list[tuple[float, float]] = []
            depths: list[float] = []
            for p_device in np.asarray(landmarks, dtype=np.float64).reshape(-1, 3):
                p_cam = np.asarray(T_cam_device @ p_device, dtype=np.float64).reshape(-1)[:3]
                if p_cam[2] <= 0:
                    continue
                px = self.final_calib.project(p_cam)
                if px is None:
                    continue
                pts_px.append((float(px[0]), float(px[1])))
                depths.append(float(p_cam[2]))
            if len(pts_px) < 4:
                continue
            arr = np.asarray(pts_px, dtype=np.float32)
            x1, y1 = arr.min(axis=0)
            x2, y2 = arr.max(axis=0)
            pad = 0.10 * max(x2 - x1, y2 - y1)
            sample = HandSample(
                bbox_xyxy=(
                    float(max(0, x1 - pad)), float(max(0, y1 - pad)),
                    float(min(w_img, x2 + pad)), float(min(h_img, y2 + pad)),
                ),
                side=side,
                score=float(getattr(hand, "confidence", 1.0) or 1.0),
                landmarks_px=arr,
                source="aria",
            )
            # Real metric depth, straight off the glasses.
            setattr(sample, "_depth_m", float(np.median(depths)))
            out.append(sample)
        return out

    def __len__(self) -> int:
        n = self.num_frames // self.stride
        return min(n, self.max_frames) if self.max_frames else n

    def __iter__(self) -> Iterator[Frame]:
        emitted = 0
        for idx in range(0, self.num_frames, self.stride):
            image_data, record = self.provider.get_image_data_by_index(self.rgb_stream_id, idx)
            if image_data is None:
                continue
            raw = image_data.to_numpy_array()
            if raw.ndim == 2:  # shouldn't happen on RGB, but be safe
                raw = cv2.cvtColor(raw, cv2.COLOR_GRAY2RGB)

            rgb = self._rectify(raw) if self.cfg.enabled else raw
            timestamp_ns = int(record.capture_timestamp_ns)

            yield Frame(
                rgb=np.ascontiguousarray(rgb),
                timestamp_ns=timestamp_ns,
                frame_idx=emitted,
                calib=self.final_calib,
                gaze=self._gaze_for(timestamp_ns),
                hands=self._hands_for(timestamp_ns),
                source="vrs",
                meta={"vrs_index": idx, "focal_px": self.focal_px},
            )
            emitted += 1
            if self.max_frames and emitted >= self.max_frames:
                return

    def close(self) -> None:
        self.provider = None
