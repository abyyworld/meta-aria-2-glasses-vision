"""Drive the sensor dashboard from a VRS file or from the live glasses.

Two providers, one output type (``AriaSnapshot``), so ``viewer.render_dashboard``
never learns where the data came from.
"""

from __future__ import annotations

import logging
import math
import threading
import time
from collections.abc import Iterator
from pathlib import Path

import cv2
import numpy as np

from .config import PipelineConfig
from .viewer import AriaSnapshot, RateMeter, SensorPanel

log = logging.getLogger(__name__)

#: Camera streams to show as thumbnails, in display order. Gen 2 has four SLAM
#: cameras and two eye-tracking cameras; any that a recording lacks are shown
#: greyed out rather than dropped.
SIDE_CAMERA_LABELS = (
    "slam-front-left",
    "slam-front-right",
    "slam-side-left",
    "slam-side-right",
    "camera-et-left",
    "camera-et-right",
)


def _fmt_vec(vec, unit: str = "", digits: int = 2) -> str:
    try:
        arr = np.asarray(vec, dtype=float).reshape(-1)[:3]
    except Exception:
        return "-"
    return " ".join(f"{v:+.{digits}f}" for v in arr) + (f" {unit}" if unit else "")


class VrsMonitor:
    """Replays every sensor stream from a .vrs alongside the RGB frames.

    Non-RGB streams are sampled *by timestamp* rather than by index, because
    each runs at its own rate and index-aligning them would silently pair an
    RGB frame with a SLAM frame from a different moment.
    """

    def __init__(self, path: str | Path, cfg: PipelineConfig, detect: bool = True) -> None:
        from .sources.vrs import VrsFrameSource

        self.cfg = cfg
        self.source = VrsFrameSource(
            path, rectify=cfg.rectify, stride=cfg.stride, max_frames=cfg.max_frames
        )
        self.provider = self.source.provider
        self.path = Path(path)

        self._side_streams: dict[str, object] = {}
        for label in SIDE_CAMERA_LABELS:
            sid = self._maybe(label)
            if sid is not None:
                self._side_streams[label] = sid

        self._vio_stream = self._maybe("vio")
        self._meters: dict[str, RateMeter] = {}
        self.detect = detect
        self._pipeline = None
        if detect:
            from .pipeline import DevicePipeline

            self._pipeline = DevicePipeline(
                cfg, focal_px=self.source.focal_px, has_aria_hands=self.source.has_hand_tracking
            )

    def _maybe(self, label: str):
        try:
            sid = self.provider.get_stream_id_from_label(label)
            if sid is None or self.provider.get_num_data(sid) == 0:
                return None
            return sid
        except Exception:
            return None

    def _meter(self, key: str) -> RateMeter:
        return self._meters.setdefault(key, RateMeter())

    def __iter__(self) -> Iterator[AriaSnapshot]:
        from .viz import draw_frame

        for frame in self.source:
            self._meter("camera-rgb").tick()

            detections, hands, interactions = [], list(frame.hands), []
            detect_ms = 0.0
            if self._pipeline is not None:
                result = self._pipeline.process(frame)
                detections = result.detections
                hands = result.hands
                interactions = result.interactions
                detect_ms = result.detect_ms

            main = draw_frame(
                frame.rgb, detections, self.cfg.viz,
                hands=hands, gaze=frame.gaze, interactions=interactions,
            )

            panels: list[SensorPanel] = []
            for label in SIDE_CAMERA_LABELS:
                sid = self._side_streams.get(label)
                image = None
                if sid is not None:
                    try:
                        data, _rec = self.provider.get_image_data_by_time_ns(
                            sid, frame.timestamp_ns
                        )
                        if data is not None:
                            image = data.to_numpy_array()
                            self._meter(label).tick()
                    except Exception:
                        image = None
                panels.append(
                    SensorPanel(name=label, image=image, rate_hz=self._meter(label).hz)
                )

            yield AriaSnapshot(
                timestamp_ns=frame.timestamp_ns,
                main=main,
                main_title=f"camera-rgb  (rectified{', ' + self.cfg.rectify.rotation if self.cfg.rectify.rotation != 'none' else ''})",
                panels=panels,
                stats=self._stats(frame, detections, hands, detect_ms),
            )

    def _stats(self, frame, detections, hands, detect_ms: float) -> list[tuple[str, str]]:
        stats: list[tuple[str, str]] = [
            ("time", f"{frame.timestamp_ns / 1e9:9.3f} s"),
            ("rgb", f"{self._meter('camera-rgb').hz:.1f} Hz"),
            ("detect", f"{detect_ms:.0f} ms"),
        ]

        if frame.gaze is not None:
            stats.append(
                ("gaze yaw/pitch", f"{math.degrees(frame.gaze.yaw_rad):+.1f} deg {math.degrees(frame.gaze.pitch_rad):+.1f} deg")
            )
            stats.append(
                ("gaze px", f"{frame.gaze.point_px[0]:.0f},{frame.gaze.point_px[1]:.0f}"
                 if frame.gaze.point_px else "off-frame")
            )
        else:
            stats.append(("gaze", "no stream"))

        if hands:
            stats.append(
                ("hands", ", ".join(f"{h.side} {h.score:.2f}" for h in hands))
            )
        else:
            stats.append(("hands", "none"))

        if detections:
            stats.append(
                ("devices", ", ".join(f"{d.label}({d.score:.2f})" for d in detections))
            )
        else:
            stats.append(("devices", "none"))

        vio = self._vio_data(frame.timestamp_ns)
        if vio is not None:
            stats.append(("vio t", vio))
        return stats

    def _vio_data(self, timestamp_ns: int) -> str | None:
        if self._vio_stream is None:
            return None
        try:
            vio = self.provider.get_vio_data_by_time_ns(self._vio_stream, timestamp_ns)
            if vio is None:
                return None
            transform = getattr(vio, "transform_odometry_bodyimu", None)
            if transform is None:
                return None
            return _fmt_vec(transform.translation(), "m")
        except Exception:
            return None

    def close(self) -> None:
        if self._pipeline is not None:
            self._pipeline.close()
        self.source.close()


class LiveMonitor:
    """Same dashboard, fed by the live stream.

    Every sensor arrives on its own callback at its own rate, so each one
    latches its newest sample and the dashboard renders whatever is current.
    Nothing blocks an SDK callback thread.
    """

    def __init__(self, cfg: PipelineConfig, serial: str | None = None,
                 profile: str = "mp_streaming_demo", interface: str = "usb",
                 port: int = 6768, detect: bool = True,
                 max_frames: int = 0, record_to_vrs: str | None = None,
                 batch_period_ms: int = 0) -> None:
        from .sources.live import LiveFrameSource

        self.cfg = cfg
        self._meters: dict[str, RateMeter] = {}
        self._images: dict[str, np.ndarray] = {}
        self._lock = threading.Lock()

        self.source = LiveFrameSource(
            serial=serial, profile=profile, interface=interface,
            port=port, rectify=cfg.rectify, max_frames=max_frames,
            record_to_vrs=record_to_vrs, batch_period_ms=batch_period_ms,
        )
        # Hook the extra camera streams the detection path does not need.
        try:
            self.source.receiver.register_slam_callback(self._on_slam)
        except Exception as exc:  # pragma: no cover - SDK dependent
            log.warning("could not register SLAM callback: %s", exc)
        try:
            self.source.receiver.register_et_callback(self._on_et)
        except Exception as exc:  # pragma: no cover
            log.warning("could not register ET callback: %s", exc)

        self._pipeline = None
        if detect:
            from .pipeline import DevicePipeline

            self._pipeline = DevicePipeline(cfg, focal_px=self.source.focal_px, has_aria_hands=True)

    def _meter(self, key: str) -> RateMeter:
        return self._meters.setdefault(key, RateMeter())

    def _stash(self, label: str, image) -> None:
        try:
            array = image.to_numpy_array() if hasattr(image, "to_numpy_array") else np.asarray(image)
        except Exception:
            return
        with self._lock:
            self._images[label] = array
        self._meter(label).tick()

    #: The live SLAM callback reports a numeric ``camera_id`` rather than a
    #: label. These are the ids observed from a Gen 2 device over USB; the
    #: name mapping follows the documented stream order (1201-1..4). An id not
    #: in this table is shown as-is rather than guessed at.
    SLAM_ID_NAMES = {
        1: "slam-front-left",
        2: "slam-front-right",
        4: "slam-side-left",
        8: "slam-side-right",
    }

    def _on_slam(self, image_data, image_record) -> None:
        cam_id = getattr(image_record, "camera_id", None)
        try:
            name = self.SLAM_ID_NAMES.get(int(cam_id), f"slam-{cam_id}")
        except (TypeError, ValueError):
            name = "slam"
        self._stash(name, image_data)

    def _on_et(self, image_data, image_record) -> None:
        self._stash("camera-et", image_data)

    def __iter__(self) -> Iterator[AriaSnapshot]:
        from .viz import draw_frame

        for frame in self.source:
            self._meter("camera-rgb").tick()

            detections, hands, interactions = [], list(frame.hands), []
            detect_ms = 0.0
            if self._pipeline is not None:
                result = self._pipeline.process(frame)
                detections, hands = result.detections, result.hands
                interactions = result.interactions
                detect_ms = result.detect_ms

            main = draw_frame(
                frame.rgb, detections, self.cfg.viz,
                hands=hands, gaze=frame.gaze, interactions=interactions,
            )

            with self._lock:
                images = dict(self._images)
            panels = [
                SensorPanel(name=k, image=v, rate_hz=self._meter(k).hz)
                for k, v in sorted(images.items())
            ]
            # Show the expected streams even before (or if) they arrive, so a
            # stream that a profile does not carry reads as "no data" rather
            # than silently vanishing from the layout.
            for expected in ("camera-et",):
                if expected not in images:
                    panels.append(SensorPanel(name=expected, image=None))
            if not panels:
                panels = [SensorPanel(name="slam / et", image=None)]

            stats = [
                ("rgb", f"{self._meter('camera-rgb').hz:.1f} Hz"),
                ("detect", f"{detect_ms:.0f} ms"),
                ("devices", ", ".join(f"{d.label}" for d in detections) or "none"),
                ("hands", ", ".join(f"{h.side} {h.score:.2f}" for h in hands) or "none"),
            ]
            if frame.gaze is not None:
                stats.append(
                    ("gaze", f"{math.degrees(frame.gaze.yaw_rad):+.1f} deg {math.degrees(frame.gaze.pitch_rad):+.1f} deg")
                )

            yield AriaSnapshot(
                timestamp_ns=frame.timestamp_ns, main=main,
                main_title="camera-rgb (live)", panels=panels, stats=stats,
            )

    def close(self) -> None:
        if self._pipeline is not None:
            self._pipeline.close()
        self.source.close()


def run_monitor(
    monitor,
    window: str = "Aria Gen 2 — what the glasses see",
    width: int = 1600,
    height: int = 900,
    display: bool = True,
    mp4_path: str | None = None,
    fps: float = 30.0,
) -> int:
    """Render a monitor's snapshots to a window and/or an MP4."""
    from .viewer import render_dashboard
    from .viz import Mp4Writer

    writer = Mp4Writer(mp4_path, fps=fps) if mp4_path else None
    if display:
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window, width, height)

    count = 0
    try:
        for snapshot in monitor:
            canvas = render_dashboard(snapshot, width=width, height=height)
            if writer is not None:
                writer.write(canvas)
            if display:
                cv2.imshow(window, canvas)
                if (cv2.waitKey(1) & 0xFF) in (ord("q"), 27):
                    break
            count += 1
    except KeyboardInterrupt:
        log.info("interrupted")
    finally:
        if writer is not None:
            writer.close()
        if display:
            cv2.destroyAllWindows()
        monitor.close()
    log.info("rendered %d dashboard frames", count)
    return count
