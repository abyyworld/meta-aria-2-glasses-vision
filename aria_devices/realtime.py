"""Realtime camera app — the hands-on test harness.

The point of this module is responsiveness. A naive loop that captures, detects
and draws in sequence runs the whole preview at detector speed, which on CPU is
a few frames a second and feels broken when you wave a phone at it.

So three stages run concurrently:

  capture thread   grabs frames as fast as the camera allows, keeping only the
                   newest (an old frame is worthless in a live preview)
  detect thread    pulls the newest frame and runs the pipeline; it is allowed
                   to be slow and to skip frames
  main thread      draws every captured frame immediately, overlaying the most
                   recent detection result

Result: the video stays at camera framerate and stays smooth while your hand
moves, and the boxes update as fast as the detector can manage. The tracker's
constant-velocity prediction hides most of the lag between the two.

cv2.imshow must be called from the main thread on macOS, which is why
rendering is not itself a worker.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import cv2

from .config import PipelineConfig
from .frames import Frame
from .pipeline import DevicePipeline, FrameResult, JsonlWriter
from .sources.video import VideoFrameSource
from .viz import Mp4Writer, draw_frame

log = logging.getLogger(__name__)


@dataclass
class _Slot:
    """A one-item mailbox: writers overwrite, readers take the newest."""

    _value: object = None
    _lock: threading.Lock = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        self._lock = threading.Lock()

    def put(self, value: object) -> None:
        with self._lock:
            self._value = value

    def get(self) -> object:
        with self._lock:
            return self._value


class FpsMeter:
    """Rolling FPS over a short window."""

    def __init__(self, window: int = 30) -> None:
        self._times: deque[float] = deque(maxlen=window)

    def tick(self) -> None:
        self._times.append(time.perf_counter())

    @property
    def fps(self) -> float:
        if len(self._times) < 2:
            return 0.0
        span = self._times[-1] - self._times[0]
        return (len(self._times) - 1) / span if span > 0 else 0.0


def run_camera(
    cfg: PipelineConfig,
    camera_index: int = 0,
    window_name: str = "aria-devices",
    width: int = 1280,
    height: int = 720,
    display: bool = True,
    mp4_path: str | None = None,
    jsonl_path: str | None = None,
    max_frames: int = 0,
    detect_every_ms: float = 0.0,
    hooks: Sequence[Callable] = (),
) -> int:
    """Run the live camera preview. Returns the number of frames displayed.

    ``detect_every_ms`` throttles the detector (0 = as fast as it can go). Worth
    raising if you want the laptop fan to stay quiet.
    """
    source = VideoFrameSource(
        camera_index, request_size=(width, height), request_fps=60.0
    )
    log.info("camera %dx%d @ %.0f fps", source.width, source.height, source.fps)

    pipeline = DevicePipeline(cfg, focal_px=source.focal_px)
    for hook in hooks:
        pipeline.add_result_hook(hook)
    pipeline.detector.warmup()

    frame_slot = _Slot()
    result_slot = _Slot()
    stop = threading.Event()

    capture_fps = FpsMeter()
    detect_fps = FpsMeter()
    display_fps = FpsMeter()

    def capture_loop() -> None:
        try:
            for frame in source:
                if stop.is_set():
                    break
                frame_slot.put(frame)
                capture_fps.tick()
        except Exception as exc:  # pragma: no cover - hardware dependent
            log.error("capture thread died: %s", exc)
        finally:
            stop.set()

    def detect_loop() -> None:
        last_idx = -1
        next_allowed = 0.0
        while not stop.is_set():
            frame = frame_slot.get()
            if frame is None or not isinstance(frame, Frame) or frame.frame_idx == last_idx:
                time.sleep(0.002)
                continue
            now = time.perf_counter()
            if now < next_allowed:
                time.sleep(0.002)
                continue
            last_idx = frame.frame_idx
            try:
                result = pipeline.process(frame)
            except Exception as exc:  # pragma: no cover - runtime robustness
                log.warning("detection failed on frame %d: %s", frame.frame_idx, exc)
                time.sleep(0.01)
                continue
            result_slot.put(result)
            detect_fps.tick()
            if detect_every_ms > 0:
                next_allowed = time.perf_counter() + detect_every_ms / 1000.0

    capture_thread = threading.Thread(target=capture_loop, name="capture", daemon=True)
    detect_thread = threading.Thread(target=detect_loop, name="detect", daemon=True)
    capture_thread.start()
    detect_thread.start()

    writer = Mp4Writer(mp4_path, fps=source.fps) if mp4_path else None
    jsonl = JsonlWriter(jsonl_path) if jsonl_path else None
    shown = 0
    last_logged_idx = -1

    if display:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, source.width, source.height)

    try:
        while not stop.is_set():
            frame = frame_slot.get()
            if not isinstance(frame, Frame):
                time.sleep(0.005)
                continue

            result = result_slot.get()
            detections = result.detections if isinstance(result, FrameResult) else []
            hands = result.hands if isinstance(result, FrameResult) else []
            interactions = result.interactions if isinstance(result, FrameResult) else []
            detect_ms = result.detect_ms if isinstance(result, FrameResult) else 0.0

            hud = [
                f"display {display_fps.fps:4.1f} fps | capture {capture_fps.fps:4.1f} fps",
                f"detect  {detect_fps.fps:4.1f} fps ({detect_ms:.0f} ms/frame)",
                f"backend {cfg.detector.backend} on {pipeline.detector.device}"
                if hasattr(pipeline.detector, "device")
                else f"backend {cfg.detector.backend}",
                "q or ESC to quit  |  s to save a still",
            ]
            canvas = draw_frame(
                frame.rgb, detections, cfg.viz, hands=hands, gaze=frame.gaze, hud=hud,
                interactions=interactions,
            )

            if writer is not None:
                writer.write(canvas)
            if jsonl is not None and isinstance(result, FrameResult):
                if result.frame_idx != last_logged_idx:
                    jsonl.write(result)
                    last_logged_idx = result.frame_idx

            if display:
                cv2.imshow(window_name, canvas)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                if key == ord("s"):
                    path = f"out/still_{int(time.time())}.png"
                    cv2.imwrite(path, canvas)
                    log.info("saved %s", path)

            display_fps.tick()
            shown += 1
            if max_frames and shown >= max_frames:
                break
            if not display:
                time.sleep(0.001)
    except KeyboardInterrupt:
        log.info("interrupted")
    finally:
        stop.set()
        capture_thread.join(timeout=1.0)
        detect_thread.join(timeout=2.0)
        if writer is not None:
            writer.close()
        if jsonl is not None:
            jsonl.close()
        if display:
            cv2.destroyAllWindows()
        pipeline.close()
        source.close()

    log.info(
        "displayed %d frames | display %.1f fps, detect %.1f fps",
        shown, display_fps.fps, detect_fps.fps,
    )
    return shown
