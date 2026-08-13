"""Offline driver: pull frames from a source, run the pipeline, write outputs."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Sequence

from .config import PipelineConfig
from .pipeline import DevicePipeline, JsonlWriter
from .sources.base import FrameSource
from .viz import Mp4Writer, draw_frame

log = logging.getLogger(__name__)


def run_offline(
    cfg: PipelineConfig,
    source: FrameSource,
    mp4_path: str | None = None,
    jsonl_path: str | None = None,
    display: bool = False,
    hooks: Sequence[Callable] = (),
) -> int:
    """Process every frame from ``source``. Returns the frame count."""
    has_aria_hands = bool(getattr(source, "has_hand_tracking", False))
    pipeline = DevicePipeline(
        cfg, focal_px=source.focal_px, has_aria_hands=has_aria_hands
    )
    for hook in hooks:
        pipeline.add_result_hook(hook)

    writer = Mp4Writer(mp4_path, fps=source.fps) if mp4_path else None
    jsonl = JsonlWriter(jsonl_path) if jsonl_path else None

    cv2 = None
    if display:
        import cv2 as _cv2

        cv2 = _cv2
        cv2.namedWindow("aria-devices", cv2.WINDOW_NORMAL)

    count = 0
    t0 = time.perf_counter()
    try:
        for frame in source:
            result = pipeline.process(frame)

            if writer is not None or display:
                canvas = draw_frame(
                    frame.rgb,
                    result.detections,
                    cfg.viz,
                    hands=result.hands,
                    gaze=frame.gaze,
                    hud=[f"frame {frame.frame_idx}  detect {result.detect_ms:.0f} ms"],
                    interactions=result.interactions,
                )
                if writer is not None:
                    writer.write(canvas)
                if display and cv2 is not None:
                    cv2.imshow("aria-devices", canvas)
                    if (cv2.waitKey(1) & 0xFF) in (ord("q"), 27):
                        break

            if jsonl is not None:
                jsonl.write(result)

            count += 1
            if count % 50 == 0:
                elapsed = time.perf_counter() - t0
                log.info("%d frames in %.1fs (%.1f fps)", count, elapsed, count / elapsed)
    finally:
        if writer is not None:
            writer.close()
        if jsonl is not None:
            jsonl.close()
        if display and cv2 is not None:
            cv2.destroyAllWindows()
        pipeline.close()
        source.close()

    elapsed = time.perf_counter() - t0
    log.info("done: %d frames in %.1fs (%.1f fps)", count, elapsed, count / max(elapsed, 1e-6))
    if mp4_path:
        log.info("wrote %s", mp4_path)
    if jsonl_path:
        log.info("wrote %s", jsonl_path)
    return count
