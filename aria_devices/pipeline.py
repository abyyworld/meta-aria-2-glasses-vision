"""The pipeline: frame in, labelled tracked detections out.

Deliberately source-agnostic — it takes ``Frame`` objects and does not care
whether they came from a VRS file, the glasses, an MP4 or a webcam.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path

from .config import HAND, PipelineConfig
from .detect import build_detector
from .detect.base import Detection, Detector
from .detect.disambiguate import WorldFixedTracker, score_detections
from .frames import Frame, HandSample
from .gaze import GazeAttributor
from .hands import (
    HandDepthEstimator,
    MediaPipeHandDetector,
    OpenVocabHandDetector,
    build_hand_detector,
)
from .interaction import (
    HandInteraction,
    InteractionEvent,
    InteractionTracker,
    analyse_interactions,
)
from .track import ByteTracker, DevicePersistence

log = logging.getLogger(__name__)


@dataclass
class FrameResult:
    """Everything produced for one frame."""

    frame_idx: int
    timestamp_ns: int
    detections: list[Detection] = field(default_factory=list)
    hands: list[HandSample] = field(default_factory=list)
    interactions: list[HandInteraction] = field(default_factory=list)
    events: list[InteractionEvent] = field(default_factory=list)
    gaze_point: tuple[float, float] | None = None
    detect_ms: float = 0.0

    def to_record(self) -> dict:
        """One JSONL record, as specified."""
        return {
            "frame_idx": self.frame_idx,
            "timestamp_ns": self.timestamp_ns,
            "detections": [
                {
                    "track_id": d.track_id,
                    "label": d.label,
                    "score": round(float(d.score), 4),
                    "bbox_xyxy": [round(float(v), 2) for v in d.bbox_xyxy],
                    "gazed_at": bool(d.gazed_at),
                    "gaze_dwell_ms": round(float(d.gaze_dwell_ms), 1),
                    "signals": d.signals,
                }
                for d in self.detections
            ],
            "hands": [
                {
                    "bbox_xyxy": [round(float(v), 2) for v in h.bbox_xyxy],
                    "side": h.side,
                    "score": round(float(h.score), 4),
                    "source": h.source,
                }
                for h in self.hands
            ],
            "interactions": [i.to_record() for i in self.interactions],
            "events": [e.to_record() for e in self.events],
            "gaze_point_px": (
                [round(v, 2) for v in self.gaze_point] if self.gaze_point else None
            ),
            "detect_ms": round(self.detect_ms, 2),
        }


class DevicePipeline:
    """Detect -> disambiguate -> track -> attribute gaze.

    One instance is stateful (tracker, dwell timers, world-fixed history), so
    use one per video and call ``reset`` between recordings.
    """

    def __init__(
        self,
        cfg: PipelineConfig,
        detector: Detector | None = None,
        focal_px: float | None = None,
        has_aria_hands: bool = False,
    ) -> None:
        self.cfg = cfg
        self.detector = detector if detector is not None else build_detector(cfg.detector)
        self.focal_px = focal_px

        self.tracker = ByteTracker(cfg.track) if cfg.track.enabled else None
        self.gaze_attributor = GazeAttributor(cfg.gaze_hit_radius_px)
        self.world_fixed = WorldFixedTracker(cfg.disambiguation)
        self.persistence = DevicePersistence(cfg.track.persist_frames)

        self._hand_backend = build_hand_detector(cfg.hands, has_aria_stream=has_aria_hands)
        self._uses_aria_hands = self._hand_backend == "aria"
        self.hand_depth = (
            HandDepthEstimator(focal_px) if focal_px and cfg.disambiguation.enable_size_prior else None
        )
        self._hooks: list[Callable[[FrameResult], None]] = []
        self._profiles = cfg.disambiguation.device_profiles
        self.interaction_tracker = InteractionTracker(
            profiles=self._profiles,
            require_gaze=cfg.require_gaze,
            gaze_grace_ms=cfg.gaze_grace_ms,
        )

    # -- integration -------------------------------------------------------
    def add_result_hook(self, hook: Callable[[FrameResult], None]) -> None:
        """Register a callback fired with every FrameResult.

        This is the integration point for embedding the detector in another
        system: push results onto a bus, a socket, a ROS topic, whatever. Hooks
        run synchronously on the processing thread, so keep them cheap — a slow
        hook throttles the pipeline. Exceptions raised by a hook are logged and
        swallowed rather than killing the run.
        """
        self._hooks.append(hook)

    def remove_result_hook(self, hook: Callable[[FrameResult], None]) -> None:
        if hook in self._hooks:
            self._hooks.remove(hook)

    # -- lifecycle ---------------------------------------------------------
    def reset(self) -> None:
        if self.tracker:
            self.tracker.reset()
        self.gaze_attributor = GazeAttributor(self.cfg.gaze_hit_radius_px)
        self.world_fixed = WorldFixedTracker(self.cfg.disambiguation)
        self.persistence.reset()
        self.interaction_tracker.reset()

    def close(self) -> None:
        if isinstance(self._hand_backend, MediaPipeHandDetector):
            self._hand_backend.close()
        self.detector.close()

    # -- per frame ---------------------------------------------------------
    def _is_world_fixed_distractor(self, det) -> bool:
        """Only suppress a stationary object if it is monitor-sized.

        Staying still is not evidence of being a distractor when the study is
        three devices parked on a desk. Physical size is the signal that
        actually separates a wall monitor from a laptop, so a measured
        device-sized diagonal vetoes the suppression outright.
        """
        cfg = self.cfg.disambiguation
        if not cfg.suppress_world_fixed or det.track_id is None:
            return False
        if not self.world_fixed.is_world_fixed(det.track_id, det.bbox_xyxy):
            return False
        diag = float(det.signals.get("diag_cm", 0.0) or 0.0)
        if 0.0 < diag < cfg.monitor_min_diag_cm:
            return False  # a real device, merely sitting still
        return True

    def process(self, frame: Frame) -> FrameResult:
        t0 = time.perf_counter()
        raw = self.detector.detect(frame.rgb)
        detect_ms = (time.perf_counter() - t0) * 1000.0

        hands = self._resolve_hands(frame, raw)

        # Depth from hand scale — the only depth we have off-Aria, and enough
        # to separate a phone from a tablet when either is being held.
        depth_fn = None
        if self.hand_depth is not None:
            self.hand_depth.update(hands)
            depth_fn = lambda det: self.hand_depth.depth_for_box(det.bbox_xyxy)  # noqa: E731

        scored = score_detections(
            raw,
            self.cfg.disambiguation,
            focal_px=self.focal_px,
            depth_fn=depth_fn,
            image_rgb=frame.rgb,
        )
        # Hands are handled by their own backend; drop any open-vocab hand
        # boxes so they are not double-counted as detections.
        scored = [d for d in scored if d.label != HAND]

        if self.tracker is not None:
            scored = self.tracker.update(scored)
            for det in scored:
                if det.track_id is not None:
                    self.world_fixed.update(det.track_id, det.bbox_xyxy)
            scored = [d for d in scored if not self._is_world_fixed_distractor(d)]
            scored = self.persistence.update(scored, frame.frame_idx)

        self.gaze_attributor.attribute(scored, frame.gaze, frame.timestamp_ns)

        # Gaze gating only makes sense where there is a gaze stream. A webcam,
        # an MP4 or a still photo has none, and requiring it there would mean
        # no device is ever attended and the pipeline emits nothing at all.
        self.interaction_tracker.require_gaze = (
            self.cfg.require_gaze and frame.gaze is not None
        )

        result = FrameResult(
            frame_idx=frame.frame_idx,
            timestamp_ns=frame.timestamp_ns,
            detections=scored,
            hands=hands,
            interactions=analyse_interactions(hands, scored, self._profiles),
            events=self.interaction_tracker.update(hands, scored, frame.timestamp_ns),
            gaze_point=frame.gaze.point_px if frame.gaze else None,
            detect_ms=detect_ms,
        )
        for hook in self._hooks:
            try:
                hook(result)
            except Exception as exc:  # a bad consumer must not kill the pipeline
                log.warning("result hook %r raised: %s", getattr(hook, "__name__", hook), exc)
        return result

    def _resolve_hands(self, frame: Frame, raw: list[Detection]) -> list[HandSample]:
        if self._uses_aria_hands:
            return list(frame.hands)  # already attached by the Aria source
        backend = self._hand_backend
        if backend is None:
            return []
        if isinstance(backend, MediaPipeHandDetector):
            try:
                return backend.detect(frame.rgb, timestamp_ms=int(frame.timestamp_ns // 1_000_000))
            except Exception as exc:  # pragma: no cover - runtime robustness
                log.warning("mediapipe hand detection failed: %s", exc)
                return []
        if isinstance(backend, OpenVocabHandDetector):
            return backend.detect_from_detections(raw)
        return []

    # -- batch -------------------------------------------------------------
    def run(self, frames: Iterable[Frame]) -> Iterator[FrameResult]:
        for frame in frames:
            yield self.process(frame)


class JsonlWriter:
    """One JSON record per frame, flushed as it goes so a crash keeps data."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.path.open("w", encoding="utf-8")
        self.records_written = 0

    def write(self, result: FrameResult) -> None:
        self._fh.write(json.dumps(result.to_record()) + "\n")
        self.records_written += 1

    def close(self) -> None:
        if not self._fh.closed:
            self._fh.close()

    def __enter__(self) -> "JsonlWriter":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
