"""Hand detection.

Three backends behind one interface, in descending order of trustworthiness:

``aria``
    Aria Gen 2's on-device hand tracking stream (``handtracking``, 371-*). This
    is the right answer whenever it exists — it is computed on the glasses from
    the SLAM cameras, costs us nothing, and gives metric 3D joints.
``mediapipe``
    For webcam / MP4 input, where there is no Aria. MediaPipe 1.0 removed the
    old ``mp.solutions`` API entirely, so this uses the Tasks HandLandmarker.
``openvocab``
    Last resort: the "human hand" prompt already in the detector's prompt set.
    Box only, no joints, and noticeably worse.

MediaPipe also gives *metric* world landmarks, which we turn into a monocular
depth estimate (see ``HandDepthEstimator``). That is what makes the physical
size prior usable off-Aria: a hand is a known-size ruler that happens to be
holding the device we are trying to measure.
"""

from __future__ import annotations

import logging
import math
import os
import urllib.request
from pathlib import Path

import numpy as np

from .config import HandConfig
from .frames import HandSample

log = logging.getLogger(__name__)

MEDIAPIPE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)

# Landmark indices in MediaPipe's 21-point hand model.
WRIST = 0
MIDDLE_MCP = 9

#: Skeleton edges for drawing.
HAND_CONNECTIONS: tuple[tuple[int, int], ...] = (
    (0, 1), (1, 2), (2, 3), (3, 4),          # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),          # index
    (5, 9), (9, 10), (10, 11), (11, 12),     # middle
    (9, 13), (13, 14), (14, 15), (15, 16),   # ring
    (13, 17), (17, 18), (18, 19), (19, 20),  # pinky
    (0, 17),                                 # palm base
)


def ensure_mediapipe_model(path: str | Path) -> Path:
    """Download the hand landmarker model on first use."""
    p = Path(path)
    if p.exists() and p.stat().st_size > 0:
        return p
    p.parent.mkdir(parents=True, exist_ok=True)
    log.info("downloading hand landmarker model -> %s", p)
    tmp = p.with_suffix(p.suffix + ".part")
    urllib.request.urlretrieve(MEDIAPIPE_MODEL_URL, tmp)  # noqa: S310 - fixed vendor URL
    os.replace(tmp, p)
    return p


class MediaPipeHandDetector:
    """MediaPipe Tasks HandLandmarker in VIDEO mode.

    VIDEO mode (rather than IMAGE) keeps MediaPipe's internal tracking alive
    between frames, which is both faster and much steadier than re-detecting
    from scratch every frame.
    """

    def __init__(self, cfg: HandConfig) -> None:
        try:
            import mediapipe as mp
            from mediapipe.tasks.python import BaseOptions
            from mediapipe.tasks.python import vision
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise ImportError(
                "mediapipe is required for hand detection: pip install 'aria-devices[detect]'"
            ) from exc

        self._mp = mp
        self.cfg = cfg
        model_path = ensure_mediapipe_model(cfg.mediapipe_model_path)

        options = vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=cfg.max_hands,
            min_hand_detection_confidence=cfg.min_confidence,
            min_hand_presence_confidence=cfg.min_confidence,
            min_tracking_confidence=cfg.min_confidence,
        )
        self._landmarker = vision.HandLandmarker.create_from_options(options)
        self._last_ts_ms = -1

    def detect(self, image_rgb: np.ndarray, timestamp_ms: int) -> list[HandSample]:
        h, w = image_rgb.shape[:2]
        # MediaPipe requires strictly increasing timestamps in VIDEO mode and
        # throws otherwise, which is easy to hit on a dropped/duplicated frame.
        if timestamp_ms <= self._last_ts_ms:
            timestamp_ms = self._last_ts_ms + 1
        self._last_ts_ms = timestamp_ms

        mp_image = self._mp.Image(
            image_format=self._mp.ImageFormat.SRGB, data=np.ascontiguousarray(image_rgb)
        )
        result = self._landmarker.detect_for_video(mp_image, timestamp_ms)

        hands: list[HandSample] = []
        for i, landmarks in enumerate(result.hand_landmarks):
            pts = np.array([[lm.x * w, lm.y * h] for lm in landmarks], dtype=np.float32)
            x1, y1 = pts.min(axis=0)
            x2, y2 = pts.max(axis=0)
            # Landmarks hug the skeleton; pad out to something box-like.
            pad = 0.12 * max(x2 - x1, y2 - y1)
            side, score = "unknown", 1.0
            if i < len(result.handedness) and result.handedness[i]:
                cat = result.handedness[i][0]
                side = str(cat.category_name).lower()
                score = float(cat.score)

            world = None
            if i < len(result.hand_world_landmarks):
                world = np.array(
                    [[lm.x, lm.y, lm.z] for lm in result.hand_world_landmarks[i]], dtype=np.float32
                )

            sample = HandSample(
                bbox_xyxy=(
                    float(max(0.0, x1 - pad)),
                    float(max(0.0, y1 - pad)),
                    float(min(w, x2 + pad)),
                    float(min(h, y2 + pad)),
                ),
                side=side,
                score=score,
                landmarks_px=pts,
                source="mediapipe",
            )
            # Stashed for HandDepthEstimator; not part of the public dataclass
            # because only this backend produces it.
            sample_world = world
            setattr(sample, "_world_landmarks", sample_world)
            hands.append(sample)
        return hands

    def close(self) -> None:
        try:
            self._landmarker.close()
        except Exception:  # pragma: no cover
            pass


class OpenVocabHandDetector:
    """Fallback that reuses the open-vocab detector's "hand" prompt.

    Produces boxes only. Kept so the pipeline still labels hands when MediaPipe
    is not installed, rather than silently dropping the requirement.
    """

    def detect_from_detections(self, detections: list) -> list[HandSample]:  # noqa: ANN001
        from .config import HAND
        from .detect.disambiguate import collapse_raw_label

        out: list[HandSample] = []
        for det in detections:
            if collapse_raw_label(det.raw_label) == HAND:
                out.append(
                    HandSample(
                        bbox_xyxy=det.bbox_xyxy, score=det.score, source="openvocab"
                    )
                )
        return out


class HandDepthEstimator:
    """Monocular depth from hand size, used to feed the physical size prior.

    MediaPipe's world landmarks are metric and centred on the hand, so the
    distance between two joints is a real length in metres. Comparing it to the
    same distance in pixels gives ``z = f * L_metres / L_pixels``.

    Accuracy is roughly +/-15% on the hand itself. That is easily enough to
    separate an 18 cm phone from a 31 cm iPad, which is the distinction the
    shape prior cannot make. It says nothing about objects far from the hand,
    so ``depth_for_box`` only answers for boxes near a hand.
    """

    #: Boxes within this multiple of the hand's diagonal are assumed to be at
    #: roughly the hand's depth (i.e. being held, or resting under it).
    NEAR_HAND_FACTOR = 2.5

    def __init__(self, focal_px: float) -> None:
        self.focal_px = float(focal_px)
        self._hands: list[tuple[tuple[float, float, float, float], float]] = []

    def update(self, hands: list[HandSample]) -> None:
        self._hands = []
        for hand in hands:
            # Aria's on-device tracker already gives metric 3D joints, so its
            # depth is measured rather than inferred. Prefer it outright.
            depth = getattr(hand, "_depth_m", None)
            if depth is None:
                depth = self._hand_depth(hand)
            if depth is not None and 0.05 < float(depth) < 10.0:
                self._hands.append((hand.bbox_xyxy, float(depth)))

    def _hand_depth(self, hand: HandSample) -> float | None:
        world = getattr(hand, "_world_landmarks", None)
        pts = hand.landmarks_px
        if world is None or pts is None or len(pts) <= MIDDLE_MCP:
            return None
        # Wrist -> middle-finger MCP is the most rigid span on the hand, so it
        # is the least sensitive to finger articulation.
        metres = float(np.linalg.norm(world[MIDDLE_MCP] - world[WRIST]))
        pixels = float(np.linalg.norm(pts[MIDDLE_MCP] - pts[WRIST]))
        if metres <= 1e-4 or pixels <= 1e-3:
            return None
        depth = self.focal_px * metres / pixels
        if not (0.05 < depth < 10.0):
            return None
        return depth

    def depth_for_box(self, box: tuple[float, float, float, float]) -> float | None:
        """Depth in metres for a box, if a hand is close enough to vouch for it."""
        if not self._hands:
            return None
        cx, cy = 0.5 * (box[0] + box[2]), 0.5 * (box[1] + box[3])
        best: tuple[float, float] | None = None
        for hbox, depth in self._hands:
            hcx, hcy = 0.5 * (hbox[0] + hbox[2]), 0.5 * (hbox[1] + hbox[3])
            hdiag = math.hypot(hbox[2] - hbox[0], hbox[3] - hbox[1])
            if hdiag <= 0:
                continue
            dist = math.hypot(cx - hcx, cy - hcy)
            if dist <= self.NEAR_HAND_FACTOR * hdiag and (best is None or dist < best[0]):
                best = (dist, depth)
        return best[1] if best else None


def build_hand_detector(cfg: HandConfig, has_aria_stream: bool = False):
    """Pick a hand backend, honouring "auto"."""
    backend = cfg.backend.lower()
    if backend == "off":
        return None
    if backend == "auto":
        backend = "aria" if has_aria_stream else "mediapipe"
    if backend == "aria":
        return "aria"  # handled inside the VRS/live source
    if backend == "mediapipe":
        try:
            return MediaPipeHandDetector(cfg)
        except Exception as exc:
            log.warning("mediapipe unavailable (%s); falling back to open-vocab hands", exc)
            return OpenVocabHandDetector()
    if backend == "openvocab":
        return OpenVocabHandDetector()
    raise ValueError(f"unknown hand backend {cfg.backend!r}")
