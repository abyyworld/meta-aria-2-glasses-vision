"""Hand-device interaction: grab/open pose, and where the hand is on a device.

Carried over from the Aria v1 study (``_Omni_Conect_Aria_Ver1-main``), which
used a ``relative_position`` on each device box to pick out which region of a
screen the hand was over, and a hand-pose enum to gate it. Only GRAB and OPEN
are kept here — the v1 MID state is dropped.

Aria and MediaPipe number their 21 landmarks differently, so every function
takes the landmark source and looks the indices up rather than assuming.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum

import numpy as np

from .detect.base import Detection
from .frames import HandSample

log = logging.getLogger(__name__)


class HandPose(Enum):
    """Only the two states the study needs."""

    OPEN = "open"
    GRAB = "grab"
    UNKNOWN = "unknown"


# Landmark index maps. MediaPipe's is the familiar 0=wrist, 4/8/12/16/20=tips
# layout; Aria's is its own ordering (fingertips first, wrist at 5).
_MEDIAPIPE_IDX = {
    "wrist": 0,
    "ref": 9,  # middle-finger MCP: the most rigid span from the wrist
    "tips": (4, 8, 12, 16, 20),
}
_ARIA_IDX = {
    "wrist": 5,
    "ref": 11,  # MIDDLE_PROXIMAL
    "tips": (0, 1, 2, 3, 4),
}


def landmark_indices(source: str) -> dict:
    return _ARIA_IDX if source == "aria" else _MEDIAPIPE_IDX


@dataclass
class HandInteraction:
    """One hand's pose, and the device it is on (if any)."""

    side: str
    pose: HandPose
    openness: float  # 0 = fist, 1 = fully splayed
    device_track_id: int | None = None
    device_label: str | None = None
    # Index into the detections list this was matched against. track_id is None
    # whenever tracking is off, so it cannot be used to identify which box was
    # matched — comparing None to None matches every device.
    device_index: int | None = None
    # Position within the device box, both in [0, 1]; None when not on a device.
    relative_xy: tuple[float, float] | None = None

    def to_record(self) -> dict:
        return {
            "side": self.side,
            "pose": self.pose.value,
            "openness": round(float(self.openness), 3),
            "device_track_id": self.device_track_id,
            "device_label": self.device_label,
            "relative_xy": (
                [round(float(v), 4) for v in self.relative_xy] if self.relative_xy else None
            ),
        }


#: Remap span for openness. Calibrated on a real open hand at an oblique angle
#: (measured reach 1.55), not on a splayed hand facing the camera (~2.2):
#: foreshortening is the normal case in egocentric video.
DEFAULT_OPENNESS_SPAN = 0.8


def hand_openness(hand: HandSample, span: float = DEFAULT_OPENNESS_SPAN) -> float:
    """How splayed the hand is, in [0, 1], scale-invariant.

    Mean fingertip distance from the wrist, divided by the wrist-to-knuckle
    span. Dividing by that span is what makes it work at any distance from the
    camera: both quantities shrink together, so the ratio does not.

    A closed fist puts the tips back near the knuckle line (reach ~1.0-1.2); an
    open hand reaches well beyond it. How far beyond depends heavily on viewing
    angle, which is what ``span`` absorbs.
    """
    pts = hand.landmarks_px
    if pts is None or len(pts) < 21:
        return float("nan")
    idx = landmark_indices(hand.source)
    wrist = np.asarray(pts[idx["wrist"]], dtype=np.float64)
    ref = np.asarray(pts[idx["ref"]], dtype=np.float64)
    scale = float(np.linalg.norm(ref - wrist))
    if scale <= 1e-3:
        return float("nan")

    tips = [np.asarray(pts[i], dtype=np.float64) for i in idx["tips"] if i < len(pts)]
    if not tips:
        return float("nan")
    mean_reach = float(np.mean([np.linalg.norm(t - wrist) for t in tips])) / scale

    if span <= 0:
        return 0.0
    return float(np.clip((mean_reach - 1.0) / span, 0.0, 1.0))


def classify_pose(
    hand: HandSample,
    grab_below: float = 0.35,
    open_above: float = 0.55,
    span: float = DEFAULT_OPENNESS_SPAN,
) -> tuple[HandPose, float]:
    """Classify a hand as GRAB or OPEN.

    The deliberate gap between the two thresholds is hysteresis-by-design: a
    hand mid-way through closing returns UNKNOWN rather than flickering between
    states, which matters because this feeds an interaction signal.
    """
    openness = hand_openness(hand, span)
    if openness != openness:  # NaN: no landmarks (open-vocab hand box only)
        return HandPose.UNKNOWN, 0.0
    if openness <= grab_below:
        return HandPose.GRAB, openness
    if openness >= open_above:
        return HandPose.OPEN, openness
    return HandPose.UNKNOWN, openness


def fingertip_centroid(hand: HandSample) -> tuple[float, float] | None:
    """Centre of the fingertips — the natural "where is this hand pointing"."""
    pts = hand.landmarks_px
    if pts is None or len(pts) < 21:
        if hand.bbox_xyxy:  # fall back to the box centre
            x1, y1, x2, y2 = hand.bbox_xyxy
            return (0.5 * (x1 + x2), 0.5 * (y1 + y2))
        return None
    idx = landmark_indices(hand.source)
    tips = [pts[i] for i in idx["tips"] if i < len(pts)]
    if not tips:
        return None
    arr = np.asarray(tips, dtype=np.float64)
    return (float(arr[:, 0].mean()), float(arr[:, 1].mean()))


def relative_position(
    box: tuple[float, float, float, float], x: float, y: float
) -> tuple[float, float] | None:
    """Where (x, y) falls inside a box, as fractions in [0, 1].

    Returns None when the point is outside. Same contract as the v1
    ``Device.relative_position``, minus the (-1, -1) sentinel.
    """
    x1, y1, x2, y2 = box
    if x < x1 or y < y1 or x > x2 or y > y2:
        return None
    w = x2 - x1
    h = y2 - y1
    if w <= 0 or h <= 0:
        return None
    return ((x - x1) / w, (y - y1) / h)


def screen_rect(
    box: tuple[float, float, float, float], profile
) -> tuple[float, float, float, float]:
    """Trim a detection box down to the device's screen rectangle.

    The detector boxes the whole physical object. A cursor has to land in
    *screen* coordinates, and for an open laptop the two are very different —
    the box includes the keyboard base, so an untrimmed mapping puts the cursor
    about a third of the way down from where the hand really is.
    """
    x1, y1, x2, y2 = box
    w = x2 - x1
    h = y2 - y1
    if w <= 0 or h <= 0:
        return box
    left, top, right, bottom = getattr(profile, "screen_inset", (0.0, 0.0, 0.0, 0.0))
    sx1 = x1 + left * w
    sy1 = y1 + top * h
    sx2 = x2 - right * w
    sy2 = y2 - bottom * h
    if sx2 <= sx1 or sy2 <= sy1:  # nonsense inset; fall back to the raw box
        return box
    return (sx1, sy1, sx2, sy2)


def normalised_distance(
    box: tuple[float, float, float, float], x: float, y: float
) -> float:
    """Distance from a point to a box, in units of the box's own diagonal.

    Scale-free on purpose: "a hand's width away" should mean the same thing
    whether the device is near or far, and dividing by the diagonal makes the
    approach threshold independent of how big the device looks.
    """
    dx = max(box[0] - x, 0.0, x - box[2])
    dy = max(box[1] - y, 0.0, y - box[3])
    if dx == 0.0 and dy == 0.0:
        return 0.0
    diag = float(np.hypot(box[2] - box[0], box[3] - box[1]))
    if diag <= 0:
        return float("inf")
    return float(np.hypot(dx, dy)) / diag


class InteractionState(Enum):
    """Transitions a hand makes with respect to one device."""

    APPROACH = "approach"  # near the device, not yet over it
    ENTER = "enter"        # first frame over the screen
    MOVE = "move"          # still over the screen
    LEAVE = "leave"        # first frame after moving off


@dataclass
class InteractionEvent:
    """What the downstream device actually consumes.

    ``x`` / ``y`` are normalised screen coordinates in [0, 1], origin top-left,
    ready to be multiplied by the target device's own resolution to place a
    cursor. They are None for APPROACH and LEAVE.
    """

    state: InteractionState
    device_label: str
    hand_side: str
    pose: HandPose
    timestamp_ns: int
    x: float | None = None
    y: float | None = None
    distance: float = 0.0  # in device diagonals; 0 when over the screen
    device_track_id: int | None = None

    def to_record(self) -> dict:
        return {
            "state": self.state.value,
            "device": self.device_label,
            "device_track_id": self.device_track_id,
            "hand": self.hand_side,
            "pose": self.pose.value,
            "x": round(self.x, 4) if self.x is not None else None,
            "y": round(self.y, 4) if self.y is not None else None,
            "distance": round(self.distance, 3),
            "timestamp_ns": self.timestamp_ns,
        }


class InteractionTracker:
    """Turns per-frame hand/device geometry into enter/move/leave events.

    An integration wants transitions, not a state dump every frame: a device
    should light up once on ENTER and clear once on LEAVE, rather than
    re-deriving that from a 30 Hz stream of booleans.

    Keyed by ``(hand_side, device_label)`` rather than track id, because the
    study has exactly one of each device class — so the class label *is* the
    routing key for which physical device gets the cursor. If a second device
    of the same class ever appears, switch this to the track id.
    """

    def __init__(self, approach_distance: float = 0.6, profiles: dict | None = None) -> None:
        self.approach_distance = approach_distance
        self.profiles = profiles or {}
        self._inside: set[tuple[str, str]] = set()

    def update(
        self,
        hands: Sequence[HandSample],
        detections: Sequence[Detection],
        timestamp_ns: int,
    ) -> list[InteractionEvent]:
        events: list[InteractionEvent] = []
        seen_inside: set[tuple[str, str]] = set()

        for hand in hands:
            point = fingertip_centroid(hand)
            if point is None:
                continue
            pose, _ = classify_pose(hand)

            for det in detections:
                profile = self.profiles.get(det.label)
                rect = screen_rect(det.bbox_xyxy, profile) if profile else det.bbox_xyxy
                key = (hand.side, det.label)
                rel = relative_position(rect, point[0], point[1])

                if rel is not None:
                    seen_inside.add(key)
                    state = (
                        InteractionState.MOVE
                        if key in self._inside
                        else InteractionState.ENTER
                    )
                    events.append(
                        InteractionEvent(
                            state=state,
                            device_label=det.label,
                            device_track_id=det.track_id,
                            hand_side=hand.side,
                            pose=pose,
                            timestamp_ns=timestamp_ns,
                            x=rel[0],
                            y=rel[1],
                            distance=0.0,
                        )
                    )
                    continue

                dist = normalised_distance(rect, point[0], point[1])
                if dist <= self.approach_distance and key not in self._inside:
                    events.append(
                        InteractionEvent(
                            state=InteractionState.APPROACH,
                            device_label=det.label,
                            device_track_id=det.track_id,
                            hand_side=hand.side,
                            pose=pose,
                            timestamp_ns=timestamp_ns,
                            distance=dist,
                        )
                    )

        # Anything that was inside last frame and is not now has left.
        for key in self._inside - seen_inside:
            hand_side, device_label = key
            events.append(
                InteractionEvent(
                    state=InteractionState.LEAVE,
                    device_label=device_label,
                    hand_side=hand_side,
                    pose=HandPose.UNKNOWN,
                    timestamp_ns=timestamp_ns,
                )
            )
        self._inside = seen_inside
        return events

    def reset(self) -> None:
        self._inside.clear()


def analyse_interactions(
    hands: Sequence[HandSample], detections: Sequence[Detection], profiles: dict | None = None
) -> list[HandInteraction]:
    """Pose-classify each hand and associate it with the device it is over.

    When a fingertip centroid falls inside several boxes the smallest wins, on
    the same logic as gaze attribution: a phone resting on a laptop should take
    priority over the laptop behind it.
    """
    out: list[HandInteraction] = []
    for hand in hands:
        pose, openness = classify_pose(hand)
        interaction = HandInteraction(side=hand.side, pose=pose, openness=openness)

        point = fingertip_centroid(hand)
        if point is not None:
            candidates = []
            for idx, det in enumerate(detections):
                profile = (profiles or {}).get(det.label)
                rect = screen_rect(det.bbox_xyxy, profile) if profile else det.bbox_xyxy
                rel = relative_position(rect, point[0], point[1])
                if rel is not None:
                    candidates.append((det.area, idx, det, rel))
            if candidates:
                _area, idx, det, rel = min(candidates, key=lambda c: c[0])
                interaction.device_track_id = det.track_id
                interaction.device_label = det.label
                interaction.device_index = idx
                interaction.relative_xy = rel
        out.append(interaction)
    return out
