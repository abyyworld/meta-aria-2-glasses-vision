"""ByteTrack-style multi-object tracking with temporal label voting.

Deliberately *not* Ultralytics' ``model.track(persist=True)``. That call is
bolted to the Ultralytics predictor, and the spec requires the detector backend
to be a config choice — OWLv2 would get no tracking at all. So this is a
self-contained implementation that takes plain ``Detection`` objects from any
backend.

The association is greedy-IoU rather than Hungarian, which avoids a scipy
dependency and is indistinguishable in quality at the handful of boxes an
egocentric desk scene produces.
"""

from __future__ import annotations

import logging
from collections import Counter, deque
from collections.abc import Sequence

from .config import TrackConfig
from .detect.base import Detection, iou

log = logging.getLogger(__name__)

Box = tuple[float, float, float, float]


class Track:
    """One tracked object, with a rolling histogram of its per-frame labels."""

    __slots__ = (
        "track_id",
        "bbox",
        "score",
        "label",
        "_votes",
        "age",
        "hits",
        "time_since_update",
        "_velocity",
        "signals",
        "gaze_frames",
    )

    def __init__(self, track_id: int, det: Detection, vote_window: int) -> None:
        self.track_id = track_id
        self.bbox: Box = det.bbox_xyxy
        self.score = det.score
        self.label = det.label
        self._votes: deque[str] = deque([det.label], maxlen=vote_window)
        self.age = 1
        self.hits = 1
        self.time_since_update = 0
        self._velocity = (0.0, 0.0)
        self.signals = dict(det.signals)
        self.gaze_frames = 0

    # -- geometry ----------------------------------------------------------
    def predict(self) -> Box:
        """Constant-velocity extrapolation of the box centre.

        Cheap, but it matters: under head rotation a device can move a long way
        between frames, and plain IoU association loses the track.
        """
        vx, vy = self._velocity
        x1, y1, x2, y2 = self.bbox
        return (x1 + vx, y1 + vy, x2 + vx, y2 + vy)

    def update(self, det: Detection) -> None:
        old_cx = 0.5 * (self.bbox[0] + self.bbox[2])
        old_cy = 0.5 * (self.bbox[1] + self.bbox[3])
        new_cx = 0.5 * (det.bbox_xyxy[0] + det.bbox_xyxy[2])
        new_cy = 0.5 * (det.bbox_xyxy[1] + det.bbox_xyxy[3])
        # Light smoothing so one jumpy frame doesn't wreck the prediction.
        vx, vy = self._velocity
        self._velocity = (0.5 * vx + 0.5 * (new_cx - old_cx), 0.5 * vy + 0.5 * (new_cy - old_cy))

        self.bbox = det.bbox_xyxy
        self.score = det.score
        self.signals = dict(det.signals)
        self._votes.append(det.label)
        self.hits += 1
        self.time_since_update = 0

    def mark_missed(self) -> None:
        self.time_since_update += 1
        self.bbox = self.predict()

    # -- label voting ------------------------------------------------------
    def voted_label(self, min_fraction: float) -> str:
        """Majority label over the vote window.

        Per-frame open-vocab labels flicker badly under motion blur. Requiring a
        clear plurality before switching means a device sitting still on a desk
        keeps one stable label instead of strobing between tablet and laptop.
        """
        if not self._votes:
            return self.label
        counts = Counter(self._votes)
        label, count = counts.most_common(1)[0]
        if count / len(self._votes) >= min_fraction:
            return label
        return self.label  # not decisive enough, keep what we had

    @property
    def vote_distribution(self) -> dict[str, float]:
        if not self._votes:
            return {}
        n = len(self._votes)
        return {k: v / n for k, v in Counter(self._votes).items()}


class ByteTracker:
    """Two-stage IoU association, as in ByteTrack.

    Pass one matches confident detections; pass two rescues tracks using the
    low-confidence detections that a single-threshold tracker would throw away.
    That second pass is what keeps a phone tracked through the motion blur of a
    head turn.
    """

    def __init__(self, cfg: TrackConfig) -> None:
        self.cfg = cfg
        self._tracks: list[Track] = []
        self._next_id = 1

    @property
    def tracks(self) -> list[Track]:
        return self._tracks

    def reset(self) -> None:
        self._tracks.clear()
        self._next_id = 1

    def update(self, detections: Sequence[Detection]) -> list[Detection]:
        """Associate detections to tracks and return them with track_ids set.

        Only tracks that have been confirmed (``min_hits``) are emitted, and the
        emitted label is the temporal majority rather than this frame's guess.
        """
        cfg = self.cfg
        for t in self._tracks:
            t.age += 1

        high = [d for d in detections if d.score >= cfg.high_thresh]
        low = [d for d in detections if cfg.low_thresh <= d.score < cfg.high_thresh]

        unmatched_tracks = list(self._tracks)

        # -- pass 1: confident detections ----------------------------------
        matched_high, unmatched_high, unmatched_tracks = _greedy_match(
            unmatched_tracks, high, cfg.match_iou
        )
        for track, det in matched_high:
            track.update(det)

        # -- pass 2: rescue with low-confidence detections ------------------
        matched_low, _unmatched_low, unmatched_tracks = _greedy_match(
            unmatched_tracks, low, cfg.match_iou
        )
        for track, det in matched_low:
            track.update(det)

        for track in unmatched_tracks:
            track.mark_missed()

        # -- spawn new tracks ----------------------------------------------
        for det in unmatched_high:
            self._tracks.append(Track(self._next_id, det, cfg.vote_window))
            self._next_id += 1

        # -- retire stale tracks -------------------------------------------
        self._tracks = [t for t in self._tracks if t.time_since_update <= cfg.max_age]

        # -- emit ------------------------------------------------------------
        out: list[Detection] = []
        for t in self._tracks:
            if t.time_since_update > 0:
                continue  # don't emit ghosts on frames where nothing matched
            if t.hits < cfg.min_hits:
                continue
            label = t.voted_label(cfg.vote_min_fraction)
            t.label = label
            det = Detection(
                bbox_xyxy=t.bbox,
                score=t.score,
                raw_label=label,
                label=label,
                track_id=t.track_id,
            )
            det.signals = dict(t.signals)
            det.signals["vote_confidence"] = round(t.vote_distribution.get(label, 0.0), 3)
            out.append(det)
        return out


def _greedy_match(
    tracks: Sequence[Track], dets: Sequence[Detection], min_iou: float
) -> tuple[list[tuple[Track, Detection]], list[Detection], list[Track]]:
    """Greedy highest-IoU-first matching.

    Returns (matched pairs, unmatched detections, unmatched tracks).
    """
    if not tracks or not dets:
        return [], list(dets), list(tracks)

    pairs: list[tuple[float, int, int]] = []
    for ti, t in enumerate(tracks):
        predicted = t.predict()
        for di, d in enumerate(dets):
            score = iou(predicted, d.bbox_xyxy)
            if score >= min_iou:
                pairs.append((score, ti, di))
    pairs.sort(reverse=True)

    used_t: set[int] = set()
    used_d: set[int] = set()
    matched: list[tuple[Track, Detection]] = []
    for _score, ti, di in pairs:
        if ti in used_t or di in used_d:
            continue
        used_t.add(ti)
        used_d.add(di)
        matched.append((tracks[ti], dets[di]))

    unmatched_dets = [d for i, d in enumerate(dets) if i not in used_d]
    unmatched_tracks = [t for i, t in enumerate(tracks) if i not in used_t]
    return matched, unmatched_dets, unmatched_tracks
