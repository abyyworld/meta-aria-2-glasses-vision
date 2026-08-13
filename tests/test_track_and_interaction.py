"""Tracker, temporal label voting, gaze attribution and hand interaction."""

from __future__ import annotations

import numpy as np
import pytest

from aria_devices.config import LAPTOP, PHONE, TABLET, TrackConfig
from aria_devices.detect.base import Detection
from aria_devices.frames import GazeSample, HandSample
from aria_devices.gaze import GazeAttributor
from aria_devices.interaction import (
    HandPose,
    analyse_interactions,
    classify_pose,
    hand_openness,
    relative_position,
)
from aria_devices.track import ByteTracker


def det(x1, y1, x2, y2, label=LAPTOP, score=0.9) -> Detection:
    return Detection(bbox_xyxy=(x1, y1, x2, y2), score=score, raw_label=label, label=label)


# ------------------------------------------------------------------ tracker
class TestByteTracker:
    def test_track_is_confirmed_after_min_hits(self):
        cfg = TrackConfig(min_hits=3)
        tracker = ByteTracker(cfg)
        box = det(0, 0, 100, 100)
        assert tracker.update([box]) == []
        assert tracker.update([box]) == []
        out = tracker.update([box])
        assert len(out) == 1
        assert out[0].track_id == 1

    def test_id_is_stable_across_motion(self):
        cfg = TrackConfig(min_hits=1)
        tracker = ByteTracker(cfg)
        ids = []
        for i in range(8):
            out = tracker.update([det(10 * i, 0, 100 + 10 * i, 100)])
            ids.extend(d.track_id for d in out)
        assert len(set(ids)) == 1, f"track id churned: {ids}"

    def test_two_objects_get_distinct_ids(self):
        cfg = TrackConfig(min_hits=1)
        tracker = ByteTracker(cfg)
        out = tracker.update([det(0, 0, 100, 100), det(500, 500, 600, 600)])
        assert len({d.track_id for d in out}) == 2

    def test_track_retires_after_max_age(self):
        cfg = TrackConfig(min_hits=1, max_age=3)
        tracker = ByteTracker(cfg)
        tracker.update([det(0, 0, 100, 100)])
        for _ in range(5):
            tracker.update([])
        assert tracker.tracks == []

    def test_low_confidence_detection_rescues_a_track(self):
        """The second ByteTrack pass: keep the track through motion blur."""
        cfg = TrackConfig(min_hits=1, high_thresh=0.5, low_thresh=0.1)
        tracker = ByteTracker(cfg)
        tracker.update([det(0, 0, 100, 100, score=0.9)])
        # A blurred frame yields only a weak detection.
        out = tracker.update([det(2, 2, 102, 102, score=0.2)])
        assert len(out) == 1
        assert out[0].track_id == 1

    def test_reset_clears_state(self):
        tracker = ByteTracker(TrackConfig(min_hits=1))
        tracker.update([det(0, 0, 100, 100)])
        tracker.reset()
        assert tracker.tracks == []


class TestLabelVoting:
    def test_majority_label_wins_over_a_single_flicker(self):
        cfg = TrackConfig(min_hits=1, vote_window=9, vote_min_fraction=0.34)
        tracker = ByteTracker(cfg)
        box = (0, 0, 100, 200)
        for _ in range(6):
            tracker.update([Detection(box, 0.9, TABLET, TABLET)])
        # One bad frame calls it a laptop.
        out = tracker.update([Detection(box, 0.9, LAPTOP, LAPTOP)])
        assert out[0].label == TABLET, "one flicker frame must not flip the label"

    def test_label_does_switch_when_evidence_persists(self):
        cfg = TrackConfig(min_hits=1, vote_window=5, vote_min_fraction=0.34)
        tracker = ByteTracker(cfg)
        box = (0, 0, 100, 200)
        for _ in range(5):
            tracker.update([Detection(box, 0.9, TABLET, TABLET)])
        for _ in range(5):
            out = tracker.update([Detection(box, 0.9, PHONE, PHONE)])
        assert out[0].label == PHONE

    def test_vote_confidence_is_reported(self):
        cfg = TrackConfig(min_hits=1, vote_window=4)
        tracker = ByteTracker(cfg)
        box = (0, 0, 100, 200)
        for _ in range(4):
            out = tracker.update([Detection(box, 0.9, TABLET, TABLET)])
        assert out[0].signals["vote_confidence"] == pytest.approx(1.0)


# --------------------------------------------------------------------- gaze
class TestGazeAttribution:
    def test_gaze_inside_a_box_marks_it(self):
        attributor = GazeAttributor(hit_radius_px=50)
        d = det(0, 0, 100, 100)
        gaze = GazeSample(0.0, 0.0, point_px=(50, 50))
        target = attributor.attribute([d], gaze, 0)
        assert target is d
        assert d.gazed_at

    def test_gaze_outside_radius_marks_nothing(self):
        attributor = GazeAttributor(hit_radius_px=10)
        d = det(0, 0, 100, 100)
        gaze = GazeSample(0.0, 0.0, point_px=(500, 500))
        assert attributor.attribute([d], gaze, 0) is None
        assert not d.gazed_at

    def test_gaze_near_a_box_still_hits_within_radius(self):
        """Eye tracking has real angular error; strict containment is too harsh."""
        attributor = GazeAttributor(hit_radius_px=50)
        d = det(0, 0, 100, 100)
        gaze = GazeSample(0.0, 0.0, point_px=(120, 50))
        assert attributor.attribute([d], gaze, 0) is d

    def test_smallest_containing_box_wins(self):
        """A phone lying on a laptop should take the gaze, not the laptop."""
        attributor = GazeAttributor()
        laptop = det(0, 0, 400, 300, LAPTOP)
        phone = det(180, 130, 230, 160, PHONE)
        target = attributor.attribute([laptop, phone], GazeSample(0, 0, point_px=(200, 145)), 0)
        assert target is phone

    def test_dwell_accumulates_while_gaze_stays(self):
        attributor = GazeAttributor()
        d = det(0, 0, 100, 100)
        d.track_id = 7
        gaze = GazeSample(0.0, 0.0, point_px=(50, 50))
        attributor.attribute([d], gaze, 0)
        attributor.attribute([d], gaze, 100_000_000)  # +100 ms
        attributor.attribute([d], gaze, 200_000_000)  # +100 ms
        assert d.gaze_dwell_ms == pytest.approx(200.0, abs=1.0)

    def test_dwell_resets_when_gaze_moves_away(self):
        attributor = GazeAttributor(hit_radius_px=10)
        a = det(0, 0, 100, 100)
        a.track_id = 1
        b = det(500, 500, 600, 600)
        b.track_id = 2
        attributor.attribute([a, b], GazeSample(0, 0, point_px=(50, 50)), 0)
        attributor.attribute([a, b], GazeSample(0, 0, point_px=(50, 50)), 100_000_000)
        attributor.attribute([a, b], GazeSample(0, 0, point_px=(550, 550)), 200_000_000)
        assert b.gazed_at
        assert b.gaze_dwell_ms == pytest.approx(100.0, abs=1.0)

    def test_no_gaze_sample_is_safe(self):
        attributor = GazeAttributor()
        assert attributor.attribute([det(0, 0, 10, 10)], None, 0) is None


# -------------------------------------------------------------- interaction
def _hand(landmarks, source="mediapipe", side="right") -> HandSample:
    arr = np.asarray(landmarks, dtype=np.float32)
    x1, y1 = arr.min(axis=0)
    x2, y2 = arr.max(axis=0)
    return HandSample(
        bbox_xyxy=(float(x1), float(y1), float(x2), float(y2)),
        side=side,
        landmarks_px=arr,
        source=source,
    )


def _synthetic_hand(reach: float) -> HandSample:
    """21 MediaPipe-ordered landmarks with fingertips at a chosen reach.

    Wrist at origin, knuckle (index 9) at distance 100, fingertips at
    ``reach * 100``. Everything else is filler.
    """
    pts = [[0.0, 0.0] for _ in range(21)]
    pts[0] = [0.0, 0.0]  # wrist
    pts[9] = [0.0, -100.0]  # middle MCP, the scale reference
    for tip in (4, 8, 12, 16, 20):
        pts[tip] = [0.0, -100.0 * reach]
    return _hand(pts)


class TestHandOpenness:
    def test_fist_scores_low(self):
        assert hand_openness(_synthetic_hand(1.0)) == pytest.approx(0.0)

    def test_open_hand_scores_high(self):
        assert hand_openness(_synthetic_hand(2.2)) == pytest.approx(1.0)

    def test_obliquely_viewed_open_hand_still_reads_as_open(self):
        """Calibrated against a real measurement: a hand reaching over a laptop
        at an angle measured reach 1.55, and must not be scored as a fist."""
        assert hand_openness(_synthetic_hand(1.55)) > 0.55

    def test_openness_is_scale_invariant(self):
        """Same gesture at twice the distance must give the same number."""
        small = _synthetic_hand(1.8)
        big = _hand(np.asarray(small.landmarks_px) * 3.0)
        assert hand_openness(small) == pytest.approx(hand_openness(big), abs=1e-6)

    def test_missing_landmarks_gives_nan(self):
        bare = HandSample(bbox_xyxy=(0, 0, 10, 10), source="openvocab")
        assert hand_openness(bare) != hand_openness(bare)  # NaN


class TestPoseClassification:
    def test_closed_hand_is_grab(self):
        pose, _ = classify_pose(_synthetic_hand(1.1))
        assert pose is HandPose.GRAB

    def test_splayed_hand_is_open(self):
        pose, _ = classify_pose(_synthetic_hand(2.0))
        assert pose is HandPose.OPEN

    def test_real_oblique_open_hand_is_open(self):
        """Regression: this exact reach was measured on a real open hand and
        used to score UNKNOWN before the span was recalibrated."""
        pose, _ = classify_pose(_synthetic_hand(1.55))
        assert pose is HandPose.OPEN

    def test_midway_hand_is_unknown_not_flickering(self):
        """The deliberate gap between thresholds; v1's MID state is dropped."""
        pose, openness = classify_pose(_synthetic_hand(1.36))
        assert pose is HandPose.UNKNOWN
        assert 0.35 < openness < 0.55

    def test_boxes_without_landmarks_are_unknown(self):
        bare = HandSample(bbox_xyxy=(0, 0, 10, 10), source="openvocab")
        pose, _ = classify_pose(bare)
        assert pose is HandPose.UNKNOWN


class TestRelativePosition:
    def test_centre_of_box_is_half_half(self):
        assert relative_position((0, 0, 100, 200), 50, 100) == pytest.approx((0.5, 0.5))

    def test_top_left_corner_is_origin(self):
        assert relative_position((10, 20, 110, 220), 10, 20) == pytest.approx((0.0, 0.0))

    def test_point_outside_returns_none(self):
        assert relative_position((0, 0, 100, 100), 150, 50) is None

    def test_degenerate_box_returns_none(self):
        assert relative_position((5, 5, 5, 5), 5, 5) is None


class TestAnalyseInteractions:
    def test_hand_over_a_device_is_associated(self):
        pts = [[200.0, 200.0] for _ in range(21)]
        pts[0] = [200.0, 300.0]
        pts[9] = [200.0, 250.0]
        hand = _hand(pts)
        device = det(100, 100, 400, 400, LAPTOP)
        device.track_id = 3
        [inter] = analyse_interactions([hand], [device])
        assert inter.device_track_id == 3
        assert inter.device_label == LAPTOP
        assert inter.relative_xy is not None

    def test_hand_away_from_devices_has_no_association(self):
        hand = _synthetic_hand(1.5)
        device = det(1000, 1000, 1200, 1200, LAPTOP)
        [inter] = analyse_interactions([hand], [device])
        assert inter.device_track_id is None
        assert inter.relative_xy is None

    def test_smallest_device_wins_when_boxes_nest(self):
        pts = [[200.0, 200.0] for _ in range(21)]
        pts[0] = [200.0, 300.0]
        pts[9] = [200.0, 250.0]
        hand = _hand(pts)
        laptop = det(0, 0, 500, 500, LAPTOP)
        laptop.track_id = 1
        phone = det(180, 180, 260, 230, PHONE)
        phone.track_id = 2
        [inter] = analyse_interactions([hand], [laptop, phone])
        assert inter.device_track_id == 2

    def test_record_is_json_safe(self):
        import json

        hand = _synthetic_hand(2.0)
        [inter] = analyse_interactions([hand], [])
        json.dumps(inter.to_record())  # must not raise
