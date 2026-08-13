"""Hand-on-device events: the signal the interaction system consumes.

Covers screen-rect trimming, approach distance, and enter/move/leave
transitions. Synthetic boxes only — no weights, no hardware.
"""

from __future__ import annotations

import numpy as np
import pytest

from aria_devices.config import LAPTOP, PHONE, TABLET, DEFAULT_DEVICE_PROFILES
from aria_devices.detect.base import Detection
from aria_devices.frames import HandSample
from aria_devices.interaction import (
    HandPose,
    InteractionState,
    InteractionTracker,
    normalised_distance,
    screen_rect,
)

PROFILES = DEFAULT_DEVICE_PROFILES


def det(x1, y1, x2, y2, label=TABLET, track_id=1) -> Detection:
    d = Detection((x1, y1, x2, y2), 0.9, label, label)
    d.track_id = track_id
    return d


def hand_at(x: float, y: float, side: str = "right") -> HandSample:
    """A hand whose fingertips sit exactly at (x, y)."""
    pts = [[x, y] for _ in range(21)]
    pts[0] = [x, y + 100.0]  # wrist below
    pts[9] = [x, y + 50.0]  # knuckle
    arr = np.asarray(pts, dtype=np.float32)
    return HandSample(
        bbox_xyxy=(x - 20, y - 20, x + 20, y + 120),
        side=side,
        landmarks_px=arr,
        source="mediapipe",
    )


# ------------------------------------------------------------- screen rect
class TestScreenRect:
    def test_laptop_screen_excludes_the_keyboard_base(self):
        """The box spans lid + base; the screen is only the top part."""
        box = (0.0, 0.0, 400.0, 300.0)
        rect = screen_rect(box, PROFILES[LAPTOP])
        assert rect[3] < box[3], "laptop screen must not reach the bottom of the box"
        # 0.38 bottom inset -> screen ends at 62% of the height.
        assert rect[3] == pytest.approx(300.0 * 0.62, abs=1.0)

    def test_tablet_trim_is_a_thin_symmetric_bezel(self):
        box = (0.0, 0.0, 200.0, 300.0)
        x1, y1, x2, y2 = screen_rect(box, PROFILES[TABLET])
        assert x1 == pytest.approx(12.0)  # 6% of 200
        assert (200.0 - x2) == pytest.approx(12.0)
        assert y1 == pytest.approx(18.0)  # 6% of 300
        assert (300.0 - y2) == pytest.approx(18.0)

    def test_phone_trim_is_smallest(self):
        box = (0.0, 0.0, 200.0, 100.0)
        phone = screen_rect(box, PROFILES[PHONE])
        tablet = screen_rect(box, PROFILES[TABLET])
        assert (phone[2] - phone[0]) > (tablet[2] - tablet[0])

    def test_degenerate_box_falls_back_to_itself(self):
        box = (5.0, 5.0, 5.0, 5.0)
        assert screen_rect(box, PROFILES[LAPTOP]) == box

    def test_cursor_lands_higher_once_the_base_is_trimmed(self):
        """The bug this inset exists to prevent."""
        from aria_devices.interaction import relative_position

        box = (0.0, 0.0, 400.0, 300.0)
        point_y = 150.0  # halfway down the whole box
        raw = relative_position(box, 200.0, point_y)
        trimmed = relative_position(screen_rect(box, PROFILES[LAPTOP]), 200.0, point_y)
        assert raw[1] == pytest.approx(0.5)
        assert trimmed[1] > 0.7, "untrimmed mapping puts the cursor far too high up"


# --------------------------------------------------------------- distance
class TestNormalisedDistance:
    def test_point_inside_is_zero(self):
        assert normalised_distance((0, 0, 100, 100), 50, 50) == 0.0

    def test_distance_is_scale_free(self):
        """Same relative offset at two scales gives the same number."""
        near = normalised_distance((0, 0, 100, 100), 150, 50)
        far = normalised_distance((0, 0, 200, 200), 300, 100)
        assert near == pytest.approx(far)

    def test_distance_grows_with_separation(self):
        a = normalised_distance((0, 0, 100, 100), 120, 50)
        b = normalised_distance((0, 0, 100, 100), 200, 50)
        assert b > a


# ----------------------------------------------------------------- events
class TestInteractionTracker:
    def test_hand_over_screen_emits_enter_then_move(self):
        tracker = InteractionTracker(profiles=PROFILES)
        device = det(0, 0, 200, 300, TABLET)
        hand = hand_at(100, 150)

        first = tracker.update([hand], [device], 0)
        assert [e.state for e in first] == [InteractionState.ENTER]

        second = tracker.update([hand], [device], 1_000_000)
        assert [e.state for e in second] == [InteractionState.MOVE]

    def test_leaving_emits_exactly_one_leave(self):
        tracker = InteractionTracker(profiles=PROFILES)
        device = det(0, 0, 200, 300, TABLET)
        tracker.update([hand_at(100, 150)], [device], 0)

        away = hand_at(5000, 5000)
        leaving = tracker.update([away], [device], 1_000_000)
        assert [e.state for e in leaving] == [InteractionState.LEAVE]

        # And nothing further on subsequent frames.
        assert tracker.update([away], [device], 2_000_000) == []

    def test_approach_fires_before_entering(self):
        tracker = InteractionTracker(approach_distance=0.6, profiles=PROFILES)
        device = det(0, 0, 200, 300, TABLET)
        # Just outside the screen rect but well within the approach radius.
        events = tracker.update([hand_at(215, 150)], [device], 0)
        assert [e.state for e in events] == [InteractionState.APPROACH]
        assert events[0].distance > 0

    def test_far_hand_emits_nothing(self):
        tracker = InteractionTracker(approach_distance=0.6, profiles=PROFILES)
        device = det(0, 0, 200, 300, TABLET)
        assert tracker.update([hand_at(9000, 9000)], [device], 0) == []

    def test_event_carries_normalised_screen_coordinates(self):
        tracker = InteractionTracker(profiles=PROFILES)
        device = det(0, 0, 200, 300, TABLET)
        rect = screen_rect(device.bbox_xyxy, PROFILES[TABLET])
        cx = 0.5 * (rect[0] + rect[2])
        cy = 0.5 * (rect[1] + rect[3])

        [event] = tracker.update([hand_at(cx, cy)], [device], 0)
        assert event.x == pytest.approx(0.5, abs=0.01)
        assert event.y == pytest.approx(0.5, abs=0.01)
        assert 0.0 <= event.x <= 1.0 and 0.0 <= event.y <= 1.0

    def test_device_label_routes_the_event(self):
        tracker = InteractionTracker(profiles=PROFILES)
        laptop = det(0, 0, 400, 300, LAPTOP, track_id=1)
        phone = det(900, 0, 1100, 100, PHONE, track_id=2)
        [event] = tracker.update([hand_at(1000, 40)], [laptop, phone], 0)
        assert event.device_label == PHONE
        assert event.device_track_id == 2

    def test_two_hands_tracked_independently(self):
        tracker = InteractionTracker(profiles=PROFILES)
        device = det(0, 0, 200, 300, TABLET)
        left = hand_at(80, 150, side="left")
        right = hand_at(120, 150, side="right")
        events = tracker.update([left, right], [device], 0)
        assert {e.hand_side for e in events} == {"left", "right"}
        assert all(e.state is InteractionState.ENTER for e in events)

    def test_one_hand_leaving_does_not_clear_the_other(self):
        tracker = InteractionTracker(profiles=PROFILES)
        device = det(0, 0, 200, 300, TABLET)
        left = hand_at(80, 150, side="left")
        right = hand_at(120, 150, side="right")
        tracker.update([left, right], [device], 0)

        events = tracker.update([left, hand_at(9000, 9000, side="right")], [device], 1)
        states = {(e.hand_side, e.state) for e in events}
        assert ("right", InteractionState.LEAVE) in states
        assert ("left", InteractionState.MOVE) in states

    def test_pose_is_carried_on_the_event(self):
        tracker = InteractionTracker(profiles=PROFILES)
        device = det(0, 0, 200, 300, TABLET)
        [event] = tracker.update([hand_at(100, 150)], [device], 0)
        assert isinstance(event.pose, HandPose)

    def test_reset_clears_inside_state(self):
        tracker = InteractionTracker(profiles=PROFILES)
        device = det(0, 0, 200, 300, TABLET)
        hand = hand_at(100, 150)
        tracker.update([hand], [device], 0)
        tracker.reset()
        # After a reset the next frame is an ENTER again, not a MOVE.
        [event] = tracker.update([hand], [device], 1)
        assert event.state is InteractionState.ENTER

    def test_record_is_json_safe_and_has_the_integration_fields(self):
        import json

        tracker = InteractionTracker(profiles=PROFILES)
        device = det(0, 0, 200, 300, TABLET)
        [event] = tracker.update([hand_at(100, 150)], [device], 12345)
        record = event.to_record()
        json.dumps(record)
        assert set(record) >= {"state", "device", "hand", "pose", "x", "y", "timestamp_ns"}


# ------------------------------------------------------------- gaze gating
class TestGazeGating:
    """A device must stay inert unless the wearer is looking at it.

    With three devices 20 cm apart, a hand reaching for the tablet crosses over
    the laptop on nearly every reach. Without gating, the laptop fires events
    the wearer never intended.
    """

    def _tracker(self, **kw):
        return InteractionTracker(profiles=PROFILES, require_gaze=True, **kw)

    def test_unlooked_device_ignores_a_hand_on_it(self):
        tracker = self._tracker()
        device = det(0, 0, 200, 300, TABLET)
        device.gazed_at = False
        assert tracker.update([hand_at(100, 150)], [device], 0) == []

    def test_looked_at_device_reacts(self):
        tracker = self._tracker()
        device = det(0, 0, 200, 300, TABLET)
        device.gazed_at = True
        [event] = tracker.update([hand_at(100, 150)], [device], 0)
        assert event.state is InteractionState.ENTER

    def test_hand_crossing_an_unlooked_device_fires_only_on_the_looked_one(self):
        """The exact 20-cm-apart failure this gating exists to prevent."""
        tracker = self._tracker()
        laptop = det(0, 0, 200, 300, "laptop", track_id=1)
        laptop.gazed_at = False
        tablet = det(220, 0, 420, 300, TABLET, track_id=2)
        tablet.gazed_at = True

        # The hand is physically on the laptop, but the wearer is looking at the
        # tablet: the laptop must stay silent. The tablet may legitimately
        # report an approach, since the hand is heading towards it.
        crossing = tracker.update([hand_at(100, 150)], [laptop, tablet], 0)
        assert all(e.device_label != "laptop" for e in crossing), (
            "laptop fired while the wearer looked elsewhere"
        )
        assert all(e.state is not InteractionState.ENTER for e in crossing)

        arriving = tracker.update([hand_at(320, 150)], [laptop, tablet], 1_000_000)
        assert [(e.device_label, e.state) for e in arriving] == [
            (TABLET, InteractionState.ENTER)
        ]

    def test_grace_period_survives_a_glance_away(self):
        """Gaze flicking to the moving hand must not kill the interaction."""
        tracker = self._tracker(gaze_grace_ms=800.0)
        device = det(0, 0, 200, 300, TABLET)
        device.gazed_at = True
        tracker.update([hand_at(100, 150)], [device], 0)

        device.gazed_at = False
        [event] = tracker.update([hand_at(100, 150)], [device], 400_000_000)  # 400 ms
        assert event.state is InteractionState.MOVE

    def test_interaction_ends_once_grace_expires(self):
        tracker = self._tracker(gaze_grace_ms=800.0)
        device = det(0, 0, 200, 300, TABLET)
        device.gazed_at = True
        tracker.update([hand_at(100, 150)], [device], 0)

        device.gazed_at = False
        events = tracker.update([hand_at(100, 150)], [device], 2_000_000_000)  # 2 s
        assert [e.state for e in events] == [InteractionState.LEAVE]

    def test_gating_is_off_by_default_for_non_aria_sources(self):
        """A webcam has no gaze stream; requiring it would emit nothing at all."""
        tracker = InteractionTracker(profiles=PROFILES)
        device = det(0, 0, 200, 300, TABLET)  # gazed_at defaults False
        assert tracker.update([hand_at(100, 150)], [device], 0)

    def test_reset_clears_gaze_memory(self):
        tracker = self._tracker()
        device = det(0, 0, 200, 300, TABLET)
        device.gazed_at = True
        tracker.update([hand_at(100, 150)], [device], 0)
        tracker.reset()
        device.gazed_at = False
        assert tracker.update([hand_at(100, 150)], [device], 1_000_000) == []
