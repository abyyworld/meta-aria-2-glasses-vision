"""The marker path, checked against ground truth rather than against itself.

Every test here builds a synthetic screen, warps it by a KNOWN homography, and
then asks the locator to invert that. The expected answer is therefore not
another run of the same code -- it is the geometry that produced the picture,
so an error in the mapping cannot cancel out.
"""

import cv2
import numpy as np
import pytest

from aria_devices.frames import HandSample
from aria_devices.interaction import InteractionState, InteractionTracker
from aria_devices.markers import (
    MARKER_ID_FOR,
    MARKER_RECT,
    ScreenLocator,
    generate_marker_png,
    marker_corners_normalized,
)

SCREEN_W, SCREEN_H = 260, 400
# A genuinely tilted, off-centre view: nothing here is fronto-parallel, which
# is the case a bounding box cannot represent and this is meant to handle.
VIEW = np.float32([[250, 180], [640, 120], [700, 690], [300, 760]])


def _screen_image(device="phone"):
    d = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    screen = np.full((SCREEN_H, SCREEN_W), 235, np.uint8)
    x0, y0, x1, y1 = MARKER_RECT
    a, b = int(x0 * SCREEN_W), int(y0 * SCREEN_H)
    c, e = int(x1 * SCREEN_W), int(y1 * SCREEN_H)
    marker = cv2.aruco.generateImageMarker(d, MARKER_ID_FOR[device], 200)
    screen[b:e, a:c] = cv2.resize(marker, (c - a, e - b),
                                  interpolation=cv2.INTER_NEAREST)
    return screen


def _warped(device="phone"):
    h_true = cv2.getPerspectiveTransform(
        np.float32([[0, 0], [SCREEN_W, 0], [SCREEN_W, SCREEN_H], [0, SCREEN_H]]),
        VIEW,
    )
    img = cv2.warpPerspective(_screen_image(device), h_true,
                              (900, 900), borderValue=255)
    return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB), h_true


def _project(h_true, sx, sy):
    """A screen-normalized point as image pixels, via the true homography."""
    p = h_true @ np.array([sx * SCREEN_W, sy * SCREEN_H, 1.0])
    return p[0] / p[2], p[1] / p[2]


class TestScreenLocator:
    def test_finds_the_marker_and_names_its_device(self):
        img, _ = _warped("phone")
        loc = ScreenLocator()
        assert loc.update(img) == {"phone"}

    @pytest.mark.parametrize("sx,sy", [(0.10, 0.10), (0.50, 0.50), (0.90, 0.85),
                                       (0.25, 0.75), (0.70, 0.30), (0.05, 0.95)])
    def test_maps_image_points_back_to_screen(self, sx, sy):
        img, h_true = _warped()
        loc = ScreenLocator()
        loc.update(img)
        got = loc.to_screen("phone", *_project(h_true, sx, sy))
        # Under 1% of the screen, at a real tilt. For scale, the bounding-box
        # approach this replaces jittered by ~4% with a perspective error on
        # top that moved with the wearer's head.
        assert abs(got[0] - sx) < 0.01
        assert abs(got[1] - sy) < 0.01

    def test_unknown_device_has_no_mapping(self):
        img, _ = _warped("phone")
        loc = ScreenLocator()
        loc.update(img)
        assert loc.to_screen("tablet", 400.0, 400.0) is None

    def test_homography_survives_the_marker_being_covered(self):
        """A hand over the marker is the most likely way it disappears, and is
        exactly when the position is still needed. The screen has not moved."""
        img, h_true = _warped()
        loc = ScreenLocator(hold_frames=5)
        loc.update(img)
        blank = np.full_like(img, 255)
        for _ in range(5):
            loc.update(blank)
        assert "phone" in loc.located()
        got = loc.to_screen("phone", *_project(h_true, 0.5, 0.5))
        assert abs(got[0] - 0.5) < 0.01
        loc.update(blank)                      # past the hold
        assert "phone" not in loc.located()

    def test_corners_are_ordered_like_the_detector_returns_them(self):
        x0, y0, x1, y1 = MARKER_RECT
        # float32, so compare approximately rather than exactly.
        assert marker_corners_normalized() == pytest.approx(
            np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]]), abs=1e-6
        )

    def test_marker_png_round_trips(self):
        png = generate_marker_png("tablet", 256)
        img = cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_GRAYSCALE)
        det = cv2.aruco.ArucoDetector(
            cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50),
            cv2.aruco.DetectorParameters())
        _, ids, _ = det.detectMarkers(img)
        assert ids is not None and int(ids.ravel()[0]) == MARKER_ID_FOR["tablet"]

    def test_stretched_markers_still_decode(self):
        """The marker is a square in NORMALIZED space, so on a real screen it is
        stretched. If that broke detection the whole design would need the
        device's aspect ratio, which is what this avoids."""
        d = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        base = cv2.aruco.generateImageMarker(d, 7, 240)
        det = cv2.aruco.ArucoDetector(d, cv2.aruco.DetectorParameters())
        for w, h in [(240, 240), (110, 240), (163, 240), (240, 110)]:
            m = cv2.resize(base, (w, h), interpolation=cv2.INTER_NEAREST)
            canvas = np.full((h + 160, w + 160), 255, np.uint8)
            canvas[80:80 + h, 80:80 + w] = m
            _, ids, _ = det.detectMarkers(canvas)
            assert ids is not None, f"failed to decode at {w}x{h}"


class TestTrackerUsesMarkers:
    def _hand_at(self, h_true, sx, sy):
        px, py = _project(h_true, sx, sy)
        lm = np.tile(np.array([px, py, 0.0], np.float32), (21, 1))
        return HandSample(side="right", landmarks_px=lm,
                          bbox_xyxy=(px - 30, py - 30, px + 30, py + 30),
                          score=0.99, source="aria")

    def test_enter_move_leave_with_accurate_positions(self):
        img, h_true = _warped()
        loc = ScreenLocator(); loc.update(img)
        tr = InteractionTracker(require_gaze=False)

        ev = tr.update([self._hand_at(h_true, 0.5, 0.5)], [], 0, locator=loc)
        assert [e.state for e in ev] == [InteractionState.ENTER]
        assert abs(ev[0].x - 0.5) < 0.01 and abs(ev[0].y - 0.5) < 0.01

        ev = tr.update([self._hand_at(h_true, 0.25, 0.70)], [], 1, locator=loc)
        assert [e.state for e in ev] == [InteractionState.MOVE]
        assert abs(ev[0].x - 0.25) < 0.01 and abs(ev[0].y - 0.70) < 0.01

        ev = tr.update([self._hand_at(h_true, 1.60, 0.50)], [], 2, locator=loc)
        assert InteractionState.LEAVE in [e.state for e in ev]

    def test_works_with_no_detections_at_all(self):
        """The marker names its own device, so a frame where the open-vocabulary
        detector found nothing still produces a usable position. Detection ran
        at 28-63% of frames; this is what stops that being the ceiling."""
        img, h_true = _warped()
        loc = ScreenLocator(); loc.update(img)
        tr = InteractionTracker(require_gaze=False)
        ev = tr.update([self._hand_at(h_true, 0.4, 0.6)], [], 0, locator=loc)
        assert ev and ev[0].device_label == "phone"
        assert abs(ev[0].x - 0.4) < 0.01

    def test_no_locator_falls_back_to_the_old_path(self):
        tr = InteractionTracker(require_gaze=False)
        assert tr.update([], [], 0) == []
