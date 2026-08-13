"""Locate device screens by fiducial marker, and map image points onto them.

WHY THIS REPLACES THE BOUNDING BOX. A hand's position on a screen was computed
as a fraction of the detector's bounding box. That cannot be accurate, and no
amount of smoothing fixes it, because the error is geometric rather than noise:

  A box is an AXIS-ALIGNED RECTANGLE. A screen is a TILTED PLANE. "40% across
  the box" is only "40% across the screen" when the device faces the camera
  square-on, which it never does. The error changes with every head movement,
  so it reads as the cursor drifting and jumping rather than as a fixed offset
  anyone would notice and correct.

  A box also JITTERS. Measured live: the laptop's edges moved by ~110px stdev
  in a 704px frame, so a perfectly still hand reported a cursor skidding across
  the screen.

A marker fixes both, because it measures something different. Four detected
corners of a known-size square give a HOMOGRAPHY: the exact projective map
between the screen plane and the image. Perspective is handled by construction
rather than assumed away, corner detection is sub-pixel, and none of it depends
on a detector agreeing with itself frame to frame.

This is the standard approach for a reason -- marker-based planar tracking from
a head-worn camera has been solved since ARToolKit. The open-vocabulary
detector is still useful for saying WHICH devices are present and roughly
where; it is simply the wrong instrument for measuring a position on a screen.

HOW THE GEOMETRY IS AGREED. The marker is drawn by the device's own web page,
so both ends must place it identically. They do, from the constants below:
``MARKER_RECT`` is in screen-normalized coordinates (0..1, origin top-left),
and ``patch_builds.py`` renders CSS percentages from the same numbers. Change
one without the other and every position is silently wrong, so they are stated
once here and imported there.

The marker is a square in NORMALIZED space, which means it looks stretched on a
non-square screen. That is deliberate and it is what removes the aspect ratio
from the problem entirely: this side never needs to know the device's shape.
Detection is unaffected -- verified at aspect ratios from 0.46 to 2.17, which
covers a portrait phone through a wide laptop.
"""

from __future__ import annotations

import logging

import cv2
import numpy as np

log = logging.getLogger(__name__)

#: 4x4 is the smallest dictionary that still has comfortable Hamming distance
#: at 50 ids, and smaller cells survive being far away and slightly blurred.
MARKER_DICT = cv2.aruco.DICT_4X4_50

#: One id per device. The id IS the routing key: the marker says which device it
#: is drawn on, so a phone can never be mistaken for a tablet, which is exactly
#: the failure the open-vocabulary detector is prone to.
MARKER_ID_FOR = {"phone": 1, "tablet": 2, "laptop": 3}
DEVICE_FOR_ID = {v: k for k, v in MARKER_ID_FOR.items()}

#: Where the marker sits on the device's own screen, normalized, origin
#: top-left, y downwards. Top-right corner, a fifth of the screen each way.
#:
#: Big enough to detect from across a desk: at ~20px/cm on this camera, a fifth
#: of a phone's short edge is about 1.5cm, roughly 30px in frame, and detection
#: wants >=20px of marker. Small enough that it does not cover the working area.
MARKER_RECT = (0.78, 0.02, 0.98, 0.22)

#: Frames a homography survives without a sighting. The hand passing over the
#: marker is the single most likely reason it disappears, and that is precisely
#: when the position is needed, so a short memory is essential rather than a
#: nicety. The screen has not moved in that time; only the view of it has.
HOLD_FRAMES = 30


def marker_corners_normalized(rect=MARKER_RECT) -> np.ndarray:
    """The marker's four corners in screen-normalized coordinates.

    Ordered to match what the ArUco detector returns: top-left, top-right,
    bottom-right, bottom-left, as the marker is drawn.
    """
    x0, y0, x1, y1 = rect
    return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=np.float32)


class ScreenLocator:
    """Tracks each device's screen plane from its marker.

    ``update`` once per frame, then ``to_screen`` for any image point.
    """

    def __init__(self, rect=MARKER_RECT, hold_frames: int = HOLD_FRAMES) -> None:
        self._rect = rect
        self._hold = hold_frames
        params = cv2.aruco.DetectorParameters()
        # Sub-pixel corner refinement. Without it corners land on whole pixels
        # and the mapped position quantises visibly on a small marker; this is
        # the difference between a cursor that settles and one that ticks.
        params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        self._detector = cv2.aruco.ArucoDetector(
            cv2.aruco.getPredefinedDictionary(MARKER_DICT), params
        )
        self._src = marker_corners_normalized(rect)
        self._screen_to_image: dict[str, np.ndarray] = {}
        self._image_to_screen: dict[str, np.ndarray] = {}
        self._age: dict[str, int] = {}

    # -- per frame ----------------------------------------------------------
    def update(self, image: np.ndarray) -> set[str]:
        """Detect markers and refresh each device's homography.

        Returns the devices seen THIS frame; devices being held from an earlier
        frame stay usable but are not included.
        """
        grey = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        corners, ids, _ = self._detector.detectMarkers(grey)

        seen: set[str] = set()
        if ids is not None:
            for quad, marker_id in zip(corners, ids.ravel()):
                device = DEVICE_FOR_ID.get(int(marker_id))
                if device is None:
                    continue
                dst = quad.reshape(4, 2).astype(np.float32)
                # Four point pairs, so this is an exact solve rather than a fit;
                # getPerspectiveTransform is the right call and cannot be thrown
                # off by an outlier the way a RANSAC fit could.
                try:
                    h = cv2.getPerspectiveTransform(self._src, dst)
                    h_inv = np.linalg.inv(h)
                except (cv2.error, np.linalg.LinAlgError):
                    # Degenerate quad: a marker seen edge-on, or three corners
                    # collapsed by blur. Keep whatever we had.
                    continue
                self._screen_to_image[device] = h
                self._image_to_screen[device] = h_inv
                self._age[device] = 0
                seen.add(device)

        for device in list(self._age):
            if device in seen:
                continue
            self._age[device] += 1
            if self._age[device] > self._hold:
                self._screen_to_image.pop(device, None)
                self._image_to_screen.pop(device, None)
                self._age.pop(device, None)
        return seen

    # -- queries ------------------------------------------------------------
    def located(self) -> set[str]:
        """Devices with a usable homography, including ones being held."""
        return set(self._image_to_screen)

    def to_screen(self, device: str, px: float, py: float):
        """An image point in that device's screen coordinates, or None.

        Returns normalized (x, y), origin top-left, matching what the rest of
        the pipeline already emits. Points outside 0..1 are returned as-is:
        deciding what counts as "on the screen" belongs to the caller, and a
        point just off the edge is useful for an approach state.
        """
        h_inv = self._image_to_screen.get(device)
        if h_inv is None:
            return None
        v = h_inv @ np.array([px, py, 1.0], dtype=np.float64)
        if abs(v[2]) < 1e-9:
            return None                 # point on the horizon; no finite mapping
        return float(v[0] / v[2]), float(v[1] / v[2])

    def screen_quad(self, device: str):
        """The screen's four corners in image pixels, for drawing. Or None."""
        h = self._screen_to_image.get(device)
        if h is None:
            return None
        unit = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float32).reshape(-1, 1, 2)
        return cv2.perspectiveTransform(unit, h).reshape(4, 2)

    def reset(self) -> None:
        self._screen_to_image.clear()
        self._image_to_screen.clear()
        self._age.clear()


def generate_marker_png(device: str, pixels: int = 512) -> bytes:
    """A PNG of one device's marker, for printing or embedding in a page."""
    if device not in MARKER_ID_FOR:
        raise ValueError(f"no marker id for {device!r}; known: {sorted(MARKER_ID_FOR)}")
    d = cv2.aruco.getPredefinedDictionary(MARKER_DICT)
    img = cv2.aruco.generateImageMarker(d, MARKER_ID_FOR[device], pixels)
    # A quiet zone is required by the detector: without white around it, the
    # marker's outer black border merges into whatever it sits on and nothing
    # is found. One cell's worth is the usual minimum; two is safer in print.
    pad = max(2, pixels // 8)
    canvas = np.full((pixels + 2 * pad, pixels + 2 * pad), 255, np.uint8)
    canvas[pad:pad + pixels, pad:pad + pixels] = img
    ok, buf = cv2.imencode(".png", canvas)
    if not ok:
        raise RuntimeError("could not encode the marker PNG")
    return buf.tobytes()
