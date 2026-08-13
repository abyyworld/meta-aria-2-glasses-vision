"""Rectification + rotation must keep image and calibration in lockstep.

Guarded behind projectaria-tools availability so the suite stays green on a
bare machine.
"""

from __future__ import annotations

import numpy as np
import pytest

projectaria = pytest.importorskip("projectaria_tools.core.calibration")
cal = projectaria

from aria_devices.config import RectifyConfig  # noqa: E402


@pytest.fixture
def linear_calib():
    """A synthetic pinhole calibration standing in for a real Aria one."""
    return cal.get_linear_camera_calibration(640, 480, 300.0, "camera-rgb")


class TestCalibRotation:
    def test_rotation_swaps_image_dimensions(self, linear_calib):
        rotated = cal.rotate_camera_calib_cw90deg(linear_calib)
        w0, h0 = (int(v) for v in linear_calib.get_image_size())
        w1, h1 = (int(v) for v in rotated.get_image_size())
        assert (w0, h0) == (640, 480)
        assert (w1, h1) == (480, 640), "a 90 degree rotation must transpose the size"

    def test_rotation_swaps_focal_lengths(self, linear_calib):
        rotated = cal.rotate_camera_calib_cw90deg(linear_calib)
        fx0, fy0 = (float(v) for v in linear_calib.get_focal_lengths())
        fx1, fy1 = (float(v) for v in rotated.get_focal_lengths())
        assert fx1 == pytest.approx(fy0)
        assert fy1 == pytest.approx(fx0)

    def test_rotation_is_not_destructive_to_principal_point(self, linear_calib):
        rotated = cal.rotate_camera_calib_cw90deg(linear_calib)
        cx, cy = (float(v) for v in rotated.get_principal_point())
        w, h = (int(v) for v in rotated.get_image_size())
        assert 0 <= cx <= w
        assert 0 <= cy <= h


class TestPixelRotationMatchesCalib:
    def test_image_and_calib_dimensions_stay_consistent(self, linear_calib):
        """The core invariant: rotate pixels and calib together or not at all."""
        from aria_devices.sources.vrs import rotate_image_cw90

        w, h = (int(v) for v in linear_calib.get_image_size())
        image = np.zeros((h, w, 3), np.uint8)

        rotated_image = rotate_image_cw90(image)
        rotated_calib = cal.rotate_camera_calib_cw90deg(linear_calib)

        calib_w, calib_h = (int(v) for v in rotated_calib.get_image_size())
        assert rotated_image.shape[:2] == (calib_h, calib_w), (
            "rotated pixels and rotated calibration disagree on size"
        )

    def test_rotation_moves_a_marker_to_the_expected_corner(self):
        """A 90 degree CW rotation sends top-left to top-right."""
        from aria_devices.sources.vrs import rotate_image_cw90

        image = np.zeros((10, 20, 3), np.uint8)
        image[0, 0] = 255  # top-left
        rotated = rotate_image_cw90(image)
        assert rotated.shape[:2] == (20, 10)
        assert rotated[0, -1].tolist() == [255, 255, 255]
        assert rotated[0, 0].tolist() == [0, 0, 0]


class TestRectifyMaps:
    def test_maps_have_destination_shape(self, linear_calib):
        from aria_devices.sources.vrs import build_rectify_maps

        dst = cal.get_linear_camera_calibration(64, 48, 40.0, "camera-rgb")
        map_x, map_y = build_rectify_maps(linear_calib, dst, cache=False)
        assert map_x.shape == (48, 64)
        assert map_y.shape == (48, 64)
        assert map_x.dtype == np.float32

    def test_identity_calibration_maps_to_itself(self):
        """Rectifying a pinhole to the same pinhole must be a no-op mapping."""
        from aria_devices.sources.vrs import build_rectify_maps

        calib = cal.get_linear_camera_calibration(32, 24, 20.0, "camera-rgb")
        map_x, map_y = build_rectify_maps(calib, calib, cache=False)
        ys, xs = np.mgrid[0:24, 0:32]
        assert np.allclose(map_x, xs, atol=1e-3)
        assert np.allclose(map_y, ys, atol=1e-3)

    def test_remap_preserves_content_under_identity(self):
        import cv2

        from aria_devices.sources.vrs import build_rectify_maps

        calib = cal.get_linear_camera_calibration(32, 24, 20.0, "camera-rgb")
        map_x, map_y = build_rectify_maps(calib, calib, cache=False)
        rng = np.random.default_rng(0)
        image = rng.integers(0, 255, (24, 32, 3), dtype=np.uint8)
        out = cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR)
        assert out.shape == image.shape
        # Bilinear sampling at exact integer coordinates is lossless.
        assert np.abs(out.astype(int) - image.astype(int)).max() <= 1


class TestRectifyConfig:
    def test_defaults_are_square_and_upright(self):
        cfg = RectifyConfig()
        assert cfg.enabled
        # No rotation: the live Gen 2 stream arrives upright. This asserted
        # "ccw90" on the strength of a comment claiming verification. A frame
        # captured off the glasses settled it: the saved image was lying on its
        # side, and one more clockwise quarter-turn made it upright, so ccw90
        # was turning an already-upright stream out of true.
        assert cfg.rotation == "none"
        assert cfg.size[0] == cfg.size[1]

    def test_focal_and_size_are_overridable(self):
        cfg = RectifyConfig(size=(512, 512), focal=280.0)
        assert cfg.size == (512, 512)
        assert cfg.focal == 280.0
