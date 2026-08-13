"""Sensor dashboard layout.

``render_dashboard`` is a pure function, so all of this runs with no glasses,
no recording and no model weights.
"""

from __future__ import annotations

import numpy as np
import pytest

from aria_devices.viewer import (
    AriaSnapshot,
    RateMeter,
    SensorPanel,
    fit_into,
    render_dashboard,
)


def img(w: int, h: int, value: int = 200, channels: int = 3) -> np.ndarray:
    shape = (h, w, channels) if channels > 1 else (h, w)
    return np.full(shape, value, np.uint8)


# ------------------------------------------------------------------- fitting
class TestFitInto:
    def test_output_is_exactly_the_requested_box(self):
        out = fit_into(img(100, 50), 200, 200)
        assert out.shape == (200, 200, 3)

    def test_aspect_ratio_is_preserved_by_letterboxing(self):
        """Stretching would misrepresent the fisheye geometry these panels
        exist to let you judge."""
        out = fit_into(img(200, 100, 255), 200, 200)
        # A 2:1 image in a square box fills the full width, half the height.
        filled_rows = np.where(out.max(axis=(1, 2)) > 100)[0]
        assert len(filled_rows) == pytest.approx(100, abs=2)

    def test_grayscale_is_converted_to_bgr(self):
        """SLAM and eye-tracking cameras are monochrome."""
        out = fit_into(img(64, 64, 128, channels=1), 64, 64)
        assert out.shape == (64, 64, 3)

    def test_none_image_yields_a_blank_panel(self):
        out = fit_into(None, 80, 60)
        assert out.shape == (60, 80, 3)

    def test_empty_image_does_not_crash(self):
        out = fit_into(np.zeros((0, 0, 3), np.uint8), 80, 60)
        assert out.shape == (60, 80, 3)

    def test_upscaling_and_downscaling_both_work(self):
        assert fit_into(img(10, 10), 200, 200).shape == (200, 200, 3)
        assert fit_into(img(2000, 2000), 100, 100).shape == (100, 100, 3)


# ------------------------------------------------------------- rate metering
class TestRateMeter:
    def test_starts_at_zero(self):
        assert RateMeter().hz == 0.0

    def test_single_sample_is_still_zero(self):
        meter = RateMeter()
        meter.tick()
        assert meter.hz == 0.0

    def test_reports_a_positive_rate_after_several_ticks(self):
        meter = RateMeter()
        for _ in range(5):
            meter.tick()
        assert meter.hz > 0

    def test_untouched_meter_is_infinitely_stale(self):
        assert RateMeter().stale_for == float("inf")

    def test_stale_for_is_small_right_after_a_tick(self):
        meter = RateMeter()
        meter.tick()
        assert meter.stale_for < 1.0


# ------------------------------------------------------------------ dashboard
class TestRenderDashboard:
    def test_returns_the_requested_canvas_size(self):
        out = render_dashboard(AriaSnapshot(main=img(640, 480)), width=1280, height=720)
        assert out.shape == (720, 1280, 3)
        assert out.dtype == np.uint8

    def test_renders_with_no_data_at_all(self):
        """Before the first frame arrives there is nothing to draw, and that
        must not be a crash."""
        out = render_dashboard(AriaSnapshot())
        assert out.shape[2] == 3

    def test_missing_streams_are_drawn_greyed_not_omitted(self):
        """A dead stream is information; hiding it hides the failure."""
        snapshot = AriaSnapshot(
            main=img(640, 480),
            panels=[
                SensorPanel("slam-front-left", img(320, 240, 180, channels=1), 30.0),
                SensorPanel("slam-front-right", None),  # never produced data
            ],
        )
        out = render_dashboard(snapshot, width=1200, height=700)
        assert out.shape == (700, 1200, 3)

    def test_all_six_side_cameras_fit(self):
        panels = [
            SensorPanel(f"cam-{i}", img(320, 240, 100 + 10 * i, channels=1), 30.0)
            for i in range(6)
        ]
        out = render_dashboard(
            AriaSnapshot(main=img(704, 704), panels=panels), width=1600, height=900
        )
        assert out.shape == (900, 1600, 3)

    def test_stats_strip_appears_when_stats_are_given(self):
        without = render_dashboard(AriaSnapshot(main=img(320, 240)), width=800, height=600)
        with_stats = render_dashboard(
            AriaSnapshot(main=img(320, 240), stats=[("rgb", "30.0 Hz")]),
            width=800, height=600,
        )
        assert without.shape == with_stats.shape
        assert not np.array_equal(without, with_stats)

    def test_many_stats_do_not_overflow_the_canvas(self):
        stats = [(f"key{i}", f"value{i}") for i in range(24)]
        out = render_dashboard(
            AriaSnapshot(main=img(320, 240), stats=stats), width=900, height=600
        )
        assert out.shape == (600, 900, 3)

    def test_tiny_canvas_is_clamped_rather_than_crashing(self):
        out = render_dashboard(AriaSnapshot(main=img(640, 480)), width=10, height=10)
        assert out.shape[0] >= 360 and out.shape[1] >= 640

    def test_panel_count_beyond_the_grid_is_truncated_safely(self):
        panels = [SensorPanel(f"c{i}", img(64, 64, channels=1), 10.0) for i in range(40)]
        out = render_dashboard(
            AriaSnapshot(main=img(320, 240), panels=panels), width=1000, height=600
        )
        assert out.shape == (600, 1000, 3)

    def test_main_view_is_actually_drawn(self):
        """A bright main image must brighten the canvas."""
        dark = render_dashboard(AriaSnapshot(main=img(640, 480, 0)), width=1000, height=600)
        bright = render_dashboard(AriaSnapshot(main=img(640, 480, 255)), width=1000, height=600)
        assert bright.mean() > dark.mean() + 20

    def test_notes_are_rendered(self):
        plain = render_dashboard(AriaSnapshot(main=img(320, 240)), width=800, height=600)
        noted = render_dashboard(
            AriaSnapshot(main=img(320, 240), notes=["eyegaze stream missing"]),
            width=800, height=600,
        )
        assert not np.array_equal(plain, noted)


class TestSensorPanel:
    def test_panel_without_image_is_not_live(self):
        assert not SensorPanel("slam-front-left").is_live

    def test_panel_with_image_is_live(self):
        assert SensorPanel("slam-front-left", img(32, 32)).is_live
