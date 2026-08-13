"""Robustness for the real study setup: three static devices, 20 cm apart.

Two failure modes are covered here, both of which delete a device that is
plainly still there:

* world-fixed suppression treating a stationary desk device as a wall monitor
* detection dropping out for a few frames when a hand occludes a screen
"""

from __future__ import annotations

import numpy as np
import pytest

from aria_devices.config import LAPTOP, PHONE, TABLET, PipelineConfig
from aria_devices.detect.base import Detection
from aria_devices.detect.disambiguate import WorldFixedTracker
from aria_devices.frames import Frame
from aria_devices.track import DevicePersistence


def det(x1, y1, x2, y2, label=TABLET, track_id=1, score=0.8, diag_cm=0.0) -> Detection:
    d = Detection((float(x1), float(y1), float(x2), float(y2)), score, label, label)
    d.track_id = track_id
    if diag_cm:
        d.signals["diag_cm"] = diag_cm
    return d


class TestDevicePersistence:
    def test_fresh_detections_pass_straight_through(self):
        p = DevicePersistence(max_age_frames=10)
        out = p.update([det(0, 0, 100, 100)], 0)
        assert len(out) == 1
        assert not out[0].persisted

    def test_device_survives_a_dropout(self):
        """The hand-occlusion case: box vanishes, device is still on the desk."""
        p = DevicePersistence(max_age_frames=10)
        p.update([det(0, 0, 100, 100, track_id=7)], 0)
        out = p.update([], 1)
        assert len(out) == 1
        assert out[0].track_id == 7
        assert out[0].persisted

    def test_persisted_box_is_marked_and_decays(self):
        """A remembered box must never masquerade as a fresh observation."""
        p = DevicePersistence(max_age_frames=10, decay=0.9)
        p.update([det(0, 0, 100, 100, score=0.8)], 0)
        first = p.update([], 1)[0]
        second = p.update([], 2)[0]
        assert first.persisted and second.persisted
        assert second.score < first.score < 0.8
        assert any("persisted" in n for n in first.notes)

    def test_persistence_expires(self):
        p = DevicePersistence(max_age_frames=3)
        p.update([det(0, 0, 100, 100)], 0)
        assert p.update([], 3)  # still within the window
        assert p.update([], 99) == []

    def test_reappearing_device_is_fresh_again(self):
        p = DevicePersistence(max_age_frames=10)
        p.update([det(0, 0, 100, 100, track_id=7)], 0)
        p.update([], 1)
        out = p.update([det(0, 0, 100, 100, track_id=7)], 2)
        assert len(out) == 1
        assert not out[0].persisted

    def test_a_ghost_never_re_anchors_the_memory(self):
        """Otherwise a device could persist forever off its own echo."""
        p = DevicePersistence(max_age_frames=3)
        p.update([det(0, 0, 100, 100)], 0)
        for f in range(1, 4):
            p.update([], f)
        assert p.update([], 10) == []

    def test_three_devices_persist_independently(self):
        """The study setup: one occluded device must not affect the others."""
        p = DevicePersistence(max_age_frames=10)
        devices = [
            det(0, 0, 100, 100, LAPTOP, track_id=1),
            det(120, 0, 200, 100, TABLET, track_id=2),
            det(220, 0, 260, 80, PHONE, track_id=3),
        ]
        p.update(devices, 0)
        out = p.update([devices[0], devices[2]], 1)  # tablet occluded by a hand
        labels = {d.label: d.persisted for d in out}
        assert labels == {LAPTOP: False, TABLET: True, PHONE: False}

    def test_reset_clears_memory(self):
        p = DevicePersistence()
        p.update([det(0, 0, 100, 100)], 0)
        p.reset()
        assert p.update([], 1) == []

    def test_untracked_detections_are_ignored(self):
        p = DevicePersistence()
        d = Detection((0.0, 0.0, 10.0, 10.0), 0.9, TABLET, TABLET)  # track_id None
        assert len(p.update([d], 0)) == 1
        assert p.update([], 1) == []


class TestWorldFixedDoesNotEatStaticDevices:
    """Regression: a stationary desk device was being deleted as a distractor."""

    def test_suppression_is_off_by_default(self):
        """The study is static devices on a desk — the opposite of a wall TV."""
        assert PipelineConfig().disambiguation.suppress_world_fixed is False

    def test_tracker_still_flags_a_motionless_box(self):
        """The underlying signal is intact; only its use is gated."""
        cfg = PipelineConfig().disambiguation
        wf = WorldFixedTracker(cfg)
        box = (100.0, 100.0, 300.0, 400.0)
        for _ in range(cfg.world_fixed_frames):
            wf.update(1, box)
        assert wf.is_world_fixed(1, box)

    def test_device_sized_object_is_never_suppressed(self):
        """A 22 cm tablet sitting still is a tablet, not a monitor."""
        from aria_devices.pipeline import DevicePipeline

        cfg = PipelineConfig()
        cfg.disambiguation.suppress_world_fixed = True  # even when enabled
        pipe = DevicePipeline.__new__(DevicePipeline)
        pipe.cfg = cfg
        pipe.world_fixed = WorldFixedTracker(cfg.disambiguation)

        box = (100.0, 100.0, 300.0, 400.0)
        for _ in range(cfg.disambiguation.world_fixed_frames):
            pipe.world_fixed.update(1, box)

        tablet = det(*box, TABLET, track_id=1, diag_cm=22.0)
        assert not pipe._is_world_fixed_distractor(tablet)

    def test_monitor_sized_object_is_still_suppressed(self):
        """The rule must keep working for what it was written for."""
        from aria_devices.pipeline import DevicePipeline

        cfg = PipelineConfig()
        cfg.disambiguation.suppress_world_fixed = True
        pipe = DevicePipeline.__new__(DevicePipeline)
        pipe.cfg = cfg
        pipe.world_fixed = WorldFixedTracker(cfg.disambiguation)

        box = (100.0, 100.0, 700.0, 500.0)
        for _ in range(cfg.disambiguation.world_fixed_frames):
            pipe.world_fixed.update(2, box)

        monitor = det(*box, LAPTOP, track_id=2, diag_cm=60.0)
        assert pipe._is_world_fixed_distractor(monitor)


class TestStaticDeskEndToEnd:
    """Three stable devices, 20 cm apart, seen for several seconds."""

    def test_devices_survive_a_long_static_run(self):
        from aria_devices.detect.base import Detector
        from aria_devices.pipeline import DevicePipeline

        boxes = {
            LAPTOP: (20.0, 120.0, 260.0, 340.0),
            TABLET: (300.0, 130.0, 450.0, 330.0),
            PHONE: (490.0, 200.0, 600.0, 265.0),
        }

        class _Static(Detector):
            prompts = ["laptop", "tablet computer", "smartphone"]

            def detect(self, image_rgb):
                return [
                    Detection(boxes[LAPTOP], 0.90, "laptop"),
                    Detection(boxes[TABLET], 0.80, "tablet computer"),
                    Detection(boxes[PHONE], 0.75, "smartphone"),
                ]

        cfg = PipelineConfig()
        cfg.hands.backend = "off"
        pipe = DevicePipeline(cfg, detector=_Static(), focal_px=600.0)

        image = np.zeros((480, 640, 3), np.uint8)
        for x1, y1, x2, y2 in boxes.values():
            image[int(y1) : int(y2), int(x1) : int(x2)] = 200

        seen = []
        for i in range(90):  # 3 s at 30 fps, twice world_fixed_frames
            result = pipe.process(Frame(rgb=image, timestamp_ns=i * 33_000_000, frame_idx=i))
            seen.append(len(result.detections))
        pipe.close()

        # Nothing may vanish just because it sat still.
        assert seen[-1] == 3, f"devices disappeared over a static run: {seen[-10:]}"
        assert min(seen[5:]) == 3
