"""Unit tests for every disambiguation rule, on synthetic boxes.

These must pass with no model weights, no network and no Aria hardware.
"""

from __future__ import annotations

import numpy as np
import pytest

from aria_devices.config import (
    LAPTOP,
    PHONE,
    TABLET,
    DeviceProfile,
    DisambiguationConfig,
)
from aria_devices.detect.base import Detection, containment, iou
from aria_devices.detect.disambiguate import (
    cluster_detections,
    collapse_raw_label,
    estimate_diagonal_cm,
    is_tablet_inside_laptop,
    keyboard_is_below,
    orientation_prior_score,
    score_detections,
    screen_on_score,
    shape_prior_score,
    size_prior_score,
    suppress_tablets_inside_laptops,
    WorldFixedTracker,
)


@pytest.fixture
def cfg() -> DisambiguationConfig:
    return DisambiguationConfig()


def det(x1, y1, x2, y2, raw="laptop", score=0.8, label="") -> Detection:
    return Detection(bbox_xyxy=(x1, y1, x2, y2), score=score, raw_label=raw, label=label)


# ---------------------------------------------------------------- geometry
class TestGeometry:
    def test_iou_identical_boxes_is_one(self):
        assert iou((0, 0, 10, 10), (0, 0, 10, 10)) == pytest.approx(1.0)

    def test_iou_disjoint_boxes_is_zero(self):
        assert iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0

    def test_iou_half_overlap(self):
        # Two 10x10 boxes overlapping in a 5x10 strip: inter 50, union 150.
        assert iou((0, 0, 10, 10), (5, 0, 15, 10)) == pytest.approx(50 / 150)

    def test_containment_full(self):
        assert containment((2, 2, 4, 4), (0, 0, 10, 10)) == pytest.approx(1.0)

    def test_containment_partial(self):
        # Inner box 4x4=16, half of it inside.
        assert containment((8, 0, 12, 4), (0, 0, 10, 10)) == pytest.approx(0.5)


# ------------------------------------------------------- label collapsing
class TestCollapse:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("laptop", LAPTOP),
            ("open laptop computer", LAPTOP),
            ("MacBook", LAPTOP),
            ("tablet computer", TABLET),
            ("iPad", TABLET),
            ("smartphone", PHONE),
            ("mobile phone held in hand", PHONE),
            ("computer keyboard", "keyboard"),
            ("computer monitor", "monitor"),
        ],
    )
    def test_known_prompts_collapse(self, raw, expected):
        assert collapse_raw_label(raw) == expected

    def test_case_and_article_insensitive(self):
        assert collapse_raw_label("  A LAPTOP ") == LAPTOP
        assert collapse_raw_label("the iPad") == TABLET

    def test_unknown_prompt_is_dropped(self):
        assert collapse_raw_label("banana") is None


# --------------------------------------------------- laptop beats tablet
class TestLaptopBeatsTablet:
    def test_tablet_fully_inside_laptop_is_suppressed(self, cfg):
        laptop = (0, 0, 400, 300)
        tablet = (50, 20, 350, 180)  # the screen half
        assert is_tablet_inside_laptop(tablet, laptop, cfg)

    def test_disjoint_tablet_survives(self, cfg):
        laptop = (0, 0, 400, 300)
        tablet = (600, 0, 800, 300)
        assert not is_tablet_inside_laptop(tablet, laptop, cfg)

    def test_high_iou_tablet_is_suppressed(self, cfg):
        # Model boxed only the screen and called the whole thing a tablet.
        laptop = (0, 0, 400, 300)
        tablet = (5, 5, 395, 295)
        assert is_tablet_inside_laptop(tablet, laptop, cfg)

    def test_suppression_removes_only_the_tablet(self, cfg):
        dets = [
            det(0, 0, 400, 300, label=LAPTOP),
            det(50, 20, 350, 180, label=TABLET),
            det(600, 0, 700, 200, label=TABLET),
        ]
        kept = suppress_tablets_inside_laptops(dets, cfg)
        labels = [d.label for d in kept]
        assert labels.count(LAPTOP) == 1
        assert labels.count(TABLET) == 1
        assert kept[1].bbox_xyxy == (600, 0, 700, 200)

    def test_no_laptop_means_no_suppression(self, cfg):
        dets = [det(50, 20, 350, 180, label=TABLET)]
        assert len(suppress_tablets_inside_laptops(dets, cfg)) == 1


# ------------------------------------------------------ keyboard adjacency
class TestKeyboardAdjacency:
    def test_keyboard_directly_below_and_aligned(self, cfg):
        screen = (100, 50, 400, 250)
        keyboard = (100, 260, 400, 330)
        assert keyboard_is_below(screen, keyboard, cfg)

    def test_keyboard_above_screen_rejected(self, cfg):
        screen = (100, 200, 400, 400)
        keyboard = (100, 100, 400, 170)
        assert not keyboard_is_below(screen, keyboard, cfg)

    def test_keyboard_too_far_below_rejected(self, cfg):
        screen = (100, 50, 400, 250)  # height 200, max gap 0.6*200 = 120
        keyboard = (100, 400, 400, 470)  # gap 150
        assert not keyboard_is_below(screen, keyboard, cfg)

    def test_keyboard_offset_sideways_rejected(self, cfg):
        screen = (100, 50, 400, 250)
        keyboard = (700, 260, 1000, 330)
        assert not keyboard_is_below(screen, keyboard, cfg)

    def test_slight_overlap_still_counts(self, cfg):
        # Real laptops show the keyboard slightly overlapping the screen base.
        screen = (100, 50, 400, 250)
        keyboard = (110, 240, 390, 320)
        assert keyboard_is_below(screen, keyboard, cfg)


# ------------------------------------------------------------ shape prior
class TestShapePrior:
    def test_inside_band_scores_one(self, cfg):
        phone = cfg.device_profiles[PHONE]
        assert shape_prior_score(0.48, phone, cfg.shape_prior_softness) == 1.0

    def test_outside_band_decays(self, cfg):
        phone = cfg.device_profiles[PHONE]
        near = shape_prior_score(0.63, phone, cfg.shape_prior_softness)
        far = shape_prior_score(0.70, phone, cfg.shape_prior_softness)
        assert 0.0 < near < 1.0
        assert far < near

    def test_far_outside_floors_at_zero(self, cfg):
        phone = cfg.device_profiles[PHONE]
        assert shape_prior_score(0.99, phone, cfg.shape_prior_softness) == 0.0

    def test_ipad_aspect_prefers_tablet_over_phone(self, cfg):
        """The real iPad A16 ratio must score tablet above phone."""
        aspect = 174.1 / 255.7  # 0.681
        tablet = shape_prior_score(aspect, cfg.device_profiles[TABLET], cfg.shape_prior_softness)
        phone = shape_prior_score(aspect, cfg.device_profiles[PHONE], cfg.shape_prior_softness)
        assert tablet > phone

    def test_iphone_aspect_prefers_phone_over_tablet(self, cfg):
        aspect = 77.6 / 163.0  # 0.476
        phone = shape_prior_score(aspect, cfg.device_profiles[PHONE], cfg.shape_prior_softness)
        tablet = shape_prior_score(aspect, cfg.device_profiles[TABLET], cfg.shape_prior_softness)
        assert phone > tablet

    def test_degenerate_box_scores_zero(self, cfg):
        assert shape_prior_score(0.0, cfg.device_profiles[PHONE], 0.12) == 0.0


# ------------------------------------------------------------- size prior
class TestSizePrior:
    def test_diagonal_from_angular_size(self):
        # 200x200 px box, focal 500 px, 1 m away -> 0.5657 m diagonal.
        diag = estimate_diagonal_cm((0, 0, 200, 200), focal_px=500.0, depth_m=1.0)
        assert diag == pytest.approx(56.57, abs=0.1)

    def test_diagonal_scales_with_depth(self):
        near = estimate_diagonal_cm((0, 0, 100, 100), 500.0, 0.5)
        far = estimate_diagonal_cm((0, 0, 100, 100), 500.0, 1.0)
        assert far == pytest.approx(2 * near)

    def test_zero_depth_is_safe(self):
        assert estimate_diagonal_cm((0, 0, 100, 100), 500.0, 0.0) == 0.0

    def test_iphone_diagonal_scores_phone_highest(self, cfg):
        diag = 18.05  # iPhone 16 Pro Max body diagonal
        scores = {
            k: size_prior_score(diag, cfg.device_profiles[k], cfg.size_prior_softness_cm)
            for k in (PHONE, TABLET, LAPTOP)
        }
        assert scores[PHONE] > scores[TABLET] >= scores[LAPTOP]

    def test_ipad_diagonal_scores_tablet_highest(self, cfg):
        diag = 30.94  # iPad A16
        scores = {
            k: size_prior_score(diag, cfg.device_profiles[k], cfg.size_prior_softness_cm)
            for k in (PHONE, TABLET, LAPTOP)
        }
        assert scores[TABLET] >= scores[LAPTOP]
        assert scores[TABLET] > scores[PHONE]

    def test_macbook_diagonal_scores_laptop_highest(self, cfg):
        diag = 37.24  # MacBook Air 13" M4 lid
        scores = {
            k: size_prior_score(diag, cfg.device_profiles[k], cfg.size_prior_softness_cm)
            for k in (PHONE, TABLET, LAPTOP)
        }
        assert scores[LAPTOP] == 1.0
        # Both smaller classes are ruled out entirely at this size.
        assert scores[TABLET] == 0.0
        assert scores[PHONE] == 0.0

    def test_size_prior_separates_phone_from_tablet(self, cfg):
        """The signal that shape cannot provide: 18 cm vs 31 cm.

        The two bands nearly touch, so the separation depends on the softness
        being tight. Assert a real margin, not just an ordering.
        """
        phone_diag, tablet_diag = 18.05, 30.94
        p = cfg.device_profiles[PHONE]
        t = cfg.device_profiles[TABLET]
        soft = cfg.size_prior_softness_cm

        assert size_prior_score(phone_diag, p, soft) == 1.0
        assert size_prior_score(tablet_diag, t, soft) == 1.0
        # The wrong class must score well under half.
        assert size_prior_score(phone_diag, t, soft) < 0.4
        assert size_prior_score(tablet_diag, p, soft) == 0.0


class TestSizeVeto:
    """A measurement is a constraint, not an opinion.

    Regression suite for a real failure: an iPad photographed at an angle
    foreshortened to aspect 1.08, landed inside the laptop shape band, made the
    orientation prior abstain, and was called a laptop at 0.61 — even though it
    measured 22 cm, which no laptop can be.
    """

    def test_22cm_object_cannot_be_a_laptop(self, cfg):
        from aria_devices.detect.disambiguate import size_veto

        assert size_veto(21.86, cfg.device_profiles[LAPTOP], cfg.size_veto_margin_cm)

    def test_22cm_object_can_be_a_tablet(self, cfg):
        from aria_devices.detect.disambiguate import size_veto

        assert not size_veto(21.86, cfg.device_profiles[TABLET], cfg.size_veto_margin_cm)

    def test_40cm_object_cannot_be_a_phone_or_tablet(self, cfg):
        from aria_devices.detect.disambiguate import size_veto

        assert size_veto(39.76, cfg.device_profiles[PHONE], cfg.size_veto_margin_cm)
        assert size_veto(39.76, cfg.device_profiles[TABLET], cfg.size_veto_margin_cm)
        assert not size_veto(39.76, cfg.device_profiles[LAPTOP], cfg.size_veto_margin_cm)

    def test_margin_absorbs_depth_error(self, cfg):
        from aria_devices.detect.disambiguate import size_veto

        # Tablet band tops out at 33; 35 is within the 4 cm margin.
        assert not size_veto(35.0, cfg.device_profiles[TABLET], cfg.size_veto_margin_cm)

    def test_no_depth_means_no_veto(self, cfg):
        from aria_devices.detect.disambiguate import size_veto

        assert not size_veto(0.0, cfg.device_profiles[LAPTOP], cfg.size_veto_margin_cm)

    # Geometry taken from the real 2560x1920 desk photo: focal 2309 px
    # (58 deg hFOV), hand-derived depth 0.481 m.
    REAL_FOCAL = 2309.0
    REAL_DEPTH = 0.481

    def test_veto_overrides_a_confident_wrong_text_score(self, cfg):
        """The exact iPad case: foreshortened to aspect 0.925 so it sits inside
        the laptop shape band, called 'laptop' at 0.94, but measuring ~22 cm."""
        laptop_hypothesis = det(1789, 813, 2560, 1526, "laptop", 0.94)
        tablet_hypothesis = det(1795, 820, 2555, 1520, "tablet computer", 0.25)
        out = score_detections(
            [laptop_hypothesis, tablet_hypothesis],
            cfg,
            focal_px=self.REAL_FOCAL,
            depth_fn=lambda d: self.REAL_DEPTH,
        )
        assert len(out) == 1
        assert out[0].label == TABLET
        assert any("size_veto" in n for n in out[0].notes)
        assert out[0].signals["diag_cm"] == pytest.approx(21.9, abs=0.5)

    def test_veto_leaves_a_correct_laptop_alone(self, cfg):
        """The MacBook from the same photo: ~40 cm, must stay a laptop."""
        box = det(0, 459, 1268, 1886, "laptop", 0.94)
        out = score_detections(
            [box], cfg, focal_px=self.REAL_FOCAL, depth_fn=lambda d: self.REAL_DEPTH
        )
        assert len(out) == 1
        assert out[0].label == LAPTOP
        assert out[0].signals["diag_cm"] == pytest.approx(39.8, abs=0.5)

    def test_real_laptop_does_not_trip_the_monitor_threshold(self, cfg):
        """40 cm must stay under monitor_min_diag_cm or the laptop vanishes."""
        assert 39.8 < cfg.monitor_min_diag_cm

    def test_absurd_depth_degrades_gracefully_instead_of_dropping(self, cfg):
        """If every class is vetoed the measurement is untrustworthy; fall back
        to soft scoring rather than deleting the detection."""
        box = det(0, 0, 400, 300, "laptop", 0.9)
        out = score_detections([box], cfg, focal_px=1900.0, depth_fn=lambda d: 0.001)
        assert len(out) == 1
        assert out[0].label in (LAPTOP, TABLET, PHONE)

    def test_veto_can_be_disabled(self, cfg):
        cfg.enable_size_veto = False
        box = det(1789, 813, 2560, 1526, "laptop", 0.94)
        out = score_detections([box], cfg, focal_px=1900.0, depth_fn=lambda d: 0.481)
        assert out[0].label == LAPTOP  # the old, wrong behaviour


# ------------------------------------------------------ orientation prior
class TestOrientationPrior:
    def test_landscape_box_matches_landscape_profile(self, cfg):
        box = (0, 0, 200, 100)  # clearly landscape
        assert orientation_prior_score(box, cfg.device_profiles[PHONE], 0.12, 0.0) == 1.0

    def test_portrait_box_fails_landscape_profile(self, cfg):
        box = (0, 0, 100, 200)
        assert orientation_prior_score(box, cfg.device_profiles[PHONE], 0.12, 0.0) == 0.0

    def test_portrait_box_matches_tablet_profile(self, cfg):
        box = (0, 0, 100, 200)
        assert orientation_prior_score(box, cfg.device_profiles[TABLET], 0.12, 0.0) == 1.0

    def test_near_square_is_neutral_not_penalised(self, cfg):
        """A steeply-angled device foreshortens to square; don't punish it."""
        box = (0, 0, 105, 100)
        assert orientation_prior_score(box, cfg.device_profiles[TABLET], 0.12, 0.0) == 0.5

    def test_any_orientation_always_scores_one(self):
        profile = DeviceProfile("x", (10, 20), (0.4, 0.6), orientation="any")
        assert orientation_prior_score((0, 0, 300, 10), profile, 0.12, 0.0) == 1.0

    def test_orientation_separates_landscape_phone_from_portrait_tablet(self, cfg):
        """The study's setup: phone horizontal, tablet vertical."""
        phone_box = (0, 0, 200, 95)
        tablet_box = (0, 0, 95, 200)
        assert orientation_prior_score(phone_box, cfg.device_profiles[PHONE], 0.12, 0.0) == 1.0
        assert orientation_prior_score(phone_box, cfg.device_profiles[TABLET], 0.12, 0.0) == 0.0
        assert orientation_prior_score(tablet_box, cfg.device_profiles[TABLET], 0.12, 0.0) == 1.0
        assert orientation_prior_score(tablet_box, cfg.device_profiles[PHONE], 0.12, 0.0) == 0.0


# ----------------------------------------------------------- screen prior
class TestScreenPrior:
    def test_bright_box_on_dark_background_scores_high(self):
        img = np.full((200, 200, 3), 20, np.uint8)
        img[60:140, 60:140] = 230
        assert screen_on_score(img, (60, 60, 140, 140), 1.10) > 0.8

    def test_dark_box_on_bright_background_scores_zero(self):
        img = np.full((200, 200, 3), 220, np.uint8)
        img[60:140, 60:140] = 15
        assert screen_on_score(img, (60, 60, 140, 140), 1.10) == 0.0

    def test_uniform_image_is_not_a_screen(self):
        img = np.full((200, 200, 3), 128, np.uint8)
        assert screen_on_score(img, (60, 60, 140, 140), 1.10) == 0.0

    def test_missing_image_is_neutral(self):
        assert screen_on_score(None, (0, 0, 10, 10), 1.10) == 0.5

    def test_degenerate_box_is_neutral(self):
        img = np.zeros((200, 200, 3), np.uint8)
        assert screen_on_score(img, (10, 10, 11, 11), 1.10) == 0.5


# ------------------------------------------------------------- clustering
class TestClustering:
    def test_near_duplicate_boxes_merge(self):
        dets = [det(0, 0, 100, 100, "laptop", 0.9), det(2, 2, 102, 102, "macbook", 0.7)]
        assert len(cluster_detections(dets)) == 1

    def test_distant_boxes_stay_separate(self):
        dets = [det(0, 0, 100, 100), det(500, 500, 600, 600)]
        assert len(cluster_detections(dets)) == 2

    def test_nested_screen_half_stays_separate(self):
        """Critical: the laptop's screen half must NOT merge, so the
        containment rule can fire on it."""
        dets = [det(0, 0, 400, 300, "laptop"), det(50, 20, 350, 180, "tablet computer")]
        assert len(cluster_detections(dets)) == 2


# ----------------------------------------------------------- world-fixed
class TestWorldFixed:
    def test_stationary_track_is_flagged(self, cfg):
        cfg.world_fixed_frames = 5
        tracker = WorldFixedTracker(cfg)
        box = (100, 100, 300, 250)
        for _ in range(5):
            tracker.update(1, box)
        assert tracker.is_world_fixed(1, box)

    def test_moving_track_is_not_flagged(self, cfg):
        cfg.world_fixed_frames = 5
        tracker = WorldFixedTracker(cfg)
        for i in range(5):
            tracker.update(1, (100 + 40 * i, 100, 300 + 40 * i, 250))
        assert not tracker.is_world_fixed(1, (260, 100, 460, 250))

    def test_short_history_is_never_flagged(self, cfg):
        cfg.world_fixed_frames = 30
        tracker = WorldFixedTracker(cfg)
        tracker.update(1, (0, 0, 10, 10))
        assert not tracker.is_world_fixed(1, (0, 0, 10, 10))


# --------------------------------------------------------- full scoring
class TestScoreDetections:
    def test_empty_input_returns_empty(self, cfg):
        assert score_detections([], cfg) == []

    def test_unknown_labels_are_dropped(self, cfg):
        assert score_detections([det(0, 0, 10, 10, "banana")], cfg) == []

    def test_laptop_survives_and_is_labelled(self, cfg):
        out = score_detections([det(0, 0, 400, 280, "laptop", 0.9)], cfg)
        assert len(out) == 1
        assert out[0].label == LAPTOP

    def test_nested_tablet_is_dropped_end_to_end(self, cfg):
        dets = [
            det(0, 0, 400, 300, "laptop", 0.85),
            det(40, 15, 360, 190, "tablet computer", 0.80),
        ]
        out = score_detections(dets, cfg)
        assert [d.label for d in out] == [LAPTOP]

    def test_keyboard_promotes_screen_to_laptop(self, cfg):
        """A portrait-ish screen with a keyboard under it is a laptop."""
        dets = [
            det(100, 50, 400, 250, "tablet computer", 0.55),
            det(100, 260, 400, 330, "computer keyboard", 0.60),
        ]
        out = score_detections(dets, cfg)
        assert len(out) == 1
        assert out[0].label == LAPTOP
        assert "keyboard_below" in out[0].notes

    def test_signals_are_populated_for_tuning(self, cfg):
        out = score_detections([det(0, 0, 400, 280, "laptop", 0.9)], cfg)
        sig = out[0].signals
        for key in ("text", "shape", "size", "orient", "screen", "aspect_portrait"):
            assert key in sig

    def test_low_score_detection_is_dropped(self, cfg):
        cfg.min_final_score = 0.95
        assert score_detections([det(0, 0, 400, 280, "laptop", 0.10)], cfg) == []

    def test_size_prior_flips_tablet_to_phone(self, cfg):
        """A box the detector called a tablet, but measured at 18 cm, is a phone.

        This is the case shape alone cannot resolve, and the reason depth is
        worth the trouble.
        """
        # 200x105 px box, focal 500, 0.45 m -> ~20 cm diagonal... tune to phone
        box = det(0, 0, 200, 96, "tablet computer", 0.55)
        depth = 0.40
        diag = estimate_diagonal_cm(box.bbox_xyxy, 500.0, depth)
        assert 12.0 < diag < 19.5, f"test setup gives {diag:.1f} cm"
        out = score_detections([box], cfg, focal_px=500.0, depth_fn=lambda d: depth)
        assert len(out) == 1
        assert out[0].label == PHONE

    def test_oversized_screen_is_suppressed_as_monitor(self, cfg):
        big = det(0, 0, 800, 500, "tablet computer", 0.7)
        # 800x500 at focal 500, 1.5 m -> ~283 cm; far beyond any laptop.
        out = score_detections([big], cfg, focal_px=500.0, depth_fn=lambda d: 1.5)
        assert out == []

    def test_monitor_text_suppresses_device_without_depth(self, cfg):
        dets = [
            det(0, 0, 500, 300, "computer monitor", 0.85),
            det(0, 0, 500, 300, "tablet computer", 0.40),
        ]
        assert score_detections(dets, cfg) == []

    def test_orientation_separates_the_two_study_devices(self, cfg):
        """Landscape phone and portrait tablet, both called 'tablet' by the model."""
        landscape = det(0, 0, 200, 95, "tablet computer", 0.5)
        portrait = det(600, 0, 695, 200, "tablet computer", 0.5)
        out = score_detections([landscape, portrait], cfg)
        by_box = {d.bbox_xyxy[0]: d.label for d in out}
        assert by_box.get(0.0) == PHONE
        assert by_box.get(600.0) == TABLET
