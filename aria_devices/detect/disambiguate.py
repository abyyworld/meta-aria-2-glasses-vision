"""Collapse noisy open-vocabulary labels into laptop / tablet / phone.

The open-vocab model is good at "there is a rectangular screen here" and bad at
which of the three it is. Everything in this module exists to fix that with
explicit, inspectable rules rather than a black box.

All the scoring helpers are pure functions over plain boxes so they can be unit
tested with synthetic data and no model weights.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Callable, Sequence

from ..config import (
    CANONICAL_DEVICES,
    HAND,
    KEYBOARD,
    LAPTOP,
    MONITOR,
    PHONE,
    PROMPT_TO_CANONICAL,
    TABLET,
    TV,
    DeviceProfile,
    DisambiguationConfig,
)
from .base import Detection, containment, iou

log = logging.getLogger(__name__)

Box = tuple[float, float, float, float]
DepthFn = Callable[[Detection], float | None]

#: Boxes merged into one physical object above this IoU. Deliberately high:
#: nesting (a laptop's screen half detected as a tablet) must survive as two
#: separate objects so the containment rule can fire on it.
CLUSTER_IOU = 0.65


# --------------------------------------------------------------------------
# Label collapsing
# --------------------------------------------------------------------------
def collapse_raw_label(raw_label: str) -> str | None:
    """Map a raw prompt onto a canonical or evidence label.

    Returns None for prompts we have no mapping for, which are then dropped.
    Matching is case-insensitive and tolerant of a trailing/leading article.
    """
    key = raw_label.strip().lower()
    if key in PROMPT_TO_CANONICAL:
        return PROMPT_TO_CANONICAL[key]
    for prefix in ("a ", "an ", "the "):
        if key.startswith(prefix) and key[len(prefix) :] in PROMPT_TO_CANONICAL:
            return PROMPT_TO_CANONICAL[key[len(prefix) :]]
    return None


# --------------------------------------------------------------------------
# Shape prior
# --------------------------------------------------------------------------
def orientation_prior_score(
    box: Box, profile: DeviceProfile, square_band: float, mismatch_score: float
) -> float:
    """Score a box's in-image orientation against the profile's expected one.

    When the study fixes each device's orientation — phone laid horizontal,
    tablet stood vertical — this becomes the strongest phone-vs-tablet signal
    available, because those two classes are otherwise nearly identical in both
    shape and text score.

    Boxes close to square score a neutral 0.5 rather than being punished: a
    device viewed at a steep angle foreshortens toward square, and we must not
    penalise it for the wearer's head position.
    """
    expected = (profile.orientation or "any").lower()
    if expected == "any":
        return 1.0
    w = max(0.0, box[2] - box[0])
    h = max(0.0, box[3] - box[1])
    if w <= 0 or h <= 0:
        return 0.5
    ratio = w / h
    if abs(ratio - 1.0) <= square_band:
        return 0.5  # ambiguous, stay neutral
    observed = "landscape" if ratio > 1.0 else "portrait"
    return 1.0 if observed == expected else mismatch_score


def screen_on_score(
    image_rgb, box: Box, min_luma_ratio: float
) -> float:
    """How much the box looks like a powered-on screen, in [0, 1].

    A lit screen is brighter than the surface around it. We compare the mean
    luma inside the box against a ring just outside it; the ratio is mapped
    onto [0, 1] with ``min_luma_ratio`` as the break-even point.

    Honest limits: this fails on a bright screen against a bright background
    (a window behind the desk), and it cannot tell a screen from a lamp. It is
    weighted low for exactly that reason — it breaks ties, it does not decide.
    """
    if image_rgb is None:
        return 0.5
    h_img, w_img = image_rgb.shape[:2]
    x1 = max(0, int(box[0]))
    y1 = max(0, int(box[1]))
    x2 = min(w_img, int(box[2]))
    y2 = min(h_img, int(box[3]))
    if x2 - x1 < 4 or y2 - y1 < 4:
        return 0.5

    inner = image_rgb[y1:y2, x1:x2]
    inner_luma = float(inner.mean())

    # Ring of the same thickness around the box.
    pad_x = max(4, (x2 - x1) // 5)
    pad_y = max(4, (y2 - y1) // 5)
    ox1 = max(0, x1 - pad_x)
    oy1 = max(0, y1 - pad_y)
    ox2 = min(w_img, x2 + pad_x)
    oy2 = min(h_img, y2 + pad_y)
    outer = image_rgb[oy1:oy2, ox1:ox2]
    outer_sum = float(outer.sum())
    outer_n = outer.size
    inner_n = inner.size
    ring_n = outer_n - inner_n
    if ring_n <= 0:
        return 0.5
    ring_luma = (outer_sum - float(inner.sum())) / ring_n
    if ring_luma <= 1e-6:
        return 1.0 if inner_luma > 1.0 else 0.5

    ratio = inner_luma / ring_luma
    if ratio <= 1.0:
        return 0.0
    if ratio >= min_luma_ratio * 1.6:
        return 1.0
    # Linear ramp from parity to comfortably-brighter.
    span = max(1e-6, min_luma_ratio * 1.6 - 1.0)
    return max(0.0, min(1.0, (ratio - 1.0) / span))


def shape_prior_score(aspect_portrait: float, profile: DeviceProfile, softness: float) -> float:
    """Score a box's aspect ratio against a device profile, in [0, 1].

    ``aspect_portrait`` is short/long edge, so this is orientation-free: a phone
    lying sideways scores the same as one held upright. Inside the profile band
    the score is 1.0; outside it decays linearly over ``softness`` ratio units
    and floors at 0.

    This is deliberately a soft signal. Perspective foreshortening at oblique
    viewing angles distorts aspect badly — a tablet tilted 60 degrees away reads
    as a phone — so it must never be able to veto on its own.
    """
    if aspect_portrait <= 0.0:
        return 0.0
    lo, hi = profile.aspect_portrait
    if lo <= aspect_portrait <= hi:
        return 1.0
    dist = (lo - aspect_portrait) if aspect_portrait < lo else (aspect_portrait - hi)
    if softness <= 0.0:
        return 0.0
    return max(0.0, 1.0 - dist / softness)


# --------------------------------------------------------------------------
# Size prior
# --------------------------------------------------------------------------
def estimate_diagonal_cm(
    box: Box, focal_px: float, depth_m: float
) -> float:
    """Physical diagonal of a box, from its angular size and a distance.

    Pinhole similar triangles on the *rectified* image: a box whose diagonal is
    ``d_px`` at distance ``z`` subtends a physical diagonal of ``d_px * z / f``.
    Only valid once the fisheye has been rectified, which is why rectification
    is not optional on the Aria path.
    """
    if focal_px <= 0.0 or depth_m <= 0.0:
        return 0.0
    w = max(0.0, box[2] - box[0])
    h = max(0.0, box[3] - box[1])
    d_px = math.hypot(w, h)
    return (d_px * depth_m / focal_px) * 100.0


def size_veto(diag_cm: float, profile: DeviceProfile, margin_cm: float) -> bool:
    """True when a measured diagonal rules this class out entirely.

    Distinct from ``size_prior_score`` on purpose. The prior is an opinion; this
    is a constraint. Once depth is measured, a 22 cm object simply cannot be a
    laptop, and no amount of text confidence should be able to say otherwise —
    which is exactly the failure this exists to stop.
    """
    if diag_cm <= 0:
        return False
    lo, hi = profile.diag_cm
    return diag_cm < (lo - margin_cm) or diag_cm > (hi + margin_cm)


def size_prior_score(diag_cm: float, profile: DeviceProfile, softness_cm: float) -> float:
    """Score a physical diagonal against a device profile, in [0, 1].

    This is the single most discriminative phone-vs-tablet signal, because the
    two are near-identical in shape and differ almost entirely in size. It is
    also the one most likely to be unavailable, since it needs depth.
    """
    if diag_cm <= 0.0:
        return 0.0
    lo, hi = profile.diag_cm
    if lo <= diag_cm <= hi:
        return 1.0
    dist = (lo - diag_cm) if diag_cm < lo else (diag_cm - hi)
    if softness_cm <= 0.0:
        return 0.0
    return max(0.0, 1.0 - dist / softness_cm)


# --------------------------------------------------------------------------
# Structural rules
# --------------------------------------------------------------------------
def is_tablet_inside_laptop(tablet: Box, laptop: Box, cfg: DisambiguationConfig) -> bool:
    """True when a tablet box is really a laptop's screen half.

    Fires on either high containment (the usual case — the screen box sits
    wholly inside the laptop box) or high IoU (the model boxed only the screen
    and called the whole thing a tablet).
    """
    if containment(tablet, laptop) >= cfg.laptop_over_tablet_containment:
        return True
    return iou(tablet, laptop) >= cfg.laptop_over_tablet_iou


def keyboard_within_lower(laptop: Box, keyboard: Box, min_overlap: float = 0.6) -> bool:
    """True when a keyboard sits inside the lower part of a laptop box.

    Distinct from ``keyboard_is_below``, which handles a screen-only box with a
    separate keyboard beneath it. An *open* laptop is usually detected as a
    single box spanning lid and base, so its keyboard is contained rather than
    adjacent — and the adjacency test rejects exactly that case.
    """
    l_x1, l_y1, l_x2, l_y2 = laptop
    k_x1, k_y1, k_x2, k_y2 = keyboard
    l_h = l_y2 - l_y1
    k_area = max(1e-6, (k_x2 - k_x1) * (k_y2 - k_y1))
    if l_h <= 0:
        return False

    ox = max(0.0, min(l_x2, k_x2) - max(l_x1, k_x1))
    oy = max(0.0, min(l_y2, k_y2) - max(l_y1, k_y1))
    if (ox * oy) / k_area < min_overlap:
        return False  # keyboard is mostly outside this laptop
    # Its top must fall in the lower half of the box: that is the lid/base seam.
    return (k_y1 - l_y1) / l_h >= 0.35


def keyboard_is_below(screen: Box, keyboard: Box, cfg: DisambiguationConfig) -> bool:
    """True when `keyboard` sits directly below `screen` and lines up in x.

    A screen with a keyboard attached below it is a laptop, not a tablet and not
    a monitor. Note the honest failure case: an external keyboard in front of a
    desktop monitor produces exactly this geometry, so this rule promotes some
    monitors to laptops. The size prior is what pulls those back.
    """
    s_x1, s_y1, s_x2, s_y2 = screen
    k_x1, k_y1, k_x2, k_y2 = keyboard
    s_h = s_y2 - s_y1
    if s_h <= 0:
        return False

    # Keyboard's top must be at or below the screen's bottom, within a gap
    # proportional to screen height.
    gap = k_y1 - s_y2
    if gap < -0.15 * s_h:  # allow slight overlap, reject a keyboard above
        return False
    if gap > cfg.keyboard_max_vgap_ratio * s_h:
        return False

    # Horizontal alignment: overlap must cover most of the narrower box.
    overlap = min(s_x2, k_x2) - max(s_x1, k_x1)
    if overlap <= 0:
        return False
    narrower = min(s_x2 - s_x1, k_x2 - k_x1)
    if narrower <= 0:
        return False
    return (overlap / narrower) >= cfg.keyboard_min_x_overlap


# --------------------------------------------------------------------------
# World-fixed (monitor / TV) detection
# --------------------------------------------------------------------------
class WorldFixedTracker:
    """Flags tracks that stay put in the world — i.e. monitors and TVs.

    Caveat, and it is a real one: under egocentric head motion a world-fixed
    monitor *does* move in the image, so image-space stillness alone is a weak
    signal. When the caller supplies a per-frame ego-rotation delta (from VIO)
    we compensate for it and the signal becomes meaningful. Without VIO this
    only catches the case where the wearer's head is also still, and the size
    threshold in ``score_detections`` does the real work.
    """

    def __init__(self, cfg: DisambiguationConfig) -> None:
        self.cfg = cfg
        self._history: dict[int, list[tuple[float, float]]] = {}

    def update(
        self,
        track_id: int,
        box: Box,
        ego_shift_px: tuple[float, float] = (0.0, 0.0),
    ) -> None:
        cx = 0.5 * (box[0] + box[2]) - ego_shift_px[0]
        cy = 0.5 * (box[1] + box[3]) - ego_shift_px[1]
        hist = self._history.setdefault(track_id, [])
        hist.append((cx, cy))
        if len(hist) > self.cfg.world_fixed_frames:
            del hist[0 : len(hist) - self.cfg.world_fixed_frames]

    def is_world_fixed(self, track_id: int, box: Box) -> bool:
        hist = self._history.get(track_id)
        if not hist or len(hist) < self.cfg.world_fixed_frames:
            return False
        diag = math.hypot(box[2] - box[0], box[3] - box[1])
        if diag <= 0:
            return False
        xs = [p[0] for p in hist]
        ys = [p[1] for p in hist]
        drift = math.hypot(max(xs) - min(xs), max(ys) - min(ys)) / diag
        return drift <= self.cfg.world_fixed_max_iou_drift

    def forget(self, track_id: int) -> None:
        self._history.pop(track_id, None)


# --------------------------------------------------------------------------
# Clustering
# --------------------------------------------------------------------------
def cluster_detections(dets: Sequence[Detection], cluster_iou: float = CLUSTER_IOU) -> list[list[int]]:
    """Group near-duplicate boxes (same object, different prompt) by IoU.

    Greedy single-link over a score-sorted list. Good enough at these box counts
    and, unlike NMS, it keeps every member so per-class text scores survive.
    """
    order = sorted(range(len(dets)), key=lambda i: -dets[i].score)
    clusters: list[list[int]] = []
    for idx in order:
        placed = False
        for cluster in clusters:
            if any(iou(dets[idx].bbox_xyxy, dets[m].bbox_xyxy) >= cluster_iou for m in cluster):
                cluster.append(idx)
                placed = True
                break
        if not placed:
            clusters.append([idx])
    return clusters


# --------------------------------------------------------------------------
# Main entry point
# --------------------------------------------------------------------------
def score_detections(
    dets: Sequence[Detection],
    cfg: DisambiguationConfig,
    focal_px: float | None = None,
    depth_fn: DepthFn | None = None,
    image_rgb=None,
) -> list[Detection]:
    """Turn raw open-vocab detections into canonically labelled ones.

    Pipeline:
      1. collapse raw prompts to canonical / evidence labels
      2. cluster near-duplicate boxes into one object each
      3. score every object against all three device hypotheses
      4. apply keyboard adjacency, then size-based monitor suppression
      5. argmax, threshold, then drop tablets nested in laptops

    ``depth_fn`` returns metres for a detection, or None when unknown. When it
    is absent the size term is dropped and its weight is redistributed across
    the text and shape terms, so the final score stays comparable to
    ``min_final_score`` either way.
    """
    if not dets:
        return []

    # --- 1. collapse -------------------------------------------------------
    collapsed: list[tuple[Detection, str]] = []
    for det in dets:
        canon = collapse_raw_label(det.raw_label)
        if canon is None:
            continue
        collapsed.append((det, canon))
    if not collapsed:
        return []

    device_items = [(d, c) for d, c in collapsed if c in CANONICAL_DEVICES]
    keyboards = [d for d, c in collapsed if c == KEYBOARD]
    screens_other = [d for d, c in collapsed if c in (MONITOR, TV)]
    hands = [d for d, c in collapsed if c == HAND]

    # Monitor/TV boxes are not devices, but they are evidence: a big screen
    # detected here is a strong hint that an overlapping "tablet" is the same
    # object and should be suppressed.
    candidates = [d for d, _ in device_items] + screens_other
    canon_of = {id(d): c for d, c in collapsed}

    # --- 2. cluster --------------------------------------------------------
    clusters = cluster_detections(candidates)

    use_size = cfg.enable_size_prior and depth_fn is not None and focal_px is not None
    use_orient = cfg.enable_orientation_prior
    use_screen = cfg.enable_screen_prior and image_rgb is not None

    # Normalise over the signals we actually have, so a missing depth estimate
    # lowers nobody's score relative to min_final_score.
    weights = {
        "text": cfg.w_text,
        "shape": cfg.w_shape,
        "size": cfg.w_size if use_size else 0.0,
        "orient": cfg.w_orient if use_orient else 0.0,
        "screen": cfg.w_screen if use_screen else 0.0,
    }
    total_w = sum(weights.values())
    if total_w > 0:
        weights = {k: v / total_w for k, v in weights.items()}
    w_text = weights["text"]
    w_shape = weights["shape"]
    w_size = weights["size"]
    w_orient = weights["orient"]
    w_screen = weights["screen"]

    results: list[Detection] = []

    for cluster in clusters:
        members = [candidates[i] for i in cluster]
        rep = max(members, key=lambda d: d.score)  # representative box

        # --- 3. per-hypothesis scoring -------------------------------------
        text: dict[str, float] = {c: 0.0 for c in CANONICAL_DEVICES}
        non_device_text = 0.0
        for m in members:
            c = canon_of[id(m)]
            if c in CANONICAL_DEVICES:
                text[c] = max(text[c], m.score)
            else:
                non_device_text = max(non_device_text, m.score)

        depth_m = depth_fn(rep) if use_size else None
        diag_cm = (
            estimate_diagonal_cm(rep.bbox_xyxy, float(focal_px), depth_m)
            if (use_size and depth_m)
            else 0.0
        )

        aspect = rep.aspect_portrait
        screen = (
            screen_on_score(image_rgb, rep.bbox_xyxy, cfg.screen_min_luma_ratio)
            if use_screen
            else 0.0
        )

        shape: dict[str, float] = {}
        size: dict[str, float] = {}
        orient: dict[str, float] = {}
        for c in CANONICAL_DEVICES:
            profile = cfg.device_profiles[c]
            shape[c] = shape_prior_score(aspect, profile, cfg.shape_prior_softness)
            size[c] = (
                size_prior_score(diag_cm, profile, cfg.size_prior_softness_cm) if diag_cm > 0 else 0.0
            )
            orient[c] = (
                orientation_prior_score(
                    rep.bbox_xyxy,
                    profile,
                    cfg.orientation_square_band,
                    cfg.orientation_mismatch_score,
                )
                if use_orient
                else 0.0
            )

        # --- 4a. keyboard adjacency ----------------------------------------
        keyboard_bonus = 0.0
        matched_keyboard = False
        keyboard_box: Box | None = None
        for kb in keyboards:
            below = keyboard_is_below(rep.bbox_xyxy, kb.bbox_xyxy, cfg)
            within = keyboard_within_lower(rep.bbox_xyxy, kb.bbox_xyxy)
            if below or within:
                matched_keyboard = True
                keyboard_bonus = cfg.keyboard_promotion_bonus
                keyboard_box = kb.bbox_xyxy
                break

        final: dict[str, float] = {}
        for c in CANONICAL_DEVICES:
            s = (
                w_text * text[c]
                + w_shape * shape[c]
                + w_size * size[c]
                + w_orient * orient[c]
                + w_screen * screen
            )
            if c == LAPTOP and matched_keyboard:
                s += keyboard_bonus
            final[c] = s

        # Hard size veto: eliminate classes the measurement rules out, but
        # only while at least one class survives — a wild depth estimate must
        # degrade to soft scoring, not delete the detection.
        allowed = list(CANONICAL_DEVICES)
        vetoed: list[str] = []
        if use_size and cfg.enable_size_veto and diag_cm > 0:
            vetoed = [
                c
                for c in CANONICAL_DEVICES
                if size_veto(diag_cm, cfg.device_profiles[c], cfg.size_veto_margin_cm)
            ]
            survivors = [c for c in CANONICAL_DEVICES if c not in vetoed]
            if survivors:
                allowed = survivors
            else:
                vetoed = []

        best = max(allowed, key=lambda c: final[c])
        best_score = final[best]

        det = Detection(
            bbox_xyxy=rep.bbox_xyxy,
            score=float(min(1.0, best_score)),
            raw_label=rep.raw_label,
            label=best,
        )
        det.signals = {
            "text": round(text[best], 4),
            "shape": round(shape[best], 4),
            "size": round(size[best], 4),
            "orient": round(orient[best], 4),
            "screen": round(screen, 4),
            "keyboard_bonus": round(keyboard_bonus, 4),
            "diag_cm": round(diag_cm, 2),
            "aspect_portrait": round(aspect, 4),
            "depth_m": round(depth_m, 3) if depth_m else 0.0,
            **{f"final_{c}": round(final[c], 4) for c in CANONICAL_DEVICES},
        }
        if matched_keyboard:
            det.notes.append("keyboard_below")

        # Measure the laptop screen instead of assuming a lid angle. The
        # keyboard's top edge is the bottom of the screen, so this adapts to
        # however far the lid happens to be open — the fixed 0.38 inset is only
        # correct near 105-115 degrees and drifts 5-10% of screen height either
        # side of that.
        if best == LAPTOP and keyboard_box is not None:
            x1, y1, x2, _y2 = det.bbox_xyxy
            w = x2 - x1
            screen_bottom = keyboard_box[1]
            if screen_bottom > y1 + 0.25 * (det.bbox_xyxy[3] - y1):
                det.screen_box = (x1 + 0.04 * w, y1 + 0.04 * w, x2 - 0.04 * w, screen_bottom)
                det.notes.append("screen_from_keyboard")

        if image_rgb is not None:
            h, w = image_rgb.shape[:2]
            x1, y1, x2, y2 = det.bbox_xyxy
            det.clipped = x1 <= 1.0 or y1 <= 1.0 or x2 >= w - 1.0 or y2 >= h - 1.0
            if det.clipped:
                det.notes.append("clipped_at_frame_edge")
        if vetoed:
            det.notes.append(f"size_veto({diag_cm:.0f}cm rules out {'+'.join(vetoed)})")
            log.debug("size veto at %.1f cm eliminated %s", diag_cm, vetoed)

        # --- 4b. monitor / TV suppression ----------------------------------
        # A screen bigger than any laptop is furniture. Only trust this when we
        # actually measured it: without depth, diag_cm is 0 and we skip.
        if diag_cm > cfg.monitor_min_diag_cm:
            det.notes.append(f"suppressed_large_screen({diag_cm:.0f}cm)")
            log.debug("drop %s: %.0f cm exceeds monitor threshold", best, diag_cm)
            continue
        # No depth available: fall back on the model saying "monitor"/"tv" more
        # confidently than it says any device class.
        if not use_size and non_device_text > max(text.values()) + 0.10:
            det.notes.append("suppressed_monitor_text")
            log.debug("drop: monitor/tv text %.2f beats device text", non_device_text)
            continue

        if best_score < cfg.min_final_score:
            log.debug("drop %s: score %.3f < %.3f", best, best_score, cfg.min_final_score)
            continue

        log.debug(
            "keep %s score=%.3f signals=%s", det.label, det.score, det.signals
        )
        results.append(det)

    # --- 5. laptop beats tablet on nesting --------------------------------
    results = suppress_tablets_inside_laptops(results, cfg)

    # Hands ride along untouched — they are not part of the device hypothesis
    # space and must not be suppressed by device rules.
    for h in hands:
        results.append(
            Detection(bbox_xyxy=h.bbox_xyxy, score=h.score, raw_label=h.raw_label, label=HAND)
        )

    return results


def suppress_tablets_inside_laptops(
    dets: Sequence[Detection], cfg: DisambiguationConfig
) -> list[Detection]:
    """Drop tablet boxes that are really the screen half of a detected laptop.

    Laptop wins on overlap, always. This is the single highest-yield rule: an
    open MacBook's lid is a near-perfect tablet as far as the detector is
    concerned.
    """
    laptops = [d for d in dets if d.label == LAPTOP]
    if not laptops:
        return list(dets)

    kept: list[Detection] = []
    for det in dets:
        if det.label != TABLET:
            kept.append(det)
            continue
        swallowed = next(
            (lp for lp in laptops if is_tablet_inside_laptop(det.bbox_xyxy, lp.bbox_xyxy, cfg)),
            None,
        )
        if swallowed is not None:
            log.debug(
                "drop tablet %s: contained in laptop %s", det.bbox_xyxy, swallowed.bbox_xyxy
            )
            continue
        kept.append(det)
    return kept
