"""Configuration dataclasses for the aria_devices pipeline.

Everything that a user might reasonably want to tune lives here, and the whole
tree round-trips through YAML so an experiment can be described by one file.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

# --------------------------------------------------------------------------
# Canonical labels
# --------------------------------------------------------------------------
# The three device classes the study cares about, plus hands. Everything else a
# detector emits is either evidence (keyboard) or a distractor to be suppressed
# (monitor, tv).
LAPTOP = "laptop"
TABLET = "tablet"
PHONE = "phone"
HAND = "hand"

CANONICAL_DEVICES = (LAPTOP, TABLET, PHONE)
CANONICAL_LABELS = (LAPTOP, TABLET, PHONE, HAND)

# Non-emitted classes that the disambiguator reasons over.
KEYBOARD = "keyboard"
MONITOR = "monitor"
TV = "tv"
EVIDENCE_LABELS = (KEYBOARD, MONITOR, TV)


# The wide prompt set handed to the open-vocabulary detector. Deliberately
# over-complete: recall first, then collapse to canonical labels in
# detect/disambiguate.py. Order matters only for readability.
DEFAULT_PROMPTS: list[str] = [
    # laptop
    "laptop",
    "open laptop computer",
    "macbook",
    "notebook computer",
    # tablet
    "tablet computer",
    "iPad",
    "tablet held in hand",
    # phone
    "smartphone",
    "mobile phone held in hand",
    "iphone",
    # evidence / distractors
    "computer monitor",
    "television screen",
    "computer keyboard",
    # hands
    "human hand",
    "hand",
]

# Maps every raw prompt onto a canonical or evidence label. A prompt missing
# from this table is dropped, so adding a prompt above without adding it here is
# a no-op rather than a crash.
PROMPT_TO_CANONICAL: dict[str, str] = {
    "laptop": LAPTOP,
    "open laptop computer": LAPTOP,
    "macbook": LAPTOP,
    "notebook computer": LAPTOP,
    "tablet computer": TABLET,
    "ipad": TABLET,
    "tablet held in hand": TABLET,
    "smartphone": PHONE,
    "mobile phone held in hand": PHONE,
    "iphone": PHONE,
    "cell phone": PHONE,
    "computer monitor": MONITOR,
    "television screen": MONITOR,
    "tv": TV,
    "computer keyboard": KEYBOARD,
    "keyboard": KEYBOARD,
    "human hand": HAND,
    "hand": HAND,
}


# --------------------------------------------------------------------------
# Physical device priors
# --------------------------------------------------------------------------
@dataclass
class DeviceProfile:
    """Physical priors for one device class.

    ``diag_cm`` is the *body* diagonal (not the marketing display diagonal),
    because that is what a bounding box actually encloses. ``aspect_portrait``
    is short-edge / long-edge, so it is always <= 1 and orientation-free: a
    landscape observation is scored against 1 / aspect_portrait.

    ``orientation`` is the expected in-image orientation: "landscape",
    "portrait", or "any". Shape and size priors overlap badly between phone and
    tablet, but if the study fixes each device's orientation then orientation
    becomes the cleanest separator available — see ``w_orient``.
    """

    name: str
    diag_cm: tuple[float, float]  # (min, max) plausible body diagonal
    aspect_portrait: tuple[float, float]  # (min, max) short/long edge ratio
    orientation: str = "any"  # "landscape" | "portrait" | "any"
    screen_on: bool = True  # expected to be an emissive, lit screen
    # Fractions of the detected box to trim to reach the *screen* rectangle,
    # as (left, top, right, bottom). The detection box is the whole physical
    # object; a cursor has to land in screen coordinates, and for an open
    # laptop those differ enormously because the box includes the keyboard
    # base. Bezels account for the small trims on tablet and phone.
    screen_inset: tuple[float, float, float, float] = (0.03, 0.03, 0.03, 0.03)
    note: str = ""


# Defaults are centred on the three devices in this study but widened enough to
# cover ordinary consumer hardware, so the pipeline is not useless on someone
# else's desk.
#
#   iPhone 16 Pro Max   163.0 x  77.6 mm  -> diag 18.1 cm, aspect 0.476
#   iPad (A16, 11")     255.7 x 174.1 mm  -> diag 30.9 cm, aspect 0.681
#   MacBook Air 13" M4  304.1 x 215.0 mm  -> lid diag 37.2 cm, aspect 0.707
#   MacBook Air 15" M4  340.4 x 237.6 mm  -> lid diag 41.5 cm, aspect 0.698
#
# Orientations below encode the study's stated setup: phone held/placed
# landscape, tablet stood vertical, MacBook open. Set any of them to "any" to
# fall back on shape and size alone.
DEFAULT_DEVICE_PROFILES: dict[str, DeviceProfile] = {
    PHONE: DeviceProfile(
        name=PHONE,
        diag_cm=(12.0, 19.5),
        aspect_portrait=(0.42, 0.60),
        orientation="landscape",
        # iPhone 16 Pro Max bezels are thin: the screen is ~96% of the body.
        screen_inset=(0.03, 0.03, 0.03, 0.03),
        note="iPhone 16 Pro Max (18.1 cm, 0.476), kept horizontal in this study",
    ),
    TABLET: DeviceProfile(
        name=TABLET,
        diag_cm=(20.0, 33.0),
        aspect_portrait=(0.62, 0.82),
        orientation="portrait",
        # iPad A16 has a visible uniform bezel, roughly 6% per side.
        screen_inset=(0.06, 0.06, 0.06, 0.06),
        note="iPad A16 11-inch (30.9 cm, 0.681), kept vertical in this study",
    ),
    LAPTOP: DeviceProfile(
        name=LAPTOP,
        diag_cm=(30.0, 46.0),
        # An open MacBook Air's lid is 215/304 = 0.707; the visible base pushes
        # the box taller, up towards 0.95. The lower bound must stay clear of
        # the phone band (0.42-0.60) or shape can never separate a laptop from
        # a phone lying on its side, which is exactly this study's setup.
        aspect_portrait=(0.55, 0.95),
        orientation="landscape",
        # The big one. An open laptop's box spans lid *and* keyboard base, so
        # the screen is only the upper part; without this trim a cursor lands
        # roughly a third of the way down from where the hand actually is.
        # 0.38 off the bottom matches a lid open near 105-115 degrees seen from
        # a seated position. Re-measure if the study fixes a different angle.
        screen_inset=(0.04, 0.04, 0.04, 0.38),
        note="MacBook Air 13\"/15\" M4 (37.2 / 41.5 cm), open",
    ),
}


@dataclass
class DisambiguationConfig:
    """Weights and thresholds for detect/disambiguate.py.

    Every rule here is a *soft* score contribution except the two containment
    rules, which are hard suppressions. See the README for measured limits.
    """

    # Final score = w_text*text + w_shape*shape + w_size*size
    #             + w_orient*orientation + w_screen*screen_on
    # Weights are normalised over whichever signals are actually available, so
    # dropping one (e.g. no depth -> no size term) does not shift every score.
    w_text: float = 0.45
    w_shape: float = 0.15
    w_size: float = 0.15
    w_orient: float = 0.15
    w_screen: float = 0.10

    # Orientation prior. With the study's fixed setup (phone landscape, tablet
    # portrait) this is the cleanest phone-vs-tablet separator there is, since
    # the two overlap heavily in both shape and text score.
    enable_orientation_prior: bool = True
    # A box this close to square is treated as orientation-ambiguous and scores
    # neutral rather than being penalised - a device viewed at a steep angle
    # foreshortens toward square.
    orientation_square_band: float = 0.12  # around aspect ratio 1.0
    orientation_mismatch_score: float = 0.0  # score when orientation is wrong

    # Screen-on prior: a powered screen is brighter and lower-texture than its
    # surroundings. Cheap to compute and a decent way to reject picture frames,
    # notebooks and closed laptops.
    enable_screen_prior: bool = True
    screen_min_luma_ratio: float = 1.10  # interior mean luma / border mean luma

    # A tablet box this contained inside a laptop box is the laptop's screen
    # half, not a separate device. Containment = intersection / tablet_area.
    laptop_over_tablet_containment: float = 0.70
    # Straight IoU also triggers the same suppression.
    laptop_over_tablet_iou: float = 0.55

    # Keyboard adjacency: a screen-like box with a keyboard directly below and
    # horizontally aligned is a laptop.
    keyboard_max_vgap_ratio: float = 0.60  # gap / screen height
    keyboard_min_x_overlap: float = 0.50  # fraction of narrower box
    keyboard_promotion_bonus: float = 0.35

    # Monitor / TV suppression.
    monitor_min_diag_cm: float = 46.0  # above this it is furniture, not a device
    # A box that barely moves in the image over this many frames while the
    # wearer's head does move is world-fixed -> a monitor, not a handheld.
    # World-fixed suppression exists to kill wall monitors and TVs. It is OFF
    # by default because the primary setup here is the opposite case: three
    # devices sitting still on a desk. A static tablet viewed by a seated
    # wearer trips this rule within ~1.5 s and the device is deleted outright.
    # Physical size (monitor_min_diag_cm) already rejects real monitors, and it
    # does so without punishing a device for staying put.
    suppress_world_fixed: bool = False
    world_fixed_frames: int = 45
    world_fixed_max_iou_drift: float = 0.10

    # Angular-size prior. Disabled unless a depth estimate is supplied.
    enable_size_prior: bool = True
    # Score falls off over this many cm outside a profile's [min, max] band.
    # The phone and tablet bands nearly touch (19.5 / 20.0 cm), so a generous
    # softness here would hand an 18 cm phone most of the tablet's score and
    # throw away the one signal that reliably separates them.
    size_prior_softness_cm: float = 3.0

    # Hard size veto. When we have a real depth measurement, a diagonal far
    # outside a class's band is not weak evidence — it is a physical
    # impossibility, and letting a confident text score override it is wrong.
    # A 22 cm object is not a laptop no matter what the detector calls it.
    # The margin absorbs depth error; the veto never fires when it would
    # eliminate every class, so a bad depth estimate degrades to soft scoring
    # rather than dropping the detection.
    enable_size_veto: bool = True
    size_veto_margin_cm: float = 4.0

    # Shape prior falloff outside the aspect band, in ratio units.
    shape_prior_softness: float = 0.12

    # Detections below this final score are dropped.
    min_final_score: float = 0.28

    device_profiles: dict[str, DeviceProfile] = field(
        default_factory=lambda: {k: dataclasses.replace(v) for k, v in DEFAULT_DEVICE_PROFILES.items()}
    )


@dataclass
class DetectorConfig:
    """Which open-vocabulary backend to run, and how."""

    # "yoloworld" | "yoloe" | "owlv2"
    backend: str = "yoloworld"
    weights: str = "yolov8s-worldv2.pt"
    prompts: list[str] = field(default_factory=lambda: list(DEFAULT_PROMPTS))

    conf_threshold: float = 0.05  # deliberately low; disambiguation does the filtering
    iou_threshold: float = 0.50
    imgsz: int = 640
    max_det: int = 60

    # "auto" resolves to cuda -> mps -> cpu.
    device: str = "auto"
    half: bool = False  # fp16; only meaningful on cuda


@dataclass
class HandConfig:
    """Hand detection.

    On Aria the on-device ``handtracking`` stream is authoritative and free, so
    it is preferred whenever the stream exists. Off Aria we fall back to
    MediaPipe, and failing that to the open-vocab detector's "hand" prompt.
    """

    # "auto" | "aria" | "mediapipe" | "openvocab" | "off"
    backend: str = "auto"
    max_hands: int = 2
    min_confidence: float = 0.40
    # Downloaded on first use if absent.
    mediapipe_model_path: str = "assets/hand_landmarker.task"
    draw_landmarks: bool = True

    # Grab/open classification. `openness` is mean fingertip reach divided by
    # the wrist-to-knuckle span, remapped onto [0, 1] over
    # [1.0, 1.0 + openness_span].
    #
    # openness_span was calibrated against a real open hand reaching over a
    # laptop at an oblique angle, which measured a reach of 1.55 — far short of
    # the ~2.2 a fully splayed hand gives when it faces the camera square on.
    # Foreshortening is the norm in egocentric video, not the exception, so the
    # span is set for that case. Widen it if you see open hands read as grabs.
    openness_span: float = 0.8
    grab_below: float = 0.35
    open_above: float = 0.55


@dataclass
class TrackConfig:
    """ByteTrack-style association + temporal label voting."""

    enabled: bool = True
    high_thresh: float = 0.45  # first association pass
    low_thresh: float = 0.10  # second (recovery) pass
    match_iou: float = 0.30
    max_age: int = 30  # frames a track survives unmatched
    min_hits: int = 3  # frames before a track is emitted

    vote_window: int = 15  # rolling label histogram length
    vote_min_fraction: float = 0.34  # majority must clear this to switch label

    # Frames a confirmed device survives without being re-detected. Devices on
    # a desk do not teleport; a hand reaching across one should not delete it.
    persist_frames: int = 10


@dataclass
class RectifyConfig:
    """Fisheye -> pinhole rectification for the Aria RGB stream.

    ``focal`` is the pinhole focal length in pixels at ``size``. Lower focal =
    wider field of view kept, at the cost of severe stretching at the edges of
    a 110-degree-plus fisheye. 350 px at 704x704 keeps roughly a 90-degree
    horizontal FOV with tolerable corner distortion; see README.
    """

    enabled: bool = True
    size: tuple[int, int] = (704, 704)  # (width, height) of the rectified image
    focal: float = 350.0

    # Aria RGB arrives rotated. "ccw90" is what the live Gen 2 stream actually
    # needs for an upright view — verified against the glasses, and contrary to
    # the clockwise rotation the original brief specified. This is not cosmetic:
    # every geometric prior downstream (especially the landscape-vs-portrait
    # orientation prior) reads a sideways image as the wrong device.
    # One of: "ccw90" | "cw90" | "none".
    rotation: str = "ccw90"
    devignette: bool = False  # needs set_devignetting_mask_folder_path
    devignetting_mask_path: str = ""
    color_correct: bool = True


@dataclass
class VizConfig:
    draw_gaze: bool = True
    draw_hands: bool = True
    draw_signals: bool = False  # per-signal score breakdown on the frame
    box_thickness: int = 2
    font_scale: float = 0.5


@dataclass
class PipelineConfig:
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    disambiguation: DisambiguationConfig = field(default_factory=DisambiguationConfig)
    hands: HandConfig = field(default_factory=HandConfig)
    track: TrackConfig = field(default_factory=TrackConfig)
    rectify: RectifyConfig = field(default_factory=RectifyConfig)
    viz: VizConfig = field(default_factory=VizConfig)

    # Gaze
    gaze_hit_radius_px: float = 90.0

    # Gaze gating: a device only reacts to hands while the wearer is looking at
    # it. This is what stops a hand crossing over the laptop on its way to the
    # tablet from firing events on the laptop — with three devices 20 cm apart,
    # that crossing happens on almost every reach.
    require_gaze: bool = True
    # Grace period after gaze leaves. Eyes saccade constantly and during a reach
    # gaze often flicks to the moving hand, so an instantaneous test would drop
    # the interaction midway through the very gesture it exists to capture.
    gaze_grace_ms: float = 800.0
    gaze_depth_m: float = 1.0  # depth at which the gaze ray is projected

    # Output
    write_mp4: bool = True
    write_jsonl: bool = True
    stride: int = 1
    max_frames: int = 0  # 0 = no limit


# --------------------------------------------------------------------------
# YAML round-trip
# --------------------------------------------------------------------------
def _to_plain(obj: Any) -> Any:
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {f.name: _to_plain(getattr(obj, f.name)) for f in dataclasses.fields(obj)}
    if isinstance(obj, dict):
        return {k: _to_plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_plain(v) for v in obj]
    return obj


def _from_plain(cls: type, data: Any) -> Any:
    """Rebuild a dataclass tree from plain YAML types.

    Unknown keys are ignored rather than fatal, so a config written by a newer
    version still loads.
    """
    if not dataclasses.is_dataclass(cls):
        return data
    kwargs: dict[str, Any] = {}
    hints = {f.name: f for f in dataclasses.fields(cls)}
    for key, value in (data or {}).items():
        f = hints.get(key)
        if f is None:
            continue
        ftype = f.type
        if ftype is DeviceProfile or ftype == "DeviceProfile":
            kwargs[key] = DeviceProfile(**value)
        elif key == "device_profiles" and isinstance(value, dict):
            kwargs[key] = {
                k: DeviceProfile(
                    name=v.get("name", k),
                    diag_cm=tuple(v["diag_cm"]),
                    aspect_portrait=tuple(v["aspect_portrait"]),
                    orientation=v.get("orientation", "any"),
                    screen_on=v.get("screen_on", True),
                    screen_inset=tuple(v.get("screen_inset", (0.03, 0.03, 0.03, 0.03))),
                    note=v.get("note", ""),
                )
                for k, v in value.items()
            }
        elif dataclasses.is_dataclass(ftype) and isinstance(value, dict):
            kwargs[key] = _from_plain(ftype, value)  # type: ignore[arg-type]
        elif key == "size" and isinstance(value, list):
            kwargs[key] = tuple(value)
        else:
            kwargs[key] = value
    return cls(**kwargs)


_NESTED = {
    "detector": DetectorConfig,
    "disambiguation": DisambiguationConfig,
    "hands": HandConfig,
    "track": TrackConfig,
    "rectify": RectifyConfig,
    "viz": VizConfig,
}


def load_config(path: str | Path) -> PipelineConfig:
    """Load a PipelineConfig from YAML. Missing sections keep their defaults."""
    raw = yaml.safe_load(Path(path).read_text()) or {}
    cfg = PipelineConfig()
    for key, value in raw.items():
        if key in _NESTED and isinstance(value, dict):
            setattr(cfg, key, _from_plain(_NESTED[key], value))
        elif hasattr(cfg, key):
            setattr(cfg, key, value)
    return cfg


def save_config(cfg: PipelineConfig, path: str | Path) -> None:
    Path(path).write_text(yaml.safe_dump(_to_plain(cfg), sort_keys=False))


def resolve_device(requested: str = "auto") -> str:
    """Resolve "auto" to the best available torch device.

    Apple silicon gets "mps", which is a real speedup over cpu for these models
    and is the relevant fast path on this machine — the spec only mentioned
    CUDA.
    """
    if requested != "auto":
        return requested
    try:
        import torch
    except ImportError:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"
