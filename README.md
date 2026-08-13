# aria_devices

Detects and labels **laptops, tablets, phones and hands** in Project Aria Gen 2
egocentric video — from recorded `.vrs` files, from a live stream off the
glasses, or from any webcam/MP4 for testing without hardware.

Built around three specific devices (iPhone 16 Pro Max, iPad A16 11-inch,
MacBook Air M4) but the detector is open-vocabulary, so it generalises to other
hardware; only the physical priors are tuned to these three.

---

## Quick start

Everything lives in a virtualenv, so **run the commands from the project
directory** using the `./aria-devices` launcher — it finds the venv itself and
needs no activation:

```bash
cd /path/to/meta-aria-2-glasses-vision
./aria-devices --help
```

A bare `aria-devices` only works after `source .venv/bin/activate`; the
launcher works either way, and from any directory if you give its full path.

```bash
./run-camera.sh
```

That opens your webcam with live detection. `q` or `Esc` quits, `s` saves a
still into `out/`.

**First run on macOS**: launch it from Terminal.app. macOS attaches the camera
permission prompt to whichever app owns the process, and an editor-embedded
terminal often cannot show it. If you get `could not open camera 0`, grant
access under System Settings → Privacy & Security → Camera, restart the
terminal, and retry.

### Install

```bash
python3.12 -m venv .venv && .venv/bin/python -m pip install -e '.[detect]'
```

Python 3.10–3.12. The live client SDK does not publish wheels for 3.13+, and
this machine's default `python3` is 3.14, which is why the venv pins 3.12.

Optional extras:

```bash
pip install -e '.[vrs]'    # offline .vrs reading  (projectaria-tools)
pip install -e '.[live]'   # live streaming        (projectaria-client-sdk)
pip install -e '.[owlv2]'  # the OWLv2 backend
pip install -e '.[dev]'    # pytest
```

---

## Commands

```bash
./aria-devices camera [--index 0] [--detect-every-ms 0]      # live webcam
./aria-devices video  INPUT.mp4 --out out/ [--mp4] [--jsonl]
./aria-devices vrs    INPUT.vrs --out out/ [--mp4] [--jsonl] [--mps DIR]
./aria-devices live   [--serial ...] [--interface usb] [--display]
./aria-devices dump-config config.yaml                       # write defaults
```

Common flags: `--backend {yoloworld,yoloe,owlv2}`, `--weights`, `--device
{auto,cpu,cuda,mps}`, `--imgsz`, `--conf`, `--hands
{auto,aria,mediapipe,openvocab,off}`, `--no-track`, `--draw-signals`,
`--stride`, `--max-frames`, `-v` / `-vv`.

### Live off the glasses

Pair once, then stream. **USB is the default and the right choice** — lower
latency and more bandwidth than Wi-Fi:

```bash
.venv/bin/aria_gen2 auth pair
./aria-devices live --interface usb --display
```

`--interface wifi_sta` or `wifi_sap` if USB-NCM is unavailable.
`--record-to-vrs out/session.vrs` saves the stream while processing it.

---

## Output

Annotated MP4, plus JSONL with one record per frame:

```json
{
  "frame_idx": 12,
  "timestamp_ns": 400000000,
  "detections": [
    {
      "track_id": 3,
      "label": "tablet",
      "score": 0.71,
      "bbox_xyxy": [412.0, 88.5, 596.2, 355.9],
      "gazed_at": true,
      "gaze_dwell_ms": 840.0,
      "signals": {"text": 0.62, "shape": 1.0, "size": 1.0, "orient": 1.0,
                  "screen": 0.83, "diag_cm": 30.4, "aspect_portrait": 0.69}
    }
  ],
  "hands": [{"bbox_xyxy": [...], "side": "right", "score": 0.94, "source": "aria"}],
  "interactions": [
    {"side": "right", "pose": "grab", "openness": 0.21,
     "device_track_id": 3, "device_label": "tablet", "relative_xy": [0.44, 0.62]}
  ],
  "gaze_point_px": [503.1, 210.7],
  "detect_ms": 28.4
}
```

`signals` is the per-rule score breakdown, for tuning weights. `-vv` logs the
same at debug level.

---

## Seeing what the glasses see

```bash
./aria-devices view recording.vrs                  # replay a recording
./aria-devices view --live --interface usb         # straight off the glasses
./aria-devices view recording.vrs --mp4 --out out/ # record the dashboard
```

One window with every sensor at once: the rectified RGB view with detections,
hands and gaze drawn on it; the four SLAM cameras; both eye-tracking cameras;
and a stats strip carrying gaze angles, hand confidences, VIO translation,
detection latency and **per-stream rate in Hz**.

Why per-stream rates rather than one FPS number: each sensor runs at its own
rate (RGB ~30 Hz, IMU up to 1 kHz), so a single figure hides the one stream
that stalled. Streams that have never produced data are drawn **greyed out with
a red dot rather than omitted** — a missing stream is the thing you are looking
for, and hiding it defeats the point. `--no-detect` shows raw sensors with the
detector switched off, which is the fastest way to tell "the model is confused"
from "the feed is broken".

`render_dashboard` is a pure function from snapshot to canvas, so the layout is
unit-tested without any hardware.

---

## The interaction output (what the rest of the system consumes)

The glasses are the sensor; the deliverable is **"a hand is at (x, y) on *this*
device"**, so the device can render a cursor there.

```bash
./aria-devices live --interface usb --events-udp 127.0.0.1:8900
```

One datagram per transition. Nothing is sent while your hands are away from
every device:

```json
{"state":"approach","device":"tablet","hand":"right","pose":"open","x":null,"y":null,"distance":0.164}
{"state":"enter",   "device":"tablet","hand":"right","pose":"open","x":0.5,"y":0.5,"distance":0.0}
{"state":"move",    "device":"tablet","hand":"right","pose":"grab","x":0.44,"y":0.62,"distance":0.0}
{"state":"leave",   "device":"tablet","hand":"right","pose":"unknown","x":null,"y":null}
```

- **`device`** is the routing key — `laptop` / `tablet` / `phone`. Since the
  study has exactly one of each, the class label *is* the device identity. If a
  second device of the same class ever appears, switch `InteractionTracker` to
  key on `device_track_id` instead.
- **`x`, `y`** are normalised **screen** coordinates in [0,1], origin top-left.
  Multiply by that device's own resolution to place the cursor.
- **`approach`** fires before the hand is over the device (within 0.6 device
  diagonals), so a device can wake or highlight before contact.
- **`pose`** is `grab` / `open` / `unknown`, for click-style gating.

Transitions, not a 30 Hz state dump: a device lights up once on `enter` and
clears once on `leave`.

### Screen rectangle vs detection box

The detector boxes the *whole physical object*. A cursor needs the *screen*, and
for an open MacBook those differ a lot — the box includes the keyboard base.
Each device profile carries a `screen_inset` that trims the box down:

| | trim (l, t, r, b) | why |
|---|---|---|
| phone | 0.03 all round | thin bezels |
| tablet | 0.06 all round | visible uniform bezel |
| laptop | 0.04, 0.04, 0.04, **0.38** | bottom 38% is the keyboard base |

Without the laptop trim a hand halfway down the box maps to y=0.50 when it
should be y≈0.81 — the cursor lands a third of the screen too high. The 0.38
figure assumes a lid open near 105–115° viewed from a seated position; if the
study fixes a different angle, re-measure it and set `screen_inset` in the
config.

**Known limit — this mapping is affine, not perspective-correct.** It assumes
the screen is roughly fronto-parallel. Viewed at a steep angle the cursor drifts
toward the near edge. Good enough for a seated user facing their devices, which
is this study's setup. The fix, if you need it, is a homography from the four
screen corners rather than an axis-aligned box; that needs corner extraction
(the screen-on brightness mask plus `cv2.minAreaRect` would get you there) and
is the natural next step if drift shows up in practice.

---

## Integrating with other systems

Three sinks, all optional:

```bash
./aria-devices live --events-udp 127.0.0.1:8900  # interaction events only  <- start here
./aria-devices camera --udp 127.0.0.1:8899       # full frame record, every frame
./aria-devices video in.mp4 --emit-stdout | your-tool
./aria-devices vrs in.vrs --out out/ --jsonl     # on disk
```

UDP mirrors what the Aria v1 study already did, but sends the structured result
instead of pixels. UDP is deliberate: a dropped frame is harmless because
another arrives ~30 ms later, whereas TCP back-pressure would stall detection.
Logs go to stderr, so `--emit-stdout` stays machine-readable.

Or embed it directly:

```python
from aria_devices import PipelineConfig
from aria_devices.pipeline import DevicePipeline
from aria_devices.sources.video import VideoFrameSource

source = VideoFrameSource("clip.mp4")
pipeline = DevicePipeline(PipelineConfig(), focal_px=source.focal_px)
pipeline.add_result_hook(lambda r: print(r.to_record()))

for result in pipeline.run(source):
    for det in result.detections:
        print(det.track_id, det.label, det.bbox_xyxy, det.gazed_at)
```

Hooks run synchronously on the processing thread — keep them cheap. A hook that
raises is logged and swallowed rather than killing the run.

---

## Why open-vocabulary instead of fine-tuning

COCO has `laptop` and `cell phone` but **no tablet class**, so a stock YOLO
structurally cannot do this task — it labels tablets as laptops or TVs. Meta's
`Aria Everyday Objects` does not help either: its 17 classes are furniture and
architecture, no personal electronics.

So the primary path is an open-vocabulary detector, prompted with a wide set and
collapsed to canonical labels by explicit rules. Default is Ultralytics
`YOLOWorld` (`yolov8s-worldv2.pt`); `yoloe` and `owlv2` are drop-in alternatives
selected by config, never hardcoded.

I agree with this call for now. The honest tradeoff: open-vocab gives you a
tablet hypothesis for free but its per-frame labels are noisy, which is why
roughly half this package is disambiguation and temporal voting rather than
detection.

---

## How the disambiguation works

Prompts are collapsed to `laptop` / `tablet` / `phone` / `hand`, with
`keyboard` / `monitor` / `tv` kept as evidence. Near-duplicate boxes are
clustered, then each cluster is scored against all three device hypotheses:

```
score = w_text·text + w_shape·shape + w_size·size + w_orient·orient + w_screen·screen
```

Weights are **normalised over the signals actually available**, so a missing
depth estimate does not drag every score below threshold.

| Signal | Weight | What it does | Where it fails |
|---|---|---|---|
| `text` | 0.45 | open-vocab confidence | conflates tablet/laptop/monitor constantly |
| `shape` | 0.15 | aspect ratio vs profile | perspective foreshortening at oblique angles |
| `size` | 0.15 | physical diagonal from angular size + depth | needs depth; absent → dropped |
| `orient` | 0.15 | landscape vs portrait vs expected | only valid if device orientation is fixed |
| `screen` | 0.10 | interior brighter than surround | fails against a bright window |

Plus two hard structural rules:

- **Laptop beats tablet on overlap.** An open MacBook's lid is a near-perfect
  tablet as far as the detector is concerned. A tablet box ≥70% contained in a
  laptop box (or ≥0.55 IoU) is dropped. Highest-yield rule in the package.
- **Keyboard adjacency.** A screen with a keyboard directly below and aligned in
  x is promoted to laptop (+0.35).

### Priors for the three study devices

| | body diagonal | aspect (short/long) | orientation |
|---|---|---|---|
| iPhone 16 Pro Max | 18.1 cm | 0.476 | landscape |
| iPad A16 11" | 30.9 cm | 0.681 | portrait |
| MacBook Air 13"/15" M4 | 37.2 / 41.5 cm | 0.71 | landscape, open |

Bands are widened around these to cover ordinary hardware. Two calibration
notes, both found by the tests rather than by inspection:

- The laptop aspect band **must not** dip below ~0.55. An earlier `(0.45, 0.85)`
  completely swallowed the phone band `(0.42, 0.60)`, making a landscape phone
  and a laptop indistinguishable by shape — exactly this study's setup.
- `size_prior_softness_cm` is 3.0, not 6.0. The phone and tablet bands nearly
  touch (19.5 / 20.0 cm); at 6 cm softness an 18 cm phone still scored 0.68 as
  a tablet, throwing away the one signal that separates them reliably.

### Where depth comes from

The size prior is the single most reliable phone-vs-tablet signal, since the two
differ mainly in size. Depth sources, best first:

1. **Aria on-device hand tracking** — 21 metric 3D landmarks in device frame,
   projected into the rectified image. Real measured depth. Used automatically
   whenever the `handtracking` stream exists.
2. **MediaPipe world landmarks** — metric hand model, so
   `z = f · L_metres / L_pixels`. Roughly ±15% on the hand, easily enough to
   separate 18 cm from 31 cm. This is what makes the size prior usable on a
   webcam.
3. **Nothing** — the term is dropped and its weight redistributed.

Depth only answers for boxes *near a hand* (within 2.5× the hand's diagonal). A
device across the room gets no size prior, by design.

**A correction to the brief:** VIO does not provide per-object depth. It gives
device pose in the world, not distance to a detected box. Genuine per-object
depth needs the MPS semi-dense point cloud (`--mps DIR`), which is offline-only.
The hand-scale route above is what actually works live.

### Monitors and TVs

Suppressed rather than promoted: anything measuring over 46 cm is furniture. A
`WorldFixedTracker` also flags boxes that stay put across many frames — but be
aware this is weak without ego-motion compensation, since under head motion a
world-fixed monitor *does* move in the image. The size threshold does the real
work; treat the world-fixed rule as a refinement.

---

## Tracking

A self-contained ByteTrack-style tracker, **not** Ultralytics'
`model.track(persist=True)`. That call is bolted to the Ultralytics predictor,
so using it would leave OWLv2 with no tracking at all and break the
backend-is-a-config-choice requirement. Two-stage IoU association with
constant-velocity prediction; the second pass rescues tracks from
low-confidence detections, which is what keeps a phone tracked through the
motion blur of a head turn.

Each track holds a rolling label histogram (15 frames) and emits the majority
label, so a device sitting still keeps one stable label instead of strobing.

---

## Image handling on the Aria RGB stream

Order matters, and getting it wrong quietly destroys detection quality:

1. Read the raw frame — fisheye, and 12 MP on Gen 2.
2. Rectify fisheye → pinhole against a linear destination calibration.
3. Rotate **90° clockwise**, pixels *and* calibration together.
4. Downscale to the detector's input size, keeping the scale factor.

**A correction to the brief:** `distort_by_calibration_and_apply_rotation` is
*not* the 90° upright helper. Its rotation argument is an `SO3` applied to the
camera ray, for stereo rectification. The upright view needs
`cv2.rotate(..., ROTATE_90_CLOCKWISE)` on the pixels alongside
`rotate_camera_calib_cw90deg` on the calibration. `rotate_camera_calib_cw90deg`
is also linear-model-only, so it must come *after* rectification.

Rectification maps are built once and cached to
`~/.cache/aria_devices/rectify_maps`. `distort_by_calibration` rebuilds its warp
on every call, which at 12 MP makes the pipeline unusable; we build a
`cv2.remap` table instead.

### Focal length tradeoff

`rectify.focal` (default 350 px at 704×704) sets how much of the fisheye FOV
survives. Lower focal keeps more of the ~110°+ field but stretches the edges
severely, and heavily stretched objects break the aspect-ratio prior. Higher
focal gives cleaner geometry over a narrower field, and devices at the edge of
view are simply lost. 350 px keeps roughly 90° horizontal with tolerable corner
distortion. Tune with `--focal` and `--rect-size`.

`--devignette` needs a separately-shipped mask folder; without one it logs a
warning and carries on. `color_correct` is on by default.

---

## Gaze

The `eyegaze` stream is projected into the rectified image through
CPF → Device → Camera. Each detection gets `gazed_at` and `gaze_dwell_ms`. A
gaze point inside a box wins; otherwise the nearest box within 90 px wins,
because eye tracking has real angular error and strict containment throws away
most of the signal on a target as small as a phone. When boxes nest, the
smallest wins — a phone on a laptop should take the gaze.

---

## Hands and interaction

| Source | Backend | Output |
|---|---|---|
| Aria (`vrs`, `live`) | on-device `handtracking` stream | 21 metric 3D joints + depth |
| Webcam / MP4 | MediaPipe Tasks `HandLandmarker` | 21 image-space joints + metric world landmarks |
| Fallback | open-vocab `"human hand"` prompt | box only |

**MediaPipe 1.0 removed `mp.solutions` entirely** — only the Tasks API remains,
so this uses `HandLandmarker` in VIDEO mode. The model downloads to
`assets/hand_landmarker.task` on first use.

Interaction, carried over from the v1 study, with **GRAB and OPEN only** (v1's
MID state dropped):

- `openness` — mean fingertip reach ÷ wrist-to-knuckle span. Scale-invariant, so
  it reads the same at any distance. Fist ≈ 0, splayed ≈ 1.
- `pose` — `grab` below 0.35, `open` above 0.55. The gap is deliberate: a hand
  mid-close returns `unknown` rather than flickering.
- `relative_xy` — where the fingertip centroid falls inside a device box, in
  [0,1]², same contract as v1's `Device.relative_position`. Drawn as a
  crosshair.

Aria and MediaPipe number their 21 landmarks differently (Aria puts fingertips
at 0–4 and the wrist at 5); the indices are looked up per source, never assumed.

---

## Performance

Measured on this machine (Apple M4, CPU/MPS, no CUDA):

- YOLO-World `yolov8s-worldv2` at 640: **~29 ms/frame (~35 fps)** detection
- With MediaPipe hands in the loop: ~15 fps end-to-end offline

`camera` mode decouples the three stages onto separate threads — capture,
detect, render — so the **preview stays at camera framerate** while detection
runs behind it at its own pace, and the tracker's velocity prediction hides most
of the lag. This is why waving your hand looks smooth even though the detector
is slower than the camera.

To go faster: `--imgsz 416`, or `--detect-every-ms 50` to throttle detection and
keep the fans quiet. `--device mps` is resolved automatically by `auto` on
Apple silicon; the brief only mentioned CUDA, but MPS is the relevant fast path
here.

---

## Testing

```bash
.venv/bin/python -m pytest -q     # 131 tests, ~0.4 s
```

Every disambiguation rule is unit tested on synthetic boxes, with no model
weights, no network and no Aria hardware. Rectify/rotate tests assert image and
calibration dimensions stay consistent against a synthetic calibration. The
end-to-end smoke test generates a short clip of coloured rectangles and asserts
the pipeline runs and emits well-formed JSONL — *not* that it detects anything,
since detection quality on synthetic rectangles is meaningless. VRS and live
tests are guarded behind `importorskip` / markers so the suite is green on a
bare machine.

---

## Known limits

- **Everything rests on the text score (0.45).** When the open-vocab model is
  confidently wrong, the geometric priors can outvote it only if depth and
  orientation are both available.
- **The orientation prior assumes the study's fixed setup.** Phone landscape,
  tablet portrait. If a device is rotated the prior actively works against you —
  set `orientation: any` in the config for that device.
- **Keyboard adjacency fires on desktop monitors** with a keyboard in front of
  them. Only the size prior pulls those back, so without depth some monitors
  will be labelled laptop.
- **The screen-on prior fails against bright backgrounds** — a lit screen with a
  window behind it is not brighter than its surround. Weighted 0.10 for this
  reason.
- **World-fixed detection is weak without VIO ego-motion compensation.**
- **Webcam focal length is a guess** (58° hFOV assumed), so depth and physical
  sizes off-Aria are approximate. On Aria the focal is known exactly.
- **No detections on synthetic rectangles.** Expected — the model needs real
  imagery. Test on real devices.

---

## If accuracy isn't good enough

Follow-up path, not built here: fine-tune a small YOLO on
[`facebookresearch/EgoObjects`](https://github.com/facebookresearch/EgoObjects)
(ICCV 2023 — large-scale egocentric, fine-grained categories including personal
electronics) to get a purpose-built 3-class head, and use the open-vocab model
only to bootstrap pseudo-labels on your own Aria recordings. That converts this
package's disambiguation rules from load-bearing into a sanity check.

---

## Layout

```
aria_devices/
  config.py         dataclasses + YAML; device profiles and weights live here
  frames.py         Frame, GazeSample, HandSample
  sources/          base.py, vrs.py, live.py, video.py
  detect/           base.py, yoloworld.py, owlv2.py, disambiguate.py
  track.py          ByteTrack + temporal label voting
  gaze.py           projection + dwell attribution
  hands.py          Aria / MediaPipe / open-vocab backends + depth
  interaction.py    grab-open pose, hand-on-device position
  viz.py            boxes, skeletons, gaze cursor, MP4
  emit.py           UDP / stdout integration sinks
  pipeline.py       detect -> disambiguate -> track -> gaze
  runner.py         offline driver
  realtime.py       threaded live camera app
  cli.py
tests/
_Omni_Conect_Aria_Ver1-main/   the v1 study, kept as reference
```

`_Omni_Conect_Aria_Ver1-main/` is retained deliberately: its UDP forwarding
pattern and `Device.relative_position` are the direct ancestors of `emit.py` and
`interaction.py`.
