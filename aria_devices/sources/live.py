"""Live streaming from paired Aria Gen 2 glasses.

Verified against projectaria-client-sdk 2.4.0. The real API differs from the
shape sketched in the brief, in ways that matter:

* ``DeviceClientConfig`` has no ``device_serial``. The serial goes in a
  ``DeviceTarget`` passed to ``connect()``.
* There is no handler hanging off the device. Streaming is a *push* model: you
  run a local HTTP server (``aria.stream_receiver.StreamReceiver``) and the
  glasses post data to it. Callbacks are registered on the receiver.
* The RGB callback takes ``(image_data, image_record)`` — two arguments, not
  three.
* The stock streaming profile is ``profile9``.

USB is the default interface (``StreamingInterface.USB_NCM``): lower latency and
more bandwidth than Wi-Fi, which matters when you want maximum frame rate.

Pair the glasses first:

    aria_gen2 auth pair
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from collections.abc import Iterator
from pathlib import Path

import cv2
import numpy as np

from ..config import RectifyConfig
from ..frames import Frame, GazeSample, HandSample
from .base import FrameSource

log = logging.getLogger(__name__)

#: Streaming profiles reported by a Gen 2 device (see `aria-devices profiles`):
#:
#:   mp_streaming_demo     machine-perception demo — the ML streams this
#:                         package lives on (hand pose, eye gaze, VIO)
#:   profile9              general purpose streaming
#:   low_latency_streaming "thermally-stable A/V streaming at low batching
#:                         periods" — the one to reach for when the glasses
#:                         start throttling
#:
#: Default is the machine-perception profile: hand tracking and gaze are not
#: optional extras here, they are the point.
DEFAULT_PROFILE = "mp_streaming_demo"
DEFAULT_PORT = 6768

#: Skin temperature above which Gen 2 starts mitigating. Past this the device
#: quietly drops frame rate and image quality, and will eventually refuse to
#: stream at all with "Cannot record due to thermal throttling" — which reads
#: like a bug in your code but is not.
THERMAL_WARN_CELSIUS = 40.0

#: Names accepted for --interface, mapped in _interface_map() against the
#: installed SDK so an unknown name fails loudly instead of silently.
INTERFACE_NAMES = ("usb", "wifi_sta", "wifi_sap")


#: Where the SDK keeps the TLS material for the local streaming server.
STREAMING_CERT_DIR = Path.home() / ".aria" / "streaming-certs" / "persistent"

#: Files that must exist and be non-empty before the receiver will start.
REQUIRED_CERT_FILES = ("subscriber.pem", "subscriber-key.pem", "root_ca.pem")

#: Written by `aria_gen2 streaming install-certs`; names the cert pair that is
#: installed on both the device and this host.
PUBLISHER_CERT_NAME_FILE = "publisher-cert-name"


def installed_cert_name(cert_dir: Path = STREAMING_CERT_DIR) -> str | None:
    """Name of the streaming cert installed on this host, if any."""
    path = cert_dir / PUBLISHER_CERT_NAME_FILE
    if not path.exists():
        return None
    name = path.read_text().strip()
    return name or None


def preflight_streaming_certs(cert_dir: Path = STREAMING_CERT_DIR) -> str:
    """Fail early, and in Python, if the streaming certificates are unusable.

    Without this the SDK throws from C++ and the whole interpreter dies with
    ``libc++abi: terminating due to uncaught exception`` — no traceback, no
    chance to catch it, and an error message that does not say what to do.

    Returns the installed cert name, which the caller must pin onto the
    streaming config. If it is left unset the *device* generates a brand new
    cert on every run and reinstalls it, overwriting these files while the
    local server is already holding the old pair — which surfaces as
    "Cert does not match private key" or a half-emptied cert directory.
    """
    fix = (
        "Run:  .venv/bin/aria_gen2 streaming install-certs\n"
        "If that fails, clear the directory first so generation starts clean:\n"
        f"  rm -rf {cert_dir}"
    )
    if not cert_dir.is_dir():
        raise RuntimeError(f"no Aria streaming certificates at {cert_dir}.\n{fix}")

    problems = []
    for name in REQUIRED_CERT_FILES:
        path = cert_dir / name
        if not path.exists():
            problems.append(f"{name} is missing")
        elif path.stat().st_size == 0:
            problems.append(f"{name} is empty (generation failed partway)")
    if problems:
        raise RuntimeError(
            "Aria streaming certificates are incomplete: " + "; ".join(problems) + f"\n{fix}"
        )

    cert_name = installed_cert_name(cert_dir)
    if cert_name is None:
        raise RuntimeError(
            f"{PUBLISHER_CERT_NAME_FILE} is missing from {cert_dir}, so there is no "
            f"installed cert to pin.\n{fix}"
        )
    return cert_name


def _interface_map() -> dict:
    import aria.sdk_gen2 as sdk_gen2

    return {
        "usb": sdk_gen2.StreamingInterface.USB_NCM,
        "wifi_sta": sdk_gen2.StreamingInterface.WIFI_STA,
        "wifi_sap": sdk_gen2.StreamingInterface.WIFI_SAP,
    }


class LiveFrameSource(FrameSource):
    """Streams RGB + eye gaze + hand poses off the glasses.

    RGB frames arrive on an SDK callback thread and are pushed into a bounded
    queue that *drops the oldest* when full. That is deliberate: in a live
    setting a stale frame is worthless, and blocking the SDK's callback thread
    to preserve one would stall the whole stream.

    Gaze and hand callbacks fire at their own rates, independent of RGB, so the
    latest sample of each is simply latched and attached to whichever frame
    comes next.
    """

    def __init__(
        self,
        serial: str | None = None,
        ip: str | None = None,
        enhance: bool = True,
        profile: str = DEFAULT_PROFILE,
        interface: str = "usb",
        rectify: RectifyConfig | None = None,
        port: int = DEFAULT_PORT,
        queue_size: int = 4,
        max_frames: int = 0,
        record_to_vrs: str | None = None,
        timeout_s: float = 20.0,
        batch_period_ms: int = 0,
    ) -> None:
        try:
            import aria.sdk_gen2 as sdk_gen2
            import aria.stream_receiver as stream_receiver
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise ImportError(
                "projectaria-client-sdk is required for live streaming: "
                "pip install 'aria-devices[live]'"
            ) from exc

        self.enhance = enhance
        self._enhance_logged = False
        if interface not in INTERFACE_NAMES:
            raise ValueError(f"interface must be one of {INTERFACE_NAMES}, got {interface!r}")

        # Must run before anything touches the SDK's streaming server.
        self.cert_name = preflight_streaming_certs()
        log.info("using installed streaming cert %s", self.cert_name)

        self._sdk = sdk_gen2
        self.cfg = rectify or RectifyConfig()
        self.max_frames = max_frames
        self.timeout_s = timeout_s
        self.fps = 30.0
        self.focal_px = None
        self.has_hand_tracking = True

        self._queue: queue.Queue = queue.Queue(maxsize=max(1, queue_size))
        self._stop = threading.Event()
        self._latest_gaze: GazeSample | None = None
        self._latest_hands: list[HandSample] = []
        self._lock = threading.Lock()
        self._frame_idx = 0
        self._rectifier: _LiveRectifier | None = None

        # -- connect --------------------------------------------------------
        client = sdk_gen2.DeviceClient()
        client.set_client_config(sdk_gen2.DeviceClientConfig())
        # AN EXPLICIT IP IS THE ONLY THING THAT REACHES THEM OVER WI-FI.
        # Bare discovery looks over USB and reports "(-4) No devices found";
        # serial lookup needs a route a phone hotspot does not provide and
        # fails "no route to host" then "device not found". Neither error hints
        # at trying an address, so the address is offered here.
        if ip:
            target = sdk_gen2.DeviceTarget(ip=ip); how = f"ip={ip}"
        elif serial:
            target = sdk_gen2.DeviceTarget(serial=serial); how = f"serial={serial}"
        else:
            target = sdk_gen2.DeviceTarget(); how = "<first available over USB>"
        log.info("connecting to Aria Gen 2 (%s)", how)

        # THE CONNECT AFTER A PREVIOUS RUN NEEDS PATIENCE, NOT A DIFFERENT
        # ADDRESS. Measured over a night of runs: a successful run is regularly
        # followed by one that cannot connect at all, three attempts running,
        # while the glasses sit in the ARP table at the same address. A minute
        # later it works. That is the previous session tearing down on the
        # device, not the network, and impatient retries diagnose it as dead
        # hardware and send you to check cables that are fine.
        backoffs = (2.0, 5.0, 10.0, 15.0, 20.0)
        last = None
        for attempt, wait in enumerate(backoffs, 1):
            try:
                self.device = client.connect(target)
                if attempt > 1:
                    log.info("connected on attempt %d", attempt)
                break
            except Exception as exc:
                last = exc
                log.warning("connect attempt %d/%d failed: %s",
                            attempt, len(backoffs), exc)
                if attempt == 2:
                    log.warning("usually the previous session still closing on "
                                "the glasses; waiting it out")
                if attempt < len(backoffs):
                    time.sleep(wait)
        else:
            raise RuntimeError(
                f"could not connect to the glasses ({how}) after "
                f"{len(backoffs)} attempts over ~{int(sum(backoffs))}s: {last}. "
                f"If they are in `arp -an` at this address, power-cycle them: "
                f"that clears a session the SDK cannot."
            )
        self._client = client
        self._warn_if_hot()

        # -- local receiver --------------------------------------------------
        server_config = sdk_gen2.HttpServerConfig()
        server_config.address = "0.0.0.0"
        server_config.port = port

        self.receiver = stream_receiver.StreamReceiver(
            enable_image_decoding=True, enable_raw_stream=False
        )
        self.receiver.set_server_config(server_config)
        if record_to_vrs:
            self.receiver.record_to_vrs(record_to_vrs)
            log.info("also recording the stream to %s", record_to_vrs)

        self.receiver.register_rgb_callback(self._on_rgb)
        self.receiver.register_eye_gaze_callback(self._on_gaze)
        self.receiver.register_hand_pose_callback(self._on_hands)
        self.receiver.register_device_calib_callback(self._on_calib)

        # -- start ------------------------------------------------------------
        streaming_config = sdk_gen2.HttpStreamingConfig()
        streaming_config.profile_name = profile
        streaming_config.streaming_interface = _interface_map()[interface]
        # Pin the already-installed cert. Without this the device mints a new
        # one per run and reinstalls it over the pair the local server is
        # already using, breaking the TLS handshake.
        streaming_config.streaming_cert_name = self.cert_name
        if batch_period_ms > 0:
            # Batching trades latency for heat: fewer, larger transmissions mean
            # fewer radio/USB wakeups, which is the main lever for surviving a
            # long session without thermal mitigation kicking in.
            streaming_config.batch_period_ms = batch_period_ms
            log.info("batching at %d ms to reduce thermal load", batch_period_ms)
        self.device.set_streaming_config(streaming_config)

        log.info("starting HTTP receiver on port %d", port)
        self.receiver.start_server()
        log.info("starting stream: profile=%s interface=%s", profile, interface)
        # A KILLED RUN LEAVES THE SESSION OPEN ON THE GLASSES, and the next is
        # refused with "(7) User session already started". Nothing in that
        # message says the cure is to stop the previous stream, so it reads as
        # dead hardware -- and it happens whenever a run is Ctrl-C'd at the
        # wrong moment, which is most of them. Tearing down takes the glasses a
        # few seconds, so back off rather than hammering.
        try:
            self.device.start_streaming()
        except Exception as exc:
            if "already started" not in str(exc).lower():
                raise
            log.warning("a previous streaming session is still open, clearing it")
            for wait in (2.0, 4.0, 6.0):
                try:
                    self.device.stop_streaming()
                except Exception:
                    pass
                time.sleep(wait)
                try:
                    self.device.start_streaming()
                    log.info("cleared the old session and started streaming")
                    break
                except Exception as retry_exc:
                    if "already started" not in str(retry_exc).lower():
                        raise
            else:
                raise RuntimeError(
                    "the glasses are still holding a streaming session from an "
                    "earlier run. Power-cycle them; that always clears it."
                ) from exc

    # -- exposure -----------------------------------------------------------
    #: Below this mean grey level a frame is underexposed. Normal indoor
    #: exposure sits around 100-140.
    DARK_MEAN = 85.0

    def _brighten(self, rgb):
        """Lift an underexposed frame before anything tries to detect in it.

        MEASURED. Frames off these glasses came in at a mean grey level of
        29.8/255 with 73% of pixels below 40 -- nearly black. The detector was
        handed that and unsurprisingly found almost nothing. Brightening takes
        it to ~113.

        CLAHE on the L channel rather than a global gain or gamma, because the
        problem is local: a lit screen beside a dark desk is exactly what a
        global curve crushes or blows out, and the screen edges are what the
        device boxes are found from.

        Adaptive, so a well-exposed frame is left untouched and this cannot
        hurt a session in a brighter room.
        """
        grey = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        mean = float(grey.mean())
        if mean >= self.DARK_MEAN:
            return rgb
        lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
        l, a_, b_ = cv2.split(lab)
        l = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(l)
        # CLAHE fixes contrast, not overall level. Very dark frames still need
        # lifting, capped so shadow noise is not amplified into edges the
        # detector reads as structure.
        if mean < 55.0:
            l = cv2.convertScaleAbs(l, alpha=min(2.2, 85.0 / max(mean, 1.0)), beta=0)
        out = cv2.cvtColor(cv2.merge((l, a_, b_)), cv2.COLOR_LAB2RGB)
        if not self._enhance_logged:
            log.info("frames are underexposed (mean %.0f/255), brightening "
                     "before detection", mean)
            self._enhance_logged = True
        return out

    # -- device health ------------------------------------------------------
    def device_status(self):
        """Latest device status, or None if it cannot be read.

        Worth polling during a long session: skin temperature and the thermal
        mitigation flag are the difference between "the detector got worse" and
        "the glasses quietly halved the frame rate".
        """
        try:
            return self.device.status()
        except Exception:  # pragma: no cover - hardware dependent
            return None

    def thermal_summary(self) -> str:
        status = self.device_status()
        if status is None:
            return "unknown"
        temp = float(getattr(status, "skin_temp_celsius", 0.0) or 0.0)
        throttled = bool(getattr(status, "thermal_mitigation_triggered", False))
        charging = bool(getattr(status, "charging", False))
        parts = [f"{temp:.1f}C"]
        if throttled:
            parts.append("THROTTLING")
        if charging:
            parts.append("charging")
        return " ".join(parts)

    def _warn_if_hot(self) -> None:
        status = self.device_status()
        if status is None:
            return
        temp = float(getattr(status, "skin_temp_celsius", 0.0) or 0.0)
        throttled = bool(getattr(status, "thermal_mitigation_triggered", False))
        charging = bool(getattr(status, "charging", False))
        if throttled:
            log.warning(
                "glasses are THERMALLY THROTTLING (skin %.1f C). Frame rate and image "
                "quality are already degraded and streaming may be refused outright. "
                "Let them cool, and prefer the low_latency_streaming profile.", temp
            )
        elif temp >= THERMAL_WARN_CELSIUS:
            log.warning("glasses are warm (skin %.1f C); throttling is close", temp)
        if charging and temp >= THERMAL_WARN_CELSIUS - 3:
            log.warning(
                "charging while streaming adds heat — for a long session, charge to "
                "full beforehand and let the battery run down during the recording"
            )

    # -- SDK callbacks (called on SDK threads) ------------------------------
    def _on_rgb(self, image_data, image_record) -> None:
        try:
            array = image_data.to_numpy_array()
        except Exception as exc:  # pragma: no cover
            log.warning("could not decode RGB frame: %s", exc)
            return
        item = (np.asarray(array), int(image_record.capture_timestamp_ns))
        try:
            self._queue.put_nowait(item)
        except queue.Full:
            # Drop the oldest — a late frame is worse than no frame here.
            try:
                self._queue.get_nowait()
                self._queue.put_nowait(item)
            except queue.Empty:  # pragma: no cover - race
                pass

    def _on_gaze(self, eyegaze_data) -> None:
        sample = GazeSample(
            yaw_rad=float(eyegaze_data.yaw),
            pitch_rad=float(eyegaze_data.pitch),
            depth_m=float(getattr(eyegaze_data, "depth", 0.0) or 0.0) or None,
        )
        if self._rectifier is not None:
            sample = self._rectifier.project_gaze(eyegaze_data) or sample
        with self._lock:
            self._latest_gaze = sample

    def _on_hands(self, handtracking_data) -> None:
        hands: list[HandSample] = []
        if self._rectifier is not None:
            hands = self._rectifier.project_hands(handtracking_data)
        with self._lock:
            self._latest_hands = hands

    def _on_calib(self, device_calib) -> None:
        """Device calibration arrives asynchronously; build the rectifier then."""
        if self._rectifier is not None:
            return
        try:
            self._rectifier = _LiveRectifier(device_calib, self.cfg)
            self.focal_px = self._rectifier.focal_px
            log.info("device calibration received; rectification ready")
        except Exception as exc:
            log.warning("could not build rectifier from device calibration: %s", exc)

    # -- iteration -----------------------------------------------------------
    def __iter__(self) -> Iterator[Frame]:
        deadline = time.time() + self.timeout_s
        while not self._stop.is_set():
            try:
                array, timestamp_ns = self._queue.get(timeout=0.5)
            except queue.Empty:
                if time.time() > deadline:
                    log.error("no frames received in %.0fs — is the device streaming?", self.timeout_s)
                    return
                continue
            deadline = time.time() + self.timeout_s

            if array.ndim == 2:
                array = cv2.cvtColor(array, cv2.COLOR_GRAY2RGB)
            rgb = self._rectifier.rectify(array) if self._rectifier else array
            if self.enhance:
                rgb = self._brighten(rgb)

            with self._lock:
                gaze = self._latest_gaze
                hands = list(self._latest_hands)

            yield Frame(
                rgb=np.ascontiguousarray(rgb),
                timestamp_ns=timestamp_ns,
                frame_idx=self._frame_idx,
                calib=self._rectifier.final_calib if self._rectifier else None,
                gaze=gaze,
                hands=hands,
                source="live",
                meta={"focal_px": self.focal_px},
            )
            self._frame_idx += 1
            if self.max_frames and self._frame_idx >= self.max_frames:
                return

    def close(self) -> None:
        self._stop.set()
        try:
            self.device.stop_streaming()
        except Exception as exc:  # pragma: no cover - hardware dependent
            log.warning("stop_streaming failed: %s", exc)
        time.sleep(0.5)
        try:
            self.receiver.stop_server()
        except Exception as exc:  # pragma: no cover
            log.warning("stop_server failed: %s", exc)


class _LiveRectifier:
    """Shares the VRS rectification path, driven by a live device calibration."""

    def __init__(self, device_calib, cfg: RectifyConfig) -> None:
        from projectaria_tools.core import calibration as cal

        from .vrs import RGB_LABEL, build_rectify_maps, rotate_calibration, rotate_image

        self.cfg = cfg
        self.device_calib = device_calib
        self.src_calib = device_calib.get_camera_calib(RGB_LABEL)
        if self.src_calib is None:
            raise RuntimeError("device calibration has no camera-rgb entry")

        w, h = cfg.size
        self.dst_calib = cal.get_linear_camera_calibration(
            w, h, cfg.focal, RGB_LABEL, self.src_calib.get_transform_device_camera()
        )
        self._map_x, self._map_y = build_rectify_maps(self.src_calib, self.dst_calib)
        self.final_calib = rotate_calibration(self.dst_calib, cfg.rotation)
        self.focal_px = float(self.final_calib.get_focal_lengths()[0])

        from ..gaze import GazeProjector

        self._gaze_projector = GazeProjector(device_calib, self.final_calib)

    def rectify(self, image: np.ndarray) -> np.ndarray:
        # rotate_image is imported INSIDE __init__, so it is a local there and
        # invisible here: every call raised "NameError: name 'rotate_image' is
        # not defined" the moment a frame arrived. The connection, the streams
        # and the detector all looked healthy -- 64 RGB frames enqueued and
        # processed -- and not one frame ever reached the pipeline.
        from .vrs import rotate_image

        if not self.cfg.enabled:
            return image
        out = cv2.remap(
            image, self._map_x, self._map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT
        )
        return rotate_image(out, self.cfg.rotation)

    def project_gaze(self, eyegaze_data) -> GazeSample | None:
        try:
            return self._gaze_projector.project(eyegaze_data)
        except Exception:  # pragma: no cover
            return None

    def project_hands(self, handtracking_data) -> list[HandSample]:
        T_cam_device = self.final_calib.get_transform_device_camera().inverse()
        w_img, h_img = (int(v) for v in self.final_calib.get_image_size())
        out: list[HandSample] = []
        for side in ("left", "right"):
            hand = getattr(handtracking_data, f"{side}_hand", None)
            if hand is None:
                continue
            landmarks = getattr(hand, "landmark_positions_device", None)
            if landmarks is None:
                continue
            pts, depths = [], []
            for p_device in np.asarray(landmarks, dtype=np.float64).reshape(-1, 3):
                p_cam = np.asarray(T_cam_device @ p_device, dtype=np.float64).reshape(-1)[:3]
                if p_cam[2] <= 0:
                    continue
                px = self.final_calib.project(p_cam)
                if px is None:
                    continue
                pts.append((float(px[0]), float(px[1])))
                depths.append(float(p_cam[2]))
            if len(pts) < 4:
                continue
            arr = np.asarray(pts, dtype=np.float32)
            x1, y1 = arr.min(axis=0)
            x2, y2 = arr.max(axis=0)
            pad = 0.10 * max(x2 - x1, y2 - y1)
            sample = HandSample(
                bbox_xyxy=(
                    float(max(0, x1 - pad)), float(max(0, y1 - pad)),
                    float(min(w_img, x2 + pad)), float(min(h_img, y2 + pad)),
                ),
                side=side,
                score=float(getattr(hand, "confidence", 1.0) or 1.0),
                landmarks_px=arr,
                source="aria",
            )
            setattr(sample, "_depth_m", float(np.median(depths)))
            out.append(sample)
        return out
