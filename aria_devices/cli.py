"""Command line interface.

    aria-devices camera [--index 0]                 live webcam preview
    aria-devices video  INPUT.mp4 --out out/        offline MP4
    aria-devices vrs    INPUT.vrs  --out out/       offline Aria recording
    aria-devices live   [--serial ...]              live off the glasses
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

from .config import PipelineConfig, load_config, save_config

log = logging.getLogger("aria_devices")


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", type=str, help="YAML config file")
    parser.add_argument("--backend", type=str, help="yoloworld | yoloe | owlv2")
    parser.add_argument("--weights", type=str, help="detector weights path or name")
    parser.add_argument("--device", type=str, help="auto | cpu | cuda | mps")
    parser.add_argument("--imgsz", type=int, help="detector input size")
    parser.add_argument("--conf", type=float, help="detector confidence threshold")
    parser.add_argument("--hands", type=str, help="auto | aria | mediapipe | openvocab | off")
    parser.add_argument("--no-track", action="store_true", help="disable tracking and voting")
    parser.add_argument("--draw-signals", action="store_true", help="draw score breakdown")
    parser.add_argument("-v", "--verbose", action="count", default=0)


def _add_output(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--out", type=str, default="out", help="output directory")
    parser.add_argument("--mp4", action="store_true", help="write annotated MP4")
    parser.add_argument("--jsonl", action="store_true", help="write per-frame JSONL")
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--display", action="store_true", help="show a live window")
    parser.add_argument(
        "--udp", type=str, metavar="HOST:PORT",
        help="stream one JSON datagram per frame (e.g. 127.0.0.1:8899)",
    )
    parser.add_argument(
        "--events-udp", type=str, metavar="HOST:PORT",
        help="stream only hand-on-device events (enter/move/leave/approach)",
    )
    parser.add_argument(
        "--emit-stdout", action="store_true",
        help="write line-delimited JSON to stdout for piping",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="aria-devices",
        description="Detect laptops, tablets, phones and hands in Aria Gen 2 video",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # -- camera -----------------------------------------------------------
    p_cam = sub.add_parser("camera", help="live webcam preview (no Aria needed)")
    p_cam.add_argument("--index", type=int, default=0, help="camera index")
    p_cam.add_argument("--width", type=int, default=1280)
    p_cam.add_argument("--height", type=int, default=720)
    p_cam.add_argument(
        "--detect-every-ms", type=float, default=0.0,
        help="throttle detection (0 = as fast as possible)",
    )
    p_cam.add_argument("--no-display", action="store_true")
    _add_output(p_cam)
    _add_common(p_cam)

    # -- image ------------------------------------------------------------
    p_img = sub.add_parser(
        "image", help="run on one or more still photos (fastest way to sanity-check)"
    )
    p_img.add_argument("input", type=str, nargs="+", help="image file(s)")
    _add_output(p_img)
    _add_common(p_img)

    # -- video ------------------------------------------------------------
    p_vid = sub.add_parser("video", help="offline MP4 / webcam recording")
    p_vid.add_argument("input", type=str)
    _add_output(p_vid)
    _add_common(p_vid)

    # -- vrs --------------------------------------------------------------
    p_vrs = sub.add_parser("vrs", help="offline Aria Gen 2 .vrs recording")
    p_vrs.add_argument("input", type=str)
    p_vrs.add_argument("--no-rectify", action="store_true", help="skip fisheye rectification")
    p_vrs.add_argument("--focal", type=float, help="rectified pinhole focal length in px")
    p_vrs.add_argument("--rect-size", type=int, help="rectified square image size")
    p_vrs.add_argument("--devignette", action="store_true")
    p_vrs.add_argument("--mps", type=str, help="MPS output folder for point-cloud depth")
    _add_output(p_vrs)
    _add_common(p_vrs)

    # -- live -------------------------------------------------------------
    p_live = sub.add_parser("live", help="live stream from paired Aria Gen 2 glasses")
    p_live.add_argument("--serial", type=str, help="device serial")
    p_live.add_argument(
        "--profile", type=str, default="mp_streaming_demo",
        help="mp_streaming_demo (ML streams) | profile9 | low_latency_streaming (coolest)",
    )
    p_live.add_argument(
        "--interface", type=str, default="usb", choices=["usb", "wifi_sta", "wifi_sap"],
        help="usb (USB-NCM) is lowest latency and highest bandwidth",
    )
    p_live.add_argument("--port", type=int, default=6768, help="local receiver port")
    p_live.add_argument(
        "--batch-ms", type=int, default=0,
        help="stream batching period; higher = cooler glasses, longer sessions",
    )
    p_live.add_argument("--record-to-vrs", type=str, help="also save the stream to a .vrs")
    _add_output(p_live)
    _add_common(p_live)

    # -- view -------------------------------------------------------------
    p_view = sub.add_parser(
        "view", help="dashboard of every Aria Gen 2 sensor at once"
    )
    p_view.add_argument(
        "input", type=str, nargs="?",
        help="a .vrs recording; omit and pass --live to use the glasses",
    )
    p_view.add_argument("--live", action="store_true", help="stream from the glasses")
    p_view.add_argument("--serial", type=str)
    p_view.add_argument("--profile", type=str, default="mp_streaming_demo")
    p_view.add_argument(
        "--interface", type=str, default="usb", choices=["usb", "wifi_sta", "wifi_sap"]
    )
    p_view.add_argument("--port", type=int, default=6768)
    p_view.add_argument("--batch-ms", type=int, default=0)
    p_view.add_argument("--width", type=int, default=1600)
    p_view.add_argument("--height", type=int, default=900)
    p_view.add_argument(
        "--no-detect", action="store_true", help="show raw sensors without running detection"
    )
    p_view.add_argument("--no-display", action="store_true")
    _add_output(p_view)
    _add_common(p_view)

    # -- listen -----------------------------------------------------------
    p_listen = sub.add_parser(
        "listen", help="print incoming UDP events (a stand-in for your receiver)"
    )
    p_listen.add_argument("--port", type=int, default=8900)
    p_listen.add_argument("--host", type=str, default="0.0.0.0")
    p_listen.add_argument("--raw", action="store_true", help="print raw JSON")

    # -- selftest ----------------------------------------------------------
    p_self = sub.add_parser(
        "selftest", help="emit a scripted hand-approach sequence, no camera or glasses"
    )
    p_self.add_argument("--events-udp", type=str, default="127.0.0.1:8900")

    # -- config -----------------------------------------------------------
    p_cfg = sub.add_parser("dump-config", help="write the default config to a YAML file")
    p_cfg.add_argument("path", type=str)

    return parser


def _resolve_config(args: argparse.Namespace) -> PipelineConfig:
    cfg = load_config(args.config) if getattr(args, "config", None) else PipelineConfig()

    if getattr(args, "backend", None):
        cfg.detector.backend = args.backend
        # Switching backend without new weights would load a YOLO checkpoint
        # into OWLv2, so pick that backend's default.
        if args.backend.lower().startswith("owl") and not getattr(args, "weights", None):
            from .detect.owlv2 import DEFAULT_OWLV2_WEIGHTS

            cfg.detector.weights = DEFAULT_OWLV2_WEIGHTS
    if getattr(args, "weights", None):
        cfg.detector.weights = args.weights
    if getattr(args, "device", None):
        cfg.detector.device = args.device
    if getattr(args, "imgsz", None):
        cfg.detector.imgsz = args.imgsz
    if getattr(args, "conf", None) is not None:
        cfg.detector.conf_threshold = args.conf
    if getattr(args, "hands", None):
        cfg.hands.backend = args.hands
    if getattr(args, "no_track", False):
        cfg.track.enabled = False
    if getattr(args, "draw_signals", False):
        cfg.viz.draw_signals = True
    if getattr(args, "stride", None):
        cfg.stride = args.stride
    if getattr(args, "max_frames", None):
        cfg.max_frames = args.max_frames

    # Rectification overrides (vrs only)
    if getattr(args, "no_rectify", False):
        cfg.rectify.enabled = False
    if getattr(args, "focal", None):
        cfg.rectify.focal = args.focal
    if getattr(args, "rect_size", None):
        cfg.rectify.size = (args.rect_size, args.rect_size)
    if getattr(args, "devignette", False):
        cfg.rectify.devignette = True

    return cfg


def _setup_logging(verbosity: int) -> None:
    level = logging.WARNING
    if verbosity == 1:
        level = logging.INFO
    elif verbosity >= 2:
        level = logging.DEBUG
    logging.basicConfig(
        level=level, format="%(asctime)s %(levelname)-7s %(name)s: %(message)s", datefmt="%H:%M:%S"
    )
    # Ultralytics is extremely chatty at INFO.
    logging.getLogger("ultralytics").setLevel(logging.WARNING)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.command == "listen":
        from .emit import listen

        listen(port=args.port, host=args.host, pretty=not args.raw)
        return 0

    if args.command == "selftest":
        return _run_selftest(args)

    if args.command == "dump-config":
        save_config(PipelineConfig(), args.path)
        print(f"wrote default config to {args.path}")
        return 0

    _setup_logging(args.verbose)
    cfg = _resolve_config(args)

    out_dir = Path(getattr(args, "out", "out"))
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.command == "image":
        return _run_images(cfg, args, out_dir)

    if args.command == "view":
        return _run_view(cfg, args, out_dir)

    stem = Path(getattr(args, "input", args.command)).stem
    mp4_path = str(out_dir / f"{stem}_annotated.mp4") if args.mp4 else None
    jsonl_path = str(out_dir / f"{stem}.jsonl") if args.jsonl else None

    hooks = _build_emitters(args)

    if args.command == "camera":
        from .realtime import run_camera

        run_camera(
            cfg,
            camera_index=args.index,
            width=args.width,
            height=args.height,
            display=not args.no_display,
            mp4_path=mp4_path,
            jsonl_path=jsonl_path,
            max_frames=args.max_frames,
            detect_every_ms=args.detect_every_ms,
            hooks=hooks,
        )
        return 0

    from .runner import run_offline

    if args.command == "video":
        from .sources.video import VideoFrameSource

        source = VideoFrameSource(args.input, stride=cfg.stride, max_frames=cfg.max_frames)
    elif args.command == "vrs":
        from .sources.vrs import VrsFrameSource

        source = VrsFrameSource(
            args.input,
            rectify=cfg.rectify,
            stride=cfg.stride,
            max_frames=cfg.max_frames,
            mps_folder=getattr(args, "mps", None),
        )
    elif args.command == "live":
        from .sources.live import LiveFrameSource

        source = LiveFrameSource(
            serial=args.serial,
            profile=args.profile,
            interface=args.interface,
            port=args.port,
            rectify=cfg.rectify,
            max_frames=cfg.max_frames,
            record_to_vrs=args.record_to_vrs,
            batch_period_ms=args.batch_ms,
        )
    else:  # pragma: no cover - argparse rejects this first
        raise SystemExit(f"unknown command {args.command}")

    run_offline(
        cfg,
        source,
        mp4_path=mp4_path,
        jsonl_path=jsonl_path,
        display=getattr(args, "display", False),
        hooks=hooks,
    )
    return 0


def _run_images(cfg: PipelineConfig, args: argparse.Namespace, out_dir: Path) -> int:
    """Annotate still photos and print what was found, per image.

    Tracking is forced off: consecutive stills are unrelated scenes, and
    temporal label voting across them would be nonsense.
    """
    import cv2

    from .pipeline import DevicePipeline, JsonlWriter
    from .sources.video import ImageFrameSource
    from .viz import draw_frame

    cfg.track.enabled = False
    source = ImageFrameSource(args.input)
    pipeline = DevicePipeline(cfg, focal_px=source.focal_px)
    jsonl = JsonlWriter(out_dir / "images.jsonl") if args.jsonl else None

    total = 0
    try:
        for frame in source:
            result = pipeline.process(frame)
            name = Path(frame.meta["path"]).stem
            canvas = draw_frame(
                frame.rgb, result.detections, cfg.viz,
                hands=result.hands, interactions=result.interactions,
                hud=[f"{name}  {result.detect_ms:.0f} ms"],
            )
            dest = out_dir / f"{name}_annotated.png"
            cv2.imwrite(str(dest), canvas)
            if jsonl is not None:
                jsonl.write(result)

            print(f"\n{name}  ->  {dest}")
            if not result.detections and not result.hands:
                print("  (nothing found)")
            for d in result.detections:
                x1, y1, x2, y2 = (round(v) for v in d.bbox_xyxy)
                extra = ""
                if d.signals.get("diag_cm"):
                    extra = f"  ~{d.signals['diag_cm']:.0f}cm"
                print(f"  {d.label:<7} {d.score:.2f}  [{x1},{y1},{x2},{y2}]{extra}")
            for h, i in zip(result.hands, result.interactions):
                on = f" on {i.device_label}" if i.device_label else ""
                print(f"  hand    {h.side:<5} {i.pose.value}{on}")
            total += 1
    finally:
        if jsonl is not None:
            jsonl.close()
        pipeline.close()

    print(f"\n{total} image(s) written to {out_dir}/")
    return 0


def _run_selftest(args: argparse.Namespace) -> int:
    """Drive the interaction pipeline with a scripted hand path.

    No camera, no glasses, no model weights — a fake detector supplies one
    device box and the hand is walked from far away, onto the screen, and off
    again. Enough to prove your receiver parses the wire format and sees the
    full approach/enter/move/leave lifecycle.
    """
    import numpy as np

    from .config import PipelineConfig
    from .detect.base import Detection, Detector
    from .emit import EventEmitter
    from .frames import Frame, HandSample
    from .pipeline import DevicePipeline

    class _FakeTablet(Detector):
        prompts = ["tablet computer"]

        def detect(self, image_rgb):
            return [Detection((100.0, 50.0, 300.0, 350.0), 0.9, "tablet computer")]

    host, _, port = args.events_udp.rpartition(":")
    emitter = EventEmitter(host or "127.0.0.1", int(port))

    cfg = PipelineConfig()
    cfg.hands.backend = "off"
    cfg.track.enabled = False
    pipeline = DevicePipeline(cfg, detector=_FakeTablet(), focal_px=600.0)
    pipeline.add_result_hook(emitter)
    pipeline._uses_aria_hands = True  # take hands straight off the frame

    image = np.zeros((400, 400, 3), np.uint8)
    image[50:350, 100:300] = 200  # a lit "screen"

    def hand_at(x: float, y: float) -> HandSample:
        pts = [[x, y] for _ in range(21)]
        pts[0] = [x, y + 100.0]
        pts[9] = [x, y + 50.0]
        return HandSample(
            (x - 20, y - 20, x + 20, y + 120), "right", 1.0,
            np.asarray(pts, np.float32), "mediapipe",
        )

    path = [900, 500, 340, 260, 200, 200, 170, 260, 340, 900]
    print(f"sending {len(path)} frames to {host or '127.0.0.1'}:{port}")
    for i, hx in enumerate(path):
        frame = Frame(rgb=image, timestamp_ns=i * 33_000_000, frame_idx=i)
        frame.hands = [hand_at(float(hx), 200.0)]
        result = pipeline.process(frame)
        for event in result.events:
            print("  sent:", event.to_record()["state"], event.to_record().get("x"))
        time.sleep(0.15)
    pipeline.close()
    print(f"done — {emitter.sent} datagram(s) sent")
    return 0


def _run_view(cfg: PipelineConfig, args: argparse.Namespace, out_dir: Path) -> int:
    """Sensor dashboard, from a recording or from the glasses."""
    from .monitor import LiveMonitor, VrsMonitor, run_monitor

    if not args.live and not args.input:
        raise SystemExit("aria-devices view needs either a .vrs path or --live")

    detect = not args.no_detect
    if args.live:
        monitor = LiveMonitor(
            cfg, serial=args.serial, profile=args.profile,
            interface=args.interface, port=args.port, detect=detect,
            max_frames=args.max_frames, batch_period_ms=args.batch_ms,
        )
        fps, stem = 30.0, "live"
    else:
        monitor = VrsMonitor(args.input, cfg, detect=detect)
        fps, stem = monitor.source.fps, Path(args.input).stem

    return run_monitor(
        monitor,
        width=args.width,
        height=args.height,
        display=not args.no_display,
        mp4_path=str(out_dir / f"{stem}_dashboard.mp4") if args.mp4 else None,
        fps=fps,
    )


def _build_emitters(args: argparse.Namespace) -> list:
    """Construct the integration sinks requested on the command line."""
    hooks: list = []
    udp = getattr(args, "udp", None)
    if udp:
        from .emit import UdpEmitter

        host, _, port = udp.rpartition(":")
        if not host or not port.isdigit():
            raise SystemExit(f"--udp expects HOST:PORT, got {udp!r}")
        hooks.append(UdpEmitter(host, int(port)))
    events_udp = getattr(args, "events_udp", None)
    if events_udp:
        from .emit import EventEmitter

        host, _, port = events_udp.rpartition(":")
        if not host or not port.isdigit():
            raise SystemExit(f"--events-udp expects HOST:PORT, got {events_udp!r}")
        hooks.append(EventEmitter(host, int(port)))
    if getattr(args, "emit_stdout", False):
        from .emit import StdoutEmitter

        hooks.append(StdoutEmitter())
    return hooks


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
