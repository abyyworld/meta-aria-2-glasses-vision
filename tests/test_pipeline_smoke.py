"""End-to-end smoke tests.

These assert the pipeline *runs* and emits well-formed output — never that it
detects anything. Detection quality on synthetic rectangles is meaningless, and
a test that asserted it would be a test that fails for the wrong reasons.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from aria_devices.config import PipelineConfig, load_config, save_config
from aria_devices.detect.base import Detection, Detector
from aria_devices.frames import Frame
from aria_devices.pipeline import DevicePipeline, JsonlWriter
from aria_devices.sources.video import VideoFrameSource, focal_from_hfov


# --------------------------------------------------------------------------
class FakeDetector(Detector):
    """Returns fixed boxes, so the pipeline is testable with no weights."""

    def __init__(self, prompts=None) -> None:
        self.prompts = prompts or ["laptop", "tablet computer", "smartphone"]
        self.calls = 0

    def detect(self, image_rgb: np.ndarray) -> list[Detection]:
        self.calls += 1
        h, w = image_rgb.shape[:2]
        return [
            Detection((0.10 * w, 0.30 * h, 0.45 * w, 0.75 * h), 0.82, "laptop"),
            Detection((0.55 * w, 0.20 * h, 0.72 * w, 0.80 * h), 0.61, "tablet computer"),
            Detection((0.78 * w, 0.55 * h, 0.95 * w, 0.68 * h), 0.55, "smartphone"),
        ]


@pytest.fixture
def clip(tmp_path: Path) -> Path:
    """A short generated clip of moving coloured rectangles."""
    path = tmp_path / "synthetic.mp4"
    w, h, fps, n = 320, 240, 10, 12
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    assert writer.isOpened()
    for i in range(n):
        img = np.full((h, w, 3), 30, np.uint8)
        dx = 2 * i
        cv2.rectangle(img, (30 + dx, 70), (140 + dx, 180), (210, 210, 215), -1)
        cv2.rectangle(img, (180, 50), (230, 190), (190, 190, 200), -1)
        cv2.rectangle(img, (250, 130), (300, 160), (120, 120, 130), -1)
        writer.write(img)
    writer.release()
    assert path.exists() and path.stat().st_size > 0
    return path


@pytest.fixture
def cfg() -> PipelineConfig:
    c = PipelineConfig()
    c.hands.backend = "off"  # no model download inside the test suite
    return c


# --------------------------------------------------------------------- source
class TestVideoSource:
    def test_reads_every_frame(self, clip):
        source = VideoFrameSource(str(clip))
        frames = list(source)
        source.close()
        assert len(frames) == 12
        assert all(isinstance(f, Frame) for f in frames)

    def test_frames_are_rgb_uint8(self, clip):
        source = VideoFrameSource(str(clip))
        frame = next(iter(source))
        source.close()
        assert frame.rgb.dtype == np.uint8
        assert frame.rgb.ndim == 3 and frame.rgb.shape[2] == 3

    def test_stride_skips_frames(self, clip):
        source = VideoFrameSource(str(clip), stride=3)
        n = len(list(source))
        source.close()
        assert n == 4

    def test_max_frames_caps_output(self, clip):
        source = VideoFrameSource(str(clip), max_frames=5)
        n = len(list(source))
        source.close()
        assert n == 5

    def test_frame_indices_are_sequential(self, clip):
        source = VideoFrameSource(str(clip))
        idxs = [f.frame_idx for f in source]
        source.close()
        assert idxs == list(range(len(idxs)))

    def test_file_is_not_mirrored(self, clip):
        source = VideoFrameSource(str(clip))
        assert source.mirror is False
        source.close()

    def test_missing_file_raises(self):
        with pytest.raises(RuntimeError):
            VideoFrameSource("/nonexistent/nope.mp4")

    def test_focal_from_hfov_is_sane(self):
        f = focal_from_hfov(1280, 58.0)
        assert 1000 < f < 1400


# ------------------------------------------------------------------ pipeline
class TestPipelineEndToEnd:
    def test_runs_over_a_clip_and_produces_results(self, clip, cfg):
        source = VideoFrameSource(str(clip))
        pipeline = DevicePipeline(cfg, detector=FakeDetector(), focal_px=source.focal_px)
        results = list(pipeline.run(source))
        pipeline.close()
        source.close()
        assert len(results) == 12

    def test_records_are_well_formed(self, clip, cfg):
        source = VideoFrameSource(str(clip))
        pipeline = DevicePipeline(cfg, detector=FakeDetector(), focal_px=source.focal_px)
        for result in pipeline.run(source):
            record = result.to_record()
            assert set(record) >= {
                "frame_idx", "timestamp_ns", "detections", "hands", "interactions",
            }
            assert isinstance(record["frame_idx"], int)
            assert isinstance(record["timestamp_ns"], int)
            for d in record["detections"]:
                assert set(d) >= {
                    "track_id", "label", "score", "bbox_xyxy", "gazed_at", "signals",
                }
                assert len(d["bbox_xyxy"]) == 4
                assert d["label"] in ("laptop", "tablet", "phone")
                assert 0.0 <= d["score"] <= 1.0
        pipeline.close()
        source.close()

    def test_jsonl_output_is_parseable(self, clip, cfg, tmp_path):
        out = tmp_path / "out.jsonl"
        source = VideoFrameSource(str(clip))
        pipeline = DevicePipeline(cfg, detector=FakeDetector(), focal_px=source.focal_px)
        with JsonlWriter(out) as writer:
            for result in pipeline.run(source):
                writer.write(result)
        pipeline.close()
        source.close()

        lines = out.read_text().strip().splitlines()
        assert len(lines) == 12
        for line in lines:
            json.loads(line)  # must not raise

    def test_tracking_assigns_stable_ids(self, clip, cfg):
        source = VideoFrameSource(str(clip))
        pipeline = DevicePipeline(cfg, detector=FakeDetector(), focal_px=source.focal_px)
        seen: set[int] = set()
        for result in pipeline.run(source):
            for d in result.detections:
                if d.track_id is not None:
                    seen.add(d.track_id)
        pipeline.close()
        source.close()
        # Three static-ish objects; ID churn would blow this well past 3.
        assert 0 < len(seen) <= 4, f"unexpected track id churn: {sorted(seen)}"

    def test_result_hook_fires_for_every_frame(self, clip, cfg):
        source = VideoFrameSource(str(clip))
        pipeline = DevicePipeline(cfg, detector=FakeDetector(), focal_px=source.focal_px)
        received = []
        pipeline.add_result_hook(received.append)
        list(pipeline.run(source))
        pipeline.close()
        source.close()
        assert len(received) == 12

    def test_a_raising_hook_does_not_kill_the_run(self, clip, cfg):
        def bad_hook(_result):
            raise RuntimeError("downstream consumer exploded")

        source = VideoFrameSource(str(clip))
        pipeline = DevicePipeline(cfg, detector=FakeDetector(), focal_px=source.focal_px)
        pipeline.add_result_hook(bad_hook)
        results = list(pipeline.run(source))
        pipeline.close()
        source.close()
        assert len(results) == 12


# ------------------------------------------------------------------- runner
class TestRunnerOutputs:
    def test_writes_mp4_and_jsonl(self, clip, cfg, tmp_path, monkeypatch):
        import aria_devices.runner as runner

        monkeypatch.setattr(runner, "DevicePipeline", _pipeline_with_fake_detector(cfg))
        mp4 = tmp_path / "annotated.mp4"
        jsonl = tmp_path / "records.jsonl"
        source = VideoFrameSource(str(clip))
        count = runner.run_offline(cfg, source, mp4_path=str(mp4), jsonl_path=str(jsonl))

        assert count == 12
        assert mp4.exists() and mp4.stat().st_size > 0
        assert len(jsonl.read_text().strip().splitlines()) == 12


def _pipeline_with_fake_detector(_cfg):
    def factory(cfg, detector=None, focal_px=None, has_aria_hands=False):
        return DevicePipeline(
            cfg, detector=FakeDetector(), focal_px=focal_px, has_aria_hands=has_aria_hands
        )

    return factory


# -------------------------------------------------------------------- config
class TestConfigRoundTrip:
    def test_yaml_round_trip_preserves_values(self, tmp_path):
        cfg = PipelineConfig()
        cfg.detector.backend = "owlv2"
        cfg.detector.imgsz = 512
        cfg.track.vote_window = 21
        cfg.disambiguation.w_text = 0.5
        path = tmp_path / "cfg.yaml"
        save_config(cfg, path)

        loaded = load_config(path)
        assert loaded.detector.backend == "owlv2"
        assert loaded.detector.imgsz == 512
        assert loaded.track.vote_window == 21
        assert loaded.disambiguation.w_text == 0.5

    def test_device_profiles_survive_round_trip(self, tmp_path):
        cfg = PipelineConfig()
        path = tmp_path / "cfg.yaml"
        save_config(cfg, path)
        loaded = load_config(path)
        phone = loaded.disambiguation.device_profiles["phone"]
        assert phone.orientation == "landscape"
        assert tuple(phone.diag_cm) == (12.0, 19.5)

    def test_partial_config_keeps_defaults(self, tmp_path):
        path = tmp_path / "partial.yaml"
        path.write_text("detector:\n  imgsz: 320\n")
        loaded = load_config(path)
        assert loaded.detector.imgsz == 320
        assert loaded.detector.backend == "yoloworld"  # untouched default

    def test_unknown_keys_are_ignored(self, tmp_path):
        path = tmp_path / "future.yaml"
        path.write_text("detector:\n  imgsz: 320\n  invented_option: 5\n")
        assert load_config(path).detector.imgsz == 320


# ----------------------------------------------------------------------- CLI
class TestCli:
    def test_parser_accepts_every_subcommand(self):
        from aria_devices.cli import build_parser

        parser = build_parser()
        for argv in (
            ["camera"],
            ["video", "in.mp4", "--out", "o", "--mp4", "--jsonl"],
            ["vrs", "in.vrs", "--out", "o", "--max-frames", "10"],
            ["live", "--interface", "usb"],
        ):
            assert parser.parse_args(argv) is not None

    def test_dump_config_writes_loadable_yaml(self, tmp_path):
        from aria_devices.cli import main

        path = tmp_path / "default.yaml"
        assert main(["dump-config", str(path)]) == 0
        assert load_config(path).detector.backend == "yoloworld"

    def test_udp_flag_is_parsed(self):
        from aria_devices.cli import build_parser

        args = build_parser().parse_args(["camera", "--udp", "127.0.0.1:8899"])
        assert args.udp == "127.0.0.1:8899"
