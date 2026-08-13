"""OWLv2 open-vocabulary backend (HuggingFace transformers).

Slower than YOLO-World on CPU by roughly an order of magnitude, but noticeably
better at long descriptive prompts ("mobile phone held in hand"), which is
exactly where the three-way device distinction is hardest. Worth having as a
swappable comparison, not as the realtime default.
"""

from __future__ import annotations

import logging

import numpy as np

from ..config import DetectorConfig, resolve_device
from .base import Detection, Detector

log = logging.getLogger(__name__)

DEFAULT_OWLV2_WEIGHTS = "google/owlv2-base-patch16-ensemble"


class Owlv2Detector(Detector):
    """OWLv2 behind the common Detector interface."""

    def __init__(self, cfg: DetectorConfig) -> None:
        try:
            import torch
            from transformers import Owlv2ForObjectDetection, Owlv2Processor
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise ImportError(
                "transformers and torch are required for the owlv2 backend: "
                "pip install 'aria-devices[owlv2]'"
            ) from exc

        self.cfg = cfg
        self.prompts = list(cfg.prompts)
        self.device = resolve_device(cfg.device)
        self._torch = torch

        weights = cfg.weights if "owlv2" in cfg.weights else DEFAULT_OWLV2_WEIGHTS
        log.info("loading owlv2 weights=%s device=%s", weights, self.device)
        self.processor = Owlv2Processor.from_pretrained(weights)
        self.model = Owlv2ForObjectDetection.from_pretrained(weights).to(self.device).eval()

        # OWLv2 wants natural-language queries; the prompt list is already that.
        self._queries = [[f"a photo of a {p}" for p in self.prompts]]

    def detect(self, image_rgb: np.ndarray) -> list[Detection]:
        torch = self._torch
        height, width = image_rgb.shape[:2]

        inputs = self.processor(
            text=self._queries, images=image_rgb, return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        # post_process_grounded_object_detection replaced the older
        # post_process_object_detection in recent transformers; fall back so
        # this works across versions.
        target_sizes = torch.tensor([[height, width]], device=self.device)
        postprocess = getattr(
            self.processor,
            "post_process_grounded_object_detection",
            getattr(self.processor, "post_process_object_detection", None),
        )
        if postprocess is None:  # pragma: no cover
            raise RuntimeError("Owlv2Processor exposes no known post-processing method")

        results = postprocess(
            outputs=outputs, threshold=self.cfg.conf_threshold, target_sizes=target_sizes
        )[0]

        out: list[Detection] = []
        for score, label_idx, box in zip(results["scores"], results["labels"], results["boxes"]):
            idx = int(label_idx)
            if idx >= len(self.prompts):
                continue
            x1, y1, x2, y2 = (float(v) for v in box.tolist())
            out.append(
                Detection(
                    bbox_xyxy=(x1, y1, x2, y2),
                    score=float(score),
                    raw_label=self.prompts[idx],
                )
            )
        return out
