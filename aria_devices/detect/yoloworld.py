"""Ultralytics YOLO-World / YOLOE open-vocabulary backend.

This is the default. COCO has ``laptop`` and ``cell phone`` but no tablet class
at all, so a closed-vocabulary model structurally cannot do this task — it will
call every tablet a laptop or a TV. An open-vocabulary model is the only way to
get a "tablet" hypothesis without training one.
"""

from __future__ import annotations

import logging

import numpy as np

from ..config import DetectorConfig, resolve_device
from .base import Detection, Detector

log = logging.getLogger(__name__)


class YoloWorldDetector(Detector):
    """YOLO-World behind the common Detector interface.

    Text embeddings for the prompt set are computed once in ``set_classes`` at
    construction, not per frame — doing it per frame costs more than the
    detection itself.
    """

    def __init__(self, cfg: DetectorConfig) -> None:
        try:
            from ultralytics import YOLO, YOLOWorld
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise ImportError(
                "ultralytics is required for the yoloworld backend: "
                "pip install 'aria-devices[detect]'"
            ) from exc

        self.cfg = cfg
        self.prompts = list(cfg.prompts)
        self.device = resolve_device(cfg.device)

        # YOLOE uses the generic YOLO loader; YOLO-World has its own class.
        loader = YOLO if cfg.backend == "yoloe" else YOLOWorld
        log.info("loading %s weights=%s device=%s", cfg.backend, cfg.weights, self.device)
        self.model = loader(cfg.weights)

        if hasattr(self.model, "set_classes"):
            if cfg.backend == "yoloe" and hasattr(self.model, "get_text_pe"):
                # YOLOE's set_classes takes precomputed prompt embeddings.
                self.model.set_classes(self.prompts, self.model.get_text_pe(self.prompts))
            else:
                self.model.set_classes(self.prompts)
        else:  # pragma: no cover - would mean a very old ultralytics
            raise RuntimeError(f"{type(self.model).__name__} has no set_classes()")

        self._names = self._resolve_names()

    def _resolve_names(self) -> dict[int, str]:
        names = getattr(self.model, "names", None)
        if isinstance(names, dict) and names:
            return {int(k): str(v) for k, v in names.items()}
        if isinstance(names, (list, tuple)) and names:
            return {i: str(v) for i, v in enumerate(names)}
        return dict(enumerate(self.prompts))

    def warmup(self) -> None:
        dummy = np.zeros((self.cfg.imgsz, self.cfg.imgsz, 3), dtype=np.uint8)
        try:
            self.detect(dummy)
        except Exception as exc:  # pragma: no cover - best effort only
            log.warning("warmup failed: %s", exc)

    def detect(self, image_rgb: np.ndarray) -> list[Detection]:
        # Ultralytics treats a raw ndarray as BGR (the cv2 convention). Our
        # Frame contract is RGB, so flip here rather than making every caller
        # remember.
        image_bgr = image_rgb[:, :, ::-1]

        results = self.model.predict(
            image_bgr,
            conf=self.cfg.conf_threshold,
            iou=self.cfg.iou_threshold,
            imgsz=self.cfg.imgsz,
            device=self.device,
            max_det=self.cfg.max_det,
            half=self.cfg.half and self.device == "cuda",
            verbose=False,
        )
        if not results:
            return []

        result = results[0]
        boxes = getattr(result, "boxes", None)
        if boxes is None or len(boxes) == 0:
            return []

        names = getattr(result, "names", None) or self._names
        if isinstance(names, (list, tuple)):
            names = dict(enumerate(names))

        xyxy = boxes.xyxy.cpu().numpy()
        conf = boxes.conf.cpu().numpy()
        cls = boxes.cls.cpu().numpy().astype(int)

        out: list[Detection] = []
        for (x1, y1, x2, y2), score, class_idx in zip(xyxy, conf, cls):
            raw = str(names.get(int(class_idx), self.prompts[int(class_idx) % len(self.prompts)]))
            out.append(
                Detection(
                    bbox_xyxy=(float(x1), float(y1), float(x2), float(y2)),
                    score=float(score),
                    raw_label=raw,
                )
            )
        return out
