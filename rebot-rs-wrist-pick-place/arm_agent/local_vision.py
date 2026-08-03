"""Lazy in-process open-vocabulary vision for the wrist servo.

SAM 3 provides unified text-prompted detection and instance masks. Grounding
DINO + SAM 2.1 and LocateAnything remain explicit compatibility backends.
"""
from __future__ import annotations

import os
import re
import threading
from pathlib import Path
from typing import Any

import cv2
import numpy as np

_BOX = re.compile(r"<box><(\d+)><(\d+)><(\d+)><(\d+)></box>")
ROOT = Path(__file__).resolve().parents[1]


def _hf_token() -> str | None:
    """Read a Hugging Face token without printing or exporting it."""
    for name in ("HF_TOKEN", "HUGGINGFACE_HUB_TOKEN",
                 "HUGGING_FACE_HUB_TOKEN"):
        value = os.getenv(name)
        if value:
            return value
    try:
        for raw in (ROOT / ".env").read_text().splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            if key.strip() in {"HF_TOKEN", "HUGGINGFACE_HUB_TOKEN",
                               "HUGGING_FACE_HUB_TOKEN"}:
                return value.strip().strip('"').strip("'") or None
    except Exception:
        pass
    return None


class LocateAnything:
    """Small local copy of the service's LocateAnything inference surface."""

    def __init__(self) -> None:
        self.model_name = os.getenv("REBOT_LA_MODEL", "nvidia/LocateAnything-3B")
        self.device = os.getenv("REBOT_LA_DEVICE", "cuda")
        self.quantization = os.getenv("REBOT_LA_QUANT", "4bit")
        self.max_size = int(os.getenv("REBOT_LA_MAX_SIZE", "336"))
        self._bundle: dict[str, Any] | None = None
        self._lock = threading.Lock()

    def _load(self) -> dict[str, Any]:
        with self._lock:
            if self._bundle is not None:
                return self._bundle
            import torch
            from transformers import AutoConfig, AutoModel, AutoProcessor, AutoTokenizer

            dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
            config = AutoConfig.from_pretrained(self.model_name, trust_remote_code=True)
            for cfg in (config, getattr(config, "text_config", None), getattr(config, "vision_config", None)):
                if cfg is not None:
                    cfg._attn_implementation = os.getenv("REBOT_LA_ATTN", "sdpa")
            kwargs: dict[str, Any] = {"config": config, "torch_dtype": dtype,
                                      "trust_remote_code": True, "low_cpu_mem_usage": True}
            if self.quantization in {"4bit", "8bit"}:
                from transformers import BitsAndBytesConfig
                kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=self.quantization == "4bit",
                    load_in_8bit=self.quantization == "8bit",
                    bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=dtype,
                    bnb_4bit_use_double_quant=True,
                )
                kwargs["device_map"] = "auto"
            tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
            processor = AutoProcessor.from_pretrained(self.model_name, trust_remote_code=True)
            model = AutoModel.from_pretrained(self.model_name, **kwargs)
            if "device_map" not in kwargs:
                model = model.to(self.device)
            model.eval()
            self._bundle = {"torch": torch, "tokenizer": tokenizer, "processor": processor,
                            "model": model, "dtype": dtype}
            return self._bundle

    def locate(self, frame_bgr: np.ndarray, description: str) -> dict[str, Any]:
        from PIL import Image

        height, width = frame_bgr.shape[:2]
        try:
            b = self._load()
            image = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
            if self.max_size > 0 and max(image.size) > self.max_size:
                scale = self.max_size / max(image.size)
                image = image.resize((round(image.width * scale), round(image.height * scale)), Image.Resampling.LANCZOS)
            messages = [{"role": "user", "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": f"Locate a single instance that matches: {description.strip()}."},
            ]}]
            processor, tokenizer, model, torch = b["processor"], b["tokenizer"], b["model"], b["torch"]
            text = processor.py_apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            images, videos = processor.process_vision_info(messages)
            inputs = processor(text=[text], images=images, videos=videos, return_tensors="pt").to(self.device)
            with torch.inference_mode():
                response = model.generate(pixel_values=inputs["pixel_values"].to(b["dtype"]),
                    input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"],
                    image_grid_hws=inputs.get("image_grid_hws"), tokenizer=tokenizer, max_new_tokens=1024,
                    use_cache=True, generation_mode="hybrid", temperature=0.0, do_sample=False,
                    top_p=0.9, repetition_penalty=1.1, verbose=False)
            answer = response[0] if isinstance(response, (tuple, list)) else response
            answer = str(answer[0] if isinstance(answer, list) else answer)
            boxes = [[int(a) / 1000 * width, int(c) / 1000 * height,
                      int(b_) / 1000 * width, int(d) / 1000 * height]
                     for a, c, b_, d in _BOX.findall(answer)]
            if not boxes:
                return {"ok": True, "found": False, "image_size": [width, height]}
            box = max(boxes, key=lambda v: (v[2] - v[0]) * (v[3] - v[1]))
            return {"ok": True, "found": True, "box": box,
                    "point_px": [(box[0] + box[2]) / 2, (box[1] + box[3]) / 2],
                    "confidence": 0.9, "image_size": [width, height]}
        except Exception as exc:  # inference errors should be a normal detection failure
            return {"ok": False, "found": False, "error": str(exc), "image_size": [width, height]}


class GroundedSam2:
    """Grounding DINO box detector with SAM 2.1 box-prompt segmentation."""

    def __init__(self) -> None:
        self.detector_name = os.getenv(
            "REBOT_GROUNDING_DINO_MODEL", "IDEA-Research/grounding-dino-base"
        )
        self.segmenter_name = os.getenv(
            "REBOT_SAM2_MODEL", "facebook/sam2.1-hiera-large"
        )
        self.device = os.getenv("REBOT_VISION_DEVICE", "cuda")
        self.box_threshold = float(os.getenv("REBOT_GROUNDING_BOX_THRESHOLD", "0.30"))
        self.text_threshold = float(os.getenv("REBOT_GROUNDING_TEXT_THRESHOLD", "0.25"))
        self._bundle: dict[str, Any] | None = None
        self._lock = threading.Lock()

    def _load(self) -> dict[str, Any]:
        with self._lock:
            if self._bundle is not None:
                return self._bundle
            import torch
            from transformers import (
                AutoModelForZeroShotObjectDetection,
                AutoProcessor,
                Sam2Model,
                Sam2Processor,
            )

            # Grounding DINO's backbone receives FP32 processor tensors and some
            # convolution paths do not autocast safely when weights are loaded
            # directly as BF16.  The pair is small enough for this 8GB GPU, so
            # FP32 is the reliable default; an explicit override remains useful
            # for hardware on which mixed precision has been validated.
            dtype_name = os.getenv("REBOT_VISION_DTYPE", "float32").strip().lower()
            dtype = torch.bfloat16 if dtype_name in {"bf16", "bfloat16"} \
                else torch.float32
            det_processor = AutoProcessor.from_pretrained(self.detector_name)
            detector = AutoModelForZeroShotObjectDetection.from_pretrained(
                self.detector_name, dtype=dtype, low_cpu_mem_usage=True
            ).to(self.device).eval()
            sam_processor = Sam2Processor.from_pretrained(self.segmenter_name)
            segmenter = Sam2Model.from_pretrained(
                self.segmenter_name, dtype=dtype, low_cpu_mem_usage=True
            ).to(self.device).eval()
            self._bundle = {
                "torch": torch,
                "dtype": dtype,
                "det_processor": det_processor,
                "detector": detector,
                "sam_processor": sam_processor,
                "segmenter": segmenter,
            }
            return self._bundle

    @staticmethod
    def _image(frame_bgr: np.ndarray):
        from PIL import Image
        return Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))

    def locate(self, frame_bgr: np.ndarray, description: str) -> dict[str, Any]:
        height, width = frame_bgr.shape[:2]
        try:
            b = self._load()
            torch = b["torch"]
            image = self._image(frame_bgr)
            query = description.strip().lower().rstrip(".") + "."
            inputs = b["det_processor"](
                images=image, text=query, return_tensors="pt"
            ).to(self.device)
            with torch.inference_mode():
                outputs = b["detector"](**inputs)
            result = b["det_processor"].post_process_grounded_object_detection(
                outputs,
                inputs.input_ids,
                threshold=self.box_threshold,
                text_threshold=self.text_threshold,
                target_sizes=[image.size[::-1]],
            )[0]
            boxes = result.get("boxes")
            scores = result.get("scores")
            if boxes is None or len(boxes) == 0:
                return {
                    "ok": True, "found": False,
                    "backend": "grounding_dino_sam2",
                    "image_size": [width, height],
                }
            best = int(torch.argmax(scores).item()) if scores is not None else 0
            box = [float(v) for v in boxes[best].detach().cpu().tolist()]
            score = float(scores[best].detach().cpu()) if scores is not None else 0.0
            return {
                "ok": True, "found": True, "box": box,
                "point_px": [(box[0] + box[2]) / 2, (box[1] + box[3]) / 2],
                "confidence": score, "backend": "grounding_dino_sam2",
                "image_size": [width, height],
            }
        except Exception as exc:
            return {
                "ok": False, "found": False, "error": str(exc),
                "backend": "grounding_dino_sam2", "image_size": [width, height],
            }

    def segment_box(self, frame_bgr: np.ndarray,
                    box: tuple[int, int, int, int]) -> np.ndarray | None:
        """Return the highest-quality SAM mask for one DINO box."""
        try:
            b = self._load()
            torch = b["torch"]
            image = self._image(frame_bgr)
            inputs = b["sam_processor"](
                images=image,
                input_boxes=[[[float(v) for v in box]]],
                return_tensors="pt",
            ).to(self.device)
            with torch.inference_mode():
                outputs = b["segmenter"](**inputs, multimask_output=True)
            masks = b["sam_processor"].post_process_masks(
                outputs.pred_masks.detach().cpu(), inputs["original_sizes"].cpu()
            )[0]
            scores = outputs.iou_scores.detach().float().cpu()
            best = int(torch.argmax(scores.reshape(-1)).item())
            candidates = masks.reshape(-1, masks.shape[-2], masks.shape[-1])
            best = min(best, candidates.shape[0] - 1)
            mask = candidates[best].numpy()
            return (mask > 0).astype(np.uint8) * 255
        except Exception:
            return None


class Sam3TextSegmenter:
    """SAM3 text-prompted instance detection and segmentation in one model."""

    def __init__(self) -> None:
        self.model_name = os.getenv("REBOT_SAM3_MODEL", "facebook/sam3")
        self.device = os.getenv("REBOT_VISION_DEVICE", "cuda")
        self.score_threshold = float(os.getenv("REBOT_SAM3_SCORE_THRESHOLD", "0.30"))
        self.mask_threshold = float(os.getenv("REBOT_SAM3_MASK_THRESHOLD", "0.50"))
        self._bundle: dict[str, Any] | None = None
        self._lock = threading.Lock()
        self._last_mask: np.ndarray | None = None
        self._last_box: tuple[int, int, int, int] | None = None

    def _load(self) -> dict[str, Any]:
        with self._lock:
            if self._bundle is not None:
                return self._bundle
            import torch
            from transformers import Sam3Model, Sam3Processor

            dtype_name = os.getenv("REBOT_VISION_DTYPE", "bfloat16").lower()
            dtype = torch.bfloat16 if self.device == "cuda" and dtype_name in {
                "bf16", "bfloat16"
            } else torch.float32
            token = _hf_token()
            processor = Sam3Processor.from_pretrained(
                self.model_name, token=token
            )
            model = Sam3Model.from_pretrained(
                self.model_name, token=token, dtype=dtype,
                low_cpu_mem_usage=True,
            ).to(self.device).eval()
            self._bundle = {
                "torch": torch, "dtype": dtype,
                "processor": processor, "model": model,
            }
            return self._bundle

    def locate(self, frame_bgr: np.ndarray, description: str) -> dict[str, Any]:
        height, width = frame_bgr.shape[:2]
        self._last_mask = None
        self._last_box = None
        try:
            b = self._load()
            image = GroundedSam2._image(frame_bgr)
            inputs = b["processor"](
                images=image, text=description.strip(), return_tensors="pt"
            ).to(self.device)
            # Image tensors should match model weights; token tensors remain int.
            for key in ("pixel_values",):
                if key in inputs:
                    inputs[key] = inputs[key].to(dtype=b["dtype"])
            with b["torch"].inference_mode():
                outputs = b["model"](**inputs)
            result = b["processor"].post_process_instance_segmentation(
                outputs,
                threshold=self.score_threshold,
                mask_threshold=self.mask_threshold,
                target_sizes=[(height, width)],
            )[0]
            scores, boxes, masks = (
                result.get("scores"), result.get("boxes"), result.get("masks")
            )
            if scores is None or boxes is None or masks is None or len(scores) == 0:
                return {
                    "ok": True, "found": False, "backend": "sam3",
                    "image_size": [width, height],
                }
            best = int(b["torch"].argmax(scores).item())
            box_values = boxes[best].detach().float().cpu().tolist()
            x1, y1, x2, y2 = (
                int(round(float(value))) for value in box_values
            )
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(width, x2), min(height, y2)
            if x2 <= x1 or y2 <= y1:
                return {"ok": True, "found": False, "backend": "sam3",
                        "image_size": [width, height]}
            mask = masks[best].detach().cpu().numpy().astype(np.uint8) * 255
            self._last_box = (x1, y1, x2, y2)
            self._last_mask = mask
            score = float(scores[best].detach().float().cpu())
            return {
                "ok": True, "found": True,
                "box": [x1, y1, x2, y2],
                "point_px": [(x1 + x2) / 2.0, (y1 + y2) / 2.0],
                "confidence": score, "backend": "sam3",
                "instances": int(len(scores)),
                "image_size": [width, height],
            }
        except Exception as exc:
            return {
                "ok": False, "found": False, "error": str(exc),
                "backend": "sam3", "image_size": [width, height],
            }

    def segment_box(self, frame_bgr: np.ndarray,
                    box: tuple[int, int, int, int]) -> np.ndarray | None:
        """Return the native mask from the immediately preceding text query."""
        if self._last_mask is None or self._last_box is None:
            return None
        a = self._last_box
        intersection = max(0, min(a[2], box[2]) - max(a[0], box[0])) * \
            max(0, min(a[3], box[3]) - max(a[1], box[1]))
        union = ((a[2] - a[0]) * (a[3] - a[1])
                 + (box[2] - box[0]) * (box[3] - box[1]) - intersection)
        return self._last_mask.copy() if intersection / max(1, union) >= 0.5 else None


_detector: LocateAnything | GroundedSam2 | Sam3TextSegmenter | None = None


def locate(frame_bgr: np.ndarray, description: str) -> dict[str, Any]:
    # Heavy vision remains opt-in so an ordinary read-only state command cannot
    # unexpectedly allocate GPU memory or download checkpoints.
    if os.getenv("REBOT_ENABLE_LOCATE_ANYTHING", "0").strip().lower() not in {"1", "true", "yes", "on"}:
        h, w = frame_bgr.shape[:2]
        return {"ok": False, "found": False, "image_size": [w, h],
                "error": "LocateAnything disabled; set REBOT_ENABLE_LOCATE_ANYTHING=1 to load it"}
    global _detector
    if _detector is None:
        backend = os.getenv("REBOT_VISION_BACKEND", "grounded_sam2").strip().lower()
        if backend == "locate_anything":
            _detector = LocateAnything()
        elif backend == "sam3":
            _detector = Sam3TextSegmenter()
        else:
            _detector = GroundedSam2()
    return _detector.locate(frame_bgr, description)


def segment_box(frame_bgr: np.ndarray,
                box: tuple[int, int, int, int]) -> np.ndarray | None:
    """Segment the most recently selected box when the backend supports masks."""
    global _detector
    segment = getattr(_detector, "segment_box", None)
    return segment(frame_bgr, box) if callable(segment) else None
