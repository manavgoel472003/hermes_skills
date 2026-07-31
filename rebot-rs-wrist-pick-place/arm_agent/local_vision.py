"""Lazy in-process open-vocabulary vision for the wrist servo.

The default backend is purpose-built Grounding DINO + SAM 2.1: DINO finds the
named whole-object box and SAM turns that box into a contour.  LocateAnything is
kept as an explicit compatibility fallback, but a 3B generative VLM is no longer
the default detector for geometric grasping.
"""
from __future__ import annotations

import os
import re
import threading
from typing import Any

import cv2
import numpy as np

_BOX = re.compile(r"<box><(\d+)><(\d+)><(\d+)><(\d+)></box>")


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
            "REBOT_SAM2_MODEL", "facebook/sam2.1-hiera-small"
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
                self.detector_name, torch_dtype=dtype, low_cpu_mem_usage=True
            ).to(self.device).eval()
            sam_processor = Sam2Processor.from_pretrained(self.segmenter_name)
            segmenter = Sam2Model.from_pretrained(
                self.segmenter_name, torch_dtype=dtype, low_cpu_mem_usage=True
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


_detector: LocateAnything | GroundedSam2 | None = None


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
        _detector = LocateAnything() if backend == "locate_anything" else GroundedSam2()
    return _detector.locate(frame_bgr, description)


def segment_box(frame_bgr: np.ndarray,
                box: tuple[int, int, int, int]) -> np.ndarray | None:
    """Segment the most recently selected box when the backend supports masks."""
    global _detector
    segment = getattr(_detector, "segment_box", None)
    return segment(frame_bgr, box) if callable(segment) else None
