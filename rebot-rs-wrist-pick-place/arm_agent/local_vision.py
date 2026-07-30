"""Lazy in-process LocateAnything detector for the single-process servo path."""
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


_detector: LocateAnything | None = None


def locate(frame_bgr: np.ndarray, description: str) -> dict[str, Any]:
    # LocateAnything is a 3B remote-code model.  Do not silently download/load it
    # from a visual-servo command: that can exhaust a shared robot workstation.
    # It is intentionally opt-in after its model cache has been prepared.
    if os.getenv("REBOT_ENABLE_LOCATE_ANYTHING", "0").strip().lower() not in {"1", "true", "yes", "on"}:
        h, w = frame_bgr.shape[:2]
        return {"ok": False, "found": False, "image_size": [w, h],
                "error": "LocateAnything disabled; set REBOT_ENABLE_LOCATE_ANYTHING=1 to load it"}
    global _detector
    if _detector is None:
        _detector = LocateAnything()
    return _detector.locate(frame_bgr, description)
