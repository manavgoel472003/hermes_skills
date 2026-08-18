"""Lazy in-process open-vocabulary vision for the wrist servo.

SAM 3 provides unified text-prompted detection and instance masks. Grounding
DINO + SAM 2.1 and LocateAnything remain explicit compatibility backends.
"""
from __future__ import annotations

import os
import re
import threading
import gc
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
        self._lock = threading.RLock()

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
    """SAM3 text-prompted segmentation, optionally with instance tracking.

    ``REBOT_VISION_TRACKER=sam3_video`` selects the checkpoint's streaming
    detector/tracker instead of treating every wrist frame as an unrelated
    image.  Once an object id is selected, a later frame is never allowed to
    silently switch to a different matching instance.  Losing that id is a
    normal fail-closed detection result; the wrist workflow may then use its
    already validated frozen path or abort.
    """

    def __init__(self) -> None:
        self.model_name = os.getenv("REBOT_SAM3_MODEL", "facebook/sam3")
        self.device = os.getenv("REBOT_VISION_DEVICE", "cuda")
        self.score_threshold = float(os.getenv("REBOT_SAM3_SCORE_THRESHOLD", "0.30"))
        self.mask_threshold = float(os.getenv("REBOT_SAM3_MASK_THRESHOLD", "0.50"))
        self.tracker = os.getenv("REBOT_VISION_TRACKER", "none").strip().lower()
        self.track_score_threshold = float(os.getenv(
            "REBOT_SAM3_TRACK_SCORE_THRESHOLD", "0.15"
        ))
        self._bundle: dict[str, Any] | None = None
        self._lock = threading.Lock()
        self._inference_lock = threading.Lock()
        self._last_mask: np.ndarray | None = None
        self._last_box: tuple[int, int, int, int] | None = None
        self._video_session: Any | None = None
        self._video_descriptions: set[str] = set()
        self._video_size: tuple[int, int] | None = None
        self._active_object_ids: dict[str, int] = {}
        self._last_results: dict[str, dict[str, Any]] = {}

    def _load(self) -> dict[str, Any]:
        with self._lock:
            if self._bundle is not None:
                return self._bundle
            import torch
            if self.tracker in {"sam3_video", "video"}:
                from transformers import Sam3VideoModel, Sam3VideoProcessor
                model_type, processor_type = Sam3VideoModel, Sam3VideoProcessor
            else:
                from transformers import Sam3Model, Sam3Processor
                model_type, processor_type = Sam3Model, Sam3Processor

            dtype_name = os.getenv("REBOT_VISION_DTYPE", "bfloat16").lower()
            dtype = torch.bfloat16 if self.device == "cuda" and dtype_name in {
                "bf16", "bfloat16"
            } else torch.float32
            token = _hf_token()
            processor = processor_type.from_pretrained(
                self.model_name, token=token
            )
            model = model_type.from_pretrained(
                self.model_name, token=token, dtype=dtype,
                low_cpu_mem_usage=True,
            ).to(self.device).eval()
            self._bundle = {
                "torch": torch, "dtype": dtype,
                "processor": processor, "model": model,
            }
            return self._bundle

    @staticmethod
    def _select_tracked_instance(
        result: dict[str, Any], description: str,
        active_object_id: int | None,
    ) -> tuple[int | None, int | None]:
        """Return ``(row, object_id)`` without changing a locked identity."""
        object_ids_value = result.get("object_ids")
        if object_ids_value is None:
            return None, active_object_id
        object_ids = [int(value) for value in object_ids_value.tolist()]
        if active_object_id is not None:
            try:
                return object_ids.index(int(active_object_id)), active_object_id
            except ValueError:
                return None, active_object_id

        prompt_ids = result.get("prompt_to_obj_ids", {}).get(description, [])
        allowed = {int(value) for value in prompt_ids} if prompt_ids \
            else set(object_ids)
        candidates = [
            index for index, object_id in enumerate(object_ids)
            if object_id in allowed
        ]
        if not candidates:
            return None, None
        scores = result.get("scores")
        if scores is None:
            row = candidates[0]
        else:
            row = max(candidates, key=lambda index: float(scores[index]))
        return row, object_ids[row]

    def _reset_video_session(
        self, bundle: dict[str, Any], image_size: tuple[int, int],
    ) -> None:
        processor = bundle["processor"]
        self._video_session = processor.init_video_session(
            inference_device=self.device,
            inference_state_device=self.device,
            processing_device=self.device,
            # Only the current frame pixels need to remain available. Tracker
            # memories, not an unbounded RGB history, carry the identity.
            video_storage_device="cpu",
            max_vision_features_cache_size=1,
            dtype=bundle["dtype"],
        )
        self._video_descriptions.clear()
        self._video_size = image_size
        self._active_object_ids.clear()
        self._last_results.clear()

    def _remember_result(
        self, description: str, result: dict[str, Any]
    ) -> dict[str, Any]:
        self._last_results[description] = result
        return result

    def tracking_lost(self, description: str) -> bool:
        normalized = " ".join(description.strip().split())
        result = self._last_results.get(normalized, {})
        return bool(result.get("tracking_lost"))

    def _locate_video(
        self, frame_bgr: np.ndarray, description: str,
    ) -> dict[str, Any]:
        height, width = frame_bgr.shape[:2]
        self._last_mask = None
        self._last_box = None
        try:
            bundle = self._load()
            image = GroundedSam2._image(frame_bgr)
            normalized_description = " ".join(description.strip().split())
            image_size = (width, height)
            with self._inference_lock:
                if self._video_session is None \
                        or image_size != self._video_size:
                    self._reset_video_session(bundle, image_size)
                if normalized_description not in self._video_descriptions:
                    bundle["processor"].add_text_prompt(
                        self._video_session, normalized_description
                    )
                    self._video_descriptions.add(normalized_description)
                inputs = bundle["processor"](
                    images=image, return_tensors="pt"
                )
                pixels = inputs["pixel_values"].to(dtype=bundle["dtype"])
                with bundle["torch"].inference_mode():
                    outputs = bundle["model"](
                        inference_session=self._video_session,
                        frame=pixels,
                    )
                result = bundle["processor"].postprocess_outputs(
                    self._video_session, outputs,
                    original_sizes=inputs["original_sizes"],
                )
                # Streaming adds preprocessed pixels to the session. They are
                # not needed after this forward pass and otherwise make a long
                # --watch session grow without bound on host memory, including
                # on frames where the target is temporarily lost.
                current_index = int(outputs.frame_idx)
                processed = getattr(
                    self._video_session, "processed_frames", None
                )
                if isinstance(processed, dict):
                    for old_index in list(processed):
                        if int(old_index) != current_index:
                            del processed[old_index]
                row, selected_id = self._select_tracked_instance(
                    result, normalized_description,
                    self._active_object_ids.get(normalized_description),
                )
                if row is None:
                    active_id = self._active_object_ids.get(
                        normalized_description
                    )
                    return self._remember_result(normalized_description, {
                        "ok": True, "found": False, "backend": "sam3",
                        "tracker": "sam3_video",
                        "tracking_lost": active_id is not None,
                        "active_object_id": active_id,
                        "image_size": [width, height],
                    })
                score = float(result["scores"][row])
                threshold = self.track_score_threshold \
                    if normalized_description in self._active_object_ids \
                    else self.score_threshold
                if score < threshold:
                    return self._remember_result(normalized_description, {
                        "ok": True, "found": False, "backend": "sam3",
                        "tracker": "sam3_video", "tracking_lost": True,
                        "active_object_id": self._active_object_ids.get(
                            normalized_description
                        ),
                        "confidence": score, "image_size": [width, height],
                    })
                self._active_object_ids[normalized_description] = int(
                    selected_id
                )
                mask = result["masks"][row].detach().cpu().numpy() \
                    .astype(np.uint8) * 255
                box_values = result["boxes"][row].detach().float().cpu().tolist()
                x1, y1, x2, y2 = (
                    int(round(float(value))) for value in box_values
                )
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(width, x2), min(height, y2)
                if x2 <= x1 or y2 <= y1 or not np.any(mask):
                    return self._remember_result(normalized_description, {
                        "ok": True, "found": False, "backend": "sam3",
                        "tracker": "sam3_video", "tracking_lost": True,
                        "active_object_id": self._active_object_ids.get(
                            normalized_description
                        ),
                        "image_size": [width, height],
                    })
                self._last_box = (x1, y1, x2, y2)
                self._last_mask = mask

                return self._remember_result(normalized_description, {
                    "ok": True, "found": True,
                    "box": [x1, y1, x2, y2],
                    "point_px": [(x1 + x2) / 2.0, (y1 + y2) / 2.0],
                    "confidence": score, "backend": "sam3",
                    "tracker": "sam3_video",
                    "active_object_id": self._active_object_ids[
                        normalized_description
                    ],
                    "instances": int(len(result["object_ids"])),
                    "image_size": [width, height],
                })
        except Exception as exc:
            return {
                "ok": False, "found": False, "error": str(exc),
                "backend": "sam3", "tracker": "sam3_video",
                "image_size": [width, height],
            }

    def locate(self, frame_bgr: np.ndarray, description: str) -> dict[str, Any]:
        if self.tracker in {"sam3_video", "video"}:
            return self._locate_video(frame_bgr, description)
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


class Sam31TextSegmenter:
    """Native SAM 3.1 text segmentation for one wrist frame at a time.

    The 8 GB RTX 4070 can run SAM 3.1 image grounding, but Meta's full video
    propagation exceeds the available VRAM.  This backend therefore uses the
    improved 3.1 checkpoint for masks and leaves temporal/3-D identity to the
    workflow's measured multi-view gates. It never silently falls back to the
    older model if loading or inference fails.
    """

    def __init__(self) -> None:
        self.device = os.getenv("REBOT_VISION_DEVICE", "cuda")
        self.score_threshold = float(os.getenv(
            "REBOT_SAM31_SCORE_THRESHOLD", "0.30"
        ))
        self._predictor: Any | None = None
        self._lock = threading.RLock()
        self._last_mask: np.ndarray | None = None
        self._last_box: tuple[int, int, int, int] | None = None
        self.bootstrap_only = os.getenv(
            "REBOT_SAM31_BOOTSTRAP_ONLY", "1"
        ).strip().lower() in {"1", "true", "yes", "on"}
        self._bootstrap_complete = False
        self._geometry_backend: Sam3TextSegmenter | None = None

    def _activate_geometry_backend(self) -> Sam3TextSegmenter:
        """Release SAM 3.1 and switch to the stable multi-view mask model."""
        if self._geometry_backend is not None:
            return self._geometry_backend
        print(
            "[vision] SAM 3.1 target anchor established; unloading it and "
            "switching to stable SAM3 masks for multi-view RGB-D geometry...",
            flush=True,
        )
        self._predictor = None
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        self._geometry_backend = Sam3TextSegmenter()
        return self._geometry_backend

    def _load(self):
        with self._lock:
            if self._predictor is not None:
                return self._predictor
            if self.device != "cuda":
                raise RuntimeError("SAM 3.1 wrist backend requires CUDA")
            from sam3.model_builder import build_sam3_predictor

            print(
                "[vision] loading SAM 3.1 multiplex checkpoint; the first "
                "load can take about 45 seconds on this machine...",
                flush=True,
            )
            # The public 3.1 checkpoint is multiplexed in groups of 16; using
            # any other value creates incompatible tensor shapes. Flash
            # Attention and compilation are deliberately optional and are not
            # required for this bounded single-frame inference path.
            self._predictor = build_sam3_predictor(
                version="sam3.1", compile=False, warm_up=False,
                max_num_objects=16, multiplex_count=16,
                use_fa3=False, use_rope_real=True,
                async_loading_frames=False,
            )
            print(
                "[vision] SAM 3.1 ready; using bounded single-frame masks "
                "(full video propagation disabled for 8GB VRAM safety)",
                flush=True,
            )
            return self._predictor

    def locate(self, frame_bgr: np.ndarray, description: str) -> dict[str, Any]:
        from PIL import Image

        if self.bootstrap_only and self._bootstrap_complete:
            result = self._activate_geometry_backend().locate(
                frame_bgr, description
            )
            if result.get("found"):
                result = {
                    **result,
                    "model_version": "sam3_after_sam3.1_anchor",
                }
            return result
        height, width = frame_bgr.shape[:2]
        self._last_mask = None
        self._last_box = None
        session_id = None
        predictor = None
        try:
            with self._lock:
                predictor = self._load()
                image = Image.fromarray(cv2.cvtColor(
                    frame_bgr, cv2.COLOR_BGR2RGB
                ))
                response = predictor.handle_request({
                    "type": "start_session",
                    "resource_path": [image],
                    "offload_video_to_cpu": True,
                })
                session_id = response["session_id"]
                response = predictor.handle_request({
                    "type": "add_prompt", "session_id": session_id,
                    "frame_index": 0, "text": description.strip(),
                    "output_prob_thresh": self.score_threshold,
                })
                output = response["outputs"]
                scores = np.asarray(output.get("out_probs", []), dtype=float)
                masks = np.asarray(output.get("out_binary_masks", []))
                boxes = np.asarray(output.get("out_boxes_xywh", []), dtype=float)
                if scores.size == 0 or masks.shape[0] == 0:
                    return {
                        "ok": True, "found": False, "backend": "sam3",
                        "model_version": "sam3.1",
                        "image_size": [width, height],
                    }
                best = int(np.argmax(scores))
                score = float(scores[best])
                if score < self.score_threshold:
                    return {
                        "ok": True, "found": False, "backend": "sam3",
                        "model_version": "sam3.1", "confidence": score,
                        "image_size": [width, height],
                    }
                mask = np.where(masks[best] > 0, 255, 0).astype(np.uint8)
                if mask.shape != (height, width) or not np.any(mask):
                    return {
                        "ok": True, "found": False, "backend": "sam3",
                        "model_version": "sam3.1",
                        "image_size": [width, height],
                    }
                x, y, box_width, box_height = boxes[best].tolist()
                x1, y1 = max(0, int(round(x))), max(0, int(round(y)))
                x2 = min(width, int(round(x + box_width)))
                y2 = min(height, int(round(y + box_height)))
                if x2 <= x1 or y2 <= y1:
                    rows, cols = np.nonzero(mask)
                    x1, x2 = int(cols.min()), int(cols.max()) + 1
                    y1, y2 = int(rows.min()), int(rows.max()) + 1
                self._last_box = (x1, y1, x2, y2)
                self._last_mask = mask
                # Keep this exact mask available to segment_box for the
                # current RGB-D observation. The next semantic query (normally
                # the complete-object confirmation) activates the stable
                # geometry model after this anchor has been consumed.
                self._bootstrap_complete = True
                return {
                    "ok": True, "found": True,
                    "box": [x1, y1, x2, y2],
                    "point_px": [(x1 + x2) / 2.0, (y1 + y2) / 2.0],
                    "confidence": score, "backend": "sam3",
                    "model_version": "sam3.1",
                    "instances": int(scores.size),
                    "image_size": [width, height],
                }
        except Exception as exc:
            return {
                "ok": False, "found": False, "error": str(exc),
                "backend": "sam3", "model_version": "sam3.1",
                "image_size": [width, height],
            }
        finally:
            if predictor is not None and session_id is not None:
                try:
                    predictor.handle_request({
                        "type": "close_session", "session_id": session_id,
                        "run_gc_collect": False,
                        "clear_cache_threshold": 1000000,
                    })
                except Exception:
                    pass

    def segment_box(self, frame_bgr: np.ndarray,
                    box: tuple[int, int, int, int]) -> np.ndarray | None:
        if self._geometry_backend is not None:
            return self._geometry_backend.segment_box(frame_bgr, box)
        if self._last_mask is None or self._last_box is None:
            return None
        a = self._last_box
        intersection = max(0, min(a[2], box[2]) - max(a[0], box[0])) * \
            max(0, min(a[3], box[3]) - max(a[1], box[1]))
        union = ((a[2] - a[0]) * (a[3] - a[1])
                 + (box[2] - box[0]) * (box[3] - box[1]) - intersection)
        return self._last_mask.copy() if intersection / max(1, union) >= 0.5 else None


_detector: LocateAnything | GroundedSam2 | Sam3TextSegmenter \
    | Sam31TextSegmenter | None = None


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
        elif backend in {"sam3_1", "sam3.1", "sam31"}:
            _detector = Sam31TextSegmenter()
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


def tracking_lost(description: str) -> bool:
    """Whether the stateful backend lost an already locked named instance."""
    global _detector
    check = getattr(_detector, "tracking_lost", None)
    return bool(check(description)) if callable(check) else False
