#!/usr/bin/env python3
"""Fine-tune the top-down gripper segmenter on original + live Orbbec labels."""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import yaml


ROOT = Path("/home/manav/rebot_lerobot")
ORIGINAL = ROOT / "datasets/organize_stationery_cosmos_workshop/gripper_segmentation/reviewed"
LIVE = ROOT / "datasets/organize_stationery_cosmos_workshop/gripper_segmentation/live_orbbec/reviewed"
DATASET = ROOT / "datasets/organize_stationery_cosmos_workshop/gripper_segmentation/yolo_topdown_gripper_orbbec_finetune"
WEIGHTS = ROOT / "runs/gripper_segmentation/topdown_yolo11n/weights/best.pt"


def stage(sources: list[Path], destination: Path) -> tuple[int, int]:
    for split in ("train", "val"):
        for kind in ("images", "labels"):
            (destination / kind / split).mkdir(parents=True, exist_ok=True)
    train = val = 0
    for source_index, source in enumerate(sources):
        images = sorted((source / "images").glob("*.jpg"))
        if len(images) < 10:
            raise RuntimeError(f"need at least 10 accepted masks in {source}; found {len(images)}")
        for index, image in enumerate(images):
            label = source / "labels" / f"{image.stem}.txt"
            if not label.exists() or not label.read_text(encoding="utf-8").strip():
                continue
            # Keep both domains in validation, not merely a final contiguous
            # tail of one recording session.
            split = "val" if index % 5 == 0 else "train"
            name = f"source{source_index}_{image.name}"
            shutil.copy2(image, destination / "images" / split / name)
            shutil.copy2(label, destination / "labels" / split / f"{Path(name).stem}.txt")
            if split == "train": train += 1
            else: val += 1
    (destination / "dataset.yaml").write_text(yaml.safe_dump({
        "path": str(destination), "train": "images/train", "val": "images/val",
        "names": {0: "gripper"},
    }, sort_keys=False), encoding="utf-8")
    return train, val


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=55)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--resume", action="store_true",
                        help="resume the interrupted fine-tune from its last checkpoint")
    args = parser.parse_args()
    if not WEIGHTS.is_file():
        raise SystemExit(f"base weights missing: {WEIGHTS}")
    train, val = stage([ORIGINAL, LIVE], DATASET)
    print(f"[finetune] staged train={train}, val={val}, dataset={DATASET}", flush=True)
    if args.prepare_only:
        return
    from ultralytics import YOLO
    run = ROOT / "runs/gripper_segmentation/topdown_yolo11n_orbbec_finetune"
    if args.resume:
        checkpoint = run / "weights/last.pt"
        if not checkpoint.is_file():
            raise SystemExit(f"resume checkpoint missing: {checkpoint}")
        print(f"[finetune] resuming {checkpoint}", flush=True)
        YOLO(str(checkpoint)).train(resume=True)
        return
    model = YOLO(str(WEIGHTS))
    model.train(
        data=str(DATASET / "dataset.yaml"), epochs=args.epochs, imgsz=640,
        batch=args.batch, device=0, workers=4, patience=16,
        project=str(ROOT / "runs/gripper_segmentation"),
        name="topdown_yolo11n_orbbec_finetune", exist_ok=True,
        degrees=10, translate=.06, scale=.15, fliplr=.5, mosaic=0.0, mixup=0.0,
    )


if __name__ == "__main__":
    main()
