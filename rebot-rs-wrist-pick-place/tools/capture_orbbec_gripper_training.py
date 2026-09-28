#!/usr/bin/env python3
"""Capture diverse top-down Orbbec gripper training frames without arm control.

The bridge must already be running.  While this program captures, manually
place the (already safely supported) arm/gripper in visibly different poses:
open/closed jaws, empty/held object, and left/centre/right workspace views.
Frames too similar to the last accepted image are skipped, so a stationary
camera does not create a redundant training set.
"""
from __future__ import annotations

import argparse
import io
import time
from pathlib import Path

import cv2
import httpx
import numpy as np


ROOT = Path("/home/manav/rebot_lerobot")
DEFAULT_OUTPUT = ROOT / "datasets/organize_stationery_cosmos_workshop/gripper_segmentation/live_orbbec/images"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=36)
    parser.add_argument("--interval", type=float, default=.75)
    parser.add_argument("--min-difference", type=float, default=.035,
                        help="minimum normalized grayscale change from the prior accepted frame")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--url", default="http://127.0.0.1:8720")
    args = parser.parse_args()
    if args.count < 1:
        raise SystemExit("--count must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    previous: np.ndarray | None = None
    accepted = attempts = 0
    max_attempts = args.count * 12
    print("[capture] camera-only. Move the actual gripper manually between views; Ctrl+C keeps accepted frames.", flush=True)
    try:
        while accepted < args.count and attempts < max_attempts:
            attempts += 1
            response = httpx.get(f"{args.url.rstrip('/')}/frame.npz", timeout=5)
            response.raise_for_status()
            data = np.load(io.BytesIO(response.content))
            color = data["color"]
            small = cv2.resize(cv2.cvtColor(color, cv2.COLOR_BGR2GRAY), (160, 120)).astype(np.float32) / 255.0
            difference = 1.0 if previous is None else float(np.mean(np.abs(small - previous)))
            if difference >= args.min_difference:
                filename = args.output / f"orbbec_gripper_{accepted + 1:03d}.jpg"
                cv2.imwrite(str(filename), color)
                previous = small
                accepted += 1
                print(f"[capture] saved {accepted}/{args.count}: {filename.name} (difference={difference:.3f})", flush=True)
            else:
                print(f"[capture] skipped near-duplicate (difference={difference:.3f}); reposition gripper", flush=True)
            time.sleep(max(.15, args.interval))
    except KeyboardInterrupt:
        pass
    print(f"[capture] complete: accepted={accepted}, attempts={attempts}, output={args.output}", flush=True)


if __name__ == "__main__":
    main()
