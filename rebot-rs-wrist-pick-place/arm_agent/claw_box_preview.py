"""Live, no-grasp preview for the hybrid Claw-box reflex.

The module deliberately configures the hybrid profile before importing the
wrist controller. It may move the arm through bounded observation poses and
returns home, but it stops before the final grasp approach and never commands
the gripper.
"""
from __future__ import annotations

import argparse
import json

from arm_agent.hybrid_profile import configure


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("object", help="specific target whose 3D box is tested")
    parser.add_argument(
        "--live", action="store_true",
        help="open the diagnostic wrist RGB-D feed on http://127.0.0.1:8731/",
    )
    parser.add_argument(
        "--reflex", choices=("off", "fallback", "force"), default="fallback",
        help="off: normal planner only; fallback: use box reflex only when needed; "
             "force: require an OBB-derived alternate candidate for this preview",
    )
    parser.add_argument(
        "--candidate-ladder", action="store_true",
        help="enumerate every safe OBB-derived terminal alternative before "
             "returning home; this is diagnostic-only and never grasps",
    )
    args = parser.parse_args()

    import os
    os.environ["REBOT_CLAW_BOX_REFLEX_FALLBACK"] = (
        "force" if args.candidate_ladder else args.reflex
    )
    if args.candidate_ladder:
        os.environ["REBOT_CLAW_BOX_REFLEX_CANDIDATE_PREVIEW"] = "1"
    configure()
    from arm_agent import wrist_servo as wrist

    if args.live:
        wrist.set_viz(wrist.make_viz(mjpeg_port=8731, mjpeg_auto_open=True))
        print("[hybrid] LIVE wrist feed -> http://127.0.0.1:8731/", flush=True)
    result = wrist.claw_box_reflex_preview(args.object, verbose=True)
    print(json.dumps(result, indent=2, default=str))
    raise SystemExit(0 if result.get("ok") else 1)


if __name__ == "__main__":
    main()
