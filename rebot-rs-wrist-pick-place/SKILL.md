---
name: rebot-rs-wrist-pick-place
description: Detect, preplan, pick, lift, drop, and home with a reBot RS arm and wrist RealSense. Use for camera-only detection checks, camera-only open-jaw calibration, or the hardware-verified fixed marker pick/place workflow. Do not use the unverified RPY bottle-pick extension.
version: 0.1.0
author: Manav Goel
metadata:
  hermes:
    tags: [robotics, rebot, realsense, pick-place, safety]
prerequisites:
  commands: [bash]
---

# reBot RS wrist pick/place

Operate the implementation from the robot workspace:

```bash
cd /home/manav/rebot_lerobot
```

Use the Python interpreter that owns the RealSense and Pinocchio dependencies:

```text
/home/manav/miniconda3/envs/rebot_rs/bin/python
```

## Safety contract

- Never zero the arm or gripper as part of this workflow.
- Never use contact-aware or stall-inferred gripper closing.
- Never bypass workspace, IK, waypoint, joint-continuity, or home verification.
- Never loosen a failed gate merely to make a target reachable.
- Treat Ctrl+C as controlled recovery: freeze, open, then home.
- If home cannot be verified, keep motor control active and tell the operator to
  support the arm before using the physical E-stop or power cutoff.
- Keep people and obstacles outside the planned arm, bottle, and drop-zone path.
- Use camera-only `--watch` before attempting a new colour/object.
- Use a light object for the first trial of a new geometry.

## Check live detection (no motion)

For a blue marker:

```bash
REBOT_WRIST_NEAR_MM=60 \
REBOT_WRIST_FAR_MM=600 \
/home/manav/miniconda3/envs/rebot_rs/bin/python \
  -m arm_agent.wrist_servo \
  --watch --object "blue marker" --live
```

Replace the object with a colour-bearing description such as `"green cap"`.
This command must not connect to or command the arm or gripper. Ctrl+C only stops
the camera stream.

## Calibrate the open-jaw visual target (no motion)

The operator manually holds the intended grasp feature in the open jaw slot. Do
not add `--run`:

```bash
REBOT_WRIST_NEAR_MM=60 \
REBOT_WRIST_FAR_MM=250 \
/home/manav/miniconda3/envs/rebot_rs/bin/python \
  -m arm_agent.wrist_servo \
  --calib-grasp-target "blue marker" --view
```

The successful result explicitly says the arm and gripper were not connected,
read, enabled, or commanded.

## Hardware-verified fixed marker pick/place

Only run after live detection is stable and the operator explicitly requests the
physical pick:

```bash
REBOT_WS_X_MAX=0.480 \
REBOT_WRIST_NEAR_MM=60 \
REBOT_WRIST_FAR_MM=600 \
REBOT_WRIST_PICK_CLOSER_MM=4 \
REBOT_WRIST_PICK_TARGET_TOL_MM=10 \
REBOT_WRIST_PICK_OPEN_BEFORE_END_M=0.040 \
REBOT_GRIPPER_SIZE_MARGIN_MM=0 \
REBOT_GRIPPER_SIZE_COMPRESSION_MM=1 \
REBOT_LOCAL_IK_POS_TOL_M=0.004 \
/home/manav/miniconda3/envs/rebot_rs/bin/python \
  -m arm_agent.wrist_servo \
  --fixed-pick-place "blue marker" --run --live
```

Expected planning evidence:

- `fit_rigid_camera_geometry` passes.
- `preplan_reachable_grasp` passes.
- `prevalidate_complete_approach` passes before approach motion begins.
- The gripper plan reports 1 mm planned position compression.
- No camera checkpoint or camera steering alters the preplanned trajectory.
- Post-lift vision confirms that the object moved rigidly with the wrist.
- The object is released over the taught drop zone and home is verified.

## New-object policy

Detection success is not proof that the complete object geometry is safe.
The current green-cap bottle trial can detect and approach the cap, but a lateral
cap approach topples the bottle. Do not retry it using larger tolerances,
additional depth, or stronger closing. A bottle needs a separately implemented
and hardware-validated, tightly clamped RPY/posture planner that accounts for the
bottle body and approaches the cap along a stable direction.

## Failure handling

Read the returned `steps` array and report the first false step. Preserve the
complete JSON output. Do not automatically rerun a physical failure.
