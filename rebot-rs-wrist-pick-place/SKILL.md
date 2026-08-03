---
name: rebot-rs-wrist-pick-place
description: Detect, segment, preplan, image-size, pick, verify, drop, and home with a reBot RS arm and wrist RealSense. Use for camera-only detection checks, camera-only open-jaw calibration, or the hardware-verified dynamic marker pick/place workflow.
version: 0.3.0
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
- Keep people outside the workspace. Static scene collision checking is not a
  human-safety system and cannot see through the arm or a carried object.
- When `REBOT_SCENE_COLLISION=1`, never bypass an empty-map, self-collision,
  table, visible-obstacle, payload, or collision-free-home refusal.
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

## Hardware-verified dynamic marker pick/place

Only run after live detection is stable and the operator explicitly requests the
physical pick:

```bash
REBOT_DYNAMIC_GRASP=1 \
REBOT_ENABLE_LOCATE_ANYTHING=1 \
REBOT_VISION_BACKEND=grounded_sam2 \
REBOT_VISION_PRIMARY=0 \
REBOT_GROUNDING_DINO_MODEL=IDEA-Research/grounding-dino-base \
REBOT_SAM2_MODEL=facebook/sam2.1-hiera-large \
REBOT_VISION_DTYPE=float32 \
REBOT_WS_X_MAX=0.490 \
REBOT_WRIST_NEAR_MM=60 \
REBOT_WRIST_FAR_MM=600 \
REBOT_WRIST_PICK_CLOSER_MM=4 \
REBOT_WRIST_PICK_TARGET_Y_OFFSET_MM=8 \
REBOT_WRIST_PICK_TARGET_TOL_MM=20 \
REBOT_WRIST_PICK_MAX_CORRECTIONS=2 \
REBOT_WRIST_PICK_MAX_CORRECTION_M=0.012 \
REBOT_WRIST_FINAL_ALIGN_GAIN=0.85 \
REBOT_WRIST_RIGID_FIT_MAX_ERROR=0.15 \
REBOT_DYNAMIC_SLOT_TOL_MM=8 \
REBOT_DYNAMIC_REMEASURE_MAX_DELTA_MM=6 \
REBOT_DYNAMIC_CLOSE_STEP_DEG=15 \
REBOT_WRIST_PICK_OPEN_BEFORE_END_M=0.040 \
REBOT_WRIST_JAW_LONG_AXIS_CAMERA_DEG=-90 \
REBOT_WRIST_GRASP_MAX_YAW_DELTA_DEG=10 \
REBOT_WRIST_GRASP_YAW_MEASUREMENT_SLOP_DEG=0.5 \
REBOT_WRIST_GRASP_RPY_STEP_DEG=2 \
REBOT_WRIST_GRASP_FINAL_APPROACH_MM=40 \
REBOT_WRIST_GRASP_MIN_FEATURE_PROMINENCE_MM=25 \
REBOT_GRIPPER_SIZE_MARGIN_MM=0 \
REBOT_GRIPPER_SIZE_COMPRESSION_MM=3 \
REBOT_DYNAMIC_MICRO_LIFT_M=0.015 \
REBOT_DYNAMIC_HOLD_TOL_MM=6 \
REBOT_DYNAMIC_RETRY_CLOSE_DEG=5 \
REBOT_DYNAMIC_MAX_RETRIES=3 \
REBOT_LOCAL_IK_POS_TOL_M=0.004 \
REBOT_ARM_SPEED_M_S=0.054 \
REBOT_ARM_MIN_MOVE_S=0.277778 \
REBOT_ARM_ANGULAR_SPEED_RAD_S=0.54 \
REBOT_JOINT_VLIM=0.90 \
REBOT_GRIPPER_MOVE_S=1.10 \
REBOT_GRIPPER_STEP_S=0.04 \
REBOT_SCENE_COLLISION=1 \
REBOT_SCENE_COLLISION_VOXEL_M=0.020 \
REBOT_SCENE_COLLISION_INFLATION_M=0.025 \
REBOT_SCENE_COLLISION_STRIDE_PX=4 \
/home/manav/miniconda3/envs/rebot_rs/bin/python \
  -m arm_agent.wrist_servo \
  --fixed-pick-place "blue marker with white body" --run --live
```

Expected planning evidence:

- `fit_rigid_camera_geometry` passes.
- `preplan_reachable_grasp` passes.
- `prevalidate_complete_approach` passes before approach motion begins.
- `scene_collision_map` reports a nonzero obstacle-column count, 20 mm voxels,
  and 25 mm inflation before the first object-approach trajectory.
- The result's `collision_scene.visualization.open_command` opens the interactive
  `.rrd`; orange points are depth returns, red boxes are inflated collision
  volumes, blue is the calibrated table, and RGB arrows are robot-frame axes.
- Continuation IK is sampled for URDF self/table/scene collision; an attached
  payload envelope is installed before the first lift.
- GPU semantic vision identifies and segments the complete object, while stable
  color/depth tracking—not a changing whole-object box—drives the Jacobian.
- `fit_rigid_camera_geometry` reports an error no greater than 0.15.
- Hardware path logs report `continuous_minimum_jerk_arrived`; the Cartesian
  safety waypoints remain prevalidated but are traversed continuously.
- `dynamic_endpoint_width` confirms the between-jaws measurement.
- `image_controlled_slow_close` reaches the bounded image-derived aperture.
- No camera checkpoint alters the long preplanned approach. A final visible miss
  may trigger only the bounded, separately IK-validated open-jaw retry described
  above.
- Post-lift vision follows the semantic body grasp region rather than a cap that
  may pivot while held.
- The object is released over the taught drop zone and home is verified.

For collision-aware scene-relative placement, prefer the robot-workspace wrapper:

```bash
./run_scene_pick_place.sh --dry "red marker" right "green marker" 0.10
./run_scene_pick_place.sh --run "red marker" right "green marker" 0.10
```

Dry mode validates perception and grasp geometry but reports base-frame collision
planning as deferred because it deliberately does not connect to read the live arm
pose. Live mode fails closed if the map or complete collision-free path cannot be
validated.

## New-object policy

Detection success is not proof that the complete object geometry is safe. The
planner ranks local body cross-sections and permits at most 10 degrees of
IK-validated wrist yaw, but it has no force/contact signal. Do not respond to a
failed novel-object grasp by bypassing fit, jaw-slot, endpoint-width, workspace,
or visual-hold gates.

## Failure handling

Read the returned `steps` array and report the first false step. Preserve the
complete JSON output. Do not automatically rerun a physical failure.
