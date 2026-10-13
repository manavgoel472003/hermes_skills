# Hermes arm controller

## Wrist-camera (eye-in-hand) pick & place — hermes demo

Say *"put the red block in the bin"* to `hermes`; the arm finds the block with the
RealSense **on its own gripper** and drops it in a spot you taught once. No table
calibration, no overhead camera. Full runbook:
[docs/WRIST_PICK_PLACE_DEMO.md](../docs/WRIST_PICK_PLACE_DEMO.md).

```bash
conda activate rebot_rs
python -m arm_agent.wrist_servo --sim                    # whole loop, no hardware
python -m arm_agent.wrist_servo --sim --view             # + annotated camera frames
python -m arm_agent.wrist_replay --spawn                 # watch the ARM move, in 3-D
python -m arm_agent.wrist_servo --snapshot --object "red block" --view
python -m arm_agent.wrist_servo --calib-tcp --run        # one time: jaws only
python -m arm_agent.wrist_servo --teach-drop --run       # one time: records the pose
python -m arm_agent.wrist_servo --pick-place "red block" --run --view
```

MCP tools for the planner: `wrist_pick`, `wrist_place`, `wrist_pick_and_place`,
`wrist_look`, `wrist_teach_drop_zone`.

`wrist_pick` now ends after the camera-verified carry lift and keeps the object
held. `wrist_place` consumes that held state, releases over the taught high drop
zone, and homes. Both calls must use the same running MCP process; the held state
is intentionally never written to disk. `wrist_pick_and_place` remains the
original all-in-one workflow.

For a standalone pick-and-hold terminal session:

```bash
./run_object_pick.sh --dry "red marker"
./run_object_pick.sh --run "red marker"
```

The standalone launcher deliberately excludes the coarse scene-placement map,
because the selected target must occupy the jaw endpoint. It still uses the
semantic detection and aligned depth to select the grasp section, size the
jaws, perform bounded final alignment, close slowly, and verify the lift. Robot
workspace/table limits, complete IK validation, URDF self-collision, and the
operator stop/recovery path remain enabled. Use `run_scene_pick_place.sh` when
scene-relative placement and scene-obstacle routing are required.

Press `Q` or `Ctrl+C` to freeze, open the gripper, home, and exit. In Hermes, say
“stop the wrist” to call `wrist_stop`; MCP stdin is reserved for protocol traffic.

Scene-relative placing is available after `wrist_pick` in the same MCP session:

```text
place it left of the green marker
place it to the right of the sponge
place it next to the bottle
place it somewhere on the table
```

Hermes maps these to `wrist_place(reference=..., relation=...)`. Left/right and
front/behind use the robot/table frame; `next_to` chooses the reachable cardinal
side with the most workspace-edge clearance. Reference localization requires at
least two stable detections from three carry-pose frames and the saved rigid camera
mapping. `run_scene_pick_place.sh` additionally enables the fail-closed RGB-D
collision gate. The ready-pose depth frame is converted to 20 mm obstacle
columns with a configurable 3 mm uncertainty inflation
base-frame obstacle columns; the requested pick target is the only excluded
contact region. Pinocchio/HPP-FCL checks the RS URDF for self-collision and checks
every sampled configuration against the calibrated table, visible clutter, and a
bounded carried-object envelope. The planner tries direct, X-first, Y-first, and
several elevated/lateral corridors, but sends nothing unless one complete route
passes. This is a static visible-scene model: unseen or moving obstacles still
require the existing Q/Ctrl+C/Hermes stop path.

The original one-call pick/place pipeline accepts the same scene destination:

```bash
./run_scene_pick_place.sh --dry "red marker" right "green marker" 0.10
./run_scene_pick_place.sh --run "red marker" right "green marker" 0.10
```

The optional final number is separation in metres. Omit it to size the spacing
from both visual footprints plus 25 mm clearance. The dry run checks detection and
grasp sizing from the current view; final reference consensus and place-path IK are
necessarily deferred until the live arm has reached the verified carry pose.

Because the camera is rigid to the gripper the TCP is *fixed in the image*, so it
is localized once (`motion_tip`) instead of tracked per frame, and the servo drives
the OBJECT to a hover target derived from an online-learned Jacobian —
`arm_agent/wrist_sim.py` renders that geometry from first principles, and checks
poses against the same Pinocchio IK the hardware uses, so `--sim` catches both bad
sign conventions and unreachable poses without hardware.

**Note:** this demo works at `REBOT_WRIST_PITCH=1.0` rad, not `arm_safe.DEF_PITCH`
(1.30). At 1.30 the RS arm reaches only ~23 mm above the table — too little to lift
a block. Scan/carry heights are derived by probing IK, not fixed offsets.

**`arm_safe` workspace default changed** from the hand-follow teleop box to a
`table` profile (`z 0.00–0.30`, `y ±0.24`): the teleop floor at z=0.183 is ~17 cm
above the table, so a descent could never reach an object. `REBOT_WORKSPACE=teleop`
restores the old box; per-axis `REBOT_WS_*` overrides also exist. The gripper is
capped at **50 %** travel for testing (`REBOT_GRIPPER_MAX_OPEN`).

## Single-process RGB-D servo (RS arm)

The depth-camera pick/place path no longer needs the FastAPI arm service. In the
`rebot_rs` environment it owns the RealSense camera, LocateAnything detector,
Pinocchio IK, and RS CAN controller in one process. The safe motion envelope,
IK verification, small-step limit, and torque-gated descent remain enforced by
`arm_agent.arm_safe`.

```bash
cd ~/rebot_lerobot
conda activate rebot_rs

# one time (already installed in this workspace)
python -m pip install 'pin==4.1.0' pyrealsense2 'transformers==4.57.6' \
  'huggingface-hub[cli,hf-transfer]>=0.34.2,<0.36.0' bitsandbytes

# no hardware: confirms imports and annotation only
python -m arm_agent.depth_servo --demo

# RealSense is the default; this does not connect or move the arm
python -m arm_agent.depth_servo --snapshot --object "red block"

# connects the RS arm only for --run; always start with --dry
python -m arm_agent.depth_servo --pick "red block" --dry --view
python -m arm_agent.depth_servo --pick "red block" --run --view --mjpeg 8730
python -m arm_agent.depth_servo --place --target "the bowl" --run --view
```

Set `REBOT_DEPTH_CAMERA=realsense` (default), or pass `--camera realsense`.
`REBOT_REALSENSE_SERIAL`, `REBOT_REALSENSE_WIDTH`, `REBOT_REALSENSE_HEIGHT`,
and `REBOT_REALSENSE_FPS` select a device/stream. `--camera orbbec` retains the
old HTTP bridge as a compatibility fallback; it is the only mode needing a
second Python process because the installed Orbbec SDK cannot run in Python 3.12.
Set `REBOT_ARM_BACKEND=service` only to use the legacy HTTP stack.
GPU semantic vision is intentionally opt-in (`REBOT_ENABLE_LOCATE_ANYTHING=1`).
The default backend is Grounding DINO Base plus SAM 2.1 Small: DINO locates the
requested whole object, SAM supplies its contour, and RGB-D supplies metric grasp
width. Set `REBOT_VISION_BACKEND=locate_anything` only for the legacy 3B generative
detector. Model downloads happen on the first run; subsequent inference uses the
local cache. Motion tracking remains color/depth-first by default because a
whole-object box is not a repeatable finite-difference keypoint;
`REBOT_VISION_PRIMARY=1` is intended for detector evaluation, not arm motion.

The local RS backend uses the shared `depth_hand_follow_demo.py` workspace:
X `[0.21, 0.43]`, Y `[-0.1445, 0.1445]`, and Z `[0.18275, 0.32725]` m. It keeps
the URDF's first-six-joint hard limits and the hand-follow 0.18 m/s target-speed
budget (9 mm maximum discrete step). Gripper targets are hard-clamped to the
hand-follow demo's 5%–70% travel window (`-0.236` to `-3.299 rad`); the regular
`open` target is `-2.72 rad`.
Relative `safe_step` and `safe_goto` commands preserve the live tool orientation;
they never implicitly reorient to the old default pitch. Local IK never performs
random retry seeding: it must converge from the current joints or it is refused.

### Markerless tip calibration

If the gripper cannot carry a coloured marker, use its bounded jaw motion as the
visual signal. This does not move an arm joint: it captures a closed RGB-D frame,
opens within the 70% cap, captures again, then restores closed. The RGB/depth
difference localises the two jaws and their midpoint is the camera-frame TCP.

```bash
python -m arm_agent.motion_tip --camera realsense --run
```

The overlay is written to `outputs/motion_tip_latest.jpg`.

Natural-language control of the reBot Arm B601: you give the arm a task in plain
language (or speech), and **Nous Hermes (via Ollama)** plans it, looks through the
camera, and carries it out with on-the-fly IK — composing a library of safe skills,
reasoning over a full scene graph, and re-planning on failure.

Built as two pieces:

- **`rebot_arm_service/`** — the "hands". A fork of
  [yuyoujiang/rebot_arm_service](https://github.com/yuyoujiang/rebot_arm_service)
  (FastAPI + the vendored `reBotArm_control_py` Pinocchio IK/CLIK + motorbridge CAN)
  extended with: multi-camera support, an open-vocabulary VLM scene graph, a
  composable skill library, and a sandboxed code escape-hatch. It is the **single
  owner of the CAN bus** and the **single place safety clamps live**.
- **`arm_agent/`** — the "brain". A thin client that turns a task into skill/tool
  calls over the service's HTTP API. Hermes never touches motors directly.

```
task (text/voice)
  -> arm_agent (Hermes planner, plan->act->verify->replan, memory)
       tools = skills + scan_scene + run_code
  -> rebot_arm_service  (skills -> clamped move_pose/gripper/IK; scene graph; sandbox)
  -> B601 over CAN
```

## What the arm can do

Open-vocabulary, multi-object, multi-step tasks, e.g. *"stack the red block on the
blue block, then put the cup to their right"*, *"sort the blocks by colour"*,
*"push the can into the corner"*, *"pour cup A into the bowl"*, *"hand me the marker"*.

Skills (each validated against a workspace box + IK, then post-condition checked):
`home, open_gripper, close_gripper, retract, move_above, grasp, pick, place_at,
place_on, place_in, place_beside, stack, push, point_at, rotate_wrist, pour,
hand_over` — plus `run_code` for motions no skill covers (restricted sandbox).

## One-time setup (on the robot PC)

Everything runs in the existing conda env `lerobot` (has pinocchio + motorbridge).

```bash
conda activate lerobot
pip install fastapi uvicorn httpx ollama faster-whisper sounddevice   # one-time

# models (Ollama)
ollama serve &
ollama pull hermes3            # the planner brain (text)
ollama pull qwen2.5vl          # the vision model (open-vocab perception)
```

Camera + arm devices (defaults): top camera `/dev/video4`, wrist `/dev/video6`,
arm on `/dev/ttyACM0`. Override with env: `REBOT_TOP_CAM_INDEX`, `REBOT_WRIST_CAM_INDEX`,
`REBOT_TOP_CAM_ROTATE180`, `REBOT_VLM_MODEL`, `REBOT_HERMES_MODEL`, `OLLAMA_HOST`.

## Calibrate the table (one-time, required for positions)

The arm has no depth sensor; object positions come from a top-camera→table
homography. Mark/measure ≥4 points on the table (base-frame x forward, y to the
arm's left, metres) and:

```bash
cd rebot_arm_service
python -m rebot_arm_service.scene.calibrate --click --camera-index 4 --table-z 0.0
# or non-interactively:
python -m rebot_arm_service.scene.calibrate --table-z 0.0 \
    --pair 120,90,0.30,0.10 --pair 520,90,0.30,-0.10 \
    --pair 520,400,0.18,-0.10 --pair 120,400,0.18,0.10
```

Writes `calib/top_cam.json`. Re-load without restarting via `POST /scene/reload-calib`.

## Run

```bash
# 1) start the service (hands)
cd rebot_arm_service
PYTHONPATH=src REBOT_TOP_CAM_INDEX=4 REBOT_WRIST_CAM_INDEX=6 \
    python -m rebot_arm_service.main           # serves on :8000

# 2a) drive it with the brain (text)
cd ..
python -m arm_agent.cli                        # REPL
python -m arm_agent.cli --once "stack the red block on the blue block"
python -m arm_agent.cli --dry-run --once "put the cup right of the marker"   # plan, no motion

# 2b) or by voice
python -m arm_agent.voice                      # push-to-talk
python -m arm_agent.voice --whisper-model small --whisper-device cpu --no-tts
```

`--dry-run` (or env `REBOT_DRY_RUN=1`) plans and validates every pose (workspace +
IK) **without moving the arm** — always smoke-test a new task this way first.

## Safety

The service is the only CAN-bus owner. Every motion path — skills *and* `run_code`
— is clamped to the workspace AABB + IK reachability + gripper limits before
executing; `POST /estop` stops the arm. `run_code` runs in a namespace with no
imports/files/network/raw-motor and hard caps on moves and wall-clock time. The
code escape-hatch is guarded, not a security sandbox — review generated motion.

## Tuning (expect to adjust on the real arm)

`config/skills.yaml` (`SkillConfig`): workspace box, `approach_pitch_rad`,
`grasp_z_offset_m`, `grasp_xy_standoff_m`, lift/place heights, gripper timing,
`push_*`, `pour_angle_rad`, hand-over pose, `wrist_refine` + gains.

**No-depth caveat:** table-plane tasks (relative placement, push, sort-into-bins)
are reliable; **precise stacking and pouring are best-effort** — they assume object
heights (`object_height_default_m`) and benefit from `wrist_refine` and a depth
camera later.

## Endpoints (service)

Primitives: `/healthz /state /connect /move/pose /move/joints /gripper /home /sim/ik /sim/fk`.
Cameras: `/camera/list`, `/camera/{top|wrist}/frame.jpg|stream.mjpg`.
Perception: `/vision/locate`, `/scene`, `/scene/calibration`, `/vision/health`.
Skills: `/skill/list`, `/skill/{name}` (`?dry_run=true`). Code: `/code/run`.
WebUI: `/webui`.
