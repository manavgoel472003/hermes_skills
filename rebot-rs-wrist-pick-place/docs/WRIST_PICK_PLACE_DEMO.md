# Wrist-camera pick & place, driven by hermes

Say *"put the red block in the bin"* to `hermes`; the B601-RS finds the block with
the RealSense **on its own gripper**, servos onto it, grasps by feel, and drops it
in the spot you taught it.

No table calibration, no homography, no overhead camera.

```
hermes (Codex planner)
  -> MCP: wrist_pick / wrist_place / wrist_pick_and_place / wrist_look
       -> arm_agent.wrist_servo   eye-in-hand servo loop
            wrist_detect  colour-first detection (LocateAnything fallback)
            arm_safe      bounds + IK verify + step cap + torque-gated descent
              -> RS arm over CAN
```

## Why the wrist camera changes the problem

With the camera bolted to the gripper, **the TCP sits at a fixed point in the
camera frame**. The overhead servo (`depth_servo`) has to re-find the gripper tip
in every frame — that is what the orange marker exists for. Here the tip *cannot*
move in the image, so it is localized **once** and cached, and each servo iteration
only has to find the object.

The catch: a static object appears to move *opposite* to the arm, inverting every
sign relative to the overhead case. Rather than reason about that, the loop
**measures** it — jog each base axis, watch the object — so the convention is
whatever the hardware actually does. The hover target then falls out with no
camera pose and no hand-eye calibration:

```
target_obj_cam = tcp_cam + APPROACH_M * J[:, 2]
```

"where the object would appear if the TCP were hovering `APPROACH_M` above it",
because `J[:, 2]` is exactly how the object shifts per metre of base lift.

## Prerequisites

- RealSense **D405** mounted on the gripper (its 7–50 cm range suits eye-in-hand;
  a D435 works if you widen `REBOT_WRIST_NEAR_MM`).
- RS arm live on CAN — see the B601-RS bring-up (gs_usb adapter at **1 Mbit**).
- The `rebot_rs` conda env (owns pyrealsense2, Pinocchio IK and the CAN controller).

```bash
cd ~/rebot_lerobot && conda activate rebot_rs
```

## Step 0 — prove the loop offline (no hardware)

Runs the real servo against a synthetic pinhole camera + fake arm in
`arm_agent/wrist_sim.py`. Do this after any change to the servo math.

```bash
python -m arm_agent.wrist_servo --sim
python -m arm_agent.wrist_servo --sim --object-xy 0.26,-0.10 --tcp-error-mm 5
```

Expect `"ok": true`, `"reason": "placed"`, and `sim.object_end_xy` equal to
`sim.drop_zone_xy`.

### Watch the actual arm (3-D replay)

`--sim --view` shows what the *camera* saw. To watch the *robot*, replay the same
trajectory through the real URDF and IK into the rerun viewer — articulated arm,
tool path, workspace, and the synchronised wrist view, all scrubbable:

```bash
python -m arm_agent.wrist_replay            # writes an .rrd and prints the view command
python -m arm_agent.wrist_replay --spawn    # open the viewer immediately
rerun outputs/wrist_sim/wrist_replay.rrd    # or open a saved one later
```

It prints the FK residual between the rendered arm and the commanded path — that
number is the check that the animation is faithful (expect well under a
millimetre). Nothing is invented: every pose is one the servo actually commanded
under `arm_safe`'s limits.

## Workspace envelope and gripper cap — read before running on hardware

**Gripper travel is capped at both ends**, enforced in `local_arm` immediately
before every motor target assignment and mirrored exactly in the sim:

| | rad | |
| --- | --- | --- |
| mechanical stop (shut) | `+0.1916` | never commanded |
| **close target — 5% open** | `-0.0170` | 0.21 rad of margin to the stop |
| **open target — 50%** | `-1.8946` | |

Closing never drives to the mechanical stop, and a close issued when the jaws are
*already* shut is **skipped** rather than re-commanded — re-driving into the stop
is what faults the motor (status 12). Raise the open cap deliberately with
`REBOT_GRIPPER_MAX_OPEN` (ceiling 0.70).

Endpoints come from the RoboStride gripper config's measured values
(`closed +0.19164`, `full open -3.98092`), **not** the round `-3*pi/2` that was
hardcoded before: that over-states travel by 13%, so a nominal "50%" against it
actually opened the jaws 61% of the way.

### Sized opening — open to fit the object

Given the jaw scale, the gripper opens only as wide as the measured object needs
rather than always swinging to the cap, and an object that cannot fit is **refused
before the arm moves**.

Radians alone cannot give millimetres, and **the URDF cannot supply it either**: it
carries the finger prismatic joints (`joint_left` 0–50 mm, `joint_right` 0–71.5 mm)
but contains **no `mimic` or `transmission` tags**, so nothing couples them to the
motor angle. The upstream repo documents no jaw stroke either. It has to be
measured once — but a ruler is enough, no camera involved:

```bash
python -m arm_agent.wrist_servo --gripper open --run       # jaws go to the 50% cap
# measure the gap with calipers, then:
python -m arm_agent.wrist_servo --set-jaw-span 80          # 80 mm at the open cap
# or from CAD full-travel geometry:
python -m arm_agent.wrist_servo --set-jaw-span 160 --at-fraction 1.0
```

Or let the camera do it — grip an object and the object's measured width against
the angle where the jaws stopped fixes the scale:

```bash
python -m arm_agent.wrist_servo --calib-jaw "red block" --run   # object in the jaws
python -m arm_agent.wrist_servo --show-calib                    # mm_per_rad, max aperture
```

Object size comes from `minAreaRect` on the detection scaled by depth, so a
diagonal object is not mistaken for a wide one. The **narrow** side decides whether
a grasp is possible at all; the **wide** side sizes the opening, so the jaws clear
the object whatever its orientation. Until calibrated, everything falls back to the
fixed 50% cap. Tune the extra gap with `REBOT_GRIPPER_CLEARANCE_MM` (default 12).

**`arm_safe` now defaults to a `table` workspace profile**, not the old
hand-follow teleop box:

| profile | x | y | z |
| --- | --- | --- | --- |
| `table` (default) | 0.20 … 0.44 | −0.24 … 0.24 | **0.00 … 0.30** |
| `teleop` (old) | 0.21 … 0.43 | −0.145 … 0.145 | **0.183 … 0.327** |

The teleop box came from the hand-follow demo, which shrinks Y/Z by 15% for a
human waving at a camera. Its floor at z=0.183 sits **~17 cm above the table** on
this robot — a descent would stop in mid-air and the gripper would close on
nothing. Measured hardware poses (x 0.263–0.401, y −0.211…0.137, z 0.015–0.282)
are all outside it but all IK-reachable.

Select with `REBOT_WORKSPACE=teleop|table`, or override any single axis with
`REBOT_WS_{X,Y,Z}_{MIN,MAX}`. **The AABB is only a coarse filter** — every move is
still IK-verified before it executes and every descent is torque-gated, so a
generous box removes no safety check; unreachable corners are refused at move time.

## Working pitch — read this before running on hardware

The demo works at **1.0 rad (57°)**, not `arm_safe.DEF_PITCH` (1.30 rad / 74°).
At that near-vertical wrist the RS arm can only reach ~23 mm above the table —
not enough to clear, grasp and lift a block, and the top of the workspace box is
unreachable entirely. 1.0 rad still aims the wrist camera down at the table while
buying ~104 mm of working height:

| tool pitch | reachable height above table |
| --- | --- |
| 1.30 rad (74°) | 23 mm — unusable |
| 1.20 rad (69°) | 50 mm |
| 1.10 rad (63°) | 68 mm |
| **1.00 rad (57°)** | **104 mm** |
| 0.80 rad (46°) | 141 mm (full box) |

Override with `REBOT_WRIST_PITCH`. Scan and carry heights are **not** fixed
offsets — they are found by asking IK for the highest reachable z at each pose, so
they adapt to whatever pitch and mount you settle on. If your camera ends up
mounted at a different angle, set the pitch that points it at the table and the
heights follow automatically.

## Picking anything, without naming it

Calibration is **once per camera mount / gripper — never per object**. The object
used during `--calib-grasp` is only a means to locate the TCP; once stored it
applies to everything.

For the object itself you do not need a name at all. A depth camera doesn't have
to *recognise* something to know it is there: fit the table plane, and anything
standing on it is an object.

```bash
python -m arm_agent.wrist_servo --pick-place any --run --live
```

`any`, `object`, `anything`, `something` (or an empty name) route to depth
segmentation, which is colour-blind and model-free — a grey block, a black cable
and a red marker are all found identically. Candidates wider than the jaws are
dropped when the jaw scale is known, and the most central of the rest wins.

The gripper's own jaws stand proud of the table too, so segmentation only accepts
a bounded height band above the plane (8–120 mm by default); otherwise the jaws
would be returned as the nearest object every time.

Naming an object still works and is more selective — `--pick "red block"` uses the
colour path and won't grab the wrong item off a crowded table.

## Step 1 — aim the camera

Leave the live view running while you aim. No motion, ever:

```bash
python -m arm_agent.wrist_servo --watch --object "red block" --live
# open http://127.0.0.1:8731/ ; Ctrl+C to stop
```

The block should be boxed with its depth labelled. **Use `--watch`, not
`--snapshot`, for the browser feed** — `--snapshot` grabs one frame and exits, and
the stream dies with the process before a browser can connect.

## Step 2 — calibrate the TCP (one time per mount)

This finds the vector from the camera lens to the grasp point. Knowing how far the
jaws open is a scalar and tells you nothing about *where* they are relative to the
lens — that 3-D offset is what the servo needs.

**From CAD, if you have the mount geometry.** The reBot-DevArm repo ships a STEP
file for the D405 mount, so the offset is knowable without any calibration:

```bash
python -m arm_agent.wrist_servo --set-tcp "-2.6,44.9,104.8"     # millimetres
python -m arm_agent.wrist_servo --check-tcp "red block" --run   # verify it
```

Two things CAD will not give you, which is why `--check-tcp` exists: the STEP
describes the camera *housing*, while the servo needs the **optical frame** (the
D405's imager centre sits inside the housing — that offset is in Intel's datasheet,
not the mount CAD), and printed brackets plus screw play add 1–3 mm against a
roughly ±10 mm grasp margin. `--check-tcp` grips an object and reports the error
without overwriting anything: under 5 mm is good, over 12 mm will cost you grasps.

**Or measure it — from a held object.** An object gripped in the jaws *is* at the
TCP, so its camera-frame position **is** the TCP's. Put the object between the
jaws, then:

```bash
python -m arm_agent.wrist_servo --calib-grasp "red block" --run --live
```

Only the gripper moves. This needs no jaw to be visible, and it is
self-consistent: the servo later drives objects to this same measured point, so
any systematic detector bias cancels instead of adding.

**Fallback — jaw motion differencing** (`motion_tip`). Closes, opens, closes, and
locates the jaws from the RGB-D difference:

```bash
python -m arm_agent.wrist_servo --calib-tcp --run
```

*If the open and closed images look identical*, this method cannot work on your
mount — the jaws are either outside the field of view or inside the D405's ~70 mm
dead zone. Check the jaws actually move first, then use `--calib-grasp`:

```bash
python -m arm_agent.wrist_servo --gripper open --run    # prints travel_percent
python -m arm_agent.wrist_servo --gripper close --run
```

Either method writes `calib/wrist_tcp.json`. Redo it whenever the camera is moved
on the gripper — a stale TCP is the most likely cause of a near-miss grasp.

## Step 3 — teach the drop zone (one time per scene)

Park the gripper where you want the release (over the bin, at release height),
then record the pose. Reads the arm; moves nothing.

```bash
python -m arm_agent.wrist_servo --teach-drop --run
# or, if you already know the numbers:
python -m arm_agent.wrist_servo --teach-drop --drop 0.24,-0.11,0.215
python -m arm_agent.wrist_servo --show-calib     # confirm both are stored
```

## Step 4 — dry run, then the real thing

```bash
python -m arm_agent.wrist_servo --pick "red block" --dry --live       # no motion
python -m arm_agent.wrist_servo --pick-place "red block" --run --live
```

`--live` opens the annotated wrist feed at **http://127.0.0.1:8731/** — open it in
a browser once the run has started (the server lives inside the process). The image updates continuously (frames are pumped while
the arm settles between steps), showing the object box, the fixed TCP crosshair,
the hover target and the live error. `--view` alone just writes frames to disk.

`--dry` reports only what is visible **from where the arm already is**; a `--run`
pick additionally searches the other scan poses.

`--pick-place` **auto-homes** once the object is released. Homing runs only after
a successful release — parking while still holding something would carry it off or
drop it somewhere unplanned — and a failed run deliberately stops where it is so
you can inspect the pose. Disable with `--no-home`; park manually any time with
`--home --run`.

Individual halves, if you want to stop in between:

```bash
python -m arm_agent.wrist_servo --pick "red block" --run --live   # leaves it held
python -m arm_agent.wrist_servo --place --run --live              # releases it
python -m arm_agent.wrist_servo --home --run
```

### Speed

Motion is deliberately slow but still smooth — each move remains a single IK +
joint-space interpolation, just given a longer duration, so nothing is jerky.

| env | default | meaning |
| --- | --- | --- |
| `REBOT_ARM_SPEED_M_S` | `0.015` | tool pace (15 mm/s) |
| `REBOT_ARM_MIN_MOVE_S` | `1.0` | floor per move, so 9 mm servo steps aren't twitches |
| `REBOT_JOINT_VLIM` | `0.25` | per-joint ceiling (rad/s), also used by homing |

At these settings a 9 mm servo step takes 1.0 s and a full pick-and-place runs a
few minutes. Raise `REBOT_ARM_SPEED_M_S` and lower `REBOT_ARM_MIN_MOVE_S` once you
trust the motion.

## Step 5 — hand it to hermes

Register the MCP server once (needs a TTY — `hermes mcp add` is interactive):

```bash
hermes mcp add rebot-arm \
  --command /home/manav/miniconda3/envs/rebot_rs/bin/python \
  --args -m arm_agent.mcp_server
```

Then just talk to it:

```
hermes chat
> can you see the red block?          -> wrist_look
> pick up the red block               -> wrist_pick
> put it in the bin                   -> wrist_place
> put the green sponge in the bin     -> wrist_pick_and_place
```

For the split workflow, keep the same Hermes/MCP process alive between
`wrist_pick` and `wrist_place`. A successful pick leaves the arm at its carry pose
with the gripper holding position. Place refuses to move after an MCP restart or
unless that same process recorded the successful pick. The original
`wrist_pick_and_place` tool remains available for one-call operation.

Set `REBOT_MCP_DRY_RUN=1` in the MCP env to let hermes rehearse a whole
conversation with zero motion.

| tool | moves the arm | needs |
| --- | --- | --- |
| `wrist_look` | no | camera |
| `wrist_teach_drop_zone` | no (reads pose) | arm connected |
| `wrist_pick` | yes; ends holding at carry pose | TCP + grasp-target calibration |
| `wrist_place` | yes; releases and homes | same-process successful pick + drop zone |
| `wrist_pick_and_place` | yes | both |

## Tuning

| env | default | what it does |
| --- | --- | --- |
| `REBOT_WRIST_CAMERA` | `realsense` | camera backend |
| `REBOT_WRIST_APPROACH` | `0.060` | hover height above the object before descending (m) |
| `REBOT_WRIST_NEAR_MM` / `REBOT_WRIST_FAR_MM` | `90` / `600` | detection depth gate — excludes the jaws and the far room |
| `REBOT_WRIST_PITCH` | `1.0` | tool pitch (rad) for deliberate moves — see above |
| `REBOT_WORKSPACE` | `table` | workspace profile: `table` or `teleop` |
| `REBOT_WS_*_MIN/MAX` | from profile | override a single workspace axis (m) |
| `REBOT_GRIPPER_MAX_OPEN` | `0.50` | max gripper travel fraction (ceiling 0.70) |
| `REBOT_WRIST_SCAN` | IK-derived | pin scanning to a single `x,y,z` pose |
| `REBOT_WRIST_TCP` | from `calib/` | override the TCP-in-camera (mm) |
| `REBOT_WRIST_DROP` | from `calib/` | override the release pose (m) |
| `REBOT_ENABLE_LOCATE_ANYTHING` | `0` | allow the open-vocab VLM fallback for non-colour names |

In-module constants: `APPROACH_M`, `ALIGN_TOL_MM` (8 mm), `GAIN` (0.6),
`JAC_STEP` (20 mm), `MAX_ITERS`, `STALL_ITERS`.

## Troubleshooting

| symptom | cause / fix |
| --- | --- |
| `wrist TCP not calibrated` | run step 2 |
| `drop zone not taught` | run step 3 |
| `'X' not found from any scan pose` | not a known colour word and the VLM is off → use a colour name, or set `REBOT_ENABLE_LOCATE_ANYTHING=1`; or the object is outside every scan view |
| `no depth-gated blob of that colour` | object outside `NEAR_MM..FAR_MM`, or the colour band misses it — check the `--snapshot` overlay |
| `Jacobian ill-conditioned` / `barely moved` | the object was mis-detected or occluded during the jogs; clear the view around it |
| `step refused: out_of_bounds` | the grasp pose is outside the safe envelope — move the object further inside the workspace |
| `ik_unreachable` / `could not reach the scan pose` | the pose is inside the box but not reachable at this pitch — lower `REBOT_WRIST_PITCH`, or move the object away from the workspace corners |
| `grasp not confirmed` | aligned but missed: usually a stale TCP calibration (redo step 2) or an object wider than the jaws |
| `did not converge` / `stalled` | detection is noisy — improve lighting, or raise `ALIGN_TOL_MM` |

## Safety

Every motion goes through `arm_agent.arm_safe`: hard workspace AABB, IK verified
*before* moving, per-move step cap, and torque-gated descent that stops on contact
rather than pushing. The gripper is hard-clamped to a 5 %–**50 %** travel window
(testing default; see above).
Nothing here bypasses that gate — and the simulator enforces the same limits
*including the same Pinocchio IK*, so a pose the real gate would refuse is refused
offline too rather than discovered on the bench.

Grasp and release are **verified, not assumed**: a grasp counts only if the jaws
stop short of full close, so a miss is reported as `grasp not confirmed` instead of
a silent success.

## Related

- `arm_agent/depth_servo.py` — the overhead-camera sibling (fixed Orbbec/RealSense).
- `arm_agent/motion_tip.py` — the markerless jaw-motion TCP localizer used by step 2.
- `arm_agent/wrist_sim.py` — the offline geometry model behind `--sim`.
