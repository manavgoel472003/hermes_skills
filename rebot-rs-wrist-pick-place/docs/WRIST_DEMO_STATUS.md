# Wrist pick & place — status, gaps, and arm safety

Snapshot as of 2026-07-29. The distinction that matters throughout: **verified on the
real arm**, **verified offline or camera-only**, and **not verified at all**. Nothing
below is marked verified unless it was actually run and its output seen.

---

## 1. Verified on hardware

| Thing | Evidence |
| --- | --- |
| **CAN bus** | gs_usb adapter enumerates as `can0`; up at **1 Mbit** with `txqueuelen 1000`; state `ERROR-ACTIVE`. The default qlen of 10 caused `ENOBUFS` and an `ERROR-PASSIVE` bus. |
| **Arm connects & reports state** | Joint angles stream in free-drive; FK of those angles gives sane tool poses. |
| **Free-drive posing** | `2_zero_and_read.py rebotarm_rs.yaml` (fresh tree) — arm and jaws movable by hand, all 7 values printed. |
| **Working tool pitch** | Measured **0.971 rad (55.6°)** at the user's own "sees the whole table" pose — within 1.7° of the 1.0 rad the code had derived from reachability alone. |
| **Scan / looking pose** | `--look --run --live` reaches **(0.320, 0.000, 0.300)**, streams the wrist view, and **auto-homes on exit** (including Ctrl+C). |
| **Slow, smooth motion** | 15 mm/s tool pace, 1 s floor per move, 0.25 rad/s per joint. No jerking observed. |
| **Live MJPEG feed** | `http://127.0.0.1:8731/` serves multipart JPEG during a run. |
| **Gripper position read-back** | Stable across separate processes (−1.91°, then −1.58° for the same physical state). |
| **Gripper responds at low gain** | Moves at **kp=8**; does **not** move at the config's original **kp=50**. Config now set to kp=8 / kd=1 for the gripper only. |
| **Gripper travel** | Free-drive sweep: `0 → 48.10 → 133.00 → 186.16 → 272.60 → 338.63` (stop). Closed ≈ **0°**, open ≈ **338.63°**; opening is the **more positive** direction. |

## 2. Verified offline (simulation) or camera-only

| Thing | Evidence |
| --- | --- |
| Full servo loop | `--sim` runs scan → Jacobian → servo → descend → grasp-verify → place → home. Position × TCP-error sweep: **15/15**. |
| Sim is kinematically honest | `SimArm.check_ik` calls the **same Pinocchio IK** as hardware, so unreachable poses are refused offline. This is what caught the DEF_PITCH problem. |
| 3-D replay | `wrist_replay` pushes the sim trajectory through the real URDF + IK: **0.00 mm mean FK error**. |
| D405 camera | 640×480 aligned RGB-D; deprojection round-trip exact to 0.0000 px. |
| Colour detector | HSV + depth gate; correctly finds/rejects on live frames. |
| Name-free detection | Depth-plane segmentation found **8 objects** on a live frame; colour-blind (red/grey/black all detected identically). |
| MCP surface | 31 tools register; 5 `wrist_*` tools present; pre-existing tools intact. |
| Failure paths | Missing TCP, absent object, unreachable pose, oversize object — all fail with actionable reasons. |

## 3. NOT verified — the real gaps

1. **No object has ever been picked.** The grasp itself is entirely unproven on hardware.
2. **Gripper open/close to a commanded angle.** The jaws have moved (once, at kp=8, and once "almost full open"), but a *controlled* move to a target with the corrected limits has not been demonstrated.
3. **TCP calibration.** The one attempt stored a garbage value (443 mm from the lens — a marker on the table). Guards were added; a good calibration has not been captured.
4. **Jaw mm-per-radian.** Never calibrated, so sized opening is inactive and the gripper falls back to the fixed limits.
5. **Torque-gated descent.** `safe_descend` depends on `joint_torques()` from the RS backend. **Nobody has confirmed the RS motors report meaningful torque.** If it always reads ~0, contact detection silently does nothing.
6. **Table height.** Unmeasured. The Z floor is set to 0.00 by inference.
7. **Drop zone.** Taught at (0.275, −0.187, 0.059) but never used in a real place.
8. **Camera framing from the scan pose.** The arm reaches it and streams, but whether the table is usefully framed is a judgement call not yet recorded.

## 4. Exact next steps

Every command runs from `~/rebot_lerobot` unless stated. Each step says what to look
for and what to do if it does not hold. Do them in order — later steps assume the
earlier ones passed.

### Step 0 — every boot or replug: bring up CAN

```bash
sudo ip link set can0 down
sudo ip link set can0 txqueuelen 1000
sudo ip link set can0 type can bitrate 1000000
sudo ip link set can0 up
ip -details -statistics link show can0 | grep -E "state|qlen"
```

**Expect** `state ERROR-ACTIVE` and `qlen 1000`.
**If** it shows `ERROR-PASSIVE` while idle → electrical (termination / a node not
ACKing). Stop; software pacing will not fix it.

### Step 1 — SAFETY GATE: is torque feedback real?

Nothing that descends is safe until this passes.

```bash
python -m arm_agent.wrist_servo --state
```

Read `joint_torque_nm` and `torque_all_zero`. Then **press gently down on the arm by
hand** and run it again.

**Expect** the torque numbers to change noticeably between the two reads.
**If** `torque_all_zero: true` both times → `safe_descend` has no contact signal and
will drive to the Z floor regardless of what is in the way. **Do not run a pick.**
Tell me and we will gate the descent on something else (commanded-vs-actual position
stall alone, plus a hard Z floor measured in step 3).

### Step 2 — does the gripper track a commanded angle?

Work up in steps; do not jump to the limit.

```bash
python -m arm_agent.wrist_servo --gripper read
python -m arm_agent.wrist_servo --gripper open  --run --to-deg 50
python -m arm_agent.wrist_servo --gripper open  --run --to-deg 150
python -m arm_agent.wrist_servo --gripper close --run --to-deg 20
```

**Expect** `position_deg` to approach `target_deg` each time, easing over ~1.5 s.
**If** it does not move → try `--gripper-kp 4 --gripper-kd 0.5`.
**If** `position_deg` moves *away* from the target → the drive polarity is inverted
relative to the encoder; stop and tell me, that needs a sign flip in the command, not
more driving.

### Step 3 — measure the table height

With the arm in free-drive, put the gripper tip down on the table where you will
place objects:

```bash
cd ~/rebot_lerobot/reBotArm_control_py_fresh
python example/2_zero_and_read.py rebotarm_rs.yaml      # jaws CLOSED before this runs
# pose the tip on the table, Ctrl+C, then:
cd ~/rebot_lerobot
python -m arm_agent.wrist_servo --state
```

Send me `tool_position_m`. Its **z** is the table height, and I will set the descent
floor from it instead of the current inferred 0.00.

> `2_zero_and_read.py` calls `set_zero()` on every joint at startup. Have the **jaws
> closed** when you launch it, or the gripper limits shift and must be re-measured.

### Step 4 — calibrate the TCP

Put the object **between the jaws**, then:

```bash
python -m arm_agent.wrist_servo --calib-grasp "blue marker" --run
```

**Expect** `ok: true` and a TCP within 20–250 mm of the lens.
**If** it refuses with "could not confirm the object is gripped" → the jaws did not
close on it (step 2 first).
**If** it refuses on distance → the detector found a *different* marker on the table;
remove it, or check the view with `--watch --object "blue marker" --live`.

### Step 5 — check the scene is visible from the scan pose

```bash
python -m arm_agent.wrist_servo --look --run --live
# open http://127.0.0.1:8731/ , Ctrl+C when done (it homes itself)
```

**Expect** the table filling a useful part of the frame and the depth inset
(bottom-right) showing surface, not black. Black = outside the D405 range or past the
600 mm gate.
**If** coverage is poor, try `--scan-index 1..4` for the four quadrant poses.

### Step 6 — dry pick, then the real one

```bash
python -m arm_agent.wrist_servo --pick "blue marker" --dry --live
python -m arm_agent.wrist_servo --pick-place any --run --live
```

`--dry` moves nothing. Keep a hand near the e-stop for the first `--run`; it takes a
few minutes at 15 mm/s. If anything looks wrong, Ctrl+C then `--home --run`.

**If it fails**, paste the `steps` list from the JSON — each stage reports its own
pass/fail reason, which localises the problem immediately.

### Step 7 — optional: sized opening

Only needed if you want the jaws to open to fit each object and to refuse oversize
ones. Skip it and the gripper just uses the fixed limits.

```bash
python -m arm_agent.wrist_servo --gripper open --run --to-deg 150
# measure the jaw gap with calipers, then:
python -m arm_agent.wrist_servo --set-jaw-span <mm>
```

### Step 8 — hand it to hermes

```bash
hermes mcp add rebot-arm \
  --command /home/manav/miniconda3/envs/rebot_rs/bin/python \
  --args -m arm_agent.mcp_server
hermes chat        # "put the red block in the bin"
```

Set `REBOT_MCP_DRY_RUN=1` in the MCP env to rehearse a whole conversation with zero
motion.

---

## 5. Arm safety

### Already implemented

- **Every motion is IK-verified before it executes** — bounds, then a Pinocchio solve that must actually reach the pose, then a per-move step cap (~9 mm). A refused move produces *no motion at all*.
- **Long moves are split** into capped, individually verified sub-steps (`safe_goto`), so the arm never lunges.
- **Torque-gated descent** stops on a torque jump or a position stall rather than pushing (**untested — see below**).
- **Gripper hard clamp, 15–150°**, non-configurable: no skill and no env override can drive the jaws to either mechanical stop.
- **Gripper ramping** — targets walk across in 30 increments over 1.5 s instead of a single jump, which is what previously slammed the CAN bus with saturated commands.
- **Already-closed guard** — a close command is skipped rather than re-driving into the stop (the cause of the motor fault, status 12).
- **Out-of-range read detection** — a jaw reading outside the measured limits is reported as a fault instead of silently trusted.
- **Slow pace** — 15 mm/s, 0.25 rad/s, 1 s minimum per move.
- **Auto-home only after a confirmed release**, never while holding an object.
- **Grasp and release are verified**, not assumed — a miss reports `grasp not confirmed`.

### Recommended before unattended running

1. **Prove torque feedback is real.** *The highest-value safety check left.* Read `joint_torque` while pressing gently on the arm; if the numbers do not move, `safe_descend` is blind and the arm will drive to the Z floor regardless of what it hits. Everything about "descend by feel" rests on this.
2. **Measure the table and set the Z floor to it.** Currently 0.00 by inference. A floor below the table is only safe *because* the torque gate is expected to stop first — which is unproven (1).
3. **Interrupt handling for `--pick`.** `--look` homes on Ctrl+C; the pick path does not. Interrupting mid-descent leaves the arm holding position over the table. Worth giving the pick the same `finally: home` treatment, or at least a documented "press Ctrl+C then run `--home --run`".
4. **A real e-stop.** Today the only abort is Ctrl+C. `POST /estop` exists in the HTTP service but not on the local path. Physical e-stop wired to motor power is the honest answer.
5. **Make the CAN settings persistent.** Bitrate and `txqueuelen 1000` are lost on reboot or replug. A udev rule or systemd unit avoids running with a 10-frame queue, which previously produced `ENOBUFS` and an `ERROR-PASSIVE` bus.
6. **Watch for `ERROR-PASSIVE` returning.** If it reappears while idle, that is electrical — bus termination (120 Ω at both ends) or a node not ACKing — and no amount of software pacing fixes it.
7. **Investigate `[safe_home] trajectory send timeout`.** Seen during homing; it completed, but it indicates a send that did not finish cleanly on the same bus the gripper shares.
8. **Investigate `ensure_mode failed ... 0x7005 not received within 1s` on joint1.** Seen at connect. A joint that has not confirmed its control mode may not respond as expected.
9. **No obstacle or self-collision model.** IK checks URDF joint limits only. The traverse to the drop zone is a straight line at carry height — keep the path between the table and the bin clear.
10. **The workspace box was widened** to x[0.20,0.44] y[±0.24] z[0.00,0.30] so the arm can reach the table at all. The AABB is a coarse filter; per-move IK verification is the real gate. Revert with `REBOT_WORKSPACE=teleop` if a tighter envelope is wanted.

---

## 6. Facts learned the hard way

- **`set_zero()` invalidates every stored angle.** It makes the *current* position the new origin. `2_zero_and_read.py` calls it at startup on all joints including the gripper. Always have the **jaws closed** when anything zeroes them, or re-measure afterwards. This killed one set of limits mid-session.
- **The gripper needs a low MIT gain.** kp=50 produced no motion at all; kp=8 works.
- **A RobStride motor only answers feedback requests once enabled.** Reading it while disabled returns nothing.
- **Two control-py trees exist and have diverged.** Use `reBotArm_control_py_fresh/` only; it takes the hardware yaml positionally (`python example/2_zero_and_read.py rebotarm_rs.yaml`).
- **`arm_agent` modules must be run from `~/rebot_lerobot`**, not from inside the control-py directory.
- **At the old default pitch (1.30 rad) the arm reaches only ~23 mm above the table** — not enough to lift anything. The demo works at 1.0 rad, and scan/carry heights are probed from IK rather than fixed offsets.
