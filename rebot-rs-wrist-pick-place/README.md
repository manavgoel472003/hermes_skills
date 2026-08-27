# reBot RS wrist-camera pick/place

This is the known-good source snapshot for the fixed, preplanned RS wrist-camera
pick/place workflow, refreshed from the hardware-tested workspace on 2026-08-11.

Pick and placement are exposed independently through the long-lived Hermes MCP
controller. A successful pick creates an in-memory hold certificate and measured
payload context; placement consumes that certificate, releases, and homes. The
combined operation remains available and uses the same two phase implementations.

## Verified hardware outcome

- The wrist camera detected and depth-sized a blue marker.
- A noisy measured camera Jacobian was projected onto a rigid translation model.
- Deterministic continuation IK prevalidated the complete Cartesian route without
  random seed branches.
- Grounding DINO Base identifies the requested whole object and SAM 2.1 Large
  supplies its mask; conjunctive multi-colour RGB-D matching requires every
  requested colour on the same nearby object, while forward/backward optical
  flow preserve instance identity during calibration jogs.
- The object is checked for jaw fit before motion and measured again between the
  open jaws at the final endpoint.
- The gripper closes through slow bounded position ramps to the image-derived
  width, then performs a 15 mm visual verification lift.
- A noisy camera Jacobian is refused above a 0.15 rigid-fit error instead of
  being allowed to create a shallow or sideways endpoint.
- Ctrl+C and failure recovery freeze before opening/homing.
- If the first emergency jaw-open is interrupted, recovery retries it at home.
- Motor disable is refused unless home is independently verified.
- Prevalidated RS waypoint chains run as one continuous 50 Hz quintic
  minimum-jerk joint trajectory, rather than stopping at each internal waypoint.
- A visible endpoint miss can trigger up to two separately IK-validated,
  12 mm-capped open-jaw corrections before the existing jaw-slot gate is applied.
- Scene-relative placement localizes a stable visible reference in the robot/table
  frame and supports left, right, front, behind, and next-to requests.
- Optional fail-closed collision checking converts RealSense depth into inflated
  base-frame obstacle columns. Pinocchio/Coal checks RS-URDF self-collision, the
  calibrated table, visible clutter, and an attached payload before motion.
- Carry planning evaluates direct, X-first, Y-first, elevated, and lateral
  corridors and refuses motion when no complete collision-free route exists.
- Every constructed map is exported as a dense, denoised, camera-colored PLY and
  interactive Rerun recording. Viewing uses 2 mm RGB voxels across the surrounding
  visible scene while collision checks retain their separate conservative map.

## Safety boundary

There is no validated force/contact signal. Closing is image-sized and
position-only, with immutable 15–150 degree command limits. Novel fragile or
unstable objects still require supervised trials; detection is not proof that an
object can be safely grasped. The collision map covers the static scene visible
from the ready-pose wrist camera; it cannot guarantee clearance from hidden or
moving obstacles.

## CAN setup

`can0` runs at **1,000,000 bit/s** — not the earlier 500,000. Run this after
booting, reconnecting the adapter, or recovering from a CAN failure. Full
detail, including motor-feedback verification, is in `CAN.md`.

```bash
sudo ip link set can0 down
sudo ip link set can0 type can bitrate 1000000
sudo ip link set can0 txqueuelen 1000
sudo ip link set can0 up
ip -details link show can0        # expect: UP, bitrate 1000000, qlen 1000
```

`txqueuelen 1000` does not survive a reboot or a `link down`, and must be
re-applied after either. The socketcan default of 10 is too small for the
burst `disable_all` sends, which surfaces as
`socketcan write failed: No buffer space available (os error 105)`.

Reset a wedged bus, then disable the motors:

```bash
sudo ip link set can0 down && sudo ip link set can0 up
sudo ip link set can0 txqueuelen 1000

cd <robot workspace>/reBotArm_control_py_fresh
python example/disable_all.py rebotarm_rs.yaml
```

`ERROR-ACTIVE` is the healthy bus state. `ERROR-WARNING`, `ERROR-PASSIVE` or
`BUS-OFF` indicate wiring, termination, or a bitrate mismatch.

## Usage

```bash
REBOT_GRIPPER_SIZE_COMPRESSION_MM=6 \
REBOT_WS_X_MAX=0.60 \
REBOT_HYBRID_TARGET_CLUSTER_RADIUS_M=0.11 \
REBOT_OBJECT_GEOMETRY_MODE=hybrid \
REBOT_SCENE_COLLISION_EXPORT_VIEW=1 \
REBOT_PAYLOAD_COLLISION=0 \
./run_scene_pick_place.sh --run \
  "<object>" <relation> "<reference>" 2>&1 | tee /tmp/pick_run.log
```

Relations, exactly as the CLI accepts them: `drop_zone` `free_space` `left`
`right` `front` `behind` `next_to` `stack`. `stack` releases at the reference
*centre* and so has no lateral error budget; the side-relative relations add
clearance and tolerate a looser reference.

Why each variable is set:

- `GRIPPER_SIZE_COMPRESSION_MM=6` — closes the jaws 6 mm inside the measured
  width so they bite. The default is 0 because `wrist_sim` treats a command
  more than 2 mm inside the object as a miss, so a raised default would report
  every simulated grasp as unheld. A round marker needs this: it has no flat to
  seat against and rolls out of a grip that merely touches it.
- `WS_X_MAX=0.60` — the workspace AABB is a coarse pre-filter only (see
  `arm_agent/arm_safe.py`); every waypoint is still IK-verified and every
  descent torque-gated. The stock 0.44 refuses reachable targets before IK runs.
- `HYBRID_TARGET_CLUSTER_RADIUS_M=0.11` — identity radius. Raise only as far as
  needed; it is what keeps a same-coloured decoy out.
- `OBJECT_GEOMETRY_MODE=hybrid` — multi-view metric object box.

`Ctrl+C` or `Q` stops at any stage and runs the clean shutdown: freeze, open
gripper, home. `kill -9` skips it and leaves the motors energised.

`arm_agent/tests.md` records the acceptance conditions, the scene-setup rules
that dominate the failure modes, and how to read a run log.

## Source layout

- `arm_agent/` — captured implementation.
- `calib/wrist_gripper.json` — measured jaw calibration (mm-per-radian span).
  Machine-specific; committed here so the branch is self-contained, and
  overridable with `REBOT_WRIST_CALIB`.
- `tests/` — safety, calibration, detection, and workflow regression tests.
- `SKILL.md` — Hermes operating contract and validated commands.
- `CAN.md` — full CAN bring-up and verification.

The implementation expects the existing robot workspace, including
`reBotArm_control_py_fresh`, its RS hardware YAML, the `rebot_rs` environment,
CAN configuration, and machine-specific calibration files. Secrets are not
committed; the gripper calibration now is, under `calib/`.

## Validation

From the robot workspace:

```bash
python -m unittest discover -s tests
```

The refreshed workspace passes the 172-test wrist workflow suite and the
10-test voxel-navigation suite. Hardware-dependent local-arm tests additionally
require the robot workspace's controller package, URDF, and private calibration
files, which are intentionally not duplicated here.
