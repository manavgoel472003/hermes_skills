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

## Source layout

- `arm_agent/` — captured implementation.
- `tests/` — safety, calibration, detection, and workflow regression tests.
- `SKILL.md` — Hermes operating contract and validated commands.

The implementation expects the existing robot workspace, including
`reBotArm_control_py_fresh`, its RS hardware YAML, the `rebot_rs` environment,
CAN configuration, and machine-specific calibration files. Calibration JSON and
secrets are not committed.

## Validation

From the robot workspace:

```bash
python -m unittest discover -s tests
```

The refreshed workspace passes the 172-test wrist workflow suite and the
10-test voxel-navigation suite. Hardware-dependent local-arm tests additionally
require the robot workspace's controller package, URDF, and private calibration
files, which are intentionally not duplicated here.
