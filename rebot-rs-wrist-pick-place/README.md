# reBot RS wrist-camera pick/place

This is the known-good source snapshot for the fixed, preplanned RS wrist-camera
pick/place workflow developed and hardware-tested through 2026-07-31.

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

## Safety boundary

There is no validated force/contact signal. Closing is image-sized and
position-only, with immutable 15–150 degree command limits. Novel fragile or
unstable objects still require supervised trials; detection is not proof that an
object can be safely grasped.

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

The snapshot passes 74 tests.
