# reBot RS wrist-camera pick/place

This is the known-good source snapshot for the fixed, preplanned RS wrist-camera
pick/place workflow developed and hardware-tested on 2026-07-30.

## Verified hardware outcome

- The wrist camera detected and depth-sized a blue marker.
- A noisy measured camera Jacobian was projected onto a rigid translation model.
- Deterministic continuation IK prevalidated the complete Cartesian route without
  random seed branches.
- The gripper opened at a preplanned waypoint, closed to detected width minus
  1 mm, lifted the marker, released it over the taught drop zone, and homed.
- Ctrl+C and failure recovery freeze before opening/homing.
- Motor disable is refused unless home is independently verified.

## Known limitation

A green bottle cap is detectable, but the current translation-only approach can
topple the bottle body. Constrained RPY/posture planning for bottle geometry is
future work and is deliberately not included in this known-good baseline.

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

The baseline passed 37 tests before publication.
