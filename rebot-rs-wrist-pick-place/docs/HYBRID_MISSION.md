# Wrist-only hybrid missions

An opt-in runtime combines Claw-style bounded mission stages and reusable carry
routes with Hermes-side wrist RGB-D perception, grasp geometry, scene collision
checks, payload-aware transport, and pick verification. It does not run two robot
controllers or transplant Claw's raw joint presets onto a different calibration.
The existing launchers and agent skill packages remain unchanged.

## Usage

Run from the repository root. The launcher uses the existing `rebot_rs` Python;
set `HYBRID_PYTHON` to select another prepared environment.

```bash
# Offline calibration/config inspection; no camera or robot import.
bash run_hybrid_mission.sh "blue marker" --plan

# Camera-backed preview and existing dry checks; motor calls are blocked.
bash run_hybrid_mission.sh "blue marker" --dry

# Scene-relative preview. Coordinates are robot-base frame: +X front, +Y left.
bash run_hybrid_mission.sh "blue marker" --routine scene_place \
  --relation left --reference "red cup" --distance-m 0.08 --dry
```

Only after checking the physical calibration, current camera mounting, cleared
workspace, and emergency stop, replace `--dry` with `--run` to move the robot.
`taught_drop` uses the existing saved wrist drop zone. `scene_place` supports
left/right/front/behind/next_to/stack; stack takes no distance.

Offline checks do **not** establish physical calibration correctness or path
feasibility. Camera dry runs do **not** certify a live trajectory or pick outcome.
No hardware run has been performed as part of this implementation.

## One shared controller for agent clients

```bash
# Persistent stdio MCP server; live requests are refused by default.
bash run_hybrid_mission.sh --mcp

# Explicitly permit a client to request physical execution.
bash run_hybrid_mission.sh --mcp --allow-motion
```

Connect the chosen agent to this command using its MCP configuration. The tools
are `hybrid_start`, `hybrid_status`, and `hybrid_stop`. Start returns a mission ID;
poll status for the result. A live start additionally requires `dry_run=false`.
The runner accepts one task at a time, preserving the wrist pipeline's held-object
state and pre-pick placement plan in one process. Diagnostics go to stderr.

Do not run the legacy MCP server, service, or another motor controller concurrently.
A lifetime lock excludes another hybrid controller, but legacy launchers do not
participate in that lock. Neither project's agent configuration is switched
automatically by this implementation.

## Behaviour and reusable motion

The wrist pipeline emits stages for observation, grasp planning, approach,
checkpoint, grasp, pick verification, held state, placement planning, carry,
release, and return home. Existing bounded grasp recovery remains inside the
wrist pipeline; the outer runner never blindly repeats a failed task.

Recipes live in `config/hybrid_missions.json`. Built-in recipes have no custom
carry points and use existing scene-aware routes. An optional `carry_via_m` list
contains up to eight base-frame XYZ points in metres, paired with the
`calibration_id` printed by `--plan`. The route is generated from the **current**
end-effector pose through these points to the measured release endpoint, then
validated against the current scene, robot, and payload. Rejected preset geometry
falls back to the ordinary validated scene route; it never bypasses safety checks.
Changed calibration/profile identity refuses the recipe until revalidated.

The identity covers calibration artifacts and REBOT environment settings, not
physical mounting or every external model/software file. Keep a specified camera
serial and revalidate after hardware, URDF, or software changes. Changing calibration
in a running controller requires restart; the profile never rewrites calibration.

## Camera and safety boundaries

- Only the wrist RealSense is used. Existing TCP, jaw target, camera rotation,
  gripper, table, and destination calibrations remain authoritative.
- Scene collision checks and measured carried-object collision geometry are
  required for both taught-drop and scene-relative missions.
- The default object geometry mode is the existing stationary RGB-D path with
  scene-frame fusion. Existing pose-aware multi-view object reconstruction remains
  opt-in via `REBOT_OBJECT_GEOMETRY_MODE=hybrid`; the word "hybrid" here names that
  separate reconstruction mode. This implementation does not add an external
  camera or continuous whole-scene remapping during transport.
- Stop is cooperative, checked at stages and motion-send loops. It requests arm
  freeze, **not** automatic release or home. Camera/inference calls may delay stop
  processing; this is not a hardware emergency stop. An uncertain physical state
  blocks further tasks and keeps the controller in the existing operator recovery
  hold instead of silently exiting into automatic shutdown motion.
- Live `succeeded` requires the wrist backend's verification, empty held-object
  state, and confirmed home. `release_verified` is separate from
  `placement_verified`, which remains false: no new post-release visual destination
  check is implemented. A successful release is not proof the object landed correctly.

## Implementation and checks

`arm_agent/hybrid_mission.py` owns recipes, calibration preflight, task ownership,
status, CLI, and MCP. `hybrid_profile.py` supplies the opt-in wrist configuration.
`mission_control.py` supplies dry-run and cancellation guards. Surgical hooks in
`wrist_servo.py` and `local_arm.py` preserve existing non-hybrid behaviour while
exposing stages, validating carry presets, and making hybrid motor loops stoppable.

Hardware-free tests in `tests/test_hybrid_mission.py` cover ownership, completion
evidence, stop behaviour, dry-run motor guards, calibration binding, preset-route
validation, sender cancellation, and simulated wrist pick/place handoff. The MCP
interface has also been smoke-tested for discovery, status, and live-request denial.

```bash
PYTHONPATH=tests:. /home/manav/miniconda3/envs/rebot_rs/bin/python -m unittest -v \
  test_hybrid_mission test_joint_plan_safety test_global_route_integration \
  test_safe_shutdown test_home_verification test_gripper_model
```

## Contact-physics benchmark

`arm_agent.hybrid_physics` is an optional MuJoCo contact benchmark for the current
`scene_place` capability. It uses the production RS URDF, actuated robot joints,
free-jointed cup and object, gravity, contact, rendered wrist RGB-D, the production
hybrid runner, and an independent post-run evaluator. The supported task is a blue,
flat-based upright marker placed robot-left of a red cup—not insertion into it.

The controller receives only rendered RGB-D and simulated encoders. Object/cup poses
are never exposed to it. Objects are neither welded nor mocap-controlled; the report
checks observed lift, displacement, release, final side of cup, and a one-second
settled physics window. A scene or planned route is not counted as task success.

```bash
# Uses the robot Python environment and local optional MuJoCo dependency bundle.
MUJOCO_GL=egl /home/manav/miniconda3/envs/rebot_rs/bin/python \
  -m arm_agent.hybrid_physics --mode hybrid --perception sam3 --object marker

# Interactive live scene. Close the MuJoCo window to end it.
DISPLAY=:0 /home/manav/miniconda3/envs/rebot_rs/bin/python \
  -m arm_agent.hybrid_physics --mode scene --object marker --viewer

# If the native GLFW viewer is unavailable, open the printed local URL instead.
MUJOCO_GL=egl /home/manav/miniconda3/envs/rebot_rs/bin/python \
  -m arm_agent.hybrid_physics --mode scene --object marker --web-viewer

# Physics invariants: free objects, no teleport action, gravity/cup contact,
# renderer contract, reproducible layouts, and no-op failure.
MUJOCO_GL=egl /home/manav/miniconda3/envs/rebot_rs/bin/python \
  -m unittest discover -s tests -p test_hybrid_physics.py -v
```

Each run writes a self-contained report, virtual-only calibration copy, rendered
initial wrist RGB-D, physics trace, command log, scene XML, replay GIF, and still
image beneath `outputs/hybrid_physics/`. The virtual camera mount is explicitly
calibrated from declared simulated geometry; it does not alter the physical wrist
calibration. The simulator uses ideal robot gravity compensation and approximate
jaw/link contact, so it is evidence of control integration—not hardware validation.

Validation on 2026-10-06: all 35 focused tests above passed. Full discovery before
the final two calibration-preflight tests ran 577 tests, with 576 passing and one
error. The error independently reproduces in the unchanged voxel planner:
`test_measured_stack_top_floor_beats_a_coarse_lower_collision_box` expects the
missing successful-route diagnostic `measured_destination_top_floor_m`.
That separate issue was not modified here. Python compilation, launcher syntax,
offline preflight, and MCP discovery/live-denial smoke checks also passed.
