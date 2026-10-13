# Hybrid manipulation: Claw motion routines with Hermes perception

Status: initial wrist-only runtime implemented; hardware validation remains pending.
See [implementation and usage](HYBRID_MISSION.md). The architecture below also
includes future extensions, not all of which are implemented. The goal is one manipulation pipeline combining the Claw
project's preset mission structure with Hermes-side perception, grasp geometry,
and outcome verification. Connecting two agent interfaces is secondary. One task
planner is sufficient; HTTP/MCP can expose the resulting skills afterward.

## Recommended blend

Use preset routines to express repeatable motion structure. Bind their targets to
measured object and destination geometry. Execute validated motion segments locally,
then refresh perception at stationary checkpoints. The task planner chooses an
objective and handles exceptions; it does not choose every joint increment.

In this design, a "reflex" is a local rule responding to measured state, such as
holding when an observation is invalid or refusing to close before jaw occupancy
is confirmed. A preset is a reusable motion routine. They are complementary but
different: a preset alone cannot detect failure, and a Python polling loop is not
a hard-real-time controller.

## What the two projects actually contain

- `rebot_arm_service/SKILL.md` describes an OpenClaw integration with the HTTP
  service. The directory contains the service, WebUI, camera APIs, mission APIs,
  scene perception, and a Python manipulation skill library. It does not contain
  an independent OpenClaw agent implementation.
- `arm_agent/` contains an Ollama planner and task executor, plus an MCP server
  usable by the external Hermes agent. These are two distinct planning entrypoints.
  Its newer wrist workflow owns camera, perception, and local arm control directly.
  Some MCP tools still call the older HTTP service.

Consequently, the older documentation describing the service as the single hardware
owner does not describe every current execution path.

## Target architecture

```text
Task planner: object + destination + relation
                     |
Mission state machine: observe -> acquire -> grasp -> carry -> place -> verify
          |                         |                          |
Hermes perception          Claw-style routines          Outcome checks
scene + identity + RGB-D    taught poses + parameterized  stationary/held/
grasp/payload geometry     motion segments               released/placed
          |                         |                          |
          +---------- current plan and task context ------------+
                                    |
                    Local execution and motion checks
                    one arm owner / one camera owner
                                    |
                                Hardware
```

The robot execution process owns the camera and hardware connection. MCP and HTTP
are interfaces to that process. Starting another agent must not start another arm
controller. Only one task may own manipulation at a time, including the interval
between picking an object and placing it.

## Capabilities to retain

| Existing source | Contribution to the hybrid |
| --- | --- |
| `rebot_arm_service/src/rebot_arm_service/service.py::_run_teddy_grasp_mission` | Explicit observe/search/grasp/place/reset sequence; extract its stage structure |
| `rebot_arm_service/config/teddy_bear_mission.yaml` | Named observation, placement, and initial joint poses plus configurable motion timing |
| `service.py::_bottle_tracking_step` | Detection confidence gates, deadband, and bounded correction pattern; replace largest-box identity and fixed axis assumptions |
| `rebot_arm_service/src/rebot_arm_service/app.py` | HTTP interface, health/status, camera endpoints, and WebUI integration |
| `rebot_arm_service/src/rebot_arm_service/skills/library.py` | Discoverable skill specifications and structured execution results; migrate compatible workflows individually |
| `arm_agent/executor.py` | Optional local plan, act, verify, and replan loop |
| `arm_agent/mcp_server.py` | Agent-facing tool names and descriptions, converted to remote client calls |
| `arm_agent/wrist_servo.py` | Wrist-camera pick, lift verification, held-object context, and placement workflows |
| `arm_agent/local_arm.py`, `arm_agent/arm_safe.py` | Existing local hardware execution and motion checks |
| `arm_agent/scene_collision.py` | Scene and payload collision checks where configured and supported |

Keep the current wrist implementation intact during the first extraction. Preserve
its configuration and checks rather than combining calibration values from different
hardware configurations. Collision checking is configuration-dependent; exposing a
tool must not imply that every collision feature is enabled.

## What the deeper code trace establishes

- The crawfish and teddy HTTP mission routes both call `start_teddy_mission`.
  Their preset runner uses `VisionService` (YOLO), rather than the separate VLM
  scene service. Semantic task selection needs an explicit connection to that runner.
- The mission moves to a taught overhead pose, opens, searches with joint 1/joint 4,
  centres the largest detection matching a class, computes a tool-ray intersection
  with a fixed Z plane, closes, visits a taught placement pose, and resets/disconnects.
  Keep the sequence concept; use Hermes geometry instead of the ray-plane grasp.
- `_move_joints_smooth` linearly updates the controller target with sleeps. It does
  not itself provide dense scene/payload collision validation or endpoint feedback
  verification. Its preset data should feed the newer validation/execution boundary.
- The successful mission path contains 19 seconds of explicit waits after grasp
  approach, closing, placement transit, opening, and reset. Replace these with
  bounded feedback/visual completion conditions where supported. Actual cycle-time
  savings must be measured, including required camera settling and inference.
- A stop breaks the search loop but does not return before the subsequent grasp
  sequence. `motion_enabled=False` also does not gate the initial preset move/open
  in this runner. Importing this function unchanged is not an integration strategy.
- Nearby grasp fallback tries positional offsets up to 20 mm for reachability.
  A replacement candidate must still grasp the selected physical object; reuse
  Hermes's measured candidate geometry rather than translating the endpoint blindly.
- Hermes already implements the intended structural blend: `FIXED_SCENE_TRAJECTORY`
  freezes a prevalidated macro approach to a visible checkpoint, where stationary
  geometry may produce a replacement final leg. `SIMPLE_GRASP_FLOW_ENABLED` adds
  bounded retry after verified release/home. Preserve and expose these mechanisms.
- `LocalArm.validate_joint_path` and `execute_joint_plan` provide scene-version,
  start-state, age, and path-integrity checks. Extend this boundary for taught paths;
  do not manufacture a plan certificate or call private controller targets from a
  new preset runner. Existing execution holds a lock during its send loop, so
  prompt cooperative cancellation requires changes inside execution, not just an
  extra HTTP stop endpoint.

## How each integration improves both sides

| Integration | Claw-side improvement | Hermes-side improvement |
| --- | --- | --- |
| Semantic target -> bounded acquisition routine | Search for the requested instance instead of the largest same-class box | Reuse known observation/search poses and structured acquisition failures |
| RGB-D grasp -> parameterized approach routine | Replace fixed grasp height and camera-axis assumptions with measured geometry | Execute repeatable approach segments; re-evaluate at declared checkpoints |
| Verified grasp -> preset carry routine | Refuse carry after an unverified grasp; account for the full held object | Reuse known transit poses/corridors after current-scene validation |
| Destination geometry -> placement routine | Support relational placement as well as a taught bin | Retain simple taught destinations for recurring jobs |
| Explicit completion checks -> next mission stage | Replace blind waits and completion assumptions with evidence | Give the task planner stable, actionable stage outcomes |
| Local recovery policy -> task-level replanning | Add context-aware bounded recovery and escalation | Resolve routine cases without a new language-model decision |

These are expected benefits, not measured performance results.

## Routine types and perception checkpoints

Maintain two kinds of motion routine:

1. **Taught joint routines:** observe, scan viewpoints, transit, named drop zone,
   and park. Each belongs to a robot model, joint-zero version, tool, and camera
   mount. Validate the connector from the live start and every segment against the
   current scene and payload; a known endpoint does not make the whole path valid.
2. **Parameterized Cartesian routines:** approach the selected grasp, align,
   close, micro-lift, clear the support, approach a destination, release, retreat.
   Bind offsets to explicit base/tool/object frames, solve IK from the current
   configuration, and validate the resulting path. Fixed structure does not imply
   fixed world coordinates.

| Stage | Main computation | Completion condition / next decision |
| --- | --- | --- |
| Observe | Validated taught observation pose; scene detection | Arm stationary, current calibrated frame, target/destination hypotheses |
| Acquire | Bounded search views and semantic identity association | Same target confirmed; otherwise stop search at its budget and report |
| Plan grasp | Hermes masks, RGB-D body geometry, jaw fit, IK/scene checks | Valid grasp plus full approach and visible checkpoint |
| Approach | Execute frozen validated macro segment | Measured endpoint reached and stationary |
| Checkpoint | Fresh same-target observation and geometry checks | Revalidate final leg; if inconsistent, bounded reacquire/replan |
| Grasp | Final approach, sized closing, micro-lift | Visual hold verification; motor angle alone is not an object certificate |
| Carry | Taught corridor candidate or computed route | Current-scene validation including held-object envelope |
| Place | Taught destination or measured support-relative target | Release pose accounts for object-to-tool offset and object size |
| Verify | Confirm release, then observe destination | Record released and placement-verified separately; update scene |
| Finish | Validated retreat/park | End task or begin the next object without mandatory disconnect |

Semantic inference establishes or reacquires identity. Local depth/geometry checks
may maintain it only while their evidence remains valid. Do not assume colour-only
tracking can replace semantic identity, or promise an inference rate without timing
the actual detector. Use the existing mask/depth stages first; any cheaper tracker
needs evaluation for identity switches and close-range occlusion.

An invalid view at a checkpoint causes a hold and bounded reacquisition, not blind
continuation. Replanning must first hold/settle and use the measured new start.
Known corridor blocked -> validate another corridor -> bounded global route search
-> fail explicitly if no route passes. A local retry must preserve the requested
object and placement relation. Freeze, release, and home are distinct decisions;
releasing a held object is not the universal response to lost vision.

## Data passed between the stages

These are proposed contracts, not implemented APIs:

- `TaskSpec`: object identity/description, destination identity or named zone,
  relation and reference frame, selected routine, retry/deadline budget.
- `Observation`: frame ID/time, camera serial and mount, calibration version,
  associated robot pose, object identity, geometry, and quality/uncertainty.
- `GraspPlan`: target observation, measured grasp frame, jaw aperture, candidate
  geometry, approach/checkpoint/final segments, active limits and scene version.
- `MotionSegment`: joint or Cartesian routine instance, permitted start, validated
  path, payload state, expiration, cancellation handling, completion predicate.
- `HeldObject`: object identity, verified/unknown/empty state, grasp attempt ID,
  tool-to-object transform, body envelope, measurement/calibration versions.
- `PlacementPlan`: named zone or reference object, support surface, desired object
  pose, derived tool release pose, carry/release/retreat segments and validation.
- `StageResult`: status, structured failure code, observations used, actual endpoint,
  verification evidence, elapsed time, retry eligibility and recovery outcome.

Existing scene IDs (`obj0`, etc.) are regenerated per scan; they are not persistent
task identities. Existing wrist held state is process-local. New contracts should
first wrap that state and preserve its lifetime, then replace the module globals
through a tested extraction. Do not reconstruct a verified hold after restart from
an old JSON record.

## Calibration and configuration prerequisites

- Bind camera selection to calibration selection. Current public `/scene` and
  `/vision/locate` routes apply `table_calib` even for other requested cameras.
- Keep table-plane homographies separate from depth extrinsics. The saved service
  table Z is 0; wrist Z is 0.01196 m. Establish tool/base origin equivalence before
  interpreting this as a measurable physical offset.
- Preserve the separate held-object TCP and open-jaw target (27.8 mm apart in the
  saved measurements). The saved wrist mapping has determinant -1 because it is a
  signed motion mapping, not a conventional rotation to paste into a transform tree.
- Legacy gripper commands use negative angles; the current wrist model commands
  within 15–150 degrees. Bind routine gripper actions to the selected calibrated
  backend, never copy motor coordinates between them.
- The Orbbec mappings report 3.3–3.7 mm fitting error across six samples but have no
  source-code consumers found in the workspace search. Validate reference-point
  semantics and held-out poses before adopting them. The saved Reachy DLT reports
  627 mm mean held-out error and is unsuitable as manipulation calibration.
- Standardize relation semantics: legacy service `front` is -X, wrist `front` is
  +X. Express canonical directions explicitly at the interface boundary.
- Put effective limits, geometry mode, collision options and grasp-flow settings
  in one selected profile. MCP and CLI currently select different defaults.

## Integration issues to resolve

1. **Shared state and ownership.** The wrist workflow keeps `_OBJECT_HELD` and
   `_FIXED_HELD_CONTEXT` in memory. Both interfaces must reach the same process.
   Restarting it invalidates that context and requires physical state reconciliation.
   A task owner must retain control through a split pick/place sequence; serializing
   individual HTTP requests alone is insufficient.
2. **Compatible skill families.** Currently MCP `pick_object` calls the HTTP skill
   library, while MCP `place_beside` calls wrist `fixed_place`. The latter requires
   a preceding wrist pick. Advertise explicit workflow families initially and reject
   incompatible combinations. Add generic aliases only when routing can preserve
   the complete pick/place workflow and its state.
3. **Hardware selection.** Select one hardware backend per deployment. Adapt older
   skills and missions to that owner before enabling them alongside wrist skills.
   Do not route a failed motion to a different hardware controller automatically.
4. **Dry runs.** Enforce execution mode centrally. Current MCP direct-control
   helpers can connect and invoke gripper/home without checking `_DRY_RUN`.
   A hybrid dry run must block those actions before hardware access and distinguish
   simulated outcomes from verified physical outcomes.
5. **Stop behavior.** Current `wrist_stop` invokes recovery that can open the
   gripper and home. Preserve this as an explicitly named recovery operation.
   A stop request during a running task needs a cooperative cancellation path that
   is not queued behind that task; define freeze and subsequent recovery separately.
6. **Configuration and access.** Workspace values and default service ports differ
   across existing entrypoints. Use one deployment configuration and expose active
   capabilities to agents. The existing OpenClaw network policy only permits a
   subset of routes; new skill routes need an explicit corresponding policy update.

## Implementation order

1. Record comparable stage traces from the current workflows; establish one
   deployment/calibration profile and canonical spatial relations. Keep physical
   baseline results distinct from simulated tests.
2. Introduce the mission state machine and task context with fake backends first.
   Use complete existing `fixed_pick`/`fixed_place` calls as coarse adapters so the
   held-object certificate stays intact. Do not add a preset approach or home
   around a function that already performs those actions internally.
3. Extract explicit observation, approach/checkpoint, grasp/verify, carry, and
   release/verify boundaries from the existing wrist implementation, preserving
   its tested geometry. Several stages are currently nested in a large function;
   this requires a careful extraction, not just registering new tool names.
4. Add taught observation/transit/destination routines at those boundaries. Route
   preset paths through backend validation and feedback completion. Add cooperative
   cancellation inside the actual sender before advertising responsive stop.
5. Connect semantic scene selection to bounded search and target confirmation.
   Add relation-aware placement and post-placement verification using the same
   task identity, calibration versions, and held-object geometry.
6. Expose the unified skills over HTTP/MCP and the WebUI. Task requests enter one
   state machine; raw legacy mission execution must not independently own the arm.

The first physical demonstration should be one known object to one taught drop
zone: taught observation -> Hermes grasp -> verified hold -> validated taught carry
and release -> visual check. Then add relational placement, clutter, and repeated
objects. This makes the benefit of each added layer observable.

## Acceptance criteria

- HTTP and MCP expose matching schemas and return consistent results.
- Dry-run manipulation, gripper, home, and recovery never connect to or command
  physical actuators; simulations do not establish a live held-object state.
- Conflicting tasks are refused while another task owns the arm, including a hold
  between pick and place. Another agent cannot silently take over that task.
- Pick/place retains one execution context; an incompatible family or missing
  context fails before motion.
- Connection failures, cancellation, and process restarts produce explicit state
  and do not replay an uncertain motion automatically.
- Existing wrist regression tests continue to pass. Hardware acceptance includes
  the marker/sponge and placement criteria recorded in `arm_agent/tests.md`.
- A cancelled acquisition never falls through into grasp; cancellation during a
  segment prevents subsequent targets and reports measured hold state.
- Target identity changes and missing checkpoint evidence prevent final approach.
- Taught paths are refused when the start, robot/calibration identity, payload, or
  scene is incompatible; current observations govern reuse.
- The same task/profile selects the same routines and limits through CLI, HTTP,
  and MCP.

Measure grasp and placement success separately, wrong-object selections, geometric
placement error, cycle time by stage, inference count, recovery success, and stop
latency. Compare both original pipelines and the hybrid on the same scenes and
hardware profile. Do not trade fewer model calls for unverified motion.

## Validation performed during this design review

The prior calibration review passed 15 focused tests covering camera mapping,
geometry, and simulated pick/place handoff. This deeper review additionally ran:

```text
PYTHONPATH=tests:. /home/manav/miniconda3/envs/rebot_rs/bin/python -m unittest -v \
  test_joint_plan_safety test_global_route_integration
```

All six tests passed. They check joint bounds, stale scene rejection, path integrity,
global-route fallback, visible-checkpoint routing, and certified-plan dispatch.
These original checks validate existing foundations. New hybrid tests and remaining
limitations are described in [implementation and usage](HYBRID_MISSION.md). Neither
establishes live grasp performance or cancellation latency.

The proposed deliverable is a behaviour integration: reusable motion routines
guided by scene-aware geometry, with explicit local completion and failure rules.
