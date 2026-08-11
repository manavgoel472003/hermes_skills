---
name: rebot-rs-wrist-place
description: Place and release the object previously secured by reBot RS wrist_pick, using a taught drop point or a scene-relative reference.
version: 1.0.0
author: Manav Goel
metadata:
  hermes:
    tags: [robotics, rebot, realsense, place, collision-avoidance, safety]
prerequisites:
  commands: [bash]
---

# reBot RS wrist place

Use the `wrist_place` MCP tool only after `wrist_pick` succeeds in the same MCP
controller process. Placement consumes the in-memory hold certificate and the
measured object/payload geometry, plans the collision-checked carry, verifies
release, retreats vertically, and homes.

Arguments:

- `relation="drop_zone"`: use the taught release point; no reference required.
- `relation="left|right|front|behind|next_to"`: provide a visible `reference`.
- `distance_m`: optional preferred separation, clamped to 0.06–0.18 m.

Directions use the robot/table frame: +X is front/away from the base and +Y is
robot-left. Scene-relative travel uses the target-excluded RGB-D occupancy map,
payload-aware clearance height, voxel X/Y routing, full-arm IK, URDF self/table
collision checks, release verification, and verified home.

Placement must refuse when there is no same-process pick certificate. Never
reconstruct held state from disk, jaw angle, or an operator assertion. Use
`wrist_stop` to abandon a held pick safely.

The original atomic `wrist_pick_and_place` tool remains available when no
decision is needed between the two phases.
