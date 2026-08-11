---
name: rebot-rs-wrist-pick
description: Pick and visually verify one named object with the reBot RS wrist camera, then keep it held for a later placement decision.
version: 1.0.0
author: Manav Goel
metadata:
  hermes:
    tags: [robotics, rebot, realsense, pick, safety]
prerequisites:
  commands: [bash]
---

# reBot RS wrist pick

Use the `wrist_pick` MCP tool for the pick phase. It detects and segments the
requested object, builds metric geometry, prevalidates the complete approach,
opens and image-sizes the jaws, closes without contact inference, verifies the
hold visually, and lifts to the carry pose.

The successful result must report all of the following:

- `ok: true`
- `object_held: true`
- `holding: true`
- `next_skill: wrist_place`
- `requires_same_controller_process: true`

Do not restart the MCP/controller process after a successful pick. The verified
hold certificate and measured payload geometry intentionally exist only in that
process. A file or caller-provided `held=true` assertion is never sufficient.

Use `wrist_stop` when the user says stop, cancel, abort, or quit. It freezes the
measured pose, opens the gripper, and homes. Do not invoke another pick while an
object is held.

For supervised terminal testing from the robot workspace:

```bash
cd /home/manav/rebot_lerobot
./run_object_pick.sh --run "yellow sponge"
```

The terminal command intentionally stays alive while holding. Press `Q` or
Ctrl+C to release and home; terminal processes cannot hand a held object safely
to a separate placement process.
