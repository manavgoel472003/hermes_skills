For the pick and place to be right, the following conditions must be met [This is for the same version of the code]:
1. The sponge and the marker should be picked by the same version of code.
2. Placements on top and left and right should happen with 80 % accuracy at least.
3. At least 4 out of 6 marker colors are detectable and work for now.


Once, this completes and passes the above conditions we can de-couple it into parts to enable more complex workflows.

Complex workflows should look like:
- I want to put 2 sugar cubes in a cup : This would require something like scene understanding, then pick one and place it over a cup, then repeat. All of these sub-steps are basically singular skills being joined into a chain or loop to enable an agent to make everything work.


sudo ip link set can0 down
sudo ip link set can0 type can bitrate 1000000        # 1 Mbit — not 500k
sudo ip link set can0 txqueuelen 1000                 # prevents ENOBUFS on disable
sudo ip link set can0 up
ip -details link show can0                            # verify: UP, 1000000, qlen 1000

sudo ip link set can0 down && sudo ip link set can0 up
sudo ip link set can0 txqueuelen 1000                 # re-apply, it resets on down

cd /home/manav/rebot_lerobot/reBotArm_control_py_fresh
python example/disable_all.py rebotarm_rs.yaml


## Working pick & place recipe

Verified 2026-08-23 11:49: `green marker` stack `white cup`, first attempt,
ok=True verified=True. Also verified `blue marker` right `green mug`.

```bash
cd /home/manav/rebot_lerobot
REBOT_GRIPPER_SIZE_COMPRESSION_MM=6 \
REBOT_WS_X_MAX=0.60 \
REBOT_HYBRID_TARGET_CLUSTER_RADIUS_M=0.11 \
REBOT_OBJECT_GEOMETRY_MODE=hybrid \
REBOT_SCENE_COLLISION_EXPORT_VIEW=1 \
REBOT_PAYLOAD_COLLISION=0 \
./run_scene_pick_place.sh --run \
  "<object>" <relation> "<reference>" 2>&1 | tee /tmp/pick_run.log
```

Relations (exact CLI values): `drop_zone` `free_space` `left` `right` `front`
`behind` `next_to` `stack`. `stack` drops at the reference *centre*, so it is
the only relation with no error budget; the side-relative ones add lateral
clearance and tolerate a looser reference.

Why each variable:
- `GRIPPER_SIZE_COMPRESSION_MM=6` — jaws close 6mm inside the measured width so
  they bite. Default is 0 because the kinematic sim treats a command >2mm inside
  the object as a miss, so a raised default reports every sim grasp unheld.
  A round marker needs this: it has no flat to seat against and rolls out of a
  grip that only touches it.
- `WS_X_MAX=0.60` — the workspace AABB is only a coarse pre-filter (see
  arm_safe.py); every waypoint is still IK-verified and every descent
  torque-gated. The stock 0.44 refused reachable targets before IK ever ran.
- `HYBRID_TARGET_CLUSTER_RADIUS_M=0.11` — identity radius. Raise only as far as
  needed: it is what keeps a same-coloured decoy out.
- `OBJECT_GEOMETRY_MODE=hybrid` — multi-view metric object box.

## Scene setup matters more than any flag

Most failures were the scene, not the code:
- **No two objects of the same colour.** A blue marker + blue foam block, or a
  green marker + green mug, produced a decoy ~200-240mm away that the identity
  gate rejected all sweep poses over. Colour cannot separate them.
- **Name a colour that exists.** SAM 3 ignores the colour word and returns
  whichever object is most salient — it offered the same green marker for
  "blue marker" and "red marker". The colour veto now refuses those, but it
  cannot conjure an object that is not there.
- **Keep the target off the frame's bottom edge.** Clipped envelopes are
  refused, and the target slides out of view as the wrist descends.
- **White/black are nameable now but not always discriminative.** A white cup
  on a pale table with a white bag in frame merged into a 248-315mm component
  and was refused. Prefer a strongly coloured reference.
- **Reachability.** Anchors past ~0.48 lose orientation freedom; sweep poses
  start failing IK and fusion starves. Aim for x ~ 0.32-0.40.

## Reading a run

```bash
grep -o "stage=discovery_cloud_export.*points=[0-9]*" logs/latest_wrist_run.log
```
Healthy: `views=5 points=32052` and `hybrid_object_fusion ok=True
extent_converged=True`. A single view with ~7000 points means the sweep was
rejected and the .rrd is not worth opening.

The .rrd is only rewritten when a run logs `discovery_cloud_export ok=True`;
otherwise the file on disk is from a previous run. Always check its timestamp.

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate rebot_rs
rerun "$(ls -t /tmp/claude-1000/*/*/scratchpad/wrist_servo/scene_collision/latest_discovery.rrd | head -1)"
```

## Escape hatches

| Variable | Effect |
|---|---|
| `REBOT_COLOR_VETO=0` | stop a single-colour prompt vetoing a semantic match |
| `REBOT_STACK_ALLOW_RGBD_REFERENCE=0` | stack requires a semantic mask again |
| `REBOT_GRIPPER_MAX_COMPRESSION_MM` | ceiling on jaw compression (default 8) |
| `REBOT_OBJECT_GEOMETRY_MODE=legacy` | bypass hybrid multi-view fusion |

`Ctrl+C` / `Q` stops at any stage and runs the clean shutdown: freeze, open
gripper, home. A `kill -9` skips it and leaves the motors energised.

