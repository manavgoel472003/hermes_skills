# Retargeting and Replay

This document covers the trajectory retargeting workflow in this workspace:

- convert a source trajectory into end-effector poses
- retarget those poses to a different arm
- save a replayable target trajectory
- replay in MeshCat sim
- replay on the actual reBot / Seeed B601 arm

The current workflow is implemented around these scripts:

- `lerobot/src/lerobot/scripts/lerobot_retarget_trajectory.py`
- `tools/run_retarget_seeed.py`
- `lerobot/src/lerobot/scripts/lerobot_replay_retargeted_seeed.py`
- `reBotArm_control_py/example/sim/replay_retargeted.py`
- `reBotArm_control_py/example/sim/replay_urdf_npz.py`


## 1. Environment Setup

Assume the existing `lerobot` conda environment in this workspace.

Activate it:

```bash
conda activate lerobot
```

Core Python dependencies used by the retargeting and sim pipeline:

```bash
python -m pip install numpy draccus meshcat pin matplotlib imageio robot_descriptions pyppeteer
```

For hardware replay on the Seeed B601 arm, install the local robot package once:

```bash
cd /home/manav/rebot_lerobot/lerobot-robot-seeed-b601
python -m pip install -e .
```

The SO101 sim/benchmark scripts expect the SO101 URDF at:

```text
/tmp/SO-ARM100/Simulation/SO101/so101_new_calib.urdf
```

If that path is missing, clone the upstream SO-ARM100 repo there or update the script arguments accordingly.


## 2. Main Retargeting Scripts

### `lerobot_retarget_trajectory.py`

Primary retargeting CLI.

Capabilities:

- source as joint trajectory or end-effector pose trajectory
- source robot as built-in `ur5` or arbitrary URDF
- target as built-in reBot/Seeed B601 preset or arbitrary URDF
- relative or absolute pose mapping
- separate gripper openness mapping
- smoothed target joint output

File:

- `lerobot/src/lerobot/scripts/lerobot_retarget_trajectory.py`


### `run_retarget_seeed.py`

Workspace-local convenience wrapper around `lerobot_retarget_trajectory.py`.

Use this when you want to run retargeting from the workspace root without manually setting `PYTHONPATH`.

File:

- `tools/run_retarget_seeed.py`


### `lerobot_replay_retargeted_seeed.py`

Hardware replay for the reBot / Seeed B601 arm.

Supports:

- `/dev/ttyACM*` via auto-selected `damiao`
- `can0` via `socketcan`
- move-to-start interpolation
- speed scaling
- probe-only mode

File:

- `lerobot/src/lerobot/scripts/lerobot_replay_retargeted_seeed.py`


### MeshCat sim replay

- reBot retargeted trajectory:
  - `reBotArm_control_py/example/sim/replay_retargeted.py`
- arbitrary URDF + `.npz`:
  - `reBotArm_control_py/example/sim/replay_urdf_npz.py`


## 3. Retargeting Logic

The retargeter does **task-space retargeting**, not joint-to-joint remapping.

Pipeline:

1. Source joints -> source end-effector pose trajectory
2. Source EE poses -> desired target EE poses
3. Desired target EE poses -> target joint trajectory via IK
4. Smoothing -> replay timestamps -> save `.npz`

There are three pose mapping modes:

- `relative`
  - replay source motion relative to the source start pose onto the target start pose
- `absolute`
  - interpret source poses directly in the target frame
- `canonical`
  - express source poses in a declared source canonical frame, then replay those same canonical coordinates in a declared target canonical frame
  - this is useful for pre-existing datasets where the original room/world placement is unknown

For unknown pre-existing datasets, start with:

- `--mapping_mode=canonical`
- `--orientation_weight=0.0` when XYZ matters more than wrist orientation

For calibrated source-to-target setups, use:

- `--mapping_mode=absolute`
- `--orientation_weight=0.0` when XYZ matters more than wrist orientation

For dataset transfer without the original physical setup, use:

- `--mapping_mode=canonical`
- `--source_canonical_frame_xyz_rpy="x,y,z,roll,pitch,yaw"`
- `--target_canonical_frame_xyz_rpy="x,y,z,roll,pitch,yaw"`

The canonical transforms are poses of the canonical task frame expressed in each robot's base frame. A practical convention is the center-front footprint of the base, with `z` up and `x` forward.

Estimate that frame from a URDF base mesh:

```bash
python /home/manav/rebot_lerobot/tools/estimate_base_front_frame_from_urdf.py \
  --urdf /path/to/robot.urdf
```

Then retarget with the printed `canonical_frame_xyz_rpy` values:

```bash
python /home/manav/rebot_lerobot/tools/run_retarget_seeed.py \
  --input-path /path/to/source_episode.npz \
  --source-mode joint_trajectory \
  --source-robot urdf \
  --source-urdf-path /path/to/source.urdf \
  --source-target-link gripper_frame_link \
  --source-joint-names joint1,joint2,joint3,joint4,joint5 \
  --input-data-key arm_joint_positions_rad \
  --input-time-key timestamps_s \
  --mapping-mode canonical \
  --source-canonical-frame-xyz-rpy "SOURCE_PRINTED_XYZ_RPY" \
  --target-canonical-frame-xyz-rpy "TARGET_PRINTED_XYZ_RPY" \
  --orientation-weight 0.0
```


## 4. Basic Retarget Commands

### 4.1 UR5-style source trajectory -> reBot/Seeed B601

From the workspace root:

```bash
python /home/manav/rebot_lerobot/tools/run_retarget_seeed.py \
  --input-path "/home/manav/rebot_lerobot/Example data (1 trajectory for each task)/tiger.pkl"
```

This writes:

```text
/home/manav/rebot_lerobot/outputs/tiger_retargeted.npz
```


### 4.2 SO101 end-effector pose recording -> reBot/Seeed B601

If the source `.npz` already contains `ee_poses_base`:

```bash
python /home/manav/rebot_lerobot/tools/run_retarget_seeed.py \
  --input-path "/home/manav/rebot_lerobot/real_data_so101/traj_20260601_182814.npz" \
  --output-path "/home/manav/rebot_lerobot/outputs/traj_20260601_182814_rebot_absolute_xyz.npz" \
  --source-mode ee_pose_trajectory \
  --input-data-key ee_poses_base \
  --input-time-key timestamps_s \
  --mapping_mode=absolute \
  --orientation_weight=0.0 \
  --ik_orientation_tolerance_rad=10.0 \
  --ik_max_iterations=120
```


### 4.3 Arbitrary URDF source -> reBot/Seeed B601

```bash
python /home/manav/rebot_lerobot/tools/run_retarget_seeed.py \
  --input-path "/path/to/source_traj.npz" \
  --output-path "/path/to/output_rebot.npz" \
  --source-mode joint_trajectory \
  --source-robot urdf \
  --source-urdf-path "/path/to/source_arm.urdf" \
  --source-target-link "ee_link" \
  --source-joint-names "joint1,joint2,joint3,joint4,joint5,joint6"
```


## 5. Replay In MeshCat Sim

### 5.1 Replay a retargeted reBot trajectory

```bash
cd /home/manav/rebot_lerobot/reBotArm_control_py
python example/sim/replay_retargeted.py \
  --trajectory /home/manav/rebot_lerobot/outputs/tiger_retargeted.npz \
  --open-browser \
  --hold-seconds 300 \
  --speed-multiplier 5.0
```


### 5.2 Replay an arbitrary URDF trajectory

```bash
cd /home/manav/rebot_lerobot/reBotArm_control_py
python example/sim/replay_urdf_npz.py \
  --trajectory /home/manav/rebot_lerobot/outputs/random_pickplace_benchmark/source_so101/task_000.npz \
  --urdf /tmp/SO-ARM100/Simulation/SO101/so101_new_calib.urdf \
  --joint-data-key joint_positions_rad \
  --joint-names-key joint_names \
  --time-key timestamps_s \
  --ee-frame gripper_frame_link \
  --open-browser
```


### 5.3 Capture a sim GIF

reBot:

```bash
cd /home/manav/rebot_lerobot/reBotArm_control_py
python example/sim/replay_retargeted.py \
  --trajectory /home/manav/rebot_lerobot/outputs/random_pickplace_benchmark/target_b601/task_000.npz \
  --capture-gif /home/manav/rebot_lerobot/outputs/task_000_rebot.gif \
  --capture-every 4 \
  --speed-multiplier 6.0 \
  --hold-seconds 0
```

SO101 / arbitrary URDF:

```bash
cd /home/manav/rebot_lerobot/reBotArm_control_py
python example/sim/replay_urdf_npz.py \
  --trajectory /home/manav/rebot_lerobot/outputs/random_pickplace_benchmark/source_so101/task_000.npz \
  --urdf /tmp/SO-ARM100/Simulation/SO101/so101_new_calib.urdf \
  --joint-data-key joint_positions_rad \
  --joint-names-key joint_names \
  --time-key timestamps_s \
  --ee-frame gripper_frame_link \
  --capture-gif /home/manav/rebot_lerobot/outputs/task_000_so101.gif \
  --capture-every 4 \
  --speed-multiplier 6.0 \
  --hold-seconds 0
```


## 6. Replay On The Actual reBot / Seeed B601 Arm

Always probe first:

```bash
cd /home/manav/rebot_lerobot/lerobot
PYTHONPATH=src python -m lerobot.scripts.lerobot_replay_retargeted_seeed \
  --trajectory_path=/home/manav/rebot_lerobot/outputs/random_pickplace_benchmark/target_b601/task_035.npz \
  --port=/dev/ttyACM0 \
  --probe_only=true
```

Then replay task `035` slowly:

```bash
cd /home/manav/rebot_lerobot/lerobot
PYTHONPATH=src python -m lerobot.scripts.lerobot_replay_retargeted_seeed \
  --trajectory_path=/home/manav/rebot_lerobot/outputs/random_pickplace_benchmark/target_b601/task_035.npz \
  --port=/dev/ttyACM0 \
  --move_to_start_duration_s=8.0 \
  --speed_multiplier=0.5 \
  --max_relative_target=6.0 \
  --pos_vel_velocity "[45,45,45,35,35,35,35]"
```

Safer first pass:

```bash
--speed_multiplier=0.35 --max_relative_target=4.0
```


## 7. Benchmark Scripts

### 7.1 SO101 -> B601 and YAM random pick/place benchmark

Generates 40 random SO101 pick/place tasks and retargets them to B601 and YAM.

File:

- `tools/benchmark_random_pickplace_retarget.py`

Run:

```bash
python /home/manav/rebot_lerobot/tools/benchmark_random_pickplace_retarget.py --num-tasks 40
```

Outputs:

- `outputs/random_pickplace_benchmark/summary.json`
- `outputs/random_pickplace_benchmark/target_metrics.csv`
- `outputs/random_pickplace_benchmark/source_so101/task_*.npz`
- `outputs/random_pickplace_benchmark/target_b601/task_*.npz`
- `outputs/random_pickplace_benchmark/target_yam/task_*.npz`


### 7.2 reBot B601 -> Franka/Panda benchmark

Uses the existing 40 B601 benchmark trajectories as the source set and retargets them to Franka.

File:

- `tools/benchmark_rebot_to_franka.py`

Run:

```bash
python /home/manav/rebot_lerobot/tools/benchmark_rebot_to_franka.py
```

Outputs:

- `outputs/rebot_to_franka_benchmark/summary.json`
- `outputs/rebot_to_franka_benchmark/metrics.csv`
- `outputs/rebot_to_franka_benchmark/comparison_all.json`


## 8. Single-Task Sim GIFs

Render individual tasks for source / reBot / YAM:

File:

- `tools/render_single_task_sim_gifs.py`

Example:

```bash
python /home/manav/rebot_lerobot/tools/render_single_task_sim_gifs.py --task-ids 0,1,2,3,4
```

Outputs:

- `outputs/random_pickplace_benchmark/single_task_gifs/source_so101_task_000.gif`
- `outputs/random_pickplace_benchmark/single_task_gifs/rebot_b601_task_000.gif`
- `outputs/random_pickplace_benchmark/single_task_gifs/yam_task_000.gif`
- etc.


## 9. Output File Conventions

Typical retargeted `.npz` contents:

- `desired_target_poses`
- `raw_target_joint_positions_rad`
- `smoothed_target_joint_positions_rad`
- `achieved_target_poses`
- `position_errors_m`
- `replay_timestamps_s`
- `target_output_joint_names`

If gripper mapping is active:

- `source_gripper_open_fraction`
- `target_action_positions_deg`
- `target_action_names`


## 10. Common Issues

### MeshCat import error

Install:

```bash
python -m pip install meshcat
```


### `pinocchio.visualize` missing

This usually means the wrong `pinocchio` package is installed.

Fix:

```bash
python -m pip uninstall -y pinocchio
python -m pip install --force-reinstall pin
```


### YAM / Franka descriptions not found

Install:

```bash
python -m pip install robot_descriptions
```


### Hardware replay does not move

Probe first:

```bash
--probe_only=true
```

For `/dev/ttyACM0`, the replay script auto-selects the `damiao` adapter. If needed, set it explicitly:

```bash
--can_adapter=damiao
```


## 11. Recommended Starting Points

If you want the shortest working path:

1. Retarget:

```bash
python /home/manav/rebot_lerobot/tools/run_retarget_seeed.py \
  --input-path "/home/manav/rebot_lerobot/Example data (1 trajectory for each task)/tiger.pkl"
```

2. Replay in sim:

```bash
cd /home/manav/rebot_lerobot/reBotArm_control_py
python example/sim/replay_retargeted.py \
  --trajectory /home/manav/rebot_lerobot/outputs/tiger_retargeted.npz \
  --open-browser \
  --speed-multiplier 5.0
```

3. Replay on hardware:

```bash
cd /home/manav/rebot_lerobot/lerobot
PYTHONPATH=src python -m lerobot.scripts.lerobot_replay_retargeted_seeed \
  --trajectory_path=/home/manav/rebot_lerobot/outputs/tiger_retargeted.npz \
  --port=/dev/ttyACM0 \
  --probe_only=true
```
