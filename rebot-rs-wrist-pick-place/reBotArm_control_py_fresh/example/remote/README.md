# Remote reBot Arm 102 → B601-RS teleoperation

This is the networked counterpart of Seeed's verified `lerobot-teleoperate`
path. The physical leader and follower use the published LeRobot plugins; this
directory only transports the leader action over a WebSocket relay and streams
an optional follower camera.

```text
leader_node.py  ── sequenced joint actions ──► relay_server.py ──► follower_node.py
 reBot Arm 102                                                        B601-RS
                                               └──── MJPEG ────► browser
```

The hardware nodes must use the clean `rebot_rs` environment built from the
[Seeed B601-RS LeRobot guide](https://wiki.seeedstudio.com/rebot_arm_b601_rs_lerobot/):

- Python 3.12
- LeRobot 0.4.4 from Seeed's repository
- `lerobot-teleoperator-rebot-arm-102==1.0.0`
- `lerobot-robot-seeed-b601==1.0.0`
- `motorbridge==0.5.0`
- `motorbridge-smart-servo==0.0.4`

The nodes enforce these minimums at startup.

## 1. Relay

Run on a reachable VM. Use TLS (`wss://`) through Caddy/nginx on an untrusted
network; the built-in relay does not provide authentication or encryption.

```bash
cd /path/to/reBotArm_control_py_fresh/example/remote
python -m pip install -r requirements-relay.txt
python relay_server.py --host 0.0.0.0 --port 8765
```

Only one leader and one follower may join a room. Browser viewers are unlimited.

## 2. Follower machine

Configure SocketCAN exactly as Seeed documents:

```bash
conda activate rebot_rs
sudo ip link set can0 down 2>/dev/null
sudo ip link set can0 type can bitrate 1000000 restart-ms 100
sudo ip link set can0 up
ip -details -statistics link show can0
```

Start the follower:

```bash
cd /home/manav/rebot_lerobot/reBotArm_control_py_fresh/example/remote
python follower_node.py \
  --relay YOUR_VM_IP:8765 \
  --room b601 \
  --arm rs \
  --port can0 \
  --can-adapter socketcan \
  --id follower1 \
  --camera 0
```

`--max-relative-target` defaults to `0` (disabled), matching Seeed's verified
local command. Do not add a small per-packet cap to fix network jitter: it makes
the target chase the feedback in visible steps. The remote node instead applies
only the newest sequenced sample at a maximum of 60 Hz and polls feedback at
10 Hz so it does not overload CAN.

## 3. Leader machine

```bash
conda activate rebot_rs
ls -l /dev/ttyUSB*

cd /home/manav/rebot_lerobot/reBotArm_control_py_fresh/example/remote
python leader_node.py \
  --relay YOUR_VM_IP:8765 \
  --room b601 \
  --port /dev/ttyUSB0 \
  --id rebot_arm_102_leader \
  --fps 60
```

The ID must match the calibration file. The verified calibration files are:

```text
~/.cache/huggingface/lerobot/calibration/robots/seeed_b601_rs_follower/follower1.json
~/.cache/huggingface/lerobot/calibration/teleoperators/rebot_arm_102_leader/rebot_arm_102_leader.json
```

Put both arms in corresponding poses before starting. Support the follower for
the first test and make small movements. `Ctrl+C` disconnects and disables the
follower motors.

## 4. Viewer

Open:

```text
http://YOUR_VM_IP:8765/view?room=b601
```

## Safety/control behavior

- Official follower mapping, joint limits, MIT gains, gripper controller, and
  thermal shutdown logic remain in the published B601 plugin.
- Packets carry a process session ID and increasing sequence number. Superseded
  and out-of-order packets are discarded rather than replayed.
- A `0.4 s` action watchdog stops issuing commands after a link stall. The RS
  controller holds the last command until shutdown/reconnection.
- Control is capped at 60 Hz; feedback defaults to 10 Hz; video runs separately.
- A room accepts only one leader and one follower.
- Relay connection is required before either node enters its normal run loop.

This transport is not a safety-rated system. Keep an operator beside the
follower with immediate access to power.

## No-hardware end-to-end test

Use three terminals in `example/remote`, all with `conda activate rebot_rs`:

```bash
python relay_server.py --host 127.0.0.1 --port 8765
```

```bash
python follower_node.py --relay 127.0.0.1:8765 --room demo --fake --camera test
```

```bash
python leader_node.py --relay 127.0.0.1:8765 --room demo --fake
```

Then open `http://127.0.0.1:8765/view?room=demo`.

## Gesture mode

The separate `operator_node.py` / `arm_node.py` gesture path remains available
in the repository `.venv`. It is not part of the physical leader/follower path
described above.
