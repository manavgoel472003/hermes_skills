# CAN setup

The reBot RS arm uses `can0` at **1,000,000 bit/s**. Run the following after
booting the station, reconnecting the CAN adapter, or recovering from a CAN
communication failure.

```bash
cd /home/manav/rebot_lerobot

sudo ip link set can0 down
sudo ip link set can0 txqueuelen 1000
sudo ip link set can0 type can bitrate 1000000
sudo ip link set can0 up

ip -details link show can0 | grep -E "qlen|bitrate|can state"
```

The verification output should report all of the following:

- `UP` and `LOWER_UP`
- `can state ERROR-ACTIVE`
- `bitrate 1000000`
- `qlen 1000`

Then verify motor feedback with the read-only state command. This does not move
the arm or gripper.

```bash
/home/manav/miniconda3/envs/rebot_rs/bin/python \
  -m arm_agent.wrist_servo --state
```

Do not continue to a motion command unless the state result has `"ok": true`
and contains feedback for all six joints and the gripper. If `can0` is up but
motor feedback is unavailable, check for an older arm process before retrying:

```bash
pgrep -af 'arm_agent.wrist_servo|run_.*pick'
```

Do not use the earlier 500,000 bit/s setting; this arm has been verified at
1,000,000 bit/s.
