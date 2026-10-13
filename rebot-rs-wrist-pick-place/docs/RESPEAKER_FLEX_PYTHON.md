# reSpeaker Flex: Python USB control

The device is controlled through its USB vendor-control interface. This reports direction of arrival (DOA) and voice activity detection (VAD); it does not require the Flex XVF_Host application.

## One-time host setup

Create an isolated environment and install PyUSB:

```bash
cd /home/manav/rebot_lerobot
python3 -m venv .venv-respeaker
.venv-respeaker/bin/python -m pip install --upgrade pip pyusb
```

Connect the reSpeaker directly to the computer (or a powered USB hub), then confirm that it appears:

```bash
lsusb | grep -i 2886
```

The expected USB vendor ID is `2886`. If this prints nothing, try another data-capable cable/port and confirm the Flex has power.

For non-root access on Linux, add a udev rule once:

```bash
echo 'SUBSYSTEM=="usb", ATTR{idVendor}=="2886", MODE="0666", GROUP="plugdev"' | sudo tee /etc/udev/rules.d/99-respeaker-flex.rules
sudo udevadm control --reload-rules
sudo udevadm trigger
```

Unplug and reconnect the device after adding the rule. Log out and in if your user is not already in the `plugdev` group.

## Run

```bash
.venv-respeaker/bin/python respeaker_flex.py
```

It prints the firmware version followed by DOA in degrees and whether speech is currently detected. Use `--interval 0.1` for 10 reads/sec. If multiple Seeed USB devices are attached, select the Flex product ID from `lsusb`, for example:

```bash
.venv-respeaker/bin/python respeaker_flex.py --pid 0x1234
```

## Troubleshooting

- `No reSpeaker found`: the host cannot see a USB device with vendor ID `2886`.
- `Access denied` / `USBError`: add the udev rule above, reconnect the device, and retry.
- Windows needs the additional package `libusb-package`; this repository’s script is aimed at Linux.
