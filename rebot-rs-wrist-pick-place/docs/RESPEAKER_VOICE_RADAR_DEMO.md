# reSpeaker Flex Voice Radar demo

Launch the interactive showcase with the Flex connected through its XMOS USB-C port:

```bash
cd /home/manav/rebot_lerobot
.venv-respeaker/bin/python respeaker_voice_radar_demo.py
```

The dashboard shows live direction of arrival, voice activity detection, a directional speech trail, AEC convergence, automatic-gain-control gain, and energy in each processing beam. The buttons change only the Flex LED effect: DOA, rainbow, breathe, or off. They do not alter the audio processing configuration.

Demo sequence:

1. Choose **DOA** to make the physical microphone LEDs show the detected direction.
2. Stand at several positions around the circular array and speak normally; the screen arrow and LED direction should track you.
3. Stop speaking to demonstrate VAD and the speech-history trail.
4. Play sound from a speaker near the Flex, then speak over it to demonstrate the beam-energy and AEC/AGC telemetry.

Press `Esc` or close the window to exit. The selected hardware LED effect remains active after exit.
