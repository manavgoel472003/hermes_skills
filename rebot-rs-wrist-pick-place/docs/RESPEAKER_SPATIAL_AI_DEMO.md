# Orbital: Spatial AI Host

This demo combines the Flex's hardware VAD and direction-of-arrival with local Whisper transcription and a local Qwen model. It listens continuously, automatically ends an utterance after 0.75 seconds of silence, preserves the median speaking direction, and renders a spatial conversation on screen. It does not move the robot.

## Install and run

```bash
cd /home/manav/rebot_lerobot
.venv-respeaker/bin/python -m pip install numpy sounddevice faster-whisper
ollama serve
```

In a second terminal, download the selected local model once:

```bash
ollama pull qwen2.5:3b
```

Then launch:

```bash
cd /home/manav/rebot_lerobot
.venv-respeaker/bin/python respeaker_spatial_ai.py
```

The first spoken interaction also downloads the selected Whisper `base` model. Use `--whisper-model small` for more accurate transcription if you have the RAM and are comfortable with its larger first download.

## Presentation flow

1. Put the circular array at the center of a small group.
2. Speak from a different side each time: “Hello Orbital, what can you hear?”
3. Show the arrow moving to each speaker, the transcribed utterance, and the short local-AI reply.
4. Ask a follow-up from another location. The LLM receives direction metadata and keeps the recent conversation locally.

Press `Esc` to stop. The demo makes no robot-motion or audio-processing configuration changes.
