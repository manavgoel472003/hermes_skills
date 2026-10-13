# Caption Compass

Caption Compass provides large, directional live captions for group conversations. It uses the Flex's VAD and 360-degree direction of arrival to place each completed caption in the corresponding side of the display. Whisper runs locally; no camera, identity tracking, or cloud API is used.

Run:

```bash
cd /home/manav/rebot_lerobot
.venv-respeaker/bin/python respeaker_caption_compass.py
```

Put the Flex in the centre of a conversation. Speak from different sides, leaving a short pause after each utterance. Captions appear in a stable, color-coded directional card and also in the caption trail. Press `C` to clear the display; press `Esc` to exit.

The first run downloads the selected Whisper `base` model. Use `--whisper-model small` for higher accuracy at the cost of a larger initial model download.
