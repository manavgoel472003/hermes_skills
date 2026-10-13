# Room Pulse

Room Pulse turns the reSpeaker Flex into a privacy-first, live conversation map. Each VAD-delimited speech turn becomes a node positioned by its measured direction. Local Whisper creates captions; local Qwen groups turns into concise topics and extracts action items. It never uses a camera or identifies people.

Run it with Ollama already running:

```bash
cd /home/manav/rebot_lerobot
.venv-respeaker/bin/python respeaker_room_pulse.py
```

For a strong demo, place the Flex in the middle of a group. Have people speak from different sides and introduce connected ideas:

1. Front: “We should make the demo accessible.”
2. Left: “I agree; add live captions for the audience.”
3. Rear: “I’ll prepare the caption layout by Friday.”

The first two turns should form a connected topic thread; the third appears in **Action Orbit**. Press `Esc` to exit.
