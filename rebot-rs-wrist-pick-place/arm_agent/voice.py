"""Voice front-end: speak a task, the arm does it.

Uses faster-whisper for speech-to-text and (optionally) a TTS engine to speak
results back. Push-to-talk by default (press Enter to start/stop a recording) so
it works without wake-word/VAD plumbing.

  python -m arm_agent.voice                       # voice REPL
  python -m arm_agent.voice --whisper-model small --dry-run
  python -m arm_agent.voice --file clip.wav        # transcribe a file and run once
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

from .service_client import ArmServiceClient
from .hermes_client import OllamaHermesClient
from .executor import AgentSession

SAMPLE_RATE = 16000


class Transcriber:
    def __init__(self, model_size: str | None = None, device: str | None = None,
                 compute_type: str | None = None) -> None:
        self.model_size = model_size or os.getenv("REBOT_WHISPER_MODEL", "base")
        self.device = device or os.getenv("REBOT_WHISPER_DEVICE", "cpu")
        self.compute_type = compute_type or ("int8" if self.device == "cpu" else "float16")
        self._model = None

    def _get_model(self):
        if self._model is None:
            from faster_whisper import WhisperModel
            self._model = WhisperModel(self.model_size, device=self.device, compute_type=self.compute_type)
        return self._model

    def transcribe_audio(self, audio: np.ndarray) -> str:
        audio = np.asarray(audio, dtype=np.float32).reshape(-1)
        segments, _info = self._get_model().transcribe(audio, language="en", vad_filter=True)
        return " ".join(seg.text.strip() for seg in segments).strip()

    def transcribe_file(self, path: str) -> str:
        segments, _info = self._get_model().transcribe(path, language="en", vad_filter=True)
        return " ".join(seg.text.strip() for seg in segments).strip()


def record_push_to_talk(samplerate: int = SAMPLE_RATE) -> np.ndarray:
    """Record from the default mic between two Enter presses."""
    import sounddevice as sd

    input("[press Enter to start speaking] ")
    frames: list[np.ndarray] = []
    stream = sd.InputStream(samplerate=samplerate, channels=1, dtype="float32")
    stream.start()
    print("[recording... press Enter to stop] ", end="", flush=True)

    import threading
    stop = threading.Event()

    def _wait_enter() -> None:
        input()
        stop.set()

    threading.Thread(target=_wait_enter, daemon=True).start()
    try:
        while not stop.is_set():
            data, _ = stream.read(samplerate // 10)
            frames.append(data.copy())
    finally:
        stream.stop()
        stream.close()
    if not frames:
        return np.zeros(0, dtype=np.float32)
    return np.concatenate(frames, axis=0).reshape(-1)


def _make_speaker():
    """Return a speak(text) callable; falls back to printing if no TTS engine."""
    try:
        import pyttsx3
        engine = pyttsx3.init()

        def speak(text: str) -> None:
            engine.say(text)
            engine.runAndWait()

        return speak
    except Exception:
        def speak(text: str) -> None:
            print(f"[arm] {text}")

        return speak


def _run_once(session: AgentSession, task: str, speak) -> None:
    """One-shot (file mode): run a task; if it asks, surface the question and stop."""
    print(f"[heard] {task}")
    result = session.run(task)
    summary = result.get("summary") or result.get("question") or result["status"]
    speak(summary)
    print(f"[{result['status']}] {summary}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Voice front-end for the reBot arm agent.")
    ap.add_argument("--url", default=None)
    ap.add_argument("--model", default=None, help="Ollama planner model (default hermes3).")
    ap.add_argument("--whisper-model", default=None, help="faster-whisper size (tiny/base/small/...).")
    ap.add_argument("--whisper-device", default=None, help="faster-whisper device (cpu/cuda/auto).")
    ap.add_argument("--whisper-compute-type", default=None, help="faster-whisper compute type (int8/float16/...).")
    ap.add_argument("--file", default=None, help="Transcribe a WAV/MP3 file and run once.")
    ap.add_argument("--no-tts", action="store_true", help="Print responses instead of using pyttsx3.")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    client = ArmServiceClient(base_url=args.url)
    planner = OllamaHermesClient(model=args.model)
    session = AgentSession(client=client, planner=planner, dry_run=args.dry_run)
    transcriber = Transcriber(
        model_size=args.whisper_model,
        device=args.whisper_device,
        compute_type=args.whisper_compute_type,
    )
    speak = (lambda text: print(f"[arm] {text}")) if args.no_tts else _make_speaker()

    try:
        if args.file:
            task = transcriber.transcribe_file(args.file)
            _run_once(session, task, speak)
            return 0

        print("Voice agent ready. Ctrl-C to quit.")
        while True:
            audio = record_push_to_talk()
            if audio.size == 0:
                continue
            task = transcriber.transcribe_audio(audio)
            if not task:
                print("[heard nothing]")
                continue
            if task.lower().strip(" .") in {"quit", "exit", "stop"}:
                break
            _voice_task(session, task, transcriber, speak)
    except KeyboardInterrupt:
        pass
    finally:
        client.close()
    return 0


def _voice_task(session: AgentSession, task: str, transcriber: Transcriber, speak) -> None:
    print(f"[heard] {task}")
    result = session.run(task)
    while result.get("status") == "ask":
        speak(result["question"])
        print(f"[arm asks] {result['question']}")
        audio = record_push_to_talk()
        answer = transcriber.transcribe_audio(audio) if audio.size else "stop"
        result = session.provide_answer(answer)
    summary = result.get("summary") or result["status"]
    speak(summary)
    print(f"[{result['status']}] {summary}")
    if result.get("executed"):
        for s in result["executed"]:
            print(f"  - {s['skill']} ok={s['ok']} {s.get('reason','')}")


if __name__ == "__main__":
    sys.exit(main())
