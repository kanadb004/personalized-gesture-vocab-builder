"""Speech output on a worker thread, so `say()` never blocks the caller, plus a bounded
message board for the GUI (O5)."""

from __future__ import annotations

import queue
import subprocess
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Callable


@dataclass
class SpokenEvent:
    text: str
    ts_ms: float


class Speaker:
    """Runs a worker thread that pulls text off a queue and speaks it, one item at a time.
    `say()` returns immediately. Backend is `pyttsx3` or the macOS `say` command line tool
    (`output.tts_backend`). `mute` drops queued items without speaking them but still fires the
    callbacks so the UI stays in sync."""

    def __init__(
        self,
        backend: str = "pyttsx3",
        rate: int = 175,
        voice: str = "default",
        on_started: Callable[[SpokenEvent], None] | None = None,
        on_finished: Callable[[SpokenEvent], None] | None = None,
    ) -> None:
        self.backend = backend
        self.rate = rate
        self.voice = voice
        self.mute = False
        self.on_started = on_started
        self.on_finished = on_finished
        self._queue: queue.Queue[str | None] = queue.Queue()
        self._engine = None
        if backend == "pyttsx3":
            import pyttsx3

            self._engine = pyttsx3.init()
            self._engine.setProperty("rate", rate)
            if voice != "default":
                self._engine.setProperty("voice", voice)
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def say(self, text: str) -> None:
        self._queue.put(text)

    def stop(self) -> None:
        self._queue.put(None)
        self._thread.join(timeout=2.0)

    def _run(self) -> None:
        while True:
            text = self._queue.get()
            if text is None:
                return
            if self.mute:
                continue
            started = SpokenEvent(text=text, ts_ms=time.time() * 1000.0)
            if self.on_started:
                self.on_started(started)
            self._speak(text)
            finished = SpokenEvent(text=text, ts_ms=time.time() * 1000.0)
            if self.on_finished:
                self.on_finished(finished)

    def _speak(self, text: str) -> None:
        if self.backend == "say":
            subprocess.run(["say", text], check=False)
        else:
            self._engine.say(text)
            self._engine.runAndWait()


class MessageBoard:
    """Bounded list of the most recently spoken messages, newest last."""

    def __init__(self, board_size: int = 5) -> None:
        self._messages: deque[str] = deque(maxlen=board_size)

    def add(self, text: str) -> None:
        self._messages.append(text)

    def latest(self, n: int | None = None) -> list[str]:
        messages = list(self._messages)
        if n is None:
            return messages
        return messages[-n:]

    def clear(self) -> None:
        self._messages.clear()
