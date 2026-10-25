"""
main.py - AyosTa application controller and entry point.

The controller coordinates the view (gui.py) and feature modules. Importing this
file does not open the camera or microphone. Speech model initialization is warmed
in a background thread while the GUI opens; devices are only activated when the
user starts a session.

Run with: py main.py
"""

from __future__ import annotations

import queue
import threading
from pathlib import Path
from typing import Any, Callable

import numpy as np

DIRECTIONS = {
    "Cebuano to Tagalog": ("Cebuano", "Tagalog"),
    "Tagalog to Cebuano": ("Tagalog", "Cebuano"),
}
SPEAKING = {"Cebuano": "ceb", "Tagalog": "tgl"}


class Controller:
    """Coordinate workers, background tasks, exports, and GUI-facing state."""

    DIRECTIONS = DIRECTIONS
    SPEAKING = SPEAKING

    def __init__(self) -> None:
        self.events: queue.Queue[tuple[str, Any]] = queue.Queue()
        self.face_worker = None
        self.recorder = None
        self.speech_worker = None
        self.session_active = False
        self.translating = False
        self.speaking = "Cebuano"
        self._words: list[str] = []
        self._closed = False
        self._devices_ready = False
        self._translation_lock = threading.Lock()
        self._preload_lock = threading.Lock()
        self._speech_engine = None  # Keep the loaded engine for the app's full lifetime.
        self._preload_error: str | None = None

        self.events.put(("status", "Opening AyosTa. Loading models and starting live previews..."))
        self._preload_thread = threading.Thread(
            target=self._preload_speech_engine, name="speech-model-preload", daemon=True
        )
        self._preload_thread.start()
        self._device_thread = threading.Thread(
            target=self._start_preview_devices, name="preview-devices-startup", daemon=True
        )
        self._device_thread.start()

    def _preload_speech_engine(self) -> None:
        """Load the actual speech model before a session starts."""
        try:
            from speech import MMSEngine

            engine = MMSEngine(language=self.SPEAKING[self.speaking])

            # This loads the real model weights and processor.
            engine.load()

            with self._preload_lock:
                if not self._closed:
                    self._speech_engine = engine

            self.events.put((
                "status",
                "Speech model loaded and ready. Preparing live preview..."
            ))

        except Exception as error:
            self._preload_error = f"{type(error).__name__}: {error}"
            self.events.put(("error", ("Speech", self._preload_error)))
            self.events.put((
                "status",
                "Speech model failed to load. Check the error message."
            ))
            
    def load(self) -> None:
        if self._model is not None:
            return  # Already loaded; reuse the model.


    def _start_preview_devices(self) -> None:
        """Start camera and microphone once. They remain open between sessions."""
        problems: list[str] = []
        try:
            from face import FaceWorker
            worker = FaceWorker()
            worker.start()
            if self._closed:
                worker.stop()
            else:
                self.face_worker = worker
        except Exception as error:
            problems.append(f"Camera: {type(error).__name__}: {error}")

        try:
            from audio import AudioRecorder
            recorder = AudioRecorder()
            recorder.start()
            if self._closed:
                recorder.stop()
            else:
                self.recorder = recorder
        except Exception as error:
            problems.append(f"Microphone: {type(error).__name__}: {error}")

        if problems:
            self.events.put(("problems", problems))
        self._devices_ready = True
        self.events.put(("status", self._preview_status()))

    @property
    def camera_on(self) -> bool:
        # Camera preview is independent of whether a session is recording.
        return self.face_worker is not None

    @property
    def can_start_session(self) -> bool:
        worker_busy = self.speech_worker is not None and self.speech_worker.is_alive()
        return (not self._closed and self._devices_ready and not self.session_active
                and not self.translating and not worker_busy)

    @property
    def can_translate(self) -> bool:
        return (
            not self._closed and not self.session_active and not self.translating
            and bool(self._words) and self.speech_worker is None
        )

    @property
    def has_audio(self) -> bool:
        recorder = self.recorder
        return bool(recorder is not None and recorder.buffer.length > 0)

    def start_session(self) -> None:
        """Enable session retention/transcription without restarting camera or mic."""
        if not self.can_start_session:
            return
        if self.recorder is None and self.face_worker is None:
            self.events.put(("status", "Preview devices are unavailable. Check Settings."))
            return

        self._words.clear()
        self.events.put(("transcript_cleared", None))
        self.events.put(("session", None))
        if self.recorder is not None:
            try:
                self.recorder.start_recording()
            except Exception as error:
                self.events.put(("error", ("Audio", str(error))))

        self.session_active = True
        self.events.put(("status", self._running_status()))
        if self.recorder is not None:
            threading.Thread(
                target=self._start_speech_background, name="speech-worker-startup", daemon=True
            ).start()
        elif self.face_worker is not None:
            self.events.put(("problems", ["Microphone unavailable; session has video only."]))

    def _start_speech_background(self) -> None:
        """Start a per-session worker using the already-loaded speech engine."""
        try:
            from speech import SpeechWorker, MMSEngine
            self._preload_thread.join()
            if self._closed or not self.session_active or self.recorder is None:
                return

            # Never overlap speech workers: a stopped worker may still be flushing its final words.
            previous = self.speech_worker
            if previous is not None and previous.is_alive():
                previous.join()
            if self._closed or not self.session_active or self.recorder is None:
                return

            with self._preload_lock:
                engine = self._speech_engine
            if engine is None:
                engine = MMSEngine(language=self.SPEAKING[self.speaking])
                with self._preload_lock:
                    self._speech_engine = engine
            elif hasattr(engine, "set_language"):
                engine.set_language(self.SPEAKING[self.speaking])

            worker = SpeechWorker(self.recorder.chunks, engine=engine)
            worker.start()
            if self._closed or not self.session_active:
                try:
                    worker.request_stop()
                except Exception:
                    pass
                return
            self.speech_worker = worker
            self.events.put(("status", self._running_status()))
        except Exception as error:
            self.events.put(("error", ("Speech", f"{type(error).__name__}: {error}")))
            self.events.put(("status", "Session recording; speech recognition failed to start."))

    def stop_session(self) -> None:
        """End the current recording only; preview devices and model stay alive."""
        if not self.session_active:
            return
        self.session_active = False
        errors: list[str] = []
        if self.recorder is not None:
            try:
                self.recorder.stop_recording()
            except Exception as error:
                errors.append(f"Audio: {type(error).__name__}: {error}")
        worker = self.speech_worker
        if worker is not None:
            try:
                worker.request_stop()
            except Exception as error:
                errors.append(f"Speech: {type(error).__name__}: {error}")
        if errors:
            self.events.put(("problems", errors))
        self.events.put(("status", "Session stopped. Finishing transcript..." if worker else
                         f"Session stopped. {len(self._words)} words captured."))
        self.events.put(("session", None))

    def _try_start(self, label: str, factory: Callable[[], Any], problems: list[str]):
        try:
            return factory()
        except Exception as error:
            problems.append(f"{label}: {type(error).__name__}: {error}")
            return None

    def set_speaking(self, language: str) -> None:
        if language not in self.SPEAKING:
            raise ValueError(f"Unsupported speaking language: {language}")
        self.speaking = language
        worker = self.speech_worker
        if worker is not None and self.session_active:
            try:
                worker.set_language(self.SPEAKING[language])
            except Exception as error:
                self.events.put(("error", ("Speech", f"Could not change language: {error}")))

    def lock_face(self) -> None:
        worker = self.face_worker
        if worker is not None:
            try:
                worker.lock_largest()
            except Exception as error:
                self.events.put(("error", ("Video", str(error))))

    def unlock_face(self) -> None:
        worker = self.face_worker
        if worker is not None:
            try:
                worker.unlock()
            except Exception as error:
                self.events.put(("error", ("Video", str(error))))

    def poll_frame(self):
        worker = self.face_worker
        if worker is None:
            return None
        try:
            return worker.frames.get_nowait()
        except queue.Empty:
            return None
        except Exception as error:
            self.events.put(("error", ("Video", f"Could not read camera frame: {error}")))
            return None

    def update(self) -> list[tuple[str, Any]]:
        """Drain worker results; called by the GUI thread."""
        updates: list[tuple[str, Any]] = []
        while True:
            try:
                updates.append(self.events.get_nowait())
            except queue.Empty:
                break

        worker = self.speech_worker
        if worker is not None:
            try:
                while True:
                    word = worker.words.get_nowait()
                    if word:
                        self._words.append(str(word))
                        updates.append(("words", [str(word)]))
            except queue.Empty:
                pass
            except Exception as error:
                updates.append(("error", ("Speech", f"Could not read transcript: {error}")))

            if not self.session_active and not worker.is_alive():
                self.speech_worker = None
                updates.append(("status", f"Session stopped. {len(self._words)} words captured. Preview remains active."))

        return updates

    def audio_state(self) -> tuple[int, int, bool]:
        """Return sample count, rate, and session-recording state for the waveform."""
        recorder = self.recorder
        if recorder is None:
            return 0, 16000, False
        try:
            active = bool(self.session_active and getattr(recorder, "is_recording", False))
            if active or recorder.buffer.length > 0:
                return int(recorder.buffer.length), int(recorder.sample_rate), active
            return int(recorder.preview_buffer.length), int(recorder.sample_rate), False
        except Exception:
            return 0, 16000, False

    def audio_peaks(self, start: int, end: int, columns: int):
        recorder = self.recorder
        if recorder is None or columns <= 0 or end <= start:
            return np.zeros(0), np.zeros(0)
        if self.session_active or recorder.buffer.length > 0:
            return recorder.buffer.peaks(start, end, columns)
        return recorder.preview_buffer.peaks(start, end, columns)

    def translate(self, direction: str) -> bool:
        if not self.can_translate:
            return False
        text = " ".join(self._words).strip()
        if not text:
            return False
        if direction not in self.DIRECTIONS:
            self.events.put(("translation_error", f"Unsupported translation direction: {direction}"))
            return False

        source, target = self.DIRECTIONS[direction]
        self.translating = True
        self.events.put(("status", "Translating. The first run may take a while..."))
        threading.Thread(
            target=self._translation_job,
            args=(text, source, target),
            name="translation-worker",
            daemon=True,
        ).start()
        return True

    def _translation_job(self, text: str, source: str, target: str) -> None:
        try:
            from language import translate_long

            result = translate_long(
                text,
                source,
                target,
                on_progress=lambda done, total: self.events.put(
                    ("translation_progress", (done, total))
                ),
            )
            self.events.put(("translation", (result, source, target)))
        except Exception as error:
            self.events.put(("translation_error", f"{type(error).__name__}: {error}"))
        finally:
            self.translating = False

    def save_audio(self, path: str | Path) -> None:
        recorder = self.recorder
        if recorder is None:
            raise ValueError("There is no recording to save.")
        samples = recorder.buffer.snapshot(0, recorder.buffer.length)
        if samples is None or len(samples) == 0:
            raise ValueError("There is no recorded session audio to save yet.")
        sample_rate = int(recorder.sample_rate)
        from exports import save_wav

        save_wav(path, samples, sample_rate)

    def export_pdf(self, path: str | Path, transcript_text: str) -> None:
        from exports import export_transcript_pdf

        export_transcript_pdf(path, transcript_text)

    def run_checks(self, deep: bool = False, progress=None):
        import syscheck

        return syscheck.run_all_checks(
            project_dir=Path(__file__).resolve().parent,
            deep=deep,
            session_active=self.session_active,
            progress=progress,
        )

    @staticmethod
    def summarize_checks(results):
        import syscheck

        return syscheck.summarize(results)

    @staticmethod
    def install_commands(results):
        import syscheck

        return syscheck.install_commands(results)

    def _preview_status(self) -> str:
        video = "camera preview on" if self.face_worker is not None else "camera unavailable"
        audio = "mic preview on" if self.recorder is not None else "mic unavailable"
        with self._preload_lock:
            model_ready = self._speech_engine is not None
        model = "speech model ready" if model_ready else "speech model loading"
        return f"Preview ready ({video}, {audio}; {model}). Click Start Session to record."

    def _running_status(self) -> str:
        video = "video on" if self.face_worker is not None else "video off"
        audio = "recording" if self.recorder is not None else "audio off"
        speech_state = getattr(self.speech_worker, "state", "starting")
        return f"Session recording ({video}, {audio}, speech: {speech_state})"

    def shutdown(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self.session_active:
            self.stop_session()
        # Only application shutdown closes devices and releases the retained model.
        for worker, method_name in (
            (self.face_worker, "stop"),
            (self.recorder, "stop"),
            (self.speech_worker, "request_stop"),
        ):
            if worker is not None:
                try:
                    getattr(worker, method_name)()
                except Exception:
                    pass
        self.face_worker = None
        self.recorder = None
        self.speech_worker = None
        self._speech_engine = None


def main() -> None:
    from gui import App

    controller = Controller()
    app = App(controller)
    app.mainloop()
    controller.shutdown()


if __name__ == "__main__":
    main()
