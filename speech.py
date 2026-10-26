"""Live speech recognition for AyosTa using Meta's MMS language model.

The model is loaded lazily by MMSEngine.load(). The method is idempotent, so the
startup preloader and SpeechWorker can safely call it without loading weights twice.
SpeechWorker consumes int16 audio chunks from AudioRecorder.chunks and publishes
recognized words through its thread-safe `words` queue.
"""
from __future__ import annotations

import queue
import threading
import time
from typing import Optional

import numpy as np

MMS_MODEL_ID = "facebook/mms-1b-all"
SAMPLE_RATE = 16000
MMS_DTYPE = "bfloat16"
DEFAULT_LANGUAGE = "tgl"
SILENCE_RMS = 300.0
PAUSE_SECONDS = 0.7
MAX_SEGMENT_SECONDS = 15.0
MIN_SEGMENT_SECONDS = 0.4


class SpeechEngine:
    """Minimal base interface for speech engines."""

    def load(self) -> None:
        return

    def set_language(self, language: str) -> None:
        return

    def accept(self, samples: np.ndarray) -> list[str]:
        return []

    def flush(self) -> list[str]:
        return []


class MMSEngine(SpeechEngine):
    """Segment-based MMS speech recognizer for Cebuano (`ceb`) and Tagalog (`tgl`)."""

    def __init__(self, language: str = DEFAULT_LANGUAGE) -> None:
        self._language = language
        self._loaded_language: Optional[str] = None
        self._processor = None
        self._model = None
        self._torch = None
        self._dtype = None
        self._load_lock = threading.RLock()
        self._segment_chunks: list[np.ndarray] = []
        self._segment_samples = 0
        self._silence_samples = 0
        self._speech_seen = False
        self._pending_language: Optional[str] = None
       

    @property
    def language(self) -> str:
        return self._language

    def load(self) -> None:
        """Load the selected language adapter/model once; safe to call repeatedly."""
        with self._load_lock:
            if (self._model is not None and self._processor is not None
                    and self._loaded_language == self._language):
                return

            import torch
            from transformers import AutoProcessor, Wav2Vec2ForCTC

            self._torch = torch
            self._dtype = getattr(torch, MMS_DTYPE, torch.float32)
            processor = None
            model = None
            try:
                processor = AutoProcessor.from_pretrained(
                    MMS_MODEL_ID,
                    target_lang=self._language,
                )
                model = Wav2Vec2ForCTC.from_pretrained(
                    MMS_MODEL_ID,
                    target_lang=self._language,
                    ignore_mismatched_sizes=True,
                    dtype=self._dtype,
                )
                model.eval()
            except Exception:
                self._processor = None
                self._model = None
                self._loaded_language = None
                raise

            self._processor = processor
            self._model = model
            self._loaded_language = self._language

    def set_language(self, language: str) -> None:
        """Change the language adapter at the next segment boundary."""
        if not isinstance(language, str) or not language.strip():
            raise ValueError("Language code must be a non-empty string.")
        language = language.strip()
        if language == self._language and self._pending_language is None:
            return
        # Defer model loading to the speech worker thread; do not block the GUI.
        self._pending_language = language

    def _apply_pending_language(self) -> None:
        if self._pending_language is None:
            return
        new_language = self._pending_language
        self._pending_language = None
        if new_language == self._language and self._loaded_language == new_language:
            return
        self._language = new_language
        # Clear old adapter references before loading the newly selected language.
        self._processor = None
        self._model = None
        self._loaded_language = None
        self.load()

    def accept(self, samples: np.ndarray) -> list[str]:
        """Accept one audio chunk and decode when a pause or maximum length is reached."""
        chunk = np.asarray(samples, dtype=np.int16).reshape(-1)
        if chunk.size == 0:
            return []

        # Apply language changes only from the worker's audio-processing thread.
        if self._pending_language is not None and not self._segment_chunks:
            self._apply_pending_language()

        rms = float(np.sqrt(np.mean(chunk.astype(np.float32) ** 2)))
        if rms >= SILENCE_RMS:
            self._speech_seen = True
            self._silence_samples = 0
            self._segment_chunks.append(chunk.copy())
            self._segment_samples += int(chunk.size)
        elif self._speech_seen:
            # Keep trailing silence so pause detection has a clear endpoint.
            self._segment_chunks.append(chunk.copy())
            self._segment_samples += int(chunk.size)
            self._silence_samples += int(chunk.size)
        else:
            # Ignore silence before speech to avoid repeatedly decoding empty audio.
            return []

        pause_limit = int(PAUSE_SECONDS * SAMPLE_RATE)
        max_limit = int(MAX_SEGMENT_SECONDS * SAMPLE_RATE)
        if self._speech_seen and (
            self._silence_samples >= pause_limit or self._segment_samples >= max_limit
        ):
            return self._decode_segment()
        return []

    def flush(self) -> list[str]:
        """Decode any pending speech segment."""
        if not self._segment_chunks or not self._speech_seen:
            self._reset_segment()
            return []
        return self._decode_segment()

    def _decode_segment(self) -> list[str]:
        chunks = self._segment_chunks
        sample_count = self._segment_samples
        self._reset_segment()

        if not chunks or sample_count < int(MIN_SEGMENT_SECONDS * SAMPLE_RATE):
            if self._pending_language is not None:
                self._apply_pending_language()
            return []

        if self._pending_language is not None:
            self._apply_pending_language()
        self.load()

        audio = np.concatenate(chunks).astype(np.float32) / 32768.0
        try:
            inputs = self._processor(
                audio,
                sampling_rate=SAMPLE_RATE,
                return_tensors="pt",
            )
            input_values = inputs.input_values.to(dtype=self._dtype)
            with self._torch.inference_mode():
                logits = self._model(input_values).logits
            token_ids = self._torch.argmax(logits, dim=-1)
            text = self._processor.batch_decode(token_ids)[0].strip()
        except Exception:
            # Keep errors visible to the worker rather than silently returning junk.
            raise
        return text.split() if text else []

    def _reset_segment(self) -> None:
        self._segment_chunks = []
        self._segment_samples = 0
        self._silence_samples = 0
        self._speech_seen = False


class SpeechWorker(threading.Thread):
    """Background worker that converts microphone chunks into recognized words."""

    def __init__(self, chunks: queue.Queue, engine: Optional[SpeechEngine] = None) -> None:
        super().__init__(name="speech-recognition-worker", daemon=True)
        self.chunks = chunks
        self.engine = engine or MMSEngine()
        self.words: queue.Queue[str] = queue.Queue()
        self.errors: queue.Queue[str] = queue.Queue()
        self._stop_event = threading.Event()
        self.state = "starting"

    def request_stop(self) -> None:
        self._stop_event.set()

    def stop(self, timeout: Optional[float] = None) -> None:
        self.request_stop()
        if self.is_alive() and threading.current_thread() is not self:
            self.join(timeout=timeout)

    def set_language(self, language: str) -> None:
        self.engine.set_language(language)

    def _publish(self, words: list[str]) -> None:
        for word in words:
            cleaned = str(word).strip()
            if cleaned:
                self.words.put(cleaned)

    def run(self) -> None:
        try:
            self.state = "loading"
            self.engine.load()
            self.state = "listening"
            while not self._stop_event.is_set():
                try:
                    chunk = self.chunks.get(timeout=0.1)
                except queue.Empty:
                    continue
                self._publish(self.engine.accept(chunk))

            self.state = "flushing"
            # Finish queued audio on explicit shutdown, then flush the final segment.
            while True:
                try:
                    chunk = self.chunks.get_nowait()
                except queue.Empty:
                    break
                self._publish(self.engine.accept(chunk))
            self._publish(self.engine.flush())
            self.state = "stopped"
        except Exception as error:
            self.state = "error"
            self.errors.put(f"{type(error).__name__}: {error}")
        finally:
            if self.state not in {"error", "stopped"}:
                self.state = "stopped"
