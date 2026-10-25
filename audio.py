"""Persistent microphone capture with a preview waveform and session-only recording.

The microphone stream is opened once when the application starts. While no session is
active, a short rolling preview buffer is updated but audio is NOT saved to the session
buffer and is NOT sent to speech recognition. start_recording()/stop_recording() only
change the capture gate; they never close or reopen the microphone.

Needs: pip install sounddevice numpy
"""

from __future__ import annotations

import queue
import threading

import numpy as np

SAMPLE_RATE = 16000
CHANNELS = 1
CHUNK_QUEUE_SIZE = 3000
INITIAL_SECONDS = 60
PREVIEW_SECONDS = 10
_EMPTY = np.zeros(0, dtype=np.float32)


class AudioBuffer:
    """Thread-safe int16 sample buffer, optionally retaining only the latest N seconds."""

    def __init__(self, sample_rate: int = SAMPLE_RATE, max_seconds: float | None = None):
        self.sample_rate = sample_rate
        self.max_samples = int(sample_rate * max_seconds) if max_seconds else None
        capacity = self.max_samples or INITIAL_SECONDS * sample_rate
        self._data = np.zeros(max(1, capacity), dtype=np.int16)
        self._length = 0
        self._lock = threading.Lock()

    @property
    def length(self) -> int:
        with self._lock:
            return self._length

    def clear(self) -> None:
        with self._lock:
            self._length = 0

    def append(self, samples: np.ndarray) -> None:
        samples = np.asarray(samples, dtype=np.int16).reshape(-1)
        count = len(samples)
        if count == 0:
            return
        with self._lock:
            if self.max_samples is not None:
                limit = self.max_samples
                if count >= limit:
                    self._data[:limit] = samples[-limit:]
                    self._length = limit
                    return
                overflow = max(0, self._length + count - limit)
                if overflow:
                    remaining = self._length - overflow
                    if remaining > 0:
                        self._data[:remaining] = self._data[overflow:self._length]
                    self._length = remaining
                self._data[self._length:self._length + count] = samples
                self._length += count
                return

            needed = self._length + count
            if needed > len(self._data):
                grown = np.zeros(max(needed, len(self._data) * 2), dtype=np.int16)
                grown[:self._length] = self._data[:self._length]
                self._data = grown
            self._data[self._length:needed] = samples
            self._length = needed

    def snapshot(self, start: int = 0, end=None) -> np.ndarray:
        with self._lock:
            end = self._length if end is None else min(int(end), self._length)
            start = max(0, int(start))
            if end <= start:
                return np.zeros(0, dtype=np.int16)
            return self._data[start:end].copy()

    def peaks(self, start: int, end: int, columns: int):
        with self._lock:
            end = min(int(end), self._length)
            start = max(0, int(start))
            count = end - start
            if count <= 0 or columns < 1:
                return _EMPTY, _EMPTY
            columns = min(int(columns), count)
            # Include every sample, including the remainder in the final bucket.
            edges = np.linspace(start, end, columns + 1, dtype=np.int64)
            mins = np.empty(columns, dtype=np.float32)
            maxs = np.empty(columns, dtype=np.float32)
            for i in range(columns):
                left, right = int(edges[i]), max(int(edges[i + 1]), int(edges[i]) + 1)
                view = self._data[left:min(right, end)]
                mins[i] = view.min() / 32768.0
                maxs[i] = view.max() / 32768.0
            return mins, maxs


class AudioRecorder:
    """One persistent mic stream with separate preview and session recording buffers."""

    def __init__(self, sample_rate: int = SAMPLE_RATE, device=None):
        self.sample_rate = sample_rate
        self.buffer = AudioBuffer(sample_rate)  # Recorded session only.
        self.preview_buffer = AudioBuffer(sample_rate, max_seconds=PREVIEW_SECONDS)
        self.chunks = queue.Queue(maxsize=CHUNK_QUEUE_SIZE)
        self._device = device
        self._stream = None
        self._running = False
        self._recording = False
        self._state_lock = threading.Lock()

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def is_recording(self) -> bool:
        with self._state_lock:
            return self._recording

    @property
    def duration_seconds(self) -> float:
        return self.buffer.length / self.sample_rate

    def start(self) -> None:
        """Open the mic stream once. Recording remains disabled until start_recording()."""
        if self._running:
            return
        try:
            import sounddevice
        except ImportError as error:
            raise RuntimeError("sounddevice is not installed. Run: pip install sounddevice") from error
        try:
            self._stream = sounddevice.InputStream(
                samplerate=self.sample_rate,
                channels=CHANNELS,
                dtype="int16",
                device=self._device,
                callback=self._on_audio,
            )
            self._stream.start()
            self._running = True
        except (sounddevice.PortAudioError, ValueError) as error:
            self._stream = None
            raise RuntimeError(f"Could not open the microphone: {error}") from error

    def start_recording(self) -> None:
        """Begin a fresh session without restarting the microphone stream."""
        if not self._running:
            raise RuntimeError("The microphone is not running.")
        self.buffer.clear()
        self._drain_chunks()
        with self._state_lock:
            self._recording = True

    def stop_recording(self) -> None:
        """Stop retaining/sending audio while leaving the microphone preview active."""
        with self._state_lock:
            self._recording = False

    def stop(self) -> None:
        """Close the microphone stream; reserved for application shutdown."""
        self.stop_recording()
        self._running = False
        if self._stream is not None:
            try:
                self._stream.stop()
            finally:
                self._stream.close()
                self._stream = None

    def _drain_chunks(self) -> None:
        while True:
            try:
                self.chunks.get_nowait()
            except queue.Empty:
                return

    def _on_audio(self, indata, frames, time_info, status) -> None:
        samples = indata[:, 0].copy()
        self.preview_buffer.append(samples)
        with self._state_lock:
            recording = self._recording
        if not recording:
            return
        self.buffer.append(samples)
        try:
            self.chunks.put_nowait(samples)
        except queue.Full:
            # Preserve realtime responsiveness; the buffer still retains the raw audio.
            pass
