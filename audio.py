"""
audio.py - live microphone capture + waveform data. ROUGH DRAFT, untested on a real mic.

What it provides
    AudioBuffer    Every raw sample of the session (int16, mono), thread-safe.
                   - peaks(): min/max per pixel column, the same method audio
                     editors use to draw waveforms. Cheap even for long recordings.
                   - snapshot(): a copy of the raw samples for real analysis
                     (RMS, FFT, silence detection, and so on).
    AudioRecorder  Opens the mic, fills the buffer, and forwards each chunk to a
                   queue that the speech worker reads.

The GUI only DRAWS what peaks() returns. All analysis should use the raw samples,
never the drawn pixels.

Needs: pip install sounddevice
       (Python 3.14 wheel support was not verified. Try: pip install sounddevice --dry-run)
"""

import queue
import threading

import numpy as np

SAMPLE_RATE = 16000       # 16 kHz mono is what most speech engines expect.
CHANNELS = 1
CHUNK_QUEUE_SIZE = 3000   # Roughly a minute of chunks. Covers model loading. When full, NEW chunks are dropped.
INITIAL_SECONDS = 60      # Buffer pre-allocation. It doubles automatically when full.

_EMPTY = np.zeros(0, dtype=np.float32)


class AudioBuffer:
    """Growing int16 store of the whole session. Memory: about 115 MB per hour at 16 kHz."""

    def __init__(self, sample_rate: int = SAMPLE_RATE):
        self.sample_rate = sample_rate
        self._data = np.zeros(INITIAL_SECONDS * sample_rate, dtype=np.int16)
        self._length = 0
        self._lock = threading.Lock()

    @property
    def length(self) -> int:
        """Number of samples stored so far."""
        with self._lock:
            return self._length

    def append(self, samples: np.ndarray) -> None:
        count = len(samples)
        with self._lock:
            needed = self._length + count
            if needed > len(self._data):
                grown = np.zeros(max(needed, len(self._data) * 2), dtype=np.int16)
                grown[:self._length] = self._data[:self._length]
                self._data = grown
            self._data[self._length:needed] = samples
            self._length = needed

    def snapshot(self, start: int = 0, end=None) -> np.ndarray:
        """Copy of raw samples [start, end). Use this for analysis."""
        with self._lock:
            end = self._length if end is None else min(end, self._length)
            start = max(0, start)
            return self._data[start:end].copy()

    def peaks(self, start: int, end: int, columns: int):
        """
        Return (mins, maxs) as float arrays in [-1.0, 1.0], one pair per column.
        If the range holds fewer samples than columns, fewer columns come back.
        Runs on a view under the lock, so long ranges are computed once, not copied.
        """
        with self._lock:
            end = min(end, self._length)
            start = max(0, start)
            count = end - start
            if count <= 0 or columns < 1:
                return _EMPTY, _EMPTY
            columns = min(columns, count)
            bucket = count // columns
            view = self._data[start:start + bucket * columns].reshape(columns, bucket)
            return view.min(axis=1) / 32768.0, view.max(axis=1) / 32768.0


class AudioRecorder:
    def __init__(self, sample_rate: int = SAMPLE_RATE, device=None):
        self.sample_rate = sample_rate
        self.buffer = AudioBuffer(sample_rate)
        self.chunks = queue.Queue(maxsize=CHUNK_QUEUE_SIZE)  # read by speech.SpeechWorker
        self._device = device   # None = system default mic
        self._stream = None
        self._running = False

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def duration_seconds(self) -> float:
        return self.buffer.length / self.sample_rate

    def start(self) -> None:
        """Open the mic. Raises RuntimeError with a readable message on failure."""
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
        except (sounddevice.PortAudioError, ValueError) as error:
            self._stream = None
            raise RuntimeError(f"Could not open the microphone: {error}") from error
        self._running = True

    def stop(self) -> None:
        """Stop capturing. The buffer is kept so the full waveform can still be shown."""
        self._running = False
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

    def _on_audio(self, indata, frames, time_info, status) -> None:
        # Runs on the audio thread. Keep it fast: no drawing, no model calls.
        # TODO: log `status` (input overflow) if dropouts matter for the analysis.
        samples = indata[:, 0].copy()  # indata is reused by PortAudio, so copy it.
        self.buffer.append(samples)
        try:
            self.chunks.put_nowait(samples)
        except queue.Full:
            pass