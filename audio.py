"""
audio.py - live microphone capture + waveform data.

What it provides
    AudioBuffer    Every raw sample of the session (int16, mono), thread-safe.
                   - peaks(): min/max per pixel column, the same method audio
                     editors use to draw waveforms. Cheap even for long recordings.
                   - snapshot(): a copy of the raw samples for real analysis
                     (RMS, FFT, silence detection, and so on).

    AudioRecorder  Opens the mic, applies optional input amplification,
                   fills the buffer, and forwards each chunk to a queue
                   that the speech worker reads.

Needs:
    pip install sounddevice numpy

Notes:
    INPUT_GAIN amplifies the captured microphone signal in software.
    It does not increase the physical sensitivity of the microphone.
"""

import queue
import threading

import numpy as np


SAMPLE_RATE = 16000
CHANNELS = 1

# Roughly a minute of chunks.
# When full, NEW chunks are dropped.
CHUNK_QUEUE_SIZE = 3000

# Buffer pre-allocation.
# It doubles automatically when full.
INITIAL_SECONDS = 60

# Microphone sensitivity / software amplification.
#
# 1.0 = original microphone level
# 1.5 = slightly louder
# 2.0 = twice the amplitude
# 3.0 = three times the amplitude
#
# Be careful with high values because they can cause clipping.
INPUT_GAIN = 2.0


_EMPTY = np.zeros(0, dtype=np.float32)


class AudioBuffer:
    """Growing int16 store of the whole recording session."""

    def __init__(self, sample_rate: int = SAMPLE_RATE):
        self.sample_rate = sample_rate

        self._data = np.zeros(
            INITIAL_SECONDS * sample_rate,
            dtype=np.int16,
        )

        self._length = 0
        self._lock = threading.Lock()

    @property
    def length(self) -> int:
        """Return the number of samples currently stored."""
        with self._lock:
            return self._length

    def append(self, samples: np.ndarray) -> None:
        """Append microphone samples to the recording buffer."""
        count = len(samples)

        if count == 0:
            return

        with self._lock:
            needed = self._length + count

            if needed > len(self._data):
                new_size = max(
                    needed,
                    len(self._data) * 2,
                )

                grown = np.zeros(
                    new_size,
                    dtype=np.int16,
                )

                grown[:self._length] = self._data[:self._length]
                self._data = grown

            self._data[self._length:needed] = samples
            self._length = needed

    def snapshot(self, start: int = 0, end=None) -> np.ndarray:
        """
        Return a copy of raw samples in the range [start, end).

        Use this for:
            - RMS
            - FFT
            - silence detection
            - speech analysis
        """
        with self._lock:
            start = max(0, start)

            if end is None:
                end = self._length
            else:
                end = min(end, self._length)

            if start >= end:
                return _EMPTY.copy()

            return self._data[start:end].copy()

    def peaks(self, start: int, end: int, columns: int):
        """
        Return minimum and maximum values for each waveform column.

        The returned values are normalized to approximately [-1.0, 1.0].
        """
        with self._lock:
            end = min(end, self._length)
            start = max(0, start)

            count = end - start

            if count <= 0 or columns < 1:
                return _EMPTY, _EMPTY

            columns = min(columns, count)

            bucket = count // columns

            if bucket < 1:
                return _EMPTY, _EMPTY

            usable_count = bucket * columns

            view = self._data[
                start:start + usable_count
            ].reshape(columns, bucket)

            mins = view.min(axis=1) / 32768.0
            maxs = view.max(axis=1) / 32768.0

            return mins, maxs


class AudioRecorder:
    """Capture microphone audio and store it in an AudioBuffer."""

    def __init__(
        self,
        sample_rate: int = SAMPLE_RATE,
        device=None,
        input_gain: float = INPUT_GAIN,
    ):
        self.sample_rate = sample_rate

        self.buffer = AudioBuffer(sample_rate)

        # Queue consumed by the speech worker.
        self.chunks = queue.Queue(
            maxsize=CHUNK_QUEUE_SIZE
        )

        # None means use the system default microphone.
        self._device = device

        # Software microphone amplification.
        self.input_gain = max(0.0, float(input_gain))

        self._stream = None
        self._running = False

    @property
    def is_running(self) -> bool:
        """Return True when microphone recording is active."""
        return self._running

    @property
    def duration_seconds(self) -> float:
        """Return the current recording duration in seconds."""
        return self.buffer.length / self.sample_rate

    def start(self) -> None:
        """Open and start the microphone."""
        try:
            import sounddevice
        except ImportError as error:
            raise RuntimeError(
                "sounddevice is not installed. "
                "Run: pip install sounddevice"
            ) from error

        try:
            self._stream = sounddevice.InputStream(
                samplerate=self.sample_rate,
                channels=CHANNELS,
                dtype="int16",
                device=self._device,
                callback=self._on_audio,
            )

            self._stream.start()

        except (
            sounddevice.PortAudioError,
            ValueError,
        ) as error:
            self._stream = None

            raise RuntimeError(
                f"Could not open the microphone: {error}"
            ) from error

        self._running = True

    def stop(self) -> None:
        """
        Stop microphone capture.

        The recorded buffer is kept so the complete waveform
        can still be displayed or analyzed.
        """
        self._running = False

        if self._stream is not None:
            try:
                self._stream.stop()
            finally:
                self._stream.close()
                self._stream = None

    def set_gain(self, gain: float) -> None:
        """
        Change the software microphone gain.

        Examples:
            recorder.set_gain(1.0)  # Original level
            recorder.set_gain(2.0)  # 2x amplitude
            recorder.set_gain(3.0)  # 3x amplitude
        """
        gain = float(gain)

        if not np.isfinite(gain) or gain < 0:
            raise ValueError(
                "Input gain must be a finite number greater than or equal to 0."
            )

        self.input_gain = gain

    def _on_audio(
        self,
        indata,
        frames,
        time_info,
        status,
    ) -> None:
        """
        Process one microphone chunk.

        This runs on the audio thread, so keep it lightweight.
        """

        # Copy because PortAudio reuses the input buffer.
        samples = indata[:, 0].copy()

        # Apply software amplification.
        if self.input_gain != 1.0:
            amplified = samples.astype(
                np.float32
            ) * self.input_gain

            # Prevent int16 overflow and distortion caused by
            # values outside the valid int16 range.
            amplified = np.clip(
                amplified,
                -32768.0,
                32767.0,
            )

            samples = amplified.astype(np.int16)

        # Store the complete recording.
        self.buffer.append(samples)

        # Send the same processed samples to the speech worker.
        try:
            self.chunks.put_nowait(samples)
        except queue.Full:
            # If the speech worker is temporarily too slow,
            # don't block the audio callback.
            pass