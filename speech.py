"""
speech.py - live speech-to-text for Cebuano and Tagalog. ROUGH DRAFT.

ENGINE: Meta MMS ("facebook/mms-1b-all"), run through transformers.
    The checkpoint lists both ceb (Cebuano) and tgl (Tagalog). One 1B-parameter
    base model is shared and a small per-language adapter is swapped in, so
    switching language mid-session is cheap.

HOW IT WORKS
    1. Audio chunks arrive from audio.py (int16, mono, 16 kHz).
    2. MMSEngine drops leading silence, collects speech, and waits for a pause
       (PAUSE_SECONDS) or MAX_SEGMENT_SECONDS.
    3. The segment is transcribed and the words go to the transcript.
    So words appear AFTER each pause, not while you are mid-sentence.

NOT VERIFIED (I could not run the real model where this was written)
    - Speed on a CPU-only machine. A 1B model may take several seconds per
      segment, so the transcript can lag behind the speaker.
    - RAM. bfloat16 needs roughly 2 GB for weights, float32 roughly 4 GB.
      YOLO runs in the same process.
    - The transformers v5 calls (target_lang, load_adapter, set_target_lang)
      follow the MMS docs. Check them against the installed version.
    - Recognition quality. MMS output has no punctuation or capitals, and no
      language model is used. Expect errors, especially on mixed-language speech.
    - The model license. Check it before this leaves the capstone.

FIRST RUN
    The model download is roughly 4 GB. Do it once BEFORE the first live session
    (audio recorded while the model loads can be dropped):
        python -c "from speech import MMSEngine; MMSEngine().load()"

TODO(hardware)
    Proper voice-activity detection instead of the energy gate below, overlap
    between segments, a language model for decoding, GPU inference, and
    streaming output so words appear mid-sentence.
"""

import queue
import threading

import numpy as np

from audio import SAMPLE_RATE

# --- MMS settings --------------------------------------------------------------
MMS_MODEL_ID = "facebook/mms-1b-all"
MMS_DTYPE = "bfloat16"       # "float32" needs about double the RAM. TODO(hardware)
DEFAULT_LANGUAGE = "tgl"     # MMS codes: "ceb" = Cebuano, "tgl" = Tagalog

# --- Segmentation (energy gate) ------------------------------------------------
SILENCE_RMS = 300            # int16 RMS below this counts as silence. Mic dependent:
                             # if nothing is transcribed, lower it; if noise triggers, raise it.
PAUSE_SECONDS = 0.7          # Silence this long ends a segment.
MAX_SEGMENT_SECONDS = 15.0   # Forces a cut during long uninterrupted speech.
MIN_SEGMENT_SECONDS = 0.4    # Shorter blips (clicks, coughs) are ignored.


class SpeechEngine:
    """Interface. The default does nothing, which keeps the GUI runnable."""

    name = "not connected"
    ready_state = "not connected"   # Shown in the status bar once load() has finished.

    def load(self) -> None:
        """Load models. Runs on the worker thread before audio is processed."""

    def set_language(self, code: str) -> None:
        """Switch the spoken language. May be called from the GUI thread."""

    def accept(self, samples: np.ndarray) -> list:
        """Receive one audio chunk. Return a list of new finalized words."""
        return []

    def flush(self) -> list:
        """Called once when the session stops. Return any words still buffered."""
        return []


class MMSEngine(SpeechEngine):
    name = "MMS Cebuano/Tagalog"
    ready_state = "listening"

    def __init__(self, language: str = DEFAULT_LANGUAGE):
        self._language = language          # Wanted language (set from the GUI thread).
        self._loaded_language = None       # Language the model currently has loaded.
        self._torch = None
        self._processor = None
        self._model = None
        self._dtype = None
        self._reset_segment()

    # ------------------------------------------------------------------ loading

    def load(self) -> None:
        # Imported here so that importing speech.py stays cheap and safe.
        import torch
        from transformers import AutoProcessor, Wav2Vec2ForCTC

        self._torch = torch
        self._dtype = getattr(torch, MMS_DTYPE)
        self._processor = AutoProcessor.from_pretrained(
            MMS_MODEL_ID, target_lang=self._language
        )
        self._model = Wav2Vec2ForCTC.from_pretrained(
            MMS_MODEL_ID,
            target_lang=self._language,
            ignore_mismatched_sizes=True,
            dtype=self._dtype,
        )
        self._model.eval()
        self._loaded_language = self._language

    def set_language(self, code: str) -> None:
        # Only records the request. The worker thread applies it before the next segment.
        self._language = code

    def _apply_pending_language(self) -> None:
        if self._language != self._loaded_language:
            self._processor.tokenizer.set_target_lang(self._language)
            self._model.load_adapter(self._language)
            self._loaded_language = self._language

    # ---------------------------------------------------------------- streaming

    def accept(self, samples: np.ndarray) -> list:
        if len(samples) == 0:
            return []
        if self._model is None:
            self.load()

        rms = float(np.sqrt(np.mean(samples.astype(np.float32) ** 2)))
        if rms >= SILENCE_RMS:
            self._speech_seen = True
            self._speech_samples += len(samples)
            self._silent_run = 0
        else:
            self._silent_run += len(samples)

        if not self._speech_seen:
            return []  # Drop leading silence.

        self._chunks.append(samples)
        self._buffered += len(samples)

        paused = self._silent_run >= PAUSE_SECONDS * SAMPLE_RATE
        too_long = self._buffered >= MAX_SEGMENT_SECONDS * SAMPLE_RATE
        if paused or too_long:
            return self._decode_segment()
        return []

    def flush(self) -> list:
        if self._model is None or not self._speech_seen:
            return []
        return self._decode_segment()

    def _decode_segment(self) -> list:
        audio = np.concatenate(self._chunks)
        speech_samples = self._speech_samples  # Loud part only, not the trailing pause.
        self._reset_segment()
        if speech_samples < MIN_SEGMENT_SECONDS * SAMPLE_RATE:
            return []

        self._apply_pending_language()
        waveform = audio.astype(np.float32) / 32768.0
        inputs = self._processor(waveform, sampling_rate=SAMPLE_RATE, return_tensors="pt")
        input_values = inputs["input_values"].to(self._dtype)
        with self._torch.no_grad():
            logits = self._model(input_values).logits
        token_ids = self._torch.argmax(logits, dim=-1)[0]
        return self._processor.decode(token_ids).split()

    def _reset_segment(self) -> None:
        self._chunks = []
        self._buffered = 0
        self._speech_samples = 0
        self._speech_seen = False
        self._silent_run = 0


class SpeechWorker(threading.Thread):
    """Pulls audio chunks, runs the engine, and pushes words for the GUI to read."""

    def __init__(self, chunks: queue.Queue, engine: SpeechEngine = None):
        super().__init__(daemon=True)
        self.chunks = chunks               # from audio.AudioRecorder.chunks
        self.engine = engine or SpeechEngine()
        self.words = queue.Queue()         # read by main.py
        self.state = "starting"            # Shown in the status bar.
        self.error = None
        self._stop_event = threading.Event()

    @property
    def engine_name(self) -> str:
        return self.engine.name

    def set_language(self, code: str) -> None:
        self.engine.set_language(code)

    def run(self) -> None:
        try:
            self.state = "loading model..."
            self.engine.load()
            self.state = self.engine.ready_state

            while not self._stop_event.is_set():
                try:
                    samples = self.chunks.get(timeout=0.2)
                except queue.Empty:
                    continue
                self._emit(self.engine.accept(samples))

            # Session ended: process what is already queued, then flush the engine.
            self.state = "finishing..."
            while True:
                try:
                    samples = self.chunks.get_nowait()
                except queue.Empty:
                    break
                self._emit(self.engine.accept(samples))
            self._emit(self.engine.flush())
            self.state = "stopped"
        except Exception as error:  # Broad on purpose: report to the GUI instead of dying silently.
            self.error = f"{type(error).__name__}: {error}"
            self.state = "failed"

    def request_stop(self) -> None:
        """Non-blocking. The GUI polls is_alive() and collects the final words afterwards."""
        self._stop_event.set()

    def stop(self) -> None:
        """Blocking version for scripts and tests. The GUI uses request_stop()."""
        self.request_stop()
        if self.is_alive():
            self.join(timeout=60.0)  # A segment can take a while to transcribe on CPU.

    def _emit(self, words: list) -> None:
        for word in words:
            self.words.put(word)