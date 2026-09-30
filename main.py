"""
main.py - GUI ONLY (Tkinter). ROUGH DRAFT, untested with a real camera and mic.

Layout (from the mockup, plus a slim toolbar that the mockup did not have):

    +----------------+----------------+
    |     video      |                |
    +----------------+   transcript   |
    |   audio wave   |                |
    +----------------+----------------+
    [Start] [Stop] | [Lock] [Unlock] | [direction] [Translate]      status

This file holds NO detection, audio, speech or translation logic. It starts worker
objects from the other modules, polls their queues, and draws what they return:

    face.py      FaceWorker      live video frames         (starts on Start session)
    audio.py     AudioRecorder   live mic + waveform data  (starts on Start session)
    speech.py    SpeechWorker    live words for the transcript
    language.py  translate_long  runs AFTER the session, on the Translate button

Every module is imported lazily and started independently, so a missing package or
device (no mic, no weights file) disables that panel only, not the whole app.
Worker threads never touch Tk. They talk to the GUI through queues only.

Needs: pip install pillow numpy   (pillow arrives with ultralytics)
Keys:  L = lock largest face, U = unlock
"""

import queue
import threading
import tkinter as tk
from tkinter import messagebox, ttk

import numpy as np
from PIL import Image, ImageTk

# --- Behavior ----------------------------------------------------------------
LIVE_WAVE_SECONDS = 10      # Live view shows the last N seconds. After Stop, the full recording.
WAVE_DISPLAY_GAIN = 3.0     # Vertical zoom for DISPLAY only. Stored samples are never changed.
VIDEO_POLL_MS = 20
SLOW_POLL_MS = 60           # Waveform, transcript, errors, translation results.
DIRECTIONS = {
    "Cebuano to Tagalog": ("Cebuano", "Tagalog"),
    "Tagalog to Cebuano": ("Tagalog", "Cebuano"),
}
SPEAKING = {"Cebuano": "ceb", "Tagalog": "tgl"}   # Language spoken in the session -> MMS code.

# --- Colors ------------------------------------------------------------------
BG = "#16181d"
PANEL = "#0e0f12"
BAR = "#1e2128"
BUTTON = "#2b303a"
BUTTON_HOVER = "#39404d"
TEXT = "#e6e6e6"
MUTED = "#8a8f98"
WAVE_COLOR = "#3ddc97"
WAVE_CENTER_LINE = "#2a2f38"
LIVE_COLOR = "#ff5c5c"


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("AyosTa")
        self.geometry("1180x740")
        self.minsize(800, 560)
        self.configure(bg=BG)

        # Session parts. Each is None when it is not running or failed to start.
        self.face_worker = None
        self.recorder = None        # Kept after Stop so the full waveform stays visible.
        self.speech_worker = None
        self.session_active = False
        self.translating = False

        self._words = []            # Live words only. Translation reads this, not the widget.
        self._video_image = None    # Tk drops images that nothing references.
        self._wave_key = None       # Skips redraws when nothing changed.
        self._reported_errors = set()
        self._last_speech_state = None
        self._translation_queue = queue.Queue()

        self._build_style()
        self._build_layout()
        self._refresh_buttons()
        self._draw_video_idle()

        self.bind("<l>", lambda event: self._lock_face())
        self.bind("<u>", lambda event: self._unlock_face())
        self.protocol("WM_DELETE_WINDOW", self._on_close)

        self.after(VIDEO_POLL_MS, self._poll_video)
        self.after(SLOW_POLL_MS, self._poll_slow)

    # ------------------------------------------------------------------ layout

    def _build_style(self) -> None:
        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure("Bar.TFrame", background=BAR)
        style.configure("Bar.TLabel", background=BAR, foreground=MUTED)
        style.configure("TButton", background=BUTTON, foreground=TEXT,
                        borderwidth=0, focusthickness=0, padding=(12, 6))
        style.map("TButton",
                  background=[("disabled", BAR), ("active", BUTTON_HOVER)],
                  foreground=[("disabled", MUTED)])
        style.configure("TCombobox", fieldbackground=BUTTON, background=BUTTON,
                        foreground=TEXT, arrowcolor=TEXT, borderwidth=0,
                        bordercolor=BUTTON, lightcolor=BUTTON, darkcolor=BUTTON,
                        selectbackground=BUTTON, selectforeground=TEXT)
        style.map("TCombobox", fieldbackground=[("readonly", BUTTON)],
                  foreground=[("readonly", TEXT)],
                  selectbackground=[("readonly", BUTTON)],
                  selectforeground=[("readonly", TEXT)])
        style.configure("Vertical.TScrollbar", background=BUTTON, troughcolor=PANEL,
                        bordercolor=PANEL, lightcolor=BUTTON, darkcolor=BUTTON,
                        borderwidth=0, arrowcolor=TEXT)
        style.map("Vertical.TScrollbar", background=[("active", BUTTON_HOVER)])
        style.configure("TSeparator", background=BUTTON)
        style.configure("Status.TLabel", background=BG, foreground=MUTED,
                        font=("Segoe UI", 9))
        self.option_add("*TCombobox*Listbox.background", PANEL)
        self.option_add("*TCombobox*Listbox.foreground", TEXT)

    def _build_layout(self) -> None:
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)

        content = tk.Frame(self, bg=BG)
        content.grid(row=0, column=0, sticky="nsew", padx=6, pady=(6, 0))
        content.rowconfigure(0, weight=1)
        content.columnconfigure((0, 1), weight=1, uniform="columns")

        # Left column: video on top, waveform below, equal height.
        left = tk.Frame(content, bg=BG)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 3))
        left.columnconfigure(0, weight=1)
        left.rowconfigure((0, 1), weight=1, uniform="rows")

        self.video_canvas = tk.Canvas(left, bg=PANEL, highlightthickness=0)
        self.video_canvas.grid(row=0, column=0, sticky="nsew", pady=(0, 3))
        self.video_canvas.bind("<Configure>", self._on_video_resize)

        self.wave_canvas = tk.Canvas(left, bg=PANEL, highlightthickness=0)
        self.wave_canvas.grid(row=1, column=0, sticky="nsew", pady=(3, 0))

        # Right column: transcript, full height.
        right = tk.Frame(content, bg=BG)
        right.grid(row=0, column=1, sticky="nsew", padx=(3, 0))
        right.columnconfigure(0, weight=1)
        right.rowconfigure(0, weight=1)

        self.transcript = tk.Text(
            right, wrap="word", state="disabled", bg=PANEL, fg=TEXT,
            relief="flat", padx=14, pady=12, font=("Segoe UI", 11),
            highlightthickness=0,
        )
        self.transcript.grid(row=0, column=0, sticky="nsew")
        scrollbar = ttk.Scrollbar(right, orient="vertical", command=self.transcript.yview)
        scrollbar.grid(row=0, column=1, sticky="ns")
        self.transcript.configure(yscrollcommand=scrollbar.set)
        self.transcript.tag_configure("divider", foreground=MUTED, spacing1=10, spacing3=6)
        self.transcript.tag_configure("translation", foreground=WAVE_COLOR)

        # Toolbar.
        toolbar = ttk.Frame(self, style="Bar.TFrame")
        toolbar.grid(row=1, column=0, sticky="ew", padx=6, pady=6)

        self.start_button = ttk.Button(toolbar, text="Start session", command=self._start_session)
        self.stop_button = ttk.Button(toolbar, text="Stop session", command=self._stop_session)
        self.lock_button = ttk.Button(toolbar, text="Lock face (L)", command=self._lock_face)
        self.unlock_button = ttk.Button(toolbar, text="Unlock (U)", command=self._unlock_face)
        self.speaking = tk.StringVar(value=next(iter(SPEAKING)))
        speaking_label = ttk.Label(toolbar, text="Speaking", style="Bar.TLabel")
        speaking_box = ttk.Combobox(toolbar, textvariable=self.speaking, state="readonly",
                                    values=list(SPEAKING), width=9)
        speaking_box.bind("<<ComboboxSelected>>", self._on_speaking_changed)
        self.direction = tk.StringVar(value=next(iter(DIRECTIONS)))
        direction_box = ttk.Combobox(toolbar, textvariable=self.direction, state="readonly",
                                     values=list(DIRECTIONS), width=20)
        self.translate_button = ttk.Button(toolbar, text="Translate transcript",
                                           command=self._translate)
        self.status = tk.StringVar(value="Ready")
        status_label = ttk.Label(self, textvariable=self.status, style="Status.TLabel")
        status_label.grid(row=2, column=0, sticky="w", padx=10, pady=(0, 6))

        def divider() -> None:
            tk.Frame(toolbar, width=1, bg=BUTTON).pack(side="left", fill="y", padx=8, pady=6)

        self.start_button.pack(side="left", padx=(6, 2), pady=4)
        self.stop_button.pack(side="left", padx=2, pady=4)
        divider()
        self.lock_button.pack(side="left", padx=2, pady=4)
        self.unlock_button.pack(side="left", padx=2, pady=4)
        divider()
        speaking_label.pack(side="left", padx=(2, 4), pady=4)
        speaking_box.pack(side="left", padx=2, pady=4)
        divider()
        direction_box.pack(side="left", padx=2, pady=4)
        self.translate_button.pack(side="left", padx=2, pady=4)

    def _refresh_buttons(self) -> None:
        def set_enabled(button: ttk.Button, enabled: bool) -> None:
            button.state(["!disabled"] if enabled else ["disabled"])

        idle = not self.session_active and not self.translating and not self._finishing_speech()
        face_live = self.session_active and self.face_worker is not None
        set_enabled(self.start_button, idle)
        set_enabled(self.stop_button, self.session_active)
        set_enabled(self.lock_button, face_live)
        set_enabled(self.unlock_button, face_live)
        set_enabled(self.translate_button, idle and bool(self._words))

    # ----------------------------------------------------------------- session

    def _start_session(self) -> None:
        # A new session replaces the previous transcript and recording.
        self._clear_transcript()
        self._words = []
        self._reported_errors = set()
        self._last_speech_state = None
        self._wave_key = None

        problems = []
        self.face_worker = self._try_start("Video", self._make_face_worker, problems)
        self.recorder = self._try_start("Audio", self._make_recorder, problems)
        self.speech_worker = None
        if self.recorder is not None:
            self.speech_worker = self._try_start("Speech", self._make_speech_worker, problems)

        if problems:
            messagebox.showwarning("Some parts could not start", "\n\n".join(problems))
        if self.face_worker is None and self.recorder is None:
            self.status.set("Nothing could start")
            self._refresh_buttons()
            return

        self.session_active = True
        self._set_running_status()
        self._refresh_buttons()

    def _try_start(self, label: str, factory, problems: list):
        """Start one part. On failure, record why and carry on without it."""
        try:
            return factory()
        except Exception as error:  # Broad on purpose: parts must fail independently.
            problems.append(f"{label}: {error}")
            return None

    def _make_face_worker(self):
        from face import FaceWorker
        worker = FaceWorker()
        worker.start()
        return worker

    def _make_recorder(self):
        from audio import AudioRecorder
        recorder = AudioRecorder()
        recorder.start()
        return recorder

    def _make_speech_worker(self):
        from speech import MMSEngine, SpeechWorker
        engine = MMSEngine(language=SPEAKING[self.speaking.get()])
        worker = SpeechWorker(self.recorder.chunks, engine=engine)
        worker.start()  # Loads the model on its own thread, so the window never freezes.
        return worker

    def _stop_session(self) -> None:
        if self.face_worker is not None:
            self.face_worker.stop()
        if self.recorder is not None:
            self.recorder.stop()
        if self.speech_worker is not None:
            # Non-blocking: the worker may still be loading or transcribing the last
            # segment. _finish_speech_if_done() collects its final words later.
            self.speech_worker.request_stop()

        self.face_worker = None
        self.session_active = False
        self._wave_key = None
        self._draw_video_idle()
        if self.speech_worker is not None:
            self.status.set("Session stopped. Finishing the transcript...")
        else:
            self.status.set(f"Session stopped. {len(self._words)} words in the transcript.")
        self._refresh_buttons()

    def _finishing_speech(self) -> bool:
        """True while a stopped session's speech worker is still producing words."""
        return self.speech_worker is not None and not self.session_active

    def _finish_speech_if_done(self) -> None:
        worker = self.speech_worker
        if worker is None or self.session_active or worker.is_alive():
            return
        self._drain_words()
        self.speech_worker = None
        self.status.set(f"Session stopped. {len(self._words)} words in the transcript.")
        self._refresh_buttons()

    def _refresh_speech_status(self) -> None:
        """Show model loading / listening without overwriting error messages."""
        worker = self.speech_worker
        if worker is None or worker.state == self._last_speech_state:
            return
        self._last_speech_state = worker.state
        if self.session_active and not worker.error:
            self._set_running_status()

    def _on_speaking_changed(self, event=None) -> None:
        language = self.speaking.get()
        other = "Tagalog" if language == "Cebuano" else "Cebuano"
        self.direction.set(f"{language} to {other}")  # Translate what was spoken.
        if self.speech_worker is not None and self.session_active:
            self.speech_worker.set_language(SPEAKING[language])

    def _set_running_status(self) -> None:
        video = "video on" if self.face_worker else "video off"
        audio = "audio on" if self.recorder else "audio off"
        speech = f"speech: {self.speech_worker.state}" if self.speech_worker else "speech off"
        self.status.set(f"Recording ({video}, {audio}, {speech})")

    def _lock_face(self) -> None:
        if self.face_worker is not None:
            self.face_worker.lock_largest()

    def _unlock_face(self) -> None:
        if self.face_worker is not None:
            self.face_worker.unlock()

    def _on_close(self) -> None:
        if self.session_active:
            self._stop_session()  # Non-blocking for speech. Daemon threads end with the app.
        self.destroy()

    # ------------------------------------------------------------ translation

    def _translate(self) -> None:
        text = " ".join(self._words).strip()
        if not text:
            self.status.set("Nothing to translate.")
            return
        source, target = DIRECTIONS[self.direction.get()]
        self.translating = True
        self._refresh_buttons()
        self.status.set("Translating. The first run loads the model and can take a while.")
        threading.Thread(target=self._translation_job, args=(text, source, target),
                         daemon=True).start()

    def _translation_job(self, text: str, source: str, target: str) -> None:
        """Runs on a worker thread. Reports back through the queue only."""
        try:
            from language import translate_long
            result = translate_long(
                text, source, target,
                on_progress=lambda done, total: self._translation_queue.put(("progress", (done, total))),
            )
            self._translation_queue.put(("done", (result, source, target)))
        except Exception as error:  # Broad on purpose: show the reason in the GUI.
            self._translation_queue.put(("error", f"{type(error).__name__}: {error}"))

    def _drain_translation(self) -> None:
        while True:
            try:
                kind, payload = self._translation_queue.get_nowait()
            except queue.Empty:
                return
            if kind == "progress":
                done, total = payload
                self.status.set(f"Translating: part {done} of {total}")
            elif kind == "done":
                result, source, target = payload
                self._append_transcript(f"\n{source} to {target}\n", "divider")
                self._append_transcript(result + "\n", "translation")
                self.translating = False
                self.status.set("Translation finished.")
                self._refresh_buttons()
            elif kind == "error":
                self.translating = False
                self.status.set("Translation failed.")
                self._refresh_buttons()
                messagebox.showerror("Translation failed", payload)

    # -------------------------------------------------------------- transcript

    def _clear_transcript(self) -> None:
        self.transcript.configure(state="normal")
        self.transcript.delete("1.0", "end")
        self.transcript.configure(state="disabled")

    def _append_transcript(self, text: str, tag: str = "") -> None:
        self.transcript.configure(state="normal")
        self.transcript.insert("end", text, tag)
        self.transcript.see("end")
        self.transcript.configure(state="disabled")

    def _drain_words(self) -> None:
        if self.speech_worker is None:
            return
        new_words = []
        while True:
            try:
                new_words.append(self.speech_worker.words.get_nowait())
            except queue.Empty:
                break
        if new_words:
            self._words.extend(new_words)
            self._append_transcript(" ".join(new_words) + " ")

    # ------------------------------------------------------------------- video

    def _poll_video(self) -> None:
        if self.face_worker is not None:
            try:
                frame = self.face_worker.frames.get_nowait()
            except queue.Empty:
                frame = None
            if frame is not None:
                self._show_frame(frame)
        self.after(VIDEO_POLL_MS, self._poll_video)

    def _show_frame(self, frame: np.ndarray) -> None:
        canvas = self.video_canvas
        canvas_width, canvas_height = canvas.winfo_width(), canvas.winfo_height()
        if canvas_width < 8 or canvas_height < 8:
            return
        height, width = frame.shape[:2]
        scale = min(canvas_width / width, canvas_height / height)
        size = (max(1, int(width * scale)), max(1, int(height * scale)))
        image = Image.fromarray(frame[:, :, ::-1])  # OpenCV gives BGR, Tk needs RGB.
        image = image.resize(size, Image.Resampling.BILINEAR)
        self._video_image = ImageTk.PhotoImage(image)
        canvas.delete("all")
        canvas.create_image(canvas_width // 2, canvas_height // 2, image=self._video_image)

    def _on_video_resize(self, event) -> None:
        if self.face_worker is None:
            self._draw_video_idle()

    def _draw_video_idle(self) -> None:
        canvas = self.video_canvas
        canvas.delete("all")
        canvas.create_text(canvas.winfo_width() // 2, canvas.winfo_height() // 2,
                           text="Video", fill=MUTED, font=("Segoe UI", 12))

    # ---------------------------------------------------------------- waveform

    def _poll_slow(self) -> None:
        self._drain_words()
        self._finish_speech_if_done()
        self._refresh_speech_status()
        self._redraw_wave()
        self._check_worker_errors()
        self._drain_translation()
        self.after(SLOW_POLL_MS, self._poll_slow)

    def _redraw_wave(self) -> None:
        """
        Editor-style waveform: one min/max pair per pixel column, drawn as a filled
        envelope. Live = last LIVE_WAVE_SECONDS. After Stop = the whole recording.
        TODO(team): zoom, scroll, selection, time ruler. The data side (audio.py)
        already supports any range through AudioBuffer.peaks(start, end, columns).
        """
        canvas = self.wave_canvas
        width, height = canvas.winfo_width(), canvas.winfo_height()
        if width < 8 or height < 8:
            return

        recorder = self.recorder
        total = recorder.buffer.length if recorder else 0
        running = bool(recorder and recorder.is_running)
        key = (width, height, total, running)
        if key == self._wave_key:
            return  # Nothing changed, so skip the work (matters for long stopped recordings).
        self._wave_key = key

        canvas.delete("all")
        middle = height // 2
        canvas.create_line(0, middle, width, middle, fill=WAVE_CENTER_LINE)

        if recorder is None or total == 0:
            label = "Audio waveform" if recorder is None else "Waiting for audio..."
            canvas.create_text(width // 2, middle - 14, text=label, fill=MUTED,
                               font=("Segoe UI", 12))
            return

        window = int(recorder.sample_rate * LIVE_WAVE_SECONDS)
        if running:
            start = max(0, total - window)
            span = max(2, int(width * min(total, window) / window))
        else:
            start = 0
            span = width

        mins, maxs = recorder.buffer.peaks(start, total, span)
        count = len(mins)
        if count < 2:
            return

        scale = height / 2 - 6
        top = middle - np.clip(maxs * WAVE_DISPLAY_GAIN, -1.0, 1.0) * scale
        bottom = middle - np.clip(mins * WAVE_DISPLAY_GAIN, -1.0, 1.0) * scale
        xs = np.arange(count) * (span / count)
        outline = np.vstack((np.column_stack((xs, top)),
                             np.column_stack((xs, bottom))[::-1]))
        canvas.create_polygon(outline.ravel().tolist(), fill=WAVE_COLOR, outline="")

        seconds = total / recorder.sample_rate
        text = f"LIVE  {seconds:.1f}s" if running else f"Stopped  {seconds:.1f}s"
        canvas.create_text(10, 8, anchor="nw", text=text, font=("Segoe UI", 9),
                           fill=LIVE_COLOR if running else MUTED)

    # ------------------------------------------------------------------ errors

    def _check_worker_errors(self) -> None:
        for label, worker in (("Video", self.face_worker), ("Speech", self.speech_worker)):
            if worker is None or not worker.error or label in self._reported_errors:
                continue
            self._reported_errors.add(label)
            self.status.set(f"{label} stopped: {worker.error}")
            messagebox.showerror(f"{label} problem", worker.error)


if __name__ == "__main__":
    App().mainloop()