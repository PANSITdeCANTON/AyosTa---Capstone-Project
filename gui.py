"""
gui.py - the VIEW (Tkinter). ROUGH DRAFT.

Layout (from the mockup, plus a slim toolbar that the mockup did not have):

    +----------------+----------------+
    |     video      |                |
    +----------------+   transcript   |
    |   audio wave   |                |
    +----------------+----------------+
    [Start] [Stop] | [Lock] [Unlock] | Speaking [..] | [direction] [Translate] [Save audio] [Export PDF]   [Settings]
    status

This file only DRAWS the window and forwards clicks to the controller (main.py).
It does NOT import face, audio, speech, language, exports or syscheck, and it never touches
a worker thread. Everything it shows comes from two controller calls:
    controller.update()       list of events (status text, new words, errors, ...)
    controller.poll_frame()   newest camera frame
Everything it does goes through controller methods (start_session, translate, save_audio, ...).

Needs: pip install pillow numpy
Keys:  L = lock largest face, U = unlock
"""

import queue
import sys
import threading
import tkinter as tk
from datetime import datetime
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import numpy as np
from PIL import Image, ImageTk

# --- Behavior ----------------------------------------------------------------
LIVE_WAVE_SECONDS = 10      # Preview shows the latest N seconds; stopped sessions show their recording.
WAVE_DISPLAY_GAIN = 3.0     # Vertical zoom for DISPLAY only. Stored samples are never changed.
VIDEO_POLL_MS = 20
SLOW_POLL_MS = 60           # Events, waveform, buttons.

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
WARN_COLOR = "#f5b942"

# Settings window: how check results are shown.
STATUS_LABELS = {"ok": "OK", "warn": "WARN", "fail": "FAIL", "skip": "--"}
STATUS_COLORS = {"ok": WAVE_COLOR, "warn": WARN_COLOR, "fail": LIVE_COLOR, "skip": MUTED}


class SettingsWindow(tk.Toplevel):
    """
    Settings: PC requirements and installed packages.
    This class only DISPLAYS results. The checks run inside the controller (and syscheck.py)
    on a worker thread, so the window stays responsive during slow checks.
    """

    def __init__(self, app: tk.Tk, controller):
        super().__init__(app)
        self.title("Settings")
        self.geometry("920x660")
        self.minsize(700, 500)
        self.configure(bg=BG)
        self.transient(app)

        self._controller = controller
        self._events = queue.Queue()                  # Worker thread -> window.
        self._results = []
        self._row_results = {}                        # Tree row id -> CheckResult.
        self._busy = False
        self._after_id = None

        self._build()
        self._start(deep=False)
        self._after_id = self.after(100, self._poll)

    # ------------------------------------------------------------------ layout

    def _build(self) -> None:
        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)

        header = tk.Frame(self, bg=BG)
        header.grid(row=0, column=0, sticky="ew", padx=12, pady=(12, 6))
        self.summary = tk.Label(header, text="Checking...", bg=BG, fg=TEXT, anchor="w",
                                font=("Segoe UI", 13, "bold"))
        self.summary.pack(fill="x")
        tk.Label(header, text=f"Python: {sys.executable}", bg=BG, fg=MUTED, anchor="w",
                 font=("Segoe UI", 9)).pack(fill="x")

        body = tk.Frame(self, bg=BG)
        body.grid(row=1, column=0, sticky="nsew", padx=12)
        body.columnconfigure(0, weight=1)
        body.rowconfigure(0, weight=1)
        self.tree = ttk.Treeview(body, columns=("status", "detail"), show="tree headings",
                                 selectmode="browse")
        self.tree.heading("#0", text="Check", anchor="w")
        self.tree.heading("status", text="Status")
        self.tree.heading("detail", text="Detail", anchor="w")
        self.tree.column("#0", width=270, stretch=False)
        self.tree.column("status", width=70, stretch=False, anchor="center")
        self.tree.column("detail", width=500)
        scrollbar = ttk.Scrollbar(body, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=scrollbar.set)
        self.tree.grid(row=0, column=0, sticky="nsew")
        scrollbar.grid(row=0, column=1, sticky="ns")
        for status, color in STATUS_COLORS.items():
            self.tree.tag_configure(status, foreground=color)
        self.tree.tag_configure("group", foreground=TEXT, font=("Segoe UI", 10, "bold"))
        self.tree.bind("<<TreeviewSelect>>", self._on_select)

        lower = tk.Frame(self, bg=BG)
        lower.grid(row=2, column=0, sticky="ew", padx=12, pady=(8, 0))
        lower.columnconfigure(0, weight=1)
        self.detail_box = self._readonly_text(lower, height=3)
        self.detail_box.grid(row=0, column=0, sticky="ew")
        tk.Label(lower, text="To install what is missing (PowerShell, one line each):",
                 bg=BG, fg=MUTED, anchor="w", font=("Segoe UI", 9)).grid(
                     row=1, column=0, sticky="w", pady=(8, 2))
        self.install_box = self._readonly_text(lower, height=4, mono=True)
        self.install_box.grid(row=2, column=0, sticky="ew")

        bar = ttk.Frame(self, style="Bar.TFrame")
        bar.grid(row=3, column=0, sticky="ew", padx=12, pady=12)
        self.recheck_button = ttk.Button(bar, text="Re-check", command=lambda: self._start(False))
        self.deep_button = ttk.Button(bar, text="Deep check (slow)", command=lambda: self._start(True))
        self.copy_button = ttk.Button(bar, text="Copy install commands", command=self._copy)
        close_button = ttk.Button(bar, text="Close", command=self.destroy)
        self.note = ttk.Label(bar, text="", style="Bar.TLabel")
        self.recheck_button.pack(side="left", padx=(6, 2), pady=4)
        self.deep_button.pack(side="left", padx=2, pady=4)
        self.copy_button.pack(side="left", padx=2, pady=4)
        close_button.pack(side="right", padx=(2, 6), pady=4)
        self.note.pack(side="right", padx=10)

    def _readonly_text(self, parent: tk.Misc, height: int, mono: bool = False) -> tk.Text:
        return tk.Text(parent, height=height, wrap="word", state="disabled", bg=PANEL, fg=TEXT,
                       relief="flat", padx=10, pady=6, highlightthickness=0,
                       font=("Consolas", 10) if mono else ("Segoe UI", 10))

    @staticmethod
    def _set_text(box: tk.Text, text: str) -> None:
        box.configure(state="normal")
        box.delete("1.0", "end")
        box.insert("1.0", text)
        box.configure(state="disabled")

    # ---------------------------------------------------------------- running

    def _start(self, deep: bool) -> None:
        if self._busy:
            return
        self._busy = True
        for button in (self.recheck_button, self.deep_button, self.copy_button):
            button.state(["disabled"])
        self.note.configure(text="")
        self.summary.configure(text="Checking...", fg=TEXT)
        threading.Thread(target=self._worker, args=(deep,), daemon=True).start()

    def _worker(self, deep: bool) -> None:
        """Runs on a worker thread. Talks to the window through the queue only."""
        try:
            results = self._controller.run_checks(
                deep=deep,
                progress=lambda message: self._events.put(("progress", message)),
            )
            self._events.put(("done", results))
        except Exception as error:  # Broad on purpose: show the reason instead of hanging.
            self._events.put(("error", f"{type(error).__name__}: {error}"))

    def _poll(self) -> None:
        while True:
            try:
                kind, payload = self._events.get_nowait()
            except queue.Empty:
                break
            if kind == "progress":
                self.summary.configure(text=payload, fg=TEXT)
            elif kind == "done":
                self._show(payload)
                self._finish()
            elif kind == "error":
                self.summary.configure(text="The check itself failed", fg=LIVE_COLOR)
                self._set_text(self.detail_box, payload)
                self._finish()
        self._after_id = self.after(100, self._poll)

    def _finish(self) -> None:
        self._busy = False
        for button in (self.recheck_button, self.deep_button, self.copy_button):
            button.state(["!disabled"])

    # ---------------------------------------------------------------- results

    def _show(self, results: list) -> None:
        self._results = results
        self._row_results = {}
        self.tree.delete(*self.tree.get_children())

        order = ["ok", "skip", "warn", "fail"]   # Worst status wins on a group row.
        groups = {}
        for result in results:
            if result.group not in groups:
                node = self.tree.insert("", "end", text=result.group, open=True, tags=("group",))
                groups[result.group] = [node, "ok"]
            group = groups[result.group]
            if order.index(result.status) > order.index(group[1]):
                group[1] = result.status
            row = self.tree.insert(group[0], "end", text=result.name,
                                   values=(STATUS_LABELS[result.status], result.detail),
                                   tags=(result.status,))
            self._row_results[row] = result
        for node, worst in groups.values():
            self.tree.set(node, "status", STATUS_LABELS[worst])

        problems, warnings = self._controller.summarize_checks(results)
        if problems:
            self.summary.configure(text=f"{problems} problem(s), {warnings} warning(s)", fg=LIVE_COLOR)
        elif warnings:
            self.summary.configure(text=f"No problems, {warnings} warning(s)", fg=WARN_COLOR)
        else:
            self.summary.configure(text="All checks passed", fg=WAVE_COLOR)

        commands = self._controller.install_commands(results)
        self._set_text(self.install_box, "\n".join(commands) if commands else "Nothing to install.")
        self._set_text(self.detail_box, "Select a row to see its full detail and suggested fix.")

    def _on_select(self, event=None) -> None:
        selection = self.tree.selection()
        result = self._row_results.get(selection[0]) if selection else None
        if result is None:
            return
        text = result.detail + (f"\nFix: {result.fix}" if result.fix else "")
        self._set_text(self.detail_box, text)

    def _copy(self) -> None:
        commands = self._controller.install_commands(self._results)
        if not commands:
            self.note.configure(text="Nothing to copy.")
            return
        self.clipboard_clear()
        self.clipboard_append("\n".join(commands))
        self.note.configure(text="Copied.")

    def destroy(self) -> None:
        if self._after_id is not None:
            try:
                self.after_cancel(self._after_id)
            except tk.TclError:
                pass
        super().destroy()


class App(tk.Tk):
    def __init__(self, controller):
        super().__init__()
        self.controller = controller
        self.title("AyosTa")
        self.geometry("1280x760")
        self.minsize(1000, 560)
        self.configure(bg=BG)

        self._video_image = None    # Tk drops images that nothing references.
        self._video_live = False    # A camera frame is currently shown.
        self._idle_label = None     # What the idle video panel currently says.
        self._wave_key = None       # Skips waveform redraws when nothing changed.
        self._button_key = None     # Skips button updates when nothing changed.
        self._settings_window = None

        self._build_style()
        self._build_layout()
        self._refresh_buttons()
        self._draw_video_idle()

        self.bind("<l>", lambda event: self.controller.lock_face())
        self.bind("<u>", lambda event: self.controller.unlock_face())
        self.protocol("WM_DELETE_WINDOW", self._on_close)

        self.after(VIDEO_POLL_MS, self._poll_video)
        self.after(SLOW_POLL_MS, self._poll_slow)
        # The controller preloads models and opens camera/mic in the background.
        # Session recording starts only when the user clicks Start Session.

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
        style.configure("Treeview", background=PANEL, fieldbackground=PANEL, foreground=TEXT,
                        borderwidth=0, rowheight=26, bordercolor=PANEL,
                        lightcolor=PANEL, darkcolor=PANEL)
        style.map("Treeview", background=[("selected", BUTTON_HOVER)],
                  foreground=[("selected", TEXT)])
        style.configure("Treeview.Heading", background=BAR, foreground=MUTED,
                        borderwidth=0, relief="flat")
        style.map("Treeview.Heading", background=[("active", BAR)])
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

        self.start_button = ttk.Button(toolbar, text="Start Session", command=self.controller.start_session)
        self.stop_button = ttk.Button(toolbar, text="Stop Session", command=self.controller.stop_session)
        self.lock_button = ttk.Button(toolbar, text="Lock face (L)", command=self.controller.lock_face)
        self.unlock_button = ttk.Button(toolbar, text="Unlock (U)", command=self.controller.unlock_face)
        self.speaking = tk.StringVar(value=self.controller.speaking)
        speaking_label = ttk.Label(toolbar, text="Speaking", style="Bar.TLabel")
        speaking_box = ttk.Combobox(toolbar, textvariable=self.speaking, state="readonly",
                                    values=list(self.controller.SPEAKING), width=9)
        speaking_box.bind("<<ComboboxSelected>>", self._on_speaking_changed)
        self.direction = tk.StringVar(value=next(iter(self.controller.DIRECTIONS)))
        direction_box = ttk.Combobox(toolbar, textvariable=self.direction, state="readonly",
                                     values=list(self.controller.DIRECTIONS), width=20)
        self.translate_button = ttk.Button(toolbar, text="Translate transcript",
                                           command=self._translate)
        self.save_audio_button = ttk.Button(toolbar, text="Save audio", command=self._save_audio)
        self.export_pdf_button = ttk.Button(toolbar, text="Export PDF", command=self._export_pdf)
        self.settings_button = ttk.Button(toolbar, text="Settings", command=self._open_settings)
        self.status = tk.StringVar(value="Starting...")
        status_label = ttk.Label(self, textvariable=self.status, style="Status.TLabel")
        status_label.grid(row=2, column=0, sticky="w", padx=10, pady=(0, 6))
        self._sync_direction()

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
        self.save_audio_button.pack(side="left", padx=2, pady=4)
        self.export_pdf_button.pack(side="left", padx=2, pady=4)
        self.settings_button.pack(side="right", padx=(2, 6), pady=4)

    def _refresh_buttons(self) -> None:
        controller = self.controller
        key = (controller.can_start_session, controller.session_active,
               controller.camera_on, controller.can_translate)
        if key == self._button_key:
            return
        self._button_key = key

        def set_enabled(button: ttk.Button, enabled: bool) -> None:
            button.state(["!disabled"] if enabled else ["disabled"])

        set_enabled(self.start_button, controller.can_start_session)
        set_enabled(self.stop_button, controller.session_active)
        set_enabled(self.lock_button, controller.camera_on)
        set_enabled(self.unlock_button, controller.camera_on)
        set_enabled(self.translate_button, controller.can_translate)

    # ----------------------------------------------------------------- actions

    def _sync_direction(self) -> None:
        language = self.speaking.get()
        other = "Tagalog" if language == "Cebuano" else "Cebuano"
        self.direction.set(f"{language} to {other}")  # Translate what was spoken.

    def _on_speaking_changed(self, event=None) -> None:
        self.controller.set_speaking(self.speaking.get())
        self._sync_direction()

    def _translate(self) -> None:
        if not self.controller.translate(self.direction.get()):
            self.status.set("Nothing to translate.")

    def _open_settings(self) -> None:
        window = self._settings_window
        if window is not None and window.winfo_exists():
            window.lift()
            window.focus_force()
            return
        self._settings_window = SettingsWindow(self, self.controller)

    def _on_close(self) -> None:
        self.controller.shutdown()
        self.destroy()

    # --------------------------------------------------------------- exports

    def _save_audio(self) -> None:
        """Ask where to save, then let the controller write the whole recording as a WAV file."""
        if not self.controller.has_audio:
            messagebox.showinfo("No recording", "There is no recording yet.")
            return
        path = filedialog.asksaveasfilename(
            title="Save full recording",
            defaultextension=".wav",
            filetypes=[("WAV audio", "*.wav")],
            initialfile=f"ayosta_recording_{datetime.now():%Y%m%d_%H%M%S}.wav",
        )
        if not path:
            return
        try:
            self.controller.save_audio(path)
        except Exception as error:
            messagebox.showerror("Could not save audio", f"{type(error).__name__}: {error}")
            return
        self.status.set(f"Audio saved: {Path(path).name}")
        messagebox.showinfo("Audio saved", f"Full recording saved to:\n{path}")

    def _export_pdf(self) -> None:
        """Ask where to save, then let the controller write the transcript shown on screen as a PDF."""
        transcript_text = self.transcript.get("1.0", "end-1c").strip()
        if not transcript_text:
            messagebox.showinfo("No transcript", "There is no transcript to export yet.")
            return
        path = filedialog.asksaveasfilename(
            title="Export meeting transcript",
            defaultextension=".pdf",
            filetypes=[("PDF document", "*.pdf")],
            initialfile=f"ayosta_transcript_{datetime.now():%Y%m%d_%H%M%S}.pdf",
        )
        if not path:
            return
        try:
            self.controller.export_pdf(path, transcript_text)
        except ImportError:
            messagebox.showerror(
                "PDF package missing",
                "Install the PDF dependency in your project environment:\n\n"
                "py -m pip install reportlab",
            )
            return
        except Exception as error:
            messagebox.showerror("Could not export PDF", f"{type(error).__name__}: {error}")
            return
        self.status.set(f"PDF exported: {Path(path).name}")
        messagebox.showinfo("PDF exported", f"Transcript PDF saved to:\n{path}")

    # ------------------------------------------------------------- controller events

    def _handle_event(self, kind: str, payload) -> None:
        if kind == "status":
            self.status.set(payload)
        elif kind == "words":
            self._append_transcript(" ".join(payload) + " ")
        elif kind == "transcript_cleared":
            self._clear_transcript()
            self._wave_key = None
        elif kind == "session":
            self._wave_key = None
        elif kind == "problems":
            messagebox.showwarning("Some parts could not start", "\n\n".join(payload))
        elif kind == "error":
            label, message = payload
            self.status.set(f"{label} problem: {message}")
            messagebox.showerror(f"{label} problem", message)
        elif kind == "translation_progress":
            done, total = payload
            self.status.set(f"Translating: part {done} of {total}")
        elif kind == "translation":
            result, source, target = payload
            self._append_transcript(f"\n{source} to {target}\n", "divider")
            self._append_transcript(result + "\n", "translation")
            self.status.set("Translation finished.")
        elif kind == "translation_error":
            self.status.set("Translation failed.")
            messagebox.showerror("Translation failed", payload)

    def _clear_transcript(self) -> None:
        self.transcript.configure(state="normal")
        self.transcript.delete("1.0", "end")
        self.transcript.configure(state="disabled")

    def _append_transcript(self, text: str, tag: str = "") -> None:
        self.transcript.configure(state="normal")
        self.transcript.insert("end", text, tag)
        self.transcript.see("end")
        self.transcript.configure(state="disabled")

    def _poll_slow(self) -> None:
        for kind, payload in self.controller.update():
            self._handle_event(kind, payload)
        self._refresh_buttons()
        self._redraw_wave()
        self.after(SLOW_POLL_MS, self._poll_slow)

    # ------------------------------------------------------------------- video

    def _poll_video(self) -> None:
        frame = self.controller.poll_frame()
        if frame is not None:
            self._show_frame(frame)
            self._video_live = True
        elif self._video_live and not self.controller.camera_on:
            self._video_live = False
            self._draw_video_idle()
        elif not self._video_live and self._idle_label != self._idle_text():
            self._draw_video_idle()
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
        if not self._video_live:
            self._draw_video_idle()

    def _idle_text(self) -> str:
        return "Starting camera..." if self.controller.camera_on else "Video"

    def _draw_video_idle(self) -> None:
        canvas = self.video_canvas
        canvas.delete("all")
        self._idle_label = self._idle_text()
        canvas.create_text(canvas.winfo_width() // 2, canvas.winfo_height() // 2,
                           text=self._idle_label, fill=MUTED, font=("Segoe UI", 12))

    # ---------------------------------------------------------------- waveform

    def _redraw_wave(self) -> None:
        """
        Editor-style waveform: one min/max pair per pixel column, drawn as a filled
        envelope. While recording, show recent session audio; while idle, show the last
        recording if available, otherwise the rolling microphone preview.
        TODO(team): zoom, scroll, selection, time ruler. The data side (audio.py)
        already supports any range through AudioBuffer.peaks(start, end, columns).
        """
        canvas = self.wave_canvas
        width, height = canvas.winfo_width(), canvas.winfo_height()
        if width < 8 or height < 8:
            return

        total, sample_rate, recording = self.controller.audio_state()
        # Preview buffers are rolling and their length eventually stops growing, so
        # include a small time bucket to keep the idle live waveform refreshing.
        key = (width, height, total, recording, int(datetime.now().timestamp() * 4))
        if key == self._wave_key:
            return  # Nothing changed, so skip the work (matters for long stopped recordings).
        self._wave_key = key

        canvas.delete("all")
        middle = height // 2
        canvas.create_line(0, middle, width, middle, fill=WAVE_CENTER_LINE)

        if total == 0:
            label = "Recording audio..." if recording else "Waiting for microphone preview..."
            canvas.create_text(width // 2, middle - 14, text=label, fill=MUTED,
                               font=("Segoe UI", 12))
            return

        window = int(sample_rate * LIVE_WAVE_SECONDS)
        if recording:
            start = max(0, total - window)
            span = max(2, int(width * min(total, window) / window))
        else:
            start = 0
            span = width

        mins, maxs = self.controller.audio_peaks(start, total, span)
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

        seconds = total / sample_rate
        text = f"RECORDING  {seconds:.1f}s" if recording else f"Preview / last recording  {seconds:.1f}s"
        canvas.create_text(10, 8, anchor="nw", text=text, font=("Segoe UI", 9),
                           fill=LIVE_COLOR if recording else MUTED)