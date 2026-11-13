"""
main.py - AyosTa GUI (Tkinter). ROUGH DRAFT, untested with a real camera and mic.

Screens and flow follow AyosTa_Prototype.html:

    Dashboard        today's numbers and the latest cases
    Case records     search and filter every saved case
    Case details     new hearing form + consent from both parties (required)
    Recording        live video (one camera sees both parties), waveform, transcript
    Assessment       the transcript, case status, save

    Dashboard -> Case details -> Recording -> Assessment -> Dashboard

This file holds NO detection, audio, speech or translation logic. It starts worker
objects from the other modules, polls their queues, and draws what they return:

    face.py      FaceWorker      live video frames         (starts when recording begins)
    audio.py     AudioRecorder   live mic + waveform data  (starts when recording begins)
    speech.py    SpeechWorker    live words for the transcript
    language.py  translate_long  runs AFTER the session, on the Translate button
    syscheck.py  run_all_checks  Settings window: PC requirements and installed packages

Every module is imported lazily and started independently, so a missing package or
device (no mic, no weights file) disables that panel only, not the whole app.
Worker threads never touch Tk. They talk to the GUI through queues only.

Saved cases: cases.json next to this file. It holds names and transcripts, so keep it
out of Git and out of shared folders. Audio and video are never written to disk.

Admin: changing a case between Open, Closed and To court needs the admin password. It is
created the first time it is needed and stored only as a salted hash in admin.json (keep that
file out of Git too). Every status change on a saved case is logged inside its record.

The original dark single-window layout is kept in main_classic.py.

Needs: pip install pillow numpy   (pillow arrives with ultralytics)
Keys:  L = lock largest face, U = unlock (Recording screen only)
"""

import hashlib
import hmac
import json
import os
import queue
import secrets
import sys
import threading
import time
import tkinter as tk
from datetime import datetime
from pathlib import Path
from tkinter import messagebox, ttk

import numpy as np
from PIL import Image, ImageTk

import syscheck  # Standard library only, so importing it is always safe.

# --- Behavior ----------------------------------------------------------------
LIVE_WAVE_SECONDS = 10      # Live view shows the last N seconds. After Stop, the full recording.
WAVE_DISPLAY_GAIN = 3.0     # Vertical zoom for DISPLAY only. Stored samples are never changed.
VIDEO_POLL_MS = 20
SLOW_POLL_MS = 60           # Waveform, transcript, errors, translation results, timer.
TOAST_MS = 2200
DIRECTIONS = {
    "Cebuano to Tagalog": ("Cebuano", "Tagalog"),
    "Tagalog to Cebuano": ("Tagalog", "Cebuano"),
}
SPEAKING = {"Cebuano": "ceb", "Tagalog": "tgl"}   # Language spoken in the session -> MMS code.
SPEAKERS = ("Complainant", "Respondent")
BARANGAY_TAG = "BRGY. LA FILIPINA  ·  LUPON TAGAPAMAYAPA"
CASES_FILE = Path(__file__).resolve().parent / "cases.json"
ADMIN_FILE = Path(__file__).resolve().parent / "admin.json"

# One camera sees both parties. Change the number if your webcam is not the first camera.
CAMERA_INDEX = 0

# Admin password.
ADMIN_MIN_PASSWORD = 8
ADMIN_UNLOCK_SECONDS = 300      # Admin stays unlocked this long after a correct password.
ADMIN_MAX_ATTEMPTS = 5          # Wrong passwords in a row before a short lockout.
ADMIN_LOCKOUT_SECONDS = 60
PBKDF2_ITERATIONS = 600_000

# Case status key -> (label, text color, background color).
STATUS_BADGES = {
    "open": ("Open", "#8A6A26", "#F6EDD9"),
    "closed": ("Closed", "#3E5533", "#E7EEE0"),
    "court": ("To court", "#7A382C", "#F5E4DF"),
}

# Steps shown while a hearing is in progress: screen key -> label.
FLOW_STEPS = {"setup": "Case details", "recording": "Recording", "report": "Assessment"}

# --- Colors ------------------------------------------------------------------
# Palette: forest green 38422B and cream F5F1E8 (MOSS and CREAM). The rest are the
# tints used in AyosTa_Prototype.html.
INK = "#232A1B"
MOSS = "#38422B"
MOSS_LIGHT = "#4C5A3B"
MOSS_PALE = "#EDEADD"
CREAM = "#F5F1E8"
PAPER = "#FCFAF4"
LINE = "#DEDACB"
GOLD = "#B8933F"
GOLD_BG = "#F6EDD9"
GOLD_LINE = "#E3CE9C"
GOLD_TEXT = "#7A5E22"
RUST = "#A24B3B"
SAGE = "#5F7A52"
DISABLED_BG = "#DAD6C6"
DISABLED_FG = "#A9A491"
VIDEO_BG = INK
VIDEO_TEXT = "#9A9682"

# Names used by SettingsWindow and the drawing code below.
BG = CREAM
PANEL = PAPER
BAR = MOSS_PALE
BUTTON = LINE
BUTTON_HOVER = MOSS_PALE
TEXT = INK
MUTED = "#7A7864"
WAVE_COLOR = "#3E6B35"
WAVE_CENTER_LINE = LINE
LIVE_COLOR = RUST
WARN_COLOR = "#8A6A26"

# Settings window: how check results are shown.
STATUS_LABELS = {"ok": "OK", "warn": "WARN", "fail": "FAIL", "skip": "--"}
STATUS_COLORS = {"ok": WAVE_COLOR, "warn": WARN_COLOR, "fail": LIVE_COLOR, "skip": MUTED}

# --- Fonts -------------------------------------------------------------------
FONT = ("Segoe UI", 10)
FONT_MEDIUM = ("Segoe UI", 10, "bold")
FONT_SMALL = ("Segoe UI", 9)
FONT_MONO = ("Consolas", 10)
FONT_MONO_SMALL = ("Consolas", 9)
FONT_WORDMARK = ("Georgia", 17, "bold")
FONT_TITLE = ("Georgia", 20, "bold")
FONT_STAT = ("Georgia", 22, "bold")


# --- Saved cases -------------------------------------------------------------

def load_cases(path: Path = CASES_FILE):
    """
    Returns (cases, warning). A missing file is normal (no cases yet). A damaged file
    is set aside as cases.damaged.json instead of being overwritten, so nothing is lost.
    """
    if not path.exists():
        return [], ""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            raise ValueError("expected a list of cases")
        cases, seen = [], set()
        for case in data:
            valid = isinstance(case, dict) and case.get("id") and case.get("status") in STATUS_BADGES
            if valid and case["id"] not in seen:   # A case id must be unique: it identifies the row.
                seen.add(case["id"])
                cases.append(case)
        return cases, ""
    except (OSError, ValueError) as error:
        backup = path.with_name(f"{path.stem}.damaged{path.suffix}")
        try:
            os.replace(path, backup)
        except OSError:
            pass
        return [], f"Could not read {path.name} ({error}). It was set aside as {backup.name}."


def save_cases(cases: list, path: Path = CASES_FILE) -> None:
    """Writes to a temporary file first, so a crash mid-write cannot corrupt the records."""
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(cases, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(temporary, path)


def _write_json_atomically(path: Path, data) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(temporary, path)


def _hash_password(password: str, salt: bytes, iterations: int) -> bytes:
    return hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iterations)


def read_admin_record(path: Path = ADMIN_FILE):
    """Returns (salt, expected_hash, iterations), or None when the file is missing or damaged."""
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
        return bytes.fromhex(record["salt"]), bytes.fromhex(record["hash"]), int(record["iterations"])
    except (OSError, ValueError, KeyError, TypeError):
        return None


def admin_state(path: Path = ADMIN_FILE) -> str:
    """'missing' (no password yet), 'ok', or 'damaged' (unreadable, so nobody can be verified)."""
    if not path.exists():
        return "missing"
    return "ok" if read_admin_record(path) is not None else "damaged"


def set_admin_password(password: str, path: Path = ADMIN_FILE) -> None:
    """Stores a salted PBKDF2 hash. The password itself is never written anywhere."""
    salt = secrets.token_bytes(16)
    _write_json_atomically(path, {
        "algorithm": "pbkdf2_sha256",
        "iterations": PBKDF2_ITERATIONS,
        "salt": salt.hex(),
        "hash": _hash_password(password, salt, PBKDF2_ITERATIONS).hex(),
    })


def check_admin_password(password: str, path: Path = ADMIN_FILE) -> bool:
    record = read_admin_record(path)
    if record is None:
        return False
    salt, expected, iterations = record
    return hmac.compare_digest(_hash_password(password, salt, iterations), expected)


def format_clock(seconds: float) -> str:
    seconds = max(0, int(seconds))
    return f"{seconds // 60:02d}:{seconds % 60:02d}"


# --- Small widget helpers ----------------------------------------------------

class Screen(tk.Frame):
    """
    One page of the app. Its card is centered and never wider than max_width pixels.
    Put content in screen.card. fill_height makes the card use the full page height.
    """

    def __init__(self, parent: tk.Misc, max_width: int, fill_height: bool = False):
        super().__init__(parent, bg=BG)
        self._max_width = max_width
        self.card = tk.Frame(self, bg=PAPER, highlightbackground=LINE, highlightcolor=LINE,
                             highlightthickness=1, padx=30, pady=18)
        if fill_height:
            self.card.pack(fill="both", expand=True, pady=(14, 12), padx=16)
        else:
            self.card.pack(fill="x", anchor="n", pady=(14, 12), padx=16)
        self.bind("<Configure>", self._center)

    def _center(self, event: tk.Event) -> None:
        side = max(16, (event.width - self._max_width) // 2)
        self.card.pack_configure(padx=side)


def make_label(parent: tk.Misc, text: str = "", fg: str = INK, font=FONT, bg: str = PAPER,
               **options) -> tk.Label:
    return tk.Label(parent, text=text, fg=fg, bg=bg, font=font, **options)


def make_wrapping_label(parent: tk.Misc, text: str = "", fg: str = INK, font=FONT,
                        bg: str = PAPER) -> tk.Label:
    """A label that re-wraps its text to the width it is given."""
    label = tk.Label(parent, text=text, fg=fg, bg=bg, font=font, anchor="w", justify="left")
    label.bind("<Configure>", lambda event: label.configure(wraplength=max(120, event.width - 4)))
    return label


def make_entry(parent: tk.Misc, variable: tk.StringVar) -> tk.Entry:
    return tk.Entry(parent, textvariable=variable, font=("Segoe UI", 11), bg=CREAM, fg=INK,
                    insertbackground=INK, relief="flat", bd=0, highlightthickness=1,
                    highlightbackground=LINE, highlightcolor=MOSS_LIGHT)


def make_badge(parent: tk.Misc, status: str) -> tk.Label:
    label, fg, bg = STATUS_BADGES[status]
    return tk.Label(parent, text=f"● {label}", fg=fg, bg=bg, font=("Segoe UI", 9, "bold"),
                    padx=10, pady=2)


class SettingsWindow(tk.Toplevel):
    """
    Settings: PC requirements and installed packages.
    This class only DISPLAYS results. All checking logic lives in syscheck.py and runs on a
    worker thread, so the window stays responsive during slow checks.
    """

    def __init__(self, app: tk.Tk, is_session_active):
        super().__init__(app)
        self.title("Settings")
        self.geometry("920x660")
        self.minsize(700, 500)
        self.configure(bg=BG)
        self.transient(app)

        self._is_session_active = is_session_active   # Callable. Camera test is skipped if True.
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
            results = syscheck.run_all_checks(
                project_dir=Path(__file__).resolve().parent,
                deep=deep,
                session_active=self._is_session_active(),
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

        problems, warnings = syscheck.summarize(results)
        if problems:
            self.summary.configure(text=f"{problems} problem(s), {warnings} warning(s)", fg=LIVE_COLOR)
        elif warnings:
            self.summary.configure(text=f"No problems, {warnings} warning(s)", fg=WARN_COLOR)
        else:
            self.summary.configure(text="All checks passed", fg=WAVE_COLOR)

        commands = syscheck.install_commands(results)
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
        commands = syscheck.install_commands(self._results)
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


class AdminDialog(tk.Toplevel):
    """
    Modal password prompt. `result` is True only when the admin password was accepted (or
    created). `submit(password, confirm)` does the checking and returns an error message,
    or None on success, so this class holds no security logic.
    """

    def __init__(self, parent: tk.Misc, creating: bool, submit):
        super().__init__(parent)
        self.title("Administrator")
        self.configure(bg=PAPER)
        self.resizable(False, False)
        self.transient(parent)
        self.result = False
        self._submit = submit
        self._creating = creating

        body = tk.Frame(self, bg=PAPER, padx=26, pady=22)
        body.pack()
        heading = "Create the admin password" if creating else "Administrator password"
        make_label(body, heading, fg=MOSS, font=("Georgia", 14, "bold"), anchor="w").pack(fill="x")
        note = ("Only the admin can change a case between Open, Closed and To court. Use at least "
                f"{ADMIN_MIN_PASSWORD} characters. It is stored as a salted hash, never as plain text."
                if creating else "Enter the admin password to change case status.")
        tk.Label(body, text=note, fg=MUTED, bg=PAPER, font=FONT_SMALL, wraplength=340,
                 justify="left", anchor="w").pack(fill="x", pady=(6, 12))

        self.password = tk.StringVar()
        self.confirm = tk.StringVar()
        make_label(body, "Password", fg=MOSS_LIGHT, font=FONT_SMALL, anchor="w").pack(fill="x", pady=(0, 4))
        self.password_entry = make_entry(body, self.password)
        self.password_entry.configure(show="*")
        self.password_entry.pack(fill="x", ipady=6)
        if creating:
            make_label(body, "Type it again", fg=MOSS_LIGHT, font=FONT_SMALL, anchor="w").pack(
                fill="x", pady=(10, 4))
            confirm_entry = make_entry(body, self.confirm)
            confirm_entry.configure(show="*")
            confirm_entry.pack(fill="x", ipady=6)

        self.error = tk.Label(body, text="", fg=RUST, bg=PAPER, font=FONT_SMALL, wraplength=340,
                              justify="left", anchor="w")
        self.error.pack(fill="x", pady=(10, 0))

        buttons = tk.Frame(body, bg=PAPER)
        buttons.pack(fill="x", pady=(14, 0))
        ttk.Button(buttons, text="Create password" if creating else "Unlock", command=self._on_submit).pack(
            side="right")
        ttk.Button(buttons, text="Cancel", style="Ghost.TButton", command=self.destroy).pack(
            side="right", padx=(0, 8))

        self.bind("<Return>", lambda event: self._on_submit())
        self.bind("<Escape>", lambda event: self.destroy())
        self.update_idletasks()
        x = parent.winfo_rootx() + (parent.winfo_width() - self.winfo_width()) // 2
        y = parent.winfo_rooty() + (parent.winfo_height() - self.winfo_height()) // 3
        self.geometry(f"+{max(0, x)}+{max(0, y)}")
        try:
            self.wait_visibility()
            self.grab_set()   # Modal: the main window ignores input until this closes.
        except tk.TclError:
            pass              # Some window managers refuse the grab. The dialog still works.
        self.password_entry.focus_set()

    def _on_submit(self) -> None:
        message = self._submit(self.password.get(), self.confirm.get())
        if message:
            self.error.configure(text=message)
            self.password.set("")
            self.confirm.set("")
            self.password_entry.focus_set()
            return
        self.result = True
        self.destroy()


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("AyosTa")
        self.geometry(f"1200x{min(800, max(620, self.winfo_screenheight() - 100))}")
        self.minsize(980, 600)
        self.configure(bg=BG)

        # Session parts. Each is None when it is not running or failed to start.
        self.face_worker = None
        self.recorder = None        # Kept after Stop so the full waveform stays visible.
        self.speech_worker = None
        self.session_active = False
        self.translating = False

        # The hearing in progress.
        self.speaker = SPEAKERS[0]
        self._hearing = {}          # Case details typed on the Case details screen.
        self._segments = []         # [{"speaker": str, "words": [str]}] in speaking order.
        self._words = []            # All live words. Translation reads this, not the widget.
        self._translation_text = ""
        self._video_image = None    # Tk drops images that nothing references.
        self._wave_key = None       # Skips redraws when nothing changed.
        self._reported_errors = set()
        self._last_speech_state = None
        self._settings_window = None
        self._translation_queue = queue.Queue()
        self._toast_after = None
        self.current_screen = "dashboard"
        self._selected_case_id = None   # Case shown in the Case records detail panel.
        self._admin_until = 0.0         # time.monotonic() until which admin stays unlocked.
        self._admin_failures = 0
        self._admin_locked_until = 0.0

        self.cases, load_warning = load_cases()

        self._build_style()
        self._build_layout()
        self._show_screen("dashboard")
        self._refresh_dashboard()
        self._refresh_records()
        self._refresh_buttons()
        self._draw_video_idle()
        self.status.set(load_warning or "Ready")

        self.bind("<l>", lambda event: self._on_face_key(event, lock=True))
        self.bind("<u>", lambda event: self._on_face_key(event, lock=False))
        self.protocol("WM_DELETE_WINDOW", self._on_close)

        self.after(VIDEO_POLL_MS, self._poll_video)
        self.after(SLOW_POLL_MS, self._poll_slow)

    # ------------------------------------------------------------------- style

    def _build_style(self) -> None:
        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure("Bar.TFrame", background=BAR)
        style.configure("Bar.TLabel", background=BAR, foreground=MUTED)
        style.configure("TButton", background=MOSS, foreground=PAPER, borderwidth=0,
                        focusthickness=0, padding=(14, 8), font=FONT_MEDIUM,
                        lightcolor=MOSS, darkcolor=MOSS, bordercolor=MOSS)
        style.map("TButton",
                  background=[("disabled", DISABLED_BG), ("active", MOSS_LIGHT)],
                  foreground=[("disabled", DISABLED_FG)],
                  bordercolor=[("disabled", DISABLED_BG)],
                  lightcolor=[("disabled", DISABLED_BG)], darkcolor=[("disabled", DISABLED_BG)])
        style.configure("Ghost.TButton", background=PAPER, foreground=MOSS, borderwidth=1,
                        focusthickness=0, padding=(14, 8), font=FONT_MEDIUM,
                        bordercolor=LINE, lightcolor=PAPER, darkcolor=PAPER)
        style.map("Ghost.TButton",
                  background=[("disabled", PAPER), ("active", MOSS_PALE)],
                  foreground=[("disabled", DISABLED_FG)])
        style.configure("TCombobox", fieldbackground=CREAM, background=CREAM, foreground=INK,
                        arrowcolor=MOSS, borderwidth=1, bordercolor=LINE,
                        lightcolor=CREAM, darkcolor=CREAM, selectbackground=CREAM,
                        selectforeground=INK, padding=4)
        style.map("TCombobox", fieldbackground=[("readonly", CREAM)],
                  foreground=[("readonly", INK)],
                  selectbackground=[("readonly", CREAM)],
                  selectforeground=[("readonly", INK)])
        style.configure("Vertical.TScrollbar", background=LINE, troughcolor=CREAM,
                        bordercolor=CREAM, lightcolor=LINE, darkcolor=LINE,
                        borderwidth=0, arrowcolor=MOSS)
        style.map("Vertical.TScrollbar", background=[("active", MOSS_PALE)])
        style.configure("TSeparator", background=LINE)
        style.configure("Status.TLabel", background=BG, foreground=MUTED, font=FONT_SMALL)
        style.configure("Treeview", background=PAPER, fieldbackground=PAPER, foreground=INK,
                        borderwidth=0, rowheight=30, bordercolor=PAPER,
                        lightcolor=PAPER, darkcolor=PAPER, font=FONT)
        style.map("Treeview", background=[("selected", MOSS_PALE)],
                  foreground=[("selected", INK)])
        style.configure("Treeview.Heading", background=CREAM, foreground=MUTED, relief="flat",
                        font=FONT_SMALL, padding=(6, 6))
        style.map("Treeview.Heading", background=[("active", CREAM)])
        self.option_add("*TCombobox*Listbox.background", PAPER)
        self.option_add("*TCombobox*Listbox.foreground", INK)
        self.option_add("*TCombobox*Listbox.selectBackground", MOSS)
        self.option_add("*TCombobox*Listbox.selectForeground", PAPER)
        self.option_add("*TCombobox*Listbox.font", FONT)

    # ------------------------------------------------------------------ layout

    def _build_layout(self) -> None:
        self.columnconfigure(0, weight=1)
        self.rowconfigure(2, weight=1)

        self._build_topbar()
        self._build_flow_header()

        self.stage = tk.Frame(self, bg=BG)
        self.stage.grid(row=2, column=0, sticky="nsew")
        self.stage.rowconfigure(0, weight=1)
        self.stage.columnconfigure(0, weight=1)

        self.screens = {
            "dashboard": self._build_dashboard(),
            "records": self._build_records(),
            "setup": self._build_setup(),
            "recording": self._build_recording(),
            "report": self._build_report(),
        }
        for screen in self.screens.values():
            screen.grid(row=0, column=0, sticky="nsew")

        self.status = tk.StringVar(value="Ready")
        ttk.Label(self, textvariable=self.status, style="Status.TLabel").grid(
            row=3, column=0, sticky="w", padx=14, pady=(0, 8))

        self.toast = tk.Label(self, text="", bg=MOSS, fg=PAPER, font=FONT_MEDIUM, padx=20, pady=10)

    def _build_topbar(self) -> None:
        bar = tk.Frame(self, bg=BG)
        bar.grid(row=0, column=0, sticky="ew")
        inner = tk.Frame(bar, bg=BG)
        inner.pack(fill="x", padx=28, pady=12)

        mark = tk.Canvas(inner, width=24, height=24, bg=BG, highlightthickness=0)
        mark.create_rectangle(10, 1, 14, 21, fill=MOSS, outline=MOSS)
        mark.create_rectangle(1, 10, 21, 14, fill=MOSS, outline=MOSS)
        mark.pack(side="left")
        tk.Label(inner, text="AyosTa", bg=BG, fg=MOSS, font=FONT_WORDMARK).pack(side="left", padx=(8, 0))

        tk.Label(inner, text=BARANGAY_TAG, bg=BG, fg=MOSS_LIGHT, font=FONT_MONO_SMALL).pack(side="right")
        tabs = tk.Frame(inner, bg=BG)
        tabs.pack(side="right", padx=20)
        self.tab_buttons = {}
        for key, text, command in (
            ("dashboard", "Dashboard", lambda: self._go_tab("dashboard")),
            ("records", "Case records", lambda: self._go_tab("records")),
            ("settings", "Settings", self._open_settings),
            ("admin", "Admin: locked", self._toggle_admin),
        ):
            button = tk.Button(tabs, text=text, command=command, relief="flat", bd=0, padx=14,
                               pady=6, font=FONT_MEDIUM, cursor="hand2", bg=BG, fg=MOSS_LIGHT,
                               activebackground=MOSS_PALE, activeforeground=MOSS)
            button.pack(side="left", padx=2)
            self.tab_buttons[key] = button
        tk.Frame(bar, height=1, bg=LINE).pack(fill="x")

    def _build_flow_header(self) -> None:
        """Stepper shown only while a hearing is in progress."""
        self.flow_header = tk.Frame(self, bg=BG)
        inner = tk.Frame(self.flow_header, bg=BG)
        inner.pack(fill="x", padx=28, pady=(14, 0))
        steps = tk.Frame(inner, bg=BG)
        steps.pack(side="left")
        self._step_widgets = {}
        for index, (key, text) in enumerate(FLOW_STEPS.items()):
            if index:
                tk.Frame(steps, width=20, height=1, bg=LINE).pack(side="left", padx=8)
            dot = tk.Label(steps, text="●", bg=BG, fg=LINE, font=("Segoe UI", 8))
            dot.pack(side="left")
            name = tk.Label(steps, text=text, bg=BG, fg="#B0AB96", font=FONT_MEDIUM)
            name.pack(side="left", padx=(6, 0))
            self._step_widgets[key] = (dot, name)
        tk.Button(inner, text="Cancel and return to dashboard", command=self._cancel_flow,
                  relief="flat", bd=0, bg=BG, fg=MUTED, activebackground=BG, activeforeground=RUST,
                  font=FONT_SMALL, cursor="hand2").pack(side="right")

    def _update_stepper(self, current: str) -> None:
        order = list(FLOW_STEPS)
        for key, (dot, name) in self._step_widgets.items():
            if order.index(key) < order.index(current):
                dot.configure(fg=SAGE)
                name.configure(fg=MOSS_LIGHT)
            elif key == current:
                dot.configure(fg=MOSS)
                name.configure(fg=MOSS)
            else:
                dot.configure(fg=LINE)
                name.configure(fg="#B0AB96")

    # -------------------------------------------------------------- navigation

    def _show_screen(self, name: str) -> None:
        self.current_screen = name
        for key, screen in self.screens.items():
            if key == name:
                screen.grid()
            else:
                screen.grid_remove()
        if name in FLOW_STEPS:
            self.flow_header.grid(row=1, column=0, sticky="ew")
            self._update_stepper(name)
        else:
            self.flow_header.grid_remove()
        for key, button in self.tab_buttons.items():
            active = key == name
            button.configure(bg=MOSS if active else BG, fg=PAPER if active else MOSS_LIGHT,
                             activebackground=MOSS_LIGHT if active else MOSS_PALE,
                             activeforeground=PAPER if active else MOSS)
        self._refresh_admin_button()

    def _go_tab(self, name: str) -> None:
        if self.current_screen in FLOW_STEPS and not self._confirm_discard():
            return
        if name == "records":
            self._refresh_records()
        self._show_screen(name)

    def _cancel_flow(self) -> None:
        if self._confirm_discard():
            self._show_screen("dashboard")

    def _confirm_discard(self) -> bool:
        """Leaving a hearing throws away its recording and transcript. Ask before doing it."""
        if self.current_screen == "setup":
            return True
        if not messagebox.askyesno(
                "Discard this hearing?",
                "The recording and transcript of this hearing will not be saved.\n\nDiscard it?"):
            return False
        self._discard_session()
        return True

    def _show_toast(self, message: str) -> None:
        self.toast.configure(text=message)
        self.toast.place(relx=0.5, rely=1.0, y=-44, anchor="s")
        self.toast.lift()
        if self._toast_after is not None:
            self.after_cancel(self._toast_after)
        self._toast_after = self.after(TOAST_MS, self.toast.place_forget)

    # --------------------------------------------------------------- dashboard

    def _build_dashboard(self) -> Screen:
        screen = Screen(self.stage, max_width=640)
        card = screen.card
        self.greeting = make_label(card, "", fg=MOSS_LIGHT, font=FONT_MONO_SMALL, anchor="w")
        self.greeting.pack(fill="x")
        make_label(card, "Today at a glance", fg=MOSS, font=FONT_TITLE, anchor="w").pack(
            fill="x", pady=(2, 18))

        tiles = tk.Frame(card, bg=PAPER)
        tiles.pack(fill="x")
        tiles.columnconfigure((0, 1, 2), weight=1, uniform="tiles")
        self.stat_labels = {}
        for column, (key, caption) in enumerate((
                ("hearings", "Hearings today"), ("open", "Open cases"), ("total", "Total cases"))):
            tile = tk.Frame(tiles, bg=CREAM, highlightbackground=LINE, highlightthickness=1,
                            padx=14, pady=12)
            tile.grid(row=0, column=column, sticky="ew", padx=(0 if column == 0 else 5, 0 if column == 2 else 5))
            number = make_label(tile, "0", fg=MOSS, font=FONT_STAT, bg=CREAM, anchor="w")
            number.pack(fill="x")
            make_label(tile, caption, fg=MOSS_LIGHT, font=FONT_SMALL, bg=CREAM, anchor="w").pack(fill="x")
            self.stat_labels[key] = number

        make_label(card, "Recent case activity (click a case to read its transcript)", fg=MOSS_LIGHT,
                   font=FONT_MEDIUM, anchor="w").pack(
            fill="x", pady=(22, 6))
        self.recent_frame = tk.Frame(card, bg=PAPER)
        self.recent_frame.pack(fill="x")

        ttk.Button(card, text="+ Start new hearing session", command=self._start_new_hearing).pack(
            fill="x", pady=(22, 0))
        return screen

    def _refresh_dashboard(self) -> None:
        hour = datetime.now().hour
        part = "morning" if hour < 12 else "afternoon" if hour < 18 else "evening"
        self.greeting.configure(text=f"Good {part}")

        today = datetime.now().strftime("%Y-%m-%d")
        self.stat_labels["hearings"].configure(
            text=str(sum(1 for case in self.cases if case.get("iso_date") == today)))
        self.stat_labels["open"].configure(
            text=str(sum(1 for case in self.cases if case["status"] == "open")))
        self.stat_labels["total"].configure(text=str(len(self.cases)))

        for child in self.recent_frame.winfo_children():
            child.destroy()
        if not self.cases:
            make_label(self.recent_frame, "No hearings recorded yet.", fg=MUTED, anchor="w").pack(
                fill="x", pady=10)
            return
        for case in self.cases[:3]:
            row = tk.Frame(self.recent_frame, bg=PAPER)
            row.pack(fill="x", pady=8)
            id_label = make_label(row, case["id"], fg=MOSS, font=FONT_MONO_SMALL, width=17, anchor="w")
            id_label.pack(side="left")
            parties_label = make_label(row, case.get("parties", ""), font=FONT_MEDIUM, anchor="w")
            parties_label.pack(side="left", fill="x", expand=True)
            badge = make_badge(row, case["status"])
            badge.pack(side="right")
            for widget in (row, id_label, parties_label, badge):
                widget.configure(cursor="hand2")
                widget.bind("<Button-1>", lambda event, case_id=case["id"]: self._open_case(case_id))
            tk.Frame(self.recent_frame, height=1, bg=LINE).pack(fill="x")

    # ------------------------------------------------------------ case records

    def _build_records(self) -> Screen:
        screen = Screen(self.stage, max_width=1160, fill_height=True)
        card = screen.card
        card.columnconfigure(0, weight=1)
        card.rowconfigure(1, weight=1)

        top = tk.Frame(card, bg=PAPER)
        top.grid(row=0, column=0, sticky="ew")
        make_label(top, "Digital record-keeping", fg=MOSS_LIGHT, font=FONT_MONO_SMALL, anchor="w").pack(fill="x")
        make_label(top, "Case records", fg=MOSS, font=FONT_TITLE, anchor="w").pack(fill="x", pady=(2, 14))
        search_row = tk.Frame(top, bg=PAPER)
        search_row.pack(fill="x", pady=(0, 6))
        self.search_var = tk.StringVar()
        search = make_entry(search_row, self.search_var)
        search.pack(side="left", fill="x", expand=True, ipady=6)
        self.filter_var = tk.StringVar(value="All statuses")
        status_filter = ttk.Combobox(search_row, textvariable=self.filter_var, state="readonly", width=14,
                                     values=["All statuses"] + [v[0] for v in STATUS_BADGES.values()])
        status_filter.pack(side="left", padx=(10, 0))
        self.search_var.trace_add("write", lambda *_: self._refresh_records())
        status_filter.bind("<<ComboboxSelected>>", lambda event: self._refresh_records())
        make_label(top, "Search by case number or name. Select a case to read its transcript.",
                   fg=MUTED, font=FONT_SMALL, anchor="w").pack(fill="x")

        body = tk.Frame(card, bg=PAPER)
        body.grid(row=1, column=0, sticky="nsew", pady=(10, 0))
        body.rowconfigure(0, weight=1)
        body.columnconfigure(0, weight=2, uniform="records")
        body.columnconfigure(1, weight=3, uniform="records")

        table = tk.Frame(body, bg=PAPER)
        table.grid(row=0, column=0, sticky="nsew", padx=(0, 12))
        table.columnconfigure(0, weight=1)
        table.rowconfigure(0, weight=1)
        self.records_tree = ttk.Treeview(table, columns=("parties", "date", "status"),
                                         show="tree headings", selectmode="browse", height=10)
        self.records_tree.heading("#0", text="CASE #", anchor="w")
        self.records_tree.heading("parties", text="PARTIES", anchor="w")
        self.records_tree.heading("date", text="DATE", anchor="w")
        self.records_tree.heading("status", text="STATUS", anchor="w")
        self.records_tree.column("#0", width=140, stretch=False)
        self.records_tree.column("parties", width=150)
        self.records_tree.column("date", width=60, stretch=False)
        self.records_tree.column("status", width=70, stretch=False)
        for key, (_, fg, _) in STATUS_BADGES.items():
            self.records_tree.tag_configure(key, foreground=fg)
        scrollbar = ttk.Scrollbar(table, orient="vertical", command=self.records_tree.yview)
        self.records_tree.configure(yscrollcommand=scrollbar.set)
        self.records_tree.grid(row=0, column=0, sticky="nsew")
        scrollbar.grid(row=0, column=1, sticky="ns")
        self.records_tree.bind("<<TreeviewSelect>>", self._on_record_selected)
        self.records_note = make_label(table, "", fg=MUTED, anchor="center")
        self.records_note.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(10, 0))

        detail = tk.Frame(body, bg=PAPER)
        detail.grid(row=0, column=1, sticky="nsew", padx=(12, 0))
        detail.columnconfigure(0, weight=1)
        detail.rowconfigure(4, weight=1)
        self.detail_title = make_label(detail, "", fg=MOSS, font=("Georgia", 14, "bold"), anchor="w")
        self.detail_title.grid(row=0, column=0, sticky="ew")
        self.detail_meta = make_wrapping_label(detail, "", fg=MUTED, font=FONT_SMALL)
        self.detail_meta.grid(row=1, column=0, sticky="ew", pady=(4, 10))
        make_label(detail, "Case status (changing it needs the admin password)", fg=MOSS_LIGHT,
                   font=FONT_SMALL, anchor="w").grid(row=2, column=0, sticky="w", pady=(0, 6))
        status_row = tk.Frame(detail, bg=PAPER)
        status_row.grid(row=3, column=0, sticky="ew", pady=(0, 12))
        self.detail_status_buttons = self._make_status_buttons(status_row, self._choose_record_status)
        transcript_box = tk.Frame(detail, bg=PAPER)
        transcript_box.grid(row=4, column=0, sticky="nsew")
        transcript_box.columnconfigure(0, weight=1)
        transcript_box.rowconfigure(0, weight=1)
        self.detail_transcript = self._make_transcript(transcript_box, row=0, height=8)
        self._show_case_detail(None)
        return screen

    def _refresh_records(self) -> None:
        query = self.search_var.get().strip().lower()
        wanted = self.filter_var.get()
        selected = self._selected_case_id
        self.records_tree.delete(*self.records_tree.get_children())
        shown = 0
        for case in self.cases:
            label = STATUS_BADGES[case["status"]][0]
            matches_text = (not query or query in case["id"].lower()
                            or query in case.get("parties", "").lower())
            if matches_text and wanted in ("All statuses", label):
                self.records_tree.insert("", "end", iid=case["id"], text=case["id"], tags=(case["status"],),
                                         values=(case.get("parties", ""), case.get("date", ""), label))
                shown += 1
        if shown:
            self.records_note.configure(text="")
        elif self.cases:
            self.records_note.configure(text="No cases match that search.")
        else:
            self.records_note.configure(text="No cases saved yet.")
        if selected is not None and self.records_tree.exists(selected):
            self.records_tree.selection_set(selected)
            self.records_tree.see(selected)
        else:
            self._selected_case_id = None
            self._show_case_detail(None)

    def _selected_case(self):
        return next((case for case in self.cases if case["id"] == self._selected_case_id), None)

    def _on_record_selected(self, event=None) -> None:
        selection = self.records_tree.selection()
        self._selected_case_id = selection[0] if selection else None
        self._show_case_detail(self._selected_case())

    def _open_case(self, case_id: str) -> None:
        """Jumps to Case records with this case selected (used by the Dashboard)."""
        self._go_tab("records")
        self.search_var.set("")
        self.filter_var.set("All statuses")
        self._selected_case_id = case_id
        self._refresh_records()

    def _show_case_detail(self, case) -> None:
        box = self.detail_transcript
        box.configure(state="normal")
        box.delete("1.0", "end")
        if case is None:
            self.detail_title.configure(text="No case selected")
            self.detail_meta.configure(text="Select a case on the left to read its transcript.")
            self._style_status_buttons(self.detail_status_buttons, None)
            for button in self.detail_status_buttons.values():
                button.configure(state="disabled")
            box.configure(state="disabled")
            return

        self.detail_title.configure(text=f"{case['id']}  \u00b7  {case.get('parties', '')}")
        lines = [f"Problem: {case.get('problem') or 'not given'}"]
        recorded = []
        if case.get("duration_seconds") is not None:
            recorded.append(f"Recorded {format_clock(case['duration_seconds'])}")
        if case.get("language"):
            recorded.append(f"spoken in {case['language']}")
        if recorded:
            lines.append(" ".join(recorded) + ".")
        history = case.get("status_history") or []
        if history:
            last = history[-1]
            lines.append(f"Last status change: {last.get('from')} to {last.get('to')} on {last.get('at')}.")
        self.detail_meta.configure(text="\n".join(lines))
        self._style_status_buttons(self.detail_status_buttons, case["status"])
        for button in self.detail_status_buttons.values():
            button.configure(state="normal")

        segments = case.get("transcript") or []
        for index, segment in enumerate(segments):
            box.insert("end", ("\n" if index else "") + f"{segment.get('speaker', '')}\n", "speaker")
            box.insert("end", segment.get("text", "") + "\n")
        if case.get("translation"):
            box.insert("end", "\nTranslation\n", "divider")
            box.insert("end", case["translation"], "translation")
        if not segments and not case.get("translation"):
            box.insert("end", "No transcript was saved for this case.", "divider")
        box.configure(state="disabled")

    def _choose_record_status(self, status: str) -> None:
        """Admin-only edit of a saved case. The change is logged inside the record."""
        case = self._selected_case()
        if case is None or status == case["status"]:
            return
        if not self._require_admin():
            return
        case.setdefault("status_history", []).append({
            "from": case["status"], "to": status, "at": datetime.now().isoformat(timespec="seconds"),
        })
        case["status"] = status
        self._persist_cases()
        self._refresh_dashboard()
        self._refresh_records()
        self._show_toast(f"Case {case['id']} marked {STATUS_BADGES[status][0]}")

    # ------------------------------------------------------------------ admin

    def _require_admin(self) -> bool:
        """True when admin is unlocked. Otherwise asks for the password (creating it on first use)."""
        if time.monotonic() < self._admin_until:
            return True
        state = admin_state()
        if state == "damaged":
            messagebox.showerror(
                "Admin password unreadable",
                f"{ADMIN_FILE.name} is damaged, so nobody can be verified.\n\n"
                f"Delete {ADMIN_FILE.name} from the project folder to create a new admin password.")
            return False
        dialog = AdminDialog(self, creating=state == "missing", submit=self._submit_admin_password)
        self.wait_window(dialog)
        self._refresh_admin_button()
        return dialog.result

    def _submit_admin_password(self, password: str, confirm: str):
        """Checks or creates the admin password. Returns an error message, or None on success."""
        now = time.monotonic()
        if admin_state() == "missing":
            if len(password) < ADMIN_MIN_PASSWORD:
                return f"Use at least {ADMIN_MIN_PASSWORD} characters."
            if password != confirm:
                return "The two passwords do not match."
            try:
                set_admin_password(password)
            except OSError as error:
                return f"Could not save the password: {error}"
        else:
            if now < self._admin_locked_until:
                return f"Too many wrong attempts. Try again in {int(self._admin_locked_until - now) + 1} seconds."
            if not check_admin_password(password):
                self._admin_failures += 1
                if self._admin_failures >= ADMIN_MAX_ATTEMPTS:
                    self._admin_failures = 0
                    self._admin_locked_until = now + ADMIN_LOCKOUT_SECONDS
                    return f"Too many wrong attempts. Locked for {ADMIN_LOCKOUT_SECONDS} seconds."
                return "Wrong password."
            self._admin_failures = 0
        self._admin_until = now + ADMIN_UNLOCK_SECONDS
        return None

    def _toggle_admin(self) -> None:
        if time.monotonic() < self._admin_until:
            self._admin_until = 0.0
            self._refresh_admin_button()
            self._show_toast("Admin locked")
        else:
            self._require_admin()

    def _refresh_admin_button(self) -> None:
        remaining = max(0.0, self._admin_until - time.monotonic())
        button = self.tab_buttons["admin"]
        if remaining > 0:
            text = f"Admin: unlocked {format_clock(remaining)}"
            colors = {"bg": GOLD_BG, "fg": GOLD_TEXT, "activebackground": GOLD_BG, "activeforeground": GOLD_TEXT}
        else:
            text = "Admin: locked"
            colors = {"bg": BG, "fg": MOSS_LIGHT, "activebackground": MOSS_PALE, "activeforeground": MOSS}
        if button.cget("text") != text or button.cget("bg") != colors["bg"]:
            button.configure(text=text, **colors)

    def _persist_cases(self) -> bool:
        try:
            save_cases(self.cases)
        except OSError as error:
            messagebox.showerror("Could not save the case",
                                 f"The change is shown in this session but was not written to disk.\n\n{error}")
            return False
        return True

    # ------------------------------------------------------------ case details

    def _build_setup(self) -> Screen:
        screen = Screen(self.stage, max_width=640)
        card = screen.card
        make_label(card, "New hearing session", fg=MOSS_LIGHT, font=FONT_MONO_SMALL, anchor="w").pack(fill="x")
        make_label(card, "Case details", fg=MOSS, font=FONT_TITLE, anchor="w").pack(fill="x", pady=(2, 10))

        self.ref_date = tk.StringVar()
        self.ref_time = tk.StringVar()
        self.complainant = tk.StringVar()
        self.respondent = tk.StringVar()
        self.problem = tk.StringVar()
        self.consent_complainant = tk.BooleanVar(value=False)
        self.consent_respondent = tk.BooleanVar(value=False)

        def field(parent: tk.Misc, caption: str, variable: tk.StringVar, hint: str = "") -> tk.Frame:
            wrapper = tk.Frame(parent, bg=PAPER)
            make_label(wrapper, caption, fg=MOSS_LIGHT, font=FONT_SMALL, anchor="w").pack(fill="x", pady=(0, 4))
            make_entry(wrapper, variable).pack(fill="x", ipady=7)
            if hint:
                make_label(wrapper, hint, fg=MUTED, font=FONT_SMALL, anchor="w").pack(fill="x", pady=(3, 0))
            return wrapper

        row = tk.Frame(card, bg=PAPER)
        row.pack(fill="x", pady=(0, 10))
        row.columnconfigure((0, 1), weight=1, uniform="ref")
        field(row, "Case reference: date filed (YYYY-MM-DD)", self.ref_date).grid(
            row=0, column=0, sticky="ew", padx=(0, 7))
        field(row, "Time filed (HH:MM, 24 hour)", self.ref_time).grid(
            row=0, column=1, sticky="ew", padx=(7, 0))

        row = tk.Frame(card, bg=PAPER)
        row.pack(fill="x", pady=(0, 10))
        row.columnconfigure((0, 1), weight=1, uniform="names")
        field(row, "Complainant name", self.complainant).grid(row=0, column=0, sticky="ew", padx=(0, 7))
        field(row, "Respondent name", self.respondent).grid(row=0, column=1, sticky="ew", padx=(7, 0))

        field(card, "What is the problem (for example: boundary marker moved onto neighboring lot)",
              self.problem).pack(
            fill="x", pady=(0, 10))

        consent = tk.Frame(card, bg=GOLD_BG, highlightbackground=GOLD_LINE, highlightthickness=1,
                           padx=18, pady=14)
        consent.pack(fill="x", pady=(2, 12))
        make_label(consent, "Consent required before recording", fg="#8A6A26", font=FONT_MEDIUM,
                   bg=GOLD_BG, anchor="w").pack(fill="x", pady=(0, 8))
        for text, variable in (("Complainant consent obtained", self.consent_complainant),
                               ("Respondent consent obtained", self.consent_respondent)):
            tk.Checkbutton(consent, text=text, variable=variable, command=self._validate_setup,
                           bg=GOLD_BG, fg=GOLD_TEXT, activebackground=GOLD_BG, activeforeground=GOLD_TEXT,
                           selectcolor=PAPER, font=FONT, anchor="w", cursor="hand2", relief="flat",
                           bd=0, highlightthickness=0, pady=3).pack(fill="x")

        self.proceed_button = ttk.Button(card, text="Proceed to recording",
                                         command=self._proceed_to_recording)
        self.proceed_button.pack(fill="x")
        self.setup_hint = make_wrapping_label(card, "", fg=MUTED, font=FONT_SMALL)
        self.setup_hint.pack(fill="x", pady=(6, 0))

        for variable in (self.ref_date, self.ref_time, self.complainant, self.respondent, self.problem,
                         self.consent_complainant, self.consent_respondent):
            variable.trace_add("write", lambda *_: self._validate_setup())
        return screen

    def _start_new_hearing(self) -> None:
        if self.session_active or self._finishing_speech() or self.translating:
            self.status.set("Wait for the previous hearing to finish first.")
            return
        now = datetime.now()
        self.ref_date.set(now.strftime("%Y-%m-%d"))
        self.ref_time.set(now.strftime("%H:%M"))
        for variable in (self.complainant, self.respondent, self.problem):
            variable.set("")
        self.consent_complainant.set(False)
        self.consent_respondent.set(False)
        self._validate_setup()
        self._show_screen("setup")

    def _validate_setup(self) -> bool:
        """Updates the Proceed button and hint. True when the hearing may start."""
        try:
            filed = datetime.strptime(f"{self.ref_date.get().strip()} {self.ref_time.get().strip()}",
                                      "%Y-%m-%d %H:%M")
        except ValueError:
            filed = None
        names_filled = bool(self.complainant.get().strip() and self.respondent.get().strip())
        consented = self.consent_complainant.get() and self.consent_respondent.get()

        ready = filed is not None and names_filled and consented
        self.proceed_button.state(["!disabled"] if ready else ["disabled"])
        if ready:
            self.setup_hint.configure(
                text="Case details are complete and both parties have consented. You can start recording.",
                fg=MUTED)
        elif filed is None:
            self.setup_hint.configure(
                text="Enter the date as YYYY-MM-DD and the time as HH:MM (24 hour), for example 2026-10-27 14:30.",
                fg=RUST)
        elif not names_filled:
            self.setup_hint.configure(
                text="Fill in the complainant name and respondent name to continue.", fg=RUST)
        else:
            self.setup_hint.configure(
                text="Both consent boxes must be checked before recording can begin.", fg=RUST)
        return ready

    def _proceed_to_recording(self) -> None:
        if not self._validate_setup():
            return
        if self._finishing_speech() or self.translating:
            self.status.set("Wait for the previous hearing to finish first.")
            return
        filed = datetime.strptime(f"{self.ref_date.get().strip()} {self.ref_time.get().strip()}",
                                  "%Y-%m-%d %H:%M")
        complainant = self.complainant.get().strip()
        respondent = self.respondent.get().strip()
        self._hearing = {
            "id": filed.strftime("%m/%d/%Y %H:%M"),
            "iso_date": filed.strftime("%Y-%m-%d"),
            "date": f"{filed:%b} {filed.day}",
            "complainant": complainant,
            "respondent": respondent,
            "parties": f"{complainant} vs. {respondent}",
            "problem": self.problem.get().strip(),
        }
        self._set_speaker(SPEAKERS[0])
        self._set_case_status("open")
        self._show_screen("recording")
        self.update_idletasks()  # Canvases need a real size before the first frame is drawn.
        if not self._start_session():
            self._show_screen("setup")

    # --------------------------------------------------------------- recording

    def _build_recording(self) -> Screen:
        screen = Screen(self.stage, max_width=1160, fill_height=True)
        card = screen.card
        card.rowconfigure(2, weight=1)
        card.columnconfigure(0, weight=1)

        header = tk.Frame(card, bg=PAPER)
        header.grid(row=0, column=0, sticky="ew")
        left = tk.Frame(header, bg=PAPER)
        left.pack(side="left")
        row = tk.Frame(left, bg=PAPER)
        row.pack(anchor="w")
        make_label(row, "Now speaking:", fg=MOSS_LIGHT, font=("Segoe UI", 11)).pack(side="left")
        self.speaker_label = make_label(row, SPEAKERS[0], fg=MOSS, font=("Segoe UI", 11, "bold"))
        self.speaker_label.pack(side="left", padx=(6, 0))
        pills = tk.Frame(left, bg=PAPER)
        pills.pack(anchor="w", pady=(8, 0))
        self.speaker_buttons = {}
        for who in SPEAKERS:
            button = tk.Button(pills, text=who, relief="flat", bd=0, font=FONT_SMALL, padx=12, pady=4,
                               cursor="hand2", command=lambda who=who: self._set_speaker(who))
            button.pack(side="left", padx=(0, 6))
            self.speaker_buttons[who] = button
        make_label(left, "Switch the speaker after a pause so the words are credited correctly.",
                   fg=MUTED, font=FONT_SMALL, anchor="w").pack(anchor="w", pady=(6, 0))

        timer = tk.Frame(header, bg=PAPER)
        timer.pack(side="right")
        self.rec_dot = make_label(timer, "●", fg=MUTED, font=("Segoe UI", 10))
        self.rec_dot.pack(side="left")
        self.rec_timer = make_label(timer, "00:00", fg=RUST, font=("Consolas", 14))
        self.rec_timer.pack(side="left", padx=(6, 0))

        tk.Frame(card, height=1, bg=LINE).grid(row=1, column=0, sticky="ew", pady=14)

        body = tk.Frame(card, bg=PAPER)
        body.grid(row=2, column=0, sticky="nsew")
        body.rowconfigure(0, weight=3)   # Camera.
        body.rowconfigure(1, weight=2)   # Waveform and live transcript.
        body.columnconfigure((0, 1), weight=1, uniform="columns")

        self.video_canvas = tk.Canvas(body, bg=VIDEO_BG, height=170, highlightthickness=0)
        self.video_canvas.grid(row=0, column=0, columnspan=2, sticky="nsew", pady=(0, 10))
        self.video_canvas.bind("<Configure>", lambda event: self._on_video_resize())

        self.wave_canvas = tk.Canvas(body, bg=CREAM, highlightbackground=LINE, highlightthickness=1,
                                     height=110)
        self.wave_canvas.grid(row=1, column=0, sticky="nsew", padx=(0, 8))

        transcript_side = tk.Frame(body, bg=PAPER)
        transcript_side.grid(row=1, column=1, sticky="nsew", padx=(8, 0))
        transcript_side.rowconfigure(1, weight=1)
        transcript_side.columnconfigure(0, weight=1)
        make_label(transcript_side, "LIVE TRANSCRIPT", fg=MUTED, font=("Segoe UI", 8, "bold"), anchor="w").grid(
            row=0, column=0, sticky="w", pady=(0, 6))
        self.live_transcript = self._make_transcript(transcript_side, row=1, height=6)

        controls = tk.Frame(card, bg=PAPER)
        controls.grid(row=3, column=0, sticky="ew", pady=(14, 0))
        self.lock_button = ttk.Button(controls, text="Lock face (L)", style="Ghost.TButton",
                                      command=self._lock_face)
        self.unlock_button = ttk.Button(controls, text="Unlock (U)", style="Ghost.TButton",
                                        command=self._unlock_face)
        self.lock_button.pack(side="left")
        self.unlock_button.pack(side="left", padx=(6, 0))
        make_label(controls, "Language spoken", fg=MUTED, font=FONT_SMALL).pack(side="left", padx=(20, 6))
        self.speaking = tk.StringVar(value=next(iter(SPEAKING)))
        speaking_box = ttk.Combobox(controls, textvariable=self.speaking, state="readonly",
                                    values=list(SPEAKING), width=9)
        speaking_box.bind("<<ComboboxSelected>>", self._on_speaking_changed)
        speaking_box.pack(side="left")
        self.end_button = ttk.Button(controls, text="End statement", command=self._end_statement)
        self.end_button.pack(side="right")
        return screen

    def _make_transcript(self, parent: tk.Misc, row: int, height: int = 10) -> tk.Text:
        """A read-only transcript box with a scrollbar and the tags the app uses."""
        box = tk.Text(parent, wrap="word", state="disabled", bg=CREAM, fg=INK, relief="flat",
                      padx=14, pady=12, font=("Segoe UI", 11), highlightthickness=1,
                      highlightbackground=LINE, height=height)
        box.grid(row=row, column=0, sticky="nsew")
        scrollbar = ttk.Scrollbar(parent, orient="vertical", command=box.yview)
        scrollbar.grid(row=row, column=1, sticky="ns")
        box.configure(yscrollcommand=scrollbar.set)
        box.tag_configure("speaker", foreground=MOSS, font=("Segoe UI", 10, "bold"), spacing1=8, spacing3=2)
        box.tag_configure("divider", foreground=MUTED, spacing1=10, spacing3=6)
        box.tag_configure("translation", foreground=WAVE_COLOR)
        return box

    def _set_speaker(self, who: str) -> None:
        self.speaker = who
        self.speaker_label.configure(text=who)
        for name, button in self.speaker_buttons.items():
            active = name == who
            button.configure(bg=MOSS if active else CREAM, fg=PAPER if active else MOSS_LIGHT,
                             activebackground=MOSS_LIGHT if active else MOSS_PALE,
                             activeforeground=PAPER if active else MOSS)

    def _update_timer(self) -> None:
        if self.current_screen != "recording":
            return
        recorder = self.recorder
        seconds = recorder.buffer.length / recorder.sample_rate if recorder else 0
        self.rec_timer.configure(text=format_clock(seconds))
        if self.session_active and int(time.monotonic() * 1.4) % 2 == 0:
            self.rec_dot.configure(fg=RUST)
        else:
            self.rec_dot.configure(fg=MUTED if not self.session_active else MOSS_PALE)

    # ----------------------------------------------------------------- session

    def _start_session(self) -> bool:
        """Starts video, audio and speech. Returns False when nothing could start."""
        # A new session replaces the previous transcript and recording.
        self._clear_transcripts()
        self._segments = []
        self._words = []
        self._translation_text = ""
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
            return False

        self.session_active = True
        self._set_running_status()
        self._refresh_buttons()
        return True

    def _try_start(self, label: str, factory, problems: list):
        """Start one part. On failure, record why and carry on without it."""
        try:
            return factory()
        except Exception as error:  # Broad on purpose: parts must fail independently.
            problems.append(f"{label}: {error}")
            return None

    def _make_face_worker(self):
        from face import FaceWorker
        worker = FaceWorker(source=CAMERA_INDEX)
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

    def _discard_session(self) -> None:
        """Stops everything and forgets the hearing. Nothing is saved."""
        if self.session_active:
            self._stop_session()
        self.speech_worker = None   # A worker still finishing is a daemon thread and ends by itself.
        self._clear_transcripts()
        self._segments = []
        self._words = []
        self._translation_text = ""
        self.status.set("Hearing discarded.")
        self._refresh_buttons()

    def _end_statement(self) -> None:
        if self.session_active:
            self._stop_session()
        self._fill_assessment()
        self._show_screen("report")

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

    def _refresh_buttons(self) -> None:
        def set_enabled(button: ttk.Button, enabled: bool) -> None:
            button.state(["!disabled"] if enabled else ["disabled"])

        busy = self._finishing_speech() or self.translating
        face_live = self.session_active and self.face_worker is not None
        set_enabled(self.lock_button, face_live)
        set_enabled(self.unlock_button, face_live)
        set_enabled(self.end_button, self.session_active)
        set_enabled(self.save_button, not self.session_active and not busy)

    def _lock_face(self) -> None:
        if self.face_worker is not None:
            self.face_worker.lock_largest()

    def _unlock_face(self) -> None:
        if self.face_worker is not None:
            self.face_worker.unlock()

    def _on_face_key(self, event: tk.Event, lock: bool) -> None:
        """L and U only mean lock and unlock on the Recording screen, never while typing."""
        if self.current_screen != "recording" or isinstance(event.widget, (tk.Entry, ttk.Combobox)):
            return
        if lock:
            self._lock_face()
        else:
            self._unlock_face()

    def _open_settings(self) -> None:
        window = self._settings_window
        if window is not None and window.winfo_exists():
            window.lift()
            window.focus_force()
            return
        self._settings_window = SettingsWindow(self, lambda: self.session_active)

    def _on_close(self) -> None:
        if self.session_active:
            self._stop_session()  # Non-blocking for speech. Daemon threads end with the app.
        self.destroy()

    # -------------------------------------------------------------- assessment

    def _build_report(self) -> Screen:
        screen = Screen(self.stage, max_width=1100)
        card = screen.card
        card.columnconfigure(0, weight=1)

        self.report_eyebrow = make_label(card, "", fg=MOSS_LIGHT, font=FONT_MONO_SMALL, anchor="w")
        self.report_eyebrow.grid(row=0, column=0, sticky="ew")
        make_label(card, "Transcript", fg=MOSS, font=FONT_TITLE, anchor="w").grid(
            row=1, column=0, sticky="ew", pady=(2, 10))

        transcript = tk.Frame(card, bg=PAPER)
        transcript.grid(row=2, column=0, sticky="nsew")
        transcript.rowconfigure(0, weight=1)
        transcript.columnconfigure(0, weight=1)
        self.report_transcript = self._make_transcript(transcript, row=0, height=14)
        # No Translate controls on this screen any more. The translation code still reads this.
        self.direction = tk.StringVar(value=next(iter(DIRECTIONS)))

        bottom = tk.Frame(card, bg=PAPER)
        bottom.grid(row=3, column=0, sticky="ew", pady=(12, 0))
        make_label(bottom, "Case status (changing it needs the admin password)", fg=MOSS_LIGHT,
                   font=FONT_SMALL, anchor="w").pack(fill="x", pady=(0, 6))
        controls_row = tk.Frame(bottom, bg=PAPER)
        controls_row.pack(fill="x")
        toggle = tk.Frame(controls_row, bg=PAPER)
        toggle.pack(side="left", fill="x", expand=True)
        self.status_choice = tk.StringVar(value="open")
        self.status_buttons = self._make_status_buttons(toggle, self._choose_status)
        self.save_button = ttk.Button(controls_row, text="Save and return to dashboard",
                                      style="Ghost.TButton", command=self._save_hearing)
        self.save_button.pack(side="right", padx=(16, 0))
        self._set_case_status("open")
        return screen

    def _make_status_buttons(self, parent: tk.Misc, command) -> dict:
        """Open / Closed / To court as three side-by-side buttons. command(status) runs on click."""
        buttons = {}
        for key, (label, _, _) in STATUS_BADGES.items():
            button = tk.Button(parent, text=label, relief="flat", bd=0, font=FONT_MEDIUM, pady=8,
                               cursor="hand2", command=lambda key=key: command(key))
            button.pack(side="left", fill="x", expand=True, padx=(0 if key == "open" else 8, 0))
            buttons[key] = button
        return buttons

    def _style_status_buttons(self, buttons: dict, current) -> None:
        for key, button in buttons.items():
            _, fg, bg = STATUS_BADGES[key]
            if key == current:
                button.configure(bg=bg, fg=fg, activebackground=bg, activeforeground=fg)
            else:
                button.configure(bg=CREAM, fg=MOSS_LIGHT, activebackground=MOSS_PALE, activeforeground=MOSS)

    def _set_case_status(self, status: str) -> None:
        """Sets the status with no password check. Used for the Open default of a new hearing."""
        self.status_choice.set(status)
        self._style_status_buttons(self.status_buttons, status)

    def _choose_status(self, status: str) -> None:
        """A click on the Assessment screen. Changing the default Open needs the admin password."""
        if status != self.status_choice.get() and self._require_admin():
            self._set_case_status(status)

    def _recorded_seconds(self) -> float:
        recorder = self.recorder
        return recorder.buffer.length / recorder.sample_rate if recorder else 0.0

    def _word_counts(self) -> dict:
        counts = {who: 0 for who in SPEAKERS}
        for segment in self._segments:
            counts[segment["speaker"]] += len(segment["words"])
        return counts

    def _fill_assessment(self) -> None:
        hearing = self._hearing
        self.report_eyebrow.configure(text=f"Case {hearing['id']}  \u00b7  {hearing['parties']}")

    def _save_hearing(self) -> None:
        if self.session_active or self._finishing_speech() or self.translating:
            return
        hearing = self._hearing
        record = {
            "id": hearing["id"],
            "iso_date": hearing["iso_date"],
            "date": hearing["date"],
            "parties": hearing["parties"],
            "complainant": hearing["complainant"],
            "respondent": hearing["respondent"],
            "problem": hearing["problem"],
            "status": self.status_choice.get(),
            "language": self.speaking.get(),
            "duration_seconds": round(self._recorded_seconds(), 1),
            "transcript": [{"speaker": s["speaker"], "text": " ".join(s["words"])} for s in self._segments],
            "translation": self._translation_text,
            "status_history": [],
        }
        self.cases = [case for case in self.cases if case["id"] != record["id"]]
        self.cases.insert(0, record)
        if self._persist_cases():
            self._show_toast(f"Case {record['id']} saved to records")
        self._refresh_dashboard()
        self._refresh_records()
        self._show_screen("dashboard")

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
                self._append_transcript(f"\n{source} to {target}\n", "divider", report_only=True)
                self._append_transcript(result + "\n", "translation", report_only=True)
                self._translation_text += f"{source} to {target}: {result}\n"
                self.translating = False
                self.status.set("Translation finished.")
                self._refresh_buttons()
            elif kind == "error":
                self.translating = False
                self.status.set("Translation failed.")
                self._refresh_buttons()
                messagebox.showerror("Translation failed", payload)

    # -------------------------------------------------------------- transcript

    def _clear_transcripts(self) -> None:
        for box in (self.live_transcript, self.report_transcript):
            box.configure(state="normal")
            box.delete("1.0", "end")
            box.configure(state="disabled")

    def _append_transcript(self, text: str, tag: str = "", report_only: bool = False) -> None:
        boxes = (self.report_transcript,) if report_only else (self.live_transcript, self.report_transcript)
        for box in boxes:
            box.configure(state="normal")
            box.insert("end", text, tag)
            box.see("end")
            box.configure(state="disabled")

    def _drain_words(self) -> None:
        if self.speech_worker is None:
            return
        new_words = []
        while True:
            try:
                new_words.append(self.speech_worker.words.get_nowait())
            except queue.Empty:
                break
        if not new_words:
            return
        if not self._segments or self._segments[-1]["speaker"] != self.speaker:
            self._segments.append({"speaker": self.speaker, "words": []})
            prefix = "\n" if len(self._segments) > 1 else ""
            self._append_transcript(f"{prefix}{self.speaker}\n", "speaker")
        self._segments[-1]["words"].extend(new_words)
        self._words.extend(new_words)
        self._append_transcript(" ".join(new_words) + " ")

    # ------------------------------------------------------------------- video

    def _poll_video(self) -> None:
        worker = self.face_worker
        if worker is not None:
            try:
                self._show_frame(worker.frames.get_nowait())
            except queue.Empty:
                pass
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

    def _on_video_resize(self) -> None:
        if self.face_worker is None:
            self._draw_video_idle()

    def _draw_video_idle(self) -> None:
        canvas = self.video_canvas
        canvas.delete("all")
        canvas.create_text(canvas.winfo_width() // 2, canvas.winfo_height() // 2,
                           text="Video", fill=VIDEO_TEXT, font=("Segoe UI", 12))

    # ---------------------------------------------------------------- waveform

    def _poll_slow(self) -> None:
        self._drain_words()
        self._finish_speech_if_done()
        self._refresh_speech_status()
        if self.current_screen == "recording":
            self._redraw_wave()
            self._update_timer()
        elif self.current_screen == "report":
            self._fill_assessment()
        self._check_worker_errors()
        self._drain_translation()
        self._refresh_admin_button()
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
