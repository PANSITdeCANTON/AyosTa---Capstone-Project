"""
exports.py - file-writing helpers for AyosTa.

This module contains no GUI code. The controller chooses when to call these
functions, and gui.py only asks the user for a destination path.
"""

import wave
from datetime import datetime
from pathlib import Path
from xml.sax.saxutils import escape

import numpy as np


def save_wav(path: str | Path, samples, sample_rate: int) -> None:
    """Write mono signed 16-bit PCM samples to a WAV file."""
    rate = int(sample_rate)
    if rate <= 0:
        raise ValueError("Sample rate must be a positive integer.")

    data = np.asarray(samples, dtype=np.int16)
    if data.size == 0:
        raise ValueError("There is no audio to save.")

    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(rate)
        wav_file.writeframes(data.tobytes())


def export_transcript_pdf(path: str | Path, text: str) -> None:
    """Write transcript text to PDF; reportlab is imported only when needed."""
    transcript = str(text).strip()
    if not transcript:
        raise ValueError("There is no transcript to export.")

    from reportlab.lib.enums import TA_LEFT
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import mm
    from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer

    document = SimpleDocTemplate(
        str(path),
        pagesize=A4,
        rightMargin=18 * mm,
        leftMargin=18 * mm,
        topMargin=18 * mm,
        bottomMargin=18 * mm,
        title="AyosTa Meeting Transcript",
    )
    styles = getSampleStyleSheet()
    body_style = ParagraphStyle(
        "TranscriptBody",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=10,
        leading=14,
        alignment=TA_LEFT,
        spaceAfter=5,
        wordWrap="CJK",
    )
    story = [
        Paragraph("AyosTa Meeting Transcript", styles["Title"]),
        Paragraph(f"Exported: {datetime.now():%Y-%m-%d %H:%M:%S}", styles["Normal"]),
        Spacer(1, 8 * mm),
    ]

    for line in transcript.splitlines():
        safe_line = escape(line) if line.strip() else "&#160;"
        story.append(Paragraph(safe_line, body_style))

    document.build(story)
