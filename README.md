# 👋 Welcome to AyosTa!

This guide will get AyosTa running on your Windows computer in a few easy steps. Everything is installed in its own little folder (`.venv`), so nothing touches the rest of your system.

---

## ✅ What You'll Need

| | |
|---|---|
| **Windows** | Windows 10 or 11 (64-bit) |
| **Python** | Python 3.12 (or 3.11), 64-bit, with the Python Launcher (`py`) |
| **Internet** | For downloading packages and AI models |
| **Devices** | A microphone and a webcam |
| **Space** | A few GB free (AI models can be big) |

💡 **Tip:** If everyone on the team uses the same Python version (3.11 or 3.12), testing goes smoothly.

---

## 📁 Your Project Folder

Keep these files together in the main **AyosTa - Capstone** folder:

- `requirements.txt`
- `README.md`
- `main.py`, `gui.py`, `audio.py`, `speech.py`, `face.py`, `language.py`, `exports.py`, `syscheck.py`
- `src\yolov8n-face.pt`

---

## 🚀 Quick Start

MUST HAVE env Folder

```.venv\Scripts\python.exe main.py```

Want to double-check everything is ready? Run:

```.venv\Scripts\python.exe syscheck.py```

For a deeper check (loads the AI models, so it takes longer):

```.venv\Scripts\python.exe syscheck.py --deep ```

Before running py main.py:

```.venv\Scripts\Activate.ps```

---

## 🧠 First Run: AI Models

Some features download their models the first time you use them, so the first run takes a little longer.

- **Speech recognition:** `facebook/mms-1b-all` via Hugging Face Transformers
- **Translation:** the GGUF model set in `language.py` (a multi-gigabyte download)
- **Face detection:** `src\yolov8n-face.pt`, already included in the project

A steady internet connection and some free disk space make this part easy.

---

## 🧩 About `llama-cpp-python` (Translation)

This package works best with a ready-made Windows build. If setup pauses here, these steps usually fix it:

1. Check your Python version through cmd:
   ```py -0p```
2. Refresh pip:
   ```.venv\Scripts\python.exe -m pip install --upgrade pip```
3. Try the install again:
   ```c.venv\Scripts\python.exe -m pip install llama-cpp-python```
4. Still stuck? Copy the full message and share it with the team, or check the [official instructions](https://github.com/abetlen/llama-cpp-python).

If you're only testing audio, speech, and face features at first, the team can set translation aside and add it later.

---

## 🙋 Need a Hand?

**`py` isn't recognized**
Install Python from the link above with the Python Launcher, then reopen Command Prompt.

**Packages didn't finish installing**
Check your internet and free space, then try again:
```cmd.venv\Scripts\python.exe -m pip install -r requirements.txt```
If a package name shows up in the message, copy the whole message to share with the team.

**Microphone or camera isn't showing up**
Check Windows privacy settings for mic/camera access, close other apps using them, and test the device in another app. USB audio devices may need their own drivers.

**A model download didn't finish**
Reconnect to the internet and open the feature again. It will pick up where it left off. Keep your personal access tokens private.

**Want a fresh start?**
Close AyosTa, rename `.venv` to `.venv-backup`, and run `setup.bat` again. Your downloaded models stay safe.

---

## 📝 Team Test Checklist

Tick these off as you go, and note your Python version, Windows version, and where anything needed attention:

- [ ] Python 3.11/3.12 detected
- [ ] Core import check passed
- [ ] `main.py` opens the app
- [ ] Microphone works
- [ ] Webcam and face model load
- [ ] Speech model downloads and loads
- [ ] Translation model downloads and loads
- [ ] Audio export and PDF export work
- [ ] `syscheck.py` shows the expected results

Installing the packages is step one. Trying each feature is how we know everything works together!

---

## 📦 Keeping Dependencies Tidy

`requirements.txt` lists AyosTa's direct dependencies, which keeps installs simple across everyone's computers. After a successful clean-machine test, the team can agree on one tested Python version and pin the package versions for release.

<details> <summary>See <code>requirements.txt</code></summary>

```text
# AyosTa direct runtime dependencies
# Recommended: 64-bit Python 3.11 or 3.12 on Windows 10/11.
# Install in the project's .venv with: python -m pip install -r requirements.txt

# Core arrays, image handling, webcam and face detection
numpy>=1.26,<3
Pillow>=10.0
opencv-python>=4.9
ultralytics>=8.3

# Microphone capture and system diagnostics
sounddevice>=0.5
psutil>=5.9

# Live speech recognition (Meta MMS; code uses Transformers v5-style APIs)
torch>=2.6
transformers>=5.0
huggingface-hub>=0.30

# GGUF translation model
llama-cpp-python>=0.3.5

# PDF transcript export
reportlab>=4.0
```

</details>
IF MORE PROBLEM ASK AI TO INSTALL ALL requirements.txt and ENV SCRIPTS ALSO YOUR PYTHON MUST BE IN PATH IN YOUR SYSTEM VARIABLES

BEFORE PUSHING MAKE SURE NO ERROR OR ILL KILL U

---

