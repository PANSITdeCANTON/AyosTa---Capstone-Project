"""
syscheck.py - system requirement and package checks. NO GUI CODE in this file.

Used by the Settings window in main.py. It also runs on its own:
    python syscheck.py            quick check
    python syscheck.py --deep     also import-tests every package and opens the camera briefly

WHAT IT CHECKS
    System        RAM, CPU cores, GPU, free disk space
    Python        version, virtual environment
    Packages      every pip package the project needs, grouped by feature, with versions
    Model files   whether the big downloads already exist on disk
    Devices       microphone; camera (deep check only, because it turns the camera on briefly)

LIMITS (read these)
    - The RAM, CPU and disk thresholds below are rough estimates. They were never measured.
    - The quick check proves a package is INSTALLED, not that it WORKS. A package can install
      and still fail to load (missing DLL, wrong Python version). The deep check imports each
      package in a separate process to catch that.
    - Where possible, names are read from the project's own files (WEIGHTS_PATH in face.py,
      MMS_MODEL_ID in speech.py, and so on) so they cannot drift. If you add a dependency,
      add it to PACKAGES below.
"""

import ast
import os
import platform
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import importlib.metadata
import importlib.util

OK, WARN, FAIL, SKIP = "ok", "warn", "fail", "skip"

# --- Thresholds (rough estimates, never measured) --------------------------------
MIN_PYTHON = (3, 10)
RAM_FAIL_GB = 6.0            # Below this, speech + video cannot realistically run together.
RAM_WARN_GB = 12.0           # Below this it is tight: expect to close other programs.
FREE_RAM_FAIL_GB = 1.5
FREE_RAM_WARN_GB = 3.0
MIN_CORES_WARN = 4
PROJECT_DISK_WARN_GB = 2.0
MMS_DOWNLOAD_GB = 4.0        # Approximate size of the speech model download.
GGUF_DOWNLOAD_GB = 2.4       # Approximate size of the translation model download.
IMPORT_TEST_TIMEOUT_SECONDS = 120

_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)  # Hides console flashes on Windows.


@dataclass
class CheckResult:
    group: str
    name: str
    status: str
    detail: str
    fix: str = ""            # Human-readable fix. Empty when there is nothing to do.
    pip_package: str = ""    # When set, install_commands() can build a pip command for it.
    pip_flags: str = ""


@dataclass(frozen=True)
class Package:
    feature: str
    pip_name: str            # Name for `pip install`. Empty when pip cannot provide it.
    import_name: str
    dist_names: tuple        # Names to look up in installed metadata. First match wins.
    min_version: str = ""
    pip_flags: str = ""
    note: str = ""


_CPU_WHEELS = "--extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu"
_OPENCV_DISTS = ("opencv-python", "opencv-python-headless", "opencv-contrib-python")

PACKAGES = [
    Package("GUI packages (main.py)", "pillow", "PIL", ("pillow",)),
    Package("GUI packages (main.py)", "numpy", "numpy", ("numpy",)),
    Package("GUI packages (main.py)", "", "tkinter", (),
            note="Ships with Python. If missing, reinstall Python with tcl/tk included."),
    Package("Video packages (face.py)", "ultralytics", "ultralytics", ("ultralytics",)),
    Package("Video packages (face.py)", "opencv-python", "cv2", _OPENCV_DISTS),
    Package("Video packages (face.py)", "torch", "torch", ("torch",)),
    Package("Audio packages (audio.py)", "sounddevice", "sounddevice", ("sounddevice",)),
    Package("Audio packages (audio.py)", "numpy", "numpy", ("numpy",)),
    Package("Speech packages (speech.py)", "torch", "torch", ("torch",)),
    # 4.30 is the minimum named in the MMS documentation.
    Package("Speech packages (speech.py)", "transformers", "transformers", ("transformers",),
            min_version="4.30"),
    Package("Translation packages (language.py)", "llama-cpp-python", "llama_cpp",
            ("llama-cpp-python",), pip_flags=_CPU_WHEELS,
            note="Wheel availability for your Python version was not verified."),
    Package("Translation packages (language.py)", "huggingface-hub", "huggingface_hub",
            ("huggingface-hub", "huggingface_hub")),
    #add here if new package included
]


# =============================================================================
# Public API
# =============================================================================

def run_all_checks(project_dir=None, deep: bool = False, session_active: bool = False,
                   progress=None) -> list:
    """
    Run every check and return a list of CheckResult, in display order.
    deep=True adds import tests and a brief camera test (skipped while a session runs).
    progress(message) is called with short status strings, if given.
    """
    project_dir = Path(project_dir) if project_dir else Path(__file__).resolve().parent

    def say(message: str) -> None:
        if progress is not None:
            progress(message)

    say("Checking model files...")
    model_results, download_gb = check_models(project_dir)

    say("Checking the system...")
    system_results = check_system(project_dir, download_gb)
    python_results = check_python()

    say("Checking packages...")
    package_results = [_check_package(package) for package in PACKAGES]
    if deep:
        package_results = _deep_check_packages(package_results, say)

    say("Checking devices...")
    camera_index = _read_literal(project_dir / "face.py", "CAMERA_INDEX", 0)
    device_results = [check_microphone()]
    if not deep:
        device_results.append(CheckResult(
            "Devices", "Camera", SKIP,
            "Not tested in the quick check, because opening the camera turns it on briefly. "
            "Use Deep check."))
    elif session_active:
        device_results.append(CheckResult(
            "Devices", "Camera", SKIP, "Skipped: a session is running and is using the camera."))
    else:
        say("Testing the camera...")
        device_results.append(check_camera(camera_index))

    return system_results + python_results + package_results + model_results + device_results


def summarize(results: list) -> tuple:
    """Return (problem_count, warning_count)."""
    problems = sum(1 for result in results if result.status == FAIL)
    warnings = sum(1 for result in results if result.status == WARN)
    return problems, warnings


def install_commands(results: list) -> list:
    """
    Build pip commands for every package that is missing, broken or too old.
    Uses the interpreter running this program, so packages land in the right environment.
    Packages needing the same flags share one command.
    """
    by_flags = {}
    for result in results:
        if result.pip_package and result.status in (FAIL, WARN):
            packages = by_flags.setdefault(result.pip_flags, [])
            if result.pip_package not in packages:
                packages.append(result.pip_package)

    prefix = _pip_prefix()
    commands = []
    for flags, packages in by_flags.items():
        parts = [prefix, "install"] + ([flags] if flags else []) + packages
        commands.append(" ".join(parts))
    return commands


# =============================================================================
# System and Python
# =============================================================================

def check_system(project_dir: Path, download_gb: float) -> list:
    results = [CheckResult("System", "Operating system", OK, platform.platform())]
    results += _check_memory()
    results.append(_check_cpu())
    results.append(_check_gpu())
    results += _check_disk(project_dir, download_gb)
    return results


def check_python() -> list:
    version_ok = sys.version_info >= MIN_PYTHON
    in_venv = sys.prefix != sys.base_prefix
    return [
        CheckResult(
            "Python", "Python version", OK if version_ok else FAIL,
            f"{platform.python_version()} ({platform.machine()})",
            fix="" if version_ok else
            f"Python {MIN_PYTHON[0]}.{MIN_PYTHON[1]} or newer is needed (my assumption)."),
        CheckResult(
            "Python", "Virtual environment", OK if in_venv else WARN,
            sys.prefix if in_venv else
            "Not running inside a virtual environment. pip installs would go to the global Python.",
            fix="" if in_venv else "Activate the project's .venv before running the app."),
        CheckResult("Python", "Interpreter", OK, sys.executable),
    ]


def _memory_bytes():
    """Return (total, available) in bytes, or None when it cannot be read."""
    try:
        import psutil
        memory = psutil.virtual_memory()
        return memory.total, memory.available
    except ImportError:
        pass
    if os.name == "nt":
        try:
            return _windows_memory()
        except Exception:  # ctypes fallback only: any failure just means "unknown".
            return None
    return None


def _windows_memory():
    import ctypes

    class MemoryStatus(ctypes.Structure):
        _fields_ = [
            ("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
            ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
            ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
            ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
            ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
        ]

    status = MemoryStatus()
    status.dwLength = ctypes.sizeof(MemoryStatus)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        return None
    return status.ullTotalPhys, status.ullAvailPhys


def _check_memory() -> list:
    memory = _memory_bytes()
    if memory is None:
        return [CheckResult("System", "Memory (RAM)", WARN,
                            "Could not read memory size.", fix="Install psutil: pip install psutil")]
    total_gb, free_gb = memory[0] / 1e9, memory[1] / 1e9

    if total_gb < RAM_FAIL_GB:
        total_status, note = FAIL, "Too little for speech and video together."
    elif total_gb < RAM_WARN_GB:
        total_status = WARN
        note = ("Tight. Video, speech and Windows share this. Close other programs. "
                "Translation loads only after you press Stop.")
    else:
        total_status, note = OK, ""

    if free_gb < FREE_RAM_FAIL_GB:
        free_status = FAIL
    elif free_gb < FREE_RAM_WARN_GB:
        free_status = WARN
    else:
        free_status = OK

    return [
        CheckResult("System", "Total RAM", total_status, f"{total_gb:.1f} GB. {note}".strip()),
        CheckResult("System", "Free RAM right now", free_status, f"{free_gb:.1f} GB available.",
                    fix="" if free_status == OK else "Close browsers and other heavy programs."),
    ]


def _check_cpu() -> CheckResult:
    cores = os.cpu_count() or 0
    detail = f"{cores or 'unknown'} logical cores ({platform.processor() or 'unknown model'})"
    if 0 < cores < MIN_CORES_WARN:
        return CheckResult("System", "CPU", WARN, detail + ". Few cores: expect slow speech and video.")
    return CheckResult("System", "CPU", OK, detail)


def _check_gpu() -> CheckResult:
    executable = shutil.which("nvidia-smi")
    if executable is None:
        return CheckResult("System", "GPU", OK,
                           "No NVIDIA GPU detected. The app runs on CPU (the current draft is CPU only).")
    try:
        completed = subprocess.run(
            [executable, "--query-gpu=name,memory.total", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10, creationflags=_NO_WINDOW)
        name = completed.stdout.strip().splitlines()[0] if completed.returncode == 0 else ""
    except (subprocess.TimeoutExpired, OSError, IndexError):
        name = ""
    if not name:
        return CheckResult("System", "GPU", WARN, "nvidia-smi is present but did not report a GPU.")
    return CheckResult("System", "GPU", OK,
                       f"{name}. Note: face.py (DEVICE) and speech.py still run on CPU until changed.")


def _check_disk(project_dir: Path, download_gb: float) -> list:
    cache_anchor = _existing_parent(_hf_cache_dir())
    cache_free = shutil.disk_usage(cache_anchor).free / 1e9
    if download_gb <= 0:
        cache_status, cache_detail = (OK if cache_free >= 1.0 else WARN), \
            f"{cache_free:.1f} GB free on {cache_anchor}. All large models are already downloaded."
    else:
        if cache_free < download_gb:
            cache_status = FAIL
        elif cache_free < download_gb + 2.0:
            cache_status = WARN
        else:
            cache_status = OK
        cache_detail = (f"{cache_free:.1f} GB free on {cache_anchor}. "
                        f"First-run model downloads need about {download_gb:.1f} GB.")

    project_free = shutil.disk_usage(_existing_parent(project_dir)).free / 1e9
    project_status = OK if project_free >= PROJECT_DISK_WARN_GB else WARN
    return [
        CheckResult("System", "Disk space for models", cache_status, cache_detail,
                    fix="" if cache_status == OK else "Free up disk space, or set HF_HOME to a bigger drive."),
        CheckResult("System", "Disk space in project folder", project_status,
                    f"{project_free:.1f} GB free."),
    ]


def _existing_parent(path: Path) -> Path:
    path = Path(path)
    while not path.exists() and path != path.parent:
        path = path.parent
    return path


# =============================================================================
# Packages
# =============================================================================

def _find_module(import_name: str) -> bool:
    try:
        return importlib.util.find_spec(import_name) is not None
    except (ImportError, ValueError):
        return False


def _installed_version(dist_names: tuple):
    for dist in dist_names:
        try:
            return dist, importlib.metadata.version(dist)
        except importlib.metadata.PackageNotFoundError:
            continue
    return None, None


def _version_tuple(text: str) -> tuple:
    numbers = []
    for part in text.split("."):
        match = re.match(r"\d+", part)
        if not match:
            break
        numbers.append(int(match.group()))
    return tuple(numbers)


def _check_package(package: Package) -> CheckResult:
    label = package.pip_name or package.import_name
    importable = _find_module(package.import_name)
    dist, version = _installed_version(package.dist_names)

    def result(status, detail, fix="", pip_flags=None):
        return CheckResult(package.feature, label, status, detail, fix=fix,
                           pip_package=package.pip_name,
                           pip_flags=package.pip_flags if pip_flags is None else pip_flags)

    if not importable and version is None:
        if package.pip_name:
            fix = " ".join(filter(None, ["Install it: pip install", package.pip_flags, package.pip_name]))
            detail = " ".join(filter(None, ["Not installed.", package.note]))
        else:
            fix = package.note or "Reinstall Python with this component."
            detail = "Not installed."
        return result(FAIL, detail, fix)

    if not importable:
        return result(FAIL, f"{dist} {version} is installed but cannot be found for import.",
                      "Reinstall it with --force-reinstall.",
                      pip_flags=" ".join(filter(None, ["--force-reinstall", package.pip_flags])))

    if version is None:
        return result(OK, "Found (version unknown)." + (f" {package.note}" if package.note else ""))

    if package.min_version and _version_tuple(version) < _version_tuple(package.min_version):
        return result(WARN, f"{dist} {version} is older than {package.min_version}, which is needed.",
                      "Upgrade it.",
                      pip_flags=" ".join(filter(None, ["--upgrade", package.pip_flags])))

    return result(OK, f"{dist} {version}")


def _import_test(import_name: str) -> tuple:
    """Import a module in a separate process. Returns (worked, error_line)."""
    try:
        completed = subprocess.run(
            [sys.executable, "-c", f"import {import_name}"],
            capture_output=True, text=True, timeout=IMPORT_TEST_TIMEOUT_SECONDS,
            creationflags=_NO_WINDOW)
    except subprocess.TimeoutExpired:
        return False, f"Import took longer than {IMPORT_TEST_TIMEOUT_SECONDS} seconds."
    except OSError as error:
        return False, f"Could not start a test process: {error}"
    if completed.returncode == 0:
        return True, ""
    lines = [line for line in completed.stderr.strip().splitlines() if line.strip()]
    return False, lines[-1] if lines else "Import failed."


def _deep_check_packages(results: list, say) -> list:
    cache = {}
    checked = []
    for package, result in zip(PACKAGES, results):
        if result.status not in (OK, WARN):
            checked.append(result)  # Already known to be missing or broken.
            continue
        if package.import_name not in cache:
            say(f"Import test: {package.import_name} (large packages can take a while)...")
            cache[package.import_name] = _import_test(package.import_name)
        worked, error = cache[package.import_name]
        if worked:
            result.detail += " | import test passed"
            checked.append(result)
        else:
            checked.append(CheckResult(
                result.group, result.name, FAIL,
                f"Installed ({result.detail}) but importing it fails: {error}",
                fix="Reinstall it with --force-reinstall. If it still fails, the package may not "
                    "support this Python version.",
                pip_package=package.pip_name,
                pip_flags=" ".join(filter(None, ["--force-reinstall", package.pip_flags]))))
    return checked


def _pip_prefix() -> str:
    # PowerShell needs the call operator (&) to run a quoted path.
    if os.name == "nt":
        return f'& "{sys.executable}" -m pip'
    return f'"{sys.executable}" -m pip'


# =============================================================================
# Model files
# =============================================================================

def _read_literal(path: Path, name: str, default):
    """Read `NAME = <literal>` from a source file without importing it (imports can be heavy)."""
    try:
        text = Path(path).read_text(encoding="utf-8")
    except OSError:
        return default
    match = re.search(rf"^{re.escape(name)}\s*=\s*(.+?)\s*(?:#.*)?$", text, re.MULTILINE)
    if not match:
        return default
    try:
        return ast.literal_eval(match.group(1))
    except (ValueError, SyntaxError):
        return default


def _hf_cache_dir() -> Path:
    try:
        from huggingface_hub import constants
        return Path(constants.HF_HUB_CACHE)
    except Exception:  # Not installed or an unexpected layout: fall back to the default.
        home = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
        return home / "hub"


def _cached_state(cache_dir: Path, repo_id: str, file_names: tuple) -> str:
    """Return "ready", "partial" or "missing" for a Hugging Face repo in the local cache."""
    folder = cache_dir / ("models--" + repo_id.replace("/", "--"))
    blobs, snapshots = folder / "blobs", folder / "snapshots"
    if not snapshots.is_dir():
        return "missing"
    for snapshot in snapshots.iterdir():
        for name in file_names:
            target = snapshot / name
            if target.exists() and target.stat().st_size > 0:
                return "ready"
    if blobs.is_dir() and any(blobs.glob("*.incomplete")):
        return "partial"
    return "missing"


def check_models(project_dir: Path) -> tuple:
    """Return (results, gigabytes_still_to_download)."""
    cache = _hf_cache_dir()
    results, to_download = [], 0.0

    # YOLO face weights: a file in the project folder, named by face.py.
    weights_name = _read_literal(project_dir / "face.py", "WEIGHTS_PATH", "yolov8n-face.pt")
    weights = Path(weights_name)
    weights = weights if weights.is_absolute() else project_dir / weights
    if weights.is_file() and weights.stat().st_size > 0:
        results.append(CheckResult("Model files", "YOLO face weights", OK,
                                   f"Found: {weights} ({weights.stat().st_size / 1e6:.1f} MB)"))
    else:
        results.append(CheckResult(
            "Model files", "YOLO face weights", FAIL,
            f"Not found at {weights}. Video cannot start without it.",
            fix="Download a YOLO face model to that path. Example (link not verified): "
                'Invoke-WebRequest -Uri "https://github.com/akanametov/yolov8-face/releases/'
                'download/v0.0.0/yolov8n-face.pt" -OutFile "yolov8n-face.pt"'))

    # Speech model (MMS).
    mms_id = _read_literal(project_dir / "speech.py", "MMS_MODEL_ID", "facebook/mms-1b-all")
    state = _cached_state(cache, mms_id, ("model.safetensors", "pytorch_model.bin"))
    results.append(_model_result(
        "Speech model (MMS)", mms_id, state, MMS_DOWNLOAD_GB,
        'python -c "from speech import MMSEngine; MMSEngine().load()"'))
    to_download += 0.0 if state == "ready" else MMS_DOWNLOAD_GB

    # Translation model (GGUF).
    repo = _read_literal(project_dir / "language.py", "MODEL_REPO",
                         "nielle003/Gemma_3_4B_Cebuano_Ilokano_Tagalog")
    file_name = _read_literal(project_dir / "language.py", "MODEL_FILE", "gemma-3-4b-FINAL.gguf")
    state = _cached_state(cache, repo, (file_name,))
    results.append(_model_result(
        "Translation model (GGUF)", repo, state, GGUF_DOWNLOAD_GB,
        'python -c "from language import load_model; load_model()"'))
    to_download += 0.0 if state == "ready" else GGUF_DOWNLOAD_GB

    return results, to_download


def _model_result(name: str, repo: str, state: str, size_gb: float, command: str) -> CheckResult:
    if state == "ready":
        return CheckResult("Model files", name, OK, f"Downloaded ({repo}).")
    if state == "partial":
        detail = f"Incomplete download ({repo}). It resumes on next use, about {size_gb:.1f} GB."
    else:
        detail = (f"Not downloaded yet ({repo}). The first use downloads about {size_gb:.1f} GB. "
                  "Do it before a live session, not during one.")
    return CheckResult("Model files", name, WARN, detail, fix="Download it now: " + command)


# =============================================================================
# Devices
# =============================================================================

def check_microphone() -> CheckResult:
    """Lists audio devices. Does not open the microphone, so it is safe during a session."""
    try:
        import sounddevice
    except ImportError:
        return CheckResult("Devices", "Microphone", WARN,
                           "Cannot check: sounddevice is not installed.")
    try:
        devices = sounddevice.query_devices()
        inputs = [device for device in devices if device["max_input_channels"] > 0]
        if not inputs:
            return CheckResult("Devices", "Microphone", FAIL, "No audio input device found.",
                               fix="Connect a microphone and check Windows sound settings.")
        try:
            default_name = sounddevice.query_devices(kind="input")["name"]
        except Exception:  # No default set: still fine, we have inputs.
            default_name = inputs[0]["name"]
        return CheckResult("Devices", "Microphone", OK,
                           f"{len(inputs)} input device(s). Default: {default_name}")
    except Exception as error:  # PortAudio can raise several unrelated errors.
        return CheckResult("Devices", "Microphone", FAIL, f"Could not list audio devices: {error}")


def check_camera(source) -> CheckResult:
    """Opens the camera briefly. Turns the camera on for a moment."""
    if isinstance(source, str):
        found = Path(source).is_file()
        return CheckResult("Devices", "Video source", OK if found else FAIL,
                           f"File {'found' if found else 'not found'}: {source}")
    try:
        import cv2
    except ImportError:
        return CheckResult("Devices", "Camera", WARN, "Cannot test: opencv is not installed.")

    capture = cv2.VideoCapture(source)
    try:
        if not capture.isOpened():
            return CheckResult("Devices", "Camera", FAIL,
                               f"Camera {source} could not be opened.",
                               fix="Check it is connected, allowed in Windows privacy settings, "
                                   "and not used by another app.")
        success, frame = capture.read()
        if not success:
            return CheckResult("Devices", "Camera", FAIL,
                               f"Camera {source} opened but gave no frame.")
        height, width = frame.shape[:2]
        return CheckResult("Devices", "Camera", OK, f"Camera {source} works ({width}x{height}).")
    finally:
        capture.release()


# =============================================================================
# Command line
# =============================================================================

def _main() -> None:
    deep = "--deep" in sys.argv
    results = run_all_checks(deep=deep, progress=lambda message: print(f"  ... {message}"))
    labels = {OK: "OK  ", WARN: "WARN", FAIL: "FAIL", SKIP: "--  "}
    group = None
    for result in results:
        if result.group != group:
            group = result.group
            print(f"\n{group}")
        print(f"  [{labels[result.status]}] {result.name}: {result.detail}")
        if result.fix and result.status in (WARN, FAIL):
            print(f"         fix: {result.fix}")
    problems, warnings = summarize(results)
    print(f"\n{problems} problem(s), {warnings} warning(s)")
    commands = install_commands(results)
    if commands:
        print("\nTo install what is missing:")
        for command in commands:
            print("  " + command)


if __name__ == "__main__":
    _main()