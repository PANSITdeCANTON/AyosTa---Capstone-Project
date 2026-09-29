"""
face.py - ROUGH DRAFT: YOLO face detection + tracking, with stubs for
micro-twitch analysis.

STATUS
    Stage 1 (detect, track, lock onto one face): written, UNTESTED.
    Stage 2 (micro-twitch measurement): NOT implemented. See TODO(stage2).

HOW TO RUN
    python face.py
    Controls: L = lock onto the largest visible face, U = unlock, Q = quit.

BEFORE RUNNING
    1. pip install ultralytics opencv-python
    2. Download a YOLO FACE weights file and set WEIGHTS_PATH below.
       Stock YOLO weights (yolov8n.pt) detect "person", not faces, and will
       not work here. Check the license of whichever weights you pick.
    3. Nothing else in the project (main.py, language.py) is touched by this file.

NOTES FOR THE TEAM WITH BETTER HARDWARE
    Everything marked TODO(hardware) was kept conservative for a CPU-only
    machine with ~8 GB RAM. Raise those values once you have a GPU.
"""

import os
import time
from dataclasses import dataclass

import cv2
from ultralytics import YOLO

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

WEIGHTS_PATH = "src/yolov8n-face.pt"  # TODO: confirm source and license of this file. ##AGPL-3.0 License: Free to use for personal, academic, and open-source projects. However, under copyleft rules, if you modify the code or deploy a service using it over a network, you must open-source your entire application under the same AGPL-3.0 terms.
CAMERA_INDEX = 0                  # Webcam index. Replace with a video file path to test offline.
TRACKER_CONFIG = "bytetrack.yaml" # Bundled with ultralytics. "botsort.yaml" is the alternative.

# TODO(hardware): raise IMAGE_SIZE (e.g. 640) and use a larger model on a GPU.
IMAGE_SIZE = 320
CONFIDENCE_THRESHOLD = 0.4
DEVICE = "cpu"                    # TODO(hardware): use 0 (first CUDA GPU) or "cuda".

WINDOW_NAME = "AyosTa Face Tracker"
FPS_SMOOTHING = 0.9               # Closer to 1.0 = smoother FPS readout.


# ---------------------------------------------------------------------------
# Data handed to the rest of AyosTa
# ---------------------------------------------------------------------------

@dataclass
class FaceObservation:
    """One tracked face in one frame. This is the unit other modules consume."""
    track_id: int
    box: tuple          # (x1, y1, x2, y2) in pixels
    confidence: float
    timestamp: float    # time.monotonic() seconds. Needed for any motion timing.

    @property
    def area(self) -> int:
        x1, y1, x2, y2 = self.box
        return max(0, x2 - x1) * max(0, y2 - y1)


# ---------------------------------------------------------------------------
# Setup helpers
# ---------------------------------------------------------------------------

def load_model(weights_path: str) -> YOLO:
    """Load YOLO weights, failing early with a clear message if the file is missing."""
    if not os.path.isfile(weights_path):
        raise FileNotFoundError(
            f"Face weights not found at '{weights_path}'. "
            "Download a YOLO face model and update WEIGHTS_PATH."
        )
    return YOLO(weights_path)


def open_camera(source) -> cv2.VideoCapture:
    """Open the webcam (int) or video file (str), failing early if it cannot be read."""
    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        raise RuntimeError(
            f"Could not open video source '{source}'. "
            "Check the camera index, permissions, or that no other app is using it."
        )
    # TODO(hardware): request a higher frame rate if the camera supports it,
    # for example capture.set(cv2.CAP_PROP_FPS, 60). Twitch analysis benefits from it.
    return capture


# ---------------------------------------------------------------------------
# Stage 1: detection and tracking
# ---------------------------------------------------------------------------

def detect_and_track(model: YOLO, frame) -> list:
    """
    Run YOLO detection plus the built-in tracker on one frame.

    persist=True keeps track IDs stable across calls. Returns an empty list when
    nothing is detected (ultralytics gives boxes.id = None in that case, which
    must be handled or the loop crashes).
    """
    results = model.track(
        frame,
        persist=True,
        tracker=TRACKER_CONFIG,
        imgsz=IMAGE_SIZE,
        conf=CONFIDENCE_THRESHOLD,
        device=DEVICE,
        verbose=False,
    )

    boxes = results[0].boxes
    if boxes is None or boxes.id is None:
        return []

    now = time.monotonic()
    track_ids = boxes.id.int().tolist()
    coordinates = boxes.xyxy.int().tolist()
    confidences = boxes.conf.tolist()

    return [
        FaceObservation(
            track_id=track_id,
            box=tuple(coordinate),
            confidence=confidence,
            timestamp=now,
        )
        for track_id, coordinate, confidence in zip(track_ids, coordinates, confidences)
    ]


def pick_largest(observations: list):
    """Return the biggest face, used as the default lock target. None if list is empty."""
    return max(observations, key=lambda obs: obs.area, default=None)


def crop_face(frame, box: tuple):
    """
    Crop a face from the frame with the box clamped to the image bounds.
    Returns None when the clamped box is empty, so callers must check.
    """
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = box
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(width, x2), min(height, y2)
    if x2 <= x1 or y2 <= y1:
        return None
    return frame[y1:y2, x1:x2]


# ---------------------------------------------------------------------------
# Stage 2: micro-twitch analysis (NOT IMPLEMENTED)
# ---------------------------------------------------------------------------

def analyze_motion(face_crop, observation: FaceObservation):
    """
    TODO(stage2): measure small facial motion on the LOCKED face and return
    per-frame data for the rest of AyosTa.

    Why this cannot be done with the YOLO box alone:
        The detector box jitters by several pixels between frames even when the
        face is still. A twitch is often smaller than that jitter, so box motion
        is not a usable signal.

    Suggested approach (unverified, needs testing):
        1. Run a facial landmark model on face_crop (dozens to hundreds of points).
           MediaPipe Face Mesh is the usual choice. Wheel support for Python 3.14
           was NOT checked. Verify with `pip install <pkg> --dry-run` first, or use
           a separate venv on Python 3.12 or 3.13.
        2. Normalize landmarks against the face itself (for example, relative to
           the inter-ocular distance and head pose) so that head movement and box
           jitter are subtracted out. Only local motion should remain.
        3. Keep a short rolling history per track_id (a deque of recent frames)
           and compute frame-to-frame displacement per landmark group
           (eyelid, brow, mouth corner, and so on).
        4. Smooth, then threshold. Thresholds need calibration on real footage,
           not guesses. Record labeled examples first.

    Decisions the team must make before coding this:
        - What counts as a "twitch"? Eyelid or brow flicker, muscle tremor,
          micro-expression, and small head movement need different landmarks
          and thresholds.
        - Frame rate: micro-expressions last roughly 40 to 200 ms. At 30 FPS that
          is only about 1 to 6 frames. A 60+ FPS camera and a GPU are strongly
          recommended.

    Suggested return shape (change freely):
        {"track_id": int, "timestamp": float, "twitch_score": float, "region": str}
    Return None while there is not enough history to measure.
    """
    return None  # Placeholder so the pipeline runs end to end.


def publish_result(result) -> None:
    """
    TODO(integration): hand the Stage 2 result to the rest of AyosTa.
    Options: return it to main.py, push to a queue.Queue read by another thread,
    or write to a shared store. Left empty until the team picks one.
    """
    return None


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

def draw_overlay(frame, observations: list, locked_id, fps: float) -> None:
    """Draw boxes, IDs, lock status and FPS directly onto the frame."""
    for obs in observations:
        is_locked = obs.track_id == locked_id
        color = (0, 255, 0) if is_locked else (255, 160, 0)
        x1, y1, x2, y2 = obs.box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = f"ID {obs.track_id} {obs.confidence:.2f}" + (" LOCKED" if is_locked else "")
        cv2.putText(frame, label, (x1, max(15, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    visible_ids = {obs.track_id for obs in observations}
    if locked_id is not None and locked_id not in visible_ids:
        cv2.putText(frame, f"LOCKED ID {locked_id}: LOST", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    cv2.putText(frame, f"FPS {fps:.1f}  [L] lock  [U] unlock  [Q] quit", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def run() -> None:
    model = load_model(WEIGHTS_PATH)
    capture = open_camera(CAMERA_INDEX)

    locked_id = None
    smoothed_fps = 0.0
    previous_time = time.monotonic()

    try:
        while True:
            success, frame = capture.read()
            if not success:
                # Camera unplugged or video file ended.
                print("No frame received. Stopping.")
                break

            observations = detect_and_track(model, frame)

            # Stage 2 only runs on the locked face, which keeps CPU cost down.
            if locked_id is not None:
                locked_obs = next(
                    (obs for obs in observations if obs.track_id == locked_id), None
                )
                if locked_obs is not None:
                    face_crop = crop_face(frame, locked_obs.box)
                    if face_crop is not None:
                        publish_result(analyze_motion(face_crop, locked_obs))
                # TODO(stage2): ByteTrack may assign a NEW id if the face is lost for
                # a while. Decide a re-acquire policy (for example, re-lock onto the
                # largest face near the last known position).

            # FPS readout, exponentially smoothed.
            current_time = time.monotonic()
            elapsed = max(current_time - previous_time, 1e-6)
            previous_time = current_time
            instant_fps = 1.0 / elapsed
            smoothed_fps = (
                instant_fps if smoothed_fps == 0.0
                else FPS_SMOOTHING * smoothed_fps + (1 - FPS_SMOOTHING) * instant_fps
            )

            draw_overlay(frame, observations, locked_id, smoothed_fps)
            cv2.imshow(WINDOW_NAME, frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("l"):
                target = pick_largest(observations)
                if target is not None:
                    locked_id = target.track_id
            if key == ord("u"):
                locked_id = None

    finally:
        # Always release the camera, even after an exception or Ctrl+C.
        capture.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    try:
        run()
    except (FileNotFoundError, RuntimeError) as error:
        print(f"Startup error: {error}")
    except KeyboardInterrupt:
        print("Interrupted.")