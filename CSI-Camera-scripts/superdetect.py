#!/usr/bin/env python3
"""
Jetson CSI Multi-Cascade Detector (Haar + optional LBP)

What this does
- Opens the Jetson CSI camera via a known-good GStreamer pipeline.
- Loads several useful Haar cascades (faces, eyes, profile, smile, body, plate, cat).
- Detects multiple categories at once and overlays:
  - A top status bar with counts for each detector
  - Bounding boxes + labels for each detection
- Includes live UI controls (trackbars) to enable/disable each detector and tune settings.

Notes
- Haar cascades are fast but can be noisy. Use Min neighbors + Min size to reduce false positives.
- Running many detectors at once costs FPS; use the toggles to keep only what you need.

Keys
- q or ESC: quit
"""

import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


# -----------------------------------------------------------------------------
# Cascade directory (update this to match your system)
# -----------------------------------------------------------------------------
# Common Jetson locations:
#   /usr/share/opencv4/haarcascades
#   /usr/local/share/opencv4/haarcascades
HAAR_CASCADE_DIRECTORY = "/usr/local/share/opencv4/haarcascades"

# A curated set from your list (all still loadable from the same folder)
CASCADE_FILES: Dict[str, str] = {
    # Faces
    "face_default": "haarcascade_frontalface_default.xml",
    "face_alt": "haarcascade_frontalface_alt.xml",
    "face_profile": "haarcascade_profileface.xml",
    # Eyes
    "eye": "haarcascade_eye.xml",
    "eye_glasses": "haarcascade_eye_tree_eyeglasses.xml",
    "left_eye": "haarcascade_lefteye_2splits.xml",
    "right_eye": "haarcascade_righteye_2splits.xml",
    # Expression / detail
    "smile": "haarcascade_smile.xml",
    # Bodies
    "upper_body": "haarcascade_upperbody.xml",
    "full_body": "haarcascade_fullbody.xml",
    "lower_body": "haarcascade_lowerbody.xml",
    # Plates (often noisy; keep toggleable)
    "plate_ru": "haarcascade_russian_plate_number.xml",
    "plate_ru_16": "haarcascade_license_plate_rus_16stages.xml",
    # Cats
    "cat_face": "haarcascade_frontalcatface.xml",
    "cat_face_ext": "haarcascade_frontalcatface_extended.xml",
}


# -----------------------------------------------------------------------------
# Camera / pipeline
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class CameraConfig:
    sensor_id: int = 0
    capture_width: int = 1920
    capture_height: int = 1080
    display_width: int = 960
    display_height: int = 540
    framerate: int = 30
    flip_method: int = 0


def build_jetson_csi_gstreamer_pipeline(camera: CameraConfig) -> str:
    """Build a Jetson CSI camera pipeline string compatible with cv2.VideoCapture."""
    return (
        "nvarguscamerasrc sensor-id=%d ! "
        "video/x-raw(memory:NVMM), width=(int)%d, height=(int)%d, framerate=(fraction)%d/1 ! "
        "nvvidconv flip-method=%d ! "
        "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink"
        % (
            camera.sensor_id,
            camera.capture_width,
            camera.capture_height,
            camera.framerate,
            camera.flip_method,
            camera.display_width,
            camera.display_height,
        )
    )


def open_camera_capture(pipeline: str) -> cv2.VideoCapture:
    """Open the camera using OpenCV + GStreamer."""
    capture = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
    if not capture.isOpened():
        raise RuntimeError("Unable to open camera (GStreamer).")
    return capture


def read_camera_frame(capture: cv2.VideoCapture) -> Optional[np.ndarray]:
    """Read one frame from the camera; return None on failure."""
    ok, frame = capture.read()
    if not ok or frame is None:
        return None
    return frame


# -----------------------------------------------------------------------------
# Utility helpers
# -----------------------------------------------------------------------------

def convert_bgr_to_gray(frame_bgr: np.ndarray) -> np.ndarray:
    """Convert a BGR image to grayscale."""
    return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)


def clamp_integer(value: int, minimum: int, maximum: int) -> int:
    """Clamp an integer into an inclusive range [minimum, maximum]."""
    return max(minimum, min(maximum, value))


def update_fps_estimate(previous_time: float, smoothed_fps: Optional[float]) -> Tuple[float, Optional[float]]:
    """Return (current_time, updated_smoothed_fps) using exponential smoothing."""
    current_time = time.time()
    delta_seconds = current_time - previous_time

    if delta_seconds > 0:
        instant_fps = 1.0 / delta_seconds
        smoothed_fps = instant_fps if smoothed_fps is None else (0.9 * smoothed_fps + 0.1 * instant_fps)

    return current_time, smoothed_fps


# -----------------------------------------------------------------------------
# Cascade loading + detection
# -----------------------------------------------------------------------------

def load_opencv_cascade(cascade_path: str) -> cv2.CascadeClassifier:
    """Load a cascade file and validate it loaded properly."""
    cascade = cv2.CascadeClassifier(cascade_path)
    if cascade.empty():
        raise FileNotFoundError(f"Failed to load cascade: {cascade_path}")
    return cascade


def build_cascade_fullpath(directory: str, filename: str) -> str:
    """Build a full path to a cascade file."""
    return directory.rstrip("/") + "/" + filename


def detect_rectangles(
    gray_frame: np.ndarray,
    cascade: cv2.CascadeClassifier,
    scale_factor: float,
    min_neighbors: int,
    min_size: Tuple[int, int],
) -> List[Tuple[int, int, int, int]]:
    """
    Run detectMultiScale and return list of rectangles (x, y, w, h).
    """
    rects = cascade.detectMultiScale(
        gray_frame,
        scaleFactor=scale_factor,
        minNeighbors=min_neighbors,
        minSize=min_size,
    )
    return rects if rects is not None else []


def choose_largest_rectangle(rectangles) -> Optional[Tuple[int, int, int, int]]:
    """Pick largest rectangle by area, or None if empty."""
    if rectangles is None or len(rectangles) == 0:
        return None
    return max(rectangles, key=lambda r: r[2] * r[3])


# -----------------------------------------------------------------------------
# UI (trackbars)
# -----------------------------------------------------------------------------

class LiveControls:
    """
    Trackbars for:
    - global detection tuning
    - enabling/disabling each detector
    - running detectors every N frames (to save CPU)
    """

    def __init__(self, window_name: str, detector_names: List[str]):
        self.window_name = window_name
        self.detector_names = detector_names

        cv2.namedWindow(self.window_name, cv2.WINDOW_AUTOSIZE)

        def _noop(_value: int) -> None:
            return

        # Global cadence: run all cascades every N frames
        cv2.createTrackbar("Run detection every N frames", self.window_name, 3, 30, _noop)

        # Global detection params
        # scaleFactor = 1.01..1.50 (slider 1..50 => 1.00 + v/100)
        cv2.createTrackbar("Scale factor x100 (+1.00)", self.window_name, 12, 50, _noop)
        cv2.createTrackbar("Min neighbors", self.window_name, 6, 20, _noop)
        cv2.createTrackbar("Min size (px)", self.window_name, 60, 400, _noop)

        # Toggles per detector
        # Start with common ones enabled; others disabled to keep FPS decent.
        default_on = {
            "face_default": 1,
            "eye": 1,
            "smile": 0,
            "face_profile": 0,
            "upper_body": 0,
            "full_body": 0,
            "plate_ru": 0,
            "cat_face": 0,
        }

        for name in detector_names:
            initial = default_on.get(name, 0)
            cv2.createTrackbar(f"Enable: {name}", self.window_name, initial, 1, _noop)

    def detection_interval_frames(self) -> int:
        raw = cv2.getTrackbarPos("Run detection every N frames", self.window_name)
        return max(1, raw)

    def scale_factor(self) -> float:
        raw = cv2.getTrackbarPos("Scale factor x100 (+1.00)", self.window_name)
        return max(1.01, 1.0 + (raw / 100.0))

    def min_neighbors(self) -> int:
        raw = cv2.getTrackbarPos("Min neighbors", self.window_name)
        return max(1, raw)

    def min_size(self) -> Tuple[int, int]:
        raw = cv2.getTrackbarPos("Min size (px)", self.window_name)
        size = max(20, raw)
        return (size, size)

    def is_enabled(self, detector_name: str) -> bool:
        return cv2.getTrackbarPos(f"Enable: {detector_name}", self.window_name) == 1


# -----------------------------------------------------------------------------
# Drawing
# -----------------------------------------------------------------------------

def draw_top_status_bar(frame_bgr: np.ndarray, status_text: str) -> None:
    """Draw a black status bar at the top with white text."""
    frame_height, frame_width = frame_bgr.shape[:2]
    cv2.rectangle(frame_bgr, (0, 0), (frame_width, 34), (0, 0, 0), -1)
    cv2.putText(
        frame_bgr,
        status_text,
        (10, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )


def draw_labeled_box(
    frame_bgr: np.ndarray,
    rect: Tuple[int, int, int, int],
    label: str,
    color: Tuple[int, int, int],
) -> None:
    """
    Draw a bounding box + label.
    Color is BGR.
    """
    x, y, w, h = rect
    cv2.rectangle(frame_bgr, (x, y), (x + w, y + h), color, 2)

    # Label background
    label_text = label
    (tw, th), _baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
    lx1, ly1 = x, max(0, y - (th + 10))
    lx2, ly2 = x + tw + 10, y
    cv2.rectangle(frame_bgr, (lx1, ly1), (lx2, ly2), (0, 0, 0), -1)
    cv2.putText(
        frame_bgr,
        label_text,
        (x + 5, y - 6),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        color,
        2,
        cv2.LINE_AA,
    )


def user_requested_exit(main_window_name: str, pressed_key: int) -> bool:
    """Quit if window closed or user pressed ESC/'q'."""
    if cv2.getWindowProperty(main_window_name, cv2.WND_PROP_AUTOSIZE) < 0:
        return True
    return pressed_key in (27, ord("q"))


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main() -> None:
    MAIN_WINDOW = "Multi Cascade (Main)"
    CONTROLS_WINDOW = "Controls"

    # 1) Load cascades
    detector_names = list(CASCADE_FILES.keys())
    cascades: Dict[str, cv2.CascadeClassifier] = {}

    for name, filename in CASCADE_FILES.items():
        path = build_cascade_fullpath(HAAR_CASCADE_DIRECTORY, filename)
        cascades[name] = load_opencv_cascade(path)

    # 2) Camera
    camera_config = CameraConfig()
    pipeline = build_jetson_csi_gstreamer_pipeline(camera_config)
    capture = open_camera_capture(pipeline)

    # 3) UI controls
    controls = LiveControls(CONTROLS_WINDOW, detector_names)

    # 4) State
    frame_index = 0
    previous_time = time.time()
    smoothed_fps: Optional[float] = None

    # Cache detections to avoid running cascades every frame
    cached_detections: Dict[str, List[Tuple[int, int, int, int]]] = {name: [] for name in detector_names}

    cv2.namedWindow(MAIN_WINDOW, cv2.WINDOW_AUTOSIZE)

    try:
        while True:
            frame_bgr = read_camera_frame(capture)
            if frame_bgr is None:
                continue

            previous_time, smoothed_fps = update_fps_estimate(previous_time, smoothed_fps)

            gray = convert_bgr_to_gray(frame_bgr)

            # Read global detection tuning
            detect_every_n = controls.detection_interval_frames()
            scale_factor = controls.scale_factor()
            min_neighbors = controls.min_neighbors()
            min_size = controls.min_size()

            # Run detection periodically (CPU saver)
            should_run_detection = (frame_index % detect_every_n == 0)

            if should_run_detection:
                for name in detector_names:
                    if not controls.is_enabled(name):
                        cached_detections[name] = []
                        continue

                    cached_detections[name] = detect_rectangles(
                        gray_frame=gray,
                        cascade=cascades[name],
                        scale_factor=scale_factor,
                        min_neighbors=min_neighbors,
                        min_size=min_size,
                    )

            # Draw all enabled detections
            counts = []
            for name in detector_names:
                if not controls.is_enabled(name):
                    continue

                detections = cached_detections.get(name, [])
                counts.append(f"{name}:{len(detections)}")

                # Color policy (keep it simple and readable)
                # Faces: yellow, eyes: green, bodies: cyan, plates: magenta, cats: orange, smile: white
                if "face" in name:
                    color = (0, 255, 255)
                elif "eye" in name:
                    color = (0, 255, 0)
                elif "body" in name:
                    color = (255, 255, 0)
                elif "plate" in name:
                    color = (255, 0, 255)
                elif "cat" in name:
                    color = (0, 165, 255)
                elif "smile" in name:
                    color = (255, 255, 255)
                else:
                    color = (200, 200, 200)

                for rect in detections:
                    x, y, w, h = rect
                    area = w * h
                    label = f"{name} w:{w} h:{h} area:{area}"
                    draw_labeled_box(frame_bgr, rect, label, color)

            # Status bar
            status = " | ".join(counts) if counts else "No detectors enabled"
            if smoothed_fps is not None:
                status = f"{status} | FPS:{smoothed_fps:.1f} | everyN:{detect_every_n} sf:{scale_factor:.2f} nn:{min_neighbors} ms:{min_size[0]}"

            draw_top_status_bar(frame_bgr, status)
            cv2.imshow(MAIN_WINDOW, frame_bgr)

            key = cv2.waitKey(1) & 0xFF
            if user_requested_exit(MAIN_WINDOW, key):
                break

            frame_index += 1

    finally:
        capture.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
