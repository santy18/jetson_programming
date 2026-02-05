#!/usr/bin/env python3
"""
Face zoom + dot tracking with live UI controls (OpenCV trackbars)

What this does
- Opens the Jetson CSI camera using a known-good GStreamer pipeline.
- Detects a face periodically using an LBP cascade.
- Tracks a single point (a "dot") using optical flow between detections.
- Shows:
  1) Main window with the live camera feed, dot, and the zoom-crop rectangle
  2) Zoom window that shows a zoomed-in view around the dot
  3) Controls window with sliders to tune everything live

Keys
- q or ESC: quit
"""

import time
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np


# -----------------------------------------------------------------------------
# Cascade location
# -----------------------------------------------------------------------------
# LBP cascades are fast and often available on Jetson builds.
# If this file doesn't exist on your system, update the path.
LBP_CASCADE_DIRECTORY = "/usr/local/share/opencv4/lbpcascades"
FACE_CASCADE_FILEPATH = LBP_CASCADE_DIRECTORY + "/lbpcascade_frontalface.xml"


# -----------------------------------------------------------------------------
# Camera configuration
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class CameraConfig:
    """All settings needed to build the CSI camera pipeline."""
    sensor_id: int = 0
    capture_width: int = 1920
    capture_height: int = 1080
    display_width: int = 960
    display_height: int = 540
    framerate: int = 30
    flip_method: int = 0


def build_jetson_csi_gstreamer_pipeline(camera: CameraConfig) -> str:
    """
    Build a Jetson CSI camera pipeline string compatible with cv2.VideoCapture.

    Why this exists:
    - The pipeline string is long and easy to mess up.
    - Keeping it in one function makes it easy to reuse and modify safely.
    """
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
    """
    Open the camera using OpenCV + GStreamer.

    Raises:
    - RuntimeError if the camera cannot be opened.
    """
    capture = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
    if not capture.isOpened():
        raise RuntimeError("Unable to open camera (GStreamer).")
    return capture


def read_camera_frame(capture: cv2.VideoCapture) -> Optional[np.ndarray]:
    """
    Read one frame from the camera.

    Returns:
    - frame (np.ndarray) when successful
    - None when a frame cannot be read
    """
    ok, frame = capture.read()
    if not ok or frame is None:
        return None
    return frame


# -----------------------------------------------------------------------------
# General helpers
# -----------------------------------------------------------------------------

def convert_bgr_to_gray(frame_bgr: np.ndarray) -> np.ndarray:
    """Convert a BGR image to grayscale (required for cascades + optical flow)."""
    return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)


def clamp_integer(value: int, minimum: int, maximum: int) -> int:
    """Clamp an integer into an inclusive range [minimum, maximum]."""
    return max(minimum, min(maximum, value))


def update_fps_estimate(previous_time: float, smoothed_fps: Optional[float]) -> Tuple[float, Optional[float]]:
    """
    Update a smoothed FPS estimate.

    Returns:
    - (current_time, updated_smoothed_fps)
    """
    current_time = time.time()
    delta_seconds = current_time - previous_time

    if delta_seconds > 0:
        instant_fps = 1.0 / delta_seconds
        if smoothed_fps is None:
            smoothed_fps = instant_fps
        else:
            # Exponential smoothing to reduce jitter
            smoothed_fps = (0.9 * smoothed_fps) + (0.1 * instant_fps)

    return current_time, smoothed_fps


def crop_image_centered_at_point(
    frame_bgr: np.ndarray,
    center_x: float,
    center_y: float,
    crop_width: int,
    crop_height: int,
) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    """
    Crop a rectangle centered at (center_x, center_y), clamped to image boundaries.

    Returns:
    - (cropped_image, crop_rectangle)
    - crop_rectangle is (x1, y1, x2, y2)
    """
    frame_height, frame_width = frame_bgr.shape[:2]

    x1 = clamp_integer(int(center_x - crop_width // 2), 0, frame_width - 1)
    y1 = clamp_integer(int(center_y - crop_height // 2), 0, frame_height - 1)

    x2 = clamp_integer(x1 + crop_width, 1, frame_width)
    y2 = clamp_integer(y1 + crop_height, 1, frame_height)

    # If clamping reduced the crop size, shift the start back so we keep the requested size.
    if (x2 - x1) < crop_width:
        x1 = clamp_integer(x2 - crop_width, 0, frame_width - 1)
    if (y2 - y1) < crop_height:
        y1 = clamp_integer(y2 - crop_height, 0, frame_height - 1)

    cropped = frame_bgr[y1:y2, x1:x2]
    return cropped, (x1, y1, x2, y2)


# -----------------------------------------------------------------------------
# Live UI controls (trackbars)
# -----------------------------------------------------------------------------

class LiveControls:
    """
    A small wrapper around OpenCV trackbars.

    OpenCV trackbars only return integers; these getters convert the raw integers
    into human-meaningful values (floats, tuples, etc).
    """

    def __init__(self, controls_window_name: str):
        self.window_name = controls_window_name
        cv2.namedWindow(self.window_name, cv2.WINDOW_AUTOSIZE)

        # OpenCV requires a callback function, even if we don't use it.
        def _noop(_value: int) -> None:
            return

        # Face detection controls
        cv2.createTrackbar("Enable face detection (0/1)", self.window_name, 1, 1, _noop)
        cv2.createTrackbar("Detect every N frames", self.window_name, 6, 60, _noop)

        # Face detector tuning
        # scaleFactor = 1.01..1.50 (slider 1..50 maps to 1.00 + slider/100)
        cv2.createTrackbar("Scale factor x100 (+1.00)", self.window_name, 10, 50, _noop)
        cv2.createTrackbar("Min neighbors", self.window_name, 6, 20, _noop)
        cv2.createTrackbar("Min face size (px)", self.window_name, 60, 300, _noop)

        # Zoom controls
        # zoomScale = 1.2..6.0 (slider 12..60 maps to slider/10)
        cv2.createTrackbar("Zoom x10", self.window_name, 25, 60, _noop)

        # Tracking controls
        # smoothing alpha = 0.00..0.90 (slider 0..90 maps to slider/100)
        cv2.createTrackbar("Smoothing x100", self.window_name, 25, 90, _noop)

        # LK window size (odd sizes are typical): 9..41
        cv2.createTrackbar("Optical flow window", self.window_name, 21, 41, _noop)

    # --- face detection toggles ---

    def is_face_detection_enabled(self) -> bool:
        """Return True if the user enabled face detection."""
        return cv2.getTrackbarPos("Enable face detection (0/1)", self.window_name) == 1

    def face_detection_interval_frames(self) -> int:
        """How often to re-run face detection (in frames)."""
        raw = cv2.getTrackbarPos("Detect every N frames", self.window_name)
        return max(1, raw)

    # --- face detection tuning ---

    def face_scale_factor(self) -> float:
        """scaleFactor for detectMultiScale: higher is faster but may miss faces."""
        raw = cv2.getTrackbarPos("Scale factor x100 (+1.00)", self.window_name)
        return max(1.01, 1.0 + (raw / 100.0))

    def face_min_neighbors(self) -> int:
        """minNeighbors for detectMultiScale: higher reduces false positives."""
        raw = cv2.getTrackbarPos("Min neighbors", self.window_name)
        return max(1, raw)

    def face_min_size(self) -> Tuple[int, int]:
        """minSize for detectMultiScale: ignore faces smaller than this."""
        raw = cv2.getTrackbarPos("Min face size (px)", self.window_name)
        size = max(20, raw)
        return (size, size)

    # --- zoom tuning ---

    def zoom_scale(self) -> float:
        """
        Zoom scale:
        - Higher value = smaller crop = more zoom
        - Lower value = larger crop = less zoom
        """
        raw = cv2.getTrackbarPos("Zoom x10", self.window_name)
        return max(1.2, raw / 10.0)

    # --- tracking tuning ---

    def smoothing_alpha(self) -> float:
        """
        Exponential smoothing for the tracked dot:
        - 0.00 = no smoothing (jittery but responsive)
        - 0.50 = smoother but laggier
        """
        raw = cv2.getTrackbarPos("Smoothing x100", self.window_name)
        raw = clamp_integer(raw, 0, 90)
        return raw / 100.0

    def optical_flow_window_size(self) -> Tuple[int, int]:
        """
        LK optical flow window size:
        - Bigger window = more stable tracking, but slower and can smear motion
        - Use an odd number (OpenCV often expects/behaves best with odd sizes)
        """
        raw = cv2.getTrackbarPos("Optical flow window", self.window_name)

        if raw % 2 == 0:
            raw += 1

        raw = clamp_integer(raw, 9, 41)
        return (raw, raw)


# -----------------------------------------------------------------------------
# Face detection + tracking
# -----------------------------------------------------------------------------

def load_opencv_cascade(cascade_path: str) -> cv2.CascadeClassifier:
    """
    Load a cascade classifier and validate it loaded correctly.

    Raises:
    - FileNotFoundError if OpenCV couldn't load the cascade file.
    """
    cascade = cv2.CascadeClassifier(cascade_path)
    if cascade.empty():
        raise FileNotFoundError(f"Failed to load cascade: {cascade_path}")
    return cascade


def detect_face_rectangles(
    gray_frame: np.ndarray,
    face_cascade: cv2.CascadeClassifier,
    scale_factor: float,
    min_neighbors: int,
    min_size: Tuple[int, int],
):
    """
    Run cascade face detection.

    Returns:
    - list of rectangles: (x, y, w, h)
    """
    return face_cascade.detectMultiScale(
        gray_frame,
        scaleFactor=scale_factor,
        minNeighbors=min_neighbors,
        minSize=min_size,
    )


def pick_largest_rectangle(rectangles) -> Optional[Tuple[int, int, int, int]]:
    """
    Choose the largest detection by area.

    This is a simple policy for "which face should we track?"
    """
    if rectangles is None or len(rectangles) == 0:
        return None
    return max(rectangles, key=lambda r: r[2] * r[3])


def create_track_point(center_x: float, center_y: float) -> np.ndarray:
    """
    Create a point in the exact shape OpenCV optical flow expects.

    Shape: (1, 1, 2)
    dtype: float32
    """
    return np.array([[[center_x, center_y]]], dtype=np.float32)


def track_point_with_optical_flow(
    previous_gray: np.ndarray,
    current_gray: np.ndarray,
    point_to_track: np.ndarray,
    window_size: Tuple[int, int],
) -> Optional[np.ndarray]:
    """
    Track a single point using Lucas–Kanade optical flow.

    Returns:
    - next_point (same shape as input) when tracking succeeds
    - None when tracking fails
    """
    next_point, status, _error = cv2.calcOpticalFlowPyrLK(
        previous_gray,
        current_gray,
        point_to_track,
        None,
        winSize=window_size,
        maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
    )

    if status is None or status[0][0] != 1:
        return None

    return next_point


def smooth_tracked_point(previous_point: np.ndarray, new_point: np.ndarray, alpha: float) -> np.ndarray:
    """
    Smooth the tracked point to reduce jitter.

    alpha closer to 0:
      - more weight on previous point (smoother, laggier)
    alpha closer to 1:
      - more weight on new point (snappier, more jitter)
    """
    prev_xy = previous_point[0, 0]
    new_xy = new_point[0, 0]
    smoothed_xy = (1.0 - alpha) * prev_xy + alpha * new_xy
    return np.array([[[smoothed_xy[0], smoothed_xy[1]]]], dtype=np.float32)


# -----------------------------------------------------------------------------
# Drawing helpers
# -----------------------------------------------------------------------------

def draw_top_status_bar(frame_bgr: np.ndarray, status_text: str) -> None:
    """Draw a black status bar at the top of the frame with white text."""
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


def draw_tracking_dot(frame_bgr: np.ndarray, x: float, y: float) -> None:
    """Draw the tracked dot as a filled yellow circle."""
    cv2.circle(frame_bgr, (int(x), int(y)), 6, (0, 255, 255), -1)


def draw_zoom_crop_rectangle(frame_bgr: np.ndarray, rect: Tuple[int, int, int, int]) -> None:
    """Draw the rectangle showing which region is being zoomed."""
    x1, y1, x2, y2 = rect
    cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (0, 255, 255), 2)


def user_requested_exit(main_window_name: str, pressed_key: int) -> bool:
    """
    Return True if:
    - The user closed the main window
    - The user pressed ESC or 'q'
    """
    if cv2.getWindowProperty(main_window_name, cv2.WND_PROP_AUTOSIZE) < 0:
        return True
    return pressed_key in (27, ord("q"))


# -----------------------------------------------------------------------------
# Main application loop
# -----------------------------------------------------------------------------

def main() -> None:
    # Window titles
    MAIN_VIEW_WINDOW = "Face Track (Main)"
    ZOOM_VIEW_WINDOW = "Face Track (Zoom)"
    CONTROLS_WINDOW = "Controls"

    # Create camera + pipeline
    camera_config = CameraConfig()
    gstreamer_pipeline = build_jetson_csi_gstreamer_pipeline(camera_config)

    # Load the face detector
    face_detector = load_opencv_cascade(FACE_CASCADE_FILEPATH)

    # Open the camera feed
    camera_capture = open_camera_capture(gstreamer_pipeline)

    # Create UI sliders
    live_controls = LiveControls(CONTROLS_WINDOW)

    # Tracking state (carried from frame to frame)
    previous_gray_frame: Optional[np.ndarray] = None
    tracked_point: Optional[np.ndarray] = None  # shape (1,1,2) float32
    frame_index = 0

    # FPS state
    previous_time = time.time()
    smoothed_fps: Optional[float] = None

    # Zoom window output size (width, height)
    zoom_window_size = (600, 600)

    # Create windows
    cv2.namedWindow(MAIN_VIEW_WINDOW, cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow(ZOOM_VIEW_WINDOW, cv2.WINDOW_AUTOSIZE)

    try:
        while True:
            # 1) Read a frame from the camera
            frame_bgr = read_camera_frame(camera_capture)
            if frame_bgr is None:
                continue

            # 2) Update FPS estimate
            previous_time, smoothed_fps = update_fps_estimate(previous_time, smoothed_fps)

            # 3) Convert to grayscale for detection + tracking
            gray_frame = convert_bgr_to_gray(frame_bgr)
            frame_height, frame_width = frame_bgr.shape[:2]

            # 4) Read current UI parameters (user can change these live)
            detection_enabled = live_controls.is_face_detection_enabled()
            detect_every_n_frames = live_controls.face_detection_interval_frames()

            scale_factor = live_controls.face_scale_factor()
            min_neighbors = live_controls.face_min_neighbors()
            min_face_size = live_controls.face_min_size()

            zoom_scale = live_controls.zoom_scale()
            smoothing_alpha = live_controls.smoothing_alpha()
            optical_flow_window = live_controls.optical_flow_window_size()

            # 5) Periodically detect a face to initialize/recover tracking
            should_run_detection = detection_enabled and (
                tracked_point is None or (frame_index % detect_every_n_frames == 0)
            )

            if should_run_detection:
                face_rectangles = detect_face_rectangles(
                    gray_frame,
                    face_detector,
                    scale_factor,
                    min_neighbors,
                    min_face_size,
                )
                largest_face = pick_largest_rectangle(face_rectangles)

                if largest_face is not None:
                    x, y, w, h = largest_face
                    face_center_x = x + w / 2.0
                    face_center_y = y + h / 2.0
                    tracked_point = create_track_point(face_center_x, face_center_y)

            # 6) Track the dot using optical flow (frame-to-frame)
            if previous_gray_frame is not None and tracked_point is not None:
                next_point = track_point_with_optical_flow(
                    previous_gray_frame,
                    gray_frame,
                    tracked_point,
                    optical_flow_window,
                )

                if next_point is None:
                    # Tracking lost; wait for the next detection to reinitialize
                    tracked_point = None
                else:
                    tracked_point = smooth_tracked_point(tracked_point, next_point, smoothing_alpha)

            previous_gray_frame = gray_frame
            frame_index += 1

            # 7) Render: main view + zoom view
            if tracked_point is not None:
                dot_x, dot_y = tracked_point[0, 0]

                # Keep the dot inside the image bounds
                dot_x = float(clamp_integer(int(dot_x), 0, frame_width - 1))
                dot_y = float(clamp_integer(int(dot_y), 0, frame_height - 1))

                # Draw the dot
                draw_tracking_dot(frame_bgr, dot_x, dot_y)

                # Compute crop size based on zoom scale
                crop_width = max(1, int(frame_width / zoom_scale))
                crop_height = max(1, int(frame_height / zoom_scale))

                # Crop around the dot and show where we're cropping from
                crop_bgr, crop_rect = crop_image_centered_at_point(
                    frame_bgr,
                    dot_x,
                    dot_y,
                    crop_width,
                    crop_height,
                )
                draw_zoom_crop_rectangle(frame_bgr, crop_rect)

                # Create the zoom image
                zoom_bgr = cv2.resize(crop_bgr, zoom_window_size, interpolation=cv2.INTER_LINEAR)
                cv2.imshow(ZOOM_VIEW_WINDOW, zoom_bgr)

                status = (
                    f"Detection: {'ON' if detection_enabled else 'OFF'} | Tracking: ON | "
                    f"Dot: ({int(dot_x)}, {int(dot_y)}) | "
                    f"sf:{scale_factor:.2f} nn:{min_neighbors} ms:{min_face_size[0]} | "
                    f"zoom:{zoom_scale:.1f} smooth:{smoothing_alpha:.2f} lk:{optical_flow_window[0]}"
                )
            else:
                # If tracking is off, show a blank zoom window
                blank_zoom = np.zeros((zoom_window_size[1], zoom_window_size[0], 3), dtype=np.uint8)
                cv2.putText(blank_zoom, "No face/dot", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
                cv2.imshow(ZOOM_VIEW_WINDOW, blank_zoom)

                status = f"Detection: {'ON' if detection_enabled else 'OFF'} | Tracking: OFF"

            if smoothed_fps is not None:
                status += f" | FPS: {smoothed_fps:.1f}"

            # 8) Draw the status bar and show the main window
            draw_top_status_bar(frame_bgr, status)
            cv2.imshow(MAIN_VIEW_WINDOW, frame_bgr)

            # 9) Exit handling
            pressed_key = cv2.waitKey(1) & 0xFF
            if user_requested_exit(MAIN_VIEW_WINDOW, pressed_key):
                break

    finally:
        # Always release camera and windows cleanly
        camera_capture.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
