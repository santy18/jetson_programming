#!/usr/bin/env python3
"""
Jetson Nano CSI camera + OpenCV object tracker (no deep learning)

Goals:
- KISS: keep the main loop small and readable.
- DRY: consolidate repeated logic into small helpers.
- Heavy comments: explain what each section does and why.

Usage:
- Run the script. A live camera window opens.
- Press 's' to select an ROI (drag a rectangle, press ENTER/SPACE).
- Press 'c' to clear/reset the tracker.
- Press 'q' or ESC to quit.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2


# -----------------------------
# 1) Camera / GStreamer pipeline
# -----------------------------
def gstreamer_pipeline(
    sensor_id: int = 0,
    capture_width: int = 1920,
    capture_height: int = 1080,
    display_width: int = 960,
    display_height: int = 540,
    framerate: int = 30,
    flip_method: int = 0,
) -> str:
    """
    Build a GStreamer pipeline string for Jetson CSI cameras (nvarguscamerasrc).

    Why this exists:
    - Jetson CSI cameras (Raspberry Pi cam, etc.) are commonly accessed via
      nvarguscamerasrc for best performance.
    - OpenCV can open a GStreamer pipeline via cv2.VideoCapture(..., CAP_GSTREAMER).

    The pipeline:
    - nvarguscamerasrc: Jetson camera source (CSI)
    - video/x-raw(memory:NVMM): keep frames in NVIDIA memory for speed
    - nvvidconv: convert + optional flip
    - videoconvert: convert to BGR for OpenCV
    - appsink: hand frames to OpenCV

    Returns:
        A pipeline string usable by cv2.VideoCapture.
    """
    return (
        "nvarguscamerasrc sensor-id=%d ! "
        "video/x-raw(memory:NVMM), width=(int)%d, height=(int)%d, framerate=(fraction)%d/1 ! "
        "nvvidconv flip-method=%d ! "
        "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink drop=1 sync=false"
        % (
            sensor_id,
            capture_width,
            capture_height,
            framerate,
            flip_method,
            display_width,
            display_height,
        )
    )


# -----------------------------
# 2) Tracker state (single source of truth)
# -----------------------------
@dataclass
class TrackerState:
    """
    Holds everything we need to track an object over time.

    Keeping all state in one place simplifies:
    - resets (just replace TrackerState())
    - debugging (print state)
    - extending later (multiple trackers, recovery, etc.)
    """

    tracker: Optional[cv2.Tracker] = None

    # Bounding box in OpenCV convention:
    # (x, y) is top-left corner, (w, h) is width/height.
    bbox: Optional[Tuple[int, int, int, int]] = None

    # Used for speed estimation (how far the bbox center moves per second).
    last_center: Optional[Tuple[float, float]] = None
    last_t: Optional[float] = None

    # Computed outputs
    px_per_sec: float = 0.0
    lost_count: int = 0


# -----------------------------
# 3) Small math + drawing helpers (DRY)
# -----------------------------
def bbox_center(bbox: Tuple[int, int, int, int]) -> Tuple[float, float]:
    """
    Compute center (cx, cy) for a bounding box.

    Args:
        bbox: (x, y, w, h)

    Returns:
        (cx, cy) in pixels
    """
    x, y, w, h = bbox
    return (x + w / 2.0, y + h / 2.0)


def update_speed_estimate(state: TrackerState, now: float) -> None:
    """
    Update px_per_sec based on center movement since last update.

    KISS approach:
    - compute distance in pixels between centers
    - divide by dt (seconds)
    - store into state

    If we don't have a previous sample, we simply skip calculation.
    """
    if state.bbox is None:
        return

    cx, cy = bbox_center(state.bbox)

    if state.last_center is not None and state.last_t is not None:
        dt = max(1e-6, now - state.last_t)  # protect against dt=0
        dx = cx - state.last_center[0]
        dy = cy - state.last_center[1]
        dist = (dx * dx + dy * dy) ** 0.5
        state.px_per_sec = dist / dt

    state.last_center = (cx, cy)
    state.last_t = now


def draw_bbox(frame, bbox: Tuple[int, int, int, int]) -> None:
    """
    Draw a bounding box + center point on the frame.

    This function prevents repeating drawing code across the script.
    """
    x, y, w, h = bbox
    cx, cy = bbox_center(bbox)

    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.circle(frame, (int(cx), int(cy)), 4, (0, 255, 0), -1)


def draw_text_with_outline(frame, text: str, org: Tuple[int, int], scale: float = 0.55) -> None:
    """
    Draw readable text by rendering a white line with a black outline.

    This makes the HUD readable on bright/dark backgrounds.
    """
    cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 1, cv2.LINE_AA)


def draw_hud(frame, state: TrackerState, fps: float) -> None:
    """
    Draw the on-screen "heads up display" (instructions + status).

    Keeping this separate:
    - avoids clutter in main loop
    - makes it easier to expand UI later
    """
    lines = [
        "Keys: s=select ROI  c=clear  q/esc=quit",
        f"FPS: {fps:.1f}",
    ]

    if state.tracker and state.bbox:
        lines.append(f"Speed: {state.px_per_sec:.1f} px/s")
        lines.append(f"Lost frames: {state.lost_count}")
    else:
        lines.append("Status: idle (press 's' to select ROI)")

    y = 20
    for text in lines:
        draw_text_with_outline(frame, text, (10, y))
        y += 22


# -----------------------------
# 4) Tracker factory (handles OpenCV version differences)
# -----------------------------
def create_tracker(preferred: str = "CSRT") -> cv2.Tracker:
    """
    Create an OpenCV tracker instance.

    Why this function exists:
    - OpenCV APIs differ across versions/builds:
      * Some have trackers on cv2.*
      * Some have trackers on cv2.legacy.*
    - Jetson images can vary depending on how OpenCV was installed/built.

    Strategy:
    - Try preferred tracker first (CSRT recommended).
    - Fall back to KCF, then MOSSE if available.
    - If nothing exists, raise a clear error.

    Args:
        preferred: "CSRT" or "KCF"

    Returns:
        An initialized tracker object (not yet tracking, just constructed).
    """
    preferred = preferred.upper()
    legacy = getattr(cv2, "legacy", None)

    def has_tracker(name: str) -> bool:
        return hasattr(cv2, f"Tracker{name}_create") or (legacy and hasattr(legacy, f"Tracker{name}_create"))

    def make_tracker(name: str) -> cv2.Tracker:
        if hasattr(cv2, f"Tracker{name}_create"):
            return getattr(cv2, f"Tracker{name}_create")()
        if legacy and hasattr(legacy, f"Tracker{name}_create"):
            return getattr(legacy, f"Tracker{name}_create")()
        raise RuntimeError(f"Tracker {name} not available")

    # 1) Preferred
    if preferred == "CSRT" and has_tracker("CSRT"):
        return make_tracker("CSRT")
    if preferred == "KCF" and has_tracker("KCF"):
        return make_tracker("KCF")

    # 2) Fallbacks (order matters: CSRT is accurate, KCF faster, MOSSE fastest but less accurate)
    for name in ["CSRT", "KCF", "MOSSE"]:
        if has_tracker(name):
            return make_tracker(name)

    raise RuntimeError(
        "No supported OpenCV trackers found. "
        "Install/build OpenCV with contrib trackers (often 'opencv-contrib-python' on non-Jetson, "
        "or a Jetson-specific OpenCV build that includes tracking)."
    )


# -----------------------------
# 5) Camera open helper (KISS + single responsibility)
# -----------------------------
def open_csi_camera() -> cv2.VideoCapture:
    """
    Open the CSI camera as an OpenCV VideoCapture using the GStreamer pipeline.

    Returns:
        cv2.VideoCapture that is ready to read frames from.

    Raises:
        RuntimeError if camera can't be opened.
    """
    pipe = gstreamer_pipeline(flip_method=0)
    cap = cv2.VideoCapture(pipe, cv2.CAP_GSTREAMER)
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open CSI camera with pipeline:\n{pipe}")
    return cap


# -----------------------------
# 6) User actions (select ROI / reset) as dedicated functions (DRY)
# -----------------------------
def select_roi_and_start_tracking(window_title: str, frame, state: TrackerState) -> TrackerState:
    """
    Let the user select an ROI on the current frame and start tracking it.

    How ROI selection works:
    - cv2.selectROI blocks while the user draws a rectangle.
    - The function returns (x, y, w, h).
    - If w/h == 0, the user canceled.

    We return a new/updated state for clarity and easy resetting.
    """
    roi = cv2.selectROI(window_title, frame, showCrosshair=True, fromCenter=False)
    x, y, w, h = [int(v) for v in roi]

    if w <= 0 or h <= 0:
        return state  # user canceled; keep current state

    new_state = TrackerState()
    new_state.tracker = create_tracker("CSRT")
    new_state.bbox = (x, y, w, h)

    # Seed speed estimation samples
    new_state.last_center = bbox_center(new_state.bbox)
    new_state.last_t = time.monotonic()

    # Start the tracker on the current frame + bbox
    new_state.tracker.init(frame, new_state.bbox)
    return new_state


def clear_tracking() -> TrackerState:
    """
    Reset tracker state to "idle".
    """
    return TrackerState()


# -----------------------------
# 7) FPS estimation helper
# -----------------------------
@dataclass
class FpsCounter:
    """
    Simple FPS counter using a moving time window.

    Why:
    - cv2.getTickCount() works too, but time.monotonic() is simple and stable.
    - Update FPS ~every 0.5s so the number doesn't flicker every frame.
    """
    last_t: float = time.monotonic()
    frames: int = 0
    fps: float = 0.0

    def update(self, now: float) -> float:
        self.frames += 1
        elapsed = now - self.last_t
        if elapsed >= 0.5:
            self.fps = self.frames / elapsed
            self.frames = 0
            self.last_t = now
        return self.fps


# -----------------------------
# 8) Main loop
# -----------------------------
def main() -> None:
    window_title = "Jetson CSI Tracker (no DL)"

    # Open camera and create a window once.
    cap = open_csi_camera()
    cv2.namedWindow(window_title, cv2.WINDOW_AUTOSIZE)

    state = TrackerState()
    fps_counter = FpsCounter()

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                # If a frame read fails, skip this iteration.
                # (Common when camera starts up or under load.)
                continue

            now = time.monotonic()
            fps = fps_counter.update(now)

            # --- Tracking update (only if we have an active tracker + bbox) ---
            if state.tracker is not None and state.bbox is not None:
                ok_trk, bbox = state.tracker.update(frame)

                if ok_trk:
                    # Tracker returned a new bbox; store it in state.
                    x, y, w, h = [int(v) for v in bbox]
                    state.bbox = (x, y, w, h)

                    # Good frame => reset "lost" counter
                    state.lost_count = 0

                    # Update speed estimate and render bbox
                    update_speed_estimate(state, now)
                    draw_bbox(frame, state.bbox)
                else:
                    # Tracker could not locate the object in this frame.
                    state.lost_count += 1
                    draw_text_with_outline(
                        frame,
                        "TRACK LOST (press 's' to reselect, 'c' to clear)",
                        (10, frame.shape[0] - 15),
                        scale=0.6,
                    )

            # --- UI overlay + show ---
            draw_hud(frame, state, fps)
            cv2.imshow(window_title, frame)

            # --- Input handling ---
            key = cv2.waitKey(1) & 0xFF

            # Exit keys
            if key in (27, ord("q")):
                break

            # Select ROI to start tracking
            if key == ord("s"):
                state = select_roi_and_start_tracking(window_title, frame, state)

            # Clear/reset tracking
            if key == ord("c"):
                state = clear_tracking()

            # If the user closes the window, exit cleanly.
            # (Jetson GTK can be quirky; AUTOSIZE is commonly used for visibility checks.)
            if cv2.getWindowProperty(window_title, cv2.WND_PROP_AUTOSIZE) < 0:
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
