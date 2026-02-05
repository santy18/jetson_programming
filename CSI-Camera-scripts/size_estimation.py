#!/usr/bin/env python3
"""
Single-camera distance + size estimation (Jetson Nano CSI + OpenCV)

What this does (KISS version):
- You pick an object in the video (ROI select).
- The script measures the object's "perceived width" in pixels (bbox width).
- Using known-object geometry, it estimates distance:

    distance = (known_object_width_real_units * focal_length_px) / perceived_width_px

Two ways to get focal_length_px:
1) QUICK (recommended to start):
   - Put the object at a known distance from the camera (e.g., 24 inches).
   - Select ROI once.
   - Compute focal_length_px from that snapshot.

2) CALIBRATED (better accuracy):
   - Calibrate the camera using a chessboard to get camera intrinsics.
   - Convert intrinsics to effective focal length in pixels.
   (A helper outline is included at bottom.)

Requirements:
- OpenCV with trackers (CSRT recommended). Falls back to KCF/MOSSE if needed.

Controls:
- 'f' : focal length calibration (select ROI at known distance)
- 's' : select ROI to track and estimate distance continuously
- 'c' : clear/reset tracking
- 'q' or ESC : quit

Units:
- You choose the real-world unit (inches, cm). Keep it consistent.
  If width is in inches and known distance is inches, the output is inches.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2


# -----------------------------
# 1) Jetson CSI camera pipeline
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
    """Return a GStreamer pipeline string for a Jetson CSI camera."""
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


def open_csi_camera() -> cv2.VideoCapture:
    """Open the CSI camera and return a ready-to-read VideoCapture."""
    pipe = gstreamer_pipeline(flip_method=0)
    cap = cv2.VideoCapture(pipe, cv2.CAP_GSTREAMER)
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open CSI camera with pipeline:\n{pipe}")
    return cap


# -----------------------------
# 2) Trackers (robust across OpenCV builds)
# -----------------------------
def create_tracker(preferred: str = "CSRT") -> cv2.Tracker:
    """
    Create a tracker instance.

    Notes:
    - Some OpenCV builds expose trackers under cv2.legacy.*
    - We try preferred then fall back.
    """
    preferred = preferred.upper()
    legacy = getattr(cv2, "legacy", None)

    def has(name: str) -> bool:
        return hasattr(cv2, f"Tracker{name}_create") or (legacy and hasattr(legacy, f"Tracker{name}_create"))

    def make(name: str) -> cv2.Tracker:
        if hasattr(cv2, f"Tracker{name}_create"):
            return getattr(cv2, f"Tracker{name}_create")()
        if legacy and hasattr(legacy, f"Tracker{name}_create"):
            return getattr(legacy, f"Tracker{name}_create")()
        raise RuntimeError(f"Tracker {name} not available")

    if preferred == "CSRT" and has("CSRT"):
        return make("CSRT")
    if preferred == "KCF" and has("KCF"):
        return make("KCF")

    for name in ["CSRT", "KCF", "MOSSE"]:
        if has(name):
            return make(name)

    raise RuntimeError(
        "No supported OpenCV trackers found. "
        "Install/build OpenCV with contrib tracking support."
    )


# -----------------------------
# 3) Core geometry (the math)
# -----------------------------
def compute_focal_length_px(known_distance: float, known_width: float, perceived_width_px: float) -> float:
    """
    Compute focal length (in pixels) from a known setup:

    focal_px = (perceived_width_px * known_distance) / known_width

    Args:
        known_distance: real-world distance from camera to object (your chosen units)
        known_width: real-world width of the object (same units as distance)
        perceived_width_px: object's width in the image (pixels)

    Returns:
        focal length in pixel units.
    """
    if known_width <= 0 or perceived_width_px <= 0:
        raise ValueError("known_width and perceived_width_px must be > 0")
    return (perceived_width_px * known_distance) / known_width


def estimate_distance(known_width: float, focal_px: float, perceived_width_px: float) -> float:
    """
    Estimate distance using the pinhole camera model:

    distance = (known_width * focal_px) / perceived_width_px
    """
    if perceived_width_px <= 0:
        return float("inf")
    return (known_width * focal_px) / perceived_width_px


# -----------------------------
# 4) Simple helpers (DRY)
# -----------------------------
def draw_text_with_outline(frame, text: str, org: Tuple[int, int], scale: float = 0.55) -> None:
    """Readable HUD text."""
    cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 1, cv2.LINE_AA)


def draw_bbox(frame, bbox: Tuple[int, int, int, int]) -> None:
    """Draw bbox rectangle."""
    x, y, w, h = bbox
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)


def select_roi(window_title: str, frame) -> Optional[Tuple[int, int, int, int]]:
    """
    Let the user draw a rectangle and return (x, y, w, h), or None if canceled.
    """
    roi = cv2.selectROI(window_title, frame, showCrosshair=True, fromCenter=False)
    x, y, w, h = [int(v) for v in roi]
    if w <= 0 or h <= 0:
        return None
    return (x, y, w, h)


@dataclass
class RangeState:
    """
    State for distance estimation.

    - known_width: real width of target object (units: inches/cm, your choice)
    - focal_px: computed focal length in pixels (from quick calibration or camera calibration)
    - tracker/bbox: used to keep tracking the same object over time
    """
    known_width: float = 3.0          # example: 3 inches (change this to your object)
    focal_px: Optional[float] = None  # computed after you press 'f'
    tracker: Optional[cv2.Tracker] = None
    bbox: Optional[Tuple[int, int, int, int]] = None
    lost_count: int = 0
    last_distance: Optional[float] = None


# -----------------------------
# 5) User actions
# -----------------------------
def start_tracking_from_roi(frame, roi: Tuple[int, int, int, int]) -> Tuple[cv2.Tracker, Tuple[int, int, int, int]]:
    """
    Create tracker and initialize on the selected ROI.
    """
    tracker = create_tracker("CSRT")
    tracker.init(frame, roi)
    return tracker, roi


def do_quick_focal_calibration(
    window_title: str,
    frame,
    state: RangeState,
    known_distance: float,
) -> RangeState:
    """
    Quick calibration procedure:
    - user puts the object at a known distance
    - user selects ROI (bbox width = perceived_width_px)
    - compute focal_px and store it

    This is usually "good enough" for single-object ranging.
    """
    roi = select_roi(window_title, frame)
    if roi is None:
        return state

    _, _, w_px, _ = roi
    focal_px = compute_focal_length_px(
        known_distance=known_distance,
        known_width=state.known_width,
        perceived_width_px=float(w_px),
    )
    state.focal_px = focal_px
    return state


def do_select_and_track(window_title: str, frame, state: RangeState) -> RangeState:
    """
    Select ROI then start tracking that object each frame.
    """
    roi = select_roi(window_title, frame)
    if roi is None:
        return state

    tracker, roi = start_tracking_from_roi(frame, roi)
    state.tracker = tracker
    state.bbox = roi
    state.lost_count = 0
    return state


def clear(state: RangeState) -> RangeState:
    """Reset runtime tracking (keep known_width and focal)."""
    return RangeState(known_width=state.known_width, focal_px=state.focal_px)


# -----------------------------
# 6) Main loop
# -----------------------------
def main() -> None:
    window_title = "Single-Camera Distance Estimation"
    cap = open_csi_camera()
    cv2.namedWindow(window_title, cv2.WINDOW_AUTOSIZE)

    # Configure these two values for your real object:
    # - known_width: real width of the object (in inches or cm)
    # - known_distance_for_cal: distance you will place it at for focal calibration
    state = RangeState(known_width=3.0)  # CHANGE ME (e.g., 8.5 for a US letter width is 8.5 inches)
    known_distance_for_cal = 24.0        # CHANGE ME (distance used when you press 'f')

    last_fps_t = time.monotonic()
    frames = 0
    fps = 0.0

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                continue

            # FPS estimate (simple + stable)
            now = time.monotonic()
            frames += 1
            if now - last_fps_t >= 0.5:
                fps = frames / (now - last_fps_t)
                frames = 0
                last_fps_t = now

            # Tracking update (if active)
            if state.tracker is not None and state.bbox is not None:
                ok_trk, bbox = state.tracker.update(frame)
                if ok_trk:
                    x, y, w, h = [int(v) for v in bbox]
                    state.bbox = (x, y, w, h)
                    state.lost_count = 0

                    draw_bbox(frame, state.bbox)

                    # Distance estimation requires focal_px
                    if state.focal_px is not None:
                        dist = estimate_distance(
                            known_width=state.known_width,
                            focal_px=state.focal_px,
                            perceived_width_px=float(w),
                        )
                        state.last_distance = dist

                        # Optional: "visual ruler" style readout near the box
                        draw_text_with_outline(frame, f"{dist:.2f} units", (x, max(20, y - 10)), 0.6)
                else:
                    state.lost_count += 1
                    draw_text_with_outline(
                        frame,
                        "TRACK LOST (press 's' to reselect or 'c' to clear)",
                        (10, frame.shape[0] - 15),
                        0.6,
                    )

            # HUD / instructions
            y = 20
            draw_text_with_outline(frame, "Keys: f=calibrate focal  s=select+track  c=clear  q/esc=quit", (10, y))
            y += 22
            draw_text_with_outline(frame, f"FPS: {fps:.1f}", (10, y))
            y += 22

            draw_text_with_outline(frame, f"Known object width: {state.known_width} units", (10, y))
            y += 22
            draw_text_with_outline(frame, f"Cal distance (for 'f'): {known_distance_for_cal} units", (10, y))
            y += 22

            if state.focal_px is None:
                draw_text_with_outline(frame, "Focal: not set (press 'f' to calibrate)", (10, y))
            else:
                draw_text_with_outline(frame, f"Focal(px): {state.focal_px:.1f}", (10, y))
            y += 22

            if state.last_distance is not None:
                draw_text_with_outline(frame, f"Last distance: {state.last_distance:.2f} units", (10, y))
                y += 22

            cv2.imshow(window_title, frame)

            # Input
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break

            # 'f' => quick focal calibration
            if key == ord("f"):
                # Put the object at known_distance_for_cal, then select ROI tightly around the object.
                state = do_quick_focal_calibration(
                    window_title=window_title,
                    frame=frame,
                    state=state,
                    known_distance=known_distance_for_cal,
                )

            # 's' => select ROI and track continuously
            if key == ord("s"):
                # For best results: choose an ROI that includes the full object width.
                state = do_select_and_track(window_title, frame, state)

            # 'c' => clear tracker
            if key == ord("c"):
                state = clear(state)

            # Window closed?
            if cv2.getWindowProperty(window_title, cv2.WND_PROP_AUTOSIZE) < 0:
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()


"""
-----------------------------
Optional: True camera calibration (higher accuracy)
-----------------------------

The "quick focal calibration" works well if:
- same resolution is used for calibration and runtime
- object is roughly front-facing (not angled)
- you only need approximate distance

If you want better math:
1) Print a chessboard (e.g., 9x6 inner corners) and take ~15+ images from different angles.
2) Use cv2.findChessboardCorners + cv2.calibrateCamera to get cameraMatrix.
3) focal lengths in pixels are cameraMatrix[0,0] (fx) and cameraMatrix[1,1] (fy).

Then you can set:
  focal_px = fx   (or average of fx, fy)

In practice, you'd write a small separate script:
- load calibration images
- compute cameraMatrix
- print fx, fy
- hardcode or load them in this tracker script
"""
