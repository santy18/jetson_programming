#!/usr/bin/env python3
"""
Virtual Tripwire / Perimeter Breach (Jetson Nano CSI + OpenCV, no deep learning)

What this builds:
- Live CSI camera feed.
- You select an object ROI to track (CSRT/KCF/MOSSE).
- A virtual "tripwire" line is drawn in the frame.
- When the tracked object crosses the line, an event triggers:
    - direction-aware: counts A->B vs B->A crossings
    - logs timestamp + count on screen

Core concepts:
- Tracking lifecycle (ROI select, update bbox, handle lost target)
- Line equation / signed side test (point relative to a directed line)
- Crossing detection via sign change over time

Controls:
- 's' : select ROI to track
- 'c' : clear/reset tracking + counters
- 'r' : rotate tripwire orientation (vertical <-> horizontal)
- 'q' or ESC : quit

Notes:
- KISS: one tracker, one tripwire
- DRY: small helpers for math, drawing, and actions

Extensions you can add later:
- multiple lines
- polygon zones (point-in-polygon)
- trigger GPIO / webhook
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
# 2) Tracker factory (OpenCV build compatibility)
# -----------------------------
def create_tracker(preferred: str = "CSRT") -> cv2.Tracker:
    """
    Create a tracker instance.

    OpenCV API differs by version/build:
    - Some provide cv2.TrackerCSRT_create
    - Some provide cv2.legacy.TrackerCSRT_create
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
# 3) Geometry: point relative to a line
# -----------------------------
Point = Tuple[float, float]
BBox = Tuple[int, int, int, int]  # x, y, w, h


def bbox_center(bbox: BBox) -> Point:
    """Return bbox center point (cx, cy)."""
    x, y, w, h = bbox
    return (x + w / 2.0, y + h / 2.0)


def signed_side_of_line(p: Point, a: Point, b: Point) -> float:
    """
    Determine which side of the directed line A->B the point P is on.

    Uses 2D cross product sign:
      cross = (Bx-Ax, By-Ay) x (Px-Ax, Py-Ay)
    - cross > 0 => P is on one side
    - cross < 0 => P is on the other side
    - cross = 0 => P is on the line

    This avoids slopes (no divide-by-zero issues for vertical lines).
    """
    ax, ay = a
    bx, by = b
    px, py = p

    abx, aby = (bx - ax), (by - ay)
    apx, apy = (px - ax), (py - ay)

    return (abx * apy) - (aby * apx)


def sign(v: float, eps: float = 1e-6) -> int:
    """
    Convert a float to a stable sign (-1, 0, +1), with a small epsilon.
    """
    if v > eps:
        return 1
    if v < -eps:
        return -1
    return 0


# -----------------------------
# 4) Drawing helpers (DRY)
# -----------------------------
def draw_text_with_outline(frame, text: str, org: Tuple[int, int], scale: float = 0.55) -> None:
    """Readable HUD text."""
    cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 1, cv2.LINE_AA)


def draw_bbox(frame, bbox: BBox) -> None:
    """Draw bbox and its center dot."""
    x, y, w, h = bbox
    cx, cy = bbox_center(bbox)
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.circle(frame, (int(cx), int(cy)), 4, (0, 255, 0), -1)


def draw_tripwire(frame, a: Point, b: Point) -> None:
    """Draw the tripwire line and endpoints."""
    ax, ay = int(a[0]), int(a[1])
    bx, by = int(b[0]), int(b[1])

    cv2.line(frame, (ax, ay), (bx, by), (0, 0, 255), 2)
    cv2.circle(frame, (ax, ay), 5, (0, 0, 255), -1)
    cv2.circle(frame, (bx, by), 5, (0, 0, 255), -1)


# -----------------------------
# 5) State: tracking + tripwire + counters
# -----------------------------
@dataclass
class Tripwire:
    """
    Defines a virtual line segment from A to B.

    orientation:
    - "vertical": a line near the center, top->bottom
    - "horizontal": a line near the center, left->right
    """
    a: Point
    b: Point


@dataclass
class AppState:
    """
    All app state in one place (easy reset, easy extension).
    """
    tracker: Optional[cv2.Tracker] = None
    bbox: Optional[BBox] = None
    lost_count: int = 0

    # Tripwire crossing detection:
    # - prev_side: previous sign of the tracked center relative to the tripwire
    prev_side: Optional[int] = None

    # Direction-aware counters:
    count_a_to_b: int = 0
    count_b_to_a: int = 0

    # Last event info:
    last_event: Optional[str] = None
    last_event_t: Optional[float] = None


# -----------------------------
# 6) User actions (ROI select / reset / tripwire config)
# -----------------------------
def select_roi(window_title: str, frame) -> Optional[BBox]:
    """
    Let user draw ROI and return (x, y, w, h). None if canceled.
    """
    roi = cv2.selectROI(window_title, frame, showCrosshair=True, fromCenter=False)
    x, y, w, h = [int(v) for v in roi]
    if w <= 0 or h <= 0:
        return None
    return (x, y, w, h)


def start_tracking(window_title: str, frame, state: AppState) -> AppState:
    """
    Select ROI and initialize tracker.
    Resets prev_side because a new track means a new crossing history.
    """
    roi = select_roi(window_title, frame)
    if roi is None:
        return state

    tracker = create_tracker("CSRT")
    tracker.init(frame, roi)

    state.tracker = tracker
    state.bbox = roi
    state.lost_count = 0
    state.prev_side = None
    return state


def clear_all(state: AppState) -> AppState:
    """
    Reset tracker + counters + last event.
    """
    return AppState()


def make_tripwire_for_frame(frame_shape, orientation: str = "vertical") -> Tripwire:
    """
    Create a default tripwire centered in the frame.

    KISS:
    - Use a fixed line near the center.
    - You can later replace this with mouse-defined lines or multiple zones.
    """
    h, w = frame_shape[:2]
    if orientation == "horizontal":
        y = h * 0.5
        return Tripwire(a=(w * 0.15, y), b=(w * 0.85, y))
    # vertical
    x = w * 0.5
    return Tripwire(a=(x, h * 0.15), b=(x, h * 0.85))


# -----------------------------
# 7) Crossing detection (direction-aware)
# -----------------------------
def update_tripwire_crossing(state: AppState, tripwire: Tripwire, center: Point, now: float) -> None:
    """
    Determine if the tracked object crossed the tripwire since last frame.

    Approach:
    - Compute which side of the line the center is on (signed cross product).
    - Convert to sign {-1, 0, +1}.
    - A crossing is a sign change from + to - or - to +.

    Direction:
    - If previous side was -1 and now is +1 => crossed in one direction
    - If previous side was +1 and now is -1 => crossed in the other direction

    Note:
    - If the point sits exactly on the line (sign=0), we ignore it to avoid double counting.
    """
    raw = signed_side_of_line(center, tripwire.a, tripwire.b)
    current = sign(raw)

    # If we have no history yet, establish baseline.
    if state.prev_side is None:
        if current != 0:
            state.prev_side = current
        return

    # Ignore ambiguous frames where center is almost exactly on the line.
    if current == 0:
        return

    # Crossing occurs on sign flip.
    if current != state.prev_side:
        # Determine direction.
        if state.prev_side == -1 and current == 1:
            state.count_a_to_b += 1
            state.last_event = "CROSS: (-) -> (+)"
        elif state.prev_side == 1 and current == -1:
            state.count_b_to_a += 1
            state.last_event = "CROSS: (+) -> (-)"
        else:
            # Shouldn't happen with sign() returning only -1/+1,
            # but keep it for safety.
            state.last_event = "CROSS: unknown"

        state.last_event_t = now
        state.prev_side = current
        return

    # No crossing; just update side.
    state.prev_side = current


# -----------------------------
# 8) Main loop
# -----------------------------
def main() -> None:
    window_title = "Tripwire / Perimeter Breach"
    cap = open_csi_camera()
    cv2.namedWindow(window_title, cv2.WINDOW_AUTOSIZE)

    state = AppState()
    orientation = "vertical"  # press 'r' to switch
    tripwire: Optional[Tripwire] = None

    # FPS measurement (lightweight)
    last_fps_t = time.monotonic()
    frames = 0
    fps = 0.0

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                continue

            now = time.monotonic()

            # Create/refresh the default tripwire once we know frame size.
            # If resolution changes, this will adapt.
            if tripwire is None:
                tripwire = make_tripwire_for_frame(frame.shape, orientation)

            # FPS update every 0.5s (stable display)
            frames += 1
            if now - last_fps_t >= 0.5:
                fps = frames / (now - last_fps_t)
                frames = 0
                last_fps_t = now

            # ---- Tracking update ----
            if state.tracker is not None and state.bbox is not None:
                ok_trk, bbox = state.tracker.update(frame)
                if ok_trk:
                    x, y, w, h = [int(v) for v in bbox]
                    state.bbox = (x, y, w, h)
                    state.lost_count = 0

                    draw_bbox(frame, state.bbox)

                    # Crossing check based on bbox center
                    center = bbox_center(state.bbox)
                    update_tripwire_crossing(state, tripwire, center, now)
                else:
                    state.lost_count += 1
                    draw_text_with_outline(
                        frame,
                        "TRACK LOST (press 's' to reselect, 'c' to clear)",
                        (10, frame.shape[0] - 15),
                        0.6,
                    )
                    # When tracking is lost, do not update prev_side. Keep it as-is.

            # ---- Draw tripwire and HUD ----
            draw_tripwire(frame, tripwire.a, tripwire.b)

            y = 20
            draw_text_with_outline(frame, "Keys: s=select ROI  c=clear  r=rotate wire  q/esc=quit", (10, y))
            y += 22
            draw_text_with_outline(frame, f"FPS: {fps:.1f}", (10, y))
            y += 22
            draw_text_with_outline(frame, f"Tripwire orientation: {orientation}", (10, y))
            y += 22
            draw_text_with_outline(frame, f"Count A->B: {state.count_a_to_b}", (10, y))
            y += 22
            draw_text_with_outline(frame, f"Count B->A: {state.count_b_to_a}", (10, y))
            y += 22

            # Show last event briefly
            if state.last_event and state.last_event_t and (now - state.last_event_t) < 2.0:
                draw_text_with_outline(frame, state.last_event, (10, y), 0.7)

            cv2.imshow(window_title, frame)

            # ---- Input ----
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break

            if key == ord("s"):
                state = start_tracking(window_title, frame, state)

            if key == ord("c"):
                state = clear_all(state)

            if key == ord("r"):
                orientation = "horizontal" if orientation == "vertical" else "vertical"
                tripwire = make_tripwire_for_frame(frame.shape, orientation)
                # When wire changes, reset prev_side to avoid accidental "instant crossing"
                state.prev_side = None

            if cv2.getWindowProperty(window_title, cv2.WND_PROP_AUTOSIZE) < 0:
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
c