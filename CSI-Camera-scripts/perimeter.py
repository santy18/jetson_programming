#!/usr/bin/env python3

# https://chatgpt.com/c/69797ba2-8598-832e-9a46-372638778e71
"""
Perimeter (zone) breach detection with OpenCV-only motion boxes (Jetson Nano CSI)

What it does:
- Shows live CSI camera feed.
- You define a perimeter (polygon) by clicking points on the frame.
- The script detects motion using background subtraction.
- For each moving object, it draws a bounding box ONLY if the object's centroid is inside the polygon.

Why this matches your request:
- "Select a perimeter" => mouse-defined polygon.
- "Draw a box around a person that goes in it" => boxes around moving objects in the perimeter.
  (Note: this is motion-based, so it's "moving object" not guaranteed "person".)

Controls:
- Left-click: add polygon point
- Right-click: remove last point
- ENTER: finalize perimeter
- 'e': edit perimeter (clear points and re-draw)
- 'b': reset background model (use if lighting changes)
- 'm': toggle mask/debug view (shows motion mask)
- 'q' or ESC: quit

KISS/DRY:
- Small single-purpose helpers (camera open, polygon handling, motion detection, rendering).
- All runtime state in dataclasses.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import numpy as np

Point = Tuple[int, int]


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
# 2) UI + state
# -----------------------------
@dataclass
class PolygonState:
    """Holds perimeter definition and edit/finalized state."""
    points: List[Point] = field(default_factory=list)
    finalized: bool = False

    def clear(self) -> None:
        self.points.clear()
        self.finalized = False

    def remove_last(self) -> None:
        if self.points:
            self.points.pop()

    def add_point(self, p: Point) -> None:
        if not self.finalized:
            self.points.append(p)

    def can_finalize(self) -> bool:
        return len(self.points) >= 3

    def as_contour(self) -> Optional[np.ndarray]:
        """
        Convert points into an OpenCV contour array shape (N, 1, 2).
        Returns None if not finalized or not enough points.
        """
        if not self.finalized or len(self.points) < 3:
            return None
        pts = np.array(self.points, dtype=np.int32).reshape((-1, 1, 2))
        return pts


@dataclass
class MotionConfig:
    """
    Tunables for motion detection.

    - min_area: filters out small noise blobs
    - blur_ksize: smoothing helps stabilize mask
    - morph_iters: clean up noise and fill holes
    """
    min_area: int = 1200
    blur_ksize: int = 5
    morph_iters: int = 2


@dataclass
class AppState:
    """All runtime state for the app."""
    polygon: PolygonState = field(default_factory=PolygonState)
    show_mask: bool = False
    last_event_t: Optional[float] = None
    breach_count: int = 0


@dataclass
class MouseContext:
    """Used by the OpenCV mouse callback to modify state."""
    window_title: str
    app: AppState


def on_mouse(event, x, y, flags, param) -> None:
    """
    Mouse callback:
    - Left click adds a point.
    - Right click removes the last point.
    """
    ctx: MouseContext = param
    if event == cv2.EVENT_LBUTTONDOWN:
        ctx.app.polygon.add_point((x, y))
    elif event == cv2.EVENT_RBUTTONDOWN:
        ctx.app.polygon.remove_last()


# -----------------------------
# 3) Geometry helpers
# -----------------------------
def point_in_polygon(contour: np.ndarray, p: Point) -> bool:
    """
    Use OpenCV pointPolygonTest:
    - returns > 0 if inside
    - returns 0 if on edge
    - returns < 0 if outside
    """
    return cv2.pointPolygonTest(contour, p, False) >= 0


def bbox_centroid(bbox: Tuple[int, int, int, int]) -> Point:
    """Centroid of bbox (x, y, w, h)."""
    x, y, w, h = bbox
    return (x + w // 2, y + h // 2)


# -----------------------------
# 4) Rendering helpers (DRY)
# -----------------------------
def draw_text_with_outline(frame, text: str, org: Point, scale: float = 0.55) -> None:
    """Readable HUD text."""
    cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 1, cv2.LINE_AA)


def draw_polygon(frame, poly: PolygonState) -> None:
    """
    Draw polygon points + lines while editing.
    If finalized, draw it as a closed contour.
    """
    # Draw points
    for p in poly.points:
        cv2.circle(frame, p, 4, (0, 0, 255), -1)

    # Draw segments
    if len(poly.points) >= 2:
        for i in range(1, len(poly.points)):
            cv2.line(frame, poly.points[i - 1], poly.points[i], (0, 0, 255), 2)

    # Close polygon if finalized
    if poly.finalized and len(poly.points) >= 3:
        cv2.line(frame, poly.points[-1], poly.points[0], (0, 0, 255), 2)


def draw_bbox(frame, bbox: Tuple[int, int, int, int], inside: bool) -> None:
    """
    Draw bbox with color based on whether centroid is inside perimeter.
    """
    x, y, w, h = bbox
    color = (0, 0, 255) if inside else (0, 255, 0)  # red if inside zone, green otherwise
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

    c = bbox_centroid(bbox)
    cv2.circle(frame, c, 3, color, -1)


# -----------------------------
# 5) Motion detection pipeline (OpenCV-only)
# -----------------------------
def build_bg_subtractor() -> cv2.BackgroundSubtractor:
    """
    Create a background subtractor (MOG2).
    Good default for indoor/outdoor motion detection without ML.
    """
    return cv2.createBackgroundSubtractorMOG2(
        history=300,
        varThreshold=25,
        detectShadows=True,
    )


def motion_mask(frame_bgr: np.ndarray, bg: cv2.BackgroundSubtractor, cfg: MotionConfig) -> np.ndarray:
    """
    Create a cleaned binary motion mask.

    Steps:
    - blur (reduce noise)
    - background subtract
    - threshold (remove shadows/low values)
    - morphology (remove noise, fill holes)
    """
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    if cfg.blur_ksize > 0:
        gray = cv2.GaussianBlur(gray, (cfg.blur_ksize, cfg.blur_ksize), 0)

    fg = bg.apply(gray)  # foreground mask (0..255)

    # Remove shadows (MOG2 shadows are usually ~127)
    _, th = cv2.threshold(fg, 200, 255, cv2.THRESH_BINARY)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel, iterations=cfg.morph_iters)
    th = cv2.morphologyEx(th, cv2.MORPH_DILATE, kernel, iterations=cfg.morph_iters)

    return th


def find_motion_bboxes(mask: np.ndarray, min_area: int) -> List[Tuple[int, int, int, int]]:
    """
    Find bounding boxes for contours in the motion mask.
    Filters small contours by area.
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    bboxes: List[Tuple[int, int, int, int]] = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        bboxes.append((x, y, w, h))

    return bboxes


# -----------------------------
# 6) Main
# -----------------------------
def main() -> None:
    window_title = "Perimeter Breach (OpenCV-only)"
    cap = open_csi_camera()

    app = AppState()
    cfg = MotionConfig(min_area=1200)

    bg = build_bg_subtractor()

    cv2.namedWindow(window_title, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(window_title, on_mouse, MouseContext(window_title=window_title, app=app))

    # FPS estimation (simple)
    last_fps_t = time.monotonic()
    frames = 0
    fps = 0.0

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                continue

            now = time.monotonic()

            # FPS
            frames += 1
            if now - last_fps_t >= 0.5:
                fps = frames / (now - last_fps_t)
                frames = 0
                last_fps_t = now

            # Draw the perimeter (editing or finalized)
            draw_polygon(frame, app.polygon)

            # If we have a finalized polygon, run motion detection and draw boxes in-zone
            contour = app.polygon.as_contour()
            mask = None

            if contour is not None:
                mask = motion_mask(frame, bg, cfg)
                bboxes = find_motion_bboxes(mask, cfg.min_area)

                breach_this_frame = False

                for bbox in bboxes:
                    c = bbox_centroid(bbox)
                    inside = point_in_polygon(contour, c)
                    draw_bbox(frame, bbox, inside=inside)

                    if inside:
                        breach_this_frame = True

                # Simple event: count a breach once per "cooldown" window
                if breach_this_frame:
                    cooldown_sec = 1.0
                    if app.last_event_t is None or (now - app.last_event_t) > cooldown_sec:
                        app.breach_count += 1
                        app.last_event_t = now

            # HUD
            y = 20
            draw_text_with_outline(
                frame,
                "Mouse: L=add point  R=undo  ENTER=finalize  |  Keys: e=edit  b=reset bg  m=mask  q=quit",
                (10, y),
                0.5,
            )
            y += 22
            draw_text_with_outline(frame, f"FPS: {fps:.1f}", (10, y))
            y += 22

            if not app.polygon.finalized:
                msg = "Perimeter: EDITING (add 3+ points, press ENTER to finalize)"
            else:
                msg = "Perimeter: FINALIZED (detecting motion inside zone)"
            draw_text_with_outline(frame, msg, (10, y))
            y += 22

            draw_text_with_outline(frame, f"Breach count: {app.breach_count}", (10, y))
            y += 22

            if contour is not None and app.last_event_t is not None and (now - app.last_event_t) < 1.5:
                draw_text_with_outline(frame, "BREACH!", (10, y), 0.8)

            # Display
            if app.show_mask and mask is not None:
                cv2.imshow(window_title, mask)
            else:
                cv2.imshow(window_title, frame)

            # Input
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break

            # ENTER to finalize polygon
            if key in (13, 10):  # Enter keys vary by environment
                if not app.polygon.finalized and app.polygon.can_finalize():
                    app.polygon.finalized = True

            # Edit perimeter
            if key == ord("e"):
                app.polygon.clear()

            # Reset background model (use after big lighting change or camera move)
            if key == ord("b"):
                bg = build_bg_subtractor()

            # Toggle mask debug view
            if key == ord("m"):
                app.show_mask = not app.show_mask

            # Window closed
            if cv2.getWindowProperty(window_title, cv2.WND_PROP_AUTOSIZE) < 0:
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
