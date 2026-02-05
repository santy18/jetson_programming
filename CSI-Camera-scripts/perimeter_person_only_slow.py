#!/usr/bin/env python3
"""
Perimeter (polygon) + YOLO person-only detection (Jetson Nano CSI)

What it does:
- Live CSI camera feed.
- You draw a perimeter polygon with the mouse.
- Runs YOLO detection each frame.
- Draws boxes ONLY for detections with class == "person" whose centroid is inside the polygon.
- Counts entries (cooldown-based) and shows "BREACH" overlay.

Controls:
- Mouse Left Click: add polygon point
- Mouse Right Click: remove last point
- ENTER: finalize perimeter
- 'e': edit perimeter (clear points and redraw)
- 'q' or ESC: quit

Assumptions:
- YOLO model weights path is configurable (default: yolov8n.pt).
- This script is code-only; install/run steps are intentionally omitted.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import numpy as np

# YOLO (Ultralytics) import
from ultralytics import YOLO

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
    pipe = gstreamer_pipeline(flip_method=0)
    cap = cv2.VideoCapture(pipe, cv2.CAP_GSTREAMER)
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open CSI camera with pipeline:\n{pipe}")
    return cap


# -----------------------------
# 2) Polygon perimeter state
# -----------------------------
@dataclass
class PolygonState:
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
        if not self.finalized or len(self.points) < 3:
            return None
        return np.array(self.points, dtype=np.int32).reshape((-1, 1, 2))


@dataclass
class AppState:
    polygon: PolygonState = field(default_factory=PolygonState)

    breach_count: int = 0
    last_breach_t: Optional[float] = None

    # For display
    last_fps_t: float = field(default_factory=time.monotonic)
    frames: int = 0
    fps: float = 0.0


@dataclass
class MouseContext:
    app: AppState


def on_mouse(event, x, y, flags, param) -> None:
    ctx: MouseContext = param
    if event == cv2.EVENT_LBUTTONDOWN:
        ctx.app.polygon.add_point((x, y))
    elif event == cv2.EVENT_RBUTTONDOWN:
        ctx.app.polygon.remove_last()


# -----------------------------
# 3) Geometry helpers
# -----------------------------
def point_in_polygon(contour: np.ndarray, p: Point) -> bool:
    # >= 0 means inside or on edge
    return cv2.pointPolygonTest(contour, p, False) >= 0


def bbox_centroid_xyxy(x1: int, y1: int, x2: int, y2: int) -> Point:
    return ((x1 + x2) // 2, (y1 + y2) // 2)


# -----------------------------
# 4) Drawing helpers
# -----------------------------
def draw_text_with_outline(frame, text: str, org: Point, scale: float = 0.55) -> None:
    cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 1, cv2.LINE_AA)


def draw_polygon(frame, poly: PolygonState) -> None:
    for p in poly.points:
        cv2.circle(frame, p, 4, (0, 0, 255), -1)

    if len(poly.points) >= 2:
        for i in range(1, len(poly.points)):
            cv2.line(frame, poly.points[i - 1], poly.points[i], (0, 0, 255), 2)

    if poly.finalized and len(poly.points) >= 3:
        cv2.line(frame, poly.points[-1], poly.points[0], (0, 0, 255), 2)


def draw_person_box(frame, x1: int, y1: int, x2: int, y2: int, inside: bool, conf: float) -> None:
    # Red when inside perimeter (breach), green otherwise
    color = (0, 0, 255) if inside else (0, 255, 0)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    c = bbox_centroid_xyxy(x1, y1, x2, y2)
    cv2.circle(frame, c, 3, color, -1)

    label = f"person {conf:.2f}"
    y_text = max(20, y1 - 8)
    draw_text_with_outline(frame, label, (x1, y_text), 0.55)


# -----------------------------
# 5) YOLO helpers
# -----------------------------
def resolve_person_class_id(model: YOLO) -> int:
    """
    Find the class id for 'person' in the loaded model.
    For COCO models, this is typically 0, but we resolve it from names to be safe.
    """
    names = model.names  # dict or list depending on version
    if isinstance(names, dict):
        for k, v in names.items():
            if v == "person":
                return int(k)
    else:
        for i, v in enumerate(names):
            if v == "person":
                return int(i)
    raise RuntimeError("Model does not contain a 'person' class in model.names")


def yolo_person_detections(frame_bgr, model: YOLO, person_id: int, conf_thres: float):
    """
    Run YOLO and yield (x1, y1, x2, y2, conf) for person detections only.
    """
    # Ultralytics expects BGR images fine; it handles preprocessing internally.
    results = model.predict(frame_bgr, conf=conf_thres, verbose=False)
    r = results[0]

    if r.boxes is None or len(r.boxes) == 0:
        return []

    boxes = r.boxes
    out = []
    # boxes.xyxy: (N,4), boxes.cls: (N,), boxes.conf: (N,)
    xyxy = boxes.xyxy.cpu().numpy()
    cls = boxes.cls.cpu().numpy().astype(int)
    conf = boxes.conf.cpu().numpy()

    for (x1, y1, x2, y2), c, cf in zip(xyxy, cls, conf):
        if c != person_id:
            continue
        out.append((int(x1), int(y1), int(x2), int(y2), float(cf)))

    return out


# -----------------------------
# 6) Main
# -----------------------------
def main() -> None:
    window_title = "Perimeter Breach (YOLO person-only)"

    # Change model_path if you want a different model file
    model_path = "yolov8n.pt"

    # Detection threshold
    conf_thres = 0.35

    cap = open_csi_camera()

    model = YOLO(model_path)
    person_id = resolve_person_class_id(model)

    app = AppState()

    cv2.namedWindow(window_title, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(window_title, on_mouse, MouseContext(app=app))

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                continue

            now = time.monotonic()

            # FPS (update every 0.5s)
            app.frames += 1
            if now - app.last_fps_t >= 0.5:
                app.fps = app.frames / (now - app.last_fps_t)
                app.frames = 0
                app.last_fps_t = now

            # Draw perimeter (editing or finalized)
            draw_polygon(frame, app.polygon)

            contour = app.polygon.as_contour()
            breach_this_frame = False

            # Run YOLO only if perimeter finalized (keeps it simpler + saves compute)
            if contour is not None:
                detections = yolo_person_detections(frame, model, person_id, conf_thres)

                for x1, y1, x2, y2, conf in detections:
                    c = bbox_centroid_xyxy(x1, y1, x2, y2)
                    inside = point_in_polygon(contour, c)

                    draw_person_box(frame, x1, y1, x2, y2, inside=inside, conf=conf)

                    if inside:
                        breach_this_frame = True

                # Debounce/cooldown counting (avoid +1 per frame while person stays inside)
                if breach_this_frame:
                    cooldown_sec = 1.0
                    if app.last_breach_t is None or (now - app.last_breach_t) > cooldown_sec:
                        app.breach_count += 1
                        app.last_breach_t = now

            # HUD
            y = 20
            draw_text_with_outline(
                frame,
                "Mouse: L=add point  R=undo  ENTER=finalize  |  Keys: e=edit  q=quit",
                (10, y),
                0.5,
            )
            y += 22
            draw_text_with_outline(frame, f"FPS: {app.fps:.1f}", (10, y))
            y += 22
            draw_text_with_outline(frame, f"Model: {model_path}  conf>={conf_thres}", (10, y))
            y += 22
            draw_text_with_outline(frame, f"Breach count: {app.breach_count}", (10, y))
            y += 22

            if not app.polygon.finalized:
                draw_text_with_outline(frame, "Perimeter: EDITING (3+ points, press ENTER)", (10, y))
            else:
                draw_text_with_outline(frame, "Perimeter: FINALIZED (person-only detection active)", (10, y))
            y += 22

            if app.last_breach_t is not None and (now - app.last_breach_t) < 1.5:
                draw_text_with_outline(frame, "BREACH!", (10, y), 0.8)

            cv2.imshow(window_title, frame)

            # Input
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break

            # ENTER to finalize perimeter
            if key in (13, 10):
                if not app.polygon.finalized and app.polygon.can_finalize():
                    app.polygon.finalized = True

            # Edit perimeter
            if key == ord("e"):
                app.polygon.clear()
                # Optional: reset counts when perimeter changes
                app.breach_count = 0
                app.last_breach_t = None

            if cv2.getWindowProperty(window_title, cv2.WND_PROP_AUTOSIZE) < 0:
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
