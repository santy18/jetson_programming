#!/usr/bin/env python3
"""
Perimeter (polygon) + YOLO person-only detection (Jetson Nano CSI) - optimized for FPS

Primary optimizations:
- Run YOLO every N frames (DETECT_EVERY).
- Run YOLO at a smaller input size (YOLO_IMGSZ).
- Restrict detections to the "person" class via classes=[person_id].
- Optionally crop inference to the polygon bounding rect (ENABLE_ROI_CROP).

Behavior:
- You draw a perimeter polygon.
- Script draws boxes only for people inside/outside perimeter (red inside, green outside).
- Breach count increments with a cooldown while at least one person is inside.

Controls:
- Mouse Left Click: add polygon point
- Mouse Right Click: remove last point
- ENTER: finalize perimeter
- 'e': edit perimeter (clear points and redraw)
- 'q' or ESC: quit

Notes:
- This is code only. It assumes Ultralytics YOLO is installed and model weights exist.
- For Jetson Nano, expect biggest gains from YOLO_IMGSZ + DETECT_EVERY + ROI crop.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import numpy as np
from ultralytics import YOLO

Point = Tuple[int, int]


# -----------------------------
# Performance knobs (tune these)
# -----------------------------
YOLO_MODEL_PATH = "yolov8n.pt"
YOLO_CONF = 0.35
YOLO_IMGSZ = 320          # try: 256, 320, 384 (smaller = faster, less accurate)
DETECT_EVERY = 4          # run YOLO every N frames (3-6 is common)
ENABLE_ROI_CROP = True    # crop inference to polygon bounding rect (big win if zone is small)
CROP_PADDING_PX = 20      # padding around crop rect

# Camera pipeline sizing (lower capture/display = faster)
CAPTURE_W, CAPTURE_H = 1280, 720
DISPLAY_W, DISPLAY_H = 1280, 720
FRAMERATE = 30
FLIP_METHOD = 0


# -----------------------------
# 1) Jetson CSI camera pipeline
# -----------------------------
def gstreamer_pipeline(
    sensor_id: int = 0,
    capture_width: int = CAPTURE_W,
    capture_height: int = CAPTURE_H,
    display_width: int = DISPLAY_W,
    display_height: int = DISPLAY_H,
    framerate: int = FRAMERATE,
    flip_method: int = FLIP_METHOD,
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
    pipe = gstreamer_pipeline()
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

    def bounding_rect(self) -> Optional[Tuple[int, int, int, int]]:
        """
        Bounding rect around the polygon points: (x, y, w, h)
        Useful to crop inference.
        """
        if not self.finalized or len(self.points) < 3:
            return None
        contour = self.as_contour()
        x, y, w, h = cv2.boundingRect(contour)
        return (x, y, w, h)


@dataclass
class AppState:
    polygon: PolygonState = field(default_factory=PolygonState)

    breach_count: int = 0
    last_breach_t: Optional[float] = None

    # FPS measurement
    last_fps_t: float = field(default_factory=time.monotonic)
    frames: int = 0
    fps: float = 0.0

    # Detection scheduling
    frame_idx: int = 0
    last_detections_xyxy: List[Tuple[int, int, int, int, float]] = field(default_factory=list)


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
    return cv2.pointPolygonTest(contour, p, False) >= 0


def bbox_centroid_xyxy(x1: int, y1: int, x2: int, y2: int) -> Point:
    return ((x1 + x2) // 2, (y1 + y2) // 2)


def clamp_rect(x: int, y: int, w: int, h: int, frame_w: int, frame_h: int) -> Tuple[int, int, int, int]:
    """
    Clamp a rect to frame boundaries.
    """
    x = max(0, x)
    y = max(0, y)
    x2 = min(frame_w, x + w)
    y2 = min(frame_h, y + h)
    w2 = max(0, x2 - x)
    h2 = max(0, y2 - y)
    return (x, y, w2, h2)


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
    color = (0, 0, 255) if inside else (0, 255, 0)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    c = bbox_centroid_xyxy(x1, y1, x2, y2)
    cv2.circle(frame, c, 3, color, -1)

    y_text = max(20, y1 - 8)
    draw_text_with_outline(frame, f"person {conf:.2f}", (x1, y_text), 0.55)


# -----------------------------
# 5) YOLO helpers
# -----------------------------
def resolve_person_class_id(model: YOLO) -> int:
    names = model.names
    if isinstance(names, dict):
        for k, v in names.items():
            if v == "person":
                return int(k)
    else:
        for i, v in enumerate(names):
            if v == "person":
                return int(i)
    raise RuntimeError("Model does not contain a 'person' class in model.names")


def yolo_person_detections_xyxy(
    image_bgr: np.ndarray,
    model: YOLO,
    person_id: int,
    conf_thres: float,
) -> List[Tuple[int, int, int, int, float]]:
    """
    Run YOLO and return list of person detections:
      [(x1, y1, x2, y2, conf), ...]
    """
    results = model.predict(
        image_bgr,
        imgsz=YOLO_IMGSZ,
        conf=conf_thres,
        classes=[person_id],  # restrict to person to cut post-processing
        verbose=False,
    )

    r = results[0]
    if r.boxes is None or len(r.boxes) == 0:
        return []

    boxes = r.boxes
    xyxy = boxes.xyxy.cpu().numpy()
    conf = boxes.conf.cpu().numpy()

    out: List[Tuple[int, int, int, int, float]] = []
    for (x1, y1, x2, y2), cf in zip(xyxy, conf):
        out.append((int(x1), int(y1), int(x2), int(y2), float(cf)))
    return out


# -----------------------------
# 6) Breach counting (debounced)
# -----------------------------
def update_breach_counter(app: AppState, any_person_inside: bool, now: float, cooldown_sec: float = 1.0) -> None:
    """
    Increment breach_count when at least one person is inside the zone,
    but at most once per cooldown period.
    """
    if not any_person_inside:
        return

    if app.last_breach_t is None or (now - app.last_breach_t) > cooldown_sec:
        app.breach_count += 1
        app.last_breach_t = now


# -----------------------------
# 7) Main
# -----------------------------
def main() -> None:
    window_title = "Perimeter Breach (YOLO person-only, optimized)"
    cap = open_csi_camera()

    model = YOLO(YOLO_MODEL_PATH)
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

            app.frame_idx += 1

            # Draw perimeter
            draw_polygon(frame, app.polygon)

            contour = app.polygon.as_contour()
            any_person_inside = False

            # Detection only once polygon is finalized
            if contour is not None:
                # Decide whether to run YOLO this frame
                run_detector = (app.frame_idx % DETECT_EVERY) == 0

                if run_detector:
                    if ENABLE_ROI_CROP:
                        # Crop inference to polygon bounding rect (big speed gain if zone is small)
                        rect = app.polygon.bounding_rect()
                        if rect is not None:
                            x, y, w, h = rect
                            x -= CROP_PADDING_PX
                            y -= CROP_PADDING_PX
                            w += 2 * CROP_PADDING_PX
                            h += 2 * CROP_PADDING_PX

                            fh, fw = frame.shape[:2]
                            x, y, w, h = clamp_rect(x, y, w, h, fw, fh)

                            if w > 0 and h > 0:
                                crop = frame[y : y + h, x : x + w]
                                dets = yolo_person_detections_xyxy(crop, model, person_id, YOLO_CONF)

                                # Map crop coords back to full-frame coords
                                mapped = [(x1 + x, y1 + y, x2 + x, y2 + y, cf) for (x1, y1, x2, y2, cf) in dets]
                                app.last_detections_xyxy = mapped
                            else:
                                app.last_detections_xyxy = []
                        else:
                            app.last_detections_xyxy = []
                    else:
                        app.last_detections_xyxy = yolo_person_detections_xyxy(frame, model, person_id, YOLO_CONF)

                # Use the last detections (whether fresh or from previous run)
                for x1, y1, x2, y2, conf in app.last_detections_xyxy:
                    c = bbox_centroid_xyxy(x1, y1, x2, y2)
                    inside = point_in_polygon(contour, c)
                    draw_person_box(frame, x1, y1, x2, y2, inside=inside, conf=conf)

                    if inside:
                        any_person_inside = True

                update_breach_counter(app, any_person_inside, now, cooldown_sec=1.0)

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
            draw_text_with_outline(
                frame,
                f"Model: {YOLO_MODEL_PATH}  imgsz={YOLO_IMGSZ}  detect_every={DETECT_EVERY}  roi_crop={ENABLE_ROI_CROP}",
                (10, y),
                0.45,
            )
            y += 22
            draw_text_with_outline(frame, f"Breach count: {app.breach_count}", (10, y))
            y += 22

            if not app.polygon.finalized:
                draw_text_with_outline(frame, "Perimeter: EDITING (3+ points, press ENTER)", (10, y))
            else:
                draw_text_with_outline(frame, "Perimeter: FINALIZED (person-only detection active)", (10, y))
            y += 22

            if app.last_breach_t is not None and (now - app.last_breach_t) < 1.2:
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
                    # Reset detection cache so first run is immediate
                    app.frame_idx = 0
                    app.last_detections_xyxy = []

            # Edit perimeter
            if key == ord("e"):
                app.polygon.clear()
                app.breach_count = 0
                app.last_breach_t = None
                app.last_detections_xyxy = []

            if cv2.getWindowProperty(window_title, cv2.WND_PROP_AUTOSIZE) < 0:
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
