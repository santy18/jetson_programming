#!/usr/bin/env python3
"""
Jetson CSI: Person detection + Face detection overlay

What it does:
- Detects PEOPLE using a DNN model (YOLO via Ultralytics if available, otherwise OpenCV DNN).
- Detects FACES using Haar cascades (fast, zero extra deps beyond OpenCV data).
- Shows a top status bar:
    "PERSON: YES/NO (n) | FACES: YES/NO (m)"

Keys:
  q / ESC  -> quit

Notes:
- Person detection needs a model.
  Option A (recommended): Ultralytics YOLO (pip install ultralytics) -> uses YOLOv8n by default.
  Option B: OpenCV DNN with YOLOv5/YOLOv8 ONNX that you provide locally.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2


# ----------------------------
# Camera / UI config
# ----------------------------

@dataclass(frozen=True)
class CameraConfig:
    capture_width: int = 1920
    capture_height: int = 1080
    display_width: int = 960
    display_height: int = 540
    framerate: int = 30
    flip_method: int = 0
    drop_frames: bool = True


@dataclass(frozen=True)
class FaceConfig:
    cascade_path: str = "/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml"
    scale_factor: float = 1.3
    min_neighbors: int = 5


@dataclass(frozen=True)
class DrawConfig:
    status_height: int = 40
    thickness: int = 2

    # BGR colors
    person_color: Tuple[int, int, int] = (0, 200, 255)  # orange-ish
    face_color: Tuple[int, int, int] = (255, 0, 0)      # blue
    status_bg: Tuple[int, int, int] = (0, 0, 0)
    status_text: Tuple[int, int, int] = (255, 255, 255)


# ----------------------------
# GStreamer pipeline
# ----------------------------

def gstreamer_pipeline(cfg: CameraConfig) -> str:
    drop = "true" if cfg.drop_frames else "false"
    return (
        "nvarguscamerasrc ! "
        f"video/x-raw(memory:NVMM), width=(int){cfg.capture_width}, height=(int){cfg.capture_height}, "
        f"framerate=(fraction){cfg.framerate}/1 ! "
        f"nvvidconv flip-method={cfg.flip_method} ! "
        f"video/x-raw, width=(int){cfg.display_width}, height=(int){cfg.display_height}, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! "
        f"appsink drop={drop}"
    )


# ----------------------------
# Face detection (Haar)
# ----------------------------

def load_face_cascade(path: str) -> cv2.CascadeClassifier:
    cascade = cv2.CascadeClassifier(path)
    if cascade.empty():
        raise FileNotFoundError(
            f"Failed to load face cascade: {path}\n"
            "Verify the file exists (opencv haarcascades path)."
        )
    return cascade


def detect_faces(gray, face_cascade: cv2.CascadeClassifier, cfg: FaceConfig):
    return face_cascade.detectMultiScale(
        gray,
        scaleFactor=cfg.scale_factor,
        minNeighbors=cfg.min_neighbors,
    )


# ----------------------------
# Person detection backends
# ----------------------------

class PersonDetector:
    """Interface for a person detector."""

    def detect(self, frame_bgr):
        """
        Returns a list of detections: [(x1,y1,x2,y2,conf), ...]
        Coordinates are pixel ints in the frame coordinate space.
        """
        raise NotImplementedError


class UltralyticsYOLOPersonDetector(PersonDetector):
    """Uses Ultralytics YOLO (if installed)."""

    def __init__(self, model_name: str = "yolov8n.pt", conf: float = 0.35):
        from ultralytics import YOLO  # noqa: WPS433 (runtime optional import)

        self.model = YOLO(model_name)
        self.conf = conf

    def detect(self, frame_bgr):
        # results[0].boxes -> xyxy, conf, cls
        results = self.model.predict(frame_bgr, conf=self.conf, verbose=False)
        r0 = results[0]
        dets = []
        if r0.boxes is None:
            return dets

        # Ultralytics class 0 is "person" for COCO models
        for b in r0.boxes:
            cls = int(b.cls[0].item())
            if cls != 0:
                continue
            x1, y1, x2, y2 = b.xyxy[0].tolist()
            conf = float(b.conf[0].item())
            dets.append((int(x1), int(y1), int(x2), int(y2), conf))
        return dets


class OpenCVDNNOnnxYOLOPersonDetector(PersonDetector):
    """
    Uses OpenCV DNN with an ONNX YOLO model you provide locally.

    Expected:
      - onnx_path: path to your YOLO ONNX model
      - names_path: COCO class names file (one per line), so we can filter "person"

    This is a fallback if you don't want Ultralytics.
    """

    def __init__(
        self,
        onnx_path: str,
        names_path: str,
        conf: float = 0.35,
        nms: float = 0.45,
        input_size: int = 640,
    ):
        self.net = cv2.dnn.readNetFromONNX(onnx_path)

        # If your OpenCV build supports CUDA, you can try these:
        # self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        # self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)

        self.conf = conf
        self.nms = nms
        self.input_size = input_size

        with open(names_path, "r", encoding="utf-8") as f:
            self.names = [line.strip() for line in f if line.strip()]

        # Find the "person" class index
        if "person" not in self.names:
            raise ValueError("Class names file must include 'person'.")
        self.person_idx = self.names.index("person")

    def detect(self, frame_bgr):
        h, w = frame_bgr.shape[:2]
        blob = cv2.dnn.blobFromImage(
            frame_bgr,
            scalefactor=1 / 255.0,
            size=(self.input_size, self.input_size),
            swapRB=True,
            crop=False,
        )
        self.net.setInput(blob)
        out = self.net.forward()

        # YOLO ONNX output formats vary by exporter.
        # This parser handles common (1, N, 85) style: [x, y, w, h, obj, class...]
        if out.ndim == 3:
            out = out[0]  # (N, K)

        boxes = []
        scores = []

        for row in out:
            obj = float(row[4])
            if obj <= 0:
                continue

            class_scores = row[5:]
            cls = int(class_scores.argmax())
            if cls != self.person_idx:
                continue

            conf = obj * float(class_scores[cls])
            if conf < self.conf:
                continue

            cx, cy, bw, bh = map(float, row[:4])

            # Convert from normalized-ish model space to image space.
            # Many exporters output coords relative to input_size; this approximation works well enough
            # for a practical overlay. If your boxes look off, you may need a model-specific decoder.
            x = (cx - bw / 2) * (w / self.input_size)
            y = (cy - bh / 2) * (h / self.input_size)
            ww = bw * (w / self.input_size)
            hh = bh * (h / self.input_size)

            x1 = max(0, int(x))
            y1 = max(0, int(y))
            x2 = min(w - 1, int(x + ww))
            y2 = min(h - 1, int(y + hh))

            boxes.append([x1, y1, x2 - x1, y2 - y1])
            scores.append(conf)

        if not boxes:
            return []

        idxs = cv2.dnn.NMSBoxes(boxes, scores, self.conf, self.nms)
        dets = []
        if len(idxs) == 0:
            return dets

        for i in idxs.flatten():
            x, y, ww, hh = boxes[i]
            dets.append((x, y, x + ww, y + hh, float(scores[i])))
        return dets


def build_person_detector() -> PersonDetector:
    """
    Pick the best available backend:
      1) Ultralytics YOLO if installed
      2) OpenCV DNN ONNX YOLO if you configured paths below
    """
    # Option A: Ultralytics (recommended)
    try:
        return UltralyticsYOLOPersonDetector(model_name="yolov8n.pt", conf=0.35)
    except Exception:
        pass

    # Option B: OpenCV DNN ONNX (you must provide files)
    # Put your files somewhere like:
    #   /home/santy/models/yolo.onnx
    #   /home/santy/models/coco.names
    onnx_path = "/home/santy/models/yolo.onnx"
    names_path = "/home/santy/models/coco.names"
    return OpenCVDNNOnnxYOLOPersonDetector(
        onnx_path=onnx_path,
        names_path=names_path,
        conf=0.35,
        nms=0.45,
        input_size=640,
    )


# ----------------------------
# Drawing helpers
# ----------------------------

def draw_status_bar(frame, persons: int, faces: int, draw_cfg: DrawConfig, fps: Optional[float] = None):
    h, w = frame.shape[:2]
    bar_h = min(draw_cfg.status_height, h)
    cv2.rectangle(frame, (0, 0), (w, bar_h), draw_cfg.status_bg, -1)

    person_txt = f"PERSON: {'YES' if persons > 0 else 'NO'} ({persons})"
    face_txt = f"FACES: {'YES' if faces > 0 else 'NO'} ({faces})"
    fps_txt = f" | FPS: {fps:.1f}" if fps is not None else ""

    text = f"{person_txt} | {face_txt}{fps_txt}"
    cv2.putText(
        frame,
        text,
        (10, int(bar_h * 0.72)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        draw_cfg.status_text,
        2,
        cv2.LINE_AA,
    )


def draw_detections(frame, person_dets, face_dets, draw_cfg: DrawConfig):
    # Persons
    for (x1, y1, x2, y2, conf) in person_dets:
        cv2.rectangle(frame, (x1, y1), (x2, y2), draw_cfg.person_color, draw_cfg.thickness)
        cv2.putText(
            frame,
            f"person {conf:.2f}",
            (x1, max(0, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            draw_cfg.person_color,
            2,
            cv2.LINE_AA,
        )

    # Faces
    for (x, y, w, h) in face_dets:
        cv2.rectangle(frame, (x, y), (x + w, y + h), draw_cfg.face_color, draw_cfg.thickness)


# ----------------------------
# Main
# ----------------------------

def main():
    window_title = "Person + Face Detect"

    cam_cfg = CameraConfig(
        capture_width=1920,
        capture_height=1080,
        display_width=960,
        display_height=540,
        framerate=30,
        flip_method=0,
        drop_frames=True,
    )
    face_cfg = FaceConfig()
    draw_cfg = DrawConfig()

    person_detector = build_person_detector()
    face_cascade = load_face_cascade(face_cfg.cascade_path)

    cap = cv2.VideoCapture(gstreamer_pipeline(cam_cfg), cv2.CAP_GSTREAMER)
    if not cap.isOpened():
        raise RuntimeError("Unable to open CSI camera via GStreamer pipeline.")

    last_t = time.time()
    fps = None

    try:
        cv2.namedWindow(window_title, cv2.WINDOW_AUTOSIZE)

        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                time.sleep(0.01)
                continue

            # FPS estimate
            now = time.time()
            dt = now - last_t
            last_t = now
            if dt > 0:
                inst = 1.0 / dt
                fps = inst if fps is None else (0.9 * fps + 0.1 * inst)

            # Face detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = detect_faces(gray, face_cascade, face_cfg)

            # Person detection
            persons = person_detector.detect(frame)

            # Draw overlays
            draw_detections(frame, persons, faces, draw_cfg)
            draw_status_bar(frame, persons=len(persons), faces=len(faces), draw_cfg=draw_cfg, fps=fps)

            # Window close detection (Jetson GTK quirk workaround)
            if cv2.getWindowProperty(window_title, cv2.WND_PROP_AUTOSIZE) < 0:
                break

            cv2.imshow(window_title, frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
# ----------------------------
# End of file     