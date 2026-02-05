#!/usr/bin/env python3
"""
CSI Camera Face + Eye Detection (Jetson)

- Captures frames via a GStreamer pipeline (nvarguscamerasrc).
- Runs Haar cascade face detection + eye detection.
- Draws overlays and displays a live preview window.

Keys:
  q / ESC  -> quit
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import cv2


# ----------------------------
# Config
# ----------------------------

@dataclass(frozen=True)
class CameraConfig:
    capture_width: int = 1920
    capture_height: int = 1080
    display_width: int = 960
    display_height: int = 540
    framerate: int = 30
    flip_method: int = 0
    drop_frames: bool = True  # appsink drop=True reduces latency


@dataclass(frozen=True)
class DetectConfig:
    # Haar cascade parameters
    face_scale_factor: float = 1.3
    face_min_neighbors: int = 5

    eye_scale_factor: float = 1.1
    eye_min_neighbors: int = 5

    # Draw params (BGR)
    face_color: tuple[int, int, int] = (255, 0, 0)
    eye_color: tuple[int, int, int] = (0, 255, 0)
    thickness: int = 2


# Adjust paths if your OpenCV install differs
FACE_CASCADE_PATH = "/usr/local/share/opencv4/haarcascades/haarcascade_frontalface_default.xml"
EYE_CASCADE_PATH = "/usr/local/share/opencv4/haarcascades/haarcascade_eye.xml"


# ----------------------------
# GStreamer pipeline
# ----------------------------

def gstreamer_pipeline(cfg: CameraConfig) -> str:
    """
    Build a GStreamer pipeline string for Jetson CSI cameras.

    Notes:
    - nvarguscamerasrc uses the Jetson camera stack (CSI).
    - appsink drop=True: prefer lower latency over processing every frame.
    """
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
# Detection helpers
# ----------------------------

def load_cascade(path: str, name: str) -> cv2.CascadeClassifier:
    cascade = cv2.CascadeClassifier(path)
    if cascade.empty():
        raise FileNotFoundError(
            f"Failed to load {name} cascade at: {path}\n"
            "Verify the path exists, or install OpenCV haarcascades."
        )
    return cascade


def detect_faces(gray, face_cascade: cv2.CascadeClassifier, cfg: DetectConfig):
    return face_cascade.detectMultiScale(
        gray,
        scaleFactor=cfg.face_scale_factor,
        minNeighbors=cfg.face_min_neighbors,
    )


def detect_eyes(roi_gray, eye_cascade: cv2.CascadeClassifier, cfg: DetectConfig):
    return eye_cascade.detectMultiScale(
        roi_gray,
        scaleFactor=cfg.eye_scale_factor,
        minNeighbors=cfg.eye_min_neighbors,
    )


# ----------------------------
# Main loop
# ----------------------------

def run_face_detect(
    cam_cfg: CameraConfig = CameraConfig(),
    det_cfg: DetectConfig = DetectConfig(),
    window_title: str = "Face Detect",
    show_fps: bool = True,
) -> None:
    face_cascade = load_cascade(FACE_CASCADE_PATH, "face")
    eye_cascade = load_cascade(EYE_CASCADE_PATH, "eye")

    cap = cv2.VideoCapture(gstreamer_pipeline(cam_cfg), cv2.CAP_GSTREAMER)
    if not cap.isOpened():
        raise RuntimeError("Unable to open CSI camera via GStreamer pipeline.")

    last_t = time.time()
    fps = 0.0

    try:
        cv2.namedWindow(window_title, cv2.WINDOW_AUTOSIZE)

        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                # Camera can occasionally return empty frames; keep looping briefly
                time.sleep(0.01)
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = detect_faces(gray, face_cascade, det_cfg)

            for (x, y, w, h) in faces:
                cv2.rectangle(
                    frame,
                    (x, y),
                    (x + w, y + h),
                    det_cfg.face_color,
                    det_cfg.thickness,
                )

                roi_gray = gray[y : y + h, x : x + w]
                roi_color = frame[y : y + h, x : x + w]

                eyes = detect_eyes(roi_gray, eye_cascade, det_cfg)
                for (ex, ey, ew, eh) in eyes:
                    cv2.rectangle(
                        roi_color,
                        (ex, ey),
                        (ex + ew, ey + eh),
                        det_cfg.eye_color,
                        det_cfg.thickness,
                    )

            if show_fps:
                now = time.time()
                dt = now - last_t
                last_t = now
                if dt > 0:
                    fps = 0.9 * fps + 0.1 * (1.0 / dt) if fps else (1.0 / dt)
                cv2.putText(
                    frame,
                    f"FPS: {fps:.1f}  Faces: {len(faces)}",
                    (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

            # Jetson default desktop (GTK) can behave oddly with WND_PROP_VISIBLE;
            # WND_PROP_AUTOSIZE is a workable proxy for "window still exists".
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
    run_face_detect(
        cam_cfg=CameraConfig(
            capture_width=1920,
            capture_height=1080,
            display_width=960,
            display_height=540,
            framerate=30,
            flip_method=0,
            drop_frames=True,
        ),
        det_cfg=DetectConfig(
            face_scale_factor=1.3,
            face_min_neighbors=5,
            eye_scale_factor=1.1,
            eye_min_neighbors=5,
        ),
        show_fps=True,
    )
