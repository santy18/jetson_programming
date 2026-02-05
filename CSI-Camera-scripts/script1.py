#!/usr/bin/env python3
"""
CSI camera face/eye detection on Jetson using OpenCV + GStreamer.

Improvements vs original:
- Validates cascade file paths and gives actionable errors
- Handles failed frame reads safely
- Adds optional FPS overlay + basic timing
- Makes parameters configurable in one place
- Uses a clean main() entrypoint and structured functions
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import cv2


@dataclass(frozen=True)
class CameraConfig:
    # CSI capture settings
    capture_width: int = 1920
    capture_height: int = 1080
    framerate: int = 30
    flip_method: int = 0

    # Display / processing size (smaller = faster)
    display_width: int = 960
    display_height: int = 540

    # Appsink behavior
    drop_frames: bool = True


@dataclass(frozen=True)
class DetectConfig:
    # Haar cascades (Jetson typically has these here)
    face_cascade_path: str = "/usr/local/share/opencv4/haarcascades/haarcascade_frontalface_default.xml"
    eye_cascade_path: str = "/usr/local/share/opencv4/haarcascades/haarcascade_eye.xml"

    # Face detection params
    scale_factor: float = 1.3
    min_neighbors: int = 5
    min_face_size: Tuple[int, int] = (40, 40)

    # Eye detection params (applied within face ROI)
    eye_scale_factor: float = 1.2
    eye_min_neighbors: int = 5
    min_eye_size: Tuple[int, int] = (15, 15)

    # UI
    window_title: str = "Face Detect"
    show_fps: bool = True


def gstreamer_pipeline(cfg: CameraConfig) -> str:
    """
    Return a GStreamer pipeline for the Jetson CSI camera (nvarguscamerasrc).

    Notes:
    - display_width/height sets the size delivered to OpenCV (important for speed).
    - drop=True helps keep latency down if processing falls behind.
    """
    drop = "True" if cfg.drop_frames else "False"

    return (
        "nvarguscamerasrc ! "
        "video/x-raw(memory:NVMM), "
        f"width=(int){cfg.capture_width}, height=(int){cfg.capture_height}, "
        f"framerate=(fraction){cfg.framerate}/1 ! "
        f"nvvidconv flip-method={cfg.flip_method} ! "
        "video/x-raw, "
        f"width=(int){cfg.display_width}, height=(int){cfg.display_height}, "
        "format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! "
        f"appsink drop={drop}"
    )


def _load_cascade(path: str, name: str) -> cv2.CascadeClassifier:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"{name} cascade not found: {path}\n"
            "If your OpenCV haarcascades live elsewhere, update DetectConfig paths."
        )

    cascade = cv2.CascadeClassifier(str(p))
    if cascade.empty():
        raise RuntimeError(
            f"Failed to load {name} cascade from: {path}\n"
            "The file exists but OpenCV couldn't parse it."
        )
    return cascade


def _overlay_fps(frame, fps: float) -> None:
    cv2.putText(
        frame,
        f"FPS: {fps:.1f}",
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )


def run_face_detect(cam_cfg: CameraConfig, det_cfg: DetectConfig) -> int:
    face_cascade = _load_cascade(det_cfg.face_cascade_path, "Face")
    eye_cascade = _load_cascade(det_cfg.eye_cascade_path, "Eye")

    cap = cv2.VideoCapture(gstreamer_pipeline(cam_cfg), cv2.CAP_GSTREAMER)
    if not cap.isOpened():
        print("Unable to open camera. Check CSI connection and GStreamer pipeline.")
        return 2

    cv2.namedWindow(det_cfg.window_title, cv2.WINDOW_AUTOSIZE)

    last_t = time.perf_counter()
    fps = 0.0

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                # If the camera hiccups, don't crash
                print("Warning: failed to read frame from camera.")
                time.sleep(0.02)
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=det_cfg.scale_factor,
                minNeighbors=det_cfg.min_neighbors,
                minSize=det_cfg.min_face_size,
            )

            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

                roi_gray = gray[y : y + h, x : x + w]
                roi_color = frame[y : y + h, x : x + w]

                eyes = eye_cascade.detectMultiScale(
                    roi_gray,
                    scaleFactor=det_cfg.eye_scale_factor,
                    minNeighbors=det_cfg.eye_min_neighbors,
                    minSize=det_cfg.min_eye_size,
                )

                for (ex, ey, ew, eh) in eyes:
                    cv2.rectangle(
                        roi_color,
                        (ex, ey),
                        (ex + ew, ey + eh),
                        (0, 255, 0),
                        2,
                    )

            # FPS (simple rolling estimate)
            now = time.perf_counter()
            dt = now - last_t
            last_t = now
            if dt > 0:
                fps = 0.9 * fps + 0.1 * (1.0 / dt) if fps > 0 else (1.0 / dt)

            if det_cfg.show_fps:
                _overlay_fps(frame, fps)

            # Show frame if window exists (works better on Jetson GTK than WND_PROP_VISIBLE)
            if cv2.getWindowProperty(det_cfg.window_title, cv2.WND_PROP_AUTOSIZE) < 0:
                break

            cv2.imshow(det_cfg.window_title, frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):  # ESC or q
                break

        return 0
    finally:
        cap.release()
        cv2.destroyAllWindows()


def main() -> int:
    cam_cfg = CameraConfig(
        capture_width=1920,
        capture_height=1080,
        display_width=960,
        display_height=540,
        framerate=30,
        flip_method=0,
        drop_frames=True,
    )

    det_cfg = DetectConfig(
        show_fps=True,
        # Tune these if you want fewer false positives:
        scale_factor=1.25,
        min_neighbors=6,
        min_face_size=(50, 50),
    )

    return run_face_detect(cam_cfg, det_cfg)


if __name__ == "__main__":
    raise SystemExit(main())
