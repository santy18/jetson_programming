#!/usr/bin/env python3
# MIT License
# Copyright (c) 2019-2022 JetsonHacks

import cv2


def gstreamer_pipeline(
    sensor_id=0,
    capture_width=1920,
    capture_height=1080,
    display_width=960,
    display_height=540,
    framerate=30,
    flip_method=0,
):
    # Match the known-working JetsonHacks pipeline (no appsink properties)
    return (
        "nvarguscamerasrc sensor-id=%d ! "
        "video/x-raw(memory:NVMM), width=(int)%d, height=(int)%d, framerate=(fraction)%d/1 ! "
        "nvvidconv flip-method=%d ! "
        "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink"
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


def load_cascade(path: str, name: str) -> cv2.CascadeClassifier:
    c = cv2.CascadeClassifier(path)
    if c.empty():
        raise FileNotFoundError(f"Failed to load {name} cascade: {path}")
    return c


def face_detect():
    window_title = "Face Detect"

    face_cascade = load_cascade(
        "/usr/local/share/opencv4/haarcascades/haarcascade_frontalface_default.xml",
        "face",
    )
    eye_cascade = load_cascade(
        "/usr/local/share/opencv4/haarcascades/haarcascade_eye.xml",
        "eye",
    )

    pipeline = gstreamer_pipeline(sensor_id=0, flip_method=0)
    print("Using pipeline:\n", pipeline)

    cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
    if not cap.isOpened():
        print("Unable to open camera")
        return

    try:
        cv2.namedWindow(window_title, cv2.WINDOW_AUTOSIZE)

        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)

            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

                roi_gray = gray[y : y + h, x : x + w]
                roi_color = frame[y : y + h, x : x + w]

                eyes = eye_cascade.detectMultiScale(roi_gray)
                for (ex, ey, ew, eh) in eyes:
                    cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

            if cv2.getWindowProperty(window_title, cv2.WND_PROP_AUTOSIZE) < 0:
                break

            cv2.imshow(window_title, frame)

            key = cv2.waitKey(10) & 0xFF
            if key in (27, ord("q")):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    face_detect()
