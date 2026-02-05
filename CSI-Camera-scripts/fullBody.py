#!/usr/bin/env python3
# MIT License
# Copyright (c) 2019-2022 JetsonHacks

import cv2


HAAR_DIR = "/usr/local/share/opencv4/haarcascades"  # change to /usr/share/opencv/haarcascades if needed


def gstreamer_pipeline(
    sensor_id=0,
    capture_width=1920,
    capture_height=1080,
    display_width=960,
    display_height=540,
    framerate=30,
    flip_method=0,
):
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


def body_detect():
    window_title = "Full Body Detect"

    body_cascade = cv2.CascadeClassifier(
        HAAR_DIR + "/haarcascade_fullbody.xml"
    )

    if body_cascade.empty():
        raise RuntimeError("Failed to load haarcascade_fullbody.xml")

    cap = cv2.VideoCapture(gstreamer_pipeline(), cv2.CAP_GSTREAMER)
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

            bodies = body_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=3,
                minSize=(60, 120),
            )

            for (x, y, w, h) in bodies:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
                cv2.putText(
                    frame,
                    "BODY",
                    (x, max(0, y - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 255),
                    2,
                )

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
    body_detect()
