import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import cv2
import cvzone
from ultralytics import YOLO
import math
import itertools


def process_frame(frame, model):
    results = model(frame)
    boxes_info = []
    centers = []
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = math.ceil(box.conf[0] * 100) / 100
            class_id = model.names[int(box.cls[0])]
            if conf > 0 and class_id == "person":
                boxes_info.append((x1, y1, x2, y2, conf, class_id))
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                centers.append((center_x, center_y))
    return frame, boxes_info, centers


def find_closest_path(centers):
    if not centers:
        return []

    unvisited = centers.copy()
    path = []

    current = unvisited.pop(0)
    path.append(current)

    while unvisited:
        closest = min(
            unvisited, key=lambda point: math.hypot(point[0] - current[0], point[1] - current[1])
        )
        path.append(closest)
        unvisited.remove(closest)
        current = closest

    return path


videos = [
    "rtsp://admin:oracle2015@10.5.0.239:554/Streaming/Channels/1",
    r"D:\NWR\videos\antre1.mp4",
]
video = videos[1]
cap = cv2.VideoCapture(video)
model = YOLO("yolo11l.pt")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Tidak dapat membaca frame dari video.")
        break

    frame_results, boxes_info, centers = process_frame(frame, model)

    for x1, y1, x2, y2, conf, class_id in boxes_info:
        cvzone.cornerRect(frame_results, (x1, y1, x2 - x1, y2 - y1), rt=0, colorC=(0, 255, 255))
        cvzone.putTextRect(frame_results, f"{class_id} {conf}", (x1, y1 - 15))

    path = find_closest_path(centers)

    for i in range(len(path) - 1):
        cv2.line(frame_results, path[i], path[i + 1], (0, 0, 255), 5)

    frame_show = cv2.resize(frame_results, (1280, 720))
    cv2.imshow("THREADING", frame_show)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("n"):
        break

cap.release()
cv2.destroyAllWindows()
