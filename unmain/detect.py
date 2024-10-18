import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import cv2
import cvzone
from ultralytics import YOLO
import math


def process_frame(frame):
    results = model(frame)
    boxes_info = []
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = math.ceil(box.conf[0] * 100) / 100
            class_id = model.names[int(box.cls[0])]
            if conf > 0:
                boxes_info.append((x1, y1, x2, y2, conf, class_id))
    return frame, boxes_info


videos = ["rtsp://admin:oracle2015@10.5.0.7:554/Streaming/Channels/1", "D:/SBHNL/Videos/AHMDL/Test/uji.mp4"]
video = videos[0]
cap = cv2.VideoCapture(video)
model = YOLO("D:/NWR/run/finishing/version2/weights/best.pt")

while True:
    ret, frame = cap.read()

    frame_results, boxes_info = process_frame(frame)
    for x1, y1, x2, y2, conf, class_id in boxes_info:
        cvzone.cornerRect(frame_results, (x1, y1, x2 - x1, y2 - y1), rt=0, colorC=(0, 255, 255))
        cvzone.putTextRect(frame_results, f"{class_id} {conf}", (x1, y1 - 15))

    frame_show = cv2.resize(frame_results, (1280, 720))
    cv2.imshow("THREADING", frame_show)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("n"):
        break
cap.release()
cv2.destroyAllWindows()
