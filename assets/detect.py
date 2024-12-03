import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import cv2
import cvzone
from ultralytics import YOLO
import math
import time


def export_frame(frame):
    results = model(frame)
    boxes_info = []
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = box.conf[0]
            class_id = model.names[int(box.cls[0])]
            if conf > 0.5:
                boxes_info.append((x1, y1, x2, y2, conf, class_id))
    return frame, boxes_info


n=1
videos = ["rtsp://admin:oracle2015@10.5.0.110:554/Streaming/Channels/1", f"D:/NWR/videos/test/broom_test_000{n}.mp4"]
cap = cv2.VideoCapture(videos[0])
model = YOLO("D:/NWR/run/g2_one/version_detect/weights/best.pt")
model.overrides["verbose"] = False

while True:
    ret, frame = cap.read()
    if not ret:
        n += 1
        # videos = ["rtsp://admin:oracle2015@10.5.0.110:554/Streaming/Channels/1", f"D:/NWR/videos/test/broom_test_000{n}.mp4"]
        cap = cv2.VideoCapture(videos[0])
        time.sleep(0.1)
        try:
            ret, frame = cap.read()
        except:
            continue

    frame_results, boxes_info = export_frame(frame)
    for x1, y1, x2, y2, conf, class_id in boxes_info:
        cvzone.cornerRect(frame_results, (x1, y1, x2 - x1, y2 - y1), rt=0, colorC=(0, 255, 255))
        cvzone.putTextRect(frame_results, f"{class_id} {conf}", (x1, y1 - 15))

    frame_show = cv2.resize(frame_results, (540, 360))
    cv2.imshow(f"THREADING {n}", frame_show)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("n"):
        break
cap.release()
cv2.destroyAllWindows()
