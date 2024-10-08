import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from ultralytics import YOLO
import cv2

model = YOLO("D:/SBHNL/Resources/Models/Pretrained/FINISHING_V1/F_V1/weights/best.pt")
rtsp_url = "rtsp://admin:oracle2015@10.5.0.7:554/Streaming/Channels/1"
cap = cv2.VideoCapture(rtsp_url)

while cap.isOpened():
    success, frame = cap.read()
    if success:
        results = model(frame)
        annotated_frame = results[0].plot()
        annotated_frame = cv2.resize(annotated_frame, (640, 360))
        cv2.imshow("YOLO11 Inference", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord("n"):
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()
