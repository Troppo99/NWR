from ultralytics import YOLO
import cv2

model = YOLO("D:/NWR/run/kon/version1/weights/last.pt")
cap = cv2.VideoCapture("D:/SBHNL/Videos/AHMDL/EDIT/kon.mp4")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    results = model.predict(frame, conf=0.5)
    annotated = results[0].plot()
    annotated = cv2.resize(annotated, (1280, 720))
    cv2.imshow("Inference", annotated)
    if cv2.waitKey(1) & 0xFF == ord("n"):
        break

cap.release()
cv2.destroyAllWindows()
