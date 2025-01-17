import cv2

detector = cv2.QRCodeDetector()

# cap = cv2.VideoCapture("rtsp://admin:oracle2015@192.168.100.65:554/Streaming/Channels/1")
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    data, bbox, _ = detector.detectAndDecode(frame)

    if bbox is not None:
        bbox = bbox.astype(int)
        n = len(bbox)
        for i in range(n):
            cv2.line(frame, tuple(bbox[i][0]), tuple(bbox[(i + 1) % n][0]), (255, 0, 0), 2)

        if data:
            cv2.putText(frame, data, tuple(bbox[0][0] + [0, -10]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        center_x = int(bbox[:, 0, 0].mean())
        center_y = int(bbox[:, 0, 1].mean())
        cv2.circle(frame, (center_x, center_y), 5, (0, 0, 255), -1)
        print(f"QR Code pusat di: x={center_x}, y={center_y}")

    cv2.imshow("Video", frame)

    if cv2.waitKey(1) & 0xFF == ord("n"):
        break

cap.release()
cv2.destroyAllWindows()
