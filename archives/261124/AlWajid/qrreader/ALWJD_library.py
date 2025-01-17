import cv2
from pyzbar import pyzbar

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    qr_codes = pyzbar.decode(frame)

    for qr in qr_codes:
        (x, y, w, h) = qr.rect
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, qr.data.decode("utf-8"), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        print(f"QR Code ditemukan di posisi: x={x}, y={y}, width={w}, height={h}")

    cv2.imshow("Video", frame)

    if cv2.waitKey(1) & 0xFF == ord("n"):
        break

cap.release()
cv2.destroyAllWindows()
