import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import cv2
import cvzone
from ultralytics import YOLO
import math


def process_frame(frame, model):
    results = model(frame)
    boxes_info = []
    centers = []
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = math.ceil(box.conf[0] * 100) / 100
            class_id = model.names[int(box.cls[0])]
            if conf > 0:
                boxes_info.append((x1, y1, x2, y2, conf, class_id))
                # Hitung titik tengah
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                centers.append((center_x, center_y))
    return frame, boxes_info, centers


def find_closest_path(centers):
    if not centers:
        return []

    # Menggunakan algoritma greedy untuk menemukan urutan koneksi terdekat
    unvisited = centers.copy()
    path = []

    # Mulai dari titik pertama
    current = unvisited.pop(0)
    path.append(current)

    while unvisited:
        # Cari titik terdekat dari titik saat ini
        closest = min(
            unvisited, key=lambda point: math.hypot(point[0] - current[0], point[1] - current[1])
        )
        path.append(closest)
        unvisited.remove(closest)
        current = closest

    return path


def nothing(x):
    pass


# Inisialisasi video dan model YOLO
videos = [
    "rtsp://admin:oracle2015@10.5.0.239:554/Streaming/Channels/1",
    r"D:\NWR\videos\antre1.mp4",
]
video = videos[1]
cap = cv2.VideoCapture(video)
model = YOLO("yolo11l.pt")

# Buat window dan trackbar untuk threshold
cv2.namedWindow("THREADING", cv2.WINDOW_NORMAL)
cv2.resizeWindow("THREADING", 1280, 720)
cv2.createTrackbar("Threshold", "THREADING", 400, 1000, nothing)  # Batas awal 400, maksimum 1000

while True:
    ret, frame = cap.read()
    if not ret:
        print("Tidak dapat membaca frame dari video.")
        break

    frame_results, boxes_info, centers = process_frame(frame, model)

    for x1, y1, x2, y2, conf, class_id in boxes_info:
        # Gambar bounding box
        # cvzone.cornerRect(frame_results, (x1, y1, x2 - x1, y2 - y1), rt=0, colorC=(0, 255, 255))
        # Tampilkan label
        # cvzone.putTextRect(frame_results, f"{class_id} {conf}", (x1, y1 - 15), scale=1, thickness=1, offset=3)
        pass

    # Mendapatkan nilai threshold dari trackbar
    threshold = cv2.getTrackbarPos("Threshold", "THREADING")

    # Menentukan urutan koneksi berdasarkan jarak terdekat
    path = find_closest_path(centers)

    # Menggambar garis antar titik tengah sesuai urutan dan threshold
    for i in range(len(path) - 1):
        pt1 = path[i]
        pt2 = path[i + 1]
        distance = math.hypot(pt2[0] - pt1[0], pt2[1] - pt1[1])
        if distance <= threshold:
            # Garis dalam batas threshold dengan ketebalan 7
            cv2.line(frame_results, pt1, pt2, (0, 255, 0), 7)  # Hijau
        # Jika jarak > threshold, jangan gambar garis

    # Menambahkan teks informasi threshold
    cvzone.putTextRect(
        frame_results,
        f"Threshold: {threshold}px",
        (10, 30),
        scale=1,
        thickness=2,
        offset=5,
        colorR=(0, 0, 0),
    )

    # Menambahkan teks informasi rentang threshold di bagian bawah window
    cvzone.putTextRect(
        frame_results,
        f"Threshold Range: 0 - 1000px",
        (10, 60),
        scale=0.7,
        thickness=2,
        offset=3,
        colorR=(0, 0, 0),
    )

    # Menambahkan slider grafis di sisi window menggunakan cvzone
    # cvzone tidak menyediakan slider grafis, jadi tetap menggunakan trackbar OpenCV di atas.

    # Resize dan tampilkan frame
    frame_show = cv2.resize(frame_results, (1280, 720))
    cv2.imshow("THREADING", frame_show)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("n"):
        break

cap.release()
cv2.destroyAllWindows()
