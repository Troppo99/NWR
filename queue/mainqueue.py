import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import cv2
import cvzone
from ultralytics import YOLO
import math
import numpy as np


def process_frame(frame, model):
    """
    Memproses frame untuk mendeteksi objek menggunakan model YOLO.
    Mengembalikan frame yang diproses, informasi bounding box, dan titik tengah objek.
    """
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
    """
    Menentukan urutan koneksi berdasarkan jarak terdekat menggunakan algoritma greedy.
    Mengembalikan path sebagai daftar titik tengah yang terurut.
    """
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


def catmull_rom_spline(P0, P1, P2, P3, n_points=100):
    """
    Menghasilkan titik-titik pada Catmull-Rom Spline antara P1 dan P2.

    Parameters:
    - P0, P1, P2, P3: Empat titik kontrol (tujuan: spline antara P1 dan P2).
    - n_points: Jumlah titik yang dihasilkan pada spline.

    Returns:
    - Array dari titik-titik pada spline.
    """
    alpha = 0.5  # Parameter untuk menentukan kekakuan spline

    def tj(ti, Pi, Pj):
        xi, yi = Pi
        xj, yj = Pj
        return ((((xj - xi) ** 2 + (yj - yi) ** 2) ** 0.5) ** alpha) + ti

    t0 = 0
    t1 = tj(t0, P0, P1)
    t2 = tj(t1, P1, P2)
    t3 = tj(t2, P2, P3)

    t = np.linspace(t1, t2, n_points).reshape(n_points, 1)

    A1 = (t1 - t) / (t1 - t0) * np.array(P0) + (t - t0) / (t1 - t0) * np.array(P1)
    A2 = (t2 - t) / (t2 - t1) * np.array(P1) + (t - t1) / (t2 - t1) * np.array(P2)
    A3 = (t3 - t) / (t3 - t2) * np.array(P2) + (t - t2) / (t3 - t2) * np.array(P3)

    B1 = (t2 - t) / (t2 - t0) * A1 + (t - t0) / (t2 - t0) * A2
    B2 = (t3 - t) / (t3 - t1) * A2 + (t - t1) / (t3 - t1) * A3

    C = (t2 - t) / (t2 - t1) * B1 + (t - t1) / (t2 - t1) * B2

    return C.astype(int)


def generate_catmull_rom_spline(path, n_points=100):
    """
    Menghasilkan semua titik pada Catmull-Rom Spline untuk seluruh path.

    Parameters:
    - path: List dari titik-titik (x, y).
    - n_points: Jumlah titik yang dihasilkan pada setiap segmen spline.

    Returns:
    - Array dari semua titik pada spline.
    """
    spline_points = []

    # Untuk Catmull-Rom, kita membutuhkan minimal 4 titik. Tambahkan duplikat pada awal dan akhir path
    if len(path) < 4:
        # Jika kurang dari 4 titik, tambahkan duplikat untuk memenuhi kebutuhan
        if len(path) == 3:
            path = [path[0]] + path + [path[-1]]
        elif len(path) == 2:
            path = [path[0]] + path + [path[-1]]
        elif len(path) == 1:
            path = [path[0], path[0], path[0], path[0]]

    # Tambahkan duplikat pada awal dan akhir path
    extended_path = [path[0]] + path + [path[-1]]

    for i in range(len(extended_path) - 3):
        P0 = extended_path[i]
        P1 = extended_path[i + 1]
        P2 = extended_path[i + 2]
        P3 = extended_path[i + 3]
        segment = catmull_rom_spline(P0, P1, P2, P3, n_points)
        spline_points.extend(segment)

    return np.array(spline_points, dtype=np.int32)


def split_path_by_threshold(path, threshold_distance):
    """
    Membagi path menjadi subpaths berdasarkan threshold jarak.

    Parameters:
    - path: List dari titik-titik (x, y).
    - threshold_distance: Jarak maksimum antara dua titik untuk tetap dalam subpath yang sama.

    Returns:
    - List dari subpaths.
    """
    if not path:
        return []

    subpaths = []
    current_subpath = [path[0]]

    for i in range(1, len(path)):
        pt_prev = current_subpath[-1]
        pt_current = path[i]
        distance = math.hypot(pt_current[0] - pt_prev[0], pt_current[1] - pt_prev[1])
        if distance <= threshold_distance:
            current_subpath.append(pt_current)
        else:
            if len(current_subpath) >= 2:
                subpaths.append(current_subpath)
            current_subpath = [pt_current]

    if len(current_subpath) >= 2:
        subpaths.append(current_subpath)

    return subpaths


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
cv2.createTrackbar("Threshold (%)", "THREADING", 40, 100, nothing)  # Batas awal 40%, maksimum 100%

# Buat window tambahan untuk deteksi "person"
cv2.namedWindow("Persons", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Persons", 640, 480)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Tidak dapat membaca frame dari video.")
        break

    frame_results, boxes_info, centers = process_frame(frame, model)

    # Membuat salinan frame untuk jendela "Persons"
    frame_persons = frame.copy()

    # Menampilkan titik-titik tengah untuk debugging (opsional)
    for center in centers:
        cv2.circle(frame_results, center, 5, (255, 0, 0), -1)  # Titik biru

    # Mendapatkan nilai threshold dari trackbar (dalam persen)
    threshold_percent = cv2.getTrackbarPos("Threshold (%)", "THREADING")

    # Hitung jarak maksimum (diagonal frame)
    frame_height, frame_width = frame.shape[:2]
    max_distance = math.hypot(frame_width, frame_height)

    # Konversi threshold persentase ke jarak piksel
    threshold_distance = (threshold_percent / 100.0) * max_distance

    # Menentukan urutan koneksi berdasarkan jarak terdekat
    path = find_closest_path(centers)

    # Membagi path menjadi subpaths berdasarkan threshold distance
    subpaths = split_path_by_threshold(path, threshold_distance)

    # Menggambar spline untuk setiap subpath
    for subpath in subpaths:
        # Menghasilkan spline points
        spline_points = generate_catmull_rom_spline(subpath, n_points=100)

        # Menggambar spline
        cv2.polylines(frame_results, [spline_points], False, (0, 255, 0), thickness=7)

        # Menambahkan indikator jarak antara titik-titik yang disambungkan
        for i in range(len(subpath) - 1):
            pt1 = subpath[i]
            pt2 = subpath[i + 1]
            distance = math.hypot(pt2[0] - pt1[0], pt2[1] - pt1[1])
            mid_point = ((pt1[0] + pt2[0]) // 2, (pt1[1] + pt2[1]) // 2)
            cvzone.putTextRect(
                frame_results,
                f"{int(distance)}px",
                mid_point,
                scale=0.6,
                thickness=1,
                offset=2,
                colorR=(0, 0, 0),
            )

    # Memproses deteksi "person" dan menggambar bounding box di jendela "Persons"
    for x1, y1, x2, y2, conf, class_id in boxes_info:
        if class_id.lower() == "person":
            # Menggambar bounding box menggunakan cvzone
            cvzone.cornerRect(frame_persons, (x1, y1, x2 - x1, y2 - y1), rt=0, colorC=(255, 0, 0))
            # Menampilkan label
            cvzone.putTextRect(
                frame_persons,
                f"{class_id} {conf}",
                (x1, y1 - 10),
                scale=1,
                thickness=1,
                offset=3,
                colorR=(0, 0, 0),
            )

    # Menambahkan teks informasi threshold di jendela utama
    cvzone.putTextRect(
        frame_results,
        f"Threshold: {threshold_percent}%",
        (10, 30),
        scale=1,
        thickness=2,
        offset=5,
        colorR=(0, 0, 0),
    )

    # Menambahkan teks informasi rentang threshold di bagian bawah window utama
    cvzone.putTextRect(
        frame_results,
        f"Threshold Range: 0% - 100%",
        (10, 60),
        scale=0.7,
        thickness=2,
        offset=3,
        colorR=(0, 0, 0),
    )

    # Menambahkan informasi total objek yang terdeteksi di jendela utama
    total_objects = len(boxes_info)
    cvzone.putTextRect(
        frame_results,
        f"Total Objects: {total_objects}",
        (10, 90),
        scale=0.7,
        thickness=2,
        offset=3,
        colorR=(0, 0, 0),
    )

    # Resize dan tampilkan frame utama
    frame_show = cv2.resize(frame_results, (1280, 720))
    cv2.imshow("THREADING", frame_show)

    # Resize dan tampilkan frame "Persons"
    frame_persons_show = cv2.resize(frame_persons, (640, 480))
    cv2.imshow("Persons", frame_persons_show)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("n"):
        break

cap.release()
cv2.destroyAllWindows()
