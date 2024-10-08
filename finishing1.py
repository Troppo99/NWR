import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import cv2
import numpy as np
from ultralytics import YOLO
import time

# Muat model YOLO
model = YOLO("D:/SBHNL/Resources/Models/Pretrained/FINISHING_V1/F_V1/weights/best.pt")
rtsp_url = "rtsp://admin:oracle2015@10.5.0.7:554/Streaming/Channels/1"
cap = cv2.VideoCapture(rtsp_url)

# Dictionary untuk menyimpan posisi bbox dari frame sebelumnya
previous_positions = {}

# Nama kelas (disesuaikan dengan model yang digunakan)
class_names = model.names  # Mendapatkan nama-nama kelas dari model YOLO


def iou(box1, box2):
    """Hitung Intersection over Union (IoU) antara dua bounding boxes."""
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2

    xA = max(x1_min, x2_min)
    yA = max(y1_min, y2_min)
    xB = min(x1_max, x2_max)
    yB = min(y1_max, y2_max)

    inter_area = max(0, xB - xA) * max(0, yB - yA)
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)

    union_area = box1_area + box2_area - inter_area
    return inter_area / union_area if union_area > 0 else 0


def calculate_pps(box1, box2, time_elapsed):
    """Hitung pergerakan dalam pixels per second (pps)."""
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2

    # Hitung pusat dari masing-masing bbox
    center1 = ((x1_min + x1_max) / 2, (y1_min + y1_max) / 2)
    center2 = ((x2_min + x2_max) / 2, (y2_min + y2_max) / 2)

    # Hitung jarak pergerakan
    distance = np.sqrt((center2[0] - center1[0]) ** 2 + (center2[1] - center1[1]) ** 2)

    # Hitung pixels per second
    return distance / time_elapsed if time_elapsed > 0 else 0


def filter_duplicates_with_iou(results, iou_threshold=0.5):
    """Filter duplikat menggunakan IoU dan simpan bbox dengan confidence tertinggi."""
    filtered_results = []
    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()  # Bounding boxes
        scores = result.boxes.conf.cpu().numpy()  # Confidence scores
        classes = result.boxes.cls.cpu().numpy()  # Class labels

        keep_indices = []
        seen_boxes = []

        for i, box in enumerate(boxes):
            add_box = True
            for j, kept_box in enumerate(seen_boxes):
                if iou(box, kept_box) > iou_threshold and classes[i] == classes[keep_indices[j]]:
                    if scores[i] > scores[keep_indices[j]]:
                        keep_indices[j] = i  # Ganti dengan bbox confidence lebih tinggi
                    add_box = False
                    break

            if add_box:
                keep_indices.append(i)
                seen_boxes.append(box)

        filtered_result = result[keep_indices]
        filtered_results.append(filtered_result)

    return filtered_results


last_time = time.time()

while cap.isOpened():
    success, frame = cap.read()
    current_time = time.time()
    time_elapsed = current_time - last_time
    last_time = current_time

    if success:
        # Lakukan inferensi menggunakan model YOLO
        results = model(frame)

        # Filter duplikat deteksi berdasarkan IoU dan confidence score
        filtered_results = filter_duplicates_with_iou(results)

        # Anotasi dan tampilkan frame
        if filtered_results:
            for result in filtered_results:
                boxes = result.boxes.xyxy.cpu().numpy()
                scores = result.boxes.conf.cpu().numpy()
                class_ids = result.boxes.cls.cpu().numpy()

                for i, box in enumerate(boxes):
                    box_id = int(class_ids[i])
                    score = scores[i]
                    class_name = class_names[box_id]  # Dapatkan nama kelas objek

                    # Hitung pps berdasarkan posisi sebelumnya
                    if box_id in previous_positions:
                        pps = calculate_pps(previous_positions[box_id], box, time_elapsed)
                    else:
                        pps = 0  # Jika tidak ada data sebelumnya

                    previous_positions[box_id] = box  # Simpan posisi saat ini sebagai posisi sebelumnya

                    # Anotasi pada frame: bounding box, nama objek, confidence, dan pps
                    x1, y1, x2, y2 = map(int, box)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 5)
                    cv2.putText(frame, f"{class_name} {pps:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)

            # Tampilkan frame yang sudah dianotasi
            annotated_frame = cv2.resize(frame, (640, 360))
            cv2.imshow("YOLO Inference with PPS and Class Names", annotated_frame)

        # Hentikan jika tombol 'n' ditekan
        if cv2.waitKey(1) & 0xFF == ord("n"):
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()
