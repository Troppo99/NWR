import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import cv2
from ultralytics import YOLO
import numpy as np
import time
import torch
import cvzone

# Parameter Konfigurasi
CONFIDENCE_THRESHOLD_BROOM = 0.9
CONFIDENCE_THRESHOLD_PERSON = 0.5
IOU_THRESHOLD = 0.1  # Minimal overlap ratio (10%)
TOLERANCE_SECONDS = 2  # Toleransi waktu dalam detik
BROOM_ABSENCE_THRESHOLD = 5  # Jika sapu tidak terdeteksi overlapping border selama 5 detik

# Set Resolusi Asli dan Resolusi Baru
original_width, original_height = 1280, 720  # Resolusi asli
new_width, new_height = 640, 360  # Resolusi baru yang lebih rendah

# Hitung Faktor Skala
scale_x = new_width / original_width
scale_y = new_height / original_height

# Mendefinisikan Borders (koordinat sudah sesuai dengan resolusi 1280x720)
borders = [[(37, 497), (155, 422), (267, 565), (119, 655)], [(172, 414), (283, 557), (450, 459), (304, 334)], [(320, 326), (465, 449), (622, 360), (451, 255)], [(129, 672), (279, 581), (358, 666), (404, 711), (153, 711)], [(296, 575), (464, 471), (541, 529), (627, 593), (454, 712), (434, 712)], [(491, 713), (645, 606), (812, 713)], [(480, 461), (637, 368), (801, 467), (647, 584)], [(664, 596), (817, 477), (987, 581), (865, 712), (839, 709)], [(896, 712), (1003, 590), (1131, 668), (1104, 711)], [(529, 283), (695, 202), (900, 311), (752, 420)], [(771, 430), (917, 321), (1099, 430), (998, 567)], [(1013, 575), (1140, 652), (1186, 570), (1216, 508), (1111, 441)]]

# Skalakan Borders sesuai dengan resolusi baru
scaled_borders = []
for border in borders:
    scaled_border = []
    for x, y in border:
        scaled_x = int(x * scale_x)
        scaled_y = int(y * scale_y)
        scaled_border.append((scaled_x, scaled_y))
    scaled_borders.append(scaled_border)

# Inisialisasi Warna untuk Setiap Border sebagai Cyan
default_border_color = (0, 255, 255)  # Cyan
highlight_border_color = (0, 255, 0)  # Hijau

# Konversi Koordinat ke Numpy Arrays
borders_pts = [np.array(border, np.int32) for border in scaled_borders]

# Struktur Data untuk Menyimpan State Border
border_states = {idx: {"sapu_time": None, "orang_time": None, "is_green": False, "person_and_broom_detected": False, "broom_overlap_time": 0.0, "last_broom_overlap_time": None} for idx in range(len(borders))}  # Menandai deteksi bersama  # Waktu akumulasi overlapping sapu  # Waktu terakhir overlapping sapu


# Variabel untuk Melacak Waktu
start_time = None
end_time = None
elapsed_time = None
broom_absence_timer_start = None  # Timer untuk ketidakhadiran sapu overlapping border


# Fungsi untuk Memproses Deteksi Sapu
def process_model_broom(frame):
    with torch.no_grad():
        results_broom = model_broom(frame, imgsz=960)  # Mengurangi ukuran input model
    return results_broom


# Fungsi untuk Memproses Deteksi Orang
def process_model_person(frame):
    with torch.no_grad():
        results_person = model_person(frame, imgsz=640)  # Mengurangi ukuran input model
    return results_person


# Fungsi untuk Mengekstrak Keypoints Sapu
def export_frame_broom(results, color, pairs, confidence_threshold=CONFIDENCE_THRESHOLD_BROOM):
    points = []
    coords = []
    keypoint_positions = []

    for result in results:
        keypoints_data = result.keypoints
        if keypoints_data is not None and keypoints_data.xy is not None and keypoints_data.conf is not None:
            if keypoints_data.shape[0] > 0:
                keypoints_array = keypoints_data.xy.cpu().numpy()  # shape (n, k, 2)
                keypoints_conf = keypoints_data.conf.cpu().numpy()  # shape (n, k)
                for keypoints_per_object, keypoints_conf_per_object in zip(keypoints_array, keypoints_conf):
                    keypoints_list = []
                    for kp, kp_conf in zip(keypoints_per_object, keypoints_conf_per_object):
                        if kp_conf >= confidence_threshold:
                            x, y = kp[0], kp[1]
                            keypoints_list.append((int(x), int(y)))
                        else:
                            keypoints_list.append(None)  # Keypoint diabaikan jika confidence rendah
                    keypoint_positions.append(keypoints_list)
                    for point in keypoints_list:
                        if point is not None:
                            points.append(point)
                    for i, j in pairs:
                        if i < len(keypoints_list) and j < len(keypoints_list):
                            if keypoints_list[i] is not None and keypoints_list[j] is not None:
                                coords.append((keypoints_list[i], keypoints_list[j], color))
            else:
                continue
    return points, coords, keypoint_positions


# Fungsi untuk Mengekstrak Bounding Box Orang
def export_frame_person(results, confidence_threshold=CONFIDENCE_THRESHOLD_PERSON):
    person_boxes = []

    for result in results:
        boxes = result.boxes
        if boxes is not None and boxes.xyxy is not None and boxes.conf is not None and boxes.cls is not None:
            # Pastikan ada deteksi sebelum melakukan iterasi
            if boxes.xyxy.shape[0] > 0:
                for box, conf, cls in zip(boxes.xyxy.cpu().numpy(), boxes.conf.cpu().numpy(), boxes.cls.cpu().numpy()):
                    if conf >= confidence_threshold and int(cls) == 0:  # Class 0 biasanya 'person'
                        x1, y1, x2, y2 = box
                        person_boxes.append((int(x1), int(y1), int(x2), int(y2)))
    return person_boxes


# Fungsi untuk Menghitung Intersection over Union (IoU)
def compute_iou(boxA, boxB):
    # boxA dan boxB adalah tuple (x1, y1, x2, y2)
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # Hitung luas overlap
    interWidth = max(0, xB - xA + 1)
    interHeight = max(0, yB - yA + 1)
    interArea = interWidth * interHeight

    # Hitung luas masing-masing bounding box
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    # Hitung IoU
    if (boxAArea + boxBArea - interArea) == 0:
        return 0
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou


# Fungsi untuk Memproses Setiap Frame
def process_frame(frame, current_time):
    global start_time, end_time, elapsed_time, broom_absence_timer_start, border_states
    # Ubah ukuran frame menjadi resolusi baru
    frame_resized = cv2.resize(frame, (new_width, new_height))

    # Deteksi Sapu
    results_broom = process_model_broom(frame_resized)
    points_broom, coords_broom, keypoint_positions = export_frame_broom(results_broom, (0, 255, 0), pairs_broom, confidence_threshold=CONFIDENCE_THRESHOLD_BROOM)

    # Deteksi Orang
    results_person = process_model_person(frame_resized)
    person_boxes = export_frame_person(results_person, confidence_threshold=CONFIDENCE_THRESHOLD_PERSON)

    # Inisialisasi Warna untuk Setiap Border sebagai Cyan
    border_colors = [default_border_color] * len(borders)

    # Flag untuk mengetahui apakah sapu overlapping dengan border mana pun
    broom_overlapping_any_border = False

    # Periksa Interaksi Sapu dan Orang dengan Border
    for border_id, border_pt in enumerate(borders_pts):
        # Cek Sapu di Border
        sapu_overlapping = False
        for keypoints_list in keypoint_positions:
            for idx in [2, 3, 4]:  # Keypoint indices yang relevan
                if idx < len(keypoints_list):
                    kp = keypoints_list[idx]
                    if kp is not None:
                        result = cv2.pointPolygonTest(border_pt, kp, False)
                        if result >= 0:
                            sapu_overlapping = True
                            broom_overlapping_any_border = True  # Sapu overlapping dengan border
                            break
            if sapu_overlapping:
                break

        # Cek Orang di Border dengan Overlap
        orang_overlapping = False
        for box in person_boxes:
            # Menghitung IoU antara bounding box orang dan border bounding box
            # Mengambil bounding box border
            x_min, y_min, w, h = cv2.boundingRect(border_pt)
            x_max = x_min + w
            y_max = y_min + h
            border_box = (x_min, y_min, x_max, y_max)
            iou = compute_iou(box, border_box)
            if iou >= IOU_THRESHOLD:
                orang_overlapping = True
                break

        # Tahap Pertama: Deteksi Orang dan Sapu Bersamaan
        if not border_states[border_id]["person_and_broom_detected"]:
            if sapu_overlapping and orang_overlapping:
                # Tandai bahwa orang dan sapu pernah terdeteksi bersama
                border_states[border_id]["person_and_broom_detected"] = True
                # Mulai mengakumulasi waktu overlapping sapu
                if sapu_overlapping:
                    border_states[border_id]["last_broom_overlap_time"] = current_time
        else:
            # Tahap Kedua: Akumulasi Waktu Overlapping Sapu
            if sapu_overlapping:
                if border_states[border_id]["last_broom_overlap_time"] is not None:
                    # Akumulasi waktu overlapping sapu
                    delta_time = current_time - border_states[border_id]["last_broom_overlap_time"]
                    border_states[border_id]["broom_overlap_time"] += delta_time
                # Perbarui waktu terakhir overlapping sapu
                border_states[border_id]["last_broom_overlap_time"] = current_time
            else:
                # Tidak overlapping, reset waktu terakhir
                border_states[border_id]["last_broom_overlap_time"] = None

            # Cek apakah waktu overlapping sapu mencapai lebih dari 0,1 detik
            if border_states[border_id]["broom_overlap_time"] >= 0.1 and not border_states[border_id]["is_green"]:
                border_states[border_id]["is_green"] = True
                border_colors[border_id] = highlight_border_color
                # Atur start_time jika belum diatur
                if start_time is None:
                    start_time = current_time
                    broom_absence_timer_start = current_time  # Start broom absence timer

        # Set Warna Border jika sudah hijau
        if border_states[border_id]["is_green"]:
            border_colors[border_id] = highlight_border_color

    # Logika untuk Timer Ketidakhadiran Sapu Overlapping Border
    green_borders_exist = any(state["is_green"] for state in border_states.values())
    if green_borders_exist:
        if broom_overlapping_any_border:
            # Reset broom absence timer jika sapu overlapping border
            broom_absence_timer_start = current_time
        else:
            # Jika sapu tidak overlapping dengan border mana pun
            if broom_absence_timer_start is None:
                broom_absence_timer_start = current_time
            elif (current_time - broom_absence_timer_start) >= BROOM_ABSENCE_THRESHOLD:
                # Sapu tidak overlapping border selama threshold waktu
                # Reset semua border dan timer
                border_states = {idx: {"sapu_time": None, "orang_time": None, "is_green": False, "person_and_broom_detected": False, "broom_overlap_time": 0.0, "last_broom_overlap_time": None} for idx in range(len(borders))}
                border_colors = [default_border_color] * len(borders)
                start_time = None
                end_time = None
                elapsed_time = None
                broom_absence_timer_start = None
    else:
        # Tidak ada border hijau, reset broom_absence_timer_start
        broom_absence_timer_start = None

    # Cek apakah semua border sudah hijau
    all_green = all(state["is_green"] for state in border_states.values())
    if all_green and start_time is not None and end_time is None:
        end_time = current_time
        elapsed_time = end_time - start_time
        # Konversi elapsed_time ke format hh:mm:ss
        hours = int(elapsed_time // 3600)
        minutes = int((elapsed_time % 3600) // 60)
        seconds = int(elapsed_time % 60)
        time_str = f"SWEEPING TIME: {hours:02d}:{minutes:02d}:{seconds:02d}"
    elif start_time is not None and end_time is None:
        # Timer masih berjalan
        elapsed_time = current_time - start_time
        # Konversi elapsed_time ke format hh:mm:ss
        hours = int(elapsed_time // 3600)
        minutes = int((elapsed_time % 3600) // 60)
        seconds = int(elapsed_time % 60)
        time_str = f"SWEEPING TIME: {hours:02d}:{minutes:02d}:{seconds:02d}"
    elif end_time is not None:
        # Timer sudah berhenti
        elapsed_time = end_time - start_time
        # Konversi elapsed_time ke format hh:mm:ss
        hours = int(elapsed_time // 3600)
        minutes = int((elapsed_time % 3600) // 60)
        seconds = int(elapsed_time % 60)
        time_str = f"SWEEPING TIME: {hours:02d}:{minutes:02d}:{seconds:02d}"

    # Gambar Keypoints dan Garis untuk Sapu
    if points_broom and coords_broom:
        for x, y, color in coords_broom:
            cv2.line(frame_resized, x, y, color, 2)
        for point in points_broom:
            cv2.circle(frame_resized, point, 4, (0, 255, 255), -1)

    # Gambar Bounding Box Orang
    if person_boxes:
        for box in person_boxes:
            x1, y1, x2, y2 = box
            cvzone.cornerRect(frame_resized, (x1, y1, x2 - x1, y2 - y1), 20, 3, 0, (255, 0, 0), (255, 0, 0))

    # Gambar Poligon dan Isi dengan Warna Transparan
    overlay = frame_resized.copy()
    alpha = 0.5  # Faktor Transparansi

    for border_pt, color in zip(borders_pts, border_colors):
        cv2.fillPoly(overlay, pts=[border_pt], color=color)

    # Blend overlay dengan frame asli
    cv2.addWeighted(overlay, alpha, frame_resized, 1 - alpha, 0, frame_resized)

    # Tampilkan Elapsed Time jika timer sudah dimulai
    if start_time is not None and elapsed_time is not None:
        cvzone.putTextRect(frame_resized, time_str, (10, 50), scale=1, thickness=2, offset=5)

    return frame_resized


if __name__ == "__main__":
    # Muat Model Deteksi Sapu dan Orang
    model_broom = YOLO("D:/SBHNL/Resources/Models/Pretrained/BROOM/B5_LARGE/weights/best.pt").to("cuda")  # Model Sapu
    model_person = YOLO("yolov8n.pt").to("cuda")  # Model Orang

    # Verifikasi bahwa model berada di GPU
    print(f"Model Broom device: {next(model_broom.model.parameters()).device}")
    print(f"Model Person device: {next(model_person.model.parameters()).device}")

    # Definisikan Sumber Video
    rtsp_url = "D:/SBHNL/Videos/AHMDL/Test/sapu_182.mp4"
    cap = cv2.VideoCapture(rtsp_url)
    if not cap.isOpened():
        print(f"Error: Cannot open video {rtsp_url}")
        exit()

    # Set Resolusi yang Diinginkan
    pairs_broom = [(0, 1), (1, 2), (2, 3), (2, 4)]  # Definisikan pasangan keypoint untuk menggambar garis

    frame_count = 0
    process_every_n_frames = 2  # Proses setiap 2 frame

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame")
            break

        frame_count += 1
        if frame_count % process_every_n_frames != 0:
            continue  # Lewati frame ini

        current_time = time.time()
        frame_resized = process_frame(frame, current_time)
        cv2.imshow("Broom and Person Detection", frame_resized)

        # Tekan 'n' untuk keluar
        if cv2.waitKey(1) & 0xFF == ord("n"):
            break

    cap.release()
    cv2.destroyAllWindows()
