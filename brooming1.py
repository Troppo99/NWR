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
BROOM_TOUCH_THRESHOLD = 0.00005  # ganti ke 0 untuk menghilangkan waktu overlapping

# Set Resolusi Asli dan Resolusi Baru
original_width, original_height = 1280, 720  # Resolusi asli
new_width, new_height = 640, 360  # Resolusi baru yang lebih rendah

# Hitung Faktor Skala
scale_x = new_width / original_width
scale_y = new_height / original_height

# Mendefinisikan Borders (koordinat sudah sesuai dengan resolusi 1280x720)
borders = [[(30, 493), (114, 439), (158, 510), (64, 567)], [(114, 439), (210, 383), (261, 448), (158, 510)], [(210, 383), (308, 326), (372, 384), (261, 448)], [(308, 326), (454, 247), (533, 296), (372, 384)], [(117, 667), (64, 567), (158, 510), (222, 601)], [(222, 601), (158, 510), (261, 448), (341, 530)], [(341, 530), (261, 448), (372, 384), (465, 459)], [(465, 459), (372, 384), (533, 296), (635, 357)], [(533, 296), (632, 247), (731, 303), (635, 357)], [(731, 303), (632, 247), (713, 208), (812, 258)], [(149, 715), (117, 667), (222, 601), (312, 713)], [(312, 713), (222, 601), (341, 530), (447, 634)], [(447, 634), (341, 530), (465, 459), (580, 547)], [(580, 547), (465, 459), (635, 357), (753, 428)], [(753, 428), (635, 357), (731, 303), (841, 365)], [(841, 365), (731, 303), (812, 258), (914, 311)], [(312, 713), (447, 634), (541, 715)], [(541, 715)], [(541, 715), (447, 634), (580, 547), (714, 641), (622, 713)], [(714, 641), (580, 547), (753, 428), (877, 506)], [(877, 506), (753, 428), (841, 365), (957, 432)], [(957, 432), (841, 365), (914, 311), (1014, 370)], [(622, 713), (714, 641), (825, 712)], [(825, 712), (714, 641), (877, 506), (996, 580), (877, 712)], [(996, 580), (877, 506), (957, 432), (1061, 496)], [(1061, 496), (957, 432), (1014, 370), (1110, 429)], [(877, 712), (996, 580), (1138, 663), (1108, 714)], [(1138, 663), (996, 580), (1061, 496), (1184, 573)], [(1184, 573), (1061, 496), (1110, 429), (1221, 504)]]

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

# Variabel untuk FPS
prev_frame_time = time.time()
fps = 0
# Tambahkan variabel global baru
first_green_time = None
is_counting = False


# Fungsi untuk Memproses Deteksi Sapu
def process_model_broom(frame):
    with torch.no_grad():
        results_broom = model_broom(frame, imgsz=960)  # Mengurangi ukuran input model
    return results_broom


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


# Fungsi untuk Memproses Setiap Frame
def process_frame(frame, current_time):
    global start_time, end_time, elapsed_time, broom_absence_timer_start, border_states, first_green_time, is_counting
    frame_resized = cv2.resize(frame, (new_width, new_height))

    results_broom = process_model_broom(frame_resized)
    points_broom, coords_broom, keypoint_positions = export_frame_broom(results_broom, (0, 255, 0), pairs_broom, confidence_threshold=CONFIDENCE_THRESHOLD_BROOM)

    # Inisialisasi warna border berdasarkan state sebelumnya
    border_colors = [(0, 255, 0) if state["is_green"] else (0, 255, 255) for state in border_states.values()]

    broom_overlapping_any_border = False

    for border_id, border_pt in enumerate(borders_pts):
        sapu_overlapping = False
        for keypoints_list in keypoint_positions:
            for idx in [2, 3, 4]:  # Hanya cek keypoint 2, 3, dan 4
                if idx < len(keypoints_list):
                    kp = keypoints_list[idx]
                    if kp is not None:
                        result = cv2.pointPolygonTest(border_pt, kp, False)
                        if result >= 0:
                            sapu_overlapping = True
                            broom_overlapping_any_border = True
                            break
            if sapu_overlapping:
                break

        if sapu_overlapping:
            if border_states[border_id]["last_broom_overlap_time"] is None:
                border_states[border_id]["last_broom_overlap_time"] = current_time
            else:
                delta_time = current_time - border_states[border_id]["last_broom_overlap_time"]
                border_states[border_id]["broom_overlap_time"] += delta_time
                border_states[border_id]["last_broom_overlap_time"] = current_time

            if border_states[border_id]["broom_overlap_time"] >= BROOM_TOUCH_THRESHOLD:
                border_states[border_id]["is_green"] = True
                border_colors[border_id] = (0, 255, 0)  # Ubah warna menjadi hijau
        else:
            border_states[border_id]["last_broom_overlap_time"] = None

    # Logika untuk Timer Ketidakhadiran Sapu Overlapping Border
    green_borders_exist = any(state["is_green"] for state in border_states.values())
    if green_borders_exist:
        if not is_counting:
            first_green_time = current_time
            is_counting = True

        if broom_overlapping_any_border:
            broom_absence_timer_start = current_time
        else:
            if broom_absence_timer_start is None:
                broom_absence_timer_start = current_time
            elif (current_time - broom_absence_timer_start) >= BROOM_ABSENCE_THRESHOLD:
                # Reset semua border dan timer
                for idx in range(len(borders)):
                    border_states[idx] = {"is_green": False, "broom_overlap_time": 0.0, "last_broom_overlap_time": None}
                    border_colors[idx] = (0, 255, 255)  # Kembalikan ke warna kuning
                first_green_time = None
                is_counting = False
                broom_absence_timer_start = None
    else:
        broom_absence_timer_start = None
        if is_counting:
            first_green_time = None
            is_counting = False

    # Gambar Keypoints dan Garis untuk Sapu
    if points_broom and coords_broom:
        for x, y, color in coords_broom:
            cv2.line(frame_resized, x, y, color, 2)
        for point in points_broom:
            cv2.circle(frame_resized, point, 4, (0, 255, 255), -1)

    # Gambar Poligon dan Isi dengan Warna Transparan
    overlay = frame_resized.copy()
    alpha = 0.5  # Faktor Transparansi

    for border_pt, color in zip(borders_pts, border_colors):
        cv2.fillPoly(overlay, pts=[border_pt], color=color)

    # Blend overlay dengan frame asli
    cv2.addWeighted(overlay, alpha, frame_resized, 1 - alpha, 0, frame_resized)

    # Tampilkan Elapsed Time jika timer sudah dimulai
    if is_counting and first_green_time is not None:
        elapsed_time = current_time - first_green_time
        minutes, seconds = divmod(int(elapsed_time), 60)
        time_str = f"Elapsed Time: {minutes:02d}:{seconds:02d}"
        cvzone.putTextRect(frame_resized, time_str, (10, 50), scale=1, thickness=2, offset=5)

    return frame_resized


if __name__ == "__main__":
    # Muat hanya Model Deteksi Sapu
    model_broom = YOLO("D:/SBHNL/Resources/Models/Pretrained/BROOM/B5_LARGE/weights/best.pt").to("cuda")  # Model Sapu
    model_broom.overrides["verbose"] = False
    # Verifikasi bahwa model berada di GPU
    print(f"Model Broom device: {next(model_broom.model.parameters()).device}")

    # Definisikan Sumber Video
    rtsp_url = "D:/SBHNL/Videos/AHMDL/Test/sapu_182(2).mp4"
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

        # Perhitungan FPS
        time_diff = current_time - prev_frame_time
        if time_diff > 0:
            fps = 1 / time_diff
        else:
            fps = 0
        prev_frame_time = current_time  # Perbarui waktu sebelumnya

        # Proses frame
        frame_resized = process_frame(frame, current_time)

        # Hitung dan Cetak Persentase Border Hijau
        total_borders = len(borders)
        green_borders = sum(1 for state in border_states.values() if state["is_green"])
        percentage_green = (green_borders / total_borders) * 100
        print(f"Persentase Border Hijau: {percentage_green:.2f}%")

        # Tampilkan Elapsed Time dan FPS
        cvzone.putTextRect(frame_resized, f"FPS: {int(fps)}", (10, 90), scale=1, thickness=2, offset=5)

        cv2.imshow("Broom and Person Detection", frame_resized)

        # Tekan 'n' untuk keluar
        if cv2.waitKey(1) & 0xFF == ord("n"):
            break

    cap.release()
    cv2.destroyAllWindows()
