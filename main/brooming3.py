import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import cv2
from ultralytics import YOLO
import numpy as np
import time
import torch
import cvzone
import pymysql
from datetime import datetime

# Parameter Konfigurasi
CONFIDENCE_THRESHOLD_BROOM = 0.9
BROOM_ABSENCE_THRESHOLD = 5  # Jika sapu tidak terdeteksi overlapping border selama 5 detik
BROOM_TOUCH_THRESHOLD = 0.00005  # ganti ke 0 untuk menghilangkan waktu overlapping

# Set Resolusi Asli dan Resolusi Baru
original_width, original_height = 1280, 720  # Resolusi asli
new_width, new_height = 640, 360  # Resolusi baru yang lebih rendah

# Hitung Faktor Skala
scale_x = new_width / original_width
scale_y = new_height / original_height

# Mendefinisikan Borders (koordinat sudah sesuai dengan resolusi 1280x720)
borders = [
    [(128, 592), (302, 604), (298, 712), (110, 714)],
    [(466, 709), (298, 712), (302, 604), (466, 609)],
    [(466, 709), (466, 609), (603, 608), (615, 709)],
    [(128, 592), (302, 604), (320, 436), (154, 435)],
    [(302, 604), (466, 609), (463, 432), (320, 436)],
    [(466, 609), (603, 608), (588, 424), (463, 432)],
    [(588, 424), (780, 415), (795, 469), (650, 475), (593, 478)],
    [(795, 469), (780, 415), (925, 405), (942, 455)],
    [(942, 455), (925, 405), (1043, 396), (1062, 446)],
    [(1062, 446), (1043, 396), (1158, 354), (1130, 435)],
    [(463, 432), (476, 415), (477, 335), (612, 328), (622, 422), (588, 424)],
    [(622, 422), (612, 328), (754, 320), (780, 415)],
    [(780, 415), (754, 320), (886, 316), (925, 405)],
    [(925, 405), (886, 316), (1002, 315), (1043, 396)],
    [(1043, 396), (1002, 315), (1129, 310), (1158, 354)],
    [(477, 335), (612, 328), (602, 250), (477, 257)],
    [(602, 250), (612, 328), (754, 320), (730, 244)],
    [(730, 244), (754, 320), (886, 316), (852, 245)],
    [(852, 245), (886, 316), (1002, 315), (962, 244)],
    [(962, 244), (1002, 315), (1129, 310), (1090, 249)],
    [(482, 193), (477, 257), (602, 250), (594, 188)],
    [(594, 188), (602, 250), (730, 244), (711, 184)],
    [(711, 184), (730, 244), (852, 245), (823, 183)],
    [(823, 183), (852, 245), (962, 244), (925, 185)],
    [(925, 185), (962, 244), (1090, 249), (1052, 192)],
    [(486, 142), (482, 193), (594, 188), (587, 135)],
    [(587, 135), (594, 188), (711, 184), (696, 131)],
    [(696, 131), (711, 184), (823, 183), (799, 135)],
    [(799, 135), (823, 183), (925, 185), (892, 136)],
    [(892, 136), (925, 185), (1052, 192), (1018, 144)],
    [(492, 81), (486, 142), (587, 135), (581, 75)],
    [(581, 75), (587, 135), (696, 131), (676, 72)],
    [(676, 72), (696, 131), (799, 135), (768, 70)],
    [(768, 70), (799, 135), (892, 136), (853, 77)],
    [(853, 77), (892, 136), (1018, 144), (965, 87)],
    [(500, 5), (492, 81), (581, 75), (573, 5)],
    [(573, 5), (581, 75), (676, 72), (653, 5)],
    [(653, 5), (676, 72), (768, 70), (730, 5)],
    [(730, 5), (768, 70), (853, 77), (798, 4)],
    [(798, 4), (853, 77), (965, 87), (893, 5)],
    [(932, 6), (893, 5), (965, 87), (1014, 92)],
    [(1014, 92), (965, 87), (1018, 144), (1063, 149)],
    [(1063, 149), (1018, 144), (1052, 192), (1090, 249), (1137, 246)],
    [(1137, 246), (1090, 249), (1129, 310), (1158, 354), (1205, 338)],
]

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
border_states = {
    idx: {
        "sapu_time": None,
        "orang_time": None,
        "is_green": False,
        "person_and_broom_detected": False,
        "broom_overlap_time": 0.0,
        "last_broom_overlap_time": None,
    }
    for idx in range(len(borders))
}  # Menandai deteksi bersama  # Waktu akumulasi overlapping sapu  # Waktu terakhir overlapping sapu

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
        if (
            keypoints_data is not None
            and keypoints_data.xy is not None
            and keypoints_data.conf is not None
        ):
            if keypoints_data.shape[0] > 0:
                keypoints_array = keypoints_data.xy.cpu().numpy()  # shape (n, k, 2)
                keypoints_conf = keypoints_data.conf.cpu().numpy()  # shape (n, k)
                for keypoints_per_object, keypoints_conf_per_object in zip(
                    keypoints_array, keypoints_conf
                ):
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
def process_frame(frame, current_time, percentage_green):
    global start_time, end_time, elapsed_time, broom_absence_timer_start, border_states, first_green_time, is_counting
    frame_resized = cv2.resize(frame, (new_width, new_height))
    results_broom = process_model_broom(frame_resized)
    points_broom, coords_broom, keypoint_positions = export_frame_broom(
        results_broom, (0, 255, 0), pairs_broom, confidence_threshold=CONFIDENCE_THRESHOLD_BROOM
    )

    # Inisialisasi warna border berdasarkan state sebelumnya
    border_colors = [
        (0, 255, 0) if state["is_green"] else (0, 255, 255) for state in border_states.values()
    ]

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
                print("reset")
                if percentage_green >= 75:
                    # Simpan gambar sebelum reset terjadi
                    print("Green border is bigger than 75% and data is sent to server")
                    # Pastikan gambar yang disimpan memiliki elemen border warna dan informasi tambahan
                    if first_green_time is not None:
                        elapsed_time = current_time - first_green_time  # Update elapsed_time sebelum dikirim
                    overlay = frame_resized.copy()
                    alpha = 0.5  # Faktor Transparansi
                    for border_pt, color in zip(borders_pts, border_colors):
                        cv2.fillPoly(overlay, pts=[border_pt], color=color)
                    cv2.addWeighted(overlay, alpha, frame_resized, 1 - alpha, 0, frame_resized)
                    minutes, seconds = divmod(int(elapsed_time), 60)
                    time_str = f"Elapsed Time: {minutes:02d}:{seconds:02d}"
                    cvzone.putTextRect(
                        frame_resized, time_str, (10, 50), scale=1, thickness=2, offset=5
                    )
                    cvzone.putTextRect(
                        frame_resized,
                        f"Persentase Border Hijau: {percentage_green:.2f}%",
                        (10, 75),
                        scale=1,
                        thickness=2,
                        offset=5,
                    )
                    cvzone.putTextRect(
                        frame_resized, f"FPS: {int(fps)}", (10, 100), scale=1, thickness=2, offset=5
                    )
                    image_path = "main/images/green_borders_image_161.jpg"
                    cv2.imwrite(image_path, frame_resized)
                    send_to_server("10.5.0.2", percentage_green, elapsed_time, image_path)

                # Reset semua border menjadi kuning
                for idx in range(len(borders)):
                    border_states[idx] = {
                        "is_green": False,
                        "broom_overlap_time": 0.0,
                        "last_broom_overlap_time": None,
                    }
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


def server_address(host):
    if host == "localhost":
        user = "root"
        password = "robot123"
        database = "report_ai_cctv"
        port = 3306
    elif host == "10.5.0.2":
        user = "robot"
        password = "robot123"
        database = "report_ai_cctv"
        port = 3307
    return user, password, database, port


def send_to_server(host, percentage_green, elapsed_time, image_path):
    try:
        user, password, database, port = server_address(host)
        connection = pymysql.connect(
            host=host, user=user, password=password, database=database, port=port
        )
        cursor = connection.cursor()
        table = "empbro"
        camera_name = "10.5.0.161"
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Membaca file gambar dalam bentuk biner
        with open(image_path, "rb") as file:
            binary_image = file.read()

        query = f"""
        INSERT INTO {table} (cam, timestamp, percentage, elapsed_time, image)
        VALUES (%s, %s, %s, %s, %s)
        """
        cursor.execute(query, (camera_name, timestamp, percentage_green, elapsed_time, binary_image))
        connection.commit()
        print(f"Data berhasil dikirim ")  # Gantikan logger dengan print
    except pymysql.MySQLError as e:
        print(f"Error saat mengirim data : {e}")  # Gantikan logger dengan print
    finally:
        if "cursor" in locals():
            cursor.close()
        if "connection" in locals():
            connection.close()


if __name__ == "__main__":
    # Muat hanya Model Deteksi Sapu
    model_broom = YOLO("D:/SBHNL/Resources/Models/Pretrained/BROOM/B5_LARGE/weights/best.pt").to(
        "cuda"
    )  # Model Sapu
    model_broom.overrides["verbose"] = False
    # Verifikasi bahwa model berada di GPU
    print(f"Model Broom device: {next(model_broom.model.parameters()).device}")

    # Definisikan Sumber Video
    rtsp_url = "D:/SBHNL/Videos/AHMDL/Test/sapu.mp4"
    # rtsp_url = "rtsp://admin:oracle2015@10.5.0.161:554/Streaming/Channels/1"
    cap = cv2.VideoCapture(rtsp_url)
    if not cap.isOpened():
        print(f"Error: Cannot open video {rtsp_url}")
        exit()

    # Set Resolusi yang Diinginkan
    pairs_broom = [
        (0, 1),
        (1, 2),
        (2, 3),
        (2, 4),
    ]  # Definisikan pasangan keypoint untuk menggambar garis

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

        # Hitung dan Cetak Persentase Border Hijau
        total_borders = len(borders)
        green_borders = sum(1 for state in border_states.values() if state["is_green"])
        percentage_green = (green_borders / total_borders) * 100

        # Proses frame
        frame_resized = process_frame(frame, current_time, percentage_green)

        cvzone.putTextRect(
            frame_resized,
            f"Persentase Border Hijau: {percentage_green:.2f}%",
            (10, 75),
            scale=1,
            thickness=2,
            offset=5,
        )
        # Tampilkan Elapsed Time dan FPS
        cvzone.putTextRect(
            frame_resized, f"FPS: {int(fps)}", (10, 100), scale=1, thickness=2, offset=5
        )
        cv2.imshow("Broom and Person Detection", frame_resized)

        # Tekan 'n' untuk keluar
        if cv2.waitKey(1) & 0xFF == ord("n"):
            break

    cap.release()
    cv2.destroyAllWindows()
