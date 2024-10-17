import os
# Mengizinkan duplikat library jika diperlukan
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import cv2
from ultralytics import YOLO
import numpy as np
import time
import torch
import cvzone
import pymysql
from datetime import datetime, timedelta
import multiprocessing
import logging
import signal
import sys

# Konfigurasi Logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


# Parameter Konfigurasi
CONFIDENCE_THRESHOLD_BROOM = 0.9
BROOM_ABSENCE_THRESHOLD = 10  # Dalam detik
BROOM_TOUCH_THRESHOLD = 0  # Ganti ke 0 untuk menghilangkan waktu overlapping
PERCENTAGE_GREEN_THRESHOLD = 50

# Set Resolusi Asli dan Resolusi Baru
new_width, new_height = 640, 360  # Resolusi baru yang lebih rendah

# Hitung Faktor Skala
scale_x = new_width / 1280
scale_y = new_height / 720

# Mendefinisikan Borders (koordinat sudah sesuai dengan resolusi 1280x720)
borders = [
    [(29, 493), (107, 444), (168, 543), (81, 598)],
    [(168, 543), (182, 533), (194, 550), (297, 487), (245, 429), (138, 491)],
    [(194, 550), (297, 487), (390, 581), (269, 654)],
    [(269, 654), (390, 581), (509, 687), (466, 714), (318, 714)],
    [(466, 714), (684, 714), (579, 642), (509, 687)],
    [(509, 687), (579, 642), (646, 595), (518, 502), (390, 581)],
    [(390, 581), (518, 502), (414, 418), (297, 487)],
    [(245, 429), (268, 418), (255, 356), (309, 324), (414, 418), (297, 487)],
    [(579, 642), (646, 595), (710, 550), (843, 637), (758, 713), (684, 714)],
    [(309, 324), (414, 418), (528, 355), (406, 271)],
    [(406, 271), (500, 235), (628, 305), (528, 355)],
    [(518, 502), (414, 418), (528, 355), (641, 428)],
    [(518, 502), (646, 595), (710, 550), (766, 506), (641, 428)],
    [(710, 550), (843, 637), (941, 544), (816, 468), (766, 506)],
    [(758, 713), (843, 637), (975, 714)],
    [(975, 714), (843, 637), (941, 544), (1056, 616)],
    [(975, 714), (1114, 713), (1143, 665), (1056, 616)],
    [(1143, 665), (1056, 616), (1116, 528), (1189, 576)],
    [(1056, 616), (1116, 528), (1011, 463), (941, 544)],
    [(816, 468), (941, 544), (1011, 463), (899, 397)],
    [(528, 355), (641, 428), (764, 349), (662, 290), (628, 305)],
    [(641, 428), (766, 506), (816, 468), (875, 419), (764, 349)],
    [(875, 419), (899, 397), (968, 339), (868, 281), (764, 349)],
    [(764, 349), (868, 281), (777, 235), (662, 290)],
    [(899, 397), (1011, 463), (1069, 396), (968, 339)],
    [(1011, 463), (1116, 528), (1160, 451), (1069, 396)],
    [(1116, 528), (1189, 576), (1228, 492), (1160, 451)],
]

# Menghapus duplikat dalam setiap sub-list sambil menjaga urutan
cleaned_borders = [list(dict.fromkeys(sublist)) for sublist in borders]

# Skalakan Borders sesuai dengan resolusi baru
scaled_borders = []
for border in cleaned_borders:
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


# Fungsi untuk Mendapatkan Parameter Server
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
    else:
        user = "robot"
        password = "robot123"
        database = "report_ai_cctv"
        port = 3307
    return user, password, database, port


# Fungsi untuk Mengirim Data ke Server
def send_to_server(host, percentage_green, elapsed_time, image_path):
    try:
        user, password, database, port = server_address(host)
        connection = pymysql.connect(
            host=host, user=user, password=password, database=database, port=port
        )
        cursor = connection.cursor()
        table = "empbro"
        camera_name = host  # Menggunakan host sebagai nama kamera
        timestamp_done = datetime.now()  # Waktu selesai
        timestamp_start = timestamp_done - timedelta(seconds=elapsed_time)

        # Format timestamps untuk database
        timestamp_done_str = timestamp_done.strftime("%Y-%m-%d %H:%M:%S")
        timestamp_start_str = timestamp_start.strftime("%Y-%m-%d %H:%M:%S")

        # Baca file gambar dalam mode biner
        with open(image_path, "rb") as file:
            binary_image = file.read()

        query = f"""
        INSERT INTO {table} (cam, timestamp_start, timestamp_done, elapsed_time, percentage, image_done)
        VALUES (%s, %s, %s, %s, %s, %s)
        """
        cursor.execute(
            query,
            (
                camera_name,
                timestamp_start_str,
                timestamp_done_str,
                elapsed_time,
                percentage_green,
                binary_image,
            ),
        )
        connection.commit()
        logging.info(f"Data berhasil dikirim dari {camera_name}")
    except pymysql.MySQLError as e:
        logging.error(f"Error saat mengirim data dari {host}: {e}")
    finally:
        if "cursor" in locals():
            cursor.close()
        if "connection" in locals():
            connection.close()


# Fungsi untuk memproses setiap RTSP stream
def process_rtsp_stream(rtsp_url, pairs_broom, image_dir, shutdown_event):
    try:
        # Muat model YOLO di setiap proses
        model_broom = YOLO(
            "D:/SBHNL/Resources/Models/Pretrained/BROOM/B5_LARGE/weights/best.pt"
        ).to("cuda")
        model_broom.overrides["verbose"] = False
        logging.info(
            f"Model Broom device: {next(model_broom.model.parameters()).device} for {rtsp_url}"
        )
    except Exception as e:
        logging.error(f"Error saat memuat model YOLO untuk {rtsp_url}: {e}")
        return

    cap = cv2.VideoCapture(rtsp_url)
    if not cap.isOpened():
        logging.error(f"Error: Tidak dapat membuka video {rtsp_url}")
        return

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
    }

    # Variabel untuk Melacak Waktu
    first_green_time = None
    is_counting = False
    broom_absence_timer_start = None  # Timer untuk ketidakhadiran sapu overlapping border

    # Variabel untuk FPS
    prev_frame_time = time.time()
    fps = 0

    while not shutdown_event.is_set():
        ret, frame = cap.read()
        if not ret:
            logging.warning(f"Gagal membaca frame dari {rtsp_url}")
            break

        # Perhitungan FPS
        current_time = time.time()
        time_diff = current_time - prev_frame_time
        if time_diff > 0:
            fps = 1 / time_diff
        else:
            fps = 0
        prev_frame_time = current_time  # Perbarui waktu sebelumnya

        # Resize frame
        frame_resized = cv2.resize(frame, (new_width, new_height))

        # Proses model broom
        with torch.no_grad():
            results_broom = model_broom(frame_resized, imgsz=960)

        # Ekstrak keypoints
        points_broom, coords_broom, keypoint_positions = [], [], []
        for result in results_broom:
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
                            if kp_conf >= CONFIDENCE_THRESHOLD_BROOM:
                                x, y = kp[0], kp[1]
                                keypoints_list.append((int(x), int(y)))
                            else:
                                keypoints_list.append(
                                    None
                                )  # Keypoint diabaikan jika confidence rendah
                        keypoint_positions.append(keypoints_list)
                        for point in keypoints_list:
                            if point is not None:
                                points_broom.append(point)
                        for i, j in pairs_broom:
                            if i < len(keypoints_list) and j < len(keypoints_list):
                                if keypoints_list[i] is not None and keypoints_list[j] is not None:
                                    coords_broom.append(
                                        (keypoints_list[i], keypoints_list[j], (0, 255, 0))
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

        # Inisialisasi percentage_green
        percentage_green = 0.0

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
                    logging.info(f"Reset pada {rtsp_url}")
                    total_borders = len(borders)
                    green_borders = sum(1 for state in border_states.values() if state["is_green"])
                    percentage_green = (green_borders / total_borders) * 100

                    if percentage_green >= PERCENTAGE_GREEN_THRESHOLD:
                        # Simpan gambar sebelum reset terjadi
                        logging.info(
                            f"Green border lebih besar dari {PERCENTAGE_GREEN_THRESHOLD}% dan data dikirim ke server"
                        )
                        if first_green_time is not None:
                            elapsed_time = current_time - first_green_time
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
                            frame_resized,
                            f"FPS: {int(fps)}",
                            (10, 100),
                            scale=1,
                            thickness=2,
                            offset=5,
                        )
                        # Menghasilkan nama file unik berdasarkan RTSP URL
                        sanitized_url = rtsp_url.split("@")[-1].replace(":", "_").replace("/", "_")
                        image_path = os.path.join(
                            image_dir, f"green_borders_image_{sanitized_url}.jpg"
                        )
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
            # Calculate percentage_green when green_borders_exist is False
            total_borders = len(borders)
            green_borders = sum(1 for state in border_states.values() if state["is_green"])
            percentage_green = (green_borders / total_borders) * 100

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

        # Tambahkan teks Persentase Border Hijau dan FPS
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

        # Tampilkan frame
        cv2.imshow(f"Broom and Person Detection - {rtsp_url}", frame_resized)

        # Tekan 'n' untuk keluar dari stream ini
        if cv2.waitKey(1) & 0xFF == ord("n"):
            logging.info(f"Process untuk {rtsp_url} menerima sinyal untuk berhenti.")
            shutdown_event.set()
            break

    cap.release()
    cv2.destroyWindow(f"Broom and Person Detection - {rtsp_url}")
    logging.info(f"Process untuk {rtsp_url} telah dihentikan.")


# Fungsi untuk menangani shutdown
def shutdown_signal(signum, frame):
    logging.info("Menerima sinyal shutdown.")
    sys.exit(0)


def main():
    # Daftarkan handler untuk sinyal shutdown
    signal.signal(signal.SIGINT, shutdown_signal)
    signal.signal(signal.SIGTERM, shutdown_signal)

    # Direktori untuk menyimpan gambar
    image_dir = "main/images"
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)

    # Definisikan Sumber RTSP
    rtsp_urls = [
        "rtsp://admin:oracle2015@10.5.0.182:554/Streaming/Channels/1",
        "rtsp://admin:oracle2015@10.5.0.170:554/Streaming/Channels/1",
        "rtsp://admin:oracle2015@10.5.0.161:554/Streaming/Channels/1",
    ]

    # Definisikan Pasangan Keypoint untuk Sapu
    pairs_broom = [
        (0, 1),
        (1, 2),
        (2, 3),
        (2, 4),
    ]

    # Buat Event untuk shutdown
    shutdown_event = multiprocessing.Event()

    # Buat dan Mulai Proses untuk Setiap RTSP
    processes = []
    for rtsp_url in rtsp_urls:
        p = multiprocessing.Process(
            target=process_rtsp_stream,
            args=(rtsp_url, pairs_broom, image_dir, shutdown_event),
            name=f"Process-{rtsp_url}",
        )
        p.start()
        processes.append(p)
        logging.info(f"Process untuk {rtsp_url} telah dimulai.")

    try:
        # Tunggu semua proses selesai
        for p in processes:
            p.join()
    except KeyboardInterrupt:
        logging.info("Menerima KeyboardInterrupt. Menghentikan semua proses...")
        shutdown_event.set()
        for p in processes:
            p.terminate()
        for p in processes:
            p.join()
        logging.info("Semua proses telah dihentikan.")


if __name__ == "__main__":
    main()
