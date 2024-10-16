import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import cv2
from ultralytics import YOLO
import numpy as np
import time
import torch
import cvzone
import pymysql
from datetime import datetime, timedelta

# Parameter Konfigurasi Umum
CONFIDENCE_THRESHOLD_BROOM = 0.9
BROOM_TOUCH_THRESHOLD_DEFAULT = 0.00005  # Default, akan diubah per kamera
PERCENTAGE_GREEN_THRESHOLD = 75

# Set Resolusi Asli dan Resolusi Baru
original_width, original_height = 1280, 720  # Resolusi asli
new_width, new_height = 960, 540  # Resolusi baru yang lebih rendah

# Hitung Faktor Skala
scale_x = new_width / original_width
scale_y = new_height / original_height


# Fungsi untuk Mendefinisikan dan Menskalakan Borders
def scale_borders(borders, scale_x, scale_y):
    scaled_borders = []
    for border in borders:
        scaled_border = []
        for x, y in border:
            scaled_x = int(x * scale_x)
            scaled_y = int(y * scale_y)
            scaled_border.append((scaled_x, scaled_y))
        scaled_borders.append(scaled_border)
    return scaled_borders


# Fungsi untuk Mendapatkan Konfigurasi Server
def server_address(host):
    if host == "localhost":
        user = "root"
        password = "robot123"
        database = "report_ai_cctv"
        port = 3306
    elif host == "10.5.0.8":
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
        camera_name = image_path.split("_")[-1].split(".")[0]  # Extract camera name from image_path
        timestamp_done = datetime.now()  # Keep as datetime object
        timestamp_start = timestamp_done - timedelta(seconds=elapsed_time)

        # Format timestamps for database insertion
        timestamp_done_str = timestamp_done.strftime("%Y-%m-%d %H:%M:%S")
        timestamp_start_str = timestamp_start.strftime("%Y-%m-%d %H:%M:%S")

        # Read the image file in binary mode
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
        print(f"Data berhasil dikirim dari {camera_name}")
    except pymysql.MySQLError as e:
        print(f"Error saat mengirim data dari {camera_name}: {e}")
    finally:
        if "cursor" in locals():
            cursor.close()
        if "connection" in locals():
            connection.close()


# Fungsi untuk Memproses Deteksi Sapu
def process_model_broom(frame, model_broom):
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


# Fungsi untuk Inisialisasi State Border
def initialize_border_states(num_borders):
    return {
        b_idx: {
            "sapu_time": None,
            "orang_time": None,
            "is_green": False,
            "person_and_broom_detected": False,
            "broom_overlap_time": 0.0,
            "last_broom_overlap_time": None,
        }
        for b_idx in range(num_borders)
    }


# Fungsi untuk Memproses Setiap Frame (Per Kamera)
def process_frame_per_camera(
    frame,
    current_time,
    percentage_green,
    borders_pts,
    border_states,
    BROOM_ABSENCE_THRESHOLD,
    camera_name,
    model_broom,
    pairs_broom,
    fps,
    first_green_time,
    is_counting,
):
    print(f"Processing camera {camera_name}, border states: {border_states}")

    # Inisialisasi warna border berdasarkan state sebelumnya
    border_colors = [
        (0, 255, 0) if state and state.get("is_green", False) else (0, 255, 255)
        for state in border_states.values()
    ]
    # Logika untuk Timer Ketidakhadiran Sapu Overlapping Border
    green_borders_exist = any(
        isinstance(state, dict) and state.get("is_green", False) for state in border_states.values()
    )

    frame_resized = cv2.resize(frame, (new_width, new_height))
    results_broom = process_model_broom(frame_resized, model_broom)
    points_broom, coords_broom, keypoint_positions = export_frame_broom(
        results_broom, (0, 255, 0), pairs_broom, confidence_threshold=CONFIDENCE_THRESHOLD_BROOM
    )

    broom_overlapping_any_border = False

    for border_id, border_pt in enumerate(borders_pts):
        sapu_overlapping = False
        # Logika deteksi sapu overlapping di sini
        # Jika sapu terdeteksi, tandai sapu_overlapping
        if sapu_overlapping:
            broom_overlapping_any_border = True
            break
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

            if border_states[border_id]["broom_overlap_time"] >= BROOM_TOUCH_THRESHOLD_DEFAULT:
                border_states[border_id]["is_green"] = True
                border_colors[border_id] = (0, 255, 0)  # Ubah warna menjadi hijau
        else:
            border_states[border_id]["last_broom_overlap_time"] = None

    if green_borders_exist:
        if not is_counting:
            first_green_time = current_time
            is_counting = True

        if broom_overlapping_any_border:
            border_states["broom_absence_timer_start"] = None
        else:
            if border_states["broom_absence_timer_start"] is None:
                border_states["broom_absence_timer_start"] = current_time
            else:
                # Cek apakah waktu ketidakhadiran sapu melebihi threshold
                if (
                current_time - border_states["broom_absence_timer_start"]
            ) >= BROOM_ABSENCE_THRESHOLD:
                    # Logika reset jika sapu tidak terdeteksi dalam waktu tertentu
                    print(f"Reset for {camera_name} due to broom absence.")
                    # Lakukan reset di sini
                    border_states["broom_absence_timer_start"] = (
                    None  # Reset timer setelah deteksi reset
                )
            if "broom_absence_timer_start" not in border_states:
                border_states["broom_absence_timer_start"] = current_time
            elif (
                current_time - border_states["broom_absence_timer_start"]
            ) >= BROOM_ABSENCE_THRESHOLD:
                # Reset semua border dan timer
                print(f"Reset untuk {camera_name}")
                if percentage_green >= PERCENTAGE_GREEN_THRESHOLD:
                    # Simpan gambar sebelum reset terjadi
                    print(
                        f"Green border lebih besar dari {PERCENTAGE_GREEN_THRESHOLD}% dan data dikirim ke server"
                    )
                    if first_green_time is not None:
                        elapsed_time = (
                            current_time - first_green_time
                        )  # Update elapsed_time sebelum dikirim
                    overlay = frame_resized.copy()
                    alpha = 0.5  # Faktor Transparansi
                    for border_pt, color in zip(borders_pts, border_colors):
                        cv2.fillPoly(overlay, pts=[border_pt], color=color)

                    # Blend overlay dengan frame asli
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
                    image_path = f"main/images/green_borders_image_{camera_name}.jpg"
                    cv2.imwrite(image_path, frame_resized)
                    send_to_server("10.5.0.8", percentage_green, elapsed_time, image_path)

                # Reset semua border menjadi kuning
                for idx in range(len(borders_pts)):
                    border_states[idx] = {
                        "is_green": False,
                        "broom_overlap_time": 0.0,
                        "last_broom_overlap_time": None,
                    }
                    border_colors[idx] = (0, 255, 255)  # Kembalikan ke warna kuning
                first_green_time = None
                is_counting = False
                border_states["broom_absence_timer_start"] = None
    else:
        border_states["broom_absence_timer_start"] = None
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

    return frame_resized, first_green_time, is_counting


if __name__ == "__main__":
    # Muat Model Deteksi Sapu
    model_broom = YOLO("D:/SBHNL/Resources/Models/Pretrained/BROOM/B5_LARGE/weights/best.pt").to(
        "cuda"
    )
    model_broom.overrides["verbose"] = False
    # Verifikasi bahwa model berada di GPU
    print(f"Model Broom device: {next(model_broom.model.parameters()).device}")

    # Definisikan Sumber Video
    rtsp_urls = [
        {
            "url": "rtsp://admin:oracle2015@10.5.0.182:554/Streaming/Channels/1",
            "borders": [
                [(30, 493), (114, 439), (158, 510), (64, 567)],
                [(114, 439), (210, 383), (261, 448), (158, 510)],
                [(210, 383), (308, 326), (372, 384), (261, 448)],
                [(308, 326), (454, 247), (533, 296), (372, 384)],
                [(117, 667), (64, 567), (158, 510), (222, 601)],
                [(222, 601), (158, 510), (261, 448), (341, 530)],
                [(341, 530), (261, 448), (372, 384), (465, 459)],
                [(465, 459), (372, 384), (533, 296), (635, 357)],
                [(533, 296), (632, 247), (731, 303), (635, 357)],
                [(731, 303), (632, 247), (713, 208), (812, 258)],
                [(149, 715), (117, 667), (222, 601), (312, 713)],
                [(312, 713), (222, 601), (341, 530), (447, 634)],
                [(447, 634), (341, 530), (465, 459), (580, 547)],
                [(580, 547), (465, 459), (635, 357), (753, 428)],
                [(753, 428), (635, 357), (731, 303), (841, 365)],
                [(841, 365), (731, 303), (812, 258), (914, 311)],
                [(312, 713), (447, 634), (541, 715)],
                [(541, 715), (447, 634), (580, 547), (714, 641), (622, 713)],
                [(714, 641), (580, 547), (753, 428), (877, 506)],
                [(877, 506), (753, 428), (841, 365), (957, 432)],
                [(957, 432), (841, 365), (914, 311), (1014, 370)],
                [(622, 713), (714, 641), (825, 712)],
                [(825, 712), (714, 641), (877, 506), (996, 580), (877, 712)],
                [(996, 580), (877, 506), (957, 432), (1061, 496)],
                [(1061, 496), (957, 432), (1014, 370), (1110, 429)],
                [(877, 712), (996, 580), (1138, 663), (1108, 714)],
                [(1138, 663), (996, 580), (1061, 496), (1184, 573)],
                [(1184, 573), (1061, 496), (1110, 429), (1221, 504)],
            ],
            "BROOM_ABSENCE_THRESHOLD": 10,
            "camera_name": "10.5.0.182",
        },
        {
            "url": "rtsp://admin:oracle2015@10.5.0.170:554/Streaming/Channels/1",
            "borders": [
                [(688, 98), (737, 100), (739, 137), (684, 136)],
                [(790, 103), (737, 100), (739, 137), (803, 140)],
                [(684, 136), (739, 137), (743, 173), (679, 170)],
                [(803, 140), (739, 137), (743, 173), (814, 177)],
                [(679, 170), (743, 173), (747, 208), (672, 205)],
                [(814, 177), (743, 173), (747, 208), (826, 214)],
                [(672, 205), (747, 208), (752, 253), (668, 249)],
                [(826, 214), (747, 208), (752, 253), (839, 258)],
                [(668, 249), (752, 253), (755, 302), (662, 299)],
                [(839, 258), (752, 253), (755, 302), (854, 305)],
                [(662, 299), (755, 302), (759, 360), (657, 355)],
                [(854, 305), (755, 302), (759, 360), (869, 362)],
                [(657, 355), (759, 360), (760, 431), (645, 429)],
                [(869, 362), (759, 360), (760, 431), (883, 436)],
                [(645, 429), (760, 431), (760, 526), (631, 520)],
                [(883, 436), (760, 431), (760, 526), (904, 529)],
                [(631, 520), (760, 526), (762, 644), (606, 639)],
                [(904, 529), (760, 526), (762, 644), (923, 644)],
                [(606, 639), (762, 644), (923, 644), (932, 710), (596, 710)],
            ],
            "BROOM_ABSENCE_THRESHOLD": 30,
            "camera_name": "10.5.0.170",
        },
    ]

    # Membuat Window dan Inisialisasi untuk Setiap Kamera
    for cam in rtsp_urls:
        window_name = f"Broom and Person Detection {cam['camera_name']}"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 960, 540)

    # Membuka Semua RTSP Streams
    caps = []
    for cam in rtsp_urls:
        cap = cv2.VideoCapture(cam["url"])
        if not cap.isOpened():
            print(f"Error: Cannot open video {cam['url']}")
            exit()
        caps.append(cap)

    # Skalakan Borders dan Konversi ke Numpy Arrays
    scaled_borders_list = []
    borders_pts_list = []
    for cam in rtsp_urls:
        scaled_borders = scale_borders(cam["borders"], scale_x, scale_y)
        scaled_borders_list.append(scaled_borders)
        borders_pts = [np.array(border, np.int32) for border in scaled_borders]
        borders_pts_list.append(borders_pts)

    # Struktur Data untuk Menyimpan State Border (Per Kamera)
    border_states_list = []
    first_green_time_list = []
    is_counting_list = []

    for cam in rtsp_urls:
        num_borders = len(cam["borders"])
        border_states = initialize_border_states(num_borders)
        border_states_list.append(border_states)
        first_green_time_list.append(None)
        is_counting_list.append(False)

    # Verifikasi Inisialisasi
    for idx, border_states in enumerate(border_states_list):
        print(f"Initial border states for camera {rtsp_urls[idx]['camera_name']}:", border_states)

    # Variabel untuk FPS
    prev_frame_time_list = [time.time()] * len(rtsp_urls)
    fps_list = [0] * len(rtsp_urls)

    # Definisikan Pasangan Keypoint untuk Sapu
    pairs_broom = [
        (0, 1),
        (1, 2),
        (2, 3),
        (2, 4),
    ]  # Definisikan pasangan keypoint untuk menggambar garis

    while True:
        for cam_idx, cam in enumerate(rtsp_urls):
            ret, frame = caps[cam_idx].read()
            if not ret:
                print(f"Failed to read frame from {cam['camera_name']}")
                continue  # Lewati ke kamera berikutnya

            current_time = time.time()

            # Perhitungan FPS
            time_diff = current_time - prev_frame_time_list[cam_idx]
            if time_diff > 0:
                fps_list[cam_idx] = 1 / time_diff
            else:
                fps_list[cam_idx] = 0
            prev_frame_time_list[cam_idx] = current_time  # Perbarui waktu sebelumnya

            # Hitung dan Cetak Persentase Border Hijau
            total_borders = len(rtsp_urls[cam_idx]["borders"])
            green_borders = sum(
                1
                for state in border_states_list[cam_idx].values()
                if state and state.get("is_green", False)
            )
            percentage_green = (green_borders / total_borders) * 100

            # Debugging: Pastikan border_states_list[cam_idx] tidak None
            if border_states_list[cam_idx] is None:
                print(f"Error: border_states_list[{cam_idx}] is None for camera {cam['camera_name']}")
                continue

            # Proses frame
            frame_processed, updated_first_green_time, updated_is_counting = (
                process_frame_per_camera(
                    frame,
                    current_time,
                    percentage_green,
                    borders_pts_list[cam_idx],
                    border_states_list[cam_idx],
                    rtsp_urls[cam_idx]["BROOM_ABSENCE_THRESHOLD"],
                    rtsp_urls[cam_idx]["camera_name"],
                    model_broom,
                    pairs_broom,
                    fps_list[cam_idx],
                    first_green_time_list[cam_idx],
                    is_counting_list[cam_idx],
                )
            )

            # Update nilai first_green_time_list dan is_counting_list
            first_green_time_list[cam_idx] = updated_first_green_time
            is_counting_list[cam_idx] = updated_is_counting

            # Tambahkan informasi Persentase Border Hijau dan FPS
            cvzone.putTextRect(
                frame_processed,
                f"Persentase Border Hijau: {percentage_green:.2f}%",
                (10, 75),
                scale=1,
                thickness=2,
                offset=5,
            )
            cvzone.putTextRect(
                frame_processed,
                f"FPS: {int(fps_list[cam_idx])}",
                (10, 100),
                scale=1,
                thickness=2,
                offset=5,
            )

            # Tampilkan Frame di Window yang Sesuai
            window_name = f"Broom and Person Detection {cam['camera_name']}"
            cv2.imshow(window_name, frame_processed)

        # Tekan 'n' untuk keluar dari loop
        if cv2.waitKey(1) & 0xFF == ord("n"):
            break

    # Release Semua VideoCapture dan Tutup Semua Jendela
    for cap in caps:
        cap.release()
    cv2.destroyAllWindows()
