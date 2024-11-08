import cv2
import numpy as np
import time
import os
from datetime import datetime, timedelta
import pickle
import mysql.connector
import threading
import queue
import logging

# Konfigurasi logging
logging.basicConfig(
    filename="D:/AI_SOURCE/CUTTING9/monitoring.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

# Konfigurasi database
db_config = {
    "host": "10.5.0.2",
    "user": "robot",
    "password": "robot123",
    "database": "report_ai_cctv",
    "port": 3307,
}

# Direktori screenshot unik untuk GUDANG_5s
screenshot_save_path_line = "D:/TRIAL/AI_SOURCE/CUTTING9/RED_LINE"
screenshot_save_path_area = "D:/TRIAL/AI_SOURCE/CUTTING9/GREEN_AREA"

# Pastikan direktori screenshot ada
os.makedirs(screenshot_save_path_line, exist_ok=True)
os.makedirs(screenshot_save_path_area, exist_ok=True)

# Penjadwalan file PKL berdasarkan waktu
schedule = [
    (7, 0, "D:/AI_SOURCE/CUTTING9/CUTTING9.pkl"),
    (8, 0, "D:/AI_SOURCE/CUTTING9/CUTTING9a.pkl"),
    (9, 0, "D:/AI_SOURCE/CUTTING9/CUTTING9b.pkl"),
    (10, 0, "D:/AI_SOURCE/CUTTING9/CUTTING9c.pkl"),
    (11, 0, "D:/AI_SOURCE/CUTTING9/CUTTING9d.pkl"),
    (12, 0, "D:/AI_SOURCE/CUTTING9/CUTTING9e.pkl"),
    (13, 0, "D:/AI_SOURCE/CUTTING9/CUTTING9f.pkl"),
    (14, 0, "D:/AI_SOURCE/CUTTING9/CUTTING9g.pkl"),
    (15, 0, "D:/AI_SOURCE/CUTTING9/CUTTING9h.pkl"),
    (16, 0, "D:/AI_SOURCE/CUTTING9/CUTTING9i.pkl"),
    (17, 0, "D:/AI_SOURCE/CUTTING9/CUTTING9j.pkl"),
    (18, 0, "D:/AI_SOURCE/CUTTING9/CUTTING9NET.pkl"),
]

# Sort schedule berdasarkan waktu
schedule = sorted(schedule, key=lambda x: (x[0], x[1]))


# Fungsi untuk memuat data garis dan polygon
def load_data(pkl_file):
    global lines, polygons, line_id_counter, polygon_id_counter
    try:
        with open(pkl_file, "rb") as f:
            data = pickle.load(f)
            lines = data.get("lines", [])
            polygons = data.get("polygons", [])
            line_id_counter = data.get("line_id_counter", 1)
            polygon_id_counter = data.get("polygon_id_counter", 1)
        logging.info(f"Data garis dan polygon telah dimuat dari {pkl_file}.")
    except FileNotFoundError:
        logging.warning(f"Tidak ada data garis dan polygon yang disimpan sebelumnya di {pkl_file}.")
    except Exception as e:
        logging.error(f"Error memuat data dari {pkl_file}: {e}")


# Fungsi untuk menyimpan data garis dan polygon
def save_data(pkl_file):
    data = {
        "lines": lines,
        "polygons": polygons,
        "line_id_counter": line_id_counter,
        "polygon_id_counter": polygon_id_counter,
    }
    try:
        with open(pkl_file, "wb") as f:
            pickle.dump(data, f)
        logging.info(f"Data garis dan polygon telah disimpan ke {pkl_file}.")
    except Exception as e:
        logging.error(f"Error menyimpan data ke {pkl_file}: {e}")


# Fungsi untuk memuat baseline histogram dari file PKL baru
def load_baseline_histogram(pkl_file):
    try:
        with open(pkl_file, "rb") as f:
            data = pickle.load(f)
            baseline_lines = data.get("lines", [])
            baseline_polygons = data.get("polygons", [])

            # Update default_histogram untuk garis
            for bl_line in baseline_lines:
                for line in lines:
                    if line["id"] == bl_line["id"]:
                        line["default_histogram"] = bl_line.get("default_histogram")
                        logging.info(f"Default histogram diperbarui untuk {line['id']}")

            # Update default_histogram untuk polygon
            for bl_polygon in baseline_polygons:
                for polygon in polygons:
                    if polygon["id"] == bl_polygon["id"]:
                        polygon["default_histogram"] = bl_polygon.get("default_histogram")
                        logging.info(f"Default histogram diperbarui untuk {polygon['id']}")
        logging.info(f"Baseline histogram telah dimuat dari {pkl_file}.")
    except FileNotFoundError:
        logging.warning(f"Baseline file {pkl_file} tidak ditemukan.")
    except Exception as e:
        logging.error(f"Error memuat baseline histogram dari {pkl_file}: {e}")


# Fungsi untuk menginisialisasi jadwal PKL
def initialize_pkl_schedule(schedule):
    current_time = datetime.now()
    last_switch_time = None
    last_pkl_path = None
    next_switch_time = None
    next_pkl_path = None

    for hour, minute, pkl_path in schedule:
        try:
            switch_time = current_time.replace(hour=hour, minute=minute, second=0, microsecond=0)
        except ValueError:
            # Jika waktu switch sudah berlalu (misalnya, 24 jam)
            switch_time = (current_time + timedelta(days=1)).replace(hour=hour, minute=minute, second=0, microsecond=0)
        if current_time >= switch_time:
            last_switch_time = switch_time
            last_pkl_path = pkl_path
        else:
            if next_switch_time is None:
                next_switch_time = switch_time
                next_pkl_path = pkl_path

    if last_pkl_path is None:
        # Semua switch time belum terjadi hari ini, gunakan file terakhir dari hari kemarin
        last_pkl_path = "D:/AI_SOURCE/CUTTING9/CUTTING9NET.pkl"
    if next_switch_time is None:
        # Tidak ada switch time tersisa hari ini, set switch time pertama besok
        first_switch = schedule[0]
        next_switch_time = (current_time + timedelta(days=1)).replace(hour=first_switch[0], minute=first_switch[1], second=0, microsecond=0)
        next_pkl_path = first_switch[2]

    return last_pkl_path, next_switch_time, next_pkl_path


# Memuat baseline histogram ke dalam lines dan polygons
def update_baseline_from_pkl(pkl_file):
    load_baseline_histogram(pkl_file)


# Fungsi callback untuk event mouse
def draw(event, x, y, flags, param):
    global drawing, start_point, end_point, lines, polygon_points, polygon_mode, current_mouse_position, line_id_counter, polygon_id_counter

    current_mouse_position = (x, y)  # Update posisi mouse saat ini

    if polygon_mode:
        if event == cv2.EVENT_LBUTTONDOWN:
            polygon_points.append((x, y))  # Tambah titik baru ke polygon
    else:
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            start_point = (x, y)
            end_point = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE and drawing:
            end_point = (x, y)
        elif event == cv2.EVENT_LBUTTONUP and drawing:
            drawing = False
            end_point = (x, y)
            lines.append(
                {
                    "id": f"LINE {line_id_counter}",
                    "start_point": start_point,
                    "end_point": end_point,
                    "default_histogram": None,
                    "line_detected": False,
                    "color": (0, 255, 255),
                }
            )
            logging.info(f"Garis {line_id_counter} ditambahkan dari {start_point} ke {end_point}.")
            line_id_counter += 1


def calculate_histogram(gray_frame, mask):
    """Hitung histogram dari area yang di-mask."""
    hist = cv2.calcHist([gray_frame], [0], mask, [256], [0, 256])
    cv2.normalize(hist, hist)
    return hist


def check_histogram_change(current_histogram, default_histogram, sensitivity=0.6):
    """Bandingkan histogram saat ini dengan histogram default menggunakan korelasi."""

    # Pastikan default_histogram adalah numpy array
    if isinstance(default_histogram, list):
        if len(default_histogram) > 0:
            # Ambil rata-rata histogram jika default_histogram adalah list dari beberapa histogram
            default_histogram = np.mean(default_histogram, axis=0)
        else:
            print("Error: Default histogram kosong.")
            return False

    # Lakukan perbandingan dengan korelasi
    hist_diff = cv2.compareHist(default_histogram, current_histogram, cv2.HISTCMP_CORREL)

    # Sensitivitas menentukan batasan perbedaan yang dapat diterima
    return hist_diff < (1.0 - sensitivity)


def start_countdown(object_id, object_type):
    """Mulai countdown selama 10 detik untuk garis atau polygon merah."""
    if object_id not in countdown_timers and object_id not in finished_countdown:
        countdown_timers[object_id] = {
            "start_time": time.time(),
            "captured": False,
            "path_capture": "0",
            "last_insert_time": 0,
            "violation_start_time": None,
            "violation_last_insert_time": 0,
            "first_violation": True,
            "is_in_violation": False,
        }  # Tambahkan flag ini
        logging.info(f"Countdown dimulai untuk {object_type} {object_id}.")


def capture_and_save_screenshot(frame, object_id, mid_point, save_path, object_type):
    """Ambil screenshot dan crop gambar dari tengah garis atau polygon dengan resolusi 600x600 piksel."""
    x, y = mid_point
    crop_size = 300  # Setengah dari ukuran 600x600
    height, width = frame.shape[:2]

    # Tentukan batas crop
    x1 = max(0, x - crop_size)
    y1 = max(0, y - crop_size)
    x2 = min(width, x + crop_size)
    y2 = min(height, y + crop_size)

    # Crop gambar
    cropped_img = frame[y1:y2, x1:x2]

    # Tentukan nama file berdasarkan waktu saat ini
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{object_type}_{object_id}_{timestamp}.jpg"

    # Path penyimpanan
    save_path = os.path.join(save_path, filename)

    # Simpan gambar
    cv2.imwrite(save_path, cropped_img)
    logging.info(f"Gambar disimpan: {save_path}")
    return save_path  # Kembalikan path untuk disimpan ke database


def insert_into_database(timestamp, cam_name, no_line, time_detect, status, path_capture):
    """Fungsi untuk memasukkan data ke tabel line_detectx."""
    try:
        sql = """INSERT INTO line_detect2 (timestamp, cam_name, no_line, time_detect, status, path_capture)
                 VALUES (%s, %s, %s, %s, %s, %s)"""
        val = (timestamp, cam_name, no_line, time_detect, status, path_capture)
        cursor.execute(sql, val)
        logging.info(f"Data dimasukkan ke database: {val}")
    except mysql.connector.Error as err:
        logging.error(f"Error inserting into database: {err}")
    except Exception as e:
        logging.error(f"Unexpected error: {e}")


def display_countdown_and_violation(frame, save_path, object_type="LINE", y_start=50, scale=(1, 1)):
    """Menangani countdown, violation, dan memasukkan data ke database secara realtime."""
    current_time = time.time()
    for object_id, timer_data in list(countdown_timers.items()):
        # Hanya proses ID yang sesuai dengan object_type
        if (object_type == "LINE" and object_id.startswith("LINE")) or (object_type == "AREA" and object_id.startswith("AREA")):
            elapsed_time = current_time - timer_data["start_time"]
            remaining_time = int(180 - elapsed_time)
            if remaining_time > 0:
                # Status: warning
                status = "warning"
                # Insert data setiap detik dengan path_capture="0"
                if int(elapsed_time) > timer_data["last_insert_time"]:
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    insert_into_database(timestamp, "CUTTING9", object_id, 1, status, "0")
                    countdown_timers[object_id]["last_insert_time"] = int(elapsed_time)

                # Capture pada detik ke-3 countdown (remaining_time == 177)
                if remaining_time == 177 and not timer_data["captured"]:
                    if object_type == "LINE":
                        for line in lines:
                            if line["id"] == object_id:
                                # Sesuaikan skala titik tengah
                                mid_point_scaled = (
                                    int((line["start_point"][0] + line["end_point"][0]) / 2 * scale[0]),
                                    int((line["start_point"][1] + line["end_point"][1]) / 2 * scale[1]),
                                )
                                path = capture_and_save_screenshot(frame, object_id, mid_point_scaled, save_path, object_type)
                                countdown_timers[object_id]["path_capture"] = path
                                countdown_timers[object_id]["captured"] = True
                                # Insert data dengan path_capture
                                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                                insert_into_database(timestamp, "CUTTING9", object_id, 1, status, path)
                    elif object_type == "AREA":
                        for polygon in polygons:
                            if polygon["id"] == object_id:
                                # Sesuaikan skala centroid
                                centroid_scaled = tuple((np.mean(polygon["points"], axis=0) * scale).astype(int))
                                path = capture_and_save_screenshot(frame, object_id, centroid_scaled, save_path, object_type)
                                countdown_timers[object_id]["path_capture"] = path
                                countdown_timers[object_id]["captured"] = True
                                # Insert data dengan path_capture
                                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                                insert_into_database(timestamp, "CUTTING9", object_id, 1, status, path)

                # Gambar countdown
                y_position = y_start
                x_position = int(frame.shape[1] - 100 * scale[0])
                cv2.putText(
                    frame,
                    f"{object_id}: {remaining_time}s",
                    (x_position, y_position),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (255, 0, 0),
                    1,
                    cv2.LINE_AA,
                )
                y_start += 15
            else:
                # Status: violation
                status = "violation"
                # Set violation_start_time jika belum di-set
                if timer_data.get("violation_start_time") is None:
                    countdown_timers[object_id]["violation_start_time"] = current_time
                    countdown_timers[object_id]["is_in_violation"] = True  # Set flag menjadi True
                    logging.info(f"Violation dimulai untuk {object_type} {object_id}.")

                violation_start_time = countdown_timers[object_id]["violation_start_time"]
                violation_elapsed = current_time - violation_start_time
                violation_seconds = int(violation_elapsed)

                # Insert data setiap detik selama violation dengan path_capture sesuai kondisi
                if violation_seconds > countdown_timers[object_id].get("violation_last_insert_time", 0):
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                    # Logika untuk path_capture
                    if countdown_timers[object_id].get("first_violation", True):
                        # Pelanggaran pertama: gunakan path_capture dari warning sebelumnya
                        path_capture = timer_data.get("path_capture", "0")
                        countdown_timers[object_id]["first_violation"] = False  # Set flag menjadi False setelah pelanggaran pertama
                        logging.info(f"Pelanggaran pertama untuk {object_id}: path_capture disamakan dengan warning sebelumnya.")
                    else:
                        # Pelanggaran berikutnya: set path_capture ke "0"
                        path_capture = "0"
                        logging.info(f"Pelanggaran berikutnya untuk {object_id}: path_capture diisi '0'.")

                    insert_into_database(timestamp, "CUTTING9", object_id, 1, status, path_capture)
                    countdown_timers[object_id]["violation_last_insert_time"] = violation_seconds

                # Update warna dan status
                if object_type == "LINE":
                    for line in lines:
                        if line["id"] == object_id:
                            line["color"] = (0, 0, 255)  # Merah
                elif object_type == "AREA":
                    for polygon in polygons:
                        if polygon["id"] == object_id:
                            polygon["color"] = (0, 0, 255)  # Merah

                # Gambar garis atau polygon dengan warna merah
                if object_type == "LINE":
                    for line in lines:
                        if line["id"] == object_id:
                            cv2.line(
                                frame,
                                (int(line["start_point"][0] * scale[0]), int(line["start_point"][1] * scale[1])),
                                (int(line["end_point"][0] * scale[0]), int(line["end_point"][1] * scale[1])),
                                line["color"],
                                1,
                            )
                            mid_point_scaled = (
                                int((line["start_point"][0] + line["end_point"][0]) / 2 * scale[0]),
                                int((line["start_point"][1] + line["end_point"][1]) / 2 * scale[1]),
                            )
                            cv2.putText(
                                frame,
                                f"{line['id']}",
                                mid_point_scaled,
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.4,
                                (255, 255, 255),
                                1,
                                cv2.LINE_AA,
                            )
                elif object_type == "AREA":
                    for polygon in polygons:
                        if polygon["id"] == object_id:
                            scaled_points = [(int(x * scale[0]), int(y * scale[1])) for x, y in polygon["points"]]
                            cv2.polylines(
                                frame,
                                [np.array(scaled_points)],
                                isClosed=True,
                                color=polygon["color"],
                                thickness=1,
                            )
                            centroid_scaled = tuple((np.mean(polygon["points"], axis=0) * scale).astype(int))
                            cv2.putText(
                                frame,
                                f"{polygon['id']}",
                                centroid_scaled,
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.4,
                                (255, 255, 255),
                                1,
                                cv2.LINE_AA,
                            )

                # Gambar pesan violation
                message = f"Terjadi Pelanggaran {violation_seconds}s Di {object_id}"
                cv2.putText(
                    frame,
                    message,
                    (20, y_start),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 255),
                    1,
                    cv2.LINE_AA,
                )
                y_start += 15

                # Tambahkan object_id ke violation_timers jika belum ada
                if object_id not in violation_timers:
                    violation_timers[object_id] = True  # Nilai tidak penting, hanya untuk penanda


def clear_data_for_deleted_object(object_id, frame, scale=(1, 1)):
    """Fungsi untuk menghapus countdown, pelanggaran, dan status screenshot saat garis atau area dihapus."""
    if object_id in countdown_timers:
        del countdown_timers[object_id]
        logging.info(f"Countdown dihapus untuk {object_id}.")
    if object_id in violation_timers:
        object_type = "LINE" if "LINE" in object_id else "AREA"
        save_path = screenshot_save_path_line if object_type == "LINE" else screenshot_save_path_area
        handle_violation_end(object_id, object_type, frame, save_path, scale)  # Pastikan fungsi ini diupdate sesuai kebutuhan
    finished_countdown.discard(object_id)
    logging.info(f"Pelanggaran dihapus untuk {object_id}.")


def handle_violation_end(object_id, object_type, frame, save_path, scale=(1, 1)):
    """Menghitung total waktu pelanggaran dan memasukkan data ke database dengan status 'done'."""
    if object_id in countdown_timers:
        timer_data = countdown_timers[object_id]

        # Set time_detect ke "1"
        time_detect = 1

        # Capture fullscreen
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"FULLSCREEN_{object_id}_{timestamp}.jpg"
        save_file_path = os.path.join(save_path, filename)

        cv2.imwrite(save_file_path, frame)  # Capture fullscreen
        logging.info(f"Fullscreen gambar disimpan: {save_file_path}")

        # Status done
        status = "done"

        # Insert data ke database
        timestamp_db = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cam_name = "CUTTING9"
        no_line = object_id

        insert_into_database(timestamp_db, cam_name, no_line, time_detect, status, save_file_path)
        logging.info(f"Pelanggaran untuk {object_id} telah selesai dan data 'done' dimasukkan ke database.")

        # Hapus data pelanggaran
        del countdown_timers[object_id]
        if object_id in violation_timers:
            del violation_timers[object_id]
        finished_countdown.discard(object_id)


def check_polygon_violation(frame, gray_frame, scale=(1, 1)):
    """Fungsi untuk memeriksa pelanggaran pada polygon (mode hijau)"""
    for polygon in polygons:
        mask = np.zeros_like(gray_frame)

        # Buat mask untuk polygon
        scaled_points = [(int(x * scale[0]), int(y * scale[1])) for x, y in polygon["points"]]
        cv2.fillPoly(mask, [np.array(scaled_points)], 255)

        # Hitung histogram dari area polygon
        current_histogram = calculate_histogram(gray_frame, mask)

        # Jika histogram baseline belum disimpan, simpan sekarang
        if polygon.get("default_histogram") is None:
            polygon["default_histogram"] = current_histogram.copy()
            logging.info(f"Baseline histogram disimpan untuk {polygon['id']}.")

        # Bandingkan histogram saat ini dengan histogram baseline
        if check_histogram_change(current_histogram, polygon["default_histogram"]):
            polygon["color"] = (0, 0, 255)  # Merah jika ada perubahan
            start_countdown(polygon["id"], "AREA")  # Mulai countdown jika ada perubahan
        else:
            polygon["color"] = (0, 255, 0)  # Hijau jika tidak ada perubahan
            if polygon["id"] in countdown_timers:
                if countdown_timers[polygon["id"]].get("is_in_violation"):
                    # Violation telah berakhir
                    handle_violation_end(polygon["id"], "AREA", frame, screenshot_save_path_area, scale)
                else:
                    # Hapus countdown jika kembali normal sebelum violation
                    del countdown_timers[polygon["id"]]
                    logging.info(f"Countdown dihapus karena {polygon['id']} kembali normal.")
                if polygon["id"] in violation_timers:
                    handle_violation_end(polygon["id"], "AREA", frame, screenshot_save_path_area, scale)

        # Gambar outline polygon
        cv2.polylines(
            frame,
            [np.array(scaled_points)],
            isClosed=True,
            color=polygon["color"],
            thickness=1,
        )

        # Tampilkan ID di tengah polygon
        centroid = tuple((np.mean(polygon["points"], axis=0) * scale).astype(int))
        cv2.putText(
            frame,
            f"{polygon['id']}",
            centroid,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )


def frame_capture_thread(rtsp_url, frame_queue, stop_event):
    """Thread untuk mengambil frame dari RTSP dan memasukkannya ke dalam queue."""
    while not stop_event.is_set():
        cap = cv2.VideoCapture(rtsp_url)
        if not cap.isOpened():
            logging.error("Tidak dapat membuka stream RTSP. Mencoba kembali dalam 5 detik...")
            cap.release()
            time.sleep(5)
            continue

        logging.info("Berhasil terhubung ke RTSP stream.")

        while not stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                logging.error("Gagal membaca frame dari RTSP. Mencoba kembali...")
                cap.release()
                time.sleep(5)
                break
            try:
                if not frame_queue.full():
                    frame_queue.put(frame, block=False)
            except queue.Full:
                logging.warning("Queue frame penuh. Melewati frame ini.")
                continue

    cap.release()
    logging.info("Thread pengambilan frame dihentikan.")


def main_monitoring():
    # Inisialisasi variabel global
    global lines, polygons, line_id_counter, polygon_id_counter
    global countdown_timers, violation_timers, finished_countdown
    global drawing, polygon_mode, start_point, end_point, polygon_points, current_mouse_position
    global cursor, db_connection

    lines = []
    polygons = []
    line_id_counter = 1
    polygon_id_counter = 1

    countdown_timers = {}
    violation_timers = {}
    finished_countdown = set()

    drawing = False
    polygon_mode = False
    start_point = None
    end_point = None
    polygon_points = []
    current_mouse_position = (0, 0)

    # Koneksi ke database
    try:
        db_connection = mysql.connector.connect(**db_config)
        db_connection.autocommit = True  # Aktifkan autocommit
        cursor = db_connection.cursor()
        logging.info("Koneksi ke database berhasil.")
    except mysql.connector.Error as err:
        logging.error(f"Error koneksi database: {err}")
        exit()

    # Panggil fungsi load_data() pada startup dengan PKL pertama sesuai jadwal
    current_pkl_file, next_switch_time, next_pkl_path = initialize_pkl_schedule(schedule)
    load_data(current_pkl_file)  # Memuat garis dan polygon dari PKL pertama
    update_baseline_from_pkl(current_pkl_file)  # Memuat baseline histogram dari PKL pertama

    cv2.namedWindow("RTSP Stream - Monitoring")
    cv2.setMouseCallback("RTSP Stream - Monitoring", draw)

    # Setup frame queue dan thread
    frame_queue = queue.Queue(maxsize=10)
    stop_event = threading.Event()
    rtsp_url = "rtsp://admin:oracle2015@10.5.0.7:554/Streaming/Channels/1"
    frame_thread = threading.Thread(target=frame_capture_thread, args=(rtsp_url, frame_queue, stop_event))
    frame_thread.daemon = True
    frame_thread.start()

    # Tentukan ukuran frame yang diinginkan
    desired_width = 854
    desired_height = 480
    scale_x = desired_width / 1280  # Asumsi ukuran asli RTSP stream adalah 1280x720
    scale_y = desired_height / 720
    scale = (scale_x, scale_y)

    while True:
        try:
            frame = frame_queue.get(timeout=1)  # Tunggu hingga frame tersedia
        except queue.Empty:
            logging.warning("Queue frame kosong. Menunggu frame berikutnya...")
            continue

        try:
            # Ubah ukuran frame sesuai skala
            frame_resized = cv2.resize(frame, (desired_width, desired_height))
            gray_frame = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)

            # Gambar garis dan deteksi perubahan menggunakan histogram
            for line in lines:
                if line["end_point"]:
                    # Buat mask untuk area garis
                    mask = np.zeros_like(gray_frame)
                    scaled_start = (int(line["start_point"][0] * scale_x), int(line["start_point"][1] * scale_y))
                    scaled_end = (int(line["end_point"][0] * scale_x), int(line["end_point"][1] * scale_y))
                    cv2.line(mask, scaled_start, scaled_end, 255, thickness=5)

                    # Hitung histogram dari area garis
                    current_histogram = calculate_histogram(gray_frame, mask)

                    # Jika histogram baseline belum disimpan, simpan sekarang
                    if line["default_histogram"] is None:
                        line["default_histogram"] = current_histogram.copy()
                        logging.info(f"Baseline histogram disimpan untuk {line['id']}.")

                    # Bandingkan histogram saat ini dengan histogram baseline
                    if check_histogram_change(current_histogram, line["default_histogram"]):
                        line["color"] = (0, 0, 255)  # Merah jika ada perubahan
                        start_countdown(line["id"], "LINE")  # Mulai countdown saat garis berubah menjadi merah
                    else:
                        line["color"] = (0, 255, 255)  # Kuning jika tidak ada perubahan
                        if line["id"] in countdown_timers:
                            if countdown_timers[line["id"]].get("is_in_violation"):
                                # Violation telah berakhir
                                handle_violation_end(line["id"], "LINE", frame_resized, screenshot_save_path_line, scale)
                            else:
                                # Hapus countdown jika kembali normal sebelum violation
                                del countdown_timers[line["id"]]
                                logging.info(f"Countdown dihapus karena {line['id']} kembali normal.")
                            if line["id"] in violation_timers:
                                handle_violation_end(line["id"], "LINE", frame_resized, screenshot_save_path_line, scale)

                    # Gambar garis
                    cv2.line(frame_resized, scaled_start, scaled_end, line["color"], 1)

                    # Tampilkan ID di tengah garis
                    mid_point = (
                        int((scaled_start[0] + scaled_end[0]) / 2),
                        int((scaled_start[1] + scaled_end[1]) / 2),
                    )
                    cv2.putText(
                        frame_resized,
                        f"{line['id']}",
                        mid_point,
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.4,
                        (255, 255, 255),
                        1,
                        cv2.LINE_AA,
                    )

            # Periksa pelanggaran pada polygon
            check_polygon_violation(frame_resized, gray_frame, scale)

            # Tampilkan countdown dan violation untuk mode kuning di kanan atas
            display_countdown_and_violation(frame_resized, screenshot_save_path_line, "LINE", y_start=30, scale=scale)

            # Tampilkan countdown dan violation untuk mode hijau (polygon)
            display_countdown_and_violation(frame_resized, screenshot_save_path_area, "AREA", y_start=100, scale=scale)

            # Gambar polygon yang sedang digambar (warna hijau)
            if polygon_mode and polygon_points:
                for i in range(len(polygon_points) - 1):
                    start = (int(polygon_points[i][0] * scale_x), int(polygon_points[i][1] * scale_y))
                    end = (int(polygon_points[i + 1][0] * scale_x), int(polygon_points[i + 1][1] * scale_y))
                    cv2.line(frame_resized, start, end, (0, 255, 0), 1)
                last_point_scaled = (int(current_mouse_position[0] * scale_x), int(current_mouse_position[1] * scale_y))
                cv2.line(frame_resized, start, last_point_scaled, (0, 255, 0), 1)

            # Gambar garis kuning sementara saat mouse sedang digerakkan
            if not polygon_mode and drawing:
                start_scaled = (int(start_point[0] * scale_x), int(start_point[1] * scale_y))
                end_scaled = (int(current_mouse_position[0] * scale_x), int(current_mouse_position[1] * scale_y))
                cv2.line(frame_resized, start_scaled, end_scaled, (0, 255, 255), 1)  # Tetap menampilkan garis kuning saat ditarik

            # Tambahkan status mode ke layar
            if polygon_mode:
                cv2.putText(
                    frame_resized,
                    "MODE: AREA",
                    (20, desired_height - 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (0, 255, 0),
                    1,
                    cv2.LINE_AA,
                )
            else:
                cv2.putText(
                    frame_resized,
                    "MODE: LINE",
                    (20, desired_height - 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (0, 255, 255),
                    1,
                    cv2.LINE_AA,
                )

            # Tampilkan hasilnya
            cv2.imshow("RTSP Stream - Monitoring", frame_resized)

            # Cek input keyboard untuk keluar atau mengubah mode
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                save_data(current_pkl_file)  # Simpan data saat program akan ditutup
                logging.info("Program dihentikan oleh pengguna. Data disimpan.")
                break
            elif key == ord("u"):
                # Ubah mode antara garis kuning dan polygon (hijau)
                if polygon_mode:
                    polygon_mode = False
                    polygon_points.clear()  # Reset polygon points setelah selesai menggambar
                    logging.info("Mode diubah ke LINE.")
                else:
                    polygon_mode = True
                    logging.info("Mode diubah ke AREA.")
            elif key == ord("r") and polygon_mode:
                # Selesaikan polygon saat tombol R ditekan
                if len(polygon_points) > 2:
                    polygon_id = f"AREA {polygon_id_counter}"
                    polygons.append(
                        {
                            "points": polygon_points.copy(),
                            "id": polygon_id,
                            "default_histogram": None,
                            "color": (0, 255, 0),
                        }
                    )
                    logging.info(f"Polygon {polygon_id} ditambahkan dengan titik: {polygon_points}.")
                    polygon_id_counter += 1
                polygon_points.clear()  # Reset polygon points setelah selesai menggambar
            elif key == ord("c"):
                # Hapus objek terakhir
                if polygon_mode and polygons:
                    deleted_polygon = polygons.pop()  # Hapus polygon terakhir
                    clear_data_for_deleted_object(deleted_polygon["id"], frame_resized, scale)  # Tambahkan frame
                    polygon_id_counter -= 1
                    logging.info(f"Polygon {deleted_polygon['id']} dihapus.")
                elif not polygon_mode and lines:
                    deleted_line = lines.pop()  # Hapus garis terakhir
                    clear_data_for_deleted_object(deleted_line["id"], frame_resized, scale)  # Tambahkan frame
                    line_id_counter -= 1
                    logging.info(f"Garis {deleted_line['id']} dihapus.")
            elif key == ord("y") or key == ord("Y"):
                # Menambahkan default pixel (hanya default_histogram) untuk semua garis dan polygon berdasarkan frame saat ini
                logging.info("Menambahkan default pixel (default_histogram) untuk semua garis dan polygon...")

                for line in lines:
                    if line["end_point"]:
                        # Buat mask untuk area garis
                        mask = np.zeros_like(gray_frame)
                        scaled_start = (int(line["start_point"][0] * scale_x), int(line["start_point"][1] * scale_y))
                        scaled_end = (int(line["end_point"][0] * scale_x), int(line["end_point"][1] * scale_y))
                        cv2.line(mask, scaled_start, scaled_end, 255, thickness=5)

                        # Hitung histogram dari area garis
                        current_histogram = calculate_histogram(gray_frame, mask)

                        # Jika histogram baseline belum disimpan, buat list baru untuk menampung default_histogram
                        if line["default_histogram"] is None:
                            line["default_histogram"] = []

                        # Jika default_histogram masih berupa numpy array (single histogram), ubah menjadi list
                        if isinstance(line["default_histogram"], np.ndarray):
                            line["default_histogram"] = [line["default_histogram"]]

                        # Tambahkan current_histogram ke list default_histogram
                        line["default_histogram"].append(current_histogram.copy())
                        logging.info(f"Default pixel (default_histogram) ditambahkan untuk {line['id']}")

                for polygon in polygons:
                    # Buat mask untuk area polygon
                    mask = np.zeros_like(gray_frame)
                    scaled_points = [(int(x * scale_x), int(y * scale_y)) for x, y in polygon["points"]]
                    cv2.fillPoly(mask, [np.array(scaled_points)], 255)

                    # Hitung histogram dari area polygon
                    current_histogram = calculate_histogram(gray_frame, mask)

                    # Jika histogram baseline belum disimpan, buat list baru untuk menampung default_histogram
                    if polygon["default_histogram"] is None:
                        polygon["default_histogram"] = []

                    # Jika default_histogram masih berupa numpy array (single histogram), ubah menjadi list
                    if isinstance(polygon["default_histogram"], np.ndarray):
                        polygon["default_histogram"] = [polygon["default_histogram"]]

                    # Tambahkan current_histogram ke list default_histogram
                    polygon["default_histogram"].append(current_histogram.copy())
                    logging.info(f"Default pixel (default_histogram) ditambahkan untuk {polygon['id']}")

                # Simpan data yang telah diperbarui (hanya default_histogram) ke file PKL yang sedang aktif
                save_data(current_pkl_file)
                logging.info("Default pixel (default_histogram) telah ditambahkan dan data disimpan.")

            # Penjadwalan PKL berdasarkan waktu
            current_datetime = datetime.now()
            if current_datetime >= next_switch_time:
                logging.info(f"Waktu switch telah tercapai: {next_switch_time.strftime('%H:%M')}. Memuat file PKL baru: {next_pkl_path}")
                # Memuat baseline histogram dari file PKL baru
                update_baseline_from_pkl(next_pkl_path)
                # Menetapkan file PKL baru sebagai yang aktif
                current_pkl_file = next_pkl_path
                # Menentukan switch time berikutnya
                for idx, (hour, minute, pkl_path) in enumerate(schedule):
                    try:
                        switch_time = current_datetime.replace(hour=hour, minute=minute, second=0, microsecond=0)
                    except ValueError:
                        switch_time = (current_datetime + timedelta(days=1)).replace(hour=hour, minute=minute, second=0, microsecond=0)
                    if current_datetime < switch_time:
                        next_switch_time = switch_time
                        next_pkl_path = pkl_path
                        break
                else:
                    # Jika tidak ada switch time tersisa hari ini, set switch time pertama besok
                    first_switch = schedule[0]
                    next_switch_time = (current_datetime + timedelta(days=1)).replace(
                        hour=first_switch[0],
                        minute=first_switch[1],
                        second=0,
                        microsecond=0,
                    )
                    next_pkl_path = first_switch[2]
                logging.info(f"Switch berikutnya dijadwalkan pada {next_switch_time.strftime('%Y-%m-%d %H:%M')} dengan file PKL: {next_pkl_path}")

        except Exception as e:
            logging.error(f"Error dalam memproses frame: {e}")

    # Setelah loop utama berakhir, lakukan cleanup
    stop_event.set()
    frame_thread.join()
    save_data(current_pkl_file)
    logging.info("Sumber daya dilepaskan dan koneksi database ditutup.")
    cursor.close()
    db_connection.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main_monitoring()
