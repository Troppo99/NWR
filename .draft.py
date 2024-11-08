import cv2
import numpy as np
import time
import os
from datetime import datetime, timedelta
import pickle  # Untuk menyimpan dan memuat data koordinat
import mysql.connector  # Untuk koneksi database
import threading  # Untuk menjalankan fungsi secara paralel
import logging  # Untuk logging
from queue import Queue, Empty  # Untuk antrian tugas

# Hapus semua handler yang ada
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

# Konfigurasi logging (Reset log setiap kali program dijalankan)
logging.basicConfig(filename="D:/AI_SOURCE/JALURCUTTING2/monitoring.log", filemode="w", level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")  # Overwrite mode untuk reset log  # Ubah ke DEBUG untuk logging lebih rinci

# Inisialisasi RTSP Stream untuk JALURCUTTING2
rtsp_url = "rtsp://admin:oracle2015@10.5.0.138:554/Streaming/Channels/1"  # Sesuaikan URL RTSP Anda
cap = cv2.VideoCapture(rtsp_url)

if not cap.isOpened():
    logging.error("Tidak dapat membuka stream RTSP.")
    exit()

# Variabel untuk menggambar garis dan polygon
drawing = False
polygon_mode = False
start_point = None
end_point = None
lines = []  # Menyimpan garis yang sudah terbentuk
polygons = []  # Menyimpan polygon yang sudah terbentuk
polygon_points = []  # Menyimpan titik-titik polygon sementara
current_mouse_position = (0, 0)  # Menyimpan posisi mouse saat ini
previous_frame = None  # Untuk menyimpan frame sebelumnya

# Counter untuk ID garis dan polygon
line_id_counter = 1
polygon_id_counter = 1

# Timer dictionary untuk countdown dan pelanggaran per ID Line dan Polygon
countdown_timers = {}
violation_timers = {}

# Lock untuk sinkronisasi akses ke countdown_timers
countdown_timers_lock = threading.Lock()

# Warna lingkaran
circle_color = (0, 255, 255)  # Kuning (BGR)

# Direktori screenshot unik untuk JALURCUTTING2
screenshot_save_path_line = "D:/AI_SOURCE/JALURCUTTING2/RED_LINE"
screenshot_save_path_area = "D:/AI_SOURCE/JALURCUTTING2/GREEN_AREA"

# Pastikan direktori screenshot ada
os.makedirs(screenshot_save_path_line, exist_ok=True)
os.makedirs(screenshot_save_path_area, exist_ok=True)

# Konfigurasi database
db_config = {"host": "10.5.0.2", "user": "robot", "password": "robot123", "database": "report_ai_cctv", "port": 3307}

# Koneksi ke database
try:
    db_connection = mysql.connector.connect(**db_config)
    db_connection.autocommit = True  # Aktifkan autocommit
    cursor = db_connection.cursor()
    logging.info("Koneksi ke database berhasil.")
except mysql.connector.Error as err:
    logging.error(f"Error koneksi database: {err}")
    exit()


# Definisikan fungsi insert_into_database di sini
def insert_into_database(timestamp, cam_name, no_line, time_detect, status, path_capture):
    """
    Fungsi untuk memasukkan data ke dalam database.

    Parameters:
        timestamp (str): Waktu kejadian dalam format 'YYYY-MM-DD HH:MM:SS'.
        cam_name (str): Nama kamera.
        no_line (str): ID garis atau area.
        time_detect (int): Status deteksi (misalnya, 1 untuk deteksi aktif).
        status (str): Status pelanggaran ('warning', 'violation', 'done').
        path_capture (str): Path ke file gambar yang diambil.
    """
    try:
        # Definisikan query SQL untuk memasukkan data ke tabel 'line_detectx'
        sql = """
            INSERT INTO line_detectx (timestamp, cam_name, no_line, time_detect, status, path_capture)
            VALUES (%s, %s, %s, %s, %s, %s)
        """
        values = (timestamp, cam_name, no_line, time_detect, status, path_capture)

        # Eksekusi query
        cursor.execute(sql, values)

        # Commit perubahan
        db_connection.commit()

        logging.info(f"Data berhasil dimasukkan ke database: {values}")
    except mysql.connector.Error as err:
        logging.error(f"Error memasukkan data ke database: {err}")
    except Exception as e:
        logging.error(f"Exception saat memasukkan data ke database: {e}")


# Penjadwalan file histogram berdasarkan waktu
schedule = [
    (7, 0, "D:/AI_SOURCE/JALURCUTTING2/JALURCUTTING2.npy"),
    # (8, 0, "D:/AI_SOURCE/JALURCUTTING2/JALURCUTTING2a.npy"),
    # (9, 0, "D:/AI_SOURCE/JALURCUTTING2/JALURCUTTING2b.npy"),
    # (10, 0, "D:/AI_SOURCE/JALURCUTTING2/JALURCUTTING2c.npy"),
    # (11, 0, "D:/AI_SOURCE/JALURCUTTING2/JALURCUTTING2d.npy"),
    # (12, 0, "D:/AI_SOURCE/JALURCUTTING2/JALURCUTTING2e.npy"),
    # (13, 0, "D:/AI_SOURCE/JALURCUTTING2/JALURCUTTING2f.npy"),
    # (14, 0, "D:/AI_SOURCE/JALURCUTTING2/JALURCUTTING2g.npy"),
    # (15, 0, "D:/AI_SOURCE/JALURCUTTING2/JALURCUTTING2h.npy"),
    # (16, 0, "D:/AI_SOURCE/JALURCUTTING2/JALURCUTTING2i.npy"),
    # (17, 0, "D:/AI_SOURCE/JALURCUTTING2/JALURCUTTING2j.npy"),
    (18, 0, "D:/AI_SOURCE/JALURCUTTING2/JALURCUTTING2NET.npy"),
    # Jadwal lainnya dapat ditambahkan di sini jika diperlukan
]

# Sort schedule berdasarkan waktu
schedule = sorted(schedule, key=lambda x: (x[0], x[1]))

# ========================================
# Konfigurasi Timer dan Sensitivitas
# ========================================

# Timer konfigurasi untuk LINE
LINE_COUNTDOWN_DURATION = 20  # Durasi countdown untuk LINE (detik)
LINE_SCREENSHOT_TIME = 15  # Waktu screenshot untuk LINE (detik sebelum countdown habis)
LINE_SENSITIVITY = 0.6  # Sensitivitas histogram untuk LINE

# Timer konfigurasi untuk AREA
AREA_COUNTDOWN_DURATION = 20  # Durasi countdown untuk AREA (detik)
AREA_SCREENSHOT_TIME = 15  # Waktu screenshot untuk AREA (detik sebelum countdown habis)
AREA_SENSITIVITY = 0.6  # Sensitivitas histogram untuk AREA

# ========================================


# TaskProcessor untuk menangani tugas asinkron
class TaskProcessor(threading.Thread):
    def __init__(self):
        super().__init__()
        self.task_queue = Queue()
        self.stop_event = threading.Event()

    def run(self):
        while not self.stop_event.is_set():
            try:
                task = self.task_queue.get(timeout=1)  # Tunggu tugas
                if task:
                    task_type = task.get("type")
                    if task_type == "save_image":
                        self.process_save_image_task(task)
                    elif task_type == "insert_db":
                        self.process_insert_db_task(task)
                self.task_queue.task_done()
            except Empty:
                continue  # Timeout terjadi, ulangi loop
            except Exception as e:
                logging.error(f"Exception in TaskProcessor: {e}")

    def stop(self):
        self.stop_event.set()

    def process_save_image_task(self, task):
        frame = task["frame"]
        object_id = task["object_id"]
        save_path = task["save_path"]
        object_type = task["object_type"]
        callback = task["callback"]
        is_fullscreen = task.get("is_fullscreen", False)

        if is_fullscreen:
            path = save_fullscreen_image(frame, object_id, save_path, object_type)
        else:
            mid_point = task["mid_point"]
            path = capture_and_save_screenshot(frame, object_id, mid_point, save_path, object_type)

        if callback:
            callback(path)

    def process_insert_db_task(self, task):
        logging.info(f"Memproses insert_db untuk {task['no_line']}")
        timestamp = task["timestamp"]
        cam_name = task["cam_name"]
        no_line = task["no_line"]
        time_detect = task["time_detect"]
        status = task["status"]
        path_capture = task["path_capture"]

        insert_into_database(timestamp, cam_name, no_line, time_detect, status, path_capture)
        logging.info(f"Tugas insert_db selesai untuk {no_line}")


# Inisialisasi TaskProcessor
task_processor = TaskProcessor()
task_processor.start()


# Fungsi untuk menyimpan koordinat garis dan polygon tanpa histogram
def save_coordinates(coordinate_file):
    # Salin lines dan polygons tanpa default_histogram
    lines_no_hist = []
    for line in lines:
        line_copy = line.copy()
        if "default_histogram" in line_copy:
            del line_copy["default_histogram"]
        lines_no_hist.append(line_copy)
    polygons_no_hist = []
    for polygon in polygons:
        polygon_copy = polygon.copy()
        if "default_histogram" in polygon_copy:
            del polygon_copy["default_histogram"]
        polygons_no_hist.append(polygon_copy)
    data = {"lines": lines_no_hist, "polygons": polygons_no_hist, "line_id_counter": line_id_counter, "polygon_id_counter": polygon_id_counter}
    try:
        with open(coordinate_file, "wb") as f:
            pickle.dump(data, f)
        logging.info(f"Coordinate data saved to {coordinate_file}")
    except Exception as e:
        logging.error(f"Error saving coordinate data to {coordinate_file}: {e}")


# Fungsi untuk memuat koordinat garis dan polygon
def load_coordinates(coordinate_file):
    global lines, polygons, line_id_counter, polygon_id_counter
    try:
        with open(coordinate_file, "rb") as f:
            data = pickle.load(f)
            lines = data.get("lines", [])
            polygons = data.get("polygons", [])
            line_id_counter = data.get("line_id_counter", 1)
            polygon_id_counter = data.get("polygon_id_counter", 1)
        logging.info(f"Coordinate data loaded from {coordinate_file}")
    except FileNotFoundError:
        logging.warning(f"No coordinate data found in {coordinate_file}")
    except Exception as e:
        logging.error(f"Error loading coordinate data from {coordinate_file}: {e}")


# Fungsi untuk menyimpan histogram menggunakan numpy files
def save_histograms(histogram_file):
    line_histograms = {}
    for line in lines:
        line_id = line["id"]
        default_histogram = line.get("default_histogram")
        if default_histogram is not None:
            line_histograms[line_id] = default_histogram
    polygon_histograms = {}
    for polygon in polygons:
        polygon_id = polygon["id"]
        default_histogram = polygon.get("default_histogram")
        if default_histogram is not None:
            polygon_histograms[polygon_id] = default_histogram
    histogram_data = {"lines": line_histograms, "polygons": polygon_histograms}
    try:
        np.save(histogram_file, histogram_data, allow_pickle=True)
        logging.info(f"Histogram data saved to {histogram_file}")
    except Exception as e:
        logging.error(f"Error saving histogram data to {histogram_file}: {e}")


# Fungsi untuk memuat histogram dari numpy files
def load_histograms(histogram_file):
    try:
        histogram_data = np.load(histogram_file, allow_pickle=True).item()
        line_histograms = histogram_data.get("lines", {})
        polygon_histograms = histogram_data.get("polygons", {})
        # Update default_histogram untuk lines
        for line in lines:
            line_id = line["id"]
            if line_id in line_histograms:
                line["default_histogram"] = line_histograms[line_id]
                logging.info(f"Default histogram updated for line {line_id}")
        # Update default_histogram untuk polygons
        for polygon in polygons:
            polygon_id = polygon["id"]
            if polygon_id in polygon_histograms:
                polygon["default_histogram"] = polygon_histograms[polygon_id]
                logging.info(f"Default histogram updated for polygon {polygon_id}")
        logging.info(f"Histogram data loaded from {histogram_file}")
    except FileNotFoundError:
        logging.warning(f"No histogram data found in {histogram_file}")
    except Exception as e:
        logging.error(f"Error loading histogram data from {histogram_file}: {e}")


# Fungsi untuk menginisialisasi jadwal histogram
def initialize_histogram_schedule(schedule):
    current_time = datetime.now()
    last_switch_time = None
    last_histogram_path = None
    next_switch_time = None
    next_histogram_path = None

    for hour, minute, histogram_path in schedule:
        try:
            switch_time = current_time.replace(hour=hour, minute=minute, second=0, microsecond=0)
        except ValueError:
            # Handle edge cases where hour/minute might be out of range
            switch_time = current_time + timedelta(days=1)
            switch_time = switch_time.replace(hour=hour % 24, minute=minute % 60, second=0, microsecond=0)

        if current_time >= switch_time:
            last_switch_time = switch_time
            last_histogram_path = histogram_path
        else:
            if next_switch_time is None:
                next_switch_time = switch_time
                next_histogram_path = histogram_path

    if last_histogram_path is None:
        # Semua switch time belum terjadi hari ini, gunakan file terakhir dari hari kemarin
        last_histogram_path = "D:/AI_SOURCE/JALURCUTTING2/JALURCUTTING2NET.npy"
    if next_switch_time is None:
        # Tidak ada switch time tersisa hari ini, set switch time pertama besok
        first_switch = schedule[0]
        next_switch_time = (current_time + timedelta(days=1)).replace(hour=first_switch[0], minute=first_switch[1], second=0, microsecond=0)
        next_histogram_path = first_switch[2]

    return last_histogram_path, next_switch_time, next_histogram_path


# Memuat histogram ke dalam lines dan polygons
def update_histograms_from_file(histogram_file):
    load_histograms(histogram_file)


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
            lines.append({"id": f"LINE {line_id_counter}", "start_point": start_point, "end_point": end_point, "default_histogram": None, "line_detected": False, "color": (0, 255, 255)})
            logging.info(f"Garis {line_id_counter} ditambahkan dari {start_point} ke {end_point}.")
            line_id_counter += 1


def calculate_histogram(gray_frame, mask):
    """Hitung histogram dari area yang di-mask."""
    hist = cv2.calcHist([gray_frame], [0], mask, [256], [0, 256])
    cv2.normalize(hist, hist)
    return hist


def check_histogram_change(current_histogram, default_histogram, sensitivity):
    """Bandingkan histogram saat ini dengan histogram default menggunakan korelasi."""

    # Pastikan default_histogram adalah numpy array
    if isinstance(default_histogram, list):
        if len(default_histogram) > 0:
            # Ambil rata-rata histogram jika default_histogram adalah list dari beberapa histogram
            default_histogram = np.mean(default_histogram, axis=0)
        else:
            logging.error("Default histogram kosong.")
            return False

    # Lakukan perbandingan dengan korelasi
    hist_diff = cv2.compareHist(default_histogram, current_histogram, cv2.HISTCMP_CORREL)

    # Sensitivitas menentukan batasan perbedaan yang dapat diterima
    result = hist_diff < (1.0 - sensitivity)
    logging.debug(f"Histogram correlation: {hist_diff}, Sensitivity Threshold: {1.0 - sensitivity}, Change Detected: {result}")
    return result


def start_countdown(object_id, object_type):
    """Mulai countdown untuk garis atau polygon merah."""
    with countdown_timers_lock:
        if object_id not in countdown_timers:
            # Tentukan durasi countdown dan waktu screenshot berdasarkan tipe objek
            if object_type == "LINE":
                countdown_duration = LINE_COUNTDOWN_DURATION
                screenshot_time = LINE_SCREENSHOT_TIME
            else:
                countdown_duration = AREA_COUNTDOWN_DURATION
                screenshot_time = AREA_SCREENSHOT_TIME

            countdown_timers[object_id] = {"start_time": time.time(), "captured": False, "path_capture": "0", "last_insert_time": 0, "violation_start_time": None, "violation_last_insert_time": 0, "first_violation": True, "is_in_violation": False, "countdown_duration": countdown_duration, "screenshot_time": screenshot_time}
            logging.info(f"Countdown dimulai untuk {object_type} {object_id} dengan durasi {countdown_duration} detik.")


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
    success = cv2.imwrite(save_path, cropped_img)
    if success:
        logging.info(f"Gambar disimpan: {save_path}")
    else:
        logging.error(f"Failed to save image to {save_path}")
    return save_path  # Kembalikan path untuk disimpan ke database


def save_fullscreen_image(frame, object_id, save_path, object_type):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"FULLSCREEN_{object_id}_{timestamp}.jpg"
    save_file_path = os.path.join(save_path, filename)

    # Pastikan direktori ada
    os.makedirs(save_path, exist_ok=True)

    success = cv2.imwrite(save_file_path, frame)
    if success:
        logging.info(f"Fullscreen gambar disimpan: {save_file_path}")
    else:
        logging.error(f"Failed to save image to {save_file_path}")
    return save_file_path


# Fungsi asinkron untuk penyimpanan screenshot
def capture_and_save_screenshot_async(frame, object_id, mid_point, save_path, object_type, callback):
    task = {"type": "save_image", "frame": frame.copy(), "object_id": object_id, "mid_point": mid_point, "save_path": save_path, "object_type": object_type, "callback": callback}  # Pastikan frame dicopy
    task_processor.task_queue.put(task)


# Fungsi asinkron untuk memasukkan data ke database
def insert_data_async(timestamp, cam_name, no_line, time_detect, status, path_capture):
    task = {"type": "insert_db", "timestamp": timestamp, "cam_name": cam_name, "no_line": no_line, "time_detect": time_detect, "status": status, "path_capture": path_capture}
    task_processor.task_queue.put(task)


def after_violation_end_callback(save_file_path, object_id, object_type):
    logging.info(f"Callback dipanggil untuk mengakhiri pelanggaran {object_id}")
    timestamp_db = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cam_name = "JALURCUTTING2"
    no_line = object_id
    time_detect = 1
    status = "done"

    insert_data_async(timestamp_db, cam_name, no_line, time_detect, status, save_file_path)
    logging.info(f"Pelanggaran untuk {object_id} telah selesai dan data 'done' dimasukkan ke database.")

    with countdown_timers_lock:
        if object_id in countdown_timers:
            countdown_timers[object_id]["is_in_violation"] = False
            logging.info(f"Menandai {object_id} sebagai tidak dalam pelanggaran lagi.")


def handle_violation_end_callback(save_file_path, object_id, object_type):
    logging.info(f"Handling violation end for {object_id}")
    after_violation_end_callback(save_file_path, object_id, object_type)


def handle_violation_end_async(object_id, object_type, frame, save_path, callback):
    logging.info(f"Menambahkan tugas async untuk mengakhiri pelanggaran {object_id}")
    frame_copy = frame.copy()
    task = {"type": "save_image", "frame": frame_copy, "object_id": object_id, "mid_point": None, "save_path": save_path, "object_type": object_type, "callback": callback, "is_fullscreen": True}  # Tidak perlu mid_point untuk fullscreen  # Tambahkan flag
    task_processor.task_queue.put(task)


def display_countdown_and_violation(frame, save_path, object_type="LINE", y_start=50):
    """Menangani countdown, violation, dan memasukkan data ke database secara realtime."""
    current_time = time.time()
    with countdown_timers_lock:
        timers_items = list(countdown_timers.items())
    for object_id, timer_data in timers_items:
        # Hanya proses ID yang sesuai dengan object_type
        if (object_type == "LINE" and object_id.startswith("LINE")) or (object_type == "AREA" and object_id.startswith("AREA")):
            elapsed_time = current_time - timer_data["start_time"]
            remaining_time = int(timer_data["countdown_duration"] - elapsed_time)
            if remaining_time > 0:
                # Status: warning
                status = "warning"
                # Insert data setiap detik dengan path_capture="0"
                if int(elapsed_time) > timer_data["last_insert_time"]:
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    insert_data_async(timestamp, "JALURCUTTING2", object_id, 1, status, "0")
                    countdown_timers[object_id]["last_insert_time"] = int(elapsed_time)

                # Capture pada waktu screenshot yang ditentukan
                screenshot_time = timer_data["screenshot_time"]
                if remaining_time == timer_data["countdown_duration"] - screenshot_time and not timer_data["captured"]:
                    # Tangkap nilai object_id dan status saat ini
                    current_object_id = object_id
                    current_status = status

                    def screenshot_callback(path, oid=current_object_id, otype=object_type):
                        with countdown_timers_lock:
                            if oid in countdown_timers:
                                countdown_timers[oid]["path_capture"] = path
                                countdown_timers[oid]["captured"] = True
                        # Insert data dengan path_capture
                        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        insert_data_async(timestamp, "JALURCUTTING2", oid, 1, current_status, path)

                    if object_type == "LINE":
                        for line in lines:
                            if line["id"] == object_id:
                                mid_point = ((line["start_point"][0] + line["end_point"][0]) // 2, (line["start_point"][1] + line["end_point"][1]) // 2)
                                capture_and_save_screenshot_async(frame.copy(), object_id, mid_point, save_path, object_type, screenshot_callback)
                    elif object_type == "AREA":
                        for polygon in polygons:
                            if polygon["id"] == object_id:
                                mid_point = np.mean(polygon["points"], axis=0).astype(int)
                                capture_and_save_screenshot_async(frame.copy(), object_id, mid_point, save_path, object_type, screenshot_callback)

                # Gambar countdown
                y_position = y_start
                x_position = frame.shape[1] - 150
                cv2.putText(frame, f"{object_id}: {remaining_time}s", (x_position, y_position), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1, cv2.LINE_AA)
                y_start += 15
            else:
                # Status: violation
                status = "violation"
                with countdown_timers_lock:
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

                        insert_data_async(timestamp, "JALURCUTTING2", object_id, 1, status, path_capture)
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
                            cv2.line(frame, line["start_point"], line["end_point"], line["color"], 1)
                            mid_point = ((line["start_point"][0] + line["end_point"][0]) // 2, (line["start_point"][1] + line["end_point"][1]) // 2)
                            cv2.putText(frame, f"{line['id']}", mid_point, cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
                elif object_type == "AREA":
                    for polygon in polygons:
                        if polygon["id"] == object_id:
                            cv2.polylines(frame, [np.array(polygon["points"])], isClosed=True, color=polygon["color"], thickness=1)
                            centroid = np.mean(polygon["points"], axis=0).astype(int)
                            cv2.putText(frame, f"{polygon['id']}", tuple(centroid), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)

                # Gambar pesan violation
                message = f"Terjadi Pelanggaran {violation_seconds}s Di {object_id}"
                cv2.putText(frame, message, (20, y_start), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                y_start += 15

                # Jika kondisi sudah normal, akhiri violation
                # Periksa apakah kondisi sudah normal (tidak ada perubahan)
                # Jika iya, panggil handle_violation_end_async
                # Namun, kondisi ini sudah diatur di fungsi deteksi utama
                # Jadi, pastikan 'is_in_violation' diatur kembali ke False di callback
                # Tambahkan logika untuk memastikan bahwa violation bisa diakhiri


def handle_violation_end_callback(save_file_path, object_id, object_type):
    logging.info(f"Handling violation end for {object_id}")
    after_violation_end_callback(save_file_path, object_id, object_type)


def handle_violation_end_async(object_id, object_type, frame, save_path, callback):
    logging.info(f"Menambahkan tugas async untuk mengakhiri pelanggaran {object_id}")
    frame_copy = frame.copy()
    task = {"type": "save_image", "frame": frame_copy, "object_id": object_id, "mid_point": None, "save_path": save_path, "object_type": object_type, "callback": callback, "is_fullscreen": True}  # Tidak perlu mid_point untuk fullscreen  # Tambahkan flag
    task_processor.task_queue.put(task)


def check_polygon_violation(frame, gray_frame):
    """Fungsi untuk memeriksa pelanggaran pada polygon (mode hijau)"""
    for polygon in polygons:
        mask = np.zeros_like(gray_frame)

        # Buat mask untuk polygon
        cv2.fillPoly(mask, [np.array(polygon["points"])], 255)

        # Hitung histogram dari area polygon
        current_histogram = calculate_histogram(gray_frame, mask)

        # Jika histogram baseline belum disimpan, simpan sekarang
        if polygon.get("default_histogram") is None:
            polygon["default_histogram"] = current_histogram.copy()
            logging.info(f"Baseline histogram disimpan untuk {polygon['id']}.")

        # Bandingkan histogram saat ini dengan histogram baseline
        if check_histogram_change(current_histogram, polygon["default_histogram"], AREA_SENSITIVITY):
            polygon["color"] = (0, 0, 255)  # Merah jika ada perubahan
            start_countdown(polygon["id"], "AREA")  # Mulai countdown jika ada perubahan
        else:
            polygon["color"] = (0, 255, 0)  # Hijau jika tidak ada perubahan
            with countdown_timers_lock:
                if polygon["id"] in countdown_timers:
                    if countdown_timers[polygon["id"]].get("is_in_violation"):
                        # Violation telah berakhir
                        handle_violation_end_async(polygon["id"], "AREA", frame, screenshot_save_path_area, lambda path, oid=polygon["id"], otype="AREA": handle_violation_end_callback(path, oid, otype))
                    else:
                        # Hapus countdown jika kembali normal sebelum violation
                        del countdown_timers[polygon["id"]]
                        logging.info(f"Countdown dihapus karena {polygon['id']} kembali normal.")

        # Gambar outline polygon
        cv2.polylines(frame, [np.array(polygon["points"])], isClosed=True, color=polygon["color"], thickness=1)

        # Tampilkan ID di tengah polygon
        centroid = np.mean(polygon["points"], axis=0).astype(int)
        cv2.putText(frame, f"{polygon['id']}", tuple(centroid), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1, cv2.LINE_AA)


def clear_data_for_deleted_object(object_id, frame):
    """Fungsi untuk menghapus countdown, pelanggaran, dan status screenshot saat garis atau area dihapus."""
    with countdown_timers_lock:
        if object_id in countdown_timers:
            del countdown_timers[object_id]
            logging.info(f"Countdown dihapus untuk {object_id}.")
        if object_id in violation_timers:
            del violation_timers[object_id]
    logging.info(f"Pelanggaran dihapus untuk {object_id}.")


# Panggil fungsi load_coordinates() pada startup
coordinate_file = "D:/AI_SOURCE/JALURCUTTING2/coordinates.pkl"
load_coordinates(coordinate_file)

# Inisialisasi jadwal histogram
current_histogram_file, next_switch_time, next_histogram_path = initialize_histogram_schedule(schedule)
update_histograms_from_file(current_histogram_file)

cv2.namedWindow("RTSP Stream - Monitoring")
cv2.setMouseCallback("RTSP Stream - Monitoring", draw)

while True:
    ret, frame = cap.read()
    if not ret:
        logging.error("Tidak dapat membaca frame dari RTSP atau stream berhenti.")
        break

    # Ubah ukuran frame menjadi 854x480
    frame = cv2.resize(frame, (854, 480))

    # Konversi frame ke grayscale untuk mendeteksi perubahan piksel
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Gambar garis dan deteksi perubahan menggunakan histogram
    for line in lines:
        if line["end_point"]:
            # Buat mask untuk area garis
            mask = np.zeros_like(gray_frame)
            cv2.line(mask, line["start_point"], line["end_point"], 255, thickness=5)

            # Hitung histogram dari area garis
            current_histogram = calculate_histogram(gray_frame, mask)

            # Jika histogram baseline belum disimpan, simpan sekarang
            if line["default_histogram"] is None:
                line["default_histogram"] = current_histogram.copy()
                logging.info(f"Baseline histogram disimpan untuk {line['id']}.")

            # Bandingkan histogram saat ini dengan histogram baseline
            if check_histogram_change(current_histogram, line["default_histogram"], LINE_SENSITIVITY):
                line["color"] = (0, 0, 255)  # Merah jika ada perubahan
                start_countdown(line["id"], "LINE")  # Mulai countdown saat garis berubah menjadi merah
            else:
                line["color"] = (0, 255, 255)  # Kuning jika tidak ada perubahan
                with countdown_timers_lock:
                    if line["id"] in countdown_timers:
                        if countdown_timers[line["id"]].get("is_in_violation"):
                            # Violation telah berakhir
                            handle_violation_end_async(line["id"], "LINE", frame, screenshot_save_path_line, lambda path, oid=line["id"], otype="LINE": handle_violation_end_callback(path, oid, otype))
                        else:
                            # Hapus countdown jika kembali normal sebelum violation
                            del countdown_timers[line["id"]]
                            logging.info(f"Countdown dihapus karena {line['id']} kembali normal.")

            # Gambar garis
            cv2.line(frame, line["start_point"], line["end_point"], line["color"], 1)

            # Tampilkan ID di tengah garis
            mid_point = ((line["start_point"][0] + line["end_point"][0]) // 2, (line["start_point"][1] + line["end_point"][1]) // 2)
            cv2.putText(frame, f"{line['id']}", mid_point, cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1, cv2.LINE_AA)

    # Periksa pelanggaran pada polygon
    check_polygon_violation(frame, gray_frame)

    # Tampilkan countdown dan violation untuk mode kuning di kanan atas
    display_countdown_and_violation(frame, screenshot_save_path_line, "LINE", y_start=30)

    # Tampilkan countdown dan violation untuk mode hijau (polygon)
    display_countdown_and_violation(frame, screenshot_save_path_area, "AREA", y_start=100)

    # Gambar polygon yang sedang digambar (warna hijau)
    if polygon_mode and polygon_points:
        for i in range(len(polygon_points) - 1):
            cv2.line(frame, polygon_points[i], polygon_points[i + 1], (0, 255, 0), 1)
        cv2.line(frame, polygon_points[-1], current_mouse_position, (0, 255, 0), 1)

    # Gambar garis kuning sementara saat mouse sedang digerakkan
    if not polygon_mode and drawing:
        cv2.line(frame, start_point, current_mouse_position, (0, 255, 255), 1)  # Tetap menampilkan garis kuning saat ditarik

    # Tambahkan status mode ke layar
    if polygon_mode:
        cv2.putText(frame, "MODE: AREA", (20, 460), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1, cv2.LINE_AA)
    else:
        cv2.putText(frame, "MODE: LINE", (20, 460), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1, cv2.LINE_AA)

    # Tampilkan hasilnya
    cv2.imshow("RTSP Stream - Monitoring", frame)

    # Cek input keyboard untuk keluar atau mengubah mode
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        # Simpan data saat program akan ditutup
        save_coordinates(coordinate_file)
        save_histograms(current_histogram_file)
        logging.info("Program dihentikan oleh pengguna. Data disimpan.")
        break
    elif key == ord("u"):
        # Ubah mode antara garis kuning dan polygon (hijau)
        if circle_color == (0, 255, 255):  # Kuning
            circle_color = (0, 255, 0)  # Hijau
            polygon_mode = True
            logging.info("Mode diubah ke AREA.")
        else:
            circle_color = (0, 255, 255)  # Kembali ke kuning
            polygon_mode = False
            polygon_points.clear()  # Reset polygon points setelah selesai menggambar
            logging.info("Mode diubah ke LINE.")
    elif key == ord("r") and polygon_mode:
        # Selesaikan polygon saat tombol R ditekan
        if len(polygon_points) > 2:
            polygon_id = f"AREA {polygon_id_counter}"
            polygons.append({"points": polygon_points.copy(), "id": polygon_id, "default_histogram": None, "color": (0, 255, 0)})
            logging.info(f"Polygon {polygon_id} ditambahkan dengan titik: {polygon_points}.")
            polygon_id_counter += 1
        polygon_points.clear()  # Reset polygon points setelah selesai menggambar
    elif key == ord("c"):
        # Hapus objek terakhir
        if polygon_mode and polygons:
            deleted_polygon = polygons.pop()  # Hapus polygon terakhir
            clear_data_for_deleted_object(deleted_polygon["id"], frame)  # Tambahkan frame
            polygon_id_counter -= 1
            logging.info(f"Polygon {deleted_polygon['id']} dihapus.")
        elif not polygon_mode and lines:
            deleted_line = lines.pop()  # Hapus garis terakhir
            clear_data_for_deleted_object(deleted_line["id"], frame)  # Tambahkan frame
            line_id_counter -= 1
            logging.info(f"Garis {deleted_line['id']} dihapus.")
    elif key == ord("y"):
        # Tambahkan default_histogram untuk LINE saja
        print("Menambahkan default pixel (default_histogram) untuk garis...")
        for line in lines:
            if line["end_point"]:
                # Buat mask untuk area garis
                mask = np.zeros_like(gray_frame)
                cv2.line(mask, line["start_point"], line["end_point"], 255, thickness=5)

                # Hitung histogram dari area garis
                current_histogram = calculate_histogram(gray_frame, mask)

                # Jika default_histogram belum ada, inisialisasi sebagai list
                if line["default_histogram"] is None:
                    line["default_histogram"] = []

                # Jika default_histogram masih berupa numpy array, ubah menjadi list
                if isinstance(line["default_histogram"], np.ndarray):
                    line["default_histogram"] = [line["default_histogram"]]

                # Tambahkan histogram saat ini ke default_histogram
                line["default_histogram"].append(current_histogram.copy())
                print(f"Default pixel (default_histogram) ditambahkan untuk {line['id']}")

        # Simpan histogram yang telah diperbarui
        save_histograms(current_histogram_file)
        print("Default pixel (default_histogram) untuk garis telah ditambahkan dan data disimpan.")
    elif key == ord("t"):
        # Tambahkan default_histogram untuk AREA saja
        print("Menambahkan default pixel (default_histogram) untuk area...")
        for polygon in polygons:
            # Buat mask untuk area polygon
            mask = np.zeros_like(gray_frame)
            cv2.fillPoly(mask, [np.array(polygon["points"])], 255)

            # Hitung histogram dari area polygon
            current_histogram = calculate_histogram(gray_frame, mask)

            # Jika default_histogram belum ada, inisialisasi sebagai list
            if polygon["default_histogram"] is None:
                polygon["default_histogram"] = []

            # Jika default_histogram masih berupa numpy array, ubah menjadi list
            if isinstance(polygon["default_histogram"], np.ndarray):
                polygon["default_histogram"] = [polygon["default_histogram"]]

            # Tambahkan histogram saat ini ke default_histogram
            polygon["default_histogram"].append(current_histogram.copy())
            print(f"Default pixel (default_histogram) ditambahkan untuk {polygon['id']}")

        # Simpan histogram yang telah diperbarui
        save_histograms(current_histogram_file)
        print("Default pixel (default_histogram) untuk area telah ditambahkan dan data disimpan.")
    elif key == ord("h"):
        # Reset default_histogram untuk LINE saja
        print("Mereset default pixel (default_histogram) untuk garis...")
        for line in lines:
            line["default_histogram"] = None
            print(f"Default pixel (default_histogram) direset untuk {line['id']}")

        # Simpan histogram yang telah diperbarui
        save_histograms(current_histogram_file)
        print("Default pixel (default_histogram) untuk garis telah direset dan data disimpan.")
    elif key == ord("g"):
        # Reset default_histogram untuk AREA saja
        print("Mereset default pixel (default_histogram) untuk area...")
        for polygon in polygons:
            polygon["default_histogram"] = None
            print(f"Default pixel (default_histogram) direset untuk {polygon['id']}")

        # Simpan histogram yang telah diperbarui
        save_histograms(current_histogram_file)
        print("Default pixel (default_histogram) untuk area telah direset dan data disimpan.")

    # Penanganan Penjadwalan Histogram berdasarkan waktu
    current_datetime = datetime.now()
    if current_datetime >= next_switch_time:
        logging.info(f"Waktu switch telah tercapai: {next_switch_time.strftime('%H:%M')}. Memuat file histogram baru: {next_histogram_path}")
        # Memuat histogram dari file baru
        update_histograms_from_file(next_histogram_path)
        # Menetapkan file histogram baru sebagai yang aktif
        current_histogram_file = next_histogram_path
        # Menentukan switch time berikutnya
        for idx, (hour, minute, histogram_path) in enumerate(schedule):
            try:
                switch_time = current_datetime.replace(hour=hour, minute=minute, second=0, microsecond=0)
            except ValueError:
                # Handle edge cases where hour/minute might be out of range
                switch_time = current_datetime + timedelta(days=1)
                switch_time = switch_time.replace(hour=hour % 24, minute=minute % 60, second=0, microsecond=0)

            if current_datetime < switch_time:
                next_switch_time = switch_time
                next_histogram_path = histogram_path
                break
        else:
            # Jika tidak ada switch time tersisa hari ini, set switch time pertama besok
            first_switch = schedule[0]
            next_switch_time = (current_datetime + timedelta(days=1)).replace(hour=first_switch[0], minute=first_switch[1], second=0, microsecond=0)
            next_histogram_path = first_switch[2]
        logging.info(f"Switch berikutnya dijadwalkan pada {next_switch_time.strftime('%Y-%m-%d %H:%M')} dengan file histogram: {next_histogram_path}")

# Rilis sumber daya dan tutup koneksi database
cap.release()
cv2.destroyAllWindows()
db_connection.close()  # Tutup koneksi database
logging.info("Sumber daya dilepaskan dan koneksi database ditutup.")

# Hentikan TaskProcessor
task_processor.stop()
task_processor.join()
