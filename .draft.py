import cv2
import numpy as np
import time
import os
from datetime import datetime, timedelta
import pickle
import mysql.connector
import threading
import logging
from queue import Queue, Empty

filename = "archives/AI_SOURCE/JALURCUTTING2/monitoring.log"
log_directory = os.path.dirname(filename)
os.makedirs(log_directory, exist_ok=True)

logging.basicConfig(filename=filename, filemode="w", level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")

rtsp_url = "rtsp://admin:oracle2015@10.5.0.138:554/Streaming/Channels/1"
cap = cv2.VideoCapture(rtsp_url)

if not cap.isOpened():
    logging.error("Tidak dapat membuka stream RTSP.")
    exit()

drawing = False
polygon_mode = False
start_point = None
end_point = None
lines = []
polygons = []
polygon_points = []
current_mouse_position = (0, 0)
previous_frame = None

line_id_counter = 1
polygon_id_counter = 1

countdown_timers = {}
violation_timers = {}

countdown_timers_lock = threading.Lock()

circle_color = (0, 255, 255)


screenshot_save_path_line = "archives/ai_source/JALURCUTTING2/RED_LINE"
screenshot_save_path_area = "archives/ai_source/JALURCUTTING2/GREEN_AREA"

os.makedirs(screenshot_save_path_line, exist_ok=True)
os.makedirs(screenshot_save_path_area, exist_ok=True)

db_config = {"host": "10.5.0.2", "user": "robot", "password": "robot123", "database": "report_ai_cctv", "port": 3307}

try:
    db_connection = mysql.connector.connect(**db_config)
    db_connection.autocommit = True
    cursor = db_connection.cursor()
    logging.info("Koneksi ke database berhasil.")
except mysql.connector.Error as err:
    logging.error(f"Error koneksi database: {err}")
    exit()


def insert_into_database(timestamp, cam_name, no_line, time_detect, status, path_capture):
    try:
        sql = """
            INSERT INTO line_detectx (timestamp, cam_name, no_line, time_detect, status, path_capture)
            VALUES (%s, %s, %s, %s, %s, %s)
        """
        values = (timestamp, cam_name, no_line, time_detect, status, path_capture)
        cursor.execute(sql, values)
        db_connection.commit()
        logging.info(f"Data berhasil dimasukkan ke database: {values}")
    except mysql.connector.Error as err:
        logging.error(f"Error memasukkan data ke database: {err}")
    except Exception as e:
        logging.error(f"Exception saat memasukkan data ke database: {e}")


schedule = [
    (7, 0, "archives/AI_SOURCE/JALURCUTTING2/JALURCUTTING2.npy"),
    (8, 0, "archives/AI_SOURCE/JALURCUTTING2/JALURCUTTING2a.npy"),
    (9, 0, "archives/AI_SOURCE/JALURCUTTING2/JALURCUTTING2b.npy"),
    (10, 0, "archives/AI_SOURCE/JALURCUTTING2/JALURCUTTING2c.npy"),
    (11, 0, "archives/AI_SOURCE/JALURCUTTING2/JALURCUTTING2d.npy"),
    (12, 0, "archives/AI_SOURCE/JALURCUTTING2/JALURCUTTING2e.npy"),
    (13, 0, "archives/AI_SOURCE/JALURCUTTING2/JALURCUTTING2f.npy"),
    (14, 0, "archives/AI_SOURCE/JALURCUTTING2/JALURCUTTING2g.npy"),
    (15, 0, "archives/AI_SOURCE/JALURCUTTING2/JALURCUTTING2h.npy"),
    (16, 0, "archives/AI_SOURCE/JALURCUTTING2/JALURCUTTING2i.npy"),
    (17, 0, "archives/AI_SOURCE/JALURCUTTING2/JALURCUTTING2j.npy"),
    (18, 0, "archives/AI_SOURCE/JALURCUTTING2/JALURCUTTING2NET.npy"),
]

schedule = sorted(schedule, key=lambda x: (x[0], x[1]))

LINE_COUNTDOWN_DURATION = 180
LINE_SCREENSHOT_TIME = 177
LINE_SENSITIVITY = 0.6

AREA_COUNTDOWN_DURATION = 180
AREA_SCREENSHOT_TIME = 177
AREA_SENSITIVITY = 0.6


class TaskProcessor(threading.Thread):
    def __init__(self):
        super().__init__()
        self.task_queue = Queue()
        self.stop_event = threading.Event()

    def run(self):
        while not self.stop_event.is_set():
            try:
                task = self.task_queue.get(timeout=1)
                if task:
                    task_type = task.get("type")
                    if task_type == "save_image":
                        self.process_save_image_task(task)
                    elif task_type == "insert_db":
                        self.process_insert_db_task(task)
                self.task_queue.task_done()
            except Empty:
                continue
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


task_processor = TaskProcessor()
task_processor.start()


def save_coordinates(coordinate_file):
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


def load_histograms(histogram_file):
    try:
        histogram_data = np.load(histogram_file, allow_pickle=True).item()
        line_histograms = histogram_data.get("lines", {})
        polygon_histograms = histogram_data.get("polygons", {})
        for line in lines:
            line_id = line["id"]
            if line_id in line_histograms:
                line["default_histogram"] = line_histograms[line_id]
                logging.info(f"Default histogram updated for line {line_id}")
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
        last_histogram_path = "archives/AI_SOURCE/JALURCUTTING2/JALURCUTTING2NET.npy"
    if next_switch_time is None:
        first_switch = schedule[0]
        next_switch_time = (current_time + timedelta(days=1)).replace(hour=first_switch[0], minute=first_switch[1], second=0, microsecond=0)
        next_histogram_path = first_switch[2]

    return last_histogram_path, next_switch_time, next_histogram_path


def update_histograms_from_file(histogram_file):
    load_histograms(histogram_file)


def draw(event, x, y, flags, param):
    global drawing, start_point, end_point, lines, polygon_points, polygon_mode, current_mouse_position, line_id_counter, polygon_id_counter

    current_mouse_position = (x, y)

    if polygon_mode:
        if event == cv2.EVENT_LBUTTONDOWN:
            polygon_points.append((x, y))
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
    hist = cv2.calcHist([gray_frame], [0], mask, [256], [0, 256])
    cv2.normalize(hist, hist)
    return hist


def check_histogram_change(current_histogram, default_histogram, sensitivity):
    if isinstance(default_histogram, list):
        if len(default_histogram) > 0:
            default_histogram = np.mean(default_histogram, axis=0)
        else:
            logging.error("Default histogram kosong.")
            return False
    hist_diff = cv2.compareHist(default_histogram, current_histogram, cv2.HISTCMP_CORREL)
    result = hist_diff < (1.0 - sensitivity)
    logging.debug(f"Histogram correlation: {hist_diff}, Sensitivity Threshold: {1.0 - sensitivity}, Change Detected: {result}")
    return result


def start_countdown(object_id, object_type):
    with countdown_timers_lock:
        if object_id not in countdown_timers:
            if object_type == "LINE":
                countdown_duration = LINE_COUNTDOWN_DURATION
                screenshot_time = LINE_SCREENSHOT_TIME
            else:
                countdown_duration = AREA_COUNTDOWN_DURATION
                screenshot_time = AREA_SCREENSHOT_TIME
            countdown_timers[object_id] = {"start_time": time.time(), "captured": False, "path_capture": "0", "last_insert_time": 0, "violation_start_time": None, "violation_last_insert_time": 0, "first_violation": True, "is_in_violation": False, "countdown_duration": countdown_duration, "screenshot_time": screenshot_time}
            logging.info(f"Countdown dimulai untuk {object_type} {object_id} dengan durasi {countdown_duration} detik.")


def capture_and_save_screenshot(frame, object_id, mid_point, save_path, object_type):
    x, y = mid_point
    crop_size = 300
    height, width = frame.shape[:2]

    x1 = max(0, x - crop_size)
    y1 = max(0, y - crop_size)
    x2 = min(width, x + crop_size)
    y2 = min(height, y + crop_size)

    cropped_img = frame[y1:y2, x1:x2]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{object_type}_{object_id}_{timestamp}.jpg"

    save_path = os.path.join(save_path, filename)

    success = cv2.imwrite(save_path, cropped_img)
    if success:
        logging.info(f"Gambar disimpan: {save_path}")
    else:
        logging.error(f"Failed to save image to {save_path}")
    return save_path


def save_fullscreen_image(frame, object_id, save_path, object_type):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"FULLSCREEN_{object_id}_{timestamp}.jpg"
    save_file_path = os.path.join(save_path, filename)

    os.makedirs(save_path, exist_ok=True)

    success = cv2.imwrite(save_file_path, frame)
    if success:
        logging.info(f"Fullscreen gambar disimpan: {save_file_path}")
    else:
        logging.error(f"Failed to save image to {save_file_path}")
    return save_file_path


def capture_and_save_screenshot_async(frame, object_id, mid_point, save_path, object_type, callback):
    task = {"type": "save_image", "frame": frame.copy(), "object_id": object_id, "mid_point": mid_point, "save_path": save_path, "object_type": object_type, "callback": callback}
    task_processor.task_queue.put(task)


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
    task = {"type": "save_image", "frame": frame_copy, "object_id": object_id, "mid_point": None, "save_path": save_path, "object_type": object_type, "callback": callback, "is_fullscreen": True}
    task_processor.task_queue.put(task)


def display_countdown_and_violation(frame, save_path, object_type="LINE", y_start=50):
    current_time = time.time()
    with countdown_timers_lock:
        timers_items = list(countdown_timers.items())
    for object_id, timer_data in timers_items:
        if (object_type == "LINE" and object_id.startswith("LINE")) or (object_type == "AREA" and object_id.startswith("AREA")):
            elapsed_time = current_time - timer_data["start_time"]
            remaining_time = int(timer_data["countdown_duration"] - elapsed_time)
            if remaining_time > 0:
                status = "warning"
                if int(elapsed_time) > timer_data["last_insert_time"]:
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    insert_data_async(timestamp, "JALURCUTTING2", object_id, 1, status, "0")
                    countdown_timers[object_id]["last_insert_time"] = int(elapsed_time)

                screenshot_time = timer_data["screenshot_time"]
                if remaining_time == timer_data["countdown_duration"] - screenshot_time and not timer_data["captured"]:
                    current_object_id = object_id
                    current_status = status

                    def screenshot_callback(path, oid=current_object_id, otype=object_type):
                        with countdown_timers_lock:
                            if oid in countdown_timers:
                                countdown_timers[oid]["path_capture"] = path
                                countdown_timers[oid]["captured"] = True
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

                y_position = y_start
                x_position = frame.shape[1] - 150
                cv2.putText(frame, f"{object_id}: {remaining_time}s", (x_position, y_position), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1, cv2.LINE_AA)
                y_start += 15
            else:
                status = "violation"
                with countdown_timers_lock:
                    if timer_data.get("violation_start_time") is None:
                        countdown_timers[object_id]["violation_start_time"] = current_time
                        countdown_timers[object_id]["is_in_violation"] = True
                        logging.info(f"Violation dimulai untuk {object_type} {object_id}.")

                    violation_start_time = countdown_timers[object_id]["violation_start_time"]
                    violation_elapsed = current_time - violation_start_time
                    violation_seconds = int(violation_elapsed)

                    if violation_seconds > countdown_timers[object_id].get("violation_last_insert_time", 0):
                        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                        if countdown_timers[object_id].get("first_violation", True):
                            path_capture = timer_data.get("path_capture", "0")
                            countdown_timers[object_id]["first_violation"] = False
                            logging.info(f"Pelanggaran pertama untuk {object_id}: path_capture disamakan dengan warning sebelumnya.")
                        else:
                            path_capture = "0"
                            logging.info(f"Pelanggaran berikutnya untuk {object_id}: path_capture diisi '0'.")

                        insert_data_async(timestamp, "JALURCUTTING2", object_id, 1, status, path_capture)
                        countdown_timers[object_id]["violation_last_insert_time"] = violation_seconds

                if object_type == "LINE":
                    for line in lines:
                        if line["id"] == object_id:
                            line["color"] = (0, 0, 255)
                elif object_type == "AREA":
                    for polygon in polygons:
                        if polygon["id"] == object_id:
                            polygon["color"] = (0, 0, 255)

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

                message = f"Terjadi Pelanggaran {violation_seconds}s Di {object_id}"
                cv2.putText(frame, message, (20, y_start), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                y_start += 15

    def handle_violation_end_callback(save_file_path, object_id, object_type):
        logging.info(f"Handling violation end for {object_id}")
        after_violation_end_callback(save_file_path, object_id, object_type)

    def handle_violation_end_async(object_id, object_type, frame, save_path, callback):
        logging.info(f"Menambahkan tugas async untuk mengakhiri pelanggaran {object_id}")
        frame_copy = frame.copy()
        task = {"type": "save_image", "frame": frame_copy, "object_id": object_id, "mid_point": None, "save_path": save_path, "object_type": object_type, "callback": callback, "is_fullscreen": True}
        task_processor.task_queue.put(task)


def check_polygon_violation(frame, gray_frame):
    for polygon in polygons:
        mask = np.zeros_like(gray_frame)
        cv2.fillPoly(mask, [np.array(polygon["points"])], 255)
        current_histogram = calculate_histogram(gray_frame, mask)

        if polygon.get("default_histogram") is None:
            polygon["default_histogram"] = current_histogram.copy()
            logging.info(f"Baseline histogram disimpan untuk {polygon['id']}.")

        if check_histogram_change(current_histogram, polygon["default_histogram"], AREA_SENSITIVITY):
            polygon["color"] = (0, 0, 255)
            start_countdown(polygon["id"], "AREA")
        else:
            polygon["color"] = (0, 255, 0)
            with countdown_timers_lock:
                if polygon["id"] in countdown_timers:
                    if countdown_timers[polygon["id"]].get("is_in_violation"):
                        handle_violation_end_async(polygon["id"], "AREA", frame, screenshot_save_path_area, lambda path, oid=polygon["id"], otype="AREA": handle_violation_end_callback(path, oid, otype))
                    else:
                        del countdown_timers[polygon["id"]]
                        logging.info(f"Countdown dihapus karena {polygon['id']} kembali normal.")

        cv2.polylines(frame, [np.array(polygon["points"])], isClosed=True, color=polygon["color"], thickness=1)
        centroid = np.mean(polygon["points"], axis=0).astype(int)
        cv2.putText(frame, f"{polygon['id']}", tuple(centroid), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1, cv2.LINE_AA)


def clear_data_for_deleted_object(object_id, frame):
    with countdown_timers_lock:
        if object_id in countdown_timers:
            del countdown_timers[object_id]
            logging.info(f"Countdown dihapus untuk {object_id}.")
        if object_id in violation_timers:
            del violation_timers[object_id]
    logging.info(f"Pelanggaran dihapus untuk {object_id}.")


load_coordinates("archives/AI_SOURCE/JALURCUTTING2/coordinates.pkl")
current_histogram_file, next_switch_time, next_histogram_path = initialize_histogram_schedule(schedule)
update_histograms_from_file(current_histogram_file)

cv2.namedWindow("RTSP Stream - Monitoring")
cv2.setMouseCallback("RTSP Stream - Monitoring", draw)

while True:
    ret, frame = cap.read()
    if not ret:
        logging.error("Tidak dapat membaca frame dari RTSP atau stream berhenti.")
        break

    frame = cv2.resize(frame, (960, 540))
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    for line in lines:
        if line["end_point"]:
            mask = np.zeros_like(gray_frame)
            cv2.line(mask, line["start_point"], line["end_point"], 255, thickness=5)
            current_histogram = calculate_histogram(gray_frame, mask)

            if line["default_histogram"] is None:
                line["default_histogram"] = current_histogram.copy()
                logging.info(f"Baseline histogram disimpan untuk {line['id']}.")

            if check_histogram_change(current_histogram, line["default_histogram"], LINE_SENSITIVITY):
                line["color"] = (0, 0, 255)
                start_countdown(line["id"], "LINE")
            else:
                line["color"] = (0, 255, 255)
                with countdown_timers_lock:
                    if line["id"] in countdown_timers:
                        if countdown_timers[line["id"]].get("is_in_violation"):
                            handle_violation_end_async(line["id"], "LINE", frame, screenshot_save_path_line, lambda path, oid=line["id"], otype="LINE": handle_violation_end_callback(path, oid, otype))
                        else:
                            del countdown_timers[line["id"]]
                            logging.info(f"Countdown dihapus karena {line['id']} kembali normal.")

            cv2.line(frame, line["start_point"], line["end_point"], line["color"], 1)
            mid_point = ((line["start_point"][0] + line["end_point"][0]) // 2, (line["start_point"][1] + line["end_point"][1]) // 2)
            cv2.putText(frame, f"{line['id']}", mid_point, cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1, cv2.LINE_AA)

    check_polygon_violation(frame, gray_frame)
    display_countdown_and_violation(frame, screenshot_save_path_line, "LINE", y_start=30)
    display_countdown_and_violation(frame, screenshot_save_path_area, "AREA", y_start=100)

    if polygon_mode and polygon_points:
        for i in range(len(polygon_points) - 1):
            cv2.line(frame, polygon_points[i], polygon_points[i + 1], (0, 255, 0), 1)
        cv2.line(frame, polygon_points[-1], current_mouse_position, (0, 255, 0), 1)

    if not polygon_mode and drawing:
        cv2.line(frame, start_point, current_mouse_position, (0, 255, 255), 1)

    if polygon_mode:
        cv2.putText(frame, "MODE: AREA", (20, 460), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1, cv2.LINE_AA)
    else:
        cv2.putText(frame, "MODE: LINE", (20, 460), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1, cv2.LINE_AA)

    cv2.imshow("RTSP Stream - Monitoring", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        save_coordinates("archives/AI_SOURCE/JALURCUTTING2/coordinates.pkl")
        save_histograms(current_histogram_file)
        logging.info("Program dihentikan oleh pengguna. Data disimpan.")
        break
    elif key == ord("u"):
        if circle_color == (0, 255, 255):
            circle_color = (0, 255, 0)
            polygon_mode = True
            logging.info("Mode diubah ke AREA.")
        else:
            circle_color = (0, 255, 255)
            polygon_mode = False
            polygon_points.clear()
            logging.info("Mode diubah ke LINE.")
    elif key == ord("r") and polygon_mode:
        if len(polygon_points) > 2:
            polygon_id = f"AREA {polygon_id_counter}"
            polygons.append({"points": polygon_points.copy(), "id": polygon_id, "default_histogram": None, "color": (0, 255, 0)})
            logging.info(f"Polygon {polygon_id} ditambahkan dengan titik: {polygon_points}.")
            polygon_id_counter += 1
        polygon_points.clear()
    elif key == ord("c"):
        if polygon_mode and polygons:
            deleted_polygon = polygons.pop()
            clear_data_for_deleted_object(deleted_polygon["id"], frame)
            polygon_id_counter -= 1
            logging.info(f"Polygon {deleted_polygon['id']} dihapus.")
        elif not polygon_mode and lines:
            deleted_line = lines.pop()
            clear_data_for_deleted_object(deleted_line["id"], frame)
            line_id_counter -= 1
            logging.info(f"Garis {deleted_line['id']} dihapus.")
    elif key == ord("y"):
        print("Menambahkan default pixel (default_histogram) untuk garis...")
        for line in lines:
            if line["end_point"]:
                mask = np.zeros_like(gray_frame)
                cv2.line(mask, line["start_point"], line["end_point"], 255, thickness=5)
                current_histogram = calculate_histogram(gray_frame, mask)
                if line["default_histogram"] is None:
                    line["default_histogram"] = []
                if isinstance(line["default_histogram"], np.ndarray):
                    line["default_histogram"] = [line["default_histogram"]]
                line["default_histogram"].append(current_histogram.copy())
                print(f"Default pixel (default_histogram) ditambahkan untuk {line['id']}")
        save_histograms(current_histogram_file)
        print("Default pixel (default_histogram) untuk garis telah ditambahkan dan data disimpan.")
    elif key == ord("t"):
        print("Menambahkan default pixel (default_histogram) untuk area...")
        for polygon in polygons:
            mask = np.zeros_like(gray_frame)
            cv2.fillPoly(mask, [np.array(polygon["points"])], 255)
            current_histogram = calculate_histogram(gray_frame, mask)
            if polygon["default_histogram"] is None:
                polygon["default_histogram"] = []
            if isinstance(polygon["default_histogram"], np.ndarray):
                polygon["default_histogram"] = [polygon["default_histogram"]]
            polygon["default_histogram"].append(current_histogram.copy())
            print(f"Default pixel (default_histogram) ditambahkan untuk {polygon['id']}")
        save_histograms(current_histogram_file)
        print("Default pixel (default_histogram) untuk area telah ditambahkan dan data disimpan.")
    elif key == ord("h"):
        print("Mereset default pixel (default_histogram) untuk garis...")
        for line in lines:
            line["default_histogram"] = None
            print(f"Default pixel (default_histogram) direset untuk {line['id']}")
        save_histograms(current_histogram_file)
        print("Default pixel (default_histogram) untuk garis telah direset dan data disimpan.")
    elif key == ord("g"):
        print("Mereset default pixel (default_histogram) untuk area...")
        for polygon in polygons:
            polygon["default_histogram"] = None
            print(f"Default pixel (default_histogram) direset untuk {polygon['id']}")
        save_histograms(current_histogram_file)
        print("Default pixel (default_histogram) untuk area telah direset dan data disimpan.")

    current_datetime = datetime.now()
    if current_datetime >= next_switch_time:
        logging.info(f"Waktu switch telah tercapai: {next_switch_time.strftime('%H:%M')}. Memuat file histogram baru: {next_histogram_path}")
        update_histograms_from_file(next_histogram_path)
        current_histogram_file = next_histogram_path
        for idx, (hour, minute, histogram_path) in enumerate(schedule):
            try:
                switch_time = current_datetime.replace(hour=hour, minute=minute, second=0, microsecond=0)
            except ValueError:
                switch_time = current_datetime + timedelta(days=1)
                switch_time = switch_time.replace(hour=hour % 24, minute=minute % 60, second=0, microsecond=0)
            if current_datetime < switch_time:
                next_switch_time = switch_time
                next_histogram_path = histogram_path
                break
        else:
            first_switch = schedule[0]
            next_switch_time = (current_datetime + timedelta(days=1)).replace(hour=first_switch[0], minute=first_switch[1], second=0, microsecond=0)
            next_histogram_path = first_switch[2]
        logging.info(f"Switch berikutnya dijadwalkan pada {next_switch_time.strftime('%Y-%m-%d %H:%M')} dengan file histogram: {next_histogram_path}")

cap.release()
cv2.destroyAllWindows()
db_connection.close()
logging.info("Sumber daya dilepaskan dan koneksi database ditutup.")

task_processor.stop()
task_processor.join()
