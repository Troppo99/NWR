import logging
import os
import mysql.connector
import threading
from queue import Queue, Empty
import pickle
import numpy as np
import cv2
import numpy as np
from datetime import datetime, timedelta
import time


class Logger:
    def __init__(self, log_file):
        log_directory = os.path.dirname(log_file)
        os.makedirs(log_directory, exist_ok=True)
        logging.basicConfig(filename=log_file, filemode="w", level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")
        self.logger = logging.getLogger()

    def info(self, message):
        self.logger.info(message)

    def error(self, message):
        self.logger.error(message)

    def warning(self, message):
        self.logger.warning(message)

    def debug(self, message):
        self.logger.debug(message)


class DatabaseHandler:
    def __init__(self, config, logger):
        self.logger = logger
        self.config = config
        self.connect()

    def connect(self):
        try:
            self.connection = mysql.connector.connect(**self.config)
            self.connection.autocommit = True
            self.cursor = self.connection.cursor()
            self.logger.info("Koneksi ke database berhasil.")
        except mysql.connector.Error as err:
            self.logger.error(f"Error koneksi database: {err}")
            raise

    def insert_into_database(self, timestamp, cam_name, no_line, time_detect, status, path_capture):
        try:
            sql = """
                INSERT INTO line_detectx (timestamp, cam_name, no_line, time_detect, status, path_capture)
                VALUES (%s, %s, %s, %s, %s, %s)
            """
            values = (timestamp, cam_name, no_line, time_detect, status, path_capture)
            self.cursor.execute(sql, values)
            self.connection.commit()
            self.logger.info(f"Data berhasil dimasukkan ke database: {values}")
        except mysql.connector.Error as err:
            self.logger.error(f"Error memasukkan data ke database: {err}")
        except Exception as e:
            self.logger.error(f"Exception saat memasukkan data ke database: {e}")

    def close(self):
        try:
            self.cursor.close()
            self.connection.close()
            self.logger.info("Koneksi database ditutup.")
        except Exception as e:
            self.logger.error(f"Error saat menutup koneksi database: {e}")


class TaskProcessor(threading.Thread):
    def __init__(self, logger, db_handler):
        super().__init__()
        self.task_queue = Queue()
        self.stop_event = threading.Event()
        self.logger = logger
        self.db_handler = db_handler

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
                self.logger.error(f"Exception in TaskProcessor: {e}")

    def stop(self):
        self.stop_event.set()

    def process_save_image_task(self, task):
        try:
            frame = task["frame"]
            object_id = task["object_id"]
            save_path = task["save_path"]
            object_type = task["object_type"]
            callback = task["callback"]
            is_fullscreen = task.get("is_fullscreen", False)

            if is_fullscreen:
                path = self.save_fullscreen_image(frame, object_id, save_path, object_type)
            else:
                mid_point = task["mid_point"]
                path = self.capture_and_save_screenshot(frame, object_id, mid_point, save_path, object_type)

            if callback:
                callback(path)
        except Exception as e:
            self.logger.error(f"Error processing save_image_task: {e}")

    def process_insert_db_task(self, task):
        try:
            self.logger.info(f"Memproses insert_db untuk {task['no_line']}")
            timestamp = task["timestamp"]
            cam_name = task["cam_name"]
            no_line = task["no_line"]
            time_detect = task["time_detect"]
            status = task["status"]
            path_capture = task["path_capture"]

            self.db_handler.insert_into_database(timestamp, cam_name, no_line, time_detect, status, path_capture)
            self.logger.info(f"Tugas insert_db selesai untuk {no_line}")
        except Exception as e:
            self.logger.error(f"Error processing insert_db_task: {e}")

    def capture_and_save_screenshot(self, frame, object_id, mid_point, save_path, object_type):
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

        full_save_path = os.path.join(save_path, filename)

        success = cv2.imwrite(full_save_path, cropped_img)
        if success:
            self.logger.info(f"Gambar disimpan: {full_save_path}")
        else:
            self.logger.error(f"Failed to save image to {full_save_path}")
        return full_save_path

    def save_fullscreen_image(self, frame, object_id, save_path, object_type):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"FULLSCREEN_{object_id}_{timestamp}.jpg"
        full_save_path = os.path.join(save_path, filename)

        os.makedirs(save_path, exist_ok=True)

        success = cv2.imwrite(full_save_path, frame)
        if success:
            self.logger.info(f"Fullscreen gambar disimpan: {full_save_path}")
        else:
            self.logger.error(f"Failed to save image to {full_save_path}")
        return full_save_path

    def add_task(self, task):
        self.task_queue.put(task)


class CoordinateManager:
    def __init__(self, coordinate_file, logger):
        self.coordinate_file = coordinate_file
        self.logger = logger

    def save_coordinates(self, lines, polygons, line_id_counter, polygon_id_counter):
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
            with open(self.coordinate_file, "wb") as f:
                pickle.dump(data, f)
            self.logger.info(f"Coordinate data saved to {self.coordinate_file}")
        except Exception as e:
            self.logger.error(f"Error saving coordinate data to {self.coordinate_file}: {e}")

    def load_coordinates(self):
        try:
            with open(self.coordinate_file, "rb") as f:
                data = pickle.load(f)
                self.logger.info(f"Coordinate data loaded from {self.coordinate_file}")
                return data.get("lines", []), data.get("polygons", []), data.get("line_id_counter", 1), data.get("polygon_id_counter", 1)
        except FileNotFoundError:
            self.logger.warning(f"No coordinate data found in {self.coordinate_file}")
            return [], [], 1, 1
        except Exception as e:
            self.logger.error(f"Error loading coordinate data from {self.coordinate_file}: {e}")
            return [], [], 1, 1


class HistogramManager:
    def __init__(self, histogram_file, logger):
        self.histogram_file = histogram_file
        self.logger = logger

    def save_histograms(self, lines, polygons):
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
            np.save(self.histogram_file, histogram_data, allow_pickle=True)
            self.logger.info(f"Histogram data saved to {self.histogram_file}")
        except Exception as e:
            self.logger.error(f"Error saving histogram data to {self.histogram_file}: {e}")

    def load_histograms(self, lines, polygons):
        try:
            histogram_data = np.load(self.histogram_file, allow_pickle=True).item()
            line_histograms = histogram_data.get("lines", {})
            polygon_histograms = histogram_data.get("polygons", {})

            for line in lines:
                line_id = line["id"]
                if line_id in line_histograms:
                    line["default_histogram"] = line_histograms[line_id]
                    self.logger.info(f"Default histogram updated untuk {line_id}")

            for polygon in polygons:
                polygon_id = polygon["id"]
                if polygon_id in polygon_histograms:
                    polygon["default_histogram"] = polygon_histograms[polygon_id]
                    self.logger.info(f"Default histogram updated untuk {polygon_id}")

            self.logger.info(f"Histogram data loaded from {self.histogram_file}")
        except FileNotFoundError:
            self.logger.warning(f"No histogram data found in {self.histogram_file}")
        except Exception as e:
            self.logger.error(f"Error loading histogram data from {self.histogram_file}: {e}")


class Detector:
    def __init__(self, logger, task_processor, db_handler, screenshot_save_path_line, screenshot_save_path_area):
        self.logger = logger
        self.task_processor = task_processor
        self.db_handler = db_handler
        self.screenshot_save_path_line = screenshot_save_path_line
        self.screenshot_save_path_area = screenshot_save_path_area

        self.countdown_timers = {}
        self.violation_timers = {}
        self.countdown_timers_lock = threading.Lock()

        self.LINE_COUNTDOWN_DURATION = 180
        self.LINE_SCREENSHOT_TIME = 177
        self.LINE_SENSITIVITY = 0.6

        self.AREA_COUNTDOWN_DURATION = 180
        self.AREA_SCREENSHOT_TIME = 177
        self.AREA_SENSITIVITY = 0.6

    def calculate_histogram(self, gray_frame, mask):
        hist = cv2.calcHist([gray_frame], [0], mask, [256], [0, 256])
        cv2.normalize(hist, hist)
        return hist

    def check_histogram_change(self, current_histogram, default_histogram, sensitivity):
        if isinstance(default_histogram, list):
            if len(default_histogram) > 0:
                default_histogram = np.mean(default_histogram, axis=0)
            else:
                self.logger.error("Default histogram kosong.")
                return False
        hist_diff = cv2.compareHist(default_histogram, current_histogram, cv2.HISTCMP_CORREL)
        result = hist_diff < (1.0 - sensitivity)
        self.logger.debug(f"Histogram correlation: {hist_diff}, Sensitivity Threshold: {1.0 - sensitivity}, Change Detected: {result}")
        return result

    def start_countdown(self, object_id, object_type):
        with self.countdown_timers_lock:
            if object_id not in self.countdown_timers:
                if object_type == "LINE":
                    countdown_duration = self.LINE_COUNTDOWN_DURATION
                    screenshot_time = self.LINE_SCREENSHOT_TIME
                else:
                    countdown_duration = self.AREA_COUNTDOWN_DURATION
                    screenshot_time = self.AREA_SCREENSHOT_TIME
                self.countdown_timers[object_id] = {"start_time": time.time(), "captured": False, "path_capture": "0", "last_insert_time": 0, "violation_start_time": None, "violation_last_insert_time": 0, "first_violation": True, "is_in_violation": False, "countdown_duration": countdown_duration, "screenshot_time": screenshot_time}
                self.logger.info(f"Countdown dimulai untuk {object_type} {object_id} dengan durasi {countdown_duration} detik.")

    def handle_violation_end_callback(self, save_file_path, object_id, object_type):
        self.logger.info(f"Handling violation end for {object_id}")
        timestamp_db = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cam_name = "JALURCUTTING2"
        no_line = object_id
        time_detect = 1
        status = "done"

        self.db_handler.insert_into_database(timestamp_db, cam_name, no_line, time_detect, status, save_file_path)
        self.logger.info(f"Pelanggaran untuk {object_id} telah selesai dan data 'done' dimasukkan ke database.")

        with self.countdown_timers_lock:
            if object_id in self.countdown_timers:
                self.countdown_timers[object_id]["is_in_violation"] = False
                self.logger.info(f"Menandai {object_id} sebagai tidak dalam pelanggaran lagi.")

    def handle_violation_end_async(self, object_id, object_type, frame, save_path, callback):
        self.logger.info(f"Menambahkan tugas async untuk mengakhiri pelanggaran {object_id}")
        frame_copy = frame.copy()
        task = {"type": "save_image", "frame": frame_copy, "object_id": object_id, "mid_point": None, "save_path": save_path, "object_type": object_type, "callback": callback, "is_fullscreen": True}
        self.task_processor.add_task(task)

    def process_line(self, frame, gray_frame, line):
        if not line.get("end_point"):
            return

        mask = np.zeros_like(gray_frame)
        cv2.line(mask, line["start_point"], line["end_point"], 255, thickness=5)
        current_histogram = self.calculate_histogram(gray_frame, mask)

        if line["default_histogram"] is None:
            line["default_histogram"] = current_histogram.copy()
            self.logger.info(f"Baseline histogram disimpan untuk {line['id']}.")

        if self.check_histogram_change(current_histogram, line["default_histogram"], self.LINE_SENSITIVITY):
            line["color"] = (0, 0, 255)
            self.start_countdown(line["id"], "LINE")
        else:
            line["color"] = (0, 255, 255)
            with self.countdown_timers_lock:
                if line["id"] in self.countdown_timers:
                    if self.countdown_timers[line["id"]].get("is_in_violation"):
                        self.handle_violation_end_async(line["id"], "LINE", frame, self.screenshot_save_path_line, lambda path, oid=line["id"], otype="LINE": self.handle_violation_end_callback(path, oid, otype))
                    else:
                        del self.countdown_timers[line["id"]]
                        self.logger.info(f"Countdown dihapus karena {line['id']} kembali normal.")

        cv2.line(frame, line["start_point"], line["end_point"], line["color"], 1)
        mid_point = ((line["start_point"][0] + line["end_point"][0]) // 2, (line["start_point"][1] + line["end_point"][1]) // 2)
        cv2.putText(frame, f"{line['id']}", mid_point, cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1, cv2.LINE_AA)

    def process_polygon(self, frame, gray_frame, polygon):
        mask = np.zeros_like(gray_frame)
        cv2.fillPoly(mask, [np.array(polygon["points"])], 255)
        current_histogram = self.calculate_histogram(gray_frame, mask)

        if polygon["default_histogram"] is None:
            polygon["default_histogram"] = current_histogram.copy()
            self.logger.info(f"Baseline histogram disimpan untuk {polygon['id']}.")

        if self.check_histogram_change(current_histogram, polygon["default_histogram"], self.AREA_SENSITIVITY):
            polygon["color"] = (0, 0, 255)
            self.start_countdown(polygon["id"], "AREA")
        else:
            polygon["color"] = (0, 255, 0)
            with self.countdown_timers_lock:
                if polygon["id"] in self.countdown_timers:
                    if self.countdown_timers[polygon["id"]].get("is_in_violation"):
                        self.handle_violation_end_async(polygon["id"], "AREA", frame, self.screenshot_save_path_area, lambda path, oid=polygon["id"], otype="AREA": self.handle_violation_end_callback(path, oid, otype))
                    else:
                        del self.countdown_timers[polygon["id"]]
                        self.logger.info(f"Countdown dihapus karena {polygon['id']} kembali normal.")

        cv2.polylines(frame, [np.array(polygon["points"])], isClosed=True, color=polygon["color"], thickness=1)
        centroid = np.mean(polygon["points"], axis=0).astype(int)
        cv2.putText(frame, f"{polygon['id']}", tuple(centroid), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1, cv2.LINE_AA)

    def display_countdown_and_violation(self, frame, save_path, object_type="LINE", y_start=50):
        current_time = time.time()
        with self.countdown_timers_lock:
            timers_items = list(self.countdown_timers.items())
        for object_id, timer_data in timers_items:
            if (object_type == "LINE" and object_id.startswith("LINE")) or (object_type == "AREA" and object_id.startswith("AREA")):
                elapsed_time = current_time - timer_data["start_time"]
                remaining_time = int(timer_data["countdown_duration"] - elapsed_time)
                if remaining_time > 0:
                    status = "warning"
                    if int(elapsed_time) > timer_data["last_insert_time"]:
                        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        self.db_handler.insert_into_database(timestamp, "JALURCUTTING2", object_id, 1, status, "0")
                        self.countdown_timers[object_id]["last_insert_time"] = int(elapsed_time)

                    screenshot_time = timer_data["screenshot_time"]
                    if remaining_time == self.countdown_timers[object_id]["countdown_duration"] - screenshot_time and not timer_data["captured"]:
                        current_object_id = object_id
                        current_status = status

                        def screenshot_callback(path, oid=current_object_id, otype=object_type):
                            with self.countdown_timers_lock:
                                if oid in self.countdown_timers:
                                    self.countdown_timers[oid]["path_capture"] = path
                                    self.countdown_timers[oid]["captured"] = True
                            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            self.db_handler.insert_into_database(timestamp, "JALURCUTTING2", oid, 1, current_status, path)

                        if object_type == "LINE":
                            # Assuming lines is accessible; you might need to pass it as a parameter or manage differently
                            pass
                        elif object_type == "AREA":
                            # Assuming polygons is accessible; you might need to pass it as a parameter or manage differently
                            pass

                        # Implement capture logic here based on object_type
                        # This part perlu diintegrasikan dengan kelas lain atau dijadikan parameter

                    y_position = y_start
                    x_position = frame.shape[1] - 150
                    cv2.putText(frame, f"{object_id}: {remaining_time}s", (x_position, y_position), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1, cv2.LINE_AA)
                    y_start += 15
                else:
                    status = "violation"
                    with self.countdown_timers_lock:
                        if timer_data.get("violation_start_time") is None:
                            self.countdown_timers[object_id]["violation_start_time"] = current_time
                            self.countdown_timers[object_id]["is_in_violation"] = True
                            self.logger.info(f"Violation dimulai untuk {object_type} {object_id}.")

                        violation_start_time = self.countdown_timers[object_id]["violation_start_time"]
                        violation_elapsed = current_time - violation_start_time
                        violation_seconds = int(violation_elapsed)

                        if violation_seconds > self.countdown_timers[object_id].get("violation_last_insert_time", 0):
                            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                            if self.countdown_timers[object_id].get("first_violation", True):
                                path_capture = self.countdown_timers[object_id].get("path_capture", "0")
                                self.countdown_timers[object_id]["first_violation"] = False
                                self.logger.info(f"Pelanggaran pertama untuk {object_id}: path_capture disamakan dengan warning sebelumnya.")
                            else:
                                path_capture = "0"
                                self.logger.info(f"Pelanggaran berikutnya untuk {object_id}: path_capture diisi '0'.")

                            self.db_handler.insert_into_database(timestamp, "JALURCUTTING2", object_id, 1, status, path_capture)
                            self.countdown_timers[object_id]["violation_last_insert_time"] = violation_seconds

                    if object_type == "LINE":
                        # Update color and annotations for LINE
                        pass
                    elif object_type == "AREA":
                        # Update color and annotations for AREA
                        pass

                    message = f"Terjadi Pelanggaran {violation_seconds}s Di {object_id}"
                    cv2.putText(frame, message, (20, y_start), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                    y_start += 15


class VideoCaptureHandler:
    def __init__(self, rtsp_url, logger):
        self.rtsp_url = rtsp_url
        self.logger = logger
        self.cap = cv2.VideoCapture(self.rtsp_url)
        if not self.cap.isOpened():
            self.logger.error("Tidak dapat membuka stream RTSP.")
            raise ConnectionError("Tidak dapat membuka stream RTSP.")

    def read_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            self.logger.error("Tidak dapat membaca frame dari RTSP atau stream berhenti.")
            return None
        return frame

    def release(self):
        self.cap.release()
        self.logger.info("VideoCapture dilepaskan.")


class GUIHandler:
    def __init__(self, logger, coordinate_manager, histogram_manager):
        self.logger = logger
        self.coordinate_manager = coordinate_manager
        self.histogram_manager = histogram_manager
        self.drawing = False
        self.polygon_mode = False
        self.start_point = None
        self.end_point = None
        self.lines = []
        self.polygons = []
        self.polygon_points = []
        self.current_mouse_position = (0, 0)
        self.line_id_counter = 1
        self.polygon_id_counter = 1

    def load_coordinates(self, coordinate_file):
        lines, polygons, line_id_counter, polygon_id_counter = self.coordinate_manager.load_coordinates()
        self.lines = lines
        self.polygons = polygons
        self.line_id_counter = line_id_counter
        self.polygon_id_counter = polygon_id_counter

    def save_coordinates(self, coordinate_file):
        self.coordinate_manager.save_coordinates(self.lines, self.polygons, self.line_id_counter, self.polygon_id_counter)

    def draw(self, event, x, y, flags, param):
        self.current_mouse_position = (x, y)

        if self.polygon_mode:
            if event == cv2.EVENT_LBUTTONDOWN:
                self.polygon_points.append((x, y))
        else:
            if event == cv2.EVENT_LBUTTONDOWN:
                self.drawing = True
                self.start_point = (x, y)
                self.end_point = (x, y)
            elif event == cv2.EVENT_MOUSEMOVE and self.drawing:
                self.end_point = (x, y)
            elif event == cv2.EVENT_LBUTTONUP and self.drawing:
                self.drawing = False
                self.end_point = (x, y)
                line = {"id": f"LINE {self.line_id_counter}", "start_point": self.start_point, "end_point": self.end_point, "default_histogram": None, "line_detected": False, "color": (0, 255, 255)}
                self.lines.append(line)
                self.logger.info(f"Garis {self.line_id_counter} ditambahkan dari {self.start_point} ke {self.end_point}.")
                self.line_id_counter += 1

    def handle_keypress(self, key, frame, gray_frame, detector):
        if key == ord("q"):
            self.save_coordinates("archives/AI_SOURCE/JALURCUTTING2/coordinates.pkl")
            self.histogram_manager.save_histograms(self.lines, self.polygons)
            self.logger.info("Program dihentikan oleh pengguna. Data disimpan.")
            return False
        elif key == ord("u"):
            if not self.polygon_mode:
                self.polygon_mode = True
                self.logger.info("Mode diubah ke AREA.")
            else:
                self.polygon_mode = False
                self.polygon_points.clear()
                self.logger.info("Mode diubah ke LINE.")
        elif key == ord("r") and self.polygon_mode:
            if len(self.polygon_points) > 2:
                polygon_id = f"AREA {self.polygon_id_counter}"
                polygon = {"points": self.polygon_points.copy(), "id": polygon_id, "default_histogram": None, "color": (0, 255, 0)}
                self.polygons.append(polygon)
                self.logger.info(f"Polygon {polygon_id} ditambahkan dengan titik: {self.polygon_points}.")
                self.polygon_id_counter += 1
            self.polygon_points.clear()
        elif key == ord("c"):
            if self.polygon_mode and self.polygons:
                deleted_polygon = self.polygons.pop()
                self.logger.info(f"Polygon {deleted_polygon['id']} dihapus.")
            elif not self.polygon_mode and self.lines:
                deleted_line = self.lines.pop()
                self.logger.info(f"Garis {deleted_line['id']} dihapus.")
        elif key == ord("y"):
            self.add_default_histogram_for_lines(frame, gray_frame)
        elif key == ord("t"):
            self.add_default_histogram_for_polygons(frame, gray_frame)
        elif key == ord("h"):
            self.reset_histogram_for_lines()
        elif key == ord("g"):
            self.reset_histogram_for_polygons()
        return True

    def add_default_histogram_for_lines(self, frame, gray_frame):
        self.logger.info("Menambahkan default histogram untuk garis...")
        for line in self.lines:
            if line["end_point"]:
                mask = np.zeros_like(gray_frame)
                cv2.line(mask, line["start_point"], line["end_point"], 255, thickness=5)
                current_histogram = self.calculate_histogram(gray_frame, mask)
                if line["default_histogram"] is None:
                    line["default_histogram"] = []
                if isinstance(line["default_histogram"], np.ndarray):
                    line["default_histogram"] = [line["default_histogram"]]
                line["default_histogram"].append(current_histogram.copy())
                self.logger.info(f"Default histogram ditambahkan untuk {line['id']}")
        self.histogram_manager.save_histograms(self.lines, self.polygons)
        self.logger.info("Default histogram untuk garis telah ditambahkan dan data disimpan.")

    def add_default_histogram_for_polygons(self, frame, gray_frame):
        self.logger.info("Menambahkan default histogram untuk area...")
        for polygon in self.polygons:
            mask = np.zeros_like(gray_frame)
            cv2.fillPoly(mask, [np.array(polygon["points"])], 255)
            current_histogram = self.calculate_histogram(gray_frame, mask)
            if polygon["default_histogram"] is None:
                polygon["default_histogram"] = []
            if isinstance(polygon["default_histogram"], np.ndarray):
                polygon["default_histogram"] = [polygon["default_histogram"]]
            polygon["default_histogram"].append(current_histogram.copy())
            self.logger.info(f"Default histogram ditambahkan untuk {polygon['id']}")
        self.histogram_manager.save_histograms(self.lines, self.polygons)
        self.logger.info("Default histogram untuk area telah ditambahkan dan data disimpan.")

    def reset_histogram_for_lines(self):
        self.logger.info("Mereset default histogram untuk garis...")
        for line in self.lines:
            line["default_histogram"] = None
            self.logger.info(f"Default histogram direset untuk {line['id']}")
        self.histogram_manager.save_histograms(self.lines, self.polygons)
        self.logger.info("Default histogram untuk garis telah direset dan data disimpan.")

    def reset_histogram_for_polygons(self):
        self.logger.info("Mereset default histogram untuk area...")
        for polygon in self.polygons:
            polygon["default_histogram"] = None
            self.logger.info(f"Default histogram direset untuk {polygon['id']}")
        self.histogram_manager.save_histograms(self.lines, self.polygons)
        self.logger.info("Default histogram untuk area telah direset dan data disimpan.")


class Monitor:
    def __init__(self, config):
        # Initialize Logger
        self.logger = Logger(config["log_file"])

        # Initialize DatabaseHandler
        self.db_handler = DatabaseHandler(config["db_config"], self.logger)

        # Initialize TaskProcessor
        self.task_processor = TaskProcessor(self.logger, self.db_handler)
        self.task_processor.start()

        # Initialize CoordinateManager and HistogramManager
        self.coordinate_manager = CoordinateManager(config["coordinate_file"], self.logger)
        self.histogram_manager = HistogramManager(config["histogram_file"], self.logger)

        # Initialize GUIHandler
        self.gui_handler = GUIHandler(self.logger, self.coordinate_manager, self.histogram_manager)
        self.gui_handler.load_coordinates(config["coordinate_file"])

        # Initialize VideoCaptureHandler
        self.video_capture = VideoCaptureHandler(config["rtsp_url"], self.logger)

        # Initialize Detector
        self.detector = Detector(self.logger, self.task_processor, self.db_handler, config["screenshot_save_path_line"], config["screenshot_save_path_area"])

        # Initialize Schedule
        self.schedule = sorted(config["schedule"], key=lambda x: (x[0], x[1]))
        self.last_histogram_path, self.next_switch_time, self.next_histogram_path = self.initialize_histogram_schedule()

        # Load initial histograms
        self.histogram_manager.load_histograms(self.gui_handler.lines, self.gui_handler.polygons)

    def initialize_histogram_schedule(self):
        current_time = datetime.now()
        last_switch_time = None
        last_histogram_path = None
        next_switch_time = None
        next_histogram_path = None

        for hour, minute, histogram_path in self.schedule:
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
            first_switch = self.schedule[0]
            next_switch_time = (current_time + timedelta(days=1)).replace(hour=first_switch[0], minute=first_switch[1], second=0, microsecond=0)
            next_histogram_path = first_switch[2]

        return last_histogram_path, next_switch_time, next_histogram_path

    def update_histogram_schedule(self):
        current_datetime = datetime.now()
        if current_datetime >= self.next_switch_time:
            self.logger.info(f"Waktu switch telah tercapai: {self.next_switch_time.strftime('%H:%M')}. Memuat file histogram baru: {self.next_histogram_path}")
            self.histogram_manager.load_histograms(self.gui_handler.lines, self.gui_handler.polygons)
            self.last_histogram_path = self.next_histogram_path
            for idx, (hour, minute, histogram_path) in enumerate(self.schedule):
                try:
                    switch_time = current_datetime.replace(hour=hour, minute=minute, second=0, microsecond=0)
                except ValueError:
                    switch_time = current_datetime + timedelta(days=1)
                    switch_time = switch_time.replace(hour=hour % 24, minute=minute % 60, second=0, microsecond=0)
                if current_datetime < switch_time:
                    self.next_switch_time = switch_time
                    self.next_histogram_path = histogram_path
                    break
            else:
                first_switch = self.schedule[0]
                self.next_switch_time = (current_datetime + timedelta(days=1)).replace(hour=first_switch[0], minute=first_switch[1], second=0, microsecond=0)
                self.next_histogram_path = first_switch[2]
            self.logger.info(f"Switch berikutnya dijadwalkan pada {self.next_switch_time.strftime('%Y-%m-%d %H:%M')} dengan file histogram: {self.next_histogram_path}")

    def run(self):
        cv2.namedWindow("RTSP Stream - Monitoring")
        cv2.setMouseCallback("RTSP Stream - Monitoring", self.gui_handler.draw)

        while True:
            frame = self.video_capture.read_frame()
            if frame is None:
                break

            frame = cv2.resize(frame, (960, 540))
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Process Lines
            for line in self.gui_handler.lines:
                self.detector.process_line(frame, gray_frame, line)

            # Process Polygons
            for polygon in self.gui_handler.polygons:
                self.detector.process_polygon(frame, gray_frame, polygon)

            # Display Countdown and Violations
            self.detector.display_countdown_and_violation(frame, self.detector.screenshot_save_path_line, "LINE", y_start=30)
            self.detector.display_countdown_and_violation(frame, self.detector.screenshot_save_path_area, "AREA", y_start=100)

            # Draw Polygon Points
            if self.gui_handler.polygon_mode and self.gui_handler.polygon_points:
                for i in range(len(self.gui_handler.polygon_points) - 1):
                    cv2.line(frame, self.gui_handler.polygon_points[i], self.gui_handler.polygon_points[i + 1], (0, 255, 0), 1)
                cv2.line(frame, self.gui_handler.polygon_points[-1], self.gui_handler.current_mouse_position, (0, 255, 0), 1)

            # Draw Line if Drawing
            if not self.gui_handler.polygon_mode and self.gui_handler.drawing:
                cv2.line(frame, self.gui_handler.start_point, self.gui_handler.current_mouse_position, (0, 255, 255), 1)

            # Indicate Mode
            mode_text = "MODE: AREA" if self.gui_handler.polygon_mode else "MODE: LINE"
            color = (0, 255, 0) if self.gui_handler.polygon_mode else (0, 255, 255)
            cv2.putText(frame, mode_text, (20, 460), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)

            cv2.imshow("RTSP Stream - Monitoring", frame)

            key = cv2.waitKey(1) & 0xFF
            if not self.gui_handler.handle_keypress(key, frame, gray_frame, self.detector):
                break

            # Update Histogram Schedule
            self.update_histogram_schedule()

        # Cleanup
        self.gui_handler.save_coordinates("archives/AI_SOURCE/JALURCUTTING2/coordinates.pkl")
        self.histogram_manager.save_histograms(self.gui_handler.lines, self.gui_handler.polygons)
        self.logger.info("Program dihentikan oleh pengguna. Data disimpan.")
        self.video_capture.release()
        cv2.destroyAllWindows()
        self.db_handler.close()
        self.task_processor.stop()
        self.task_processor.join()
        self.logger.info("Sumber daya dilepaskan dan koneksi database ditutup.")


if __name__ == "__main__":
    config = {
        "log_file": "archives/AI_SOURCE/JALURCUTTING2/monitoring.log",
        "rtsp_url": "rtsp://admin:oracle2015@10.5.0.138:554/Streaming/Channels/1",
        "db_config": {"host": "10.5.0.2", "user": "robot", "password": "robot123", "database": "report_ai_cctv", "port": 3307},
        "coordinate_file": "archives/AI_SOURCE/JALURCUTTING2/coordinates.pkl",
        "histogram_file": "archives/AI_SOURCE/JALURCUTTING2/JALURCUTTING2NET.npy",
        "screenshot_save_path_line": "archives/ai_source/JALURCUTTING2/RED_LINE",
        "screenshot_save_path_area": "archives/ai_source/JALURCUTTING2/GREEN_AREA",
        "schedule": [
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
        ],
    }

    monitor = Monitor(config)
    monitor.run()
