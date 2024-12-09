import os
import cv2
from ultralytics import YOLO
import torch
import cvzone
import time
import threading
import queue
import math
import numpy as np
from shapely.geometry import Polygon
import pymysql
from datetime import datetime


class BroomDetector:
    def __init__(
        self,
        BROOM_CONFIDENCE_THRESHOLD=0.0,  # set confidence 0
        rtsp_url=None,
        camera_name=None,
        window_size=(540, 360),
        display=True,
    ):
        self.BROOM_CONFIDENCE_THRESHOLD = BROOM_CONFIDENCE_THRESHOLD
        self.window_width, self.window_height = window_size
        self.new_width, self.new_height = (960, 540)
        self.prev_frame_time = 0
        self.fps = 0
        self.camera_name = camera_name
        self.borders, self.ip_camera = self.camera_config()
        self.display = display
        if not self.display:
            print(f"B`{self.camera_name} : >>>Display is disabled!<<<")

        self.is_local_file = False
        if rtsp_url is not None:
            self.rtsp_url = rtsp_url
            if os.path.isfile(rtsp_url):
                self.is_local_file = True
                cap = cv2.VideoCapture(self.rtsp_url)
                self.video_fps = cap.get(cv2.CAP_PROP_FPS)
                if not self.video_fps or math.isnan(self.video_fps):
                    self.video_fps = 25
                cap.release()
                print(f"B`{self.camera_name} : Local video file detected. FPS: {self.video_fps}")
            else:
                self.is_local_file = False
                print(f"B`{self.camera_name} : RTSP stream detected. URL: {self.rtsp_url}")
                self.video_fps = None
        else:
            self.rtsp_url = f"rtsp://admin:oracle2015@{self.ip_camera}:554/Streaming/Channels/1"
            self.video_fps = None
            self.is_local_file = False

        self.frame_queue = queue.Queue(maxsize=10)
        self.stop_event = threading.Event()
        self.frame_thread = None

        # Gunakan model last.pt sesuai permintaan
        self.broom_model = YOLO("D:/NWR/run/kon/version1/weights/last.pt").to("cuda")
        self.broom_model.overrides["verbose"] = False
        print(f"Model Broom device: {next(self.broom_model.model.parameters()).device}")

        self.no_detection_duration = 3
        self.violation_start_time = None
        self.last_detected_time = None
        self.timestamp_start = None

    def camera_config(self):
        config = {
            "CUTTING8": {
                "borders": [],
                "borders": [[(46, 256), (136, 224), (581, 295), (1109, 398), (1275, 436), (1275, 719), (989, 719), (682, 613), (437, 499), (238, 385), (111, 306)]],
                "ip": "10.5.0.95",
            },
        }
        if self.camera_name not in config:
            return [], None
        original_borders = config[self.camera_name]["borders"]
        ip = config[self.camera_name]["ip"]
        scaled_borders = []
        for border_group in original_borders:
            scaled_group = []
            for x, y in border_group:
                scaled_x = int(x * (960 / 1280))
                scaled_y = int(y * (540 / 720))
                scaled_group.append((scaled_x, scaled_y))
            if len(scaled_group) >= 3:
                polygon = Polygon(scaled_group)
                if polygon.is_valid:
                    scaled_borders.append(polygon)
        return scaled_borders, ip

    def frame_capture(self):
        while not self.stop_event.is_set():
            cap = cv2.VideoCapture(self.rtsp_url)
            if not cap.isOpened():
                print(f"B`{self.camera_name} : Failed to open stream. Retrying in 5 seconds...")
                cap.release()
                time.sleep(5)
                continue
            while not self.stop_event.is_set():
                ret, frame = cap.read()
                if not ret:
                    print(f"B`{self.camera_name} : Failed to read frame. Reconnecting in 5 seconds...")
                    cap.release()
                    time.sleep(5)
                    break
                try:
                    self.frame_queue.put(frame, timeout=1)
                except queue.Full:
                    pass
            cap.release()

    def process_model(self, frame):
        # conf=0 karena user minta confidence 0
        with torch.no_grad():
            results = self.broom_model.predict(frame, conf=0.5)
        return results

    def export_frame(self, results):
        boxes_info = []
        overlap_detected = False
        for result in results:
            if result.masks is None:
                continue
            for poly_xy in result.masks.xy:
                if len(poly_xy) < 3:
                    # Jika titik polygon kurang dari 3, tidak dapat membentuk polygon yang valid
                    continue
                polygon = Polygon(poly_xy)
                if polygon.is_empty or not polygon.is_valid:
                    # Lewati polygon kosong atau tidak valid
                    continue
                poly_area = polygon.area
                intersection_area_sum = 0.0
                for border in self.borders:
                    if polygon.intersects(border):
                        inter = polygon.intersection(border)
                        if not inter.is_empty:
                            intersection_area_sum += inter.area
                inside = False
                if intersection_area_sum > 0.5 * poly_area:
                    inside = True
                    overlap_detected = True

                # Pastikan polygon tidak kosong sebelum mengambil centroid
                if not polygon.is_empty:
                    c = polygon.centroid
                    boxes_info.append((poly_xy, inside, (c.x, c.y)))
        return boxes_info, overlap_detected

    def process_frame(self, frame, current_time):
        frame_resized = cv2.resize(frame, (self.new_width, self.new_height))
        results = self.process_model(frame_resized)
        boxes_info, overlap_detected = self.export_frame(results)

        any_inside = any(bi[1] for bi in boxes_info)
        if any_inside:
            self.last_detected_time = current_time
            if self.violation_start_time is None:
                self.violation_start_time = current_time
        else:
            if self.last_detected_time is not None:
                if (current_time - self.last_detected_time) > self.no_detection_duration:
                    self.violation_start_time = None
                    self.last_detected_time = None

        if overlap_detected and self.timestamp_start is None:
            self.timestamp_start = datetime.now()

        if self.display:
            self.draw_borders(frame_resized)

            # Buat overlay untuk menggambar polygon dengan warna solid
            overlay = frame_resized.copy()

            for poly_xy, inside, (cx, cy) in boxes_info:
                pts = np.array(poly_xy, np.int32).reshape((-1, 1, 2))
                if inside:
                    # Violation
                    cv2.fillPoly(overlay, [pts], (0, 70, 255))
                else:
                    # Warning
                    cv2.fillPoly(overlay, [pts], (0, 255, 255))

            # Campurkan overlay dengan frame_resized dengan transparansi 50%
            alpha = 0.5
            cv2.addWeighted(overlay, alpha, frame_resized, 1 - alpha, 0, frame_resized)

            # Setelah transparansi diaplikasikan, baru tulis teks di atasnya
            for poly_xy, inside, (cx, cy) in boxes_info:
                if inside:
                    # Hitung waktu violation
                    if self.violation_start_time is not None:
                        elapsed = current_time - self.violation_start_time
                        hh = int(elapsed // 3600)
                        mm = int((elapsed % 3600) // 60)
                        ss = int(elapsed % 60)
                        timer_str = f"{hh:02}:{mm:02}:{ss:02}"
                        cvzone.putTextRect(frame_resized, timer_str, (int(cx), int(cy) - 40), scale=1, thickness=2, offset=5, colorR=(0, 70, 255), colorT=(255, 255, 255))
                    cvzone.putTextRect(frame_resized, "Violation!", (int(cx), int(cy) - 10), scale=1, thickness=2, offset=5, colorR=(0, 70, 255), colorT=(255, 255, 255))
                else:
                    cvzone.putTextRect(frame_resized, "Warning!", (int(cx), int(cy) - 10), scale=1, thickness=2, offset=5, colorR=(0, 255, 255), colorT=(0, 0, 0))

        return frame_resized, overlap_detected

    def draw_borders(self, frame):
        if not self.borders:
            return
        for border_polygon in self.borders:
            if border_polygon.geom_type != "Polygon":
                continue
            pts = np.array(border_polygon.exterior.coords, np.int32)
            pts = pts.reshape((-1, 1, 2))
            cv2.polylines(frame, [pts], isClosed=True, color=(0, 255, 255), thickness=2)

    def box_to_polygon(self, x1, y1, x2, y2):
        from shapely.geometry import box

        return box(x1, y1, x2, y2)

    def check_conditions(self, percentage, overlap_detected, current_time, frame_resized):
        pass

    def capture_and_send(self, frame_resized, percentage, current_time):
        cvzone.putTextRect(frame_resized, f"Overlap: {percentage:.2f}%", (10, 50), scale=1, thickness=2, offset=5)
        cvzone.putTextRect(frame_resized, f"FPS: {int(self.fps)}", (10, 75), scale=1, thickness=2, offset=5)
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        image_path = f"images/cleaned_area_{self.camera_name}_{timestamp_str}.jpg"
        os.makedirs(os.path.dirname(image_path), exist_ok=True)
        cv2.imwrite(image_path, frame_resized)
        self.send_to_server(percentage, image_path, self.timestamp_start)
        self.timestamp_start = None

    def send_to_server(self, percentage, image_path, timestamp_start, host="10.5.0.2"):
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
                raise ValueError(f"Invalid host: {host}")
            return user, password, database, port

        try:
            user, password, database, port = server_address(host)
            connection = pymysql.connect(host=host, user=user, password=password, database=database, port=port)
            cursor = connection.cursor()
            table = "empbro"
            category = "Menyapu Lantai"
            camera_name = self.camera_name
            timestamp_done = datetime.now()
            timestamp_done_str = timestamp_done.strftime("%Y-%m-%d %H:%M:%S")
            timestamp_start_str = timestamp_start.strftime("%Y-%m-%d %H:%M:%S") if timestamp_start else None
            with open(image_path, "rb") as file:
                binary_image = file.read()
            query = f"""
            INSERT INTO {table} (cam, category, timestamp_start, timestamp_done, percentage, image_done)
            VALUES (%s, %s, %s, %s, %s, %s)
            """
            cursor.execute(query, (camera_name, category, timestamp_start_str, timestamp_done_str, percentage, binary_image))
            connection.commit()
            print(f"B`{self.camera_name} : Broom data successfully sent to server.")
        except pymysql.MySQLError as e:
            print(f"B`{self.camera_name} : Error sending broom data to server: {e}")
        finally:
            if "cursor" in locals():
                cursor.close()
            if "connection" in locals():
                connection.close()

    def main(self):
        process_every_n_frames = 2
        frame_count = 0
        if self.display:
            window_name = f"BROOM : {self.camera_name}"
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(window_name, self.window_width, self.window_height)
        if self.is_local_file:
            cap = cv2.VideoCapture(self.rtsp_url)
            frame_delay = int(1000 / self.video_fps)
            while cap.isOpened():
                start_time = time.time()
                ret, frame = cap.read()
                if not ret:
                    print(f"B`{self.camera_name} : End of video file or cannot read the frame. Restarting...")
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                frame_count += 1
                if frame_count % process_every_n_frames != 0:
                    continue
                current_time = time.time()
                time_diff = current_time - self.prev_frame_time
                self.fps = 1 / time_diff if time_diff > 0 else 0
                self.prev_frame_time = current_time
                frame_resized, overlap_detected = self.process_frame(frame, current_time)
                percentage = 0
                if self.display:
                    cvzone.putTextRect(frame_resized, f"FPS: {int(self.fps)}", (10, 75), scale=1, thickness=2, offset=5)
                self.check_conditions(percentage, overlap_detected, current_time, frame_resized)
                if self.display:
                    cv2.imshow(window_name, frame_resized)
                    processing_time = (time.time() - start_time) * 1000
                    adjusted_delay = max(int(frame_delay - processing_time), 1)
                    key = cv2.waitKey(adjusted_delay) & 0xFF
                    if key == ord("n"):
                        break
                else:
                    time.sleep(0.01)
            cap.release()
            if self.display:
                cv2.destroyAllWindows()
        else:
            self.frame_thread = threading.Thread(target=self.frame_capture)
            self.frame_thread.daemon = True
            self.frame_thread.start()
            while True:
                if self.stop_event.is_set():
                    break
                try:
                    frame = self.frame_queue.get(timeout=5)
                except queue.Empty:
                    continue
                frame_count += 1
                if frame_count % process_every_n_frames != 0:
                    continue
                current_time = time.time()
                time_diff = current_time - self.prev_frame_time
                self.fps = 1 / time_diff if time_diff > 0 else 0
                self.prev_frame_time = current_time
                frame_resized, overlap_detected = self.process_frame(frame, current_time)
                percentage = 0
                if self.display:
                    cvzone.putTextRect(frame_resized, f"FPS: {int(self.fps)}", (10, 75), scale=1, thickness=2, offset=5)
                self.check_conditions(percentage, overlap_detected, current_time, frame_resized)
                if self.display:
                    cv2.imshow(window_name, frame_resized)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("n"):
                        self.stop_event.set()
                        break
                else:
                    time.sleep(0.01)
            if self.display:
                cv2.destroyAllWindows()
            self.frame_thread.join()


def run_broom(camera_name, window_size=(320, 240), rtsp_url=None, display=True):
    detector = BroomDetector(
        camera_name=camera_name,
        rtsp_url=rtsp_url,
        window_size=window_size,
        display=display,
    )
    detector.main()


if __name__ == "__main__":
    run_broom(
        camera_name="CUTTING8",
        rtsp_url="videos/test/kon.mp4",
        display=True,
        window_size=(640, 480),
    )
