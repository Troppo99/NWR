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
import threading
import queue
import math
from shapely.geometry import Polygon, box
from shapely.ops import unary_union
from shapely.geometry import JOIN_STYLE


class CarpalDetector:
    def __init__(
        self,
        CARPAL_CONFIDENCE_THRESHOLD=0.5,
        carpal_model="models/yolo11l-pose.pt",
        camera_name=None,
        new_size=(960, 540),
        rtsp_url=None,
        window_size=(540, 360),
        display=False,
    ):
        self.CARPAL_CONFIDENCE_THRESHOLD = CARPAL_CONFIDENCE_THRESHOLD
        self.window_width, self.window_height = window_size
        self.new_width, self.new_height = new_size
        self.start_time = None
        self.end_time = None
        self.elapsed_time = None
        self.carpal_absence_timer_start = None
        self.prev_frame_time = time.time()
        self.fps = 0
        self.first_green_time = None
        self.is_counting = False
        self.camera_name = camera_name
        self.display = display
        self.rtsp_url = rtsp_url
        self.video_fps = None
        self.is_local_file = False
        self.borders, self.ip_camera = self.camera_config()
        if self.display is False:
            print(f"C`{self.camera_name} : >>>Display is disabled!<<<")
        if rtsp_url is not None:
            self.rtsp_url = rtsp_url
            if os.path.isfile(rtsp_url):
                self.is_local_file = True
                cap = cv2.VideoCapture(self.rtsp_url)
                self.video_fps = cap.get(cv2.CAP_PROP_FPS)
                if not self.video_fps or math.isnan(self.video_fps):
                    self.video_fps = 25
                cap.release()
                print(f"C`{self.camera_name} : Local video file detected. FPS: {self.video_fps}")
            else:
                self.is_local_file = False
                print(f"C`{self.camera_name} : RTSP stream detected. URL: {self.rtsp_url}")
                self.video_fps = None
        else:
            self.rtsp_url = f"rtsp://admin:oracle2015@{self.ip_camera}:554/Streaming/Channels/1"
            self.video_fps = None
            self.is_local_file = False
        self.show_text = True
        self.frame_queue = queue.Queue(maxsize=10)
        self.stop_event = threading.Event()
        self.frame_thread = None
        self.total_border_area = sum(border.area for border in self.borders) if self.borders else 0
        self.union_polygon = None
        self.total_area = 0
        self.last_overlap_time = time.time()
        self.area_cleared = False
        self.start_no_overlap_time_high = None
        self.start_no_overlap_time_low = None
        self.detection_paused = False
        self.detection_resume_time = None
        self.detection_pause_duration = 10
        self.timestamp_start = None
        self.carpal_model = YOLO(carpal_model).to("cuda")
        self.carpal_model.overrides["verbose"] = False
        print(f"Model Carpal device: {next(self.carpal_model.model.parameters()).device}")

    def camera_config(self):
        config = {
            "OFFICE1K": {
                "borders": [
                    [(24, 90), (137, 33), (233, -2), (250, -1), (243, 118), (63, 248), (30, 253), (24, 90)],
                ],
                "ip": "10.5.0.170",
            },
            "OFFICE2K": {
                "borders": [
                    [(65, 261), (-3, 305), (-2, 37), (66, -2), (299, -1), (301, 137), (229, 172), (223, 16), (196, 3), (47, 76), (65, 261)],
                ],
                "ip": "10.5.0.182",
            },
            "OFFICE3K": {
                "borders": [
                    [(1105, 154), (1206, 285), (1221, 245), (1242, 145), (1255, 44), (1201, 0), (1124, 0), (1105, 154)],
                    [(0, 193), (43, 371), (0, 476), (0, 193)],
                    [(10, 93), (71, 28), (78, 147), (100, 258), (59, 340), (44, 292), (19, 174), (10, 93)],
                ],
                "ip": "10.5.0.161",
            },
            "HALAMANDEPAN1K": {
                "borders": [
                    [(613, 192), (607, 287), (824, 320), (828, 207), (613, 192)],
                    [(854, 208), (1139, 243), (1131, 375), (851, 325), (854, 208)],
                    [(482, 193), (479, 270), (585, 284), (589, 192), (482, 193)],
                    [(1162, 250), (1277, 270), (1277, 404), (1156, 379), (1162, 250)],
                ],
                "ip": "10.5.0.236",
            },
            "EKSPEDISI1": {
                "borders": [[(429, 223), (616, 211), (882, 210), (884, 77), (785, 71), (786, 60), (628, 59), (427, 73), (429, 223)]],
                "ip": "10.5.0.155",
            },
            "HALAMANDEPAN1C": {
                "borders": [[(156, 329), (115, 322), (102, 280), (120, 244), (150, 226), (177, 242), (187, 263), (203, 260), (231, 257), (239, 280), (240, 319), (223, 345), (204, 354), (170, 340)]],
                "ip": "10.5.0.236",
            },
        }
        if self.camera_name not in config:
            raise ValueError(f"Camera name '{self.camera_name}' not found in configuration.")
        original_borders = config[self.camera_name]["borders"]
        ip = config[self.camera_name]["ip"]
        scaled_borders = []
        for border_group in original_borders:
            scaled_group = []
            for x, y in border_group:
                scaled_x = int(x * (self.new_width / 1280))
                scaled_y = int(y * (self.new_height / 720))
                scaled_group.append((scaled_x, scaled_y))
            if len(scaled_group) >= 3:
                polygon = Polygon(scaled_group)
                if polygon.is_valid:
                    scaled_borders.append(polygon)
                else:
                    print(f"C`{self.camera_name} : Invalid polygon for camera {self.camera_name}, skipping.")
            else:
                print(f"C`{self.camera_name} : Not enough points to form a polygon for camera {self.camera_name}, skipping.")
        return scaled_borders, ip

    def process_model(self, frame):
        with torch.no_grad():
            results = self.carpal_model(frame, stream=True, imgsz=960)
        return results

    def export_frame(self, results, color, pairs):
        points = []
        coords = []
        keypoint_positions = []
        confidence_threshold = self.CARPAL_CONFIDENCE_THRESHOLD
        for result in results:
            keypoints_data = result.keypoints
            if keypoints_data is not None and keypoints_data.xy is not None and keypoints_data.conf is not None:
                if keypoints_data.shape[0] > 0:
                    keypoints_array = keypoints_data.xy.cpu().numpy()
                    keypoints_conf = keypoints_data.conf.cpu().numpy()
                    for keypoints_per_object, keypoints_conf_per_object in zip(keypoints_array, keypoints_conf):
                        keypoints_list = []
                        for kp, kp_conf in zip(keypoints_per_object, keypoints_conf_per_object):
                            if kp_conf >= confidence_threshold:
                                x, y = kp[0], kp[1]
                                keypoints_list.append((int(x), int(y)))
                            else:
                                keypoints_list.append(None)
                        if len(keypoints_list) > 9 and keypoints_list[7] and keypoints_list[9]:
                            kp7 = keypoints_list[7]
                            kp9 = keypoints_list[9]
                            vx = kp9[0] - kp7[0]
                            vy = kp9[1] - kp7[1]
                            norm = (vx**2 + vy**2) ** 0.5
                            if norm != 0:
                                vx /= norm
                                vy /= norm
                                extension_length = 20
                                x_new = int(kp9[0] + vx * extension_length)
                                y_new = int(kp9[1] + vy * extension_length)
                                keypoints_list[9] = (x_new, y_new)
                        if len(keypoints_list) > 10 and keypoints_list[8] and keypoints_list[10]:
                            kp8 = keypoints_list[8]
                            kp10 = keypoints_list[10]
                            vx = kp10[0] - kp8[0]
                            vy = kp10[1] - kp8[1]
                            norm = (vx**2 + vy**2) ** 0.5
                            if norm != 0:
                                vx /= norm
                                vy /= norm
                                extension_length = 20
                                x_new = int(kp10[0] + vx * extension_length)
                                y_new = int(kp10[1] + vy * extension_length)
                                keypoints_list[10] = (x_new, y_new)
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

    def keypoint_to_polygon(self, x, y, size=5):
        x1 = x - size
        y1 = y - size
        x2 = x + size
        y2 = y + size
        return box(x1, y1, x2, y2)

    def update_union_polygon(self, new_polygons):
        if new_polygons:
            if self.union_polygon is None:
                self.union_polygon = unary_union(new_polygons)
            else:
                self.union_polygon = unary_union([self.union_polygon] + new_polygons)
            self.union_polygon = self.union_polygon.simplify(tolerance=0.5, preserve_topology=True)
            if not self.union_polygon.is_valid:
                self.union_polygon = self.union_polygon.buffer(0, join_style=JOIN_STYLE.mitre)
            self.total_area = self.union_polygon.area

    def draw_segments(self, frame):
        overlay = frame.copy()
        if self.union_polygon is not None and not self.union_polygon.is_empty:
            if self.union_polygon.geom_type == "Polygon":
                coords = np.array(self.union_polygon.exterior.coords, np.int32)
                coords = coords.reshape((-1, 1, 2))
                cv2.fillPoly(overlay, [coords], (0, 255, 0))
            elif self.union_polygon.geom_type == "MultiPolygon":
                for poly in self.union_polygon.geoms:
                    if poly.is_empty:
                        continue
                    coords = np.array(poly.exterior.coords, np.int32)
                    coords = coords.reshape((-1, 1, 2))
                    cv2.fillPoly(overlay, [coords], (0, 255, 0))
        cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

    def draw_borders(self, frame):
        if not self.borders:
            return
        for border_polygon in self.borders:
            if border_polygon.geom_type != "Polygon":
                continue
            pts = np.array(border_polygon.exterior.coords, np.int32)
            pts = pts.reshape((-1, 1, 2))
            cv2.polylines(frame, [pts], isClosed=True, color=(0, 255, 255), thickness=2)

    def frame_capture(self):
        rtsp_url = self.rtsp_url
        while not self.stop_event.is_set():
            cap = cv2.VideoCapture(rtsp_url)
            if not cap.isOpened():
                print(f"C`{self.camera_name} : Failed to open stream. Retrying in 5 seconds...")
                cap.release()
                time.sleep(5)
                continue
            while not self.stop_event.is_set():
                ret, frame = cap.read()
                if not ret:
                    print(f"C`{self.camera_name} : Failed to read frame. Reconnecting in 5 seconds...")
                    cap.release()
                    time.sleep(5)
                    break
                try:
                    self.frame_queue.put(frame, block=False)
                except queue.Full:
                    pass
            cap.release()

    def process_frame(self, frame, current_time, pairs_human):
        frame_resized = cv2.resize(frame, (self.new_width, self.new_height))
        if self.detection_paused:
            if current_time >= self.detection_resume_time:
                self.detection_paused = False
                print(f"C`{self.camera_name} : Resuming detection after {self.detection_pause_duration}-second pause.")
            else:
                self.draw_borders(frame_resized)
                return frame_resized
        results = self.process_model(frame_resized)
        points, coords, keypoint_positions = self.export_frame(results, (0, 255, 0), pairs_human)
        new_polygons = []
        overlap_detected = False
        for keypoints_list in keypoint_positions:
            for idx in [9, 10]:
                if idx < len(keypoints_list):
                    kp = keypoints_list[idx]
                    if kp is not None:
                        kp_x, kp_y = kp
                        kp_polygon = self.keypoint_to_polygon(kp_x, kp_y, size=5)
                        for border_polygon in self.borders:
                            if kp_polygon.intersects(border_polygon):
                                intersection = kp_polygon.intersection(border_polygon)
                                if not intersection.is_empty:
                                    overlap_detected = True
                                    new_polygons.append(intersection)
        if overlap_detected and self.timestamp_start is None:
            self.timestamp_start = datetime.now()
        self.update_union_polygon(new_polygons)
        percentage = (self.total_area / self.total_border_area) * 100 if self.total_border_area > 0 else 0
        self.draw_segments(frame_resized)
        self.draw_borders(frame_resized)
        if self.display:
            if keypoint_positions:
                for x, y, color in coords:
                    cv2.line(frame_resized, x, y, color, 2)
                for keypoints_list in keypoint_positions:
                    for idx, point in enumerate(keypoints_list):
                        if point is not None:
                            if idx == 9 or idx == 10:
                                radius = 10
                            else:
                                radius = 3
                            cv2.circle(frame_resized, point, radius, (0, 255, 255), -1)
        cvzone.putTextRect(
            frame_resized,
            f"Percentage of Overlap: {percentage:.2f}%",
            (10, self.new_height - 50),
            scale=1,
            thickness=2,
            offset=5,
        )
        cvzone.putTextRect(frame_resized, f"FPS: {int(self.fps)}", (10, self.new_height - 75), scale=1, thickness=2, offset=5)
        self.check_conditions(percentage, overlap_detected, current_time, frame_resized)
        return frame_resized

    def check_conditions(self, percentage, overlap_detected, current_time, frame_resized):
        if percentage >= 90:
            self.union_polygon = None
            self.total_area = 0
            print(f"C`{self.camera_name} : Percentage >= 90%")
            self.capture_and_send(frame_resized, percentage, current_time)
            self.timestamp_start = None
            self.detection_paused = True
            self.detection_resume_time = current_time + self.detection_pause_duration
            self.start_no_overlap_time_high = None
            self.start_no_overlap_time_low = None
            return
        elif percentage >= 50:
            if not overlap_detected:
                if self.start_no_overlap_time_low is None:
                    self.start_no_overlap_time_low = current_time
                elif current_time - self.start_no_overlap_time_low >= 60:
                    self.union_polygon = None
                    self.total_area = 0
                    print(f"C`{self.camera_name} : Percentage >= 50%")
                    self.capture_and_send(frame_resized, percentage, current_time)
                    self.start_no_overlap_time_low = None
            else:
                self.start_no_overlap_time_low = None
        elif percentage >= 5:
            if not overlap_detected:
                if self.start_no_overlap_time_low is None:
                    self.start_no_overlap_time_low = current_time
                elif current_time - self.start_no_overlap_time_low >= 30:
                    self.union_polygon = None
                    self.total_area = 0
                    print(f"C`{self.camera_name} : Percentage >= 5%")
                    self.start_no_overlap_time_low = None
            else:
                self.start_no_overlap_time_low = None
        else:
            if not overlap_detected:
                if self.start_no_overlap_time_low is None:
                    self.start_no_overlap_time_low = current_time
                elif current_time - self.start_no_overlap_time_low >= 5:
                    self.union_polygon = None
                    self.total_area = 0
                    print(f"C`{self.camera_name} : Percentage < 5%")
                    self.start_no_overlap_time_low = None
            else:
                self.start_no_overlap_time_low = None
        if self.detection_paused:
            if current_time >= self.detection_resume_time:
                self.detection_paused = False
                print(f"C`{self.camera_name} : Resuming detection after {self.detection_pause_duration}-second pause.")

    def capture_and_send(self, frame_resized, percentage, current_time):
        cvzone.putTextRect(frame_resized, f"Overlap: {percentage:.2f}%", (10, 50), scale=1, thickness=2, offset=5)
        cvzone.putTextRect(frame_resized, f"FPS: {int(self.fps)}", (10, 75), scale=1, thickness=2, offset=5)
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        image_path = f"images/carpal_cleaned_{self.camera_name}_{timestamp_str}.jpg"
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
            category = "Mengelap Kaca"
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
            print(f"C`{self.camera_name} : Carpal data successfully sent to server.")
        except pymysql.MySQLError as e:
            print(f"C`{self.camera_name} : Error sending carpal data to server: {e}")
        finally:
            if "cursor" in locals():
                cursor.close()
            if "connection" in locals():
                connection.close()

    def main(self):
        pairs_human = [
            (0, 1),
            (0, 2),
            (1, 2),
            (2, 4),
            (1, 3),
            (4, 6),
            (3, 5),
            (5, 6),
            (6, 8),
            (8, 10),
            (5, 7),
            (7, 9),
            (6, 12),
            (12, 11),
            (11, 5),
            (12, 14),
            (14, 16),
            (11, 13),
            (13, 15),
        ]
        process_every_n_frames = 2
        frame_count = 0
        if self.display:
            window_name = f"CARPAL : {self.camera_name}"
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(window_name, self.window_width, self.window_height)
        if self.is_local_file:
            cap = cv2.VideoCapture(self.rtsp_url)
            frame_delay = int(1000 / self.video_fps)
            while cap.isOpened():
                start_time = time.time()
                ret, frame = cap.read()
                if not ret:
                    print(f"C`{self.camera_name} : End of video file or cannot read the frame.")
                    break
                frame_count += 1
                if frame_count % process_every_n_frames != 0:
                    continue
                current_time = time.time()
                time_diff = current_time - self.prev_frame_time
                if time_diff > 0:
                    self.fps = 1 / time_diff
                else:
                    self.fps = 0
                self.prev_frame_time = current_time
                frame_resized = self.process_frame(frame, current_time, pairs_human)
                if self.display:
                    cv2.imshow(window_name, frame_resized)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("n"):
                        break
                    elif key == ord("s"):
                        self.show_text = not self.show_text
                else:
                    time.sleep(0.01)
                processing_time = (time.time() - start_time) * 1000
                adjusted_delay = max(int(frame_delay - processing_time), 1)
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
                if time_diff > 0:
                    self.fps = 1 / time_diff
                else:
                    self.fps = 0
                self.prev_frame_time = current_time
                frame_resized = self.process_frame(frame, current_time, pairs_human)
                if self.display:
                    cv2.imshow(window_name, frame_resized)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("n"):
                        self.stop_event.set()
                        break
                    elif key == ord("s"):
                        self.show_text = not self.show_text
                else:
                    time.sleep(0.01)
            if self.display:
                cv2.destroyAllWindows()
            self.frame_thread.join()


def run_carpal(
    camera_name,
    window_size=(320, 240),
    rtsp_url=None,
    display=True,
):
    detector = CarpalDetector(
        camera_name=camera_name,
        window_size=window_size,
        rtsp_url=rtsp_url,
        display=display,
    )
    detector.main()


if __name__ == "__main__":
    run_carpal(
        camera_name="HALAMANDEPAN1C",
        display=True,
    )
