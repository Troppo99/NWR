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
from shapely.geometry import Polygon, box, Point
from shapely.ops import unary_union
from shapely.geometry import JOIN_STYLE


class BaseDetector:
    def __init__(
        self,
        camera_name,
        new_size=(960, 540),
        rtsp_url=None,
        window_size=(540, 360),
        display=False,
    ):
        self.window_width, self.window_height = window_size
        self.new_width, self.new_height = new_size
        self.prev_frame_time = time.time()
        self.fps = 0
        self.camera_name = camera_name
        self.display = display
        self.rtsp_url = rtsp_url
        self.video_fps = None
        self.is_local_file = False
        self.show_text = True
        self.frame_queue = queue.Queue(maxsize=10)
        self.stop_event = threading.Event()
        self.frame_thread = None
        self.detection_paused = False
        self.detection_resume_time = None

        # Initialize union polygon for accumulated overlapping areas
        self.union_polygon = None  # Initialize the union polygon
        self.total_area = 0  # Accumulated area

        # Variables for tracking the conditions
        self.last_overlap_time = time.time()
        self.area_cleared = False

        # Initialize time trackers
        self.start_no_overlap_time_high = None
        self.start_no_overlap_time_low = None

        if self.display is False:
            print(f">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>\n\tDisplay is not running!\n<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")

        self.borders, self.ip_camera = self.camera_config()
        self.setup_rtsp()

        # Compute total border area
        self.total_border_area = sum(border.area for border in self.borders) if self.borders else 0

    def setup_rtsp(self):
        if self.rtsp_url is not None:
            self.rtsp_url = self.rtsp_url
            if os.path.isfile(self.rtsp_url):
                self.is_local_file = True
                cap = cv2.VideoCapture(self.rtsp_url)
                self.video_fps = cap.get(cv2.CAP_PROP_FPS)
                if not self.video_fps or math.isnan(self.video_fps):
                    self.video_fps = 25
                cap.release()
                print(f"Local video file detected. FPS: {self.video_fps}")
            else:
                self.is_local_file = False
                print(f"RTSP stream detected. URL: {self.rtsp_url}")
                self.video_fps = None
        else:
            self.rtsp_url = f"rtsp://admin:oracle2015@{self.ip_camera}:554/Streaming/Channels/1"
            self.video_fps = None
            self.is_local_file = False

    def frame_capture(self):
        rtsp_url = self.rtsp_url
        while not self.stop_event.is_set():
            cap = cv2.VideoCapture(rtsp_url)
            if not cap.isOpened():
                print("Failed to open stream. Retrying in 5 seconds...")
                cap.release()
                time.sleep(5)
                continue

            while not self.stop_event.is_set():
                ret, frame = cap.read()
                if not ret:
                    print("Stream failed to read. Ensure the URL is correct.")
                    cap.release()
                    time.sleep(5)
                    break
                try:
                    self.frame_queue.put(frame, block=False)
                except queue.Full:
                    pass
            cap.release()

    def camera_config(self):
        # This method should be implemented in child classes
        raise NotImplementedError

    def process_model(self, frame):
        # This method should be implemented in child classes
        raise NotImplementedError

    def process_frame(self, frame, current_time):
        # This method should be implemented in child classes
        raise NotImplementedError

    def check_conditions(self, percentage, overlap_detected, current_time, frame_resized):
        # This method should be implemented in child classes
        raise NotImplementedError

    def capture_and_send(self, frame_resized, percentage, current_time):
        # This method should be implemented in child classes
        raise NotImplementedError

    def send_to_server(self, percentage, image_path, host="10.5.0.2"):
        # This method can be shared
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
            table = "empbro"  # Replace with your table name
            category = self.category
            camera_name = self.camera_name
            timestamp_done = datetime.now()
            timestamp_done_str = timestamp_done.strftime("%Y-%m-%d %H:%M:%S")

            with open(image_path, "rb") as file:
                binary_image = file.read()

            query = f"""
            INSERT INTO {table} (cam, category, timestamp_done, percentage, image_done)
            VALUES (%s, %s, %s, %s, %s)
            """
            cursor.execute(query, (camera_name, category, timestamp_done_str, percentage, binary_image))
            connection.commit()
            print(f"{self.category} data successfully sent to server.")
        except pymysql.MySQLError as e:
            print(f"Error sending {self.category} data to server: {e}")
        finally:
            if "cursor" in locals():
                cursor.close()
            if "connection" in locals():
                connection.close()

    def main(self):
        # This method should be implemented in child classes
        raise NotImplementedError


class BroomDetector(BaseDetector):
    def __init__(
        self,
        camera_name,
        rtsp_url=None,
        window_size=(540, 360),
        display=False,
        broom_model="broom_model.pt",
        new_size=(960, 540),
        BROOM_CONFIDENCE_THRESHOLD=0.5,
    ):
        self.BROOM_CONFIDENCE_THRESHOLD = BROOM_CONFIDENCE_THRESHOLD
        super().__init__(camera_name, new_size, rtsp_url, window_size, display)
        self.category = "Menyapu Lantai"
        self.model = YOLO(broom_model).to("cuda")
        self.model.overrides["verbose"] = False
        print(f"Model Broom device: {next(self.model.model.parameters()).device}")

        # Variables specific to BroomDetector
        self.pairs = [
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 0),
        ]  # Adjust as needed for your model

    def camera_config(self):
        config = {
            "OFFICE1": {
                "borders": [
                    [(100, 100), (200, 100), (200, 200), (100, 200), (100, 100)],
                ],
                "ip": "10.5.0.170",
            },
            # Add configurations for other cameras
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
                    print(f"Invalid polygon for camera {self.camera_name}, skipping.")
            else:
                print(f"Not enough points to form a polygon for camera {self.camera_name}, skipping.")

        return scaled_borders, ip

    def process_model(self, frame):
        with torch.no_grad():
            results = self.model(frame, stream=True)
        return results

    def export_frame(self, results, color):
        points = []
        coords = []
        confidence_threshold = self.BROOM_CONFIDENCE_THRESHOLD

        for result in results:
            boxes = result.boxes
            if boxes is not None and boxes.xyxy is not None and boxes.conf is not None:
                for box in boxes:
                    conf = box.conf.cpu().numpy()[0]
                    if conf >= confidence_threshold:
                        x1, y1, x2, y2 = box.xyxy.cpu().numpy()[0]
                        points.append((int((x1 + x2) / 2), int((y1 + y2) / 2)))
                        coords.append(((int(x1), int(y1)), (int(x2), int(y2)), color))
        return points, coords

    def keypoint_to_polygon(self, x1, y1, x2, y2):
        return box(x1, y1, x2, y2)

    def update_union_polygon(self, new_polygons):
        if new_polygons:
            if self.union_polygon is None:
                self.union_polygon = unary_union(new_polygons)
            else:
                self.union_polygon = unary_union([self.union_polygon] + new_polygons)
            # Simplify the union polygon to reduce complexity
            self.union_polygon = self.union_polygon.simplify(tolerance=0.5, preserve_topology=True)
            # Ensure the polygon remains valid
            if not self.union_polygon.is_valid:
                self.union_polygon = self.union_polygon.buffer(0, join_style=JOIN_STYLE.mitre)
            self.total_area = self.union_polygon.area

    def draw_segments(self, frame):
        overlay = frame.copy()
        # Draw the union polygon
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

    def process_frame(self, frame, current_time):
        frame_resized = cv2.resize(frame, (self.new_width, self.new_height))
        results = self.process_model(frame_resized)
        points, coords = self.export_frame(results, (0, 255, 0))

        new_polygons = []
        overlap_detected = False

        for coord in coords:
            (x1, y1), (x2, y2), color = coord
            kp_polygon = self.keypoint_to_polygon(x1, y1, x2, y2)
            for border_polygon in self.borders:
                if kp_polygon.intersects(border_polygon):
                    intersection = kp_polygon.intersection(border_polygon)
                    if not intersection.is_empty:
                        overlap_detected = True
                        new_polygons.append(intersection)

        self.update_union_polygon(new_polygons)

        percentage = (self.total_area / self.total_border_area) * 100 if self.total_border_area > 0 else 0

        self.draw_segments(frame_resized)
        self.draw_borders(frame_resized)

        # Draw bounding boxes
        if self.display:
            for coord in coords:
                (x1, y1), (x2, y2), color = coord
                cv2.rectangle(frame_resized, (x1, y1), (x2, y2), color, 2)

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
        # Implement conditions similar to CarpalDetector if needed
        pass  # Add your condition checks here

    def capture_and_send(self, frame_resized, percentage, current_time):
        # Implement capture and send functionality
        pass  # Add your capture and send code here

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
                    print("End of video file or cannot read the frame.")
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

                frame_resized = self.process_frame(frame, current_time)
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

                frame_resized = self.process_frame(frame, current_time)
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


class CarpalDetector(BaseDetector):
    def __init__(
        self,
        CARPAL_ABSENCE_THRESHOLD=10,
        CARPAL_TOUCH_THRESHOLD=0,
        CARPAL_PERCENTAGE_GREEN_THRESHOLD=50,
        CARPAL_CONFIDENCE_THRESHOLD=0.5,
        carpal_model="yolo11l-pose.pt",
        camera_name=None,
        new_size=(960, 540),
        rtsp_url=None,
        window_size=(540, 360),
        display=False,
    ):
        self.CARPAL_CONFIDENCE_THRESHOLD = CARPAL_CONFIDENCE_THRESHOLD
        self.CARPAL_ABSENCE_THRESHOLD = CARPAL_ABSENCE_THRESHOLD
        self.CARPAL_TOUCH_THRESHOLD = CARPAL_TOUCH_THRESHOLD
        self.CARPAL_PERCENTAGE_GREEN_THRESHOLD = CARPAL_PERCENTAGE_GREEN_THRESHOLD
        super().__init__(camera_name, new_size, rtsp_url, window_size, display)
        self.category = "Mengelap Kaca"
        self.model = YOLO(carpal_model).to("cuda")
        self.model.overrides["verbose"] = False
        print(f"Model Carpal device: {next(self.model.model.parameters()).device}")

        # Variables specific to CarpalDetector
        self.pairs_human = [
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

    def camera_config(self):
        config = {
            "OFFICE1": {
                "borders": [
                    [(24, 90), (137, 33), (233, -2), (250, -1), (243, 118), (63, 248), (30, 253), (24, 90)],
                ],
                "ip": "10.5.0.170",
            },
            # Add configurations for other cameras
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
                    print(f"Invalid polygon for camera {self.camera_name}, skipping.")
            else:
                print(f"Not enough points to form a polygon for camera {self.camera_name}, skipping.")

        return scaled_borders, ip

    def process_model(self, frame):
        with torch.no_grad():
            results = self.model(frame, stream=True, imgsz=960)
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

                        # Extend points 9 and 10
                        # Point 9
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
                        # Point 10
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

    def keypoint_to_polygon(self, x, y, size=10):
        # Create a circular polygon (buffer around the point)
        return Point(x, y).buffer(size)

    def update_union_polygon(self, new_polygons):
        if new_polygons:
            if self.union_polygon is None:
                self.union_polygon = unary_union(new_polygons)
            else:
                self.union_polygon = unary_union([self.union_polygon] + new_polygons)
            # Simplify the union polygon to reduce complexity
            self.union_polygon = self.union_polygon.simplify(tolerance=0.5, preserve_topology=True)
            # Ensure the polygon remains valid
            if not self.union_polygon.is_valid:
                self.union_polygon = self.union_polygon.buffer(0, join_style=JOIN_STYLE.mitre)
            self.total_area = self.union_polygon.area

    def draw_segments(self, frame):
        overlay = frame.copy()
        # Draw the union polygon
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

    def process_frame(self, frame, current_time):
        frame_resized = cv2.resize(frame, (self.new_width, self.new_height))
        results = self.process_model(frame_resized)
        points, coords, keypoint_positions = self.export_frame(results, (0, 255, 0), self.pairs_human)

        new_polygons = []
        overlap_detected = False

        for keypoints_list in keypoint_positions:
            for idx in [9, 10]:
                if idx < len(keypoints_list):
                    kp = keypoints_list[idx]
                    if kp is not None:
                        kp_x, kp_y = kp
                        # Create a circular polygon around the keypoint
                        kp_polygon = self.keypoint_to_polygon(kp_x, kp_y, size=10)  # Adjust size as needed
                        for border_polygon in self.borders:
                            if kp_polygon.intersects(border_polygon):
                                intersection = kp_polygon.intersection(border_polygon)
                                if not intersection.is_empty:
                                    overlap_detected = True
                                    new_polygons.append(intersection)

        self.update_union_polygon(new_polygons)

        percentage = (self.total_area / self.total_border_area) * 100 if self.total_border_area > 0 else 0

        self.draw_segments(frame_resized)
        self.draw_borders(frame_resized)

        # Draw keypoints and lines
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
        # New condition: If percentage >= 90%, reset immediately and pause detection
        if percentage >= 90:
            # Reset polygons and print message
            self.union_polygon = None
            self.total_area = 0
            print("All borders covered - Percentage >= 90%")
            # Capture frame and send data
            self.capture_and_send(frame_resized, percentage, current_time)
            # Pause detection for 10 seconds
            self.detection_paused = True
            self.detection_resume_time = current_time + 10  # 10-second pause
            # Reset time trackers
            self.start_no_overlap_time_high = None
            self.start_no_overlap_time_low = None
            return  # Skip further checks

        # Condition when overlap percentage >= 80%
        if percentage >= 80:
            if not overlap_detected:
                if self.start_no_overlap_time_high is None:
                    self.start_no_overlap_time_high = current_time
                elif current_time - self.start_no_overlap_time_high >= 30:
                    # Reset polygons and print message
                    self.union_polygon = None
                    self.total_area = 0
                    print("High percentage area cleaned")
                    # Capture frame and send data
                    self.capture_and_send(frame_resized, percentage, current_time)
                    self.start_no_overlap_time_high = None
            else:
                self.start_no_overlap_time_high = None

        # Condition when overlap percentage >= 50%
        elif percentage >= 50:
            if not overlap_detected:
                if self.start_no_overlap_time_low is None:
                    self.start_no_overlap_time_low = current_time
                elif current_time - self.start_no_overlap_time_low >= 20:
                    # Reset polygons
                    self.union_polygon = None
                    self.total_area = 0
                    print("Medium percentage area cleaned")
                    self.start_no_overlap_time_low = None
            else:
                self.start_no_overlap_time_low = None

        # Condition when overlap percentage >= 10%
        elif percentage >= 10:
            if not overlap_detected:
                if self.start_no_overlap_time_low is None:
                    self.start_no_overlap_time_low = current_time
                elif current_time - self.start_no_overlap_time_low >= 10:
                    # Reset polygons
                    self.union_polygon = None
                    self.total_area = 0
                    print("Low percentage area cleaned")
                    self.start_no_overlap_time_low = None
            else:
                self.start_no_overlap_time_low = None

        # Condition when overlap percentage < 10%
        else:
            if not overlap_detected:
                if self.start_no_overlap_time_low is None:
                    self.start_no_overlap_time_low = current_time
                elif current_time - self.start_no_overlap_time_low >= 3:
                    # Reset polygons
                    self.union_polygon = None
                    self.total_area = 0
                    print("Very low percentage area cleaned")
                    self.start_no_overlap_time_low = None
            else:
                self.start_no_overlap_time_low = None

        # Handle detection pause
        if self.detection_paused:
            if current_time >= self.detection_resume_time:
                self.detection_paused = False
                print("Resuming detection after 10-second pause.")

    def capture_and_send(self, frame_resized, percentage, current_time):
        # Add text overlays before saving
        cvzone.putTextRect(frame_resized, f"Overlap: {percentage:.2f}%", (10, 50), scale=1, thickness=2, offset=5)
        cvzone.putTextRect(frame_resized, f"FPS: {int(self.fps)}", (10, 75), scale=1, thickness=2, offset=5)

        # Save the frame to an image file
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        image_path = f"images/carpal_cleaned_{self.camera_name}_{timestamp_str}.jpg"
        os.makedirs(os.path.dirname(image_path), exist_ok=True)
        cv2.imwrite(image_path, frame_resized)
        # Send data to server
        self.send_to_server(percentage, image_path)

    def main(self):
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
                    print("End of video file or cannot read the frame.")
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

                frame_resized = self.process_frame(frame, current_time)
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

                frame_resized = self.process_frame(frame, current_time)
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


def run_broom(
    camera_name,
    window_size=(320, 240),
    rtsp_url=None,
    display=True,
):
    detector = BroomDetector(
        camera_name=camera_name,
        window_size=window_size,
        rtsp_url=rtsp_url,
        display=display,
    )
    detector.main()


def run_carpal(
    camera_name,
    CARPAL_ABSENCE_THRESHOLD=30,
    CARPAL_TOUCH_THRESHOLD=0,
    CARPAL_PERCENTAGE_GREEN_THRESHOLD=50,
    window_size=(320, 240),
    rtsp_url=None,
    display=True,
):
    detector = CarpalDetector(
        CARPAL_ABSENCE_THRESHOLD=CARPAL_ABSENCE_THRESHOLD,
        CARPAL_TOUCH_THRESHOLD=CARPAL_TOUCH_THRESHOLD,
        CARPAL_PERCENTAGE_GREEN_THRESHOLD=CARPAL_PERCENTAGE_GREEN_THRESHOLD,
        camera_name=camera_name,
        window_size=window_size,
        rtsp_url=rtsp_url,
        display=display,
    )
    detector.main()


# if __name__ == "__main__":