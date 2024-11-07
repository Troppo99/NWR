import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import cv2
from ultralytics import YOLO
import torch
import cvzone
import time
import threading
import queue
import math
import numpy as np
from shapely.geometry import Polygon, box
from shapely.ops import unary_union
from shapely.geometry import JOIN_STYLE
import pymysql
from datetime import datetime, timedelta


class BroomDetector:
    def __init__(
        self,
        BROOM_CONFIDENCE_THRESHOLD=0.5,
        rtsp_url=None,
        camera_name=None,
        window_size=(540, 360),
    ):
        self.BROOM_CONFIDENCE_THRESHOLD = BROOM_CONFIDENCE_THRESHOLD
        self.window_width, self.window_height = window_size
        self.new_width, self.new_height = (960, 540)
        self.prev_frame_time = 0
        self.fps = 0
        self.union_polygon = None  # Initialize the union polygon
        self.total_area = 0
        self.camera_name = camera_name
        self.borders, self.ip_camera = self.camera_config()
        self.total_border_area = sum(border.area for border in self.borders) if self.borders else 0
        if rtsp_url is not None:
            self.rtsp_url = rtsp_url
            if os.path.isfile(rtsp_url):
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

        self.frame_queue = queue.Queue(maxsize=10)
        self.stop_event = threading.Event()
        self.frame_thread = None

        self.broom_model = YOLO("broom6l.pt").to("cuda")
        self.broom_model.overrides["verbose"] = False
        print(f"Model Broom device: {next(self.broom_model.model.parameters()).device}")

        # Variables for tracking the conditions
        self.last_overlap_time = time.time()
        self.area_cleared = False

        # Initialize time trackers
        self.start_no_overlap_time_high = None
        self.start_no_overlap_time_low = None

        # Variables for detection pause
        self.detection_paused = False
        self.detection_resume_time = None

    def camera_config(self):
        config = {
            # OFFICE
            "OFFICE1": {
                "borders": [[(688, 93), (791, 96), (845, 275), (872, 383), (905, 536), (933, 712), (591, 712), (633, 443), (662, 254), (688, 93)]],
                "ip": "10.5.0.170",
            },
            "OFFICE2": {
                "borders": [[(24, 496), (107, 442), (134, 492), (264, 416), (250, 358), (503, 232), (633, 309), (783, 233), (1028, 369), (1073, 328), (1244, 442), (1207, 541), (1153, 642), (1105, 718), (319, 718), (179, 538), (71, 603), (24, 496)]],
                "ip": "10.5.0.182",
            },
            "OFFICE3": {
                "borders": [[(160, 411), (127, 572), (109, 716), (601, 716), (590, 462), (1121, 418), (1163, 248), (1030, 94), (936, 3), (500, 3), (484, 176), (476, 403), (160, 411)]],
                "ip": "10.5.0.161",
            },
            # LINE
            "SEWING1": {
                "borders": [[(397, 715), (991, 715), (961, 555), (908, 348), (852, 186), (793, 43), (777, 2), (608, 1), (568, 110), (569, 180), (560, 194), (536, 195), (445, 493), (397, 715)]],
                "ip": "10.5.0.201",
            },
            "SEWING2": {
                "borders": [[(623, 60), (698, 60), (733, 202), (786, 494), (914, 486), (943, 581), (803, 597), (818, 714), (569, 714), (575, 607), (430, 605), (448, 504), (581, 505), (623, 60)]],
                "ip": "10.5.0.217",
            },
            "SEWING3": {
                "borders": [[(561, 76), (652, 79), (673, 478), (1002, 479), (1273, 472), (1276, 544), (1006, 577), (679, 584), (683, 714), (433, 714), (454, 567), (381, 560), (407, 459), (471, 464), (512, 251), (561, 76)]],
                "ip": "10.5.0.214",
            },
            "SEWING4": {
                "borders": [[(584, 40), (672, 41), (686, 128), (683, 210), (706, 211), (739, 406), (776, 718), (480, 718), (508, 466), (559, 170), (584, 40)]],
                "ip": "10.5.0.146",
            },
            "SEWING5": {
                "borders": [[(628, 107), (698, 107), (759, 442), (873, 436), (905, 555), (872, 562), (878, 586), (780, 598), (795, 712), (582, 712), (586, 601), (526, 601), (549, 452), (595, 452), (614, 250), (628, 107)]],
                "ip": "10.5.0.252",
            },
            "SEWING6": {
                "borders": [[(631, 205), (696, 206), (724, 360), (756, 569), (775, 710), (567, 710), (625, 225), (631, 205)]],
                "ip": "10.5.0.159",
            },
            "SEWING8": {
                "borders": [[(115, 540), (51, 638), (166, 665), (340, 691), (596, 704), (804, 691), (879, 681), (841, 539), (781, 544), (698, 174), (629, 174), (605, 553), (405, 559), (115, 540)]],
                "ip": "10.5.0.219",
            },
            "SEWINGBACK1": {
                "borders": [[(726, 312), (831, 316), (835, 335), (835, 335), (846, 334), (935, 581), (974, 718), (504, 718), (497, 659), (291, 617), (248, 676), (88, 627), (168, 544), (363, 585), (560, 612), (711, 622), (726, 312)]],
                "ip": "10.5.0.183",
            },
            "SEWINGBACK2": {
                "borders": [[(663, 258), (815, 255), (873, 431), (915, 575), (951, 718), (536, 719), (663, 258)]],
                "ip": "10.5.0.195",
            },
            "LINEMANUAL10": {
                "borders": [[(555, 235), (624, 235), (697, 716), (504, 717), (526, 451), (555, 235)]],
                "ip": "10.5.0.205",
            },
            "LINEMANUAL14": {
                "borders": [[(634, 145), (722, 147), (780, 371), (850, 717), (526, 717), (587, 367), (634, 145)]],
                "ip": "10.5.0.202",
            },
            "LINEMANUAL15": {
                "borders": [[(93, 517), (189, 533), (163, 579), (308, 600), (495, 616), (782, 614), (802, 718), (233, 718), (101, 684), (62, 560), (93, 517)]],
                "ip": "10.5.0.207",
            },
            # GUDANG
            "GUDANGACC1": {
                "borders": [[(698, 95), (819, 126), (864, 281), (902, 451), (926, 594), (938, 716), (680, 716), (695, 368), (698, 95)]],
                "ip": "10.5.0.110",
            },
            "GUDANGACC4": {
                "borders": [[(512, 94), (735, 84), (811, 287), (876, 492), (930, 715), (318, 714), (368, 500), (427, 316), (512, 94)]],
                "ip": "10.5.0.180",
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
                scaled_x = int(x * (960 / 1280))
                scaled_y = int(y * (540 / 720))
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
                    print("Failed to read frame. Reconnecting in 5 seconds...")
                    cap.release()
                    time.sleep(5)
                    break
                try:
                    self.frame_queue.put(frame, timeout=1)
                except queue.Full:
                    pass
            cap.release()

    def process_model(self, frame):
        with torch.no_grad():
            results = self.broom_model(frame, stream=True, imgsz=960)
        return results

    def export_frame(self, results):
        new_polygons = []
        overlap_detected = False
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                dp = 0.2
                x1 = x1 - dp * (x2 - x1)
                y1 = y1 - dp * (y2 - y1)
                x2 = x2 + dp * (x2 - x1)
                y2 = y2 + dp * (y2 - y1)
                conf = box.conf[0]
                class_id = self.broom_model.names[int(box.cls[0])]
                if conf > self.BROOM_CONFIDENCE_THRESHOLD:
                    bbox_polygon = self.box_to_polygon(x1, y1, x2, y2)
                    for border in self.borders:
                        if bbox_polygon.intersects(border):
                            intersection = bbox_polygon.intersection(border)
                            if not intersection.is_empty:
                                overlap_detected = True
                                if intersection.geom_type in [
                                    "Polygon",
                                    "MultiPolygon",
                                ]:
                                    new_polygons.append(intersection)
        return new_polygons, overlap_detected

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

        # Check if detection is paused
        if self.detection_paused:
            # Check if it's time to resume detection
            if current_time >= self.detection_resume_time:
                self.detection_paused = False
                print("Resuming detection after 10-second pause.")
            else:
                # Skip detection and return the frame as is
                self.draw_borders(frame_resized)
                return frame_resized, False

        results = self.process_model(frame_resized)
        new_polygons, overlap_detected = self.export_frame(results)
        self.update_union_polygon(new_polygons)
        if new_polygons:
            for intersection_polygon in new_polygons:
                if intersection_polygon.geom_type == "Polygon":
                    x, y, w, h = self.polygon_to_bbox(intersection_polygon)
                    cvzone.cornerRect(frame_resized, (x, y, w, h), l=10, t=2, colorR=(0, 255, 255), colorC=(255, 255, 255))
                    area = intersection_polygon.area
                    cvzone.putTextRect(frame_resized, f"Area: {int(area)}", (x, y - 10), scale=0.5, thickness=1, offset=0, colorR=(0, 255, 255), colorT=(0, 0, 0))
                elif intersection_polygon.geom_type == "MultiPolygon":
                    for poly in intersection_polygon.geoms:
                        x, y, w, h = self.polygon_to_bbox(poly)
                        cvzone.cornerRect(frame_resized, (x, y, w, h), l=10, t=2, colorR=(0, 255, 255), colorC=(255, 255, 255))
                        area = poly.area
                        cvzone.putTextRect(frame_resized, f"Area: {int(area)}", (x, y - 10), scale=0.5, thickness=1, offset=0, colorR=(0, 255, 255), colorT=(0, 0, 0))
        self.draw_segments(frame_resized)
        self.draw_borders(frame_resized)
        return frame_resized, overlap_detected

    def box_to_polygon(self, x1, y1, x2, y2):
        return box(x1, y1, x2, y2)

    def polygon_to_bbox(self, polygon):
        minx, miny, maxx, maxy = polygon.bounds
        return int(minx), int(miny), int(maxx - minx), int(maxy - miny)

    def check_conditions(self, percentage, overlap_detected, current_time, frame_resized):
        # New condition: If percentage >= 90%, reset immediately and pause detection
        if percentage >= 90:
            # Reset polygons and print message
            self.union_polygon = None
            self.total_area = 0
            print("AREA DIBERSIHKAN - Percentage >= 90%")
            # Capture frame and send data
            self.capture_and_send(frame_resized, percentage, current_time)
            # Pause detection for 10 seconds
            self.detection_paused = True
            self.detection_resume_time = current_time + 60  # 10-second pause
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
                    print("AREA DIBERSIHKAN")
                    # Capture frame and send data
                    self.capture_and_send(frame_resized, percentage, current_time)
                    self.start_no_overlap_time_high = None
            else:
                self.start_no_overlap_time_high = None

        # Condition when overlap percentage < 80%
        elif percentage >= 50:
            if not overlap_detected:
                if self.start_no_overlap_time_low is None:
                    self.start_no_overlap_time_low = current_time
                elif current_time - self.start_no_overlap_time_low >= 20:
                    # Reset polygons
                    self.union_polygon = None
                    self.total_area = 0
                    self.start_no_overlap_time_low = None
            else:
                self.start_no_overlap_time_low = None
        elif percentage >= 10:
            if not overlap_detected:
                if self.start_no_overlap_time_low is None:
                    self.start_no_overlap_time_low = current_time
                elif current_time - self.start_no_overlap_time_low >= 10:
                    # Reset polygons
                    self.union_polygon = None
                    self.total_area = 0
                    self.start_no_overlap_time_low = None
            else:
                self.start_no_overlap_time_low = None
        else:
            if not overlap_detected:
                if self.start_no_overlap_time_low is None:
                    self.start_no_overlap_time_low = current_time
                elif current_time - self.start_no_overlap_time_low >= 3:
                    # Reset polygons
                    self.union_polygon = None
                    self.total_area = 0
                    self.start_no_overlap_time_low = None
            else:
                self.start_no_overlap_time_low = None

    def capture_and_send(self, frame_resized, percentage, current_time):
        # Add text overlays before saving
        cvzone.putTextRect(frame_resized, f"Overlap: {percentage:.2f}%", (10, 50), scale=1, thickness=2, offset=5)
        cvzone.putTextRect(frame_resized, f"FPS: {int(self.fps)}", (10, 75), scale=1, thickness=2, offset=5)

        # Save the frame to an image file
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        image_path = f"images/cleaned_area_{self.camera_name}_{timestamp_str}.jpg"
        os.makedirs(os.path.dirname(image_path), exist_ok=True)
        cv2.imwrite(image_path, frame_resized)
        # Send data to server
        self.send_to_server(percentage, image_path)

    def send_to_server(self, percentage, image_path, host="10.5.0.2"):
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
            category = "Menyapu Lantai"
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
            print(f"Broom data successfully sent to server.")
        except pymysql.MySQLError as e:
            print(f"Error sending broom data to server: {e}")
        finally:
            if "cursor" in locals():
                cursor.close()
            if "connection" in locals():
                connection.close()

    def main(self):
        process_every_n_frames = 2
        frame_count = 0

        window_name = f"Broom Detection"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, self.window_width, self.window_height)

        if self.video_fps is not None:
            cap = cv2.VideoCapture(self.rtsp_url)
            frame_delay = int(1000 / self.video_fps)

            while cap.isOpened():
                start_time = time.time()
                ret, frame = cap.read()
                if not ret:
                    break
                frame_count += 1
                if frame_count % process_every_n_frames != 0:
                    continue
                current_time = time.time()
                time_diff = current_time - self.prev_frame_time
                self.fps = 1 / time_diff if time_diff > 0 else 0
                self.prev_frame_time = current_time
                frame_resized, overlap_detected = self.process_frame(frame, current_time)

                percentage = (self.total_area / self.total_border_area) * 100 if self.total_border_area > 0 else 0

                # Draw text overlays
                cvzone.putTextRect(frame_resized, f"Overlap: {percentage:.2f}%", (10, 50), scale=1, thickness=2, offset=5)
                cvzone.putTextRect(frame_resized, f"FPS: {int(self.fps)}", (10, 75), scale=1, thickness=2, offset=5)

                # Check conditions
                self.check_conditions(percentage, overlap_detected, current_time, frame_resized)

                cv2.imshow(window_name, frame_resized)
                processing_time = (time.time() - start_time) * 1000
                adjusted_delay = max(int(frame_delay - processing_time), 1)
                key = cv2.waitKey(adjusted_delay) & 0xFF
                if key == ord("n"):
                    break
            cap.release()
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
                percentage = (self.total_area / self.total_border_area) * 100 if self.total_border_area > 0 else 0

                # Draw text overlays
                cvzone.putTextRect(frame_resized, f"Overlap: {percentage:.2f}%", (10, 50), scale=1, thickness=2, offset=5)
                cvzone.putTextRect(frame_resized, f"FPS: {int(self.fps)}", (10, 75), scale=1, thickness=2, offset=5)

                # Check conditions
                self.check_conditions(percentage, overlap_detected, current_time, frame_resized)

                cv2.imshow(window_name, frame_resized)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("n"):
                    self.stop_event.set()
                    break
            cv2.destroyAllWindows()
            self.frame_thread.join()


def run_broom(camera_name, window_size=(320, 240), rtsp_url=None):
    detector = BroomDetector(
        camera_name=camera_name,
        rtsp_url=rtsp_url,
        window_size=window_size,
    )
    detector.main()


if __name__ == "__main__":
    run_broom(
        camera_name="SEWING1",
        # window_size=(980, 600),
        # rtsp_url="D:/NWR/videos/test/broom_test_0002.mp4",
        # rtsp_url="videos/brooming1.mp4",
    )
