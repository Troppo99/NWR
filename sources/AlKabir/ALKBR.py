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
        display=True,
    ):
        self.BROOM_CONFIDENCE_THRESHOLD = BROOM_CONFIDENCE_THRESHOLD
        self.window_width, self.window_height = window_size
        self.new_width, self.new_height = (960, 540)
        self.prev_frame_time = 0
        self.fps = 0
        self.union_polygon = None
        self.total_area = 0
        self.camera_name = camera_name
        self.borders, self.ip_camera = self.camera_config()
        self.total_border_area = sum(border.area for border in self.borders) if self.borders else 0
        self.display = display
        if self.display is False:
            print(f"B`{self.camera_name} : >>>Display is disabled!<<<")
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
        self.broom_model = YOLO("models/broom6l.pt").to("cuda")
        self.broom_model.overrides["verbose"] = False
        print(f"Model Broom device: {next(self.broom_model.model.parameters()).device}")
        self.last_overlap_time = time.time()
        self.area_cleared = False
        self.start_no_overlap_time_high = None
        self.start_no_overlap_time_low = None
        self.detection_paused = False
        self.detection_resume_time = None
        self.detection_pause_duration = 10
        self.timestamp_start = None
        # New variable `to track the time when coverage first reached 90%
        self.start_high_coverage_time = None

    def camera_config(self):
        config = {
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
            "SEWINGOFFICE": {
                "borders": [[(397, 715), (991, 715), (961, 555), (908, 348), (852, 186), (793, 43), (777, 2), (608, 1), (568, 110), (569, 180), (560, 194), (536, 195), (445, 493), (397, 715)]],
                "ip": "10.5.0.201",
            },
            "SEWING1": {
                "borders": [[(1028, 476), (1062, 563), (801, 597), (574, 608), (597, 314), (618, 81), (704, 79), (759, 322), (788, 495), (1028, 476)]],
                "ip": "10.5.0.217",
            },
            "SEWING2": {
                "borders": [[(642, 79), (721, 77), (797, 512), (812, 630), (580, 637), (519, 632), (533, 513), (599, 515), (642, 79)]],
                "ip": "10.5.0.151",
            },
            "SEWING3": {
                "borders": [[(405, 456), (470, 463), (500, 306), (544, 135), (565, 57), (652, 60), (659, 204), (675, 476), (680, 586), (452, 569), (383, 562), (405, 456), (405, 456)]],
                "ip": "10.5.0.214",
            },
            "SEWING4": {
                "borders": [[(584, 40), (672, 41), (686, 128), (683, 210), (706, 211), (739, 406), (776, 718), (480, 718), (508, 466), (559, 170), (584, 40)]],
                "ip": "10.5.0.146",
            },
            "SEWING6": {
                "borders": [[(544, 451), (532, 604), (584, 605), (780, 599), (758, 441), (694, 81), (628, 85), (594, 452), (544, 451)]],
                "ip": "10.5.0.252",
            },
            "SEWING5": {
                "borders": [[(631, 205), (696, 206), (724, 360), (756, 569), (775, 710), (567, 710), (625, 225), (631, 205)]],
                "ip": "10.5.0.159",
            },
            "SEWING7": {
                "borders": [[(628, 180), (697, 176), (732, 332), (779, 542), (845, 538), (880, 678), (800, 692), (593, 704), (605, 548), (628, 180)]],
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
                "borders": [[(63, 559), (165, 578), (351, 604), (538, 619), (774, 618), (801, 616), (823, 721), (229, 720), (102, 681), (63, 559)]],
                "ip": "10.5.0.207",
            },
            "GUDANGACC1": {
                "borders": [[(698, 95), (819, 126), (864, 281), (902, 451), (926, 594), (938, 716), (680, 716), (695, 368), (698, 95)]],
                "ip": "10.5.0.110",
            },
            "GUDANGACC2": {
                "borders": [[(880, 478), (921, 481), (916, 540), (866, 645), (708, 635), (880, 478)]],
                "ip": "10.5.0.107",
            },
            "GUDANGACC3": {
                "borders": [[(794, 409), (823, 481), (249, 445), (65, 660), (-1, 626), (1, 465), (92, 391), (143, 402), (175, 373), (794, 409)]],
                "ip": "10.5.0.108",
            },
            "GUDANGACC4": {
                "borders": [[(512, 94), (735, 84), (811, 287), (876, 492), (930, 715), (318, 714), (368, 500), (427, 316), (512, 94)]],
                "ip": "10.5.0.180",
            },
            "CUTTING1": {
                "borders": [[(16, 373), (55, 348), (251, 521), (431, 646), (306, 718), (129, 716), (102, 673), (67, 607), (35, 534), (67, 518), (16, 373)]],
                "ip": "10.5.0.66",
            },
            "CUTTING2": {
                "borders": [[(7, 317), (253, 246), (683, 368), (90, 574), (6, 437), (7, 317)]],
                "ip": "10.5.0.58",
            },
            "CUTTING3": {
                "borders": [[(4, 408), (284, 329), (485, 456), (610, 529), (755, 605), (883, 668), (1002, 561), (1100, 460), (1156, 391), (1252, 434), (1219, 535), (1131, 717), (6, 717), (4, 408)]],
                "ip": "10.5.0.92",
            },
            "CUTTING4": {
                "borders": [[(6, 196), (4, 555), (65, 720), (1277, 716), (1275, 533), (881, 355), (244, 580), (153, 462), (84, 350), (6, 196)]],
                "ip": "10.5.0.101",
            },
            "CUTTING5": {
                "borders": [[(1201, 287), (1104, 248), (804, 347), (668, 393), (450, 466), (142, 560), (205, 718), (459, 717), (398, 646), (603, 574), (782, 498), (920, 433), (1062, 507), (1154, 433), (1033, 376), (1201, 287)]],
                "ip": "10.5.0.87",
            },
            "CUTTING8": {
                "borders": [[(17, 399), (79, 373), (136, 453), (215, 542), (331, 653), (407, 718), (119, 718), (61, 589), (32, 475), (17, 399)]],
                "ip": "10.5.0.95",
            },
            "CUTTING9": {
                "borders": [[(599, 247), (722, 247), (740, 325), (827, 715), (520, 717), (599, 247)]],
                "ip": "10.5.0.120",
            },
            "CUTTING10": {
                "borders": [[(658, 282), (736, 282), (802, 571), (639, 570), (658, 282)]],
                "ip": "10.5.0.143",
            },
            "GUDANGKAIN1": {  # CAMERA OFFLINE
                "borders": [],
                "ip": "10.5.0.109",
            },
            "GUDANGKAIN2": {  # CAMERA OFFLINE
                "borders": [],
                "ip": "10.5.0.110",
            },
            "FOLDING1": {
                "borders": [],
                "ip": "10.5.0.111",
            },
            "FOLDING2": {
                "borders": [],
                "ip": "10.5.0.112",
            },
            "FOLDING3": {
                "borders": [],
                "ip": "10.5.0.113",
            },
            "METALDET1": {
                "borders": [],
                "ip": "10.5.0.114",
            },
            "FREEMETAL1": {
                "borders": [],
                "ip": "10.5.0.115",
            },
            "FREEMETAL2": {
                "borders": [],
                "ip": "10.5.0.116",
            },
            "BUFFER1": {
                "borders": [],
                "ip": "10.5.0.116",
            },
            "INNERBOX1": {
                "borders": [],
                "ip": "10.5.0.117",
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
                    print(f"B`{self.camera_name} : Invalid polygon for camera {self.camera_name}, skipping.")
            else:
                print(f"B`{self.camera_name} : Not enough points to form a polygon for camera {self.camera_name}, skipping.")
        return scaled_borders, ip

    def frame_capture(self):
        rtsp_url = self.rtsp_url
        while not self.stop_event.is_set():
            cap = cv2.VideoCapture(rtsp_url)
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
                                if intersection.geom_type in ["Polygon", "MultiPolygon"]:
                                    new_polygons.append(intersection)
        return new_polygons, overlap_detected

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

    def process_frame(self, frame, current_time):
        frame_resized = cv2.resize(frame, (self.new_width, self.new_height))
        if self.detection_paused:
            if current_time >= self.detection_resume_time:
                self.detection_paused = False
                print(f"B`{self.camera_name} : Resuming detection after {self.detection_pause_duration}-second pause.")
            else:
                self.draw_borders(frame_resized)
                return frame_resized, False
        results = self.process_model(frame_resized)
        new_polygons, overlap_detected = self.export_frame(results)
        self.update_union_polygon(new_polygons)
        if overlap_detected and self.timestamp_start is None:
            self.timestamp_start = datetime.now()
        if new_polygons and self.display:
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
        if self.display:
            self.draw_segments(frame_resized)
            self.draw_borders(frame_resized)
        return frame_resized, overlap_detected

    def box_to_polygon(self, x1, y1, x2, y2):
        return box(x1, y1, x2, y2)

    def polygon_to_bbox(self, polygon):
        minx, miny, maxx, maxy = polygon.bounds
        return int(minx), int(miny), int(maxx - minx), int(maxy - miny)

    def check_conditions(self, percentage, overlap_detected, current_time, frame_resized):
        # Check for 90% or above coverage with a 5-second confirmation period
        if percentage >= 90:
            if self.start_high_coverage_time is None:
                self.start_high_coverage_time = current_time
            if current_time - self.start_high_coverage_time >= 5:  # Coverage stays >= 90% for 5 seconds
                self.union_polygon = None
                self.total_area = 0
                print(f"B`{self.camera_name} : Percentage >= 90% confirmed after 5 seconds.")
                self.capture_and_send(frame_resized, percentage, current_time)
                self.timestamp_start = None
                self.detection_paused = True
                self.detection_resume_time = current_time + self.detection_pause_duration
                self.start_no_overlap_time_high = None
                self.start_no_overlap_time_low = None
                self.start_high_coverage_time = None
                return
        else:
            self.start_high_coverage_time = None

        if 50 <= percentage < 90:
            if not overlap_detected:
                if self.start_no_overlap_time_low is None:
                    self.start_no_overlap_time_low = current_time
                elif current_time - self.start_no_overlap_time_low >= 60:
                    self.union_polygon = None
                    self.total_area = 0
                    print(f"B`{self.camera_name} : Percentage >= 50%")
                    self.capture_and_send(frame_resized, percentage, current_time)
                    self.start_no_overlap_time_low = None
            else:
                self.start_no_overlap_time_low = None
        elif 5 <= percentage < 50:
            if not overlap_detected:
                if self.start_no_overlap_time_low is None:
                    self.start_no_overlap_time_low = current_time
                elif current_time - self.start_no_overlap_time_low >= 30:
                    self.union_polygon = None
                    self.total_area = 0
                    print(f"B`{self.camera_name} : Percentage >= 5%")
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
                    print(f"B`{self.camera_name} : Percentage < 5%")
                    self.start_no_overlap_time_low = None
            else:
                self.start_no_overlap_time_low = None

        if self.detection_paused:
            if current_time >= self.detection_resume_time:
                self.detection_paused = False
                print(f"B`{self.camera_name} : Resuming detection after {self.detection_pause_duration}-second pause.")

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
        if self.video_fps is not None:
            cap = cv2.VideoCapture(self.rtsp_url)
            frame_delay = int(1000 / self.video_fps)
            while cap.isOpened():
                start_time = time.time()
                ret, frame = cap.read()
                if not ret:
                    print(f"B`{self.camera_name} : End of video file or cannot read the frame.")
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
                if self.display:
                    cvzone.putTextRect(frame_resized, f"Overlap: {percentage:.2f}%", (10, 50), scale=1, thickness=2, offset=5)
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
                percentage = (self.total_area / self.total_border_area) * 100 if self.total_border_area > 0 else 0
                if self.display:
                    cvzone.putTextRect(frame_resized, f"Overlap: {percentage:.2f}%", (10, 50), scale=1, thickness=2, offset=5)
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
        display=True,
        # rtsp_url="D:/NWR/videos/test/broom_test_0002.mp4",
    )
