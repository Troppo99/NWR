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
            "HALAMANDEPAN1": {
                "borders": [[(611, 191), (717, 194), (828, 202), (828, 163), (717, 160), (614, 159), (611, 191)], [(856, 169), (991, 181), (1135, 197), (1134, 239), (990, 218), (855, 205), (856, 169)]],
                "ip": "10.5.0.236",
            },
            "EKSPEDISI1": {
                "borders": [[(94, 47), (101, 100), (176, 58), (173, 129), (282, 55), (418, 42), (669, 24), (897, 36), (1050, 110), (1278, 275), (1272, 8), (219, 0), (94, 47)]],
                "ip": "10.5.0.155",
            },
            "OFFICE1": {
                "borders": [[(206, 0), (-1, 92), (-1, -2), (206, 0)], [(1036, 3), (1280, 149), (1278, 40), (1183, -1), (1036, 3)]],
                "ip": "10.5.0.170",
            },
            "GUDANGACC2": {
                "borders": [[(1059, 302), (1275, 257), (1278, 117), (1059, 226), (1059, 302)], [(4, 273), (337, 267), (342, 212), (4, 191), (4, 273)]],
                "ip": "10.5.0.107",
            },
            "GUDANGACC3": {  # CAMERA OFFLINE
                "borders": [],
                "ip": "10.5.0.108",
            },
            "GUDANGKAIN3": {  # CAMERA OFFLINE
                "borders": [],
                "ip": "10.5.0.111",
            },
            "GUDANGKAIN4": {  # CAMERA OFFLINE
                "borders": [],
                "ip": "10.5.0.112",
            },
            "GUDANGKAIN5": {  # CAMERA OFFLINE
                "borders": [],
                "ip": "10.5.0.113",
            },
            "FOLDING1": {
                "borders": [[(3, 189), (402, 81), (789, 27), (790, 6), (585, 13), (345, 38), (0, 126), (3, 189)], [(874, 32), (874, 7), (1052, 25), (1057, 52), (874, 32)], [(1070, 58), (1069, 28), (1129, 37), (1279, 83), (1276, 150), (1136, 71), (1070, 58)]],
                "ip": "10.5.0.7",
            },
            "FOLDING2": {
                "borders": [],
                "ip": "10.5.0.",
            },
            "FOLDING3": {
                "borders": [],
                "ip": "10.5.0.114",
            },
            "METALDET1": {
                "borders": [],
                "ip": "10.5.0.115",
            },
            "KANTIN1": {
                "borders": [],
                "ip": "10.5.0.228",
            },
            "KANTIN2": {
                "borders": [],
                "ip": "10.5.0.239",
            },
            "HALAMANBELAKANG2": {
                "borders": [],
                "ip": "10.5.0.43",
            },
            "JALURCUTTING1": {
                "borders": [],
                "ip": "10.5.0.116",
            },
            "JALURCUTTING2": {
                "borders": [],
                "ip": "10.5.0.117",
            },
            "CUTTING2": {
                "borders": [],
                "ip": "10.5.0.118",
            },
            "CUTTING4": {
                "borders": [],
                "ip": "10.5.0.119",
            },
            "CUTTING8": {
                "borders": [],
                "ip": "10.5.0.120",
            },
            "CUTTING9": {
                "borders": [[(240, 179), (231, 222), (369, 204), (560, 190), (764, 184), (761, 145), (497, 151), (240, 179)], [(931, 151), (924, 182), (1116, 199), (1273, 224), (1268, 200), (1117, 171), (931, 151)], [(429, 64), (428, 74), (648, 65), (648, 52), (429, 64)], [(727, 54), (726, 64), (941, 81), (936, 69), (727, 54)], [(995, 77), (1148, 106), (1142, 119), (996, 92), (995, 77)]],
                "ip": "10.5.0.120",
            },
            "CUTTING10": {
                "borders": [[(137, 194), (350, 136), (349, 155), (138, 214), (137, 194)], [(376, 198), (453, 163), (456, 177), (378, 212)], [(512, 146), (512, 146), (539, 135), (542, 145), (514, 156)], [(1091, 192), (1184, 224), (1186, 244), (1090, 206)], [(999, 164), (956, 147), (950, 156), (992, 172)]],
                "ip": "10.5.0.143",
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
            category = "Membersihkan Ramat"
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
        camera_name="CUTTING10",
        display=True,
    )
