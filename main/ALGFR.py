import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import cv2
from ultralytics import YOLO
import numpy as np
import time
import torch
import cvzone
import pymysql
from datetime import datetime, timedelta
import threading
import queue


class CarpalDetector:
    def __init__(
        self,
        CARPAL_ABSENCE_THRESHOLD=10,
        CARPAL_TOUCH_THRESHOLD=0,
        CARPAL_PERCENTAGE_GREEN_THRESHOLD=50,
        CARPAL_CONFIDENCE_THRESHOLD=0.5,
        carpal_model="yolo11l-pose.pt",
        camera_name=None,
        new_size=None,
        rtsp_url=None,
    ):
        self.CARPAL_CONFIDENCE_THRESHOLD = CARPAL_CONFIDENCE_THRESHOLD
        self.CARPAL_ABSENCE_THRESHOLD = CARPAL_ABSENCE_THRESHOLD
        self.CARPAL_TOUCH_THRESHOLD = CARPAL_TOUCH_THRESHOLD
        self.CARPAL_PERCENTAGE_GREEN_THRESHOLD = CARPAL_PERCENTAGE_GREEN_THRESHOLD
        self.new_width, self.new_height = new_size
        self.scale_x = self.new_width / 1280
        self.scale_y = self.new_height / 720
        self.scaled_borders = []
        self.start_time = None
        self.end_time = None
        self.elapsed_time = None
        self.carpal_absence_timer_start = None
        self.prev_frame_time = time.time()
        self.fps = 0
        self.first_green_time = None
        self.is_counting = False
        self.camera_name = camera_name
        if rtsp_url is None:
            self.rtsp_url = f"rtsp://admin:oracle2015@{camera_name}:554/Streaming/Channels/1"
        else:
            self.rtsp_url = rtsp_url
        self.borders, self.idx = self.camera_config(camera_name)
        self.show_text = True
        self.frame_queue = queue.Queue(maxsize=10)
        self.stop_event = threading.Event()
        self.frame_thread = None

        for border in self.borders:
            scaled_border = []
            for x, y in border:
                scaled_x = int(x * self.scale_x)
                scaled_y = int(y * self.scale_y)
                scaled_border.append((scaled_x, scaled_y))
            self.scaled_borders.append(scaled_border)

        self.border_states = {
            idx: {
                "carpal_time": None,
                "is_green": False,
                "carpal_overlap_time": 0.0,
                "last_carpal_overlap_time": None,
            }
            for idx in range(len(self.borders))
        }
        self.borders_pts = [np.array(border, np.int32) for border in self.scaled_borders]
        self.carpal_model = YOLO(carpal_model).to("cuda")
        self.carpal_model.overrides["verbose"] = False
        print(f"Model Carpal device: {next(self.carpal_model.model.parameters()).device}")

    def camera_config(self, camera_name):
        config = {
            "10.5.0.182": [
                [(1, 305), (65, 261), (56, 188), (1, 221)],
                [(1, 221), (56, 188), (51, 127), (1, 155)],
                [(51, 127), (47, 74), (109, 46), (109, 1), (72, 1), (1, 37), (1, 155)],
                [(228, 173), (304, 139), (304, 86), (230, 119)],
                [(230, 119), (304, 86), (302, 38), (227, 71)],
                [(227, 71), (302, 38), (301, 1), (200, 1), (190, 4), (224, 14)],
            ],
            "10.5.0.170": [
                [(30, 258), (60, 253), (58, 211), (24, 217)],
                [(58, 211), (56, 169), (23, 173), (24, 217)],
                [(23, 173), (56, 169), (57, 132), (20, 139)],
                [(20, 139), (57, 132), (58, 78), (21, 97)],
                [(69, 76), (102, 61), (99, 109), (68, 122)],
                [(68, 122), (99, 109), (99, 142), (66, 161)],
                [(66, 161), (99, 142), (99, 178), (67, 197)],
                [(67, 197), (99, 178), (103, 219), (68, 249)],
                [(114, 54), (148, 36), (146, 78), (113, 98)],
                [(113, 98), (146, 78), (145, 117), (113, 133)],
                [(113, 133), (145, 117), (141, 151), (112, 169)],
                [(112, 169), (141, 151), (142, 191), (113, 214)],
                [(160, 35), (183, 25), (181, 66), (158, 75)],
                [(158, 75), (181, 66), (179, 98), (155, 111)],
                [(155, 111), (179, 98), (178, 134), (153, 148)],
                [(153, 148), (178, 134), (181, 165), (153, 184)],
                [(197, 21), (219, 12), (214, 46), (192, 55)],
                [(192, 55), (214, 46), (214, 83), (191, 93)],
                [(191, 93), (214, 83), (213, 112), (191, 126)],
                [(191, 126), (213, 112), (211, 143), (191, 160)],
                [(230, 9), (247, 1), (245, 33), (226, 43)],
                [(226, 43), (245, 33), (243, 69), (226, 77)],
                [(226, 77), (243, 69), (244, 97), (225, 110)],
                [(225, 110), (244, 97), (243, 122), (221, 137)],
                [(275, 86), (285, 81), (286, 61), (274, 64)],
                [(274, 64), (286, 61), (288, 1), (288, 1), (278, 2)],
                [(285, 81), (308, 79), (309, 33), (292, 33)],
                [(292, 33), (309, 33), (311, 4), (288, 1)],
                [(308, 79), (328, 72), (331, 34), (309, 33)],
                [(309, 33), (331, 34), (333, 2), (311, 4)],
                [(328, 72), (350, 62), (350, 30), (331, 34)],
                [(331, 34), (350, 30), (352, 2), (333, 2)],
                [(350, 62), (369, 53), (369, 25), (350, 30)],
                [(350, 30), (369, 25), (371, 1), (352, 2)],
                [(369, 53), (387, 44), (385, 20), (369, 25)],
                [(369, 25), (385, 20), (387, 3), (371, 1)],
            ],
            "10.5.0.161": [
                [(2, 517), (30, 424), (-1, 345)],
                [(-1, 345), (30, 424), (46, 386), (-1, 208)],
                [(63, 344), (98, 267), (88, 214), (42, 304), (45, 318), (52, 308)],
                [(42, 304), (88, 214), (79, 165), (30, 251)],
                [(30, 251), (79, 165), (71, 112), (19, 180)],
                [(19, 180), (71, 112), (67, 37), (7, 98), (19, 180)],
                [(1050, 0), (1075, 21), (1078, 0)],
                [(1103, 166), (1151, 231), (1161, 186), (1109, 121)],
                [(1109, 121), (1161, 186), (1170, 134), (1115, 82)],
                [(1115, 82), (1170, 134), (1176, 88), (1116, 44)],
                [(1116, 44), (1176, 88), (1178, 38), (1119, 10)],
                [(1119, 10), (1178, 38), (1179, 0), (1119, -3)],
                [(1208, 302), (1161, 235), (1169, 195), (1221, 257)],
                [(1221, 257), (1169, 195), (1180, 140), (1235, 187)],
                [(1235, 187), (1180, 140), (1189, 89), (1246, 126)],
                [(1246, 126), (1189, 89), (1188, 42), (1252, 66)],
                [(1252, 66), (1188, 42), (1193, 6), (1256, 23)],
                [(1256, 23), (1193, 6), (1254, 4)],
            ],
        }
        camera_names = list(config.keys())
        indices = {name: idx + 1 for idx, name in enumerate(camera_names)}
        return config[camera_name], indices[camera_name]

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
            if (
                keypoints_data is not None
                and keypoints_data.xy is not None
                and keypoints_data.conf is not None
            ):
                if keypoints_data.shape[0] > 0:
                    keypoints_array = keypoints_data.xy.cpu().numpy()
                    keypoints_conf = keypoints_data.conf.cpu().numpy()
                    for keypoints_per_object, keypoints_conf_per_object in zip(
                        keypoints_array, keypoints_conf
                    ):
                        keypoints_list = []
                        for kp, kp_conf in zip(keypoints_per_object, keypoints_conf_per_object):
                            if kp_conf >= confidence_threshold:
                                x, y = kp[0], kp[1]
                                keypoints_list.append((int(x), int(y)))
                            else:
                                keypoints_list.append(None)
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
                    print("Stream gagal dibaca. Pastikan URL stream benar.")
                    cap.release()
                    time.sleep(5)
                    break
                try:
                    self.frame_queue.put(frame, block=False)
                except queue.Full:
                    pass
            cap.release()

    def process_frame(self, frame, current_time, percentage_green, pairs_human):
        frame_resized = cv2.resize(frame, (self.new_width, self.new_height))
        results = self.process_model(frame_resized)
        points, coords, keypoint_positions = self.export_frame(results, (0, 255, 0), pairs_human)

        border_colors = [
            (0, 255, 0) if state["is_green"] else (0, 255, 255)
            for state in self.border_states.values()
        ]

        carpal_overlapping_any_border = False

        for border_id, border_pt in enumerate(self.borders_pts):
            carpal_overlapping = False
            for keypoints_list in keypoint_positions:
                for idx in [9, 10]:
                    if idx < len(keypoints_list):
                        kp = keypoints_list[idx]
                        if kp is not None:
                            result = cv2.pointPolygonTest(border_pt, kp, False)
                            if result >= 0:
                                carpal_overlapping = True
                                carpal_overlapping_any_border = True
                                break
                if carpal_overlapping:
                    break

            if carpal_overlapping:
                if self.border_states[border_id]["last_carpal_overlap_time"] is None:
                    self.border_states[border_id]["last_carpal_overlap_time"] = current_time
                else:
                    delta_time = (
                        current_time - self.border_states[border_id]["last_carpal_overlap_time"]
                    )
                    self.border_states[border_id]["carpal_overlap_time"] += delta_time
                    self.border_states[border_id]["last_carpal_overlap_time"] = current_time

                if (
                    self.border_states[border_id]["carpal_overlap_time"]
                    >= self.CARPAL_TOUCH_THRESHOLD
                ):
                    self.border_states[border_id]["is_green"] = True
                    border_colors[border_id] = (0, 255, 0)
            else:
                self.border_states[border_id]["last_carpal_overlap_time"] = None

        green_borders_exist = any(state["is_green"] for state in self.border_states.values())
        if green_borders_exist:
            if not self.is_counting:
                self.first_green_time = current_time
                self.is_counting = True
            if carpal_overlapping_any_border:
                self.carpal_absence_timer_start = current_time
            else:
                if self.carpal_absence_timer_start is None:
                    self.carpal_absence_timer_start = current_time
                elif (
                    current_time - self.carpal_absence_timer_start
                ) >= self.CARPAL_ABSENCE_THRESHOLD:
                    print(f"Resetting carpal borders in percentage {percentage_green:.2f}%")
                    if percentage_green >= self.CARPAL_PERCENTAGE_GREEN_THRESHOLD:
                        print(
                            f"Green carpal border is bigger than {self.CARPAL_PERCENTAGE_GREEN_THRESHOLD}% and data is sent to server"
                        )
                        if self.first_green_time is not None:
                            self.elapsed_time = current_time - self.first_green_time
                        overlay = frame_resized.copy()
                        alpha = 0.5
                        for border_pt, color in zip(self.borders_pts, border_colors):
                            cv2.fillPoly(overlay, pts=[border_pt], color=color)
                        cv2.addWeighted(overlay, alpha, frame_resized, 1 - alpha, 0, frame_resized)
                        minutes, seconds = divmod(int(self.elapsed_time), 60)
                        time_str = f"Elapsed Time: {minutes:02d}:{seconds:02d}"
                        if self.show_text:
                            cvzone.putTextRect(
                                frame_resized, time_str, (10, self.new_height-100), scale=1, thickness=2, offset=5
                            )
                            cvzone.putTextRect(
                                frame_resized,
                                f"Percentage of Green Border: {percentage_green:.2f}%",
                                (10, self.new_height-50),
                                scale=1,
                                thickness=2,
                                offset=5,
                            )
                            cvzone.putTextRect(
                                frame_resized,
                                f"FPS: {int(self.fps)}",
                                (10, self.new_height-75),
                                scale=1,
                                thickness=2,
                                offset=5,
                            )
                        image_path = "main/images/green_borders_image_182.jpg"
                        cv2.imwrite(image_path, frame_resized)
                        self.send_to_server(
                            "10.5.0.2", percentage_green, self.elapsed_time, image_path
                        )
                    for idx in range(len(self.borders)):
                        self.border_states[idx] = {
                            "is_green": False,
                            "carpal_overlap_time": 0.0,
                            "last_carpal_overlap_time": None,
                        }
                        border_colors[idx] = (0, 255, 255)
                    self.first_green_time = None
                    self.is_counting = False
                    self.carpal_absence_timer_start = None
        else:
            self.carpal_absence_timer_start = None
            if self.is_counting:
                self.first_green_time = None
                self.is_counting = False

        if percentage_green == 100:
            print("Percentage carpal green is 100%, performing immediate reset and data send.")
            if self.first_green_time is not None:
                self.elapsed_time = current_time - self.first_green_time
            overlay = frame_resized.copy()
            alpha = 0.5
            for border_pt, color in zip(self.borders_pts, border_colors):
                cv2.fillPoly(overlay, pts=[border_pt], color=color)
            cv2.addWeighted(overlay, alpha, frame_resized, 1 - alpha, 0, frame_resized)
            minutes, seconds = divmod(int(self.elapsed_time), 60)
            time_str = f"Elapsed Time: {minutes:02d}:{seconds:02d}"
            if self.show_text:
                cvzone.putTextRect(
                    frame_resized, time_str, (10, self.new_height-100), scale=1, thickness=2, offset=5
                )
                cvzone.putTextRect(
                    frame_resized,
                    f"Percentage of Green Border: {percentage_green:.2f}%",
                    (10, self.new_height-50),
                    scale=1,
                    thickness=2,
                    offset=5,
                )
                cvzone.putTextRect(
                    frame_resized, f"FPS: {int(self.fps)}", (10, self.new_height-75), scale=1, thickness=2, offset=5
                )
            image_path = "main/images/green_borders_image_182.jpg"
            cv2.imwrite(image_path, frame_resized)
            self.send_to_server("10.5.0.2", percentage_green, self.elapsed_time, image_path)

            for idx in range(len(self.borders)):
                self.border_states[idx] = {
                    "is_green": False,
                    "carpal_overlap_time": 0.0,
                    "last_carpal_overlap_time": None,
                }
                border_colors[idx] = (0, 255, 255)
            self.first_green_time = None
            self.is_counting = False
            self.carpal_absence_timer_start = None

        if points and coords:
            for x, y, color in coords:
                cv2.line(frame_resized, x, y, color, 2)
            for point in points:
                cv2.circle(frame_resized, point, 4, (0, 255, 255), -1)
        overlay = frame_resized.copy()
        alpha = 0.5
        for border_pt, color in zip(self.borders_pts, border_colors):
            cv2.fillPoly(overlay, pts=[border_pt], color=color)
        cv2.addWeighted(overlay, alpha, frame_resized, 1 - alpha, 0, frame_resized)
        if self.is_counting and self.first_green_time is not None:
            self.elapsed_time = current_time - self.first_green_time
            minutes, seconds = divmod(int(self.elapsed_time), 60)
            time_str = f"Elapsed Time: {minutes:02d}:{seconds:02d}"
            if self.show_text:
                cvzone.putTextRect(
                    frame_resized, time_str, (10, self.new_height-100), scale=1, thickness=2, offset=5
                )

        return frame_resized

    def send_to_server(self, host, percentage_green, elapsed_time, image_path):
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
            return user, password, database, port

        try:
            user, password, database, port = server_address(host)
            connection = pymysql.connect(
                host=host, user=user, password=password, database=database, port=port
            )
            cursor = connection.cursor()
            table = "empbro"
            camera_name = self.camera_name
            timestamp_done = datetime.now()
            timestamp_start = timestamp_done - timedelta(seconds=elapsed_time)

            timestamp_done_str = timestamp_done.strftime("%Y-%m-%d %H:%M:%S")
            timestamp_start_str = timestamp_start.strftime("%Y-%m-%d %H:%M:%S")

            # **Define the parameter time to compare with (e.g., 09:00:00)**
            parameter_time_str = "09:00:00"
            parameter_time = datetime.strptime(parameter_time_str, "%H:%M:%S").time()

            # **Extract the time portion of timestamp_done**
            timestamp_done_time = timestamp_done.time()

            # **Compare and set isdiscipline**
            if timestamp_done_time > parameter_time:
                isdiscipline = "Tidak disiplin"
            else:
                isdiscipline = "Disiplin"

            with open(image_path, "rb") as file:
                binary_image = file.read()

            query = f"""
            INSERT INTO {table} (cam, timestamp_start, timestamp_done, percentage, image_done, isdiscipline)
            VALUES (%s, %s, %s, %s, %s, %s)
            """
            cursor.execute(
                query,
                (
                    camera_name,
                    timestamp_start_str,
                    timestamp_done_str,
                    percentage_green,
                    binary_image,
                    isdiscipline,
                ),
            )
            connection.commit()
            print(f"Carpal data successfully sent to server.")
        except pymysql.MySQLError as e:
            print(f"Error sending carpal data to server: {e}")
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

        self.frame_thread = threading.Thread(target=self.frame_capture)
        self.frame_thread.daemon = True
        self.frame_thread.start()

        window_name = f"RUN{self.idx} : {self.camera_name}"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, self.new_width, self.new_height)

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
            total_borders = len(self.borders)
            green_borders = sum(1 for state in self.border_states.values() if state["is_green"])
            percentage_green = (green_borders / total_borders) * 100
            frame_resized = self.process_frame(frame, current_time, percentage_green, pairs_human)
            if self.show_text:
                cvzone.putTextRect(
                    frame_resized,
                    f"Percentage of Green Border: {percentage_green:.2f}%",
                    (10, self.new_height-50),
                    scale=1,
                    thickness=2,
                    offset=5,
                )
                cvzone.putTextRect(
                    frame_resized, f"FPS: {int(self.fps)}", (10, self.new_height-75), scale=1, thickness=2, offset=5
                )
            cv2.imshow(window_name, frame_resized)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("n"):
                self.stop_event.set()
                break
            elif key == ord("s"):
                self.show_text = not self.show_text
        cv2.destroyAllWindows()
        self.frame_thread.join()

def run_carpal(CARPAL_ABSENCE_THRESHOLD, CARPAL_TOUCH_THRESHOLD, CARPAL_PERCENTAGE_GREEN_THRESHOLD, camera_name, new_size=(360, 202)):
    detector = CarpalDetector(
        CARPAL_ABSENCE_THRESHOLD=CARPAL_ABSENCE_THRESHOLD,
        CARPAL_TOUCH_THRESHOLD=CARPAL_TOUCH_THRESHOLD,
        CARPAL_PERCENTAGE_GREEN_THRESHOLD=CARPAL_PERCENTAGE_GREEN_THRESHOLD,
        camera_name=camera_name,
        new_size=new_size,
    )

    detector.main()

if __name__ == "__main__":
    run_carpal(
        CARPAL_ABSENCE_THRESHOLD=10,
        CARPAL_TOUCH_THRESHOLD=0,
        CARPAL_PERCENTAGE_GREEN_THRESHOLD=50,
        camera_name="10.5.0.182",
        new_size=(960, 540),
    )
