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
import math


class MotorDetector:
    def __init__(
        self,
        MOTOR_ABSENCE_THRESHOLD=10,
        MOTOR_TOUCH_THRESHOLD=0,
        MOTOR_PERCENTAGE_YELLOW_THRESHOLD=50,
        MOTOR_CONFIDENCE_THRESHOLD=0.5,
        motor_model="yolo11l.pt",
        camera_name=None,
        new_size=(960, 540),
        rtsp_url=None,
        window_size=(540, 360),
    ):
        self.MOTOR_CONFIDENCE_THRESHOLD = MOTOR_CONFIDENCE_THRESHOLD
        self.MOTOR_ABSENCE_THRESHOLD = MOTOR_ABSENCE_THRESHOLD
        self.MOTOR_TOUCH_THRESHOLD = MOTOR_TOUCH_THRESHOLD
        self.MOTOR_PERCENTAGE_YELLOW_THRESHOLD = MOTOR_PERCENTAGE_YELLOW_THRESHOLD
        self.window_width, self.window_height = window_size
        self.new_width, self.new_height = new_size
        self.scale_x = self.new_width / 1280
        self.scale_y = self.new_height / 720
        self.scaled_borders = []
        self.start_time = None
        self.end_time = None
        self.elapsed_time = None
        self.motor_absence_timer_start = None
        self.prev_frame_time = time.time()
        self.fps = 0
        self.first_yellow_time = None
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
                "motor_time": None,
                "is_yellow": False,
                "motor_overlap_time": 0.0,
                "last_motor_overlap_time": None,
            }
            for idx in range(len(self.borders))
        }
        self.borders_pts = [np.array(border, np.int32) for border in self.scaled_borders]
        self.motor_model = YOLO(motor_model).to("cuda")
        self.motor_model.overrides["verbose"] = False
        print(f"Model Motor device: {next(self.motor_model.model.parameters()).device}")

    def camera_config(self, camera_name):
        config = {
            "10.5.0.206": [
                [(0, 719), (0, 429), (413, 429), (414, 722)],
                [(414, 722), (413, 429), (844, 423), (867, 721)],
                [(867, 721), (844, 423), (1281, 420), (1288, 744)],
                [(0, 429), (0, 391), (150, 261), (465, 259), (413, 429)],
                [(413, 429), (465, 259), (861, 262), (844, 423)],
                [(844, 423), (861, 262), (1248, 287), (1276, 302), (1281, 420)],
                [(150, 261), (269, 184), (491, 183), (465, 259)],
                [(465, 259), (491, 183), (853, 205), (861, 262)],
                [(861, 262), (853, 205), (915, 215), (915, 191), (1077, 212), (1152, 229), (1248, 287)],
            ],
        }
        camera_names = list(config.keys())
        indices = {name: idx + 1 for idx, name in enumerate(camera_names)}
        return config[camera_name], indices[camera_name]

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

    def send_to_server(self, host, percentage_yellow, elapsed_time, image_path):
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
            connection = pymysql.connect(host=host, user=user, password=password, database=database, port=port)
            cursor = connection.cursor()
            table = "empbro"
            activity = "Mengelap Kaca"
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
            INSERT INTO {table} (cam, activity, timestamp_start, timestamp_done, percentage, image_done, isdiscipline)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            """
            cursor.execute(
                query,
                (
                    camera_name,
                    activity,
                    timestamp_start_str,
                    timestamp_done_str,
                    percentage_yellow,
                    binary_image,
                    isdiscipline,
                ),
            )
            connection.commit()
            print(f"Motor data successfully sent to server.")
        except pymysql.MySQLError as e:
            print(f"Error sending motor data to server: {e}")
        finally:
            if "cursor" in locals():
                cursor.close()
            if "connection" in locals():
                connection.close()

    def process_model(self, frame):
        with torch.no_grad():
            results = self.motor_model(frame, stream=True, imgsz=1280)
        return results

    def export_frame(self, results):
        boxes_info = []
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = box.conf[0]
                class_id = self.motor_model.names[int(box.cls[0])]
                if conf > self.MOTOR_CONFIDENCE_THRESHOLD and class_id == "motorcycle":
                    boxes_info.append((x1, y1, x2, y2, conf, class_id))
        return boxes_info

    def process_frame(self, frame, current_time, percentage_yellow):
        frame_resized = cv2.resize(frame, (self.new_width, self.new_height))
        results = self.process_model(frame_resized)
        boxes_info = self.export_frame(results)

        border_colors = [(0, 255, 255) if state["is_yellow"] else (0, 255, 0) for state in self.border_states.values()]

        motor_overlapping_any_border = False

        for border_id, border_pt in enumerate(self.borders_pts):
            motor_overlapping = False
            border_polygon = np.array(border_pt, dtype=np.int32)
            border_contour = border_polygon.reshape((-1, 1, 2))

            for x1, y1, x2, y2, conf, class_id in boxes_info:
                if class_id == "motorcycle":
                    # Membuat polygon dari bounding box
                    box_polygon = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.int32)
                    box_contour = box_polygon.reshape((-1, 1, 2))

                    # Menghitung area intersection antara bounding box dan border
                    intersection = cv2.intersectConvexConvex(border_contour, box_contour)[1]
                    if intersection is not None and cv2.contourArea(intersection) > 0:
                        motor_overlapping = True
                        motor_overlapping_any_border = True
                        break

            if motor_overlapping:
                if self.border_states[border_id]["last_motor_overlap_time"] is None:
                    self.border_states[border_id]["last_motor_overlap_time"] = current_time
                else:
                    delta_time = current_time - self.border_states[border_id]["last_motor_overlap_time"]
                    self.border_states[border_id]["motor_overlap_time"] += delta_time
                    self.border_states[border_id]["last_motor_overlap_time"] = current_time

                if self.border_states[border_id]["motor_overlap_time"] >= self.MOTOR_TOUCH_THRESHOLD:
                    self.border_states[border_id]["is_yellow"] = True
                    border_colors[border_id] = (0, 255, 255)
            else:
                self.border_states[border_id]["last_motor_overlap_time"] = None

        yellow_borders_exist = any(state["is_yellow"] for state in self.border_states.values())
        if yellow_borders_exist:
            if not self.is_counting:
                self.first_yellow_time = current_time
                self.is_counting = True
            if motor_overlapping_any_border:
                self.motor_absence_timer_start = current_time
            else:
                if self.motor_absence_timer_start is None:
                    self.motor_absence_timer_start = current_time
                elif (current_time - self.motor_absence_timer_start) >= self.MOTOR_ABSENCE_THRESHOLD:
                    print(f"Resetting motor borders in percentage {percentage_yellow:.2f}%")
                    if percentage_yellow >= self.MOTOR_PERCENTAGE_YELLOW_THRESHOLD:
                        print(f"Yellow motor border is bigger than {self.MOTOR_PERCENTAGE_YELLOW_THRESHOLD}% and data is sent to server")
                        if self.first_yellow_time is not None:
                            self.elapsed_time = current_time - self.first_yellow_time
                        overlay = frame_resized.copy()
                        alpha = 0.5
                        for border_pt, color in zip(self.borders_pts, border_colors):
                            cv2.fillPoly(overlay, pts=[border_pt], color=color)
                        cv2.addWeighted(overlay, alpha, frame_resized, 1 - alpha, 0, frame_resized)
                        minutes, seconds = divmod(int(self.elapsed_time), 60)
                        time_str = f"Elapsed Time: {minutes:02d}:{seconds:02d}"
                        if self.show_text:
                            cvzone.putTextRect(frame_resized, time_str, (10, self.new_height - 100), scale=1, thickness=2, offset=5)
                            cvzone.putTextRect(
                                frame_resized,
                                f"Percentage of Yellow Border: {percentage_yellow:.2f}%",
                                (10, self.new_height - 50),
                                scale=1,
                                thickness=2,
                                offset=5,
                            )
                            cvzone.putTextRect(
                                frame_resized,
                                f"FPS: {int(self.fps)}",
                                (10, self.new_height - 75),
                                scale=1,
                                thickness=2,
                                offset=5,
                            )
                        image_path = "main/images/yellow_borders_image_182.jpg"
                        cv2.imwrite(image_path, frame_resized)
                        self.send_to_server("10.5.0.2", percentage_yellow, self.elapsed_time, image_path)
                    for idx in range(len(self.borders)):
                        self.border_states[idx] = {
                            "is_yellow": False,
                            "motor_overlap_time": 0.0,
                            "last_motor_overlap_time": None,
                        }
                        border_colors[idx] = (0, 255, 255)
                    self.first_yellow_time = None
                    self.is_counting = False
                    self.motor_absence_timer_start = None
        else:
            self.motor_absence_timer_start = None
            if self.is_counting:
                self.first_yellow_time = None
                self.is_counting = False

        if percentage_yellow == 100:
            print("Percentage motor yellow is 100%, performing immediate reset and data send.")
            if self.first_yellow_time is not None:
                self.elapsed_time = current_time - self.first_yellow_time
            overlay = frame_resized.copy()
            alpha = 0.5
            for border_pt, color in zip(self.borders_pts, border_colors):
                cv2.fillPoly(overlay, pts=[border_pt], color=color)
            cv2.addWeighted(overlay, alpha, frame_resized, 1 - alpha, 0, frame_resized)
            minutes, seconds = divmod(int(self.elapsed_time), 60)
            time_str = f"Elapsed Time: {minutes:02d}:{seconds:02d}"
            if self.show_text:
                cvzone.putTextRect(frame_resized, time_str, (10, self.new_height - 100), scale=1, thickness=2, offset=5)
                cvzone.putTextRect(
                    frame_resized,
                    f"Percentage of Yellow Border: {percentage_yellow:.2f}%",
                    (10, self.new_height - 50),
                    scale=1,
                    thickness=2,
                    offset=5,
                )
                cvzone.putTextRect(frame_resized, f"FPS: {int(self.fps)}", (10, self.new_height - 75), scale=1, thickness=2, offset=5)
            image_path = "main/images/yellow_borders_image_182.jpg"
            cv2.imwrite(image_path, frame_resized)
            self.send_to_server("10.5.0.2", percentage_yellow, self.elapsed_time, image_path)

            for idx in range(len(self.borders)):
                self.border_states[idx] = {
                    "is_yellow": False,
                    "motor_overlap_time": 0.0,
                    "last_motor_overlap_time": None,
                }
                border_colors[idx] = (0, 255, 255)
            self.first_yellow_time = None
            self.is_counting = False
            self.motor_absence_timer_start = None

        if boxes_info:
            for x1, y1, x2, y2, conf, class_id in boxes_info:
                cvzone.putTextRect(frame_resized, f"{class_id} {conf:.2f}", (x1, y1), scale=1, thickness=2, offset=5)
                cvzone.cornerRect(frame_resized, (x1, y1, x2 - x1, y2 - y1), l=10, t=2, colorR=(0, 255, 0))
        overlay = frame_resized.copy()
        alpha = 0.5
        for border_pt, color in zip(self.borders_pts, border_colors):
            cv2.fillPoly(overlay, pts=[border_pt], color=color)
        cv2.addWeighted(overlay, alpha, frame_resized, 1 - alpha, 0, frame_resized)
        if self.is_counting and self.first_yellow_time is not None:
            self.elapsed_time = current_time - self.first_yellow_time
            minutes, seconds = divmod(int(self.elapsed_time), 60)
            time_str = f"Elapsed Time: {minutes:02d}:{seconds:02d}"
            if self.show_text:
                cvzone.putTextRect(frame_resized, time_str, (10, self.new_height - 100), scale=1, thickness=2, offset=5)
        return frame_resized

    def main(self):
        process_every_n_frames = 2
        frame_count = 0

        self.frame_thread = threading.Thread(target=self.frame_capture)
        self.frame_thread.daemon = True
        self.frame_thread.start()

        window_name = f"Motor Detector"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, self.window_width, self.window_height)

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
            yellow_borders = sum(1 for state in self.border_states.values() if state["is_yellow"])
            percentage_yellow = (yellow_borders / total_borders) * 100
            frame_resized = self.process_frame(frame, current_time, percentage_yellow)
            if self.show_text:
                cvzone.putTextRect(
                    frame_resized,
                    f"Percentage of Yellow Border: {percentage_yellow:.2f}%",
                    (10, self.new_height - 50),
                    scale=1,
                    thickness=2,
                    offset=5,
                )
                cvzone.putTextRect(frame_resized, f"FPS: {int(self.fps)}", (10, self.new_height - 75), scale=1, thickness=2, offset=5)
            cv2.imshow(window_name, frame_resized)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("n"):
                self.stop_event.set()
                break
            elif key == ord("s"):
                self.show_text = not self.show_text
        cv2.destroyAllWindows()
        self.frame_thread.join()


def run_motor(MOTOR_ABSENCE_THRESHOLD, MOTOR_TOUCH_THRESHOLD, MOTOR_PERCENTAGE_YELLOW_THRESHOLD, MOTOR_CONFIDENCE_THRESHOLD, camera_name, window_size=(540, 360)):
    detector = MotorDetector(
        MOTOR_ABSENCE_THRESHOLD=MOTOR_ABSENCE_THRESHOLD,
        MOTOR_TOUCH_THRESHOLD=MOTOR_TOUCH_THRESHOLD,
        MOTOR_PERCENTAGE_YELLOW_THRESHOLD=MOTOR_PERCENTAGE_YELLOW_THRESHOLD,
        MOTOR_CONFIDENCE_THRESHOLD=MOTOR_CONFIDENCE_THRESHOLD,
        camera_name=camera_name,
        window_size=window_size,
    )

    detector.main()


if __name__ == "__main__":
    run_motor(
        MOTOR_ABSENCE_THRESHOLD=10,
        MOTOR_TOUCH_THRESHOLD=0,
        MOTOR_PERCENTAGE_YELLOW_THRESHOLD=50,
        MOTOR_CONFIDENCE_THRESHOLD=0,
        camera_name="10.5.0.206",
        window_size=(540, 360),
    )
