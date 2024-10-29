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
        MOTOR_TOUCH_THRESHOLD=3,
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
        self.window_width, self.window_height = window_size
        self.new_width, self.new_height = new_size
        self.scale_x = self.new_width / 1280
        self.scale_y = self.new_height / 720
        self.scaled_borders = []
        self.prev_frame_time = time.time()
        self.fps = 0
        self.camera_name = camera_name
        self.rtsp_url = rtsp_url
        self.video_fps = None  # Initialize video FPS
        self.is_local_file = False  # Flag to indicate if rtsp_url is a local file
        self.borders, self.ip_camera, self.idx = self.camera_config()

        if rtsp_url is not None:
            self.rtsp_url = rtsp_url  # Keep the provided rtsp_url
            if os.path.isfile(rtsp_url):
                # It's a local file
                self.is_local_file = True
                cap = cv2.VideoCapture(self.rtsp_url)
                self.video_fps = cap.get(cv2.CAP_PROP_FPS)
                if not self.video_fps or math.isnan(self.video_fps):
                    self.video_fps = 25  # Default FPS if unable to get
                cap.release()
                print(f"Local video file detected. FPS: {self.video_fps}")
            else:
                # Assume it's an RTSP stream
                self.is_local_file = False
                print(f"RTSP stream detected. URL: {self.rtsp_url}")
                self.video_fps = None
        else:
            # rtsp_url is None
            self.rtsp_url = f"rtsp://admin:oracle2015@{self.ip_camera}:554/Streaming/Channels/1"
            self.video_fps = None
            self.is_local_file = False

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
                "is_yellow": False,
                "motor_overlap_time": 0.0,
                "last_motor_overlap_time": None,
                "motor_absence_timer_start": None,
                "first_yellow_time": None,
                "is_counting": False,
                "image_start": None,
                "image_done": None,
                "db_id": None,  # Add this line
            }
            for idx in range(len(self.borders))
        }

        self.borders_pts = [np.array(border, np.int32) for border in self.scaled_borders]
        self.motor_model = YOLO(motor_model).to("cuda")
        self.motor_model.overrides["verbose"] = False
        print(f"Model Motor device: {next(self.motor_model.model.parameters()).device}")

    def camera_config(self):
        config = {
            "EXPEDISI2": {
                "borders": [
                    [(0, 719), (0, 429), (413, 429), (414, 722)],
                    [(414, 722), (413, 429), (844, 423), (867, 721)],
                    [(867, 721), (844, 423), (1281, 420), (1288, 744)],
                    [(0, 429), (0, 391), (179, 266), (465, 259), (413, 429)],
                    [(413, 429), (465, 259), (861, 262), (844, 423)],
                    [(844, 423), (861, 262), (1248, 287), (1276, 302), (1281, 420)],
                    [(179, 266), (280, 196), (492, 189), (465, 259)],
                    [(465, 259), (492, 189), (853, 205), (861, 262)],
                    [(861, 262), (853, 205), (915, 215), (918, 201), (1069, 221), (1156, 241), (1248, 287)],
                ],
                "ip": "10.5.0.206",
            },
            "KANTIN2": {
                "borders": [
                    [(619, 1), (689, -1), (694, 47), (603, 48)],
                    [(748, 1), (689, -1), (694, 47), (788, 47)],
                    [(603, 48), (694, 47), (698, 126), (572, 123)],
                    [(788, 47), (694, 47), (698, 126), (847, 124)],
                    [(572, 123), (698, 126), (704, 236), (536, 231)],
                    [(847, 124), (698, 126), (704, 236), (923, 235)],
                    [(536, 231), (704, 236), (717, 401), (475, 401)],
                    [(923, 235), (704, 236), (717, 401), (1021, 400)],
                    [(475, 401), (717, 401), (730, 712), (383, 714)],
                    [(1021, 400), (717, 401), (730, 712), (1164, 712)],
                ],
                "ip": "10.5.0.239",
            },
            "KANTIN1": {
                "borders": [
                    [(-3, 549), (126, 584), (127, 716), (1, 716)],
                    [(126, 584), (392, 618), (393, 715), (127, 716)],
                    [(392, 618), (657, 626), (657, 714), (393, 715)],
                    [(657, 626), (902, 612), (902, 716), (657, 714)],
                    [(902, 612), (1112, 582), (1112, 714), (902, 716)],
                    [(1112, 582), (1278, 543), (1279, 715), (1112, 714)],
                ],
                "ip": "10.5.0.228",
            },
            "HALAMAN1": {
                "borders": [
                    [(1, 716), (2, 592), (137, 592), (135, 716)],
                    [(135, 716), (137, 592), (270, 591), (268, 718)],
                    [(268, 718), (270, 591), (404, 593), (405, 718)],
                    [(405, 718), (404, 593), (529, 594), (531, 719)],
                    [(531, 719), (529, 594), (636, 594), (638, 717)],
                    [(638, 717), (636, 594), (745, 595), (745, 717)],
                    [(745, 717), (745, 595), (862, 597), (866, 719)],
                    [(866, 719), (862, 597), (990, 596), (993, 718)],
                    [(993, 718), (990, 596), (1128, 599), (1129, 716)],
                    [(1129, 716), (1128, 599), (1277, 601), (1277, 716)],
                    [(1128, 599), (1128, 513), (1276, 547), (1277, 601)],
                    [(990, 596), (989, 484), (1128, 513), (1128, 599)],
                    [(862, 597), (860, 454), (989, 484), (990, 596)],
                    [(745, 595), (748, 482), (859, 485), (862, 597)],
                    [(860, 454), (859, 485), (748, 482), (751, 428)],
                    [(745, 595), (636, 594), (641, 480), (748, 482)],
                    [(529, 594), (529, 594), (528, 475), (641, 480), (636, 594)],
                    [(404, 593), (406, 476), (528, 475), (529, 594)],
                    [(270, 591), (276, 473), (406, 476), (404, 593)],
                    [(137, 592), (142, 472), (276, 473), (270, 591)],
                    [(2, 592), (6, 483), (142, 472), (137, 592)],
                    [(641, 480), (642, 401), (751, 428), (748, 482)],
                    [(528, 475), (530, 374), (642, 401), (641, 480)],
                    [(406, 476), (409, 338), (530, 374), (528, 475)],
                    [(276, 473), (278, 423), (321, 414), (323, 396), (307, 376), (285, 368), (288, 310), (409, 338), (406, 476)],
                    [(142, 472), (149, 359), (218, 412), (278, 423), (276, 473)],
                    [(6, 483), (5, 351), (123, 340), (149, 359), (142, 472)],
                    [(5, 351), (2, 290), (106, 273), (107, 306), (107, 327), (123, 340)],
                ],
                "ip": "10.5.0.236",
            },
            "GERBANG1": {
                "borders": [],
                "ip": "10.5.0.245",
            },
        }

        borders = config[self.camera_name]["borders"]
        ip = config[self.camera_name]["ip"]
        camera_names = list(config.keys())
        indices = {name: idx + 1 for idx, name in enumerate(camera_names)}
        index = indices[self.camera_name]
        return borders, ip, index

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
                    print("Failed to read stream. Retrying...")
                    cap.release()
                    time.sleep(5)
                    break
                try:
                    self.frame_queue.put(frame, block=False)
                except queue.Full:
                    pass
            cap.release()

    def send_to_server(
        self,
        host,
        category,
        border_id,
        timestamp_start,
        timestamp_done,
        image_start,
        image_done,
        db_id=None,
    ):
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
            camera_name = self.camera_name

            if db_id is None:
                # Insert new record
                query = f"""
                INSERT INTO {table} (cam, category, border, timestamp_start, image_start)
                VALUES (%s, %s, %s, %s, %s)
                """
                data = (
                    camera_name,
                    category,
                    border_id,
                    timestamp_start.strftime("%Y-%m-%d %H:%M:%S") if timestamp_start else None,
                    image_start,
                )
                cursor.execute(query, data)
                connection.commit()
                db_id = cursor.lastrowid  # Get the auto-incremented ID
                print(f"Data successfully inserted to server for border {border_id} with ID {db_id}.")
                return db_id  # Return the db_id
            else:
                # Update existing record
                query = f"""
                UPDATE {table}
                SET timestamp_done=%s, image_done=%s
                WHERE id=%s
                """
                data = (
                    timestamp_done.strftime("%Y-%m-%d %H:%M:%S") if timestamp_done else None,
                    image_done,
                    db_id,
                )
                cursor.execute(query, data)
                connection.commit()
                print(f"Data successfully updated to server for border {border_id} with ID {db_id}.")
                return db_id
        except pymysql.MySQLError as e:
            print(f"Error sending data to server: {e}")
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

    def process_frame(self, frame, current_time):
        frame_resized = cv2.resize(frame, (self.new_width, self.new_height))
        results = self.process_model(frame_resized)
        boxes_info = self.export_frame(results)

        border_colors = []

        for border_id, border_pt in enumerate(self.borders_pts):
            motor_overlapping = False
            border_polygon = np.array(border_pt, dtype=np.int32)
            border_contour = border_polygon.reshape((-1, 1, 2))

            for x1, y1, x2, y2, conf, class_id in boxes_info:
                if class_id == "motorcycle":
                    # Create polygon from bounding box
                    box_polygon = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.int32)
                    box_contour = box_polygon.reshape((-1, 1, 2))

                    # Compute intersection area between bounding box and border
                    intersection = cv2.intersectConvexConvex(border_contour, box_contour)[1]
                    if intersection is not None and cv2.contourArea(intersection) > 0:
                        motor_overlapping = True
                        break

            state = self.border_states[border_id]

            if motor_overlapping:
                if state["last_motor_overlap_time"] is None:
                    state["last_motor_overlap_time"] = current_time
                else:
                    delta_time = current_time - state["last_motor_overlap_time"]
                    state["motor_overlap_time"] += delta_time
                    state["last_motor_overlap_time"] = current_time

                if state["motor_overlap_time"] >= self.MOTOR_TOUCH_THRESHOLD:
                    if not state["is_yellow"]:
                        # Border turns yellow for the first time
                        state["is_yellow"] = True
                        state["first_yellow_time"] = current_time
                        state["is_counting"] = True

                        # Capture image_start
                        image_start = frame_resized.copy()
                        overlay = image_start.copy()
                        alpha = 0.5
                        cv2.fillPoly(overlay, pts=[border_pt], color=(0, 255, 255))
                        cv2.addWeighted(overlay, alpha, image_start, 1 - alpha, 0, image_start)
                        image_start_path = f"main/images/border_{border_id}_start.jpg"
                        cv2.imwrite(image_start_path, image_start)
                        with open(image_start_path, "rb") as file:
                            binary_image_start = file.read()

                        # Send data to server with image_start and get db_id
                        db_id = self.send_to_server(
                            host="10.5.0.2",
                            category="Motorcycle Detected",
                            border_id=border_id,
                            timestamp_start=datetime.now(),
                            timestamp_done=None,
                            image_start=binary_image_start,
                            image_done=None,
                            db_id=None,
                        )
                        # Store db_id in state
                        state["db_id"] = db_id

                # Reset absence timer
                state["motor_absence_timer_start"] = None

            else:
                if state["last_motor_overlap_time"] is not None:
                    state["last_motor_overlap_time"] = None
                    # Start absence timer
                    state["motor_absence_timer_start"] = current_time

                elif state["motor_absence_timer_start"] is not None:
                    absence_time = current_time - state["motor_absence_timer_start"]
                    if absence_time >= self.MOTOR_ABSENCE_THRESHOLD:
                        if not state["is_yellow"]:
                            # Reset accumulated overlap time
                            state["motor_overlap_time"] = 0.0
                            state["motor_absence_timer_start"] = None
                        else:
                            # Violation ends, reset the border
                            state["is_yellow"] = False
                            state["motor_overlap_time"] = 0.0
                            state["last_motor_overlap_time"] = None
                            state["motor_absence_timer_start"] = None

                            # Capture image_done
                            image_done = frame_resized.copy()
                            overlay = image_done.copy()
                            alpha = 0.5
                            cv2.fillPoly(overlay, pts=[border_pt], color=(0, 255, 0))
                            cv2.addWeighted(overlay, alpha, image_done, 1 - alpha, 0, image_done)
                            image_done_path = f"main/images/border_{border_id}_done.jpg"
                            cv2.imwrite(image_done_path, image_done)
                            with open(image_done_path, "rb") as file:
                                binary_image_done = file.read()

                            # Send data to server with image_done
                            if state["db_id"] is not None:
                                self.send_to_server(
                                    host="10.5.0.2",
                                    category="Motorcycle Detected",
                                    border_id=border_id,
                                    timestamp_start=None,
                                    timestamp_done=datetime.now(),
                                    image_start=None,
                                    image_done=binary_image_done,
                                    db_id=state["db_id"],  # Use the db_id to update the record
                                )
                                # Clear db_id after updating
                                state["db_id"] = None
                            else:
                                print(f"No db_id found for border {border_id} when ending violation.")

                            # Reset counting variables
                            state["first_yellow_time"] = None
                            state["is_counting"] = False

            # Update border_colors
            if state["is_yellow"]:
                border_colors.append((0, 255, 255))
            else:
                border_colors.append((0, 255, 0))

        # Drawing boxes and overlays
        if boxes_info:
            for x1, y1, x2, y2, conf, class_id in boxes_info:
                # Draw bounding boxes
                cvzone.cornerRect(
                    frame_resized,
                    (x1, y1, x2 - x1, y2 - y1),
                    l=10,
                    t=2,
                    colorR=(0, 255, 255),
                    colorC=(255, 255, 255),
                )

        overlay = frame_resized.copy()
        alpha = 0.5
        for border_pt, color in zip(self.borders_pts, border_colors):
            cv2.fillPoly(overlay, pts=[border_pt], color=color)
        cv2.addWeighted(overlay, alpha, frame_resized, 1 - alpha, 0, frame_resized)

        # Optionally, display per-border elapsed times
        if self.show_text:
            for border_id, state in self.border_states.items():
                if state["is_counting"]:
                    elapsed_time = current_time - state["first_yellow_time"]
                    minutes, seconds = divmod(int(elapsed_time), 60)
                    time_str = f"Border {border_id}: {minutes:02d}:{seconds:02d}"
                    cvzone.putTextRect(
                        frame_resized,
                        time_str,
                        (10, self.new_height - 100 - 25 * border_id),
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

        return frame_resized

    def main(self):
        process_every_n_frames = 2
        frame_count = 0

        window_name = f"MOTOR{self.idx} : {self.camera_name}"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, self.window_width, self.window_height)

        if self.video_fps is not None:
            # Local video file, process frames in the main thread
            cap = cv2.VideoCapture(self.rtsp_url)
            frame_delay = int(1000 / self.video_fps)  # Delay in milliseconds

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
                if self.show_text:
                    cvzone.putTextRect(
                        frame_resized,
                        f"FPS: {int(self.fps)}",
                        (10, self.new_height - 75),
                        scale=1,
                        thickness=2,
                        offset=5,
                    )
                cv2.imshow(window_name, frame_resized)
                processing_time = (time.time() - start_time) * 1000  # Convert to milliseconds
                adjusted_delay = max(int(frame_delay - processing_time), 1)
                key = cv2.waitKey(adjusted_delay) & 0xFF
                if key == ord("n"):
                    break
                elif key == ord("s"):
                    self.show_text = not self.show_text
            cap.release()
            cv2.destroyAllWindows()
        else:
            # RTSP stream, use frame capture thread
            self.frame_thread = threading.Thread(target=self.frame_capture)
            self.frame_thread.daemon = True
            self.frame_thread.start()

            while True:
                start_time = time.time()
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
                if self.show_text:
                    cvzone.putTextRect(
                        frame_resized,
                        f"FPS: {int(self.fps)}",
                        (10, self.new_height - 75),
                        scale=1,
                        thickness=2,
                        offset=5,
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


def run_motor(MOTOR_ABSENCE_THRESHOLD, MOTOR_TOUCH_THRESHOLD, MOTOR_CONFIDENCE_THRESHOLD, camera_name, window_size=(540, 360), rtsp_url=None):
    detector = MotorDetector(
        MOTOR_ABSENCE_THRESHOLD=MOTOR_ABSENCE_THRESHOLD,
        MOTOR_TOUCH_THRESHOLD=MOTOR_TOUCH_THRESHOLD,
        MOTOR_CONFIDENCE_THRESHOLD=MOTOR_CONFIDENCE_THRESHOLD,
        camera_name=camera_name,
        window_size=window_size,
        rtsp_url=rtsp_url,
    )

    detector.main()


if __name__ == "__main__":
    run_motor(
        MOTOR_ABSENCE_THRESHOLD=3,
        MOTOR_TOUCH_THRESHOLD=3,
        MOTOR_CONFIDENCE_THRESHOLD=0,
        camera_name="EXPEDISI2",
        window_size=(540, 360),
        # rtsp_url="videos/1028(2).mp4",
    )
