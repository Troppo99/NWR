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
import queue


class opencvSection:
    def __init__(self, video):
        self.video = video
        self.cap = cv2.VideoCapture(video)

    def connect_to_stream(self):
        while not self.cap.isOpened():
            self.cap.release()
            time.sleep(5)
            self.cap = cv2.VideoCapture(self.video)
        return self.cap

    def read_frame(self, frame_queue, stop_flag):
        while not stop_flag.is_set():
            ret, frame = self.cap.read()
            if not ret:
                self.cap.release()
                time.sleep(5)
                self.cap = self.connect_to_stream()
                continue
            try:
                frame_queue.put(frame, block=False)
            except queue.Full:
                pass


class yoloSection:
    def __init__(self, model_broom):
        self.model_broom = model_broom

    def process_model_broom(self, frame):
        with torch.no_grad():
            results_broom = self.model_broom(frame, imgsz=960)
        return results_broom

    def export_frame_broom(self, results, color, pairs, confidence_threshold=CONFIDENCE_THRESHOLD_BROOM):
        points = []
        coords = []
        keypoint_positions = []

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


class serverSection:
    def __init__(self, host, user, password, database, port):
        self.host = host
        self.user = user
        self.password = password
        self.database = database
        self.port = port

    def send_to_server(self, percentage_green, elapsed_time, image_path, table="empbro", camera_name="10.5.0.182"):
        try:
            connection = pymysql.connect(
                host=self.host, user=self.user, password=self.password, database=self.database, port=self.port
            )
            cursor = connection.cursor()
            timestamp_done = datetime.now()
            timestamp_start = timestamp_done - timedelta(seconds=elapsed_time)
            timestamp_done_str = timestamp_done.strftime("%Y-%m-%d %H:%M:%S")
            timestamp_start_str = timestamp_start.strftime("%Y-%m-%d %H:%M:%S")

            with open(image_path, "rb") as file:
                binary_image = file.read()

            query = f"""
            INSERT INTO {table} (cam, timestamp_start, timestamp_done, elapsed_time, percentage, image_done)
            VALUES (%s, %s, %s, %s, %s, %s)
            """
            cursor.execute(
                query,
                (
                    camera_name,
                    timestamp_start_str,
                    timestamp_done_str,
                    elapsed_time,
                    percentage_green,
                    binary_image,
                ),
            )
            connection.commit()
            print(f"Data berhasil dikirim ")
        except pymysql.MySQLError as e:
            print(f"Error saat mengirim data : {e}")
        finally:
            if "cursor" in locals():
                cursor.close()
            if "connection" in locals():
                connection.close()

def run_detection(video_path):
    pass

def thread_read_frame(video_path):
    pass    

def camera(name):
    configurations = {
        "10.5.0.161": "161",
        "10.5.0.170": "170",
        "10.5.0.182": "182",
    }
    video = f"rtsp://admin:robot123@10.5.0.{configurations[name][0]}/Streaming/Channels/1"
    model = YOLO("broom5l.pt")
    return video, model
