import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import cv2
from ultralytics import YOLO
import numpy as np
import time
import queue
import threading


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


def run_detection(camera_name, stop_flag=None):
    video_path, _ = camera(camera_name)
    frame_queue = queue.Queue(maxsize=10)
    opencv = opencvSection(video_path)
    cap = opencv.connect_to_stream()
    thread = thread_read_frame(opencv, frame_queue, stop_flag)
    while not stop_flag.is_set():
        if not frame_queue.empty():
            frame = frame_queue.get()
            cv2.imshow(f"ALKBR TESTING - {camera_name}", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("n"):
                stop_flag.set()
                break
    cap.release()
    cv2.destroyAllWindows()
    thread.join()


def thread_read_frame(opencv, frame_queue, stop_flag):
    thread = threading.Thread(target=opencv.read_frame, args=(frame_queue, stop_flag))
    thread.daemon = True
    thread.start()
    return thread


def camera(name):
    configurations = {
        "10.5.0.161": "161",
        "10.5.0.170": "170",
        "10.5.0.182": "182",
    }
    video = f"rtsp://admin:oracle2015@10.5.0.{configurations[name]}:554/Streaming/Channels/1"
    model = YOLO("broom5l.pt")
    return video, model


if __name__ == "__main__":
    camera_names = ["10.5.0.161", "10.5.0.170", "10.5.0.182"]
    threads = []
    stop_flags = []

    for camera_name in camera_names:
        stop_flag = threading.Event()
        stop_flags.append(stop_flag)
        thread = threading.Thread(target=run_detection, args=(camera_name, stop_flag))
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()
