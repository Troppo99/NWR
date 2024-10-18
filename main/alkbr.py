import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import cv2
from ultralytics import YOLO
import numpy as np
import time
import queue
import threading
import torch
import cvzone


# Konfigurasi umum
CONFIDENCE_THRESHOLD_BROOM = 0.9
new_width, new_height = 960, 540
scale_x = new_width / 1280
scale_y = new_height / 720
pairs_broom = [(0, 1), (1, 2), (2, 3), (2, 4)]


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

    def read_frame(self, frame_queue, model, stop_flag):
        while not stop_flag.is_set():
            ret, frame = self.cap.read()
            if not ret:
                self.cap.release()
                time.sleep(5)
                self.cap = self.connect_to_stream()
                continue

            # Resize frame untuk efisiensi deteksi
            frame_resized = cv2.resize(frame, (new_width, new_height))

            # Proses deteksi menggunakan model YOLO
            with torch.no_grad():
                results_broom = model(frame_resized, imgsz=960)
            points, coords, _ = export_frame_broom(results_broom, (0, 255, 0), pairs_broom)

            # Tambahkan hasil deteksi ke frame
            if points and coords:
                for x, y, color in coords:
                    cv2.line(frame_resized, x, y, color, 2)
                for point in points:
                    cv2.circle(frame_resized, point, 4, (0, 255, 255), -1)

            try:
                # Memasukkan frame yang telah dianalisis ke dalam queue untuk diproses di thread utama
                frame_queue.put((self.video, frame_resized), block=False)
            except queue.Full:
                pass

        self.cap.release()


def run_detection(camera_name, frame_queue, stop_flag):
    video_path, model = camera(camera_name)
    opencv = opencvSection(video_path)
    cap = opencv.connect_to_stream()

    # Memulai membaca frame dan melakukan inferensi
    opencv.read_frame(frame_queue, model, stop_flag)


def display_frames(frame_queue, stop_flag, threads):
    # Menampilkan frame di thread utama
    while not stop_flag.is_set():
        if not frame_queue.empty():
            camera_name, frame = frame_queue.get()
            # Tampilkan frame pada window yang berbeda untuk setiap kamera
            cv2.imshow(f"ALKBR TESTING - {camera_name}", frame)
            if cv2.waitKey(1) & 0xFF == ord("n"):
                stop_flag.set()
                break

    # Tunggu semua thread pengambil frame selesai sebelum keluar
    for thread in threads:
        thread.join()

    cv2.destroyAllWindows()


def camera(name):
    configurations = {
        "10.5.0.161": "161",
        "10.5.0.170": "170",
        "10.5.0.182": "182",
    }
    video = f"rtsp://admin:oracle2015@10.5.0.{configurations[name]}:554/Streaming/Channels/1"
    model = YOLO("broom5l.pt").to("cuda")  # Menggunakan model YOLO dan mengarahkannya ke GPU untuk performa lebih baik
    model.overrides["verbose"] = False
    return video, model


def export_frame_broom(results, color, pairs, confidence_threshold=CONFIDENCE_THRESHOLD_BROOM):
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


if __name__ == "__main__":
    camera_names = ["10.5.0.161", "10.5.0.170", "10.5.0.182"]
    frame_queue = queue.Queue(maxsize=20)
    stop_flag = threading.Event()
    threads = []

    # Membuat thread untuk pengambilan frame dan melakukan inferensi pada setiap kamera
    for camera_name in camera_names:
        thread = threading.Thread(target=run_detection, args=(camera_name, frame_queue, stop_flag))
        threads.append(thread)
        thread.start()

    # Mengelola display frame di thread utama
    try:
        display_frames(frame_queue, stop_flag, threads)
    finally:
        stop_flag.set()
        for thread in threads:
            thread.join()
