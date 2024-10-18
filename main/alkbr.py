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
                frame_queue.put((self.video, frame), block=False)
            except queue.Full:
                pass

        self.cap.release()  # Pastikan untuk melepaskan sumber daya kamera saat thread berhenti


def run_detection(camera_name, frame_queue, stop_flag):
    # Menghubungkan ke kamera dan memulai streaming
    video_path, _ = camera(camera_name)
    opencv = opencvSection(video_path)
    cap = opencv.connect_to_stream()

    # Mengambil frame secara berulang dan memasukkannya ke queue
    opencv.read_frame(frame_queue, stop_flag)


def display_frames(frame_queue, stop_flag, threads):
    # Menampilkan frame di thread utama
    while not stop_flag.is_set():
        if not frame_queue.empty():
            camera_name, frame = frame_queue.get()
            # Tampilkan frame pada window yang berbeda untuk setiap kamera
            cv2.imshow(f"ALKBR TESTING - {camera_name}", frame)

            # Jika tombol 'n' ditekan, stop_flag diatur untuk menghentikan semua operasi
            if cv2.waitKey(1) & 0xFF == ord("n"):
                stop_flag.set()
                break

    # Tunggu semua thread pengambil frame selesai sebelum keluar
    for thread in threads:
        thread.join()

    # Tutup semua jendela OpenCV
    cv2.destroyAllWindows()


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
    frame_queue = queue.Queue(maxsize=20)
    stop_flag = threading.Event()
    threads = []

    # Membuat thread untuk pengambilan frame setiap kamera
    for camera_name in camera_names:
        thread = threading.Thread(target=run_detection, args=(camera_name, frame_queue, stop_flag))
        threads.append(thread)
        thread.start()

    # Mengelola display frame di thread utama
    try:
        display_frames(frame_queue, stop_flag, threads)
    finally:
        stop_flag.set()  # Set stop flag agar semua thread berhenti
        for thread in threads:
            thread.join()  # Pastikan semua thread telah dihentikan dengan benar
