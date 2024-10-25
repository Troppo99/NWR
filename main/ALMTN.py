import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import cv2
from ultralytics import YOLO
import time
import torch
import cvzone
import threading
import queue
import math


class MotorDetector:
    def __init__(
        self,
        MOTOR_CONFIDENCE_THRESHOLD=0.5,
        motor_model="yolo11l.pt",
        camera_name=None,
        new_size=(960, 540),
        rtsp_url=None,
        window_size=(540, 360),
    ):
        self.MOTOR_CONFIDENCE_THRESHOLD = MOTOR_CONFIDENCE_THRESHOLD
        self.window_width, self.window_height = window_size
        self.new_width, self.new_height = new_size
        self.scale_x = self.new_width / 1280
        self.scale_y = self.new_height / 720
        self.camera_name = camera_name
        if rtsp_url is None:
            self.rtsp_url = f"rtsp://admin:oracle2015@{camera_name}:554/Streaming/Channels/1"
        else:
            self.rtsp_url = rtsp_url
        self.frame_queue = queue.Queue(maxsize=10)
        self.stop_event = threading.Event()
        self.frame_thread = None
        self.motor_model = YOLO(motor_model).to("cuda")
        self.motor_model.overrides["verbose"] = False
        print(f"Model motor device: {next(self.motor_model.model.parameters()).device}")

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

    def process_model(self, frame):
        with torch.no_grad():
            results = self.motor_model(frame, stream=True, imgsz=1280)
        return results

    def export_frame(self, results):
        boxes_info = []
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = math.ceil(box.conf[0] * 100) / 100
                class_id = self.motor_model.names[int(box.cls[0])]
                if conf > self.MOTOR_CONFIDENCE_THRESHOLD and class_id == "motorcycle":
                    boxes_info.append((x1, y1, x2, y2, conf, class_id))
        return boxes_info

    def process_frame(self, frame):
        frame_resized = cv2.resize(frame, (self.new_width, self.new_height))
        results = self.process_model(frame_resized)
        boxes_info = self.export_frame(results)
        return frame_resized, boxes_info

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

            frame_resized, boxes_info = self.process_frame(frame)
            for x1, y1, x2, y2, conf, class_id in boxes_info:
                cvzone.putTextRect(frame_resized, f"{class_id} {conf:.2f}", (x1, y1), scale=1, thickness=2, offset=5)
                cvzone.cornerRect(frame_resized, (x1, y1, x2 - x1, y2 - y1), l=10, t=2, colorR=(0, 255, 0))
            cv2.imshow(window_name, frame_resized)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("n"):
                self.stop_event.set()
                break
            elif key == ord("s"):
                self.show_text = not self.show_text
        cv2.destroyAllWindows()
        self.frame_thread.join()


def run_motor(MOTOR_CONFIDENCE_THRESHOLD, camera_name, window_size=(540, 360)):
    detector = MotorDetector(
        MOTOR_CONFIDENCE_THRESHOLD=MOTOR_CONFIDENCE_THRESHOLD,
        camera_name=camera_name,
        window_size=window_size,
    )

    detector.main()


if __name__ == "__main__":
    run_motor(
        MOTOR_CONFIDENCE_THRESHOLD=0,
        camera_name="10.5.0.206",
        window_size=(540, 360),
    )
