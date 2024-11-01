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


class BroomDetector:
    def __init__(
        self,
        BROOM_CONFIDENCE_THRESHOLD=0.5,
        rtsp_url=None,
        camera_name=None,
        window_size=(540, 360),
        new_size=(960, 540),
        bbox_duration=5,
    ):
        self.BROOM_CONFIDENCE_THRESHOLD = BROOM_CONFIDENCE_THRESHOLD
        self.window_width, self.window_height = window_size
        self.new_width, self.new_height = new_size
        self.prev_frame_time = 0
        self.fps = 0
        self.bbox_duration = bbox_duration
        self.active_bboxes = []
        self.total_area = 0
        self.camera_name = camera_name
        self.ip_camera = self.camera_config()
        if rtsp_url is not None:
            self.rtsp_url = rtsp_url
            if os.path.isfile(rtsp_url):
                self.is_local_file = True
                cap = cv2.VideoCapture(self.rtsp_url)
                self.video_fps = cap.get(cv2.CAP_PROP_FPS)
                if not self.video_fps or math.isnan(self.video_fps):
                    self.video_fps = 25
                cap.release()
                print(f"Local video file detected. FPS: {self.video_fps}")
            else:
                self.is_local_file = False
                print(f"RTSP stream detected. URL: {self.rtsp_url}")
                self.video_fps = None
        else:
            self.rtsp_url = f"rtsp://admin:oracle2015@{self.ip_camera}:554/Streaming/Channels/1"
            self.video_fps = None
            self.is_local_file = False

        self.frame_queue = queue.Queue(maxsize=10)
        self.stop_event = threading.Event()
        self.frame_thread = None

        self.broom_model = YOLO("broom6l.pt").to("cuda")
        self.broom_model.overrides["verbose"] = False
        print(f"Model Broom device: {next(self.broom_model.model.parameters()).device}")

    def camera_config(self):
        config = {
            "OFFICE1": "10.5.0.170",
            "OFFICE2": "10.5.0.182",
            "OFFICE3": "10.5.0.161",
        }
        return config[self.camera_name]

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
                    print("Failed to read frame. Reconnecting in 5 seconds...")
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
            results = self.broom_model(frame)
        return results

    def export_frame(self, results, current_time):
        boxes_info = []
        self.total_area = 0
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = box.conf[0]
                class_id = self.broom_model.names[int(box.cls[0])]
                if conf > self.BROOM_CONFIDENCE_THRESHOLD:
                    area = (x2 - x1) * (y2 - y1)
                    self.total_area += area
                    self.active_bboxes.append(((x1, y1, x2, y2), current_time))
                    boxes_info.append((x1, y1, x2, y2, area, class_id))
        return boxes_info

    def draw_segments(self, frame, current_time):
        overlay = frame.copy()
        for (x1, y1, x2, y2), start_time in self.active_bboxes:
            if current_time - start_time < self.bbox_duration:
                cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), -1)
            else:
                self.active_bboxes.remove(((x1, y1, x2, y2), start_time))
        cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

    def process_frame(self, frame, current_time):
        frame_resized = cv2.resize(frame, (self.new_width, self.new_height))
        results = self.process_model(frame_resized)
        boxes_info = self.export_frame(results, current_time)
        if boxes_info:
            for x1, y1, x2, y2, area, class_id in boxes_info:
                cvzone.cornerRect(frame_resized, (x1, y1, x2 - x1, y2 - y1), l=10, t=2, colorR=(0, 255, 255), colorC=(255, 255, 255))
                cvzone.putTextRect(frame_resized, f"{class_id} {area:.2f}", (x1, y1 + 6), scale=0.5, thickness=1, offset=0, colorR=(0, 255, 255), colorT=(0, 0, 0))
        self.draw_segments(frame_resized, current_time)
        return frame_resized

    def main(self):
        process_every_n_frames = 2
        frame_count = 0

        window_name = f"Broom Detection"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, self.window_width, self.window_height)

        if self.video_fps is not None:
            cap = cv2.VideoCapture(self.rtsp_url)
            frame_delay = int(1000 / self.video_fps)

            while cap.isOpened():
                start_time = time.time()
                ret, frame = cap.read()
                if not ret:
                    break
                frame_count += 1
                if frame_count % process_every_n_frames != 0:
                    continue
                current_time = time.time()
                time_diff = current_time - self.prev_frame_time
                self.fps = 1 / time_diff if time_diff > 0 else 0
                self.prev_frame_time = current_time
                frame_resized = self.process_frame(frame, current_time)

                cvzone.putTextRect(frame_resized, f"Total Area: {self.total_area}", (10, 50), scale=1, thickness=2, offset=5)
                cvzone.putTextRect(frame_resized, f"FPS: {int(self.fps)}", (10, 75), scale=1, thickness=2, offset=5)
                cv2.imshow(window_name, frame_resized)
                processing_time = (time.time() - start_time) * 1000
                adjusted_delay = max(int(frame_delay - processing_time), 1)
                key = cv2.waitKey(adjusted_delay) & 0xFF
                if key == ord("n"):
                    break
            cap.release()
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
                self.fps = 1/time_diff if time_diff > 0 else 0
                self.prev_frame_time = current_time
                frame_resized = self.process_frame(frame, current_time)
                cvzone.putTextRect(frame_resized, f"Total Area: {self.total_area}", (10, 50), scale=1, thickness=2, offset=5)
                cvzone.putTextRect(frame_resized, f"FPS: {int(self.fps)}", (10, 75), scale=1, thickness=2, offset=5)
                cv2.imshow(window_name, frame_resized)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("n"):
                    self.stop_event.set()
                    break
            cv2.destroyAllWindows()
            self.frame_thread.join()


def run_broom(camera_name, window_size=(540, 360), rtsp_url=None):
    detector = BroomDetector(
        camera_name=camera_name,
        rtsp_url=rtsp_url,
        window_size=window_size,
    )
    detector.main()


if __name__ == "__main__":
    run_broom(
        camera_name="OFFICE2",
        window_size=(960, 540),
        # rtsp_url="D:/NWR/videos/test/broom_test_0002.mp4",
        rtsp_url="videos/test1.mp4"
    )
