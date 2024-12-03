import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import cv2
import cvzone
from ultralytics import YOLO
import math
import numpy as np
import time
import threading
import queue
import torch
import pymysql
from datetime import datetime
import math


class PuddleDetector:
    def __init__(
        self,
        PUDDLE_CONFIDENCE_THRESHOLD=0.5,
        puddle_model="models/puddle.pt",
        camera_name=None,
        new_size=(960, 540),
        rtsp_url=None,
        window_size=(960, 540),
        icon_paths={
            "check_icon": "D:/SBHNL/Images/AHMDL/Icon/check_icon.png",
            "puddle_icon": "D:/SBHNL/Images/AHMDL/Icon/puddle_icon.png",
        },
    ):
        self.PUDDLE_CONFIDENCE_THRESHOLD = PUDDLE_CONFIDENCE_THRESHOLD
        self.window_width, self.window_height = window_size
        self.new_width, self.new_height = new_size
        self.scale_x = self.new_width / 1280
        self.scale_y = self.new_height / 720
        self.prev_frame_time = time.time()
        self.fps = 0
        self.camera_name = camera_name
        self.rtsp_url = rtsp_url
        self.video_fps = None
        self.is_local_file = False
        self.borders, self.ip_camera, self.idx = self.camera_config()

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

        self.show_text = True
        self.frame_queue = queue.Queue(maxsize=10)
        self.stop_event = threading.Event()
        self.frame_thread = None

        self.puddle_model = YOLO(puddle_model).to("cuda")
        self.puddle_model.overrides["verbose"] = False
        print(f"Model Puddle device: {next(self.puddle_model.model.parameters()).device}")

        if icon_paths is None:
            icon_paths = {
                "check_icon": "check_icon.png",
                "puddle_icon": "puddle_icon.png",
            }

        # icon read
        check_icon = cv2.imread(icon_paths["check_icon"], cv2.IMREAD_UNCHANGED)
        puddle_icon = cv2.imread(icon_paths["puddle_icon"], cv2.IMREAD_UNCHANGED)

        # icon size config
        icon_height = 100
        icon_width = int(check_icon.shape[1] * (icon_height / check_icon.shape[0]))

        # Resize all icons to the same dimensions
        self.check_icon_resized = cv2.resize(check_icon, (icon_width, icon_height))
        self.puddle_icon_resized = cv2.resize(puddle_icon, (icon_width, icon_height))

        # bgr and alpha config
        self.check_bgr = self.check_icon_resized[:, :, :3]
        self.check_alpha = self.check_icon_resized[:, :, 3] / 255.0

        self.puddle_bgr = self.puddle_icon_resized[:, :, :3]
        self.puddle_alpha = self.puddle_icon_resized[:, :, 3] / 255.0

    def camera_config(self):
        config = {
            "EKSPEDISI2": {
                "borders": [],
                "ip": "10.5.0.206",
            },
            "HALAMANDEPAN1": {
                "borders": [],
                "ip": "10.5.0.236",
            },
            "KANTIN1": {
                "borders": [],
                "ip": "10.5.0.228",
            },
            "KANTIN2": {
                "borders": [],
                "ip": "10.5.0.239",
            },
            "HALAMANBELAKANG1": {
                "borders": [],
                "ip": "10.5.0.61",
            },
            "HALAMANBELAKANG2": {
                "borders": [],
                "ip": "10.5.0.43",
            },
        }
        borders = config[self.camera_name]["borders"]
        ip = config[self.camera_name]["ip"]
        camera_names = list(config.keys())
        indices = {name: idx + 1 for idx, name in enumerate(camera_names)}
        index = indices.get(self.camera_name, 0)
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

    def process_model(self, frame):
        with torch.no_grad():
            results = self.puddle_model(frame, stream=True, imgsz=960)
        return results

    def export_frame(self, results):
        boxes_info = []
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = box.conf[0]
                class_id = self.puddle_model.names[int(box.cls[0])]
                if conf > self.PUDDLE_CONFIDENCE_THRESHOLD and class_id == "puddle":
                    boxes_info.append((x1, y1, x2, y2, conf, class_id))
        return boxes_info

    def overlay_image_alpha(self, img, img_overlay, pos, alpha_mask):
        x, y = pos
        h, w = img_overlay.shape[0], img_overlay.shape[1]
        if y + h > img.shape[0] or x + w > img.shape[1]:
            h = min(h, img.shape[0] - y)
            w = min(w, img.shape[1] - x)
            img_overlay = img_overlay[:h, :w]
            alpha_mask = alpha_mask[:h, :w]
        img[y : y + h, x : x + w] = (img[y : y + h, x : x + w] * (1 - alpha_mask[:, :, None]) + img_overlay * alpha_mask[:, :, None]).astype(np.uint8)

    def process_frame(self, frame):
        frame_resized = cv2.resize(frame, (self.new_width, self.new_height))
        results = self.process_model(frame_resized)
        boxes_info = self.export_frame(results)

        total_objects = len(boxes_info)
        icon_width = self.check_icon_resized.shape[1]
        text_x_position = icon_width

        self.overlay_image_alpha(frame_resized, self.check_bgr if total_objects == 0 else self.puddle_bgr, (text_x_position - icon_width, 40), self.check_alpha if total_objects == 0 else self.puddle_alpha)
        status_message = "It is dry :)" if total_objects == 0 else "It is flooded!"
        cvzone.putTextRect(frame_resized, status_message, (text_x_position - 10, 100), scale=2, thickness=4, colorR=(235, 183, 23), colorT=(10, 240, 100) if total_objects == 0 else (100, 10, 240), offset=7)

        return frame_resized

    def main(self):
        process_every_n_frames = 2
        frame_count = 0

        window_name = f"PUDDLE{self.idx} : {self.camera_name}"
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

                frame_processed = self.process_frame(frame)

                if self.show_text:
                    cvzone.putTextRect(frame_processed, f"FPS: {int(self.fps)}", (10, self.new_height - 11), scale=2, thickness=4, offset=7, colorR=(235, 183, 23))

                cv2.imshow(window_name, frame_processed)
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

                frame_processed = self.process_frame(frame)

                if self.show_text:
                    cvzone.putTextRect(frame_processed, f"FPS: {int(self.fps)}", (10, self.new_height - 11), scale=2, thickness=4, offset=7, colorR=(235, 183, 23))

                cv2.imshow(window_name, frame_processed)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("n"):
                    self.stop_event.set()
                    break
                elif key == ord("s"):
                    self.show_text = not self.show_text
            cv2.destroyAllWindows()
            self.frame_thread.join()


def run_puddle(
    PUDDLE_CONFIDENCE_THRESHOLD,
    camera_name,
    window_size=(540, 360),
    rtsp_url=None,
):
    detector = PuddleDetector(
        PUDDLE_CONFIDENCE_THRESHOLD=PUDDLE_CONFIDENCE_THRESHOLD,
        camera_name=camera_name,
        window_size=window_size,
        rtsp_url=rtsp_url,
    )
    detector.main()


if __name__ == "__main__":
    run_puddle(
        PUDDLE_CONFIDENCE_THRESHOLD=0,
        camera_name="KANTIN1",
        window_size=(320, 240),
    )
