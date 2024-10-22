import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import cv2
from ultralytics import YOLO
import numpy as np
import time
import torch
import queue
from concurrent.futures import ThreadPoolExecutor


class CarpalDetector:
    def __init__(self):
        self.model = YOLO("yolo11l-pose.pt")
        self.model.overrides["verbose"] = False
        self.stop_flag = False
        self.keypoint_overlap = False
        self.keypoint_absence_timer_start = None
        self.cap = cv2.VideoCapture("rtsp://admin:oracle2015@10.5.0.170:554/Streaming/Channels/1")
        self.borders_pane = [
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
        ]

    def process_model(self, frame):
        with torch.no_grad():
            results = self.model(frame, stream=True, imgsz=960)
        return results

    def read_frames(self, frame_queue):
        while not self.stop_flag:
            ret, frame = self.cap.read()
            if not ret:
                print("Stream gagal dibaca. Pastikan URL stream benar.")
                break
            try:
                frame_queue.put(frame, block=False)
            except queue.Full:
                pass
            time.sleep(0.01)

    def draw_pose(self, frame, keypoint_coords, pairs, green_pairs, blue_pairs, pink_pairs, orange_pairs, scaled_borders_pts, border_states):
        for i, coord in enumerate(keypoint_coords):
            if coord:
                x, y = coord
                if i == 9:
                    kp7 = keypoint_coords[7]
                    kp9 = keypoint_coords[9]
                    if kp7 and kp9:
                        vx = kp9[0] - kp7[0]
                        vy = kp9[1] - kp7[1]
                        norm = (vx**2 + vy**2) ** 0.5
                        if norm != 0:
                            vx /= norm
                            vy /= norm
                            extension_length = 50
                            x_new = int(kp9[0] + vx * extension_length)
                            y_new = int(kp9[1] + vy * extension_length)
                            x, y = x_new, y_new
                    radius = 30
                    point = (x, y)
                    overlap = False
                    for idx, border in enumerate(scaled_borders_pts):
                        dist = cv2.pointPolygonTest(border, point, True)
                        if dist >= -radius:
                            border_states[idx]["is_green"] = True
                            overlap = True
                    if overlap:
                        self.keypoint_overlap = True  # Keypoints are overlapping with some border
                    cv2.circle(frame, (x, y), radius, (0, 255, 255), -1)
                elif i == 10:
                    kp8 = keypoint_coords[8]
                    kp10 = keypoint_coords[10]
                    if kp8 and kp10:
                        vx = kp10[0] - kp8[0]
                        vy = kp10[1] - kp8[1]
                        norm = (vx**2 + vy**2) ** 0.5
                        if norm != 0:
                            vx /= norm
                            vy /= norm
                            extension_length = 50
                            x_new = int(kp10[0] + vx * extension_length)
                            y_new = int(kp10[1] + vy * extension_length)
                            x, y = x_new, y_new
                    radius = 30
                    point = (x, y)
                    overlap = False
                    for idx, border in enumerate(scaled_borders_pts):
                        dist = cv2.pointPolygonTest(border, point, True)
                        if dist >= -radius:
                            border_states[idx]["is_green"] = True
                            overlap = True
                    if overlap:
                        self.keypoint_overlap = True  # Keypoints are overlapping with some border
                    cv2.circle(frame, (x, y), radius, (0, 255, 255), -1)
                else:
                    radius = 5
                    cv2.circle(frame, (x, y), radius, (0, 255, 255), -1)
        for i, j in pairs:
            if keypoint_coords[i] and keypoint_coords[j]:
                if (i, j) in green_pairs or (j, i) in green_pairs:
                    color = (0, 255, 0)
                elif (i, j) in blue_pairs or (j, i) in blue_pairs:
                    color = (255, 255, 0)
                elif (i, j) in pink_pairs or (j, i) in pink_pairs:
                    color = (200, 0, 255)
                elif (i, j) in orange_pairs or (j, i) in orange_pairs:
                    color = (60, 190, 255)
                else:
                    color = (255, 0, 0)
                cv2.line(frame, keypoint_coords[i], keypoint_coords[j], color, 2)

    def process_frames(self, frame_queue):
        pairs = [(0, 1), (0, 2), (1, 2), (2, 4), (1, 3), (4, 6), (3, 5), (5, 6), (6, 8), (8, 10), (5, 7), (7, 9), (6, 12), (12, 11), (11, 5), (12, 14), (14, 16), (11, 13), (13, 15)]
        green_pairs = {(0, 1), (0, 2), (1, 2), (1, 3), (2, 4), (3, 5), (4, 6)}
        blue_pairs = {(8, 10), (6, 8), (5, 6), (5, 7), (7, 9)}
        pink_pairs = {(6, 12), (11, 12), (5, 11)}
        orange_pairs = {(14, 16), (12, 14), (11, 13), (13, 15)}

        # Initialize border states
        border_states = [{"is_green": False} for _ in self.borders_pane]
        self.keypoint_absence_timer_start = None
        first_frame = True

        while not self.stop_flag:
            if not frame_queue.empty():
                frame = frame_queue.get()

                if first_frame:
                    height, width, _ = frame.shape
                    scale_x = width / 1280
                    scale_y = height / 720
                    scaled_borders_pts = []
                    for border in self.borders_pane:
                        scaled_border = []
                        for x, y in border:
                            scaled_x = int(x * scale_x)
                            scaled_y = int(y * scale_y)
                            scaled_border.append((scaled_x, scaled_y))
                        scaled_borders_pts.append(np.array(scaled_border, np.int32))
                    first_frame = False

                results = self.process_model(frame)

                current_time = time.time()
                self.keypoint_overlap = False  # Reset overlap flag

                for result in results:
                    keypoints_data = result.keypoints.data

                    for keypoints in keypoints_data:
                        keypoint_coords = [(int(x), int(y)) if confidence > 0.5 else None for x, y, confidence in keypoints]
                        self.draw_pose(frame, keypoint_coords, pairs, green_pairs, blue_pairs, pink_pairs, orange_pairs, scaled_borders_pts, border_states)

                # Check for reset condition
                if any(state["is_green"] for state in border_states):
                    if self.keypoint_overlap:
                        self.keypoint_absence_timer_start = None
                    else:
                        if self.keypoint_absence_timer_start is None:
                            self.keypoint_absence_timer_start = current_time
                        elif (current_time - self.keypoint_absence_timer_start) >= 3:
                            # Reset all borders to yellow
                            print("Resetting all borders to yellow after 3 seconds of no overlap.")
                            for state in border_states:
                                state["is_green"] = False
                            self.keypoint_absence_timer_start = None
                else:
                    self.keypoint_absence_timer_start = None  # No green borders, no need to reset

                overlay = frame.copy()
                alpha = 0.5
                for idx, border in enumerate(scaled_borders_pts):
                    if border_states[idx]["is_green"]:
                        color = (0, 255, 0)  # Green
                    else:
                        color = (0, 255, 255)  # Yellow
                    cv2.fillPoly(overlay, [border], color)
                cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

                frame_show = cv2.resize(frame, (1280, 720))
                cv2.imshow("THREADPOOL EXECUTOR - Pose Detection with Borders", frame_show)

                if cv2.waitKey(1) & 0xFF == ord("n"):
                    print("Keluar dari aplikasi.")
                    self.stop_flag = True
                    break
            else:
                time.sleep(0.005)

    def main(self):
        if not self.cap.isOpened():
            print("Gagal membuka stream video. Periksa URL atau koneksi.")
            return

        frame_queue = queue.Queue(maxsize=20)

        with ThreadPoolExecutor(max_workers=2) as executor:
            executor.submit(self.read_frames, frame_queue)
            self.process_frames(frame_queue)

        executor.shutdown(wait=True)
        self.cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    carpal = CarpalDetector()
    carpal.main()
