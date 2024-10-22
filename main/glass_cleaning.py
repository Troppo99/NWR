import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
import torch
import cv2
from ultralytics import YOLO
from concurrent.futures import ThreadPoolExecutor
import queue
import time
import numpy as np

stop_flag = False
borders = [[(48, 341), (77, 334), (70, 302), (41, 310)], [(77, 334), (108, 326), (101, 292), (70, 302)], [(108, 326), (141, 320), (133, 283), (101, 292)]]


def process_model(frame):
    with torch.no_grad():
        results = model(frame, stream=True, imgsz=960)
    return results


def read_frames(cap, frame_queue):
    global stop_flag
    while not stop_flag:
        ret, frame = cap.read()
        if not ret:
            print("Stream gagal dibaca. Pastikan URL stream benar.")
            break
        try:
            frame_queue.put(frame, block=False)
        except queue.Full:
            pass
        time.sleep(0.01)


def draw_pose(frame, keypoint_coords, pairs, green_pairs, blue_pairs, pink_pairs, orange_pairs, scaled_borders_pts, border_states):
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
                for idx, border in enumerate(scaled_borders_pts):
                    if cv2.pointPolygonTest(border, point, False) >= 0:
                        border_states[idx] = True
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
                for idx, border in enumerate(scaled_borders_pts):
                    if cv2.pointPolygonTest(border, point, False) >= 0:
                        border_states[idx] = True
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


def process_frames(frame_queue):
    global stop_flag
    pairs = [(0, 1), (0, 2), (1, 2), (2, 4), (1, 3), (4, 6), (3, 5), (5, 6), (6, 8), (8, 10), (5, 7), (7, 9), (6, 12), (12, 11), (11, 5), (12, 14), (14, 16), (11, 13), (13, 15)]
    green_pairs = {(0, 1), (0, 2), (1, 2), (1, 3), (2, 4), (3, 5), (4, 6)}
    blue_pairs = {(8, 10), (6, 8), (5, 6), (5, 7), (7, 9)}
    pink_pairs = {(6, 12), (11, 12), (5, 11)}
    orange_pairs = {(14, 16), (12, 14), (11, 13), (13, 15)}

    border_states = [False] * len(borders)
    first_frame = True
    while not stop_flag:
        if not frame_queue.empty():
            frame = frame_queue.get()

            if first_frame:
                height, width, _ = frame.shape
                scale_x = width / 1280
                scale_y = height / 720
                scaled_borders_pts = []
                for border in borders:
                    scaled_border = []
                    for x, y in border:
                        scaled_x = int(x * scale_x)
                        scaled_y = int(y * scale_y)
                        scaled_border.append((scaled_x, scaled_y))
                    scaled_borders_pts.append(np.array(scaled_border, np.int32))
                first_frame = False

            results = process_model(frame)

            for result in results:
                keypoints_data = result.keypoints.data

                for keypoints in keypoints_data:
                    keypoint_coords = [(int(x), int(y)) if confidence > 0.5 else None for x, y, confidence in keypoints]
                    draw_pose(frame, keypoint_coords, pairs, green_pairs, blue_pairs, pink_pairs, orange_pairs, scaled_borders_pts, border_states)

            overlay = frame.copy()
            alpha = 0.5
            for idx, border in enumerate(scaled_borders_pts):
                if border_states[idx]:
                    color = (0, 255, 0)
                else:
                    color = (0, 255, 255)
                cv2.fillPoly(overlay, [border], color)
            cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

            border_states = [False] * len(borders)

            frame_show = cv2.resize(frame, (1280, 720))
            cv2.imshow("THREADPOOL EXECUTOR - Pose Detection with Borders", frame_show)

            if cv2.waitKey(1) & 0xFF == ord("n"):
                print("Keluar dari aplikasi.")
                stop_flag = True
                break
        else:
            time.sleep(0.005)


def main():
    global stop_flag, model
    cap = cv2.VideoCapture("rtsp://admin:oracle2015@192.168.100.65:554/Streaming/Channels/1")
    if not cap.isOpened():
        print("Gagal membuka stream video. Periksa URL atau koneksi.")
        return

    model = YOLO("yolo11l-pose.pt").to("cuda")
    model.overrides["verbose"] = False
    frame_queue = queue.Queue(maxsize=20)

    with ThreadPoolExecutor(max_workers=2) as executor:
        executor.submit(read_frames, cap, frame_queue)
        process_frames(frame_queue)

    executor.shutdown(wait=True)
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
