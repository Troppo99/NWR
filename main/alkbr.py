import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import cv2
from ultralytics import YOLO
import numpy as np
import time
import threading
import queue
import torch

start_time = time.time()
x = 0

# Configuration
CONFIDENCE_THRESHOLD_BROOM = 0.9
new_width, new_height = 640, 360
pairs_broom = [(0, 1), (1, 2), (2, 3), (2, 4)]
process_every_n_frames = 5

# Define configurations globally
configurations = {
    "10.5.0.161": "161",
    "10.5.0.170": "170",
    "10.5.0.182": "182",
}


def camera(name):
    if name not in configurations:
        return None
    # Since 'name' is already the IP address, we can use it directly
    video = f"rtsp://admin:oracle2015@{name}:554/Streaming/Channels/1"
    return video


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


def read_frames(camera_name, frame_queue, stop_flag):
    video_path = camera(camera_name)
    if video_path is None:
        print(f"Camera {camera_name} not found in configurations.")
        return

    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, new_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, new_height)
    frame_count = 0

    while not stop_flag.is_set():
        ret, frame = cap.read()
        if not ret:
            cap.release()
            time.sleep(1)
            cap = cv2.VideoCapture(video_path)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, new_width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, new_height)
            continue

        frame_count += 1
        if frame_count % process_every_n_frames == 0:
            frame_resized = cv2.resize(frame, (new_width, new_height))
            try:
                frame_queue.put((camera_name, frame_resized), block=False)
            except queue.Full:
                pass
        else:
            # Skip frame processing
            pass

    cap.release()


def inference_worker(frame_queue, result_queue, stop_flag):
    # Load model
    start_model_load_time = time.time()
    model = YOLO("broom5l.pt").to("cuda")  # Use a smaller model if necessary
    model.overrides["verbose"] = False
    end_model_load_time = time.time()
    model_load_time = end_model_load_time - start_model_load_time
    print(f"Model loaded in {model_load_time:.2f} seconds")

    while not stop_flag.is_set():
        try:
            camera_name, frame = frame_queue.get(timeout=1)
            # Start timing inference
            start_inference_time = time.time()

            # Run detection
            with torch.no_grad():
                results_broom = model(frame)

            # End timing inference
            end_inference_time = time.time()
            inference_time = end_inference_time - start_inference_time
            print(f"Inference time for {camera_name}: {inference_time:.2f} seconds")

            # Process detection results
            points, coords, _ = export_frame_broom(results_broom, (0, 255, 0), pairs_broom)

            # Add detection results to frame
            if points and coords:
                for x1, y1, color in coords:
                    cv2.line(frame, x1, y1, color, 2)
                for point in points:
                    cv2.circle(frame, point, 4, (0, 255, 255), -1)

            result_queue.put((camera_name, frame))
        except queue.Empty:
            continue


def display_frames(result_queue, stop_flag, camera_names):
    global x
    window_names = {name: f"ALKBR TESTING - {name}" for name in camera_names}
    while not stop_flag.is_set():
        try:
            camera_name, frame = result_queue.get(timeout=1)
            cv2.imshow(window_names[camera_name], frame)
            if x == 0:
                total_run_time = time.time() - start_time
                print(f"Waktu total menunggu window muncul: {total_run_time:.2f} detik")
                x = 1
            if cv2.waitKey(1) & 0xFF == ord("n"):
                total_run_time = time.time() - start_time
                print(f"Waktu total dari start hingga 'n' ditekan: {total_run_time:.2f} detik")
                stop_flag.set()
                break
        except queue.Empty:
            continue

    cv2.destroyAllWindows()


if __name__ == "__main__":
    camera_names = list(configurations.keys())
    frame_queue = queue.Queue(maxsize=50)
    result_queue = queue.Queue(maxsize=50)
    stop_flag = threading.Event()
    threads = []

    # Start frame reading threads
    for camera_name in camera_names:
        t = threading.Thread(target=read_frames, args=(camera_name, frame_queue, stop_flag))
        t.start()
        threads.append(t)

    # Start inference thread
    inference_thread = threading.Thread(target=inference_worker, args=(frame_queue, result_queue, stop_flag))
    inference_thread.start()
    threads.append(inference_thread)

    # Start display in main thread
    try:
        display_frames(result_queue, stop_flag, camera_names)
    finally:
        stop_flag.set()
        for t in threads:
            t.join()
