import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import cv2
from ultralytics import YOLO
import numpy as np
import time
import multiprocessing
import torch

start_time = time.time()
x = 0

# Configuration
CONFIDENCE_THRESHOLD_BROOM = 0.9
new_width, new_height = 640, 480
pairs_broom = [(0, 1), (1, 2), (2, 3), (2, 4)]
process_every_n_frames = 5


def opencv_section(video, frame_queue, stop_flag):
    cap = cv2.VideoCapture(video)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, new_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, new_height)

    # Load model inside the process
    start_model_load_time = time.time()
    model = YOLO("broom5l.pt").to("cuda")  # Use a smaller model if necessary
    model.overrides["verbose"] = False
    end_model_load_time = time.time()
    model_load_time = end_model_load_time - start_model_load_time
    print(f"Model load time for {video}: {model_load_time:.2f} seconds")

    frame_count = 0

    while not stop_flag.is_set():
        ret, frame = cap.read()
        if not ret:
            cap.release()
            time.sleep(1)
            cap = cv2.VideoCapture(video)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, new_width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, new_height)
            continue

        frame_count += 1
        if frame_count % process_every_n_frames != 0:
            continue

        frame_resized = cv2.resize(frame, (new_width, new_height))

        # Start timing inference
        start_inference_time = time.time()

        # Run detection
        with torch.no_grad():
            results_broom = model(frame_resized)

        # End timing inference
        end_inference_time = time.time()
        inference_time = end_inference_time - start_inference_time
        print(f"Inference time for {video}: {inference_time:.2f} seconds")

        # Process detection results
        points, coords, _ = export_frame_broom(results_broom, (0, 255, 0), pairs_broom)

        # Add detection results to frame
        if points and coords:
            for x1, y1, color in coords:
                cv2.line(frame_resized, x1, y1, color, 2)
            for point in points:
                cv2.circle(frame_resized, point, 4, (0, 255, 255), -1)

        try:
            frame_queue.put((video, frame_resized), block=False)
        except multiprocessing.queues.Full:
            pass

    cap.release()


def display_frames(frame_queue, stop_flag, processes):
    global x
    window_names = {}
    while not stop_flag.is_set():
        try:
            if not frame_queue.empty():
                camera_name, frame = frame_queue.get()
                if camera_name not in window_names:
                    window_names[camera_name] = f"ALKBR TESTING - {camera_name}"

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
            else:
                time.sleep(0.01)
        except KeyboardInterrupt:
            stop_flag.set()
            break

    for process in processes:
        process.join()

    cv2.destroyAllWindows()


def camera(name):
    configurations = {
        "10.5.0.161": "161",
        "10.5.0.170": "170",
        "10.5.0.182": "182",
    }
    video = f"rtsp://admin:oracle2015@10.5.0.{configurations[name]}:554/Streaming/Channels/1"
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


if __name__ == "__main__":
    camera_names = ["10.5.0.161", "10.5.0.170", "10.5.0.182"]
    frame_queue = multiprocessing.Queue(maxsize=20)
    stop_flag = multiprocessing.Event()
    processes = []

    # Create processes for each camera
    for camera_name in camera_names:
        video_path = camera(camera_name)
        process = multiprocessing.Process(target=opencv_section, args=(video_path, frame_queue, stop_flag))
        processes.append(process)
        process.start()

    # Manage display frames in the main process
    try:
        display_frames(frame_queue, stop_flag, processes)
    finally:
        stop_flag.set()
        for process in processes:
            process.join()
