import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
import cv2
from ultralytics import YOLO
from concurrent.futures import ThreadPoolExecutor
import queue
import time

stop_flag = False


def read_frames(cap, frame_queue):
    global stop_flag
    while not stop_flag:
        ret, frame = cap.read()
        if not ret:
            print("Stream gagal dibaca. Pastikan URL stream benar.")
            break
        try:
            # Menambahkan frame langsung ke antrian tanpa GPU
            frame_queue.put(frame, block=False)
        except queue.Full:
            print("Antrian penuh, frame dibuang")
        # Mengurangi sleep agar lebih sinkron dengan frame rate video
        time.sleep(0.01)


def draw_pose(frame, keypoint_coords, pairs, green_pairs, blue_pairs, pink_pairs, orange_pairs):
    for i, coord in enumerate(keypoint_coords):
        if coord:
            x, y = coord
            cv2.circle(frame, (x, y), 5, (0, 255, 255), -1)  # Gambar titik keypoint

    for i, j in pairs:
        if keypoint_coords[i] and keypoint_coords[j]:
            if (i, j) in green_pairs or (j, i) in green_pairs:
                color = (0, 255, 0)  # Hijau untuk bagian tubuh atas
            elif (i, j) in blue_pairs or (j, i) in blue_pairs:
                color = (255, 255, 0)  # Biru untuk bagian kaki
            elif (i, j) in pink_pairs or (j, i) in pink_pairs:
                color = (200, 0, 255)  # Pink untuk bagian torso
            elif (i, j) in orange_pairs or (j, i) in orange_pairs:
                color = (60, 190, 255)  # Oranye untuk bagian tangan
            else:
                color = (255, 0, 0)  # Merah untuk bagian lainnya
            cv2.line(frame, keypoint_coords[i], keypoint_coords[j], color, 2)  # Gambar garis antar keypoint


def process_frames(frame_queue):
    global stop_flag
    pairs = [(0, 1), (0, 2), (1, 2), (2, 4), (1, 3), (4, 6), (3, 5), (5, 6), (6, 8), (8, 10), (5, 7), (7, 9), (6, 12), (12, 11), (11, 5), (12, 14), (14, 16), (11, 13), (13, 15)]
    green_pairs = {(0, 1), (0, 2), (1, 2), (1, 3), (2, 4), (3, 5), (4, 6)}
    blue_pairs = {(8, 10), (6, 8), (5, 6), (5, 7), (7, 9)}
    pink_pairs = {(6, 12), (11, 12), (5, 11)}
    orange_pairs = {(14, 16), (12, 14), (11, 13), (13, 15)}

    while not stop_flag:
        if not frame_queue.empty():
            frame = frame_queue.get()
            results = model(frame, stream=True)

            for result in results:
                keypoints_data = result.keypoints.data

                for keypoints in keypoints_data:
                    keypoint_coords = [(int(x), int(y)) if confidence > 0.5 else None for x, y, confidence in keypoints]
                    draw_pose(frame, keypoint_coords, pairs, green_pairs, blue_pairs, pink_pairs, orange_pairs)

            frame_show = cv2.resize(frame, (1280, 720))
            cv2.imshow("THREADPOOL EXECUTOR - Pose Detection", frame_show)

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
    frame_queue = queue.Queue(maxsize=20)

    with ThreadPoolExecutor(max_workers=2) as executor:
        executor.submit(read_frames, cap, frame_queue)
        process_frames(frame_queue)

    executor.shutdown(wait=True)
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
