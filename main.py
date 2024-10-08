import cv2
import cvzone
from ultralytics import YOLO
import math
import threading
import queue
import time


def read_frames(cap, video, frame_queue, stop_flag):
    while not stop_flag.is_set():
        ret, frame = cap.read()
        if not ret:
            print("Request timed out.")
            cap.release()
            time.sleep(5)
            cap = connect_to_stream(video)
            continue
        try:
            frame_queue.put(frame, block=False)
        except queue.Full:
            print("Antrian penuh, frame dibuang")


def export_frame(frame):
    results = model(frame)
    boxes_info = []
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = math.ceil(box.conf[0] * 100) / 100
            class_id = model.names[int(box.cls[0])]
            if conf > 0:
                boxes_info.append((x1, y1, x2, y2, conf, class_id))
    return frame, boxes_info


def process_frames(frame_queue, stop_flag):
    while not stop_flag.is_set():
        if not frame_queue.empty():
            frame = frame_queue.get()
            frame_results, boxes_info = export_frame(frame)
            for x1, y1, x2, y2, conf, class_id in boxes_info:
                cvzone.cornerRect(frame_results, (x1, y1, x2 - x1, y2 - y1), rt=0, colorC=(0, 255, 255))
                cvzone.putTextRect(frame_results, f"{class_id} {conf}", (x1, y1 - 15))

            frame_show = cv2.resize(frame_results, (1280, 720))
            cv2.imshow("ADVCANCE BSML VERSION 2", frame_show)

            if cv2.waitKey(1) & 0XFF == ord('n'):
                stop_flag.set()
                break


def connect_to_stream(video):
    cap = cv2.VideoCapture(video)
    while not cap.isOpened():
        print("Request timed out.")
        cap.release()
        time.sleep(5)
        cap = cv2.VideoCapture(video)
    return cap


# Main program
# video = "Videos/cutting2.mp4"
video = "rtsp://admin:oracle2015@192.168.100.27:554/Streaming/Channels/1"
cap = connect_to_stream(video)
model = YOLO("Resources/Models/Pretrained/CUTTING/C1_V1/weights/best.pt")
frame_queue = queue.Queue(maxsize=5)
stop_flag = threading.Event()

reader_thread = threading.Thread(target=read_frames, args=(cap, video, frame_queue, stop_flag))
reader_thread.daemon = True
reader_thread.start()

process_frames(frame_queue, stop_flag)

cap.release()
cv2.destroyAllWindows()
