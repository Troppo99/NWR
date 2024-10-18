from ALKBR import display_thread, run_detection
import threading
import queue

camera_names = ["10.5.0.161", "10.5.0.170", "10.5.0.182"]
frame_queue = queue.Queue(maxsize=20)
stop_flag = threading.Event()
threads = []
for camera_name in camera_names:
    thread = threading.Thread(target=run_detection, args=(camera_name, frame_queue, stop_flag))
    threads.append(thread)
    thread.start()

display_thread_instance = threading.Thread(target=display_thread, args=(frame_queue, stop_flag))
display_thread_instance.start()

for thread in threads:
    thread.join()
display_thread_instance.join()
