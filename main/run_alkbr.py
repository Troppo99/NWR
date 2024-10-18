from ALKBR import run_detection
import threading

camera_names = ["10.5.0.161", "10.5.0.170", "10.5.0.182"]
threads = []
stop_flags = []

for camera_name in camera_names:
    stop_flag = threading.Event()
    stop_flags.append(stop_flag)
    thread = threading.Thread(target=run_detection, args=(camera_name, stop_flag))
    threads.append(thread)
    thread.start()

for thread in threads:
    thread.join()
