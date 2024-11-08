from NEW_ALKBR import run_broom
from NEW_ALGFR import run_carpal
import multiprocessing


def start_processes(target_func, cameras, window_size=(320, 240), rtsp_url=None):
    processes = []
    for camera in cameras:
        if rtsp_url:
            p = multiprocessing.Process(target=target_func, args=(camera, window_size, rtsp_url))
        else:
            p = multiprocessing.Process(target=target_func, args=(camera, window_size))
        processes.append(p)
        p.start()
    return processes


if __name__ == "__main__":
    broom_list_cameras = ["SEWING1", "SEWING2", "SEWING3", "SEWING4", "SEWING5", "SEWING6", "SEWING8", "OFFICE1", "OFFICE2", "OFFICE3"]
    carpal_list_cameras = ["OFFICE1", "OFFICE2", "OFFICE3"]

    broom_window_size = (320, 240)
    carpal_window_size = (320, 240)

    broom_processes = []
    if broom_list_cameras:
        broom_processes = start_processes(
            target_func=run_broom,
            cameras=broom_list_cameras,
            window_size=broom_window_size,
        )

    carpal_processes = []
    if carpal_list_cameras:
        carpal_processes = start_processes(
            target_func=run_carpal,
            cameras=carpal_list_cameras,
            window_size=carpal_window_size,
        )

    all_processes = broom_processes + carpal_processes

    for p in all_processes:
        p.join()
