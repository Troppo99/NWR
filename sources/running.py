from sources.AlGhaffar.carpal_detection import run_carpal
from sources.AlKabir.broom_detection import run_broom
import multiprocessing


def start_processes(target_func, cameras, window_size=(320, 240), rtsp_url=None, display=True):
    processes = []
    for camera in cameras:
        if rtsp_url:
            p = multiprocessing.Process(target=target_func, args=(camera, window_size, rtsp_url, display))
        else:
            p = multiprocessing.Process(target=target_func, args=(camera, window_size, None, display))
        processes.append(p)
        p.start()
    return processes


if __name__ == "__main__":
    broom_list_cameras = [
        "SEWINGOFFICE",
        "SEWING1",
        "SEWING2",
    ]
    carpal_list_cameras = [
        "OFFICE1K",
        "OFFICE2K",
        "OFFICE3K",
    ]
    display = True
    broom_processes = start_processes(target_func=run_broom, cameras=broom_list_cameras, display=display)
    carpal_processes = start_processes(target_func=run_carpal, cameras=carpal_list_cameras, display=display)

    all_processes = broom_processes + carpal_processes

    for p in all_processes:
        p.join()
