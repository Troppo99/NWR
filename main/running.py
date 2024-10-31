from ALKBR import run_broom
from ALGFR import run_carpal
import multiprocessing


def start_processes(target_func, cameras, args, size):
    processes = []
    for camera in cameras:
        p = multiprocessing.Process(target=target_func, args=(*args, camera, size))
        processes.append(p)
        p.start()
    return processes


if __name__ == "__main__":
    broom_list_cameras = ["OFFICE1", "OFFICE2", "OFFICE3"]
    carpal_list_cameras = ["OFFICE1", "OFFICE2", "OFFICE3"]
    
    broom_args = (30, 0.005, 80)
    carpal_args = (30, 0.005, 80)

    broom_processes = start_processes(run_broom, broom_list_cameras, broom_args, size=(480, 320))
    carpal_processes = start_processes(run_carpal, carpal_list_cameras, carpal_args, size=(480, 320))

    for p in broom_processes + carpal_processes:
        p.join()
