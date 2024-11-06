from ALKBR import run_broom
from ALGFR import run_carpal
from ALMTN import run_motor
import multiprocessing


def start_processes(target_func, cameras, args=(), kwargs={}):
    processes = []
    for camera in cameras:
        process_kwargs = kwargs.copy()
        process_kwargs["camera_name"] = camera
        p = multiprocessing.Process(target=target_func, args=args, kwargs=process_kwargs)
        processes.append(p)
        p.start()
    return processes


if __name__ == "__main__":
    display = True
    broom_list_cameras = ["SEWING1", "SEWING2", "SEWING3", "SEWING4", "SEWING5", "SEWING6", "SEWING8", "OFFICE1", "OFFICE2", "OFFICE3"]
    carpal_list_cameras = ["OFFICE1", "OFFICE2", "OFFICE3"]
    motor_list_cameras = ["HALAMAN1", "GERBANG1", "KANTIN1", "KANTIN2", "EXPEDISI2"]

    broom_args = (30, 0.005, 80)
    broom_kwargs = {"display": display, "window_size": (320, 240)}

    carpal_args = (30, 0.005, 80)
    carpal_kwargs = {"display": display, "window_size": (320, 240)}

    motor_args = (30, 0.005, 80)
    motor_kwargs = {"display": display, "window_size": (320, 240)}

    broom_processes = start_processes(run_broom, broom_list_cameras, args=broom_args, kwargs=broom_kwargs)
    carpal_processes = start_processes(run_carpal, carpal_list_cameras, args=carpal_args, kwargs=carpal_kwargs)
    motor_processes = start_processes(run_motor, motor_list_cameras, args=motor_args, kwargs=motor_kwargs)

    for p in broom_processes + carpal_processes + motor_processes:
        p.join()
