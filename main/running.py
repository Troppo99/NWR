# caller_script.py
from NEW_ALKBR import run_broom
from ALGFR import run_carpal
from ALMTN import run_motor
import multiprocessing


def start_processes(target_func, cameras, window_size=(320, 240), rtsp_url=None):
    """
    Starts multiple processes for a given target function and list of cameras.

    :param target_func: The function to run in each process.
    :param cameras: A list of camera names.
    :param window_size: A tuple specifying the window size.
    :param rtsp_url: Optional RTSP URL to override default.
    :return: A list of multiprocessing.Process objects.
    """
    processes = []
    for camera in cameras:
        # Ensure args is a tuple. Pass rtsp_url only if it's provided.
        if rtsp_url:
            p = multiprocessing.Process(target=target_func, args=(camera, window_size, rtsp_url))
        else:
            p = multiprocessing.Process(target=target_func, args=(camera, window_size))
        processes.append(p)
        p.start()
    return processes


if __name__ == "__main__":
    # Define camera lists
    broom_list_cameras = ["SEWING1", "SEWING2", "SEWING3", "SEWING4", "SEWING5", "SEWING6", "SEWING8", "OFFICE1", "OFFICE2", "OFFICE3"]
    carpal_list_cameras = ["OFFICE1", "OFFICE2", "OFFICE3"]
    motor_list_cameras = ["HALAMAN1", "GERBANG1", "KANTIN1", "KANTIN2", "EXPEDISI2"]

    # Define window sizes per target function (customize as needed)
    broom_window_size = (320, 240)  # Example size for broom
    carpal_window_size = (320, 240)  # Example size for carpal
    motor_window_size = (320, 240)  # Example size for motor

    # Optional: Define RTSP URLs per camera if needed
    # Example RTSP URL override (if applicable)
    # rtsp_url_broom = "rtsp://username:password@camera_ip:554/stream"
    # rtsp_url_carpal = "rtsp://username:password@camera_ip:554/stream"
    # rtsp_url_motor = "rtsp://username:password@camera_ip:554/stream"

    # Start processes for broom cameras
    broom_processes = []
    if broom_list_cameras:
        broom_processes = start_processes(
            target_func=run_broom,
            cameras=broom_list_cameras,
            window_size=broom_window_size,
            # rtsp_url=rtsp_url_broom  # Uncomment if RTSP URL override is needed
        )

    # Start processes for carpal cameras
    carpal_processes = []
    if carpal_list_cameras:
        carpal_processes = start_processes(
            target_func=run_carpal,
            cameras=carpal_list_cameras,
            window_size=carpal_window_size,
            # rtsp_url=rtsp_url_carpal  # Uncomment if RTSP URL override is needed
        )

    # Start processes for motor cameras
    motor_processes = []
    if motor_list_cameras:
        motor_processes = start_processes(
            target_func=run_motor,
            cameras=motor_list_cameras,
            window_size=motor_window_size,
            # rtsp_url=motor_rtsp_url  # Uncomment if RTSP URL override is needed
        )

    # Combine all processes
    all_processes = broom_processes + carpal_processes + motor_processes

    # Wait for all processes to complete
    for p in all_processes:
        p.join()
