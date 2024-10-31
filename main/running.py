from ALKBR import run_broom
from ALGFR import run_carpal
import multiprocessing

if __name__ == "__main__":
    size = (480, 320)
    list_cameras = ["OFFICE1", "OFFICE2", "OFFICE3"]
    # broom
    broom1_process = multiprocessing.Process(target=run_broom, args=(10, 0, 50, list_cameras[0], size))
    broom2_process = multiprocessing.Process(target=run_broom, args=(10, 0, 50, list_cameras[1], size))
    broom3_process = multiprocessing.Process(target=run_broom, args=(10, 0, 50, list_cameras[2], size))

    # carpal
    carpal1_process = multiprocessing.Process(target=run_carpal, args=(10, 0, 50, list_cameras[0], size))
    carpal2_process = multiprocessing.Process(target=run_carpal, args=(10, 0, 50, list_cameras[1], size))
    carpal3_process = multiprocessing.Process(target=run_carpal, args=(10, 0, 50, list_cameras[2], size))

    # start broom
    broom1_process.start()
    broom2_process.start()
    broom3_process.start()


    # start carpal
    carpal1_process.start()
    carpal2_process.start()
    carpal3_process.start()

    # join broom
    broom1_process.join()
    broom2_process.join()
    broom3_process.join()


    # join carpal
    carpal1_process.join()
    carpal2_process.join()
    carpal3_process.join()
