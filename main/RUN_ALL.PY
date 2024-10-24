from ALKBR import run_broom
from ALGFR import run_carpal
import multiprocessing

if __name__ == "__main__":
    carpal1_process = multiprocessing.Process(target=run_carpal, args=(10, 0, 50, "10.5.0.182", (960, 540)))
    broom1_process = multiprocessing.Process(target=run_broom, args=(10, 0, 50, "10.5.0.182", (960, 540)))
    carpal2_process = multiprocessing.Process(target=run_carpal, args=(10, 0, 50, "10.5.0.170", (960, 540)))
    broom2_process = multiprocessing.Process(target=run_broom, args=(10, 0, 50, "10.5.0.170", (960, 540)))
    carpal3_process = multiprocessing.Process(target=run_carpal, args=(10, 0, 50, "10.5.0.161", (960, 540)))
    broom3_process = multiprocessing.Process(target=run_broom, args=(10, 0, 50, "10.5.0.161", (960, 540)))

    carpal1_process.start()
    broom1_process.start()
    carpal2_process.start()
    broom2_process.start()
    carpal3_process.start()
    broom3_process.start()

    carpal1_process.join()
    broom1_process.join()
    carpal2_process.join()
    broom2_process.join()
    carpal3_process.join()
    broom3_process.join()
