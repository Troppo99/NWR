from ALKBR import run_broom
from ALGFR import run_carpal
import multiprocessing

if __name__ == "__main__":
    # hybrid alkbr and algfr
    broom1_process = multiprocessing.Process(target=run_broom, args=(10, 0, 50, "10.5.0.182", (960, 540)))
    broom2_process = multiprocessing.Process(target=run_broom, args=(10, 0, 50, "10.5.0.170", (960, 540)))
    broom3_process = multiprocessing.Process(target=run_broom, args=(10, 0, 50, "10.5.0.161", (960, 540)))
    carpal1_process = multiprocessing.Process(target=run_carpal, args=(10, 0, 50, "10.5.0.182", (960, 540)))
    carpal2_process = multiprocessing.Process(target=run_carpal, args=(10, 0, 50, "10.5.0.170", (960, 540)))
    carpal3_process = multiprocessing.Process(target=run_carpal, args=(10, 0, 50, "10.5.0.161", (960, 540)))

    # only alkbr
    broom4_process = multiprocessing.Process(target=run_broom, args=(10, 0, 50, "10.5.0.110", (960, 540)))
    broom5_process = multiprocessing.Process(target=run_broom, args=(10, 0, 50, "10.5.0.180", (960, 540)))
    broom6_process = multiprocessing.Process(target=run_broom, args=(10, 0, 50, "10.5.0.185", (960, 540)))
    broom7_process = multiprocessing.Process(target=run_broom, args=(10, 0, 50, "10.5.0.146", (960, 540)))
    broom8_process = multiprocessing.Process(target=run_broom, args=(10, 0, 50, "10.5.0.183", (960, 540)))
    broom9_process = multiprocessing.Process(target=run_broom, args=(10, 0, 50, "10.5.0.195", (960, 540)))
    broom10_process = multiprocessing.Process(target=run_broom, args=(10, 0, 50, "10.5.0.201", (960, 540)))

    # start alkbr
    broom1_process.start()
    broom2_process.start()
    broom3_process.start()
    broom4_process.start()
    broom5_process.start()
    broom6_process.start()
    broom7_process.start()
    broom8_process.start()
    broom9_process.start()
    broom10_process.start()

    # start algfr
    carpal1_process.start()
    carpal2_process.start()
    carpal3_process.start()

    # join alkbr
    broom1_process.join()
    broom2_process.join()
    broom3_process.join()
    broom4_process.join()
    broom5_process.join()
    broom6_process.join()
    broom7_process.join()
    broom8_process.join()
    broom9_process.join()
    broom10_process.join()

    # join algfr
    carpal1_process.join()
    carpal2_process.join()
    carpal3_process.join()
