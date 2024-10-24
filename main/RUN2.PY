from ALKBR import run_broom
from ALGFR import run_carpal
import multiprocessing

if __name__ == "__main__":
    carpal_process = multiprocessing.Process(target=run_carpal, args=(10, 0, 50, "10.5.0.170", (960, 540)))
    broom_process = multiprocessing.Process(target=run_broom, args=(10, 0, 50, "10.5.0.170", (960, 540)))

    carpal_process.start()
    broom_process.start()

    carpal_process.join()
    broom_process.join()
