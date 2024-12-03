import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO("models/yolo11l.pt")
    model.train(
        data="D:/NWR/datasets/Paper_Two/data.yaml",
        epochs=100,
        imgsz=640,
        project="run/paper",
        name="ONE&TWO",
        device="cuda",
        batch=16,
        resume=False,
        amp=True,
    )
