import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO("D:/NWR/run/broom/version_detect/weights/last.pt")
    model.train(
        data="D:/NWR/datasets/BROOM_DETECT.v1i.yolov11/data.yaml",
        epochs=100,
        imgsz=640,
        project="run/broom",
        name="version_detect",
        device="cuda",
        batch=16,
        resume=True,
        amp=True,  # Use 'amp' instead of 'fp16' to enable mixed-precision training
    )
