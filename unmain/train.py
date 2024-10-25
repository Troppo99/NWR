import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO("yolo11l-pose.pt")
    model.train(
        data="D:/SBHNL/Images/AHMDL/ALKBR/BROOM_V5/data.yaml",
        epochs=100,
        imgsz=640,
        project="run/broom5_yolo11",
        name="version1",
        device="cuda",
        batch=16,
        resume=False,
        amp=True,  # Use 'amp' instead of 'fp16' to enable mixed-precision training
    )
