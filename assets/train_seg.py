from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO("models/yolo11l-seg.pt")

    model.train(
        task="segment",
        data="D:/NWR/datasets/KON2.v1i.yolov11/data.yaml",
        epochs=100,
        imgsz=640,
        project="run/kon2",
        name="version1",
        device="cuda",
        batch=16,
        resume=False,
        amp=True,
    )
