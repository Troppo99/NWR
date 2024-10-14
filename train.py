from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO("yolov11l.pt")
    model.train(
        data="D:/SBHNL/Images/BSML/Datasets/Finishing/finishing_united/data.yaml",
        epochs=50,
        imgsz=640,
        project="run/finishing",
        name="version1",
        device="cuda",
        batch=16,
        resume=False,
        fp16=True,
    )
