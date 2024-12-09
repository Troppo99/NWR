from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO(r"D:\NWR\run\kon\version1\weights\last.pt")

    model.train(
        task="segment",
        data="D:/NWR/datasets/kon/data.yaml",
        epochs=100,
        imgsz=640,
        project="run/kon",
        name="version1",
        device="cuda",
        batch=16,
        resume=True,
        amp=True,
    )
