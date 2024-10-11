from ultralytics import YOLO

if __name__ == "__main__":
    model_large = YOLO("yolov11l.pt")
    model_large.train(data="", epochs=50, imgsz=640, project="run/finishing", name="version1", device="cuda", resume=False)
