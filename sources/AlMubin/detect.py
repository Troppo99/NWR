import cv2
import cvzone
from ultralytics import YOLO


class YOLOVideoProcessor:
    def __init__(self, video_path, model_path, display_size=(540, 360), confidence_threshold=0.5):
        self.video_path = video_path
        self.model_path = model_path
        self.display_width, self.display_height = display_size
        self.confidence_threshold = confidence_threshold
        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            raise IOError(f"Error: Tidak dapat membuka video {self.video_path}")
        self.model = YOLO(self.model_path)
        self.model.overrides["verbose"] = False
        self.window_name = "YOLO Detection"

    def export_frame(self, frame):
        results = self.model(frame)
        boxes_info = []
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = box.conf[0]
                class_id = self.model.names[int(box.cls[0])]
                if conf > self.confidence_threshold:
                    boxes_info.append((x1, y1, x2, y2, conf, class_id))
        return frame, boxes_info

    def draw_boxes(self, frame, boxes_info):
        for x1, y1, x2, y2, conf, class_id in boxes_info:
            cvzone.cornerRect(frame, (x1, y1, x2 - x1, y2 - y1), rt=0, colorC=(0, 255, 255))
            cvzone.putTextRect(frame, f"{class_id} {conf:.2f}", (x1, y1 - 10))
        return frame

    def resize_frame(self, frame):
        return cv2.resize(frame, (self.display_width, self.display_height))

    def display_video(self):
        cv2.namedWindow(self.window_name)

        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("Video selesai atau tidak dapat membaca frame.")
                break

            frame_results, boxes_info = self.export_frame(frame)
            frame_results = self.draw_boxes(frame_results, boxes_info)
            frame_show = self.resize_frame(frame_results)
            cv2.imshow(self.window_name, frame_show)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                print("Keluar dari program.")
                break

        self.cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    video_path = "D:/SBHNL/Videos/AHMDL/EDIT/MOT.mp4"
    model_path = "D:/NWR/run/cutting_engine/version1/weights/best.pt"

    try:
        processor = YOLOVideoProcessor(video_path=video_path, model_path=model_path)
        processor.display_video()
    except Exception as e:
        print(e)
