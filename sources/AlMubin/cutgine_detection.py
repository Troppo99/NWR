import sys
import cv2
import cvzone
from ultralytics import YOLO
from PyQt6.QtCore import Qt, QTimer, QThread, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap, QFont
from PyQt6.QtWidgets import QApplication, QLabel, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout

# Konstanta
PROCESS_WIDTH = 960
PROCESS_HEIGHT = 540
DISPLAY_WIDTH = 1280
DISPLAY_HEIGHT = 720
CONFIDENCE_THRESHOLD = 0.5


class YOLOVideoProcessor(QThread):
    frame_updated = pyqtSignal(QImage)

    def __init__(self, video_path, model_path):
        super().__init__()
        self.video_path = video_path
        self.model_path = model_path
        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            raise IOError(f"Error: Tidak dapat membuka video {self.video_path}")
        self.model = YOLO(self.model_path)
        self.model.overrides["verbose"] = False
        self.running = True

    def run(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                print("Video selesai atau tidak dapat membaca frame. Mengulang video.")
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Restart video
                continue

            # Resize frame untuk pemrosesan
            frame_resized = cv2.resize(frame, (PROCESS_WIDTH, PROCESS_HEIGHT))

            # Deteksi objek dengan YOLO
            results = self.model(frame_resized)
            boxes_info = []

            for result in results:
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = box.conf[0]
                    # class_id = self.model.names[int(box.cls[0])]  # Tidak digunakan
                    if conf > CONFIDENCE_THRESHOLD:
                        boxes_info.append((x1, y1, x2, y2, conf))

            # Gambar bounding boxes dan label "Warning!" jika ada deteksi
            if boxes_info:
                for x1, y1, x2, y2, conf in boxes_info:
                    cvzone.cornerRect(frame_resized, (x1, y1, x2 - x1, y2 - y1), rt=0, colorC=(0, 0, 255))  # Merah
                    cvzone.putTextRect(frame_resized, "Warning!", (x1, y1 - 10), scale=1, thickness=2, offset=5, colorB=(0, 0, 255))

                # Tambahkan teks informasi tetap di pojok kiri atas
                cvzone.putTextRect(frame_resized, "Pelanggaran menggunakan mesin potong manual", (10, 30), scale=1, thickness=2, offset=5, colorB=(255, 255, 255), colorT=(0, 0, 0))

            # Resize frame untuk tampilan
            frame_display = cv2.resize(frame_resized, (DISPLAY_WIDTH, DISPLAY_HEIGHT))

            # Convert frame ke QImage
            rgb_image = cv2.cvtColor(frame_display, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
            self.frame_updated.emit(qt_image)

            # Delay sesuai FPS video
            fps = self.cap.get(cv2.CAP_PROP_FPS)
            if fps > 0:
                self.msleep(int(1000 / fps))
            else:
                self.msleep(30)

    def stop(self):
        self.running = False
        self.wait()
        self.cap.release()


class MainWindow(QMainWindow):
    def __init__(self, video_path, model_path):
        super().__init__()
        self.setWindowTitle("YOLO Object Detection with Warning")
        self.setGeometry(100, 100, DISPLAY_WIDTH, DISPLAY_HEIGHT)

        # Central widget
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        # Layout utama
        self.main_layout = QHBoxLayout()
        self.central_widget.setLayout(self.main_layout)

        # Video display
        self.video_label = QLabel()
        self.video_label.setFixedSize(DISPLAY_WIDTH, DISPLAY_HEIGHT)
        self.video_label.setStyleSheet("background-color: black;")
        self.main_layout.addWidget(self.video_label)

        # Inisialisasi processor video
        self.processor = YOLOVideoProcessor(video_path, model_path)
        self.processor.frame_updated.connect(self.update_frame)
        self.processor.start()

    def update_frame(self, qt_image):
        self.video_label.setPixmap(QPixmap.fromImage(qt_image))

    def closeEvent(self, event):
        # Hentikan processor video saat menutup aplikasi
        self.processor.stop()
        event.accept()


def main():
    app = QApplication(sys.argv)

    # Path ke video lokal dan model YOLO
    video_path = "videos/test/cutgine.mp4"
    model_path = "run/cutting_engine/version1/weights/best.pt"

    window = MainWindow(video_path, model_path)
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
