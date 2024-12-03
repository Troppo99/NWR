import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
import sys
import cv2
import numpy as np
import time
from PyQt6.QtWidgets import QApplication, QMainWindow, QLabel, QWidget, QGridLayout, QScrollArea
from PyQt6.QtGui import QPixmap, QImage
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from functools import partial
from ultralytics import YOLO
import cvzone
import queue


class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)
    stream_failed_signal = pyqtSignal(str)
    stream_reconnecting_signal = pyqtSignal(str)

    def __init__(self, rtsp_url):
        super().__init__()
        self.rtsp_url = rtsp_url
        self._run_flag = True
        self.model = YOLO("models/yolo11l.pt")
        self.model.overrides["verbose"] = False

    def run(self):
        while self._run_flag:
            cap = cv2.VideoCapture(self.rtsp_url)
            if not cap.isOpened():
                self.stream_failed_signal.emit(self.rtsp_url)
                self.stream_reconnecting_signal.emit(self.rtsp_url)
                cap.release()
                time.sleep(5)
                continue
            while self._run_flag:
                ret, frame = cap.read()
                if not ret:
                    self.stream_failed_signal.emit(self.rtsp_url)
                    self.stream_reconnecting_signal.emit(self.rtsp_url)
                    cap.release()
                    time.sleep(5)
                    break
                results = self.model(frame)
                boxes_info = []
                for result in results:
                    for box in result.boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        conf = box.conf[0]
                        class_id = self.model.names[int(box.cls[0])]
                        if conf > 0.5:
                            boxes_info.append((x1, y1, x2, y2, conf, class_id))
                for x1, y1, x2, y2, conf, class_id in boxes_info:
                    cvzone.cornerRect(frame, (x1, y1, x2 - x1, y2 - y1), rt=0, colorC=(0, 255, 255))
                    cvzone.putTextRect(frame, f"{class_id} {conf:.2f}", (x1, y1 - 15))
                self.change_pixmap_signal.emit(frame)
        cap.release()

    def stop(self):
        self._run_flag = False
        self.wait()


class MainWindow(QMainWindow):
    def __init__(self, rtsp_urls, columns=2):
        super().__init__()
        self.setWindowTitle("RTSP Stream Viewer")
        self.rtsp_urls = rtsp_urls
        self.columns = columns
        self.threads = []
        self.init_ui()
        self.start_threads()
        self.resize(1280, 720)

    def init_ui(self):
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.grid_widget = QWidget()
        self.grid_layout = QGridLayout()
        self.grid_widget.setLayout(self.grid_layout)
        self.scroll_area.setWidget(self.grid_widget)
        main_layout = QGridLayout()
        main_layout.addWidget(self.scroll_area, 0, 0)
        self.central_widget.setLayout(main_layout)
        self.labels = []
        for idx, url in enumerate(self.rtsp_urls):
            row = idx // self.columns
            col = idx % self.columns
            label = QLabel()
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            label.setStyleSheet("background-color: black; color: white;")
            label.setFixedSize(640, 360)
            label.setText(f"Connecting...\n{url}")
            label.setScaledContents(True)
            self.grid_layout.addWidget(label, row, col)
            self.labels.append(label)

    def start_threads(self):
        for idx, url in enumerate(self.rtsp_urls):
            thread = VideoThread(url)
            thread.change_pixmap_signal.connect(partial(self.update_image, label_index=idx))
            thread.stream_failed_signal.connect(lambda u, l=idx: self.handle_stream_failure(l, u))
            thread.stream_reconnecting_signal.connect(lambda u, l=idx: self.handle_reconnecting(l, u))
            thread.start()
            self.threads.append(thread)

    def update_image(self, cv_img, label_index):
        qt_img = self.convert_cv_qt(cv_img)
        self.labels[label_index].setPixmap(qt_img)
        self.labels[label_index].setText("")
        self.labels[label_index].setStyleSheet("background-color: black; color: white;")

    def handle_stream_failure(self, label_index, rtsp_url):
        self.labels[label_index].setText(f"Stream Error:\n{rtsp_url}")
        self.labels[label_index].setStyleSheet("background-color: red; color: white;")
        self.labels[label_index].setPixmap(QPixmap())

    def handle_reconnecting(self, label_index, rtsp_url):
        self.labels[label_index].setText(f"Reconnecting...\n{rtsp_url}")
        self.labels[label_index].setStyleSheet("background-color: yellow; color: black;")
        self.labels[label_index].setPixmap(QPixmap())

    def convert_cv_qt(self, cv_img):
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        q_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        p = q_image.scaled(640, 360, Qt.AspectRatioMode.KeepAspectRatio)
        return QPixmap.fromImage(p)

    def closeEvent(self, event):
        for thread in self.threads:
            thread.stop()
        event.accept()


def main():
    rtsp_urls = [
        "rtsp://admin:oracle2015@10.5.0.7:554/Streaming/Channels/1",
        "rtsp://admin:oracle2015@10.5.0.170:554/Streaming/Channels/1",
        "rtsp://admin:oracle2015@10.5.0.161:554/Streaming/Channels/1",
        "rtsp://admin:oracle2015@10.5.0.182:554/Streaming/Channels/1",
    ]

    app = QApplication(sys.argv)
    window = MainWindow(rtsp_urls, columns=2)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
