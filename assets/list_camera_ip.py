import sys
import cv2
import numpy as np
from PyQt6.QtWidgets import QApplication, QWidget, QPushButton, QLabel, QVBoxLayout, QHBoxLayout, QListWidget, QMessageBox
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtCore import Qt, QThread, pyqtSignal

# Konstanta ukuran frame
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720

# Daftar IP kamera untuk masing-masing NVR
nvr_a_ips = [(66, "cutting 1"), (58, "cutting 2"), (92, "cutting 3"), (101, "cutting 4"), (87, "cutting 5"), (43, "cutting 6"), (61, "cutting 7"), (95, "cutting 8"), (120, "cutting 9"), (143, "cutting 10"), (10, "gudang kain 4"), (13, "gudang kain 3"), (4, "gudang kain 1"), (51, "gudang kain 2"), (7, "folding 1"), (82, "metal detector 1")]

nvr_b_ips = [(110, "gudang acc 1"), (107, "gudang acc 2"), (168, "gudang acc 3"), (180, "gudang acc 4"), (12, "buffer 1"), (30, "inner box"), (121, "folding 3 (back)"), (105, "folding 2 (side)"), (185, "metal detector 2"), (155, "load expedisi"), (181, "gudang kain 5"), (104, "free metal 1"), (123, "free metal 2"), (170, "office 1"), (182, "office 2"), (161, "office 3")]

nvr_c_ips = [(150, "no detect"), (159, "line 5-6"), (146, "lorong line 4"), (183, "adm prod 1"), (195, "adm prod 2"), (201, "lorong line 1"), (202, "adm prod 3"), (206, "luar expedisi"), (205, "line persiapan 1"), (207, "line persiapan 2"), (214, "line 3-4"), (219, "line 7-8"), (217, "line 1-2"), (228, "kantin luar"), (239, "kantin mushola"), (252, "line 6-7")]

nvr_d_ips = [(236, "no detect"), (245, "no detect")]


class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)

    def __init__(self, ip_address):
        super().__init__()
        self.ip_address = ip_address
        self._run_flag = True

    def run(self):
        rtsp_url = f"rtsp://admin:oracle2015@10.5.0.{self.ip_address}:554/Streaming/Channels/1"
        cap = cv2.VideoCapture(rtsp_url)
        while self._run_flag:
            ret, cv_img = cap.read()
            if ret:
                self.change_pixmap_signal.emit(cv_img)
        cap.release()

    def stop(self):
        self._run_flag = False
        self.wait()


class CCTVViewer(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("CCTV Viewer")
        self.setGeometry(100, 100, FRAME_WIDTH, FRAME_HEIGHT)

        # Inisialisasi variabel
        self.current_thread = None
        self.current_camera_ip = None

        # Layout utama
        main_layout = QHBoxLayout()

        # Panel tombol
        button_layout = QVBoxLayout()

        self.btn_nvr_a = QPushButton("NVR A")
        self.btn_nvr_b = QPushButton("NVR B")
        self.btn_nvr_c = QPushButton("NVR C")
        self.btn_nvr_d = QPushButton("NVR D")
        self.btn_hide = QPushButton("Hide Utama")

        button_layout.addWidget(self.btn_nvr_a)
        button_layout.addWidget(self.btn_nvr_b)
        button_layout.addWidget(self.btn_nvr_c)
        button_layout.addWidget(self.btn_nvr_d)
        button_layout.addWidget(self.btn_hide)
        button_layout.addStretch()

        # Menghubungkan tombol dengan fungsi
        self.btn_nvr_a.clicked.connect(lambda: self.show_ip_list("A"))
        self.btn_nvr_b.clicked.connect(lambda: self.show_ip_list("B"))
        self.btn_nvr_c.clicked.connect(lambda: self.show_ip_list("C"))
        self.btn_nvr_d.clicked.connect(lambda: self.show_ip_list("D"))
        self.btn_hide.clicked.connect(self.hide_all)

        # Panel daftar IP kamera
        self.list_widget = QListWidget()
        self.list_widget.clicked.connect(self.select_camera)

        # Panel video
        self.video_label = QLabel()
        self.video_label.setFixedSize(640, 480)
        self.video_label.setStyleSheet("background-color: black;")

        # Tambahkan widget ke layout utama
        main_layout.addLayout(button_layout)
        main_layout.addWidget(self.list_widget)
        main_layout.addWidget(self.video_label)

        self.setLayout(main_layout)

        # Inisialisasi video_thread
        self.video_thread = None

    def show_ip_list(self, nvr):
        self.list_widget.clear()
        if nvr == "A":
            for ip, desc in nvr_a_ips:
                self.list_widget.addItem(f"{desc} : 10.5.0.{ip}")
        elif nvr == "B":
            for ip, desc in nvr_b_ips:
                self.list_widget.addItem(f"{desc} : 10.5.0.{ip}")
        elif nvr == "C":
            for ip, desc in nvr_c_ips:
                self.list_widget.addItem(f"{desc} : 10.5.0.{ip}")
        elif nvr == "D":
            for ip, desc in nvr_d_ips:
                self.list_widget.addItem(f"{desc} : 10.5.0.{ip}")

    def hide_all(self):
        self.list_widget.clear()
        self.stop_video()

    def select_camera(self, index):
        item_text = self.list_widget.currentItem().text()
        ip_address = item_text.split(":")[-1].strip()
        ip = ip_address.split(".")[-1]
        self.start_video(int(ip))

    def start_video(self, ip_address):
        self.stop_video()
        self.current_camera_ip = ip_address
        self.video_thread = VideoThread(ip_address)
        self.video_thread.change_pixmap_signal.connect(self.update_image)
        self.video_thread.start()

    def stop_video(self):
        if hasattr(self, "video_thread") and self.video_thread is not None:
            if self.video_thread.isRunning():
                self.video_thread.stop()
            self.video_label.clear()
            self.video_label.setStyleSheet("background-color: black;")
            self.video_thread = None

    def closeEvent(self, event):
        self.stop_video()
        event.accept()

    def update_image(self, cv_img):
        """Mengubah frame OpenCV menjadi format yang bisa ditampilkan di QLabel."""
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        scaled_image = qt_image.scaled(self.video_label.width(), self.video_label.height(), Qt.AspectRatioMode.KeepAspectRatio)
        self.video_label.setPixmap(QPixmap.fromImage(scaled_image))


def main():
    app = QApplication(sys.argv)
    viewer = CCTVViewer()
    viewer.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
