import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
import sys
import cv2
import numpy as np
import time
import threading
import queue
import subprocess
from PyQt6.QtWidgets import QApplication, QMainWindow, QLabel, QWidget, QGridLayout, QScrollArea
from PyQt6.QtGui import QPixmap, QImage
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from ultralytics import YOLO
import cvzone
import torch


def get_ffmpeg_process(rtsp_url, width=640, height=360):
    # Perintah FFmpeg untuk mentranskode stream ke H.264, resize, dan output sebagai raw BGR24
    command = ["ffmpeg", "-i", rtsp_url, "-vf", f"scale={width}:{height}", "-c:v", "rawvideo", "-pix_fmt", "bgr24", "-f", "rawvideo", "-"]  # Resize frame
    return subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=10**8)


class PaperDetector(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)

    def __init__(self, rtsp_url, camera_name, width=640, height=360):
        super().__init__()
        self.rtsp_url = rtsp_url
        self.camera_name = camera_name
        self.width = width
        self.height = height
        self._run_flag = True
        self.frame_queue = queue.Queue(maxsize=10)
        self.stop_event = threading.Event()
        self.model = YOLO("D:/NWR/run/paper/THREE/weights/best.pt").to("cuda")
        self.model.overrides["verbose"] = False
        print(f"Model device: {next(self.model.model.parameters()).device}")

        # Inisialisasi status border
        self.border_status = {}
        # Warna asli border
        self.original_border_color = (255, 255, 0)  # Biru muda
        # Warna border ketika overlapping lebih dari 5 detik
        self.alert_border_color = (0, 0, 255)  # Merah

        # Inisialisasi FFmpeg process
        self.ffmpeg_process = get_ffmpeg_process(self.rtsp_url, self.width, self.height)

    def export_frame(self, results):
        boxes_info = []
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = box.conf[0]
                class_id = self.model.names[int(box.cls[0])]
                if conf > 0:
                    boxes_info.append((x1, y1, x2, y2, conf, class_id))
        return boxes_info

    def process_model(self, frame):
        with torch.no_grad():
            results = self.model.predict(frame, stream=True, imgsz=960)
        return results

    def overlay_image_alpha(self, img, img_overlay, pos, alpha_mask):
        x, y = pos
        h, w = img_overlay.shape[0], img_overlay.shape[1]
        if y + h > img.shape[0] or x + w > img.shape[1]:
            h = min(h, img.shape[0] - y)
            w = min(w, img.shape[1] - x)
            img_overlay = img_overlay[:h, :w]
            alpha_mask = alpha_mask[:h, :w]
        img[y : y + h, x : x + w] = (img[y : y + h, x : x + w] * (1 - alpha_mask[:, :, None]) + img_overlay * alpha_mask[:, :, None]).astype(np.uint8)

    def is_valid_frame(self, frame):
        # Contoh validasi: cek variansi warna
        variance = np.var(frame)
        if variance < 100:  # Threshold ini bisa disesuaikan
            return False
        return True

    def process_frame(self, frame):
        # Pastikan frame dapat dimodifikasi
        frame = frame.copy()

        results = self.process_model(frame)
        boxes_info = self.export_frame(results)

        # Definisikan border untuk masing-masing kamera
        border_camera1 = [
            [(int(x * self.width / 1280), int(y * self.height / 720)) for (x, y) in [(1022, 400), (1101, 405), (1151, 546), (1013, 552), (1013, 552), (1043, 443)]],
            [(int(x * self.width / 1280), int(y * self.height / 720)) for (x, y) in [(440, 347), (573, 354), (559, 417), (559, 535), (364, 516)]],
            [(int(x * self.width / 1280), int(y * self.height / 720)) for (x, y) in [(219, 329), (116, 473), (116, 473), (241, 500), (340, 341)]],
            [(int(x * self.width / 1280), int(y * self.height / 720)) for (x, y) in [(71, 322), (1, 409), (1, 449), (49, 458), (153, 331)]],
            [(int(x * self.width / 1280), int(y * self.height / 720)) for (x, y) in [(301, 121), (360, 118), (320, 152), (256, 157), (256, 157)]],
            [(int(x * self.width / 1280), int(y * self.height / 720)) for (x, y) in [(423, 109), (501, 110), (460, 152), (381, 150), (381, 150)]],
            [(int(x * self.width / 1280), int(y * self.height / 720)) for (x, y) in [(580, 109), (556, 152), (556, 152), (650, 157), (653, 114)]],
        ]

        border_camera2 = [
            [(int(x * self.width / 1280), int(y * self.height / 720)) for (x, y) in [(576, 74), (691, 99), (689, 184), (538, 148)]],
            [(int(x * self.width / 1280), int(y * self.height / 720)) for (x, y) in [(538, 148), (689, 184), (683, 335), (491, 292)]],
            [(int(x * self.width / 1280), int(y * self.height / 720)) for (x, y) in [(460, 422), (685, 469), (673, 723), (386, 718)]],
        ]

        border_camera3 = [[(int(x * self.width / 1280), int(y * self.height / 720)) for (x, y) in [(1072, 511), (1174, 490), (1257, 708), (1219, 719), (1141, 719)]]]

        # Tentukan apakah kamera adalah Camera1 atau Camera2
        if self.camera_name == "OFFICE1":
            borders = border_camera1
        elif self.camera_name == "GUDANGACC1":
            borders = border_camera2
        elif self.camera_name == "OFFICE3":
            borders = border_camera3
        else:
            borders = []

        current_time = time.time()

        # Inisialisasi status untuk setiap border jika belum ada
        for idx in range(len(borders)):
            if idx not in self.border_status:
                self.border_status[idx] = {"is_overlapping": False, "overlap_start_time": None, "overlap_end_time": None, "current_color": self.original_border_color}

        # Gambarkan setiap border
        for idx, border in enumerate(borders):
            border_info = self.border_status[idx]

            # Cek apakah ada objek yang tumpang tindih dengan border ini
            overlap = False
            for x1, y1, x2, y2, conf, class_id in boxes_info:
                if self.overlap_check_single_border(x1, y1, x2, y2, border):
                    overlap = True
                    # Gambarkan bounding box (opsional)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 1)
                    break  # Tidak perlu mengecek lebih lanjut jika sudah ada overlap

            # Update status overlapping
            if overlap:
                if not border_info["is_overlapping"]:
                    # Mulai overlapping
                    border_info["is_overlapping"] = True
                    border_info["overlap_start_time"] = current_time
                    border_info["overlap_end_time"] = None
                else:
                    # Sudah overlapping, cek durasi
                    duration = current_time - border_info["overlap_start_time"]
                    if duration >= 5 and border_info["current_color"] != self.alert_border_color:
                        # Ubah warna border menjadi merah
                        border_info["current_color"] = self.alert_border_color
            else:
                if border_info["is_overlapping"]:
                    # Overlapping baru saja berhenti
                    border_info["is_overlapping"] = False
                    border_info["overlap_end_time"] = current_time
                else:
                    if border_info["overlap_end_time"] is not None:
                        # Cek apakah sudah 3 detik tanpa overlapping
                        no_overlap_duration = current_time - border_info["overlap_end_time"]
                        if no_overlap_duration >= 3:
                            # Kembalikan warna border ke warna asal
                            border_info["current_color"] = self.original_border_color
                            border_info["overlap_end_time"] = None

            # Tentukan teks yang akan ditampilkan
            if border_info["current_color"] == self.alert_border_color:
                text = f"{idx+1}: Warning"
                color = (0, 0, 255)  # Merah untuk "Warning"
            else:
                text = f"{idx+1}: Clear"
                color = (55, 205, 0)  # Hijau untuk "Clear"

            # Gambarkan border dengan warna sesuai status
            try:
                cv2.polylines(frame, [np.array(border)], True, border_info["current_color"], 2)
            except Exception as e:
                print(f"{self.camera_name} : Error drawing polylines: {e}")

            # Tentukan posisi untuk teks (misalnya, pada titik pertama border)
            text_position = border[0]

            # Tambahkan teks pada frame
            cvzone.putTextRect(frame, text, text_position, colorT=(255, 255, 255), scale=1, thickness=2, offset=3, colorR=color)

        # Kembalikan frame yang telah diproses
        return frame

    def overlap_check_single_border(self, x1, y1, x2, y2, border):
        # Periksa apakah salah satu sudut kotak berada di dalam border
        rect = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)
        polygon = np.array(border, dtype=np.int32)
        for point in rect:
            x = float(point[0])
            y = float(point[1])
            if cv2.pointPolygonTest(polygon, (x, y), False) >= 0:
                return True
        return False

    def frame_capture(self):
        ffmpeg = self.ffmpeg_process
        frame_size = self.width * self.height * 3  # 640 * 360 * 3
        while not self.stop_event.is_set():
            # Baca frame dari stdout FFmpeg
            raw_frame = ffmpeg.stdout.read(frame_size)
            if len(raw_frame) != frame_size:
                # Baca stderr FFmpeg untuk melihat error
                stderr_output = ffmpeg.stderr.read(1024).decode("utf-8")
                print(f"{self.camera_name} : Failed to read frame from FFmpeg.")
                print(f"FFmpeg stderr: {stderr_output}")
                break
            try:
                frame = np.frombuffer(raw_frame, np.uint8).reshape((self.height, self.width, 3))
            except ValueError as e:
                print(f"{self.camera_name} : Frame reshaping error: {e}")
                continue

            # Validasi frame
            if not self.is_valid_frame(frame):
                print(f"{self.camera_name} : Invalid frame detected.")
                continue

            try:
                self.frame_queue.put(frame, timeout=1)
            except queue.Full:
                pass
        ffmpeg.terminate()

    def run(self):
        threading.Thread(target=self.frame_capture, daemon=True).start()
        while self._run_flag:
            try:
                frame = self.frame_queue.get(timeout=5)
            except queue.Empty:
                continue

            frame_processed = self.process_frame(frame)
            self.change_pixmap_signal.emit(frame_processed)

    def stop(self):
        self._run_flag = False
        self.stop_event.set()
        self.ffmpeg_process.terminate()
        self.wait()


class MainWindow(QMainWindow):
    def __init__(self, rtsp_urls, camera_names, columns=2):
        super().__init__()
        self.setWindowTitle("RTSP Stream Viewer with YOLO Inference")
        self.rtsp_urls = rtsp_urls
        self.camera_names = camera_names
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
            camera_name = self.camera_names[idx] if idx < len(self.camera_names) else f"Camera{idx+1}"
            # Sesuaikan resolusi sesuai dengan yang diinginkan (640 width)
            thread = PaperDetector(url, camera_name, width=640, height=360)
            thread.change_pixmap_signal.connect(lambda cv_img, label_index=idx: self.update_image(cv_img, label_index))
            thread.start()
            self.threads.append(thread)

    def update_image(self, cv_img, label_index):
        qt_img = self.convert_cv_qt(cv_img)
        self.labels[label_index].setPixmap(qt_img)
        self.labels[label_index].setText("")
        self.labels[label_index].setStyleSheet("background-color: black; color: white;")

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
        "rtsp://admin:oracle2015@10.5.0.170:554/Streaming/Channels/1",
        "rtsp://admin:oracle2015@10.5.0.110:554/Streaming/Channels/1",
        "rtsp://admin:oracle2015@10.5.0.161:554/Streaming/Channels/1",
        "rtsp://admin:oracle2015@10.5.0.7:554/Streaming/Channels/1",
        # Tambahkan RTSP URL lainnya di sini
    ]
    camera_names = [
        "OFFICE1",
        "GUDANGACC1",
        "OFFICE3",
        "FOLDING1",
        # Tambahkan nama kamera sesuai RTSP URL
    ]

    app = QApplication(sys.argv)
    window = MainWindow(rtsp_urls, camera_names)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
