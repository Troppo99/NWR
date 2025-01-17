import sys
import cv2
import cv2.aruco as aruco
import numpy as np
import os
import json
import time
import math

from PyQt6.QtCore import Qt, pyqtSignal, QThread, QTimer
from PyQt6.QtWidgets import (
    QApplication,
    QMainWindow,
    QLabel,
    QWidget,
    QGridLayout,
    QVBoxLayout,
    QSizePolicy,
    QGroupBox,
    QHBoxLayout,
)
from PyQt6.QtGui import QImage, QPixmap


def load_calibration(file_path="sources/Al-Wajid/calibration/camera_calibration.npz"):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File kalibrasi tidak ditemukan: {file_path}")

    with np.load(file_path) as X:
        camera_matrix, dist_coeffs = X["camera_matrix"], X["dist_coeffs"]

    print("Parameter Kalibrasi Berhasil Dimuat.")
    print("Camera Matrix:")
    print(camera_matrix)
    print("Distortion Coefficients:")
    print(dist_coeffs)

    return camera_matrix, dist_coeffs


class ArucoDetector:
    def __init__(self, dictionary_type=aruco.DICT_4X4_100):
        self.aruco_dict = aruco.getPredefinedDictionary(dictionary_type)
        self.parameters = aruco.DetectorParameters()

    def detect_markers(self, gray_frame):
        corners, ids, rejected = aruco.detectMarkers(gray_frame, self.aruco_dict, parameters=self.parameters)
        return corners, ids, rejected

    def estimate_pose(self, corners, ids, marker_length, camera_matrix, dist_coeffs):
        if ids is not None and len(ids) > 0:
            rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(corners, marker_length, camera_matrix, dist_coeffs)
            return rvecs, tvecs
        else:
            return None, None


class OverlayManager:
    def __init__(self, overlay_config_path, desired_size=(100, 100)):
        self.overlay_images = self.load_overlay_config(overlay_config_path, desired_size)

    def load_overlay_config(self, config_path, desired_size):
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"File konfigurasi overlay tidak ditemukan: {config_path}")

        with open(config_path, "r") as f:
            overlay_config = json.load(f)

        overlay_dict = {}
        for marker_id, image_path in overlay_config.items():
            if not os.path.exists(image_path):
                print(f"Overlay image untuk marker ID {marker_id} tidak ditemukan: {image_path}")
                continue

            overlay = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
            if overlay is None:
                print(f"Gagal membaca gambar overlay untuk marker ID {marker_id}: {image_path}")
                continue

            if overlay.shape[2] == 4:
                overlay_rgb = overlay[:, :, :3]
                overlay_alpha = overlay[:, :, 3]
            else:
                overlay_rgb = overlay
                overlay_alpha = np.ones(overlay_rgb.shape[:2], dtype=np.uint8) * 255

            if (overlay_rgb.shape[1], overlay_rgb.shape[0]) != desired_size:
                overlay_rgb = cv2.resize(overlay_rgb, desired_size, interpolation=cv2.INTER_AREA)
                overlay_alpha = cv2.resize(overlay_alpha, desired_size, interpolation=cv2.INTER_AREA)

            overlay_dict[int(marker_id)] = (overlay_rgb, overlay_alpha)

        return overlay_dict

    def apply_overlay(self, frame, marker_id, marker_corners):
        if marker_id not in self.overlay_images:
            return frame

        overlay_image, overlay_alpha = self.overlay_images[marker_id]

        center_x = int(np.mean(marker_corners[:, 0]))
        center_y = int(np.mean(marker_corners[:, 1]))

        overlay_height, overlay_width = overlay_image.shape[:2]
        top_left_x = center_x - overlay_width // 2
        top_left_y = center_y - overlay_height // 2

        frame_height, frame_width = frame.shape[:2]
        if top_left_x < 0 or top_left_y < 0 or top_left_x + overlay_width > frame_width or top_left_y + overlay_height > frame_height:
            print(f"ArucoDetectorApp: Overlay untuk marker ID {marker_id} melebihi batas frame.")
            return frame

        roi = frame[
            top_left_y : top_left_y + overlay_height,
            top_left_x : top_left_x + overlay_width,
        ]

        alpha_overlay = overlay_alpha.astype(float) / 255.0
        alpha_frame = 1.0 - alpha_overlay

        for c in range(0, 3):
            roi[:, :, c] = alpha_overlay * overlay_image[:, :, c] + alpha_frame * roi[:, :, c]

        frame[
            top_left_y : top_left_y + overlay_height,
            top_left_x : top_left_x + overlay_width,
        ] = roi

        text = "THIS IS WORK ORDER"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        font_color = (0, 0, 0)
        thickness = 1
        line_type = cv2.LINE_AA
        (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
        text_x = top_left_x + (overlay_width - text_width) // 2
        text_y = top_left_y + overlay_height + text_height + 5
        if text_y + baseline > frame_height:
            print(f"Teks melebihi batas bawah frame untuk marker ID {marker_id}.")
            return frame

        cv2.rectangle(frame, (text_x - 2, text_y - text_height - 2), (text_x + text_width + 2, text_y + baseline + 2), (0, 255, 255), cv2.FILLED)
        cv2.putText(frame, text, (text_x, text_y), font, font_scale, font_color, thickness, line_type)

        return frame


class VideoProcessor(QThread):
    frame_processed = pyqtSignal(QImage, dict)

    def __init__(
        self,
        source,
        overlay_manager,
        detector,
        camera_matrix,
        dist_coeffs,
        marker_length=0.05,
        scale_factor=1.5,
        skip_frames=5,
        camera_location="",
    ):
        super().__init__()
        self.source = source
        self.overlay_manager = overlay_manager
        self.detector = detector
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs
        self.marker_length = marker_length
        self.scale_factor = scale_factor
        self.skip_frames = skip_frames
        self._running = True
        self.camera_location = camera_location

        self.prev_position_wo = None
        self.status_wo = "Tidak terdeteksi"
        self.motion_start_time = None
        self.stationary_start_time = None
        self.total_moving_time = 0
        self.total_stationary_time = 0
        self.speed = 0

        self.movement_threshold = 5

    def run(self):
        while self._running:
            cap = cv2.VideoCapture(self.source)
            if not cap.isOpened():
                print(f"VideoProcessor: Tidak dapat membuka sumber video: {self.source}. Mencoba lagi dalam 5 detik.")
                cap.release()
                time.sleep(5)
                continue

            frame_counter = 0

            while self._running:
                ret, frame = cap.read()
                if not ret:
                    print(f"VideoProcessor: Gagal membaca frame dari sumber: {self.source}. Reconnecting dalam 5 detik...")
                    cap.release()
                    time.sleep(5)
                    break

                if frame_counter % (self.skip_frames + 1) != 0:
                    frame_counter += 1
                    continue

                frame_counter += 1

                try:
                    frame_resized = cv2.resize(frame, (1280, 720), interpolation=cv2.INTER_AREA)
                    gray = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)

                    corners, ids, rejected = self.detector.detect_markers(gray)
                    info = {
                        "camera_location": self.camera_location,
                        "status_wo": self.status_wo,
                        "detected": False,
                        "durasi_bergerak": 0.0,
                        "durasi_diam": 0.0,
                        "kecepatan": 0.0,
                    }

                    if ids is not None and len(ids) > 0:
                        ids = ids.flatten().astype(int)

                        rvecs, tvecs = self.detector.estimate_pose(corners, ids, self.marker_length, self.camera_matrix, self.dist_coeffs)

                        for i, (rvec, tvec) in enumerate(zip(rvecs, tvecs)):
                            marker_id = ids[i]
                            if marker_id in self.overlay_manager.overlay_images:
                                frame_resized = self.overlay_manager.apply_overlay(frame_resized, marker_id, corners[i][0])

                                if marker_id == 0:
                                    info["detected"] = True
                                    current_position = (
                                        int(np.mean(corners[i][0][:, 0])),
                                        int(np.mean(corners[i][0][:, 1])),
                                    )
                                    if self.prev_position_wo is not None:
                                        distance = math.hypot(
                                            current_position[0] - self.prev_position_wo[0],
                                            current_position[1] - self.prev_position_wo[1],
                                        )
                                        self.speed = distance

                                        if distance > self.movement_threshold:
                                            if self.status_wo != "Bergerak":
                                                if self.stationary_start_time:
                                                    duration = time.time() - self.stationary_start_time
                                                    self.total_stationary_time += duration
                                                    self.stationary_start_time = None
                                                self.motion_start_time = time.time()
                                                self.status_wo = "Bergerak"
                                            else:
                                                self.speed = distance
                                        else:
                                            if self.status_wo != "Diam":
                                                if self.motion_start_time:
                                                    duration = time.time() - self.motion_start_time
                                                    self.total_moving_time += duration
                                                    self.motion_start_time = None
                                                self.stationary_start_time = time.time()
                                                self.status_wo = "Diam"
                                            else:
                                                pass
                                    else:
                                        self.motion_start_time = time.time()
                                        self.status_wo = "Bergerak"

                                    self.prev_position_wo = current_position
                            else:
                                print(f"VideoProcessor: Tidak ada overlay untuk marker ID {marker_id}")

                        print(f"VideoProcessor: Detected IDs in {self.source}: {ids}")
                    else:
                        if self.status_wo != "Tidak terdeteksi":
                            if self.status_wo == "Bergerak" and self.motion_start_time:
                                duration = time.time() - self.motion_start_time
                                self.total_moving_time += duration
                                self.motion_start_time = None
                            elif self.status_wo == "Diam" and self.stationary_start_time:
                                duration = time.time() - self.stationary_start_time
                                self.total_stationary_time += duration
                                self.stationary_start_time = None
                            self.status_wo = "Tidak terdeteksi"
                            self.prev_position_wo = None
                            self.speed = 0

                        print(f"VideoProcessor: No markers detected in {self.source}.")

                    if self.status_wo == "Bergerak":
                        if self.motion_start_time:
                            duration = time.time() - self.motion_start_time
                            info["durasi_bergerak"] = duration
                            info["kecepatan"] = self.speed
                    elif self.status_wo == "Diam":
                        if self.stationary_start_time:
                            duration = time.time() - self.stationary_start_time
                            info["durasi_diam"] = duration
                            info["kecepatan"] = self.speed

                    total_time = self.total_moving_time + self.total_stationary_time
                    if total_time > 0:
                        avg_moving = (self.total_moving_time / total_time) * 100
                        avg_stationary = (self.total_stationary_time / total_time) * 100
                        info["average_moving"] = f"Rata-rata bergerak: {avg_moving:.1f}%"
                        info["average_stationary"] = f"Rata-rata diam: {avg_stationary:.1f}%"
                    else:
                        info["average_moving"] = "Rata-rata bergerak: 0.0%"
                        info["average_stationary"] = "Rata-rata diam: 0.0%"

                    # Validasi frame sebelum mengirim
                    if frame_resized is not None and frame_resized.size != 0:
                        rgb_image = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
                        h, w, ch = rgb_image.shape
                        bytes_per_line = ch * w
                        qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
                        self.frame_processed.emit(qt_image, info)
                    else:
                        print(f"VideoProcessor: Frame kosong atau rusak dari {self.source}, melewati frame.")

                except Exception as e:
                    print(f"VideoProcessor: Error memproses frame dari {self.source}: {e}")
                    continue

                time.sleep(0.03)

    def stop(self):
        self._running = False
        self.wait()


class ArucoApp(QMainWindow):
    def __init__(
        self,
        overlay_config_path,
        calibration_file,
        marker_length=0.05,
        scale_factor=1.5,
        sources=None,
    ):
        super().__init__()
        self.setWindowTitle("History Pergerakan WO")
        self.setFixedSize(1600, 800)

        self.camera_matrix, self.dist_coeffs = load_calibration(calibration_file)

        self.detector = ArucoDetector(aruco.DICT_4X4_100)
        self.overlay_manager = OverlayManager(overlay_config_path)

        if sources is None:
            self.sources = [
                ("rtsp://admin:oracle2015@192.168.100.65:554/Streaming/Channels/1", "ROBOTIC"),
                ("rtsp://admin:oracle2015@192.168.100.18:554/Streaming/Channels/1", "PINTU KELUAR"),
                ("rtsp://admin:oracle2015@192.168.100.2:554/Streaming/Channels/1", "LINE TENGAH"),
                ("rtsp://admin:oracle2015@10.5.0.170:554/Streaming/Channels/1", "OFFICE1"),
                ("rtsp://admin:oracle2015@10.5.0.182:554/Streaming/Channels/1", "OFFICE2"),
                ("rtsp://admin:oracle2015@10.5.0.161:554/Streaming/Channels/1", "OFFICE3"),
                ("rtsp://admin:oracle2015@10.5.0.201:554/Streaming/Channels/1", "SEWINGOFFICE"),
                ("rtsp://admin:oracle2015@10.5.0.217:554/Streaming/Channels/1", "SEWING1"),
                ("rtsp://admin:oracle2015@10.5.0.151:554/Streaming/Channels/1", "SEWING2"),
            ]
        else:
            self.sources = sources

        self.setup_ui()

        self.video_processors = []
        for source, location in self.sources:
            processor = VideoProcessor(
                source=source,
                overlay_manager=self.overlay_manager,
                detector=self.detector,
                camera_matrix=self.camera_matrix,
                dist_coeffs=self.dist_coeffs,
                marker_length=marker_length,
                scale_factor=scale_factor,
                skip_frames=2,  # Contoh: proses setiap 3 frame (skip 2)
                camera_location=location,
            )
            processor.frame_processed.connect(self.update_frame)
            self.video_processors.append(processor)

        for processor in self.video_processors:
            processor.start()

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_average)
        self.timer.start(5000)

        self.detection_order = {}
        self.current_detection_number = 1

    def setup_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        main_grid = QGridLayout()
        central_widget.setLayout(main_grid)

        video_grid = QGridLayout()
        main_grid.addLayout(video_grid, 0, 0, 1, 1)

        self.video_labels = []

        for i, (source, location) in enumerate(self.sources):
            row = i // 3
            col = i % 3

            video_group = QGroupBox(f"Video {i + 1} - {location}")
            video_group_layout = QVBoxLayout()
            video_group.setLayout(video_group_layout)
            video_grid.addWidget(video_group, row, col)

            video_label = QLabel()
            video_label.setStyleSheet("background-color: black;")
            video_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
            video_group_layout.addWidget(video_label)
            self.video_labels.append(video_label)

        info_panel = QVBoxLayout()
        info_panel.setContentsMargins(10, 10, 10, 10)
        main_grid.addLayout(info_panel, 0, 1, 1, 1)

        info_panel_widget = QWidget()
        info_panel_widget.setLayout(info_panel)
        info_panel_widget.setFixedWidth(500)
        main_grid.addWidget(info_panel_widget, 0, 1, 1, 1)

        main_grid.setColumnStretch(0, 3)
        main_grid.setColumnStretch(1, 1)

        info_title = QLabel("History Pergerakan WO")
        info_title.setAlignment(Qt.AlignmentFlag.AlignLeft)
        info_title.setStyleSheet("font-weight: bold; font-size: 16px;")
        info_panel.addWidget(info_title)

        # 1. Grup Tabel Pelacakan WO
        tracking_table_group = QGroupBox("Tabel Pelacakan WO")
        tracking_table_layout = QVBoxLayout()
        tracking_table_group.setLayout(tracking_table_layout)
        info_panel.addWidget(tracking_table_group)

        self.history_grid = QGridLayout()
        self.history_grid.setSpacing(5)
        tracking_table_layout.addLayout(self.history_grid)

        # Header Tabel Pelacakan WO
        headers = ["Kamera", "Status WO", "Durasi Diam", "Kecepatan"]
        for col, header in enumerate(headers):
            header_label = QLabel(f"<b>{header}</b>")
            header_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
            header_label.setFixedWidth(100)
            self.history_grid.addWidget(header_label, 0, col)

        # Baris Data Tabel Pelacakan WO
        self.history_widgets = []
        for i, (source, location) in enumerate(self.sources):
            row = i + 1
            camera_label = QLabel(location)
            camera_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
            camera_label.setFixedWidth(100)
            self.history_grid.addWidget(camera_label, row, 0)

            status_label = QLabel("Tidak terdeteksi")
            status_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
            status_label.setFixedWidth(100)
            self.history_grid.addWidget(status_label, row, 1)

            durasi_diam_label = QLabel("-")
            durasi_diam_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
            durasi_diam_label.setFixedWidth(100)
            self.history_grid.addWidget(durasi_diam_label, row, 2)

            kecepatan_label = QLabel("-")
            kecepatan_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
            kecepatan_label.setFixedWidth(100)
            self.history_grid.addWidget(kecepatan_label, row, 3)

            self.history_widgets.append(
                {
                    "status_label": status_label,
                    "durasi_diam_label": durasi_diam_label,
                    "kecepatan_label": kecepatan_label,
                }
            )

        # 2. Grup Ringkasan Pelacakan WO
        average_group = QGroupBox("Ringkasan Pelacakan WO")
        average_layout = QVBoxLayout()
        average_group.setLayout(average_layout)
        average_group.setFixedWidth(480)
        info_panel.addWidget(average_group)

        self.average_label = QLabel("Rata-rata bergerak: 0.0%\nRata-rata diam: 0.0%")
        self.average_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        self.average_label.setFixedWidth(480)
        average_layout.addWidget(self.average_label)

        # 3. Grup Status Deteksi WO
        status_detection_group = QGroupBox("Status Deteksi WO")
        status_detection_layout = QVBoxLayout()
        status_detection_group.setLayout(status_detection_layout)
        info_panel.addWidget(status_detection_group)

        # Header Tabel Status Deteksi WO
        status_headers = ["Kamera", "Deteksi ke-", "Timestamp"]
        self.status_detection_grid = QGridLayout()
        self.status_detection_grid.setSpacing(5)
        status_detection_layout.addLayout(self.status_detection_grid)

        for col, header in enumerate(status_headers):
            header_label = QLabel(f"<b>{header}</b>")
            header_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
            header_label.setFixedWidth(150)
            self.status_detection_grid.addWidget(header_label, 0, col)

        # Baris Data Status Deteksi WO
        self.status_detection_widgets = []
        for i, (source, location) in enumerate(self.sources):
            row = i + 1
            camera_label = QLabel(location)
            camera_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
            camera_label.setFixedWidth(150)
            self.status_detection_grid.addWidget(camera_label, row, 0)

            deteksi_ke_label = QLabel("-")
            deteksi_ke_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
            deteksi_ke_label.setFixedWidth(150)
            self.status_detection_grid.addWidget(deteksi_ke_label, row, 1)

            timestamp_label = QLabel("-")
            timestamp_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
            timestamp_label.setFixedWidth(150)
            self.status_detection_grid.addWidget(timestamp_label, row, 2)

            self.status_detection_widgets.append(
                {
                    "deteksi_ke_label": deteksi_ke_label,
                    "timestamp_label": timestamp_label,
                }
            )

    def update_frame(self, qt_image, info):
        sender = self.sender()
        if sender not in self.video_processors:
            return
        index = self.video_processors.index(sender)
        if index >= len(self.video_labels):
            return

        self.video_labels[index].setPixmap(
            QPixmap.fromImage(qt_image).scaled(
                self.video_labels[index].width(),
                self.video_labels[index].height(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )

        history = self.history_widgets[index]
        history["status_label"].setText(f"{info['status_wo']}")
        history["durasi_diam_label"].setText(f"{info['durasi_diam']:.1f}" if info["durasi_diam"] > 0 else "-")
        history["kecepatan_label"].setText(f"{info['kecepatan']:.1f}" if info["kecepatan"] > 0 else "-")

        # Pembaruan Status Deteksi WO
        if info["detected"] and index not in self.detection_order:
            self.detection_order[index] = self.current_detection_number
            self.current_detection_number += 1
            deteksi_ke = self.detection_order[index]
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            self.status_detection_widgets[index]["deteksi_ke_label"].setText(f"Deteksi ke-{deteksi_ke}")
            self.status_detection_widgets[index]["timestamp_label"].setText(timestamp)
        elif not info["detected"]:
            if index in self.detection_order:
                pass
            else:
                self.status_detection_widgets[index]["deteksi_ke_label"].setText("-")
                self.status_detection_widgets[index]["timestamp_label"].setText("-")
        else:
            if index in self.detection_order:
                deteksi_ke = self.detection_order[index]
                timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                self.status_detection_widgets[index]["deteksi_ke_label"].setText(f"Deteksi ke-{deteksi_ke}")
                self.status_detection_widgets[index]["timestamp_label"].setText(timestamp)

    def update_average(self):
        total_moving = 0
        total_stationary = 0
        count = 0
        for processor in self.video_processors:
            total_moving += processor.total_moving_time
            total_stationary += processor.total_stationary_time
            count += 1

        total_time = total_moving + total_stationary
        if total_time > 0:
            avg_moving = (total_moving / total_time) * 100
            avg_stationary = (total_stationary / total_time) * 100
        else:
            avg_moving = 0.0
            avg_stationary = 0.0

        self.average_label.setText(f"Rata-rata bergerak: {avg_moving:.1f}%\nRata-rata diam: {avg_stationary:.1f}%")

    def closeEvent(self, event):
        for processor in self.video_processors:
            processor.stop()
        event.accept()


def run_aruco_app():
    app = QApplication(sys.argv)
    overlay_config_path = "sources/Al-Wajid/overlay_config.json"
    calibration_file = "sources/Al-Wajid/calibration/camera_calibration.npz"
    marker_length = 0.05
    scale_factor = 2

    sources = [
        ("rtsp://admin:oracle2015@192.168.100.65:554/Streaming/Channels/1", "ROBOTIC"),
        ("rtsp://admin:oracle2015@192.168.100.18:554/Streaming/Channels/1", "PINTU KELUAR"),
        ("rtsp://admin:oracle2015@192.168.100.2:554/Streaming/Channels/1", "LINE TENGAH"),
        ("rtsp://admin:oracle2015@10.5.0.170:554/Streaming/Channels/1", "OFFICE1"),
        ("rtsp://admin:oracle2015@10.5.0.182:554/Streaming/Channels/1", "OFFICE2"),
        ("rtsp://admin:oracle2015@10.5.0.161:554/Streaming/Channels/1", "OFFICE3"),
        ("rtsp://admin:oracle2015@10.5.0.201:554/Streaming/Channels/1", "SEWINGOFFICE"),
        ("rtsp://admin:oracle2015@10.5.0.217:554/Streaming/Channels/1", "SEWING1"),
        ("rtsp://admin:oracle2015@10.5.0.151:554/Streaming/Channels/1", "SEWING2"),
    ]

    window = ArucoApp(
        overlay_config_path=overlay_config_path,
        calibration_file=calibration_file,
        marker_length=marker_length,
        scale_factor=scale_factor,
        sources=sources,
    )
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    run_aruco_app()
