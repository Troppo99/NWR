import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
import threading
import queue
import time
import os
from PyQt6.QtWidgets import QApplication, QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout, QGridLayout, QMessageBox
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtCore import Qt, QTimer, QThread, pyqtSignal, pyqtSlot


class AnomalyDetection(QThread):
    # Sinyal untuk mengirim frame yang telah diproses ke GUI
    frame_processed = pyqtSignal(np.ndarray)

    def __init__(self):
        super().__init__()
        self.rois = [[(57, 465), (225, 430), (236, 514), (220, 557), (78, 594)], [(387, 758), (472, 734), (480, 820), (393, 850)]]
        # Path folder untuk gambar referensi
        self.reference_folder = "D:/NWR/sources/AlFaruq/media/"
        self.reference_filename = "room0.jpg"
        self.reference_path = os.path.join(self.reference_folder, self.reference_filename)
        self.reference_img = cv2.imread(self.reference_path)
        if self.reference_img is None:
            raise ValueError(f"Tidak dapat membaca gambar referensi dari {self.reference_path}")

        self.cap = cv2.VideoCapture("rtsp://admin:oracle2015@192.168.100.65:554/Streaming/Channels/1")
        # Alternatif sumber video lokal
        # self.cap = cv2.VideoCapture("C:/path/to/local/video.mp4")

        ret, frame = self.cap.read()
        if not ret:
            raise ValueError("Tidak dapat membaca frame dari sumber video.")
        self.frame_height, self.frame_width = frame.shape[:2]
        self.reference_img = cv2.resize(self.reference_img, (self.frame_width, self.frame_height))
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        self.reference_display = self.reference_img.copy()
        for roi in self.rois:
            cv2.polylines(self.reference_display, [np.array(roi, dtype=np.int32)], isClosed=True, color=(0, 255, 0), thickness=2)

        self.precomputed_masks = []
        self.bounding_boxes = []
        self.cropped_polygons = []
        for roi in self.rois:
            mask = self.create_polygon_mask(self.reference_img.shape[:2], roi)
            x, y, w, h = cv2.boundingRect(np.array(roi, dtype=np.int32))
            self.bounding_boxes.append((x, y, w, h))
            cropped_polygon = [(pt[0] - x, pt[1] - y) for pt in roi]
            self.cropped_polygons.append(cropped_polygon)
            cropped_mask = self.create_polygon_mask((h, w), cropped_polygon)
            self.precomputed_masks.append(cropped_mask)

        self.capture_queue = queue.Queue(maxsize=10)
        self.process_queue = queue.Queue(maxsize=10)
        self.stop_event = threading.Event()
        self.frame_counter = 0
        self.frame_count = 0
        self.start_time = time.time()
        self.lock = threading.Lock()  # Untuk sinkronisasi pembaruan referensi

        self.latest_frame = None  # Menyimpan frame terbaru yang diproses
        self.latest_original_frame = None  # Menyimpan frame asli terbaru

    def create_polygon_mask(self, image_shape, polygon):
        mask = np.zeros(image_shape, dtype=np.uint8)
        pts = np.array([polygon], dtype=np.int32)
        cv2.fillPoly(mask, pts, 255)
        return mask

    def align_images(self, reference_roi, target_roi, max_features=500, good_match_percent=0.15):
        try:
            gray_ref = cv2.cvtColor(reference_roi, cv2.COLOR_BGR2GRAY)
            gray_target = cv2.cvtColor(target_roi, cv2.COLOR_BGR2GRAY)
            orb = cv2.ORB_create(max_features)
            keypoints_ref, descriptors_ref = orb.detectAndCompute(gray_ref, None)
            keypoints_target, descriptors_target = orb.detectAndCompute(gray_target, None)
            if descriptors_ref is None or descriptors_target is None:
                print("Tidak ditemukan deskriptor dalam salah satu ROI.")
                return target_roi

            matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
            matches = matcher.match(descriptors_ref, descriptors_target, None)
            if len(matches) == 0:
                print("Tidak ada kecocokan fitur ditemukan.")
                return target_roi

            matches = sorted(matches, key=lambda x: x.distance)
            num_good_matches = int(len(matches) * good_match_percent)
            matches = matches[:num_good_matches]
            if len(matches) < 4:
                print("Tidak cukup kecocokan untuk menghitung homografi.")
                return target_roi

            points_ref = np.zeros((len(matches), 2), dtype=np.float32)
            points_target = np.zeros((len(matches), 2), dtype=np.float32)

            for i, match in enumerate(matches):
                points_ref[i, :] = keypoints_ref[match.queryIdx].pt
                points_target[i, :] = keypoints_target[match.trainIdx].pt

            h, mask = cv2.findHomography(points_target, points_ref, cv2.RANSAC)

            if h is None:
                print("Homografi tidak dapat dihitung.")
                return target_roi

            height, width, channels = reference_roi.shape
            aligned_target = cv2.warpPerspective(target_roi, h, (width, height))

            return aligned_target
        except Exception as e:
            print(f"Error dalam align_images: {e}")
            return target_roi

    def run(self):
        try:
            # Mulai thread capture_frames
            self.capture_thread = threading.Thread(target=self.capture_frames)
            self.capture_thread.start()
            self.process_frames()
            self.capture_thread.join()
        except Exception as e:
            print(f"Error dalam run: {e}")

    def capture_frames(self):
        try:
            while not self.stop_event.is_set():
                ret, frame = self.cap.read()
                if not ret:
                    print("Tidak dapat membaca frame dari video.")
                    self.stop_event.set()
                    break

                self.frame_counter += 1
                if self.frame_counter % 2 != 0:
                    continue

                try:
                    self.capture_queue.put(frame, timeout=1)
                except queue.Full:
                    print("Frame queue penuh. Melewati frame.")
                    continue
        except Exception as e:
            print(f"Error dalam capture_frames: {e}")

    def process_frames(self):
        try:
            while not self.stop_event.is_set():
                try:
                    frame = self.capture_queue.get(timeout=1)
                except queue.Empty:
                    continue

                output = frame.copy()

                with self.lock:
                    for idx, roi in enumerate(self.rois):
                        x, y, w, h = self.bounding_boxes[idx]
                        cropped_polygon = self.cropped_polygons[idx]
                        mask = self.precomputed_masks[idx]
                        reference_cropped = cv2.bitwise_and(self.reference_img[y : y + h, x : x + w], self.reference_img[y : y + h, x : x + w], mask=mask)
                        target_cropped = cv2.bitwise_and(frame[y : y + h, x : x + w], frame[y : y + h, x : x + w], mask=mask)
                        if reference_cropped.size == 0 or target_cropped.size == 0:
                            continue

                        aligned_target_cropped = self.align_images(reference_cropped, target_cropped)
                        if aligned_target_cropped is None:
                            continue

                        try:
                            gray_ref_roi = cv2.cvtColor(reference_cropped, cv2.COLOR_BGR2GRAY)
                            gray_aligned_roi = cv2.cvtColor(aligned_target_cropped, cv2.COLOR_BGR2GRAY)
                            (score, diff) = ssim(gray_ref_roi, gray_aligned_roi, full=True)
                            diff = (diff * 255).astype("uint8")
                            thresh = cv2.threshold(diff, 50, 255, cv2.THRESH_BINARY_INV)[1]
                            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
                            thresh = cv2.dilate(thresh, kernel, iterations=2)
                            thresh = cv2.erode(thresh, kernel, iterations=1)
                            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                            for contour in contours:
                                if cv2.contourArea(contour) > 100:
                                    (cx, cy, cw, ch) = cv2.boundingRect(contour)
                                    cv2.rectangle(output, (x + cx, y + cy), (x + cx + cw, y + cy + ch), (0, 0, 255), 2)
                        except Exception as e:
                            print(f"Error dalam proses ROI {idx}: {e}")

                # Update FPS
                self.frame_count += 1
                if self.frame_count % 30 == 0:
                    elapsed_time = time.time() - self.start_time
                    fps = self.frame_count / elapsed_time
                    print(f"FPS: {fps:.2f}")

                # Menyimpan frame terbaru untuk tampilan dan kalibrasi
                with self.lock:
                    self.latest_frame = output.copy()
                    self.latest_original_frame = frame.copy()  # Menyimpan frame asli

                # Emit frame yang telah diproses ke GUI
                self.frame_processed.emit(output)
        except Exception as e:
            print(f"Error dalam process_frames: {e}")

    @pyqtSlot(np.ndarray)
    def update_reference_image(self, new_reference_frame):
        try:
            with self.lock:
                # Resize frame ke ukuran referensi
                resized_reference = cv2.resize(new_reference_frame, (self.frame_width, self.frame_height))
                self.reference_img = resized_reference
                self.reference_display = self.reference_img.copy()
                for roi in self.rois:
                    cv2.polylines(self.reference_display, [np.array(roi, dtype=np.int32)], isClosed=True, color=(0, 255, 0), thickness=2)

                # Recompute masks, bounding boxes, dan cropped polygons
                self.precomputed_masks = []
                self.bounding_boxes = []
                self.cropped_polygons = []
                for roi in self.rois:
                    mask = self.create_polygon_mask(self.reference_img.shape[:2], roi)
                    x, y, w, h = cv2.boundingRect(np.array(roi, dtype=np.int32))
                    self.bounding_boxes.append((x, y, w, h))
                    cropped_polygon = [(pt[0] - x, pt[1] - y) for pt in roi]
                    self.cropped_polygons.append(cropped_polygon)
                    cropped_mask = self.create_polygon_mask((h, w), cropped_polygon)
                    self.precomputed_masks.append(cropped_mask)

            print("Gambar referensi berhasil diperbarui.")
        except Exception as e:
            print(f"Error dalam update_reference_image: {e}")


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Anomaly Detection")
        self.setFixedSize(1280, 720)  # Tetapkan ukuran tetap 1280x720

        # Inisialisasi AnomalyDetection sebagai QThread
        self.ad = AnomalyDetection()

        # Koneksi sinyal frame_processed dari AnomalyDetection ke slot update_video
        self.ad.frame_processed.connect(self.update_video)

        # Mulai thread AnomalyDetection
        self.ad.start()

        # QLabel untuk menampilkan video
        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setStyleSheet("background-color: black;")

        # QLabel untuk menampilkan gambar referensi
        self.reference_label = QLabel()
        self.reference_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.reference_label.setStyleSheet("background-color: gray;")
        self.reference_label.setVisible(False)  # Sembunyikan secara default

        # Tombol untuk menampilkan/menyembunyikan referensi
        self.show_reference_button = QPushButton("Show Reference")
        self.show_reference_button.clicked.connect(self.toggle_reference)

        # Tombol untuk menjalankan/menghentikan deteksi
        self.run_stop_button = QPushButton("Stop")
        self.run_stop_button.clicked.connect(self.toggle_running)

        # Tombol untuk kalibrasi referensi
        self.calibrate_button = QPushButton("Calibrate")
        self.calibrate_button.clicked.connect(self.calibrate_reference)

        # Layout grid
        self.grid_layout = QGridLayout()
        self.grid_layout.setContentsMargins(5, 5, 5, 5)
        self.grid_layout.setSpacing(5)

        # Tambahkan video_label ke grid, spanning 2 kolom awalnya
        self.grid_layout.addWidget(self.video_label, 0, 0, 1, 2)
        # Tambahkan reference_label ke grid, tetapi sembunyikan
        self.grid_layout.addWidget(self.reference_label, 1, 0, 1, 2)

        # Layout untuk tombol
        button_layout = QHBoxLayout()
        button_layout.addWidget(self.show_reference_button)
        button_layout.addWidget(self.run_stop_button)
        button_layout.addWidget(self.calibrate_button)  # Tambahkan tombol calibrate

        # Layout utama
        main_layout = QVBoxLayout()
        main_layout.addLayout(self.grid_layout)
        main_layout.addLayout(button_layout)
        self.setLayout(main_layout)

        # Variabel untuk menyimpan frame terbaru
        self.latest_frame = None

        # Inisialisasi show_reference
        self.show_reference = False  # Tambahkan ini

    def toggle_reference(self):
        self.show_reference = not self.show_reference
        if self.show_reference:
            # Ubah layout: video_label dan reference_label dalam grid 2x2
            self.grid_layout.removeWidget(self.video_label)
            self.video_label.setParent(None)
            self.grid_layout.removeWidget(self.reference_label)
            self.reference_label.setParent(None)

            # Tambahkan video_label dan reference_label ke grid 2x2
            self.grid_layout.addWidget(self.video_label, 0, 0, 1, 1)
            self.grid_layout.addWidget(self.reference_label, 0, 1, 1, 1)

            # Atur ulang reference_label dengan gambar referensi
            with self.ad.lock:
                ref_image = self.ad.reference_display.copy()
            height, width, channel = ref_image.shape
            bytes_per_line = 3 * width
            q_img = QImage(ref_image.data, width, height, bytes_per_line, QImage.Format.Format_RGB888).rgbSwapped()
            pixmap = QPixmap.fromImage(q_img)
            self.reference_label.setPixmap(pixmap.scaled(self.reference_label.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
            self.reference_label.setVisible(True)

            self.show_reference_button.setText("Hide Reference")
        else:
            # Ubah layout kembali ke single view
            self.grid_layout.removeWidget(self.video_label)
            self.video_label.setParent(None)
            self.grid_layout.removeWidget(self.reference_label)
            self.reference_label.setParent(None)

            # Tambahkan video_label spanning 2 kolom
            self.grid_layout.addWidget(self.video_label, 0, 0, 1, 2)
            self.grid_layout.addWidget(self.reference_label, 1, 0, 1, 2)
            self.reference_label.setVisible(False)

            self.show_reference_button.setText("Show Reference")

    def toggle_running(self):
        if self.ad.isRunning():
            # Hentikan thread
            self.ad.stop_event.set()
            self.run_stop_button.setText("Run")
            self.ad.wait()
            print("Deteksi anomali dihentikan.")
        else:
            # Mulai thread baru jika sudah berhenti
            if not self.ad.isRunning():
                self.ad = AnomalyDetection()
                self.ad.frame_processed.connect(self.update_video)
                self.ad.start()
                print("Deteksi anomali dimulai.")
            self.run_stop_button.setText("Stop")
        # Tidak perlu menyimpan status running secara manual karena menggunakan isRunning()

    def calibrate_reference(self):
        # Ambil frame asli terbaru dari AnomalyDetection
        with self.ad.lock:
            if self.ad.latest_original_frame is not None:
                frame = self.ad.latest_original_frame.copy()
                print("Frame tersedia untuk kalibrasi.")
            else:
                frame = None
                print("Frame tidak tersedia untuk kalibrasi.")

        if frame is not None:
            try:
                # Simpan frame sebagai gambar referensi baru
                timestamp = time.strftime("%Y%m%d-%H%M%S")
                new_reference_filename = f"room_{timestamp}.jpg"
                new_reference_path = os.path.join(self.ad.reference_folder, new_reference_filename)

                # Menyimpan gambar referensi baru
                cv2.imwrite(new_reference_path, frame)
                print(f"Gambar referensi baru disimpan sebagai {new_reference_path}")

                # Update reference image di AnomalyDetection
                self.ad.update_reference_image(frame)

                # Jika reference sedang ditampilkan, perbarui tampilannya
                if self.show_reference:
                    with self.ad.lock:
                        ref_image = self.ad.reference_display.copy()
                    height, width, channel = ref_image.shape
                    bytes_per_line = 3 * width
                    q_img = QImage(ref_image.data, width, height, bytes_per_line, QImage.Format.Format_RGB888).rgbSwapped()
                    pixmap = QPixmap.fromImage(q_img)
                    self.reference_label.setPixmap(pixmap.scaled(self.reference_label.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
                    print("Reference_label diperbarui dengan gambar baru.")

                QMessageBox.information(self, "Calibrate", "Gambar referensi telah berhasil diperbarui.")
            except Exception as e:
                print(f"Error saat kalibrasi: {e}")
                QMessageBox.warning(self, "Calibrate", f"Terjadi kesalahan saat kalibrasi: {e}")
        else:
            QMessageBox.warning(self, "Calibrate", "Tidak ada frame yang tersedia untuk kalibrasi.")

    @pyqtSlot(np.ndarray)
    def update_video(self, frame):
        try:
            # Simpan frame terbaru untuk kalibrasi
            with self.ad.lock:
                self.ad.latest_frame = frame.copy()
                print("latest_frame diperbarui.")

            # Tampilkan frame di video_label
            height, width, channel = frame.shape
            bytes_per_line = 3 * width
            q_img = QImage(frame.data, width, height, bytes_per_line, QImage.Format.Format_RGB888).rgbSwapped()
            pixmap = QPixmap.fromImage(q_img)
            scaled_pixmap = pixmap.scaled(self.video_label.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
            self.video_label.setPixmap(scaled_pixmap)
        except Exception as e:
            print(f"Error dalam update_video: {e}")

    def closeEvent(self, event):
        # Pastikan thread berhenti saat window ditutup
        self.ad.stop_event.set()
        self.ad.wait()
        event.accept()


if __name__ == "__main__":
    app = QApplication([])
    window = MainWindow()
    window.show()
    app.exec()
