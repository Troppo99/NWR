import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
import threading
import queue
import time
from PyQt6.QtWidgets import QApplication, QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout, QGridLayout
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtCore import Qt, QTimer


class AnomalyDetection(threading.Thread):
    def __init__(self):
        super().__init__()
        self.rois = [[(57, 465), (225, 430), (236, 514), (220, 557), (78, 594)], [(387, 758), (472, 734), (480, 820), (393, 850)]]
        # self.rois = [[(1252, 687), (2007, 795), (3157, 995), (3155, 1710), (2632, 1695), (2622, 1575), (2245, 1572), (812, 1022), (1242, 910)]]
        self.reference_img = cv2.imread("D:/NWR/sources/AlFaruq/media/room0.jpg")
        self.cap = cv2.VideoCapture("rtsp://admin:oracle2015@192.168.100.65:554/Streaming/Channels/1")
        # self.cap = cv2.VideoCapture("C:/Users/Public/iVMS-4200 Site/UserData/Video/Camera_01_10.5.0.111_10.5.0.111_20241203141002/Camera_01_10.5.0.111_10.5.0.111_20241203075308_20241203075410_21256616.mp4")
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
        self.running = False

    def create_polygon_mask(self, image_shape, polygon):
        mask = np.zeros(image_shape, dtype=np.uint8)
        pts = np.array([polygon], dtype=np.int32)
        cv2.fillPoly(mask, pts, 255)
        return mask

    def align_images(self, reference_roi, target_roi, max_features=500, good_match_percent=0.15):
        gray_ref = cv2.cvtColor(reference_roi, cv2.COLOR_BGR2GRAY)
        gray_target = cv2.cvtColor(target_roi, cv2.COLOR_BGR2GRAY)
        orb = cv2.ORB_create(max_features)
        keypoints_ref, descriptors_ref = orb.detectAndCompute(gray_ref, None)
        keypoints_target, descriptors_target = orb.detectAndCompute(gray_target, None)
        if descriptors_ref is None or descriptors_target is None:
            # print("Tidak ditemukan deskriptor dalam salah satu ROI.")
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
            # print("Tidak cukup kecocokan untuk menghitung homografi.")
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

    def capture_frames(self):
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

    def process_frames(self):
        while not self.stop_event.is_set():
            try:
                frame = self.capture_queue.get(timeout=1)
            except queue.Empty:
                continue

            output = frame.copy()

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

            # Update FPS
            self.frame_count += 1
            if self.frame_count % 30 == 0:
                elapsed_time = time.time() - self.start_time
                fps = self.frame_count / elapsed_time
                print(f"FPS: {fps:.2f}")

            if cv2.waitKey(1) & 0xFF == ord("n"):
                self.stop_event.set()
                break

            # Masukkan frame yang telah diproses ke antrean proses
            try:
                self.process_queue.put((frame, output), timeout=1)
            except queue.Full:
                print("Process queue penuh. Melewati frame.")

    def get_frame(self):
        try:
            return self.process_queue.get_nowait()
        except queue.Empty:
            return None, None

    def run(self):
        self.stop_event.clear()
        capture_thread = threading.Thread(target=self.capture_frames)
        capture_thread.start()
        self.process_frames()
        capture_thread.join()


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Anomaly Detection")
        self.setFixedSize(1280, 720)  # Tetapkan ukuran tetap 1280x720

        self.ad = AnomalyDetection()
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

        # Layout utama
        main_layout = QVBoxLayout()
        main_layout.addLayout(self.grid_layout)
        main_layout.addLayout(button_layout)
        self.setLayout(main_layout)

        # Timer untuk memperbarui video
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_video)
        self.timer.start(30)  # Update setiap 30 ms (~33 FPS)

        self.show_reference = False

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

            # Atur ulang reference_label
            ref_image = self.ad.reference_display
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
        if self.ad.running:
            self.ad.stop_event.set()
            self.run_stop_button.setText("Run")
        else:
            if not self.ad.is_alive():
                self.ad = AnomalyDetection()
                self.ad.start()
            else:
                self.ad.stop_event.clear()
            self.run_stop_button.setText("Stop")
        self.ad.running = not self.ad.running

    def update_video(self):
        frame, output = self.ad.get_frame()
        if frame is not None and output is not None:
            # Tampilkan output frame
            display_frame = output

            # Resize frame untuk menyesuaikan QLabel tanpa mengubah resolusi internal
            height, width, channel = display_frame.shape
            bytes_per_line = 3 * width
            q_img = QImage(display_frame.data, width, height, bytes_per_line, QImage.Format.Format_RGB888).rgbSwapped()
            pixmap = QPixmap.fromImage(q_img)
            scaled_pixmap = pixmap.scaled(self.video_label.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
            self.video_label.setPixmap(scaled_pixmap)

            if self.show_reference:
                # Tampilkan gambar referensi di reference_label (sudah diatur saat toggle_reference)
                pass

    def closeEvent(self, event):
        # Pastikan thread berhenti saat window ditutup
        self.ad.stop_event.set()
        self.ad.join()
        event.accept()


if __name__ == "__main__":
    app = QApplication([])
    window = MainWindow()
    window.show()
    app.exec()
