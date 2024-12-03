import sys
import os
import cv2
import shutil
import numpy as np
from PyQt6.QtWidgets import (
    QApplication,
    QMainWindow,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QHBoxLayout,
    QWidget,
)
from PyQt6.QtGui import QPixmap, QImage
from PyQt6.QtCore import Qt


class ImageViewer(QMainWindow):
    def __init__(self, folder_path):
        super().__init__()
        self.setWindowTitle("Image Viewer")

        self.folder_path = folder_path
        self.hapus_folder = os.path.join(folder_path, "hapus")
        if not os.path.exists(self.hapus_folder):
            os.makedirs(self.hapus_folder)

        # Dapatkan daftar file gambar
        self.image_files = [f for f in os.listdir(folder_path) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
        self.image_files.sort()
        self.total_images = len(self.image_files)
        self.current_index = 0

        self.status_dict = {}  # Melacak status gambar ('Keep' atau 'Delete')

        # Koordinat border yang telah ditentukan
        self.border = [[(595, 651), (599, 413), (651, 173), (644, 362), (595, 651)], [(877, 369), (892, 191), (1010, 445), (965, 611), (933, 610), (877, 369)], [(816, 154), (844, 248), (853, 118), (817, 58), (816, 154)], [(681, 57), (673, 98), (665, 214), (678, 140), (681, 57)]]

        # Inisialisasi antarmuka
        self.init_ui()

        if self.total_images == 0:
            self.show_message(f"Tidak ada gambar yang valid di folder: {folder_path}")
        else:
            self.display_image()

    def init_ui(self):
        # Widget utama
        widget = QWidget()
        self.setCentralWidget(widget)

        # Layout
        main_layout = QVBoxLayout()
        image_layout = QHBoxLayout()
        control_layout = QHBoxLayout()

        # Label untuk menampilkan gambar
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setMinimumSize(640, 480)
        image_layout.addWidget(self.image_label)

        # Tombol kontrol
        self.keep_button = QPushButton("Keep (K)")
        self.delete_button = QPushButton("Delete (D)")
        self.previous_button = QPushButton("Previous (←)")
        self.next_button = QPushButton("Next (→)")
        self.quit_button = QPushButton("Quit (Q)")

        self.keep_button.clicked.connect(self.mark_keep)
        self.delete_button.clicked.connect(self.mark_delete)
        self.previous_button.clicked.connect(self.previous_image)
        self.next_button.clicked.connect(self.next_image)
        self.quit_button.clicked.connect(self.quit_app)

        control_layout.addWidget(self.keep_button)
        control_layout.addWidget(self.delete_button)
        control_layout.addWidget(self.previous_button)
        control_layout.addWidget(self.next_button)
        control_layout.addWidget(self.quit_button)

        main_layout.addLayout(image_layout)
        main_layout.addLayout(control_layout)

        widget.setLayout(main_layout)

    def display_image(self):
        image_name = self.image_files[self.current_index]
        image_path = os.path.join(self.folder_path, image_name)
        status = self.status_dict.get(image_name, "")

        # Membaca gambar menggunakan OpenCV
        image = cv2.imread(image_path)
        if image is None:
            self.show_message(f"Error: Tidak dapat membuka gambar {image_path}")
            return

        # Mengubah ukuran gambar jika perlu
        image = self.resize_image(image, max_size=(1280, 720))

        # Menggambar poligon pada gambar
        image = self.draw_polygon(image, self.border)

        # Konversi gambar OpenCV (BGR) ke QImage (RGB)
        height, width, channel = image.shape
        bytes_per_line = 3 * width
        q_image = QImage(image.data.tobytes(), width, height, bytes_per_line, QImage.Format.Format_BGR888)

        # Konversi QImage ke QPixmap
        pixmap = QPixmap.fromImage(q_image)

        # Menampilkan pixmap di label
        self.image_label.setPixmap(pixmap)

        # Memperbarui judul jendela dengan status
        self.setWindowTitle(f"Image Viewer - {self.current_index + 1}/{self.total_images} {status}")

    def resize_image(self, image, max_size=(1280, 720)):
        height, width = image.shape[:2]
        max_width, max_height = max_size
        if width > max_width or height > max_height:
            scaling_factor = min(max_width / width, max_height / height)
            new_size = (int(width * scaling_factor), int(height * scaling_factor))
            image = cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)
        return image

    def draw_polygon(self, image, borders):
        # Menskalakan titik jika gambar diubah ukurannya
        height, width = image.shape[:2]
        scale_x = width / 1280  # Lebar asli
        scale_y = height / 720  # Tinggi asli

        for points in borders:
            scaled_points = [(int(x * scale_x), int(y * scale_y)) for x, y in points]
            # Menggambar poligon pada gambar
            pts = np.array(scaled_points, np.int32)
            pts = pts.reshape((-1, 1, 2))
            cv2.polylines(image, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
        return image

    def mark_keep(self):
        image_name = self.image_files[self.current_index]
        self.status_dict[image_name] = "Keep"
        self.next_image()

    def mark_delete(self):
        image_name = self.image_files[self.current_index]
        self.status_dict[image_name] = "Delete"
        self.next_image()

    def previous_image(self):
        if self.current_index > 0:
            self.current_index -= 1
            self.display_image()

    def next_image(self):
        if self.current_index < self.total_images - 1:
            self.current_index += 1
            self.display_image()
        else:
            self.show_message("Akhir dari gambar.")

    def quit_app(self):
        # Memindahkan semua gambar yang ditandai untuk dihapus
        self.move_marked_images()
        QApplication.quit()

    def move_marked_images(self):
        for image_name, status in self.status_dict.items():
            if status == "Delete":
                self.move_to_hapus(image_name)

    def move_to_hapus(self, image_name):
        src_path = os.path.join(self.folder_path, image_name)
        dest_path = os.path.join(self.hapus_folder, image_name)
        try:
            shutil.move(src_path, dest_path)
            print(f"Memindahkan {image_name} ke folder 'hapus'.")
        except Exception as e:
            print(f"Error memindahkan {image_name}: {e}")

    def show_message(self, message):
        self.statusBar().showMessage(message, 5000)  # Menampilkan pesan selama 5 detik
        print(message)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_K:
            self.mark_keep()
        elif event.key() == Qt.Key.Key_D:
            self.mark_delete()
        elif event.key() == Qt.Key.Key_Period:
            self.previous_image()
        elif event.key() == Qt.Key.Key_Comma:
            self.next_image()
        elif event.key() == Qt.Key.Key_Q:
            self.quit_app()


def main(path=None):
    import argparse

    parser = argparse.ArgumentParser(description="Image Viewer Application")
    parser.add_argument(
        "folder",
        nargs="?",
        default=f"//10.5.0.2/server_15_juta/NWR'S ROOM/WALLPAPER/{path}",
        help="Path ke folder yang berisi gambar",
    )
    args = parser.parse_args()

    folder_path = args.folder
    if not os.path.exists(folder_path):
        print(f"Folder {folder_path} tidak ditemukan.")
        sys.exit(1)

    app = QApplication(sys.argv)
    viewer = ImageViewer(folder_path)
    viewer.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
