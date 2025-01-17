import cv2
import cv2.aruco as aruco
import numpy as np
import os


class ArucoDetector:
    def __init__(self, dictionary_type=aruco.DICT_4X4_100):
        self.aruco_dict = self.get_predefined_dictionary(dictionary_type)
        self.parameters = self.get_detector_parameters()

    def get_predefined_dictionary(self, dictionary_type):
        try:
            return aruco.getPredefinedDictionary(dictionary_type)
        except AttributeError:
            try:
                return aruco.Dictionary_get(dictionary_type)
            except AttributeError:
                raise AttributeError("Fungsi untuk mendapatkan dictionary Aruco tidak ditemukan. " "Pastikan Anda menginstal opencv-contrib-python.")

    def get_detector_parameters(self):
        try:
            return aruco.DetectorParameters()
        except AttributeError:
            try:
                return aruco.DetectorParameters_create()
            except AttributeError:
                raise AttributeError("Fungsi untuk membuat DetectorParameters tidak ditemukan. " "Pastikan Anda menginstal opencv-contrib-python.")

    def detect_markers(self, gray_frame):
        try:
            detector = aruco.ArucoDetector(self.aruco_dict, self.parameters)
            corners, ids, rejected = detector.detectMarkers(gray_frame)
        except AttributeError:
            corners, ids, rejected = aruco.detectMarkers(gray_frame, self.aruco_dict, parameters=self.parameters)
        return corners, ids, rejected

    def estimate_pose(self, corners, ids, marker_length, camera_matrix, dist_coeffs):
        try:
            rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(corners, marker_length, camera_matrix, dist_coeffs)
        except AttributeError:
            print("Fungsi estimatePoseSingleMarkers tidak ditemukan.")
            rvecs, tvecs = [], []
        return rvecs, tvecs


class OverlayManager:
    def __init__(self, overlay_path, desired_size=(600, 608)):
        self.overlay_image, self.overlay_alpha = self.load_overlay(overlay_path, desired_size)

    def load_overlay(self, path, desired_size):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Gambar overlay tidak ditemukan: {path}")

        overlay = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if overlay is None:
            raise IOError("Gagal membaca gambar overlay. Periksa path dan format gambar.")

        if overlay.shape[2] == 4:
            overlay_rgb = overlay[:, :, :3]
            overlay_alpha = overlay[:, :, 3]
        else:
            overlay_rgb = overlay
            overlay_alpha = np.ones(overlay_rgb.shape[:2], dtype=np.uint8) * 255

        if (overlay_rgb.shape[1], overlay_rgb.shape[0]) != desired_size:
            overlay_rgb = cv2.resize(overlay_rgb, desired_size, interpolation=cv2.INTER_AREA)
            overlay_alpha = cv2.resize(overlay_alpha, desired_size, interpolation=cv2.INTER_AREA)

        return overlay_rgb, overlay_alpha

    def overlay_image_alpha_func(self, img, img_overlay, pos, alpha_mask):
        x, y = pos

        h, w = img_overlay.shape[:2]

        y1, y2 = max(0, y), min(img.shape[0], y + h)
        x1, x2 = max(0, x), min(img.shape[1], x + w)

        y1o, y2o = max(0, -y), min(h, img.shape[0] - y)
        x1o, x2o = max(0, -x), min(w, img.shape[1] - x)

        if y1 >= y2 or x1 >= x2 or y1o >= y2o or x1o >= x2o:
            return

        img_crop = img[y1:y2, x1:x2]
        img_overlay_crop = img_overlay[y1o:y2o, x1o:x2o]
        alpha = alpha_mask[y1o:y2o, x1o:x2o, None] / 255.0

        img[y1:y2, x1:x2] = (1.0 - alpha) * img_crop + alpha * img_overlay_crop

    def apply_overlay(self, frame, corners):
        top_left, top_right, bottom_right, bottom_left = corners

        width = int(np.linalg.norm(top_right - top_left))
        height = int(np.linalg.norm(top_left - bottom_left))

        scale = min(width / self.overlay_image.shape[1], height / self.overlay_image.shape[0])

        overlay_resized = cv2.resize(self.overlay_image, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        alpha_resized = cv2.resize(self.overlay_alpha, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

        center_x = int((top_left[0] + bottom_right[0]) / 2 - overlay_resized.shape[1] / 2)
        center_y = int((top_left[1] + bottom_right[1]) / 2 - overlay_resized.shape[0] / 2)

        self.overlay_image_alpha_func(frame, overlay_resized, (center_x, center_y), alpha_resized)


class Camera:
    def __init__(self, source=0):
        # self.cap = cv2.VideoCapture(source)
        self.cap = cv2.VideoCapture("rtsp://admin:oracle2015@192.168.100.65:554/Streaming/Channels/1")
        if not self.cap.isOpened():
            raise IOError("Tidak dapat membuka webcam. Periksa koneksi atau driver webcam Anda.")

    def get_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            raise IOError("Gagal membaca frame dari webcam.")
        return frame

    def release(self):
        self.cap.release()


class ArucoApp:
    def __init__(self, overlay_path, marker_id_to_overlay=19, marker_length=0.05):
        self.detector = ArucoDetector(aruco.DICT_4X4_100)
        self.overlay_manager = OverlayManager(overlay_path)
        self.camera = Camera(0)
        self.marker_id_to_overlay = marker_id_to_overlay
        self.marker_length = marker_length
        self.camera_matrix = np.array([[800, 0, 320], [0, 800, 240], [0, 0, 1]], dtype=float)
        self.dist_coeffs = np.zeros((5, 1))

    def run(self):
        print("Mulai streaming video dari webcam. Tekan 'n' untuk keluar.")
        while True:
            try:
                frame = self.camera.get_frame()
            except IOError as e:
                print(e)
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            corners, ids, rejected = self.detector.detect_markers(gray)

            if ids is not None and len(ids) > 0:
                rvecs, tvecs = self.detector.estimate_pose(corners, ids, self.marker_length, self.camera_matrix, self.dist_coeffs)
                aruco.drawDetectedMarkers(frame, corners, ids)

                for i, (rvec, tvec) in enumerate(zip(rvecs, tvecs)):
                    try:
                        if hasattr(aruco, "drawAxis"):
                            aruco.drawAxis(frame, self.camera_matrix, self.dist_coeffs, rvec, tvec, self.marker_length)
                        elif hasattr(aruco, "drawFrameAxes"):
                            aruco.drawFrameAxes(frame, self.camera_matrix, self.dist_coeffs, rvec, tvec, self.marker_length)
                    except AttributeError:
                        print("Fungsi drawAxis atau drawFrameAxes tidak ditemukan.")

                    if ids[i][0] == self.marker_id_to_overlay:
                        self.overlay_manager.apply_overlay(frame, corners[i][0])

                print(f"Detected IDs: {ids.flatten()}")
            else:
                print("No markers detected.")

            cv2.imshow("Aruco Markers", frame)

            if cv2.waitKey(1) & 0xFF == ord("n"):
                break

        self.camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    overlay_image_path = r"D:\NWR\main\Al-Wajid\images\starbucks.png"
    app = ArucoApp(overlay_path=overlay_image_path, marker_id_to_overlay=0, marker_length=0.05)
    app.run()
