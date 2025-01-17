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

    def apply_overlay_perspective(self, frame, marker_corners):
        h, w = self.overlay_image.shape[:2]
        src_pts = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32)
        dst_pts = np.array(marker_corners, dtype=np.float32)
        homography, status = cv2.findHomography(src_pts, dst_pts)

        if homography is not None:
            warped_overlay = cv2.warpPerspective(self.overlay_image, homography, (frame.shape[1], frame.shape[0]))
            warped_alpha = cv2.warpPerspective(self.overlay_alpha, homography, (frame.shape[1], frame.shape[0]))
            warped_alpha = cv2.merge([warped_alpha, warped_alpha, warped_alpha])
            alpha_mask = warped_alpha.astype(float) / 255.0
            frame = frame * (1 - alpha_mask) + warped_overlay * alpha_mask
            frame = frame.astype(np.uint8)

        return frame


class Camera:
    def __init__(self, source=0):
        self.cap = cv2.VideoCapture(source)
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
                ids = ids.flatten().astype(int)

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

                    if ids[i] == self.marker_id_to_overlay:
                        frame = self.overlay_manager.apply_overlay_perspective(frame, corners[i][0])

                print(f"Detected IDs: {ids}")
            else:
                print("No markers detected.")

            cv2.imshow("Aruco Markers", frame)

            if cv2.waitKey(1) & 0xFF == ord("n"):
                break

        self.camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    overlay_image_path = r"D:\NWR\main\Al-Wajid\starbucks.png"
    app = ArucoApp(overlay_path=overlay_image_path, marker_id_to_overlay=19, marker_length=0.05)
    app.run()
