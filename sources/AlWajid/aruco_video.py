import cv2
import cv2.aruco as aruco
import numpy as np

print(f"OpenCV Version: {cv2.__version__}")

try:
    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_100)  # Menggunakan dictionary 4x4_100, artinya 4x4 dengan 100 marker (0-99)
except AttributeError:
    try:
        aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_100)
    except AttributeError:
        print("Fungsi untuk mendapatkan dictionary Aruco tidak ditemukan.")
        print("Pastikan Anda menginstal opencv-contrib-python.")
        exit()

try:
    parameters = aruco.DetectorParameters()
except AttributeError:
    try:
        parameters = aruco.DetectorParameters_create()
    except AttributeError:
        print("Fungsi untuk membuat DetectorParameters tidak ditemukan.")
        print("Pastikan Anda menginstal opencv-contrib-python.")
        exit()

camera_matrix = np.array([[800, 0, 320], [0, 800, 240], [0, 0, 1]], dtype=float)
dist_coeffs = np.zeros((5, 1))  # Asumsi tanpa distorsi

cap = cv2.VideoCapture(0)
# cap = cv2.VideoCapture("rtsp://admin:oracle2015@192.168.100.65:554/Streaming/Channels/1")

if not cap.isOpened():
    print("Tidak dapat membuka webcam. Periksa koneksi atau driver webcam Anda.")
    exit()

print("Mulai streaming video dari webcam. Tekan 'q' untuk keluar.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Gagal membaca frame dari webcam.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    try:
        detector = aruco.ArucoDetector(aruco_dict, parameters)
        corners, ids, rejected = detector.detectMarkers(gray)
    except AttributeError:
        corners, ids, rejected = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

    if ids is not None and len(ids) > 0:
        try:
            rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(corners, 0.05, camera_matrix, dist_coeffs)
        except AttributeError:
            print("Fungsi estimatePoseSingleMarkers tidak ditemukan.")
            rvecs, tvecs = [], []

        for rvec, tvec in zip(rvecs, tvecs):
            try:
                if hasattr(aruco, "drawAxis"):
                    aruco.drawAxis(frame, camera_matrix, dist_coeffs, rvec, tvec, 0.05)
                elif hasattr(aruco, "drawFrameAxes"):
                    aruco.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvec, tvec, 0.05)
            except AttributeError:
                print("Fungsi drawAxis atau drawFrameAxes tidak ditemukan.")

        aruco.drawDetectedMarkers(frame, corners, ids)
        print(f"Detected IDs: {ids.flatten()}")

    else:
        print("No markers detected.")

    cv2.imshow("Aruco Markers", frame)

    if cv2.waitKey(1) & 0xFF == ord("n"):
        break

cap.release()
cv2.destroyAllWindows()
