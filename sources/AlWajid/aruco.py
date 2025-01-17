import cv2
import cv2.aruco as aruco

print(f"OpenCV Version: {cv2.__version__}")

try:
    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
except AttributeError:
    try:
        aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_50)
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

image = cv2.imread("main/Al-Wajid/charuco_board.png")

if image is None:
    print("Image not found. Check the path.")
    exit()

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

try:
    detector = aruco.ArucoDetector(aruco_dict, parameters)
except AttributeError:
    print("ArucoDetector tidak ditemukan di modul cv2.aruco.")
    print("Pastikan Anda menggunakan versi OpenCV yang mendukung ArucoDetector.")
    exit()

corners, ids, rejected = detector.detectMarkers(gray)

if ids is not None and len(ids) > 0:
    aruco.drawDetectedMarkers(image, corners, ids)
    print(f"Detected IDs: {ids.flatten()}")
else:
    print("No markers detected.")

cv2.imshow("Aruco Markers", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
