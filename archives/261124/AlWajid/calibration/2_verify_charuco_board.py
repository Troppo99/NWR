import cv2
import cv2.aruco as aruco


def count_markers(image_path, dictionary_type=aruco.DICT_4X4_100):
    img = cv2.imread(image_path)
    if img is None:
        print("Gagal membaca gambar. Periksa path yang diberikan.")
        return

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    aruco_dict = aruco.getPredefinedDictionary(dictionary_type)
    corners, ids, rejected = aruco.detectMarkers(gray, aruco_dict)

    if ids is not None:
        print(f"Jumlah Marker yang Terdeteksi: {len(ids)}")
    else:
        print("Tidak ada marker yang terdeteksi.")

    img_markers = aruco.drawDetectedMarkers(img.copy(), corners, ids)
    cv2.imshow("Detected Markers", img_markers)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    image_path = "main/Al-Wajid/calibrataion/charuco_board.png"
    count_markers(image_path)
