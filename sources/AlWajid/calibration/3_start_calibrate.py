import cv2
import cv2.aruco as aruco
import numpy as np


def calibrate_camera_charuco(charuco_board, aruco_dict, num_frames=20):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise IOError("Tidak dapat membuka webcam. Periksa koneksi atau driver webcam Anda.")

    all_corners = []
    all_ids = []
    image_size = None

    print("Mulai kalibrasi kamera.")
    print(f"Silakan atur dan tembak Charuco Board. Anda perlu menangkap sekitar {num_frames} gambar.")
    print("Tekan 'c' untuk menangkap gambar, 'q' untuk keluar.")

    while len(all_corners) < num_frames:
        ret, frame = cap.read()
        if not ret:
            print("Gagal membaca frame dari webcam.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, rejected = aruco.detectMarkers(gray, aruco_dict)

        if ids is not None:
            ret_charuco, charuco_corners, charuco_ids = aruco.interpolateCornersCharuco(markerCorners=corners, markerIds=ids, image=gray, board=charuco_board)
            if ret_charuco > 0:
                aruco.drawDetectedCornersCharuco(frame, charuco_corners, charuco_ids)

        cv2.imshow("Kalibrasi Kamera", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("c"):
            if ids is not None:
                ret_charuco, charuco_corners, charuco_ids = aruco.interpolateCornersCharuco(markerCorners=corners, markerIds=ids, image=gray, board=charuco_board)
                if ret_charuco > 0:
                    all_corners.append(charuco_corners)
                    all_ids.append(charuco_ids)
                    print(f"Gambar {len(all_corners)}/{num_frames} berhasil ditangkap.")
                else:
                    print("Charuco corners tidak terdeteksi. Coba lagi.")
            else:
                print("Marker tidak terdeteksi. Coba lagi.")
        elif key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

    if len(all_corners) < num_frames:
        print("Kalibrasi gagal. Tidak cukup gambar untuk kalibrasi.")
        return None, None

    ret, camera_matrix, dist_coeffs, rvecs, tvecs = aruco.calibrateCameraCharuco(charucoCorners=all_corners, charucoIds=all_ids, board=charuco_board, imageSize=gray.shape[::-1], cameraMatrix=None, distCoeffs=None)

    if ret:
        print("Kalibrasi Kamera Selesai.")
        print("Camera Matrix:")
        print(camera_matrix)
        print("Distortion Coefficients:")
        print(dist_coeffs)
        return camera_matrix, dist_coeffs
    else:
        print("Kalibrasi gagal.")
        return None, None


if __name__ == "__main__":
    squaresX = 5
    squaresY = 7
    squareLength = 0.04
    markerLength = 0.02
    dictionary_type = aruco.DICT_4X4_100

    aruco_dict = aruco.getPredefinedDictionary(dictionary_type)
    charuco_board = aruco.CharucoBoard(size=(squaresX, squaresY), squareLength=squareLength, markerLength=markerLength, dictionary=aruco_dict)

    camera_matrix, dist_coeffs = calibrate_camera_charuco(charuco_board, aruco_dict, num_frames=20)

    if camera_matrix is not None and dist_coeffs is not None:
        np.savez("main/Al-Wajid/calibration/camera_calibration.npz", camera_matrix=camera_matrix, dist_coeffs=dist_coeffs)
        print("Hasil kalibrasi disimpan di 'main/Al-Wajid/calibration/camera_calibration.npz'")
