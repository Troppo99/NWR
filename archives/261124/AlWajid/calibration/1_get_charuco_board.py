import cv2
import cv2.aruco as aruco


def generate_charuco_board(squaresX=5, squaresY=7, squareLength=0.04, markerLength=0.02, dictionary_type=aruco.DICT_4X4_100, save_path="main/Al-Wajid/calibrataion/charuco_board.png"):
    aruco_dict = aruco.getPredefinedDictionary(dictionary_type)
    charuco_board = aruco.CharucoBoard(size=(squaresX, squaresY), squareLength=squareLength, markerLength=markerLength, dictionary=aruco_dict)
    img = aruco.drawPlanarBoard(board=charuco_board, outSize=(2000, 2000), img=None, marginSize=0, borderBits=1)
    cv2.imwrite(save_path, img)
    print(f"Charuco Board telah disimpan di {save_path}")
    cv2.imshow(save_path, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    generate_charuco_board()
