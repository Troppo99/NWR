import sys
import cv2
import cvzone
from ultralytics import YOLO
import time

# Konstanta
PROCESS_WIDTH = 960
PROCESS_HEIGHT = 540
DISPLAY_WIDTH = 1280
DISPLAY_HEIGHT = 720
CONFIDENCE_THRESHOLD = 0.5


def main():
    # Path ke video lokal dan model YOLO
    video_path = "videos/test/cutgine.mp4"
    model_path = "run/cutting_engine/version1/weights/best.pt"

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Error: Tidak dapat membuka video {video_path}")

    model = YOLO(model_path)
    model.overrides["verbose"] = False

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30  # fallback jika fps tidak terdeteksi

    cv2.namedWindow("Inference", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Inference", DISPLAY_WIDTH, DISPLAY_HEIGHT)

    while True:
        ret, frame = cap.read()
        if not ret:
            # Jika video habis atau gagal baca frame, restart ke awal
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        # Resize frame untuk pemrosesan
        frame_resized = cv2.resize(frame, (PROCESS_WIDTH, PROCESS_HEIGHT))

        # Deteksi objek dengan YOLO
        results = model(frame_resized)
        boxes_info = []

        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = box.conf[0]
                if conf > CONFIDENCE_THRESHOLD:
                    boxes_info.append((x1, y1, x2, y2, conf))

        # Gambar bounding boxes dan label "Warning!" jika ada deteksi
        if boxes_info:
            for x1, y1, x2, y2, conf in boxes_info:
                cvzone.cornerRect(frame_resized, (x1, y1, x2 - x1, y2 - y1), rt=0, colorC=(0, 0, 255))  # Merah
                cvzone.putTextRect(frame_resized, "Warning!", (x1, y1 - 10), scale=1, thickness=2, offset=5, colorB=(0, 0, 255))
            cvzone.putTextRect(frame_resized, "Pelanggaran menggunakan mesin potong manual", (10, 30), scale=1, thickness=2, offset=5, colorB=(255, 255, 255), colorT=(0, 0, 0))

        # Resize frame untuk tampilan
        frame_display = cv2.resize(frame_resized, (DISPLAY_WIDTH, DISPLAY_HEIGHT))

        cv2.imshow("Inference", frame_display)

        # Delay sesuai FPS video
        delay = int(1000 / fps)
        key = cv2.waitKey(delay) & 0xFF
        if key == ord("n"):  # Tekan ESC untuk keluar
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
