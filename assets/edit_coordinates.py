import json
import cv2
import numpy as np

# Konstanta skala
ORIGINAL_WIDTH = 3200
ORIGINAL_HEIGHT = 1800
DISPLAY_WIDTH = 1280
DISPLAY_HEIGHT = 720
SCALE_X = DISPLAY_WIDTH / ORIGINAL_WIDTH
SCALE_Y = DISPLAY_HEIGHT / ORIGINAL_HEIGHT

# Global Variables untuk interaksi mouse
selected_office = None
selected_roi = None
selected_point = None
dragging = False


def load_json(file_path):
    with open(file_path, "r") as f:
        data = json.load(f)
    return data


def save_json(data, file_path):
    with open(file_path, "w") as f:
        json.dump(data, f, indent=4)


def scale_coords(coords, scale_x=SCALE_X, scale_y=SCALE_Y):
    return [[int(x * scale_x), int(y * scale_y)] for [x, y] in coords]


def descale_coords(coords, scale_x=SCALE_X, scale_y=SCALE_Y):
    return [[int(x / scale_x), int(y / scale_y)] for [x, y] in coords]


def draw_rois(frame, rois, color=(0, 255, 0), thickness=2, selected_roi=None, selected_point=None):
    for roi_idx, roi in enumerate(rois):
        pts = np.array(roi, np.int32).reshape((-1, 1, 2))
        if roi_idx == selected_roi:
            cv2.polylines(frame, [pts], isClosed=True, color=(0, 0, 255), thickness=thickness)
        else:
            cv2.polylines(frame, [pts], isClosed=True, color=color, thickness=thickness)
        # Gambar titik-titik
        for point_idx, point in enumerate(roi):
            if roi_idx == selected_roi and point_idx == selected_point:
                cv2.circle(frame, tuple(point), 10, (255, 0, 0), -1)  # Titik yang dipilih berwarna biru
            else:
                cv2.circle(frame, tuple(point), 5, (0, 0, 255), -1)
    return frame


def mouse_callback(event, x, y, flags, param):
    global selected_office, selected_roi, selected_point, dragging, data, json_file

    if selected_office is None:
        return

    office_data = data[selected_office]
    rois = office_data["rois"]
    scaled_rois = [scale_coords(roi) for roi in rois]

    if event == cv2.EVENT_LBUTTONDOWN:
        # Cek apakah klik berada dekat dengan titik mana pun
        for roi_idx, roi in enumerate(scaled_rois):
            for point_idx, point in enumerate(roi):
                distance = np.sqrt((x - point[0]) ** 2 + (y - point[1]) ** 2)
                if distance < 10:
                    selected_roi = roi_idx
                    selected_point = point_idx
                    dragging = True
                    print(f"Memilih ROI {roi_idx}, Titik {point_idx}")
                    return

    elif event == cv2.EVENT_MOUSEMOVE:
        if dragging and selected_roi is not None and selected_point is not None:
            # Geser titik
            dx = int(x / SCALE_X) - data[selected_office]["rois"][selected_roi][selected_point][0]
            dy = int(y / SCALE_Y) - data[selected_office]["rois"][selected_roi][selected_point][1]
            # Update koordinat di resolusi asli
            data[selected_office]["rois"][selected_roi][selected_point][0] = int(x / SCALE_X)
            data[selected_office]["rois"][selected_roi][selected_point][1] = int(y / SCALE_Y)
            print(f"Geser ROI {selected_roi}, Titik {selected_point} ke ({int(x / SCALE_X)}, {int(y / SCALE_Y)})")

    elif event == cv2.EVENT_LBUTTONUP:
        if dragging:
            dragging = False
            print(f"Selesai menggeser ROI {selected_roi}, Titik {selected_point}")
            selected_roi = None
            selected_point = None
            # Setelah menggeser, simpan perubahan
            save_json(data, json_file)
            print("Perubahan disimpan ke JSON.")


def main():
    global selected_office, data, json_file

    # Path ke file JSON
    json_file = "sources/AlFaruq/camera_conf.json"
    data = load_json(json_file)

    offices = list(data.keys())
    current_office_idx = 0

    while True:
        selected_office = offices[current_office_idx]
        office_data = data[selected_office]
        ip_camera = office_data["ip_camera"]
        rtsp_url = f"rtsp://admin:oracle2015@{ip_camera}:554/Streaming/Channels/1"

        cap = cv2.VideoCapture(rtsp_url)
        if not cap.isOpened():
            print(f"Cannot open stream for {selected_office}. Skipping to next office.")
            current_office_idx = (current_office_idx + 1) % len(offices)
            continue

        cv2.namedWindow(selected_office)
        cv2.setMouseCallback(selected_office, mouse_callback)

        print(f"Menampilkan {selected_office}.")
        print("Tekan 'n' untuk next office, 'p' untuk previous office, 'q' untuk quit.")

        while True:
            ret, frame = cap.read()
            if not ret:
                print(f"Failed to grab frame from {selected_office}.")
                break

            # Resize frame ke resolusi tampilan
            frame_display = cv2.resize(frame, (DISPLAY_WIDTH, DISPLAY_HEIGHT))

            # Dapatkan ROIs dan skala
            rois = office_data["rois"]
            scaled_rois = [scale_coords(roi) for roi in rois]

            # Gambar ROIs
            frame_display = draw_rois(frame_display, scaled_rois, selected_roi=selected_roi, selected_point=selected_point)

            cv2.imshow(selected_office, frame_display)

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                cap.release()
                cv2.destroyAllWindows()
                print("Keluar dan menyimpan perubahan.")
                save_json(data, json_file)
                return
            elif key == ord("n"):
                cap.release()
                cv2.destroyAllWindows()
                current_office_idx = (current_office_idx + 1) % len(offices)
                break
            elif key == ord("p"):
                cap.release()
                cv2.destroyAllWindows()
                current_office_idx = (current_office_idx - 1) % len(offices)
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
