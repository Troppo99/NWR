import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
import torch
import torch.nn as nn
import cv2
import numpy as np
import threading
import queue
import time


class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU(True),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 3, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


def overlay_image_alpha(img, img_overlay, pos, alpha_mask):
    x, y = pos
    h, w = img_overlay.shape[:2]
    if y + h > img.shape[0] or x + w > img.shape[1]:
        return
    slice_y = slice(y, y + h)
    slice_x = slice(x, x + w)
    img[slice_y, slice_x, :] = alpha_mask * img_overlay[:, :, :3] + (1 - alpha_mask) * img[slice_y, slice_x, :]


def infer_on_frame(frame, threshold=30, MIN_AREA=150, MAX_AREA=10000):
    resized_frame = cv2.resize(frame, (640, 360))
    frame_rgb = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
    img_tensor = torch.from_numpy(frame_rgb).permute(2, 0, 1).float().div(255.0).unsqueeze(0).to(device)

    with torch.no_grad():
        reconstructed_img = model(img_tensor)

    difference = torch.abs(img_tensor - reconstructed_img)
    diff_gray = 0.2989 * difference[:, 0, :, :] + 0.5870 * difference[:, 1, :, :] + 0.1140 * difference[:, 2, :, :]
    binary_diff = (diff_gray > (threshold / 255.0)).float()

    binary_diff_np = binary_diff.squeeze(0).cpu().numpy().astype(np.uint8) * 255

    contours, _ = cv2.findContours(binary_diff_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not hasattr(infer_on_frame, "border_mask"):
        border_polygon_np = np.array(border, dtype=np.int32)
        inner_border_np = np.array(inner_border, dtype=np.int32)
        infer_on_frame.border_mask = np.zeros((360, 640), dtype=np.uint8)
        cv2.fillPoly(infer_on_frame.border_mask, [border_polygon_np], 255)
        cv2.fillPoly(infer_on_frame.border_mask, [inner_border_np], 0)

    valid_polygons_count = 0

    contours_touching_border = [contour for contour in contours if is_contour_partially_inside_border(contour, infer_on_frame.border_mask)]

    for contour in contours_touching_border:
        clipped_contour = clip_contour_to_border(contour, infer_on_frame.border_mask)

        if clipped_contour is not None and len(clipped_contour) > 0:
            clipped_area = cv2.contourArea(clipped_contour)

            if clipped_area < MIN_AREA or clipped_area > MAX_AREA:
                continue

            valid_polygons_count += 1

            hull = cv2.convexHull(clipped_contour)
            cv2.drawContours(resized_frame, [hull], -1, (0, 255, 255), 2)

    # cv2.putText(resized_frame,f'Threshold: {threshold}',(30, 100), cv2.FONT_HERSHEY_SIMPLEX,0.6,(0, 255, 0),2)
    # cv2.putText(resized_frame,f'Min Area: {MIN_AREA}',(30, 120),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0, 255, 0),2)
    print(f"Threshold - Min Area: {threshold} - {MIN_AREA}")

    border_array = np.array(border, np.int32)
    cv2.polylines(
        resized_frame,
        [border_array],
        isClosed=True,
        color=(255, 105, 180),
        thickness=2,
    )

    inner_border_array = np.array(inner_border, np.int32)
    cv2.polylines(
        resized_frame,
        [inner_border_array],
        isClosed=True,
        color=(255, 105, 180),
        thickness=2,
    )

    return resized_frame, valid_polygons_count


def is_contour_partially_inside_border(contour, border_mask):
    for point in contour:
        x, y = int(point[0][0]), int(point[0][1])
        if border_mask[y, x] != 0:
            return True
    return False


def clip_contour_to_border(contour, border_mask):
    clipped_contour = [point for point in contour if border_mask[int(point[0][1]), int(point[0][0])] != 0]
    return np.array(clipped_contour, dtype=np.int32) if len(clipped_contour) > 0 else None


def read_frames(cap, frame_queue, stop_flag):
    # Hapus penggunaan fps dan frame_time
    while not stop_flag.is_set():
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to read frame from stream.")
            stop_flag.set()
            break
        try:
            frame_queue.put(frame, block=False)
        except queue.Full:
            pass


def process_frames(frame_queue, stop_flag, sensitivity):
    MIN_AREA = 20

    attention_icon = cv2.imread("D:/SBHNL/Images/AHMDL/Icon/attention_icon.png", cv2.IMREAD_UNCHANGED)
    check_icon = cv2.imread("D:/SBHNL/Images/AHMDL/Icon/check_icon.png", cv2.IMREAD_UNCHANGED)

    icon_height = 50
    attention_icon_resized = cv2.resize(
        attention_icon,
        (
            int(attention_icon.shape[1] * (icon_height / attention_icon.shape[0])),
            icon_height,
        ),
        interpolation=cv2.INTER_AREA,
    )
    check_icon_resized = cv2.resize(
        check_icon,
        (
            int(check_icon.shape[1] * (icon_height / check_icon.shape[0])),
            icon_height,
        ),
        interpolation=cv2.INTER_AREA,
    )

    attention_bgr = attention_icon_resized[:, :, :3]
    attention_alpha = attention_icon_resized[:, :, 3] / 255.0

    check_bgr = check_icon_resized[:, :, :3]
    check_alpha = check_icon_resized[:, :, 3] / 255.0

    is_disturbed = False
    disturbance_start_time = None
    disturbance_buffer_start_time = None
    safe_buffer_start_time = None

    while not stop_flag.is_set():
        try:
            frame = frame_queue.get(timeout=0.1)
        except queue.Empty:
            continue

        processed_frame, valid_polygons_count = infer_on_frame(frame, threshold=sensitivity, MIN_AREA=MIN_AREA)

        current_time = time.time()

        if valid_polygons_count > 0:
            # There are yellow polygons
            if not is_disturbed:
                if disturbance_buffer_start_time is None:
                    disturbance_buffer_start_time = current_time
                else:
                    if current_time - disturbance_buffer_start_time >= 3:
                        is_disturbed = True
                        disturbance_start_time = current_time
                        safe_buffer_start_time = None
            else:
                disturbance_buffer_start_time = None
                safe_buffer_start_time = None
        else:
            # No yellow polygons
            if is_disturbed:
                if safe_buffer_start_time is None:
                    safe_buffer_start_time = current_time
                else:
                    if current_time - safe_buffer_start_time >= 5:
                        is_disturbed = False
                        disturbance_start_time = None
                        disturbance_buffer_start_time = None
            else:
                disturbance_buffer_start_time = None
                safe_buffer_start_time = None

        if is_disturbed:
            if disturbance_start_time is not None:
                duration = int(current_time - disturbance_start_time)
            else:
                duration = 0
            hours = duration // 3600
            minutes = (duration % 3600) // 60
            seconds = duration % 60
            duration_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
            message = f"AREA TERGANGGU SELAMA {duration_str}"
            icon_bgr = attention_bgr
            icon_alpha = attention_alpha
            text_color = (0, 0, 255)
        else:
            message = "AREA BERSIH"
            icon_bgr = check_bgr
            icon_alpha = check_alpha
            text_color = (0, 255, 0)

        icon_x = 10
        icon_y = 30
        text_x = icon_x + icon_bgr.shape[1]
        text_y = icon_y + int(icon_bgr.shape[0] / 2) + 8

        overlay_image_alpha(processed_frame, icon_bgr, (icon_x, icon_y), icon_alpha[:, :, None])

        cv2.putText(
            processed_frame,
            message,
            (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            text_color,
            2,
        )

        cv2.imshow("Processed Video", processed_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("n") or key == ord("N"):
            stop_flag.set()
            break

        if key == ord("o"):
            sensitivity = max(0, sensitivity - 1)
        elif key == ord("p"):
            sensitivity = min(255, sensitivity + 1)

        if key == ord("l"):
            MIN_AREA = max(0, MIN_AREA - 10)
        elif key == ord("k"):
            MIN_AREA = MIN_AREA + 10


def process_video_with_threading(video_path):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error: Cannot open video {video_path}")
        return
    else:
        print(f"Successfully opened video {video_path}")

    frame_queue = queue.Queue(maxsize=10)
    stop_flag = threading.Event()

    sensitivity = 17

    reader_thread = threading.Thread(target=read_frames, args=(cap, frame_queue, stop_flag))
    reader_thread.daemon = True
    reader_thread.start()

    process_frames(frame_queue, stop_flag, sensitivity)

    reader_thread.join()
    cap.release()
    cv2.destroyAllWindows()


# Pemanggilan model tanpa weights_only
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = Autoencoder().to(device)
state_dict = torch.load(r"D:\NWR\sources\AlAhad\images\wallpaper1\Models\Wallpaper\wallpaper_v1.pth")

model.load_state_dict(state_dict)
model.eval()

border = [(int(x / 2), int(y / 2)) for (x, y) in [(99, 6), (436, 5), (456, 361), (159, 378), (128, 246), (109, 137), (99, 6)]]

inner_border = [(int(x / 2), int(y / 2)) for (x, y) in [(209, 24), (281, 20), (283, 113), (216, 119), (209, 24)]]

# video_path = r"C:\Users\Public\iVMS-4200 Site\UserData\Video\Depan_Office_163_163_20241129090712\Depan_Office_163_163_20241129071902_20241129072021_77073.mp4"
video_path = "rtsp://admin:oracle2015@10.5.0.161:554/Streaming/Channels/1"
process_video_with_threading(video_path)
