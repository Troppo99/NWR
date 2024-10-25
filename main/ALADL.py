import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
import cv2
import cvzone
from ultralytics import YOLO
import math
import numpy as np


def process_frame(frame):
    results = model(frame)
    boxes_info = []
    for result in results:
        for result_box in results:
            for box in result_box.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = math.ceil(box.conf[0] * 100) / 100
                class_id = model.names[int(box.cls[0])]
                if conf > 0:
                    boxes_info.append((x1, y1, x2, y2, conf, class_id))
    return frame, boxes_info


def overlay_image_alpha(img, img_overlay, pos, alpha_mask):
    x, y = pos

    # Image overlay region
    h, w = img_overlay.shape[0], img_overlay.shape[1]

    # Overlaying
    img[y : y + h, x : x + w] = (img[y : y + h, x : x + w] * (1 - alpha_mask) + img_overlay * alpha_mask).astype(np.uint8)


videos = ["rtsp://admin:oracle2015@10.5.0.206:554/Streaming/Channels/1", "D:/SBHNL/Videos/AHMDL/Test/0926(1).mp4"]
video = videos[1]
cap = cv2.VideoCapture(video)
model = YOLO("D:/SBHNL/Resources/Models/Pretrained/PUDDLE/P_V3/weights/best.pt")

# Load the external PNG images with alpha channel (4th channel)
icon_image = cv2.imread("D:/SBHNL/Images/AHMDL/Icon/puddle_icon.png", cv2.IMREAD_UNCHANGED)
icon_height = 100  # Resize all icons to this height
icon_width = int(icon_image.shape[1] * (icon_height / icon_image.shape[0]))  # Scale the width proportionally
icon_image_resized = cv2.resize(icon_image, (icon_width, icon_height))

# Load the check icon and attention icon
check_icon = cv2.imread("D:/SBHNL/Images/AHMDL/Icon/check_icon.png", cv2.IMREAD_UNCHANGED)
attention_icon = cv2.imread("D:/SBHNL/Images/AHMDL/Icon/attention_icon.png", cv2.IMREAD_UNCHANGED)

# Resize both icons to match the icon_height (same as the main icon)
check_icon_resized = cv2.resize(check_icon, (icon_width, icon_height))
attention_icon_resized = cv2.resize(attention_icon, (icon_width, icon_height))

# Split the PNG images into BGR and Alpha channels
icon_bgr = icon_image_resized[:, :, :3]  # BGR channels
icon_alpha = icon_image_resized[:, :, 3] / 255.0  # Alpha channel (normalized to range 0-1)

check_bgr = check_icon_resized[:, :, :3]
check_alpha = check_icon_resized[:, :, 3] / 255.0

attention_bgr = attention_icon_resized[:, :, :3]
attention_alpha = attention_icon_resized[:, :, 3] / 255.0

while True:
    ret, frame = cap.read()

    frame_results, boxes_info = process_frame(frame)

    # Menampilkan jumlah objek terdeteksi di sisi kanan
    total_objects = len(boxes_info)
    text_x_position = icon_width + 30  # Text starts 30px to the right of the resized image

    # Overlay the icon image on the frame without the background (using alpha mask)
    overlay_image_alpha(frame, icon_bgr, (20, 120), icon_alpha[:, :, None])

    # Display the total objects on the right side of the image
    cvzone.putTextRect(frame_results, f"TOTAL PUDDLES : {total_objects}", (text_x_position, 180), scale=2.5, thickness=2, colorR=(235, 183, 23))

    # Conditional display for check or attention icon and message
    if total_objects == 0:
        # Display the check icon to the left of the text
        overlay_image_alpha(frame_results, check_bgr, (text_x_position - icon_width, 220), check_alpha[:, :, None])
        # Display the "Waterproofing" text next to the check icon
        cvzone.putTextRect(frame_results, "It is Dry :)", (text_x_position, 280), scale=2.5, thickness=2, colorR=(0, 255, 0))
    else:
        # Display the attention icon to the left of the text
        overlay_image_alpha(frame_results, attention_bgr, (text_x_position - icon_width, 220), attention_alpha[:, :, None])
        # Display the "It is Dry :))" text next to the attention icon
        cvzone.putTextRect(frame_results, "It is Flooded!", (text_x_position, 280), scale=2.5, thickness=2, colorR=(0, 0, 255))

    # Menampilkan kotak dan label dari objek yang terdeteksi
    for x1, y1, x2, y2, conf, class_id in boxes_info:
        cvzone.cornerRect(frame_results, (x1, y1, x2 - x1, y2 - y1), rt=0, colorC=(0, 255, 255))
        cvzone.putTextRect(frame_results, f"{class_id}", (x1, y1 - 15), colorR=(235, 183, 23), thickness=2, scale=1)

    # Display the final frame
    frame_show = cv2.resize(frame_results, (1280, 720))
    cv2.imshow("THREADING", frame_show)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("n"):
        break

cap.release()
cv2.destroyAllWindows()
