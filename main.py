import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import cv2
import numpy as np
from ultralytics import YOLO


# Muat model YOLO
model = YOLO("D:/SBHNL/Resources/Models/Pretrained/FINISHING_V1/F_V1/weights/best.pt")
rtsp_url = "rtsp://admin:oracle2015@10.5.0.7:554/Streaming/Channels/1"
cap = cv2.VideoCapture(rtsp_url)


def iou(box1, box2):
    """Compute Intersection over Union (IoU) between two bounding boxes."""
    # Unpack the bounding boxes
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2

    # Calculate intersection coordinates
    xA = max(x1_min, x2_min)
    yA = max(y1_min, y2_min)
    xB = min(x1_max, x2_max)
    yB = min(y1_max, y2_max)

    # Compute area of intersection
    inter_area = max(0, xB - xA) * max(0, yB - yA)

    # Compute area of both bounding boxes
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)

    # Compute union area
    union_area = box1_area + box2_area - inter_area

    # Return IoU value
    return inter_area / union_area if union_area > 0 else 0


def filter_duplicates_with_iou(results, iou_threshold=0.5):
    """Filter duplicates using IoU and keep the bbox with the highest confidence."""
    filtered_results = []
    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()  # Bounding boxes
        scores = result.boxes.conf.cpu().numpy()  # Confidence scores
        classes = result.boxes.cls.cpu().numpy()  # Class labels

        keep_indices = []
        seen_boxes = []

        for i, box in enumerate(boxes):
            add_box = True
            for j, kept_box in enumerate(seen_boxes):
                # Calculate IoU between current box and already seen boxes
                if iou(box, kept_box) > iou_threshold and classes[i] == classes[keep_indices[j]]:
                    # If IoU is high and they belong to the same class, compare confidence scores
                    if scores[i] > scores[keep_indices[j]]:
                        keep_indices[j] = i  # Replace with higher confidence box
                    add_box = False
                    break

            if add_box:
                keep_indices.append(i)
                seen_boxes.append(box)

        filtered_result = result[keep_indices]
        filtered_results.append(filtered_result)

    return filtered_results


while cap.isOpened():
    success, frame = cap.read()
    if success:
        # Lakukan inferensi menggunakan model YOLO
        results = model(frame)

        # Filter duplikat deteksi berdasarkan IoU dan confidence score
        filtered_results = filter_duplicates_with_iou(results)

        # Anotasi dan tampilkan frame
        if filtered_results:
            annotated_frame = filtered_results[0].plot()  # Menggunakan hasil yang telah difilter
            annotated_frame = cv2.resize(annotated_frame, (640, 360))
            cv2.imshow("YOLO11 Inference", annotated_frame)

        # Hentikan jika tombol 'n' ditekan
        if cv2.waitKey(1) & 0xFF == ord("n"):
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()
