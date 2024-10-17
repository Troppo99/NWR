import cv2
import numpy as np

# List of borders (replace with your own borders)
borders = [
    [(30, 493), (114, 439), (158, 510), (64, 567)],
    [(114, 439), (210, 383), (261, 448), (158, 510)],
    [(210, 383), (308, 326), (372, 384), (261, 448)],
    [(308, 326), (454, 247), (533, 296), (372, 384)],
    [(117, 667), (64, 567), (158, 510), (222, 601)],
    [(222, 601), (158, 510), (261, 448), (341, 530)],
    [(341, 530), (261, 448), (372, 384), (465, 459)],
    [(465, 459), (372, 384), (533, 296), (635, 357)],
    [(533, 296), (632, 247), (731, 303), (635, 357)],
    [(731, 303), (632, 247), (713, 208), (812, 258)],
    [(149, 715), (117, 667), (222, 601), (312, 713)],
    [(312, 713), (222, 601), (341, 530), (447, 634)],
    [(447, 634), (341, 530), (465, 459), (580, 547)],
    [(580, 547), (465, 459), (635, 357), (753, 428)],
    [(753, 428), (635, 357), (731, 303), (841, 365)],
    [(841, 365), (731, 303), (812, 258), (914, 311)],
    [(312, 713), (447, 634), (541, 715)],
    [(541, 715), (447, 634), (580, 547), (714, 641), (622, 713)],
    [(714, 641), (580, 547), (753, 428), (877, 506)],
    [(877, 506), (753, 428), (841, 365), (957, 432)],
    [(957, 432), (841, 365), (914, 311), (1014, 370)],
    [(622, 713), (714, 641), (825, 712)],
    [(825, 712), (714, 641), (877, 506), (996, 580), (877, 712)],
    [(996, 580), (877, 506), (957, 432), (1061, 496)],
    [(1061, 496), (957, 432), (1014, 370), (1110, 429)],
    [(877, 712), (996, 580), (1138, 663), (1108, 714)],
    [(1138, 663), (996, 580), (1061, 496), (1184, 573)],
    [(1184, 573), (1061, 496), (1110, 429), (1221, 504)],
]

# RTSP stream URL
rtsp_url = "rtsp://admin:oracle2015@10.5.0.182:554/Streaming/Channels/1"

# Global variables for dragging
dragging = False
selected_point = None
selected_border = None

# Window settings
display_width = 1280
display_height = 720

# Magnet threshold (in pixels)
magnet_threshold = 10


# Function to calculate the distance between two points
def distance(point1, point2):
    return np.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)


# Function to find the nearest point to (x, y) within magnet threshold
def find_nearest_point(x, y):
    global borders
    nearest_point = None
    min_distance = float("inf")

    for border in borders:
        for point in border:
            dist = distance((x, y), point)
            if dist < magnet_threshold and dist < min_distance:
                nearest_point = point
                min_distance = dist

    return nearest_point


# Function to draw all borders (lines and points)
def draw_borders(img):
    for border in borders:
        for i in range(len(border)):
            cv2.line(img, border[i], border[(i + 1) % len(border)], (255, 0, 0), 2)
        for point in border:
            cv2.circle(img, point, 5, (0, 255, 0), -1)


# Function to find the closest point to the mouse click
def find_closest_point(x, y):
    global borders
    min_dist = float("inf")
    closest_point = None
    closest_border = None
    for border_idx, border in enumerate(borders):
        for point_idx, point in enumerate(border):
            dist = np.linalg.norm(np.array([x, y]) - np.array(point))
            if dist < min_dist:
                min_dist = dist
                closest_point = (border_idx, point_idx)
                closest_border = border
    return closest_point if min_dist < 10 else None, closest_border


# Mouse callback to handle dragging and moving points with magnet effect
def mouse_callback(event, x, y, flags, param):
    global dragging, selected_point, selected_border

    if event == cv2.EVENT_LBUTTONDOWN:
        closest_point, closest_border = find_closest_point(x, y)
        if closest_point:
            selected_point = closest_point
            selected_border = closest_border
            dragging = True

    elif event == cv2.EVENT_MOUSEMOVE:
        if dragging and selected_point:
            border_idx, point_idx = selected_point
            nearest_point = find_nearest_point(x, y)
            if nearest_point is not None:
                borders[border_idx][point_idx] = nearest_point
            else:
                borders[border_idx][point_idx] = (x, y)

    elif event == cv2.EVENT_LBUTTONUP:
        dragging = False
        selected_point = None


# Function to remove duplicate points in borders
def remove_duplicate_points(borders):
    cleaned_borders = []
    for border in borders:
        cleaned_border = []
        for point in border:
            if point not in cleaned_border:
                cleaned_border.append(point)
        cleaned_borders.append(cleaned_border)
    return cleaned_borders


# Function to print the borders in the desired format
def print_borders(borders):
    borders = remove_duplicate_points(borders)
    print("borders = [")
    for border in borders:
        print("    [", end="")
        print(", ".join(f"({x}, {y})" for x, y in border), end="")
        print("],")
    print("]")


# Set up OpenCV window and mouse callback
cv2.namedWindow("Borders Editor")
cv2.setMouseCallback("Borders Editor", mouse_callback)

# Capture RTSP stream
cap = cv2.VideoCapture(rtsp_url)

if not cap.isOpened():
    print("Error: Could not open RTSP stream.")
    exit()

while True:
    ret, frame = cap.read()

    if not ret:
        print("Error: Failed to read frame from RTSP stream.")
        break

    # Resize the frame to fit the display window
    frame = cv2.resize(frame, (display_width, display_height))

    # Draw all borders on the video frame
    draw_borders(frame)

    # Show the video with borders
    cv2.imshow("Borders Editor", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):  # Press 'q' to quit
        print_borders(borders)  # Print borders in the desired format before quitting
        break
    elif key == ord("s"):  # Press 's' to save edited borders
        print_borders(borders)  # Print borders in the desired format

# Clean up
cap.release()
cv2.destroyAllWindows()
