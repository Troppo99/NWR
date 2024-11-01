import cv2
import math
from shapely.geometry import Polygon, LineString, Point
from shapely.ops import triangulate
import numpy as np

# RTSP stream URL atau path video lokal
video_path = "rtsp://admin:oracle2015@10.5.0.182:554/Streaming/Channels/1"

# Initialize variables for storing keypoints
chains = []  # List to store all chains of keypoints
dragging = False
preview_point = None
current_polygon = None  # To store the completed polygon

# Magnet threshold distance (in pixels)
magnet_threshold = 10

# Desired display resolution
display_width = 1280
display_height = 720

# Variable to store current number of subdivisions
current_polygon_subdivisions = 4  # Default to 4


# Function to calculate the distance between two points
def distance(point1, point2):
    return math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)


# Function to find the nearest point to the preview_point within the magnet threshold
def find_nearest_point(preview_point):
    nearest_point = None
    min_distance = float("inf")

    # Search in all chains (previous points)
    for chain in chains:
        for point in chain:
            dist = distance(preview_point, point)
            if dist < magnet_threshold and dist < min_distance:
                nearest_point = point
                min_distance = dist

    return nearest_point


# Mouse callback function to create keypoints and draw connecting lines
def create_keypoint(event, x, y, flags, param):
    global chains, frame, dragging, preview_point, current_polygon, current_polygon_subdivisions

    if event == cv2.EVENT_LBUTTONDOWN:
        dragging = True
        preview_point = (x, y)

    elif event == cv2.EVENT_MOUSEMOVE:
        if dragging:
            preview_point = (x, y)
            nearest_point = find_nearest_point(preview_point)
            if nearest_point is not None:
                preview_point = nearest_point

            # Create a copy of the frame to show the preview
            frame_copy = frame.copy()
            # Draw all existing chains
            draw_chains(frame_copy)
            # Draw the preview line and point
            if len(chains) > 0 and len(chains[-1]) > 0:
                cv2.line(frame_copy, chains[-1][-1], preview_point, (255, 0, 0), 2)
            cv2.circle(frame_copy, preview_point, 5, (0, 255, 0), -1)
            cv2.imshow("Video", frame_copy)

    elif event == cv2.EVENT_LBUTTONUP:
        if dragging:
            nearest_point = find_nearest_point((x, y))
            if nearest_point is not None:
                x, y = nearest_point  # Snap to nearest point

            dragging = False
            if len(chains) == 0 or len(chains[-1]) == 0:
                chains.append([(x, y)])
            else:
                chains[-1].append((x, y))
                cv2.line(frame, chains[-1][-2], chains[-1][-1], (255, 0, 0), 2)
            cv2.circle(frame, chains[-1][-1], 5, (0, 255, 0), -1)
            cv2.imshow("Video", frame)

            # Check if the last point is near the first point to close the polygon
            if len(chains[-1]) >= 4:
                first_point = chains[-1][0]
                last_point = chains[-1][-1]
                if distance(first_point, last_point) < magnet_threshold:
                    # Close the polygon
                    chains[-1][-1] = first_point  # Ensure it's exactly the first point
                    cv2.line(frame, last_point, first_point, (0, 0, 255), 2)  # Red line to indicate closure
                    cv2.imshow("Video", frame)

                    # Create a Shapely polygon
                    polygon_points = chains[-1]
                    polygon = Polygon(polygon_points)
                    if polygon.is_valid and polygon.area > 0:
                        current_polygon = polygon
                        print(f"Poligon selesai: {polygon_points}")
                        print_borders()

                        # Ask user for number of subdivisions
                        n = get_subdivision_input()
                        if n:
                            subdivide_polygon_grid(polygon, img=frame, n_subdivisions=n)
                    else:
                        print("Poligon tidak valid atau area nol.")
                        chains.pop()  # Remove the invalid polygon


# Function to draw all chains
def draw_chains(img):
    # Draw all chains
    for chain in chains:
        if len(chain) > 0:
            # Draw lines
            for i in range(1, len(chain)):
                cv2.line(img, chain[i - 1], chain[i], (255, 0, 0), 2)
            # Draw circles at keypoints
            for point in chain:
                cv2.circle(img, point, 5, (0, 255, 0), -1)


# Function to undo the last point from the current chain or from the previous chains
def undo_last_point():
    global chains, current_polygon
    if len(chains) > 0:
        if len(chains[-1]) > 0:
            chains[-1].pop()
            redraw_frame()
        if len(chains[-1]) == 0:
            chains.pop()
            redraw_frame()
        if current_polygon and not chains:
            current_polygon = None


# Function to redraw the frame (without needing to capture a new frame)
def redraw_frame():
    global frame, original_frame, current_polygon, current_polygon_subdivisions
    frame = original_frame.copy()
    draw_chains(frame)
    if current_polygon:
        subdivide_polygon_grid(current_polygon, img=frame, n_subdivisions=current_polygon_subdivisions)
    cv2.imshow("Video", frame)


# Function to print all chains and their points
def print_chains():
    print("\nChains and Points:")
    non_empty_chain_count = 0  # Variable to count non-empty chains
    for idx, chain in enumerate(chains):
        if len(chain) > 0:  # Only count and print non-empty chains
            non_empty_chain_count += 1
            print(f"Chain {non_empty_chain_count}:")  # Print non-empty chain number
            for i, point in enumerate(chain):
                print(f"  Point {i + 1}: ({point[0]}, {point[1]})")


# Function to print the borders at the end of the program
def print_borders():
    borders = [[(int(p[0]), int(p[1])) for p in chain] for chain in chains if len(chain) > 0]
    print(f"borders = {borders}")


# Function to get user input for number of subdivisions
def get_subdivision_input():
    """
    Function to get user input for number of subdivisions.
    For simplicity, using console input.
    """
    try:
        n = int(input("Masukkan jumlah pembagian (misal 4, 9, 16): "))
        if n < 1:
            print("Jumlah pembagian harus minimal 1.")
            return None
        # Check if n is a perfect square
        sqrt_n = math.sqrt(n)
        if not sqrt_n.is_integer():
            print("Jumlah pembagian harus merupakan bilangan kuadrat sempurna (misal 4, 9, 16).")
            return None
        return n
    except ValueError:
        print("Input tidak valid. Silakan masukkan angka integer.")
        return None


# Function to subdivide quadrilateral into grid-based subdivisions
def subdivide_polygon_grid(polygon, img, n_subdivisions=4):
    """
    Subdivide the quadrilateral polygon into n_subdivisions parts with grid-based subdivision.

    :param polygon: Shapely Polygon object (quadrilateral)
    :param img: OpenCV image to draw the subdivisions
    :param n_subdivisions: Number of desired subdivisions (must be a perfect square)
    """
    if polygon.area < n_subdivisions:
        print(f"Area poligon terlalu kecil untuk dibagi menjadi {n_subdivisions} bagian.")
        return

    # Check if polygon is quadrilateral
    coords = list(polygon.exterior.coords)[:-1]  # Remove the duplicate first/last point
    if len(coords) != 4:
        print("Poligon bukan quadrilateral. Tidak dapat dibagi menjadi grid.")
        return

    # Check if n_subdivisions is a perfect square
    sqrt_n = math.sqrt(n_subdivisions)
    if not sqrt_n.is_integer():
        print("Jumlah pembagian harus merupakan bilangan kuadrat sempurna (misal 4, 9, 16).")
        return

    grid_rows = grid_cols = int(sqrt_n)

    A, B, C, D = coords

    # Function to compute equally spaced points along a side
    def compute_division_points(p1, p2, divisions):
        points = []
        for i in range(divisions + 1):
            t = i / divisions
            point = interpolate(p1, p2, t)
            points.append(point)
        return points

    # Function to interpolate between two points
    def interpolate(p1, p2, t):
        return (p1[0] + (p2[0] - p1[0]) * t, p1[1] + (p2[1] - p1[1]) * t)

    # Compute division points on each side
    top_div = compute_division_points(A, B, grid_cols)
    bottom_div = compute_division_points(D, C, grid_cols)
    left_div = compute_division_points(A, D, grid_rows)
    right_div = compute_division_points(B, C, grid_rows)

    # Now, for each row and column, interpolate points to define grid cells
    # Compute intermediate points between top_div and bottom_div for each column
    grid_points = []
    for row in range(grid_rows + 1):
        t = row / grid_rows
        row_points = []
        for col in range(grid_cols + 1):
            # Interpolate between top_div[col] and bottom_div[col] based on t
            p = interpolate(top_div[col], bottom_div[col], t)
            row_points.append(p)
        grid_points.append(row_points)

    # Now, define sub-polygons
    borders = []
    for row in range(grid_rows):
        for col in range(grid_cols):
            top_left = grid_points[row][col]
            top_right = grid_points[row][col + 1]
            bottom_right = grid_points[row + 1][col + 1]
            bottom_left = grid_points[row + 1][col]
            sub_border = [top_left, top_right, bottom_right, bottom_left]
            # Apply rounding to each coordinate
            sub_border_int = [(int(round(p[0])), int(round(p[1]))) for p in sub_border]
            borders.append(sub_border_int)

    # Print borders
    print(f"borders = {borders}")

    # Draw sub-polygons with different colors
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255), (128, 0, 128), (0, 128, 128), (128, 128, 0), (0, 0, 128)]  # Red  # Green  # Blue  # Cyan  # Magenta  # Yellow  # Purple  # Teal  # Olive  # Navy

    for idx, border in enumerate(borders):
        pts = np.array(border, np.int32)
        pts = pts.reshape((-1, 1, 2))
        color = colors[idx % len(colors)]
        cv2.fillPoly(img, [pts], color)
        cv2.polylines(img, [pts], True, (0, 0, 0), 2)
        # Calculate and print area
        sub_polygon = Polygon(border)
        print(f"Area {idx + 1}: {sub_polygon.area}")

    # Update the image with subdivisions
    cv2.imshow("Video", img)

    # Update the borders variable to include all sub-polygons
    print(f"borders = {borders}")


# Open the video stream
cap = cv2.VideoCapture(video_path)

# Check if the video stream is opened successfully
if not cap.isOpened():
    print("Error: Could not open video stream.")
    exit()

# Create a window and set the mouse callback function
cv2.namedWindow("Video")
cv2.setMouseCallback("Video", create_keypoint)

# Read the first frame to use as the base for redrawing
ret, frame = cap.read()
if not ret:
    print("Failed to grab frame")
    exit()
frame = cv2.resize(frame, (display_width, display_height))
original_frame = frame.copy()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        cap.release()
        cap = cv2.VideoCapture(video_path)
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame after reconnecting")
            break

    # Resize the frame to desired resolution
    frame = cv2.resize(frame, (display_width, display_height))
    original_frame = frame.copy()

    # Draw all the chains on the frame
    draw_chains(frame)

    # If dragging, update the preview
    if dragging and preview_point is not None:
        frame_copy = frame.copy()
        # Draw the preview line and point
        if len(chains) > 0 and len(chains[-1]) > 0:
            cv2.line(frame_copy, chains[-1][-1], preview_point, (255, 0, 0), 2)
        cv2.circle(frame_copy, preview_point, 5, (0, 255, 0), -1)
        cv2.imshow("Video", frame_copy)
    else:
        cv2.imshow("Video", frame)

    key = cv2.waitKey(1) & 0xFF  # Change waitKey to 1 for real-time display
    if key == ord("n"):  # Press 'n' to exit the program
        print_borders()  # Print the final borders in the required format
        break
    elif key == 13:  # Press 'Enter' to save current chain and start a new chain
        # Only append a new chain if the last one has points
        if len(chains) > 0 and len(chains[-1]) > 0:
            print_chains()  # Print the chains and points
            chains.append([])  # Start a new chain
    elif key == ord("a"):  # Press 'a' to delete all points
        chains = []
        current_polygon = None
        redraw_frame()
    elif key == ord("f"):  # Press 'f' to undo the last point
        undo_last_point()

# Release the video capture object and close display windows
cap.release()
cv2.destroyAllWindows()
