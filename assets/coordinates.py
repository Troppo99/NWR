import cv2
import math

# RTSP stream URL
video_path = "rtsp://admin:oracle2015@10.5.0.143:554/Streaming/Channels/1"

# Initialize variables for storing keypoints
chains = []  # List to store all chains of keypoints
dragging = False
preview_point = None

# Magnet threshold distance (in pixels)
magnet_threshold = 10

# Desired display resolution
display_width = 1280
display_height = 720


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
    global chains, frame, dragging, preview_point

    if event == cv2.EVENT_LBUTTONDOWN:
        # Start dragging for the first point as well
        dragging = True
        preview_point = (x, y)

    elif event == cv2.EVENT_MOUSEMOVE:
        if dragging:
            # Update the preview point
            preview_point = (x, y)
            # Check for magnet effect
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
            # Check for magnet effect one last time before saving the point
            nearest_point = find_nearest_point((x, y))
            if nearest_point is not None:
                x, y = nearest_point  # Snap to nearest point

            # Place the point, whether first or subsequent
            dragging = False
            if len(chains) == 0 or len(chains[-1]) == 0:
                # If chains is empty or the last chain is empty, start a new chain
                chains.append([(x, y)])
            else:
                # Append the point to the current chain
                chains[-1].append((x, y))
                # Draw line between the last point and this new point
                cv2.line(frame, chains[-1][-2], chains[-1][-1], (255, 0, 0), 2)
            # Draw the point
            cv2.circle(frame, chains[-1][-1], 5, (0, 255, 0), -1)
            cv2.imshow("Video", frame)


# Function to draw all chains
def draw_chains(img):
    # Draw previous chains
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
    global chains
    if len(chains) > 0:
        if len(chains[-1]) > 0:
            # Remove the last point from the current chain
            chains[-1].pop()
        # If the current chain is empty after removing, remove the chain
        if len(chains[-1]) == 0:
            chains.pop()


# Function to print all chains and their points
def print_chains():
    print("\nChains and Points:")
    non_empty_chain_count = 0  # Variabel untuk menghitung chain yang tidak kosong
    for idx, chain in enumerate(chains):
        if len(chain) > 0:  # Hanya hitung dan print jika chain tidak kosong
            non_empty_chain_count += 1
            print(f"Chain {non_empty_chain_count}:")  # Cetak nomor chain yang tidak kosong
            for i, point in enumerate(chain):
                print(f"  Point {i + 1}: ({point[0]}, {point[1]})")


# Function to print the borders at the end of the program
def print_borders():
    borders = [[(p[0], p[1]) for p in chain] for chain in chains if len(chain) > 0]
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

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        cap = cv2.VideoCapture(video_path)
        ret, frame = cap.read()

    # Resize the frame to 1280x720p for display
    frame = cv2.resize(frame, (display_width, display_height))

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
    if key == ord("n") or key == ord("N"):  # Press 'n' to exit the program
        print_borders()  # Print the final borders in the required format
        break
    elif key == 13:  # Press 'Enter' to save current chain and start a new chain
        # Only append a new chain if the last one has points
        if len(chains) > 0 and len(chains[-1]) > 0:
            print_chains()  # Print the chains and points
            chains.append([])  # Start a new chain
    elif key == ord("a"):  # Press 'a' to delete all points
        chains = []
    elif key == ord("f"):  # Press 'f' to undo the last point
        undo_last_point()

# Release the video capture object and close display windows
cap.release()
cv2.destroyAllWindows()
