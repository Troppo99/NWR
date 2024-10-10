import cv2
import numpy as np

# List of borders (replace with your own borders)
borders = [
    [(128, 592), (302, 604), (298, 712), (110, 714)],
    [(466, 709), (298, 712), (302, 604), (466, 609)],
    [(466, 709), (466, 609), (603, 608), (615, 709)],
    [(128, 592), (302, 604), (320, 436), (154, 435)],
    [(302, 604), (466, 609), (463, 432), (320, 436)],
    [(466, 609), (603, 608), (588, 424), (463, 432)],
    [(588, 424), (780, 415), (795, 469), (650, 475), (650, 475), (593, 478)],
    [(795, 469), (780, 415), (925, 405), (942, 455), (795, 469), (795, 469), (795, 469)],
    [(942, 455), (925, 405), (1043, 396), (1062, 446)],
    [(1062, 446), (1043, 396), (1171, 383), (1130, 435), (1130, 435)],
    [(463, 432), (476, 415), (477, 335), (612, 328), (622, 422), (588, 424)],
    [(622, 422), (612, 328), (754, 320), (780, 415)],
    [(780, 415), (754, 320), (886, 316), (925, 405)],
    [(925, 405), (886, 316), (1002, 315), (1043, 396)],
    [(1043, 396), (1002, 315), (1129, 310), (1171, 383)],
    [(477, 335), (612, 328), (602, 250), (477, 257)],
    [(602, 250), (612, 328), (754, 320), (730, 244)],
    [(730, 244), (754, 320), (886, 316), (852, 245)],
    [(852, 245), (886, 316), (1002, 315), (962, 244)],
    [(962, 244), (1002, 315), (1129, 310), (1090, 249)],
    [(482, 193), (477, 257), (602, 250), (594, 188)],
    [(594, 188), (602, 250), (730, 244), (711, 184)],
    [(711, 184), (730, 244), (852, 245), (823, 183)],
    [(823, 183), (852, 245), (962, 244), (925, 185)],
    [(925, 185), (962, 244), (1090, 249), (1052, 192)],
    [(486, 142), (482, 193), (594, 188), (587, 135)],
    [(587, 135), (594, 188), (711, 184), (696, 131)],
    [(696, 131), (711, 184), (823, 183), (799, 135)],
    [(799, 135), (823, 183), (925, 185), (892, 136)],
    [(892, 136), (925, 185), (1052, 192), (1018, 144)],
    [(492, 81), (486, 142), (587, 135), (581, 75)],
    [(581, 75), (587, 135), (696, 131), (676, 72)],
    [(676, 72), (696, 131), (799, 135), (768, 70)],
    [(768, 70), (799, 135), (892, 136), (853, 77)],
    [(853, 77), (892, 136), (1018, 144), (965, 87)],
    [(500, 5), (492, 81), (581, 75), (573, 5)],
    [(573, 5), (581, 75), (676, 72), (653, 5)],
    [(653, 5), (676, 72), (768, 70), (730, 5)],
    [(730, 5), (768, 70), (853, 77), (798, 4)],
    [(798, 4), (853, 77), (965, 87), (893, 5)],
    [(932, 6), (893, 5), (965, 87), (1014, 92)],
    [(1014, 92), (965, 87), (1018, 144), (1063, 149)],
    [(1063, 149), (1018, 144), (1052, 192), (1090, 249), (1137, 246)],
    [(1137, 246), (1090, 249), (1129, 310), (1171, 383), (1177, 346), (1205, 338)],
]

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

while True:
    # Create an empty canvas
    canvas = np.zeros((display_height, display_width, 3), dtype=np.uint8)

    # Draw all borders on the canvas
    draw_borders(canvas)

    # Show the image
    cv2.imshow("Borders Editor", canvas)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):  # Press 'q' to quit
        print_borders(borders)  # Print borders in the desired format before quitting
        break
    elif key == ord("s"):  # Press 's' to save edited borders
        print_borders(borders)  # Print borders in the desired format

# Clean up
cv2.destroyAllWindows()
