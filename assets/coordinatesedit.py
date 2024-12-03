import cv2
import numpy as np


def round_borders(borders):
    rounded_borders = []
    for border in borders:
        rounded_border = [(round(x), round(y)) for x, y in border]
        rounded_borders.append(rounded_border)
    return rounded_borders


# List of borders (replace with your own borders)
borders = [
    [(25, 497), (66, 471), (92, 523), (47, 552)],
    [(66, 471), (107, 445), (136, 494), (92, 523)],
    [(47, 552), (92, 523), (118, 575), (69, 606)],
    [(92, 523), (136, 494), (166, 544), (118, 575)],
    [(166, 544), (155, 526), (300, 445), (314, 460)],
    [(155, 526), (145, 509), (286, 429), (300, 445)],
    [(145, 509), (134, 491), (272, 414), (286, 429)],
    [(314, 460), (300, 445), (445, 363), (462, 377)],
    [(300, 445), (286, 429), (428, 350), (445, 363)],
    [(286, 429), (272, 414), (411, 336), (428, 350)],
    [(462, 377), (445, 363), (590, 282), (610, 293)],
    [(445, 363), (428, 350), (569, 270), (590, 282)],
    [(428, 350), (411, 336), (549, 259), (569, 270)],
    [(549, 259), (524, 246), (391, 316), (407, 339)],
    [(524, 246), (498, 232), (376, 294), (391, 316)],
    [(407, 339), (391, 316), (259, 387), (265, 419)],
    [(391, 316), (376, 294), (253, 355), (259, 387)],
    [(610, 293), (524, 342), (559, 366), (647, 315)],
    [(524, 342), (438, 390), (471, 417), (559, 366)],
    [(438, 390), (351, 439), (382, 469), (471, 417)],
    [(351, 439), (265, 487), (294, 520), (382, 469)],
    [(265, 487), (179, 536), (206, 571), (294, 520)],
    [(647, 315), (559, 366), (594, 390), (684, 337)],
    [(559, 366), (471, 417), (504, 444), (594, 390)],
    [(471, 417), (382, 469), (413, 498), (504, 444)],
    [(382, 469), (294, 520), (323, 552), (413, 498)],
    [(294, 520), (206, 571), (233, 606), (323, 552)],
    [(684, 337), (594, 390), (628, 415), (720, 358)],
    [(594, 390), (504, 444), (536, 471), (628, 415)],
    [(504, 444), (413, 498), (445, 528), (536, 471)],
    [(413, 498), (323, 552), (353, 584), (445, 528)],
    [(323, 552), (233, 606), (261, 641), (353, 584)],
    [(720, 358), (628, 415), (663, 439), (757, 380)],
    [(628, 415), (536, 471), (569, 499), (663, 439)],
    [(536, 471), (445, 528), (476, 558), (569, 499)],
    [(445, 528), (353, 584), (382, 617), (476, 558)],
    [(353, 584), (261, 641), (288, 676), (382, 617)],
    [(757, 380), (663, 439), (698, 464), (794, 402)],
    [(663, 439), (569, 499), (602, 526), (698, 464)],
    [(569, 499), (476, 558), (507, 587), (602, 526)],
    [(476, 558), (382, 617), (411, 649), (507, 587)],
    [(382, 617), (288, 676), (315, 711), (411, 649)],
    [(794, 402), (825, 379), (787, 356), (755, 379)],
    [(825, 379), (856, 356), (820, 334), (787, 356)],
    [(856, 356), (888, 332), (852, 312), (820, 334)],
    [(888, 332), (919, 309), (884, 290), (852, 312)],
    [(755, 379), (787, 356), (749, 334), (716, 356)],
    [(787, 356), (820, 334), (783, 313), (749, 334)],
    [(820, 334), (852, 312), (816, 292), (783, 313)],
    [(852, 312), (884, 290), (850, 270), (816, 292)],
    [(716, 356), (749, 334), (711, 312), (676, 332)],
    [(749, 334), (783, 313), (746, 292), (711, 312)],
    [(783, 313), (816, 292), (781, 272), (746, 292)],
    [(816, 292), (850, 270), (816, 251), (781, 272)],
    [(676, 332), (711, 312), (673, 290), (637, 309)],
    [(711, 312), (746, 292), (709, 270), (673, 290)],
    [(746, 292), (781, 272), (745, 251), (709, 270)],
    [(781, 272), (816, 251), (781, 232), (745, 251)],
    [(794, 402), (825, 379), (868, 404), (838, 429)],
    [(825, 379), (856, 356), (897, 380), (868, 404)],
    [(856, 356), (888, 332), (926, 356), (897, 380)],
    [(888, 332), (919, 309), (955, 331), (926, 356)],
    [(838, 429), (868, 404), (910, 430), (882, 456)],
    [(868, 404), (897, 380), (937, 404), (910, 430)],
    [(897, 380), (926, 356), (964, 379), (937, 404)],
    [(926, 356), (955, 331), (992, 354), (964, 379)],
    [(882, 456), (910, 430), (952, 456), (927, 482)],
    [(910, 430), (937, 404), (977, 429), (952, 456)],
    [(937, 404), (964, 379), (1002, 402), (977, 429)],
    [(964, 379), (992, 354), (1028, 376), (1002, 402)],
    [(927, 482), (952, 456), (994, 481), (971, 509)],
    [(952, 456), (977, 429), (1018, 454), (994, 481)],
    [(977, 429), (1002, 402), (1041, 426), (1018, 454)],
    [(1002, 402), (1028, 376), (1064, 398), (1041, 426)],
    [(971, 509), (1018, 539), (1039, 510), (994, 481)],
    [(1018, 539), (1064, 569), (1084, 540), (1039, 510)],
    [(1064, 569), (1111, 599), (1129, 569), (1084, 540)],
    [(1111, 599), (1158, 629), (1174, 598), (1129, 569)],
    [(994, 481), (1039, 510), (1061, 482), (1018, 454)],
    [(1039, 510), (1084, 540), (1104, 510), (1061, 482)],
    [(1084, 540), (1129, 569), (1148, 539), (1104, 510)],
    [(1129, 569), (1174, 598), (1191, 568), (1148, 539)],
    [(1018, 454), (1061, 482), (1082, 454), (1041, 426)],
    [(1061, 482), (1104, 510), (1124, 481), (1082, 454)],
    [(1104, 510), (1148, 539), (1166, 509), (1124, 481)],
    [(1148, 539), (1191, 568), (1208, 537), (1166, 509)],
    [(1041, 426), (1082, 454), (1104, 425), (1064, 398)],
    [(1082, 454), (1124, 481), (1144, 452), (1104, 425)],
    [(1124, 481), (1166, 509), (1184, 479), (1144, 452)],
    [(1166, 509), (1208, 537), (1224, 506), (1184, 479)],
    [(794, 402), (750, 430), (799, 460), (841, 430)],
    [(750, 430), (707, 458), (757, 490), (799, 460)],
    [(707, 458), (663, 486), (715, 520), (757, 490)],
    [(663, 486), (620, 514), (674, 550), (715, 520)],
    [(620, 514), (576, 542), (632, 580), (674, 550)],
    [(841, 430), (799, 460), (847, 490), (887, 457)],
    [(799, 460), (757, 490), (807, 522), (847, 490)],
    [(757, 490), (715, 520), (767, 554), (807, 522)],
    [(715, 520), (674, 550), (728, 586), (767, 554)],
    [(674, 550), (632, 580), (688, 619), (728, 586)],
    [(887, 457), (847, 490), (896, 519), (934, 485)],
    [(847, 490), (807, 522), (858, 554), (896, 519)],
    [(807, 522), (767, 554), (820, 588), (858, 554)],
    [(767, 554), (728, 586), (781, 623), (820, 588)],
    [(728, 586), (688, 619), (743, 657), (781, 623)],
    [(934, 485), (896, 519), (944, 549), (980, 512)],
    [(896, 519), (858, 554), (908, 586), (944, 549)],
    [(858, 554), (820, 588), (872, 622), (908, 586)],
    [(820, 588), (781, 623), (835, 659), (872, 622)],
    [(781, 623), (743, 657), (799, 696), (835, 659)],
    [(980, 512), (944, 549), (993, 579), (1027, 540)],
    [(944, 549), (908, 586), (958, 618), (993, 579)],
    [(908, 586), (872, 622), (924, 656), (958, 618)],
    [(872, 622), (835, 659), (889, 695), (924, 656)],
    [(835, 659), (799, 696), (855, 734), (889, 695)],
    [(1158, 629), (1092, 584), (1062, 627), (1133, 674)],
    [(1092, 584), (1027, 540), (991, 581), (1062, 627)],
    [(1133, 674), (1062, 627), (1032, 670), (1108, 718)],
    [(1062, 627), (991, 581), (955, 622), (1032, 670)],
]
borders = round_borders(borders)

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
