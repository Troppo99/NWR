import cv2
import math
from shapely.geometry import Polygon, LineString
from shapely.ops import split
import numpy as np

video_path = "rtsp://admin:oracle2015@10.5.0.170:554/Streaming/Channels/1"
chains = []
dragging = False
preview_point = None
current_polygon = None
magnet_threshold = 10
display_width = 1280
display_height = 720


def distance(point1, point2):
    return math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)


def find_nearest_point(preview_point):
    nearest_point = None
    min_distance = float("inf")
    for chain in chains:
        for point in chain:
            dist = distance(preview_point, point)
            if dist < magnet_threshold and dist < min_distance:
                nearest_point = point
                min_distance = dist

    return nearest_point


def create_keypoint(event, x, y, flags, param):
    global chains, frame, dragging, preview_point, current_polygon

    if event == cv2.EVENT_LBUTTONDOWN:
        dragging = True
        preview_point = (x, y)

    elif event == cv2.EVENT_MOUSEMOVE:
        if dragging:
            preview_point = (x, y)
            nearest_point = find_nearest_point(preview_point)
            if nearest_point is not None:
                preview_point = nearest_point

            frame_copy = frame.copy()
            draw_chains(frame_copy)
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

            if len(chains[-1]) >= 4:
                first_point = chains[-1][0]
                last_point = chains[-1][-1]
                if distance(first_point, last_point) < magnet_threshold:
                    chains[-1][-1] = first_point
                    cv2.line(frame, last_point, first_point, (0, 0, 255), 2)
                    cv2.imshow("Video", frame)

                    polygon_points = chains[-1]
                    polygon = Polygon(polygon_points)
                    if polygon.is_valid and polygon.area > 0:
                        current_polygon = polygon
                        print(f"Poligon selesai: {polygon_points}")
                        print_borders()

                        subdivide_polygon(polygon, frame)
                    else:
                        print("Poligon tidak valid atau area nol.")
                        chains.pop()


def draw_chains(img):
    for chain in chains:
        if len(chain) > 0:
            for i in range(1, len(chain)):
                cv2.line(img, chain[i - 1], chain[i], (255, 0, 0), 2)
            for point in chain:
                cv2.circle(img, point, 5, (0, 255, 0), -1)


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


def redraw_frame():
    global frame, original_frame, current_polygon
    frame = original_frame.copy()
    draw_chains(frame)
    if current_polygon:
        subdivide_polygon(current_polygon, frame)
    cv2.imshow("Video", frame)


def print_chains():
    print("\nChains and Points:")
    non_empty_chain_count = 0
    for idx, chain in enumerate(chains):
        if len(chain) > 0:
            non_empty_chain_count += 1
            print(f"Chain {non_empty_chain_count}:")
            for i, point in enumerate(chain):
                print(f"  Point {i + 1}: ({point[0]}, {point[1]})")


def print_borders():
    borders = [[(p[0], p[1]) for p in chain] for chain in chains if len(chain) > 0]
    print(f"borders = {borders}")


def subdivide_polygon(polygon, img):
    if polygon.area < 4:
        print("Area poligon terlalu kecil untuk dibagi.")
        return

    if len(polygon.exterior.coords) - 1 != 4:
        print("Poligon bukan quadrilateral. Tidak dapat dibagi menjadi empat area.")
        return

    coords = list(polygon.exterior.coords)[:-1]

    A, B, C, D = coords

    def midpoint(p1, p2):
        return ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)

    AB_mid = midpoint(A, B)
    BC_mid = midpoint(B, C)
    CD_mid = midpoint(C, D)
    DA_mid = midpoint(D, A)

    def line_intersection(p1, p2, p3, p4):
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = p3
        x4, y4 = p4

        denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if denom == 0:
            return None

        Px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / denom
        Py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / denom
        return (int(Px), int(Py))

    intersection = line_intersection(AB_mid, CD_mid, BC_mid, DA_mid)
    if intersection is None:
        print("Tidak dapat menemukan titik potong. Poligon mungkin tidak simetris.")
        return

    border1 = [A, AB_mid, intersection, DA_mid]
    border2 = [AB_mid, B, BC_mid, intersection]
    border3 = [intersection, BC_mid, C, CD_mid]
    border4 = [DA_mid, intersection, CD_mid, D]

    borders = [border1, border2, border3, border4]
    print(f"borders = {borders}")

    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]
    for idx, border in enumerate(borders):
        pts = np.array(border, np.int32)
        pts = pts.reshape((-1, 1, 2))
        cv2.fillPoly(img, [pts], colors[idx % len(colors)])
        cv2.polylines(img, [pts], True, (0, 0, 0), 2)
        sub_polygon = Polygon(border)
        print(f"Area {idx + 1}: {sub_polygon.area}")

    cv2.imshow("Video", img)


cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video stream.")
    exit()

cv2.namedWindow("Video")
cv2.setMouseCallback("Video", create_keypoint)

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

    frame = cv2.resize(frame, (display_width, display_height))
    original_frame = frame.copy()

    draw_chains(frame)

    if dragging and preview_point is not None:
        frame_copy = frame.copy()
        if len(chains) > 0 and len(chains[-1]) > 0:
            cv2.line(frame_copy, chains[-1][-1], preview_point, (255, 0, 0), 2)
        cv2.circle(frame_copy, preview_point, 5, (0, 255, 0), -1)
        cv2.imshow("Video", frame_copy)
    else:
        cv2.imshow("Video", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("n"):
        print_borders()
        break
    elif key == 13:
        if len(chains) > 0 and len(chains[-1]) > 0:
            print_chains()
            chains.append([])
    elif key == ord("a"):
        chains = []
        current_polygon = None
        redraw_frame()
    elif key == ord("f"):
        undo_last_point()

cap.release()
cv2.destroyAllWindows()
