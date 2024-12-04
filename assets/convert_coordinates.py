rois = [[(501, 275), (803, 318), (1263, 398), (1262, 684), (1053, 678), (1049, 630), (898, 629), (325, 409), (497, 364)]]
resolution_inital = (1280, 720)
resolution_target = (3200, 1800)

for roi in rois:
    new_roi = []
    for point in roi:
        x, y = point
        x = int(x * resolution_target[0] / resolution_inital[0])
        y = int(y * resolution_target[1] / resolution_inital[1])
        point = (x, y)
        new_roi.append(point)
    print([new_roi])
