import itertools

import cv2
import numpy as np

# color-ranges in HSV
RED_A = [(0, 70, 60), (10, 255, 255)]
RED_B = [(170, 70, 60), (180, 255, 255)]
BLUE = [(87, 80, 20), (132, 255, 255)]
YELLOW = [(10, 100, 190), (32, 240, 255)]


def filter_boxes(boxes):
    for box_a, box_b in itertools.combinations(boxes[:], 2):
        if (box_a in boxes) and (box_b in boxes):
            box_intersection = intersection(box_a, box_b)
            area_overlap = box_intersection[2] * box_intersection[3]
            area_a = box_a[2] * box_a[3]
            area_b = box_b[2] * box_b[3]
            iou = area_overlap / (area_a + area_b - area_overlap)

            # check if one is completely inside the other
            if area_overlap == area_a:
                boxes.remove(box_a)
                continue
            if area_overlap == area_b:
                boxes.remove(box_b)
                continue

            if iou > 0.01:
                boxes.remove(box_b)
                boxes.remove(box_a)

    return boxes


def enlarge_boxes(boxes, expand_ratio):
    new_boxes = []
    for i in range(0, len(boxes)):
        new_box = []
        additional_width = boxes[i][2] // expand_ratio
        additional_height = boxes[i][3] // expand_ratio
        new_box.append(int(boxes[i][0] - additional_width / 2))
        new_box.append(int(boxes[i][1] - additional_height / 2))
        new_box.append(int(boxes[i][2] + additional_width))
        new_box.append(int(boxes[i][3] + additional_height))
        # TODO: a_max not valid!
        new_box = np.clip(new_box, a_min=0, a_max=1024)
        new_boxes.append(new_box)

    return new_boxes


def equalize_luminance(image):
    image = cv2.cvtColor(src=image, code=cv2.COLOR_BGR2YUV)
    image[:, :, 0] = cv2.equalizeHist(src=image[:, :, 0])
    return cv2.cvtColor(src=image, code=cv2.COLOR_YUV2RGB)


def filter_by_color(image, color_ranges):
    image = cv2.cvtColor(src=image, code=cv2.COLOR_RGB2HSV)
    mask = np.zeros(shape=(image.shape[0], image.shape[1]), dtype=np.uint8)

    for color_range in color_ranges:
        mask = mask | cv2.inRange(src=image, lowerb=color_range[0], upperb=color_range[1])
    return mask


def concatenate_blobs(image, size=(5, 3)):
    # connect in height -> Einbahnstraßenschild
    kernel = np.ones(shape=size, dtype=np.uint8)
    # DILATE -> worse
    # OPEN -> worse
    # ERODE -> worse
    return cv2.morphologyEx(src=image, op=cv2.MORPH_CLOSE, kernel=kernel)


def keep_blobs_of_area(image, min_area, max_area, min_ar, max_ar):
    # find all your connected components
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(image=image, connectivity=8)
    tops = stats[1:, 1]
    widths = stats[1:, 2]
    heights = stats[1:, 3]
    sizes = stats[1:, 4]
    nb_components = nb_components - 1

    img2 = np.zeros(shape=output.shape, dtype=np.uint8)
    for i in range(0, nb_components):
        if min_area <= sizes[i] <= max_area:
            if tops[i] > 0:
                # check the aspect ratio for a given blob
                # handle two signs on top of each other.
                # this is mostly the case for red warning signs
                # 1.3, 0.3 for red
                if max_ar >= (widths[i] / heights[i]) >= min_ar:
                    img2[output == i + 1] = 255

    return img2


def detect_circles(image, boxes):
    # alternativly: simple blob detector
    circles = cv2.HoughCircles(image=image, method=cv2.HOUGH_GRADIENT, dp=1.0, minDist=50, param1=170, param2=10,
                               minRadius=7,
                               maxRadius=40)

    if circles is not None:
        circles = np.uint16(np.around(circles))
        for circle in circles[0, :]:
            if circle[2] < circle[0] and circle[2] < circle[1]:
                box = [circle[0] - circle[2], circle[1] - circle[2], circle[2] * 2, circle[2] * 2]
                boxes.append(box)

    return boxes


def thresholded_to_zero(image, threshold):
    image_adjusted = np.copy(image)
    mask = cv2.cvtColor(src=image_adjusted, code=cv2.COLOR_RGB2GRAY)
    mask = cv2.threshold(src=mask, thresh=threshold, maxval=255, type=cv2.THRESH_BINARY)[1]
    pixels_ooi = np.where(mask == 0)
    image_adjusted[pixels_ooi[0], pixels_ooi[1], :] = [0, 0, 0]
    return image_adjusted


def detect(image, boxes, min_ar, max_ar, corners=None, min_extent=None):
    # image is in grayscale
    contours, hierarchy = cv2.findContours(image=image, mode=cv2.RETR_LIST, method=cv2.CHAIN_APPROX_SIMPLE)
    for i in range(0, len(contours)):
        contour = contours[i]
        area_contour = cv2.contourArea(contour=contour)

        box = cv2.boundingRect(array=contour)
        area_box = box[2] * box[3]
        extent = float(area_contour) / area_box
        ar = box[2] / box[3]

        peri = cv2.arcLength(curve=contour, closed=True)
        approx = cv2.approxPolyDP(curve=contour, epsilon=0.06 * peri, closed=True)

        if min_ar <= ar <= max_ar:
            if corners is None:
                if min_extent is None:
                    boxes.append(box)
                elif extent >= min_extent and (area_contour > 50):
                    boxes.append(box)
            elif corners == len(approx):
                if min_extent is None:
                    boxes.append(box)
                elif (extent >= min_extent) and (area_contour > 50):
                    boxes.append(box)

    return boxes


def intersection(box_a, box_b):
    x = max(box_a[0], box_b[0])
    y = max(box_a[1], box_b[1])
    width = min(box_a[0] + box_a[2], box_b[0] + box_b[2]) - x
    height = min(box_a[1] + box_a[3], box_b[1] + box_b[3]) - y
    if width < 0 or height < 0:
        return 0, 0, 0, 0
    return x, y, width, height


def crop_to_square_and_resize(image, desired_shape):
    height = image.shape[0]
    width = image.shape[1]
    crop_size = 0

    if height > width:
        crop_size = width
    else:
        crop_size = height

    start_x = width // 2 - (crop_size // 2)
    start_y = height // 2 - (crop_size // 2)
    image = image[start_y:start_y + crop_size, start_x:start_x + crop_size]

    # resize to shape_input
    return cv2.resize(image, (desired_shape[1], desired_shape[0]), interpolation=cv2.INTER_AREA)
