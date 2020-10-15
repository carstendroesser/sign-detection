import cv2

from detector.detection_utils import thresholded_to_zero, filter_by_color, keep_blobs_of_area, concatenate_blobs, \
    equalize_luminance, detect, detect_circles, RED_A, RED_B, YELLOW, BLUE, filter_boxes, enlarge_boxes


def detect_signs(image_orig):
    image_adjusted = equalize_luminance(image=image_orig)
    image_adjusted = cv2.medianBlur(src=image_adjusted, ksize=3)

    boxes = []

    # YELLOW
    mask_yellow = thresholded_to_zero(image=image_adjusted, threshold=120)
    mask_yellow = filter_by_color(image=mask_yellow, color_ranges=[YELLOW])
    mask_yellow = keep_blobs_of_area(image=mask_yellow, min_area=70, max_area=1000, min_ar=0.8, max_ar=1.2)
    mask_yellow = concatenate_blobs(mask_yellow)
    boxes = detect(image=mask_yellow, boxes=boxes, min_ar=0.8, max_ar=1.2, min_extent=0.5)

    # RED
    mask_red = filter_by_color(image=image_adjusted, color_ranges=[RED_A, RED_B])
    mask_red = concatenate_blobs(image=mask_red)
    mask_red = keep_blobs_of_area(image=mask_red, min_area=100, max_area=2000, min_ar=0.3, max_ar=1.3)

    # detect
    boxes = detect(image=mask_red, boxes=boxes, min_ar=0.8, max_ar=1.2, corners=3)
    boxes = detect_circles(image=mask_red, boxes=boxes)

    # BLUE
    mask_blue = filter_by_color(image=image_adjusted, color_ranges=[BLUE])
    mask_blue = concatenate_blobs(image=mask_blue)
    mask_blue = keep_blobs_of_area(image=mask_blue, min_area=100, max_area=2000, min_ar=0.8, max_ar=1.2)
    boxes = detect_circles(image=mask_blue, boxes=boxes)

    # clean-up boxes to remove false-positives
    boxes = filter_boxes(boxes=boxes)
    boxes = enlarge_boxes(boxes=boxes, expand_ratio=2)
    return boxes
