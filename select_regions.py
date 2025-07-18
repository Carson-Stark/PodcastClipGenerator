import cv2
import numpy as np

def select_regions(input_frame, initial_regions=None):
    import cv2
    import numpy as np

    selected_boxes = []
    drawing_box = False
    start_x, start_y = -1, -1
    image = np.copy(input_frame)

    def draw_boxes(event, x, y, flags, param):
        nonlocal drawing_box, start_x, start_y, selected_boxes, image

        if event == cv2.EVENT_LBUTTONDOWN:
            drawing_box = True
            start_x, start_y = x, y

        elif event == cv2.EVENT_MOUSEMOVE:
            if drawing_box:
                img_copy = np.copy(image)
                cv2.rectangle(img_copy, (start_x, start_y), (x, y), (0, 255, 0), 2)
                cv2.imshow('Select Regions', img_copy)

        elif event == cv2.EVENT_LBUTTONUP:
            drawing_box = False
            cv2.rectangle(image, (start_x, start_y), (x, y), (0, 255, 0), 2)
            selected_boxes.append((start_x, start_y, x, y))
            cv2.imshow('Select Regions', image)

    if initial_regions:
        for box in initial_regions:
            x1, y1, x2, y2 = box
            cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)

    cv2.namedWindow('Select Regions')
    cv2.setMouseCallback('Select Regions', draw_boxes)

    while True:
        cv2.imshow('Select Regions', image)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):
            break
        elif key == 27:
            selected_boxes = []
            break

    cv2.destroyAllWindows()
    return selected_boxes
