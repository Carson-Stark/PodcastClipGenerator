import cv2
import numpy as np

# Global variables to track mouse events
selected_boxes = []
drawing_box = False
start_x, start_y = -1, -1

def draw_boxes(event, x, y, flags, param):
    global drawing_box, start_x, start_y, selected_boxes

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing_box = True
        start_x, start_y = x, y

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing_box:
            img_copy = np.copy(image)
            cv2.rectangle(img_copy, (start_x, start_y), (x, y), (0, 255, 0), 2)
            cv2.imshow('image', img_copy)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing_box = False
        cv2.rectangle(image, (start_x, start_y), (x, y), (0, 255, 0), 2)
        selected_boxes.append((start_x, start_y, x, y))

        cv2.imshow('image', image)

def save_selected_regions(image, boxes, save_path):
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = box
        region = image[y1:y2, x1:x2]
        cv2.imwrite(f'{save_path}/region_{i+1}.jpg', region)

if __name__ == "__main__":
    video_path = 'test.mp4'
    save_path = 'selected_regions'
    
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    if not ret:
        print("Error: Unable to read the video.")
        exit()
    
    cv2.namedWindow('image')
    cv2.setMouseCallback('image', draw_boxes)
    
    image = np.copy(frame)
    
    while True:
        cv2.imshow('image', image)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):
            save_selected_regions(image, selected_boxes, save_path)
            print("Selected regions saved successfully.")
        elif key == 27:  # Press ESC to exit
            break
    
    cap.release()
    cv2.destroyAllWindows()
