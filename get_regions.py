import cv2
import numpy as np

# Load YOLO
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Load class labels
classes = []
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Function to perform object detection on a frame
def detect_people(frame):
    height, width, channels = frame.shape

    # Preprocess frame
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    boxes = []
    confidences = []
    # Process outputs
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5 and class_id == 0:  # 0 corresponds to person class
                # Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height) - 50
                w = int(detection[2] * width)
                h = int(detection[3] * height) + 50

                # Calculate bounding box coordinates
                x1 = int(center_x - w / 2)
                y1 = int(center_y - h / 2)
                x2 = int(center_x + w / 2)
                y2 = int(center_y + h / 2)

                # Ensure the region is within the frame boundaries
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(width, x2)
                y2 = min(height, y2)

                boxes.append([x1, y1, x2, y2])
                confidences.append(float(confidence))


    # Non-maximum suppression to remove redundant overlapping boxes
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.5)

    # Draw bounding boxes
    """for i in range(len(boxes)):
        if i in indexes:
            x1, y1, x2, y2 = boxes[i]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)"""

    #cv2.imshow("Frame", frame)
    #cv2.waitKey(0)

    #return boxes if in indexes
    return [boxes[i] for i in range(len(boxes)) if i in indexes]

def crop_boxes(image, box):
    im_height, im_width = image.shape[:2]
    x1, y1, x2, y2 = box
    width = x2 - x1
    height = y2 - y1
    aspect_ratio = 9 / 16
    base_width = min(width, height * aspect_ratio)
    base_height = min(height, width / aspect_ratio)
    scalars = [(1, 1), (1, 2), (1.5, 1.5)]
    resized_boxes = []

    for size in scalars:
        scalar_height, scalar_width = size

        # Calculate target dimensions while preserving aspect ratio
        target_height = min (im_height, int(base_height * scalar_height))
        target_width = int(target_height * aspect_ratio / scalar_height * scalar_width)

        # Calculate the center of the bounding box
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2

        # Calculate the new coordinates of the bounding box
        new_x1 = int(center_x - target_width / 2)
        new_y1 = int(center_y - target_height / 2)
        new_x2 = int(center_x + target_width / 2)
        new_y2 = int(center_y + target_height / 2)

        # Ensure the new coordinates are within the image boundaries but keep the aspect ratio
        if new_x1 < 0:
            new_x1 = 0
            new_x2 = new_x1 + target_width
        if new_x2 > im_width:
            new_x2 = im_width
            new_x1 = new_x2 - target_width
        if new_y1 < 0:
            new_y1 = 0
            new_y2 = new_y1 + target_height
        if new_y2 > im_height:
            new_y2 = im_height
            new_y1 = new_y2 - target_height

        resized_boxes.append((new_x1, new_y1, new_x2, new_y2))

    return resized_boxes


def get_regions(frame):
    regions = []
    detections = detect_people(frame)
    detections.sort(key=lambda x: x[0])
    for detection in detections:
        crops = crop_boxes(frame, detection)
        regions.append(crops)
    return regions

def save_regions(regions):
    with open("regions.txt", "w") as f:
        for region_set in regions:
            f.write(f"{len(region_set)}\n")
            for region in region_set:
                f.write(f"{region[0]} {region[1]} {region[2]} {region[3]}\n")


# Open video file
cap = cv2.VideoCapture("input.mp4")

# Check if video file opened successfully
if not cap.isOpened():
    print("Error: Couldn't open video file.")
    exit()

regions = []
regions_set = False
# Process video frame by frame
ret, frame = cap.read()

# Perform object detection on the frame
regions = get_regions(frame)
save_regions(regions)

# Display the frame
colors = [(0, 255, 0), (0, 0, 255), (255, 0, 0)]
for r, region in enumerate(regions):
    for i, box in enumerate(region):
        x1, y1, x2, y2 = box
        cv2.rectangle(frame, (x1, y1), (x2, y2), colors[i], 2)
        #label index
        cv2.putText(frame, str(r), (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, colors[i], 2)

cv2.imshow("Frame", frame)
cv2.waitKey(0)

# Release video capture and close all windows
cap.release()
cv2.destroyAllWindows()