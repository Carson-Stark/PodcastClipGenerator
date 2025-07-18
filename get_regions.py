import cv2
import numpy as np
from moviepy.editor import VideoFileClip

# Load YOLO model and config
net = cv2.dnn.readNet("models/yolov3.weights", "models/yolov3.cfg")
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Load class labels
classes = []
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

def detect_people(frame):
    height, width, channels = frame.shape

    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    boxes = []
    confidences = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5 and class_id == 0:  # person class
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height) - 50
                w = int(detection[2] * width)
                h = int(detection[3] * height) + 50

                x1 = int(center_x - w / 2)
                y1 = int(center_y - h / 2)
                x2 = int(center_x + w / 2)
                y2 = int(center_y + h / 2)

                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(width, x2)
                y2 = min(height, y2)

                boxes.append([x1, y1, x2, y2])
                confidences.append(float(confidence))

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.5)

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

        target_height = min(im_height, int(base_height * scalar_height))
        target_width = int(target_height * aspect_ratio / scalar_height * scalar_width)

        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2

        new_x1 = int(center_x - target_width / 2)
        new_y1 = int(center_y - target_height / 2)
        new_x2 = int(center_x + target_width / 2)
        new_y2 = int(center_y + target_height / 2)

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

def get_first_frame(input_video_path):

    video = VideoFileClip(input_video_path)
    return np.array(video.get_frame(0))

def get_regions(first_frame):

    regions = []
    detections = detect_people(first_frame)
    detections.sort(key=lambda x: x[0])
    for detection in detections:
        crops = crop_boxes(first_frame, detection)
        regions.append(crops)

    return regions

def save_regions(regions, output_file="outputs/regions.txt"):
    with open(output_file, "w") as f:
        for region_set in regions:
            f.write(f"{len(region_set)}\n")
            for region in region_set:
                f.write(f"{region[0]} {region[1]} {region[2]} {region[3]}\n")

    print(f"Regions saved to {output_file}")

    return output_file

def show_regions(frame, regions):

    colors = [(0, 255, 0), (0, 0, 255), (255, 0, 0)]
    for r, region in enumerate(regions):
        for i, box in enumerate(region):
            x1, y1, x2, y2 = box
            print(box)
            cv2.rectangle(frame, (x1, y1), (x2, y2), colors[i], 2)
            #label index
            cv2.putText(frame, str(r), (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, colors[i], 2)

    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    cv2.imshow("Frame", frame)
    cv2.waitKey(0)

    cv2.destroyAllWindows()

if __name__ == "__main__":
    input_video_path = "input.mp4"  # Example input video
    output_regions_path = "outputs/regions.txt"  # File to save regions

    frame = get_first_frame(input_video_path)
    regions = get_regions(frame)
    show_regions(frame, regions)
    save_regions(regions, output_regions_path)