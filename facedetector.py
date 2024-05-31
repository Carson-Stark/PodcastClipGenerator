import mediapipe as mp
import cv2

BaseOptions = mp.tasks.BaseOptions
FaceDetector = mp.tasks.vision.FaceDetector
FaceDetectorOptions = mp.tasks.vision.FaceDetectorOptions
VisionRunningMode = mp.tasks.vision.RunningMode

net = cv2.dnn.readNetFromCaffe("deploy.prototxt", "res10_300x300_ssd_iter_140000_fp16.caffemodel")

# Create a face detector instance with the video mode:
options = FaceDetectorOptions(
    base_options=BaseOptions(model_asset_path='/path/to/model.task'),
    running_mode=VisionRunningMode.VIDEO)
with FaceDetector.create_from_options(options) as detector:

    cap = cv2.VideoCapture("test.mp4")

    while True:

        # Read frame from video
        ret, frame = cap.read()
        if not ret:
            break

        timestamp = cap.get(cv2.CAP_PROP_POS_MSEC)

         # Use DNN for face detection
        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
        net.setInput(blob)
        detections = net.forward()

        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.2:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                if speaker == i:
                    cv2.rectangle(frame, (startX, startY), (endX, endY), (255, 0, 0), 2)
                else:
                    cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)

                crop = frame[startY:endY, startX:endX]
                crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.array(crop))
                face_landmarker_result = landmarker.detect(mp_image)
                crop = cv2.cvtColor(crop, cv2.COLOR_RGB2BGR)
                crop = draw_landmarks_on_image(crop, face_landmarker_result)
                frame[startY:endY, startX:endX] = crop
     

        cv2.imshow("Frame", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    