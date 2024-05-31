import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2
from mediapipe import solutions
import cv2
import numpy as np
from moviepy.editor import VideoFileClip
from pyAudioAnalysis import audioBasicIO as aIO
from pyAudioAnalysis import audioSegmentation as aS

model_path = 'face_landmarker.task'

BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

def draw_landmarks_on_image(rgb_image, detection_result):
  face_landmarks_list = detection_result.face_landmarks
  annotated_image = np.copy(rgb_image)

  # Loop through the detected faces to visualize\.
  for idx in range(len(face_landmarks_list)):
    face_landmarks = face_landmarks_list[idx]

    # Draw the face landmarks.
    face_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
    face_landmarks_proto.landmark.extend([
      landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in face_landmarks
    ])

    solutions.drawing_utils.draw_landmarks(
        image=annotated_image,
        landmark_list=face_landmarks_proto,
        connections=mp.solutions.face_mesh.FACEMESH_CONTOURS,
        landmark_drawing_spec=None,
        connection_drawing_spec=mp.solutions.drawing_styles
        .get_default_face_mesh_contours_style())
    solutions.drawing_utils.draw_landmarks(
        image=annotated_image,
        landmark_list=face_landmarks_proto,
        connections=mp.solutions.face_mesh.FACEMESH_IRISES,
          landmark_drawing_spec=None,
          connection_drawing_spec=mp.solutions.drawing_styles
          .get_default_face_mesh_iris_connections_style())

  return annotated_image

regions_set = False
regions = []
intervals = []
last_speakers = 0
single_intervals = []
speaker_start = 0

def load_speaker_diarization():
    with open("speaker_diarization.txt", "r") as f:
        for line in f:
            start, end, speaker = map(float, line.split())
            if end - start < 1:
                continue
            intervals.append((start, end, int(speaker)))

def get_current_speakers(timestamp):
    speakers = []
    for interval in intervals:
        if interval[0] <= timestamp <= interval[1]:
            speakers.append(interval[2])

    return speakers

def load_regions():
    with open("regions.txt", "r") as f:
        for line in f:
            num_boxes = int(line)
            boxes = []
            for _ in range(num_boxes):
                x1, y1, x2, y2 = map(int, f.readline().split())
                boxes.append((x1, y1, x2, y2))
            regions.append(boxes)

def within_region(region, x, y):
    x1, y1, x2, y2 = region
    return x1 <= x <= x2 and y1 <= y <= y2

def get_person_index(x, y):
    for i, region in enumerate(regions):
        if within_region(region[0], x, y):
            return i

    return -1


def fit_speakers(concat_frame, frame, speakers, speaking, regions):
    speakers_added = 0
    for i, region in enumerate(regions):
        if speaking[i]:
            if speakers == 1:
                crop_size = region[2]
                crop = frame[crop_size[1]:crop_size[3], crop_size[0]:crop_size[2]]
                crop = cv2.resize(crop, (1080, 1920))
                concat_frame = crop
            elif speakers == 2 and speakers_added == 0:
                crop_size = region[1]
                crop = frame[crop_size[1]:crop_size[3], crop_size[0]:crop_size[2]]
                crop = cv2.resize(crop, (1080, 960))
                concat_frame[:960, :1080] = crop
            elif speakers == 2 and speakers_added == 1:
                crop_size = region[1]
                crop = frame[crop_size[1]:crop_size[3], crop_size[0]:crop_size[2]]
                crop = cv2.resize(crop, (1080, 960))
                concat_frame[960:, :1080] = crop
            elif speakers == 3 and speakers_added == 0:
                crop_size = region[1]
                crop = frame[crop_size[1]:crop_size[3], crop_size[0]:crop_size[2]]
                crop = cv2.resize(crop, (1080, 960))
                concat_frame[:960, :1080] = crop
            elif speakers == 3 and speakers_added == 1:
                crop_size = region[0]
                crop = frame[crop_size[1]:crop_size[3], crop_size[0]:crop_size[2]]
                crop = cv2.resize(crop, (540, 960))
                concat_frame[960:, :540] = crop
            elif speakers == 3 and speakers_added == 2:
                crop_size = region[0]
                crop = frame[crop_size[1]:crop_size[3], crop_size[0]:crop_size[2]]
                crop = cv2.resize(crop, (540, 960))
                concat_frame[960:, 540:] = crop
            elif speakers == 4 and speakers_added == 0:
                crop_size = region[0]
                crop = frame[crop_size[1]:crop_size[3], crop_size[0]:crop_size[2]]
                crop = cv2.resize(crop, (540, 960))
                concat_frame[:960, :540] = crop
            elif speakers == 4 and speakers_added == 1:
                crop_size = region[0]
                crop = frame[crop_size[1]:crop_size[3], crop_size[0]:crop_size[2]]
                crop = cv2.resize(crop, (540, 960))
                concat_frame[:960, 540:] = crop
            elif speakers == 4 and speakers_added == 2:
                crop_size = region[0]
                crop = frame[crop_size[1]:crop_size[3], crop_size[0]:crop_size[2]]
                crop = cv2.resize(crop, (540, 960))
                concat_frame[960:, :540] = crop
            elif speakers == 4 and speakers_added == 3:
                crop_size = region[0]
                crop = frame[crop_size[1]:crop_size[3], crop_size[0]:crop_size[2]]
                crop = cv2.resize(crop, (540, 960))
                concat_frame[960:, 540:] = crop

            speakers_added += 1

    return concat_frame

net = cv2.dnn.readNetFromCaffe("deploy.prototxt", "res10_300x300_ssd_iter_140000_fp16.caffemodel")

options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.IMAGE,
    min_face_detection_confidence=0.1,
    num_faces=1,
    min_face_presence_confidence=0.5,
    min_tracking_confidence=0.5,
)

video_file = "input.mp4"

# Extract audio from video
video = VideoFileClip(video_file)
audio = video.audio
audio.write_audiofile("audio.wav")

load_regions()
load_speaker_diarization()

output_video = cv2.VideoWriter("output.mp4", cv2.VideoWriter_fourcc(*'mp4v'), 30, (1080, 1920))

with FaceLandmarker.create_from_options(options) as landmarker:
    cap = cv2.VideoCapture(video_file)

    speaking = [False] * len(regions)
    mouth_open = [False] * len(regions)
    closed_frames = [0] * len(regions)
    open_frames = [0] * len(regions)
    speakers = 0
    concat_frame = np.zeros((1920, 1080, 3), np.uint8)
    timestamp = 0
    speaker_start = 0

    while True:

        # Read frame from video
        ret, frame = cap.read()
        if not ret:
            break

        timestamp = cap.get(cv2.CAP_PROP_POS_MSEC)
        frame_number = cap.get(cv2.CAP_PROP_POS_FRAMES)

        if frame_number % 1 == 0:
            # Use DNN for face detection
            """h, w = frame.shape[:2]
            blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
            net.setInput(blob)
            detections = net.forward()

            for i in range(detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                if confidence > 0.2:
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (startX, startY, endX, endY) = box.astype("int")
                    person = get_person_index(startX, endY)
                    if person == -1:
                        continue

                    crop = frame[startY:endY, startX:endX]
                    crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.array(crop))
                    face_landmarker_result = landmarker.detect(mp_image)
                    crop = cv2.cvtColor(crop, cv2.COLOR_RGB2BGR)
                    crop = draw_landmarks_on_image(crop, face_landmarker_result)
                    frame[startY:endY, startX:endX] = crop

                    if len(face_landmarker_result.face_landmarks) == 0:
                        continue

                    mouth_distance = face_landmarker_result.face_landmarks[0][14].y - face_landmarker_result.face_landmarks[0][13].y

                    if mouth_distance > 0.02:
                        open_frames[person] += 1
                        closed_frames[person] = 0
                    else:
                        closed_frames[person] += 1
                        open_frames[person] = 0

                    if closed_frames[person] > 5:
                        mouth_open[person] = False
                        cv2.rectangle(frame, (startX, startY), (endX, endY), (255, 0, 0), 2)
                    elif open_frames[person] > 5:
                        mouth_open[person] = True
                        cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)"""

        index_speaking = get_current_speakers(timestamp / 1000)

        if len(index_speaking) != 0:
            speakers = len(index_speaking)
            if speakers != last_speakers:
                if speakers == 1:
                    single_intervals.append((speaker_start, timestamp))
                last_speakers = speakers
                speaker_start = timestamp
            last_speakers = speakers
            speaking = [False] * len(regions)
            for i in index_speaking:
                speaking[i] = True
            """for i in range(len(mouth_open)):
                if mouth_open[i]:
                    speaking[i] = True
                    speakers += 1"""
        

        concat_frame = fit_speakers(concat_frame, frame, speakers, speaking, regions)

        output_video.write(concat_frame)
        cv2.imshow("Frame", concat_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        #rounded and correct progress bar
        print(f"Progress: {round((cap.get(cv2.CAP_PROP_POS_FRAMES) / cap.get(cv2.CAP_PROP_FRAME_COUNT)) * 100, 2)}%", end="\r")

    if speakers > 1:
        single_intervals.append((speaker_start, timestamp))

    cap.release()
    output_video.release()
    cv2.destroyAllWindows()


#add audio to video
video = VideoFileClip("output.mp4")
video = video.set_audio(audio)
video.write_videofile("output_audio.mp4")

with open("single_intervals.txt", "w") as f:
    for interval in single_intervals:
        f.write(f"{interval[0]} {interval[1]}\n")
