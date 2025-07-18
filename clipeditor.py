import mediapipe as mp
import cv2
import numpy as np
from moviepy.editor import VideoFileClip


def edit_clips(video_file, selected_regions, diarization_results, output_path="outputs/output.mp4", visualize=False):
    model_path = "models/face_landmarker.task"

    BaseOptions = mp.tasks.BaseOptions
    FaceLandmarker = mp.tasks.vision.FaceLandmarker
    FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    regions = selected_regions
    intervals = diarization_results

    options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=VisionRunningMode.IMAGE,
        min_face_detection_confidence=0.1,
        num_faces=1,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    video = VideoFileClip(video_file)
    audio = video.audio

    output_video = cv2.VideoWriter("outputs/output_noaudio.mp4", cv2.VideoWriter_fourcc(*'mp4v'), 30, (1080, 1920))

    with FaceLandmarker.create_from_options(options) as landmarker:
        cap = cv2.VideoCapture(video_file)

        speaking = [False] * len(regions)
        last_speakers = 0
        single_intervals = []
        speaker_start = 0
        concat_frame = np.zeros((1920, 1080, 3), np.uint8)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            timestamp = cap.get(cv2.CAP_PROP_POS_MSEC)
            frame_number = cap.get(cv2.CAP_PROP_POS_FRAMES)

            index_speaking = []
            for interval in intervals:
                if interval[0] * 1000 <= timestamp <= interval[1] * 1000:
                    index_speaking.append(interval[2])

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
                    if i < len(speaking):
                        speaking[i] = True

            concat_frame = fit_speakers(concat_frame, frame, speakers, speaking, regions)

            output_video.write(concat_frame)
            # For modularization, do not show GUI
            if visualize:
                cv2.imshow("Frame", concat_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            print(f"Progress: {round((cap.get(cv2.CAP_PROP_POS_FRAMES) / cap.get(cv2.CAP_PROP_FRAME_COUNT)) * 100, 2)}%", end="\r")

        if last_speakers > 1:
            single_intervals.append((speaker_start, timestamp))

        cap.release()
        output_video.release()
        cv2.destroyAllWindows()

    # Add audio to video
    video = VideoFileClip("outputs/output_noaudio.mp4")
    video = video.set_audio(audio)
    video.write_videofile(output_path)

    return output_path

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

def load_speaker_diarization(diarization_path):
    intervals = []
    with open(diarization_path, "r") as f:
        for line in f:
            start, end, speaker = map(float, line.split())
            if end - start < 1:
                continue
            intervals.append((start, end, int(speaker)))
    return intervals

def load_regions(regions_path):
    regions = []
    with open(regions_path, "r") as f:
        for line in f:
            num_boxes = int(line)
            boxes = []
            for _ in range(num_boxes):
                x1, y1, x2, y2 = map(int, f.readline().split())
                boxes.append((x1, y1, x2, y2))
            regions.append(boxes)
    return regions

if __name__ == "__main__":
    selected_regions = load_regions("outputs/regions.txt")
    diarization_results = load_speaker_diarization("outputs/speaker_diarization.txt") #saved automatically by gui_diarization
    output_path = edit_clips("input.mp4", selected_regions, diarization_results, visualize=True)
    print(f"Edited video saved at: {output_path}")