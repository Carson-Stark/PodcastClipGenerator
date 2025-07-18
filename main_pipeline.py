import json
from captions import generate_captions
from facedetector import detect_faces
from select_regions import select_regions
from speaker_diarization import perform_speaker_diarization
from clipeditor import edit_clips
from get_regions import get_regions, get_first_frame
from gui_diarization import run_app as run_gui_diarization
import moviepy.editor as mpy

def main():
    input_video_path = "input.mp4"
    audio_output_path = "outputs/audio.wav"

    print("Starting pipeline...")

    # Step 1: Region detection (people detection)
    print("Detecting people regions...")
    first_frame = get_first_frame(input_video_path)
    regions = get_regions(first_frame)

    # Step 2: Speaker diarization using GUI diarization
    print("Performing speaker diarization with GUI...")
    diarization_results = run_gui_diarization(input_video_path)

    # Step 3: Clip editing
    print("Editing clips...")
    edited_video_path = edit_clips(input_video_path, regions, diarization_results)

    # Step 4: Transcription and caption generation
    print("Generating captions...")
    captions_video = generate_captions(edited_video_path, audio_output_path)

    print(f"Pipeline completed. Final captioned video at {captions_video}")

if __name__ == "__main__":
    main()
