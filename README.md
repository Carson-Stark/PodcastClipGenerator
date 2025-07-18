# Podcast Clips Pipeline

![clips](https://github.com/user-attachments/assets/9b5aa818-7fd1-48b7-aef6-f3f66a50f949)

## Overview
This repository provides a comprehensive pipeline for processing podcast clips. It includes transcription, face detection, region selection, speaker diarization, caption generation, and video editing. The pipeline is modular, allowing users to run individual scripts or the entire pipeline.

- **Original Project Date:** April 2024

## Features
- Automatic transcription using OpenAI's Whisper model
- Speaker diarization with face tracking and manual override tools
- Region detection using Mediapipe + OpenCV
- Configurable GUI for customizing which subject is in focus at what time
- Intelligent word-aligned caption generation and formatting
- Exports final vertical videos with dynamic face framing and styled subtitles

## Example Output Video

Input was the raw landscape video.

https://github.com/user-attachments/assets/6d3e1fc5-b69f-4b3f-b904-8a502d1db476

## Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/Carson-Stark/PodcastClipGenerator
   cd PodcastClips
   ```

2. Create virtual environment

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Download YOLOv3 files

   1. **cfg**  
      `wget https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3.cfg`

   2. **weights**  
      `wget https://pjreddie.com/media/files/yolov3.weights`

   3. **COCO labels**  
      `wget https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names`

   Place these in the `models/` folder
   ```
   └──models/
      ├── yolov3.cfg
      ├── yolov3.weights
      └── coco.names
   ```

   
## Usage
Run the main pipeline:
```bash
python main_pipeline.py
```

It may be easier to run each script individually:

If you don't want to configure the file names, name your input video `input.mp4` and place it into the project folder. Outputs will be saved in the `outputs/` folder by default, but you can configure this inside each script.

1. Get person regions
   ```
   python get_regions.py
   ```
   - **Input:** input.mp4 (video file).
   - **Output:** regions.txt (detected regions).

2. Create diarization

   Use the gui tool to set the intervals where you want each person in focus. The top bar corresponds to the leftmost person.
   ```
   python gui_diarization.py
   ```
   - **Input:** input.mp4 (video file).
   - **Output:** Speaker intervals saved in `speaker_diarization.txt`.

4. Run the clip editor
   ```
   python clipeditor.py
   ```
   - **Input:** `regions.txt` and `speaker_diarization.txt`.
   - **Output:** `output.mp4` (edited video with audio).

5. Add captions
   ```
   python captions.py
   ```
   - **Input:** input.mp4 (video file).
   - **Output:** output_with_captions.mp4 (captioned video).


## Modules
- `captions.py`: Handles transcription and caption generation.
- `clipeditor.py`: Edits video clips based on regions and speaker intervals.
- `facedetector.py`:  Detects faces in video using Mediapipe.
- `get_regions.py`: Detects regions of interest using YOLO.
- `gui_diarization.py`: Provides a GUI for speaker diarization.
- `select_regions.py`: Allows manual region selection.
- `speaker_diarization.py`: Maps speakers to intervals.
