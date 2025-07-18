from pyannote.audio import Pipeline
from moviepy.editor import VideoFileClip
import torch

def perform_speaker_diarization(config, video_path):
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token=config.get("hf_auth_token", None)
    )

    # send pipeline to GPU (when available)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pipeline.to(device)

    video = VideoFileClip(video_path)

    # get audio from video
    audio = video.audio
    audio_path = "temp_audio.wav"
    audio.write_audiofile(audio_path)

    # apply pretrained pipeline
    diarization = pipeline(audio_path)
    intervals = diarization.get_timeline()

    # print the result
    longest_interval = {}
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        if speaker not in longest_interval:
            longest_interval[speaker] = turn
        else:
            if turn.duration > longest_interval[speaker].duration:
                longest_interval[speaker] = turn

    # For modularization, we avoid interactive clip preview and input
    # Instead, map speakers to IDs automatically or via config
    mapped_speakers = {speaker: speaker for speaker in longest_interval.keys()}

    # Prepare diarization results as list of tuples (start, end, speaker_id)
    diarization_results = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        diarization_results.append((turn.start, turn.end, mapped_speakers[speaker]))

    # Return diarization results
    return diarization_results
