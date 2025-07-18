import whisperx
import moviepy.editor as mpy
from moviepy.editor import CompositeVideoClip, TextClip
import numpy as np
import pygame

def generate_captions(input_video_path, audio_output_path, video_output_path="output_with_captions.mp4"):

    device = "cpu"
    batch_size = 16
    compute_type = "int8"  # or "float16" or "int8_float16"
    enable_transcript_correction = False
    whisper_model = "large-v2"

    # Load original video
    input_video = mpy.VideoFileClip(input_video_path)
    frame_size = input_video.size

    # Load original audio
    input_audio = input_video.audio
    input_audio.write_audiofile(audio_output_path)

    # Transcribe with whisper
    model = whisperx.load_model(whisper_model, device, compute_type=compute_type)
    audio = whisperx.load_audio(audio_output_path)
    result = model.transcribe(audio, batch_size=batch_size)

    # Allow user to correct the transcript with UI (optional)
    if enable_transcript_correction:
        import PySimpleGUI as sg
        layout = [[sg.Multiline('\n'.join(segment['text'] for segment in result['segments']), size=(60, 20), key='transcript')],
                  [sg.OK(), sg.Cancel()]]
        window = sg.Window('Transcript Correction', layout)

        while True:
            event, values = window.read()
            if event in (sg.WINDOW_CLOSED, 'Cancel'):
                break
            elif event == 'OK':
                corrected_transcript = values['transcript'].split('\n')
                for i, segment in enumerate(result['segments']):
                    segment['text'] = corrected_transcript[i]
                break

        window.close()

    # Align whisper output
    model_a, metadata = whisperx.load_align_model(language_code=result['language'], device=device)
    result = whisperx.align(result['segments'], model_a, metadata, audio, device, return_char_alignments=False)

    def inside_center_intervals(time, centered_intervals):
        for interval in centered_intervals:
            if interval[0] <= time and interval[1] >= time:
                return True
        return False

    centered_intervals = []

    def create_caption(textJSON, framesize, font="Arial-Black", color='white', highlight_color='rgb(0,255,0)', stroke_color='black', stroke_width=3):
        word_clips = []
        line_clips = []
        xy_textclips_positions = []

        frame_width = framesize[0]
        frame_height = framesize[1]
        x_pos = 0
        line_width = 0
        y_pos = frame_height * 2 / 3

        x_buffer = frame_width * 1 / 5

        max_line_width = frame_width - 2 * (x_buffer)

        fontsize = int(frame_width * 0.06)
        last_start = textJSON['words'][0]['start']

        for index, wordJSON in enumerate(textJSON['words']):
            if 'end' not in wordJSON:
                wordJSON['start'] = xy_textclips_positions[-1]['end']
                wordJSON['end'] = xy_textclips_positions[-1]['end']

            if inside_center_intervals(last_start, centered_intervals):
                y_pos = frame_height * 1 / 2 - 50
            else:
                y_pos = frame_height * 2 / 3

            duration = wordJSON['end'] - wordJSON['start']
            word_clip = TextClip(wordJSON['word'], font=font, fontsize=fontsize, color=color, stroke_color=stroke_color, stroke_width=stroke_width).set_start(last_start).set_duration(duration)
            word_clip_space = TextClip(" ", font=font, fontsize=fontsize, color=color).set_start(last_start).set_duration(duration)
            word_width, word_height = word_clip.size
            space_width, space_height = word_clip_space.size
            if line_width + word_width + space_width <= max_line_width:
                xy_textclips_positions.append({
                    "x_pos": x_pos,
                    "y_pos": y_pos,
                    "width": word_width,
                    "height": word_height,
                    "word": wordJSON['word'],
                    "start": wordJSON['start'],
                    "end": wordJSON['end'],
                    "duration": duration
                })

                word_clip = word_clip.set_position((x_pos, y_pos))
                word_clip_space = word_clip_space.set_position((x_pos + word_width, y_pos))
                line_clips.append((index, word_clip))

                x_pos = x_pos + word_width + space_width
                line_width = line_width + word_width + space_width
            else:
                if len(line_clips) > 0:
                    x_pos = (frame_width - line_width) / 2

                    for i, clip in line_clips:
                        clip = clip.set_position((x_pos + xy_textclips_positions[i]["x_pos"], y_pos)).set_duration(wordJSON['start'] - last_start)
                        xy_textclips_positions[i]["x_pos"] = x_pos + xy_textclips_positions[i]["x_pos"]
                        word_clips.append(clip)
                    line_clips = []

                x_pos = 0

                last_start = wordJSON['start']

                if inside_center_intervals(last_start, centered_intervals):
                    y_pos = frame_height * 1 / 2 - 50
                else:
                    y_pos = frame_height * 2 / 3

                xy_textclips_positions.append({
                    "x_pos": x_pos,
                    "y_pos": y_pos,
                    "width": word_width,
                    "height": word_height,
                    "word": wordJSON['word'],
                    "start": wordJSON['start'],
                    "end": wordJSON['end'],
                    "duration": duration
                })

                word_clip = word_clip.set_start(last_start).set_position((x_pos, y_pos))
                word_clip_space = word_clip_space.set_start(last_start).set_position((x_pos + word_width, y_pos))
                line_clips.append((index, word_clip))

                line_width = word_width + space_width
                x_pos = word_width + space_width

        if len(line_clips) > 0:
            x_pos = (frame_width - line_width) / 2
            end = xy_textclips_positions[-1]["end"]
            for i, clip in line_clips:
                clip = clip.set_position((x_pos + xy_textclips_positions[i]["x_pos"], y_pos)).set_duration(end - last_start)
                xy_textclips_positions[i]["x_pos"] = x_pos + xy_textclips_positions[i]["x_pos"]
                word_clips.append(clip)
            line_clips = []

        for highlight_word in xy_textclips_positions:
            word_clip_highlight = TextClip(highlight_word['word'], fontsize=fontsize, font=font, color=highlight_color, stroke_color=stroke_color, stroke_width=stroke_width).set_start(highlight_word['start']).set_duration(highlight_word['duration'])
            word_clip_highlight = word_clip_highlight.set_position((highlight_word['x_pos'], highlight_word['y_pos']))
            word_clips.append(word_clip_highlight)

        return word_clips, xy_textclips_positions

    all_linelevel_splits = []
    linelevel_subtitles = result['segments']

    for line in linelevel_subtitles:
        out_clips, positions = create_caption(line, frame_size)
        all_linelevel_splits.extend(out_clips)

    final_video = CompositeVideoClip([input_video] + all_linelevel_splits)
    final_video = final_video.set_audio(input_audio)

    # Instead of writing to file here, return the final video clip object
    final_video.write_videofile(video_output_path)

    return final_video

import os
if __name__ == "__main__":
    input_video_path = "output.mp4"  # Example input video
    audio_output_path = "audio.wav"  # Example audio output to be written
    output_video_path = "output_with_captions.mp4"

    if os.path.exists(audio_output_path):
        try:
            os.remove(audio_output_path)          # remove old copy
        except PermissionError:
            pygame.mixer.quit()          # drop any lock held by the previous run
            os.remove(audio_output_path)

    final_video = generate_captions(input_video_path, audio_output_path, output_video_path)
    print(f"Captioned video saved to '{output_video_path}")