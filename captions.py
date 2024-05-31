import whisperx
import gc 
import moviepy.editor as mpy
from moviepy.editor import *
import json
import numpy as np

device = "cpu" 
audio_file = "audio.wav"
video_file = "output_audio.mp4"
batch_size = 16 # reduce if low on GPU mem
compute_type = "int8" # change to "int8" if low on GPU mem (may reduce accuracy)

centered_intervals = []
with open("single_intervals.txt", "r") as f:
    for line in f:
        start, end = line.split()
        centered_intervals.append((float(start) / 1000, float(end) / 1000))

# Load original video
input_video = mpy.VideoFileClip(video_file)
frame_size = input_video.size

# Load original audio
input_audio = input_video.audio
input_audio.write_audiofile(audio_file)

# 1. Transcribe with original whisper (batched)
model = whisperx.load_model("large-v2", device, compute_type=compute_type)

# save model to local path (optional)
# model_dir = "/path/"
# model = whisperx.load_model("large-v2", device, compute_type=compute_type, download_root=model_dir)

audio = whisperx.load_audio(audio_file)
result = model.transcribe(audio, batch_size=batch_size)
print(result)

#allow user to correct the transcript with UI

"""import PySimpleGUI as sg

# Create a window with a multiline input for the transcript and OK/Cancel buttons
layout = [[sg.Multiline('\n'.join(segment['text'] for segment in result['segments']), size=(60, 20), key='transcript')],
          [sg.OK(), sg.Cancel()]]
window = sg.Window('Transcript Correction', layout)

# Event loop
while True:
    event, values = window.read()
    if event in (sg.WINDOW_CLOSED, 'Cancel'):
        break
    elif event == 'OK':
        # Update the transcript with the corrected text
        corrected_transcript = values['transcript'].split('\n')
        for i, segment in enumerate(result['segments']):
            segment['text'] = corrected_transcript[i]
        break

window.close()"""

# 2. Align whisper output
model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)

print(result["segments"]) # after alignment

def inside_center_intervals(time):
    for interval in centered_intervals:
        if interval[0] <= time and interval[1] >= time:
            return True
    return False

def add_shadow(txt_clip, opacity=0.7, offset=(4, 4), txt_color='white'):
    shadow = txt_clip.set_color("black").set_opacity(.75).set_position(offset).resize(1.1)
    return CompositeVideoClip([shadow, txt_clip])

def create_caption(textJSON, framesize,font = "Arial-Black",color='white', highlight_color='rgb(0,255,0)',stroke_color='black',stroke_width=3):

    word_clips = []
    line_clips = []
    xy_textclips_positions = []

    frame_width = framesize[0]
    frame_height = framesize[1]
    x_pos = 0
    line_width = 0  # Total width of words in the current line
    y_pos = frame_height*2/3

    x_buffer = frame_width*1/5

    max_line_width = frame_width - 2 * (x_buffer)

    fontsize = int(frame_width * 0.06) #6 percent of video width
    last_start = textJSON['words'][0]['start']

    for index, wordJSON in enumerate(textJSON['words']):
        if 'end' not in wordJSON:
            wordJSON['start'] = xy_textclips_positions[-1]['end']
            wordJSON['end'] = xy_textclips_positions[-1]['end']

        if inside_center_intervals(last_start):
            y_pos = frame_height*1/2 - 50
        else:
            y_pos = frame_height*2/3
        
        duration = wordJSON['end']-wordJSON['start']
        word_clip = TextClip(wordJSON['word'],font=font,fontsize=fontsize, color=color,stroke_color=stroke_color,stroke_width=stroke_width).set_start(last_start).set_duration(duration)
        word_clip_space = TextClip(" ",font=font,fontsize=fontsize, color=color).set_start(last_start).set_duration(duration)
        word_width, word_height = word_clip.size
        space_width, space_height = word_clip_space.size
        if line_width + word_width + space_width <= max_line_width:
            # Store info of each word_clip created
            xy_textclips_positions.append({
                "x_pos":x_pos,
                "y_pos": y_pos,
                "width" : word_width,
                "height" : word_height,
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
            #total clip
            if len(line_clips) > 0:
                x_pos = (frame_width - line_width) / 2

                for i, clip in line_clips:
                    clip = clip.set_position((x_pos + xy_textclips_positions[i]["x_pos"], y_pos)).set_duration(wordJSON['start'] - last_start)
                    xy_textclips_positions[i]["x_pos"] = x_pos + xy_textclips_positions[i]["x_pos"]
                    word_clips.append(clip)
                #line_clip = CompositeVideoClip(line_clips)
                #line_clip.set_position("center")
                line_clips = []

            # Move to the next line
            x_pos = 0

            last_start = wordJSON['start']

            if inside_center_intervals(last_start):
                y_pos = frame_height*1/2 - 50
            else:
                y_pos = frame_height*2/3

            # Store info of each word_clip created
            xy_textclips_positions.append({
                "x_pos":x_pos,
                "y_pos": y_pos,
                "width" : word_width,
                "height" : word_height,
                "word": wordJSON['word'],
                "start": wordJSON['start'],
                "end": wordJSON['end'],
                "duration": duration
            })

            word_clip = word_clip.set_start(last_start).set_position((x_pos, y_pos))
            word_clip_space = word_clip_space.set_start(last_start).set_position((x_pos+ word_width , y_pos))
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
        #line_clip = CompositeVideoClip(line_clips)
        #line_clip.set_position("center")
        #word_clips.extend(line_clips)
        line_clips = []


    for highlight_word in xy_textclips_positions:

        word_clip_highlight = TextClip(highlight_word['word'], fontsize=fontsize, font=font, color=highlight_color,stroke_color=stroke_color,stroke_width=stroke_width).set_start(highlight_word['start']).set_duration(highlight_word['duration'])
        word_clip_highlight = word_clip_highlight.set_position((highlight_word['x_pos'], highlight_word['y_pos']))
        #word_clip_highlight = add_shadow(word_clip_highlight)
        word_clips.append(word_clip_highlight)

    return word_clips, xy_textclips_positions

all_linelevel_splits=[]
linelevel_subtitles = result['segments']

for line in linelevel_subtitles:
  out_clips, positions = create_caption(line, frame_size)

  all_linelevel_splits.extend(out_clips)

input_video_duration = input_video.duration

final_video = CompositeVideoClip([input_video] + all_linelevel_splits)

# Set the audio of the final video to be the same as the input video
final_video = final_video.set_audio(input_video.audio)

# Write to file
final_video.write_videofile("captions.mp4")

