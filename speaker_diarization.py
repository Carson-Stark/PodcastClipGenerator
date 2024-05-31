import os
from pyannote.audio import Pipeline
from moviepy.editor import VideoFileClip

pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.1",
    use_auth_token="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")

# send pipeline to GPU (when available)
import torch

pipeline.to(torch.device("cuda"))

video = VideoFileClip("input.mp4")

#get audio from video
audio = video.audio
#save audio to file
audio.write_audiofile("audio.wav")

# apply pretrained pipeline
diarization = pipeline("audio.wav")
intervals = diarization.get_timeline()

# print the result
longest_interval = {}
for turn, _, speaker in diarization.itertracks(yield_label=True):
    if speaker not in longest_interval:
        longest_interval[speaker] = turn
    else:
        if turn.duration > longest_interval[speaker].duration:
            longest_interval[speaker] = turn
    
    print(f"start={turn.start:.1f}s stop={turn.end:.1f}s speaker_{speaker}")

mapped_speakers = {}
for speaker, turn in longest_interval.items():
    clip = video.subclip(turn.start, min (turn.start + 5, turn.end))
    clip.preview()
    clip.close()
    speaker_name = input("Enter speaker number: ")
    print (f"Speaker {speaker} is {speaker_name}")
    mapped_speakers[speaker] = int(speaker_name)

#save the intervals
with open("speaker_diarization.txt", "w") as f:
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        f.write(f"{turn.start} {turn.end} {mapped_speakers[speaker]}\n")

import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt

# Create a Tkinter window
root = tk.Tk()
root.title("Speaker Diarization Timeline")

# Create a Figure and set its size
fig = plt.figure(figsize=(8, 4))

# Create a subplot for the timeline
ax = fig.add_subplot(111)

# Plot the speaker diarization results on the timeline
for turn, _, speaker in diarization.itertracks(yield_label=True):
    start_time = turn.start
    end_time = turn.end
    ax.plot([start_time, end_time], [speaker, speaker], color='b', linewidth=10)

# Set labels and title
ax.set_xlabel('Time (s)')
ax.set_ylabel('Speaker')
ax.set_title('Speaker Diarization Timeline')

# Create a Tkinter canvas
canvas = FigureCanvasTkAgg(fig, master=root)
canvas_widget = canvas.get_tk_widget()
canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=1)

# Run the Tkinter event loop
root.mainloop()
