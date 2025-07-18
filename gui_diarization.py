import tkinter as tk
from moviepy.editor import VideoFileClip
import pygame
from PIL import Image, ImageTk

class HorizontalBarsApp(tk.Tk):
    def __init__(self, video_path, save_path="outputs/speaker_diarization.txt"):
        super().__init__()
        self.title("Resizable Horizontal Bars")

        pygame.mixer.init()

        self.canvas = tk.Canvas(self, width=960, height=540 + 180)
        self.canvas.pack()

        self.video_path = video_path
        self.save_path = save_path
        self.video_clip = VideoFileClip(video_path)
        self.audio = self.video_clip.audio
        self.audio.write_audiofile("audio.wav")
        pygame.mixer.music.load("audio.wav")
        self.total_frames = int(self.video_clip.duration * self.video_clip.fps)
        self.time = 0
        self.start_time = 0
        self.playing = False

        self.tracks = []
        self.intervals = {}
        self.scrubers = []
        self.selected_interval = None  # Track currently selected interval

        self.create_tracks()
        self.create_controls()

        self.bind("<Delete>", self.delete_interval)

        frame = self.video_clip.get_frame(0)
        frame = Image.fromarray(frame)
        frame = frame.resize((960, 540))
        frame = ImageTk.PhotoImage(frame)
        self.canvas.create_image(0, 0, image=frame, anchor="nw")
        self.canvas.image = frame

        self.diarization_results = None

    def create_tracks(self):
        for i in range(4):
            frame = tk.Frame(self.canvas)
            frame.place(x=0, y=560 + i * 40, width=960, height=20)
            canvas = tk.Canvas(frame, width=800, bg="white", highlightthickness=0)
            add_button = tk.Button(frame, text="Add Interval", command=lambda idx=i: self.add_interval(idx))
            add_button.pack(side="right", padx=20)
            canvas.pack(side="left", padx=(20,0))
            scrub = canvas.create_rectangle(2, 2, 5, 18, fill="red")
            self.scrubers.append(scrub)
            self.tracks.append(canvas)
            self.intervals[canvas] = []

    def make_draggable(self, track, interval):
        track.tag_bind(interval, "<ButtonPress-1>", self.on_press)
        track.tag_bind(interval, "<B1-Motion>", self.on_drag)
        track.tag_bind(interval, "<ButtonRelease-1>", self.on_release)

    def on_press(self, event):
        interval = event.widget.find_closest(event.x, event.y)[0]
        for track, track_intervals in self.intervals.items():
            for track_interval in track_intervals:
                track.itemconfig(track_interval, fill="blue")

        event.widget.itemconfig(interval, fill="green")

        self.selected_interval = (event.widget, interval)

        self._drag_data = {
            "x": event.x,
            "item": interval,
            "side": None
        }

        coords = event.widget.coords(interval)
        if event.x - coords[0] < 10:  # If mouse is close to left side
            self._drag_data["side"] = "left"
        elif coords[2] - event.x < 10:  # If mouse is close to right side
            self._drag_data["side"] = "right"

    def on_drag(self, event):
        dx = event.x - self._drag_data["x"]
        item = self._drag_data["item"]
        side = self._drag_data["side"]
        coords = event.widget.coords(item)
        self._drag_data["x"] = event.x

        if side == "left":
            for i, track in enumerate(self.tracks):
                for intervals in self.intervals[track]:
                    if track != event.widget:
                        coords2 = track.coords(intervals)
                        if abs(coords[0] + dx - coords2[2]) < 10 and abs(event.x - coords[0]) < 10:
                            dx = coords2[2] - coords[0]
            new_width = coords[2] - coords[0] - dx
            if new_width >= 10:  # Minimum width
                event.widget.coords(item, coords[0] + dx, coords[1], coords[2], coords[3])
        elif side == "right":
            for i, track in enumerate(self.tracks):
                for intervals in self.intervals[track]:
                    if track != event.widget:
                        coords2 = track.coords(intervals)
                        if abs(coords[2] + dx - coords2[0]) < 10 and abs(event.x - coords[2]) < 10:
                            dx = coords2[0] - coords[2]
            new_width = coords[2] - coords[0] + dx
            if new_width >= 10:  # Minimum width
                event.widget.coords(item, coords[0], coords[1], coords[2] + dx, coords[3])

        self.update()
        self.update_idletasks()

    def on_release(self, event):
        self._drag_data = {}

    def add_interval(self, idx):
        track = self.tracks[idx]
        
        #current pos to end of the video
        x = self.time / self.video_clip.duration * 800
        interval = track.create_rectangle(x, 2, x+90, 18, fill="blue")
        self.make_draggable(track, interval)
        track.tag_lower(interval)
        self.intervals[track].append(interval)

    def delete_interval(self, event):
        if self.selected_interval is not None:
            track = self.selected_interval[0]
            track.delete(self.selected_interval[1])
            self.intervals[track].remove(self.selected_interval[1])
            self.selected_interval = None  # Reset selected interval after deletion

    def create_controls(self):
        scale_frame = tk.Frame(self)
        scale_frame.pack(side="top", fill="x")
        self.scale = tk.Scale(scale_frame, length=800, from_=0, to=self.total_frames-1, orient=tk.HORIZONTAL, command=self.set_frame)
        self.scale.pack(side="left", padx=20)
        button_frame = tk.Frame(self)
        button_frame.pack(side="bottom", padx=20)
        self.play_button = tk.Button(button_frame, text="Play", command=self.toggle_play)
        self.play_button.pack(side="left", pady=10)
        self.save_button = tk.Button(button_frame, text="Save", command=self.save)
        self.save_button.pack(side="left", pady=10)
        tk.Button(button_frame, text="Quit", command=self.on_quit).pack(side="left", pady=10)

    def save(self):
        with open(self.save_path, "w") as f:
            for i, track in enumerate(self.tracks):
                intervals = self.intervals[track]
                for interval in intervals:
                    coords = track.coords(interval)
                    start_time = coords[0] / 800 * self.video_clip.duration
                    end_time = coords[2] / 800 * self.video_clip.duration
                    f.write(f"{start_time} {end_time} {i}\n")
            print(f"Saved speaker diarization to {self.save_path}")

    def set_frame(self, frame_num):
        self.time = int(frame_num) / self.video_clip.fps
        self.current_frame = int(frame_num)
        frame = self.video_clip.get_frame(self.time)
        frame = Image.fromarray(frame)
        frame = frame.resize((960, 540))
        frame = ImageTk.PhotoImage(frame)
        self.canvas.create_image(0, 0, image=frame, anchor="nw")
        self.canvas.image = frame
        self.scale.set(frame_num)

        for i, track in enumerate(self.tracks):
            track.coords(self.scrubers[i], 2 + self.current_frame / self.total_frames * 800, 2, 5 + self.current_frame / self.total_frames * 800, 18)


    def toggle_play(self):
        if not self.playing:
            self.play_button.config(text="Pause")
            self.playing = True
            # Play audio at the corresponding time
            pygame.mixer.music.load("audio.wav")
            pygame.mixer.music.play(start=self.time)
            self.start_time = self.time
        else:
            self.playing = False
            self.play_button.config(text="Play")
            pygame.mixer.music.pause()
            self.time = self.start_time + pygame.mixer.music.get_pos() / 1000

    def update_video(self):
        if self.playing:
            self.time = self.start_time + pygame.mixer.music.get_pos() / 1000
            current_frame = int(self.time * self.video_clip.fps)
            self.set_frame(current_frame)

    def get_diarization_results(self):
        diarization_results = []
        for i, track in enumerate(self.tracks):
            intervals = self.intervals[track]
            for interval in intervals:
                coords = track.coords(interval)
                start_time = coords[0] / 800 * self.video_clip.duration
                end_time = coords[2] / 800 * self.video_clip.duration
                diarization_results.append((start_time, end_time, i))
        return diarization_results
    
    def on_quit(self):
        """Collect the intervals while widgets still exist, then leave the loop."""
        self.diarization_results = self.get_diarization_results()
        self.save()
        pygame.mixer.music.stop()          # be nice, stop the sound
        self.video_clip.close()
        self.quit()                        # ends mainloop but keeps objects alive

def run_app(input_video_path):
    app = HorizontalBarsApp(input_video_path)
    app.after(100, app.update_video)
    app.mainloop()                # blocks until user clicks “Quit” (on_quit)
    results = app.diarization_results
    app.destroy()                 # now it’s safe to destroy widgets
    return results

if __name__ == "__main__":
    diar = run_app("input.mp4")
    for start, end, track in diar:
        print(f"Track {track}: {start:6.2f} s – {end:6.2f} s")
