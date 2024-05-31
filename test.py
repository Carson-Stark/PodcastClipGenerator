from moviepy.editor import *

# Function to create pop-in animation
def pop_in_animation(text, font='Arial', fontsize=50, duration=2, fps=30):
    # Create the text clip
    text_clip = TextClip(text, fontsize=fontsize, font=font, color='white')

    # Calculate scale keyframes for pop-in animation
    scale_factor = 0.2  # Initial scale factor
    keyframes = [(0, scale_factor), (duration/2, 1.2), (duration, 1)]  # Scale factors at different time points

    # Define a function to animate scale
    def scale_animation(t):
        return scale_factor + (t/duration) * (1 - scale_factor)

    # Animate the scale
    scaled_clips = [text_clip.set_duration(duration).fl(lambda gf, t: gf(t).resize(scale_animation(t)))]

    # Composite the clips
    final_clip = CompositeVideoClip(scaled_clips, size=text_clip.size)

    return final_clip.set_duration(duration).set_fps(fps)

# Create the pop-in animation
animation = pop_in_animation("Pop-in Animation")

# Preview the animation
animation.preview()