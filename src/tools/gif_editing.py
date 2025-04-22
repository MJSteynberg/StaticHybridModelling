from PIL import Image, ImageSequence

def adjust_gif_speed(in_path, out_path, cutoff1, cutoff2, cutoff3):
    """
    Adjusts the frame speeds of a GIF.
    - slow_factor: multiplier to slow down the first 'cutoff' frames.
    - fast_factor: multiplier to speed up the remaining frames.
    """
    im = Image.open(in_path)
    frames = []
    durations = []

    # Process frames one by one, adjusting duration per frame
    for i, frame in enumerate(ImageSequence.Iterator(im)):
        duration = frame.info.get('duration', 100)  # default to 100ms if not specified
        if i < cutoff1:
            new_duration = int(duration * 0.8)
        elif i < cutoff2:
            new_duration = int(duration * 0.5)
        elif i < cutoff3:
            new_duration = int(duration * 0.3)
        else:
            new_duration = int(duration * 0.1)
        
        frames.append(frame.copy())
        durations.append(new_duration)

    # Save the modified GIF with the new durations
    frames[0].save(out_path, save_all=True, append_images=frames[1:], duration=durations, loop=0)

if __name__ == "__main__":
    input_gif = r"results\experiment_baseline\experiment_baseline_new.gif"     # Update with your gif path
    output_gif = r"results\experiment_baseline\experiment_baseline_new_edited.gif"   # Output path for the modified gif
    adjust_gif_speed(input_gif, output_gif, cutoff1=200, cutoff2=500, cutoff3 = 1000)