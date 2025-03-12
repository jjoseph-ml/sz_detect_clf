import os
import cv2
import ast


def process_video(video_path):
    # Compute output video path with _aug inserted before extension
    base, ext = os.path.splitext(video_path)
    output_path = base + '_aug' + ext

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Failed to open video: {video_path}")
        return

    # Retrieve properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Define codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    if not out.isOpened():
        print(f"Failed to open writer for: {output_path}")
        cap.release()
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Perform horizontal flip
        flipped_frame = cv2.flip(frame, 1)
        out.write(flipped_frame)
        
    cap.release()
    out.release()
    print(f"Augmented video saved as: {output_path}")


def main():
    # Hardcode the annotation file path - fix the typo in preprocessng
    annotation_file = "preprocessing/video_annotations/video_annotations.txt"
    if not os.path.exists(annotation_file):
        print(f"Annotation file not found: {annotation_file}")
        return

    with open(annotation_file, 'r') as f:
        lines = f.readlines()

    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            # Each line is a tuple like ('videos/7942GT00.mp4', "7942GT00", '06348578', [306, 365])
            data = ast.literal_eval(line)
            video_relative_path = data[0]  # e.g. 'videos/7942GT00.mp4'
        except Exception as e:
            print(f"Failed to parse line: {line}. Error: {e}")
            continue

        # Hardcode the video path by concatenation
        # The annotation already includes 'videos/' in the path, so we need to handle that
        if video_relative_path.startswith("videos/"):
            video_name = video_relative_path.split("/")[1]  # Extract just the filename
            video_path = "preprocessing/videos/" + video_name
        else:
            video_path = "preprocessing/" + video_relative_path
        
        if os.path.exists(video_path):
            print(f"Processing video: {video_path}")
            process_video(video_path)
        else:
            print(f"Video not found, skipping: {video_path}")


if __name__ == "__main__":
    main() 