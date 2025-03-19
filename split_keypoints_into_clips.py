import os
import os.path as osp
import ast
import numpy as np
import pickle
import pandas as pd
from tqdm import tqdm
import traceback
import warnings
import logging
import sys

# Suppress specific warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

# Configure logging to only show errors
logging.getLogger().setLevel(logging.ERROR)

def read_keypoints_file(keypoints_path):
    """Read keypoints from a pickle file."""
    try:
        with open(keypoints_path, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        print(f"Error reading keypoints file {keypoints_path}:")
        print(traceback.format_exc())
        raise

def read_video_annotations(annotation_file):
    """Read and parse the video annotation file."""
    annotations = {}
    try:
        with open(annotation_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                # Parse the tuple: ('videos/7942GT00.mp4', "7942GT00", '06348578', [306, 365])
                data = ast.literal_eval(line)
                video_path = data[0]  # e.g. 'videos/7942GT00.mp4'
                video_name = osp.basename(video_path)  # e.g. '7942GT00.mp4'
                video_id = data[1]    # e.g. "7942GT00"
                patient_id = data[2]  # e.g. '06348578'
                time_range = data[3]  # e.g. [306, 365] - seizure start and end times in seconds
                
                # Store annotation with video name as key
                annotations[video_name] = {
                    'video_id': video_id,
                    'patient_id': patient_id,
                    'time_range': time_range  # This is in seconds, not frames
                }
                
                # Also store for augmented version
                base, ext = osp.splitext(video_name)
                aug_video_name = f"{base}_aug{ext}"
                annotations[aug_video_name] = {
                    'video_id': video_id,
                    'patient_id': patient_id,
                    'time_range': time_range
                }
                
        return annotations
    except Exception as e:
        print(f"Error reading annotation file {annotation_file}:")
        print(traceback.format_exc())
        raise

def determine_clip_label(start_frame, end_frame, seizure_range, fps=30.0):
    """
    Determine if a clip contains seizure activity based on frame indices and seizure time range.
    
    Args:
        start_frame: Start frame index of the clip
        end_frame: End frame index of the clip
        seizure_range: List containing seizure start and end times in seconds
        fps: Frames per second (default: 30.0)
        
    Returns:
        1 if the clip contains seizure activity, 0 otherwise
    """
    # Calculate clip start and end times in seconds
    clip_start_time = start_frame / fps
    clip_end_time = end_frame / fps
    
    # Get seizure start and end times in seconds from annotation
    seizure_start_time = seizure_range[0]
    seizure_end_time = seizure_range[1]
    
    # Check for any overlap between clip time range and seizure time range
    is_seizure = not (clip_end_time <= seizure_start_time or clip_start_time >= seizure_end_time)
    
    return 1 if is_seizure else 0

def process_all_keypoints(keypoints_dir, output_dir, annotation_file, frames_per_clip=100):
    """Process all keypoint files, split into clips, and save with labels."""
    
    # Clean .pkl files from output directory
    if os.path.exists(output_dir):
        print(f"Cleaning .pkl files from output directory: {output_dir}")
        for file in os.listdir(output_dir):
            if file.endswith('.pkl'):
                file_path = os.path.join(output_dir, file)
                try:
                    os.unlink(file_path)
                except Exception as e:
                    print(f"Error deleting {file_path}: {e}")
        
        # Wait for user confirmation
        response = input("\nDirectory cleaned. Press Enter to continue or 'q' to quit: ")
        if response.lower() == 'q':
            print("Operation cancelled by user")
            sys.exit(0)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Read video annotations
    print("Reading video annotations...")
    annotations = read_video_annotations(annotation_file)
    
    # Get list of all keypoint files
    keypoint_files = [f for f in os.listdir(keypoints_dir) if f.endswith('.pkl')]
    print(f"Found {len(keypoint_files)} keypoint files to process")
    
    # Track statistics
    total_clips = 0
    seizure_clips = 0
    non_seizure_clips = 0
    failed_videos = []
    
    # Create a list to store clip annotations (clip_path, label)
    clip_annotations = []
    
    # Process each keypoint file
    for keypoint_file in tqdm(keypoint_files, desc="Processing keypoint files"):
        try:
            # Get video name from keypoint file
            video_name = keypoint_file.replace('.pkl', '.mp4')
            
            # Skip if annotation not found
            if video_name not in annotations:
                print(f"Warning: No annotation found for {video_name}, skipping")
                failed_videos.append((video_name, "No annotation found"))
                continue
            
            # Read keypoints
            keypoints_path = os.path.join(keypoints_dir, keypoint_file)
            pose_results = read_keypoints_file(keypoints_path)
            
            # Get video properties from annotation
            patient_id = annotations[video_name]['patient_id']
            video_id = annotations[video_name]['video_id']
            seizure_range = annotations[video_name]['time_range']  # Frame range for seizure
            
            # Calculate number of clips
            num_frames = len(pose_results)
            num_clips = num_frames // frames_per_clip
            
            print(f"Processing {video_name}: {num_frames} frames, {num_clips} clips")
            
            # Process each clip
            for clip_idx in range(num_clips):
                start_frame = clip_idx * frames_per_clip
                end_frame = start_frame + frames_per_clip
                
                # Skip if we don't have enough frames
                if end_frame > num_frames:
                    continue
                
                # Create annotation dictionary using logic from extract_poses_from_clips.ipynb
                anno = dict()
                
                # Extract frames for this clip
                clip_pose_results = pose_results[start_frame:end_frame]
                
                # Get number of frames and keypoints
                clip_num_frames = len(clip_pose_results)
                num_points = clip_pose_results[0]['keypoints'].shape[1]
                
                # Initialize arrays for keypoints and scores
                keypoints = np.zeros((1, clip_num_frames, num_points, 2), dtype=np.float32)
                scores = np.zeros((1, clip_num_frames, num_points), dtype=np.float32)
                
                # Fill arrays with data from pose_results
                for f_idx, frm_pose in enumerate(clip_pose_results):
                    keypoints[0, f_idx] = frm_pose['keypoints'][0]
                    scores[0, f_idx] = frm_pose['keypoint_scores'][0]
                
                # Set clip name
                anno['frame_dir'] = f"{patient_id}_{video_id}_Seg_{clip_idx}"
                
                # Add suffix for augmented videos to avoid filename collisions
                if '_aug.mp4' in video_name:
                    anno['frame_dir'] += "_aug"
                
                # Set image shape (using default values since we don't have direct access to frames)
                # These are common dimensions for the videos in the dataset
                anno['img_shape'] = (480, 640)  # height, width
                anno['original_shape'] = (480, 640)  # height, width
                
                # Set total frames
                anno['total_frames'] = clip_num_frames
                
                # Set keypoints and scores
                anno['keypoint'] = keypoints
                anno['keypoint_score'] = scores
                
                # Determine if this clip contains seizure
                anno['label'] = determine_clip_label(start_frame, end_frame, seizure_range)
                
                # Generate output path
                output_path = os.path.join(output_dir, f"{anno['frame_dir']}.pkl")
                
                # Update statistics
                total_clips += 1
                if anno['label'] == 1:
                    seizure_clips += 1
                else:
                    non_seizure_clips += 1
                
                # Save clip
                with open(output_path, 'wb') as f:
                    pickle.dump(anno, f)
                
                # Add to clip annotations list
                clip_annotations.append((output_path, anno['label']))
            
        except Exception as e:
            print(f"Failed to process {keypoint_file}: {str(e)}")
            failed_videos.append((keypoint_file, str(e)))
    
    # Save clip annotations to a text file
    clip_annotations_file = os.path.join('preprocessing', 'video_annotations', 'clip_annotations.txt')
    print(f"Saving clip annotations to {clip_annotations_file}")
    
    # Sort clip annotations to ensure segments are in order
    # Sort by patient ID, video ID, and then segment number
    sorted_clip_annotations = sorted(clip_annotations, key=lambda x: (
        # Extract patient_id, video_id, and segment number from the path
        os.path.basename(x[0]).split('_')[0],  # patient_id
        os.path.basename(x[0]).split('_')[1],  # video_id
        "_aug" in os.path.basename(x[0]),      # augmented videos after originals
        int(os.path.basename(x[0]).split('_Seg_')[1].split('.')[0].replace("_aug", ""))  # segment number
    ))
    
    # Remove duplicates (in case there are any)
    unique_paths = set()
    unique_clip_annotations = []
    for clip_path, label in sorted_clip_annotations:
        if clip_path not in unique_paths:
            unique_paths.add(clip_path)
            unique_clip_annotations.append((clip_path, label))
    
    with open(clip_annotations_file, 'w') as f:
        for clip_path, label in unique_clip_annotations:
            f.write(f"{clip_path} {label}\n")
    
    # Print summary
    print("\n=== Processing Summary ===")
    print(f"Total keypoint files processed: {len(keypoint_files) - len(failed_videos)}")
    print(f"Total clips generated: {total_clips}")
    print(f"Seizure clips: {seizure_clips}")
    print(f"Non-seizure clips: {non_seizure_clips}")
    print(f"Clip annotations saved to: {clip_annotations_file}")
    
    # Print failed videos
    if failed_videos:
        print("\nFailed videos:")
        for video, error in failed_videos:
            print(f"{video}: {error}")
        
        # Save failed videos to file
        with open(os.path.join(output_dir, 'failed_videos.txt'), 'w') as f:
            for video, error in failed_videos:
                f.write(f"{video}: {error}\n")

if __name__ == '__main__':
    # Configuration
    config = {
        'keypoints_dir': 'preprocessing/video_keypoints',
        'output_dir': 'preprocessing/clip_keypoints',
        'annotation_file': 'preprocessing/video_annotations/video_annotations.txt',
        'frames_per_clip': 90  # Same as in clip_and_annotate.ipynb
    }
    
    # Run processing
    process_all_keypoints(
        config['keypoints_dir'],
        config['output_dir'],
        config['annotation_file'],
        config['frames_per_clip']
    )
