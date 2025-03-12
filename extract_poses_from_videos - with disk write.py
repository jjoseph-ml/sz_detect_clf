import os
import os.path as osp
import ast
import numpy as np
import pandas as pd
import cv2
import mmengine
from tempfile import TemporaryDirectory
from tqdm import tqdm
import traceback
import warnings
import logging
import sys
import pickle
import torch
import torch.nn as nn
from mmpose.structures import InstanceData

# Suppress specific warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

# Configure logging to only show errors
logging.getLogger('mmengine').setLevel(logging.ERROR)
logging.getLogger('mmcv').setLevel(logging.ERROR)
logging.getLogger('mmpose').setLevel(logging.ERROR)

def get_flipped_bbox(bbox, frame_width):
    """
    Flip bbox coordinates horizontally for augmented videos.
    Args:
        bbox: numpy array with [xmin, ymin, xmax, ymax]
        frame_width: width of the video frame
    Returns:
        numpy array with flipped coordinates
    """
    flipped_bbox = bbox.copy()
    # Flip x coordinates
    flipped_bbox[0] = frame_width - bbox[2]  # new xmin = width - old xmax
    flipped_bbox[2] = frame_width - bbox[0]  # new xmax = width - old xmin
    return flipped_bbox

def get_frame_width(video_path):
    """Get frame width from video."""
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        cap.release()
        return width
    except Exception as e:
        print(f"Error getting frame width for {video_path}:")
        print(traceback.format_exc())
        raise

def get_init_boxes(frame_length, video_name, video_path, bbox_df):
    """Create a list of initial bounding boxes using the correct bbox for the video."""
    try:
        # Check if this is an augmented video
        is_augmented = '_aug' in video_name
        
        # Get the original video name (without _aug if present)
        orig_video_name = video_name.replace('_aug', '')
        
        # Get the bbox for this video from DataFrame
        bbox_row = bbox_df[bbox_df['video'] == orig_video_name]
        
        if not bbox_row.empty:
            # Extract coordinates from the DataFrame row
            bbox = np.array([[
                bbox_row['xmin'].iloc[0],
                bbox_row['ymin'].iloc[0],
                bbox_row['xmax'].iloc[0],
                bbox_row['ymax'].iloc[0]
            ]])
        else:
            # Default bbox if not found
            print(f"Warning: No bbox found for {orig_video_name}, using default")
            bbox = np.array([[251.92479, 88.36366, 554.54114, 418.58438]])
        
        # If this is an augmented video, flip the bbox
        if is_augmented:
            frame_width = get_frame_width(video_path)
            bbox[0] = get_flipped_bbox(bbox[0], frame_width)
            
        # Create the list with frame length copies
        det_results = [bbox.copy() for _ in range(frame_length)]
        return det_results
    except Exception as e:
        print(f"Error in get_init_boxes for {video_name}:")
        print(traceback.format_exc())
        raise

def frame_extract(video_path, out_dir):
    """
    Extract frames from a video file and save them to disk.
    
    Args:
        video_path: Path to the video file
        out_dir: Directory to save frames to
        
    Returns:
        tuple: (frames, frame_paths) where frames is a list of frame arrays and 
               frame_paths is a list of paths to the saved frames
    """
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        # Get total frame count for progress bar
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        print(f"Extracting frames from video ({total_frames} frames, {fps:.2f} fps)")
        
        frames = []
        frame_paths = []
        frame_idx = 0
        
        # Create output directory if needed
        if out_dir is not None:
            os.makedirs(out_dir, exist_ok=True)
        
        # Create progress bar for frame extraction
        with tqdm(total=total_frames, desc="Extracting frames") as pbar:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frames.append(frame)
                
                # Write frame to disk
                frame_filename = osp.join(out_dir, f'frame_{frame_idx:06d}.jpg')
                cv2.imwrite(frame_filename, frame)
                frame_paths.append(frame_filename)
                
                frame_idx += 1
                pbar.update(1)
        
        cap.release()
        return frames, frame_paths
    except Exception as e:
        print(f"Error extracting frames from {video_path}:")
        print(traceback.format_exc())
        raise

def pose_inference(pose_config, pose_checkpoint, frames, det_results, device):
    """Run pose inference on frames."""
    try:
        from mmaction.apis import pose_inference as mmaction_pose_inference
        from mmpose.apis import inference_topdown, init_model
        from mmpose.structures import PoseDataSample, merge_data_samples
        
        # Create a progress bar wrapper for pose inference
        print(f"Running pose inference on {len(frames)} frames")
        
        # Initialize the pose model
        if isinstance(pose_config, nn.Module):
            model = pose_config
        else:
            model = init_model(pose_config, pose_checkpoint, device)
        
        results = []
        data_samples = []
        
        # Create progress bar for pose inference
        with tqdm(total=len(frames), desc="Pose inference") as pbar:
            for i, (frame, det) in enumerate(zip(frames, det_results)):
                # Run inference directly on the frame
                pose_data_samples = inference_topdown(model, frame, det[..., :4], bbox_format='xyxy')
                pose_data_sample = merge_data_samples(pose_data_samples)
                pose_data_sample.dataset_meta = model.dataset_meta
                
                # Make fake pred_instances if needed
                if not hasattr(pose_data_sample, 'pred_instances'):
                    num_keypoints = model.dataset_meta['num_keypoints']
                    pred_instances_data = dict(
                        keypoints=np.empty(shape=(0, num_keypoints, 2)),
                        keypoints_scores=np.empty(shape=(0, 17), dtype=np.float32),
                        bboxes=np.empty(shape=(0, 4), dtype=np.float32),
                        bbox_scores=np.empty(shape=(0), dtype=np.float32))
                    pose_data_sample.pred_instances = InstanceData(**pred_instances_data)
                
                poses = pose_data_sample.pred_instances.to_dict()
                results.append(poses)
                data_samples.append(pose_data_sample)
                pbar.update(1)
        
        return results, data_samples
    except Exception as e:
        print(f"Error in pose inference:")
        print(traceback.format_exc())
        raise

def create_visualization(frames, data_samples, output_filename, output_viz_dir, pose_config):
    """
    Create visualization video with pose keypoints.
    
    Args:
        frames: List of video frames
        data_samples: Pose data samples
        output_filename: Base filename for output
        output_viz_dir: Directory to save visualization
        pose_config: Path to pose config file
    """
    try:
        from mmengine.registry import VISUALIZERS
        from mmengine.utils import track_iter_progress
        import mmcv
        
        print("Generating visualization...")
        pose_config_obj = mmengine.Config.fromfile(pose_config)
        visualizer = VISUALIZERS.build(pose_config_obj.visualizer)
        visualizer.set_dataset_meta(data_samples[0].dataset_meta)
        
        # Get dimensions from the first frame
        height, width = frames[0].shape[:2]
        
        # Initialize video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_filename = os.path.join(output_viz_dir, f"{output_filename}_viz.mp4")
        out = cv2.VideoWriter(out_filename, fourcc, 24, (width, height))
        
        # Create visualization with progress bar
        with tqdm(total=len(frames), desc="Creating visualization") as pbar:
            for d, f in zip(data_samples, frames):
                f = mmcv.imconvert(f, 'bgr', 'rgb')
                visualizer.add_datasample(
                    'result',
                    f,
                    data_sample=d,
                    draw_gt=False,
                    draw_heatmap=False,
                    draw_bbox=True,
                    show=False,
                    wait_time=0,
                    out_file=None,
                    kpt_thr=0.3)
                vis_frame = visualizer.get_image()
                
                # Convert RGB to BGR for OpenCV
                vis_frame = cv2.cvtColor(vis_frame, cv2.COLOR_RGB2BGR)
                
                # Write frame
                out.write(vis_frame)
                pbar.update(1)
        
        # Release video writer
        out.release()
        print(f'Visualization saved to {out_filename}')
        return out_filename
    except Exception as e:
        print(f"Error in create_visualization:")
        print(traceback.format_exc())
        raise

def pose_extraction(video_path, pose_config, pose_checkpoint, device, bbox_df, output_viz_dir=None):
    """Extract poses from a video."""
    try:
        print(f"\n{'='*20} Processing {osp.basename(video_path)} {'='*20}")
        
        # Get video name and output filename
        video_name = osp.basename(video_path)
        output_filename = osp.splitext(video_name)[0]
        
        # Create a temporary directory for frame extraction
        with TemporaryDirectory() as temp_dir:
            print(f"Extracting frames to temporary directory: {temp_dir}")
            
            # Extract frames to the temporary directory
            frames, frame_paths = frame_extract(video_path, out_dir=temp_dir)
            
            # Get bounding boxes using the video name and path (for frame width)
            det_results = get_init_boxes(len(frames), video_name, video_path, bbox_df)
            
            # Get pose results
            pose_results, data_samples = pose_inference(pose_config, pose_checkpoint,
                                                      frames, det_results, device)
            
            if pose_results is None:
                raise ValueError("Pose inference returned None")
            
            # Save raw pose data if output directory is provided
            if output_viz_dir is not None:
                # Create directory for raw pose data
                pose_data_dir = os.path.join(output_viz_dir, "pose_data")
                os.makedirs(pose_data_dir, exist_ok=True)
                
                # Save pose results
                pose_data_file = os.path.join(pose_data_dir, f"{output_filename}_pose_data.pkl")
                with open(pose_data_file, 'wb') as f:
                    pickle.dump(pose_results, f)
                print(f"Raw pose data saved to {pose_data_file}")
                
                # Copy frames to permanent location if needed
                frames_out_dir = os.path.join(output_viz_dir, f"{output_filename}_frames")
                os.makedirs(frames_out_dir, exist_ok=True)
                
                print(f"Copying frames from temp directory to: {frames_out_dir}")
                for i, frame_path in enumerate(frame_paths):
                    if os.path.exists(frame_path):
                        dest_path = os.path.join(frames_out_dir, f'frame_{i:06d}.jpg')
                        # Use cv2 to read and write instead of copying to ensure compatibility
                        frame = cv2.imread(frame_path)
                        cv2.imwrite(dest_path, frame)
            
            # Print the 100th frame pose results to a text file for debugging
            if len(pose_results) > 100:
                debug_dir = os.path.dirname(output_viz_dir) if output_viz_dir else "."
                os.makedirs(debug_dir, exist_ok=True)
                debug_file = os.path.join(debug_dir, f"{output_filename}_frame100_raw.txt")
                with open(debug_file, 'w') as f:
                    # Get the 100th frame pose result and write it directly to the file
                    frame_100 = pose_results[100]
                    f.write(str(frame_100))
                    print(f"Raw frame 100 data saved to {debug_file}")
            
            # Generate visualization if output directory is provided
            if output_viz_dir is not None:
                #create_visualization(frames, data_samples, output_filename, output_viz_dir, pose_config)
                pass
        
        print(f"{'='*20} Completed {osp.basename(video_path)} {'='*20}\n")
        
        # Return only the raw pose_results
        return pose_results
        
    except Exception as e:
        print(f"Error in pose_extraction for {video_path}:")
        print(traceback.format_exc())
        raise

def process_all_videos(annotation_file, output_dir, pose_config, pose_checkpoint, device, 
                     bbox_file, output_viz_dir=None):
    """Process all videos listed in the annotation file."""
    
    # Create output directories if they don't exist
    os.makedirs(output_dir, exist_ok=True)
    if output_viz_dir:
        os.makedirs(output_viz_dir, exist_ok=True)
    
    # Load bounding box data
    print("Loading bbox data...")
    bbox_df = pd.read_csv(bbox_file)
    bbox_df['video'] = bbox_df['video'].str.strip()
    
    # Track failed videos
    failed_videos = []
    # Track skipped videos (already processed)
    skipped_videos = []
    # Track successfully processed videos
    processed_videos = []
    
    # Read annotation file
    with open(annotation_file, 'r') as f:
        lines = f.readlines()
    
    print(f"Found {len(lines)} videos to process")
    
    # Process each video with progress bar
    for line in tqdm(lines, desc="Processing videos"):
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

        # Construct the video path
        if video_relative_path.startswith("videos/"):
            video_name = video_relative_path.split("/")[1]  # Extract just the filename
            video_path = os.path.join("preprocessing/videos", video_name)
        else:
            video_path = os.path.join("preprocessing", video_relative_path)
        
        # Also check for the augmented version
        base, ext = os.path.splitext(video_path)
        aug_video_path = base + '_aug' + ext
        
        # Process original video if it exists
        if os.path.exists(video_path):
            output_path = os.path.join(output_dir, f"{osp.splitext(video_name)[0]}.pkl")
            
            # Skip if keypoints already exist
            if os.path.exists(output_path):
                print(f"Keypoints already exist for {video_path}, skipping")
                skipped_videos.append(video_path)
            else:
                try:
                    print(f"Processing video: {video_path}")
                    pose_results = pose_extraction(video_path, pose_config, pose_checkpoint, 
                                         device, bbox_df, output_viz_dir)
                    if pose_results is not None:
                        # Save pose_results directly
                        with open(output_path, 'wb') as f:
                            pickle.dump(pose_results, f)
                        processed_videos.append(video_path)
                    else:
                        failed_videos.append((video_path, "Null pose results"))
                except Exception as e:
                    print(f"Failed to process {video_path}: {str(e)}")
                    failed_videos.append((video_path, str(e)))
        else:
            print(f"Video not found, skipping: {video_path}")
        
        # Process augmented video if it exists
        if os.path.exists(aug_video_path):
            aug_video_name = osp.basename(aug_video_path)
            output_path = os.path.join(output_dir, f"{osp.splitext(aug_video_name)[0]}.pkl")
            
            # Skip if keypoints already exist
            if os.path.exists(output_path):
                print(f"Keypoints already exist for {aug_video_path}, skipping")
                skipped_videos.append(aug_video_path)
            else:
                try:
                    print(f"Processing augmented video: {aug_video_path}")
                    pose_results = pose_extraction(aug_video_path, pose_config, pose_checkpoint, 
                                         device, bbox_df, output_viz_dir)
                    if pose_results is not None:
                        # Save pose_results directly
                        with open(output_path, 'wb') as f:
                            pickle.dump(pose_results, f)
                        processed_videos.append(aug_video_path)
                    else:
                        failed_videos.append((aug_video_path, "Null pose results"))
                except Exception as e:
                    print(f"Failed to process {aug_video_path}: {str(e)}")
                    failed_videos.append((aug_video_path, str(e)))
        else:
            print(f"Augmented video not found, skipping: {aug_video_path}")
    
    # Print summary
    print("\n=== Processing Summary ===")
    print(f"Total videos found: {len(lines)}")
    print(f"Videos processed successfully: {len(processed_videos)}")
    print(f"Videos skipped (already processed): {len(skipped_videos)}")
    print(f"Videos failed: {len(failed_videos)}")
    
    # Print failed videos details
    if failed_videos:
        print("\nFailed videos summary:")
        for video, error in failed_videos:
            print(f"{video}: {error}")
        
        # Save failed videos to file
        with open(os.path.join(output_dir, 'failed_videos.txt'), 'w') as f:
            for video, error in failed_videos:
                f.write(f"{video}: {error}\n")

if __name__ == '__main__':
    # Configuration
    config = {
        'annotation_file': 'preprocessing/video_annotations/video_annotations.txt',
        'output_dir': 'preprocessing/video_keypoints',
        'output_viz_dir': 'preprocessing/video_keypoints/visualized_videos',
        'device': 'cuda:0',
        'pose_config': 'mmpose/configs/wholebody_2d_keypoint/topdown_heatmap/coco-wholebody/td-hm_hrnet-w48_dark-8xb32-210e_coco-wholebody-384x288.py',
        'pose_checkpoint': 'https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w48_coco_wholebody_384x288_dark-f5726563_20200918.pth',
        'bbox_file': 'preprocessing/video_annotations/initial_bboxes.csv'
    }
    
    # Run processing
    process_all_videos(
        config['annotation_file'],
        config['output_dir'],
        config['pose_config'],
        config['pose_checkpoint'],
        config['device'],
        config['bbox_file'],
        config['output_viz_dir']
    ) 