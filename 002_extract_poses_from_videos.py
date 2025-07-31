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
import time
from mmengine.structures import InstanceData

# Suppress specific warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

# Configure logging to only show errors
logging.getLogger('mmengine').setLevel(logging.ERROR)
logging.getLogger('mmcv').setLevel(logging.ERROR)
logging.getLogger('mmpose').setLevel(logging.ERROR)

def get_optimal_device():
    """Find the optimal device (GPU or CPU) for running the model."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    return device

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

def get_video_info(video_path):
    """Get video information including frame count, width, height, and fps."""
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        cap.release()
        return {
            'frame_count': frame_count,
            'width': width,
            'height': height,
            'fps': fps
        }
    except Exception as e:
        print(f"Error getting video info for {video_path}:")
        print(traceback.format_exc())
        raise

def get_init_bbox(video_name, video_path, bbox_df):
    """Get the initial bounding box for a video."""
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
            
        return bbox
    except Exception as e:
        print(f"Error in get_init_bbox for {video_name}:")
        print(traceback.format_exc())
        raise

def process_single_frame(model, frame, det, device):
    """Process a single frame for pose estimation."""
    try:
        from mmpose.apis import inference_topdown
        from mmpose.structures import merge_data_samples
        
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
        
        # Extract pose data
        poses = pose_data_sample.pred_instances.to_dict()
        
        return poses, pose_data_sample
    except Exception as e:
        print(f"Error in process_single_frame:")
        print(traceback.format_exc())
        raise

def pose_extraction_frame_by_frame(video_path, pose_config, pose_checkpoint, device, bbox_df, output_viz_dir=None):
    """Extract poses from a video frame by frame without storing all frames in memory."""
    try:
        from mmpose.apis import init_model
        
        total_start_time = time.time()
        print(f"\n{'='*20} Processing {osp.basename(video_path)} {'='*20}")
        
        # Get video information
        video_info = get_video_info(video_path)
        frame_count = video_info['frame_count']
        fps = video_info['fps']
        print(f"Video info: {frame_count} frames, {fps:.2f} fps")
        
        # Get bounding box for the video
        video_name = osp.basename(video_path)
        bbox_start_time = time.time()
        bbox = get_init_bbox(video_name, video_path, bbox_df)
        bbox_end_time = time.time()
        bbox_time = bbox_end_time - bbox_start_time
        print(f"Bounding box preparation completed in {bbox_time:.2f} seconds")
        
        # Initialize the pose model
        model_init_start = time.time()
        model = init_model(pose_config, pose_checkpoint, device)
        model_init_end = time.time()
        model_init_time = model_init_end - model_init_start
        print(f"Model initialization completed in {model_init_time:.2f} seconds")
        
        # Open the video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        # Initialize visualization if needed
        visualizer = None
        out = None
        if output_viz_dir is not None:
            from mmengine.registry import VISUALIZERS
            import mmcv
            
            output_filename = osp.splitext(osp.basename(video_path))[0]
            pose_config_obj = mmengine.Config.fromfile(pose_config)
            visualizer = VISUALIZERS.build(pose_config_obj.visualizer)
            
            # Get dimensions from the first frame
            ret, first_frame = cap.read()
            if not ret:
                raise ValueError(f"Could not read first frame from {video_path}")
            
            height, width = first_frame.shape[:2]
            
            # Initialize video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out_filename = os.path.join(output_viz_dir, f"{output_filename}_viz.mp4")
            out = cv2.VideoWriter(out_filename, fourcc, fps, (width, height))
            
            # Reset video to start
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        # Process frames one by one
        results = []
        frame_times = []
        frame_idx = 0
        
        # Create progress bar for frame processing
        with tqdm(total=frame_count, desc="Processing frames") as pbar:
            while True:
                # Time each frame
                frame_start = time.time()
                
                # Read frame
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process frame
                det = bbox.copy()  # Use the same bbox for all frames
                poses, pose_data_sample = process_single_frame(model, frame, det, device)
                
                # Store results
                results.append(poses)
                
                # Create visualization if needed
                if visualizer is not None and out is not None:
                    # Set dataset meta if this is the first frame
                    if frame_idx == 0:
                        visualizer.set_dataset_meta(pose_data_sample.dataset_meta)
                    
                    # Create visualization
                    f = mmcv.imconvert(frame, 'bgr', 'rgb')
                    visualizer.add_datasample(
                        'result',
                        f,
                        data_sample=pose_data_sample,
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
                
                # Record frame time
                frame_end = time.time()
                frame_time = frame_end - frame_start
                frame_times.append(frame_time)
                
                # Save debug info for frame 100
                if frame_idx == 100:
                    output_filename = osp.splitext(osp.basename(video_path))[0]
                    debug_dir = os.path.dirname(output_viz_dir) if output_viz_dir else "."
                    os.makedirs(debug_dir, exist_ok=True)
                    debug_file = os.path.join(debug_dir, f"{output_filename}_frame100_raw.txt")
                    with open(debug_file, 'w') as f:
                        f.write(str(poses))
                        print(f"Raw frame 100 data saved to {debug_file}")
                
                frame_idx += 1
                pbar.update(1)
        
        # Release resources
        cap.release()
        if out is not None:
            out.release()
            print(f'Visualization saved to {os.path.join(output_viz_dir, f"{output_filename}_viz.mp4")}')
        
        # Calculate statistics
        total_end_time = time.time()
        total_time = total_end_time - total_start_time
        
        if frame_times:
            avg_frame_time = sum(frame_times) / len(frame_times)
            max_frame_time = max(frame_times)
            min_frame_time = min(frame_times)
            
            print(f"Frame processing statistics:")
            print(f"  Average frame processing time: {avg_frame_time:.4f} seconds")
            print(f"  Min/Max frame times: {min_frame_time:.4f}/{max_frame_time:.4f} seconds")
        
        print(f"Time breakdown for {osp.basename(video_path)}:")
        print(f"  Bounding box prep: {bbox_time:.2f} seconds ({bbox_time/total_time*100:.1f}%)")
        print(f"  Model initialization: {model_init_time:.2f} seconds ({model_init_time/total_time*100:.1f}%)")
        print(f"  Total processing time: {total_time:.2f} seconds")
        print(f"{'='*20} Completed {osp.basename(video_path)} {'='*20}\n")
        
        return results
        
    except Exception as e:
        print(f"Error in pose_extraction_frame_by_frame for {video_path}:")
        print(traceback.format_exc())
        raise

def process_all_videos(annotation_file, output_dir, pose_config, pose_checkpoint, device, 
                     bbox_file, output_viz_dir=None):
    """Process all videos listed in the annotation file."""
    
    overall_start_time = time.time()
    
    # Create output directories if they don't exist
    os.makedirs(output_dir, exist_ok=True)
    if output_viz_dir:
        os.makedirs(output_viz_dir, exist_ok=True)
    
    # Load bounding box data
    bbox_load_start = time.time()
    print("Loading bbox data...")
    bbox_df = pd.read_csv(bbox_file)
    bbox_df['video'] = bbox_df['video'].str.strip()
    bbox_load_end = time.time()
    print(f"Bbox data loaded in {bbox_load_end - bbox_load_start:.2f} seconds")
    
    # Track failed videos
    failed_videos = []
    # Track skipped videos (already processed)
    skipped_videos = []
    # Track successfully processed videos
    processed_videos = []
    
    # Track timing statistics
    extraction_times = []
    saving_times = []
    
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
                    extraction_start = time.time()
                    pose_results = pose_extraction_frame_by_frame(video_path, pose_config, pose_checkpoint, 
                                         device, bbox_df, output_viz_dir)
                    extraction_end = time.time()
                    extraction_time = extraction_end - extraction_start
                    extraction_times.append(extraction_time)
                    
                    if pose_results is not None:
                        # Save pose_results directly
                        save_start = time.time()
                        with open(output_path, 'wb') as f:
                            pickle.dump(pose_results, f)
                        save_end = time.time()
                        save_time = save_end - save_start
                        saving_times.append(save_time)
                        print(f"Results saved in {save_time:.2f} seconds")
                        
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
                    extraction_start = time.time()
                    pose_results = pose_extraction_frame_by_frame(aug_video_path, pose_config, pose_checkpoint, 
                                         device, bbox_df, output_viz_dir)
                    extraction_end = time.time()
                    extraction_time = extraction_end - extraction_start
                    extraction_times.append(extraction_time)
                    
                    if pose_results is not None:
                        # Save pose_results directly
                        save_start = time.time()
                        with open(output_path, 'wb') as f:
                            pickle.dump(pose_results, f)
                        save_end = time.time()
                        save_time = save_end - save_start
                        saving_times.append(save_time)
                        print(f"Results saved in {save_time:.2f} seconds")
                        
                        processed_videos.append(aug_video_path)
                    else:
                        failed_videos.append((aug_video_path, "Null pose results"))
                except Exception as e:
                    print(f"Failed to process {aug_video_path}: {str(e)}")
                    failed_videos.append((aug_video_path, str(e)))
        else:
            print(f"Augmented video not found, skipping: {aug_video_path}")
    
    overall_end_time = time.time()
    overall_time = overall_end_time - overall_start_time
    
    # Print summary
    print("\n=== Processing Summary ===")
    print(f"Total videos found: {len(lines)}")
    print(f"Videos processed successfully: {len(processed_videos)}")
    print(f"Videos skipped (already processed): {len(skipped_videos)}")
    print(f"Videos failed: {len(failed_videos)}")
    
    # Print timing summary
    if extraction_times:
        avg_extraction_time = sum(extraction_times) / len(extraction_times)
        print(f"\n=== Timing Summary ===")
        print(f"Average extraction time per video: {avg_extraction_time:.2f} seconds")
        if saving_times:
            avg_saving_time = sum(saving_times) / len(saving_times)
            print(f"Average saving time per video: {avg_saving_time:.2f} seconds")
        print(f"Total processing time: {overall_time:.2f} seconds")
    
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
        'device': get_optimal_device(),  # Automatically find the best device
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