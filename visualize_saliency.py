#!/usr/bin/env python
"""
Script to generate and visualize saliency maps for the trained models.
Uses trained models from k-fold cross validation to show which input features
are most important for predictions.
"""

import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import pickle
from pathlib import Path
import mmcv
from mmengine.config import Config
from mmaction.apis import init_recognizer
from mmengine.runner import Runner
import os
import json
import re
import subprocess

def find_best_model(fold_dir: Path) -> Path:
    """
    Find the best model checkpoint in the fold directory.
    """
    checkpoints = list(fold_dir.glob('best_acc_top1_epoch*.pth'))
    if not checkpoints:
        raise FileNotFoundError(f"No best model checkpoint found in {fold_dir}")
    
    if len(checkpoints) > 1:
        # If multiple checkpoints exist, get the one with the highest epoch number
        return max(checkpoints, key=lambda x: int(str(x).split('epoch_')[-1].split('.')[0]))
    
    return checkpoints[0]

def load_model_and_data(fold: int) -> list:
    """
    Load model and all test data for a specific fold.
    Maps predictions to clip names from the annotation file.
    """
    # Find best model checkpoint
    fold_dir = Path(f'k_fold/work_dirs/fold{fold}')
    try:
        checkpoint = find_best_model(fold_dir)
        print(f"Using checkpoint: {checkpoint}")
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Error for fold {fold}: {e}")
    
    config_file = f'k_fold/stgcn/stgcn_fold{fold}.py'

    video_ids = get_videos_in_test_split(config_file)
    print(f"Video IDs: {video_ids}")

    # Filter annotations for the specific video ID
    filter_annotations_by_video(video_ids[0], config_file)  # Keep only clips for the first video ID
    
    # Load the config file
    cfg = Config.fromfile(config_file)
    
    # Initialize model using MMAction2 API
    model = init_recognizer(cfg, str(checkpoint), device='cuda')
    model.eval()
    
    # Build the test dataset from config
    test_dataloader = Runner.build_dataloader(cfg.test_dataloader)
    
    # Load predictions to get total number of samples
    pred_file = f'k_fold/test_results/fold{fold}_predictions.pkl'
    with open(pred_file, 'rb') as f:
        predictions = pickle.load(f)
    
    # Get the clip names from the annotation file
    annotation_file = get_annotation_file_from_config(config_file)
    with open(annotation_file, 'rb') as f:
        annotations = pickle.load(f)
    clip_names = annotations['split']['xsub_test']
    
    print(f"Number of predictions: {len(predictions)}")
    print(f"Number of clips in annotation file: {len(clip_names)}")
    
    # If the number of predictions matches the number of clips, we can map them directly
    use_clip_names = len(predictions) == len(clip_names)
    if use_clip_names:
        print("Number of predictions matches number of clips - will use clip names for filenames")
    else:
        print(f"Warning: Number of predictions ({len(predictions)}) doesn't match number of clips ({len(clip_names)})")
        print("Will use sample indices for filenames instead")
    
    results = []
    for sample_idx in range(len(predictions)):  # Iterate over all samples
        # Get to the specific sample in dataloader
        for i, data_batch in enumerate(test_dataloader):
            if i == sample_idx:
                break
        
        # Get the input data
        input_data = data_batch['inputs']
        if isinstance(input_data, (list, tuple)):
            input_data = input_data[0]
        
        # Reshape input data
        if isinstance(input_data, torch.Tensor):
            input_data = input_data[0:1]  # Keep only first clip
            input_data = input_data.unsqueeze(1)
            input_data = input_data.cuda()
        
        # Get ground truth label
        true_label = predictions[sample_idx]['gt_label'].item()
        
        # Get the clip name if available and numbers match
        clip_name = None
        if use_clip_names and sample_idx < len(clip_names):
            clip_name = clip_names[sample_idx]
        
        results.append((model, input_data, true_label, sample_idx, clip_name))
    
    return results

def compute_saliency_map(model, input_tensor):
    """
    Compute saliency map for given input and model.
    """
    # Ensure input tensor is on GPU and requires grad
    input_tensor = input_tensor.cuda()
    input_tensor.requires_grad_()
    
    # Forward pass
    with torch.enable_grad():
        output = model(input_tensor, return_loss=False)
        print(f"Raw output type: {type(output)}")
        print(f"Raw output shape/size: {output.shape if isinstance(output, torch.Tensor) else [o.shape for o in output]}")
        
        # Handle different output formats
        if isinstance(output, (list, tuple)):
            output = output[0]  # Take first element if it's a list/tuple
        
        # For output shape [N, M, T, C, V], average over T dimension
        output = output.mean(dim=2)  # Now [N, M, C, V]
        output = output.squeeze(1)   # Remove M dimension -> [N, C, V]
        
        print(f"Processed output shape: {output.shape}")
        print(f"Output values min: {output.min().item()}, max: {output.max().item()}")
        
        # Get predictions across all dimensions
        pred_scores = output.mean(dim=-1)  # Average over keypoints -> [N, C]
        
        # Convert to binary classification (if needed)
        if pred_scores.size(1) != 2:
            print(f"Warning: Model output has {pred_scores.size(1)} classes, converting to binary")
            # Assuming class 0 is negative and rest are positive
            binary_scores = torch.zeros((pred_scores.size(0), 2), device=pred_scores.device)
            binary_scores[:, 0] = pred_scores[:, 0]  # Negative class
            binary_scores[:, 1] = pred_scores[:, 1:].sum(dim=1)  # Positive class
            pred_scores = binary_scores
        
        # Apply softmax to get probabilities
        pred_probs = torch.softmax(pred_scores, dim=1)[0]
        pred_class = pred_probs.argmax().item()
        print(f"Binary prediction scores: {pred_probs}")
        print(f"Predicted class: {pred_class} ({'Positive' if pred_class == 1 else 'Negative'})")
        
        # Get the score for the predicted class
        class_score = pred_scores[0, pred_class]
        print(f"Class score: {class_score.item()}")
        
        # Compute gradients
        class_score.backward()
        
        # Get gradients and convert to saliency map
        saliency = input_tensor.grad.abs()
        # Aggregate across M and C dimensions to get [T, V] saliency map
        saliency = saliency.mean(dim=1).mean(dim=-1)  
        saliency = saliency.squeeze()  # Remove all extra dimensions
        
        # Apply slight smoothing to make the map more visually interpretable
        saliency = torch.nn.functional.avg_pool2d(
            saliency.unsqueeze(0).unsqueeze(0), 
            kernel_size=3, 
            stride=1, 
            padding=1
        ).squeeze()
        
        print(f"Saliency map shape: {saliency.shape}")
        print(f"Saliency values min: {saliency.min().item()}, max: {saliency.max().item()}")
        
    return saliency.detach().cpu().numpy()

def plot_saliency(input_data, saliency_map, true_label, pred_label, save_path, model=None, clip_name=None):
    """
    Plot the saliency map and save it to a file.
    
    Args:
        input_data (torch.Tensor): Input data tensor
        saliency_map (torch.Tensor): Saliency map tensor
        true_label (int): True label (0 or 1)
        pred_label (int): Predicted label (0 or 1)
        save_path (Path): Path to save the plot
        model (nn.Module, optional): Model used for prediction
        clip_name (str, optional): Name of the clip
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # Create main title with clip name
    clip_display = clip_name if clip_name else "Unknown Clip"
    clip_type = "Seizure" if true_label == 1 else "Non-Seizure"
    main_title = f"Clip: {clip_display}\nType: {clip_type}"
    fig.suptitle(main_title, fontsize=12, y=1.02)
    
    # Define keypoint groups according to COCO WholeBody
    keypoint_groups = {
        'Body': list(range(0, 17)),      # Body keypoints
        'Feet': list(range(17, 23)),     # Foot keypoints (3 per foot)
        'Face': list(range(23, 91)),     # Face keypoints (68 points)
        'Left Hand': list(range(91, 112)), # Left hand keypoints (21 points)
        'Right Hand': list(range(112, 133)) # Right hand keypoints (21 points)
    }
    
    # Calculate motion between frames
    input_vis = input_data.squeeze().detach().cpu().numpy()  # Shape: (T, V, C)
    
    # Ensure input_vis has the expected shape
    if input_vis.ndim != 3:
        raise ValueError(f"Expected input shape (T, V, C), but got {input_vis.shape}")
    
    # Calculate Euclidean distance between consecutive frames for each keypoint
    motion_data = np.sqrt(np.sum(np.diff(input_vis, axis=0)**2, axis=-1))  # Shape: (T-1, V)
    
    # Add a zero frame for the first frame to maintain the same number of frames
    motion_data = np.concatenate(([np.zeros(motion_data.shape[1])], motion_data), axis=0)  # Shape: (T, V)
    
    # Normalize motion data for visualization
    motion_data = (motion_data - motion_data.min()) / (motion_data.max() - motion_data.min() + 1e-8)
    
    # Transpose the motion data to have keypoints on y-axis and time on x-axis
    motion_data = motion_data.T
    
    # Plot motion data
    im1 = ax1.imshow(motion_data, aspect='auto', cmap='viridis')
    
    ax1.set_title('Motion Between Frames')
    ax1.set_ylabel('Keypoints')
    ax1.set_xlabel('Time (frames)')
    plt.colorbar(im1, ax=ax1)
    
    # Add frame numbers on x-axis
    ax1.set_xticks(np.arange(0, motion_data.shape[1], 10))
    ax1.set_xticklabels([f"{i}" for i in range(0, motion_data.shape[1], 10)])
    
    # Add keypoint group labels on y-axis
    group_positions = []
    group_labels = []
    for group_name, indices in keypoint_groups.items():
        if indices:
            mid_point = (indices[0] + indices[-1]) / 2
            group_positions.append(mid_point)
            group_labels.append(f"{group_name}\n({len(indices)} pts)")
    
    # Set keypoint ticks
    ax1.set_yticks(group_positions)
    ax1.set_yticklabels(group_labels)
    
    # Plot saliency map
    if saliency_map.ndim > 2:
        saliency_map = saliency_map.squeeze()
    saliency_norm = (saliency_map - saliency_map.min()) / (saliency_map.max() - saliency_map.min() + 1e-8)
    
    # Transpose the saliency map
    saliency_norm = saliency_norm.T
    
    im2 = ax2.imshow(saliency_norm, aspect='auto', cmap='hot')
    
    # Set title without clip name, just prediction info
    ax2.set_title(f'Saliency Map\nTrue: {"Seizure" if true_label == 1 else "Non-Seizure"}, ' + 
                  f'Pred: {"Seizure" if pred_label == 1 else "Non-Seizure"}')
    
    ax2.set_ylabel('Keypoints')
    ax2.set_xlabel('Time (frames)')
    plt.colorbar(im2, ax=ax2)
    
    # Add frame numbers on x-axis
    ax2.set_xticks(np.arange(0, saliency_norm.shape[1], 10))
    ax2.set_xticklabels([f"{i}" for i in range(0, saliency_norm.shape[1], 10)])
    
    # Add same keypoint group labels on y-axis
    ax2.set_yticks(group_positions)
    ax2.set_yticklabels(group_labels)
    
    # Add prediction confidence
    with torch.no_grad():
        pred = model(input_data.cuda(), return_loss=False)
        if isinstance(pred, (list, tuple)):
            pred = pred[0]
        pred = pred.mean(dim=2).squeeze(1)
        pred_scores = pred.mean(dim=-1)
        if pred_scores.size(1) > 2:
            binary_scores = torch.zeros((pred_scores.size(0), 2), device=pred_scores.device)
            binary_scores[:, 0] = pred_scores[:, 0]
            binary_scores[:, 1] = pred_scores[:, 1:].sum(dim=1)
            pred_scores = binary_scores
        probs = torch.softmax(pred_scores, dim=1)[0]
    
    confidence = float(probs[pred_label])
    plt.figtext(0.02, 0.02, f'Prediction confidence: {confidence:.2%}', 
                fontsize=10, bbox=dict(facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def get_annotation_file_from_config(config_file: str) -> str:
    """
    Get the annotation file path from the configuration file.
    """
    cfg = Config.fromfile(config_file)
    return cfg.ann_file

def get_videos_in_test_split(config_file: str) -> list:
    """
    Get video IDs from the test split in the annotation file.

    """
    # Get the annotation file path from the config
    annotation_file = get_annotation_file_from_config(config_file)
    
    # Load the annotation file
    with open(annotation_file, 'rb') as f:  # Open in binary mode for pickle
        annotations = pickle.load(f)
    
    # Extract video IDs from the test split
    test_videos = annotations['split']['xsub_test']
    
    # Extract unique video IDs from the clip names
    video_ids = set()
    for clip in test_videos:
        parts = clip.split('_')
        if len(parts) >= 2:
            video_id = parts[1]  # Extract the video ID
            video_ids.add(video_id)
    
    return sorted(list(video_ids))

def find_all_clips_for_video(video_id: str, config_file: str):
    """
    Find all clips for a specific video from preprocessing/clip_keypoints directory
    and add them to xsub_test in order, avoiding augmented clips.
    """
    # Define the preprocessing directory
    preprocessing_dir = Path('preprocessing/clip_keypoints')
    
    # Check if directory exists
    if not preprocessing_dir.exists():
        print(f"Warning: {preprocessing_dir} does not exist. Using existing clips only.")
        return []
    
    # Find all clip files for the specified video ID
    # Files are named like 05463487_79519000_Seg_1.pkl
    clip_files = []
    for item in preprocessing_dir.glob(f'*_{video_id}_Seg_*.pkl'):
        if item.is_file() and "_aug" not in item.name:
            clip_files.append(item)
    
    # Extract clip names (without .pkl extension)
    clip_names = [file.stem for file in clip_files]
    print(f"Found {len(clip_names)} clips for video {video_id}")
    
    # Create a list of (clip_name, segment_number) tuples for sorting
    clip_with_segment = []
    for clip in clip_names:
        try:
            # Split by '_Seg_' and take the part after it
            seg_part = clip.split('_Seg_')[1]
            # Convert to integer (handle any additional parts by splitting again)
            if '_' in seg_part:
                seg_num = int(seg_part.split('_')[0])
            else:
                seg_num = int(seg_part)
            clip_with_segment.append((clip, seg_num))
            #print(f"Clip: {clip}, Segment: {seg_num}")
        except (IndexError, ValueError) as e:
            # If parsing fails, use a large number to sort it at the end
            print(f"Error parsing segment number for {clip}: {e}")
            clip_with_segment.append((clip, 999999))
    
    # Sort by segment number
    clip_with_segment.sort(key=lambda x: x[1])
    
    # Extract sorted clip names
    sorted_clip_names = [item[0] for item in clip_with_segment]
    
    print(f"Found {len(sorted_clip_names)} non-augmented clips for video {video_id} in preprocessing directory")
    print(f"First 5 sorted clips: {sorted_clip_names[:5]}")
    print(f"Last 5 sorted clips: {sorted_clip_names[-5:]}")
    
    return sorted_clip_names

def filter_annotations_by_video(video_id: str, config_file: str, output_dir: Path = None):
    """
    Filter the annotation file to keep clips of only one specified video 
    and save it back to the same file. Removes augmented clips.
    
    Args:
        video_id (str): The video ID to filter for
        config_file (str): Path to the configuration file
        output_dir (Path, optional): Directory to save text annotation file
    """
    # Get the annotation file path from the config
    annotation_file = get_annotation_file_from_config(config_file)
    
    # Load the annotation file
    with open(annotation_file, 'rb') as f:  # Open in binary mode for pickle
        annotations = pickle.load(f)
    
    # Find all clips for this video from preprocessing directory
    all_clips = find_all_clips_for_video(video_id, config_file)
    print(f"Found {len(all_clips)} clips for video {video_id} in preprocessing directory")

    # Create filtered annotations
    filtered_annotations = {
        'split': {
            'xsub_test': all_clips,
            'xsub_train': annotations['split'].get('xsub_train', []),  # Keep other splits unchanged
        },
        'annotations': []
    }
    
    # Filter the annotations to keep only those for the specified video ID and non-augmented
    existing_frame_dirs = set()
    for annotation in annotations['annotations']:
        if video_id in annotation['frame_dir'] and "_aug" not in annotation['frame_dir']:
            filtered_annotations['annotations'].append(annotation)
            existing_frame_dirs.add(annotation['frame_dir'])
    
    # Check if we need to add new annotations for clips found in preprocessing
    for clip in all_clips:
        if clip not in existing_frame_dirs:
            print(f"Warning: No annotation found for clip {clip}. This clip will be included in the split but may not have annotation data.")
    
    # Save the filtered annotations back to the same file
    with open(annotation_file, 'wb') as f:
        pickle.dump(filtered_annotations, f)

    # Extract fold number from config file path
    fold_num = "unknown"
    if "fold" in config_file:
        fold_num = config_file.split("fold")[-1].split(".")[0]
    
    # Determine where to save the text annotation file
    if output_dir is None:
        # Use default location
        txt_annotation_file = f'k_fold/saliency_maps/{video_id}/annotations_fold{fold_num}.txt'
    else:
        # Use provided output directory
        txt_annotation_file = output_dir / f'annotations_fold{fold_num}.txt'
    
    # Ensure the directory exists
    os.makedirs(os.path.dirname(txt_annotation_file), exist_ok=True)
    
    # Write the annotations to the text file
    write_annotations_to_txt(filtered_annotations, txt_annotation_file)
    
    print(f"Filtered annotations saved to {annotation_file} with clips for video ID: {video_id}")
    print(f"Human-readable annotations saved to {txt_annotation_file}")
    print(f"Total non-augmented clips: {len(all_clips)}")
    print(f"Total non-augmented annotations: {len(filtered_annotations['annotations'])}")
    
    return filtered_annotations

def write_annotations_to_txt(annotations, output_file):
    """
    Write the annotations to a text file in a JSON-like format.
    
    Args:
        annotations (dict): The annotations dictionary
        output_file (str): Path to the output text file
    """
    with open(output_file, 'w') as f:
        f.write("{\n")
        
        # Write split information
        f.write("    \"split\":\n")
        f.write("        {\n")
        
        # Write each split
        splits = list(annotations['split'].keys())
        for i, split_name in enumerate(splits):
            clips = annotations['split'][split_name]
            f.write(f"            '{split_name}':\n")
            f.write("                [")
            
            # For xsub_test, show all clips
            if split_name == 'xsub_test':
                for j, clip in enumerate(clips):
                    if j % 3 == 0:  # Start a new line every 3 clips for readability
                        f.write("\n                    ")
                    f.write(f"'{clip}'")
                    if j < len(clips) - 1:
                        f.write(", ")
            else:
                # For other splits, show a limited number
                max_clips_to_show = 20
                for j, clip in enumerate(clips[:max_clips_to_show]):
                    if j % 5 == 0:  # Start a new line every 5 clips for readability
                        f.write("\n                    ")
                    f.write(f"'{clip}'")
                    if j < min(len(clips), max_clips_to_show) - 1:
                        f.write(", ")
                
                # Indicate if there are more clips
                if len(clips) > max_clips_to_show:
                    f.write(f"\n                    ... and {len(clips) - max_clips_to_show} more clips")
            
            f.write("\n                ]")
            if i < len(splits) - 1:
                f.write(",")
            f.write("\n")
        
        f.write("        },\n\n")
        
        # Write annotation details
        f.write("    \"annotations\":\n")
        f.write("        [\n")
        
        # Write each annotation
        for i, ann in enumerate(annotations['annotations']):
            f.write("            {\n")
            
            # Write annotation fields
            f.write(f"                'frame_dir': '{ann['frame_dir']}',\n")
            f.write(f"                'label': {ann['label']},\n")
            
            # Add other fields if they exist
            if 'img_shape' in ann:
                f.write(f"                'img_shape': {ann['img_shape']},\n")
            if 'original_shape' in ann:
                f.write(f"                'original_shape': {ann['original_shape']},\n")
            if 'total_frames' in ann:
                f.write(f"                'total_frames': {ann['total_frames']},\n")
            
            # Handle keypoint data - show shape instead of full array
            if 'keypoint' in ann:
                kp_shape = np.array(ann['keypoint']).shape
                f.write(f"                'keypoint': array with shape {kp_shape},\n")
            if 'keypoint_score' in ann:
                kp_score_shape = np.array(ann['keypoint_score']).shape
                f.write(f"                'keypoint_score': array with shape {kp_score_shape}")
                if i < len(annotations['annotations']) - 1:
                    f.write(",\n")
                else:
                    f.write("\n")
            
            f.write("            }")
            if i < len(annotations['annotations']) - 1:
                f.write(",\n")
            else:
                f.write("\n")
        
        f.write("        ]\n")
        f.write("}\n")

def run_testing_for_fold(fold: int) -> bool:
    """
    Run the testing script for a specific fold to generate prediction files.
    This should be called after modifying the annotation file.
    
    Args:
        fold (int): The fold number to run testing for
        
    Returns:
        bool: True if testing was successful, False otherwise
    """
    print(f"\nRunning testing for fold {fold} to generate prediction files...")
    
    # Path to the testing script
    testing_script = 'run_kfold_testing.py'
    
    # Build command
    cmd = [
        'python', testing_script,
        '--folds', str(fold)
    ]
    
    try:
        # Run the testing process
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"Testing completed successfully for fold {fold}")
        print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error running testing for fold {fold}: {e}")
        print(f"stdout: {e.stdout}")
        print(f"stderr: {e.stderr}")
        return False

def process_fold_with_testing(fold: int, output_dir: Path):
    """
    Process a single fold with testing:
    1. Filter annotations for the specific video ID
    2. Run testing to generate predictions
    3. Generate saliency maps
    
    Args:
        fold (int): The fold number to process
        output_dir (Path): Directory to save saliency maps
    """
    try:
        # Get the video ID for this fold
        config_file = f'k_fold/stgcn/stgcn_fold{fold}.py'
        video_ids = get_videos_in_test_split(config_file)
        current_video_id = video_ids[0]  # Use the first video ID
        print(f"Processing video ID: {current_video_id}")
        
        # Create a video-specific output directory
        video_output_dir = output_dir / current_video_id
        video_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Filter annotations for the specific video ID
        filter_annotations_by_video(current_video_id, config_file, video_output_dir)
        
        # Run testing for this fold to generate prediction files
        '''testing_success = run_testing_for_fold(fold)
        if not testing_success:
            print(f"Warning: Testing failed for fold {fold}. Saliency maps may be incomplete.")
            return'''
        
        # Load model and all test data
        samples = load_model_and_data(fold)
        print(f"Number of samples: {len(samples)}")
        for i, (_, _, true_label, sample_idx, clip_name) in enumerate(samples[:5]):  # Show first 5 samples
            print(f"Sample {i}: idx={sample_idx}, label={true_label}, clip_name={clip_name}")
    
        #sys.exit() 
        
        for model, input_data, true_label, sample_idx, clip_name in samples:
            print(f"\nProcessing sample {sample_idx}")
            if clip_name:
                print(f"Clip: {clip_name}")
            print(f"True label: {true_label}")
            
            # Compute saliency map
            saliency_map = compute_saliency_map(model, input_data)
            
            # Get prediction
            with torch.no_grad():
                pred = model(input_data, return_loss=False)
                if isinstance(pred, (list, tuple)):
                    pred = pred[0]
                pred = pred.mean(dim=2).squeeze(1)
                pred_scores = pred.mean(dim=-1)
                if pred_scores.size(1) > 2:
                    binary_scores = torch.zeros((pred_scores.size(0), 2), device=pred_scores.device)
                    binary_scores[:, 0] = pred_scores[:, 0]
                    binary_scores[:, 1] = pred_scores[:, 1:].sum(dim=1)
                    pred_scores = binary_scores
                pred_probs = torch.softmax(pred_scores, dim=1)[0]
                pred_label = pred_probs.argmax().item()
            
            # Create save path with clip name if available
            if clip_name:
                save_path = video_output_dir / f'saliency_map_{clip_name}.png'
            else:
                save_path = video_output_dir / f'saliency_map_fold_{fold}_sample_{sample_idx}.png'
            
            # Plot and save
            plot_saliency(
                input_data,
                saliency_map,
                true_label,
                pred_label,
                save_path,
                model,
                clip_name
            )
            
            if clip_name:
                print(f"Generated saliency map for clip {clip_name}")
            else:
                print(f"Generated saliency map for sample {sample_idx}")
        
        print(f"All saliency maps for video {current_video_id} saved to {video_output_dir}")
        
    except Exception as e:
        print(f"Error processing fold {fold}: {e}")
        import traceback
        traceback.print_exc()

def main():
    # Create output directory
    output_dir = Path('k_fold/saliency_maps')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process each fold
    n_folds = 3
    
    for fold in range(n_folds):
        print(f"\nProcessing fold {fold}...")
        process_fold_with_testing(fold, output_dir)

if __name__ == '__main__':
    main() 