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
        except (IndexError, ValueError) as e:
            # If parsing fails, use a large number to sort it at the end
            print(f"Error parsing segment number for {clip}: {e}")
            clip_with_segment.append((clip, 999999))
    
    # Sort by segment number
    clip_with_segment.sort(key=lambda x: x[1])
    
    # Extract sorted clip names
    sorted_clip_names = [item[0] for item in clip_with_segment]
    
    print(f"Found {len(sorted_clip_names)} non-augmented clips for video {video_id}")
    
    return sorted_clip_names

def filter_annotations_by_video(video_id: str, config_file: str, output_dir: Path = None):
    """
    Filter the annotation file to keep clips of only one specified video 
    and save it back to the same file. Keeps all seizure clips plus 5 clips
    before and after each continuous seizure segment.
    """
    # Get the annotation file path from the config
    annotation_file = get_annotation_file_from_config(config_file)
    
    # Load the annotation file
    with open(annotation_file, 'rb') as f:
        annotations = pickle.load(f)
    
    # Find all clips for this video from preprocessing directory
    all_clips = find_all_clips_for_video(video_id, config_file)
    print(f"Found {len(all_clips)} clips for video {video_id} in preprocessing directory")

    # Create a mapping of clip names to labels
    clip_to_label = {}
    for annotation in annotations['annotations']:
        if video_id in annotation['frame_dir'] and "_aug" not in annotation['frame_dir']:
            clip_name = annotation['frame_dir']
            label = annotation['label']
            clip_to_label[clip_name] = label
    
    # Identify seizure clips and their indices
    seizure_indices = []
    for i, clip in enumerate(all_clips):
        if clip_to_label.get(clip) == 1:  # 1 = seizure
            seizure_indices.append(i)
    
    if not seizure_indices:
        print("No seizure clips found. Keeping all clips.")
        indices_to_keep = list(range(len(all_clips)))
    else:
        # Find continuous seizure segments
        seizure_segments = []
        current_segment = [seizure_indices[0]]
        
        for i in range(1, len(seizure_indices)):
            # If this index is consecutive with the previous one, add to current segment
            if seizure_indices[i] == seizure_indices[i-1] + 1:
                current_segment.append(seizure_indices[i])
            else:
                # This is the start of a new segment
                seizure_segments.append(current_segment)
                current_segment = [seizure_indices[i]]
        
        # Add the last segment
        if current_segment:
            seizure_segments.append(current_segment)
        
        print(f"Found {len(seizure_segments)} continuous seizure segments")
        
        # Determine indices to keep (seizure clips + context)
        indices_to_keep = set()
        for segment in seizure_segments:
            # Add 5 clips before the segment
            start_idx = max(0, segment[0] - 5)
            # Add 5 clips after the segment
            end_idx = min(len(all_clips) - 1, segment[-1] + 5)
            
            # Add all indices in the range [start_idx, end_idx]
            indices_to_keep.update(range(start_idx, end_idx + 1))
        
        indices_to_keep = sorted(indices_to_keep)
        
        print(f"Keeping {len(indices_to_keep)} clips out of {len(all_clips)} total clips")
        print(f"This includes {len(seizure_indices)} seizure clips and {len(indices_to_keep) - len(seizure_indices)} context clips")
    
    # Filter the clips list to keep only the selected indices
    filtered_clips = [all_clips[i] for i in indices_to_keep]
    
    # Create filtered annotations
    filtered_annotations = {
        'split': {
            'xsub_test': filtered_clips,
            'xsub_train': annotations['split'].get('xsub_train', []),  # Keep other splits unchanged
        },
        'annotations': []
    }
    
    # Filter the annotations to keep only those for the specified clips
    filtered_clips_set = set(filtered_clips)
    label_counts = {0: 0, 1: 0}
    
    for annotation in annotations['annotations']:
        if annotation['frame_dir'] in filtered_clips_set:
            filtered_annotations['annotations'].append(annotation)
            
            # Count labels
            label = annotation['label']
            label_counts[label] = label_counts.get(label, 0) + 1
    
    print(f"Label distribution in filtered annotations: {label_counts}")
    
    # Save the filtered annotations back to the same file
    with open(annotation_file, 'wb') as f:
        pickle.dump(filtered_annotations, f)
    
    print(f"Filtered annotations saved to {annotation_file} with clips for video ID: {video_id}")
    print(f"Total clips kept: {len(filtered_clips)}")
    print(f"Total annotations: {len(filtered_annotations['annotations'])}")
    
    return filtered_annotations

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
    
    # Check label distribution
    label_counts = {0: 0, 1: 0}
    for pred in predictions:
        label = pred['gt_label'].item()
        label_counts[label] = label_counts.get(label, 0) + 1
    
    print(f"Label distribution in predictions: {label_counts}")
    
    # Get the clip names from the annotation file
    annotation_file = get_annotation_file_from_config(config_file)
    with open(annotation_file, 'rb') as f:
        annotations = pickle.load(f)
    clip_names = annotations['split']['xsub_test']
    
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
        print(f"Output shape: {output.shape}")
        # Handle different output formats
        if isinstance(output, (list, tuple)):
            output = output[0]  # Take first element if it's a list/tuple
        
        # For output shape [N, M, T, C, V], average over T dimension
        output = output.mean(dim=2)  # Now [N, M, C, V]
        output = output.squeeze(1)   # Remove M dimension -> [N, C, V]
        
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
        
        # Get the score for the predicted class
        class_score = pred_scores[0, pred_class]
        
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
        
    return saliency.detach().cpu().numpy()

def calculate_motion_data(input_vis):
    """
    Calculate motion data (velocity and acceleration) from input keypoint data.
    """
    # Calculate Euclidean distance between consecutive frames for each keypoint
    motion_data = np.sqrt(np.sum(np.diff(input_vis, axis=0)**2, axis=-1))  # Shape: (T-1, V)
    
    # Add a zero frame for the first frame to maintain the same number of frames
    motion_data = np.concatenate(([np.zeros(motion_data.shape[1])], motion_data), axis=0)  # Shape: (T, V)
    
    # Normalize motion data for visualization
    motion_data = (motion_data - motion_data.min()) / (motion_data.max() - motion_data.min() + 1e-8)
    
    # Transpose the motion data to have keypoints on y-axis and time on x-axis
    motion_data = motion_data.T

    return motion_data

def calculate_motion_data_without_outliers(input_vis, outlier_percentile=95):
    """
    Calculate motion data with filtering of extreme movements (top 5%).
    
    """
    # Calculate Euclidean distance between consecutive frames for each keypoint
    motion_data = np.sqrt(np.sum(np.diff(input_vis, axis=0)**2, axis=-1))
    
    # Add a zero frame for the first frame to maintain the same number of frames
    motion_data = np.concatenate(([np.zeros(motion_data.shape[1])], motion_data), axis=0)
    
    # Find the threshold value at the specified percentile
    threshold = np.percentile(motion_data, outlier_percentile)
    
    # Cap values above the threshold
    capped_motion = motion_data.copy()
    capped_motion[capped_motion > threshold] = threshold
    
    # Normalize the capped motion data for visualization
    capped_motion = (capped_motion - capped_motion.min()) / (capped_motion.max() - capped_motion.min() + 1e-8)
    
    # Transpose for visualization
    capped_motion = capped_motion.T
    
    return capped_motion

def plot_saliency(input_data, saliency_map, true_label, pred_label, save_path, model=None, clip_name=None, confidence=None):
    """
    Plot the saliency map and save it to a file.
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
    
    motion_data = calculate_motion_data_without_outliers(input_vis)

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
    if confidence is None and model is not None:
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
    
    if confidence is not None:
        plt.figtext(0.02, 0.02, f'Prediction confidence: {confidence:.2%}', 
                    fontsize=10, bbox=dict(facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def run_testing_for_fold(fold: int) -> bool:
    """
    Run the testing script for a specific fold to generate prediction files.
    This should be called after modifying the annotation file.

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

def print_test_clips_with_labels(annotation_file: str):
    """
    Print the contents of xsub_test from the annotation file along with their labels.
    """
    # Load the annotation file
    with open(annotation_file, 'rb') as f:
        annotations = pickle.load(f)
    
    # Get the test clips
    test_clips = annotations['split']['xsub_test']
    
    # Create a mapping of clip names to labels
    clip_to_label = {}
    for ann in annotations['annotations']:
        clip_name = ann['frame_dir']
        label = ann['label']
        clip_to_label[clip_name] = label
    
    # Print header
    print("\nTest clips in annotation file:")
    print(f"{'Index':<6} {'Clip Name':<50} {'Label':<10}")
    print("-" * 70)
    
    # Print each clip with its label
    for i, clip in enumerate(test_clips):
        label = clip_to_label.get(clip, "Unknown")
        label_str = "Seizure" if label == 1 else "Non-seizure" if label == 0 else "Unknown"
        print(f"{i:<6} {clip:<50} {label_str:<10}")
    
    # Print summary
    seizure_count = sum(1 for clip in test_clips if clip_to_label.get(clip) == 1)
    non_seizure_count = sum(1 for clip in test_clips if clip_to_label.get(clip) == 0)
    unknown_count = sum(1 for clip in test_clips if clip not in clip_to_label)
    
    print("-" * 70)
    print(f"Total clips: {len(test_clips)}")
    print(f"Seizure clips: {seizure_count}")
    print(f"Non-seizure clips: {non_seizure_count}")
    print(f"Unknown label clips: {unknown_count}")

def print_prediction_file_contents(fold: int):
    """
    Print the contents of the prediction file generated by the testing script.
    Shows prediction scores, ground truth labels, and prediction accuracy.
    
    Returns:
        dict: A mapping from sample index to prediction information
    """
    pred_file = f'k_fold/test_results/fold{fold}_predictions.pkl'
    
    # Check if prediction file exists
    if not os.path.exists(pred_file):
        print(f"Error: Prediction file {pred_file} not found.")
        return {}
    
    # Load the prediction file
    with open(pred_file, 'rb') as f:
        predictions = pickle.load(f)
    
    # Print header
    print("\nPrediction file contents:")
    print(f"{'Index':<6} {'GT Label':<10} {'Pred Label':<10} {'Confidence':<12} {'Frame Dir' if 'frame_dir' in predictions[0] else ''}")
    print("-" * 70)
    
    # Track statistics
    correct = 0
    total = len(predictions)
    confusion_matrix = {
        'true_positive': 0,  # Correctly predicted seizure
        'false_positive': 0, # Non-seizure predicted as seizure
        'true_negative': 0,  # Correctly predicted non-seizure
        'false_negative': 0  # Seizure predicted as non-seizure
    }
    
    # Create prediction mapping
    pred_mapping = {}
    
    # Print each prediction
    for i, pred in enumerate(predictions):
        gt_label = pred['gt_label'].item()
        
        # Get prediction scores
        if 'pred_score' in pred:
            scores = pred['pred_score']
            if isinstance(scores, torch.Tensor):
                scores = scores.cpu().numpy()
            
            # Handle different score formats
            if len(scores.shape) > 1:
                scores = scores.mean(axis=0)  # Average over multiple clips if needed
            
            # Get binary prediction
            if len(scores) > 2:
                # Convert to binary (class 0 vs rest)
                binary_scores = np.zeros(2)
                binary_scores[0] = scores[0]
                binary_scores[1] = np.sum(scores[1:])
                scores = binary_scores
            
            # Apply softmax to get probabilities
            exp_scores = np.exp(scores - np.max(scores))
            probs = exp_scores / exp_scores.sum()
            
            pred_label = np.argmax(probs)
            confidence = probs[pred_label]
        else:
            # If no scores available, use label directly
            pred_label = pred.get('pred_label', -1)
            confidence = 1.0  # Default confidence
        
        # Store in mapping
        pred_mapping[i] = {
            'gt_label': gt_label,
            'pred_label': pred_label,
            'confidence': confidence,
            'frame_dir': pred.get('frame_dir', '')
        }
        
        # Update statistics
        if gt_label == pred_label:
            correct += 1
            if gt_label == 1:
                confusion_matrix['true_positive'] += 1
            else:
                confusion_matrix['true_negative'] += 1
        else:
            if pred_label == 1:
                confusion_matrix['false_positive'] += 1
            else:
                confusion_matrix['false_negative'] += 1
        
        # Format labels for display
        gt_str = "Seizure" if gt_label == 1 else "Non-seizure"
        pred_str = "Seizure" if pred_label == 1 else "Non-seizure"
        
        # Print row
        frame_dir = pred.get('frame_dir', '')
        print(f"{i:<6} {gt_str:<10} {pred_str:<10} {confidence:.4f}      {frame_dir}")
    
    # Print summary
    print("-" * 70)
    accuracy = correct / total if total > 0 else 0
    print(f"Total predictions: {total}")
    print(f"Correct predictions: {correct}")
    print(f"Accuracy: {accuracy:.4f} ({correct}/{total})")
    
    # Print confusion matrix
    print("\nConfusion Matrix:")
    print(f"True Positive: {confusion_matrix['true_positive']} (correctly identified seizures)")
    print(f"False Positive: {confusion_matrix['false_positive']} (non-seizures misclassified as seizures)")
    print(f"True Negative: {confusion_matrix['true_negative']} (correctly identified non-seizures)")
    print(f"False Negative: {confusion_matrix['false_negative']} (seizures misclassified as non-seizures)")
    
    # Calculate additional metrics
    precision = 0
    recall = 0
    
    if confusion_matrix['true_positive'] + confusion_matrix['false_positive'] > 0:
        precision = confusion_matrix['true_positive'] / (confusion_matrix['true_positive'] + confusion_matrix['false_positive'])
        print(f"Precision: {precision:.4f}")
    
    if confusion_matrix['true_positive'] + confusion_matrix['false_negative'] > 0:
        recall = confusion_matrix['true_positive'] / (confusion_matrix['true_positive'] + confusion_matrix['false_negative'])
        print(f"Recall: {recall:.4f}")
    
    if precision + recall > 0:
        f1 = 2 * (precision * recall) / (precision + recall)
        print(f"F1 Score: {f1:.4f}")
    
    return pred_mapping

def process_fold_with_testing(fold: int, output_dir: Path):
    """
    Process a single fold with testing:
    1. Filter annotations for the specific video ID
    2. Run testing to generate predictions
    3. Generate saliency maps
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
        filtered_annotations = filter_annotations_by_video(current_video_id, config_file, video_output_dir)
        
        # Print the test clips with their labels
        annotation_file = get_annotation_file_from_config(config_file)
        print_test_clips_with_labels(annotation_file)
        
        # Run testing for this fold to generate prediction files
        testing_success = run_testing_for_fold(fold)
        if not testing_success:
            print(f"Warning: Testing failed for fold {fold}. Saliency maps may be incomplete.")
            return
        
        # Print the contents of the prediction file and get the prediction mapping
        pred_mapping = print_prediction_file_contents(fold)
        
        # Load model and all test data
        samples = load_model_and_data(fold)
        print(f"Number of samples: {len(samples)}")
        
        # Identify seizure clips
        seizure_clips = [(sample_idx, clip_name) for _, _, true_label, sample_idx, clip_name in samples if true_label == 1]
        print(f"Found {len(seizure_clips)} seizure clips in samples")
        
        # Process samples
        for model, input_data, true_label, sample_idx, clip_name in samples:
            print(f"Processing sample {sample_idx}")
            
            # Compute saliency map
            saliency_map = compute_saliency_map(model, input_data)
            
            # Use the prediction from the file instead of recomputing
            if sample_idx in pred_mapping:
                pred_info = pred_mapping[sample_idx]
                pred_label = pred_info['pred_label']
                confidence = pred_info['confidence']
            else:
                # Fallback to computing prediction if not found in mapping
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
                    confidence = float(pred_probs[pred_label])
            
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
                clip_name,
                confidence
            )
            
            print(f"Generated saliency map for {'clip ' + clip_name if clip_name else 'sample ' + str(sample_idx)}")
        
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