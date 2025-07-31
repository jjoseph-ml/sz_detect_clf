import torch
import numpy as np
import pickle
from pathlib import Path
import mmcv
from mmengine.config import Config
import os
import json
import re
import subprocess
import cv2
import shutil
import matplotlib.pyplot as plt

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



def frame_extract(video_path):
    """Extract frames from a video file."""
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        
        cap.release()
        return frames
    except Exception as e:
        print(f"Error extracting frames from {video_path}:")
        import traceback
        traceback.print_exc()
        raise



def rdbu_color(val):
    """
    Convert a value in the range [-1, 1] to an RGB color using the RdBu colormap.
    Positive values are red, negative values are blue, and zero is white.
    This matches matplotlib's RdBu_r colormap.
    """
    val = max(-1, min(1, val))  # Clamp to [-1, 1]
    
    # RdBu colormap colors
    if val < 0:  # Blue for negative
        # Scale from white to blue
        intensity = -val
        r = 1.0 - intensity * 0.8
        g = 1.0 - intensity * 0.8
        b = 1.0
    elif val > 0:  # Red for positive
        # Scale from white to red
        intensity = val
        r = 1.0
        g = 1.0 - intensity * 0.8
        b = 1.0 - intensity * 0.8
    else:  # White for zero
        r, g, b = 1.0, 1.0, 1.0
    
    # Convert to 0-255 range for OpenCV
    return (int(b * 255), int(g * 255), int(r * 255))

def load_visibility_data():
    """
    Load visibility data from the CSV file.
    Returns a dictionary mapping clip names to their low visibility percentage.
    """
    visibility_data = {}
    csv_path = "preprocessing/visibility_analysis_clips.csv"
    
    if not os.path.exists(csv_path):
        print(f"Warning: Visibility data file {csv_path} not found")
        return visibility_data
    
    try:
        with open(csv_path, 'r') as f:
            # Skip header
            next(f)
            for line in f:
                parts = line.strip().split(',')
                if len(parts) >= 7:
                    # Extract clip name without .pkl extension
                    clip_name = parts[0].replace('.pkl', '')
                    low_vis_percentage = float(parts[6])
                    visibility_data[clip_name] = low_vis_percentage
        
        print(f"Loaded visibility data for {len(visibility_data)} clips")
        return visibility_data
    except Exception as e:
        print(f"Error loading visibility data: {e}")
        return {}

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
        tuple: A tuple containing:
            - dict: A mapping from sample index to prediction information.
            - float or None: The calculated accuracy percentage, or None if no predictions.
    """
    pred_file = f'k_fold/test_results/fold{fold}_predictions.pkl'
    if not os.path.exists(pred_file):
        print(f"Prediction file not found: {pred_file}")
        return {}, None # Return empty mapping and None for accuracy

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
        
        # Get prediction scores and label directly from the prediction file
        if 'pred_score' in pred and 'pred_label' in pred:
            scores = pred['pred_score']
            if isinstance(scores, torch.Tensor):
                scores = scores.cpu().numpy()
            
            # Get the predicted label from the file
            pred_label = pred['pred_label'].item()
            
            # Get confidence directly from the scores
            confidence = float(scores[pred_label])
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
    accuracy_percent = accuracy * 100 if total > 0 else None # Calculate percentage
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
    
    return pred_mapping, accuracy_percent # Return mapping and accuracy percentage

