#!/usr/bin/env python
"""
Model loading functions for saliency analysis.
"""

import os
import torch
import pickle
from pathlib import Path
import mmcv
from mmengine.config import Config
from mmaction.apis import init_recognizer
from mmengine.runner import Runner
import shutil
import re

from .utils import get_annotation_file_from_config

def find_best_model(fold_dir: Path) -> Path:
    """Find the best model checkpoint in the fold directory."""
    checkpoints = list(fold_dir.glob('best_acc_top1_epoch*.pth'))
    if not checkpoints:
        raise FileNotFoundError(f"No best model checkpoint found in {fold_dir}")
    
    if len(checkpoints) > 1:
        # If multiple checkpoints exist, get the one with the highest epoch number
        return max(checkpoints, key=lambda x: int(str(x).split('epoch_')[-1].split('.')[0]))
    
    return checkpoints[0]

def load_best_fold_model(best_fold: int):
    """Load the best fold model using the all videos config."""
    config_file = 'k_fold/stgcn/stgcnpp_all_videos.py'
    
    fold_dir = Path(f'k_fold/work_dirs/fold{best_fold}')
    checkpoint = find_best_model(fold_dir)
    
    cfg = Config.fromfile(config_file)
    model = init_recognizer(cfg, str(checkpoint), device='cuda')
    model.eval()
    print(f"Loaded model from {checkpoint} using {config_file}")
    return model

def update_config_annotation_file(config_file: str, new_annotation_file: str):
    """Update the annotation file path in the config file."""
    # Read the config file
    with open(config_file, 'r') as f:
        config_content = f.read()
    
    # Find and replace the annotation file line
    # Look for lines like: ann_file = 'path/to/annotation.pkl'
    pattern = r"ann_file\s*=\s*['\"][^'\"]*['\"]"
    replacement = f"ann_file = '{new_annotation_file}'"
    
    updated_content = re.sub(pattern, replacement, config_content)
    
    # Write the updated config back
    with open(config_file, 'w') as f:
        f.write(updated_content)
    
    print(f"Updated config file {config_file} to use annotation file: {new_annotation_file}")

def load_model_and_data_for_video_specific(fold: int, expected_clips: list) -> list:
    """
    Load model and test data for a specific set of clips from the general annotation file.
    """
    # Find best model checkpoint
    fold_dir = Path(f'k_fold/work_dirs/fold{fold}')
    try:
        checkpoint = find_best_model(fold_dir)
        print(f"Using checkpoint: {checkpoint}")
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Error for fold {fold}: {e}")
    
    # Use all videos config file
    config_file = 'k_fold/stgcn/stgcnpp_all_videos.py'
    
    # Load the config file
    cfg = Config.fromfile(config_file)
    
    # Initialize model using MMAction2 API
    model = init_recognizer(cfg, str(checkpoint), device='cuda')
    model.eval()
    
    # Build the test dataset from config
    test_dataloader = Runner.build_dataloader(cfg.test_dataloader)
    
    # Get the clip names from the annotation file
    annotation_file = get_annotation_file_from_config(config_file)
    with open(annotation_file, 'rb') as f:
        annotations = pickle.load(f)
    all_clip_names = annotations['split']['xsub_test']
    
    print(f"Total clips in annotation: {len(all_clip_names)}")
    print(f"Expected clips: {len(expected_clips)}")
    
    # Create a mapping of clip names to their indices in the full annotation
    clip_to_index = {}
    for i, clip_name in enumerate(all_clip_names):
        clip_to_index[clip_name] = i
    
    # Get the indices for our expected clips
    expected_indices = [clip_to_index[clip_name] for clip_name in expected_clips]
    expected_indices.sort()  # Sort to maintain order
    
    results = []
    for sample_idx in expected_indices:
        clip_name = all_clip_names[sample_idx]
        
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
        
        # Get ground truth label from annotation
        true_label = None
        for annotation in annotations['annotations']:
            if annotation['frame_dir'] == clip_name:
                true_label = annotation['label']
                break
        
        if true_label is None:
            print(f"Warning: Could not find label for clip {clip_name}")
            continue
        
        results.append((model, input_data, true_label, sample_idx, clip_name))
    
    print(f"Loaded {len(results)} samples")
    return results

def load_video_test_data(video_id: str, best_fold: int):
    """Load test data for a specific video from the all videos annotation file."""
    config_file = 'k_fold/stgcn/stgcnpp_all_videos.py'
    
    annotation_file = get_annotation_file_from_config(config_file)
    
    if not os.path.exists(annotation_file):
        raise FileNotFoundError(f"Annotation file not found: {annotation_file}")
    
    # Load the annotation file
    with open(annotation_file, 'rb') as f:
        annotations = pickle.load(f)
    
    # Filter clips for this specific video
    video_clips = []
    for clip_name in annotations['split']['xsub_test']:
        if video_id in clip_name:
            video_clips.append(clip_name)
    
    print(f"Found {len(video_clips)} clips for video {video_id} in all videos annotation")
    
    if len(video_clips) == 0:
        raise ValueError(f"No clips found for video {video_id} in all videos annotation")
    
    # Create a mapping of clip names to their indices in the original annotation
    clip_to_index = {}
    for i, clip_name in enumerate(annotations['split']['xsub_test']):
        clip_to_index[clip_name] = i
    
    # Get the indices for our video clips
    video_clip_indices = [clip_to_index[clip_name] for clip_name in video_clips]
    
    # Load model and test data using the general config
    samples = load_model_and_data_for_video_specific(best_fold, video_clips)
    
    return samples

def load_model_from_checkpoint(checkpoint_path: str):
    """Load model from a specific checkpoint file."""
    config_file = 'k_fold/stgcn/stgcnpp_all_videos.py'
    
    cfg = Config.fromfile(config_file)
    model = init_recognizer(cfg, checkpoint_path, device='cuda')
    model.eval()
    print(f"Loaded model from {checkpoint_path} using {config_file}")
    return model

def load_model_and_data_for_video_specific_with_checkpoint(checkpoint_path: str, expected_clips: list) -> list:
    """
    Load model and test data for a specific set of clips using a checkpoint file.
    """
    # Use all videos config file
    config_file = 'k_fold/stgcn/stgcnpp_all_videos.py'
    
    # Load the config file
    cfg = Config.fromfile(config_file)
    
    # Initialize model using MMAction2 API with the checkpoint
    model = init_recognizer(cfg, checkpoint_path, device='cuda')
    model.eval()
    
    # Build the test dataset from config
    test_dataloader = Runner.build_dataloader(cfg.test_dataloader)
    
    # Get the clip names from the annotation file
    annotation_file = get_annotation_file_from_config(config_file)
    with open(annotation_file, 'rb') as f:
        annotations = pickle.load(f)
    all_clip_names = annotations['split']['xsub_test']
    
    print(f"Total clips in annotation: {len(all_clip_names)}")
    print(f"Expected clips: {len(expected_clips)}")
    
    # Create a mapping of clip names to their indices in the full annotation
    clip_to_index = {}
    for i, clip_name in enumerate(all_clip_names):
        clip_to_index[clip_name] = i
    
    # Get the indices for our expected clips
    expected_indices = [clip_to_index[clip_name] for clip_name in expected_clips]
    expected_indices_set = set(expected_indices)  # For fast lookup
    expected_indices.sort()  # Sort to maintain order
    
    # Create a mapping of clip names to their labels for fast lookup
    clip_to_label = {}
    for annotation in annotations['annotations']:
        clip_to_label[annotation['frame_dir']] = annotation['label']
    
    # Collect all needed samples in a single pass through the dataloader
    results = []
    print(f"Loading {len(expected_indices)} samples from dataloader (this may take a moment)...")
    
    for i, data_batch in enumerate(test_dataloader):
        if i in expected_indices_set:
            clip_name = all_clip_names[i]
            
            # Get the input data
            input_data = data_batch['inputs']
            if isinstance(input_data, (list, tuple)):
                input_data = input_data[0]
            
            # Reshape input data
            if isinstance(input_data, torch.Tensor):
                input_data = input_data[0:1]  # Keep only first clip
                input_data = input_data.unsqueeze(1)
                input_data = input_data.cuda()
            
            # Get ground truth label from annotation
            true_label = clip_to_label.get(clip_name)
            
            if true_label is None:
                print(f"Warning: Could not find label for clip {clip_name}")
                continue
            
            results.append((model, input_data, true_label, i, clip_name))
            
            # Progress indicator
            if len(results) % 50 == 0:
                print(f"      Progress: {len(results)}/{len(expected_indices)} samples loaded...")
            
            # Early exit if we've collected all needed samples
            if len(results) >= len(expected_indices):
                break
    
    # Sort results by original expected_clips order
    clip_order = {clip_name: idx for idx, clip_name in enumerate(expected_clips)}
    results.sort(key=lambda x: clip_order.get(x[4], 999999))
    
    print(f"Loaded {len(results)} samples")
    return results

def load_video_test_data_with_checkpoint(video_id: str, checkpoint_path: str):
    """Load test data for a specific video using a checkpoint file."""
    config_file = 'k_fold/stgcn/stgcnpp_all_videos.py'
    
    annotation_file = get_annotation_file_from_config(config_file)
    
    if not os.path.exists(annotation_file):
        raise FileNotFoundError(f"Annotation file not found: {annotation_file}")
    
    # Load the annotation file
    with open(annotation_file, 'rb') as f:
        annotations = pickle.load(f)
    
    # Filter clips for this specific video
    video_clips = []
    for clip_name in annotations['split']['xsub_test']:
        if video_id in clip_name:
            video_clips.append(clip_name)
    
    print(f"Found {len(video_clips)} clips for video {video_id} in all videos annotation")
    
    if len(video_clips) == 0:
        raise ValueError(f"No clips found for video {video_id} in all videos annotation")
    
    # Create a mapping of clip names to their indices in the original annotation
    clip_to_index = {}
    for i, clip_name in enumerate(annotations['split']['xsub_test']):
        clip_to_index[clip_name] = i
    
    # Get the indices for our video clips
    video_clip_indices = [clip_to_index[clip_name] for clip_name in video_clips]
    
    # Load model and test data using the checkpoint
    samples = load_model_and_data_for_video_specific_with_checkpoint(checkpoint_path, video_clips)
    
    return samples 