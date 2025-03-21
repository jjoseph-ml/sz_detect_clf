#!/usr/bin/env python
"""
Script to generate and visualize saliency maps for the trained models.
Uses trained models from k-fold cross validation to show which input features
are most important for predictions.
"""

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

def find_best_model(fold_dir: Path) -> Path:
    """
    Find the best model checkpoint in the fold directory.
    
    Args:
        fold_dir: Directory containing model checkpoints
    
    Returns:
        Path to the best model checkpoint
    """
    checkpoints = list(fold_dir.glob('best_acc_top1_epoch*.pth'))
    if not checkpoints:
        raise FileNotFoundError(f"No best model checkpoint found in {fold_dir}")
    
    if len(checkpoints) > 1:
        # If multiple checkpoints exist, get the one with the highest epoch number
        return max(checkpoints, key=lambda x: int(str(x).split('epoch_')[-1].split('.')[0]))
    
    return checkpoints[0]

def load_model_and_data(fold: int, num_random_samples: int = 5) -> list:
    """
    Load model and randomly selected test data for a specific fold.
    
    Args:
        fold: Fold number
        num_random_samples: Number of random samples to select from test set
    
    Returns:
        list of tuples: [(model, input_data, true_label, sample_idx), ...]
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
    filtered_annotations = filter_annotations_by_video(video_ids[0], config_file)
    
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
    
    # Randomly select samples
    total_samples = len(predictions)
    random_indices = np.random.choice(total_samples, num_random_samples, replace=False)
    print(f"Selected random sample indices: {random_indices}")
    
    results = []
    for sample_idx in random_indices:
        # Get to the specific sample in dataloader
        for i, data_batch in enumerate(test_dataloader):
            #print(f"data batch : {data_batch}")
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
        
        results.append((model, input_data, true_label, sample_idx))
    
    return results

def compute_saliency_map(model, input_tensor):
    """
    Compute saliency map for given input and model.
    
    Args:
        model: MMAction2 model
        input_tensor: Input tensor with shape (N, M, T, V, C)
    
    Returns:
        Saliency map as numpy array
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

def plot_saliency(input_data, saliency_map, true_label, pred_label, save_path, model, frame_dir=None):
    """
    Plot original input and saliency map side by side with time on x-axis.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # Extract detailed clip information from frame_dir
    clip_info = "Unknown Clip"
    if frame_dir:
        try:
            # Assuming frame_dir format: path/to/frames/patient_id_video_id_clip_id/
            clip_name = os.path.basename(frame_dir)
            parts = clip_name.split('_')
            if len(parts) >= 3:
                patient_id = parts[0]
                video_id = parts[1]
                clip_id = '_'.join(parts[2:])  # Join remaining parts in case of underscores
                clip_info = f"Patient: {patient_id}\nVideo: {video_id}\nClip: {clip_id}"
            else:
                clip_info = clip_name
        except Exception as e:
            print(f"Error parsing frame directory: {e}")
            clip_info = os.path.basename(frame_dir)
    
    # Create main title with detailed clip info
    clip_type = "Seizure" if true_label == 1 else "Non-Seizure"
    main_title = f"{clip_info}\nType: {clip_type}"
    fig.suptitle(main_title, fontsize=12, y=1.02)
    
    # Define keypoint groups according to COCO WholeBody
    keypoint_groups = {
        'Body': list(range(0, 17)),      # Body keypoints
        'Feet': list(range(17, 23)),     # Foot keypoints (3 per foot)
        'Face': list(range(23, 91)),     # Face keypoints (68 points)
        'Left Hand': list(range(91, 112)), # Left hand keypoints (21 points)
        'Right Hand': list(range(112, 133)) # Right hand keypoints (21 points)
    }
    
    # Detailed body keypoint labels
    body_keypoints = {
        0: 'nose',
        1: 'left_eye', 2: 'right_eye',
        3: 'left_ear', 4: 'right_ear',
        5: 'left_shoulder', 6: 'right_shoulder',
        7: 'left_elbow', 8: 'right_elbow',
        9: 'left_wrist', 10: 'right_wrist',
        11: 'left_hip', 12: 'right_hip',
        13: 'left_knee', 14: 'right_knee',
        15: 'left_ankle', 16: 'right_ankle'
    }
    
    # Plot original input - transposed to have time on x-axis
    input_vis = input_data.squeeze()
    input_vis = input_vis.detach().cpu().numpy()
    input_vis = np.sqrt(np.sum(input_vis**2, axis=-1))
    input_vis = (input_vis - input_vis.min()) / (input_vis.max() - input_vis.min() + 1e-8)
    
    # Transpose the data to swap axes
    input_vis = input_vis.T
    
    im1 = ax1.imshow(input_vis, aspect='auto', cmap='viridis')
    
    ax1.set_title('Input Skeleton Motion')
    ax1.set_ylabel('Keypoints')
    ax1.set_xlabel('Time (frames)')
    plt.colorbar(im1, ax=ax1)
    
    # Add frame numbers on x-axis
    ax1.set_xticks(np.arange(0, input_vis.shape[1], 10))
    ax1.set_xticklabels([f"{i}" for i in range(0, input_vis.shape[1], 10)])
    
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
    
    # Plot saliency map - transposed to have time on x-axis
    if saliency_map.ndim > 2:
        saliency_map = saliency_map.squeeze()
    saliency_norm = (saliency_map - saliency_map.min()) / (saliency_map.max() - saliency_map.min() + 1e-8)
    
    # Transpose the saliency map
    saliency_norm = saliency_norm.T
    
    im2 = ax2.imshow(saliency_norm, aspect='auto', cmap='hot')
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
    
    Args:
        config_file: Path to the configuration file.
    
    Returns:
        Path to the annotation file.
    """
    cfg = Config.fromfile(config_file)
    return cfg.ann_file

def get_videos_in_test_split(config_file: str) -> list:
    """
    Get video IDs from the test split in the annotation file.
    
    Args:
        config_file: Path to the configuration file.
    
    Returns:
        List of unique video IDs in the test split.
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

def filter_annotations_by_video(video_id: str, config_file: str):
    """
    Filter the annotation file to keep clips of only one specified video and save it back to the same file.
    
    Args:
        video_id: The video ID to filter by (e.g., '7953A100').
        config_file: Path to the configuration file to get the annotation file path.
    """
    # Get the annotation file path from the config
    annotation_file = get_annotation_file_from_config(config_file)
    
    # Load the annotation file
    with open(annotation_file, 'rb') as f:  # Open in binary mode for pickle
        annotations = pickle.load(f)
    
    # Filter clips for the specified video ID
    filtered_clips = []
    for clip in annotations['split']['xsub_test']:
        if video_id in clip:
            filtered_clips.append(clip)
    
    # Sort the filtered clips
    filtered_clips.sort()
    
    # Create a new annotations structure
    filtered_annotations = {
        'split': {
            'xsub_test': filtered_clips,
            'xsub_train': annotations['split'].get('xsub_train', []),  # Keep other splits unchanged
            # You can add other splits if needed
        },
        'annotations': []
    }
    
    # Filter the annotations to keep only those for the specified video ID
    for annotation in annotations['annotations']:
        if video_id in annotation['frame_dir']:
            filtered_annotations['annotations'].append(annotation)
    
    # Save the filtered annotations back to the same file
    with open(annotation_file, 'wb') as f:
        pickle.dump(filtered_annotations, f)
    
    print(f"Filtered annotations saved to {annotation_file} with clips for video ID: {video_id}")

def main():
    # Create output directory
    output_dir = Path('k_fold/saliency_maps')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process each fold
    n_folds = 3
    num_random_samples = 5  # Number of random samples to process per fold
    
    for fold in range(n_folds):
        print(f"\nProcessing fold {fold}...")
        
        try:
            # Load model and random samples
            samples = load_model_and_data(fold, num_random_samples)
            
            for model, input_data, true_label, sample_idx in samples:
                print(f"\nProcessing sample {sample_idx}")
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
                
                # Plot and save
                save_path = output_dir / f'saliency_map_fold_{fold}_sample_{sample_idx}.png'
                plot_saliency(
                    input_data,
                    saliency_map,
                    true_label,
                    pred_label,
                    save_path,
                    model
                )
                
                print(f"Generated saliency map for sample {sample_idx}")
            
        except Exception as e:
            print(f"Error processing fold {fold}: {e}")
            import traceback
            traceback.print_exc()
            continue

if __name__ == '__main__':
    main() 