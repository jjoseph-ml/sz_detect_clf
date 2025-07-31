#!/usr/bin/env python
"""
Script to test all videos using the best fold's model.
This script calls the MMAction2 tools/test.py script with the appropriate config files.
"""

import os
import sys
import subprocess
import argparse
from datetime import datetime
from pathlib import Path
import torch
from typing import List
import pickle

# Add MMAction2 to Python path if needed
mmaction2_path = os.path.join(os.path.dirname(__file__), 'mmaction2')
if mmaction2_path not in sys.path:
    sys.path.append(mmaction2_path)

def check_gpu_availability() -> List[int]:
    """Check GPU availability and return list of available GPU indices."""
    if not torch.cuda.is_available():
        print("Warning: No GPU available - testing will use CPU")
        return []
    
    gpu_count = torch.cuda.device_count()
    gpu_indices = list(range(gpu_count))
    print(f"Found {gpu_count} GPU(s):")
    for i in gpu_indices:
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    return gpu_indices

def find_best_checkpoint(fold_dir: Path) -> Path:
    """Find the best checkpoint file in the fold directory."""
    checkpoint_files = list(fold_dir.glob("best_*.pth"))
    if not checkpoint_files:
        raise FileNotFoundError(f"No best checkpoint found in {fold_dir}")
    
    if len(checkpoint_files) > 1:
        print(f"Warning: Multiple best checkpoints found in {fold_dir}, using {checkpoint_files[0]}")
    
    return checkpoint_files[0]

def create_all_clips_annotation_file(best_fold: int) -> Path:
    """Create a combined annotation file with all non-augmented clips from clip_annotations.txt."""
    import pickle
    
    # Load the fold's annotation file to get the structure and existing annotations
    fold_annotation_file = Path(f'k_fold/data/skeleton/bcm_master_annotation_fold{best_fold}.pkl')
    
    if not fold_annotation_file.exists():
        raise FileNotFoundError(f"Fold annotation file not found: {fold_annotation_file}")
    
    print(f"Loading fold annotation structure from: {fold_annotation_file}")
    
    with open(fold_annotation_file, 'rb') as f:
        fold_data = pickle.load(f)
    
    # Read all clips from clip_annotations.txt
    clip_annotations_file = Path('preprocessing/video_annotations/clip_annotations.txt')
    
    if not clip_annotations_file.exists():
        raise FileNotFoundError(f"Clip annotations file not found: {clip_annotations_file}")
    
    print(f"Reading all clips from: {clip_annotations_file}")
    
    all_clips = []
    clip_to_label = {}
    
    with open(clip_annotations_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            parts = line.split()
            if len(parts) != 2:
                continue
            
            clip_path, label_str = parts
            clip_name = Path(clip_path).stem  # Remove .pkl extension
            
            # Skip augmented clips
            if '_aug' in clip_name:
                continue
            
            try:
                label = int(label_str)
                all_clips.append(clip_name)
                clip_to_label[clip_name] = label
            except ValueError:
                print(f"Warning: Invalid label '{label_str}' for clip {clip_name}")
                continue
    
    print(f"Found {len(all_clips)} non-augmented clips from clip_annotations.txt")
    
    # Create new annotation data with all clips in the test split
    # Build new annotations structure that preserves correct labels and includes keypoint data
    new_annotations = []
    print(f"Loading keypoint data for {len(all_clips)} clips...")
    
    for i, clip_name in enumerate(all_clips):
        if i % 1000 == 0:  # Progress indicator every 1000 clips
            print(f"  Processed {i}/{len(all_clips)} clips...")
            
        if clip_name in clip_to_label:
            original_label = clip_to_label[clip_name]
            # Convert label 9 (mixed frames) to label 1 (seizure) for the annotation file
            final_label = 1 if original_label == 9 else original_label
            
            # Load keypoint data from the individual pickle file
            keypoint_file = Path(f'preprocessing/clip_keypoints/{clip_name}.pkl')
            if keypoint_file.exists():
                try:
                    with open(keypoint_file, 'rb') as f:
                        keypoint_data = pickle.load(f)
                    
                    # Create annotation with keypoint data
                    annotation = {
                        'frame_dir': clip_name,
                        'label': final_label,
                        'img_shape': keypoint_data.get('img_shape', (480, 640)),
                        'original_shape': keypoint_data.get('original_shape', (480, 640)),
                        'total_frames': keypoint_data.get('total_frames', 90),
                        'keypoint': keypoint_data['keypoint'],
                        'keypoint_score': keypoint_data.get('keypoint_score', None)
                    }
                    new_annotations.append(annotation)
                except Exception as e:
                    print(f"Warning: Error loading keypoint data for {clip_name}: {e}")
                    # Fallback to basic annotation without keypoint data
                    annotation = {
                        'frame_dir': clip_name,
                        'label': final_label
                    }
                    new_annotations.append(annotation)
            else:
                print(f"Warning: Keypoint file not found for {clip_name}")
                # Fallback to basic annotation without keypoint data
                annotation = {
                    'frame_dir': clip_name,
                    'label': final_label
                }
                new_annotations.append(annotation)
        else:
            print(f"Warning: Clip {clip_name} not found in clip_to_label mapping")
    
    # Log the label distribution for debugging
    final_label_counts = {}
    original_label_counts = {}
    for annotation in new_annotations:
        final_label = annotation['label']
        final_label_counts[final_label] = final_label_counts.get(final_label, 0) + 1
        
        # Get original label for comparison
        clip_name = annotation['frame_dir']
        if clip_name in clip_to_label:
            original_label = clip_to_label[clip_name]
            original_label_counts[original_label] = original_label_counts.get(original_label, 0) + 1
    
    print(f"Original label distribution: {original_label_counts}")
    print(f"Final annotation label distribution (label 9 → 1): {final_label_counts}")
    
    combined_data = {
        'split': {
            'xsub_train': [],
            'xsub_val': [],
            'xsub_test': all_clips  # All non-augmented clips go to test split
        },
        'annotations': new_annotations  # Use new annotations with correct labels
    }
    
    # Save combined annotation file in the same directory as original
    combined_file = fold_annotation_file.parent / f"bcm_master_annotation_all.pkl"
    
    print(f"Creating combined annotation file: {combined_file}")
    print(f"Test split will contain {len(all_clips)} clips")
    
    with open(combined_file, 'wb') as f:
        pickle.dump(combined_data, f)
    
    print(f"Combined annotation file created with all non-augmented clips")
    print(f"Note: Using existing annotations structure from fold {best_fold}")
    
    return combined_file

def modify_config_for_all_videos(config_file: Path, best_fold: int) -> Path:
    """Modify config file to use all videos annotation file."""
    annotation_file = create_all_clips_annotation_file(best_fold)
    # Keep config files in original location to avoid import issues
    modified_config = config_file.parent / f"stgcnpp_all_videos.py"
    
    # Read the original config
    with open(config_file, 'r') as f:
        config_content = f.read()
    
    # Replace the annotation file path to use the all videos annotation file
    old_annotation_pattern = f"bcm_master_annotation_fold{best_fold}.pkl"
    new_annotation_pattern = annotation_file.name
    
    modified_content = config_content.replace(old_annotation_pattern, new_annotation_pattern)
    
    # Write the modified config
    with open(modified_config, 'w') as f:
        f.write(modified_content)
    
    print(f"Created modified config: {modified_config}")
    return modified_config

def enhance_predictions_with_clip_names(predictions_file: Path, annotation_file: Path, output_file: Path):
    """Enhance the raw predictions with clip names for easy linking."""
    print(f"Enhancing predictions with clip names...")
    
    # Validate input files exist
    if not predictions_file.exists():
        raise FileNotFoundError(f"Predictions file not found: {predictions_file}")
    
    if not annotation_file.exists():
        raise FileNotFoundError(f"Annotation file not found: {annotation_file}")
    
    # Load the raw predictions
    with open(predictions_file, 'rb') as f:
        raw_predictions = pickle.load(f)
    
    if not raw_predictions:
        raise ValueError(f"No predictions found in {predictions_file}")
    
    # Load the annotation file to get clip names
    with open(annotation_file, 'rb') as f:
        annotation_data = pickle.load(f)
    
    # Get clip names from the test split (which contains all clips)
    clip_names = annotation_data['split']['xsub_test']
    
    # Verify we have the same number of predictions and clips
    if len(raw_predictions) != len(clip_names):
        print(f"Warning: Number of predictions ({len(raw_predictions)}) doesn't match number of clips ({len(clip_names)})")
        num_to_process = min(len(raw_predictions), len(clip_names))
    else:
        num_to_process = len(raw_predictions)
    
    print(f"Processing {num_to_process} predictions with clip names")
    
    # Create enhanced predictions with clip names
    enhanced_predictions = []
    for i in range(num_to_process):
        clip_name = clip_names[i]
        pred = raw_predictions[i]
        
        # Create enhanced prediction with clip name
        enhanced_pred = {
            'clip_name': clip_name,
            'gt_label': pred['gt_label'],
            'pred_label': pred['pred_label'],
            'pred_score': pred['pred_score']
        }
        enhanced_predictions.append(enhanced_pred)
    
    # Save enhanced predictions
    with open(output_file, 'wb') as f:
        pickle.dump(enhanced_predictions, f)
    
    print(f"Enhanced predictions saved to: {output_file}")
    print(f"Format: List of dictionaries with 'clip_name', 'gt_label', 'pred_label', 'pred_score'")
    
    return enhanced_predictions

def save_predictions_as_csv(enhanced_predictions, csv_output_file: Path):
    """Save enhanced predictions as a CSV file for easy analysis."""
    import pandas as pd
    import numpy as np
    
    print(f"Saving predictions as CSV...")
    
    # Convert predictions to a format suitable for CSV
    csv_data = []
    for pred in enhanced_predictions:
        # Extract clip information
        clip_name = pred['clip_name']
        
        # Parse clip name to extract components
        parts = clip_name.split('_')
        if len(parts) >= 3:
            patient_id = parts[0]
            video_id = parts[1]
            seg_part = parts[2] if len(parts) == 3 else parts[2] + '_' + parts[3]
            is_augmented = '_aug' in seg_part
            seg_num = seg_part.replace('Seg_', '').replace('_aug', '')
        else:
            patient_id = 'unknown'
            video_id = 'unknown'
            seg_num = 'unknown'
            is_augmented = False
        
        # Convert tensors to values
        gt_label = pred['gt_label'].item() if hasattr(pred['gt_label'], 'item') else pred['gt_label']
        pred_label = pred['pred_label'].item() if hasattr(pred['pred_label'], 'item') else pred['pred_label']
        
        # Get prediction scores
        scores = pred['pred_score']
        if hasattr(scores, 'cpu'):
            scores = scores.cpu().numpy()
        elif hasattr(scores, 'numpy'):
            scores = scores.numpy()
        
        # Ensure scores is a numpy array
        if not isinstance(scores, np.ndarray):
            scores = np.array(scores)
        
        # Get confidence for predicted class
        confidence = float(scores[pred_label])
        
        # Calculate if prediction is correct
        correct = 1 if gt_label == pred_label else 0
        
        # Create row data
        row = {
            'clip_name': clip_name,
            'patient_id': patient_id,
            'video_id': video_id,
            'segment_num': seg_num,
            'is_augmented': is_augmented,
            'ground_truth': gt_label,
            'predicted': pred_label,
            'confidence': confidence,
            'score_class_0': float(scores[0]),
            'score_class_1': float(scores[1]),
            'correct': correct
        }
        csv_data.append(row)
    
    # Create DataFrame and save to CSV
    df = pd.DataFrame(csv_data)
    df.to_csv(csv_output_file, index=False)
    
    print(f"CSV predictions saved to: {csv_output_file}")
    print(f"CSV columns: {list(df.columns)}")
    
    # Print summary statistics
    total_predictions = len(df)
    accuracy = df['correct'].mean()
    print(f"\nSummary:")
    print(f"Total predictions: {total_predictions}")
    print(f"Overall accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"Correct predictions: {df['correct'].sum()}")
    print(f"Incorrect predictions: {total_predictions - df['correct'].sum()}")
    
    return df

def main():
    parser = argparse.ArgumentParser(description='Run testing using specified best model on all videos')
    parser.add_argument('--checkpoint', type=str, required=True,
                      help='Path to the best model checkpoint file')
    parser.add_argument('--dry-run', action='store_true', 
                      help='Print commands without executing them')
    args, test_args = parser.parse_known_args()
    
    # Use the direct path to test.py in mmaction2
    test_script = os.path.join(mmaction2_path, 'tools', 'test.py')
    
    # Hardcoded paths
    config_dir = Path('k_fold/stgcn')
    work_dir_base = Path('k_fold/work_dirs')
    
    # Always use all_video_testing as base directory
    test_results_dir = Path('all_video_testing')
    
    # Create subdirectory for results
    results_dir = test_results_dir / 'all'
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Use provided checkpoint
    checkpoint_file = Path(args.checkpoint)
    
    if not checkpoint_file.exists():
        print(f"Error: Model checkpoint file not found: {checkpoint_file}")
        return 1
    
    # Find any available config file (since all configs are the same)
    config_files = list(config_dir.glob("stgcnpp_fold*.py"))
    if not config_files:
        print(f"Error: No config files found in {config_dir}")
        return 1
    
    # Use the first available config file
    best_fold_config = config_files[0]
    print(f"Using config file: {best_fold_config}")
    print(f"Using model checkpoint: {checkpoint_file}")
    
    # Extract fold number from config filename for annotation file creation
    config_name = best_fold_config.stem
    if 'fold' in config_name:
        best_fold = int(config_name.split('fold')[-1])
    else:
        # If no fold number in config name, use 0 as default
        best_fold = 0
        print(f"Warning: Could not extract fold number from config name, using fold {best_fold}")
    
    # Modify config to use all videos
    modified_config = modify_config_for_all_videos(best_fold_config, best_fold)
    
    # Set up output files
    raw_output_file = results_dir / f"best_model_all_predictions_raw.pkl"
    enhanced_output_file = results_dir / f"best_model_all_predictions.pkl"
    csv_output_file = results_dir / "all_predictions.csv"
    
    print(f"\n{'='*80}")
    print(f"Testing all videos using specified best model")
    print(f"  Config: {modified_config}")
    print(f"  Checkpoint: {checkpoint_file}")
    print(f"  Raw Output: {raw_output_file}")
    print(f"  Enhanced Output: {enhanced_output_file}")
    print(f"{'='*80}")
    
    # Check if tools/test.py exists
    if not os.path.isfile(test_script):
        print(f"Error: Testing script '{test_script}' not found")
        return 1
    
    # Check GPU availability
    available_gpus = check_gpu_availability()
    
    # Set environment variables for GPU testing
    if available_gpus:
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, available_gpus))
    
    # Build command
    cmd = [
        'python', test_script,
        str(modified_config),
        str(checkpoint_file),
        '--dump', str(raw_output_file)
    ] + test_args
    
    print(f"Running command: {' '.join(cmd)}")
    
    if args.dry_run:
        print("Dry run - command not executed")
        return 0
    
    # Run testing
    start_time = datetime.now()
    print(f"Starting testing at {start_time}")
    
    try:
        # Set PYTHONPATH to include mmaction2 directory
        env = os.environ.copy()
        env['PYTHONPATH'] = f"{mmaction2_path}:{env.get('PYTHONPATH', '')}"
        
        # Run the testing process
        result = subprocess.run(cmd, check=True, env=env)
        
        # Enhance predictions with clip names
        annotation_file = Path(f'k_fold/data/skeleton/bcm_master_annotation_all.pkl')
        enhanced_predictions = enhance_predictions_with_clip_names(raw_output_file, annotation_file, enhanced_output_file)
        
        # Save predictions as CSV
        save_predictions_as_csv(enhanced_predictions, csv_output_file)
        
        end_time = datetime.now()
        duration = end_time - start_time
        print(f"Testing completed successfully in {duration}")
        print(f"Raw results saved to: {raw_output_file}")
        print(f"Enhanced results saved to: {enhanced_output_file}")
        
    except subprocess.CalledProcessError as e:
        print(f"Error running testing: {e}")
        return 1
    
    return 0

if __name__ == '__main__':
    sys.exit(main()) 