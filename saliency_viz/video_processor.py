#!/usr/bin/env python
"""
Video processing functions for saliency analysis.
"""

import os
import json
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from typing import Dict, List, Tuple
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, 
    confusion_matrix, precision_score, recall_score
)

from .data_loader import filter_predictions_by_video, find_clip_data
from .model_loader import load_best_fold_model, load_video_test_data, load_model_from_checkpoint, load_video_test_data_with_checkpoint
from .saliency import compute_saliency_map
from .visualization import (
    plot_saliency,
    calculate_and_save_body_part_saliency, create_body_part_saliency_histogram
)
from .viz_saliency_video_overlay import (
    create_saliency_video, create_whole_video_aggregated_saliency
)
from .viz_saliency_timeline import plot_keypoint_saliency_over_time_whole_video


def calculate_comprehensive_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_scores: np.ndarray) -> Dict[str, float]:
    """
    Calculate comprehensive performance metrics matching 005d_run_fold_prediction_analysis.py.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels  
        y_scores: Prediction scores/probabilities for positive class
    
    Returns:
        Dictionary containing all metrics
    """
    # Calculate confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    # Calculate all metrics
    metrics = {
        'Accuracy': accuracy_score(y_true, y_pred),
        'F1_Score': f1_score(y_true, y_pred),
        'AUC_ROC': roc_auc_score(y_true, y_scores),
        'Sensitivity': tp / (tp + fn) if (tp + fn) > 0 else 0,  # True Positive Rate / Recall
        'Specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,  # True Negative Rate
        'PPV': tp / (tp + fp) if (tp + fp) > 0 else 0,  # Positive Predictive Value / Precision
        'NPV': tn / (tn + fn) if (tn + fn) > 0 else 0,  # Negative Predictive Value
        'Balanced_Accuracy': (tp / (tp + fn) + tn / (tn + fp)) / 2 if (tp + fn) > 0 and (tn + fp) > 0 else 0,
        'TP': int(tp),
        'TN': int(tn),
        'FP': int(fp),
        'FN': int(fn),
        'Total': len(y_true)
    }
    
    return metrics


def calculate_video_metrics(video_predictions: pd.DataFrame) -> Dict[str, float]:
    """
    Calculate comprehensive metrics for a single video.
    
    Args:
        video_predictions: DataFrame with 'ground_truth', 'predicted', 'confidence' columns
    
    Returns:
        Dictionary containing all metrics
    """
    if len(video_predictions) == 0:
        return None
    
    # Extract arrays
    y_true = video_predictions['ground_truth'].values
    y_pred = video_predictions['predicted'].values
    y_scores = video_predictions['confidence'].values
    
    # Calculate metrics
    metrics = calculate_comprehensive_metrics(y_true, y_pred, y_scores)
    
    # Add class distribution
    class_counts = video_predictions['ground_truth'].value_counts().to_dict()
    metrics['Class_Distribution'] = {
        0: class_counts.get(0, 0),
        1: class_counts.get(1, 0)
    }
    
    return metrics


def process_video_by_task(best_fold: int, video_id: str, predictions_df, output_dir: Path, task='all'):
    """
    Process a single video for specific tasks.
    
    Args:
        best_fold: The fold number to use
        video_id: The video ID to process
        predictions_df: DataFrame with predictions
        output_dir: Output directory
        task: The task to perform:
            - 'histograms': Create body part saliency histograms
            - 'saliency_maps': Create static PNG plots of saliency maps
            - 'saliency_videos': Create individual clip videos AND whole video visualization
            - 'whole_video': Create only whole video visualization
            - 'all': Create all outputs
    """
    print(f"Processing video ID: {video_id} for fold {best_fold} (task: {task})")
    
    # Filter predictions for this video
    video_predictions = filter_predictions_by_video(predictions_df, video_id)
    
    if len(video_predictions) == 0:
        print(f"No predictions found for video {video_id}")
        return
    
    # Create organized output directory structure using new structure
    videos_dir = output_dir / "videos"
    video_output_dir = videos_dir / video_id
    video_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories for different output types
    saliency_maps_dir = video_output_dir / "saliency_maps"
    histograms_dir = video_output_dir / "histograms"
    clips_dir = histograms_dir / "clips"
    aggregated_dir = histograms_dir / "aggregated"
    videos_clips_dir = video_output_dir / "individual_clips"
    whole_video_dir = video_output_dir  # Put whole video at same level as individual_clips
    data_dir = video_output_dir / "data"
    
    # Create all directories
    for dir_path in [saliency_maps_dir, clips_dir, aggregated_dir, videos_clips_dir, data_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    # Load best fold model
    model = load_best_fold_model(best_fold)
    
    # Load test data for this video
    test_data = load_video_test_data(video_id, best_fold)
    print(f"Loaded {len(test_data)} test samples for video {video_id}")
    
    # Process each clip
    video_clips = {}
    processed_clips = 0
    
    for _, pred_row in video_predictions.iterrows():
        clip_name = pred_row['clip_name']
        
        # Find corresponding test data
        clip_data = find_clip_data(test_data, clip_name)
        if clip_data is None:
            print(f"Warning: No test data found for clip {clip_name}, skipping.")
            continue
        
        # Check if keypoint file exists
        keypoint_path = f"preprocessing/clip_keypoints/{clip_name}.pkl"
        if not os.path.exists(keypoint_path):
            print(f"Warning: Keypoint file not found for clip {clip_name}, skipping.")
            continue
        
        print(f"Processing clip {clip_name}")
        
        # Compute saliency map
        saliency_map = compute_saliency_map(model, clip_data['input_data'])
        
        # Store clip data with existing predictions
        video_clips[clip_name] = {
            'saliency_map': saliency_map,
            'true_label': pred_row['ground_truth'],
            'pred_label': pred_row['predicted'],
            'confidence': pred_row['confidence']
        }
        
        # Generate task-specific outputs
        if task in ['saliency_maps', 'all']:
            # Create save path for saliency maps
            save_path = saliency_maps_dir / f'{clip_name}.png'
            
            # Plot and save saliency map
            plot_saliency(
                clip_data['input_data'],
                saliency_map,
                pred_row['ground_truth'],
                pred_row['predicted'],
                save_path,
                model,
                clip_name,
                pred_row['confidence']
            )
        
        if task in ['saliency_videos', 'all']:
            # Create saliency video with prediction probability
            print(f"Creating saliency video for {clip_name}...")
            result = create_saliency_video(clip_name, saliency_map, videos_clips_dir, 
                                         pred_label=pred_row['predicted'], 
                                         confidence=pred_row['confidence'])
            if result:
                print(f"  ✓ Saliency video created: {result}")
            else:
                print(f"  ✗ Failed to create saliency video for {clip_name}")
        
        if task in ['histograms', 'all']:
            # Create body part saliency histogram (per-clip)
            create_body_part_saliency_histogram(
                clip_name, saliency_map, pred_row['ground_truth'], pred_row['predicted'], 
                clips_dir, confidence=pred_row['confidence']
            )
        
        processed_clips += 1
        print(f"Generated outputs for clip {clip_name}")
    
    print(f"Processed {processed_clips} clips for video {video_id}")
    
    # Calculate comprehensive video-level metrics
    video_metrics = calculate_video_metrics(video_predictions)
    
    if video_metrics:
        print(f"\nVideo Performance Metrics:")
        print(f"  Accuracy: {video_metrics['Accuracy']:.3f}")
        print(f"  F1 Score: {video_metrics['F1_Score']:.3f}")
        print(f"  AUC-ROC: {video_metrics['AUC_ROC']:.3f}")
        print(f"  Sensitivity: {video_metrics['Sensitivity']:.3f}")
        print(f"  Specificity: {video_metrics['Specificity']:.3f}")
        print(f"  Balanced Accuracy: {video_metrics['Balanced_Accuracy']:.3f}")
        print(f"  Class Distribution: Non-Seizure={video_metrics['Class_Distribution'][0]}, Seizure={video_metrics['Class_Distribution'][1]}")
        print(f"  Confusion Matrix: TP={video_metrics['TP']}, TN={video_metrics['TN']}, FP={video_metrics['FP']}, FN={video_metrics['FN']}")
    else:
        video_metrics = None
        print("No predictions available for metrics calculation")
    
    # Save video-level data
    if video_clips:
        # Save saliency data as JSON
        saliency_data = {
            'video_id': video_id,
            'fold': best_fold,
            'total_clips': len(video_clips),
            'clips': {}
        }
        
        # Add comprehensive metrics if available
        if video_metrics:
            saliency_data['metrics'] = video_metrics
        
        for clip_name, clip_data in video_clips.items():
            # Convert numpy arrays to lists for JSON serialization
            saliency_data['clips'][clip_name] = {
                'true_label': int(clip_data['true_label']),
                'pred_label': int(clip_data['pred_label']),
                'confidence': float(clip_data['confidence']),
                'saliency_shape': clip_data['saliency_map'].shape,
                'saliency_stats': {
                    'mean': float(clip_data['saliency_map'].mean()),
                    'std': float(clip_data['saliency_map'].std()),
                    'min': float(clip_data['saliency_map'].min()),
                    'max': float(clip_data['saliency_map'].max())
                }
            }
        
        # Save to data directory
        data_file = data_dir / f"saliency_data.json"
        with open(data_file, 'w') as f:
            json.dump(saliency_data, f, indent=2)
        
        # Create analysis summary
        summary_file = data_dir / f"analysis_summary.txt"
        with open(summary_file, 'w') as f:
            f.write(f"Saliency Analysis Summary\n")
            f.write(f"=======================\n")
            f.write(f"Video ID: {video_id}\n")
            f.write(f"Fold: {best_fold}\n")
            f.write(f"Total Clips: {len(video_clips)}\n")
            f.write(f"Tasks Completed: {task}\n")
            
            if video_metrics:
                f.write(f"\nPerformance Metrics:\n")
                f.write(f"  Accuracy: {video_metrics['Accuracy']:.3f}\n")
                f.write(f"  F1 Score: {video_metrics['F1_Score']:.3f}\n")
                f.write(f"  AUC-ROC: {video_metrics['AUC_ROC']:.3f}\n")
                f.write(f"  Sensitivity: {video_metrics['Sensitivity']:.3f}\n")
                f.write(f"  Specificity: {video_metrics['Specificity']:.3f}\n")
                f.write(f"  Balanced Accuracy: {video_metrics['Balanced_Accuracy']:.3f}\n")
                f.write(f"  PPV (Precision): {video_metrics['PPV']:.3f}\n")
                f.write(f"  NPV: {video_metrics['NPV']:.3f}\n")
                f.write(f"  Class Distribution: Non-Seizure={video_metrics['Class_Distribution'][0]}, Seizure={video_metrics['Class_Distribution'][1]}\n")
                f.write(f"  Confusion Matrix: TP={video_metrics['TP']}, TN={video_metrics['TN']}, FP={video_metrics['FP']}, FN={video_metrics['FN']}\n")
            
            f.write(f"\nOutput Directories:\n")
            f.write(f"- Saliency Maps: {saliency_maps_dir}\n")
            f.write(f"- Histograms: {histograms_dir}\n")
            f.write(f"- Individual Clips: {videos_clips_dir}\n")
            f.write(f"- Whole Video: {whole_video_dir}\n")
            f.write(f"- Data: {data_dir}\n")
    
    # Generate per-video histograms and whole video visualizations
    if video_clips:
        if task in ['histograms', 'all']:
            # Generate per-video body part saliency analysis
            print(f"\nCreating per-video body part saliency analysis for video {video_id}")
            calculate_and_save_body_part_saliency(
                best_fold, video_id, video_clips, aggregated_dir
            )
        
        if task in ['whole_video', 'saliency_videos', 'all']:
            print(f"\nCreating AGGREGATED saliency whole video visualization for fold {best_fold}, video {video_id}")
            # Generate three videos with different normalization modes
            normalization_modes = ["per_clip", "whole_video_per_keypoint", "per_clip_per_keypoint"]
            for mode in normalization_modes:
                try:
                    print(f"\n   Generating whole video with {mode} normalization...")
                    create_whole_video_aggregated_saliency(
                        best_fold, video_id, video_clips, whole_video_dir,
                        video_metrics=video_metrics, normalization_mode=mode
                    )
                    print(f"   ✅ Whole video ({mode} normalization) saved to: {whole_video_dir}")
                except Exception as e:
                    print(f"   ⚠️  Error generating whole video with {mode} normalization: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
    
    print(f"All outputs for video {video_id} saved to {video_output_dir}")

def create_analysis_directory_structure(output_dir: Path):
    """
    Create the analysis directory structure for cross-cutting analysis.
    
    Args:
        output_dir: Base output directory (k_fold/saliency_maps)
    """
    print(f"\nCreating analysis directory structure")
    
    # Create analysis directory
    analysis_dir = output_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories for different types of analysis
    fold_comparisons_dir = analysis_dir / "fold_comparisons"
    class_discrimination_dir = analysis_dir / "class_discrimination"
    statistical_summaries_dir = analysis_dir / "statistical_summaries"
    
    # Create all directories
    for dir_path in [fold_comparisons_dir, class_discrimination_dir, statistical_summaries_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    # Create README file explaining the structure
    readme_file = analysis_dir / "README.md"
    with open(readme_file, 'w') as f:
        f.write(f"# Saliency Analysis Directory Structure\n\n")
        f.write(f"This directory contains cross-cutting analysis across videos and patients.\n\n")
        f.write(f"## Directory Structure\n\n")
        f.write(f"### `fold_comparisons/`\n")
        f.write(f"- Comparison of saliency patterns across different folds\n")
        f.write(f"- Fold-to-fold consistency analysis\n")
        f.write(f"- Cross-validation stability metrics\n\n")
        f.write(f"### `class_discrimination/`\n")
        f.write(f"- Analysis of which body parts best discriminate between classes\n")
        f.write(f"- Feature importance rankings\n")
        f.write(f"- Class-specific saliency patterns\n\n")
        f.write(f"### `statistical_summaries/`\n")
        f.write(f"- Overall statistics across all videos/patients\n")
        f.write(f"- Performance metrics summaries\n")
        f.write(f"- Statistical significance tests\n\n")
        f.write(f"## Usage\n\n")
        f.write(f"These directories are populated when running comprehensive analysis across multiple videos and patients.\n")
    
    print(f"Created analysis directory structure at {analysis_dir}")


def create_per_patient_analysis(best_fold: int, predictions_df, output_dir: Path):
    """
    Create per-patient body part saliency analysis by aggregating data across all videos for each patient.
    This function should be called after processing all videos to create patient-level aggregations.
    
    Args:
        best_fold: The fold number
        predictions_df: Full predictions DataFrame with all videos
        output_dir: Base output directory (k_fold/saliency_maps)
    """
    print(f"\nCreating per-patient body part saliency analysis for fold {best_fold}")
    
    # Create patients directory using new structure
    patients_dir = output_dir / "patients"
    patients_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all unique patients
    patients = predictions_df['patient_id'].unique()
    print(f"Found {len(patients)} patients: {patients}")
    
    for patient_id in patients:
        print(f"\nProcessing patient {patient_id}")
        
        # Get all videos for this patient
        patient_videos = predictions_df[predictions_df['patient_id'] == patient_id]['video_id'].unique()
        print(f"  Found {len(patient_videos)} videos for patient {patient_id}: {patient_videos}")
        
        # Create patient-specific directory (without 'patient_' prefix)
        patient_dir = patients_dir / str(patient_id)
        patient_dir.mkdir(exist_ok=True)
        
        # Create subdirectories for patient-level analysis
        summary_dir = patient_dir / "summary"
        average_maps_dir = patient_dir / "average_maps"
        body_part_analysis_dir = patient_dir / "body_part_analysis"
        cross_video_comparison_dir = patient_dir / "cross_video_comparison"
        
        # Create all directories
        for dir_path in [summary_dir, average_maps_dir, body_part_analysis_dir, cross_video_comparison_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Create a summary of what per-patient analysis would contain
        summary_file = summary_dir / f"analysis_summary.txt"
        with open(summary_file, 'w') as f:
            f.write(f"Per-Patient Body Part Saliency Analysis Summary\n")
            f.write(f"==============================================\n")
            f.write(f"Patient ID: {patient_id}\n")
            f.write(f"Fold: {best_fold}\n")
            f.write(f"Total Videos: {len(patient_videos)}\n")
            f.write(f"Videos: {', '.join(patient_videos)}\n")
            f.write(f"\nDirectory Structure:\n")
            f.write(f"- Summary: {summary_dir}\n")
            f.write(f"- Average Maps: {average_maps_dir}\n")
            f.write(f"- Body Part Analysis: {body_part_analysis_dir}\n")
            f.write(f"- Cross-Video Comparison: {cross_video_comparison_dir}\n")
            f.write(f"\nNote: Full per-patient analysis requires processing all videos for this patient.\n")
            f.write(f"This would aggregate body part saliency data across all {len(patient_videos)} videos.\n")
            f.write(f"\nFuture implementation would include:\n")
            f.write(f"- Aggregated body part saliency across all patient videos\n")
            f.write(f"- Patient-level saliency patterns and trends\n")
            f.write(f"- Comparison of saliency patterns between patients\n")
            f.write(f"- Cross-video consistency analysis\n")
        
        # Create patient-level metadata
        metadata = {
            'patient_id': str(patient_id),
            'fold': best_fold,
            'total_videos': len(patient_videos),
            'videos': list(patient_videos),
            'analysis_status': 'summary_only',
            'created_date': str(Path().cwd()),
            'directories': {
                'summary': str(summary_dir),
                'average_maps': str(average_maps_dir),
                'body_part_analysis': str(body_part_analysis_dir),
                'cross_video_comparison': str(cross_video_comparison_dir)
            }
        }
        
        metadata_file = summary_dir / f"metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"  Created per-patient analysis structure for patient {patient_id}")
    
    print(f"Completed per-patient analysis setup for {len(patients)} patients")


def process_video_by_task_with_checkpoint(checkpoint_path: str, video_id: str, predictions_df, output_dir: Path, task='all', moving_avg_window=10):
    """
    Process a single video for saliency analysis using a specific checkpoint.
    This is a checkpoint-based version of process_video_by_task.
    
    Args:
        checkpoint_path: Path to the model checkpoint file
        video_id: Video ID to process
        predictions_df: DataFrame with predictions
        output_dir: Output directory
        task: Task to perform ('histograms', 'saliency_maps', 'saliency_videos', 'whole_video', 'keypoint_timeline', 'all')
        moving_avg_window: Window size for moving average smoothing in body part timeline plots (default: 10 frames)
    """
    print(f"\n{'='*60}")
    print(f"🎬 PROCESSING VIDEO: {video_id}")
    print(f"📁 Checkpoint: {checkpoint_path}")
    print(f"{'='*60}")
    
    print(f"\n📋 PHASE 1: SETUP & DATA PREPARATION")
    print(f"   1. Loading prediction data from CSV file...")
    
    # Filter predictions for this video
    video_predictions = filter_predictions_by_video(predictions_df, video_id)
    
    if len(video_predictions) == 0:
        print(f"❌ No predictions found for video {video_id}")
        return
    
    print(f"   2. Found {len(video_predictions)} clips with predictions")
    print(f"   3. Creating prediction mapping (clip_name → ground_truth, predicted, confidence)...")
    
    # Create a mapping from clip_name to prediction data from CSV
    clip_predictions = {}
    for _, row in video_predictions.iterrows():
        clip_predictions[row['clip_name']] = {
            'true_label': row['ground_truth'],
            'pred_label': row['predicted'],
            'confidence': row['confidence']
        }
    
    print(f"   4. Loading trained ST-GCN model from checkpoint...")
    # Load model from checkpoint
    model = load_model_from_checkpoint(checkpoint_path)
    
    print(f"   5. Loading keypoint test data for all video clips...")
    # Load test data for this video using the checkpoint
    test_data = load_video_test_data_with_checkpoint(video_id, checkpoint_path)
    
    if not test_data:
        print(f"No test data found for video {video_id}")
        return
    
    print(f"   6. Setting up output directory structure...")
    # Create video-specific output directory
    video_output_dir = output_dir / video_id
    video_output_dir.mkdir(parents=True, exist_ok=True)
    print(f"   ✅ Phase 1 complete: {len(test_data)} clips ready for processing")
    
    # Initialize video_clips variable (will be populated in various task sections)
    video_clips = None
    
    # Process based on task (keypoint plots can run independently at the end)
    if task in ['histograms', 'all']:
        print(f"\n📈 STEP 1/4: Creating body part saliency histograms")
        print(f"   Computing saliency maps for {len(test_data)} clips...")
        # Note: This function expects (fold, video_id, video_clips, output_dir)
        # We need to create a video_clips dictionary from test_data
        video_clips = {}
        for model_instance, input_data, true_label, sample_idx, clip_name in test_data:
            # Compute saliency map for this clip
            saliency_map = compute_saliency_map(model_instance, input_data)
            # Get prediction
            with torch.no_grad():
                pred_output = model_instance(input_data, return_loss=False)
                
                # Handle different output formats
                if isinstance(pred_output, (list, tuple)):
                    pred_output = pred_output[0]
                
                # Process the output to get class prediction
                # Standard processing for all mode
                output = pred_output.mean(dim=2)  # Average over temporal dimension
                output = output.squeeze(1)  # Remove batch dimension
                pred_scores = output.mean(dim=-1)  # Average over spatial dimension
                
                # Handle different class configurations
                if pred_scores.size(1) != 2:
                    # Convert to binary classification
                    binary_scores = torch.zeros((pred_scores.size(0), 2), device=pred_scores.device)
                    binary_scores[:, 0] = pred_scores[:, 0]
                    binary_scores[:, 1] = pred_scores[:, 1:].sum(dim=1)
                    pred_scores = binary_scores
                
                pred_label = torch.argmax(pred_scores, dim=1).item()
            
            # Use predictions from CSV instead of computing fresh
            if clip_name in clip_predictions:
                pred_data = clip_predictions[clip_name]
                video_clips[clip_name] = {
                    'true_label': pred_data['true_label'],
                    'pred_label': pred_data['pred_label'],
                    'saliency_map': saliency_map
                }
                # Debug output removed for cleaner logs
            else:
                print(f"    Warning: No prediction data found for clip {clip_name}")
                # Fallback to computed prediction if CSV data not available
                video_clips[clip_name] = {
                    'true_label': true_label,
                    'pred_label': pred_label,
                    'saliency_map': saliency_map
                }
        
        calculate_and_save_body_part_saliency(None, video_id, video_clips, video_output_dir)
        print(f"   ✅ Body part saliency histograms saved to: {video_output_dir}")
        # Note: create_body_part_saliency_histogram is for individual clips, not entire videos
        # The aggregated analysis above already provides video-level body part saliency
    
    if task in ['saliency_maps', 'all']:
        print(f"\n🖼️  STEP 2/4: Creating static saliency maps")
        print(f"   Generating plots for {len(test_data)} clips...")
        for model_instance, input_data, true_label, sample_idx, clip_name in test_data:
            saliency_map = compute_saliency_map(model_instance, input_data)
            # Use predictions from CSV for static plots
            if clip_name in clip_predictions:
                pred_data = clip_predictions[clip_name]
                save_path = video_output_dir / f"{clip_name}_saliency_plot.png"
                plot_saliency(input_data, saliency_map, pred_data['true_label'], pred_data['pred_label'], save_path, clip_name=clip_name, confidence=pred_data['confidence'])
            else:
                # Fallback to computed prediction
                with torch.no_grad():
                    pred_output = model_instance(input_data, return_loss=False)
                    if isinstance(pred_output, (list, tuple)):
                        pred_output = pred_output[0]
                    output = pred_output.mean(dim=2)
                    output = output.squeeze(1)
                    pred_scores = output.mean(dim=-1)
                    if pred_scores.size(1) != 2:
                        binary_scores = torch.zeros((pred_scores.size(0), 2), device=pred_scores.device)
                        binary_scores[:, 0] = pred_scores[:, 0]
                        binary_scores[:, 1] = pred_scores[:, 1:].sum(dim=1)
                        pred_scores = binary_scores
                    pred_label = torch.argmax(pred_scores, dim=1).item()
                    confidence = torch.softmax(pred_scores, dim=1).max().item()
                save_path = video_output_dir / f"{clip_name}_saliency_plot.png"
                plot_saliency(input_data, saliency_map, true_label, pred_label, save_path, clip_name=clip_name, confidence=confidence)
        print(f"   ✅ Static saliency maps saved to: {video_output_dir}")
    
    if task in ['saliency_videos', 'whole_video', 'all']:
        print(f"\n🎥 STEP 3/4: Creating individual saliency videos")
        print(f"   Processing {len(test_data)} clips...")
        
        # Create separate folder for individual video clips
        clips_output_dir = video_output_dir / "individual_clips"
        clips_output_dir.mkdir(parents=True, exist_ok=True)
        print(f"   📁 Individual clips will be saved to: {clips_output_dir}")
        
        for i, (model_instance, input_data, true_label, sample_idx, clip_name) in enumerate(test_data):
            # Show progress every 50 clips
            if i % 50 == 0:
                print(f"   Progress: {i}/{len(test_data)} clips processed...")
            # Compute saliency map for this clip
            saliency_map = compute_saliency_map(model_instance, input_data)
            # Get prediction
            with torch.no_grad():
                pred_output = model_instance(input_data, return_loss=False)
                
                # Handle different output formats
                if isinstance(pred_output, (list, tuple)):
                    pred_output = pred_output[0]
                
                # Process the output to get class prediction
                output = pred_output.mean(dim=2)  # Average over temporal dimension
                output = output.squeeze(1)  # Remove batch dimension
                pred_scores = output.mean(dim=-1)  # Average over spatial dimension
                
                # Handle different class configurations
                if pred_scores.size(1) != 2:
                    # Convert to binary classification
                    binary_scores = torch.zeros((pred_scores.size(0), 2), device=pred_scores.device)
                    binary_scores[:, 0] = pred_scores[:, 0]
                    binary_scores[:, 1] = pred_scores[:, 1:].sum(dim=1)
                    pred_scores = binary_scores
                
                pred_label = torch.argmax(pred_scores, dim=1).item()
            
            # Use predictions from CSV for saliency video
            if clip_name in clip_predictions:
                pred_data = clip_predictions[clip_name]
                create_saliency_video(clip_name, saliency_map, clips_output_dir, pred_label=pred_data['pred_label'])
            else:
                create_saliency_video(clip_name, saliency_map, clips_output_dir, pred_label=pred_label)
        print(f"   ✅ Individual saliency videos saved to: {clips_output_dir}")
        
        if task in ['whole_video', 'all']:
            print(f"\n🎬 STEP 4/4: Creating whole video aggregated saliency")
            print(f"\n🧠 PHASE 2: SALIENCY COMPUTATION")
            print(f"   7. Computing gradient-based saliency maps for each clip...")
            print(f"      Processing {len(test_data)} clips to identify keypoint importance...")
            
            # Create video_clips dictionary for whole video processing using CSV predictions
            if video_clips is None:
                video_clips = {}
            for i, (model_instance, input_data, true_label, sample_idx, clip_name) in enumerate(test_data):
                if i % 50 == 0:  # Progress update every 50 clips
                    print(f"      Progress: {i}/{len(test_data)} clips processed...")
                saliency_map = compute_saliency_map(model_instance, input_data)
                
                # Use predictions from CSV instead of computing fresh
                if clip_name in clip_predictions:
                    pred_data = clip_predictions[clip_name]
                    video_clips[clip_name] = {
                        'true_label': pred_data['true_label'],
                        'pred_label': pred_data['pred_label'],
                        'saliency_map': saliency_map,
                        'confidence': pred_data['confidence']
                    }
                    # Debug output removed for cleaner logs
                else:
                    print(f"    Warning: No prediction data found for clip {clip_name}")
                    # Fallback to computed prediction if CSV data not available
                    with torch.no_grad():
                        pred_output = model_instance(input_data, return_loss=False)
                        if isinstance(pred_output, (list, tuple)):
                            pred_output = pred_output[0]
                        output = pred_output.mean(dim=2)
                        output = output.squeeze(1)
                        pred_scores = output.mean(dim=-1)
                        if pred_scores.size(1) != 2:
                            binary_scores = torch.zeros((pred_scores.size(0), 2), device=pred_scores.device)
                            binary_scores[:, 0] = pred_scores[:, 0]
                            binary_scores[:, 1] = pred_scores[:, 1:].sum(dim=1)
                            pred_scores = binary_scores
                        pred_label = torch.argmax(pred_scores, dim=1).item()
                        confidence = torch.softmax(pred_scores, dim=1).max().item()
                    
                    video_clips[clip_name] = {
                        'true_label': true_label,
                        'pred_label': pred_label,
                        'saliency_map': saliency_map,
                        'confidence': confidence
                    }
            
            print(f"   ✅ Phase 2 complete: Saliency maps computed for all clips")
            
            print(f"\n📊 PHASE 3: WHOLE VIDEO AGGREGATION")
            print(f"   8. Calculating video-level performance metrics...")
            video_metrics = calculate_video_metrics(video_predictions)
            if video_metrics:
                print(f"      Video Performance: AUC={video_metrics['AUC_ROC']:.3f}, F1={video_metrics['F1_Score']:.3f}, Accuracy={video_metrics['Accuracy']:.3f}")
            else:
                print(f"      Warning: Could not calculate video metrics")
                video_metrics = None
            
            # Generate three videos with different normalization modes
            normalization_modes = ["per_clip", "whole_video_per_keypoint", "per_clip_per_keypoint"]
            for mode in normalization_modes:
                try:
                    print(f"\n   Generating whole video with {mode} normalization...")
                    create_whole_video_aggregated_saliency(
                        None, video_id, video_clips, video_output_dir,
                        video_metrics=video_metrics, normalization_mode=mode
                    )
                    print(f"   ✅ Whole video ({mode} normalization) saved to: {video_output_dir}")
                except Exception as e:
                    print(f"   ⚠️  Error generating whole video with {mode} normalization: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
    
    # Generate keypoint saliency plots if requested
    if task in ['keypoint_timeline', 'all']:
        # Ensure video_clips is populated (create if not already created)
        if video_clips is None or not video_clips:
            print(f"\n📊 Generating keypoint saliency plots - computing saliency maps...")
            video_clips = {}
            for i, (model_instance, input_data, true_label, sample_idx, clip_name) in enumerate(test_data):
                if i % 50 == 0:
                    print(f"      Progress: {i}/{len(test_data)} clips processed...")
                saliency_map = compute_saliency_map(model_instance, input_data)
                
                if clip_name in clip_predictions:
                    pred_data = clip_predictions[clip_name]
                    video_clips[clip_name] = {
                        'true_label': pred_data['true_label'],
                        'pred_label': pred_data['pred_label'],
                        'saliency_map': saliency_map,
                        'confidence': pred_data['confidence']
                    }
                else:
                    # Fallback to computed prediction
                    with torch.no_grad():
                        pred_output = model_instance(input_data, return_loss=False)
                        if isinstance(pred_output, (list, tuple)):
                            pred_output = pred_output[0]
                        output = pred_output.mean(dim=2)
                        output = output.squeeze(1)
                        pred_scores = output.mean(dim=-1)
                        if pred_scores.size(1) != 2:
                            binary_scores = torch.zeros((pred_scores.size(0), 2), device=pred_scores.device)
                            binary_scores[:, 0] = pred_scores[:, 0]
                            binary_scores[:, 1] = pred_scores[:, 1:].sum(dim=1)
                            pred_scores = binary_scores
                        pred_label = torch.argmax(pred_scores, dim=1).item()
                        confidence = torch.softmax(pred_scores, dim=1).max().item()
                    
                    video_clips[clip_name] = {
                        'true_label': true_label,
                        'pred_label': pred_label,
                        'saliency_map': saliency_map,
                        'confidence': confidence
                    }
        
        # Generate keypoint saliency plots
        print(f"\n📈 Generating keypoint saliency over time plots...")
        plot_keypoint_saliency_over_time_whole_video(video_id, video_clips, video_output_dir, moving_avg_window=moving_avg_window)
        print(f"   ✅ Keypoint saliency plots saved to: {video_output_dir / 'keypoint_saliency_timeline'}")
    
    # Generate keypoint statistics analysis if requested
    if task in ['keypoint_statistics', 'all']:
        # Ensure video_clips is populated (create if not already created)
        if video_clips is None or not video_clips:
            print(f"\n📊 Generating keypoint statistics - computing saliency maps...")
            video_clips = {}
            for i, (model_instance, input_data, true_label, sample_idx, clip_name) in enumerate(test_data):
                if i % 50 == 0:
                    print(f"      Progress: {i}/{len(test_data)} clips processed...")
                saliency_map = compute_saliency_map(model_instance, input_data)
                
                if clip_name in clip_predictions:
                    pred_data = clip_predictions[clip_name]
                    video_clips[clip_name] = {
                        'true_label': pred_data['true_label'],
                        'pred_label': pred_data['pred_label'],
                        'saliency_map': saliency_map,
                        'confidence': pred_data['confidence']
                    }
                else:
                    # Fallback to computed prediction
                    with torch.no_grad():
                        pred_output = model_instance(input_data, return_loss=False)
                        if isinstance(pred_output, (list, tuple)):
                            pred_output = pred_output[0]
                        output = pred_output.mean(dim=2)
                        output = output.squeeze(1)
                        pred_scores = output.mean(dim=-1)
                        if pred_scores.size(1) != 2:
                            binary_scores = torch.zeros((pred_scores.size(0), 2), device=pred_scores.device)
                            binary_scores[:, 0] = pred_scores[:, 0]
                            binary_scores[:, 1] = pred_scores[:, 1:].sum(dim=1)
                            pred_scores = binary_scores
                        pred_label = torch.argmax(pred_scores, dim=1).item()
                        confidence = torch.softmax(pred_scores, dim=1).max().item()
                    
                    video_clips[clip_name] = {
                        'true_label': true_label,
                        'pred_label': pred_label,
                        'saliency_map': saliency_map,
                        'confidence': confidence
                    }
        
        # Generate keypoint statistics
        print(f"\n📊 Generating keypoint statistical analysis...")
        from .viz_saliency_statistics import analyze_keypoint_statistics
        analyze_keypoint_statistics(video_id, video_clips, video_output_dir)
        print(f"   ✅ Keypoint statistics saved to: {video_output_dir / 'keypoint_statistics'}")
    
    print(f"\n🎉 COMPLETED: Video {video_id} processed successfully!")
    print(f"📁 All outputs saved to: {video_output_dir}")
    print(f"{'='*60}") 