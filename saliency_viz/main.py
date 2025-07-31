#!/usr/bin/env python
"""
Main orchestration module for saliency analysis.
"""

from pathlib import Path

from .data_loader import load_predictions_from_csv
from .video_processor import process_video_by_task, create_per_patient_analysis, create_analysis_directory_structure


def process_single_video(best_fold: int, video_id: str, task='all'):
    """
    Process a single video for saliency analysis.
    
    Args:
        best_fold: The fold number to use
        video_id: The specific video ID to process
        task: The task to perform:
            - 'histograms': Create body part saliency histograms
            - 'saliency_maps': Create static PNG plots of saliency maps
            - 'saliency_videos': Create individual clip videos AND whole video visualization
            - 'whole_video': Create only whole video visualization
            - 'all': Create all outputs
    """
    print(f"Starting saliency analysis for single video: {video_id} (fold {best_fold}, task: {task})")
    
    # Create output directory
    output_dir = Path('k_fold/saliency_maps')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load predictions from CSV
    try:
        predictions_df = load_predictions_from_csv(best_fold)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please ensure the CSV file exists and run the testing script first.")
        return
    
    # Check if the specified video exists in the predictions
    if video_id not in predictions_df['video_id'].unique():
        available_videos = predictions_df['video_id'].unique()
        print(f"Error: Video ID '{video_id}' not found in predictions.")
        print(f"Available video IDs: {sorted(available_videos)}")
        return
    
    print(f"Found video {video_id} in predictions dataset")
    
    # Process the single video
    try:
        process_video_by_task(best_fold, video_id, predictions_df, output_dir, task=task)
        print(f"\nCompleted saliency analysis for video {video_id}")
        
        # Create analysis directory structure for future use
        create_analysis_directory_structure(output_dir)
        
    except Exception as e:
        print(f"Error processing video {video_id}: {e}")
        import traceback
        traceback.print_exc()


def main(best_fold: int, task='all'):
    """
    Main function to process all videos using existing predictions from CSV.
    """
    print(f"Starting saliency analysis using existing predictions for fold {best_fold} (task: {task})")
    
    # Create output directory
    output_dir = Path('k_fold/saliency_maps')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load predictions from CSV
    try:
        predictions_df = load_predictions_from_csv(best_fold)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please ensure the CSV file exists and run the testing script first.")
        return
    
    # Get unique video IDs from CSV
    video_ids = predictions_df['video_id'].unique()
    print(f"Found {len(video_ids)} unique video IDs: {video_ids}")
    
    # Convert numpy array to list for easier indexing
    video_ids_list = video_ids.tolist()
    
    # Process each video
    for i, video_id in enumerate(video_ids_list):
        print("=" * 80)
        print(f"Processing video: {video_id}  Number: {i + 1}")
        print("=" * 80)
        
        try:
            process_video_by_task(best_fold, video_id, predictions_df, output_dir, task=task)
        except Exception as e:
            print(f"Error processing video {video_id}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Create per-patient analysis after processing all videos
    if task in ['histograms', 'all']:
        print("\n" + "=" * 80)
        print("Creating per-patient analysis")
        print("=" * 80)
        create_per_patient_analysis(best_fold, predictions_df, output_dir)
    
    # Create analysis directory structure for cross-cutting analysis
    print("\n" + "=" * 80)
    print("Creating analysis directory structure")
    print("=" * 80)
    create_analysis_directory_structure(output_dir)
    
    print(f"\nCompleted saliency analysis for all videos in fold {best_fold}")
