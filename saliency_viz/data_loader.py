#!/usr/bin/env python
"""
Data loading functions for saliency analysis.
"""

import os
import pandas as pd
from pathlib import Path
import pickle

def load_predictions_from_csv(best_fold: int):
    """Load predictions from the CSV file and return as DataFrame."""
    csv_path = f"all_video_testing/all/all_predictions.csv"
    
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} predictions from {csv_path}")
    return df

def filter_predictions_by_video(predictions_df, video_id: str):
    """Filter predictions for a specific video ID."""
    video_predictions = predictions_df[predictions_df['video_id'] == video_id].copy()
    print(f"Found {len(video_predictions)} predictions for video {video_id}")
    return video_predictions

def find_clip_data(test_data, clip_name: str):
    """Find test data for a specific clip name."""
    for model, input_data, true_label, sample_idx, test_clip_name in test_data:
        if test_clip_name == clip_name:
            return {
                'model': model,
                'input_data': input_data,
                'true_label': true_label,
                'sample_idx': sample_idx,
                'clip_name': test_clip_name
            }
    return None 