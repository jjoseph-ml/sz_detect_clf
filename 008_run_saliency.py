#!/usr/bin/env python
"""
Entry point script to run the saliency visualization using existing predictions.
This script should be placed in the parent directory (where the saliency_viz folder is located).
"""

import os
import sys
import argparse
import pandas as pd
from pathlib import Path

# Add the current directory to the Python path if needed
# This ensures the saliency_viz package can be found
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Add the local mmaction2 folder to Python path to use the same version as training
mmaction2_path = os.path.join(os.path.dirname(__file__), 'mmaction2')
if mmaction2_path not in sys.path:
    sys.path.insert(0, mmaction2_path)

# Import the main functions from the package
from saliency_viz.main import main, process_single_video, main_with_checkpoint, process_single_video_with_checkpoint

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run saliency analysis using existing predictions')
    parser.add_argument('--checkpoint', type=str, required=True,
                      help='Path to the model checkpoint file')
    parser.add_argument('--task', choices=['histograms', 'saliency_maps', 'saliency_videos', 'whole_video', 'keypoint_timeline', 'keypoint_statistics', 'all'], default='all',
                      help='Task to perform: "histograms" (body part histograms), "saliency_maps" (static plots), "saliency_videos" (individual clips + whole video), "whole_video" (whole video only), "keypoint_timeline" (keypoint saliency over time plots), "keypoint_statistics" (keypoint importance ranking and activation sequences), or "all"')
    parser.add_argument('--video-id', type=str, default=None,
                      help='Process only a specific video ID. If not provided, processes all videos.')
    parser.add_argument('--moving-avg-window', type=int, default=10,
                      help='Window size for moving average smoothing in body part timeline plots (default: 10 frames, ~0.33 seconds at 30 FPS)')
    args = parser.parse_args()
    
    if args.video_id:
        # Process single video
        process_single_video_with_checkpoint(args.checkpoint, args.video_id, task=args.task, moving_avg_window=args.moving_avg_window)
    else:
        # Process all videos
        main_with_checkpoint(args.checkpoint, task=args.task, moving_avg_window=args.moving_avg_window)