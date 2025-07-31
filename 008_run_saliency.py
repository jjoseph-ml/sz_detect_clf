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

# Import the main function from the package
from saliency_viz.main import main, process_single_video

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run saliency analysis using existing predictions')
    parser.add_argument('--best-fold', type=int, required=True,
                      help='Index of the best fold to use for saliency analysis')
    parser.add_argument('--task', choices=['histograms', 'saliency_maps', 'saliency_videos', 'whole_video', 'all'], default='all',
                      help='Task to perform: "histograms" (body part histograms), "saliency_maps" (static plots), "saliency_videos" (individual clips + whole video), "whole_video" (whole video only), or "all"')
    parser.add_argument('--video-id', type=str, default=None,
                      help='Process only a specific video ID. If not provided, processes all videos.')
    args = parser.parse_args()
    
    if args.video_id:
        # Process single video
        process_single_video(args.best_fold, args.video_id, task=args.task)
    else:
        # Process all videos
        main(args.best_fold, task=args.task)