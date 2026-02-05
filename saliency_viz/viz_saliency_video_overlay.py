"""
Functions for creating video overlays with saliency maps.
"""

import os
import cv2
import numpy as np
import pickle
from pathlib import Path

from .utils import frame_extract, rdbu_color, load_visibility_data


def create_saliency_video(clip_name, saliency_map, output_dir, top_percent=10, pred_label=None, confidence=None):
    try:
        video_path = f"preprocessing/video_clips/{clip_name}.mp4"
        keypoint_path = f"preprocessing/clip_keypoints/{clip_name}.pkl"
        # Create output directory (output_dir is already the clips directory)
        # Handle both string and Path inputs
        if isinstance(output_dir, str):
            output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)
        output_video = output_dir / f"{clip_name}_saliency.mp4"
        
        # Check if files exist
        video_exists = os.path.exists(video_path)
        keypoint_exists = os.path.exists(keypoint_path)
        
        # Only show detailed output for every 50 clips or when there are errors
        show_details = (not video_exists or not keypoint_exists)  # Always show errors
        
        if show_details:
            print(f"    Checking files for {clip_name}:")
            print(f"      Video: {video_path} - {'✓' if video_exists else '✗'}")
            print(f"      Keypoints: {keypoint_path} - {'✓' if keypoint_exists else '✗'}")
            print(f"      Output: {output_video}")
        
        if not video_exists or not keypoint_exists:
            print(f"    Video or keypoint file not found for {clip_name}")
            return None
        
        frames = frame_extract(video_path)
        if not frames:
            return None
        
        with open(keypoint_path, 'rb') as f:
            keypoint_data = pickle.load(f)
        
        keypoints = None
        keypoint_scores = None
        if isinstance(keypoint_data, dict):
            if 'keypoint' in keypoint_data:
                keypoints = keypoint_data['keypoint']
                if 'keypoint_score' in keypoint_data:
                    keypoint_scores = keypoint_data['keypoint_score']
            elif 'keypoints' in keypoint_data:
                keypoints = keypoint_data['keypoints']
        
        if keypoints is None:
            return None
        
        if len(keypoints.shape) == 4:
            keypoints = keypoints[0]
        
        max_abs_val = np.max(np.abs(saliency_map))
        saliency_norm = saliency_map / (max_abs_val + 1e-8)
        
        abs_saliency = np.abs(saliency_norm)
        # COMMENTED OUT: 10% filtering logic - now showing all keypoints
        # threshold = np.percentile(abs_saliency.flatten(), 100 - top_percent)
        threshold = 0.0  # Show all keypoints (no filtering)
        
        # Debug: Print threshold info
        total_values = abs_saliency.size
        values_above_threshold = np.sum(abs_saliency >= threshold)
        percentage_shown = (values_above_threshold / total_values) * 100
        print(f"    Threshold: {threshold:.4f}, Values above threshold: {values_above_threshold}/{total_values} ({percentage_shown:.1f}%)")
        
        height, width = frames[0].shape[:2]
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_video), fourcc, 30, (width, height))
        
        for i, frame in enumerate(frames):
            if i >= len(saliency_norm) or i >= len(keypoints):
                break
                
            vis_frame = frame.copy()
            
            legend_height = 30
            legend_width = 200
            legend_x = width - legend_width - 10
            legend_y = 10
            
            for x in range(legend_width):
                val = -1 + 2 * (x / legend_width)
                color = rdbu_color(val)
                cv2.line(vis_frame, (legend_x + x, legend_y), (legend_x + x, legend_y + legend_height), color, 1)
            
            cv2.putText(vis_frame, "Negative", (legend_x, legend_y + legend_height + 20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, rdbu_color(-1), 1)
            cv2.putText(vis_frame, "Positive", (legend_x + legend_width - 50, legend_y + legend_height + 20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, rdbu_color(1), 1)
            cv2.putText(vis_frame, "Neutral", (legend_x + legend_width//2 - 25, legend_y + legend_height + 20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, rdbu_color(0), 1)
            
            frame_keypoints = keypoints[i]
            frame_saliency = saliency_norm[i]
            
            for j, kpt in enumerate(frame_keypoints):
                if j >= saliency_norm.shape[1]:
                    continue
                
                saliency_val = frame_saliency[j]
                
                if abs(saliency_val) < threshold:
                    continue
                
                x, y = int(kpt[0]), int(kpt[1])
                
                if 0 <= x < width and 0 <= y < height:
                    valid_point = True
                    if keypoint_scores is not None:
                        valid_point = keypoint_scores[0, i, j] > 0.3
                    
                    if valid_point:
                        color = rdbu_color(saliency_val)
                        base_radius = 2
                        scale_factor = 2
                        radius = int(base_radius + abs(saliency_val) * scale_factor)
                        cv2.circle(vis_frame, (x, y), radius, color, -1)
            
            if pred_label is not None and confidence is not None:
                pred_text = f"Prediction: {'Seizure' if pred_label == 1 else 'Non-Seizure'}"
                prob_text = f"Probability: {confidence:.2%}"
                
                cv2.rectangle(vis_frame, (10, 10), (310, 70), (0, 0, 0), -1)
                cv2.rectangle(vis_frame, (10, 10), (310, 70), (255, 255, 255), 1)
                
                cv2.putText(vis_frame, pred_text, (20, 35), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(vis_frame, prob_text, (20, 60), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            out.write(vis_frame)
        
        out.release()
        return output_video
        
    except Exception as e:
        print(f"Error creating saliency video for {clip_name}: {e}")
        import traceback
        traceback.print_exc()
        return None


def _calculate_percentile_normalization_per_keypoint_1d(clip_averaged_saliency, percentile_low=5, percentile_high=95):
    """
    Calculate percentile-based normalization for 1D input (single clip case).
    For 1D input, calculate percentiles across all keypoints.
    
    Args:
        clip_averaged_saliency (np.ndarray): 1D array of shape (num_keypoints,) with saliency values
        percentile_low (int): Lower percentile to use (default: 5)
        percentile_high (int): Upper percentile to use (default: 95)
    
    Returns:
        float: vmax value for normalization (same for all keypoints in single clip)
    """
    # Calculate percentiles for positive and negative values separately
    positive_values = clip_averaged_saliency[clip_averaged_saliency > 0]
    negative_values = clip_averaged_saliency[clip_averaged_saliency < 0]
    
    if len(positive_values) > 0 and len(negative_values) > 0:
        pos_percentile = np.percentile(positive_values, percentile_high)
        neg_percentile = abs(np.percentile(negative_values, 100 - percentile_low))
        vmax = max(pos_percentile, neg_percentile)
    elif len(positive_values) > 0:
        vmax = np.percentile(positive_values, percentile_high)
    elif len(negative_values) > 0:
        vmax = abs(np.percentile(negative_values, 100 - percentile_low))
    else:
        # Fallback to absolute max if no positive/negative values
        vmax = max(abs(clip_averaged_saliency.min()), abs(clip_averaged_saliency.max()))
    
    # Ensure vmax is not zero
    if vmax < 1e-8:
        vmax = max(abs(clip_averaged_saliency.min()), abs(clip_averaged_saliency.max()))
        if vmax < 1e-8:
            vmax = 1.0  # Default fallback
    
    return vmax


def normalize_per_clip(clip_averaged_saliency, percentile=95):
    """
    Normalize saliency values per clip using percentile-based normalization.
    
    Args:
        clip_averaged_saliency (np.ndarray): 1D array of shape (num_keypoints,) - averaged saliency per keypoint for one clip
        percentile (int): Percentile to use for normalization (default: 95)
    
    Returns:
        tuple: (normalized_array, "PER-CLIP", norm_factor) where normalized_array is clipped to [-1, 1]
    """
    try:
        if clip_averaged_saliency.size == 0:
            print("Warning: Empty saliency array in normalize_per_clip, returning original array")
            return clip_averaged_saliency, "PER-CLIP", 1.0
        
        # Calculate percentile
        percentile_val = np.percentile(np.abs(clip_averaged_saliency), percentile)
        
        # Handle zero-division
        if percentile_val < 1e-8:
            max_abs_val = np.max(np.abs(clip_averaged_saliency))
            if max_abs_val < 1e-8:
                norm_factor = 1.0
            else:
                norm_factor = max_abs_val
        else:
            norm_factor = percentile_val
        
        # Normalize
        normalized_array = clip_averaged_saliency / norm_factor
        normalized_array = np.clip(normalized_array, -1.0, 1.0)
        
        print(f"   Normalization factor (per-clip, {percentile}th percentile): {norm_factor:.6f}")
        
        return normalized_array, "PER-CLIP", norm_factor
    
    except Exception as e:
        print(f"Warning: Error in normalize_per_clip: {e}, using default normalization")
        return clip_averaged_saliency, "PER-CLIP", 1.0


def normalize_whole_video_per_keypoint(all_clip_averaged_dict, sorted_clips, percentile_low=5, percentile_high=95):
    """
    Normalize saliency values per keypoint across the whole video.
    Uses two-pass approach: first collect all data, then calculate per-keypoint normalization.
    
    Args:
        all_clip_averaged_dict (dict): Dictionary mapping clip_name to 1D array (num_keypoints,)
        sorted_clips (list): List of sorted clip names
        percentile_low (int): Lower percentile to use (default: 5)
        percentile_high (int): Upper percentile to use (default: 95)
    
    Returns:
        tuple: (normalized_dict, normalization_ranges) where:
            - normalized_dict: maps clip_name to normalized 1D array
            - normalization_ranges: 2D array (num_keypoints, 2) with [vmin, vmax] per keypoint
    """
    try:
        if not all_clip_averaged_dict or len(sorted_clips) == 0:
            print("Warning: Empty clip data in normalize_whole_video_per_keypoint")
            return {}, np.zeros((0, 2))
        
        # First pass: Stack all arrays into 2D array (num_keypoints, num_clips)
        num_keypoints = None
        stacked_arrays = []
        valid_clips = []
        
        for clip_name in sorted_clips:
            if clip_name not in all_clip_averaged_dict:
                continue
            clip_data = all_clip_averaged_dict[clip_name]
            if clip_data.size == 0:
                continue
            
            if num_keypoints is None:
                num_keypoints = clip_data.shape[0]
            elif clip_data.shape[0] != num_keypoints:
                print(f"Warning: Keypoint count mismatch for clip {clip_name}, skipping")
                continue
            
            stacked_arrays.append(clip_data)
            valid_clips.append(clip_name)
        
        if len(stacked_arrays) == 0:
            print("Warning: No valid clips found in normalize_whole_video_per_keypoint")
            return {}, np.zeros((num_keypoints if num_keypoints else 0, 2))
        
        # Stack into 2D array: (num_keypoints, num_clips)
        saliency_heatmap = np.stack(stacked_arrays, axis=1)  # Shape: (num_keypoints, num_clips)
        
        # Calculate normalization ranges per keypoint
        normalization_ranges = np.zeros((num_keypoints, 2))  # [vmin, vmax] for each keypoint
        
        for keypoint_idx in range(num_keypoints):
            keypoint_saliency = saliency_heatmap[keypoint_idx, :]  # Get all clips for this keypoint
            
            # Calculate percentiles for positive and negative values separately
            positive_values = keypoint_saliency[keypoint_saliency > 0]
            negative_values = keypoint_saliency[keypoint_saliency < 0]
            
            if len(positive_values) > 0 and len(negative_values) > 0:
                pos_percentile = np.percentile(positive_values, percentile_high)
                neg_percentile = abs(np.percentile(negative_values, 100 - percentile_low))
                vmax = max(pos_percentile, neg_percentile)
            elif len(positive_values) > 0:
                vmax = np.percentile(positive_values, percentile_high)
            elif len(negative_values) > 0:
                vmax = abs(np.percentile(negative_values, 100 - percentile_low))
            else:
                # Fallback to absolute max
                vmax = max(abs(keypoint_saliency.min()), abs(keypoint_saliency.max()))
            
            # Ensure vmax is not zero
            if vmax < 1e-8:
                vmax = max(abs(keypoint_saliency.min()), abs(keypoint_saliency.max()))
                if vmax < 1e-8:
                    vmax = 1.0  # Default fallback
            
            vmin = -vmax
            normalization_ranges[keypoint_idx, 0] = vmin
            normalization_ranges[keypoint_idx, 1] = vmax
        
        # Second pass: Apply normalization per keypoint to each clip
        normalized_dict = {}
        for clip_name, clip_data in zip(valid_clips, stacked_arrays):
            normalized_array = np.zeros_like(clip_data)
            for keypoint_idx in range(num_keypoints):
                vmin, vmax = normalization_ranges[keypoint_idx, 0], normalization_ranges[keypoint_idx, 1]
                if vmax > 1e-8:
                    normalized_array[keypoint_idx] = np.clip(clip_data[keypoint_idx] / vmax, -1.0, 1.0)
                else:
                    normalized_array[keypoint_idx] = clip_data[keypoint_idx]
            normalized_dict[clip_name] = normalized_array
        
        # Log statistics
        vmax_range = normalization_ranges[:, 1]
        print(f"   Per-keypoint normalization (whole video): vmax ranges from {vmax_range.min():.6f} to {vmax_range.max():.6f}")
        print(f"   Applied normalization to {len(normalized_dict)} clips")
        
        return normalized_dict, normalization_ranges
    
    except Exception as e:
        print(f"Warning: Error in normalize_whole_video_per_keypoint: {e}")
        import traceback
        traceback.print_exc()
        return {}, np.zeros((0, 2))


def normalize_per_clip_per_keypoint(clip_averaged_saliency, percentile_low=5, percentile_high=95):
    """
    Normalize saliency values per keypoint within a single clip.
    For 1D input, calculate percentiles across all keypoints and apply same factor.
    
    Args:
        clip_averaged_saliency (np.ndarray): 1D array of shape (num_keypoints,) - averaged saliency per keypoint for one clip
        percentile_low (int): Lower percentile to use (default: 5)
        percentile_high (int): Upper percentile to use (default: 95)
    
    Returns:
        tuple: (normalized_array, normalization_ranges) where:
            - normalized_array: normalized 1D array clipped to [-1, 1]
            - normalization_ranges: 2D array (num_keypoints, 2) with [vmin, vmax] (same for all keypoints)
    """
    try:
        if clip_averaged_saliency.size == 0:
            print("Warning: Empty saliency array in normalize_per_clip_per_keypoint")
            return clip_averaged_saliency, np.zeros((0, 2))
        
        num_keypoints = clip_averaged_saliency.shape[0]
        
        # Calculate vmax from all keypoints (treat as single "frame")
        vmax = _calculate_percentile_normalization_per_keypoint_1d(
            clip_averaged_saliency, percentile_low, percentile_high
        )
        
        # Apply same normalization factor to all keypoints
        vmin = -vmax
        normalization_ranges = np.zeros((num_keypoints, 2))
        normalization_ranges[:, 0] = vmin
        normalization_ranges[:, 1] = vmax
        
        # Normalize
        if vmax > 1e-8:
            normalized_array = np.clip(clip_averaged_saliency / vmax, -1.0, 1.0)
        else:
            normalized_array = clip_averaged_saliency
        
        print(f"   Normalization factor (per-clip per-keypoint): {vmax:.6f}")
        
        return normalized_array, normalization_ranges
    
    except Exception as e:
        print(f"Warning: Error in normalize_per_clip_per_keypoint: {e}")
        return clip_averaged_saliency, np.zeros((num_keypoints if 'num_keypoints' in locals() else 0, 2))


def draw_seizure_period_bar(vis_frame, is_seizure_period, current_frame_num, total_frames, 
                            seizure_frame_ranges, height, width, bottom_text_height=60, previous_bar_end=None):
    """
    Draw a yellow bar indicating seizure period that expands as the video progresses through the seizure.
    The bar is positioned on the video timeline, starting at the seizure start position and extending
    to the current frame position. After the seizure period ends, the bar continues to show the full
    seizure period length.
    
    For example, if seizure is from frames 100-200 (out of 1000 total frames):
    - At frame 50: no bar (before seizure)
    - At frame 100: bar starts at 10% of width (seizure start), extends to 10% (just started, 1 pixel wide)
    - At frame 105: bar starts at 10% of width, extends to 10.5% of width (5 frames elapsed)
    - At frame 150: bar starts at 10% of width, extends to 15% of width (50 frames elapsed)
    - At frame 200: bar starts at 10% of width, extends to 20% of width (full seizure period)
    - At frame 250: bar still shows from 10% to 20% (full seizure period remains visible)
    
    Args:
        vis_frame (np.ndarray): The frame to draw on (will be modified in place)
        is_seizure_period (bool): True if this frame is in a seizure period, False otherwise
        current_frame_num (int): Current frame number in the whole video (1-indexed)
        total_frames (int): Total number of frames in the entire video
        seizure_frame_ranges (list): List of (start_frame, end_frame) tuples for all seizure periods
        height (int): Frame height
        width (int): Frame width
        bottom_text_height (int): Height of bottom text area to position bar above it
        previous_bar_end (dict): Dictionary to track previous bar end positions per seizure period
    
    Returns:
        None (modifies vis_frame in place)
    """
    if total_frames == 0 or not seizure_frame_ranges:
        return  # Don't draw anything if no frames or no seizure periods
    
    bar_height = 10
    bar_y_start = height - bottom_text_height - bar_height
    bar_y_end = height - bottom_text_height
    
    # Convert to 0-indexed for calculations (current_frame_num is 1-indexed)
    frame_idx = current_frame_num - 1
    
    # Process all seizure periods
    for seizure_start, seizure_end in seizure_frame_ranges:
        # Calculate where the seizure period starts on the timeline (as percentage of video width)
        seizure_start_percent = seizure_start / total_frames
        seizure_end_percent = (seizure_end + 1) / total_frames  # +1 because end is inclusive
        
        # Bar always starts at the seizure start position on the timeline
        bar_x_start = int(seizure_start_percent * width)
        
        # Calculate where the seizure period ends (for boundary checking)
        seizure_end_x = int(seizure_end_percent * width)
        
        # Determine bar end position based on current frame position relative to seizure period
        if frame_idx < seizure_start:
            # Before seizure period: don't draw anything for this seizure period
            continue
        elif frame_idx <= seizure_end:
            # During seizure period: bar extends from seizure start to current frame position
            current_frame_percent = (frame_idx + 1) / total_frames  # +1 because we want to include current frame
            calculated_bar_x_end = int(current_frame_percent * width)
            
            # Ensure bar doesn't exceed the seizure end position
            calculated_bar_x_end = min(calculated_bar_x_end, seizure_end_x)
            
            # Ensure smooth expansion: bar should always be at least as big as previous frame
            # This prevents the bar from shrinking or jumping backwards
            if previous_bar_end is not None:
                seizure_key = (seizure_start, seizure_end)
                if seizure_key in previous_bar_end:
                    prev_end = previous_bar_end[seizure_key]
                    # Ensure bar grows (or stays same) - never shrinks
                    calculated_bar_x_end = max(calculated_bar_x_end, prev_end)
            
            bar_x_end = calculated_bar_x_end
            
            # Update previous bar end for next frame
            if previous_bar_end is not None:
                seizure_key = (seizure_start, seizure_end)
                previous_bar_end[seizure_key] = bar_x_end
        else:
            # After seizure period: show full seizure period length (from start to end)
            bar_x_end = seizure_end_x
            
            # Update previous bar end to full length (so it stays at full length)
            if previous_bar_end is not None:
                seizure_key = (seizure_start, seizure_end)
                previous_bar_end[seizure_key] = bar_x_end
        
        # Ensure bar is at least 2 pixels wide for visibility (even at the very start)
        if bar_x_end - bar_x_start < 2:
            bar_x_end = bar_x_start + 2
            # But don't exceed seizure end
            if bar_x_end > seizure_end_x:
                bar_x_end = seizure_end_x
        
        # Draw yellow bar positioned on the timeline
        seizure_bar_color = (0, 255, 255)  # Yellow in BGR
        cv2.rectangle(vis_frame, (bar_x_start, bar_y_start), (bar_x_end, bar_y_end), seizure_bar_color, -1)
        # Add 1-pixel dark border for better visibility
        cv2.rectangle(vis_frame, (bar_x_start, bar_y_start), (bar_x_end, bar_y_end), (0, 0, 0), 1)


def draw_seizure_timeline_bar(vis_frame, current_frame_num, total_frames, 
                              seizure_frame_ranges, height, width, bottom_text_height=60):
    """
    Draw a yellow bar indicating full seizure period(s) and a red inverted triangle showing current frame position.
    The yellow bar always shows the complete seizure period(s) regardless of current frame position.
    The red triangle moves along the timeline to indicate the current frame position.
    
    For example, if seizure is from frames 100-200 (out of 1000 total frames):
    - At frame 50: yellow bar from 10% to 20% of width (full seizure), red triangle at 5% (current frame)
    - At frame 105: yellow bar from 10% to 20% of width (full seizure), red triangle at 10.5% (current frame)
    - At frame 200: yellow bar from 10% to 20% of width (full seizure), red triangle at 20% (current frame)
    - At frame 250: yellow bar from 10% to 20% of width (full seizure), red triangle at 25% (current frame)
    
    Args:
        vis_frame (np.ndarray): The frame to draw on (will be modified in place)
        current_frame_num (int): Current frame number in the whole video (1-indexed)
        total_frames (int): Total number of frames in the entire video
        seizure_frame_ranges (list): List of (start_frame, end_frame) tuples for all seizure periods
        height (int): Frame height
        width (int): Frame width
        bottom_text_height (int): Height of bottom text area to position bar above it
    
    Returns:
        None (modifies vis_frame in place)
    """
    if total_frames == 0 or not seizure_frame_ranges:
        return  # Don't draw anything if no frames or no seizure periods
    
    bar_height = 10
    bar_y_start = height - bottom_text_height - bar_height
    bar_y_end = height - bottom_text_height
    
    # Draw yellow bars for all seizure periods (always full length, drawn on every frame)
    for seizure_start, seizure_end in seizure_frame_ranges:
        # Calculate where the seizure period starts and ends on the timeline (as percentage of video width)
        seizure_start_percent = seizure_start / total_frames
        seizure_end_percent = (seizure_end + 1) / total_frames  # +1 because end is inclusive
        
        # Bar always shows full seizure period from start to end
        bar_x_start = int(seizure_start_percent * width)
        bar_x_end = int(seizure_end_percent * width)
        
        # Ensure bar is at least 2 pixels wide for visibility
        if bar_x_end - bar_x_start < 2:
            bar_x_end = bar_x_start + 2
            # But don't exceed frame width
            if bar_x_end > width:
                bar_x_end = width
        
        # Draw yellow bar positioned on the timeline
        seizure_bar_color = (0, 255, 255)  # Yellow in BGR
        cv2.rectangle(vis_frame, (bar_x_start, bar_y_start), (bar_x_end, bar_y_end), seizure_bar_color, -1)
        # Add 1-pixel dark border for better visibility
        cv2.rectangle(vis_frame, (bar_x_start, bar_y_start), (bar_x_end, bar_y_end), (0, 0, 0), 1)
    
    # Draw red inverted triangle at current frame position
    # Calculate current frame position on timeline
    current_frame_percent = current_frame_num / total_frames
    triangle_x = int(current_frame_percent * width)
    
    # Ensure triangle is within frame bounds
    triangle_x = max(0, min(triangle_x, width - 1))
    
    # Triangle parameters: inverted triangle pointing down
    triangle_height = 8
    triangle_base_width = 10
    triangle_top_y = bar_y_start - triangle_height // 2  # Position closer to the yellow bar
    triangle_bottom_y = bar_y_start + bar_height // 2  # Point extends into the middle of the yellow bar
    
    # Define triangle points: top-left, top-right, bottom-center (pointing down)
    triangle_points = np.array([
        [triangle_x - triangle_base_width // 2, triangle_top_y],      # Top-left
        [triangle_x + triangle_base_width // 2, triangle_top_y],      # Top-right
        [triangle_x, triangle_bottom_y]                                # Bottom point
    ], np.int32)
    
    # Draw red inverted triangle
    red_color = (0, 0, 255)  # Red in BGR
    cv2.fillPoly(vis_frame, [triangle_points], red_color)
    # Add 1-pixel dark border for better visibility
    cv2.drawContours(vis_frame, [triangle_points], -1, (0, 0, 0), 1)


def draw_prediction_timeline_bar(vis_frame, current_frame_num, total_frames, 
                                  prediction_frame_ranges, height, width, bottom_text_height=60):
    """
    Draw a magenta bar indicating predicted seizure period(s).
    The magenta bar always shows the complete predicted seizure period(s) regardless of current frame position.
    Positioned above the yellow seizure bar with a small gap.
    
    For example, if prediction is from frames 100-200 (out of 1000 total frames):
    - At frame 50: magenta bar from 10% to 20% of width (full prediction)
    - At frame 105: magenta bar from 10% to 20% of width (full prediction)
    - At frame 200: magenta bar from 10% to 20% of width (full prediction)
    - At frame 250: magenta bar from 10% to 20% of width (full prediction)
    
    Args:
        vis_frame (np.ndarray): The frame to draw on (will be modified in place)
        current_frame_num (int): Current frame number in the whole video (1-indexed)
        total_frames (int): Total number of frames in the entire video
        prediction_frame_ranges (list): List of (start_frame, end_frame) tuples for all predicted seizure periods
        height (int): Frame height
        width (int): Frame width
        bottom_text_height (int): Height of bottom text area to position bar above it
    
    Returns:
        None (modifies vis_frame in place)
    """
    if total_frames == 0 or not prediction_frame_ranges:
        return  # Don't draw anything if no frames or no prediction periods
    
    bar_height = 10
    gap_between_bars = 3  # Gap between prediction bar and seizure bar
    # Position prediction bar above the seizure bar
    # Seizure bar is at: height - bottom_text_height - bar_height
    # Prediction bar should be above it with a gap
    seizure_bar_y_start = height - bottom_text_height - bar_height
    bar_y_start = seizure_bar_y_start - bar_height - gap_between_bars
    bar_y_end = bar_y_start + bar_height
    
    # Draw magenta bars for all prediction periods (always full length, drawn on every frame)
    for pred_start, pred_end in prediction_frame_ranges:
        # Calculate where the prediction period starts and ends on the timeline (as percentage of video width)
        pred_start_percent = pred_start / total_frames
        pred_end_percent = (pred_end + 1) / total_frames  # +1 because end is inclusive
        
        # Bar always shows full prediction period from start to end
        bar_x_start = int(pred_start_percent * width)
        bar_x_end = int(pred_end_percent * width)
        
        # Ensure bar is at least 2 pixels wide for visibility
        if bar_x_end - bar_x_start < 2:
            bar_x_end = bar_x_start + 2
            # But don't exceed frame width
            if bar_x_end > width:
                bar_x_end = width
        
        # Draw magenta bar positioned on the timeline
        prediction_bar_color = (255, 0, 255)  # Magenta in BGR
        cv2.rectangle(vis_frame, (bar_x_start, bar_y_start), (bar_x_end, bar_y_end), prediction_bar_color, -1)
        # Add 1-pixel dark border for better visibility
        cv2.rectangle(vis_frame, (bar_x_start, bar_y_start), (bar_x_end, bar_y_end), (0, 0, 0), 1)


def create_whole_video_aggregated_saliency(fold, video_id, video_clips, output_dir,
                                           low_percentile=5, high_percentile=95,
                                           video_metrics=None, use_global_normalization=False,
                                           normalization_mode="per_clip"):
    """
    Create a whole-video visualization where saliency for each keypoint is averaged
    across all frames within its clip. This aggregated value is then normalized
    and displayed consistently for that keypoint throughout its clip's duration.
    
    Args:
        fold (int): The fold number.
        video_id (str): The video ID.
        video_clips (dict): Dictionary of clip data {clip_name: {saliency_map, true_label, pred_label, confidence}}.
        output_dir (Path): Base output directory (should be .../videos/{video_id}/videos)
        low_percentile (int): Lower percentile for normalization (default: 5)
        high_percentile (int): Upper percentile for normalization (default: 95)
        video_metrics (dict, optional): Metrics to display on video overlay.
        use_global_normalization (bool): Deprecated parameter, ignored if normalization_mode is provided.
        normalization_mode (str): Normalization mode - "per_clip", "whole_video_per_keypoint", or "per_clip_per_keypoint" (default: "per_clip")
    """
    print(f"\n--- Starting Aggregated Saliency Visualization ---")
    print(f"   Normalization mode: {normalization_mode}")
    print(f"   Percentiles: {low_percentile}-{high_percentile}")
    
    # Load visibility data (optional, but good for consistency if used in text)
    visibility_data = load_visibility_data()

    # Create output directory structure (output_dir should already be the whole_video directory)
    whole_video_dir = output_dir
    whole_video_dir.mkdir(parents=True, exist_ok=True)

    # Get all clip names and sort them by segment number
    clip_names = list(video_clips.keys())
    print(f"   Sorting clips chronologically by segment number...")
    
    # Create mapping of clip names to their temporal position
    clip_timeline = {}
    for clip in clip_names:
        # Extract segment number from clip name (e.g., "05463487_79519000_Seg_1")
        try:
            seg_part = clip.split('_Seg_')[1]
            if '_' in seg_part:
                seg_num = int(seg_part.split('_')[0])
            else:
                seg_num = int(seg_part)
            clip_timeline[clip] = seg_num
        except (IndexError, ValueError):
            print(f"   Warning: Error parsing segment number for {clip}")
            clip_timeline[clip] = 999999
    
    # Sort clips by their temporal position
    sorted_clips = sorted(clip_names, key=lambda x: clip_timeline.get(x, 999999))
    print(f"   ✅ Sorted {len(sorted_clips)} clips chronologically")
    
    # Generate video file with name based on normalization mode
    output_video = whole_video_dir / f"{video_id}_whole_video_saliency_{normalization_mode}.mp4"
    
    # Initialize video writer (will set dimensions after loading first frame)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = None
    
    # For whole_video_per_keypoint mode: collect all clip-averaged saliency values first
    if normalization_mode == "whole_video_per_keypoint":
        print(f"   Collecting all clip-averaged saliency values for whole-video per-keypoint normalization...")
        all_clip_averaged_dict = {}
        for clip_name in sorted_clips:
            try:
                saliency_map = video_clips[clip_name]['saliency_map']
                clip_averaged_saliency = np.mean(saliency_map, axis=0)  # Average across time dimension
                all_clip_averaged_dict[clip_name] = clip_averaged_saliency
            except Exception as e:
                print(f"   Warning: Error processing clip {clip_name} for normalization: {e}")
                continue
        
        print(f"   Calculating per-keypoint normalization ranges across all clips...")
        normalized_dict, normalization_ranges = normalize_whole_video_per_keypoint(
            all_clip_averaged_dict, sorted_clips, low_percentile, high_percentile
        )
        print(f"   ✅ Pre-computed normalization for {len(normalized_dict)} clips")
    
    print(f"\n💾 Generating video with {normalization_mode} normalization...")
    print(f"   Loading original video clips and keypoint data...")
    print(f"   Overlaying saliency heatmaps on keypoints...")
    print(f"   Adding prediction labels and confidence scores as text overlays...")
    print(f"   Concatenating all clips into one continuous video...")
    
    # Calculate total video frames and identify seizure period frame ranges
    print(f"   Calculating total video duration and seizure period ranges...")
    total_frames = 0
    seizure_frame_ranges = []  # List of (start_frame, end_frame) tuples for seizure periods
    prediction_frame_ranges = []  # List of (start_frame, end_frame) tuples for predicted seizure periods
    current_frame = 0
    clip_frame_counts = {}  # Track frame count per clip for accurate frame indexing
    
    for clip_name in sorted_clips:
        try:
            video_path = f"preprocessing/video_clips/{clip_name}.mp4"
            if os.path.exists(video_path):
                frames = frame_extract(video_path)
                num_frames = len(frames)
                true_label = video_clips[clip_name]['true_label']
                pred_label = video_clips[clip_name].get('pred_label', None)
                confidence = video_clips[clip_name].get('confidence', 0.0)
                
                clip_frame_counts[clip_name] = num_frames
                
                if true_label == 1:  # Seizure period
                    seizure_start = current_frame
                    seizure_end = current_frame + num_frames - 1
                    seizure_frame_ranges.append((seizure_start, seizure_end))
                    print(f"      Seizure clip {clip_name}: frames {seizure_start} to {seizure_end} ({num_frames} frames)")
                
                # Check for predicted seizure periods (pred_label == 1 and confidence >= 0.5)
                if pred_label == 1 and confidence >= 0.5:
                    pred_start = current_frame
                    pred_end = current_frame + num_frames - 1
                    prediction_frame_ranges.append((pred_start, pred_end))
                    print(f"      Predicted seizure clip {clip_name}: frames {pred_start} to {pred_end} ({num_frames} frames, confidence: {confidence:.3f})")
                
                current_frame += num_frames
                total_frames += num_frames
        except Exception as e:
            print(f"   Warning: Error calculating frames for {clip_name}: {e}")
            continue
    
    print(f"   Total video frames: {total_frames}")
    if seizure_frame_ranges:
        print(f"   Seizure periods: {seizure_frame_ranges}")
    else:
        print(f"   No seizure periods found in this video")
    if prediction_frame_ranges:
        print(f"   Predicted seizure periods: {prediction_frame_ranges}")
    else:
        print(f"   No predicted seizure periods found in this video")
    
    # Initialize cumulative frame counter for frame/time display
    cumulative_frame = 0
    fps = 30  # Frames per second for time calculation
    
    # Process each clip in temporal order
    for clip_idx, clip_name in enumerate(sorted_clips):
        try:
            # Get low visibility percentage if available
            low_vis_percentage = visibility_data.get(clip_name, None)
            
            # Get saliency map
            saliency_map = video_clips[clip_name]['saliency_map']
            
            # Average saliency values across all frames in this clip
            # This gives us one saliency value per keypoint for the entire clip
            clip_averaged_saliency = np.mean(saliency_map, axis=0)  # Average across time dimension
            
            # Apply normalization based on mode
            if normalization_mode == "whole_video_per_keypoint":
                # Use pre-computed normalized values
                if clip_name in normalized_dict:
                    saliency_norm = normalized_dict[clip_name]
                else:
                    print(f"   Warning: Clip {clip_name} not found in normalized_dict, skipping")
                    continue
            elif normalization_mode == "per_clip":
                saliency_norm, norm_method, norm_factor = normalize_per_clip(clip_averaged_saliency, percentile=95)
            elif normalization_mode == "per_clip_per_keypoint":
                saliency_norm, normalization_ranges = normalize_per_clip_per_keypoint(
                    clip_averaged_saliency, low_percentile, high_percentile
                )
            else:
                print(f"   Warning: Unknown normalization_mode '{normalization_mode}', using per_clip")
                saliency_norm, norm_method, norm_factor = normalize_per_clip(clip_averaged_saliency, percentile=95)
            
            # Get prediction info - THIS IS RETRIEVED PER CLIP, NOT PER FRAME
            true_label = video_clips[clip_name]['true_label']
            pred_label = video_clips[clip_name]['pred_label']
            confidence = video_clips[clip_name]['confidence']
            
            # Debug: Log true_label for first few clips and seizure clips
            if clip_idx < 5 or true_label == 1 or (clip_idx % 50 == 0):
                print(f"   Clip {clip_idx+1}/{len(sorted_clips)} {clip_name}: true_label={true_label} (will draw bar: {true_label == 1})")
            
            # Load video frames and keypoints for this clip
            video_path = f"preprocessing/video_clips/{clip_name}.mp4"
            keypoint_path = f"preprocessing/clip_keypoints/{clip_name}.pkl"
            
            if not os.path.exists(video_path) or not os.path.exists(keypoint_path):
                print(f"   Warning: Missing video or keypoint file for {clip_name}, skipping.")
                continue
            
            frames = frame_extract(video_path)
            if not frames:
                print(f"   Warning: No frames extracted from {video_path}, skipping.")
                continue
            
            try:
                with open(keypoint_path, 'rb') as f:
                    keypoint_data = pickle.load(f)
            except Exception as e:
                print(f"   Warning: Error loading keypoint file {keypoint_path}: {e}, skipping.")
                continue
            
            # Extract keypoints (handle different structures)
            keypoints = None
            if isinstance(keypoint_data, dict) and 'keypoint' in keypoint_data:
                keypoints = keypoint_data['keypoint']
            elif isinstance(keypoint_data, np.ndarray):
                keypoints = keypoint_data
            
            if keypoints is None:
                print(f"   Warning: Could not extract keypoints from {keypoint_path}, skipping.")
                continue
            
            keypoints = np.array(keypoints)  # Ensure numpy array
            
            # Adjust keypoints shape to (T, V, C)
            if keypoints.ndim == 5:  # (N, M, T, V, C)
                keypoints = keypoints[0, 0]  # Take first instance, first person
            elif keypoints.ndim == 4:  # (M, T, V, C)
                keypoints = keypoints[0]  # Take first person
            elif keypoints.ndim != 3:
                print(f"   Warning: Unexpected keypoint dimensions {keypoints.shape} for {clip_name}, skipping.")
                continue
            
            # Ensure number of frames match
            if len(frames) != keypoints.shape[0]:
                print(f"   Warning: Frame count mismatch for {clip_name} (Frames: {len(frames)}, Keypoints: {keypoints.shape[0]}), skipping.")
                continue
            
            # Visualize frames for this clip
            for i in range(len(frames)):
                vis_frame = frames[i].copy()
                height, width, _ = vis_frame.shape
                
                # Initialize video writer with frame dimensions if not done yet
                if video_writer is None:
                    video_writer = cv2.VideoWriter(str(output_video), fourcc, 30, (width, height))
                    print(f"   ✅ Initialized video writer: {width}x{height}")
                
                keypoints_frame = keypoints[i]  # Keypoints for this frame (V, C)
                
                # Draw keypoints with saliency colors
                for j in range(keypoints_frame.shape[0]):  # Iterate over keypoints (V)
                    x, y = int(keypoints_frame[j, 0]), int(keypoints_frame[j, 1])
                    if x <= 0 or y <= 0 or x >= width or y >= height:  # Skip if out of bounds
                        continue
                    
                    # Get the averaged normalized saliency value for this keypoint (same for all frames in clip)
                    if j < len(saliency_norm):
                        saliency_val = saliency_norm[j]  # Use the clip-averaged saliency value
                        color = rdbu_color(saliency_val)
                        
                        # Calculate radius based on saliency magnitude (better scaling with cap)
                        radius = int(1 + min(abs(saliency_val), 2.0) * 1.5)  # Cap at 2.0 and use 1.5 scale
                        
                        cv2.circle(vis_frame, (x, y), radius, color, -1)
                
                # Add text overlays in specified positions
                # Top left: Ground truth and prediction
                pred_text = f'Pred: {"Seizure" if pred_label == 1 else "Non-Seizure"} ({confidence:.2f})'
                true_text = f'True: {"Seizure" if true_label == 1 else "Non-Seizure"}'
                cv2.putText(vis_frame, true_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
                cv2.putText(vis_frame, pred_text, (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
                
                # Determine if this is a seizure period using the same logic as the text overlay
                # NOTE: true_label is retrieved ONCE per clip (above), then used for ALL frames in that clip
                # This is correct because each clip has a single true_label value
                # The bar should ONLY appear for frames in clips where true_label == 1
                is_seizure_period = (true_label == 1)  # Same check used for true_text above
                
                # Calculate current frame number in the whole video
                current_frame_num = cumulative_frame + i + 1
                
                # Debug: Log first frame of each clip to verify true_label is correct
                if i == 0 and (clip_idx < 5 or true_label == 1):
                    print(f"      Frame {current_frame_num} (clip {clip_name}, frame {i+1}/{len(frames)}): is_seizure_period={is_seizure_period}, true_label={true_label}")
                
                # Top right: Legend and frame counter
                legend_text = "Red: Positive, Blue: Negative"
                legend_x = width - 250
                cv2.putText(vis_frame, legend_text, (legend_x, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
                
                # Frame counter and time indicator (below legend)
                time_seconds = current_frame_num / fps
                minutes = int(time_seconds // 60)
                seconds = time_seconds % 60
                frame_time_text = f"Frame: {current_frame_num} | Time: {minutes:02d}:{seconds:05.2f}"
                cv2.putText(vis_frame, frame_time_text, (legend_x, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
                
                # Draw prediction and seizure period indicator bars above text overlays
                # The magenta bar shows predicted seizure periods, the yellow bar shows actual seizure periods.
                # A red inverted triangle moves along the timeline to indicate the current frame position.
                # Bottom text includes: clip name, metrics (AUC, F1), and progress indicator
                bottom_text_height = 60  # Approximate height for all bottom text elements
                # Draw prediction bar first (above seizure bar)
                draw_prediction_timeline_bar(vis_frame, current_frame_num, total_frames, 
                                            prediction_frame_ranges, height, width, bottom_text_height)
                # Draw seizure bar (below prediction bar)
                draw_seizure_timeline_bar(vis_frame, current_frame_num, total_frames, 
                                         seizure_frame_ranges, height, width, bottom_text_height)
                
                # Bottom left: Clip name
                clip_text = f'Clip: {clip_name}'
                cv2.putText(vis_frame, clip_text, (10, height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
                
                # Bottom right: AUC and F1 score, and progress indicator (if available in video_metrics)
                metrics_x = width - 150
                metrics_y_start = height - 30
                
                if video_metrics is not None:
                    auc_text = f'AUC: {video_metrics.get("AUC_ROC", "N/A"):.3f}' if "AUC_ROC" in video_metrics else 'AUC: N/A'
                    f1_text = f'F1: {video_metrics.get("F1_Score", "N/A"):.3f}' if "F1_Score" in video_metrics else 'F1: N/A'
                    cv2.putText(vis_frame, auc_text, (metrics_x, metrics_y_start), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
                    cv2.putText(vis_frame, f1_text, (metrics_x, metrics_y_start + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
                
                # Progress indicator
                progress_percent = ((clip_idx + 1) / len(sorted_clips)) * 100
                progress_text = f"Progress: {progress_percent:.1f}% ({clip_idx + 1}/{len(sorted_clips)})"
                cv2.putText(vis_frame, progress_text, (metrics_x, metrics_y_start + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
                
                # Write frame
                video_writer.write(vis_frame)
            
            # Update cumulative frame count after processing all frames in this clip
            cumulative_frame += len(frames)
            
            # Show progress every 50 clips to reduce output
            if (clip_idx + 1) % 50 == 0 or clip_idx == 0 or clip_idx == len(sorted_clips) - 1:
                print(f"   Processed clip {clip_idx+1}/{len(sorted_clips)}: {clip_name}")
        
        except Exception as e:
            print(f"   Warning: Error processing clip {clip_name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Release video writer
    if video_writer is not None:
        video_writer.release()
        print(f"   ✅ Whole video visualization ({normalization_mode} normalization) saved to {output_video}")
    else:
        print("   ⚠️  No frames were processed for the whole video visualization.")

