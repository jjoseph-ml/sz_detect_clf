"""
Functions for creating saliency timeline visualizations (line plots and heatmaps) for keypoints over time.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.patches import Rectangle


def get_body_part_groups():
    """
    Get body part groups mapping (group name -> (start_idx, end_idx)).
    
    Returns:
        dict: Dictionary mapping body part group names to (start_index, end_index) tuples
    """
    return {
        'Body': (0, 16),
        'Left Foot': (17, 19),
        'Right Foot': (20, 22),
        'Face Contour': (23, 39),
        'Right Eyebrow': (40, 44),
        'Left Eyebrow': (45, 49),
        'Nose': (50, 58),
        'Right Eye': (59, 64),
        'Left Eye': (65, 70),
        'Mouth Outer': (71, 82),
        'Mouth Inner': (83, 90),
        'Left Hand': (91, 111),
        'Right Hand': (112, 132)
    }


def get_coco_wholebody_keypoint_names():
    """
    Get COCO whole body keypoint names mapping (index -> name).
    
    Returns:
        dict: Dictionary mapping keypoint index (0-132) to keypoint name
    """
    # COCO whole body keypoint structure:
    # 0-16: 17 body keypoints
    # 17-22: 6 foot keypoints
    # 23-90: 68 face keypoints
    # 91-132: 42 hand keypoints (21 per hand)
    
    keypoint_names = {}
    
    # Body keypoints (0-16)
    body_names = [
        'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
        'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
        'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
        'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
    ]
    for i, name in enumerate(body_names):
        keypoint_names[i] = name
    
    # Foot keypoints (17-22)
    foot_names = [
        'left_big_toe', 'left_small_toe', 'left_heel',
        'right_big_toe', 'right_small_toe', 'right_heel'
    ]
    for i, name in enumerate(foot_names):
        keypoint_names[17 + i] = name
    
    # Face keypoints (23-90) - 68 keypoints grouped into major regions
    # Standard 68-point facial landmark structure:
    # 0-16: Face contour/jawline (17 points)
    # 17-21: Right eyebrow (5 points)
    # 22-26: Left eyebrow (5 points)
    # 27-35: Nose (9 points)
    # 36-41: Right eye (6 points)
    # 42-47: Left eye (6 points)
    # 48-59: Mouth outer contour (12 points)
    # 60-67: Mouth inner contour (8 points)
    
    face_region_map = {
        'face_contour': list(range(17)),      # f0-f16
        'R_eyebrow': list(range(17, 22)),     # f17-f21
        'L_eyebrow': list(range(22, 27)),      # f22-f26
        'nose': list(range(27, 36)),          # f27-f35
        'R_eye': list(range(36, 42)),        # f36-f41
        'L_eye': list(range(42, 48)),         # f42-f47
        'mouth_outer': list(range(48, 60)),   # f48-f59
        'mouth_inner': list(range(60, 68))     # f60-f67
    }
    
    # Map each face keypoint to its region name
    for region_name, indices in face_region_map.items():
        for local_idx in indices:
            global_idx = 23 + local_idx  # Face keypoints start at index 23
            keypoint_names[global_idx] = region_name
    
    # Hand keypoints (91-132) - 42 keypoints (21 per hand)
    # Left hand (91-111)
    left_hand_names = [
        'left_hand_root', 'left_thumb1', 'left_thumb2', 'left_thumb3', 'left_thumb4',
        'left_forefinger1', 'left_forefinger2', 'left_forefinger3', 'left_forefinger4',
        'left_middle_finger1', 'left_middle_finger2', 'left_middle_finger3', 'left_middle_finger4',
        'left_ring_finger1', 'left_ring_finger2', 'left_ring_finger3', 'left_ring_finger4',
        'left_pinky_finger1', 'left_pinky_finger2', 'left_pinky_finger3', 'left_pinky_finger4'
    ]
    for i, name in enumerate(left_hand_names):
        keypoint_names[91 + i] = name
    
    # Right hand (112-132)
    right_hand_names = [
        'right_hand_root', 'right_thumb1', 'right_thumb2', 'right_thumb3', 'right_thumb4',
        'right_forefinger1', 'right_forefinger2', 'right_forefinger3', 'right_forefinger4',
        'right_middle_finger1', 'right_middle_finger2', 'right_middle_finger3', 'right_middle_finger4',
        'right_ring_finger1', 'right_ring_finger2', 'right_ring_finger3', 'right_ring_finger4',
        'right_pinky_finger1', 'right_pinky_finger2', 'right_pinky_finger3', 'right_pinky_finger4'
    ]
    for i, name in enumerate(right_hand_names):
        keypoint_names[112 + i] = name
    
    return keypoint_names


def plot_keypoint_saliency_over_time_whole_video(video_id, video_clips, output_dir, moving_avg_window=10):
    """
    Create line plots showing raw saliency values over time for each keypoint across the entire video.
    Seizure sections are highlighted with background shading.
    
    IMPORTANT: This function uses RAW saliency values without any normalization.
    Normalization is only applied to heatmaps (see create_keypoint_saliency_heatmap()).
    
    Args:
        video_id (str): Video ID
        video_clips (dict): Dictionary of clip data {clip_name: {saliency_map, true_label, pred_label, confidence}}
        output_dir (Path): Output directory (should be k_fold/saliency_maps/{video_id})
        moving_avg_window (int): Window size for moving average smoothing in body part timeline plots (default: 10 frames)
    """
    print(f"\n--- Creating Keypoint Saliency Over Time Plots for Video: {video_id} ---")
    
    # Create output subdirectory
    timeline_dir = output_dir / "keypoint_saliency_timeline"
    timeline_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {timeline_dir}")
    
    # Get all clip names and sort them chronologically by segment number
    clip_names = list(video_clips.keys())
    
    # Create mapping of clip names to their temporal position (same logic as create_whole_video_aggregated_saliency)
    clip_timeline = {}
    for clip in clip_names:
        try:
            seg_part = clip.split('_Seg_')[1]
            if '_' in seg_part:
                seg_num = int(seg_part.split('_')[0])
            else:
                seg_num = int(seg_part)
            clip_timeline[clip] = seg_num
        except (IndexError, ValueError):
            print(f"Warning: Error parsing segment number for {clip}, placing at end")
            clip_timeline[clip] = 999999
    
    # Sort clips by their temporal position
    sorted_clips = sorted(clip_names, key=lambda x: clip_timeline.get(x, 999999))
    print(f"Sorted {len(sorted_clips)} clips chronologically")
    
    # Determine number of keypoints from first clip's saliency map
    if not sorted_clips:
        print("Warning: No clips found. Cannot generate keypoint plots.")
        return
    
    first_clip_data = video_clips[sorted_clips[0]]
    first_saliency_map = first_clip_data['saliency_map']
    
    if first_saliency_map.ndim != 2:
        print(f"Warning: Expected saliency map with shape (T, V), but got {first_saliency_map.shape}")
        return
    
    num_keypoints = first_saliency_map.shape[1]
    print(f"Detected {num_keypoints} keypoints. Generating plots for all keypoints...")
    
    # Loop through all keypoints
    for keypoint_idx in range(num_keypoints):
        if (keypoint_idx + 1) % 10 == 0 or keypoint_idx == 0:
            print(f"  Generating plot {keypoint_idx + 1}/{num_keypoints}...")
        
        # Collect saliency values and seizure frame information across all clips
        all_saliency_values = []
        seizure_frame_mask = []
        current_frame = 0
        
        for clip_name in sorted_clips:
            clip_data = video_clips[clip_name]
            saliency_map = clip_data['saliency_map']
            true_label = clip_data['true_label']
            
            # Validate saliency map shape
            if saliency_map.ndim != 2 or saliency_map.shape[1] != num_keypoints:
                print(f"Warning: Skipping clip {clip_name} due to shape mismatch")
                continue
            
            # Extract saliency values for this keypoint (raw, no normalization)
            # NOTE: Timeline plots always use raw values - normalization is only for heatmaps
            keypoint_saliency = saliency_map[:, keypoint_idx]
            
            # Append to the continuous timeline (raw values preserved)
            all_saliency_values.extend(keypoint_saliency.tolist())
            
            # Mark frames as seizure (true_label == 1) or non-seizure (true_label == 0)
            num_frames = len(keypoint_saliency)
            is_seizure = (true_label == 1)
            seizure_frame_mask.extend([is_seizure] * num_frames)
            
            current_frame += num_frames
        
        if not all_saliency_values:
            print(f"Warning: No saliency values found for keypoint {keypoint_idx}")
            continue
        
        # Convert to numpy arrays for plotting
        saliency_array = np.array(all_saliency_values)
        seizure_mask = np.array(seizure_frame_mask)
        frame_numbers = np.arange(len(saliency_array))
        
        # Create the plot with two subplots stacked vertically
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=False)
        
        # ===== TOP SUBPLOT: Line plot showing saliency over time =====
        # Highlight seizure sections with background shading
        seizure_frames = frame_numbers[seizure_mask]
        if len(seizure_frames) > 0:
            # Find continuous seizure regions
            seizure_regions = []
            start = None
            for i, is_seizure in enumerate(seizure_mask):
                if is_seizure and start is None:
                    start = i
                elif not is_seizure and start is not None:
                    seizure_regions.append((start, i - 1))
                    start = None
            if start is not None:
                seizure_regions.append((start, len(seizure_mask) - 1))
            
            # Highlight each seizure region
            for start_frame, end_frame in seizure_regions:
                ax1.axvspan(start_frame, end_frame, alpha=0.3, color='red', label='Seizure' if start_frame == seizure_regions[0][0] else '')
        
        # Plot the saliency values (raw, no normalization)
        # NOTE: Timeline plots use raw values - normalization is only applied to heatmaps
        ax1.plot(frame_numbers, saliency_array, linewidth=1, color='blue', alpha=0.7)
        
        # Add horizontal reference line at y=0
        ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5, linewidth=1)
        
        # Set labels and title for top subplot
        ax1.set_xlabel('Frame Number', fontsize=12)
        ax1.set_ylabel('Raw Saliency Value', fontsize=12)
        ax1.set_title(f'{video_id} - Keypoint {keypoint_idx} Saliency Over Time', fontsize=13, fontweight='bold')
        
        # Add grid for better readability
        ax1.grid(True, alpha=0.3)
        
        # Add legend if seizure regions exist
        if len(seizure_frames) > 0:
            ax1.legend(loc='upper right')
        
        # Set y-axis limits to show full range (including negative values)
        y_min = saliency_array.min()
        y_max = saliency_array.max()
        y_margin = (y_max - y_min) * 0.05 if y_max != y_min else 0.1
        ax1.set_ylim(y_min - y_margin, y_max + y_margin)
        
        # ===== BOTTOM SUBPLOT: Histogram showing distribution of saliency values =====
        # Create histogram of saliency values
        n_bins = min(50, len(saliency_array) // 10) if len(saliency_array) > 50 else 20  # Adaptive bin count
        n_bins = max(10, n_bins)  # Minimum 10 bins
        
        counts, bins, patches = ax2.hist(saliency_array, bins=n_bins, edgecolor='black', alpha=0.7, color='steelblue')
        
        # Color bars based on positive/negative values
        for i, (count, bin_left, bin_right, patch) in enumerate(zip(counts, bins[:-1], bins[1:], patches)):
            bin_center = (bin_left + bin_right) / 2
            if bin_center > 0:
                patch.set_facecolor('green')
                patch.set_alpha(0.6)
            elif bin_center < 0:
                patch.set_facecolor('red')
                patch.set_alpha(0.6)
            else:
                patch.set_facecolor('gray')
                patch.set_alpha(0.6)
        
        # Add vertical reference line at x=0
        ax2.axvline(x=0, color='gray', linestyle='--', alpha=0.5, linewidth=1)
        
        # Set labels and title for bottom subplot
        ax2.set_xlabel('Raw Saliency Value', fontsize=12)
        ax2.set_ylabel('Frequency (Number of Frames)', fontsize=12)
        ax2.set_title('Distribution of Saliency Values', fontsize=12, fontweight='bold')
        
        # Add grid for better readability
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Add statistics text box
        mean_val = np.mean(saliency_array)
        std_val = np.std(saliency_array)
        median_val = np.median(saliency_array)
        stats_text = f'Mean: {mean_val:.4f}\nStd: {std_val:.4f}\nMedian: {median_val:.4f}'
        ax2.text(0.98, 0.98, stats_text, transform=ax2.transAxes, 
                fontsize=10, verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        
        # Save the plot
        output_path = timeline_dir / f"{video_id}_keypoint_{keypoint_idx:03d}_saliency_over_time.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    print(f"✅ Generated {num_keypoints} keypoint saliency plots saved to: {timeline_dir}")
    
    # Generate heatmap visualizations for all keypoints (4 normalization modes)
    print(f"\n--- Creating Keypoint Saliency Heatmaps for Video: {video_id} ---")
    
    # 1. Percentile global normalization
    print(f"\n[1/4] Generating percentile global normalization heatmap...")
    try:
        create_keypoint_saliency_heatmap(video_id, video_clips, sorted_clips, num_keypoints, timeline_dir, 
                                        normalize_per_keypoint=False, normalization_method='percentile')
        print(f"✅ Percentile global normalization heatmap completed")
    except Exception as e:
        print(f"❌ Error generating percentile global heatmap: {e}")
        import traceback
        traceback.print_exc()
    
    # 2. Percentile per-keypoint normalization
    print(f"\n[2/4] Generating percentile per-keypoint normalization heatmap...")
    try:
        create_keypoint_saliency_heatmap(video_id, video_clips, sorted_clips, num_keypoints, timeline_dir, 
                                        normalize_per_keypoint=True, normalization_method='percentile')
        print(f"✅ Percentile per-keypoint normalization heatmap completed")
    except Exception as e:
        print(f"❌ Error generating percentile per-keypoint heatmap: {e}")
        import traceback
        traceback.print_exc()
    
    # 3. Z-score global normalization
    print(f"\n[3/4] Generating z-score global normalization heatmap...")
    try:
        create_keypoint_saliency_heatmap(video_id, video_clips, sorted_clips, num_keypoints, timeline_dir, 
                                        normalize_per_keypoint=False, normalization_method='zscore')
        print(f"✅ Z-score global normalization heatmap completed")
    except Exception as e:
        print(f"❌ Error generating z-score global heatmap: {e}")
        import traceback
        traceback.print_exc()
    
    # 4. Z-score per-keypoint normalization
    print(f"\n[4/4] Generating z-score per-keypoint normalization heatmap...")
    try:
        create_keypoint_saliency_heatmap(video_id, video_clips, sorted_clips, num_keypoints, timeline_dir, 
                                        normalize_per_keypoint=True, normalization_method='zscore')
        print(f"✅ Z-score per-keypoint normalization heatmap completed")
    except Exception as e:
        print(f"❌ Error generating z-score per-keypoint heatmap: {e}")
        import traceback
        traceback.print_exc()
    
    print(f"\n✅ All 4 heatmaps generated successfully!")
    
    # Optional: Generate grouped body part visualizations
    try:
        plot_grouped_body_part_visualizations(video_id, video_clips, output_dir, moving_avg_window=moving_avg_window)
    except Exception as e:
        print(f"Warning: Could not generate grouped body part visualizations: {e}")
        import traceback
        traceback.print_exc()


def calculate_percentile_normalization(saliency_heatmap, percentile_low=5, percentile_high=95):
    """
    Calculate percentile-based normalization values for a saliency heatmap.
    This focuses on the more interesting values while still showing the full range.
    Normalization is calculated across ALL keypoints and frames (global normalization).
    
    Args:
        saliency_heatmap (np.ndarray): 2D array of shape (num_keypoints, num_frames) with saliency values
        percentile_low (int): Lower percentile to use (default: 5)
        percentile_high (int): Upper percentile to use (default: 95)
    
    Returns:
        tuple: (vmin, vmax) - Normalized value range for colormap
    """
    # Calculate percentiles for positive and negative values separately
    positive_values = saliency_heatmap[saliency_heatmap > 0]
    negative_values = saliency_heatmap[saliency_heatmap < 0]
    
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
        vmax = max(abs(saliency_heatmap.min()), abs(saliency_heatmap.max()))
    
    # Ensure vmax is not zero
    if vmax < 1e-8:
        vmax = max(abs(saliency_heatmap.min()), abs(saliency_heatmap.max()))
        if vmax < 1e-8:
            vmax = 1.0  # Default fallback
    
    vmin = -vmax
    
    return vmin, vmax


def calculate_percentile_normalization_absolute(saliency_heatmap, percentile_low=5, percentile_high=95):
    """
    Calculate percentile-based normalization values for absolute saliency values.
    This focuses on the more interesting values while still showing the full range.
    Normalization is calculated across ALL keypoints and frames (global normalization).
    For absolute values, uses single-sided percentile calculation.
    
    Args:
        saliency_heatmap (np.ndarray): 2D array of shape (num_keypoints, num_frames) with non-negative saliency values
        percentile_low (int): Lower percentile to use (default: 5)
        percentile_high (int): Upper percentile to use (default: 95)
    
    Returns:
        tuple: (vmin, vmax) - Normalized value range for colormap (vmin will be 0.0)
    """
    # For absolute values, use single-sided percentile calculation
    vmax = np.percentile(saliency_heatmap, percentile_high)
    
    # Ensure vmax is not zero
    if vmax < 1e-8:
        vmax = saliency_heatmap.max()
        if vmax < 1e-8:
            vmax = 1.0  # Default fallback
    
    vmin = 0.0
    
    return vmin, vmax


def calculate_percentile_normalization_per_keypoint(saliency_heatmap, percentile_low=5, percentile_high=95):
    """
    Calculate percentile-based normalization values per keypoint (per row).
    Each keypoint gets its own normalization range, which helps visualize activity
    in keypoints with lower overall saliency values.
    
    Args:
        saliency_heatmap (np.ndarray): 2D array of shape (num_keypoints, num_frames) with saliency values
        percentile_low (int): Lower percentile to use (default: 5)
        percentile_high (int): Upper percentile to use (default: 95)
    
    Returns:
        np.ndarray: 2D array of shape (num_keypoints, 2) where each row is [vmin, vmax] for that keypoint
    """
    num_keypoints = saliency_heatmap.shape[0]
    normalization_ranges = np.zeros((num_keypoints, 2))  # [vmin, vmax] for each keypoint
    
    for keypoint_idx in range(num_keypoints):
        keypoint_saliency = saliency_heatmap[keypoint_idx, :]  # Get all frames for this keypoint
        
        # Calculate percentiles for positive and negative values separately for this keypoint
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
            # Fallback to absolute max if no positive/negative values
            vmax = max(abs(keypoint_saliency.min()), abs(keypoint_saliency.max()))
        
        # Ensure vmax is not zero
        if vmax < 1e-8:
            vmax = max(abs(keypoint_saliency.min()), abs(keypoint_saliency.max()))
            if vmax < 1e-8:
                vmax = 1.0  # Default fallback
        
        vmin = -vmax
        normalization_ranges[keypoint_idx, 0] = vmin
        normalization_ranges[keypoint_idx, 1] = vmax
    
    return normalization_ranges


def calculate_zscore_normalization(saliency_heatmap, num_std=3):
    """
    Calculate z-score normalization values for a saliency heatmap.
    Uses mean and standard deviation across ALL keypoints and frames (global normalization).
    
    Args:
        saliency_heatmap (np.ndarray): 2D array of shape (num_keypoints, num_frames) with saliency values
        num_std (float): Number of standard deviations to use for the range (default: 3)
    
    Returns:
        tuple: (vmin, vmax) - Normalized value range for colormap in z-score units
    """
    # Calculate mean and standard deviation across all values
    mean_val = np.mean(saliency_heatmap)
    std_val = np.std(saliency_heatmap)
    
    # Use symmetric range: mean ± num_std * std
    vmax = num_std * std_val if std_val > 1e-8 else 1.0
    vmin = -vmax
    
    return vmin, vmax


def calculate_zscore_normalization_per_keypoint(saliency_heatmap, num_std=3):
    """
    Calculate z-score normalization values per keypoint (per row).
    Each keypoint gets its own normalization range based on its mean and std.
    
    Args:
        saliency_heatmap (np.ndarray): 2D array of shape (num_keypoints, num_frames) with saliency values
        num_std (float): Number of standard deviations to use for the range (default: 3)
    
    Returns:
        np.ndarray: 2D array of shape (num_keypoints, 2) where each row is [vmin, vmax] in z-score units
    """
    num_keypoints = saliency_heatmap.shape[0]
    normalization_ranges = np.zeros((num_keypoints, 2))  # [vmin, vmax] for each keypoint
    
    for keypoint_idx in range(num_keypoints):
        keypoint_saliency = saliency_heatmap[keypoint_idx, :]  # Get all frames for this keypoint
        
        # Calculate mean and standard deviation for this keypoint
        mean_val = np.mean(keypoint_saliency)
        std_val = np.std(keypoint_saliency)
        
        # Use symmetric range: mean ± num_std * std
        if std_val > 1e-8:
            vmax = num_std * std_val
        else:
            # Fallback if std is too small
            vmax = max(abs(keypoint_saliency.min() - mean_val), abs(keypoint_saliency.max() - mean_val))
            if vmax < 1e-8:
                vmax = 1.0
        
        vmin = -vmax
        normalization_ranges[keypoint_idx, 0] = vmin
        normalization_ranges[keypoint_idx, 1] = vmax
    
    return normalization_ranges


def create_keypoint_saliency_heatmap(video_id, video_clips, sorted_clips, num_keypoints, output_dir, 
                                     normalize_per_keypoint=False, normalization_method='percentile',
                                     percentile_low=5, percentile_high=95, zscore_num_std=3):
    """
    Create a heatmap showing saliency values for all keypoints over time across the entire video.
    
    IMPORTANT: This function applies normalization to enhance visualization contrast.
    The individual timeline plots (plot_keypoint_saliency_over_time_whole_video) use raw values.
    
    Args:
        video_id (str): Video ID
        video_clips (dict): Dictionary of clip data {clip_name: {saliency_map, true_label, pred_label, confidence}}
        sorted_clips (list): List of clip names sorted chronologically
        num_keypoints (int): Number of keypoints (133)
        output_dir (Path): Output directory for saving the heatmap
        normalize_per_keypoint (bool): If True, normalize each keypoint independently. If False, use global normalization (default: False)
        normalization_method (str): 'percentile' or 'zscore' (default: 'percentile')
        percentile_low (int): Lower percentile for percentile normalization (default: 5)
        percentile_high (int): Upper percentile for percentile normalization (default: 95)
        zscore_num_std (float): Number of standard deviations for z-score normalization (default: 3)
    """
    norm_mode_str = "per-keypoint" if normalize_per_keypoint else "global"
    print(f"Building 2D saliency array (keypoints × frames) for {norm_mode_str} normalization...")
    
    # Build 2D array: rows = keypoints (0-132), columns = frames
    all_saliency_matrix = []  # Will be list of lists, each inner list is one frame's saliency for all keypoints
    seizure_frame_mask = []
    
    # Process each clip chronologically
    for clip_name in sorted_clips:
        clip_data = video_clips[clip_name]
        saliency_map = clip_data['saliency_map']
        true_label = clip_data['true_label']
        
        # Validate saliency map shape
        if saliency_map.ndim != 2 or saliency_map.shape[1] != num_keypoints:
            print(f"Warning: Skipping clip {clip_name} due to shape mismatch")
            continue
        
        # saliency_map shape is (T, V) where T=time/frames, V=keypoints
        # We need to transpose it to (V, T) and then append columns (frames)
        # For each frame, we want all keypoint values
        num_frames = saliency_map.shape[0]
        
        # Append each frame as a column (all keypoint values for that frame)
        for frame_idx in range(num_frames):
            # Get saliency values for all keypoints at this frame
            frame_saliency = saliency_map[frame_idx, :]  # Shape: (V,)
            all_saliency_matrix.append(frame_saliency)
            
            # Mark if this frame is part of a seizure clip
            is_seizure = (true_label == 1)
            seizure_frame_mask.append(is_seizure)
    
    if not all_saliency_matrix:
        print("Warning: No saliency data found. Cannot create heatmap.")
        return
    
    # Convert to numpy array: shape will be (num_frames, num_keypoints)
    # Then transpose to (num_keypoints, num_frames) for heatmap
    saliency_array = np.array(all_saliency_matrix)  # Shape: (num_frames, num_keypoints)
    saliency_heatmap = saliency_array.T  # Shape: (num_keypoints, num_frames)
    
    seizure_mask = np.array(seizure_frame_mask)
    num_frames = saliency_heatmap.shape[1]
    
    print(f"Heatmap shape: {saliency_heatmap.shape} (keypoints × frames)")
    print(f"Total frames: {num_frames}")
    print(f"Saliency range: [{saliency_heatmap.min():.6f}, {saliency_heatmap.max():.6f}]")
    
    # Find continuous seizure regions
    seizure_regions = []
    start = None
    for i, is_seizure in enumerate(seizure_mask):
        if is_seizure and start is None:
            start = i
        elif not is_seizure and start is not None:
            seizure_regions.append((start, i - 1))
            start = None
    if start is not None:
        seizure_regions.append((start, len(seizure_mask) - 1))
    
    print(f"Found {len(seizure_regions)} seizure region(s)")
    
    # Create the heatmap figure
    fig, ax = plt.subplots(figsize=(16, 10))
    
    # Apply normalization based on the selected mode and method
    if normalization_method == 'zscore':
        if normalize_per_keypoint:
            # Per-keypoint z-score normalization
            print(f"Using per-keypoint z-score normalization ({zscore_num_std} std)...")
            normalization_ranges = calculate_zscore_normalization_per_keypoint(
                saliency_heatmap, zscore_num_std
            )
            
            # Create a normalized copy where each row is z-score normalized
            normalized_heatmap = np.zeros_like(saliency_heatmap)
            for keypoint_idx in range(num_keypoints):
                keypoint_data = saliency_heatmap[keypoint_idx, :]
                mean_val = np.mean(keypoint_data)
                std_val = np.std(keypoint_data)
                
                if std_val > 1e-8:
                    # Z-score: (x - mean) / std
                    z_scores = (keypoint_data - mean_val) / std_val
                    # Clip to ±num_std range
                    normalized_heatmap[keypoint_idx, :] = np.clip(
                        z_scores, -zscore_num_std, zscore_num_std
                    )
                else:
                    normalized_heatmap[keypoint_idx, :] = 0.0
            
            # For display, use fixed range based on num_std
            vmin_display, vmax_display = -zscore_num_std, zscore_num_std
            heatmap_to_plot = normalized_heatmap
            
            # Print summary statistics
            vmax_range = normalization_ranges[:, 1]
            print(f"Per-keypoint z-score: vmax ranges from {vmax_range.min():.4f} to {vmax_range.max():.4f}")
            print(f"Display range: [{vmin_display:.4f}, {vmax_display:.4f}] (z-score units)")
        else:
            # Global z-score normalization
            print(f"Using global z-score normalization ({zscore_num_std} std)...")
            mean_val = np.mean(saliency_heatmap)
            std_val = np.std(saliency_heatmap)
            
            # Z-score normalize the entire heatmap
            if std_val > 1e-8:
                z_scores = (saliency_heatmap - mean_val) / std_val
                heatmap_to_plot = np.clip(z_scores, -zscore_num_std, zscore_num_std)
            else:
                heatmap_to_plot = np.zeros_like(saliency_heatmap)
            
            vmin_display, vmax_display = -zscore_num_std, zscore_num_std
            print(f"Global z-score: mean={mean_val:.4f}, std={std_val:.4f}")
            print(f"Display range: [{vmin_display:.4f}, {vmax_display:.4f}] (z-score units)")
    else:  # percentile normalization
        if normalize_per_keypoint:
            # Per-keypoint percentile normalization
            print(f"Using per-keypoint normalization (percentile {percentile_low}-{percentile_high})...")
            normalization_ranges = calculate_percentile_normalization_per_keypoint(
                saliency_heatmap, percentile_low, percentile_high
            )
            
            # Create a normalized copy of the heatmap where each row is normalized to [-1, 1]
            normalized_heatmap = np.zeros_like(saliency_heatmap)
            for keypoint_idx in range(num_keypoints):
                vmin, vmax = normalization_ranges[keypoint_idx, 0], normalization_ranges[keypoint_idx, 1]
                keypoint_data = saliency_heatmap[keypoint_idx, :]
                
                # Normalize to [-1, 1] range
                if vmax > 1e-8:  # Avoid division by zero
                    # Map [vmin, vmax] to [-1, 1] preserving zero
                    normalized_heatmap[keypoint_idx, :] = np.clip(
                        keypoint_data / vmax, -1.0, 1.0
                    )
                else:
                    normalized_heatmap[keypoint_idx, :] = keypoint_data
            
            # For display, use fixed range [-1, 1] since we've normalized each row
            vmin_display, vmax_display = -1.0, 1.0
            heatmap_to_plot = normalized_heatmap
            
            # Print summary statistics
            vmax_range = normalization_ranges[:, 1]
            print(f"Per-keypoint normalization: vmax ranges from {vmax_range.min():.4f} to {vmax_range.max():.4f}")
            print(f"Display range: [{vmin_display:.4f}, {vmax_display:.4f}] (normalized)")
        else:
            # Global percentile normalization
            print(f"Using global normalization (percentile {percentile_low}-{percentile_high})...")
            vmin_display, vmax_display = calculate_percentile_normalization(
                saliency_heatmap, percentile_low, percentile_high
            )
            heatmap_to_plot = saliency_heatmap
            print(f"Global normalization: vmin={vmin_display:.4f}, vmax={vmax_display:.4f}")
    
    # Create heatmap with RdBu (Red-Blue) diverging colormap
    # RdBu goes from blue (negative) through white (zero) to red (positive)
    im = ax.imshow(heatmap_to_plot, aspect='auto', cmap='RdBu', vmin=vmin_display, vmax=vmax_display, 
                   interpolation='nearest', origin='lower')
    
    # Highlight seizure regions with vertical red shading
    for start_frame, end_frame in seizure_regions:
        ax.axvspan(start_frame - 0.5, end_frame + 0.5, alpha=0.2, color='red', zorder=0)
    
    # Set axis labels and title
    ax.set_xlabel('Frame Number', fontsize=14, fontweight='bold')
    ax.set_ylabel('Keypoint Index', fontsize=14, fontweight='bold')
    norm_scope = 'Per-Keypoint' if normalize_per_keypoint else 'Global'
    norm_method_str = 'Z-Score' if normalization_method == 'zscore' else 'Percentile'
    ax.set_title(f'Saliency Heatmap: {video_id}\nAll Keypoints Over Time ({norm_scope} {norm_method_str})', 
                fontsize=16, fontweight='bold', pad=20)
    
    # Set y-axis ticks to show keypoint names
    # Get keypoint names mapping
    keypoint_names = get_coco_wholebody_keypoint_names()
    
    # Show keypoint names on y-axis with smart spacing
    # Show all keypoints but use abbreviated names and smaller font
    yticks = np.arange(0, num_keypoints)
    yticklabels = []
    
    for i in yticks:
        if i in keypoint_names:
            name = keypoint_names[i]
            # Abbreviate names for readability
            # Face regions are already grouped, so use them as-is with abbreviation
            if name in ['face_contour', 'R_eyebrow', 'L_eyebrow', 'nose', 'R_eye', 'L_eye', 'mouth_outer', 'mouth_inner']:
                # Face region names - already abbreviated
                name_abbrev = name.replace('face_contour', 'face_cntr').replace('mouth_outer', 'mouth_out')
                name_abbrev = name_abbrev.replace('mouth_inner', 'mouth_in')
                yticklabels.append(name_abbrev)
            else:
                # Abbreviate other keypoint names
                name_abbrev = name.replace('left_', 'L_').replace('right_', 'R_')
                name_abbrev = name_abbrev.replace('_finger', '').replace('_toe', '')
                name_abbrev = name_abbrev.replace('forefinger', 'fore').replace('middle_finger', 'mid')
                name_abbrev = name_abbrev.replace('ring_finger', 'ring').replace('pinky_finger', 'pink')
                name_abbrev = name_abbrev.replace('hand_root', 'wrist')
                name_abbrev = name_abbrev.replace('shoulder', 'shldr').replace('elbow', 'elb')
                name_abbrev = name_abbrev.replace('wrist', 'wrst').replace('ankle', 'ank')
                name_abbrev = name_abbrev.replace('big_toe', 'btoe').replace('small_toe', 'stoe')
                yticklabels.append(name_abbrev)
        else:
            yticklabels.append(str(i + 1))
    
    # Set ticks with all keypoint names (small font to fit)
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels, fontsize=6)  # Small font to fit all 133 names
    
    # Add body part group labels to the left of keypoint labels
    body_part_groups = get_body_part_groups()
    
    # Create secondary y-axis for group labels (positioned on the left)
    ax2 = ax.twinx()
    ax2.set_ylim(ax.get_ylim())  # Match the y-limits of the main axis
    
    # Set group labels at the midpoint of each group's keypoint range
    group_yticks = []
    group_yticklabels = []
    
    for group_name, (start_idx, end_idx) in body_part_groups.items():
        # Calculate midpoint of the group range
        midpoint = (start_idx + end_idx) / 2.0
        group_yticks.append(midpoint)
        group_yticklabels.append(group_name)
    
    # Set the group labels on the secondary axis (left side)
    ax2.set_yticks(group_yticks)
    ax2.set_yticklabels(group_yticklabels, fontsize=11, fontweight='bold')
    ax2.tick_params(axis='y', which='major', pad=15, left=True, right=False)  # Show ticks on left only
    
    # Move the secondary y-axis to the left side
    ax2.yaxis.set_ticks_position('left')
    ax2.yaxis.set_label_position('left')
    
    # Hide the right spine of the secondary axis
    ax2.spines['right'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax2.spines['bottom'].set_visible(False)
    
    # Keep the original keypoint labels on the right side
    ax.yaxis.set_ticks_position('right')
    ax.yaxis.set_label_position('right')
    
    # Set x-axis ticks (show every 2000 frames or similar to avoid crowding)
    if num_frames > 5000:
        xtick_spacing = max(2000, num_frames // 10)
    elif num_frames > 1000:
        xtick_spacing = max(500, num_frames // 10)
    else:
        xtick_spacing = max(100, num_frames // 10)
    
    xticks = np.arange(0, num_frames, xtick_spacing)
    ax.set_xticks(xticks)
    ax.set_xticklabels([str(int(x)) for x in xticks], fontsize=10)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.02, pad=0.02)
    if normalization_method == 'zscore':
        if normalize_per_keypoint:
            cbar.set_label('Z-Score (per keypoint)', fontsize=12, fontweight='bold', rotation=270, labelpad=20)
        else:
            cbar.set_label('Z-Score', fontsize=12, fontweight='bold', rotation=270, labelpad=20)
    else:
        if normalize_per_keypoint:
            cbar.set_label('Normalized Saliency Value (per keypoint)', fontsize=12, fontweight='bold', rotation=270, labelpad=20)
        else:
            cbar.set_label('Raw Saliency Value', fontsize=12, fontweight='bold', rotation=270, labelpad=20)
    cbar.ax.tick_params(labelsize=10)
    
    # Add text annotation showing saliency range and normalization info
    actual_range = f'Actual Range: [{saliency_heatmap.min():.4f}, {saliency_heatmap.max():.4f}]'
    if normalization_method == 'zscore':
        mean_val = np.mean(saliency_heatmap)
        std_val = np.std(saliency_heatmap)
        if normalize_per_keypoint:
            display_range = f'Display Range: [{vmin_display:.4f}, {vmax_display:.4f}] (z-score per keypoint)'
            range_text = f'{actual_range}\nMean: {mean_val:.4f}, Std: {std_val:.4f}\n{display_range}\n(Using ±{zscore_num_std} std per keypoint)'
        else:
            display_range = f'Display Range: [{vmin_display:.4f}, {vmax_display:.4f}] (z-score)'
            range_text = f'{actual_range}\nMean: {mean_val:.4f}, Std: {std_val:.4f}\n{display_range}\n(Using ±{zscore_num_std} std globally)'
    else:
        if normalize_per_keypoint:
            display_range = f'Display Range: [{vmin_display:.4f}, {vmax_display:.4f}] (normalized per keypoint)'
            range_text = f'{actual_range}\n{display_range}\n(Using {percentile_low}-{percentile_high} percentile per keypoint)'
        else:
            display_range = f'Display Range: [{vmin_display:.4f}, {vmax_display:.4f}]'
            range_text = f'{actual_range}\n{display_range}\n(Using {percentile_low}-{percentile_high} percentile globally)'
    ax.text(0.02, 0.98, range_text, transform=ax.transAxes, 
            fontsize=10, verticalalignment='top', horizontalalignment='left',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
    
    # Add legend for seizure regions if they exist
    if len(seizure_regions) > 0:
        legend_elements = [
            Rectangle((0, 0), 1, 1, facecolor='red', alpha=0.2, edgecolor='red', linewidth=2, label='Seizure Period')
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=11, framealpha=0.9)
    
    # Add grid for better readability (subtle)
    ax.grid(True, alpha=0.1, linestyle='--', linewidth=0.5)
    
    plt.tight_layout()
    
    # Save the heatmap with appropriate filename based on normalization mode and method
    if normalization_method == 'zscore':
        if normalize_per_keypoint:
            output_path = output_dir / f"{video_id}_keypoint_saliency_heatmap_zscore_per_keypoint.png"
        else:
            output_path = output_dir / f"{video_id}_keypoint_saliency_heatmap_zscore_global.png"
    else:
        if normalize_per_keypoint:
            output_path = output_dir / f"{video_id}_keypoint_saliency_heatmap_percentile_per_keypoint.png"
        else:
            output_path = output_dir / f"{video_id}_keypoint_saliency_heatmap_percentile_global.png"
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')  # High DPI for publication quality
    plt.close()
    
    print(f"✅ Heatmap saved to: {output_path}")


def plot_grouped_body_part_saliency_timeline(video_id, video_clips, output_dir, moving_avg_window=10):
    """
    Create time series plots with body part groups instead of individual keypoints.
    Groups keypoints by body parts and averages saliency values spatially (across keypoints in each group).
    Creates separate plots for mean and median averaging.
    
    IMPORTANT: This function uses RAW saliency values without any normalization.
    
    Args:
        video_id (str): Video ID
        video_clips (dict): Dictionary of clip data {clip_name: {saliency_map, true_label, pred_label, confidence}}
        output_dir (Path): Output directory (should be k_fold/saliency_maps/{video_id})
        moving_avg_window (int): Window size for moving average smoothing (default: 100 frames)
    """
    print(f"\n--- Creating Grouped Body Part Saliency Timeline Plots for Video: {video_id} ---")
    
    # Create output subdirectory
    timeline_dir = output_dir / "body_part_saliency_timeline"
    timeline_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {timeline_dir}")
    
    # Get all clip names and sort them chronologically by segment number
    clip_names = list(video_clips.keys())
    
    # Create mapping of clip names to their temporal position
    clip_timeline = {}
    for clip in clip_names:
        try:
            seg_part = clip.split('_Seg_')[1]
            if '_' in seg_part:
                seg_num = int(seg_part.split('_')[0])
            else:
                seg_num = int(seg_part)
            clip_timeline[clip] = seg_num
        except (IndexError, ValueError):
            print(f"Warning: Error parsing segment number for {clip}, placing at end")
            clip_timeline[clip] = 999999
    
    # Sort clips by their temporal position
    sorted_clips = sorted(clip_names, key=lambda x: clip_timeline.get(x, 999999))
    print(f"Sorted {len(sorted_clips)} clips chronologically")
    
    # Determine number of keypoints from first clip's saliency map
    if not sorted_clips:
        print("Warning: No clips found. Cannot generate body part plots.")
        return
    
    first_clip_data = video_clips[sorted_clips[0]]
    first_saliency_map = first_clip_data['saliency_map']
    
    if first_saliency_map.ndim != 2:
        print(f"Warning: Expected saliency map with shape (T, V), but got {first_saliency_map.shape}")
        return
    
    num_keypoints = first_saliency_map.shape[1]
    print(f"Detected {num_keypoints} keypoints. Grouping by body parts...")
    
    # Get body part groups and convert ranges to keypoint indices
    body_part_groups = get_body_part_groups()
    body_part_indices = {}
    for group_name, (start_idx, end_idx) in body_part_groups.items():
        # Convert to inclusive indices
        body_part_indices[group_name] = list(range(start_idx, end_idx + 1))
        # Filter to only include valid keypoint indices
        body_part_indices[group_name] = [idx for idx in body_part_indices[group_name] if idx < num_keypoints]
    
    print(f"Grouped into {len(body_part_indices)} body part groups")
    
    # Collect saliency values for each body part group across all clips
    body_part_saliency_mean = {group_name: [] for group_name in body_part_indices.keys()}
    body_part_saliency_median = {group_name: [] for group_name in body_part_indices.keys()}
    seizure_frame_mask = []
    
    for clip_name in sorted_clips:
        clip_data = video_clips[clip_name]
        saliency_map = clip_data['saliency_map']
        true_label = clip_data['true_label']
        
        # Validate saliency map shape
        if saliency_map.ndim != 2 or saliency_map.shape[1] != num_keypoints:
            print(f"Warning: Skipping clip {clip_name} due to shape mismatch")
            continue
        
        num_frames = saliency_map.shape[0]
        
        # For each frame, compute mean and median saliency for each body part group
        for frame_idx in range(num_frames):
            frame_saliency = saliency_map[frame_idx, :]  # Shape: (V,)
            
            # Compute mean and median for each body part group
            for group_name, keypoint_indices in body_part_indices.items():
                if len(keypoint_indices) > 0:
                    # Extract saliency values for keypoints in this group
                    group_saliency = frame_saliency[keypoint_indices]
                    # Compute mean and median
                    body_part_saliency_mean[group_name].append(np.mean(group_saliency))
                    body_part_saliency_median[group_name].append(np.median(group_saliency))
            
            # Mark if this frame is part of a seizure clip
            is_seizure = (true_label == 1)
            seizure_frame_mask.append(is_seizure)
    
    if not seizure_frame_mask:
        print("Warning: No saliency data found. Cannot generate body part plots.")
        return
    
    num_frames = len(seizure_frame_mask)
    frame_numbers = np.arange(num_frames)
    seizure_mask = np.array(seizure_frame_mask)
    
    # Find continuous seizure regions
    seizure_regions = []
    start = None
    for i, is_seizure in enumerate(seizure_mask):
        if is_seizure and start is None:
            start = i
        elif not is_seizure and start is not None:
            seizure_regions.append((start, i - 1))
            start = None
    if start is not None:
        seizure_regions.append((start, len(seizure_mask) - 1))
    
    # Create plots for mean and median
    for aggregation_type, body_part_saliency in [('mean', body_part_saliency_mean), ('median', body_part_saliency_median)]:
        # Create the plot with two subplots stacked vertically
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10), sharex=False)
        
        # ===== TOP SUBPLOT: Line plot showing saliency over time =====
        # Highlight seizure sections with background shading
        if len(seizure_regions) > 0:
            for start_frame, end_frame in seizure_regions:
                ax1.axvspan(start_frame, end_frame, alpha=0.3, color='red', 
                           label='Seizure' if start_frame == seizure_regions[0][0] else '')
        
        # Plot lines for each body part group
        colors = plt.cm.tab20(np.linspace(0, 1, len(body_part_indices)))
        for idx, (group_name, saliency_values) in enumerate(body_part_saliency.items()):
            if len(saliency_values) == num_frames:
                saliency_array = np.array(saliency_values)
                ax1.plot(frame_numbers, saliency_array, linewidth=2, 
                        color=colors[idx], alpha=0.8, label=group_name)
        
        # Add horizontal reference line at y=0
        ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5, linewidth=1)
        
        # Set labels and title for top subplot
        ax1.set_xlabel('Frame Number', fontsize=12)
        ax1.set_ylabel('Raw Saliency Value', fontsize=12)
        ax1.set_title(f'{video_id} - Body Part Groups Saliency Over Time ({aggregation_type.capitalize()})', 
                     fontsize=14, fontweight='bold')
        
        # Add grid and legend
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc='best', fontsize=9, ncol=2)
        
        # Set y-axis limits to show full range
        all_values = []
        for saliency_values in body_part_saliency.values():
            if len(saliency_values) == num_frames:
                all_values.extend(saliency_values)
        if all_values:
            y_min = min(all_values)
            y_max = max(all_values)
            y_margin = (y_max - y_min) * 0.05 if y_max != y_min else 0.1
            ax1.set_ylim(y_min - y_margin, y_max + y_margin)
        
        # ===== BOTTOM SUBPLOT: Histogram showing distribution of saliency values =====
        # Combine all body part saliency values for histogram
        all_saliency_for_hist = []
        for saliency_values in body_part_saliency.values():
            if len(saliency_values) == num_frames:
                all_saliency_for_hist.extend(saliency_values)
        
        if all_saliency_for_hist:
            saliency_array = np.array(all_saliency_for_hist)
            
            # Create histogram
            n_bins = min(50, len(saliency_array) // 10) if len(saliency_array) > 50 else 20
            n_bins = max(10, n_bins)
            
            counts, bins, patches = ax2.hist(saliency_array, bins=n_bins, edgecolor='black', alpha=0.7, color='steelblue')
            
            # Color bars based on positive/negative values
            for i, (count, bin_left, bin_right, patch) in enumerate(zip(counts, bins[:-1], bins[1:], patches)):
                bin_center = (bin_left + bin_right) / 2
                if bin_center > 0:
                    patch.set_facecolor('green')
                    patch.set_alpha(0.6)
                elif bin_center < 0:
                    patch.set_facecolor('red')
                    patch.set_alpha(0.6)
                else:
                    patch.set_facecolor('gray')
                    patch.set_alpha(0.6)
            
            # Add vertical reference line at x=0
            ax2.axvline(x=0, color='gray', linestyle='--', alpha=0.5, linewidth=1)
            
            # Set labels and title
            ax2.set_xlabel('Raw Saliency Value', fontsize=12)
            ax2.set_ylabel('Frequency (Number of Frames)', fontsize=12)
            ax2.set_title('Distribution of Saliency Values (All Body Parts)', fontsize=12, fontweight='bold')
            
            # Add grid
            ax2.grid(True, alpha=0.3, axis='y')
            
            # Add statistics text box
            mean_val = np.mean(saliency_array)
            std_val = np.std(saliency_array)
            median_val = np.median(saliency_array)
            stats_text = f'Mean: {mean_val:.4f}\nStd: {std_val:.4f}\nMedian: {median_val:.4f}'
            ax2.text(0.98, 0.98, stats_text, transform=ax2.transAxes, 
                    fontsize=10, verticalalignment='top', horizontalalignment='right',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        
        # Save the plot
        output_path = timeline_dir / f"{video_id}_grouped_body_parts_{aggregation_type}_timeline.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✅ Saved {aggregation_type} timeline plot: {output_path}")
    
    # Generate individual plots for each body part group
    print(f"\n--- Creating Individual Body Part Timeline Plots ---")
    for aggregation_type, body_part_saliency in [('mean', body_part_saliency_mean)]:
        for group_name, saliency_values in body_part_saliency.items():
            if len(saliency_values) != num_frames:
                continue
            
            saliency_array = np.array(saliency_values)
            
            # Create the plot with two subplots stacked vertically
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=False)
            
            # ===== TOP SUBPLOT: Line plot showing saliency over time =====
            # Highlight seizure sections with background shading
            if len(seizure_regions) > 0:
                for start_frame, end_frame in seizure_regions:
                    ax1.axvspan(start_frame, end_frame, alpha=0.3, color='red', 
                               label='Seizure' if start_frame == seizure_regions[0][0] else '')
            
            # Plot the saliency values (raw, no normalization)
            ax1.plot(frame_numbers, saliency_array, linewidth=1, color='blue', alpha=0.7, label='Raw Data')
            
            # Calculate and plot moving average (simple rolling mean)
            if len(saliency_array) > moving_avg_window:
                # Use numpy convolution for moving average
                window = np.ones(moving_avg_window) / moving_avg_window
                moving_avg = np.convolve(saliency_array, window, mode='same')
                ax1.plot(frame_numbers, moving_avg, linewidth=2, color='red', alpha=0.9, 
                        label=f'Moving Average (window={moving_avg_window})')
            elif len(saliency_array) > 1:
                # If data is shorter than window, use a smaller window (half the data length)
                effective_window = max(1, len(saliency_array) // 4)
                window = np.ones(effective_window) / effective_window
                moving_avg = np.convolve(saliency_array, window, mode='same')
                ax1.plot(frame_numbers, moving_avg, linewidth=2, color='red', alpha=0.9, 
                        label=f'Moving Average (window={effective_window})')
            
            # Add horizontal reference line at y=0
            ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5, linewidth=1)
            
            # Set labels and title for top subplot
            ax1.set_xlabel('Frame Number', fontsize=12)
            ax1.set_ylabel('Raw Saliency Value', fontsize=12)
            ax1.set_title(f'{video_id} - {group_name} Saliency Over Time ({aggregation_type.capitalize()})', 
                         fontsize=13, fontweight='bold')
            
            # Add grid for better readability
            ax1.grid(True, alpha=0.3)
            
            # Add legend (always show, includes raw data and moving average)
            ax1.legend(loc='best', fontsize=9)
            
            # Set y-axis limits to show full range (including negative values)
            y_min = saliency_array.min()
            y_max = saliency_array.max()
            y_margin = (y_max - y_min) * 0.05 if y_max != y_min else 0.1
            ax1.set_ylim(y_min - y_margin, y_max + y_margin)
            
            # ===== BOTTOM SUBPLOT: Histogram showing distribution of saliency values =====
            # Create histogram of saliency values
            n_bins = min(50, len(saliency_array) // 10) if len(saliency_array) > 50 else 20
            n_bins = max(10, n_bins)
            
            counts, bins, patches = ax2.hist(saliency_array, bins=n_bins, edgecolor='black', alpha=0.7, color='steelblue')
            
            # Color bars based on positive/negative values
            for i, (count, bin_left, bin_right, patch) in enumerate(zip(counts, bins[:-1], bins[1:], patches)):
                bin_center = (bin_left + bin_right) / 2
                if bin_center > 0:
                    patch.set_facecolor('green')
                    patch.set_alpha(0.6)
                elif bin_center < 0:
                    patch.set_facecolor('red')
                    patch.set_alpha(0.6)
                else:
                    patch.set_facecolor('gray')
                    patch.set_alpha(0.6)
            
            # Add vertical reference line at x=0
            ax2.axvline(x=0, color='gray', linestyle='--', alpha=0.5, linewidth=1)
            
            # Set labels and title for bottom subplot
            ax2.set_xlabel('Raw Saliency Value', fontsize=12)
            ax2.set_ylabel('Frequency (Number of Frames)', fontsize=12)
            ax2.set_title('Distribution of Saliency Values', fontsize=12, fontweight='bold')
            
            # Add grid for better readability
            ax2.grid(True, alpha=0.3, axis='y')
            
            # Add statistics text box
            mean_val = np.mean(saliency_array)
            std_val = np.std(saliency_array)
            median_val = np.median(saliency_array)
            stats_text = f'Mean: {mean_val:.4f}\nStd: {std_val:.4f}\nMedian: {median_val:.4f}'
            ax2.text(0.98, 0.98, stats_text, transform=ax2.transAxes, 
                    fontsize=10, verticalalignment='top', horizontalalignment='right',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
            
            plt.tight_layout()
            
            # Save the plot (sanitize group name for filename)
            safe_group_name = group_name.replace(' ', '_').lower()
            output_path = timeline_dir / f"{video_id}_body_part_{safe_group_name}_{aggregation_type}_timeline_moving_avg_{moving_avg_window}.png"
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close()
    
    print(f"✅ Generated individual body part timeline plots saved to: {timeline_dir}")
    print(f"✅ Generated grouped body part saliency timeline plots saved to: {timeline_dir}")


def create_grouped_body_part_saliency_heatmap(video_id, video_clips, sorted_clips, num_keypoints, output_dir, 
                                               normalize_per_group=False, normalization_method='percentile',
                                               percentile_low=5, percentile_high=95, zscore_num_std=3,
                                               temporal_avg_window=None, use_absolute_values=False):
    """
    Create a heatmap showing saliency values for body part groups over time across the entire video.
    Groups keypoints by body parts and averages saliency values spatially (across keypoints in each group).
    
    IMPORTANT: This function applies normalization to enhance visualization contrast.
    
    Args:
        video_id (str): Video ID
        video_clips (dict): Dictionary of clip data {clip_name: {saliency_map, true_label, pred_label, confidence}}
        sorted_clips (list): List of clip names sorted chronologically
        num_keypoints (int): Number of keypoints
        output_dir (Path): Output directory for saving the heatmap
        normalize_per_group (bool): If True, normalize each body part group independently. If False, use global normalization (default: False)
        normalization_method (str): 'percentile' or 'zscore' (default: 'percentile')
        percentile_low (int): Lower percentile for percentile normalization (default: 5)
        percentile_high (int): Upper percentile for percentile normalization (default: 95)
        zscore_num_std (float): Number of standard deviations for z-score normalization (default: 3)
        temporal_avg_window (int, optional): Window size for temporal averaging (moving average). If None, no temporal averaging is applied (default: None)
        use_absolute_values (bool): If True, use absolute values of saliency (magnitude only). If False, use signed values (default: False)
    """
    norm_mode_str = "per-group" if normalize_per_group else "global"
    print(f"Building 2D saliency array (body part groups × frames) for {norm_mode_str} normalization...")
    
    # Get body part groups and convert ranges to keypoint indices
    body_part_groups = get_body_part_groups()
    body_part_indices = {}
    group_names = []
    for group_name, (start_idx, end_idx) in body_part_groups.items():
        # Convert to inclusive indices
        indices = list(range(start_idx, end_idx + 1))
        # Filter to only include valid keypoint indices
        indices = [idx for idx in indices if idx < num_keypoints]
        if len(indices) > 0:
            body_part_indices[group_name] = indices
            group_names.append(group_name)
    
    num_groups = len(group_names)
    print(f"Grouped into {num_groups} body part groups")
    
    # Build 2D array: rows = body part groups, columns = frames
    all_saliency_matrix = []  # Will be list of lists, each inner list is one frame's saliency for all groups
    seizure_frame_mask = []
    
    # Process each clip chronologically
    for clip_name in sorted_clips:
        clip_data = video_clips[clip_name]
        saliency_map = clip_data['saliency_map']
        true_label = clip_data['true_label']
        
        # Validate saliency map shape
        if saliency_map.ndim != 2 or saliency_map.shape[1] != num_keypoints:
            print(f"Warning: Skipping clip {clip_name} due to shape mismatch")
            continue
        
        num_frames = saliency_map.shape[0]
        
        # Append each frame as a column (all body part group values for that frame)
        for frame_idx in range(num_frames):
            frame_saliency = saliency_map[frame_idx, :]  # Shape: (V,)
            
            # Compute mean saliency for each body part group
            frame_group_saliency = []
            for group_name in group_names:
                keypoint_indices = body_part_indices[group_name]
                if len(keypoint_indices) > 0:
                    group_saliency = frame_saliency[keypoint_indices]
                    frame_group_saliency.append(np.mean(group_saliency))
                else:
                    frame_group_saliency.append(0.0)
            
            all_saliency_matrix.append(frame_group_saliency)
            
            # Mark if this frame is part of a seizure clip
            is_seizure = (true_label == 1)
            seizure_frame_mask.append(is_seizure)
    
    if not all_saliency_matrix:
        print("Warning: No saliency data found. Cannot create heatmap.")
        return
    
    # Convert to numpy array: shape will be (num_frames, num_groups)
    # Then transpose to (num_groups, num_frames) for heatmap
    saliency_array = np.array(all_saliency_matrix)  # Shape: (num_frames, num_groups)
    saliency_heatmap = saliency_array.T  # Shape: (num_groups, num_frames)
    
    # Apply absolute value transformation if requested
    if use_absolute_values:
        saliency_heatmap = np.abs(saliency_heatmap)
        print(f"Applied absolute value transformation. New saliency range: [{saliency_heatmap.min():.6f}, {saliency_heatmap.max():.6f}]")
    
    # Apply temporal averaging (moving average) if requested
    if temporal_avg_window is not None and temporal_avg_window > 1:
        print(f"Applying temporal averaging (moving average, window={temporal_avg_window})...")
        smoothed_heatmap = np.zeros_like(saliency_heatmap)
        for group_idx in range(saliency_heatmap.shape[0]):
            group_data = saliency_heatmap[group_idx, :]
            if len(group_data) > temporal_avg_window:
                # Use numpy convolution for moving average
                window = np.ones(temporal_avg_window) / temporal_avg_window
                smoothed_heatmap[group_idx, :] = np.convolve(group_data, window, mode='same')
            elif len(group_data) > 1:
                # If data is shorter than window, use a smaller window
                effective_window = max(1, len(group_data) // 4)
                window = np.ones(effective_window) / effective_window
                smoothed_heatmap[group_idx, :] = np.convolve(group_data, window, mode='same')
            else:
                smoothed_heatmap[group_idx, :] = group_data
        saliency_heatmap = smoothed_heatmap
        print(f"Temporal averaging applied. New saliency range: [{saliency_heatmap.min():.6f}, {saliency_heatmap.max():.6f}]")
    
    seizure_mask = np.array(seizure_frame_mask)
    num_frames = saliency_heatmap.shape[1]
    
    print(f"Heatmap shape: {saliency_heatmap.shape} (body part groups × frames)")
    print(f"Total frames: {num_frames}")
    print(f"Saliency range: [{saliency_heatmap.min():.6f}, {saliency_heatmap.max():.6f}]")
    
    # Find continuous seizure regions
    seizure_regions = []
    start = None
    for i, is_seizure in enumerate(seizure_mask):
        if is_seizure and start is None:
            start = i
        elif not is_seizure and start is not None:
            seizure_regions.append((start, i - 1))
            start = None
    if start is not None:
        seizure_regions.append((start, len(seizure_mask) - 1))
    
    print(f"Found {len(seizure_regions)} seizure region(s)")
    
    # Create the heatmap figure
    fig, ax = plt.subplots(figsize=(16, 8))
    
    # Apply normalization based on the selected mode and method
    if normalization_method == 'zscore':
        if normalize_per_group:
            # Per-group z-score normalization
            print(f"Using per-group z-score normalization ({zscore_num_std} std)...")
            normalized_heatmap = np.zeros_like(saliency_heatmap)
            for group_idx in range(num_groups):
                group_data = saliency_heatmap[group_idx, :]
                mean_val = np.mean(group_data)
                std_val = np.std(group_data)
                
                if std_val > 1e-8:
                    z_scores = (group_data - mean_val) / std_val
                    if use_absolute_values:
                        # For absolute values, clip to [0, zscore_num_std] since values are non-negative
                        normalized_heatmap[group_idx, :] = np.clip(z_scores, 0.0, zscore_num_std)
                    else:
                        normalized_heatmap[group_idx, :] = np.clip(z_scores, -zscore_num_std, zscore_num_std)
                else:
                    normalized_heatmap[group_idx, :] = 0.0
            
            if use_absolute_values:
                vmin_display, vmax_display = 0.0, zscore_num_std
            else:
                vmin_display, vmax_display = -zscore_num_std, zscore_num_std
            heatmap_to_plot = normalized_heatmap
            print(f"Display range: [{vmin_display:.4f}, {vmax_display:.4f}] (z-score units)")
        else:
            # Global z-score normalization
            print(f"Using global z-score normalization ({zscore_num_std} std)...")
            mean_val = np.mean(saliency_heatmap)
            std_val = np.std(saliency_heatmap)
            
            if std_val > 1e-8:
                z_scores = (saliency_heatmap - mean_val) / std_val
                if use_absolute_values:
                    # For absolute values, clip to [0, zscore_num_std] since values are non-negative
                    heatmap_to_plot = np.clip(z_scores, 0.0, zscore_num_std)
                else:
                    heatmap_to_plot = np.clip(z_scores, -zscore_num_std, zscore_num_std)
            else:
                heatmap_to_plot = np.zeros_like(saliency_heatmap)
            
            if use_absolute_values:
                vmin_display, vmax_display = 0.0, zscore_num_std
            else:
                vmin_display, vmax_display = -zscore_num_std, zscore_num_std
            print(f"Global z-score: mean={mean_val:.4f}, std={std_val:.4f}")
            print(f"Display range: [{vmin_display:.4f}, {vmax_display:.4f}] (z-score units)")
    else:  # percentile normalization
        if normalize_per_group:
            # Per-group percentile normalization
            print(f"Using per-group normalization (percentile {percentile_low}-{percentile_high})...")
            normalized_heatmap = np.zeros_like(saliency_heatmap)
            for group_idx in range(num_groups):
                group_data = saliency_heatmap[group_idx, :]
                
                if use_absolute_values:
                    # For absolute values, use single-sided percentile calculation
                    vmax = np.percentile(group_data, percentile_high)
                    if vmax < 1e-8:
                        vmax = group_data.max()
                        if vmax < 1e-8:
                            vmax = 1.0
                    # Normalize to [0, 1] range
                    if vmax > 1e-8:
                        normalized_heatmap[group_idx, :] = np.clip(group_data / vmax, 0.0, 1.0)
                    else:
                        normalized_heatmap[group_idx, :] = group_data
                else:
                    # Calculate percentiles for positive and negative values separately
                    positive_values = group_data[group_data > 0]
                    negative_values = group_data[group_data < 0]
                    
                    if len(positive_values) > 0 and len(negative_values) > 0:
                        pos_percentile = np.percentile(positive_values, percentile_high)
                        neg_percentile = abs(np.percentile(negative_values, 100 - percentile_low))
                        vmax = max(pos_percentile, neg_percentile)
                    elif len(positive_values) > 0:
                        vmax = np.percentile(positive_values, percentile_high)
                    elif len(negative_values) > 0:
                        vmax = abs(np.percentile(negative_values, 100 - percentile_low))
                    else:
                        vmax = max(abs(group_data.min()), abs(group_data.max()))
                    
                    if vmax < 1e-8:
                        vmax = max(abs(group_data.min()), abs(group_data.max()))
                        if vmax < 1e-8:
                            vmax = 1.0
                    
                    # Normalize to [-1, 1] range
                    if vmax > 1e-8:
                        normalized_heatmap[group_idx, :] = np.clip(group_data / vmax, -1.0, 1.0)
                    else:
                        normalized_heatmap[group_idx, :] = group_data
            
            if use_absolute_values:
                vmin_display, vmax_display = 0.0, 1.0
            else:
                vmin_display, vmax_display = -1.0, 1.0
            heatmap_to_plot = normalized_heatmap
            print(f"Display range: [{vmin_display:.4f}, {vmax_display:.4f}] (normalized)")
        else:
            # Global percentile normalization
            print(f"Using global normalization (percentile {percentile_low}-{percentile_high})...")
            if use_absolute_values:
                vmin_display, vmax_display = calculate_percentile_normalization_absolute(
                    saliency_heatmap, percentile_low, percentile_high
                )
            else:
                vmin_display, vmax_display = calculate_percentile_normalization(
                    saliency_heatmap, percentile_low, percentile_high
                )
            heatmap_to_plot = saliency_heatmap
            print(f"Global normalization: vmin={vmin_display:.4f}, vmax={vmax_display:.4f}")
    
    # Create heatmap with appropriate colormap
    # Use 'viridis' for absolute values, 'RdBu' for signed values
    colormap = 'viridis' if use_absolute_values else 'RdBu'
    im = ax.imshow(heatmap_to_plot, aspect='auto', cmap=colormap, vmin=vmin_display, vmax=vmax_display, 
                   interpolation='nearest', origin='lower')
    
    # Highlight seizure regions with vertical gray shading
    for start_frame, end_frame in seizure_regions:
        ax.axvspan(start_frame - 0.5, end_frame + 0.5, alpha=0.3, color='gray', zorder=0)
    
    # Set axis labels and title
    ax.set_xlabel('Frame Number', fontsize=14, fontweight='bold')
    ax.set_ylabel('Body Part Group', fontsize=14, fontweight='bold')
    norm_scope = 'Per-Group' if normalize_per_group else 'Global'
    norm_method_str = 'Z-Score' if normalization_method == 'zscore' else 'Percentile'
    temporal_str = f' (Temporal Avg: {temporal_avg_window})' if temporal_avg_window is not None else ''
    absolute_str = ' (Absolute Values)' if use_absolute_values else ''
    ax.set_title(f'Body Part Saliency Heatmap: {video_id}\nAll Body Part Groups Over Time ({norm_scope} {norm_method_str}{temporal_str}{absolute_str})', 
                fontsize=16, fontweight='bold', pad=20)
    
    # Set y-axis ticks to show body part group names
    yticks = np.arange(0, num_groups)
    ax.set_yticks(yticks)
    ax.set_yticklabels(group_names, fontsize=11)
    
    # Set x-axis ticks (show every 2000 frames or similar to avoid crowding)
    if num_frames > 5000:
        xtick_spacing = max(2000, num_frames // 10)
    elif num_frames > 1000:
        xtick_spacing = max(500, num_frames // 10)
    else:
        xtick_spacing = max(100, num_frames // 10)
    
    xticks = np.arange(0, num_frames, xtick_spacing)
    ax.set_xticks(xticks)
    ax.set_xticklabels([str(int(x)) for x in xticks], fontsize=10)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.02, pad=0.02)
    if use_absolute_values:
        if normalization_method == 'zscore':
            if normalize_per_group:
                cbar.set_label('Absolute Saliency Z-Score (per group)', fontsize=12, fontweight='bold', rotation=270, labelpad=20)
            else:
                cbar.set_label('Absolute Saliency Z-Score', fontsize=12, fontweight='bold', rotation=270, labelpad=20)
        else:
            if normalize_per_group:
                cbar.set_label('Absolute Saliency Value (per group)', fontsize=12, fontweight='bold', rotation=270, labelpad=20)
            else:
                cbar.set_label('Absolute Saliency Value', fontsize=12, fontweight='bold', rotation=270, labelpad=20)
    else:
        if normalization_method == 'zscore':
            if normalize_per_group:
                cbar.set_label('Z-Score (per group)', fontsize=12, fontweight='bold', rotation=270, labelpad=20)
            else:
                cbar.set_label('Z-Score', fontsize=12, fontweight='bold', rotation=270, labelpad=20)
        else:
            if normalize_per_group:
                cbar.set_label('Normalized Saliency Value (per group)', fontsize=12, fontweight='bold', rotation=270, labelpad=20)
            else:
                cbar.set_label('Raw Saliency Value', fontsize=12, fontweight='bold', rotation=270, labelpad=20)
    cbar.ax.tick_params(labelsize=10)
    
    # Range information text box removed - colorbar provides scale information
    # Legend removed - seizure regions are clearly visible via gray shading
    # Add grid for better readability (subtle)
    ax.grid(True, alpha=0.1, linestyle='--', linewidth=0.5)
    
    plt.tight_layout()
    
    # Save the heatmap with appropriate filename based on normalization mode and method
    temporal_suffix = f"_temporal_avg_{temporal_avg_window}" if temporal_avg_window is not None else ""
    absolute_suffix = "_absolute" if use_absolute_values else ""
    if normalization_method == 'zscore':
        if normalize_per_group:
            output_path = output_dir / f"{video_id}_body_part_saliency_heatmap_zscore_per_group{absolute_suffix}{temporal_suffix}.png"
        else:
            output_path = output_dir / f"{video_id}_body_part_saliency_heatmap_zscore_global{absolute_suffix}{temporal_suffix}.png"
    else:
        if normalize_per_group:
            output_path = output_dir / f"{video_id}_body_part_saliency_heatmap_percentile_per_group{absolute_suffix}{temporal_suffix}.png"
        else:
            output_path = output_dir / f"{video_id}_body_part_saliency_heatmap_percentile_global{absolute_suffix}{temporal_suffix}.png"
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')  # High DPI for publication quality
    plt.close()
    
    print(f"✅ Heatmap saved to: {output_path}")


def plot_grouped_body_part_visualizations(video_id, video_clips, output_dir, moving_avg_window=10):
    """
    Wrapper function that generates all grouped body part visualizations (timeline plots and heatmaps).
    Creates the output directory and calls both timeline and heatmap functions.
    
    Args:
        video_id (str): Video ID
        video_clips (dict): Dictionary of clip data {clip_name: {saliency_map, true_label, pred_label, confidence}}
        output_dir (Path): Output directory (should be k_fold/saliency_maps/{video_id})
        moving_avg_window (int): Window size for moving average smoothing in timeline plots (default: 100 frames)
    """
    print(f"\n{'='*80}")
    print(f"Generating Grouped Body Part Visualizations for Video: {video_id}")
    print(f"{'='*80}")
    
    # Create output directory
    body_part_dir = output_dir / "body_part_saliency_timeline"
    body_part_dir.mkdir(parents=True, exist_ok=True)
    
    # Get sorted clips (needed for heatmap)
    clip_names = list(video_clips.keys())
    clip_timeline = {}
    for clip in clip_names:
        try:
            seg_part = clip.split('_Seg_')[1]
            if '_' in seg_part:
                seg_num = int(seg_part.split('_')[0])
            else:
                seg_num = int(seg_part)
            clip_timeline[clip] = seg_num
        except (IndexError, ValueError):
            clip_timeline[clip] = 999999
    
    sorted_clips = sorted(clip_names, key=lambda x: clip_timeline.get(x, 999999))
    
    # Determine number of keypoints
    if not sorted_clips:
        print("Warning: No clips found. Cannot generate body part visualizations.")
        return
    
    first_clip_data = video_clips[sorted_clips[0]]
    first_saliency_map = first_clip_data['saliency_map']
    
    if first_saliency_map.ndim != 2:
        print(f"Warning: Expected saliency map with shape (T, V), but got {first_saliency_map.shape}")
        return
    
    num_keypoints = first_saliency_map.shape[1]
    
    # Generate timeline plots (mean and median)
    try:
        plot_grouped_body_part_saliency_timeline(video_id, video_clips, output_dir, moving_avg_window=moving_avg_window)
    except Exception as e:
        print(f"❌ Error generating timeline plots: {e}")
        import traceback
        traceback.print_exc()
    
    # Generate heatmaps (per-group normalization only - 4 variants)
    print(f"\n--- Creating Body Part Saliency Heatmaps for Video: {video_id} ---")
    
    # 1. Percentile per-group normalization
    print(f"\n[1/4] Generating percentile per-group normalization heatmap...")
    try:
        create_grouped_body_part_saliency_heatmap(video_id, video_clips, sorted_clips, num_keypoints, body_part_dir, 
                                                   normalize_per_group=True, normalization_method='percentile')
        print(f"✅ Percentile per-group normalization heatmap completed")
    except Exception as e:
        print(f"❌ Error generating percentile per-group heatmap: {e}")
        import traceback
        traceback.print_exc()
    
    # 2. Z-score per-group normalization
    print(f"\n[2/4] Generating z-score per-group normalization heatmap...")
    try:
        create_grouped_body_part_saliency_heatmap(video_id, video_clips, sorted_clips, num_keypoints, body_part_dir, 
                                                   normalize_per_group=True, normalization_method='zscore')
        print(f"✅ Z-score per-group normalization heatmap completed")
    except Exception as e:
        print(f"❌ Error generating z-score per-group heatmap: {e}")
        import traceback
        traceback.print_exc()
    
    # Generate heatmaps with temporal averaging (per-group normalization only - 2 variants)
    print(f"\n--- Creating Body Part Saliency Heatmaps with Temporal Averaging (window={moving_avg_window}) for Video: {video_id} ---")
    
    # 3. Percentile per-group normalization with temporal averaging
    print(f"\n[3/4] Generating percentile per-group normalization heatmap with temporal averaging...")
    try:
        create_grouped_body_part_saliency_heatmap(video_id, video_clips, sorted_clips, num_keypoints, body_part_dir, 
                                                   normalize_per_group=True, normalization_method='percentile',
                                                   temporal_avg_window=moving_avg_window)
        print(f"✅ Percentile per-group normalization heatmap with temporal averaging completed")
    except Exception as e:
        print(f"❌ Error generating percentile per-group heatmap with temporal averaging: {e}")
        import traceback
        traceback.print_exc()
    
    # 4. Z-score per-group normalization with temporal averaging
    print(f"\n[4/4] Generating z-score per-group normalization heatmap with temporal averaging...")
    try:
        create_grouped_body_part_saliency_heatmap(video_id, video_clips, sorted_clips, num_keypoints, body_part_dir, 
                                                   normalize_per_group=True, normalization_method='zscore',
                                                   temporal_avg_window=moving_avg_window)
        print(f"✅ Z-score per-group normalization heatmap with temporal averaging completed")
    except Exception as e:
        print(f"❌ Error generating z-score per-group heatmap with temporal averaging: {e}")
        import traceback
        traceback.print_exc()
    
    # Generate absolute value heatmaps (4 variants matching the signed variants)
    print(f"\n--- Creating Body Part Saliency Heatmaps (Absolute Values) for Video: {video_id} ---")
    
    # 5. Percentile per-group normalization (absolute values)
    print(f"\n[5/8] Generating percentile per-group normalization heatmap (absolute values)...")
    try:
        create_grouped_body_part_saliency_heatmap(video_id, video_clips, sorted_clips, num_keypoints, body_part_dir, 
                                                   normalize_per_group=True, normalization_method='percentile',
                                                   use_absolute_values=True)
        print(f"✅ Percentile per-group normalization heatmap (absolute values) completed")
    except Exception as e:
        print(f"❌ Error generating percentile per-group heatmap (absolute values): {e}")
        import traceback
        traceback.print_exc()
    
    # 6. Z-score per-group normalization (absolute values)
    print(f"\n[6/8] Generating z-score per-group normalization heatmap (absolute values)...")
    try:
        create_grouped_body_part_saliency_heatmap(video_id, video_clips, sorted_clips, num_keypoints, body_part_dir, 
                                                   normalize_per_group=True, normalization_method='zscore',
                                                   use_absolute_values=True)
        print(f"✅ Z-score per-group normalization heatmap (absolute values) completed")
    except Exception as e:
        print(f"❌ Error generating z-score per-group heatmap (absolute values): {e}")
        import traceback
        traceback.print_exc()
    
    # 7. Percentile per-group normalization with temporal averaging (absolute values)
    print(f"\n[7/8] Generating percentile per-group normalization heatmap with temporal averaging (absolute values)...")
    try:
        create_grouped_body_part_saliency_heatmap(video_id, video_clips, sorted_clips, num_keypoints, body_part_dir, 
                                                   normalize_per_group=True, normalization_method='percentile',
                                                   temporal_avg_window=moving_avg_window, use_absolute_values=True)
        print(f"✅ Percentile per-group normalization heatmap with temporal averaging (absolute values) completed")
    except Exception as e:
        print(f"❌ Error generating percentile per-group heatmap with temporal averaging (absolute values): {e}")
        import traceback
        traceback.print_exc()
    
    # 8. Z-score per-group normalization with temporal averaging (absolute values)
    print(f"\n[8/8] Generating z-score per-group normalization heatmap with temporal averaging (absolute values)...")
    try:
        create_grouped_body_part_saliency_heatmap(video_id, video_clips, sorted_clips, num_keypoints, body_part_dir, 
                                                   normalize_per_group=True, normalization_method='zscore',
                                                   temporal_avg_window=moving_avg_window, use_absolute_values=True)
        print(f"✅ Z-score per-group normalization heatmap with temporal averaging (absolute values) completed")
    except Exception as e:
        print(f"❌ Error generating z-score per-group heatmap with temporal averaging (absolute values): {e}")
        import traceback
        traceback.print_exc()
    
    print(f"\n✅ All grouped body part visualizations generated successfully!")
    print(f"   Output directory: {body_part_dir}")
