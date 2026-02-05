"""
Functions for statistical analysis of saliency data to understand body dynamics during seizures.
All analyses use RAW saliency values without normalization.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from collections import defaultdict, Counter
import itertools

from .viz_saliency_timeline import get_coco_wholebody_keypoint_names


def analyze_keypoint_importance_ranking(video_id, video_clips, sorted_clips, num_keypoints, output_dir):
    """
    Analyze and rank keypoints by importance during seizure periods using multiple metrics.
    Uses raw saliency values (no normalization).
    
    Args:
        video_id (str): Video ID
        video_clips (dict): Dictionary of clip data {clip_name: {saliency_map, true_label, pred_label, confidence}}
        sorted_clips (list): List of clip names sorted chronologically
        num_keypoints (int): Number of keypoints (133)
        output_dir (Path): Output directory for saving results
    """
    print(f"\n--- Analyzing Keypoint Importance Ranking for Video: {video_id} ---")
    
    # Get keypoint names
    keypoint_names = get_coco_wholebody_keypoint_names()
    
    # Collect raw saliency values for each keypoint during seizure periods
    keypoint_seizure_saliency = {i: [] for i in range(num_keypoints)}
    
    for clip_name in sorted_clips:
        clip_data = video_clips[clip_name]
        saliency_map = clip_data['saliency_map']
        true_label = clip_data['true_label']
        
        # Only analyze seizure clips
        if true_label != 1:
            continue
        
        # Validate saliency map shape
        if saliency_map.ndim != 2 or saliency_map.shape[1] != num_keypoints:
            continue
        
        # Extract saliency values for all keypoints during this seizure clip
        for keypoint_idx in range(num_keypoints):
            keypoint_saliency = saliency_map[:, keypoint_idx]
            keypoint_seizure_saliency[keypoint_idx].extend(keypoint_saliency.tolist())
    
    # Calculate metrics for each keypoint
    metrics_data = []
    
    for keypoint_idx in range(num_keypoints):
        saliency_values = np.array(keypoint_seizure_saliency[keypoint_idx])
        
        if len(saliency_values) == 0:
            # No seizure data for this keypoint
            metrics_data.append({
                'keypoint_idx': keypoint_idx,
                'keypoint_name': keypoint_names.get(keypoint_idx, f'keypoint_{keypoint_idx}'),
                'mean_abs_saliency': 0.0,
                'max_saliency': 0.0,
                'std_saliency': 0.0,
                'mean_saliency': 0.0,
                'num_seizure_frames': 0
            })
            continue
        
        # Calculate all metrics
        mean_abs = np.mean(np.abs(saliency_values))
        max_val = np.max(saliency_values)
        std_val = np.std(saliency_values)
        mean_val = np.mean(saliency_values)
        
        metrics_data.append({
            'keypoint_idx': keypoint_idx,
            'keypoint_name': keypoint_names.get(keypoint_idx, f'keypoint_{keypoint_idx}'),
            'mean_abs_saliency': mean_abs,
            'max_saliency': max_val,
            'std_saliency': std_val,
            'mean_saliency': mean_val,
            'num_seizure_frames': len(saliency_values)
        })
    
    # Create DataFrame for easier manipulation
    df = pd.DataFrame(metrics_data)
    
    # Rank by each metric
    df['rank_mean_abs'] = df['mean_abs_saliency'].rank(ascending=False, method='min').astype(int)
    df['rank_max'] = df['max_saliency'].rank(ascending=False, method='min').astype(int)
    df['rank_std'] = df['std_saliency'].rank(ascending=False, method='min').astype(int)
    df['rank_mean'] = df['mean_saliency'].abs().rank(ascending=False, method='min').astype(int)
    
    # Save CSV with all rankings
    csv_path = output_dir / f"{video_id}_keypoint_importance_ranking.csv"
    df.to_csv(csv_path, index=False)
    print(f"✅ Rankings saved to: {csv_path}")
    
    # Create visualization: 2x2 grid with all 4 metrics
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'Keypoint Importance Ranking: {video_id}\n(During Seizure Periods - Raw Saliency)', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    # Top N keypoints to show (default: top 20)
    top_n = 20
    
    # Top-left: Mean Absolute Saliency
    ax = axes[0, 0]
    df_sorted = df.nlargest(top_n, 'mean_abs_saliency')
    y_pos = np.arange(len(df_sorted))
    bars = ax.barh(y_pos, df_sorted['mean_abs_saliency'], color='steelblue', alpha=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([keypoint_names.get(idx, f'KP{idx}') for idx in df_sorted['keypoint_idx']], fontsize=8)
    ax.set_xlabel('Mean Absolute Saliency', fontsize=11, fontweight='bold')
    ax.set_title(f'Top {top_n} Keypoints by Mean Absolute Saliency', fontsize=12, fontweight='bold')
    ax.text(0.5, -0.15, 
            'What: Average magnitude of saliency during seizures. Higher = more consistently important.\n'
            'How: Calculated as mean(|saliency|) for all frames in seizure clips (true_label=1).',
            transform=ax.transAxes, fontsize=8, verticalalignment='top', ha='center',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.5))
    ax.grid(True, alpha=0.3, axis='x')
    ax.invert_yaxis()
    # Add value labels on bars
    for i, (idx, val) in enumerate(zip(df_sorted['keypoint_idx'], df_sorted['mean_abs_saliency'])):
        ax.text(val, i, f' {val:.3f}', va='center', fontsize=7)
    
    # Top-right: Maximum Saliency
    ax = axes[0, 1]
    df_sorted = df.nlargest(top_n, 'max_saliency')
    y_pos = np.arange(len(df_sorted))
    bars = ax.barh(y_pos, df_sorted['max_saliency'], color='coral', alpha=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([keypoint_names.get(idx, f'KP{idx}') for idx in df_sorted['keypoint_idx']], fontsize=8)
    ax.set_xlabel('Maximum Saliency', fontsize=11, fontweight='bold')
    ax.set_title(f'Top {top_n} Keypoints by Maximum Saliency', fontsize=12, fontweight='bold')
    ax.text(0.5, -0.15, 
            'What: Peak saliency value reached during seizures. Higher = strongest single activation.\n'
            'How: Calculated as max(saliency) across all frames in seizure clips (true_label=1).',
            transform=ax.transAxes, fontsize=8, verticalalignment='top', ha='center',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightcoral', alpha=0.5))
    ax.grid(True, alpha=0.3, axis='x')
    ax.invert_yaxis()
    # Add value labels on bars
    for i, (idx, val) in enumerate(zip(df_sorted['keypoint_idx'], df_sorted['max_saliency'])):
        ax.text(val, i, f' {val:.3f}', va='center', fontsize=7)
    
    # Bottom-left: Standard Deviation
    ax = axes[1, 0]
    df_sorted = df.nlargest(top_n, 'std_saliency')
    y_pos = np.arange(len(df_sorted))
    bars = ax.barh(y_pos, df_sorted['std_saliency'], color='mediumseagreen', alpha=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([keypoint_names.get(idx, f'KP{idx}') for idx in df_sorted['keypoint_idx']], fontsize=8)
    ax.set_xlabel('Standard Deviation of Saliency', fontsize=11, fontweight='bold')
    ax.set_title(f'Top {top_n} Keypoints by Saliency Variability (Std)', fontsize=12, fontweight='bold')
    ax.text(0.5, -0.15, 
            'What: Variability of saliency values during seizures. Higher = more dynamic/changing activity.\n'
            'How: Calculated as std(saliency) for all frames in seizure clips (true_label=1).',
            transform=ax.transAxes, fontsize=8, verticalalignment='top', ha='center',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.5))
    ax.grid(True, alpha=0.3, axis='x')
    ax.invert_yaxis()
    # Add value labels on bars
    for i, (idx, val) in enumerate(zip(df_sorted['keypoint_idx'], df_sorted['std_saliency'])):
        ax.text(val, i, f' {val:.3f}', va='center', fontsize=7)
    
    # Bottom-right: Mean Saliency (preserving sign)
    ax = axes[1, 1]
    df_sorted = df.nlargest(top_n, 'mean_saliency', key=lambda x: x.abs())
    y_pos = np.arange(len(df_sorted))
    colors = ['red' if val < 0 else 'blue' for val in df_sorted['mean_saliency']]
    bars = ax.barh(y_pos, df_sorted['mean_saliency'], color=colors, alpha=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([keypoint_names.get(idx, f'KP{idx}') for idx in df_sorted['keypoint_idx']], fontsize=8)
    ax.set_xlabel('Mean Saliency (Preserving Sign)', fontsize=11, fontweight='bold')
    ax.set_title(f'Top {top_n} Keypoints by Mean Saliency Magnitude', fontsize=12, fontweight='bold')
    ax.text(0.5, -0.15, 
            'What: Average saliency preserving positive/negative sign. Red=negative, Blue=positive influence.\n'
            'How: Calculated as mean(saliency) for all frames in seizure clips (true_label=1), preserving sign.',
            transform=ax.transAxes, fontsize=8, verticalalignment='top', ha='center',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lavender', alpha=0.5))
    ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    ax.grid(True, alpha=0.3, axis='x')
    ax.invert_yaxis()
    # Add value labels on bars
    for i, (idx, val) in enumerate(zip(df_sorted['keypoint_idx'], df_sorted['mean_saliency'])):
        ax.text(val, i, f' {val:.3f}', va='center', fontsize=7)
    
    # Add overall description below the plots
    description_text = (
        "These rankings identify which keypoints show the strongest saliency signals during seizure periods. "
        "Higher values indicate keypoints that are more important for the model's seizure detection. "
        "All metrics are calculated using raw saliency values (no normalization) from frames where true_label=1."
    )
    fig.text(0.5, 0.01, description_text, ha='center', va='bottom', fontsize=9, 
             style='italic', wrap=True, bbox=dict(boxstyle='round,pad=0.5', facecolor='wheat', alpha=0.3))
    
    plt.tight_layout(rect=[0, 0.05, 1, 0.99])
    
    # Save the figure
    output_path = output_dir / f"{video_id}_keypoint_importance_ranking.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Importance ranking visualization saved to: {output_path}")
    
    # Print top 5 keypoints for each metric
    print(f"\n📊 Top 5 Keypoints by Each Metric:")
    print(f"  Mean Absolute Saliency:")
    top_mean_abs = df.nlargest(5, 'mean_abs_saliency')
    for _, row in top_mean_abs.iterrows():
        print(f"    {row['keypoint_name']:20s} (idx {row['keypoint_idx']:3d}): {row['mean_abs_saliency']:.4f}")
    
    print(f"  Maximum Saliency:")
    top_max = df.nlargest(5, 'max_saliency')
    for _, row in top_max.iterrows():
        print(f"    {row['keypoint_name']:20s} (idx {row['keypoint_idx']:3d}): {row['max_saliency']:.4f}")
    
    print(f"  Standard Deviation:")
    top_std = df.nlargest(5, 'std_saliency')
    for _, row in top_std.iterrows():
        print(f"    {row['keypoint_name']:20s} (idx {row['keypoint_idx']:3d}): {row['std_saliency']:.4f}")


def analyze_keypoint_activation_sequences(video_id, video_clips, sorted_clips, num_keypoints, output_dir):
    """
    Analyze keypoint activation sequences using keypoint-specific thresholds.
    Uses raw saliency values (no normalization).
    
    Args:
        video_id (str): Video ID
        video_clips (dict): Dictionary of clip data {clip_name: {saliency_map, true_label, pred_label, confidence}}
        sorted_clips (list): List of clip names sorted chronologically
        num_keypoints (int): Number of keypoints (133)
        output_dir (Path): Output directory for saving results
    """
    print(f"\n--- Analyzing Keypoint Activation Sequences for Video: {video_id} ---")
    
    # Get keypoint names
    keypoint_names = get_coco_wholebody_keypoint_names()
    
    # Step 1: Build continuous timeline and calculate thresholds
    print("Building continuous timeline and calculating keypoint-specific thresholds...")
    
    # Collect all saliency values for each keypoint across entire video
    keypoint_all_saliency = {i: [] for i in range(num_keypoints)}
    frame_to_clip = []  # Track which clip each frame belongs to
    frame_to_seizure = []  # Track if each frame is seizure
    current_frame = 0
    
    for clip_name in sorted_clips:
        clip_data = video_clips[clip_name]
        saliency_map = clip_data['saliency_map']
        true_label = clip_data['true_label']
        
        if saliency_map.ndim != 2 or saliency_map.shape[1] != num_keypoints:
            continue
        
        num_frames = saliency_map.shape[0]
        is_seizure = (true_label == 1)
        
        for frame_idx in range(num_frames):
            for keypoint_idx in range(num_keypoints):
                keypoint_all_saliency[keypoint_idx].append(saliency_map[frame_idx, keypoint_idx])
            frame_to_clip.append(clip_name)
            frame_to_seizure.append(is_seizure)
            current_frame += 1
    
    # Calculate keypoint-specific thresholds: mean + 2*std for each keypoint
    keypoint_thresholds = {}
    for keypoint_idx in range(num_keypoints):
        saliency_values = np.array(keypoint_all_saliency[keypoint_idx])
        if len(saliency_values) > 0:
            mean_val = np.mean(saliency_values)
            std_val = np.std(saliency_values)
            threshold = mean_val + 2 * std_val
            keypoint_thresholds[keypoint_idx] = threshold
        else:
            keypoint_thresholds[keypoint_idx] = 0.0
    
    print(f"Calculated thresholds for {len(keypoint_thresholds)} keypoints")
    
    # Step 2: Identify activation events
    print("Identifying activation events...")
    
    activation_events = {i: [] for i in range(num_keypoints)}  # frame numbers where activated
    activation_data = []  # List of (frame, keypoint_idx, saliency_value, is_seizure)
    
    current_frame = 0
    for clip_name in sorted_clips:
        clip_data = video_clips[clip_name]
        saliency_map = clip_data['saliency_map']
        true_label = clip_data['true_label']
        
        if saliency_map.ndim != 2 or saliency_map.shape[1] != num_keypoints:
            continue
        
        num_frames = saliency_map.shape[0]
        is_seizure = (true_label == 1)
        
        for frame_idx in range(num_frames):
            for keypoint_idx in range(num_keypoints):
                saliency_value = saliency_map[frame_idx, keypoint_idx]
                threshold = keypoint_thresholds[keypoint_idx]
                
                if saliency_value > threshold:
                    activation_events[keypoint_idx].append(current_frame)
                    activation_data.append({
                        'frame': current_frame,
                        'keypoint_idx': keypoint_idx,
                        'keypoint_name': keypoint_names.get(keypoint_idx, f'keypoint_{keypoint_idx}'),
                        'saliency_value': saliency_value,
                        'threshold': threshold,
                        'is_seizure': is_seizure,
                        'clip_name': clip_name
                    })
            current_frame += 1
    
    # Save activation data to CSV
    activation_df = pd.DataFrame(activation_data)
    csv_path = output_dir / f"{video_id}_keypoint_activation_data.csv"
    activation_df.to_csv(csv_path, index=False)
    print(f"✅ Activation data saved to: {csv_path}")
    
    # Step 3: Create visualizations
    
    # 3.1 Activation Timeline
    print("Creating activation timeline visualization...")
    fig, ax = plt.subplots(figsize=(16, 10))
    
    # Plot activation events
    for keypoint_idx in range(num_keypoints):
        if len(activation_events[keypoint_idx]) > 0:
            frames = activation_events[keypoint_idx]
            y_pos = [keypoint_idx] * len(frames)
            ax.scatter(frames, y_pos, s=10, alpha=0.6, c='blue', marker='|')
    
    # Highlight seizure regions
    seizure_regions = []
    start = None
    for i, is_seizure in enumerate(frame_to_seizure):
        if is_seizure and start is None:
            start = i
        elif not is_seizure and start is not None:
            seizure_regions.append((start, i - 1))
            start = None
    if start is not None:
        seizure_regions.append((start, len(frame_to_seizure) - 1))
    
    for start_frame, end_frame in seizure_regions:
        ax.axvspan(start_frame - 0.5, end_frame + 0.5, alpha=0.2, color='red', zorder=0, label='Seizure Period' if start_frame == seizure_regions[0][0] else '')
    
    # Set labels and title
    ax.set_xlabel('Frame Number', fontsize=14, fontweight='bold')
    ax.set_ylabel('Keypoint Index', fontsize=14, fontweight='bold')
    ax.set_title(f'Keypoint Activation Timeline: {video_id}\n(Activation = Saliency > Keypoint-Specific Threshold)', 
                fontsize=16, fontweight='bold', pad=20)
    
    # Set y-axis ticks with keypoint names (show every 10th for readability)
    yticks = np.arange(0, num_keypoints, 10)
    yticklabels = []
    for i in yticks:
        if i in keypoint_names:
            name = keypoint_names[i]
            # Abbreviate for display
            name_abbrev = name.replace('left_', 'L_').replace('right_', 'R_')
            name_abbrev = name_abbrev.replace('_finger', '').replace('_toe', '')
            name_abbrev = name_abbrev.replace('forefinger', 'fore').replace('middle_finger', 'mid')
            name_abbrev = name_abbrev.replace('ring_finger', 'ring').replace('pinky_finger', 'pink')
            name_abbrev = name_abbrev.replace('hand_root', 'wrist')
            name_abbrev = name_abbrev.replace('shoulder', 'shldr').replace('elbow', 'elb')
            name_abbrev = name_abbrev.replace('wrist', 'wrst').replace('ankle', 'ank')
            name_abbrev = name_abbrev.replace('big_toe', 'btoe').replace('small_toe', 'stoe')
            if name_abbrev.startswith('face_'):
                name_abbrev = name_abbrev.replace('face_contour', 'face_cntr')
            yticklabels.append(f"{i}:{name_abbrev}")
        else:
            yticklabels.append(str(i))
    
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels, fontsize=7)
    ax.set_ylim(-0.5, num_keypoints - 0.5)
    
    if len(seizure_regions) > 0:
        ax.legend(loc='upper right', fontsize=11)
    
    ax.grid(True, alpha=0.3)
    # Add description below the plot
    ax.text(0.5, -0.08, 
            'What: Each vertical line marks when a keypoint\'s saliency exceeds its threshold. '
            'Red shaded regions indicate seizure periods. Dense vertical bands show synchronized activations.\n'
            'How: For each keypoint, threshold = mean(saliency) + 2×std(saliency) calculated across entire video. '
            'Activation occurs when saliency > threshold for that keypoint.',
            transform=ax.transAxes, fontsize=9, verticalalignment='top', ha='center',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='wheat', alpha=0.3))
    
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    
    timeline_path = output_dir / f"{video_id}_keypoint_activation_timeline.png"
    plt.savefig(timeline_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Activation timeline saved to: {timeline_path}")
    
    # 3.2 Activation Frequency Bar Chart
    print("Creating activation frequency visualization...")
    fig, ax = plt.subplots(figsize=(14, 8))
    
    activation_counts = {i: len(activation_events[i]) for i in range(num_keypoints)}
    keypoint_indices = list(activation_counts.keys())
    counts = list(activation_counts.values())
    keypoint_labels = [keypoint_names.get(i, f'KP{i}') for i in keypoint_indices]
    
    # Sort by frequency
    sorted_data = sorted(zip(keypoint_indices, counts, keypoint_labels), key=lambda x: x[1], reverse=True)
    sorted_indices, sorted_counts, sorted_labels = zip(*sorted_data)
    
    # Show top 30 keypoints
    top_n = min(30, len(sorted_indices))
    bars = ax.barh(range(top_n), sorted_counts[:top_n], color='steelblue', alpha=0.7)
    ax.set_yticks(range(top_n))
    ax.set_yticklabels([sorted_labels[i] for i in range(top_n)], fontsize=8)
    ax.set_xlabel('Number of Activation Events', fontsize=12, fontweight='bold')
    ax.set_ylabel('Keypoint', fontsize=12, fontweight='bold')
    ax.set_title(f'Keypoint Activation Frequency: {video_id}\n(Top {top_n} Most Frequently Activated Keypoints)', 
                fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    ax.invert_yaxis()
    
    # Add value labels
    for i, count in enumerate(sorted_counts[:top_n]):
        ax.text(count, i, f' {count}', va='center', fontsize=8)
    
    # Add description below the plot
    ax.text(0.5, -0.1, 
            'What: Total number of times each keypoint exceeded its activation threshold. '
            'Higher frequency indicates more persistent or recurring activation patterns.\n'
            'How: Count of frames where saliency > (mean + 2×std) for each keypoint across entire video.',
            transform=ax.transAxes, fontsize=9, verticalalignment='top', ha='center',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='wheat', alpha=0.3))
    
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    frequency_path = output_dir / f"{video_id}_keypoint_activation_frequency.png"
    plt.savefig(frequency_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Activation frequency chart saved to: {frequency_path}")
    
    # 3.3 Seizure vs Non-Seizure Comparison (Normalized by Frame Count)
    print("Creating seizure vs non-seizure comparison (normalized by frame count)...")
    
    seizure_activations = defaultdict(int)
    non_seizure_activations = defaultdict(int)
    
    # Count frames in each period
    seizure_frame_count = sum(1 for is_sz in frame_to_seizure if is_sz)
    non_seizure_frame_count = sum(1 for is_sz in frame_to_seizure if not is_sz)
    
    print(f"  Seizure frames: {seizure_frame_count}, Non-seizure frames: {non_seizure_frame_count}")
    
    for event in activation_data:
        keypoint_idx = event['keypoint_idx']
        if event['is_seizure']:
            seizure_activations[keypoint_idx] += 1
        else:
            non_seizure_activations[keypoint_idx] += 1
    
    # Normalize by frame count to get activation rates (events per frame)
    seizure_rates = {}
    non_seizure_rates = {}
    
    all_keypoints = set(seizure_activations.keys()) | set(non_seizure_activations.keys())
    
    for kp in all_keypoints:
        if seizure_frame_count > 0:
            seizure_rates[kp] = seizure_activations.get(kp, 0) / seizure_frame_count
        else:
            seizure_rates[kp] = 0.0
        
        if non_seizure_frame_count > 0:
            non_seizure_rates[kp] = non_seizure_activations.get(kp, 0) / non_seizure_frame_count
        else:
            non_seizure_rates[kp] = 0.0
    
    # Get top keypoints by total activation rate
    total_rates = {kp: seizure_rates.get(kp, 0) + non_seizure_rates.get(kp, 0) 
                   for kp in all_keypoints}
    top_keypoints = sorted(all_keypoints, key=lambda x: total_rates[x], reverse=True)[:20]
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    x_pos = np.arange(len(top_keypoints))
    seizure_rate_values = [seizure_rates.get(kp, 0) for kp in top_keypoints]
    non_seizure_rate_values = [non_seizure_rates.get(kp, 0) for kp in top_keypoints]
    
    width = 0.35
    bars1 = ax.bar(x_pos - width/2, seizure_rate_values, width, label='Seizure Period', color='red', alpha=0.7)
    bars2 = ax.bar(x_pos + width/2, non_seizure_rate_values, width, label='Non-Seizure Period', color='blue', alpha=0.7)
    
    ax.set_xlabel('Keypoint', fontsize=12, fontweight='bold')
    ax.set_ylabel('Activation Rate (Events per Frame)', fontsize=12, fontweight='bold')
    ax.set_title(f'Keypoint Activation: Seizure vs Non-Seizure Comparison (Normalized)\n{video_id}', 
                fontsize=14, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels([keypoint_names.get(kp, f'KP{kp}') for kp in top_keypoints], 
                       rotation=45, ha='right', fontsize=8)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add description below the plot
    ax.text(0.5, -0.12, 
            'What: Comparison of activation rate (events per frame) during seizure (red) vs non-seizure (blue) periods. '
            'Keypoints with higher red bars are more specific to seizure events.\n'
            'How: Activation rate = (number of activations) / (number of frames) for each period. '
            'Normalized by frame count to account for different period durations. '
            'Threshold: mean(saliency) + 2×std(saliency) per keypoint.',
            transform=ax.transAxes, fontsize=9, verticalalignment='top', ha='center',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='wheat', alpha=0.3))
    
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    comparison_path = output_dir / f"{video_id}_keypoint_activation_comparison.png"
    plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Seizure vs non-seizure comparison saved to: {comparison_path}")
    
    # 3.4 Activation Co-occurrence Heatmap
    print("Creating activation co-occurrence heatmap...")
    
    # Calculate co-occurrence: how often keypoints activate together (within same frame)
    cooccurrence_matrix = np.zeros((num_keypoints, num_keypoints))
    
    # Group activations by frame
    frame_activations = defaultdict(set)
    for event in activation_data:
        frame_activations[event['frame']].add(event['keypoint_idx'])
    
    # Count co-occurrences
    for frame, activated_keypoints in frame_activations.items():
        if len(activated_keypoints) > 1:
            for kp1, kp2 in itertools.combinations(activated_keypoints, 2):
                cooccurrence_matrix[kp1, kp2] += 1
                cooccurrence_matrix[kp2, kp1] += 1
    
    # Create heatmap for top keypoints only (for readability)
    # Use total activation counts (not rates) for selecting top keypoints for co-occurrence
    total_activation_counts = {kp: seizure_activations.get(kp, 0) + non_seizure_activations.get(kp, 0) 
                               for kp in all_keypoints}
    top_kp_by_total = sorted(all_keypoints, key=lambda x: total_activation_counts[x], reverse=True)[:30]
    top_kp_indices = sorted(top_kp_by_total)
    
    cooccurrence_subset = cooccurrence_matrix[np.ix_(top_kp_indices, top_kp_indices)]
    
    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(cooccurrence_subset, cmap='YlOrRd', aspect='auto', interpolation='nearest')
    
    # Set ticks and labels
    ax.set_xticks(range(len(top_kp_indices)))
    ax.set_yticks(range(len(top_kp_indices)))
    ax.set_xticklabels([keypoint_names.get(kp, f'KP{kp}') for kp in top_kp_indices], 
                       rotation=90, ha='right', fontsize=7)
    ax.set_yticklabels([keypoint_names.get(kp, f'KP{kp}') for kp in top_kp_indices], fontsize=7)
    
    ax.set_xlabel('Keypoint', fontsize=12, fontweight='bold')
    ax.set_ylabel('Keypoint', fontsize=12, fontweight='bold')
    ax.set_title(f'Keypoint Activation Co-occurrence: {video_id}\n(Top 30 Most Activated Keypoints)', 
                fontsize=14, fontweight='bold')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Number of Co-occurrences', fontsize=11, fontweight='bold')
    
    # Add description below the plot
    ax.text(0.5, -0.1, 
            'What: Heatmap showing how often keypoints activate together (same frame). '
            'Brighter colors indicate stronger co-activation patterns. Diagonal shows self-co-occurrence.\n'
            'How: For each frame, count pairs of keypoints that both exceed their thresholds simultaneously. '
            'Matrix entry (i,j) = number of frames where both keypoint i and j are activated.',
            transform=ax.transAxes, fontsize=9, verticalalignment='top', ha='center',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='wheat', alpha=0.3))
    
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    cooccurrence_path = output_dir / f"{video_id}_keypoint_activation_cooccurrence.png"
    plt.savefig(cooccurrence_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Co-occurrence heatmap saved to: {cooccurrence_path}")
    
    # 3.5 Activation Sequences (common patterns)
    print("Analyzing activation sequences...")
    
    # Find common activation sequences (which keypoints activate in sequence)
    # Look for patterns where keypoints activate within a short time window (e.g., 10 frames)
    sequence_window = 10
    sequences = []
    
    # Group activations by time windows
    for frame_start in range(0, len(frame_to_seizure), sequence_window):
        frame_end = min(frame_start + sequence_window, len(frame_to_seizure))
        window_activations = set()
        
        for event in activation_data:
            if frame_start <= event['frame'] < frame_end:
                window_activations.add(event['keypoint_idx'])
        
        if len(window_activations) >= 2:
            sequences.append(sorted(window_activations))
    
    # Find most common sequences
    sequence_counts = Counter(tuple(seq) for seq in sequences)
    most_common_sequences = sequence_counts.most_common(10)
    
    # Create sequence visualization
    fig, ax = plt.subplots(figsize=(14, 8))
    
    if most_common_sequences:
        y_positions = np.arange(len(most_common_sequences))
        sequence_labels = []
        counts = []
        
        for seq_tuple, count in most_common_sequences:
            seq_list = list(seq_tuple)
            seq_names = [keypoint_names.get(kp, f'KP{kp}') for kp in seq_list]
            # Create compact label
            label = ' → '.join(seq_names[:5])  # Show first 5 keypoints
            if len(seq_names) > 5:
                label += f' ... (+{len(seq_names)-5} more)'
            sequence_labels.append(label)
            counts.append(count)
        
        bars = ax.barh(y_positions, counts, color='mediumpurple', alpha=0.7)
        ax.set_yticks(y_positions)
        ax.set_yticklabels(sequence_labels, fontsize=9)
        ax.set_xlabel('Frequency of Sequence', fontsize=12, fontweight='bold')
        ax.set_ylabel('Keypoint Activation Sequence', fontsize=12, fontweight='bold')
        ax.set_title(f'Most Common Keypoint Activation Sequences: {video_id}\n(Within {sequence_window}-frame windows)', 
                    fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
        ax.invert_yaxis()
        
        # Add value labels
        for i, count in enumerate(counts):
            ax.text(count, i, f' {count}', va='center', fontsize=9)
        
        # Add description below the plot
        ax.text(0.5, -0.12, 
                f'What: Most frequent combinations of keypoints activating within {sequence_window}-frame windows. '
                'Sequences show which keypoints tend to activate together, revealing coordinated body movements.\n'
                f'How: Slide a {sequence_window}-frame window across the video. For each window, collect all activated keypoints. '
                'Count frequency of each unique combination. Sequences with ≥2 keypoints are shown.',
                transform=ax.transAxes, fontsize=9, verticalalignment='top', ha='center',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='wheat', alpha=0.3))
    else:
        ax.text(0.5, 0.5, 'No common activation sequences found', 
               ha='center', va='center', fontsize=14, transform=ax.transAxes)
        ax.set_title(f'Keypoint Activation Sequences: {video_id}', fontsize=14, fontweight='bold')
    
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    sequence_path = output_dir / f"{video_id}_keypoint_activation_sequences.png"
    plt.savefig(sequence_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Activation sequences saved to: {sequence_path}")
    
    # Print summary statistics
    print(f"\n📊 Activation Analysis Summary:")
    print(f"  Total activation events: {len(activation_data)}")
    print(f"  Keypoints with activations: {len([kp for kp, events in activation_events.items() if len(events) > 0])}")
    print(f"  Activations during seizures: {sum(1 for e in activation_data if e['is_seizure'])} (rate: {sum(1 for e in activation_data if e['is_seizure'])/seizure_frame_count if seizure_frame_count > 0 else 0:.4f} per frame)")
    print(f"  Activations during non-seizure: {sum(1 for e in activation_data if not e['is_seizure'])} (rate: {sum(1 for e in activation_data if not e['is_seizure'])/non_seizure_frame_count if non_seizure_frame_count > 0 else 0:.4f} per frame)")
    print(f"  Seizure frames: {seizure_frame_count}, Non-seizure frames: {non_seizure_frame_count}")
    print(f"  Most frequently activated keypoint: {keypoint_names.get(max(activation_counts, key=activation_counts.get), 'unknown')}")


def analyze_keypoint_statistics(video_id, video_clips, output_dir):
    """
    Main orchestration function for keypoint statistical analysis.
    
    Args:
        video_id (str): Video ID
        video_clips (dict): Dictionary of clip data {clip_name: {saliency_map, true_label, pred_label, confidence}}
        output_dir (Path): Base output directory (will create keypoint_statistics/ subdirectory)
    """
    print(f"\n{'='*60}")
    print(f"📊 KEYPOINT STATISTICAL ANALYSIS: {video_id}")
    print(f"{'='*60}")
    
    # Create output subdirectory
    stats_dir = output_dir / "keypoint_statistics"
    stats_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {stats_dir}")
    
    # Get all clip names and sort them chronologically
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
        print("Warning: No clips found. Cannot perform statistical analysis.")
        return
    
    first_clip_data = video_clips[sorted_clips[0]]
    first_saliency_map = first_clip_data['saliency_map']
    
    if first_saliency_map.ndim != 2:
        print(f"Warning: Expected saliency map with shape (T, V), but got {first_saliency_map.shape}")
        return
    
    num_keypoints = first_saliency_map.shape[1]
    print(f"Detected {num_keypoints} keypoints")
    
    # Run both analyses
    try:
        analyze_keypoint_importance_ranking(video_id, video_clips, sorted_clips, num_keypoints, stats_dir)
    except Exception as e:
        print(f"❌ Error in keypoint importance ranking: {e}")
        import traceback
        traceback.print_exc()
    
    try:
        analyze_keypoint_activation_sequences(video_id, video_clips, sorted_clips, num_keypoints, stats_dir)
    except Exception as e:
        print(f"❌ Error in keypoint activation sequences: {e}")
        import traceback
        traceback.print_exc()
    
    print(f"\n✅ Keypoint statistical analysis completed for {video_id}")
    print(f"📁 All outputs saved to: {stats_dir}")

