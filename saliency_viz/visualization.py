"""
Functions for visualizing saliency maps.
"""

import os
import cv2
import numpy as np
import pickle
from pathlib import Path
import re
import matplotlib.pyplot as plt
import torch
import pandas as pd
import json # For saving aggregated data

from .utils import frame_extract, rdbu_color, load_visibility_data
from .saliency import calculate_motion_data_without_outliers


def plot_saliency(input_data, saliency_map, true_label, pred_label, save_path, model=None, clip_name=None, confidence=None):
    # Create a comprehensive 2x3 subplot layout
    fig = plt.figure(figsize=(24, 16))
    
    clip_display = clip_name if clip_name else "Unknown Clip"
    clip_type = "Seizure" if true_label == 1 else "Non-Seizure"
    pred_type = "Seizure" if pred_label == 1 else "Non-Seizure"
    conf_text = f" (Conf: {confidence:.3f})" if confidence else ""
    main_title = f"Comprehensive Saliency Analysis: {clip_display}\nTrue: {clip_type} | Pred: {pred_type}{conf_text}"
    fig.suptitle(main_title, fontsize=14, y=0.95)
    
    keypoint_groups = {
        'Body': list(range(0, 17)),
        'Feet': list(range(17, 23)),
        'Face': list(range(23, 91)),
        'Left Hand': list(range(91, 112)),
        'Right Hand': list(range(112, 133))
    }
    
    input_vis = input_data.squeeze().detach().cpu().numpy()
    
    if input_vis.ndim != 3:
        raise ValueError(f"Expected input shape (T, V, C), but got {input_vis.shape}")
    
    motion_data = calculate_motion_data_without_outliers(input_vis)
    
    if saliency_map.ndim > 2:
        saliency_map = saliency_map.squeeze()
    
    max_abs_val = np.max(np.abs(saliency_map))
    saliency_norm = saliency_map / (max_abs_val + 1e-8)
    
    # Calculate motion magnitude per frame (sum across all keypoints for each frame)
    motion_magnitude = np.sqrt(np.sum(motion_data**2, axis=0))  # Shape: (T,)
    # Calculate saliency intensity per frame (mean across all keypoints for each frame)
    saliency_intensity = np.mean(np.abs(saliency_norm), axis=0)  # Shape: (T,)
    
    # Ensure both arrays have the same length
    min_length = min(len(motion_magnitude), len(saliency_intensity))
    motion_magnitude = motion_magnitude[:min_length]
    saliency_intensity = saliency_intensity[:min_length]
    
    # 1. Motion Heatmap (Top Left)
    ax1 = plt.subplot(2, 3, 1)
    im1 = ax1.imshow(motion_data, aspect='auto', cmap='viridis')
    ax1.set_title('1. Motion Between Frames', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Keypoints')
    ax1.set_xlabel('Time (frames)')
    plt.colorbar(im1, ax=ax1, shrink=0.8)
    
    # 2. Saliency Heatmap (Top Middle)
    ax2 = plt.subplot(2, 3, 2)
    saliency_viz = saliency_norm.T
    im2 = ax2.imshow(saliency_viz, aspect='auto', cmap='seismic', vmin=-1, vmax=1)
    ax2.set_title('2. Saliency Heatmap', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Keypoints')
    ax2.set_xlabel('Time (frames)')
    plt.colorbar(im2, ax=ax2, shrink=0.8)
    
    # 3. Motion-Saliency Correlation (Top Right)
    ax3 = plt.subplot(2, 3, 3)
    
    # Calculate correlation with error handling
    try:
        if len(motion_magnitude) > 1 and len(saliency_intensity) > 1:
            correlation = np.corrcoef(motion_magnitude, saliency_intensity)[0, 1]
            if np.isnan(correlation):
                correlation = 0.0
        else:
            correlation = 0.0
    except:
        correlation = 0.0
    
    ax3.scatter(motion_magnitude, saliency_intensity, alpha=0.6, s=20)
    ax3.set_title(f'3. Motion-Saliency Correlation\n(r = {correlation:.3f})', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Motion Magnitude')
    ax3.set_ylabel('Saliency Intensity')
    ax3.grid(True, alpha=0.3)
    
    # Add trend line with error handling
    try:
        if len(motion_magnitude) > 1 and len(saliency_intensity) > 1:
            z = np.polyfit(motion_magnitude, saliency_intensity, 1)
            p = np.poly1d(z)
            ax3.plot(motion_magnitude, p(motion_magnitude), "r--", alpha=0.8, linewidth=2)
    except:
        pass  # Skip trend line if polyfit fails
    
    # 4. Temporal Saliency Traces (Bottom Left)
    ax4 = plt.subplot(2, 3, 4)
    colors = ['red', 'blue', 'green', 'orange', 'purple']
    for i, (group_name, indices) in enumerate(keypoint_groups.items()):
        if indices and len(indices) > 0:
            group_saliency = np.mean(saliency_norm[:, indices], axis=1)
            ax4.plot(group_saliency, label=group_name, color=colors[i % len(colors)], linewidth=2)
    
    ax4.set_title('4. Temporal Saliency Traces', fontsize=12, fontweight='bold')
    ax4.set_xlabel('Time (frames)')
    ax4.set_ylabel('Average Saliency')
    ax4.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax4.grid(True, alpha=0.3)
    ax4.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    
    # 5. Keypoint-Specific Saliency Bars (Bottom Middle)
    ax5 = plt.subplot(2, 3, 5)
    group_names = []
    group_saliencies = []
    for group_name, indices in keypoint_groups.items():
        if indices and len(indices) > 0:
            group_saliency = np.mean(np.abs(saliency_norm[:, indices]))
            group_names.append(group_name)
            group_saliencies.append(group_saliency)
    
    bars = ax5.bar(group_names, group_saliencies, color=['red', 'blue', 'green', 'orange', 'purple'][:len(group_names)])
    ax5.set_title('5. Keypoint Group Saliency', fontsize=12, fontweight='bold')
    ax5.set_ylabel('Average |Saliency|')
    ax5.tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for bar, val in zip(bars, group_saliencies):
        ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{val:.3f}', ha='center', va='bottom', fontsize=10)
    
    # 6. Motion-Saliency Overlay (Bottom Right)
    ax6 = plt.subplot(2, 3, 6)
    
    # Create overlay: motion as background, saliency as overlay
    motion_bg = ax6.imshow(motion_data, aspect='auto', cmap='viridis', alpha=0.7)
    
    # Create saliency overlay with transparency
    saliency_overlay = np.zeros_like(saliency_viz)
    saliency_overlay[saliency_viz > 0] = saliency_viz[saliency_viz > 0]  # Only positive saliency
    im6 = ax6.imshow(saliency_overlay, aspect='auto', cmap='Reds', alpha=0.6, vmin=0, vmax=1)
    
    ax6.set_title('6. Motion + Saliency Overlay', fontsize=12, fontweight='bold')
    ax6.set_ylabel('Keypoints')
    ax6.set_xlabel('Time (frames)')
    
    # Add colorbar for saliency overlay
    cbar6 = plt.colorbar(im6, ax=ax6, shrink=0.8)
    cbar6.set_label('Positive Saliency', rotation=270, labelpad=15)
    
    # Set common properties for all subplots
    for ax in [ax1, ax2, ax4, ax6]:
        ax.set_xticks(np.arange(0, motion_data.shape[1], max(1, motion_data.shape[1]//10)))
        ax.set_xticklabels([f"{i}" for i in range(0, motion_data.shape[1], max(1, motion_data.shape[1]//10))])
        
        group_positions = []
        group_labels = []
        for group_name, indices in keypoint_groups.items():
            if indices:
                mid_point = (indices[0] + indices[-1]) / 2
                group_positions.append(mid_point)
                group_labels.append(f"{group_name}\n({len(indices)})")
        
        ax.set_yticks(group_positions)
        ax.set_yticklabels(group_labels)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


# --- NEW FUNCTION: Average Saliency Map Calculation ---
def calculate_and_save_average_saliency(fold, video_id, video_clips, output_dir):
    """
    Calculates the average saliency map for correctly classified examples
    of each class within a video's clips. Saves the average maps as .npy
    files and generates simple visualizations.

    Args:
        fold (int): The fold number.
        video_id (str): The video ID.
        video_clips (dict): Dictionary of clip data
                           {clip_name: {saliency_map, true_label, pred_label, ...}}.
        output_dir (Path): Base output directory (e.g., 'k_fold/saliency_maps').
    """
    print(f"\n--- Calculating Average Saliency Maps for Video: {video_id} (Fold {fold}) ---")

    # Define class labels (assuming 0: Non-Seizure, 1: Seizure)
    class_labels = {0: 'Non-Seizure', 1: 'Seizure'}
    saliency_sums = {label: None for label in class_labels}
    saliency_counts = {label: 0 for label in class_labels}
    map_shape = None

    if not video_clips:
        print("Warning: No video clips data provided. Cannot calculate average saliency.")
        return

    # Accumulate saliency maps for correctly classified clips
    for clip_name, clip_data in video_clips.items():
        true_label = clip_data.get('true_label')
        pred_label = clip_data.get('pred_label')
        saliency_map = clip_data.get('saliency_map') # Expected shape: (T, V) or similar

        if saliency_map is None or true_label is None or pred_label is None:
            print(f"Warning: Missing data for clip {clip_name}. Skipping.")
            continue

        # Check if prediction is correct
        if true_label == pred_label:
            current_class = true_label
            if current_class in class_labels:
                if saliency_sums[current_class] is None:
                    # Initialize sum array with zeros based on first map's shape
                    map_shape = saliency_map.shape
                    saliency_sums[current_class] = np.zeros_like(saliency_map, dtype=np.float64)
                    print(f"Initialized average map for class {class_labels[current_class]} with shape {map_shape}")

                # Ensure shapes match before adding (important if maps could vary)
                if saliency_map.shape == map_shape:
                    saliency_sums[current_class] += saliency_map
                    saliency_counts[current_class] += 1
                else:
                    print(f"Warning: Shape mismatch for clip {clip_name} ({saliency_map.shape} vs {map_shape}). Skipping.")

    # Create a dedicated directory for average maps within the video's output
    avg_map_dir = output_dir / video_id / "average_saliency"
    avg_map_dir.mkdir(parents=True, exist_ok=True)

    # Calculate and save average maps
    for label, class_name in class_labels.items():
        count = saliency_counts[label]
        if count > 0:
            average_map = saliency_sums[label] / count
            print(f"Calculated average saliency for class '{class_name}' from {count} correctly classified clips.")

            # --- Save the raw average map ---
            save_path_npy = avg_map_dir / f"avg_saliency_map_fold{fold}_{video_id}_class_{label}_{class_name}.npy"
            np.save(save_path_npy, average_map)
            print(f"Saved raw average map to: {save_path_npy}")

            # --- Generate and save a visualization ---
            save_path_png = avg_map_dir / f"avg_saliency_map_fold{fold}_{video_id}_class_{label}_{class_name}.png"
            try:
                plt.figure(figsize=(10, 4))
                # Determine robust color limits based on the average map itself
                vmax = np.percentile(np.abs(average_map), 99) # Use 99th percentile of absolute values for scaling
                vmin = -vmax
                plt.imshow(average_map, cmap='RdBu', aspect='auto', vmin=vmin, vmax=vmax)
                plt.colorbar(label='Average Saliency')
                plt.title(f"Average Saliency Map - Class: {class_name}\nVideo: {video_id} (Fold {fold}) - {count} clips")
                plt.xlabel("Keypoints / Vertices")
                plt.ylabel("Frames / Time")
                plt.tight_layout()
                plt.savefig(save_path_png, dpi=150)
                plt.close()
                print(f"Saved average map visualization to: {save_path_png}")
            except Exception as e:
                print(f"Error generating visualization for class {class_name}: {e}")

        else:
            print(f"No correctly classified clips found for class '{class_name}'. Cannot calculate average map.")

    print(f"--- Finished Average Saliency Map Calculation for Video: {video_id} ---")

# --- NEW FUNCTION: Body Part Saliency Aggregation ---
def calculate_and_save_body_part_saliency(fold, video_id, video_clips, output_dir):
    """
    Calculates the average POSITIVE and average NEGATIVE saliency contributions
    aggregated by body part for correctly classified examples of each class.
    Saves the aggregated data and generates a comparative bar chart showing both contributions.

    Args:
        fold (int): The fold number.
        video_id (str): The video ID.
        video_clips (dict): Dictionary of clip data
                           {clip_name: {saliency_map, true_label, pred_label, ...}}.
        output_dir (Path): Base output directory (e.g., 'k_fold/saliency_maps').
    """
    print(f"\n--- Calculating Average Positive/Negative Body Part Saliency for Video: {video_id} (Fold {fold}) ---")

    # Define body part groups (consistent with plot_saliency)
    keypoint_groups = {
        'Body': list(range(0, 17)),
        'Feet': list(range(17, 23)),
        'Face': list(range(23, 91)),
        'Left Hand': list(range(91, 112)),
        'Right Hand': list(range(112, 133))
    }
    # Ensure all keypoints are covered (optional check)
    all_indices = set(idx for indices in keypoint_groups.values() for idx in indices)
    max_index = max(all_indices) if all_indices else -1
    print(f"Keypoint groups cover indices up to {max_index}")

    # Define class labels
    class_labels = {0: 'Non-Seizure', 1: 'Seizure'}
    
    # Store sums of *average positive* and *average negative* saliency per part per clip
    # Structure: sums[class_label][part_name]['positive'/'negative']
    body_part_sums = {
        label: {part: {'positive': 0.0, 'negative': 0.0} for part in keypoint_groups}
        for label in class_labels
    }
    # Store counts of clips per class
    clip_counts = {label: 0 for label in class_labels}
    num_keypoints = -1 # To store the number of keypoints from the first valid map

    if not video_clips:
        print("Warning: No video clips data provided. Cannot calculate body part saliency.")
        return

    # Accumulate aggregated saliency for correctly classified clips
    for clip_name, clip_data in video_clips.items():
        true_label = clip_data.get('true_label')
        pred_label = clip_data.get('pred_label')
        saliency_map = clip_data.get('saliency_map') # Expected shape: (T, V)

        if saliency_map is None or true_label is None or pred_label is None:
            continue

        # Store number of keypoints if not already set
        if num_keypoints == -1:
            if saliency_map.ndim == 2 and saliency_map.shape[1] > 0:
                 num_keypoints = saliency_map.shape[1]
                 print(f"Detected {num_keypoints} keypoints from saliency map shape {saliency_map.shape}")
            else:
                print(f"Warning: Invalid saliency map shape {saliency_map.shape} for clip {clip_name}. Cannot determine keypoint count yet.")
                continue # Skip if shape is wrong

        # Check if prediction is correct
        if true_label == pred_label:
            current_class = true_label
            if current_class in class_labels:
                # Ensure map shape is consistent
                if saliency_map.ndim != 2 or saliency_map.shape[1] != num_keypoints:
                    print(f"Warning: Saliency map shape mismatch for clip {clip_name} ({saliency_map.shape} vs expected (T, {num_keypoints})). Skipping.")
                    continue

                # Calculate average positive and negative saliency for each body part in this clip
                clip_part_avg = {part: {'positive': 0.0, 'negative': 0.0} for part in keypoint_groups}

                for part_name, indices in keypoint_groups.items():
                    # Filter indices to be within the valid range [0, num_keypoints-1]
                    valid_indices = [idx for idx in indices if 0 <= idx < num_keypoints]
                    if not valid_indices:
                        continue

                    # Select columns (keypoints) for this part
                    part_saliency = saliency_map[:, valid_indices] # (T, num_valid_indices_in_part)

                    # Separate positive and negative values
                    positive_values = part_saliency[part_saliency > 0]
                    negative_values = part_saliency[part_saliency < 0]

                    # Calculate mean of positive values (will be >= 0)
                    mean_pos = np.mean(positive_values) if positive_values.size > 0 else 0.0
                    # Calculate mean of negative values (will be <= 0)
                    mean_neg = np.mean(negative_values) if negative_values.size > 0 else 0.0

                    clip_part_avg[part_name]['positive'] = mean_pos
                    clip_part_avg[part_name]['negative'] = mean_neg

                # Add this clip's average part saliencies (pos/neg) to the running sum for the class
                for part_name in keypoint_groups:
                    body_part_sums[current_class][part_name]['positive'] += clip_part_avg[part_name]['positive']
                    body_part_sums[current_class][part_name]['negative'] += clip_part_avg[part_name]['negative']

                clip_counts[current_class] += 1

    # Create a dedicated directory for these results (fixed: no deep nesting)
    body_part_dir = output_dir
    body_part_dir.mkdir(parents=True, exist_ok=True)

    # Calculate final average positive and negative body part saliency per class
    average_body_part_saliency = {
        class_labels[label]: {part: {'positive': 0.0, 'negative': 0.0} for part in keypoint_groups}
        for label in class_labels
    }
    plot_data = {'Part': list(keypoint_groups.keys())}
    plot_categories = [] # To store column names for the plot

    for label, class_name in class_labels.items():
        count = clip_counts[label]
        class_averages = {part: {'positive': 0.0, 'negative': 0.0} for part in keypoint_groups}
        pos_col_name = f"{class_name} Positive"
        neg_col_name = f"{class_name} Negative"
        plot_categories.extend([pos_col_name, neg_col_name])
        class_pos_values = []
        class_neg_values = []

        if count > 0:
            print(f"Calculating average Pos/Neg body part saliency for class '{class_name}' from {count} clips.")
            for part_name in keypoint_groups:
                avg_pos = body_part_sums[label][part_name]['positive'] / count
                avg_neg = body_part_sums[label][part_name]['negative'] / count
                class_averages[part_name]['positive'] = avg_pos
                class_averages[part_name]['negative'] = avg_neg
                class_pos_values.append(avg_pos)
                class_neg_values.append(avg_neg)
            average_body_part_saliency[class_name] = class_averages
            plot_data[pos_col_name] = class_pos_values
            plot_data[neg_col_name] = class_neg_values
        else:
            print(f"No correctly classified clips found for class '{class_name}'. Cannot calculate average Pos/Neg body part saliency.")
            average_body_part_saliency[class_name] = class_averages # Already initialized with zeros
            plot_data[pos_col_name] = [0.0] * len(keypoint_groups)
            plot_data[neg_col_name] = [0.0] * len(keypoint_groups)

    # --- Save the aggregated data ---
    save_path_json = body_part_dir / f"avg_pos_neg_body_part_saliency_fold{fold}_{video_id}.json"
    try:
        # Convert numpy types to Python native types for JSON serialization
        def convert_numpy_types(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, (np.int32, np.int64)):
                return int(obj)
            elif isinstance(obj, dict):
                return {key: convert_numpy_types(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            else:
                return obj
        
        converted_data = convert_numpy_types(average_body_part_saliency)
        with open(save_path_json, 'w') as f:
            json.dump(converted_data, f, indent=4)
        print(f"Saved average Pos/Neg body part saliency data to: {save_path_json}")
    except Exception as e:
        print(f"Error saving body part saliency data to JSON: {e}")

    # --- Generate and save a comparison bar chart (Positive vs Negative) ---
    save_path_png = body_part_dir / f"avg_pos_neg_body_part_saliency_fold{fold}_{video_id}_comparison.png"
    try:
        df = pd.DataFrame(plot_data)
        if df.empty or len(df) == 0:
             print("Cannot generate plot: No data available.")
             return

        # Plotting - Use pandas to create grouped bars
        ax = df.plot(x='Part', y=plot_categories, kind='bar', figsize=(15, 8), rot=45) # Wider figure, rotate labels

        plt.title(f"Average Positive vs. Negative Saliency per Body Part\nVideo: {video_id} (Fold {fold})")
        plt.ylabel("Average Saliency Contribution")
        plt.xlabel("Body Part")
        # Add a horizontal line at y=0 for reference
        plt.axhline(0, color='grey', linewidth=0.8, linestyle='--')
        plt.legend(title='Class & Contribution Type', bbox_to_anchor=(1.02, 1), loc='upper left') # Adjust legend position
        plt.tight_layout(rect=[0, 0, 0.85, 1]) # Adjust layout to make space for legend
        plt.savefig(save_path_png, dpi=150)
        plt.close()
        print(f"Saved Pos/Neg body part saliency comparison chart to: {save_path_png}")
    except Exception as e:
        print(f"Error generating body part saliency comparison chart: {e}")
        import traceback
        traceback.print_exc()

    # --- NEW: Generate and save absolute contribution histogram ---
    save_path_abs_png = body_part_dir / f"avg_absolute_body_part_saliency_fold{fold}_{video_id}.png"
    try:
        # Calculate absolute contributions for each class
        abs_plot_data = {'Part': list(keypoint_groups.keys())}
        
        for label, class_name in class_labels.items():
            abs_col_name = f"{class_name} Absolute"
            abs_values = []
            
            for part_name in keypoint_groups:
                # Calculate absolute contribution (sum of positive and negative)
                pos_val = average_body_part_saliency[class_name][part_name]['positive']
                neg_val = abs(average_body_part_saliency[class_name][part_name]['negative'])
                abs_contribution = pos_val + neg_val
                abs_values.append(abs_contribution)
            
            abs_plot_data[abs_col_name] = abs_values
        
        # Create absolute contribution plot
        df_abs = pd.DataFrame(abs_plot_data)
        if not df_abs.empty:
            # Plotting - Use pandas to create grouped bars
            ax = df_abs.plot(x='Part', y=[col for col in df_abs.columns if col != 'Part'], 
                           kind='bar', figsize=(12, 7), rot=45)
            
            plt.title(f"Average Absolute Saliency Contribution per Body Part\nVideo: {video_id} (Fold {fold})")
            plt.ylabel("Average Absolute Saliency Contribution")
            plt.xlabel("Body Part")
            plt.legend(title='Class', bbox_to_anchor=(1.02, 1), loc='upper left')
            plt.tight_layout(rect=[0, 0, 0.85, 1])
            plt.savefig(save_path_abs_png, dpi=150)
            plt.close()
            print(f"Saved absolute body part saliency chart to: {save_path_abs_png}")
        else:
            print("Cannot generate absolute plot: No data available.")
            
    except Exception as e:
        print(f"Error generating absolute body part saliency chart: {e}")
        import traceback
        traceback.print_exc()

    print(f"--- Finished Pos/Neg Body Part Saliency Aggregation for Video: {video_id} ---")

def analyze_class_discrimination(fold, video_id, video_clips, output_dir):
    """
    Identifies which body parts show the greatest difference in saliency between classes.
    This helps determine which body regions are most important for classification.
    
    Args:
        fold (int): The fold number.
        video_id (str): The video ID.
        video_clips (dict): Dictionary of clip data.
        output_dir (Path): Base output directory.
    """
    print(f"\n--- Analyzing Class Discrimination by Body Part for Video: {video_id} (Fold {fold}) ---")

    # Define body part groups (same as in other functions)
    keypoint_groups = {
        'Body': list(range(0, 17)),
        'Feet': list(range(17, 23)),
        'Face': list(range(23, 91)),
        'Left Hand': list(range(91, 112)),
        'Right Hand': list(range(112, 133))
    }
    
    # Define class labels
    class_labels = {0: 'Non-Seizure', 1: 'Seizure'}
    
    # First, calculate average saliency per body part per class
    body_part_sums = {label: {part: 0.0 for part in keypoint_groups} for label in class_labels}
    clip_counts = {label: 0 for label in class_labels}
    num_keypoints = -1
    
    # Process clips to calculate average saliency per body part per class
    for clip_name, clip_data in video_clips.items():
        true_label = clip_data.get('true_label')
        pred_label = clip_data.get('pred_label')
        saliency_map = clip_data.get('saliency_map')
        
        if saliency_map is None or true_label is None or pred_label is None:
            continue
            
        # Initialize num_keypoints if not set
        if num_keypoints == -1:
            if saliency_map.ndim == 2 and saliency_map.shape[1] > 0:
                num_keypoints = saliency_map.shape[1]
                print(f"Detected {num_keypoints} keypoints from saliency map shape {saliency_map.shape}")
            else:
                print(f"Warning: Invalid saliency map shape {saliency_map.shape} for clip {clip_name}. Cannot determine keypoint count yet.")
                continue
        
        # Only use correctly classified clips
        if true_label == pred_label:
            current_class = true_label
            if current_class in class_labels:
                # Ensure map shape is consistent
                if saliency_map.ndim != 2 or saliency_map.shape[1] != num_keypoints:
                    print(f"Warning: Saliency map shape mismatch for clip {clip_name} ({saliency_map.shape} vs expected (T, {num_keypoints})). Skipping.")
                    continue
                    
                # Calculate average absolute saliency for each body part
                for part_name, indices in keypoint_groups.items():
                    valid_indices = [idx for idx in indices if 0 <= idx < num_keypoints]
                    if not valid_indices:
                        continue
                        
                    part_saliency = np.abs(saliency_map[:, valid_indices])
                    mean_saliency = np.mean(part_saliency) if part_saliency.size > 0 else 0.0
                    body_part_sums[current_class][part_name] += mean_saliency
                
                clip_counts[current_class] += 1
    
    # Calculate average saliency per body part per class
    average_saliency = {class_labels[label]: {} for label in class_labels}
    for label, class_name in class_labels.items():
        count = clip_counts[label]
        if count > 0:
            print(f"Calculating average saliency for class '{class_name}' from {count} clips.")
            for part_name in keypoint_groups:
                average_saliency[class_name][part_name] = body_part_sums[label][part_name] / count
        else:
            print(f"No correctly classified clips found for class '{class_name}'. Cannot calculate average saliency.")
            for part_name in keypoint_groups:
                average_saliency[class_name][part_name] = 0.0
    
    # Calculate the absolute difference between classes for each body part
    class_diff = {}
    for part in keypoint_groups:
        seizure_val = average_saliency['Seizure'].get(part, 0)
        non_seizure_val = average_saliency['Non-Seizure'].get(part, 0)
        
        # Calculate absolute difference (discrimination power)
        class_diff[part] = abs(seizure_val - non_seizure_val)
    
    # Sort parts by discrimination power (highest to lowest)
    sorted_parts = sorted(class_diff.items(), key=lambda x: x[1], reverse=True)
    
    # Create output directory
    discrimination_dir = output_dir / video_id / "class_discrimination"
    discrimination_dir.mkdir(parents=True, exist_ok=True)
    
    # Save the discrimination data as JSON
    discrimination_data = {
        "discrimination_power": class_diff,
        "average_saliency": average_saliency,
        "sorted_parts": [{"part": p, "discrimination_power": v} for p, v in sorted_parts]
    }
    
    json_path = discrimination_dir / f"class_discrimination_fold{fold}_{video_id}.json"
    try:
        with open(json_path, 'w') as f:
            json.dump(discrimination_data, f, indent=4)
        print(f"Saved class discrimination data to: {json_path}")
    except Exception as e:
        print(f"Error saving class discrimination data: {e}")
    
    # Create a bar chart of discrimination power
    try:
        plt.figure(figsize=(10, 6))
        parts = [p for p, _ in sorted_parts]
        values = [v for _, v in sorted_parts]
        
        bars = plt.bar(parts, values, color='skyblue')
        
        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                    f'{height:.3f}', ha='center', va='bottom', rotation=0)
        
        plt.title(f"Body Parts by Class Discrimination Power\nVideo: {video_id} (Fold {fold})")
        plt.ylabel("Absolute Difference in Saliency Between Classes")
        plt.xlabel("Body Part")
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        # Save the plot
        plt.savefig(discrimination_dir / f"class_discrimination_power_fold{fold}_{video_id}.png", dpi=150)
        plt.close()
        print(f"Saved class discrimination power chart to: {discrimination_dir}")
    except Exception as e:
        print(f"Error generating class discrimination chart: {e}")
        import traceback
        traceback.print_exc()
    
    # Create a comparative bar chart showing both classes side by side
    try:
        plt.figure(figsize=(12, 7))
        
        # Prepare data for grouped bar chart
        x = np.arange(len(parts))
        width = 0.35
        
        # Get values for each class in the same order as sorted_parts
        seizure_values = [average_saliency['Seizure'].get(part, 0) for part, _ in sorted_parts]
        non_seizure_values = [average_saliency['Non-Seizure'].get(part, 0) for part, _ in sorted_parts]
        
        # Create grouped bars
        plt.bar(x - width/2, seizure_values, width, label='Seizure', color='red', alpha=0.7)
        plt.bar(x + width/2, non_seizure_values, width, label='Non-Seizure', color='blue', alpha=0.7)
        
        plt.title(f"Average Saliency by Body Part and Class\nVideo: {video_id} (Fold {fold})")
        plt.ylabel("Average Absolute Saliency")
        plt.xlabel("Body Part")
        plt.xticks(x, parts, rotation=45, ha='right')
        plt.legend()
        plt.tight_layout()
        
        # Save the comparative plot
        plt.savefig(discrimination_dir / f"class_comparison_fold{fold}_{video_id}.png", dpi=150)
        plt.close()
        print(f"Saved class comparison chart to: {discrimination_dir}")
    except Exception as e:
        print(f"Error generating class comparison chart: {e}")
        import traceback
        traceback.print_exc()
    
    print(f"--- Finished Class Discrimination Analysis for Video: {video_id} ---")

def create_body_part_saliency_histogram(clip_name, saliency_map, true_label, pred_label, output_dir, confidence=None):
    """
    Creates a histogram showing the total absolute saliency contribution grouped by body parts for a specific clip.
    
    Args:
        clip_name (str): Name of the clip
        saliency_map (ndarray): Saliency map with shape (T, V) where T is time and V is vertices/keypoints
        true_label (int): Ground truth label (0=Non-Seizure, 1=Seizure)
        pred_label (int): Predicted label (0=Non-Seizure, 1=Seizure)
        output_dir (Path): Base output directory
        confidence (float, optional): Prediction confidence
    
    Returns:
        Path: Path to the saved histogram image
    """
    # Define body part groups (consistent with other functions)
    keypoint_groups = {
        'Body': list(range(0, 17)),
        'Feet': list(range(17, 23)),
        'Face': list(range(23, 91)),
        'Left Hand': list(range(91, 112)),
        'Right Hand': list(range(112, 133))
    }
    
    # Create output directory
    histogram_dir = output_dir / "saliency_histograms"
    histogram_dir.mkdir(exist_ok=True, parents=True)
    
    # Output file path
    output_path = histogram_dir / f"{clip_name}_saliency_histogram.png"
    
    # Check if saliency map has expected shape
    if saliency_map.ndim != 2:
        print(f"Warning: Expected saliency map with shape (T, V), but got {saliency_map.shape}. Skipping histogram.")
        return None
    
    # Calculate absolute saliency for each body part
    body_part_stats = {}
    
    for part_name, indices in keypoint_groups.items():
        # Filter indices to be within the valid range
        valid_indices = [idx for idx in indices if 0 <= idx < saliency_map.shape[1]]
        if not valid_indices:
            continue
            
        # Extract saliency values for this body part
        part_saliency = saliency_map[:, valid_indices]
        
        # Calculate statistics
        body_part_stats[part_name] = {
            'sum_absolute': np.sum(np.abs(part_saliency)),
            'num_keypoints': len(valid_indices)
        }
    
    # Sort body parts by absolute saliency for consistent ordering
    sorted_parts = sorted(body_part_stats.keys(), 
                         key=lambda x: body_part_stats[x]['sum_absolute'],
                         reverse=True)
    
    # Create figure for the single histogram
    plt.figure(figsize=(10, 6))
    
    # Get absolute values for each body part
    abs_values = [body_part_stats[part]['sum_absolute'] for part in sorted_parts]
    
    # Create a colormap based on the values
    colors = plt.cm.viridis(np.array(abs_values) / max(abs_values) if max(abs_values) > 0 else np.zeros(len(abs_values)))
    
    # Create the bar chart
    x = np.arange(len(sorted_parts))
    width = 0.7
    bars = plt.bar(x, abs_values, width, color=colors)
    
    # Add title and labels
    plt.title(f"Total Absolute Saliency by Body Part - Clip: {clip_name}\n"
              f"True: {'Seizure' if true_label == 1 else 'Non-Seizure'}, "
              f"Pred: {'Seizure' if pred_label == 1 else 'Non-Seizure'}"
              f"{f' ({confidence:.2f})' if confidence is not None else ''}", 
              fontsize=12)
    
    plt.ylabel('Sum of Absolute Saliency')
    plt.xlabel('Body Part')
    plt.xticks(x, sorted_parts, rotation=45, ha='right')
    
    # Add keypoint counts as text
    for i, part in enumerate(sorted_parts):
        plt.text(i, abs_values[i] + max(abs_values) * 0.02, 
                f"{body_part_stats[part]['num_keypoints']} pts", 
                ha='center', va='bottom', fontsize=9)
    
    # Add a note about interpretation
    plt.figtext(0.5, 0.01, 
               "Absolute saliency represents the overall impact of each body part\n"
               "on the model's decision, regardless of direction (positive or negative).", 
               ha='center', fontsize=9, bbox=dict(facecolor='white', alpha=0.8))
    
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved body part saliency histogram to {output_path}")
    return output_path


