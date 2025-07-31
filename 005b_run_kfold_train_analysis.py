from pathlib import Path
from typing import Tuple, Dict, List
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def find_log_files(fold_dir: Path) -> List[Path]:
    """Find all log files in the fold directory, sorted by timestamp."""
    log_files = []
    
    # Find all timestamped directories (format: YYYYMMDD_HHMMSS)
    timestamp_dirs = [d for d in fold_dir.glob('*_*') if d.is_dir() and len(d.name) == 15]
    
    if not timestamp_dirs:
        raise FileNotFoundError(f"No timestamped directories found in {fold_dir}")
    
    # Sort directories by timestamp (oldest first)
    timestamp_dirs.sort(key=lambda x: x.name)
    
    # Look for .log files in each directory
    for timestamp_dir in timestamp_dirs:
        log_file = timestamp_dir / f"{timestamp_dir.name}.log"
        if log_file.exists():
            log_files.append(log_file)
            print(f"Found log file: {log_file}")
    
    if not log_files:
        raise FileNotFoundError(f"No log files found in {fold_dir}")
    
    return log_files

def find_log_file(fold_dir: Path) -> Path:
    """Find the most recent log file in the fold directory (backward compatibility)."""
    log_files = find_log_files(fold_dir)
    return log_files[-1]  # Return the most recent log file

def validate_and_clean_metrics(train_metrics: Dict[str, List[float]], val_metrics: Dict[str, List[float]]) -> Tuple[Dict[str, List[float]], Dict[str, List[float]]]:
    """Validate and clean extracted metrics to handle potential issues from resumed training."""
    
    # Make copies to avoid modifying original data
    train_clean = {k: v.copy() for k, v in train_metrics.items()}
    val_clean = {k: v.copy() for k, v in val_metrics.items()}
    
    # Check for length mismatches
    train_len = len(train_clean['loss'])
    val_len = len(val_clean['loss'])
    
    if train_len != val_len:
        print(f"  Warning: Training epochs ({train_len}) != Validation epochs ({val_len})")
        
        # Truncate to the shorter length
        min_len = min(train_len, val_len)
        if min_len > 0:
            print(f"  Truncating both to {min_len} epochs")
            for key in train_clean:
                train_clean[key] = train_clean[key][:min_len]
            for key in val_clean:
                val_clean[key] = val_clean[key][:min_len]
        else:
            print("  Error: No valid epochs found")
            return {}, {}
    
    # Check for invalid values (NaN, inf, negative)
    for metric_name, values in train_clean.items():
        invalid_indices = []
        for i, val in enumerate(values):
            if not (isinstance(val, (int, float)) and val >= 0 and not np.isnan(val) and not np.isinf(val)):
                invalid_indices.append(i)
        
        if invalid_indices:
            print(f"  Warning: Found {len(invalid_indices)} invalid {metric_name} values in training metrics")
            # Remove invalid values
            for i in reversed(invalid_indices):
                for key in train_clean:
                    train_clean[key].pop(i)
                for key in val_clean:
                    val_clean[key].pop(i)
    
    for metric_name, values in val_clean.items():
        invalid_indices = []
        for i, val in enumerate(values):
            if not (isinstance(val, (int, float)) and val >= 0 and not np.isnan(val) and not np.isinf(val)):
                invalid_indices.append(i)
        
        if invalid_indices:
            print(f"  Warning: Found {len(invalid_indices)} invalid {metric_name} values in validation metrics")
            # Remove invalid values
            for i in reversed(invalid_indices):
                for key in train_clean:
                    train_clean[key].pop(i)
                for key in val_clean:
                    val_clean[key].pop(i)
    
    final_len = len(train_clean['loss'])
    if final_len > 0:
        print(f"  Final clean dataset: {final_len} epochs")
    else:
        print("  Error: No valid epochs remaining after cleaning")
    
    return train_clean, val_clean

def extract_metrics_from_latest_complete_session(log_files: List[Path]) -> Tuple[Dict[str, List[float]], Dict[str, List[float]]]:
    """Extract metrics from the most recent log file that contains complete training data."""
    
    # Sort log files by timestamp (latest first)
    sorted_log_files = sorted(log_files, key=lambda x: x.name, reverse=True)
    
    print(f"Looking for complete training session among {len(sorted_log_files)} log files...")
    
    for i, log_file in enumerate(sorted_log_files):
        print(f"Checking log file {i+1}/{len(sorted_log_files)}: {log_file.name}")
        
        # Count epochs in this file
        epoch_count = 0
        has_training_data = False
        
        try:
            with open(log_file, 'r') as f:
                for line in f:
                    if 'Saving checkpoint at' in line:
                        epoch_match = re.search(r'Saving checkpoint at (\d+) epochs?', line)
                        if epoch_match:
                            epoch_count = max(epoch_count, int(epoch_match.group(1)))
                    elif 'Epoch(train)' in line:
                        has_training_data = True
        except Exception as e:
            print(f"    Error reading {log_file}: {e}")
            continue
        
        print(f"    Found {epoch_count} epochs, has training data: {has_training_data}")
        
        # If this file has training data and epochs, use it
        if has_training_data and epoch_count > 0:
            print(f"    Using {log_file.name} as the complete training session")
            return extract_metrics_from_logs([log_file])
    
    # If no complete session found, fall back to original method
    print("    No complete training session found, using all files with deduplication")
    return extract_metrics_from_logs(log_files)

def extract_metrics_from_logs(log_files: List[Path]) -> Tuple[Dict[str, List[float]], Dict[str, List[float]]]:
    """Extract training and validation metrics from multiple log files, handling resumed training."""
    train_metrics = {
        'loss': [],
        'accuracy': []
    }
    val_metrics = {
        'loss': [],
        'accuracy': []
    }
    
    print(f"Processing {len(log_files)} log files...")
    
    for i, log_file in enumerate(log_files):
        print(f"Reading log file {i+1}/{len(log_files)}: {log_file.name}")
        
        current_epoch_metrics = {
            'loss': None,
            'accuracy': None
        }
        
        # Track epochs found in this file to detect duplicates
        epochs_in_this_file = set()
        
        try:
            with open(log_file, 'r') as f:
                for line in f:
                    # Check for end of epoch
                    if 'Saving checkpoint at' in line:
                        # Extract epoch number from checkpoint line
                        epoch_match = re.search(r'Saving checkpoint at (\d+) epochs?', line)
                        if epoch_match:
                            epoch_num = int(epoch_match.group(1))
                            
                            # Check if this epoch was already seen in a previous file
                            if epoch_num in epochs_in_this_file:
                                print(f"    Warning: Duplicate epoch {epoch_num} detected in same file")
                                continue
                            
                            epochs_in_this_file.add(epoch_num)
                            
                            # Check if this epoch was already processed from a previous log file
                            if len(train_metrics['loss']) >= epoch_num:
                                print(f"    Skipping epoch {epoch_num} - already processed from previous log file")
                                continue
                            
                            # Save the last metrics from this epoch
                            if current_epoch_metrics['loss'] is not None:
                                train_metrics['loss'].append(current_epoch_metrics['loss'])
                                train_metrics['accuracy'].append(current_epoch_metrics['accuracy'])
                                print(f"    Added training metrics for epoch {epoch_num} - Loss: {current_epoch_metrics['loss']:.4f}, "
                                      f"Acc: {current_epoch_metrics['accuracy']:.4f}")
                                # Reset current epoch metrics
                                current_epoch_metrics['loss'] = None
                                current_epoch_metrics['accuracy'] = None
                        continue
                    
                    # Extract training metrics (keep updating until end of epoch)
                    if 'Epoch(train)' in line:
                        loss_match = re.search(r'loss: ([\d.]+)', line)
                        acc_match = re.search(r'top1_acc: ([\d.]+)', line)
                        
                        if loss_match and acc_match:
                            current_epoch_metrics['loss'] = float(loss_match.group(1))
                            current_epoch_metrics['accuracy'] = float(acc_match.group(1))
                    
                    # Extract validation metrics
                    elif 'Epoch(val)' in line and 'acc/top1:' in line:
                        loss_match = re.search(r'loss/loss_cls: ([\d.]+)', line)
                        acc_match = re.search(r'acc/top1: ([\d.]+)', line)
                        
                        if loss_match and acc_match:
                            loss_val = float(loss_match.group(1))
                            acc_val = float(acc_match.group(1))
                            
                            # Only add validation metrics if we haven't exceeded the expected epoch count
                            if len(val_metrics['loss']) < len(train_metrics['loss']) + 1:
                                val_metrics['loss'].append(loss_val)
                                val_metrics['accuracy'].append(acc_val)
                                print(f"    Added validation metrics - Loss: {loss_val:.4f}, Acc: {acc_val:.4f}")
        
        except Exception as e:
            print(f"Error reading log file {log_file}: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"Found {len(train_metrics['loss'])} training epochs and {len(val_metrics['loss'])} validation epochs")
    
    # Validate and clean the metrics
    train_metrics, val_metrics = validate_and_clean_metrics(train_metrics, val_metrics)
    
    return train_metrics, val_metrics

def extract_metrics_from_log(log_file: Path) -> Tuple[Dict[str, List[float]], Dict[str, List[float]]]:
    """Extract training and validation metrics from a single log file (backward compatibility)."""
    return extract_metrics_from_logs([log_file])

def plot_training_curves(fold_metrics: Dict[int, Tuple[Dict, Dict]], output_dir: Path) -> None:
    """Create training and validation curves for each fold."""
    plt.style.use('default')
    
    # Calculate grid dimensions
    n_folds = len(fold_metrics)
    n_cols = 3  # Number of columns in the grid
    n_rows = (n_folds + n_cols - 1) // n_cols  # Ceiling division
    
    # Create figure with subplots for each fold
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    
    # Colors for train and validation
    train_color = '#1f77b4'
    val_color = '#ff7f0e'
    
    # Plot each fold
    for idx, (fold_idx, (train_metrics, val_metrics)) in enumerate(fold_metrics.items()):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row, col]
        
        epochs = range(1, len(train_metrics['loss']) + 1)
        
        # Plot loss
        ln1 = ax.plot(epochs, train_metrics['loss'], '--', color=train_color, label='Train Loss')
        ax.plot(epochs, val_metrics['loss'], '-', color=train_color, alpha=0.5, label='Val Loss')
        
        # Plot accuracy on secondary y-axis
        ax2 = ax.twinx()
        ln2 = ax2.plot(epochs, train_metrics['accuracy'], '--', color=val_color, label='Train Acc')
        ax2.plot(epochs, val_metrics['accuracy'], '-', color=val_color, alpha=0.5, label='Val Acc')
        
        # Set labels and title
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss', color=train_color)
        ax2.set_ylabel('Accuracy', color=val_color)
        ax.set_title(f'Fold {fold_idx}', pad=10)
        
        # Customize ticks
        ax.tick_params(axis='y', labelcolor=train_color)
        ax2.tick_params(axis='y', labelcolor=val_color)
        
        # Add grid
        ax.grid(True, linestyle='--', alpha=0.3)
        
        # Combine legends
        lns = ln1 + ln2
        labs = [l.get_label() for l in lns]
        ax.legend(lns, labs, loc='center right')
    
    # Remove empty subplots
    for idx in range(n_folds, n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        fig.delaxes(axes[row, col])
    
    # Add overall title
    fig.suptitle('Training and Validation Metrics by Fold', fontsize=14, y=1.02)
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(output_dir / 'training_curves_by_fold.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Also create separate files for each fold
    for fold_idx, (train_metrics, val_metrics) in fold_metrics.items():
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
        epochs = range(1, len(train_metrics['loss']) + 1)
        
        # Plot loss
        ax1.plot(epochs, train_metrics['loss'], '--', color=train_color, label='Train')
        ax1.plot(epochs, val_metrics['loss'], '-', color=train_color, label='Validation')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title(f'Fold {fold_idx} - Loss', pad=10)
        ax1.grid(True, linestyle='--', alpha=0.7)
        ax1.legend()
        
        # Plot accuracy
        ax2.plot(epochs, train_metrics['accuracy'], '--', color=val_color, label='Train')
        ax2.plot(epochs, val_metrics['accuracy'], '-', color=val_color, label='Validation')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.set_title(f'Fold {fold_idx} - Accuracy', pad=10)
        ax2.grid(True, linestyle='--', alpha=0.7)
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig(output_dir / f'fold{fold_idx}_curves.png', dpi=300, bbox_inches='tight')
        plt.close()

def plot_boxplots(fold_metrics: Dict[int, Tuple[Dict, Dict]], output_dir: Path) -> None:
    """Create boxplots of metrics across folds."""
    plt.style.use('default')  # Use default style
    
    # Prepare data for boxplots
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    
    for train_metrics, val_metrics in fold_metrics.values():
        train_losses.extend(train_metrics['loss'])
        val_losses.extend(val_metrics['loss'])
        train_accs.extend(train_metrics['accuracy'])
        val_accs.extend(val_metrics['accuracy'])
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    # Plot loss boxplots
    loss_data = [train_losses, val_losses]
    bp1 = ax1.boxplot(loss_data, labels=['Train', 'Validation'], patch_artist=True)
    ax1.set_title('Loss Distribution', fontsize=12, pad=10)
    ax1.set_ylabel('Loss')
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    # Plot accuracy boxplots
    acc_data = [train_accs, val_accs]
    bp2 = ax2.boxplot(acc_data, labels=['Train', 'Validation'], patch_artist=True)
    ax2.set_title('Accuracy Distribution', fontsize=12, pad=10)
    ax2.set_ylabel('Accuracy')
    ax2.grid(True, linestyle='--', alpha=0.7)
    
    # Add colors to the boxplots
    colors = ['#1f77b4', '#ff7f0e']
    for bplot, color in [(bp1, colors[0]), (bp2, colors[1])]:
        for patch in bplot['boxes']:
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'metric_distributions.png', dpi=300)
    plt.close()

def extract_metrics_from_resumed_training(log_files: List[Path]) -> Tuple[Dict[str, List[float]], Dict[str, List[float]]]:
    """Extract metrics by reading log files in reverse chronological order, handling resumed training."""
    
    # Sort log files by timestamp (latest first)
    sorted_log_files = sorted(log_files, key=lambda x: x.name, reverse=True)
    
    print(f"Processing {len(sorted_log_files)} log files in reverse chronological order...")
    
    train_metrics = {
        'loss': [],
        'accuracy': []
    }
    val_metrics = {
        'loss': [],
        'accuracy': []
    }
    
    # Track which epochs we've already found to avoid duplicates
    found_epochs = set()
    
    for i, log_file in enumerate(sorted_log_files):
        print(f"Reading log file {i+1}/{len(sorted_log_files)}: {log_file.name}")
        
        # Check if this file contains resume information
        has_resume_info = False
        resumed_from_epoch = None
        
        # First pass: check for resume information
        try:
            with open(log_file, 'r') as f:
                for line in f:
                    if 'Auto resumed from the latest checkpoint' in line:
                        has_resume_info = True
                        print(f"    Found resume information in {log_file.name}")
                        
                        # Extract the checkpoint epoch if possible
                        checkpoint_match = re.search(r'epoch_(\d+)\.pth', line)
                        if checkpoint_match:
                            resumed_from_epoch = int(checkpoint_match.group(1))
                            print(f"    Resumed from epoch {resumed_from_epoch}")
                        break
        except Exception as e:
            print(f"    Error checking resume info: {e}")
            continue
        
        # Second pass: extract metrics from this file
        current_epoch_metrics = {
            'loss': None,
            'accuracy': None
        }
        
        epochs_in_this_file = []
        
        try:
            with open(log_file, 'r') as f:
                for line in f:
                    # Check for end of epoch
                    if 'Saving checkpoint at' in line:
                        epoch_match = re.search(r'Saving checkpoint at (\d+) epochs?', line)
                        if epoch_match:
                            epoch_num = int(epoch_match.group(1))
                            
                            # Only process epochs we haven't seen before
                            if epoch_num not in found_epochs:
                                epochs_in_this_file.append(epoch_num)
                                found_epochs.add(epoch_num)
                                
                                # Save the last metrics from this epoch
                                if current_epoch_metrics['loss'] is not None:
                                    train_metrics['loss'].append(current_epoch_metrics['loss'])
                                    train_metrics['accuracy'].append(current_epoch_metrics['accuracy'])
                                    print(f"    Added training metrics for epoch {epoch_num} - Loss: {current_epoch_metrics['loss']:.4f}, "
                                          f"Acc: {current_epoch_metrics['accuracy']:.4f}")
                                    # Reset current epoch metrics
                                    current_epoch_metrics['loss'] = None
                                    current_epoch_metrics['accuracy'] = None
                            else:
                                print(f"    Skipping epoch {epoch_num} - already processed")
                        continue
                    
                    # Extract training metrics (keep updating until end of epoch)
                    if 'Epoch(train)' in line:
                        loss_match = re.search(r'loss: ([\d.]+)', line)
                        acc_match = re.search(r'top1_acc: ([\d.]+)', line)
                        
                        if loss_match and acc_match:
                            current_epoch_metrics['loss'] = float(loss_match.group(1))
                            current_epoch_metrics['accuracy'] = float(acc_match.group(1))
                    
                    # Extract validation metrics
                    elif 'Epoch(val)' in line and 'acc/top1:' in line:
                        loss_match = re.search(r'loss/loss_cls: ([\d.]+)', line)
                        acc_match = re.search(r'acc/top1: ([\d.]+)', line)
                        
                        if loss_match and acc_match:
                            loss_val = float(loss_match.group(1))
                            acc_val = float(acc_match.group(1))
                            
                            # Only add validation metrics if we haven't exceeded the expected epoch count
                            if len(val_metrics['loss']) < len(train_metrics['loss']) + 1:
                                val_metrics['loss'].append(loss_val)
                                val_metrics['accuracy'].append(acc_val)
                                print(f"    Added validation metrics - Loss: {loss_val:.4f}, Acc: {acc_val:.4f}")
        
        except Exception as e:
            print(f"    Error reading metrics: {e}")
            continue
        
        print(f"    Found {len(epochs_in_this_file)} new epochs: {sorted(epochs_in_this_file)}")
        
        # If we found epoch 1, we have the complete training history
        if 1 in epochs_in_this_file:
            print(f"    Found epoch 1 in {log_file.name} - training history complete")
            break
        
        # If this file has resume info but no epoch 1, continue to previous file
        if has_resume_info and not epochs_in_this_file:
            print(f"    File has resume info but no new epochs - continuing to previous file")
            continue
    
    # Sort the metrics by epoch number (they might be out of order due to reverse processing)
    if train_metrics['loss']:
        # Create a mapping of epoch numbers to metrics
        epoch_metrics = {}
        for i in range(len(train_metrics['loss'])):
            epoch_num = i + 1  # Assuming epochs are sequential
            epoch_metrics[epoch_num] = {
                'train_loss': train_metrics['loss'][i],
                'train_acc': train_metrics['accuracy'][i],
                'val_loss': val_metrics['loss'][i] if i < len(val_metrics['loss']) else None,
                'val_acc': val_metrics['accuracy'][i] if i < len(val_metrics['accuracy']) else None
            }
        
        # Reconstruct sorted lists
        sorted_epochs = sorted(epoch_metrics.keys())
        train_metrics['loss'] = [epoch_metrics[ep]['train_loss'] for ep in sorted_epochs]
        train_metrics['accuracy'] = [epoch_metrics[ep]['train_acc'] for ep in sorted_epochs]
        val_metrics['loss'] = [epoch_metrics[ep]['val_loss'] for ep in sorted_epochs if epoch_metrics[ep]['val_loss'] is not None]
        val_metrics['accuracy'] = [epoch_metrics[ep]['val_acc'] for ep in sorted_epochs if epoch_metrics[ep]['val_acc'] is not None]
    
    print(f"Found {len(train_metrics['loss'])} training epochs and {len(val_metrics['loss'])} validation epochs")
    
    # Validate and clean the metrics
    train_metrics, val_metrics = validate_and_clean_metrics(train_metrics, val_metrics)
    
    return train_metrics, val_metrics

def main():
    # Paths
    work_dir = Path('k_fold/work_dirs')  # Relative path
    output_dir = Path('k_fold/training_analysis')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all fold directories
    fold_dirs = [d for d in work_dir.glob('fold*') if d.is_dir()]
    # Sort by fold number (natural sorting) instead of alphabetical
    fold_dirs.sort(key=lambda x: int(x.name.replace('fold', '')))
    
    if not fold_dirs:
        print(f"Error: No fold directories found in {work_dir}")
        return 1
    
    print(f"Found {len(fold_dirs)} fold directories")
    
    # Process each fold's logs
    fold_metrics = {}
    for fold_dir in fold_dirs:
        fold_idx = int(fold_dir.name.replace('fold', ''))
        
        try:
            log_files = find_log_files(fold_dir)
            
            # Detect if this is a resumed training scenario
            if len(log_files) > 1:
                print(f"\nProcessing fold {fold_idx} - RESUMED TRAINING DETECTED")
                print(f"  Found {len(log_files)} log files:")
                for i, log_file in enumerate(log_files):
                    print(f"    {i+1}. {log_file.parent.name}/{log_file.name}")
            else:
                print(f"\nProcessing fold {fold_idx} - SINGLE TRAINING SESSION")
                print(f"  Log file: {log_files[0].parent.name}/{log_files[0].name}")
            
            train_metrics, val_metrics = extract_metrics_from_resumed_training(log_files)
            
            if not train_metrics['loss'] or not val_metrics['loss']:
                print(f"Warning: No metrics found in log files for fold {fold_idx}")
                continue
            
            fold_metrics[fold_idx] = (train_metrics, val_metrics)
            
            # Print summary for this fold
            print(f"  Found {len(train_metrics['loss'])} epochs of training data")
            print(f"  Final training loss: {train_metrics['loss'][-1]:.4f}")
            print(f"  Final validation loss: {val_metrics['loss'][-1]:.4f}")
            print(f"  Best validation accuracy: {max(val_metrics['accuracy']):.4f}")
            
        except FileNotFoundError as e:
            print(f"Warning: {e}")
            continue
        except Exception as e:
            print(f"Error processing fold {fold_idx}: {e}")
            continue
    
    if not fold_metrics:
        print("Error: No metrics could be calculated")
        return 1
    
    # Create visualizations
    print("\nGenerating visualizations...")
    plot_training_curves(fold_metrics, output_dir)
    plot_boxplots(fold_metrics, output_dir)
    print(f"Visualizations saved in: {output_dir}")
    
    # Save comprehensive metrics to CSV
    print("\nSaving metrics to CSV...")
    
    # Create a comprehensive DataFrame with all metrics
    max_epochs = max(len(train_metrics['loss']) for train_metrics, _ in fold_metrics.values())
    
    # Initialize DataFrame with proper column names
    columns = []
    for fold_idx in sorted(fold_metrics.keys()):
        columns.extend([
            f'Fold{fold_idx}_Train_Loss',
            f'Fold{fold_idx}_Train_Accuracy', 
            f'Fold{fold_idx}_Val_Loss',
            f'Fold{fold_idx}_Val_Accuracy'
        ])
    
    metrics_df = pd.DataFrame(index=range(max_epochs), columns=columns)
    
    # Fill in the data
    for fold_idx, (train_metrics, val_metrics) in fold_metrics.items():
        epochs = len(train_metrics['loss'])
        
        # Training metrics
        metrics_df[f'Fold{fold_idx}_Train_Loss'].iloc[:epochs] = train_metrics['loss']
        metrics_df[f'Fold{fold_idx}_Train_Accuracy'].iloc[:epochs] = train_metrics['accuracy']
        
        # Validation metrics
        metrics_df[f'Fold{fold_idx}_Val_Loss'].iloc[:epochs] = val_metrics['loss']
        metrics_df[f'Fold{fold_idx}_Val_Accuracy'].iloc[:epochs] = val_metrics['accuracy']
    
    # Add epoch numbers
    metrics_df.insert(0, 'Epoch', range(1, max_epochs + 1))
    
    # Save to CSV
    csv_path = output_dir / 'comprehensive_training_metrics.csv'
    metrics_df.to_csv(csv_path, index=False)
    print(f"Comprehensive metrics saved to: {csv_path}")
    
    # Also save a summary statistics file
    summary_data = []
    for fold_idx, (train_metrics, val_metrics) in fold_metrics.items():
        summary_data.append({
            'Fold': fold_idx,
            'Total_Epochs': len(train_metrics['loss']),
            'Final_Train_Loss': train_metrics['loss'][-1],
            'Final_Val_Loss': val_metrics['loss'][-1],
            'Best_Val_Accuracy': max(val_metrics['accuracy']),
            'Final_Train_Accuracy': train_metrics['accuracy'][-1],
            'Final_Val_Accuracy': val_metrics['accuracy'][-1],
            'Min_Train_Loss': min(train_metrics['loss']),
            'Min_Val_Loss': min(val_metrics['loss'])
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_path = output_dir / 'training_summary.csv'
    summary_df.to_csv(summary_path, index=False)
    print(f"Training summary saved to: {summary_path}")
    
    return 0

if __name__ == '__main__':
    exit(main()) 