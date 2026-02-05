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
            return {'loss': [], 'accuracy': []}, {'loss': [], 'accuracy': []}
    
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
        return {'loss': [], 'accuracy': []}, {'loss': [], 'accuracy': []}
    
    return train_clean, val_clean

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
        
        try:
            with open(log_file, 'r') as f:
                for line in f:
                    # Check for end of epoch
                    if 'Saving checkpoint at' in line:
                        # Save the last metrics from this epoch
                        if current_epoch_metrics['loss'] is not None:
                            train_metrics['loss'].append(current_epoch_metrics['loss'])
                            train_metrics['accuracy'].append(current_epoch_metrics['accuracy'])
                            print(f"  Added training metrics - Loss: {current_epoch_metrics['loss']:.4f}, "
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
                        loss_match = re.search(r'(?:loss/loss_cls|val/loss_cls): ([\d.]+)', line)
                        acc_match = re.search(r'acc/top1: ([\d.]+)', line)
                        
                        if loss_match and acc_match:
                            loss_val = float(loss_match.group(1))
                            acc_val = float(acc_match.group(1))
                            val_metrics['loss'].append(loss_val)
                            val_metrics['accuracy'].append(acc_val)
                            print(f"  Added validation metrics - Loss: {loss_val:.4f}, Acc: {acc_val:.4f}")
        
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

def plot_training_curves(training_metrics: Dict, output_dir: Path, mode: str = 'kfold') -> None:
    """Create training and validation curves for each training run."""
    plt.style.use('default')
    
    # Calculate grid dimensions
    n_runs = len(training_metrics)
    n_cols = 3  # Number of columns in the grid
    n_rows = (n_runs + n_cols - 1) // n_cols  # Ceiling division
    
    # Create figure with subplots for each training run
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    
    # Colors for train and validation
    train_color = '#1f77b4'
    val_color = '#ff7f0e'
    
    # Plot each training run
    for idx, (dir_idx, (train_metrics, val_metrics)) in enumerate(training_metrics.items()):
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
        if mode == 'kfold':
            ax.set_title(f'Fold {dir_idx}', pad=10)
        else:
            ax.set_title(f'Cross-Site Test {dir_idx}', pad=10)
        
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
    for idx in range(n_runs, n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        fig.delaxes(axes[row, col])
    
    # Add overall title
    if mode == 'kfold':
        fig.suptitle('Training and Validation Metrics by Fold', fontsize=14, y=1.02)
    else:
        fig.suptitle('Training and Validation Metrics by Cross-Site Test', fontsize=14, y=1.02)
    
    # Adjust layout and save
    plt.tight_layout()
    if mode == 'kfold':
        filename = 'training_curves_by_fold.png'
    else:
        filename = 'training_curves_by_cross_site.png'
    plt.savefig(output_dir / filename, dpi=300, bbox_inches='tight')
    plt.close()

    # Also create separate files for each training run
    for dir_idx, (train_metrics, val_metrics) in training_metrics.items():
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
        epochs = range(1, len(train_metrics['loss']) + 1)
        
        # Plot loss
        ax1.plot(epochs, train_metrics['loss'], '--', color=train_color, label='Train')
        ax1.plot(epochs, val_metrics['loss'], '-', color=train_color, label='Validation')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.grid(True, linestyle='--', alpha=0.7)
        ax1.legend()
        
        # Plot accuracy
        ax2.plot(epochs, train_metrics['accuracy'], '--', color=val_color, label='Train')
        ax2.plot(epochs, val_metrics['accuracy'], '-', color=val_color, label='Validation')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.grid(True, linestyle='--', alpha=0.7)
        ax2.legend()
        
        if mode == 'kfold':
            ax1.set_title(f'Fold {dir_idx} - Loss', pad=10)
            ax2.set_title(f'Fold {dir_idx} - Accuracy', pad=10)
            filename = f'fold{dir_idx}_curves.png'
        else:
            ax1.set_title(f'Cross-Site Test {dir_idx} - Loss', pad=10)
            ax2.set_title(f'Cross-Site Test {dir_idx} - Accuracy', pad=10)
            filename = f'cross_site_test_{dir_idx}_curves.png'
        
        plt.tight_layout()
        plt.savefig(output_dir / filename, dpi=300, bbox_inches='tight')
        plt.close()

def plot_boxplots(training_metrics: Dict, output_dir: Path, mode: str = 'kfold') -> None:
    """Create boxplots of metrics across folds."""
    plt.style.use('default')  # Use default style
    
    # Prepare data for boxplots
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    
    for train_metrics, val_metrics in training_metrics.values():
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

def main():
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run k-fold or cross-site training analysis')
    parser.add_argument('--mode', type=str, choices=['kfold', 'cross_site'], default='kfold',
                        help='Analysis mode: kfold or cross_site (default: kfold)')
    args = parser.parse_args()
    
    # Paths
    work_dir = Path('k_fold/work_dirs')  # Relative path
    output_dir = Path('k_fold/training_analysis')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find training directories based on mode
    if args.mode == 'kfold':
        training_dirs = sorted([d for d in work_dir.glob('fold*') if d.is_dir()])
        dir_pattern = "fold directories"
    else:  # cross_site mode
        training_dirs = sorted([d for d in work_dir.glob('cross_site_test_*') if d.is_dir()])
        dir_pattern = "cross-site training directories"
    
    if not training_dirs:
        print(f"Error: No {dir_pattern} found in {work_dir}")
        return 1
    
    print(f"Found {len(training_dirs)} {dir_pattern}")
    
    # Process each directory's logs
    training_metrics = {}
    for training_dir in training_dirs:
        if args.mode == 'kfold':
            dir_idx = int(training_dir.name.replace('fold', ''))
            dir_name = f"fold {dir_idx}"
        else:  # cross_site mode
            # Extract site name from directory name (e.g., "cross_site_test_ucla" -> "ucla")
            if 'cross_site_test_' in training_dir.name:
                site_name = training_dir.name.split('cross_site_test_')[1]
                dir_idx = site_name
                dir_name = f"cross-site test {site_name}"
            else:
                dir_idx = training_dir.name
                dir_name = f"cross-site test {training_dir.name}"
        
        try:
            log_files = find_log_files(training_dir)
            
            # Detect if this is a resumed training scenario
            if len(log_files) > 1:
                print(f"\nProcessing {dir_name} - RESUMED TRAINING DETECTED")
                print(f"  Found {len(log_files)} log files:")
                for i, log_file in enumerate(log_files):
                    print(f"    {i+1}. {log_file.parent.name}/{log_file.name}")
            else:
                print(f"\nProcessing {dir_name} - SINGLE TRAINING SESSION")
                print(f"  Log file: {log_files[0].parent.name}/{log_files[0].name}")
            
            train_metrics, val_metrics = extract_metrics_from_logs(log_files)
            
            if not train_metrics['loss'] or not val_metrics['loss']:
                print(f"Warning: No metrics found in log files for {dir_name}")
                continue
            
            training_metrics[dir_idx] = (train_metrics, val_metrics)
            
            # Print summary for this training run
            print(f"  Found {len(train_metrics['loss'])} epochs of training data")
            print(f"  Final training loss: {train_metrics['loss'][-1]:.4f}")
            print(f"  Final validation loss: {val_metrics['loss'][-1]:.4f}")
            print(f"  Best validation accuracy: {max(val_metrics['accuracy']):.4f}")
            
        except FileNotFoundError as e:
            print(f"Warning: {e}")
            continue
        except Exception as e:
            print(f"Error processing {dir_name}: {e}")
            continue
    
    if not training_metrics:
        print("Error: No metrics could be calculated")
        return 1
    
    # Create visualizations
    print("\nGenerating visualizations...")
    plot_training_curves(training_metrics, output_dir, args.mode)
    plot_boxplots(training_metrics, output_dir, args.mode)
    print(f"Visualizations saved in: {output_dir}")
    
    # Save comprehensive metrics to CSV
    print("\nSaving metrics to CSV...")
    
    # Create a comprehensive DataFrame with all metrics
    max_epochs = max(len(train_metrics['loss']) for train_metrics, _ in training_metrics.values())
    
    # Initialize DataFrame with proper column names
    columns = []
    for dir_idx in sorted(training_metrics.keys()):
        if args.mode == 'kfold':
            prefix = f'Fold{dir_idx}'
        else:
            prefix = f'CrossSite_{dir_idx}'
        columns.extend([
            f'{prefix}_Train_Loss',
            f'{prefix}_Train_Accuracy', 
            f'{prefix}_Val_Loss',
            f'{prefix}_Val_Accuracy'
        ])
    
    metrics_df = pd.DataFrame(index=range(max_epochs), columns=columns)
    
    # Fill in the data
    for dir_idx, (train_metrics, val_metrics) in training_metrics.items():
        epochs = len(train_metrics['loss'])
        
        if args.mode == 'kfold':
            prefix = f'Fold{dir_idx}'
        else:
            prefix = f'CrossSite_{dir_idx}'
        
        # Training metrics
        metrics_df[f'{prefix}_Train_Loss'].iloc[:epochs] = train_metrics['loss']
        metrics_df[f'{prefix}_Train_Accuracy'].iloc[:epochs] = train_metrics['accuracy']
        
        # Validation metrics
        metrics_df[f'{prefix}_Val_Loss'].iloc[:epochs] = val_metrics['loss']
        metrics_df[f'{prefix}_Val_Accuracy'].iloc[:epochs] = val_metrics['accuracy']
    
    # Add epoch numbers
    metrics_df.insert(0, 'Epoch', range(1, max_epochs + 1))
    
    # Save to CSV
    csv_path = output_dir / 'comprehensive_training_metrics.csv'
    metrics_df.to_csv(csv_path, index=False)
    print(f"Comprehensive metrics saved to: {csv_path}")
    
    # Also save a summary statistics file
    summary_data = []
    for dir_idx, (train_metrics, val_metrics) in training_metrics.items():
        summary_data.append({
            'Fold': dir_idx,
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