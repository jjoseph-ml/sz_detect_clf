from pathlib import Path
from typing import Tuple, Dict, List
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def find_log_file(fold_dir: Path) -> Path:
    """Find the most recent log file in the fold directory."""
    # Find all timestamped directories (format: YYYYMMDD_HHMMSS)
    timestamp_dirs = [d for d in fold_dir.glob('*_*') if d.is_dir() and len(d.name) == 15]
    
    if not timestamp_dirs:
        raise FileNotFoundError(f"No timestamped directories found in {fold_dir}")
    
    # Get the latest directory
    latest_dir = max(timestamp_dirs, key=lambda x: x.name)
    
    # Look for .log file in the latest directory
    log_file = latest_dir / f"{latest_dir.name}.log"
    
    if not log_file.exists():
        raise FileNotFoundError(f"No log file found in {latest_dir}")
    
    print(f"Found log file: {log_file}")
    return log_file

def extract_metrics_from_log(log_file: Path) -> Tuple[Dict[str, List[float]], Dict[str, List[float]]]:
    """Extract training and validation metrics from a log file."""
    train_metrics = {
        'loss': [],
        'accuracy': []
    }
    val_metrics = {
        'loss': [],
        'accuracy': []
    }
    
    current_epoch_metrics = {
        'loss': None,
        'accuracy': None
    }
    
    print("Reading log file...")
    try:
        with open(log_file, 'r') as f:
            for line in f:
                # Check for end of epoch
                if 'Saving checkpoint at' in line:
                    # Save the last metrics from this epoch
                    if current_epoch_metrics['loss'] is not None:
                        train_metrics['loss'].append(current_epoch_metrics['loss'])
                        train_metrics['accuracy'].append(current_epoch_metrics['accuracy'])
                        print(f"Added training metrics - Loss: {current_epoch_metrics['loss']:.4f}, "
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
                        val_metrics['loss'].append(loss_val)
                        val_metrics['accuracy'].append(acc_val)
                        print(f"Added validation metrics - Loss: {loss_val:.4f}, Acc: {acc_val:.4f}")
    
    except Exception as e:
        print(f"Error reading log file {log_file}: {e}")
        import traceback
        traceback.print_exc()
    
    print(f"Found {len(train_metrics['loss'])} training epochs and {len(val_metrics['loss'])} validation epochs")
    
    if len(train_metrics['loss']) != len(val_metrics['loss']):
        print("Warning: Number of training and validation epochs don't match!")
    
    return train_metrics, val_metrics

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

def main():
    # Paths
    work_dir = Path('k_fold/work_dirs')  # Relative path
    output_dir = Path('k_fold/training_analysis')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all fold directories
    fold_dirs = sorted([d for d in work_dir.glob('fold*') if d.is_dir()])
    
    if not fold_dirs:
        print(f"Error: No fold directories found in {work_dir}")
        return 1
    
    print(f"Found {len(fold_dirs)} fold directories")
    
    # Process each fold's logs
    fold_metrics = {}
    for fold_dir in fold_dirs:
        fold_idx = int(fold_dir.name.replace('fold', ''))
        
        try:
            log_file = find_log_file(fold_dir)
            print(f"\nProcessing fold {fold_idx} logs from {log_file.name}...")
            
            train_metrics, val_metrics = extract_metrics_from_log(log_file)
            
            if not train_metrics['loss'] or not val_metrics['loss']:
                print(f"Warning: No metrics found in log file for fold {fold_idx}")
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
    
    # Save metrics to CSV
    metrics_df = pd.DataFrame({
        f'Fold {fold_idx} Train Loss': train_metrics['loss']
        for fold_idx, (train_metrics, _) in fold_metrics.items()
    })
    metrics_df.to_csv(output_dir / 'training_metrics.csv')
    print(f"Training metrics saved to: {output_dir / 'training_metrics.csv'}")
    
    return 0

if __name__ == '__main__':
    exit(main()) 