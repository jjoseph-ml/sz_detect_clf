#!/usr/bin/env python
"""
Script to analyze predictions from k-fold cross validation testing.
Calculates and reports various performance metrics for each fold and overall averages.
"""

import os
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, 
    confusion_matrix, precision_score, recall_score, roc_curve, auc
)
import matplotlib.pyplot as plt
import seaborn as sns

def load_predictions(file_path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load predictions from a pickle file and extract relevant arrays."""
    with open(file_path, 'rb') as f:
        predictions = pickle.load(f)
    
    y_true = []
    y_pred = []
    y_scores = []
    
    for pred in predictions:
        y_true.append(pred['gt_label'].item())
        y_pred.append(pred['pred_label'].item())
        # Get probability score for positive class (class 1)
        y_scores.append(pred['pred_score'][1].item())
    
    return np.array(y_true), np.array(y_pred), np.array(y_scores)

def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_scores: np.ndarray) -> Dict[str, float]:
    """Calculate various performance metrics."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    # Calculate metrics
    metrics = {
        'Accuracy': accuracy_score(y_true, y_pred),
        'F1 Score': f1_score(y_true, y_pred),
        'AUC-ROC': roc_auc_score(y_true, y_scores),
        'Sensitivity': tp / (tp + fn),  # True Positive Rate / Recall
        'Specificity': tn / (tn + fp),  # True Negative Rate
        'PPV': tp / (tp + fp),  # Positive Predictive Value / Precision
        'NPV': tn / (tn + fn),  # Negative Predictive Value
    }
    
    return metrics

def format_metrics_table(fold_metrics: Dict[int, Dict[str, float]]) -> pd.DataFrame:
    """Format metrics into a pandas DataFrame with fold columns and metric rows."""
    # Create DataFrame with fold columns
    df = pd.DataFrame(fold_metrics).T
    
    # Calculate averages
    averages = df.mean()
    
    # Add a row for averages at the bottom
    df.loc['Average'] = averages
    
    # Format values to 4 decimal places
    df = df.round(4)
    
    # Create string representation with separator line
    result = df.to_string()
    lines = result.split('\n')
    # Insert separator line before the last line (averages)
    separator = '-' * len(lines[0])
    lines.insert(-1, separator)
    
    # Print the formatted table
    print("\nPerformance Metrics by Fold:")
    print("="*80)
    print('\n'.join(lines))
    
    return df

def plot_metrics_comparison(df: pd.DataFrame, output_dir: Path) -> None:
    """Create a bar plot comparing metrics across folds."""
    # Drop the Average row for plotting
    df_plot = df.drop('Average')
    
    # Create figure with larger size
    plt.figure(figsize=(12, 6))
    
    # Create grouped bar plot
    x = np.arange(len(df_plot.index))
    width = 0.1
    metrics = df_plot.columns
    
    for i, metric in enumerate(metrics):
        plt.bar(x + i*width, df_plot[metric], width, label=metric)
    
    plt.xlabel('Fold')
    plt.ylabel('Score')
    plt.title('Performance Metrics Comparison Across Folds')
    plt.xticks(x + width * (len(metrics)-1)/2, df_plot.index)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    # Save plot
    plt.savefig(output_dir / 'metrics_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_roc_curves(fold_predictions: Dict[int, Tuple[np.ndarray, np.ndarray, np.ndarray]], 
                   output_dir: Path) -> None:
    """Create ROC curves for all folds."""
    plt.figure(figsize=(8, 8))
    
    # Plot ROC curve for each fold
    for fold_idx, (y_true, _, y_scores) in fold_predictions.items():
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'Fold {fold_idx} (AUC = {roc_auc:.3f})')
    
    # Plot random classifier line
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves for All Folds')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.tight_layout()
    
    # Save plot
    plt.savefig(output_dir / 'roc_curves.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_confusion_matrices(fold_predictions: Dict[int, Tuple[np.ndarray, np.ndarray, np.ndarray]], 
                          output_dir: Path) -> None:
    """Create heatmap of confusion matrices for all folds."""
    n_folds = len(fold_predictions)
    fig, axes = plt.subplots(1, n_folds, figsize=(5*n_folds, 4))
    if n_folds == 1:
        axes = [axes]
    
    for ax, (fold_idx, (y_true, y_pred, _)) in zip(axes, fold_predictions.items()):
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_title(f'Fold {fold_idx}')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'confusion_matrices.png', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    # Directory containing prediction files
    predictions_dir = Path('k_fold/test_results')
    
    if not predictions_dir.exists():
        print(f"Error: Predictions directory not found: {predictions_dir}")
        return 1
    
    # Find all prediction files with natural sorting
    def natural_sort_key(filename):
        """Extract fold number for natural sorting"""
        if 'fold' in filename.name:
            fold_part = filename.name.split('fold')[1].split('_')[0]
            return int(fold_part)
        return 0
    
    prediction_files = sorted(predictions_dir.glob('fold*_predictions.pkl'), key=natural_sort_key)
    
    if not prediction_files:
        print(f"Error: No prediction files found in {predictions_dir}")
        return 1
    
    print(f"Found {len(prediction_files)} prediction files")
    
    # Store predictions for visualization
    fold_predictions = {}
    
    # Calculate metrics for each fold
    fold_metrics = {}
    for pred_file in prediction_files:
        fold_idx = int(pred_file.stem.split('fold')[1].split('_')[0])
        print(f"\nProcessing fold {fold_idx}...")
        
        try:
            # Load and process predictions
            y_true, y_pred, y_scores = load_predictions(pred_file)
            metrics = calculate_metrics(y_true, y_pred, y_scores)
            fold_metrics[fold_idx] = metrics
            fold_predictions[fold_idx] = (y_true, y_pred, y_scores)
            
            # Print confusion matrix for this fold
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
            print("\nConfusion Matrix:")
            print(f"TN: {tn:4d}  FP: {fp:4d}")
            print(f"FN: {fn:4d}  TP: {tp:4d}")
            
        except Exception as e:
            print(f"Error processing fold {fold_idx}: {e}")
            continue
    
    if not fold_metrics:
        print("Error: No metrics could be calculated")
        return 1
    
    # Create formatted table of results and print it
    results_table = format_metrics_table(fold_metrics)
    
    # Save results to CSV
    output_file = predictions_dir / 'fold_metrics.csv'
    results_table.to_csv(output_file)
    print(f"\nResults saved to: {output_file}")
    
    # Create visualizations
    print("\nGenerating visualizations...")
    plot_metrics_comparison(results_table, predictions_dir)
    plot_roc_curves(fold_predictions, predictions_dir)
    plot_confusion_matrices(fold_predictions, predictions_dir)
    print(f"Visualizations saved in: {predictions_dir}")
    
    return 0

if __name__ == '__main__':
    exit(main()) 