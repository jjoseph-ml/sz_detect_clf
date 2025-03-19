#!/usr/bin/env python
"""
Script to generate and visualize saliency maps for the trained models.
Uses trained models from k-fold cross validation to show which input features
are most important for predictions.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import pickle
from pathlib import Path
import mmcv
from mmengine.config import Config
from mmaction.apis import init_recognizer
from mmengine.runner import Runner

def find_best_model(fold_dir: Path) -> Path:
    """
    Find the best model checkpoint in the fold directory.
    
    Args:
        fold_dir: Directory containing model checkpoints
    
    Returns:
        Path to the best model checkpoint
    """
    checkpoints = list(fold_dir.glob('best_acc_top1_epoch*.pth'))
    if not checkpoints:
        raise FileNotFoundError(f"No best model checkpoint found in {fold_dir}")
    
    if len(checkpoints) > 1:
        # If multiple checkpoints exist, get the one with the highest epoch number
        return max(checkpoints, key=lambda x: int(str(x).split('epoch_')[-1].split('.')[0]))
    
    return checkpoints[0]

def load_model_and_data(fold: int) -> tuple:
    """
    Load model and test data for a specific fold.
    
    Args:
        fold: Fold number
    
    Returns:
        tuple: (model, input_data, true_label)
    """
    # Find best model checkpoint
    fold_dir = Path(f'k_fold/work_dirs/fold{fold}')
    try:
        checkpoint = find_best_model(fold_dir)
        print(f"Using checkpoint: {checkpoint}")
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Error for fold {fold}: {e}")
    
    config_file = f'k_fold/stgcn/stgcn_fold{fold}.py'
    
    # Load the config file
    cfg = Config.fromfile(config_file)
    
    # Initialize model using MMAction2 API
    model = init_recognizer(cfg, str(checkpoint), device='cuda')
    model.eval()
    
    # Build the test dataset from config
    from mmengine.registry import DATASETS
    test_dataloader = Runner.build_dataloader(cfg.test_dataloader)
    
    # Get first batch of test data
    for data_batch in test_dataloader:
        break
    
    # Load predictions to get ground truth labels
    pred_file = f'k_fold/test_results/fold{fold}_predictions.pkl'
    with open(pred_file, 'rb') as f:
        predictions = pickle.load(f)
    
    # Get first sample's ground truth label
    true_label = predictions[0]['gt_label'].item()
    
    # Get the input data from the batch and move to GPU
    input_data = data_batch['inputs']
    if isinstance(input_data, (list, tuple)):
        input_data = input_data[0]
    
    # Reshape input data to match STGCN requirements (N, M, T, V, C)
    if isinstance(input_data, torch.Tensor):
        # Take only the first sample (num_clips=10 in test_pipeline)
        input_data = input_data[0:1]  # Keep only first clip
        # Add M dimension: [1, 90, 133, 3] -> [1, 1, 90, 133, 3]
        input_data = input_data.unsqueeze(1)
        # Move to GPU
        input_data = input_data.cuda()
        print(f"Reshaped input data shape: {input_data.shape}")
    
    return model, input_data, true_label

def compute_saliency_map(model, input_tensor):
    """
    Compute saliency map for given input and model.
    
    Args:
        model: MMAction2 model
        input_tensor: Input tensor with shape (N, M, T, V, C)
    
    Returns:
        Saliency map as numpy array
    """
    # Ensure input tensor is on GPU and requires grad
    input_tensor = input_tensor.cuda()
    input_tensor.requires_grad_()
    
    # Forward pass
    with torch.enable_grad():
        output = model(input_tensor, return_loss=False)
        print(f"Raw output type: {type(output)}")
        print(f"Raw output shape/size: {output.shape if isinstance(output, torch.Tensor) else [o.shape for o in output]}")
        
        # Handle different output formats
        if isinstance(output, (list, tuple)):
            output = output[0]  # Take first element if it's a list/tuple
        
        # For output shape [N, M, T, C, V], average over T dimension
        output = output.mean(dim=2)  # Now [N, M, C, V]
        output = output.squeeze(1)   # Remove M dimension -> [N, C, V]
        
        print(f"Processed output shape: {output.shape}")
        print(f"Output values min: {output.min().item()}, max: {output.max().item()}")
        
        # Get predictions across all dimensions
        pred_scores = output.mean(dim=-1)  # Average over keypoints -> [N, C]
        
        # Convert to binary classification (if needed)
        if pred_scores.size(1) != 2:
            print(f"Warning: Model output has {pred_scores.size(1)} classes, converting to binary")
            # Assuming class 0 is negative and rest are positive
            binary_scores = torch.zeros((pred_scores.size(0), 2), device=pred_scores.device)
            binary_scores[:, 0] = pred_scores[:, 0]  # Negative class
            binary_scores[:, 1] = pred_scores[:, 1:].sum(dim=1)  # Positive class
            pred_scores = binary_scores
        
        # Apply softmax to get probabilities
        pred_probs = torch.softmax(pred_scores, dim=1)[0]
        pred_class = pred_probs.argmax().item()
        print(f"Binary prediction scores: {pred_probs}")
        print(f"Predicted class: {pred_class} ({'Positive' if pred_class == 1 else 'Negative'})")
        
        # Get the score for the predicted class
        class_score = pred_scores[0, pred_class]
        print(f"Class score: {class_score.item()}")
        
        # Compute gradients
        class_score.backward()
        
        # Get gradients and convert to saliency map
        saliency = input_tensor.grad.abs()
        # Aggregate across M and C dimensions to get [T, V] saliency map
        saliency = saliency.mean(dim=1).mean(dim=-1)  
        saliency = saliency.squeeze()  # Remove all extra dimensions
        
        # Apply slight smoothing to make the map more visually interpretable
        saliency = torch.nn.functional.avg_pool2d(
            saliency.unsqueeze(0).unsqueeze(0), 
            kernel_size=3, 
            stride=1, 
            padding=1
        ).squeeze()
        
        print(f"Saliency map shape: {saliency.shape}")
        print(f"Saliency values min: {saliency.min().item()}, max: {saliency.max().item()}")
        
    return saliency.detach().cpu().numpy()

def plot_saliency(input_data, saliency_map, true_label, pred_label, save_path, model):
    """
    Plot original input and saliency map side by side.
    
    Args:
        input_data: Original input data (N, M, T, V, C)
        saliency_map: Generated saliency map (T, V)
        true_label: Ground truth label
        pred_label: Predicted label
        save_path: Path to save visualization
        model: The trained model for confidence computation
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot original input - combine all channels to show skeleton better
    # Debug print to check shapes
    print(f"Input data shape before processing: {input_data.shape}")
    input_vis = input_data.squeeze()  # Remove N and M dimensions if present
    print(f"Input data shape after squeeze: {input_vis.shape}")
    input_vis = input_vis.detach().cpu().numpy()  # Convert to numpy
    input_vis = np.sqrt(np.sum(input_vis**2, axis=-1))  # Combine channels using magnitude
    print(f"Final input visualization shape: {input_vis.shape}")
    
    # Normalize input visualization
    input_vis = (input_vis - input_vis.min()) / (input_vis.max() - input_vis.min() + 1e-8)
    
    im1 = ax1.imshow(input_vis, aspect='auto', cmap='viridis')
    ax1.set_title('Input Skeleton Motion')
    ax1.set_xlabel('Keypoints')
    ax1.set_ylabel('Time (frames)')
    plt.colorbar(im1, ax=ax1)
    
    # Add frame numbers
    ax1.set_yticks(np.arange(0, input_vis.shape[0], 10))
    ax1.set_yticklabels([f"{i}" for i in range(0, input_vis.shape[0], 10)])
    
    # Add keypoint numbers
    ax1.set_xticks(np.arange(0, input_vis.shape[1], 10))
    ax1.set_xticklabels([f"{i}" for i in range(0, input_vis.shape[1], 10)])
    
    # Ensure saliency map is 2D and normalize it
    if saliency_map.ndim > 2:
        saliency_map = saliency_map.squeeze()
    print(f"Saliency map shape for plotting: {saliency_map.shape}")
    
    # Normalize saliency map to [0, 1]
    saliency_norm = (saliency_map - saliency_map.min()) / (saliency_map.max() - saliency_map.min() + 1e-8)
    
    # Plot saliency map with increased contrast
    im2 = ax2.imshow(saliency_norm, aspect='auto', cmap='hot')
    ax2.set_title(f'Saliency Map\nTrue: {"Positive" if true_label == 1 else "Negative"}, ' + 
                  f'Pred: {"Positive" if pred_label == 1 else "Negative"}')
    ax2.set_xlabel('Keypoints')
    ax2.set_ylabel('Time (frames)')
    plt.colorbar(im2, ax=ax2)
    
    # Add frame numbers
    ax2.set_yticks(np.arange(0, saliency_map.shape[0], 10))
    ax2.set_yticklabels([f"{i}" for i in range(0, saliency_map.shape[0], 10)])
    
    # Add keypoint numbers
    ax2.set_xticks(np.arange(0, saliency_map.shape[1], 10))
    ax2.set_xticklabels([f"{i}" for i in range(0, saliency_map.shape[1], 10)])
    
    # Add text showing prediction confidence
    with torch.no_grad():
        pred = model(input_data.cuda(), return_loss=False)
        if isinstance(pred, (list, tuple)):
            pred = pred[0]
        pred = pred.mean(dim=2).squeeze(1)  # Average over time, remove M dimension
        pred_scores = pred.mean(dim=-1)  # Average over keypoints
        if pred_scores.size(1) != 2:
            binary_scores = torch.zeros((pred_scores.size(0), 2), device=pred_scores.device)
            binary_scores[:, 0] = pred_scores[:, 0]
            binary_scores[:, 1] = pred_scores[:, 1:].sum(dim=1)
            pred_scores = binary_scores
        probs = torch.softmax(pred_scores, dim=1)[0]
        
    confidence = float(probs[pred_label])
    plt.figtext(0.02, 0.02, f'Prediction confidence: {confidence:.2%}', 
                fontsize=10, bbox=dict(facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def main():
    # Create output directory
    output_dir = Path('k_fold/saliency_maps')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process each fold
    n_folds = 3
    
    for fold in range(n_folds):
        print(f"\nProcessing fold {fold}...")
        
        try:
            # Load model and data
            model, input_data, true_label = load_model_and_data(fold)
            print(f"True label: {true_label}")
            
            # Compute saliency map
            saliency_map = compute_saliency_map(model, input_data)
            
            # Get prediction
            with torch.no_grad():
                pred = model(input_data, return_loss=False)
                if isinstance(pred, (list, tuple)):
                    pred = pred[0]
                
                # Average over time and keypoints dimensions
                pred = pred.mean(dim=2)  # Average over time
                pred = pred.squeeze(1)    # Remove M dimension
                pred_scores = pred.mean(dim=-1)  # Average over keypoints
                
                # Convert to binary classification
                if pred_scores.size(1) > 2:
                    binary_scores = torch.zeros((pred_scores.size(0), 2), device=pred_scores.device)
                    binary_scores[:, 0] = pred_scores[:, 0]  # Negative class
                    binary_scores[:, 1] = pred_scores[:, 1:].sum(dim=1)  # Positive class
                    pred_scores = binary_scores
                
                pred_probs = torch.softmax(pred_scores, dim=1)[0]
                pred_label = pred_probs.argmax().item()
                print(f"Binary prediction probabilities: {pred_probs}")
                print(f"Predicted label: {pred_label}")
            
            # Plot and save
            save_path = output_dir / f'saliency_map_fold_{fold}.png'
            plot_saliency(
                input_data,
                saliency_map,
                true_label,
                pred_label,
                save_path,
                model
            )
            
            print(f"Generated saliency map for fold {fold}")
            
        except Exception as e:
            print(f"Error processing fold {fold}: {e}")
            import traceback
            traceback.print_exc()
            continue

if __name__ == '__main__':
    main() 