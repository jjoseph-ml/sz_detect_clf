import torch
import numpy as np

def compute_saliency_map(model, input_tensor):
    """
    Compute saliency map using gradient-based method.
    
    Args:
        model: The trained model
        input_tensor: Input tensor for saliency computation
    
    Returns:
        numpy.ndarray: Saliency map
    """
    input_tensor = input_tensor.cuda()
    input_tensor.requires_grad_()
    
    with torch.enable_grad():
        output = model(input_tensor, return_loss=False)
        print(f"Output shape: {output.shape}")
        
        # Handle different output formats
        if isinstance(output, (list, tuple)):
            output = output[0]
        
        # Standard processing for all mode
        output = output.mean(dim=2)
        output = output.squeeze(1)
        
        pred_scores = output.mean(dim=-1)
        
        # Handle different class configurations
        if pred_scores.size(1) != 2:
            print(f"Warning: Model output has {pred_scores.size(1)} classes, converting to binary")
            binary_scores = torch.zeros((pred_scores.size(0), 2), device=pred_scores.device)
            binary_scores[:, 0] = pred_scores[:, 0]
            binary_scores[:, 1] = pred_scores[:, 1:].sum(dim=1)
            pred_scores = binary_scores
        
        pred_probs = torch.softmax(pred_scores, dim=1)[0]
        pred_class = pred_probs.argmax().item()
        
        class_score = pred_scores[0, pred_class]
        
        class_score.backward()
        
        # Get raw gradients without absolute value
        saliency = input_tensor.grad
        
        # Standard processing for all mode
        saliency = saliency.mean(dim=1).mean(dim=-1)
        
        saliency = saliency.squeeze()
        
    return saliency.detach().cpu().numpy()


def calculate_motion_data(input_vis):
    """
    Calculate motion data (velocity and acceleration) from input keypoint data.
    """
    # Calculate Euclidean distance between consecutive frames for each keypoint
    motion_data = np.sqrt(np.sum(np.diff(input_vis, axis=0)**2, axis=-1))  # Shape: (T-1, V)
    
    # Add a zero frame for the first frame to maintain the same number of frames
    motion_data = np.concatenate(([np.zeros(motion_data.shape[1])], motion_data), axis=0)  # Shape: (T, V)
    
    # Normalize motion data for visualization
    motion_data = (motion_data - motion_data.min()) / (motion_data.max() - motion_data.min() + 1e-8)
    
    # Transpose the motion data to have keypoints on y-axis and time on x-axis
    motion_data = motion_data.T

    return motion_data


def calculate_motion_data_without_outliers(input_vis, outlier_percentile=95):
    """
    Calculate motion data with filtering of extreme movements (top 5%).
    
    """
    # Calculate Euclidean distance between consecutive frames for each keypoint
    motion_data = np.sqrt(np.sum(np.diff(input_vis, axis=0)**2, axis=-1))
    
    # Add a zero frame for the first frame to maintain the same number of frames
    motion_data = np.concatenate(([np.zeros(motion_data.shape[1])], motion_data), axis=0)
    
    # Find the threshold value at the specified percentile
    threshold = np.percentile(motion_data, outlier_percentile)
    
    # Cap values above the threshold
    capped_motion = motion_data.copy()
    capped_motion[capped_motion > threshold] = threshold
    
    # Normalize the capped motion data for visualization
    capped_motion = (capped_motion - capped_motion.min()) / (capped_motion.max() - capped_motion.min() + 1e-8)
    
    # Transpose for visualization
    capped_motion = capped_motion.T
    
    return capped_motion
