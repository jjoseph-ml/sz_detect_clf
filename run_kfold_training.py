#!/usr/bin/env python
"""
Script to run training for k-fold cross validation.
This script calls the MMAction2 tools/train.py script with the appropriate config files.
"""

import os
import sys
import subprocess
import argparse
from datetime import datetime
from pathlib import Path
import torch
from typing import List

# Add MMAction2 to Python path if needed
mmaction2_path = os.path.join(os.path.dirname(__file__), 'mmaction2')
if mmaction2_path not in sys.path:
    sys.path.append(mmaction2_path)

# Import and register all mmaction modules
'''try:
    import mmaction
    from mmaction.registry import MODELS
    from mmaction.models import *  # This is important to register all models
    print(f"MMAction2 version: {mmaction.__version__}")
    print("Available models:", list(MODELS.module_dict.keys()))
except ImportError:
    print("Error: MMAction2 not found in Python path")
    sys.exit(1)'''

def check_gpu_availability() -> List[int]:
    """Check GPU availability and return list of available GPU indices."""
    if not torch.cuda.is_available():
        print("Warning: No GPU available - training will use CPU")
        return []
    
    gpu_count = torch.cuda.device_count()
    gpu_indices = list(range(gpu_count))
    print(f"Found {gpu_count} GPU(s):")
    for i in gpu_indices:
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    return gpu_indices

def cleanup_checkpoints(work_dir_base: Path, folds_to_run: List[int]) -> None:
    """Clean up .pth checkpoint files from previous runs for specified folds.
    
    Args:
        work_dir_base: Base directory containing fold subdirectories
        folds_to_run: List of fold indices to clean up
    """
    total_deleted = 0
    for fold_idx in folds_to_run:
        fold_dir = work_dir_base / f"fold{fold_idx}"
        if not fold_dir.exists():
            continue
            
        # Find and delete all .pth files in the fold directory
        checkpoint_files = list(fold_dir.glob("*.pth"))
        for checkpoint in checkpoint_files:
            try:
                checkpoint.unlink()
                total_deleted += 1
                print(f"Deleted checkpoint: {checkpoint}")
            except Exception as e:
                print(f"Error deleting {checkpoint}: {e}")
    
    if total_deleted > 0:
        print(f"\nCleanup complete. Deleted {total_deleted} checkpoint files.")
    else:
        print("\nNo checkpoint files found to clean up.")

def main():
    parser = argparse.ArgumentParser(description='Run k-fold cross-validation training')
    parser.add_argument('--folds', type=str, default='all', 
                      help='Comma-separated list of fold indices to run, or "all"')
    parser.add_argument('--dry-run', action='store_true', 
                      help='Print commands without executing them')
    parser.add_argument('--no-cleanup', action='store_true',
                      help='Skip cleanup of previous checkpoint files')
    args, train_args = parser.parse_known_args()
    
    # Use the direct path to train.py in mmaction2
    train_script = os.path.join(mmaction2_path, 'tools', 'train.py')
    
    # Hardcoded paths
    config_dir = Path('k_fold/stgcn')
    work_dir_base = Path('k_fold/work_dirs')
    
    # Find all fold config files
    config_files = sorted([str(f) for f in config_dir.glob("stgcn_fold*.py")])
    
    if not config_files:
        print(f"Error: No fold configuration files found in {config_dir}")
        print("Please run setup_k_fold.py first to create the necessary files.")
        return 1
    
    print(f"Found {len(config_files)} fold configuration files:")
    for i, config_file in enumerate(config_files):
        print(f"  Fold {i}: {config_file}")
    
    # Determine which folds to run
    if args.folds == 'all':
        folds_to_run = list(range(len(config_files)))
    else:
        try:
            folds_to_run = [int(fold) for fold in args.folds.split(',')]
            # Check for invalid fold indices
            invalid_folds = [f for f in folds_to_run if f < 0 or f >= len(config_files)]
            if invalid_folds:
                print(f"Error: Invalid fold indices: {invalid_folds}")
                print(f"Valid fold indices are 0-{len(config_files)-1}")
                return 1
        except ValueError:
            print(f"Error: Invalid fold format. Use comma-separated integers (e.g., '0,1,2') or 'all'")
            return 1
    
    print(f"\nWill run training for {len(folds_to_run)} folds: {', '.join(map(str, folds_to_run))}")
    
    # Check if tools/train.py exists
    if not os.path.isfile(train_script):
        print(f"Error: Training script '{train_script}' not found")
        return 1
    
    # Check GPU availability
    available_gpus = check_gpu_availability()
    
    # Set environment variables for GPU training
    if available_gpus:
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, available_gpus))
        os.environ['LOCAL_RANK'] = '0'  # Set local rank for single GPU training
    
    # Clean up previous checkpoints unless --no-cleanup is specified
    if not args.no_cleanup and not args.dry_run:
        print("\nCleaning up previous checkpoint files...")
        cleanup_checkpoints(work_dir_base, folds_to_run)
    
    # Run training for each fold
    start_time = datetime.now()
    print(f"Starting training at {start_time}")
    
    for fold_idx in folds_to_run:
        config_file = config_files[fold_idx]
        work_dir = work_dir_base / f"fold{fold_idx}"
        
        # Ensure work directory exists
        os.makedirs(work_dir, exist_ok=True)
        
        print(f"\n{'='*80}")
        print(f"Training fold {fold_idx}")
        print(f"  Config: {config_file}")
        print(f"  Work directory: {work_dir}")
        print(f"{'='*80}")
        
        # Build command with work directory and any additional train.py arguments
        cmd = [
            'python',
            train_script,
            config_file,
            '--work-dir', str(work_dir)
           # '--resume','k_fold/work_dirs/best_pretrained_model.pth'
        ] + train_args
        
        print(f"Running command: {' '.join(cmd)}")
        
        if args.dry_run:
            print("Dry run - command not executed")
            continue
        
        try:
            fold_start_time = datetime.now()
            print(f"Started at: {fold_start_time}")
            
            # Set PYTHONPATH to include mmaction2 directory
            env = os.environ.copy()
            env['PYTHONPATH'] = f"{mmaction2_path}:{env.get('PYTHONPATH', '')}"
            
            # Run the training process with the modified environment
            subprocess.run(cmd, check=True, env=env)
            
            fold_end_time = datetime.now()
            fold_duration = fold_end_time - fold_start_time
            print(f"Fold {fold_idx} completed in {fold_duration}")
            print(f"{'='*80}")
        except subprocess.CalledProcessError as e:
            print(f"Error running training for fold {fold_idx}: {e}")
            print("Continuing with next fold...")
    
    end_time = datetime.now()
    total_duration = end_time - start_time
    print(f"\nTraining completed at {end_time}")
    print(f"Total duration: {total_duration}")
    
    return 0

if __name__ == '__main__':
    sys.exit(main()) 