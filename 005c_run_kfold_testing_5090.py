#!/usr/bin/env python
"""
Script to run testing for k-fold cross validation models.
This script calls the MMAction2 tools/test.py script with the appropriate config files.
VERSION 5090: Fixed PyTorch 2.6+ checkpoint loading compatibility issue.
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

def check_gpu_availability() -> List[int]:
    """Check GPU availability and return list of available GPU indices."""
    if not torch.cuda.is_available():
        print("Warning: No GPU available - testing will use CPU")
        return []
    
    gpu_count = torch.cuda.device_count()
    gpu_indices = list(range(gpu_count))
    print(f"Found {gpu_count} GPU(s):")
    for i in gpu_indices:
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    return gpu_indices

def find_best_checkpoint(fold_dir: Path) -> Path:
    """Find the best checkpoint file in the fold directory."""
    checkpoint_files = list(fold_dir.glob("best_*.pth"))
    if not checkpoint_files:
        raise FileNotFoundError(f"No best checkpoint found in {fold_dir}")
    
    if len(checkpoint_files) > 1:
        print(f"Warning: Multiple best checkpoints found in {fold_dir}, using {checkpoint_files[0]}")
    
    return checkpoint_files[0]

def fix_checkpoint_compatibility(checkpoint_path: Path) -> Path:
    """
    Fix PyTorch 2.6+ checkpoint compatibility issues by creating a cleaned version.
    This handles the 'weights_only' loading issue without modifying the original checkpoint.
    """
    try:
        # Try to load with weights_only=True first (PyTorch 2.6+ default)
        checkpoint = torch.load(str(checkpoint_path), map_location='cpu', weights_only=True)
        print(f"Checkpoint loaded successfully with weights_only=True")
        return checkpoint_path
    except Exception as e:
        if "weights_only" in str(e) or "UnpicklingError" in str(e):
            print(f"PyTorch compatibility issue detected: {e}")
            print("Attempting to fix by loading with weights_only=False...")
            
            try:
                # Try loading with weights_only=False (less secure but more compatible)
                checkpoint = torch.load(str(checkpoint_path), map_location='cpu', weights_only=False)
                print(f"Checkpoint loaded successfully with weights_only=False")
                
                # Create a cleaned checkpoint with only weights
                cleaned_checkpoint = {}
                for key, value in checkpoint.items():
                    if isinstance(value, torch.Tensor):
                        cleaned_checkpoint[key] = value
                    elif key in ['state_dict', 'model', 'optimizer']:
                        # Keep essential keys that might contain tensors
                        if isinstance(value, dict):
                            cleaned_checkpoint[key] = value
                        else:
                            cleaned_checkpoint[key] = value
                
                # Save cleaned checkpoint to a temporary file
                temp_checkpoint_path = checkpoint_path.parent / f"{checkpoint_path.stem}_cleaned_5090.pth"
                torch.save(cleaned_checkpoint, temp_checkpoint_path)
                print(f"Created cleaned checkpoint: {temp_checkpoint_path}")
                
                return temp_checkpoint_path
                
            except Exception as e2:
                print(f"Failed to load checkpoint even with weights_only=False: {e2}")
                print("Falling back to original checkpoint path...")
                return checkpoint_path
        else:
            print(f"Unexpected error loading checkpoint: {e}")
            return checkpoint_path

def main():
    parser = argparse.ArgumentParser(description='Run k-fold cross-validation or cross-site testing (VERSION 5090)')
    parser.add_argument('--folds', type=str, default='all', 
                      help='Comma-separated list of fold indices to test, or "all"')
    parser.add_argument('--mode', type=str, choices=['kfold', 'cross_site'], default='kfold',
                      help='Testing mode: kfold or cross_site (default: kfold)')
    parser.add_argument('--checkpoint', type=str, default=None,
                      help='Path to specific checkpoint file to use for all folds (overrides automatic checkpoint finding)')
    parser.add_argument('--dry-run', action='store_true', 
                      help='Print commands without executing them')
    parser.add_argument('--fix-checkpoints', action='store_true', default=True,
                      help='Automatically fix PyTorch checkpoint compatibility issues (default: True)')
    args, test_args = parser.parse_known_args()
    
    # Validate checkpoint file if provided
    specific_checkpoint = None
    if args.checkpoint:
        checkpoint_path = Path(args.checkpoint)
        if not checkpoint_path.exists():
            print(f"Error: Specified checkpoint file '{checkpoint_path}' does not exist")
            return 1
        if not checkpoint_path.is_file():
            print(f"Error: Specified checkpoint path '{checkpoint_path}' is not a file")
            return 1
        specific_checkpoint = checkpoint_path
        print(f"Using specific checkpoint for all folds: {specific_checkpoint}")
    else:
        print("No specific checkpoint provided - will use best checkpoint for each fold")
    
    # Use the direct path to test.py in mmaction2
    test_script = os.path.join(mmaction2_path, 'tools', 'test.py')
    
    # Hardcoded paths
    config_dir = Path('k_fold/stgcn')
    work_dir_base = Path('k_fold/work_dirs')
    test_results_dir = Path('k_fold/test_results')
    
    # Create test results directory
    test_results_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine config file pattern based on mode
    if args.mode == 'kfold':
        config_pattern = "stgcnpp_fold*.py"
        setup_script = "004_setup_k_fold.py"
        
        # Find all fold config files with natural sorting
        def natural_sort_key(filename):
            """Extract fold number for natural sorting"""
            if 'fold' in filename.name:
                fold_part = filename.name.split('fold')[1].split('.')[0]
                return int(fold_part)
            return 0
        
        config_files = sorted([f for f in config_dir.glob(config_pattern)], key=natural_sort_key)
        config_files = [str(f) for f in config_files]
        
        if not config_files:
            print(f"Error: No fold configuration files found in {config_dir}")
            print(f"Please run {setup_script} first to create the necessary files.")
            return 1
        
        print(f"Found {len(config_files)} fold configuration files:")
        for i, config_file in enumerate(config_files):
            print(f"  Fold {i}: {config_file}")
            
    else:  # cross_site mode
        config_pattern = "stgcnpp_cross_site_test_*.py"
        setup_script = "004_setup_k_fold_cross_site.py"
        
        config_files = list(config_dir.glob(config_pattern))
        config_files = [str(f) for f in config_files]
        
        if not config_files:
            print(f"Error: No cross-site configuration files found in {config_dir}")
            print(f"Please run {setup_script} first to create the necessary files.")
            return 1
        
        print(f"Found {len(config_files)} cross-site configuration files:")
        for config_file in config_files:
            print(f"  - {config_file}")
    
    # Determine which folds/configs to test
    if args.mode == 'kfold':
        if args.folds == 'all':
            folds_to_test = list(range(len(config_files)))
        else:
            try:
                folds_to_test = [int(fold) for fold in args.folds.split(',')]
                # Check for invalid fold indices
                invalid_folds = [f for f in folds_to_test if f < 0 or f >= len(config_files)]
                if invalid_folds:
                    print(f"Error: Invalid fold indices: {invalid_folds}")
                    print(f"Valid fold indices are 0-{len(config_files)-1}")
                    return 1
            except ValueError:
                print(f"Error: Invalid fold format. Use comma-separated integers (e.g., '0,1,2') or 'all'")
                return 1
    else:  # cross_site mode
        # For cross-site, we always test all configs (there's typically only one)
        folds_to_test = list(range(len(config_files)))
        if args.folds != 'all':
            print("Warning: --folds argument ignored in cross-site mode. Testing all cross-site configs.")
    
    print(f"\nWill run testing for {len(folds_to_test)} folds: {', '.join(map(str, folds_to_test))}")
    
    # Check if tools/test.py exists
    if not os.path.isfile(test_script):
        print(f"Error: Testing script '{test_script}' not found")
        return 1
    
    # Check GPU availability
    available_gpus = check_gpu_availability()
    
    # Set environment variables for GPU testing
    if available_gpus:
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, available_gpus))
    
    # Run testing for each fold
    start_time = datetime.now()
    print(f"Starting testing at {start_time}")
    
    results = []
    for fold_idx in folds_to_test:
        config_file = config_files[fold_idx]
        
        # Determine work directory name based on mode
        if args.mode == 'kfold':
            work_dir = work_dir_base / f"fold{fold_idx}"
        else:  # cross_site mode
            # Extract site name from config filename (e.g., "stgcnpp_cross_site_test_ucla.py" -> "ucla")
            config_name = Path(config_file).stem
            if 'cross_site_test_' in config_name:
                site_name = config_name.split('cross_site_test_')[1]
                work_dir = work_dir_base / f"cross_site_test_{site_name}"
            else:
                work_dir = work_dir_base / f"cross_site_test_{fold_idx}"
        
        # Use specific checkpoint if provided, otherwise find best checkpoint for this fold
        if specific_checkpoint:
            checkpoint_file = specific_checkpoint
        else:
            try:
                checkpoint_file = find_best_checkpoint(work_dir)
            except FileNotFoundError as e:
                print(f"Error: {e}")
                if args.mode == 'kfold':
                    print(f"Skipping fold {fold_idx}")
                else:
                    print(f"Skipping cross-site config {fold_idx}")
                continue
        
        # Fix checkpoint compatibility if enabled
        if args.fix_checkpoints:
            if args.mode == 'kfold':
                print(f"Checking checkpoint compatibility for fold {fold_idx}...")
            else:
                print(f"Checking checkpoint compatibility for cross-site config {fold_idx}...")
            checkpoint_file = fix_checkpoint_compatibility(checkpoint_file)
        
        # Set up output file for this fold/config
        if args.mode == 'kfold':
            output_file = test_results_dir / f"fold{fold_idx}_predictions.pkl"
        else:  # cross_site mode
            config_name = Path(config_file).stem
            if 'cross_site_test_' in config_name:
                site_name = config_name.split('cross_site_test_')[1]
                output_file = test_results_dir / f"cross_site_test_{site_name}_predictions.pkl"
            else:
                output_file = test_results_dir / f"cross_site_test_{fold_idx}_predictions.pkl"
        
        print(f"\n{'='*80}")
        if args.mode == 'kfold':
            print(f"Testing fold {fold_idx}")
        else:
            print(f"Testing cross-site config {fold_idx}")
        print(f"  Config: {config_file}")
        print(f"  Checkpoint: {checkpoint_file}")
        print(f"  Output: {output_file}")
        print(f"{'='*80}")
        
        # Build command
        cmd = [
            'python', test_script,
            config_file,
            str(checkpoint_file),
            '--dump', str(output_file)
        ] + test_args
        
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
            
            # Run the testing process
            result = subprocess.run(cmd, check=True, env=env)
            results.append((fold_idx, result.returncode == 0))
            
            fold_end_time = datetime.now()
            fold_duration = fold_end_time - fold_start_time
            print(f"Fold {fold_idx} completed in {fold_duration}")
            print(f"{'='*80}")
        except subprocess.CalledProcessError as e:
            print(f"Error running testing for fold {fold_idx}: {e}")
            results.append((fold_idx, False))
            print("Continuing with next fold...")
    
    # Print summary
    end_time = datetime.now()
    total_duration = end_time - start_time
    print(f"\nTesting completed at {end_time}")
    print(f"Total duration: {total_duration}")
    
    print("\nResults Summary:")
    print("="*50)
    successful_folds = sum(1 for _, success in results if success)
    print(f"Successfully tested {successful_folds}/{len(results)} folds")
    for fold_idx, success in results:
        status = "Success" if success else "Failed"
        print(f"Fold {fold_idx}: {status}")
    
    return 0

if __name__ == '__main__':
    sys.exit(main())


