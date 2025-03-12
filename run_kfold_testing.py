#!/usr/bin/env python
"""
Script to run testing for k-fold cross validation models.
This script calls the MMAction2 tools/test.py script with the appropriate config files.
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

def main():
    parser = argparse.ArgumentParser(description='Run k-fold cross-validation testing')
    parser.add_argument('--folds', type=str, default='all', 
                      help='Comma-separated list of fold indices to test, or "all"')
    parser.add_argument('--dry-run', action='store_true', 
                      help='Print commands without executing them')
    args, test_args = parser.parse_known_args()
    
    # Use the direct path to test.py in mmaction2
    test_script = os.path.join(mmaction2_path, 'tools', 'test.py')
    
    # Hardcoded paths
    config_dir = Path('k_fold/stgcn')
    work_dir_base = Path('k_fold/work_dirs')
    test_results_dir = Path('k_fold/test_results')
    
    # Create test results directory
    test_results_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all fold config files
    config_files = sorted([str(f) for f in config_dir.glob("stgcn_fold*.py")])
    
    if not config_files:
        print(f"Error: No fold configuration files found in {config_dir}")
        print("Please run setup_k_fold.py first to create the necessary files.")
        return 1
    
    print(f"Found {len(config_files)} fold configuration files:")
    for i, config_file in enumerate(config_files):
        print(f"  Fold {i}: {config_file}")
    
    # Determine which folds to test
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
        work_dir = work_dir_base / f"fold{fold_idx}"
        
        try:
            checkpoint_file = find_best_checkpoint(work_dir)
        except FileNotFoundError as e:
            print(f"Error: {e}")
            print(f"Skipping fold {fold_idx}")
            continue
        
        # Set up output file for this fold
        output_file = test_results_dir / f"fold{fold_idx}_predictions.pkl"
        
        print(f"\n{'='*80}")
        print(f"Testing fold {fold_idx}")
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