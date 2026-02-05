#!/usr/bin/env python
"""
Script to run the complete k-fold cross validation or cross-site validation pipeline.
This script runs all the necessary steps in sequence:
1. Training
2. Training analysis
3. Testing
4. Prediction analysis
5. Report generation

Usage:
    python 005_run_kfold_pipeline.py --mode kfold      # For k-fold validation
    python 005_run_kfold_pipeline.py --mode cross_site  # For cross-site validation
    python 005_run_kfold_pipeline.py                   # Default: k-fold
"""

import os
import sys
import subprocess
import time
import argparse
from datetime import datetime
from pathlib import Path

def run_script(script_command: str) -> bool:
    """Run a Python script and return True if successful."""
    print(f"\n{'='*80}")
    print(f"Running {script_command}...")
    print(f"{'='*80}")
    
    # Split the command into individual arguments
    cmd_parts = script_command.split()
    
    # Check if the script file exists
    script_path = cmd_parts[0]
    if not os.path.exists(script_path):
        print(f"Error: Script file not found: {script_path}")
        return False
    
    # Try different Python executables
    python_executables = ['python3', 'python', '/usr/bin/python3', '/usr/bin/python']
    
    for python_exe in python_executables:
        try:
            # Create new command with python executable
            full_cmd = [python_exe] + cmd_parts
            result = subprocess.run(full_cmd, check=True)
            print(f"\n{script_command} completed successfully")
            return True
        except FileNotFoundError:
            continue  # Try next Python executable
        except subprocess.CalledProcessError as e:
            print(f"Error running {script_command}: {e}")
            return False
    
    print("Error: No working Python executable found")
    return False

def check_config_files(mode):
    """Check if the required config files exist for the specified mode."""
    config_dir = Path('k_fold/stgcn')
    
    if mode == 'kfold':
        config_files = list(config_dir.glob("stgcnpp_fold*.py"))
        if not config_files:
            print(f"Error: No k-fold configuration files found in {config_dir}")
            print("Please run 004_setup_k_fold.py first to create the necessary files.")
            return False
        print(f"Found {len(config_files)} k-fold configuration files")
    else:  # cross_site
        config_files = list(config_dir.glob("stgcnpp_cross_site_test_*.py"))
        if not config_files:
            print(f"Error: No cross-site configuration files found in {config_dir}")
            print("Please run 004_setup_k_fold_cross_site.py first to create the necessary files.")
            return False
        print(f"Found {len(config_files)} cross-site configuration files:")
        for config in config_files:
            print(f"  - {config.name}")
    
    return True

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run k-fold or cross-site validation pipeline')
    parser.add_argument('--mode', type=str, choices=['kfold', 'cross_site'], default='kfold',
                        help='Pipeline mode: kfold or cross_site (default: kfold)')
    args = parser.parse_args()
    
    # Ensure we're in the correct directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    # Check if required config files exist
    if not check_config_files(args.mode):
        return 1
    
    # Determine which scripts to run based on mode
    if args.mode == 'kfold':
        scripts = [
            '005A_run_kfold_training.py --folds all',
            '005b_run_kfold_train_analysis.py',
            '005c_run_kfold_testing_5090.py',
            '005d_run_fold_prediction_analysis.py',
            '005E_generate_kfold_report.py'
        ]
        pipeline_name = "k-fold validation"
    else:  # cross_site
        scripts = [
            #'005A_run_kfold_training.py --folds all --mode cross_site',
            '005b_run_kfold_train_analysis.py --mode cross_site',
            '005c_run_kfold_testing_5090.py --mode cross_site',
            '005d_run_fold_prediction_analysis.py --mode cross_site',
            '005E_generate_kfold_report.py --mode cross_site'
        ]
        pipeline_name = "cross-site validation"
    
    # Start time
    start_time = datetime.now()
    print(f"Starting {pipeline_name} pipeline at {start_time}")
    
    # Run each script in sequence
    for i, script_name in enumerate(scripts):
        if not run_script(script_name):
            print(f"\n{pipeline_name} pipeline failed at {script_name}")
            return 1
        
        # Add delay between scripts (except after the last one)
        if i < len(scripts) - 1:
            print("\nWaiting 5 seconds before next script...")
            time.sleep(5)
    
    # End time
    end_time = datetime.now()
    total_duration = end_time - start_time
    print(f"\n{pipeline_name} pipeline completed at {end_time}")
    print(f"Total duration: {total_duration}")
    
    return 0

if __name__ == '__main__':
    sys.exit(main()) 