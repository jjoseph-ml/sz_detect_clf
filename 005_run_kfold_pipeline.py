#!/usr/bin/env python
"""
Script to run the complete k-fold cross validation pipeline.
This script runs all the necessary steps in sequence:
1. Training
2. Training analysis
3. Testing
4. Prediction analysis
5. Report generation
"""

import os
import sys
import subprocess
import time
from datetime import datetime

def run_script(script_name: str) -> bool:
    """Run a Python script and return True if successful."""
    print(f"\n{'='*80}")
    print(f"Running {script_name}...")
    print(f"{'='*80}")
    
    try:
        result = subprocess.run(['python', script_name], check=True)
        print(f"\n{script_name} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\nError running {script_name}: {e}")
        return False

def main():
    # List of scripts to run in sequence
    scripts = [
        #'005A_run_kfold_training.py',
        '005B_run_kfold_train_analysis.py',
        '005C_run_kfold_testing.py',
        '005D_run_fold_prediction_analysis.py',
        '005E_generate_kfold_report.py'
    ]
    
    # Start time
    start_time = datetime.now()
    print(f"\nStarting k-fold pipeline at {start_time}")
    
    # Run each script in sequence
    for i, script_name in enumerate(scripts):
        if not run_script(script_name):
            print(f"\nPipeline failed at {script_name}")
            return 1
        
        # Add delay between scripts (except after the last one)
        if i < len(scripts) - 1:
            print("\nWaiting 20 seconds before next script...")
            time.sleep(20)
    
    # End time
    end_time = datetime.now()
    total_duration = end_time - start_time
    print(f"\nPipeline completed at {end_time}")
    print(f"Total duration: {total_duration}")
    
    return 0

if __name__ == '__main__':
    sys.exit(main()) 