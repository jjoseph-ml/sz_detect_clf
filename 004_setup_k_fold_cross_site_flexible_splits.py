#!/usr/bin/env python3
"""
Unified script for creating k-fold pickle files with flexible train/val/test split configurations.

PREREQUISITE: 004_setup_k_fold.py must be run prior for each individual site (BCM and UCLA) 
before running this script. This script expects the site-specific fold files to already exist.

This script combines functionality from:
- 004_setup_k_fold_combined_from_sites.py (combines all splits from both sites)
- 004_setup_k_fold_combined_from_sites_test_1.py (site-specific test splits)

Supports all 27 combinations of train/val/test splits:
- train: bcm, ucla, combined
- val: bcm, ucla, combined  
- test: bcm, ucla, combined

Usage:
    python 004_setup_k_fold_cross_site_flexible_splits.py --train combined --val combined --test bcm
    python 004_setup_k_fold_cross_site_flexible_splits.py --train bcm --val bcm --test ucla
    python 004_setup_k_fold_cross_site_flexible_splits.py --train ucla --val ucla --test bcm --force
"""

import os
import pickle
import glob
import argparse
from pathlib import Path
from collections import defaultdict

def validate_site_fold_counts(bcm_dir, ucla_dir):
    """
    Validate that both sites have the same number of fold files
    
    Args:
        bcm_dir: Path to BCM data directory
        ucla_dir: Path to UCLA data directory
        
    Returns:
        tuple: (bcm_folds, ucla_folds, max_folds) or (None, None, None) if validation fails
    """
    print("Validating fold counts...")
    
    # Find all fold files in each directory
    bcm_pattern = os.path.join(bcm_dir, "bcm_master_annotation_fold*.pkl")
    ucla_pattern = os.path.join(ucla_dir, "ucla_master_annotation_fold*.pkl")
    
    bcm_files = sorted(glob.glob(bcm_pattern))
    ucla_files = sorted(glob.glob(ucla_pattern))
    
    print(f"BCM fold files found: {len(bcm_files)}")
    print(f"UCLA fold files found: {len(ucla_files)}")
    
    if len(bcm_files) != len(ucla_files):
        print(f"ERROR: Mismatch in fold counts!")
        print(f"   BCM: {len(bcm_files)} folds")
        print(f"   UCLA: {len(ucla_files)} folds")
        return None, None, None
    
    # Extract fold numbers and validate they match
    bcm_folds = []
    ucla_folds = []
    
    for bcm_file in bcm_files:
        filename = os.path.basename(bcm_file)
        fold_num = int(filename.split('fold')[1].split('.')[0])
        bcm_folds.append(fold_num)
    
    for ucla_file in ucla_files:
        filename = os.path.basename(ucla_file)
        fold_num = int(filename.split('fold')[1].split('.')[0])
        ucla_folds.append(fold_num)
    
    bcm_folds.sort()
    ucla_folds.sort()
    
    if bcm_folds != ucla_folds:
        print(f"ERROR: Fold numbers don't match!")
        print(f"   BCM folds: {bcm_folds}")
        print(f"   UCLA folds: {ucla_folds}")
        return None, None, None
    
    print(f"SUCCESS: Both sites have {len(bcm_files)} folds with matching numbers: {bcm_folds}")
    return bcm_files, ucla_files, len(bcm_files)

def load_site_data(filepath):
    """
    Load pickle data from a site file
    
    Args:
        filepath: Path to pickle file
        
    Returns:
        dict: Loaded pickle data or None if error
    """
    try:
        with open(filepath, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        print(f"ERROR: Error loading {filepath}: {e}")
        return None

def analyze_patient_video_distribution(data, site_name, fold_num):
    """
    Analyze patient and video distribution in a dataset
    
    Args:
        data: Pickle data structure
        site_name: Name of the site (BCM/UCLA/COMBINED)
        fold_num: Fold number
        
    Returns:
        dict: Analysis results
    """
    analysis = {
        'site': site_name,
        'fold': fold_num,
        'patients': set(),
        'videos': set(),
        'patient_clip_counts': defaultdict(int),
        'video_clip_counts': defaultdict(int),
        'split_stats': {},
        'fold_patients': {}  # New: track patients per fold
    }
    
    # Analyze each split
    for split_name in ['xsub_train', 'xsub_val', 'xsub_test']:
        clips = data['split'][split_name]
        split_patients = set()
        split_videos = set()
        
        for clip_name in clips:
            # Extract patient and video from clip name
            # Format: "patient_video_Seg_X" or "sXXXX_sXXXXszXX_Seg_X"
            parts = clip_name.split('_')
            if len(parts) >= 2:
                patient = parts[0]
                video = f"{parts[0]}_{parts[1]}"
                
                split_patients.add(patient)
                split_videos.add(video)
                analysis['patients'].add(patient)
                analysis['videos'].add(video)
                analysis['patient_clip_counts'][patient] += 1
                analysis['video_clip_counts'][video] += 1
        
        analysis['split_stats'][split_name] = {
            'clips': len(clips),
            'patients': len(split_patients),
            'videos': len(split_videos),
            'patient_list': sorted(split_patients),
            'video_list': sorted(split_videos)
        }
        
        # Store fold-specific patient information
        analysis['fold_patients'][split_name] = sorted(split_patients)
    
    return analysis

def print_foldwise_patient_analysis(all_fold_analyses, train_site, val_site, test_site):
    """
    Print comprehensive fold-wise patient analysis
    
    Args:
        all_fold_analyses: List of analysis results for all folds
        train_site: Training site configuration
        val_site: Validation site configuration  
        test_site: Test site configuration
    """
    print("\n" + "=" * 100)
    print("COMPREHENSIVE FOLD-WISE PATIENT ANALYSIS")
    print("=" * 100)
    print(f"Configuration: train={train_site}, val={val_site}, test={test_site}")
    print("=" * 100)
    
    # Collect all patients across folds for each split
    all_train_patients = set()
    all_val_patients = set()
    all_test_patients = set()
    
    fold_summary = []
    
    for fold_analysis in all_fold_analyses:
        fold_num = fold_analysis['fold']
        
        train_patients = set(fold_analysis['fold_patients']['xsub_train'])
        val_patients = set(fold_analysis['fold_patients']['xsub_val'])
        test_patients = set(fold_analysis['fold_patients']['xsub_test'])
        
        all_train_patients.update(train_patients)
        all_val_patients.update(val_patients)
        all_test_patients.update(test_patients)
        
        fold_summary.append({
            'fold': fold_num,
            'train_patients': train_patients,
            'val_patients': val_patients,
            'test_patients': test_patients,
            'train_count': len(train_patients),
            'val_count': len(val_patients),
            'test_count': len(test_patients)
        })
    
    # Print fold-by-fold breakdown
    print(f"\nFOLD-BY-FOLD PATIENT BREAKDOWN:")
    print("-" * 80)
    for fold_info in fold_summary:
        print(f"\nFold {fold_info['fold']}:")
        print(f"  Train patients ({fold_info['train_count']}): {sorted(fold_info['train_patients'])}")
        print(f"  Val patients   ({fold_info['val_count']}): {sorted(fold_info['val_patients'])}")
        print(f"  Test patients  ({fold_info['test_count']}): {sorted(fold_info['test_patients'])}")
        
        # Check for patient overlap within fold
        train_val_overlap = fold_info['train_patients'] & fold_info['val_patients']
        train_test_overlap = fold_info['train_patients'] & fold_info['test_patients']
        val_test_overlap = fold_info['val_patients'] & fold_info['test_patients']
        
        if train_val_overlap:
            print(f"    ⚠️  Train-Val overlap: {sorted(train_val_overlap)}")
        if train_test_overlap:
            print(f"    ⚠️  Train-Test overlap: {sorted(train_test_overlap)}")
        if val_test_overlap:
            print(f"    ⚠️  Val-Test overlap: {sorted(val_test_overlap)}")
    
    # Print overall statistics
    print(f"\nOVERALL PATIENT STATISTICS:")
    print("-" * 50)
    print(f"Total unique train patients across all folds: {len(all_train_patients)}")
    print(f"Total unique val patients across all folds: {len(all_val_patients)}")
    print(f"Total unique test patients across all folds: {len(all_test_patients)}")
    
    # Check for cross-fold overlaps
    train_val_overlap = all_train_patients & all_val_patients
    train_test_overlap = all_train_patients & all_test_patients
    val_test_overlap = all_val_patients & all_test_patients
    
    print(f"\nCROSS-FOLD OVERLAP ANALYSIS:")
    print("-" * 50)
    if train_val_overlap:
        print(f"Train-Val overlap: {len(train_val_overlap)} patients: {sorted(train_val_overlap)}")
    else:
        print("Train-Val overlap: None ✅")
        
    if train_test_overlap:
        print(f"Train-Test overlap: {len(train_test_overlap)} patients: {sorted(train_test_overlap)}")
    else:
        print("Train-Test overlap: None ✅")
        
    if val_test_overlap:
        print(f"Val-Test overlap: {len(val_test_overlap)} patients: {sorted(val_test_overlap)}")
    else:
        print("Val-Test overlap: None ✅")
    
    # Patient distribution analysis
    print(f"\nPATIENT DISTRIBUTION ANALYSIS:")
    print("-" * 50)
    
    # Count how many folds each patient appears in
    patient_fold_counts = defaultdict(int)
    for fold_info in fold_summary:
        for patient in fold_info['train_patients']:
            patient_fold_counts[f"{patient}_train"] += 1
        for patient in fold_info['val_patients']:
            patient_fold_counts[f"{patient}_val"] += 1
        for patient in fold_info['test_patients']:
            patient_fold_counts[f"{patient}_test"] += 1
    
    # Show patients that appear in multiple folds
    multi_fold_patients = {k: v for k, v in patient_fold_counts.items() if v > 1}
    if multi_fold_patients:
        print(f"Patients appearing in multiple folds:")
        for patient_split, count in sorted(multi_fold_patients.items()):
            print(f"  {patient_split}: {count} folds")
    else:
        print("No patients appear in multiple folds ✅")

def validate_combination(train_site, val_site, test_site):
    """
    Validate the combination using simple logic: train == val is recommended
    
    Args:
        train_site: Training site configuration
        val_site: Validation site configuration
        test_site: Test site configuration
        
    Returns:
        tuple: (is_experimental, warnings)
    """
    warnings = []
    is_experimental = False
    
    # Simple rule: train == val is recommended, train != val is experimental
    if train_site != val_site:
        is_experimental = True
        warnings.append("Validation mismatch: Training and validation use different sites")
        warnings.append("This may lead to unreliable model selection")
    
    return is_experimental, warnings

def print_combination_guidance(train_site, val_site, test_site, is_experimental, warnings):
    """
    Print guidance for the combination
    
    Args:
        train_site: Training site configuration
        val_site: Validation site configuration
        test_site: Test site configuration
        is_experimental: Whether this is an experimental combination
        warnings: List of warnings
    """
    print(f"\nCOMBINATION ANALYSIS:")
    print("-" * 50)
    print(f"Configuration: train={train_site}, val={val_site}, test={test_site}")
    
    if is_experimental:
        print(f"Status: EXPERIMENTAL ⚠️")
        print(f"Warnings:")
        for warning in warnings:
            print(f"  ⚠️  {warning}")
    else:
        print(f"Status: RECOMMENDED ✅")
        print(f"No warnings detected.")

def ask_user_confirmation(train_site, val_site, test_site, is_experimental):
    """
    Ask user for confirmation to proceed with experimental combinations
    
    Args:
        train_site: Training site configuration
        val_site: Validation site configuration
        test_site: Test site configuration
        is_experimental: Whether this is an experimental combination
        
    Returns:
        bool: True if user wants to proceed, False otherwise
    """
    if not is_experimental:
        return True  # No confirmation needed for recommended combinations
    
    print(f"\n" + "=" * 60)
    print("EXPERIMENTAL COMBINATION DETECTED")
    print("=" * 60)
    print(f"You are about to use an experimental combination:")
    print(f"  train={train_site}, val={val_site}, test={test_site}")
    print(f"\nThis combination may lead to unreliable model selection because")
    print(f"training and validation use different sites.")
    print(f"\nDo you want to continue? (y/n): ", end="")
    
    while True:
        response = input().strip().lower()
        if response in ['y', 'yes']:
            print("Proceeding with experimental combination...")
            return True
        elif response in ['n', 'no']:
            print("Operation cancelled.")
            return False
        else:
            print("Please enter 'y' for yes or 'n' for no: ", end="")

def merge_site_data_flexible(bcm_data, ucla_data, fold_num, train_site, val_site, test_site, show_analysis=True):
    """
    Merge data from BCM and UCLA sites with flexible split configuration
    
    Args:
        bcm_data: BCM pickle data
        ucla_data: UCLA pickle data  
        fold_num: Fold number for reporting
        train_site: Training site configuration ('bcm', 'ucla', 'combined')
        val_site: Validation site configuration ('bcm', 'ucla', 'combined')
        test_site: Test site configuration ('bcm', 'ucla', 'combined')
        show_analysis: Whether to show detailed analysis
        
    Returns:
        dict: Merged data structure
    """
    print(f"  Merging fold {fold_num} with train={train_site}, val={val_site}, test={test_site}...")
    
    # Validate data structures
    if not bcm_data or not ucla_data:
        print(f"ERROR: Missing data for fold {fold_num}")
        return None
    
    # Check that both have the expected structure
    required_keys = ['split', 'annotations']
    for key in required_keys:
        if key not in bcm_data or key not in ucla_data:
            print(f"ERROR: Missing '{key}' key in fold {fold_num}")
            return None
    
    # Analyze individual sites if requested
    if show_analysis:
        bcm_analysis = analyze_patient_video_distribution(bcm_data, "BCM", fold_num)
        ucla_analysis = analyze_patient_video_distribution(ucla_data, "UCLA", fold_num)
    
    # Merge split data based on configuration
    merged_split = {}
    
    # Helper function to get split data
    def get_split_data(site_config, split_name):
        if site_config == 'bcm':
            return bcm_data['split'][split_name]
        elif site_config == 'ucla':
            return ucla_data['split'][split_name]
        elif site_config == 'combined':
            return bcm_data['split'][split_name] + ucla_data['split'][split_name]
        else:
            print(f"ERROR: Invalid site configuration '{site_config}'")
            return None
    
    # Configure each split
    for split_name in ['xsub_train', 'xsub_val', 'xsub_test']:
        if split_name == 'xsub_train':
            site_config = train_site
        elif split_name == 'xsub_val':
            site_config = val_site
        else:  # xsub_test
            site_config = test_site
        
        split_data = get_split_data(site_config, split_name)
        if split_data is None:
            return None
        
        merged_split[split_name] = split_data
        
        # Print split information
        if site_config == 'combined':
            bcm_count = len(bcm_data['split'][split_name])
            ucla_count = len(ucla_data['split'][split_name])
            total_count = len(split_data)
            print(f"    {split_name}: Combined (BCM={bcm_count}, UCLA={ucla_count}, Total={total_count})")
        else:
            count = len(split_data)
            print(f"    {split_name}: {site_config.upper()} only ({count} clips)")
    
    # Merge annotations based on which sites are used
    used_sites = set([train_site, val_site, test_site])
    merged_annotations = []
    
    if 'bcm' in used_sites or 'combined' in used_sites:
        merged_annotations.extend(bcm_data['annotations'])
    if 'ucla' in used_sites or 'combined' in used_sites:
        merged_annotations.extend(ucla_data['annotations'])
    
    print(f"    annotations: {len(merged_annotations)} total")
    
    # Create merged dataset
    merged_data = {
        'split': merged_split,
        'annotations': merged_annotations
    }
    
    # Analyze combined data if requested
    if show_analysis:
        combined_analysis = analyze_patient_video_distribution(merged_data, "COMBINED", fold_num)
        return merged_data, bcm_analysis, ucla_analysis, combined_analysis
    
    return merged_data

def save_merged_data(merged_data, output_file):
    """
    Save merged data to pickle file
    
    Args:
        merged_data: Merged data structure
        output_file: Output file path
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        with open(output_file, 'wb') as f:
            pickle.dump(merged_data, f)
        
        print(f"SUCCESS: Saved: {output_file}")
        return True
    except Exception as e:
        print(f"ERROR: Error saving {output_file}: {e}")
        return False

def create_flexible_split_files(bcm_dir, ucla_dir, output_dir, train_site, val_site, test_site):
    """
    Create pickle files with flexible split configuration
    
    Args:
        bcm_dir: Path to BCM data directory
        ucla_dir: Path to UCLA data directory  
        output_dir: Path to output directory for combined files
        train_site: Training site configuration
        val_site: Validation site configuration
        test_site: Test site configuration
        
    Returns:
        list: List of created output files
    """
    print(f"Creating flexible split files...")
    print(f"Configuration: train={train_site}, val={val_site}, test={test_site}")
    print("=" * 60)
    
    # Validate fold counts
    bcm_files, ucla_files, num_folds = validate_site_fold_counts(bcm_dir, ucla_dir)
    if bcm_files is None:
        return []
    
    created_files = []
    all_fold_analyses = []
    
    # Process each fold
    for i in range(num_folds):
        print(f"\nProcessing fold {i}...")
        
        # Load data from both sites
        bcm_data = load_site_data(bcm_files[i])
        ucla_data = load_site_data(ucla_files[i])
        
        # Merge the data with flexible configuration
        result = merge_site_data_flexible(bcm_data, ucla_data, i, train_site, val_site, test_site, show_analysis=True)
        
        if isinstance(result, tuple):
            merged_data, bcm_analysis, ucla_analysis, combined_analysis = result
            all_fold_analyses.append(combined_analysis)
        else:
            merged_data = result
        
        if merged_data is None:
            print(f"ERROR: Failed to merge fold {i}")
            continue
        
        # Save merged data
        output_file = os.path.join(output_dir, f"bcm_master_annotation_fold{i}.pkl")
        if save_merged_data(merged_data, output_file):
            created_files.append(output_file)
    
    # Print comprehensive fold-wise analysis
    if all_fold_analyses:
        print_foldwise_patient_analysis(all_fold_analyses, train_site, val_site, test_site)
    
    return created_files

def print_summary(created_files, bcm_dir, ucla_dir, train_site, val_site, test_site):
    """
    Print summary of the merging process
    
    Args:
        created_files: List of created output files
        bcm_dir: BCM source directory
        ucla_dir: UCLA source directory
        train_site: Training site configuration
        val_site: Validation site configuration
        test_site: Test site configuration
    """
    print("\n" + "=" * 60)
    print("MERGE SUMMARY")
    print("=" * 60)
    
    print(f"Source directories:")
    print(f"  BCM: {bcm_dir}")
    print(f"  UCLA: {ucla_dir}")
    
    print(f"\nSplit configuration:")
    print(f"  Train: {train_site.upper()}")
    print(f"  Val:   {val_site.upper()}")
    print(f"  Test:  {test_site.upper()}")
    
    print(f"\nCreated {len(created_files)} files:")
    for file in created_files:
        print(f"  {file}")
    
    if created_files:
        print(f"\nSUCCESS: Successfully created flexible split pickle files!")
        print(f"   Files are ready for k-fold training with the specified configuration")
    else:
        print(f"\nERROR: No files were created due to errors")

def parse_arguments():
    """
    Parse command line arguments
    
    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="Create k-fold pickle files with flexible train/val/test split configurations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Recommended combinations (no confirmation needed):
  python 004_setup_k_fold_cross_site_flexible_splits.py --train combined --val combined --test bcm
  python 004_setup_k_fold_cross_site_flexible_splits.py --train combined --val combined --test ucla
  python 004_setup_k_fold_cross_site_flexible_splits.py --train bcm --val bcm --test bcm
  python 004_setup_k_fold_cross_site_flexible_splits.py --train bcm --val bcm --test ucla
  python 004_setup_k_fold_cross_site_flexible_splits.py --train ucla --val ucla --test bcm
  
  # Experimental combinations (will ask for confirmation):
  python 004_setup_k_fold_cross_site_flexible_splits.py --train bcm --val ucla --test bcm
  python 004_setup_k_fold_cross_site_flexible_splits.py --train combined --val bcm --test ucla
        """
    )
    
    parser.add_argument(
        '--train',
        type=str,
        choices=['bcm', 'ucla', 'combined'],
        required=True,
        help='Training split configuration (bcm, ucla, or combined)'
    )
    
    parser.add_argument(
        '--val',
        type=str,
        choices=['bcm', 'ucla', 'combined'],
        required=True,
        help='Validation split configuration (bcm, ucla, or combined)'
    )
    
    parser.add_argument(
        '--test',
        type=str,
        choices=['bcm', 'ucla', 'combined'],
        required=True,
        help='Test split configuration (bcm, ucla, or combined)'
    )
    
    parser.add_argument(
        '--bcm_dir',
        type=str,
        default='k_fold/data/bcm',
        help='Path to BCM data directory (default: k_fold/data/bcm)'
    )
    
    parser.add_argument(
        '--ucla_dir',
        type=str,
        default='k_fold/data/ucla',
        help='Path to UCLA data directory (default: k_fold/data/ucla)'
    )
    
    parser.add_argument(
        '--output_dir',
        type=str,
        default='k_fold/data/skeleton',
        help='Path to output directory (default: k_fold/data/skeleton)'
    )
    
    return parser.parse_args()

def main():
    """
    Main function to create flexible split pickle files
    """
    args = parse_arguments()
    
    print("K-Fold Cross-Site Flexible Split Configuration")
    print("=" * 60)
    
    # Configuration
    bcm_dir = args.bcm_dir
    ucla_dir = args.ucla_dir
    output_dir = args.output_dir
    train_site = args.train
    val_site = args.val
    test_site = args.test
    
    print(f"BCM directory: {bcm_dir}")
    print(f"UCLA directory: {ucla_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Configuration: train={train_site}, val={val_site}, test={test_site}")
    
    # Validate input directories exist
    if not os.path.exists(bcm_dir):
        print(f"ERROR: BCM directory not found: {bcm_dir}")
        return
    
    if not os.path.exists(ucla_dir):
        print(f"ERROR: UCLA directory not found: {ucla_dir}")
        return
    
    # Validate combination and check for warnings
    is_experimental, warnings = validate_combination(train_site, val_site, test_site)
    print_combination_guidance(train_site, val_site, test_site, is_experimental, warnings)
    
    # Ask for user confirmation if experimental
    if not ask_user_confirmation(train_site, val_site, test_site, is_experimental):
        return
    
    # Create flexible split files
    created_files = create_flexible_split_files(bcm_dir, ucla_dir, output_dir, train_site, val_site, test_site)
    
    # Print summary
    print_summary(created_files, bcm_dir, ucla_dir, train_site, val_site, test_site)

if __name__ == "__main__":
    main()
