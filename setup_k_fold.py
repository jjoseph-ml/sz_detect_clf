import os
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from collections import defaultdict
from tqdm import tqdm
import shutil
from pathlib import Path
import re

def organize_by_patient(input_folder):
    """
    Organize data by patient ID from pickle files
    
    Args:
        input_folder: Folder containing pickle files with frame data
        
    Returns:
        Dictionary mapping patient IDs to their data
    """
    print("Loading and organizing files by patient...")
    
    # Read all pickle files
    pkl_files = [f for f in os.listdir(input_folder) if f.endswith('.pkl')]
    
    # Create patient-based grouping
    patient_data = defaultdict(lambda: {'frame_dirs': [], 'labels': [], 'files': [], 'videos': set()})
    
    for filename in tqdm(pkl_files, desc="Processing files"):
        patient_id = filename.split('_')[0]
        video_id = filename.split('_')[1]  # Get the video ID part (e.g., 7941EC00)
        
        try:
            with open(os.path.join(input_folder, filename), 'rb') as f:
                pkl_data = pickle.load(f)
                
                # Store data grouped by patient
                patient_data[patient_id]['frame_dirs'].append(pkl_data['frame_dir'])
                patient_data[patient_id]['labels'].append(pkl_data['label'])
                patient_data[patient_id]['files'].append(filename)
                patient_data[patient_id]['videos'].add(video_id)
        except Exception as e:
            print(f"Error loading {filename}: {e}")
    
    return patient_data

def load_all_annotations(input_folder):
    """
    Load all annotation data from pickle files
    
    Args:
        input_folder: Folder containing pickle files
        
    Returns:
        List of all annotations
    """
    pkl_files = [f for f in os.listdir(input_folder) if f.endswith('.pkl')]
    all_annotations = []
    
    for filename in tqdm(pkl_files, desc="Loading annotations"):
        try:
            with open(os.path.join(input_folder, filename), 'rb') as f:
                pkl_data = pickle.load(f)
                all_annotations.append(pkl_data)
        except Exception as e:
            print(f"Error loading {filename}: {e}")
    
    return all_annotations

def calculate_class_distribution(patient_data):
    """
    Calculate class distribution (seizure ratio) for each patient
    
    Args:
        patient_data: Dictionary mapping patient IDs to their data
        
    Returns:
        Dictionary mapping patient IDs to their seizure ratio
    """
    patient_class_dist = {}
    
    for patient, data in patient_data.items():
        labels = data['labels']
        if not labels:
            patient_class_dist[patient] = 0
            continue
            
        class_1_count = sum(1 for label in labels if label == 1)
        class_1_ratio = class_1_count / len(labels)
        patient_class_dist[patient] = class_1_ratio
    
    return patient_class_dist

def balance_split_clips(patient_ids, patient_data, minority_ratio=1.0):
    """
    Balance split clips and return video IDs
    
    Args:
        patient_ids: List of patient IDs for this split
        patient_data: Dictionary mapping patient IDs to their data
        minority_ratio: Target ratio between majority and minority class
        
    Returns:
        List of balanced clips and list of videos in split
    """
    class_0_clips = []
    class_1_clips = []
    videos_in_split = set()
    
    # Collect all clips and videos for this split
    for patient_id in patient_ids:
        if patient_id not in patient_data:
            continue
            
        # Add all unique videos for this patient
        videos_in_split.update(patient_data[patient_id]['videos'])
        
        # Collect clips by class
        for frame_dir, label in zip(patient_data[patient_id]['frame_dirs'], 
                                  patient_data[patient_id]['labels']):
            if label == 0:
                class_0_clips.append(frame_dir)
            else:
                class_1_clips.append(frame_dir)
    
    # Keep all class 1 clips (minority class)
    num_class_1 = len(class_1_clips)
    
    if num_class_1 == 0:
        return class_0_clips, sorted(list(videos_in_split))
        
    # Sample from class 0 to achieve desired ratio
    num_class_0_to_keep = int(num_class_1 / minority_ratio)
    
    if num_class_0_to_keep > len(class_0_clips):
        print(f"Warning: Requested {num_class_0_to_keep} class 0 clips but only {len(class_0_clips)} available")
        sampled_class_0_clips = class_0_clips
    else:
        sampled_indices = np.random.choice(
            range(len(class_0_clips)),
            size=num_class_0_to_keep,
            replace=False
        )
        sampled_class_0_clips = [class_0_clips[i] for i in sampled_indices]
    
    return list(sampled_class_0_clips) + class_1_clips, sorted(list(videos_in_split))

def count_labels(split_dirs, annotations):
    """
    Count labels for a split
    
    Args:
        split_dirs: List of frame directories in this split
        annotations: List of all annotations
        
    Returns:
        Bincount of labels
    """
    labels = []
    for ann in annotations:
        if ann['frame_dir'] in split_dirs:
            labels.append(ann['label'])
    
    if not labels:
        return np.bincount([0, 0])  # Return [1, 1] if empty
        
    return np.bincount(labels, minlength=2)

def create_fold_annotation_files(input_folder, output_dir, k=5, minority_ratio=1.0, patients_per_test=2):
    """
    Create k-fold annotation files with stratified patient splits
    
    Args:
        input_folder: Folder containing pickle files
        output_dir: Base directory for output files
        k: Number of folds
        minority_ratio: Target ratio between majority and minority class
        patients_per_test: Number of patients to include in each test set
        
    Returns:
        List of annotation file paths and total number of patients
    """
    np.random.seed(42)  # For reproducibility
    
    # Load and organize files by patient
    patient_data = organize_by_patient(input_folder)
    patient_ids = list(patient_data.keys())
    
    print(f"\nFound {len(patient_ids)} patients: {', '.join(sorted(patient_ids))}")
    
    # Calculate class distribution for each patient
    patient_class_dist = calculate_class_distribution(patient_data)
    
    print("\nPatient class distributions (seizure ratio):")
    for patient in sorted(patient_ids):
        labels = patient_data[patient]['labels']
        class_0_count = sum(1 for label in labels if label == 0)
        class_1_count = sum(1 for label in labels if label == 1)
        total_clips = len(labels)
        
        print(f"  Patient {patient}:")
        print(f"    Total clips: {total_clips}")
        print(f"    Class 0 (normal): {class_0_count} ({(class_0_count/total_clips)*100:.1f}%)")
        print(f"    Class 1 (seizure): {class_1_count} ({(class_1_count/total_clips)*100:.1f}%)")
        print(f"    Seizure ratio: {patient_class_dist[patient]:.2f}")
    
    # Sort patients by seizure ratio for stratification
    sorted_patients = sorted(patient_ids, key=lambda p: patient_class_dist[p])
    
    # Calculate number of patients for test and validation sets
    total_patients = len(sorted_patients)
    patients_per_val = patients_per_test
    
    if patients_per_test + patients_per_val > total_patients:
        raise ValueError(f"Not enough patients ({total_patients}) for test ({patients_per_test}) "
                       f"and validation ({patients_per_val}) sets")
    
    # Create k different splits of the data
    fold_size = total_patients // k
    
    # Load all annotations once
    all_annotations = load_all_annotations(input_folder)
    
    # Create output directory if it doesn't exist
    annotation_dir = os.path.join(output_dir, 'data/skeleton')
    os.makedirs(annotation_dir, exist_ok=True)
    
    annotation_files = []
    
    # Create k fold configurations
    for fold in range(k):
        # Calculate indices for this fold
        start_idx = fold * fold_size
        end_idx = start_idx + fold_size if fold < k - 1 else total_patients
        
        # Get test patients for this fold
        test_indices = list(range(start_idx, end_idx))
        test_patients = [sorted_patients[i] for i in test_indices]
        
        # Get validation patients (next fold_size patients, wrapping around if necessary)
        val_start = end_idx
        val_end = val_start + patients_per_val
        if val_end > total_patients:
            val_indices = list(range(val_start, total_patients)) + list(range(0, val_end - total_patients))
        else:
            val_indices = list(range(val_start, val_end))
        val_patients = [sorted_patients[i] for i in val_indices]
        
        # Get training patients (remaining patients)
        train_patients = [p for p in sorted_patients if p not in test_patients and p not in val_patients]
        
        # Balance each split
        train_clips, train_videos = balance_split_clips(train_patients, patient_data, minority_ratio)
        val_clips, val_videos = balance_split_clips(val_patients, patient_data, minority_ratio)
        test_clips, test_videos = balance_split_clips(test_patients, patient_data, minority_ratio)
        
        # Count labels for reporting
        train_labels = count_labels(train_clips, all_annotations)
        val_labels = count_labels(val_clips, all_annotations)
        test_labels = count_labels(test_clips, all_annotations)
        
        # Create fold dataset
        fold_dataset = {
            'split': {
                'xsub_train': train_clips,
                'xsub_val': val_clips,
                'xsub_test': test_clips
            },
            'annotations': all_annotations
        }
        
        # Save fold annotation file
        output_file = os.path.join(annotation_dir, f"bcm_master_annotation_fold{fold}.pkl")
        with open(output_file, 'wb') as f:
            pickle.dump(fold_dataset, f)
        
        annotation_files.append(output_file)
            
        print(f"\nFold {fold} annotation file created: {output_file}")
        print(f"  Training patients ({len(train_patients)}): {', '.join(sorted(train_patients))}")
        print(f"    Normal clips: {train_labels[0]}, Seizure clips: {train_labels[1]}")
        print(f"    Ratio (normal:seizure): {train_labels[0]/max(train_labels[1], 1):.2f}:1")
        
        print(f"  Validation patients ({len(val_patients)}): {', '.join(sorted(val_patients))}")
        print(f"    Normal clips: {val_labels[0]}, Seizure clips: {val_labels[1]}")
        print(f"    Ratio (normal:seizure): {val_labels[0]/max(val_labels[1], 1):.2f}:1")
        
        print(f"  Testing patients ({len(test_patients)}): {', '.join(sorted(test_patients))}")
        print(f"    Normal clips: {test_labels[0]}, Seizure clips: {test_labels[1]}")
        print(f"    Ratio (normal:seizure): {test_labels[0]/max(test_labels[1], 1):.2f}:1")
        
    # At the end of the function, return total_patients along with annotation_files
    return annotation_files, total_patients

def create_fold_config_files(base_config_path, output_dir, k=5, epochs=10):
    """
    Create k configuration files for training
    
    Args:
        base_config_path: Path to base configuration file
        output_dir: Base directory for output files
        k: Number of folds
        epochs: Number of epochs for training (default: 10)
        
    Returns:
        List of configuration file paths
    """
    # Ensure config output directory exists
    config_output_dir = os.path.join(output_dir, 'stgcn')
    os.makedirs(config_output_dir, exist_ok=True)
    
    # Ensure work directories exist
    for fold in range(k):
        os.makedirs(os.path.join(output_dir, f"work_dirs/fold{fold}"), exist_ok=True)
    
    # Get the base filename without extension
    base_filename = os.path.basename(base_config_path)
    base_name, _ = os.path.splitext(base_filename)
    
    # If the base config is stgcn_joint_motion.py, we also need to handle stgcn_joint.py
    if base_name == 'stgcn_joint_motion':
        base_joint_path = os.path.join(os.path.dirname(base_config_path), 'stgcn_joint.py')
        
        # Load base joint config file
        with open(base_joint_path, 'r') as f:
            base_joint_content = f.read()
        
        # Create k joint config files
        for fold in range(k):
            fold_joint_content = base_joint_content
            
            # Update annotation file path using regex
            fold_joint_content = re.sub(
                r"ann_file\s*=\s*'[^']*'",
                f"ann_file = '{output_dir}/data/skeleton/bcm_master_annotation_fold{fold}.pkl'",
                fold_joint_content
            )
            
            # Update the number of epochs in train_cfg
            if "max_epochs=" in fold_joint_content:
                fold_joint_content = re.sub(
                    r'max_epochs=\d+',
                    f'max_epochs={epochs}',
                    fold_joint_content
                )
            
            # Update optimizer configuration
            optim_config = """
optim_wrapper = dict(
    optimizer=dict(
        type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0005, nesterov=True))
"""
            # Check if optim_wrapper already exists
            if "optim_wrapper = dict(" not in fold_joint_content:
                # Add it before auto_scale_lr if it exists
                if "auto_scale_lr = dict(" in fold_joint_content:
                    fold_joint_content = fold_joint_content.replace(
                        "auto_scale_lr = dict(",
                        f"{optim_config}\nauto_scale_lr = dict("
                    )
                else:
                    # Otherwise add it at the end
                    fold_joint_content += f"\n{optim_config}"
            
            # Write the fold-specific joint config file
            fold_joint_path = os.path.join(config_output_dir, f'stgcn_joint.py')
            with open(fold_joint_path, 'w') as f:
                f.write(fold_joint_content)
    
    # Load base config file
    with open(base_config_path, 'r') as f:
        base_config_content = f.read()
    
    config_files = []
    
    # Create k config files
    for fold in range(k):
        fold_config_content = base_config_content
        
        # Update annotation file path using regex
        fold_config_content = re.sub(
            r"ann_file\s*=\s*'[^']*'",
            f"ann_file = '{output_dir}/data/skeleton/bcm_master_annotation_fold{fold}.pkl'",
            fold_config_content
        )
        
        # Update the number of epochs in train_cfg
        if "max_epochs=" in fold_config_content:
            # Use regex to replace max_epochs value
            fold_config_content = re.sub(
                r'max_epochs=\d+',
                f'max_epochs={epochs}',
                fold_config_content
            )
        else:
            print(f"Warning: Could not find max_epochs in config file for fold {fold}")
        
        # Remove any existing work_dir setting
        if "work_dir =" in fold_config_content:
            fold_config_content = fold_config_content.replace(
                "work_dir =", 
                "# work_dir ="
            )
        
        # Update optimizer configuration
        optim_config = """
optim_wrapper = dict(
    optimizer=dict(
        type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0005, nesterov=True))
"""
        # Check if optim_wrapper already exists
        if "optim_wrapper = dict(" not in fold_config_content:
            # Add it before auto_scale_lr if it exists
            if "auto_scale_lr = dict(" in fold_config_content:
                fold_config_content = fold_config_content.replace(
                    "auto_scale_lr = dict(",
                    f"{optim_config}\nauto_scale_lr = dict("
                )
            else:
                # Otherwise add it at the end
                fold_config_content += f"\n{optim_config}"
        
        # Update val_evaluator to include both metrics
        if "val_evaluator = [dict(type='AccMetric')]" in fold_config_content:
            fold_config_content = fold_config_content.replace(
                "val_evaluator = [dict(type='AccMetric')]",
                "val_evaluator = [dict(type='AccMetric'), dict(type='LossMetric')]"
            )
        
        # Write the fold-specific config file
        fold_config_path = os.path.join(config_output_dir, f'{base_name}_fold{fold}.py')
        with open(fold_config_path, 'w') as f:
            f.write(fold_config_content)
        
        config_files.append(fold_config_path)
            
        print(f"Created config file for fold {fold}: {fold_config_path}")
    
    return config_files

def setup_k_fold_cross_validation(input_folder, base_config_path, output_dir='k_fold', k=5, epochs=10, patients_per_test=2):
    """
    Setup k-fold cross-validation 
    
    Args:
        input_folder: Folder containing pickle files
        base_config_path: Path to base configuration file
        output_dir: Base directory for all k-fold related files
        k: Number of folds
        epochs: Number of epochs for training (default: 10)
        patients_per_test: Number of patients to include in each test set
        
    Returns:
        Lists of annotation files and config files
    """
    print(f"Preparing {k}-fold cross-validation setup under {output_dir}/...")
    
    # Create the base output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Step 1: Create fold annotation files
    print("\nCreating fold annotation files...")
    annotation_files, total_patients = create_fold_annotation_files(
        input_folder=input_folder,
        output_dir=output_dir,
        k=k,
        minority_ratio=1.0,
        patients_per_test=patients_per_test
    )
    
    # Step 2: Create fold config files
    print("\nCreating fold configuration files...")
    config_files = create_fold_config_files(
        base_config_path=base_config_path,
        output_dir=output_dir,
        k=k,
        epochs=epochs
    )
    
    # Print config files for easy reference
    print("\nCreated the following files:")
    print("Annotation files:")
    for file in annotation_files:
        print(f"  {file}")
    
    print("\nConfiguration files:")
    for file in config_files:
        print(f"  {file}")
    
    print(f"\nSetup complete! All files are organized under the {output_dir}/ directory.")
    print("You can now use run_kfold_training.py to train the models:")
    
    # Add summary section
    print("\nK-FOLD SUMMARY")
    print("==============")
    print(f"Total number of folds: {k}")
    print(f"Patients per set in each fold:")
    print(f"  - Test set: {patients_per_test} patients")
    print(f"  - Validation set: {patients_per_test} patients")
    print(f"  - Training set: {total_patients - (2 * patients_per_test)} patients")
    print(f"\nTotal number of patients: {total_patients}")
    print("Each patient appears:")
    print(f"  - Once in test set")
    print(f"  - Once in validation set")
    print(f"  - {k-2} times in training set")
    
    return annotation_files, config_files

if __name__ == "__main__":
    # Configuration parameters
    #input_folder = 'preprocessing/clip_annotations/balanced_dataset'
    input_folder = 'preprocessing/clip_keypoints'
    base_config_path = 'stgcn/stgcnpp_joint-motion.py'
    output_dir = 'k_fold'  # All k-fold related files will be under this directory
    k = 3  # Number of folds
    epochs = 15 # Number of epochs for training
    patients_per_test = 1  # Number of patients in each test set
    
    # Setup k-fold cross-validation
    annotation_files, config_files = setup_k_fold_cross_validation(
        input_folder=input_folder,
        base_config_path=base_config_path,
        output_dir=output_dir,
        k=k,
        epochs=epochs,
        patients_per_test=patients_per_test
    )