# Script Execution Order and Dependencies

## 1. 001_augment_videos.py

The script reads video annotations from `preprocessing/video_annotations/video_annotations.txt`, processes each video by applying a horizontal flip transformation, and saves the augmented versions with an `_aug` suffix while preserving the original video properties.

## 2. 002_extract_poses_from_videos.py

Extracts human pose keypoints from videos using a pre-trained pose estimation model (HRNet with COCO-wholebody dataset).

### Features:
- **Video Processing**: Reads videos from an annotation file and processes them frame-by-frame
- **Pose Detection**: Uses MMPose (HRNet model) to detect 133 keypoints per person in each frame
- **Bounding Box Handling**: Uses predefined bounding boxes for each video, with special handling for augmented videos (horizontal flips)
- **Visualization**: Optionally creates output videos showing the detected poses
- **Batch Processing**: Handles both original and augmented videos, skipping already processed ones

## 3. 003_split_keypoints_into_clips.py

This script processes human pose keypoint data extracted from videos and splits them into smaller clips for machine learning training using STGCN model of MMaction2.

### Functionality:
- Splits videos into clips of a specified length (default 90 frames, about 3 seconds at 30fps)
- Labels each clip as seizure (1) or non-seizure (0) based on whether the clip's time range overlaps with the annotated seizure period
- Generates annotation files that map each clip file path to its seizure/non-seizure label
- **Partial seizure detection**: When using `--use-partial-labels`, clips that partially overlap with seizure periods are labeled as 9

### Usage:
```bash
# Basic usage (labels: 0, 1)
python 003_split_keypoints_into_clips.py

# With partial seizure detection (labels: 0, 1, 9)
python 003_split_keypoints_into_clips.py --use-partial-labels
```

### Output Types:
- **Filtered clips**: Processed into a standardized format expected by MMAaction2 STGCN model
- **Raw clips**: Preserves the original keypoint data structure (Not used by the training pipeline)

## 4. 004_setup_k_fold.py

This script sets up k-fold cross-validation for a seizure detection model using skeleton keypoint data. It organizes video clip data by patient ID, filters annotations to only include valid seizure (class 1) and normal (class 0) labels, and creates balanced training/validation/test splits for each fold.

### Key Features:
- Ensures that each patient appears exactly once in the test set and once in the validation set across all folds
- Patients appear in multiple training sets
- Generates both annotation files (containing the data splits) and configuration files (for model training) for each fold
- Provides options to control parameters like number of epochs, clip length, and feature types
- Provides detailed statistics about class distribution and patient-wise seizure ratios

### Input/Output:
- **Input**: `preprocessing/clip_annotations.txt`
- **Output**: 
  - Annotation files: `k_fold/data/skeleton/bcm_master_annotation_fold{0-4}.pkl`
  - Configuration files: `k_fold/stgcn/stgcnpp_fold{0-4}.py`

## 4a. 004_setup_k_fold_cross_site_flexible_splits.py

Unified script for creating k-fold pickle files with flexible train/val/test split configurations for cross-site experiments.

**PREREQUISITE**: `004_setup_k_fold.py` must be run prior for each individual site (BCM and UCLA) before running this script.

### Key Features:
- Supports all 27 combinations of train/val/test splits:
  - train: bcm, ucla, combined
  - val: bcm, ucla, combined
  - test: bcm, ucla, combined
- Combines site-specific fold files created by `004_setup_k_fold.py`
- Validates that both sites have the same number of fold files

### Usage:
```bash
python 004_setup_k_fold_cross_site_flexible_splits.py --train combined --val combined --test bcm
python 004_setup_k_fold_cross_site_flexible_splits.py --train bcm --val bcm --test ucla
python 004_setup_k_fold_cross_site_flexible_splits.py --train ucla --val ucla --test bcm --force
```

## 5. 005A_run_kfold_training.py

This script runs k-fold cross-validation training for a pose-based action recognition model (ST-GCN) using MMAction2.

### Functionality:
- **Manages fold configurations**: Finds all fold config files and determines which folds to train
- **Cleans up previous runs**: Optionally deletes old checkpoint files from previous training sessions
- **Runs training sequentially**: For each fold, it creates a work directory and calls MMAction2's train.py with the appropriate config file

### Usage Examples:
```bash
# Run all folds
python run_kfold_training.py

# Single fold
python 005A_run_kfold_training.py --folds 0

# Multiple specific folds
python 005A_run_kfold_training.py --folds 0,1,2

# Non-sequential folds
python 005A_run_kfold_training.py --folds 1,3,4

# Clean up previous checkpoints
python 005A_run_kfold_training.py --cleanup

# Dry run - no command execution
python 005A_run_kfold_training.py --dry-run
```

## 6. 005b_run_kfold_train_analysis.py

This script processes training logs from multiple k-fold experiments and generates comprehensive analysis and visualizations of the training performance.

### Processing Steps:
- Finds training log files in timestamped directories within each fold's work directory
- Extracts training and validation metrics (loss and accuracy) from MMAction2 training logs
- Creates plots showing loss and accuracy over epochs for each fold and saves them as PNG files
- Exports all metrics for all folds in a CSV
- Creates a summary file with key statistics for each fold (final loss, best accuracy, etc.)

### Outputs:
Generated in `k_fold/training_analysis/` directory:
- `training_curves_by_fold.png` - Combined visualization of all folds
- `fold{X}_curves.png` - Individual plots for each fold
- `metric_distributions.png` - Boxplots showing metric distributions
- `comprehensive_training_metrics.csv` - All raw metrics data
- `training_summary.csv` - Summary statistics for each fold

## 7. 005c_run_kfold_testing.py / 005c_run_kfold_testing_5090.py

This script runs testing for pose-based seizure detection models that were trained using k-fold cross-validation. It uses the MMAction2 framework to evaluate the trained models on their respective test sets.

**Note**: `005c_run_kfold_testing_5090.py` is the recommended version with PyTorch 2.6+ checkpoint compatibility fixes and cross-site testing support.

### Functionality:
- Finds the best trained checkpoint for each fold
- Runs the MMAction2 test.py script with the appropriate configuration
- Generates prediction results for each fold
- Saves prediction results as pickle files (`fold{0-4}_predictions.pkl`)

### Version 5090 Features:
- **PyTorch 2.6+ Compatibility**: Automatically fixes checkpoint loading issues with `weights_only` parameter
- **Cross-Site Testing Support**: `--mode` argument supports both `kfold` and `cross_site` testing modes
- **Checkpoint Fixing**: `--fix-checkpoints` flag (default: True) handles compatibility issues automatically

### Usage (5090 version):
```bash
# K-fold testing (default)
python 005c_run_kfold_testing_5090.py --folds all

# Cross-site testing
python 005c_run_kfold_testing_5090.py --mode cross_site

# Specific folds
python 005c_run_kfold_testing_5090.py --folds 0,1,2
```

## 8. 005d_run_fold_prediction_analysis.py

Reads pickle files containing model predictions from each fold and extracts ground truth labels, predicted labels, and prediction scores.

### Metrics Computed:
For each fold, it computes:
- **Accuracy**: Overall correct predictions
- **F1 Score**: Harmonic mean of precision and recall
- **AUC-ROC**: Area under the ROC curve
- **Sensitivity**: True positive rate (recall)
- **Specificity**: True negative rate
- **PPV**: Positive predictive value (precision)
- **NPV**: Negative predictive value

### Outputs:
- Saves results to a CSV file (`fold_metrics.csv`)
- Creates visualizations:
  - ROC Curves
  - Confusion matrices

## 9. 005E_generate_kfold_report.py

Creates a professional HTML report that consolidates training and testing results from k-fold cross-validation experiments into a single, easy-to-view document.

## 10. 005_run_kfold_pipeline.py

This script acts as a pipeline orchestrator that automates the complete k-fold cross-validation workflow. It runs the following scripts sequentially:

1. `005A_run_kfold_training.py`
2. `005b_run_kfold_train_analysis.py`
3. `005c_run_kfold_testing.py`
4. `005d_run_fold_prediction_analysis.py`
5. `005E_generate_kfold_report.py`

## 11. 006_test_all_videos.py

This script tests a specified model checkpoint on all videos with seizure events.

### Functionality:
- **All Videos Testing**: Tests the specified model checkpoint on all videos that contain seizure events
- **Checkpoint-Based**: Uses a user-specified checkpoint file instead of automatically selecting the best fold
- **Comprehensive Output**: Generates raw predictions, enhanced predictions with clip names, and CSV results

### Usage Examples:
```bash
# Test using a specific checkpoint file
python 006_test_all_videos.py --checkpoint k_fold/work_dirs/fold0/best_top1_acc_epoch_50.pth


# Dry run to see commands without execution
python 006_test_all_videos.py --checkpoint k_fold/work_dirs/fold2/best_top1_acc_epoch_60.pth --dry-run
```

### Outputs:
- **Unified Directory**: All results saved to `all_video_testing/all/` folder
- **Files Created**:
  - `best_model_all_predictions_raw.pkl` - Raw predictions from MMAction2
  - `best_model_all_predictions.pkl` - Enhanced predictions with clip names
  - `all_predictions.csv` - CSV file with comprehensive metadata
  - Modified config files for all-video testing

### Key Features:
- **Checkpoint Flexibility**: Can use any trained model checkpoint
- **Automatic Directory Creation**: Creates output directories if they don't exist
- **Comprehensive Metadata**: CSV includes patient ID, video ID, segment info, and prediction details
- **GPU Support**: Automatically detects and uses available GPUs
- **Resume Capability**: Can handle interrupted testing sessions

## 12. 008_run_saliency.py

This script runs a gradient-based saliency analysis to understand which parts of the input (human pose keypoints) are most important for the model's predictions.

### Functionality:
- **Saliency Analysis**: Computes gradients with respect to input keypoints to identify important body regions
- **Multiple Output Types**: Generates different types of saliency visualizations
- **Video-Specific Analysis**: Can process single videos or all videos in the dataset
- **Comprehensive Metrics**: Calculates performance metrics for each video

### Task Options:
- **`histograms`**: Create body part saliency histograms showing which body regions are most important
- **`saliency_maps`**: Create static PNG plots of saliency maps for individual clips
- **`saliency_videos`**: Create individual clip videos AND whole video visualization
- **`whole_video`**: Create only whole video visualization (aggregated saliency)
- **`all`**: Create all outputs (default)

### Usage Examples:
```bash
# Process all videos with all outputs
python 008_run_saliency.py --best-fold 0 --task all

# Process only histograms for all videos
python 008_run_saliency.py --best-fold 1 --task histograms

# Process only saliency maps for all videos
python 008_run_saliency.py --best-fold 2 --task saliency_maps

# Process only saliency videos for all videos
python 008_run_saliency.py --best-fold 0 --task saliency_videos

# Process only whole video visualization
python 008_run_saliency.py --best-fold 1 --task whole_video

# Process a single specific video
python 008_run_saliency.py --best-fold 0 --task all --video-id video_001

# Process histograms for a single video
python 008_run_saliency.py --best-fold 1 --task histograms --video-id video_002
```

### Output Structure:
Generated in `k_fold/saliency_maps/` directory:
- **`videos/{video_id}/`**: Organized by video ID
  - **`saliency_maps/`**: Static PNG plots of saliency maps for each clip
  - **`histograms/clips/`**: Individual clip body part saliency histograms
  - **`histograms/aggregated/`**: Aggregated body part saliency across all clips
  - **`videos/clips/`**: Saliency visualization videos for individual clips
  - **`videos/whole_video/`**: Whole video saliency visualization
  - **`data/`**: Raw saliency data and metrics

### Analysis Features:
- **Body Part Analysis**: Identifies which body regions (head, arms, legs, torso) are most important
- **Movement Correlation**: Correlates saliency with movement patterns over time
- **Keypoint Highlighting**: Highlights important keypoints in each frame
- **Temporal Analysis**: Shows saliency changes over the entire video timeline
- **Performance Metrics**: Calculates accuracy, F1-score, AUC-ROC, sensitivity, specificity for each video
- **Per-Patient Analysis**: Creates aggregated analysis across patients

### Prerequisites:
- Must have run `006_test_all_videos.py` first to generate predictions CSV
- Requires trained model checkpoints from k-fold training
- Needs keypoint data files in `preprocessing/clip_keypoints/`