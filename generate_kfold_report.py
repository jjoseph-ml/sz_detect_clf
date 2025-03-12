#!/usr/bin/env python
"""
Script to generate an HTML report combining k-fold cross validation results and plots.
"""

from pathlib import Path
import pandas as pd
import json
from datetime import datetime
import shutil
import re

def extract_training_config(config_file_path):
    """Extract training configuration details from a config file.
    
    This function handles base file references and extracts configuration parameters
    from both the current file and its base files.
    """
    config = {}
    processed_files = set()  # To avoid circular references
    
    def process_config_file(file_path):
        if file_path in processed_files:
            return
        
        processed_files.add(file_path)
        
        try:
            with open(file_path, 'r') as f:
                content = f.read()
                
                # Check for base file reference
                base_match = re.search(r'_base_\s*=\s*[\'"]([^\'"]+)[\'"]', content)
                if base_match:
                    base_file = base_match.group(1)
                    # Handle relative paths
                    if not base_file.startswith('/'):
                        base_dir = Path(file_path).parent
                        base_path = base_dir / base_file
                    else:
                        base_path = Path(base_file)
                    
                    if base_path.exists():
                        process_config_file(base_path)
                    else:
                        print(f"Warning: Base file {base_path} not found")
                
                # Extract max_epochs
                max_epochs_match = re.search(r'max_epochs=(\d+)', content)
                if max_epochs_match and 'max_epochs' not in config:
                    config['max_epochs'] = int(max_epochs_match.group(1))
                
                # Extract learning rate
                lr_match = re.search(r'lr=([0-9.]+)', content)
                if lr_match and 'learning_rate' not in config:
                    config['learning_rate'] = float(lr_match.group(1))
                
                # Extract momentum
                momentum_match = re.search(r'momentum=([0-9.]+)', content)
                if momentum_match and 'momentum' not in config:
                    config['momentum'] = float(momentum_match.group(1))
                
                # Extract batch size
                batch_size_match = re.search(r'batch_size=(\d+)', content)
                if batch_size_match and 'batch_size' not in config:
                    config['batch_size'] = int(batch_size_match.group(1))
                    
                # Extract optimizer type
                optimizer_match = re.search(r'optimizer=dict\(\s*type=[\'"]([\w]+)[\'"]', content)
                if optimizer_match and 'optimizer_type' not in config:
                    config['optimizer_type'] = optimizer_match.group(1)
                    
                # Extract weight decay
                weight_decay_match = re.search(r'weight_decay=([0-9.]+)', content)
                if weight_decay_match and 'weight_decay' not in config:
                    config['weight_decay'] = float(weight_decay_match.group(1))
                    
                # Extract nesterov flag if present
                nesterov_match = re.search(r'nesterov=(True|False)', content)
                if nesterov_match and 'nesterov' not in config:
                    config['nesterov'] = nesterov_match.group(1) == 'True'
                    
                # Extract train_cfg with max_epochs
                train_cfg_match = re.search(r'train_cfg\s*=\s*dict\([^)]*max_epochs=(\d+)', content, re.DOTALL)
                if train_cfg_match and 'max_epochs' not in config:
                    config['max_epochs'] = int(train_cfg_match.group(1))
                    
                # Extract optim_wrapper with optimizer details
                optim_wrapper_match = re.search(r'optim_wrapper\s*=\s*dict\(\s*optimizer=dict\([^)]*\)\)', content, re.DOTALL)
                if optim_wrapper_match:
                    optim_text = optim_wrapper_match.group(0)
                    
                    # Extract optimizer type from optim_wrapper
                    opt_type_match = re.search(r'type=[\'"]([\w]+)[\'"]', optim_text)
                    if opt_type_match and 'optimizer_type' not in config:
                        config['optimizer_type'] = opt_type_match.group(1)
                    
                    # Extract learning rate from optim_wrapper
                    opt_lr_match = re.search(r'lr=([0-9.]+)', optim_text)
                    if opt_lr_match and 'learning_rate' not in config:
                        config['learning_rate'] = float(opt_lr_match.group(1))
                    
                    # Extract momentum from optim_wrapper
                    opt_momentum_match = re.search(r'momentum=([0-9.]+)', optim_text)
                    if opt_momentum_match and 'momentum' not in config:
                        config['momentum'] = float(opt_momentum_match.group(1))
                    
                    # Extract weight decay from optim_wrapper
                    opt_wd_match = re.search(r'weight_decay=([0-9.]+)', optim_text)
                    if opt_wd_match and 'weight_decay' not in config:
                        config['weight_decay'] = float(opt_wd_match.group(1))
                    
                    # Extract nesterov from optim_wrapper
                    opt_nesterov_match = re.search(r'nesterov=(True|False)', optim_text)
                    if opt_nesterov_match and 'nesterov' not in config:
                        config['nesterov'] = opt_nesterov_match.group(1) == 'True'
                
        except Exception as e:
            print(f"Warning: Could not extract config from {file_path}: {e}")
    
    # Start processing from the given config file
    process_config_file(config_file_path)
    
    return config

def create_html_report(training_dir: Path, test_results_file: Path, output_file: Path) -> None:
    """Create an HTML report combining training plots and test results."""
    
    # Create output directory if it doesn't exist
    output_dir = output_file.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Read test results
    print("Reading test results...")
    test_metrics = pd.read_csv(test_results_file)
    
    # Filter out the "Average" row if it exists
    # The first column might be unnamed (index column)
    first_col = test_metrics.columns[0]
    if "Average" in test_metrics[first_col].values:
        test_metrics = test_metrics[test_metrics[first_col] != "Average"]
    
    # Get the actual number of folds from the test results
    num_folds = len(test_metrics)
    print(f"Found {num_folds} folds in the test results")
    
    # Extract training configuration from the STGCN configuration files
    # First try fold-specific config, then fall back to base configs
    config_file_paths = [
        Path('k_fold/stgcn/stgcn_fold0.py'),  # Try fold-specific config first
        Path('k_fold/stgcn/stgcn_joint.py'),  # Then try base configs
        Path('k_fold/stgcn/stgcn_joint_motion.py')
    ]
    
    training_config = {}
    for config_path in config_file_paths:
        if config_path.exists():
            print(f"Extracting training configuration from {config_path}...")
            training_config = extract_training_config(config_path)
            if training_config:  # If we got some config values, stop searching
                print(f"Extracted training configuration: {training_config}")
                break
    
    # Copy individual fold plots based on actual number of folds
    print("Copying images to report directory...")
    # No more overview images to copy
    
    # Copy individual fold plots
    for fold_idx in range(num_folds):  # Use actual number of folds
        src = training_dir / f'fold{fold_idx}_curves.png'
        if src.exists():
            shutil.copy2(src, output_dir / f'fold{fold_idx}_curves.png')
            print(f"Copied fold{fold_idx}_curves.png")
        else:
            print(f"Warning: fold{fold_idx}_curves.png not found in {training_dir}")
    
    print("Generating HTML report...")
    # HTML template
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>K-Fold Cross Validation Results</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                max-width: 1200px;
                margin: 0 auto;
                padding: 20px;
                background-color: #f5f5f5;
            }}
            .container {{
                background-color: white;
                padding: 20px;
                border-radius: 5px;
                box-shadow: 0 2px 5px rgba(0,0,0,0.1);
                margin-bottom: 20px;
            }}
            h1, h2 {{
                color: #333;
                border-bottom: 2px solid #eee;
                padding-bottom: 10px;
            }}
            .metrics-table {{
                width: 100%;
                border-collapse: collapse;
                margin: 20px 0;
            }}
            .metrics-table th, .metrics-table td {{
                padding: 12px;
                text-align: center;
                border: 1px solid #ddd;
            }}
            .metrics-table th {{
                background-color: #f8f9fa;
            }}
            .config-table {{
                width: 48%;
                border-collapse: collapse;
                margin: 20px 0;
                display: inline-table;
                vertical-align: top;
                box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            }}
            .config-table th, .config-table td {{
                padding: 12px;
                text-align: left;
                border: 1px solid #ddd;
            }}
            .config-table th {{
                background-color: #f8f9fa;
                width: 40%;
            }}
            .config-table caption {{
                font-weight: bold;
                padding: 10px;
                background-color: #f0f0f0;
                border: 1px solid #ddd;
                border-bottom: none;
            }}
            .config-container {{
                display: flex;
                justify-content: space-between;
                flex-wrap: wrap;
                gap: 20px;
            }}
            .plot-container {{
                margin: 20px 0;
                text-align: center;
            }}
            .plot-container img {{
                max-width: 100%;
                height: auto;
                margin: 10px 0;
            }}
            .fold-section {{
                margin: 30px 0;
                padding: 20px;
                background-color: white;
                border-radius: 5px;
                box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            }}
            .timestamp {{
                color: #666;
                font-size: 0.9em;
                margin-top: 20px;
                text-align: right;
            }}
            .best-value {{
                font-weight: bold;
                color: #28a745;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>K-Fold Cross Validation Results</h1>
            
            <h2>Training Configuration</h2>
            <div class="config-container">
                <table class="config-table">
                    <caption>Training Parameters</caption>
                    <tr>
                        <th>Parameter</th>
                        <th>Value</th>
                    </tr>
                    <tr>
                        <td>Max Epochs</td>
                        <td>{training_config.get('max_epochs', 'N/A')}</td>
                    </tr>
                    <tr>
                        <td>Batch Size</td>
                        <td>{training_config.get('batch_size', 'N/A')}</td>
                    </tr>
                </table>
                
                <table class="config-table">
                    <caption>Optimizer Settings</caption>
                    <tr>
                        <th>Parameter</th>
                        <th>Value</th>
                    </tr>
                    <tr>
                        <td>Type</td>
                        <td>{training_config.get('optimizer_type', 'N/A')}</td>
                    </tr>
                    <tr>
                        <td>Learning Rate</td>
                        <td>{training_config.get('learning_rate', 'N/A')}</td>
                    </tr>
                    <tr>
                        <td>Momentum</td>
                        <td>{training_config.get('momentum', 'N/A')}</td>
                    </tr>
                    <tr>
                        <td>Weight Decay</td>
                        <td>{training_config.get('weight_decay', 'N/A')}</td>
                    </tr>
                    <tr>
                        <td>Nesterov</td>
                        <td>{training_config.get('nesterov', 'N/A')}</td>
                    </tr>
                </table>
            </div>
            
            <h2>Performance by Fold</h2>
            <table class="metrics-table">
                <tr>
                    <th>Fold</th>
        """
    
    # Add metric names to header
    metrics = ['Accuracy', 'F1 Score', 'AUC-ROC', 'Sensitivity', 'Specificity', 'PPV', 'NPV']
    for metric in metrics:
        html_content += f"<th>{metric}</th>"
    html_content += "</tr>"
    
    # Add rows for each fold
    for fold_idx in range(len(test_metrics)):
        html_content += f"""
                <tr>
                    <td>{fold_idx}</td>
        """
        for metric in metrics:
            value = test_metrics.loc[fold_idx, metric]
            # Check if this is the best value for this metric
            is_best = value == test_metrics[metric].max()
            cell_class = 'best-value' if is_best else ''
            html_content += f"""
                    <td class="{cell_class}">{value:.4f}</td>
            """
        html_content += "</tr>"
    
    # Add average row
    html_content += """
                <tr style="border-top: 2px solid #ddd;">
                    <td><strong>Average</strong></td>
        """
    for metric in metrics:
        avg_value = test_metrics[metric].mean()
        html_content += f"""
                    <td><strong>{avg_value:.4f}</strong></td>
        """
    html_content += """
                </tr>
            </table>
            
            <h2>Individual Fold Results</h2>
    """
    
    # Add individual fold sections with updated image paths
    for fold_idx in range(len(test_metrics)):
        html_content += f"""
            <div class="fold-section">
                <h3>Fold {fold_idx}</h3>
                <div class="plot-container">
                    <img src="./fold{fold_idx}_curves.png" alt="Fold {fold_idx} Training Curves">
                </div>
            </div>
        """
    
    # Add timestamp and close tags
    html_content += f"""
            <div class="timestamp">
                Report generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
            </div>
        </div>
    </body>
    </html>
    """
    
    # Write the HTML file
    print(f"Writing HTML report to {output_file}")
    with open(output_file, 'w') as f:
        f.write(html_content)

def main():
    print("Starting report generation...")
    
    # Paths
    training_dir = Path('k_fold/training_analysis')
    test_results_file = Path('k_fold/test_results/fold_metrics.csv')
    output_dir = Path('k_fold/report')
    output_file = output_dir / 'k_fold_report.html'
    
    print(f"Using following paths:")
    print(f"  Training analysis dir: {training_dir}")
    print(f"  Test results file: {test_results_file}")
    print(f"  Output directory: {output_dir}")
    
    # Check if required directories and files exist
    if not training_dir.exists():
        print(f"Error: Training analysis directory not found: {training_dir}")
        return 1
    
    if not test_results_file.exists():
        print(f"Error: Test results file not found: {test_results_file}")
        return 1
    
    try:
        create_html_report(training_dir, test_results_file, output_file)
        print("\nReport generation completed successfully!")
        print(f"Report saved to: {output_file}")
        print("You can open it in a web browser to view the results.")
        return 0
    except Exception as e:
        print(f"\nError generating report: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == '__main__':
    exit(main()) 