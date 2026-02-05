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

try:
    import pdfkit
    PDFKIT_AVAILABLE = True
except ImportError:
    PDFKIT_AVAILABLE = False
    print("Warning: pdfkit not available. PDF generation will be skipped.")
    print("Install with: pip install pdfkit")
    print("Also requires wkhtmltopdf: https://wkhtmltopdf.org/downloads.html")

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
                
                # Extract learning rate - look for both direct lr and in optim_wrapper
                lr_match = re.search(r'lr=([0-9.e+-]+)', content)
                if lr_match and 'learning_rate' not in config:
                    try:
                        config['learning_rate'] = float(lr_match.group(1))
                    except ValueError:
                        # Handle scientific notation like 1e-7
                        config['learning_rate'] = lr_match.group(1)
                
                # Extract batch size
                batch_size_match = re.search(r'batch_size=(\d+)', content)
                if batch_size_match and 'batch_size' not in config:
                    config['batch_size'] = int(batch_size_match.group(1))
                    
                # Extract optimizer type
                optimizer_match = re.search(r'optimizer=dict\(\s*type=[\'"]([\w]+)[\'"]', content)
                if optimizer_match and 'optimizer_type' not in config:
                    config['optimizer_type'] = optimizer_match.group(1)
                    
                # Extract features (feats)
                feats_match = re.search(r'feats=\[([^\]]+)\]', content)
                if feats_match and 'feats' not in config:
                    config['feats'] = feats_match.group(1)
                
                                # Extract clip length
                clip_len_match = re.search(r'clip_len=(\d+)', content)
                if clip_len_match and 'clip_len' not in config:
                    config['clip_len'] = int(clip_len_match.group(1))
                
                # Extract dropout
                dropout_match = re.search(r'dropout=([0-9.]+)', content)
                if dropout_match and 'dropout' not in config:
                    config['dropout'] = float(dropout_match.group(1))
                
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
                    opt_lr_match = re.search(r'lr=([0-9.e+-]+)', optim_text)
                    if opt_lr_match and 'learning_rate' not in config:
                        try:
                            config['learning_rate'] = float(opt_lr_match.group(1))
                        except ValueError:
                            # Handle scientific notation like 1e-7
                            config['learning_rate'] = opt_lr_match.group(1)
                    
                    # Extract weight decay from optim_wrapper
                    opt_wd_match = re.search(r'weight_decay=([0-9.e+-]+)', optim_text)
                    if opt_wd_match and 'weight_decay' not in config:
                        try:
                            config['weight_decay'] = float(opt_wd_match.group(1))
                        except ValueError:
                            # Handle scientific notation like 1e-7
                            config['weight_decay'] = opt_wd_match.group(1)
                
        except Exception as e:
            print(f"Warning: Could not extract config from {file_path}: {e}")
    
    # Start processing from the given config file
    process_config_file(config_file_path)
    
    return config

def html_to_pdf(html_file: Path, pdf_file: Path) -> bool:
    """Convert HTML file to PDF using pdfkit."""
    if not PDFKIT_AVAILABLE:
        print("pdfkit not available, skipping PDF generation")
        return False
    
    try:
        print(f"Converting HTML to PDF: {pdf_file}")
        print(f"HTML file exists: {html_file.exists()}")
        print(f"HTML file size: {html_file.stat().st_size} bytes")
        
        # Configure pdfkit options for better output
        options = {
            'page-size': 'A4',
            'margin-top': '0in',
            'margin-right': '0in',
            'margin-bottom': '0in',
            'margin-left': '0in',
            'encoding': "UTF-8",
            'no-outline': None,
            'enable-local-file-access': None,
            'print-media-type': None,
            'dpi': 300,
            'image-quality': 100
        }
        
        # Try to find wkhtmltopdf executable
        import subprocess
        wkhtmltopdf_path = None
        
        # First try to find it in PATH
        try:
            result = subprocess.run(['wkhtmltopdf', '--version'], capture_output=True, text=True)
            if result.returncode == 0:
                wkhtmltopdf_path = 'wkhtmltopdf'
                print(f"wkhtmltopdf version: {result.stdout.strip()}")
            else:
                print("wkhtmltopdf found but returned error")
        except FileNotFoundError:
            print("wkhtmltopdf not found in PATH, trying common Windows locations...")
        
        # If not in PATH, try common Windows installation locations
        if not wkhtmltopdf_path:
            common_paths = [
                r"C:\Program Files\wkhtmltopdf\bin\wkhtmltopdf.exe",
                r"C:\Program Files (x86)\wkhtmltopdf\bin\wkhtmltopdf.exe",
                r"C:\wkhtmltopdf\bin\wkhtmltopdf.exe",
                r"C:\wkhtmltopdf\wkhtmltopdf.exe"
            ]
            
            for path in common_paths:
                if Path(path).exists():
                    wkhtmltopdf_path = path
                    print(f"Found wkhtmltopdf at: {path}")
                    break
        
        if not wkhtmltopdf_path:
            print("Error: wkhtmltopdf not found!")
            print("Please install wkhtmltopdf from: https://wkhtmltopdf.org/downloads.html")
            print("Common installation paths:")
            print("  - C:\\Program Files\\wkhtmltopdf\\bin\\")
            print("  - C:\\wkhtmltopdf\\bin\\")
            return False
        
        # Configure pdfkit with the found path
        config = pdfkit.configuration(wkhtmltopdf=wkhtmltopdf_path)
        
        pdfkit.from_file(str(html_file), str(pdf_file), options=options, configuration=config)
        print(f"PDF generated successfully: {pdf_file}")
        print(f"PDF file exists: {pdf_file.exists()}")
        print(f"PDF file size: {pdf_file.stat().st_size} bytes")
        return True
    except Exception as e:
        print(f"Error generating PDF: {e}")
        import traceback
        traceback.print_exc()
        return False

def create_html_report(training_dir: Path, test_results_file: Path, output_file: Path, mode: str = 'kfold') -> None:
    """Create an HTML report combining training plots and test results."""
    
    # Create output directory if it doesn't exist
    output_dir = output_file.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Read test results
    print("Reading test results...")
    test_metrics = pd.read_csv(test_results_file, index_col=0)
    
    # Filter out the "Average" row if it exists
    if "Average" in test_metrics.index:
        test_metrics = test_metrics[test_metrics.index != "Average"]
    
    # Get the actual number of runs from the test results
    num_runs = len(test_metrics)
    if mode == 'kfold':
        print(f"Found {num_runs} folds in the test results")
        run_type = "folds"
    else:
        print(f"Found {num_runs} cross-site tests in the test results")
        run_type = "cross-site tests"
    
    # Try to read validation metrics if available
    validation_metrics = None
    training_summary = None
    try:
        validation_file = Path('k_fold/training_analysis/comprehensive_training_metrics.csv')
        training_summary_file = Path('k_fold/training_analysis/training_summary.csv')
        
        if validation_file.exists():
            print("Reading validation metrics...")
            validation_metrics = pd.read_csv(validation_file)
            print(f"Found validation metrics for {len(validation_metrics)} epochs")
        
        if training_summary_file.exists():
            print("Reading training summary...")
            training_summary = pd.read_csv(training_summary_file)
            print(f"Found training summary for {len(training_summary)} folds")
            
    except Exception as e:
        print(f"Warning: Could not read validation metrics: {e}")
        validation_metrics = None
        training_summary = None
    
    # Extract training configuration from the STGCN configuration files
    # First try run-specific config, then fall back to base configs
    if mode == 'kfold':
        config_file_paths = [
            Path('k_fold/stgcn/stgcnpp_fold0.py'),  # Try fold-specific config first
            #Path('k_fold/stgcn/stgcn_joint.py'),  # Then try base configs
            #Path('k_fold/stgcn/stgcn_joint_motion.py')
        ]
    else:  # cross_site mode
        # Find cross-site config files
        cross_site_configs = list(Path('k_fold/stgcn').glob('stgcnpp_cross_site_test_*.py'))
        if cross_site_configs:
            config_file_paths = [cross_site_configs[0]]  # Use the first cross-site config found
        else:
            config_file_paths = []
    
    training_config = {}
    for config_path in config_file_paths:
        if config_path.exists():
            print(f"Extracting training configuration from {config_path}...")
            training_config = extract_training_config(config_path)
            if training_config:  # If we got some config values, stop searching
                print(f"Extracted training configuration: {training_config}")
                break
    
    # Read the config file content for display in the report
    config_content = ""
    if mode == 'kfold':
        config_path = Path('k_fold/stgcn/stgcnpp_fold0.py')
    else:  # cross_site mode
        cross_site_configs = list(Path('k_fold/stgcn').glob('stgcnpp_cross_site_test_*.py'))
        config_path = cross_site_configs[0] if cross_site_configs else None
    
    if config_path and config_path.exists():
        print(f"Reading config file: {config_path}")
        try:
            with open(config_path, 'r') as f:
                config_content = f.read()
        except Exception as e:
            print(f"Warning: Could not read config file: {e}")
            config_content = "Error reading config file"
    else:
        print(f"Warning: Config file not found: {config_path}")
        config_content = "Config file not found"
    
    # Copy individual run plots based on mode
    print("Copying images to report directory...")
    
    # Copy individual run plots - look for all available run plots
    if mode == 'kfold':
        run_plots = list(training_dir.glob('fold*_curves.png'))
        run_plots.sort()  # Sort them for consistent ordering
        
        for run_plot in run_plots:
            # Extract fold number from filename (e.g., "fold4_curves.png" -> "4")
            fold_match = re.search(r'fold(\d+)_curves\.png', run_plot.name)
            if fold_match:
                fold_num = fold_match.group(1)
                dst = output_dir / f'fold{fold_num}_curves.png'
                shutil.copy2(run_plot, dst)
                print(f"Copied {run_plot.name} -> {dst.name}")
            else:
                print(f"Warning: Could not parse fold number from {run_plot.name}")
    else:  # cross_site mode
        run_plots = list(training_dir.glob('cross_site_test_*_curves.png'))
        run_plots.sort()  # Sort them for consistent ordering
        
        for run_plot in run_plots:
            # Extract site name from filename (e.g., "cross_site_test_ucla_curves.png" -> "ucla")
            site_match = re.search(r'cross_site_test_([^_]+)_curves\.png', run_plot.name)
            if site_match:
                site_name = site_match.group(1)
                dst = output_dir / f'cross_site_test_{site_name}_curves.png'
                shutil.copy2(run_plot, dst)
                print(f"Copied {run_plot.name} -> {dst.name}")
            else:
                print(f"Warning: Could not parse site name from {run_plot.name}")
    
    if not run_plots:
        print(f"Warning: No individual {'fold' if mode == 'kfold' else 'cross-site'} plots found in training directory")
    
    # Copy overview training plots (excluding the cluttered combined plot)
    overview_plots = [
        'metric_distributions.png'
    ]
    
    for plot_name in overview_plots:
        src = training_dir / plot_name
        if src.exists():
            shutil.copy2(src, output_dir / plot_name)
            print(f"Copied {plot_name}")
        else:
            print(f"Warning: {plot_name} not found in {training_dir}")
    
    # Copy ROC curves plot
    roc_src = Path('k_fold/test_results/roc_curves.png')
    if roc_src.exists():
        shutil.copy2(roc_src, output_dir / 'roc_curves.png')
        print(f"Copied roc_curves.png")
    else:
        print(f"Warning: roc_curves.png not found in {roc_src}")
    
    print("Generating HTML report...")
    # HTML template
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>{'K-Fold Cross Validation' if mode == 'kfold' else 'Cross-Site Validation'} Results</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                max-width: 1200px;
                margin: 0 auto;
                padding: 20px;
                background-color: white;
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
                font-size: 0.85em;
            }}
            .metrics-table th, .metrics-table td {{
                padding: 10px 8px;
                text-align: center;
                border: 1px solid #ddd;
            }}
            .metrics-table th {{
                background-color: #f8f9fa;
            }}

            .config-table-container {{
                margin: 20px 0;
                text-align: center;
            }}
            .config-table {{
                width: 100%;
                border-collapse: collapse;
                margin: 0 auto;
                background-color: white;
                border-radius: 5px;
                overflow: hidden;
                box-shadow: 0 2px 5px rgba(0,0,0,0.1);
                font-size: 0.85em;
            }}
            .config-table th {{
                background-color: #f8f9fa;
                color: #495057;
                font-weight: bold;
                padding: 10px 6px;
                border: 1px solid #dee2e6;
                text-align: center;
                font-size: 0.85em;
            }}
            .config-table td {{
                padding: 10px 6px;
                border: 1px solid #dee2e6;
                text-align: center;
                background-color: white;
                font-weight: 500;
            }}
            /* PDF-specific styles */
            @media print {{
                .config-table {{
                    page-break-inside: avoid;
                    border-collapse: collapse;
                }}
                .config-table th, .config-table td {{
                    border: 1px solid #000;
                }}
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
                page-break-inside: avoid;
            }}
            .plot-section {{
                margin: 30px 0;
                padding: 20px;
                background-color: white;
                border-radius: 5px;
                box-shadow: 0 2px 5px rgba(0,0,0,0.1);
                page-break-inside: avoid;
            }}
            .validation-section {{
                margin: 30px 0;
                padding: 20px;
                background-color: #f8f9fa;
                border-radius: 5px;
                border: 1px solid #e9ecef;
                page-break-inside: avoid;
            }}
            .validation-section h3 {{
                color: #495057;
                margin-top: 0;
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
            .worst-value {{
                font-weight: bold;
                color: #dc3545;
            }}
            .config-code {{
                background-color: #f8f9fa;
                border: 1px solid #e9ecef;
                border-radius: 5px;
                padding: 20px;
                margin: 20px 0;
                font-family: 'Courier New', monospace;
                font-size: 12px;
                line-height: 1.4;
                overflow-x: auto;
                white-space: pre-wrap;
                word-wrap: break-word;
            }}
            .config-section {{
                margin: 30px 0;
                padding: 20px;
                background-color: white;
                border-radius: 5px;
                box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            }}
            .note-box {{
                background-color: #e7f3ff;
                border-left: 4px solid #2196F3;
                padding: 15px;
                margin: 20px 0;
                border-radius: 5px;
            }}
            .note-box p {{
                margin: 0;
                color: #1976D2;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>{'K-Fold Cross Validation' if mode == 'kfold' else 'Cross-Site Validation'} Results</h1>
            
            <h2>Training Configuration</h2>
            <div class="config-table-container">
                <table class="config-table">
                    <tr>
                        <th>Epochs</th>
                        <th>Batch Size</th>
                        <th>Learning Rate</th>
                        <th>Optimizer</th>
                    </tr>
                    <tr>
                        <td>{training_config.get('max_epochs', 'N/A')}</td>
                        <td>{training_config.get('batch_size', 'N/A')}</td>
                        <td>{training_config.get('learning_rate', 'N/A')}</td>
                        <td>{training_config.get('optimizer_type', 'N/A')}</td>
                    </tr>
                    <tr>
                        <th>Features</th>
                        <th>Clip Length</th>
                        <th>Weight Decay</th>
                        <th>Dropout</th>
                    </tr>
                    <tr>
                        <td>{training_config.get('feats', 'N/A')}</td>
                        <td>{training_config.get('clip_len', 'N/A')}</td>
                        <td>{training_config.get('weight_decay', 'N/A')}</td>
                        <td>{training_config.get('dropout', 'N/A')}</td>
                    </tr>
                </table>
            </div>
            
            <div class="note-box">
                <p><strong>Note:</strong> For complete training configuration details, please refer to the configuration file at the bottom of this report.</p>
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
    for i, fold_idx in enumerate(test_metrics.index):
        html_content += f"""
                <tr>
                    <td>{fold_idx}</td>
        """
        for metric in metrics:
            value = test_metrics.loc[fold_idx, metric]
            # Check if this is the best or worst value for this metric
            is_best = value == test_metrics[metric].max()
            is_worst = value == test_metrics[metric].min()
            
            if is_best:
                cell_class = 'best-value'
            elif is_worst:
                cell_class = 'worst-value'
            else:
                cell_class = ''
                
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
            
            <h2>Training & Validation Metrics</h2>
        """
    
    # Add validation metrics if available
    if training_summary is not None and not training_summary.empty:
        html_content += """
            <div class="validation-section">
                <h3>Training Summary by Fold</h3>
                <table class="metrics-table">
                    <tr>
                        <th>Fold</th>
                        <th>Total Epochs</th>
                        <th>Best Val Accuracy</th>
                        <th>Best Checkpoint</th>
                        <th>Final Val Accuracy</th>
                        <th>Final Train Accuracy</th>
                        <th>Min Val Loss</th>
                        <th>Min Train Loss</th>
                    </tr>
        """
        
        for _, row in training_summary.iterrows():
            # Calculate best checkpoint epoch from comprehensive metrics if available
            best_checkpoint = "N/A"
            if validation_metrics is not None and not validation_metrics.empty:
                fold_id = row['Fold']
                if mode == 'kfold':
                    fold_num = int(fold_id)  # Convert to integer for k-fold
                    val_acc_col = f'Fold{fold_num}_Val_Accuracy'
                else:  # cross_site mode
                    val_acc_col = f'CrossSite_{fold_id}_Val_Accuracy'
                
                if val_acc_col in validation_metrics.columns:
                    # Find the epoch with best validation accuracy
                    val_acc_values = validation_metrics[val_acc_col].dropna()  # Remove NaN values
                    if not val_acc_values.empty:
                        best_acc = val_acc_values.max()
                        best_epoch = validation_metrics.loc[validation_metrics[val_acc_col] == best_acc, 'Epoch'].iloc[0]
                        best_checkpoint = f"Epoch {int(best_epoch)}"
            
            html_content += f"""
                    <tr>
                        <td>{fold_id}</td>
                        <td>{row['Total_Epochs']}</td>
                        <td>{row['Best_Val_Accuracy']:.4f}</td>
                        <td>{best_checkpoint}</td>
                        <td>{row['Final_Val_Accuracy']:.4f}</td>
                        <td>{row['Final_Train_Accuracy']:.4f}</td>
                        <td>{row['Min_Val_Loss']:.4f}</td>
                        <td>{row['Min_Train_Loss']:.4f}</td>
                    </tr>
            """
        
        html_content += """
                </table>
            </div>
        """
    

    
    html_content += f"""
            <h2>Individual {'Fold' if mode == 'kfold' else 'Cross-Site Test'} Results</h2>
        """
    
    # Add individual run sections with updated image paths
    # Use the actual run plots that were copied
    if mode == 'kfold':
        run_plots = list(training_dir.glob('fold*_curves.png'))
        # Sort numerically by fold number instead of alphabetically
        run_plots.sort(key=lambda x: int(re.search(r'fold(\d+)_curves\.png', x.name).group(1)))
        
        for run_plot in run_plots:
            # Extract fold number from filename
            fold_match = re.search(r'fold(\d+)_curves\.png', run_plot.name)
            if fold_match:
                fold_num = fold_match.group(1)
                html_content += f"""
                    <div class="fold-section">
                        <h3>Fold {fold_num}</h3>
                        <div class="plot-container">
                            <img src="./fold{fold_num}_curves.png" alt="Fold {fold_num} Training Curves">
                        </div>
                    </div>
                """
    else:  # cross_site mode
        run_plots = list(training_dir.glob('cross_site_test_*_curves.png'))
        run_plots.sort()  # Sort alphabetically by site name
        
        for run_plot in run_plots:
            # Extract site name from filename
            site_match = re.search(r'cross_site_test_([^_]+)_curves\.png', run_plot.name)
            if site_match:
                site_name = site_match.group(1)
                html_content += f"""
                    <div class="fold-section">
                        <h3>Cross-Site Test {site_name}</h3>
                        <div class="plot-container">
                            <img src="./cross_site_test_{site_name}_curves.png" alt="Cross-Site Test {site_name} Training Curves">
                        </div>
                    </div>
                """
    
    # Add overview training plots (excluding the cluttered combined plot)
    html_content += f"""
            <div class="plot-section">
                <h2>Training Overview</h2>
                <div class="plot-container">
                    <img src="./metric_distributions.png" alt="Metric Distributions Across Folds">
                </div>
            </div>
        """
    
    # Add ROC curves plot
    html_content += f"""
            <div class="plot-section">
                <h2>ROC Curves</h2>
                <div class="plot-container">
                    <img src="./roc_curves.png" alt="ROC Curves for All Folds">
                </div>
            </div>
        """
    
    # Add configuration file content
    html_content += f"""
            <div class="config-section">
                <h2>{'Fold' if mode == 'kfold' else 'Cross-Site'} Configuration File</h2>
                <p>Below is the complete configuration file used for training:</p>
                <div class="config-code">{config_content}</div>
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
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Generate k-fold or cross-site validation report')
    parser.add_argument('--mode', type=str, choices=['kfold', 'cross_site'], default='kfold',
                        help='Report mode: kfold or cross_site (default: kfold)')
    args = parser.parse_args()
    
    print("Starting report generation...")
    
    # Paths
    training_dir = Path('k_fold/training_analysis')
    
    # Determine test results file based on mode
    if args.mode == 'kfold':
        test_results_file = Path('k_fold/test_results/fold_metrics.csv')
        report_prefix = 'k_fold_report'
    else:  # cross_site mode
        test_results_file = Path('k_fold/test_results/cross_site_metrics.csv')
        report_prefix = 'cross_site_report'
    
    output_dir = Path('k_fold/report')
    
    # Manual wkhtmltopdf path (uncomment and modify if needed)
    # WKHTMLTOPDF_PATH = r"C:\wkhtmltopdf\bin\wkhtmltopdf.exe"
    WKHTMLTOPDF_PATH = None
    
    # Generate timestamped filenames
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    html_filename = f'{report_prefix}_{timestamp}.html'
    pdf_filename = f'{report_prefix}_{timestamp}.pdf'
    
    output_html = output_dir / html_filename
    output_pdf = output_dir / pdf_filename
    
    print(f"Using following paths:")
    print(f"  Training analysis dir: {training_dir}")
    print(f"  Test results file: {test_results_file}")
    print(f"  Output directory: {output_dir}")
    print(f"  HTML output: {output_html}")
    print(f"  PDF output: {output_pdf}")
    
    # Check if required directories and files exist
    if not training_dir.exists():
        print(f"Error: Training analysis directory not found: {training_dir}")
        return 1
    
    if not test_results_file.exists():
        print(f"Error: Test results file not found: {test_results_file}")
        return 1
    
    try:
        # Generate HTML report
        create_html_report(training_dir, test_results_file, output_html, args.mode)
        print(f"\nHTML report generated successfully: {output_html}")
        
        # Generate PDF report
        if PDFKIT_AVAILABLE:
            # If manual path is specified, try to use it
            if WKHTMLTOPDF_PATH and Path(WKHTMLTOPDF_PATH).exists():
                print(f"Using manual wkhtmltopdf path: {WKHTMLTOPDF_PATH}")
                # Override the automatic detection in html_to_pdf function
                import pdfkit
                config = pdfkit.configuration(wkhtmltopdf=WKHTMLTOPDF_PATH)
                try:
                    options = {
                        'page-size': 'A4',
                        'margin-top': '0in',
                        'margin-right': '0in',
                        'margin-bottom': '0in',
                        'margin-left': '0in',
                        'encoding': "UTF-8",
                        'no-outline': None,
                        'enable-local-file-access': None
                    }
                    pdfkit.from_file(str(output_html), str(output_pdf), options=options, configuration=config)
                    print(f"PDF report generated successfully: {output_pdf}")
                except Exception as e:
                    print(f"PDF generation with manual path failed: {e}")
                    # Fall back to automatic detection
                    if html_to_pdf(output_html, output_pdf):
                        print(f"PDF report generated successfully: {output_pdf}")
                    else:
                        print("PDF generation failed")
            else:
                if html_to_pdf(output_html, output_pdf):
                    print(f"PDF report generated successfully: {output_pdf}")
                else:
                    print("PDF generation failed")
        else:
            print("PDF generation skipped (pdfkit not available)")
        
        print("\nReport generation completed successfully!")
        print(f"HTML report: {output_html}")
        if PDFKIT_AVAILABLE:
            print(f"PDF report: {output_pdf}")
        print("You can open the HTML file in a web browser to view the results.")
        return 0
    except Exception as e:
        print(f"\nError generating report: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == '__main__':
    exit(main()) 