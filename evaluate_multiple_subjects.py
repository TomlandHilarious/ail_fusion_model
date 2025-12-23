#!/usr/bin/env python3
"""
Multiple Subject Evaluation Script for EEG Attention Prediction

This script runs within-subject training for multiple subjects and compiles
performance metrics to generate a comprehensive evaluation report.
"""

import os
import sys
import time
import json
import datetime
import warnings
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from collections import defaultdict

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from modeling.eeg_modeling import EEGAttentionPredictor

def load_and_evaluate_cached_dataset(cache_path, batch_size=32, epochs=50, within_subject=False, test_subjects=None,
                               lr=0.001, use_class_weights=True, weight_decay=1e-4, data_augmentation=False,
                               patience=20, debug=False):
    """
    Load a cached dataset and train/evaluate the EEG attention predictor model.
    
    Args:
        cache_path: Path to the cached dataset
        batch_size: Batch size for training
        epochs: Number of training epochs
        lr: Learning rate
        weight_decay: L2 regularization strength
        use_class_weights: Whether to use class weights to handle imbalanced data
        data_augmentation: Whether to apply data augmentation techniques
        patience: Early stopping patience (default increased from 10 to 20 for extended training)
        debug: Whether to print debug information
        
    Returns:
        Trained EEGAttentionPredictor instance
    """
    print(f"\n{'='*60}\nLoading cached dataset from: {cache_path}\n{'='*60}")
    
    if not os.path.exists(cache_path):
        raise FileNotFoundError(f"Cache file not found: {cache_path}")
    
    # Load the cached dataset
    print(f"Loading cached dataset from {cache_path}...")
    try:
        # Load the data
        print("Attempting to load the dataset file...")
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning, message="torch.load with pickle")
            cache_data = torch.load(cache_path)
            
        # Check if this is directly a list of samples or a dictionary with 'samples' key
        if isinstance(cache_data, dict) and 'samples' in cache_data:
            samples = cache_data['samples']
        else:
            samples = cache_data
            
        print(f"Dataset loaded successfully with {len(samples)} samples")
        
        # Create a simple dataset from the cached samples
        class CachedDataset(torch.utils.data.Dataset):
            def __init__(self, samples):
                self.samples = samples
                
            def __len__(self):
                return len(self.samples)
                
            def __getitem__(self, idx):
                return self.samples[idx]
        
        # Create dataset instance
        dataset = CachedDataset(samples)
        
        # Initialize the predictor
        num_eeg_channels = dataset[0]['eeg'].shape[0]  # Get number of channels from the first sample
        predictor = EEGAttentionPredictor(eeg_channels=num_eeg_channels)
        
        # Train the model with proper subject and question period splitting
        print("\nStarting training with proper subject and question period separation...")
        try:
            # Train with extended parameters
            history = predictor.train(
                dataset=dataset,
                batch_size=batch_size,
                epochs=epochs,  # Use the extended epochs parameter
                lr=lr,
                patience=patience,  # Use the increased patience parameter
                weight_decay=weight_decay,
                use_scheduler=True,
                within_subject=within_subject,
                test_subjects=test_subjects  # This ensures no data leakage between train/val sets
            )
        except Exception as e:
            print(f"ERROR during training: {e}")
            import traceback
            traceback.print_exc()
            raise
        
        # Plot training history
        predictor.plot_training_history()
        
        return predictor
        
    except Exception as e:
        print(f"ERROR loading dataset: {e}")
        import traceback
        traceback.print_exc()
        return None

def parse_arguments():
    parser = argparse.ArgumentParser(description="Evaluate EEG attention prediction across multiple subjects")
    parser.add_argument("--cache-path", type=str, 
                        default="/media/volume/sdb/ail_project/processed_eeg_fall_2024/dataset_cache/attend_deep/win_2.0/eeg_raw/excl_226_240_241_242_244_247_252_259_266_274_277/samples_86820a8a.pt",
                        help="Path to the cached dataset file")
    parser.add_argument("--num-subjects", type=int, default=0, 
                        help="Number of subjects to evaluate (0 for all subjects)")
    parser.add_argument("--epochs", type=int, default=100,
                        help="Number of training epochs per subject")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Batch size for training")
    parser.add_argument("--output-dir", type=str, default="./subject_evaluation_results",
                        help="Directory to save results")
    parser.add_argument("--min-samples", type=int, default=1000,
                        help="Minimum number of samples required to include a subject")
    return parser.parse_args()

def setup_output_directory(output_dir):
    """Create output directory if it doesn't exist"""
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = os.path.join(output_dir, f"subject_results_{timestamp}")
    os.makedirs(results_path, exist_ok=True)
    return results_path

def load_and_analyze_dataset(cache_path):
    """Load the dataset and identify subjects with the most samples"""
    print(f"Loading dataset from {cache_path}...")
    
    # Load dataset
    if cache_path.endswith('.pt'):
        import torch
        cache_data = torch.load(cache_path)
        if isinstance(cache_data, dict) and 'samples' in cache_data:
            samples = cache_data['samples']
        else:
            samples = cache_data
    else:
        raise ValueError(f"Unsupported cache format: {cache_path}")
    
    # Count samples per subject
    subject_counts = defaultdict(int)
    for sample in samples:
        subject_id = sample['sub']
        subject_counts[subject_id] += 1
    
    # Sort subjects by sample count (descending)
    subjects_by_count = sorted(subject_counts.items(), key=lambda x: x[1], reverse=True)
    print(f"Found {len(subjects_by_count)} unique subjects in the dataset")
    
    return samples, subjects_by_count

def evaluate_subjects(args, results_path):
    """Run evaluation for multiple subjects and compile results"""
    # Load dataset and get subjects with most samples
    samples, subjects_by_count = load_and_analyze_dataset(args.cache_path)
    
    # Select subjects based on criteria
    if args.num_subjects > 0:
        # Select top N subjects
        filtered_subjects = subjects_by_count[:args.num_subjects]
    else:
        # Select all subjects with minimum sample count
        filtered_subjects = [(subject, count) for subject, count in subjects_by_count 
                            if count >= args.min_samples]
    
    selected_subjects = [subject for subject, _ in filtered_subjects]
    print(f"Selected {len(selected_subjects)} subjects for evaluation")
    print(f"Subject IDs: {selected_subjects}")
    
    # Prepare results storage
    results = {
        "subject_id": [],
        "sample_count": [],
        "best_val_accuracy": [],
        "best_val_loss": [],
        "train_accuracy": [],
        "generalization_gap": [],
        "convergence_epoch": [],
        "final_epoch": [],
        "early_stopping_triggered": [],
        "late_improvement": [],
        "accuracy_at_epoch_40": [],
        "improvement_after_40": [],
        "training_time_min": [],
        "model_path": []  # Added to track model paths for reference
    }
    
    # Create a models directory within results_path
    models_dir = os.path.join(results_path, "individual_models")
    os.makedirs(models_dir, exist_ok=True)
    print(f"Models will be saved to: {models_dir}")
    
    # Run evaluation for each subject
    for subject_idx, (subject_id, sample_count) in enumerate(filtered_subjects):
        print(f"\n===== Evaluating Subject {subject_id} ({sample_count} samples) - {subject_idx+1}/{len(filtered_subjects)} =====")
        
        start_time = time.time()
        
        # Train individual EEG conformer for this subject with extended parameters
        predictor = load_and_evaluate_cached_dataset(
            cache_path=args.cache_path,
            batch_size=args.batch_size,
            epochs=args.epochs,
            within_subject=True,
            test_subjects=[subject_id],  # This will make the subject be selected for within-subject training
            patience=20  # Increased patience for extended training
        )
        
        training_time = (time.time() - start_time) / 60  # in minutes
        
        # Extract results
        if hasattr(predictor, 'best_metrics') and predictor.best_metrics:
            # Basic metrics
            results["subject_id"].append(subject_id)
            results["sample_count"].append(sample_count)
            best_val_acc = predictor.best_metrics.get("val_accuracy", float('nan'))
            best_train_acc = predictor.best_metrics.get("train_accuracy", float('nan'))
            results["best_val_accuracy"].append(best_val_acc)
            results["best_val_loss"].append(predictor.best_metrics.get("val_loss", float('nan')))
            results["train_accuracy"].append(best_train_acc)
            results["generalization_gap"].append(best_val_acc - best_train_acc)  # Fix: Calculate generalization gap as val_acc - train_acc
            
            # Convergence metrics
            convergence_epoch = predictor.best_metrics.get("epoch", float('nan'))
            results["convergence_epoch"].append(convergence_epoch)
            results["final_epoch"].append(predictor.epoch)
            results["early_stopping_triggered"].append(predictor.epoch < args.epochs)
            results["training_time_min"].append(training_time)
            
            # Extended training analysis metrics
            late_improvement = convergence_epoch > 40 if not np.isnan(convergence_epoch) else False
            results["late_improvement"].append(late_improvement)
            
            # If we have the detailed history of validation accuracies
            if hasattr(predictor, 'val_accs') and len(predictor.val_accs) > 40:
                # Get accuracy at epoch 40 (index 39)
                acc_at_40 = predictor.val_accs[39]
                results["accuracy_at_epoch_40"].append(acc_at_40)
                # Calculate improvement after epoch 40
                results["improvement_after_40"].append(best_val_acc - acc_at_40)
            else:
                results["accuracy_at_epoch_40"].append(float('nan'))
                results["improvement_after_40"].append(float('nan'))
            
            # Save individual subject model with structured directory
            model_path = os.path.join(models_dir, f"subject_{subject_id}_model.pt")
            torch.save(predictor.model.state_dict(), model_path)
            results["model_path"].append(model_path)
            
            # Save learning curves for this subject
            if hasattr(predictor, 'train_losses') and hasattr(predictor, 'val_losses'):
                plt.figure(figsize=(15, 10))
                
                # Plot losses
                plt.subplot(2, 2, 1)
                plt.plot(predictor.train_losses, label='Training Loss')
                plt.plot(predictor.val_losses, label='Validation Loss')
                plt.axvline(x=convergence_epoch, color='r', linestyle='--', 
                           label=f'Best Epoch: {convergence_epoch}')
                if len(predictor.train_losses) > 40:
                    plt.axvline(x=40, color='g', linestyle=':', 
                               label='Epoch 40 (Standard Training)')
                plt.title(f'Subject {subject_id} - Loss Curves')
                plt.xlabel('Epoch')
                plt.ylabel('Loss')
                plt.legend()
                
                # Plot accuracies
                plt.subplot(2, 2, 2)
                plt.plot(predictor.train_accs, label='Training Accuracy')
                plt.plot(predictor.val_accs, label='Validation Accuracy')
                plt.axvline(x=convergence_epoch, color='r', linestyle='--')
                if len(predictor.train_accs) > 40:
                    plt.axvline(x=40, color='g', linestyle=':')
                plt.title(f'Subject {subject_id} - Accuracy Curves')
                plt.xlabel('Epoch')
                plt.ylabel('Accuracy')
                plt.legend()
                
                # Add extended training zoom plot if we have enough epochs
                if len(predictor.val_accs) > 40:
                    plt.subplot(2, 2, 3)
                    # Zoom in on the extended part of training (after epoch 40)
                    x_extended = list(range(40, len(predictor.val_accs)))
                    plt.plot(x_extended, predictor.train_losses[40:], label='Training Loss')
                    plt.plot(x_extended, predictor.val_losses[40:], label='Validation Loss')
                    if convergence_epoch >= 40:
                        plt.axvline(x=convergence_epoch, color='r', linestyle='--')
                    plt.title('Extended Training Period (Epochs 40+)')
                    plt.xlabel('Epoch')
                    plt.ylabel('Loss')
                    plt.legend()
                    
                    plt.subplot(2, 2, 4)
                    plt.plot(x_extended, predictor.train_accs[40:], label='Training Accuracy')
                    plt.plot(x_extended, predictor.val_accs[40:], label='Validation Accuracy')
                    if convergence_epoch >= 40:
                        plt.axvline(x=convergence_epoch, color='r', linestyle='--')
                    plt.title('Extended Training Accuracy (Epochs 40+)')
                    plt.xlabel('Epoch')
                    plt.ylabel('Accuracy')
                    plt.legend()
                
                plt.tight_layout()
                plt.savefig(os.path.join(results_path, f"subject_{subject_id}_learning_curves.png"))
                plt.close()
                
                # Save raw metrics data for potential further analysis
                metrics_data = {
                    'train_loss': predictor.train_losses,
                    'val_loss': predictor.val_losses,
                    'train_acc': predictor.train_accs,
                    'val_acc': predictor.val_accs,
                    'best_epoch': convergence_epoch,
                    'final_epoch': predictor.epoch
                }
                with open(os.path.join(results_path, f"subject_{subject_id}_metrics.json"), 'w') as f:
                    json.dump(metrics_data, f)
        else:
            print(f"Warning: No metrics available for subject {subject_id}")
    
    return results

def generate_report(results, results_path):
    """Generate a comprehensive report with tables and visualizations"""
    # Convert results to DataFrame
    df = pd.DataFrame(results)
    
    # Sort by validation accuracy (descending)
    df = df.sort_values(by='best_val_accuracy', ascending=False)
    
    # Calculate summary statistics
    summary = {
        "mean": df.mean(),
        "std": df.std(),
        "min": df.min(),
        "max": df.max(),
        "median": df.median()
    }
    
    # Save results to CSV
    df.to_csv(os.path.join(results_path, "subject_results.csv"), index=False)
    
    # Create summary table
    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(os.path.join(results_path, "summary_statistics.csv"))
    
    # Generate markdown report
    generate_markdown_report(df, summary, results_path)
    
    # Generate visualizations
    plt.figure(figsize=(12, 8))
    
    # Accuracy comparison plot
    plt.subplot(2, 2, 1)
    plt.bar(df["subject_id"].astype(str), df["best_val_accuracy"] * 100)
    plt.axhline(y=50, color='r', linestyle='--', label='Chance level')
    plt.title("Validation Accuracy by Subject")
    plt.xlabel("Subject ID")
    plt.ylabel("Accuracy (%)")
    plt.legend()
    plt.ylim([0, 100])
    
    # Generalization gap plot
    plt.subplot(2, 2, 2)
    plt.bar(df["subject_id"].astype(str), df["generalization_gap"] * 100)
    plt.title("Generalization Gap by Subject")
    plt.xlabel("Subject ID")
    plt.ylabel("Gap (Val - Train) (%)")
    plt.axhline(y=0, color='r', linestyle='--')
    
    # Sample count vs accuracy
    plt.subplot(2, 2, 3)
    plt.scatter(df["sample_count"], df["best_val_accuracy"] * 100)
    plt.title("Accuracy vs Sample Count")
    plt.xlabel("Number of Samples")
    plt.ylabel("Validation Accuracy (%)")
    
    # Convergence speed
    plt.subplot(2, 2, 4)
    plt.bar(df["subject_id"].astype(str), df["convergence_epoch"])
    plt.title("Convergence Speed by Subject")
    plt.xlabel("Subject ID")
    plt.ylabel("Best Epoch")
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_path, "subject_comparison.png"))
    
    # Generate HTML report
    html_report = f"""
    <html>
    <head>
        <title>EEG Attention Prediction - Subject Analysis Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; }}
            table {{ border-collapse: collapse; width: 100%; margin-bottom: 30px; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
            tr:nth-child(even) {{ background-color: #f9f9f9; }}
            h1, h2 {{ color: #333; }}
            .summary {{ background-color: #e7f3fe; padding: 15px; margin-bottom: 20px; border-left: 6px solid #2196F3; }}
        </style>
    </head>
    <body>
        <h1>EEG Attention Prediction - Subject Analysis Report</h1>
        <p>Generated on {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
        
        <div class="summary">
            <h2>Summary Statistics</h2>
            <p>Average validation accuracy: {summary_stats['val_accuracy_mean']:.2f}% (± {summary_stats['val_accuracy_std']:.2f}%)</p>
            <p>Best subject accuracy: {summary_stats['val_accuracy_max']:.2f}%</p>
            <p>Average generalization gap: {summary_stats['gen_gap_mean']:.2f}%</p>
            <p>Average convergence epoch: {summary_stats['convergence_epoch_mean']:.1f}</p>
        </div>
        
        <h2>Individual Subject Results</h2>
        <table>
            <tr>
                <th>Subject ID</th>
                <th>Sample Count</th>
                <th>Val Accuracy (%)</th>
                <th>Train Accuracy (%)</th>
                <th>Generalization Gap (%)</th>
                <th>Best Epoch</th>
            </tr>
    """
    
    for _, row in df.iterrows():
        html_report += f"""
            <tr>
                <td>{row['subject_id']}</td>
                <td>{row['sample_count']}</td>
                <td>{row['best_val_accuracy']*100:.2f}</td>
                <td>{row['train_accuracy']*100:.2f}</td>
                <td>{row['generalization_gap']*100:.2f}</td>
                <td>{row['convergence_epoch']}</td>
            </tr>
        """
    
    html_report += """
        </table>
        
        <h2>Visualizations</h2>
        <img src="subject_comparison.png" alt="Subject Comparisons" style="width:100%; max-width:1000px;">
        
        <h2>Analysis and Recommendations</h2>
        <p>The analysis shows considerable variability in prediction performance across subjects.
           This supports the use of within-subject training as a strategy to address individual differences in EEG patterns.</p>
        
        <p>Factors that may influence prediction accuracy:</p>
        <ul>
            <li>Sample count - more data generally leads to better models</li>
            <li>Signal quality - some subjects may have cleaner EEG signals</li>
            <li>Individual differences in attention-related neural signatures</li>
            <li>Temporal stability of attention patterns within a session</li>
        </ul>
        
        <p>Recommendations for further improvement:</p>
        <ul>
            <li>Enhanced preprocessing to reduce noise and artifacts</li>
            <li>Feature engineering focused on known attention-related EEG patterns</li>
            <li>Subject-specific hyperparameter tuning</li>
            <li>Model architecture optimizations for EEG time-series data</li>
            <li>Ensemble methods combining predictions across temporal windows</li>
        </ul>
    </body>
    </html>
    """
    
    with open(os.path.join(results_path, "report.html"), "w") as f:
        f.write(html_report)
    
    print(f"\nReport generated and saved to {results_path}")
    print(f"Summary: Average validation accuracy: {summary['mean']['best_val_accuracy']*100:.2f}% (± {summary['std']['best_val_accuracy']*100:.2f}%)")
    print(f"Markdown report saved to {os.path.join(results_path, 'eeg_performance_report.md')}")

def generate_markdown_report(df, summary, results_path):
    """Generate a detailed markdown report of subject performance and save as README.md"""
    # Generate the report content
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    md_content = f"""# EEG Attention Prediction - Individual Subject Conformers

This report summarizes the performance of individual EEG conformers trained for each subject ({len(df)} subjects total) with extended training epochs.

**Generated on:** {timestamp}

## Summary Statistics

| Metric | Mean | Std | Min | Max | Median |
| ------ | ---- | --- | --- | --- | ------ |
| Validation Accuracy (%) | {summary_stats['val_accuracy_mean']*100:.2f} | {summary_stats['val_accuracy_std']*100:.2f} | {summary_stats['val_accuracy_min']*100:.2f} | {summary_stats['val_accuracy_max']*100:.2f} | {summary_stats['val_accuracy_median']*100:.2f} |
| Validation Loss | {summary_stats['val_loss_mean']:.4f} | {summary_stats['val_loss_std']:.4f} | {summary_stats['val_loss_min']:.4f} | {summary_stats['val_loss_max']:.4f} | {summary_stats['val_loss_median']:.4f} |
| Training Accuracy (%) | {summary_stats['train_accuracy_mean']*100:.2f} | {summary_stats['train_accuracy_std']*100:.2f} | {summary_stats['train_accuracy_min']*100:.2f} | {summary_stats['train_accuracy_max']*100:.2f} | {summary_stats['train_accuracy_median']*100:.2f} |
| Generalization Gap (%) | {summary_stats['gen_gap_mean']*100:.2f} | {summary_stats['gen_gap_std']*100:.2f} | {summary_stats['gen_gap_min']*100:.2f} | {summary_stats['gen_gap_max']*100:.2f} | {summary_stats['gen_gap_median']*100:.2f} |
| Best Epoch | {summary_stats['convergence_epoch_mean']:.1f} | {summary_stats['convergence_epoch_std']:.1f} | {summary_stats['convergence_epoch_min']} | {summary_stats['convergence_epoch_max']} | {summary_stats['convergence_epoch_median']} |
| Training Time (min/subject) | {summary_stats['training_time_mean']:.2f} | {summary_stats['training_time_std']:.2f} | {summary_stats['training_time_min']:.2f} | {summary_stats['training_time_max']:.2f} | {summary_stats['training_time_median']:.2f} |

## Individual Subject Performance

Results are sorted by validation accuracy (descending):

| Subject ID | Sample Count | Val Accuracy (%) | Train Accuracy (%) | Gen. Gap (%) | Best Epoch | Early Stop | Late Improve |
| ---------- | ------------ | --------------- | ----------------- | ------------ | ---------- | ---------- | ----------- |
"""

    # Add rows for each subject
    for _, row in df.iterrows():
        early_stop = "Yes" if row['early_stopping_triggered'] else "No"
        late_improve = "Yes" if row['late_improvement'] else "No"
        md_content += f"| {row['subject_id']} | {row['sample_count']} | {row['best_val_accuracy']*100:.2f} | {row['train_accuracy']*100:.2f} | {row['generalization_gap']*100:.2f} | {row['convergence_epoch']} | {early_stop} | {late_improve} |\n"

    # Add analysis of extended training benefits
    late_improvement_subjects = df[df['late_improvement'] == True]
    percentage_late_improve = (len(late_improvement_subjects) / len(df) * 100) if len(df) > 0 else 0
    
    # Calculate average improvement from extended training
    improvement_after_40_avg = df['improvement_after_40'].mean() * 100  # convert to percentage
    
    md_content += f"""

## Extended Training Analysis

### Late Improvement Patterns
- {len(late_improvement_subjects)} out of {len(df)} subjects ({percentage_late_improve:.1f}%) showed best performance after epoch 40
- Average accuracy improvement after epoch 40: {improvement_after_40_avg:.2f}%
- Subject(s) with most improvement from extended training:
"""

    # Add details for subjects with significant late improvements
    # Sort by improvement_after_40 and take top 3 or fewer if not enough data
    top_improvers = df.sort_values('improvement_after_40', ascending=False)
    top_n = min(3, len(top_improvers))
    
    for i in range(top_n):
        if top_improvers.iloc[i]['improvement_after_40'] > 0:
            subject = top_improvers.iloc[i]['subject_id']
            epoch = top_improvers.iloc[i]['convergence_epoch']
            acc_at_40 = top_improvers.iloc[i]['accuracy_at_epoch_40'] * 100
            final_acc = top_improvers.iloc[i]['best_val_accuracy'] * 100
            improvement = top_improvers.iloc[i]['improvement_after_40'] * 100
            md_content += f"  - Subject {subject}: Improved from {acc_at_40:.2f}% at epoch 40 to {final_acc:.2f}% at epoch {epoch} (+{improvement:.2f}%)\n"

    # Add analysis section with focus on individual models
    md_content += f"""

## Individual Subject Model Analysis

### Top Performing Models

The following subjects achieved the highest validation accuracy with their individual conformers:

| Rank | Subject ID | Validation Accuracy (%) | Training Accuracy (%) | Convergence Epoch | Samples |
| ---- | ---------- | ---------------------- | -------------------- | ----------------- | ------- |
"""    
    
    # Add top performing models table content
    # Get top 5 subjects by validation accuracy
    top_performers = df.sort_values('best_val_accuracy', ascending=False).head(5)
    for i, (_, row) in enumerate(top_performers.iterrows()):
        md_content += f"| {i+1} | {row['subject_id']} | {row['best_val_accuracy']*100:.2f} | {row['train_accuracy']*100:.2f} | {row['convergence_epoch']} | {row['sample_count']} |\n"
    
    # Continue with general performance analysis
    md_content += f"""

### Model Location
Individual EEG conformer models for each subject are saved in the `individual_models` directory. Each model is named `subject_XXX_model.pt`.

## General Performance Analysis

### Performance Patterns
- The best performing subject reached {summary['max']['best_val_accuracy']*100:.2f}% validation accuracy
- Average validation accuracy across subjects: {summary['mean']['best_val_accuracy']*100:.2f}% (± {summary['std']['best_val_accuracy']*100:.2f}%)
- {(df['early_stopping_triggered'] == True).sum()} subjects triggered early stopping before completing all training epochs

### Model Convergence
- Average convergence epoch: {summary['mean']['convergence_epoch']:.1f}
- Fastest converging subject reached best validation performance at epoch {summary['min']['convergence_epoch']}
- Slowest converging subject reached best validation performance at epoch {summary['max']['convergence_epoch']}

### Data and Training Patterns
- Average number of samples per subject: {summary['mean']['sample_count']:.1f} (± {summary['std']['sample_count']:.1f})
- Correlation between sample count and validation accuracy: {df[['sample_count', 'best_val_accuracy']].corr().iloc[0,1]:.3f}
- Average training time per subject: {summary['mean']['training_time_min']:.2f} minutes

## Extended Training Conclusions

1. **Benefits of Longer Training**: {percentage_late_improve:.1f}% of subjects benefited from training beyond 40 epochs, with an average improvement of {improvement_after_40_avg:.2f}%.

2. **Computational Cost vs. Benefit**: Extended training increased computation time by approximately {summary['mean']['training_time_min']:.2f} minutes per subject, which {"appears justified" if improvement_after_40_avg > 1.0 else "may not be justified"} given the observed performance improvements.

3. **Subject Variability**: Subjects show high variability in when they achieve peak performance, suggesting that adaptive epoch counts based on validation curves may be more efficient than fixed epoch counts.

4. **Individual vs. Group Models**: Training individual EEG conformers for each subject allows for personalized models that can adapt to unique EEG patterns of each participant.

## Recommendations

1. **Architecture Optimization**: The current model achieves modest performance. Consider testing alternative architectures (EEGNet, DeepConvNet) specifically designed for EEG data.

2. **Feature Engineering**: Explore frequency-domain features or advanced signal processing techniques to extract more meaningful patterns from raw EEG.

3. **Adaptive Training Duration**: Implement dynamic early stopping strategies that adapt to individual subject learning curves.

4. **Hyperparameter Tuning**: Run systematic grid search for optimal learning rates, batch sizes, and model dimensions for each individual subject model.

5. **Data Quality**: Implement more aggressive artifact removal and signal filtering to improve signal-to-noise ratio.

6. **Ensemble Methods**: Create an ensemble using the individual subject models to see if combining their predictions improves overall performance.

7. **Transfer Learning**: Explore using pre-trained models from high-performing subjects as starting points for subjects with less data.

## Learning Curves

Learning curve plots for each subject are available in the results directory. These include extended training analysis with separate visualization of performance after epoch 40.
"""

    # Save both as a performance report and as README.md for easier viewing in repositories
    with open(os.path.join(results_path, 'eeg_performance_report.md'), 'w') as f:
        f.write(md_content)
    
    # Also save as README.md for easy viewing
    with open(os.path.join(results_path, 'README.md'), 'w') as f:
        f.write(md_content)

def main():
    args = parse_arguments()
    results_path = setup_output_directory(args.output_dir)
    
    start_time = time.time()
    print(f"Starting individual subject EEG conformer training at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Training with {args.epochs} epochs per subject")
    
    # Run evaluations with individual models for each subject
    results = evaluate_subjects(args, results_path)
    
    # Generate comprehensive report with individual and summary statistics
    generate_report(results, results_path)
    
    elapsed_time = time.time() - start_time
    print(f"Training and evaluation completed in {elapsed_time/60:.2f} minutes")
    print(f"Results and individual models saved to: {results_path}")

if __name__ == "__main__":
    main()
