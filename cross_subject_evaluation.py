#!/usr/bin/env python3
"""
Cross-Subject (Inter-Subject) Evaluation Script for EEG Attention Prediction

This script performs Leave-One-Subject-Out (LOSO) cross-validation where:
- Train on N-1 subjects
- Test on the held-out subject
- Repeat for all subjects

This tests the model's ability to generalize to completely unseen individuals.
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
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from modeling.eeg_modeling import EEGAttentionPredictor

def load_dataset(cache_path):
    """Load the cached dataset"""
    print(f"Loading dataset from {cache_path}...")
    
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        cache_data = torch.load(cache_path)
        
    if isinstance(cache_data, dict) and 'samples' in cache_data:
        samples = cache_data['samples']
    else:
        samples = cache_data
        
    print(f"Dataset loaded successfully with {len(samples)} samples")
    return samples

def get_subject_info(samples):
    """Extract subject information from samples"""
    subject_counts = defaultdict(int)
    for sample in samples:
        subject_id = sample['sub']
        subject_counts[subject_id] += 1
    
    subjects_by_count = sorted(subject_counts.items(), key=lambda x: x[1], reverse=True)
    print(f"Found {len(subjects_by_count)} unique subjects")
    
    return subjects_by_count, subject_counts

def cross_subject_evaluation(cache_path, test_subject_id, epochs=50, batch_size=32, 
                            lr=0.0001, patience=15, weight_decay=1e-4):
    """
    Perform cross-subject evaluation for a single held-out subject.
    
    Args:
        cache_path: Path to cached dataset
        test_subject_id: Subject ID to hold out for testing
        epochs: Number of training epochs
        batch_size: Batch size
        lr: Learning rate
        patience: Early stopping patience
        weight_decay: L2 regularization
        
    Returns:
        Dictionary with evaluation metrics
    """
    print(f"\n{'='*80}")
    print(f"Cross-Subject Evaluation: Testing on Subject {test_subject_id}")
    print(f"{'='*80}\n")
    
    # Load dataset
    samples = load_dataset(cache_path)
    
    # Create dataset wrapper
    class CachedDataset(torch.utils.data.Dataset):
        def __init__(self, samples):
            self.samples = samples
            
        def __len__(self):
            return len(self.samples)
            
        def __getitem__(self, idx):
            return self.samples[idx]
    
    dataset = CachedDataset(samples)
    
    # Split into train (all other subjects) and test (held-out subject)
    train_indices = []
    test_indices = []
    
    for idx, sample in enumerate(samples):
        if sample['sub'] == test_subject_id:
            test_indices.append(idx)
        else:
            train_indices.append(idx)
    
    print(f"Training samples: {len(train_indices)} (from {len(set([samples[i]['sub'] for i in train_indices]))} subjects)")
    print(f"Test samples: {len(test_indices)} (from subject {test_subject_id})")
    
    if len(test_indices) == 0:
        print(f"WARNING: No test samples found for subject {test_subject_id}")
        return None
    
    # Create train/test subsets
    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    test_dataset = torch.utils.data.Subset(dataset, test_indices)
    
    # Initialize model
    num_eeg_channels = dataset[0]['eeg'].shape[0]
    predictor = EEGAttentionPredictor(
        eeg_channels=num_eeg_channels,
        emb_size=64,
        depth=4,
        dropout=0.3
    )
    
    # Train the model (using within_subject=False for cross-subject)
    print("\nTraining model on all subjects except the test subject...")
    start_time = time.time()
    
    try:
        # For cross-subject, we need to manually handle the train/val split
        # We'll use a portion of the training subjects for validation
        
        # Get unique training subjects
        train_subjects = list(set([samples[i]['sub'] for i in train_indices]))
        
        # Split training subjects into train and validation (80/20)
        np.random.seed(42)
        np.random.shuffle(train_subjects)
        split_idx = int(len(train_subjects) * 0.8)
        train_subj = train_subjects[:split_idx]
        val_subj = train_subjects[split_idx:]
        
        # Create train and validation indices
        final_train_indices = [i for i in train_indices if samples[i]['sub'] in train_subj]
        final_val_indices = [i for i in train_indices if samples[i]['sub'] in val_subj]
        
        print(f"Training subjects: {len(train_subj)}")
        print(f"Validation subjects: {len(val_subj)}")
        print(f"Training samples: {len(final_train_indices)}")
        print(f"Validation samples: {len(final_val_indices)}")
        
        # Create final datasets
        final_train_dataset = torch.utils.data.Subset(dataset, final_train_indices)
        final_val_dataset = torch.utils.data.Subset(dataset, final_val_indices)
        
        # Manual training loop for cross-subject
        predictor.model.train()
        predictor.optimizer = torch.optim.AdamW(predictor.model.parameters(), lr=lr, weight_decay=weight_decay)
        predictor.criterion = torch.nn.CrossEntropyLoss()
        
        # Create data loaders
        train_loader = torch.utils.data.DataLoader(
            final_train_dataset, 
            batch_size=batch_size, 
            shuffle=True
        )
        val_loader = torch.utils.data.DataLoader(
            final_val_dataset, 
            batch_size=batch_size, 
            shuffle=False
        )
        
        best_val_acc = 0
        best_val_loss = float('inf')
        patience_counter = 0
        train_losses = []
        val_losses = []
        train_accs = []
        val_accs = []
        
        for epoch in range(epochs):
            # Training
            predictor.model.train()
            train_loss = 0
            train_correct = 0
            train_total = 0
            
            for batch in train_loader:
                eeg_data = batch['eeg'].to(predictor.device)
                labels = batch['label'].to(predictor.device)
                
                # Preprocess
                eeg_data = predictor.preprocess_eeg_data(eeg_data)
                
                # Forward
                _, outputs = predictor.model(eeg_data)
                loss = predictor.criterion(outputs, labels)
                
                # Backward
                predictor.optimizer.zero_grad()
                loss.backward()
                predictor.optimizer.step()
                
                # Metrics
                train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                train_total += labels.size(0)
                train_correct += (predicted == labels).sum().item()
            
            train_loss /= len(train_loader)
            train_acc = 100 * train_correct / train_total
            
            # Validation
            predictor.model.eval()
            val_loss = 0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for batch in val_loader:
                    eeg_data = batch['eeg'].to(predictor.device)
                    labels = batch['label'].to(predictor.device)
                    
                    eeg_data = predictor.preprocess_eeg_data(eeg_data)
                    _, outputs = predictor.model(eeg_data)
                    loss = predictor.criterion(outputs, labels)
                    
                    val_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()
            
            val_loss /= len(val_loader)
            val_acc = 100 * val_correct / val_total
            
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            train_accs.append(train_acc)
            val_accs.append(val_acc)
            
            print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% - Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            
            # Early stopping
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_val_loss = val_loss
                best_train_acc = train_acc
                best_epoch = epoch + 1
                patience_counter = 0
                # Save best model
                best_model_state = predictor.model.state_dict().copy()
            else:
                patience_counter += 1
                
            if patience_counter >= patience:
                print(f"\nEarly stopping at epoch {epoch+1}")
                break
        
        # Restore best model
        predictor.model.load_state_dict(best_model_state)
        
        training_time = (time.time() - start_time) / 60
        
        print(f"\nTraining completed in {training_time:.2f} minutes")
        print(f"Best validation accuracy: {best_val_acc:.2f}% at epoch {best_epoch}")
        
    except Exception as e:
        print(f"ERROR during training: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    # Test on held-out subject
    print(f"\nTesting on held-out subject {test_subject_id}...")
    predictor.model.eval()
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False
    )
    
    test_correct = 0
    test_total = 0
    test_loss = 0
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for batch in test_loader:
            eeg_data = batch['eeg'].to(predictor.device)
            labels = batch['label'].to(predictor.device)
            
            eeg_data = predictor.preprocess_eeg_data(eeg_data)
            _, outputs = predictor.model(eeg_data)
            loss = predictor.criterion(outputs, labels)
            
            test_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item()
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    test_loss /= len(test_loader)
    test_acc = 100 * test_correct / test_total
    
    print(f"Test accuracy on subject {test_subject_id}: {test_acc:.2f}%")
    print(f"Test loss: {test_loss:.4f}")
    
    # Calculate additional metrics
    from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
    
    f1 = f1_score(all_labels, all_predictions, average='weighted')
    precision = precision_score(all_labels, all_predictions, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_predictions, average='weighted', zero_division=0)
    cm = confusion_matrix(all_labels, all_predictions)
    
    print(f"F1 Score: {f1:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"Confusion Matrix:\n{cm}")
    
    # Return results
    results = {
        'subject_id': test_subject_id,
        'test_accuracy': test_acc,
        'test_loss': test_loss,
        'test_f1': f1,
        'test_precision': precision,
        'test_recall': recall,
        'best_val_accuracy': best_val_acc,
        'best_val_loss': best_val_loss,
        'best_train_accuracy': best_train_acc,
        'best_epoch': best_epoch,
        'training_time_min': training_time,
        'train_samples': len(final_train_indices),
        'val_samples': len(final_val_indices),
        'test_samples': len(test_indices),
        'confusion_matrix': cm.tolist(),
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_accs': train_accs,
        'val_accs': val_accs
    }
    
    return results

def run_loso_evaluation(cache_path, output_dir, epochs=50, batch_size=32, 
                       max_subjects=None, min_samples=1000):
    """
    Run Leave-One-Subject-Out cross-validation.
    
    Args:
        cache_path: Path to cached dataset
        output_dir: Directory to save results
        epochs: Number of training epochs per fold
        batch_size: Batch size
        max_subjects: Maximum number of subjects to evaluate (None for all)
        min_samples: Minimum samples required per subject
    """
    # Create output directory
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join(output_dir, f"cross_subject_results_{timestamp}")
    os.makedirs(results_dir, exist_ok=True)
    
    print(f"Results will be saved to: {results_dir}")
    
    # Load dataset and get subject info
    samples = load_dataset(cache_path)
    subjects_by_count, subject_counts = get_subject_info(samples)
    
    # Filter subjects by minimum sample count
    eligible_subjects = [(subj, count) for subj, count in subjects_by_count 
                        if count >= min_samples]
    
    if max_subjects:
        eligible_subjects = eligible_subjects[:max_subjects]
    
    print(f"\nEvaluating {len(eligible_subjects)} subjects with LOSO cross-validation")
    print(f"Subjects: {[s[0] for s in eligible_subjects]}")
    
    # Run LOSO for each subject
    all_results = []
    
    for idx, (subject_id, sample_count) in enumerate(eligible_subjects):
        print(f"\n{'#'*80}")
        print(f"LOSO Fold {idx+1}/{len(eligible_subjects)}: Testing Subject {subject_id} ({sample_count} samples)")
        print(f"{'#'*80}")
        
        result = cross_subject_evaluation(
            cache_path=cache_path,
            test_subject_id=subject_id,
            epochs=epochs,
            batch_size=batch_size
        )
        
        if result:
            all_results.append(result)
            
            # Save individual result
            result_file = os.path.join(results_dir, f"subject_{subject_id}_results.json")
            with open(result_file, 'w') as f:
                # Convert numpy arrays to lists for JSON serialization
                json_result = {k: v.tolist() if isinstance(v, np.ndarray) else v 
                              for k, v in result.items()}
                json.dump(json_result, f, indent=2)
    
    # Compile summary statistics
    if all_results:
        summary = compile_summary(all_results, results_dir)
        print(f"\n{'='*80}")
        print("CROSS-SUBJECT EVALUATION SUMMARY")
        print(f"{'='*80}")
        print(summary)
        
        # Save summary
        summary_file = os.path.join(results_dir, "summary.txt")
        with open(summary_file, 'w') as f:
            f.write(summary)
        
        # Create summary CSV
        df = pd.DataFrame(all_results)
        csv_file = os.path.join(results_dir, "cross_subject_results.csv")
        df.to_csv(csv_file, index=False)
        print(f"\nResults saved to: {csv_file}")
    
    return all_results

def compile_summary(results, results_dir):
    """Compile summary statistics from all results"""
    test_accs = [r['test_accuracy'] for r in results]
    test_f1s = [r['test_f1'] for r in results]
    
    summary = f"""
Cross-Subject (LOSO) Evaluation Results
========================================

Number of subjects evaluated: {len(results)}

Test Accuracy Statistics:
  Mean: {np.mean(test_accs):.2f}%
  Std:  {np.std(test_accs):.2f}%
  Min:  {np.min(test_accs):.2f}%
  Max:  {np.max(test_accs):.2f}%
  Median: {np.median(test_accs):.2f}%

F1 Score Statistics:
  Mean: {np.mean(test_f1s):.4f}
  Std:  {np.std(test_f1s):.4f}
  Min:  {np.min(test_f1s):.4f}
  Max:  {np.max(test_f1s):.4f}

Per-Subject Results:
"""
    
    for r in results:
        summary += f"  Subject {r['subject_id']}: Test Acc = {r['test_accuracy']:.2f}%, F1 = {r['test_f1']:.4f}\n"
    
    return summary

def main():
    parser = argparse.ArgumentParser(description="Cross-Subject (LOSO) EEG Attention Evaluation")
    parser.add_argument("--cache-path", type=str,
                       default="/media/volume/sdb/ail_project/processed_eeg_fall_2024/dataset_cache/attend_deep/win_2.0/eeg_raw/excl_226_240_241_242_244_247_252_259_266_274_277/samples_86820a8a.pt",
                       help="Path to cached dataset")
    parser.add_argument("--output-dir", type=str,
                       default="/media/volume/sdb/ail_project/cross_subject_results",
                       help="Output directory for results")
    parser.add_argument("--epochs", type=int, default=50,
                       help="Number of training epochs per fold")
    parser.add_argument("--batch-size", type=int, default=32,
                       help="Batch size")
    parser.add_argument("--max-subjects", type=int, default=None,
                       help="Maximum number of subjects to evaluate (None for all)")
    parser.add_argument("--min-samples", type=int, default=1000,
                       help="Minimum samples required per subject")
    
    args = parser.parse_args()
    
    # Run LOSO evaluation
    results = run_loso_evaluation(
        cache_path=args.cache_path,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        max_subjects=args.max_subjects,
        min_samples=args.min_samples
    )
    
    print("\nCross-subject evaluation completed!")

if __name__ == "__main__":
    main()
