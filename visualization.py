import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter, defaultdict
from typing import Dict, List, Tuple, Any
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA
import os
from datetime import datetime

# Path to the cached datasets
deep_attend_dataset_path = "/media/volume/sdb/ail_project/processed_eeg_fall_2024/dataset_cache/attend_deep/win_2.0/eeg_raw/excl_226_240_241_242_244_247_252_259_266_274_277/samples_86820a8a.pt"
shallow_attend_dataset_path = "/media/volume/sdb/ail_project/processed_eeg_fall_2024/dataset_cache/attend_shallow/win_2.0/eeg_raw/excl_226_240_241_242_244_247_252_259_266_274_277/samples_918ab9c7.pt"

# Create output directory for visualizations with timestamp
def create_output_directory():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"/media/volume/sdb/ail_project/visualizations/lda_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    print(f"Created output directory: {output_dir}")
    return output_dir

def load_dataset(path):
    print(f"Loading dataset from {path}")
    try:
        data = torch.load(path)
        print("Dataset loaded successfully")
        return data
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None

def explore_dataset_structure(data):
    print("\n==== Dataset Structure =====")
    print(f"Type: {type(data)}")
    if isinstance(data, dict):
        print(f"Keys: {data.keys()}")
  
    if 'metadata' in data:
        metadata = data['metadata']
        print(f"\nMetadata keys: {list(metadata.keys())}")
        
    if 'samples' in data:
        samples = data['samples']
        print(f"\nSamples count: {len(samples)}")
        if samples:
            # Check the first sample
            first_sample = samples[0]
            print(f"Sample structure (first sample keys): {first_sample.keys()}")
            
            # Show the shape of each field
            for key, value in first_sample.items():
                if isinstance(value, torch.Tensor):
                    print(f"  {key}: tensor of shape {tuple(value.shape)}")
                else:
                    print(f"  {key}: {type(value)}")
    
    print("\n==== End Dataset Structure ====")

def extract_metadata(dataset):
    """Extract metadata from samples"""
    if not dataset or 'samples' not in dataset:
        return {}
    
    samples = dataset['samples']
    if not samples:
        return {}
    
    # Extract subject IDs and question periods
    subject_ids = []
    question_periods = []
    labels = []
    
    for sample in samples:
        if 'sub' in sample:
            subject_ids.append(sample['sub'])
        if 'question_period' in sample:
            question_periods.append(sample['question_period'])
        if 'label' in sample:
            labels.append(sample['label'].item() if isinstance(sample['label'], torch.Tensor) else sample['label'])
    
    # Count occurrences
    subject_counts = Counter(subject_ids)
    qp_counts = Counter(question_periods)
    label_counts = Counter(labels)
    
    # Group samples by subject
    samples_by_subject = defaultdict(list)
    for i, sample in enumerate(samples):
        if 'sub' in sample:
            samples_by_subject[sample['sub']].append(i)
    
    return {
        'subject_counts': subject_counts,
        'question_period_counts': qp_counts,
        'label_counts': label_counts,
        'samples_by_subject': dict(samples_by_subject),
        'total_samples': len(samples)
    }

def compare_datasets(deep_metadata, shallow_metadata):
    """Compare deep and shallow attend datasets"""
    print("\nDataset Comparison:")
    deep_subjects = set(deep_metadata.get('subject_counts', {}).keys())
    shallow_subjects = set(shallow_metadata.get('subject_counts', {}).keys())
    
    common_subjects = deep_subjects.intersection(shallow_subjects)
    only_deep = deep_subjects - shallow_subjects
    only_shallow = shallow_subjects - deep_subjects
    
    print(f"  Common subjects: {len(common_subjects)}")
    print(f"  Only in deep: {len(only_deep)}")
    print(f"  Only in shallow: {len(only_shallow)}")
    
    # Provide summary statistics instead of per-subject comparison
    if common_subjects:
        deep_counts = deep_metadata.get('subject_counts', {})
        shallow_counts = shallow_metadata.get('subject_counts', {})
        
        deep_sample_counts = [deep_counts.get(subject, 0) for subject in common_subjects]
        shallow_sample_counts = [shallow_counts.get(subject, 0) for subject in common_subjects]
        
        # Check if sample counts match across datasets
        counts_match = all(d == s for d, s in zip(deep_sample_counts, shallow_sample_counts))
        if counts_match:
            print("  Sample counts match exactly between deep and shallow datasets for all subjects")
        else:
            # Only show summary statistics if they don't match
            diff_counts = [d - s for d, s in zip(deep_sample_counts, shallow_sample_counts)]
            print(f"  Sample count differences: min={min(diff_counts)}, max={max(diff_counts)}, avg={sum(diff_counts)/len(diff_counts):.2f}")
    
    # Compare label distributions
    deep_label_counts = deep_metadata.get('label_counts', {})
    shallow_label_counts = shallow_metadata.get('label_counts', {})
    
    deep_total = sum(deep_label_counts.values())
    shallow_total = sum(shallow_label_counts.values())
    
    if deep_total > 0 and shallow_total > 0:
        print("\nLabel distribution comparison:")
        all_labels = sorted(set(list(deep_label_counts.keys()) + list(shallow_label_counts.keys())))
        for label in all_labels:
            deep_count = deep_label_counts.get(label, 0)
            shallow_count = shallow_label_counts.get(label, 0)
            deep_pct = deep_count / deep_total * 100 if deep_total > 0 else 0
            shallow_pct = shallow_count / shallow_total * 100 if shallow_total > 0 else 0
            print(f"  Label {label}: Deep={deep_count} ({deep_pct:.1f}%), Shallow={shallow_count} ({shallow_pct:.1f}%)")

def visualize_lda(dataset, title="LDA Visualization", max_samples=10000, output_dir=None):
    """Create LDA visualization to examine separation between classes"""
    if not dataset or 'samples' not in dataset:
        print("No samples found in dataset")
        return
    
    samples = dataset['samples']
    metadata = dataset.get('metadata', {})
    
    if not samples:
        print("Empty samples list")
        return
    
    print(f"\nPreparing LDA visualization for {title}...")
    
    # Extract EEG data and labels
    X_list = []
    y_list = []
    subject_list = []
    
    # Get channel names from metadata for better visualization
    eeg_cols = metadata.get('eeg_cols', [])
    eeg_channel_info = metadata.get('eeg_channel_info', {})

    # Sample count to avoid memory issues
    sample_count = min(len(samples), max_samples)
    indices = np.random.choice(len(samples), sample_count, replace=False)
    
    for idx in indices:
        sample = samples[idx]
        if 'eeg' in sample and 'label' in sample:
            eeg = sample['eeg']
            label = sample['label']
            subject = sample['sub'] if 'sub' in sample else 'unknown'
            
            if isinstance(eeg, torch.Tensor):
                eeg = eeg.numpy()
            if isinstance(label, torch.Tensor):
                label = label.item()
            
            # Reshape EEG data for feature extraction
            # We'll use flatten features of channels and time
            if eeg.ndim == 3:  # (batch, channels, time)
                eeg = eeg.squeeze(0)  # Remove batch dimension if present
            
            eeg = np.squeeze(eeg)            # (C, T)  e.g. (4, 512)
            vec = eeg.reshape(-1)            # 1‑D (C*T,)  = 2048  for Muse
            X_list.append(vec)               
            y_list.append(label)
            subject_list.append(subject)
    
    if not X_list:
        print("No valid samples with both EEG and labels found")
        return
    
    # Convert to numpy arrays
    X = np.array(X_list)
    y = np.array(y_list)
    subjects = np.array(subject_list)
    # apply pca
    pipe = make_pipeline(
        StandardScaler(),
        PCA(n_components=100, random_state=0),   # keep top 100 PCs (tune!)
        LinearDiscriminantAnalysis(solver='svd')  # svd solver supports transform
    )
    X_lda = pipe.fit_transform(X, y)

    # Determine number of unique classes
    n_classes = len(np.unique(y))
    
    print(f"Performing LDA on {len(X)} samples with {X.shape[1]} features")
    print(f"Found {n_classes} unique classes")
    print(f"Applied dimensionality reduction: {X.shape[1]} → {X_lda.shape[1]} features")
    
    # Check dimensionality of LDA output
    n_components = X_lda.shape[1]
    
    # If we only have 1 component (binary classification), add a second dimension
    # with small random jitter for better visualization
    if n_components == 1:
        print("Adding random jitter as second dimension for binary classification")
        # Add small random jitter as the second dimension
        rng = np.random.RandomState(42)  # For reproducibility
        jitter = rng.normal(0, 0.1, size=len(X_lda))
        X_lda = np.column_stack((X_lda, jitter))
        
    print(f"LDA transformation complete: {X_lda.shape}")
    
    # Plot the results
    plt.figure(figsize=(10, 8))
    
    # Plot by class
    classes = np.unique(y)
    colors = ['blue', 'red', 'green', 'purple', 'orange']  # Add more colors if needed
    
    # Get class names from metadata if available
    class_names = {}
    if n_classes == 2:  # Binary classification
        class_names = {0: "Not Attending", 1: "Attending"}
    
    for i, cls in enumerate(classes):
        mask = (y == cls)
        plt.scatter(X_lda[mask, 0], X_lda[mask, 1], 
                   c=colors[i % len(colors)], 
                   label=f'{class_names.get(cls, f"Class {cls}")}', 
                   alpha=0.7)
    
    # Add a title and labels
    plt.title(f'LDA Projection of EEG Data - {title}')
    plt.xlabel('LD1')
    plt.ylabel('LD2')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    # Save the plot
    output_path = os.path.join(output_dir, f"{title.lower().replace(' ', '_')}_lda_visualization.png")
    plt.savefig(output_path)
    plt.close()
    
    print(f"LDA visualization saved as {output_path}")
    
    # Create additional visualization showing LDA by subject
    plt.figure(figsize=(12, 10))
    
    # Get unique subjects and assign colors
    unique_subjects = np.unique(subjects)
    num_subjects = len(unique_subjects)
    
    # Use a colormap for a large number of subjects
    cmap = plt.cm.get_cmap('tab20' if num_subjects <= 20 else 'viridis', num_subjects)
    
    # Plot by subject
    for i, subj in enumerate(unique_subjects):
        mask = (subjects == subj)
        plt.scatter(X_lda[mask, 0], X_lda[mask, 1], 
                   color=cmap(i % num_subjects),
                   label=f'Subject {subj}', 
                   alpha=0.5,
                   marker='o',
                   s=30)
    
    # If too many subjects, limit the legend to avoid overcrowding
    if num_subjects > 20:
        plt.legend(loc='upper right', ncol=2, fontsize='small')
    else:
        plt.legend(loc='best', ncol=3, fontsize='small')
    
    plt.title(f'LDA Projection of EEG Data by Subject - {title}')
    plt.xlabel('LD1')
    plt.ylabel('LD2')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    # Save the plot
    output_path = os.path.join(output_dir, f"{title.lower().replace(' ', '_')}_lda_by_subject.png")
    plt.savefig(output_path)
    plt.close()
    
    print(f"LDA by subject visualization saved as {output_path}")
    
    # Return the LDA model and transformed data
    return pipe, X_lda, y

if __name__ == "__main__":
    # Create output directory for visualizations
    output_dir = create_output_directory()
    
    # Load datasets
    deep_attend_dataset = load_dataset(deep_attend_dataset_path)
    shallow_attend_dataset = load_dataset(shallow_attend_dataset_path)
    
    # Explore structure
    print("\n=== DEEP ATTEND DATASET ===")
    explore_dataset_structure(deep_attend_dataset)
    
    print("\n=== SHALLOW ATTEND DATASET ===")
    explore_dataset_structure(shallow_attend_dataset)
    
    # Extract metadata
    deep_metadata = extract_metadata(deep_attend_dataset)
    shallow_metadata = extract_metadata(shallow_attend_dataset)
    
    # Compare datasets
    compare_datasets(deep_metadata, shallow_metadata)
    
    # Create LDA visualizations for both datasets
    print("\nPerforming LDA visualization for Deep Attend dataset...")
    deep_lda, deep_lda_features, deep_labels = visualize_lda(deep_attend_dataset, "Deep Attend", output_dir=output_dir)
    
    print("\nPerforming LDA visualization for Shallow Attend dataset...")
    shallow_lda, shallow_lda_features, shallow_labels = visualize_lda(shallow_attend_dataset, "Shallow Attend", output_dir=output_dir)
    
    print(f"\nAll visualizations have been saved to: {output_dir}")
