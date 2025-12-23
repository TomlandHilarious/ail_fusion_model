#!/usr/bin/env python3
"""
Interactive LDA visualization of EEG data using Plotly
"""
import torch
import numpy as np
import pandas as pd
import os
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA

# Path to the cached datasets - same as in visualization.py
deep_attend_dataset_path = "/media/volume/sdb/ail_project/processed_eeg_fall_2024/dataset_cache/attend_deep/win_2.0/eeg_raw/excl_226_240_241_242_244_247_252_259_266_274_277/samples_86820a8a.pt"
shallow_attend_dataset_path = "/media/volume/sdb/ail_project/processed_eeg_fall_2024/dataset_cache/attend_shallow/win_2.0/eeg_raw/excl_226_240_241_242_244_247_252_259_266_274_277/samples_918ab9c7.pt"

def load_dataset(path):
    """Load dataset from cache"""
    print(f"Loading dataset from {path}")
    try:
        # Use the default loading method without weights_only for now
        # This will show a warning but will work with the existing dataset files
        data = torch.load(path)  
        print("Dataset loaded successfully")
        return data
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        return None

def interactive_lda_visualization(dataset, title="Interactive LDA Visualization", max_samples=10000, output_dir=None):
    """Create interactive LDA visualization with Plotly"""
    if not dataset or 'samples' not in dataset:
        print("No samples found in dataset")
        return None
    
    samples = dataset['samples']
    metadata = dataset.get('metadata', {})
    
    if not samples:
        print("Empty samples list")
        return None
    
    print(f"\nPreparing interactive LDA visualization for {title}...")
    
    # Randomly sample data if needed
    if len(samples) > max_samples:
        import random
        random.seed(42)  # For reproducibility
        samples = random.sample(samples, max_samples)
    
    # Extract EEG data and labels
    X_list = []
    y_list = []
    subject_list = []
    
    for sample in samples:
        eeg = sample.get('eeg')
        label = sample.get('label')
        subject = sample.get('sub')
        
        if eeg is not None and label is not None and subject is not None:
            # Convert label to integer if it's a tensor
            if torch.is_tensor(label):
                label = label.item()
            
            # Reshape EEG data for feature extraction
            if torch.is_tensor(eeg):
                eeg = eeg.cpu().numpy()  # Convert PyTorch tensor to numpy
                
            if isinstance(eeg, np.ndarray):
                if eeg.ndim == 3:  # (batch, channels, time)
                    eeg = np.squeeze(eeg, axis=0)  # Remove batch dimension if present
                
                # Make sure we have a consistent shape
                eeg = np.squeeze(eeg)            # (C, T)  e.g. (4, 512)
                vec = eeg.reshape(-1)            # 1‑D (C*T,)  = 2048  for Muse
                X_list.append(vec)               
                y_list.append(label)
                subject_list.append(subject)
            else:
                print(f"Skipping sample with unexpected EEG type: {type(eeg)}")
                continue
    
    # Convert to numpy arrays
    X = np.array(X_list)
    y = np.array(y_list)
    subjects = np.array(subject_list)
    
    # Apply PCA + LDA pipeline
    pipe = make_pipeline(
        StandardScaler(),
        PCA(n_components=100, random_state=0),   # keep top 100 PCs (tune!)
        LinearDiscriminantAnalysis(solver='svd')  # svd solver supports transform
    )
    X_lda = pipe.fit_transform(X, y)
    
    # Determine number of unique classes
    n_classes = len(np.unique(y))
    unique_subjects = np.unique(subjects)
    
    print(f"Performing LDA on {len(X)} samples with {X.shape[1]} features")
    print(f"Found {n_classes} unique classes and {len(unique_subjects)} unique subjects")
    print(f"Applied dimensionality reduction: {X.shape[1]} → {X_lda.shape[1]} features")
    
    # Check dimensionality of LDA output
    n_components = X_lda.shape[1]
    
    # If we only have 1 component (binary classification), add a second dimension
    # with small random jitter for better visualization
    if n_components == 1:
        print("Adding random jitter as second dimension for binary classification")
        np.random.seed(42)  # For reproducibility
        jitter = np.random.normal(0, 0.1, size=len(X_lda))
        X_lda = np.column_stack((X_lda, jitter))
    
    print(f"LDA transformation complete: {X_lda.shape}")
    
    # Create a DataFrame for easy plotting with Plotly
    df = pd.DataFrame({
        'LDA1': X_lda[:, 0],
        'LDA2': X_lda[:, 1] if X_lda.shape[1] > 1 else np.zeros(X_lda.shape[0]),
        'Class': ['Attending' if label == 1 else 'Not Attending' for label in y],
        'Subject': subjects
    })
    
    # Create interactive plots with Plotly
    
    # 1. Class separation plot
    fig1 = px.scatter(
        df, 
        x='LDA1', 
        y='LDA2', 
        color='Class',
        color_discrete_map={'Attending': 'blue', 'Not Attending': 'red'},
        title=f"{title}: Class Separation",
        labels={'LDA1': 'LDA Component 1', 'LDA2': 'LDA Component 2 (with jitter)'},
        hover_data=['Subject']
    )
    
    # 2. Subject clustering plot
    fig2 = px.scatter(
        df, 
        x='LDA1', 
        y='LDA2', 
        color='Subject',
        title=f"{title}: Subject Clustering",
        labels={'LDA1': 'LDA Component 1', 'LDA2': 'LDA Component 2 (with jitter)'},
        hover_data=['Class']
    )
    
    # Create a subplot figure combining both visualizations
    fig = make_subplots(
        rows=1, cols=2, 
        subplot_titles=(f"Class Separation", f"Subject Clustering"),
        horizontal_spacing=0.1
    )
    
    # Add traces from individual plots to the subplots
    for trace in fig1.data:
        fig.add_trace(trace, row=1, col=1)
    
    for trace in fig2.data:
        fig.add_trace(trace, row=1, col=2)
    
    # Update layout
    fig.update_layout(
        title_text=f"Interactive LDA Visualization - {title}",
        height=800,
        width=1600,
        legend_title_text='',
    )
    
    # Save interactive HTML file
    if output_dir:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        filename = title.lower().replace(' ', '_')
        output_path = os.path.join(output_dir, f"{filename}_interactive_lda.html")
        fig.write_html(output_path)
        print(f"Interactive visualization saved as {output_path}")
    
    return fig

if __name__ == "__main__":
    # Create output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"/media/volume/sdb/ail_project/visualizations/interactive_lda_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    print(f"Created output directory: {output_dir}")
    
    # Load datasets
    deep_attend_dataset = load_dataset(deep_attend_dataset_path)
    shallow_attend_dataset = load_dataset(shallow_attend_dataset_path)
    
    # Create visualizations
    print("\nPerforming interactive LDA visualization for Deep Attend dataset...")
    deep_fig = interactive_lda_visualization(deep_attend_dataset, "Deep Attend", output_dir=output_dir)
    
    print("\nPerforming interactive LDA visualization for Shallow Attend dataset...")
    shallow_fig = interactive_lda_visualization(shallow_attend_dataset, "Shallow Attend", output_dir=output_dir)
    
    print(f"\nAll interactive visualizations have been saved to: {output_dir}")
    print("Open the HTML files in a web browser to explore the visualizations interactively")
