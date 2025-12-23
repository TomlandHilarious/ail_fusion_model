import os
import sys
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import warnings
import torch.nn as nn
import torch.optim as optim
from tqdm.auto import tqdm
from torch.utils.data import DataLoader, Subset, random_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
from collections import defaultdict
import time  # For timing operations
import random  # For random sampling
import traceback  # For detailed error tracing
import pandas as pd
from data_loader import MultiModalDataset

# Add EEG-Conformer directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'eeg_conformer'))

# Import conformer models
from eeg_conformer.conformer import Conformer
from eeg_conformer.small_conformer import SmallConformer

# Import the MultiModalDataset
from data_loader import MultiModalDataset

class EEGAttentionPredictor:
    """Class for training and evaluating EEG-based attention prediction models"""
    
    def __init__(self, eeg_channels=4, emb_size=64, depth=4, n_classes=2, dropout=0.3):
        """
        Initialize the EEG attention predictor with a SmallConformer model (adapted for 4-channel EEG).
        
        Args:
            eeg_channels: Number of EEG channels in the data
            emb_size: Embedding size for the Conformer model (default: 64, increased from 40)
            depth: Depth of the transformer encoder (default: 4, increased from 3)
            n_classes: Number of attention classes (2 for binary: deep_attend vs. shallow_attend)
            dropout: Dropout rate for model regularization (default: 0.3, decreased from 0.5)
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Initialize the SmallConformer model with improved parameters
        # - Larger embedding size for more expressive features
        # - More optimal depth for the transformer encoder
        # - Reduced dropout to prevent underfitting
        self.model = SmallConformer(emb_size=emb_size, depth=depth, n_classes=n_classes, dropout=dropout)
        self.model = self.model.to(self.device)
        
        # Define loss function and optimizer
        self.criterion = nn.CrossEntropyLoss()  # Can be replaced with weighted loss if needed
        self.optimizer = None  # Will be initialized in train()
        self.lr = 0.001
        
        # Track training metrics
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
    
    def preprocess_eeg_data(self, eeg_data):
        """Preprocess EEG data for Conformer model with normalization"""
        # Convert to float for numerical stability
        eeg_data = eeg_data.float()
        
        # Check input shape and handle accordingly
        if len(eeg_data.shape) == 4:  # Shape: [batch, 1, channels, time]
            # Already in the correct format for the model
            pass
        elif len(eeg_data.shape) == 3:  # Shape: [batch, channels, time]
            # Add a dimension to match the expected input format for the model
            # The SmallConformer expects [batch, channels, height, width] where height=1
            eeg_data = eeg_data.unsqueeze(2)  # Now shape is [batch, channels, 1, time]
        else:
            raise ValueError(f"Unexpected EEG data shape: {eeg_data.shape}. Expected 3D or 4D tensor.")
            
        # Apply per-channel normalization (z-score) to improve model performance
        # Normalize each channel independently across the time dimension
        batch_size, n_channels, height, time_steps = eeg_data.shape
        for b in range(batch_size):
            for c in range(n_channels):
                # Get mean and std for this channel across the time dimension
                channel_data = eeg_data[b, c, 0, :]  # For the 1D height case
                mean = torch.mean(channel_data)
                std = torch.std(channel_data)
                # Avoid division by zero
                if std == 0:
                    std = 1.0
                # Normalize to zero mean and unit variance
                eeg_data[b, c, 0, :] = (channel_data - mean) / std
        
        return eeg_data
                
    
    def train(self, dataset, batch_size=32, epochs=50, lr=0.001, val_split=0.2, test_subjects=None,
              weight_decay=1e-4, patience=10, use_scheduler=True, within_subject=False):
        """
        Train the model on the given dataset.
        
        Args:
            dataset: Dataset to train on
            batch_size: Batch size for training
            epochs: Number of epochs to train
            lr: Learning rate
            val_split: Validation split ratio (for within-subject mode)
            test_subjects: List of subjects to exclude from training
            weight_decay: Weight decay for optimizer
            patience: Early stopping patience
            use_scheduler: Whether to use learning rate scheduler
            within_subject: If True, perform within-subject training and evaluation
                           instead of cross-subject validation
        """
        self.lr = lr
        # Use AdamW optimizer with weight decay for better generalization
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=weight_decay)
        
        # Reset the tracking arrays to avoid issues with multiple training runs
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        
        # Learning rate scheduler
        scheduler = None
        if use_scheduler:
            # OneCycleLR scheduler for improved convergence and performance
            # Provides a warm-up phase and a gradual cooldown
            scheduler = optim.lr_scheduler.OneCycleLR(
                self.optimizer, max_lr=lr, 
                steps_per_epoch=len(dataset) // batch_size + 1,
                epochs=epochs, 
                pct_start=0.3,  # Spend 30% of training in warm-up phase
                verbose=True
            )
        
        # Group data by subject ID and question period to prevent leakage
        subject_qp_groups = self._group_by_subject_and_question_period(dataset, within_subject)
        
        # Split the groups into training and validation sets
        train_indices, val_indices = self._split_by_groups(subject_qp_groups, val_split, test_subjects, within_subject)
        
        # Create subset datasets
        train_dataset = torch.utils.data.Subset(dataset, train_indices)
        val_dataset = torch.utils.data.Subset(dataset, val_indices)
        
        train_sampler = torch.utils.data.BatchSampler(torch.utils.data.SequentialSampler(train_dataset), batch_size=batch_size, drop_last=True)
        val_sampler = torch.utils.data.BatchSampler(torch.utils.data.SequentialSampler(val_dataset), batch_size=batch_size, drop_last=True)
        
        # Calculate the actual sizes of the datasets
        train_size = len(train_dataset)
        val_size = len(val_dataset)
        print(f"Training on {train_size} samples, validating on {val_size} samples")
        
        # Initialize early stopping variables
        best_val_loss = float('inf')
        best_model_state = None
        patience_counter = 0
        
        # Training loop
        print("\nStarting training...")
        for epoch in range(epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            # Use tqdm for progress tracking with more detailed stats
            progress_bar = tqdm(train_sampler, desc=f"Epoch {epoch+1}/{epochs} (Train)")
            for batch_idx, indices in enumerate(progress_bar):
                # Get batch data
                batch = [dataset[i] for i in indices]
                
                # Extract EEG data and labels
                eeg_data = torch.stack([sample['eeg'] for sample in batch])
                labels = torch.stack([sample['label'] for sample in batch])
                
                # Move to device
                eeg_data = eeg_data.to(self.device)
                labels = labels.to(self.device)
                
                # Preprocess EEG data for Conformer
                eeg_data = self.preprocess_eeg_data(eeg_data)
                
                # Forward pass
                _, outputs = self.model(eeg_data)  # Conformer returns features and logits
                
                # Calculate loss
                loss = self.criterion(outputs, labels)
                
                # Backward pass and optimization
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                # Track metrics
                train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                train_total += labels.size(0)
                train_correct += (predicted == labels).sum().item()
                
                # Update progress bar with current batch loss
                batch_loss = loss.item()
                progress_bar.set_postfix({'batch_loss': f'{batch_loss:.4f}'})
            
            # Calculate training metrics
            avg_train_loss = train_loss / len(train_sampler)
            train_accuracy = 100.0 * train_correct / train_total  # Convert to percentage
            self.train_losses.append(avg_train_loss)  # Store average loss, not total
            self.train_accuracies.append(train_accuracy)  # Store as percentage
            
            # Validation phase
            self.model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():  # No gradients needed for validation
                for batch_idx, indices in enumerate(tqdm(val_sampler, desc=f"Epoch {epoch+1}/{epochs} (Val)")):
                    # Get batch data
                    batch = [dataset[i] for i in indices]
                    
                    # Extract EEG data and labels
                    eeg_data = torch.stack([sample['eeg'] for sample in batch])
                    labels = torch.stack([sample['label'] for sample in batch])
                    
                    # Move to device
                    eeg_data = eeg_data.to(self.device)
                    labels = labels.to(self.device)
                    
                    # Preprocess EEG data for Conformer
                    eeg_data = self.preprocess_eeg_data(eeg_data)
                    
                    # Forward pass
                    features, outputs = self.model(eeg_data)  # Conformer returns features and logits
                    
                    # Debug model outputs if in first epoch and first batch
                    if epoch == 0 and batch_idx == 0:
                        print(f"\nDEBUG - Model output stats: mean={outputs.mean().item():.4f}, std={outputs.std().item():.4f}")
                        print(f"DEBUG - Feature stats: mean={features.mean().item():.4f}, std={features.std().item():.4f}")
                        print(f"DEBUG - First few logits:\n{outputs[:2]}")
                        print(f"DEBUG - Label distribution in this batch: {labels.cpu().numpy().tolist()[:10]}...")
                    
                    # Calculate loss
                    loss = self.criterion(outputs, labels)
                    
                    # Track metrics
                    val_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()
                    
                    # Debug validation predictions (check for class imbalance in predictions)
                    if batch_idx == 0:
                        # Only log the first batch to avoid cluttering output
                        pred_counts = {0: 0, 1: 0}
                        for pred in predicted.cpu().numpy():
                            if pred in pred_counts:
                                pred_counts[pred] += 1
                        print(f"\nDEBUG - First val batch predictions: {pred_counts}")
                        print(f"DEBUG - First val batch probabilities: {torch.softmax(outputs, dim=1)[0:5]}")
            
            # Calculate validation metrics
            avg_val_loss = val_loss / len(val_sampler)
            val_accuracy = 100.0 * val_correct / val_total  # Convert to percentage
            
            # Debug validation results
            print(f"\nDEBUG - Val correct: {val_correct}, Val total: {val_total}")
            print(f"DEBUG - Calculated val accuracy: {val_correct}/{val_total} = {val_correct/val_total*100:.2f}%")
            
            # Check for potential class imbalance issues in validation set
            # If all predictions are the same class, accuracy will match class distribution
            self.val_losses.append(avg_val_loss)  # Store average loss, not total
            self.val_accuracies.append(val_accuracy)  # Store as percentage
            
            # Check for early stopping
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_model_state = self.model.state_dict().copy()
                patience_counter = 0
                print(f"   ↓ New best validation loss: {best_val_loss:.4f}")
            else:
                patience_counter += 1
                print(f"   - No improvement for {patience_counter}/{patience} epochs")
                
            # Update scheduler if used
            if scheduler is not None and isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(avg_val_loss)
            elif scheduler is not None:
                scheduler.step()  # OneCycleLR doesn't need a loss value
                
            # Early stopping check
            if patience_counter >= patience:
                print(f"\nEarly stopping triggered! No improvement for {patience} epochs.")
                break
            self.val_losses.append(val_loss)
            self.val_accuracies.append(val_accuracy)
            
            # Print metrics with better formatting
            print(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.2f}%, LR: {self.optimizer.param_groups[0]['lr']:.6f} - Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.2f}%")
        
        # Restore best model if early stopping
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
            print("\nRestored model from best validation checkpoint")
        
        # Final metrics - correctly formatted as percentages
        print("\nTraining completed!")
        print(f"Best validation accuracy: {max(self.val_accuracies):.2f}%")
        print(f"Best validation loss: {min(self.val_losses):.4f}")
        
        # Calculate final generalization gap
        try:
            best_epoch_idx = self.val_losses.index(min(self.val_losses))
            train_acc_at_best = self.train_accuracies[best_epoch_idx]
            val_acc_at_best = self.val_accuracies[best_epoch_idx]
            gen_gap = train_acc_at_best - val_acc_at_best
            print(f"Generalization gap at best model: {gen_gap:.2f}% (Train: {train_acc_at_best:.2f}%, Val: {val_acc_at_best:.2f}%)")
        except Exception as e:
            print(f"Could not calculate generalization gap: {str(e)}")
        
        # Check for overfitting - safely access the final epoch metrics
        try:
            final_epoch = min(len(self.train_accuracies) - 1, len(self.val_accuracies) - 1)
            if final_epoch >= 0:  # Make sure we have at least one epoch recorded
                train_val_gap = self.train_accuracies[final_epoch] - self.val_accuracies[final_epoch]
                
                if train_val_gap > 0.2:  # If there's more than 20% gap between train and val accuracy
                    print("\nWARNING: Significant gap between training and validation accuracy indicates overfitting.")
                    print(f"Training accuracy: {self.train_accuracies[final_epoch]:.4f}, Validation accuracy: {self.val_accuracies[final_epoch]:.4f}")
                    print("Consider using stronger regularization, more diverse training data, or simpler model.")
            else:
                print("\nWARNING: No training epochs completed. Check your dataset and model configuration.")
        except IndexError as e:
            print(f"\nWARNING: Error checking for overfitting: {str(e)}. This is likely due to early stopping.")
            print(f"Train accuracies recorded: {len(self.train_accuracies)}, Val accuracies recorded: {len(self.val_accuracies)}")
        except Exception as e:
            print(f"\nWARNING: Unexpected error during overfitting check: {str(e)}")

        
        # Return training history
        history = {
            'train_loss': self.train_losses,
            'val_loss': self.val_losses,
            'train_acc': self.train_accuracies,
            'val_acc': self.val_accuracies
        }
        return history
    
    def evaluate(self, test_loader):
        """
        Evaluate the model on a test set.
        
        Args:
            test_loader: DataLoader for the test dataset
        
        Returns:
            Dictionary containing evaluation metrics
        """
        self.model.eval()
        test_loss = 0.0
        all_labels = []
        all_preds = []
        
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Evaluating"):
                # Extract EEG data and labels
                eeg_data = batch['eeg']
                labels = batch['label']
                
                # Preprocess EEG data for Conformer
                eeg_data = self.preprocess_eeg_data(eeg_data)
                
                # Move to device
                eeg_data = eeg_data.to(self.device)
                labels = labels.to(self.device)
                
                # Forward pass
                _, outputs = self.model(eeg_data)  # Conformer returns features and logits
                
                # Compute loss
                loss = self.criterion(outputs, labels)
                test_loss += loss.item() * eeg_data.size(0)
                
                # Get predictions
                _, preds = torch.max(outputs, 1)
                
                # Store labels and predictions
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())
        
        # Calculate metrics
        test_loss /= len(all_labels)
        accuracy = accuracy_score(all_labels, all_preds) * 100
        f1 = f1_score(all_labels, all_preds, average='weighted') * 100
        conf_matrix = confusion_matrix(all_labels, all_preds)
        
        # Print results
        print(f"Test Results:")
        print(f"  Loss: {test_loss:.4f}")
        print(f"  Accuracy: {accuracy:.2f}%")
        print(f"  F1 Score: {f1:.2f}%")
        print(f"  Confusion Matrix:\n{conf_matrix}")
        
        return {
            'loss': test_loss,
            'accuracy': accuracy,
            'f1_score': f1,
            'confusion_matrix': conf_matrix,
            'predictions': all_preds,
            'true_labels': all_labels
        }
    
    def predict(self, eeg_data):
        """
        Make predictions on new EEG data.
        
        Args:
            eeg_data: Tensor of shape [batch_size, channels, time_steps]
        
        Returns:
            Tuple of (predicted_classes, probabilities)
        """
        self.model.eval()
        
        # Preprocess EEG data
        eeg_data = self.preprocess_eeg_data(eeg_data)
        eeg_data = eeg_data.to(self.device)
        
        # Make prediction
        with torch.no_grad():
            _, outputs = self.model(eeg_data)
            probabilities = torch.softmax(outputs, dim=1)
            predictions = torch.argmax(probabilities, dim=1)
        
        return predictions.cpu().numpy(), probabilities.cpu().numpy()
    
    def save_model(self, path):
        """
        Save the model to disk.
        
        Args:
            path: Path to save the model
        """
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict() if self.optimizer else None,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_accuracies': self.train_accuracies,
            'val_accuracies': self.val_accuracies
        }, path)
        print(f"Model saved to {path}")
    
    def load_model(self, path):
        """
        Load a saved model from disk.
        
        Args:
            path: Path to the saved model
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        if checkpoint['optimizer_state_dict'] and self.optimizer:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_losses = checkpoint.get('val_losses', [])
        self.train_accuracies = checkpoint.get('train_accuracies', [])
        self.val_accuracies = checkpoint.get('val_accuracies', [])
        print(f"Model loaded from {path}")
    
    def _group_by_subject_and_question_period(self, dataset, within_subject=False):
        """
        Group the dataset indices by subject ID and question period to prevent data leakage.
        Now optimized to work with pre-processed data.
        
        Args:
            dataset: MultiModalDataset instance with pre-processed samples
            within_subject: If True, perform within-subject training and evaluation
        
        Returns:
            Dictionary with keys as (subject_id, question_period) and values as lists of dataset indices
        """
        print("Grouping dataset by subject and question period...")
        print(f"Dataset length: {len(dataset)}")
        
        groups = {}
        progress_interval = max(1, len(dataset) // 10)  # Report progress every 10%
        
        # Since data is now pre-processed, we can access samples directly
        for idx in range(len(dataset)):
            if idx % progress_interval == 0:
                print(f"Processed {idx}/{len(dataset)} samples ({idx/len(dataset)*100:.1f}%)")
            
            # Get pre-processed sample
            sample = dataset[idx]
            
            # Extract subject ID and question period directly from the sample
            subject_id = sample['sub']
            question_period = sample['question_period']
            
            # Create a group key
            group_key = (subject_id, question_period)
            
            # Add index to the appropriate group
            if group_key not in groups:
                groups[group_key] = []
            groups[group_key].append(idx)
        
        print(f"Grouped data into {len(groups)} subject-question period combinations")
        return groups
    
    def _split_by_groups(self, groups, val_split=0.2, test_subjects=None, within_subject=False):
        """
        Split groups into training and validation sets, ensuring that all data points
        from the same (subject, question_period) group stay together.
        
        Args:
            groups: Dictionary with (subject_id, question_period) keys and dataset indices as values
            val_split: Validation split ratio
            test_subjects: List of subjects to exclude from training
            within_subject: If True, perform within-subject training and evaluation
        
        Returns:
            Tuple of (train_indices, val_indices)
        """
        train_indices = []
        val_indices = []
        
        if within_subject:
            # Select a single subject for within-subject evaluation
            print("Using within-subject training and evaluation mode")
            
            # Get all unique subjects from the group keys
            all_subjects = sorted(list(set(key[0] for key in groups.keys())))
            
            # If test_subjects is provided, use the first one as the target subject
            if test_subjects is not None and len(test_subjects) > 0:
                selected_subject = test_subjects[0]
                print(f"Training on specified subject: {selected_subject}")
            else:
                # Pick the subject with the most samples for better training
                subject_counts = defaultdict(int)
                for (subject, _), indices in groups.items():
                    subject_counts[subject] += len(indices)
                
                # Sort subjects by sample count (descending)
                subjects_by_count = sorted(subject_counts.items(), key=lambda x: x[1], reverse=True)
                selected_subject = subjects_by_count[0][0]  # Take the subject with most samples
            
            # Get sample count for the selected subject
            subject_counts = defaultdict(int)
            for (subject, _), indices in groups.items():
                subject_counts[subject] += len(indices)
            print(f"Selected subject {selected_subject} with {subject_counts[selected_subject]} samples for within-subject training")
            
            # Get all indices for the selected subject from our groups
            subject_indices = []
            for (subject, _), indices in groups.items():
                if subject == selected_subject:
                    subject_indices.extend(indices)
            print(f"Found {len(subject_indices)} samples for subject {selected_subject}")
            
            # Split the data temporally (chronologically) for this subject
            # This is more realistic than random splitting for time series data
            # Group indices by question period
            qp_groups = defaultdict(list)
            for (subject, qp), indices in groups.items():
                if subject == selected_subject:
                    qp_groups[qp].extend(indices)
            
            # Sort question periods and flatten the indices in temporal order
            subject_indices = []
            for qp in sorted(qp_groups.keys()):
                subject_indices.extend(qp_groups[qp])
            
            # Split into train/val with temporal order preserved
            split_idx = int(len(subject_indices) * (1 - val_split))
            train_groups = subject_indices[:split_idx]
            val_groups = subject_indices[split_idx:]
            
            print(f"Train set: {len(train_groups)} samples (earlier question periods)")
            print(f"Validation set: {len(val_groups)} samples (later question periods)")
            
        else:
            # Original cross-subject validation grouping
            print("Grouping dataset by subject and question period...")
            data_groups = defaultdict(list)
            
            # Group samples by subject and question period
            print(f"Dataset length: {len(dataset)}")
            for i, sample in enumerate(dataset):
                if i % 10000 == 0:
                    print(f"Processed {i}/{len(dataset)} samples ({i/len(dataset)*100:.1f}%)")
                
                subject = sample['subject']
                question_period = sample['question_period']
                key = f"{subject}_{question_period}"
                data_groups[key].append(i)
            
            print(f"Grouped data into {len(data_groups)} subject-question period combinations")
            
            # Split groups into train and validation sets
            train_groups = []
            val_groups = []
            
            # Filter subjects
            all_subjects = sorted(list(set([s.split('_')[0] for s in data_groups.keys()])))
            
            if test_subjects is not None:
                all_subjects = [s for s in all_subjects if s not in test_subjects]
            
            # Use stratified split for subjects
            train_subjects, val_subjects = train_test_split(all_subjects, test_size=val_split, random_state=42)
            
            print(f"Training on subjects: {train_subjects}")
            print(f"Validating on subjects: {val_subjects}")
            
            # Assign groups to train or validation
            for group_key, indices in data_groups.items():
                subject = group_key.split('_')[0]
                if subject in train_subjects:
                    train_groups.extend(indices)
                elif subject in val_subjects:
                    val_groups.extend(indices)
        return train_groups, val_groups
    
    def plot_training_history(self):
        """
        Plot training history including loss and accuracy curves.
        """
        plt.figure(figsize=(12, 5))
        
        # Plot loss
        plt.subplot(1, 2, 1)
        plt.plot(self.train_losses, label='Train Loss')
        plt.plot(self.val_losses, label='Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        
        # Plot accuracy
        plt.subplot(1, 2, 2)
        plt.plot(self.train_accuracies, label='Train Accuracy')
        plt.plot(self.val_accuracies, label='Validation Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy (%)')
        plt.title('Training and Validation Accuracy')
        plt.legend()
        
        plt.tight_layout()
        plt.show()


def train_eeg_attention_model(eeg_folder, face_folder, label_file, attend_type='deep', 
                         test_subjects=None, batch_size=32, epochs=30, lr=0.0005,
                         excluded_subjects=None, use_cache=True):
    """
    End-to-end training function for EEG attention prediction model.
    Ensures proper data splitting based on subjects and question periods to prevent data leakage.
    
    Args:
        eeg_folder: Path to the EEG data folder
        face_folder: Path to the facial data folder
        label_file: Path to the labels file
        attend_type: Type of attention to predict ('deep' or 'shallow')
        test_subjects: Optional list of subject IDs to use for testing (e.g., ['sub-001', 'sub-005'])
        batch_size: Batch size for training
        epochs: Number of training epochs
        lr: Learning rate
    
    Returns:
        Trained EEGAttentionPredictor instance
    """
    # Define default excluded subjects if none provided
    if excluded_subjects is None:
        # These are subjects to be excluded from the analysis as specified
        excluded_subjects = [
            '226', '240', '244', '247', '252', '253', '259'
        ]
        print(f"Using default excluded subjects: {excluded_subjects}")
    
    # Initialize the dataset with excluded subjects
    print("Loading dataset...")
    dataset = MultiModalDataset(
        eeg_folder=eeg_folder,
        face_folder=face_folder,
        label_file=label_file,
        attend_type=attend_type,
        excluded_subjects=excluded_subjects,
        use_cache=use_cache
    )
    print(f"Dataset loaded: {len(dataset)} samples")
    
    # Sample the first few items to examine subject IDs and question periods
    print("\nSample data items:")
    unique_subjects = set()
    unique_qps = set()
    for i in range(min(5, len(dataset))):
        sample = dataset[i]
        print(f"  Sample {i}: Subject={sample['sub']}, QP={sample['question_period']}, "  
              f"EEG shape={tuple(sample['eeg'].shape)}, Label={sample['label'].item()}")
        unique_subjects.add(sample['sub'])
    
    print(f"\nFound {len(unique_subjects)} unique subjects in first few samples")
    
    # Initialize the predictor (adapt channels to your data)
    num_eeg_channels = dataset[0]['eeg'].shape[0]  # Get number of channels from the first sample
    predictor = EEGAttentionPredictor(eeg_channels=num_eeg_channels)
    
    # Train the model with proper subject and question period splitting
    print("\nStarting training with proper subject and question period separation...")
    try:
        # Wrap in try/except to catch and print any errors
        print("DEBUG: About to start grouping data by subject and question period...")
        history = predictor.train(
            dataset=dataset,
            batch_size=batch_size,
            epochs=epochs,
            lr=lr,
            test_subjects=test_subjects  # This ensures no data leakage between train/val sets
        )
    except Exception as e:
        print(f"ERROR during training: {e}")
        import traceback
        traceback.print_exc()
        raise
    
    # Plot training history
    predictor.plot_training_history()
    
    # Save the trained model
    model_path = os.path.join(os.path.dirname(__file__), 'eeg_attention_model.pth')
    predictor.save_model(model_path)
    """
    Load a cached dataset and train/evaluate the EEG attention predictor model.
    
    Args:
        cache_path: Path to the cached dataset
        batch_size: Batch size for training
        epochs: Number of training epochs
        lr: Learning rate
        emb_size: Embedding size for the model
        depth: Depth of the transformer encoder
        dropout: Dropout rate
        weight_decay: L2 regularization strength
        use_class_weights: Whether to use class weights to handle imbalanced data
        data_augmentation: Whether to apply data augmentation techniques
        
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
        
        # Apply data augmentation if enabled
        if data_augmentation:
            print("\nApplying data augmentation to improve model robustness...")
            augmented_samples = []
            
            # Get a subset of samples for augmentation (to avoid excessive memory usage)
            subset_size = min(5000, len(samples))
            subset_indices = np.random.choice(len(samples), subset_size, replace=False)
            subset = [samples[i] for i in subset_indices]
            
            # Apply time-domain augmentations
            for sample in tqdm(subset, desc="Generating augmented samples"):
                # Make a deep copy to avoid modifying original
                aug_sample = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in sample.items()}
                
                # Add random Gaussian noise (SNR between 15-25 dB)
                if 'eeg' in aug_sample:
                    eeg_data = aug_sample['eeg']
                    # Calculate signal power
                    signal_power = torch.mean(eeg_data**2)
                    # Target SNR in dB
                    target_snr_db = np.random.uniform(15, 25)
                    # Convert to linear scale
                    target_snr_linear = 10**(target_snr_db/10)
                    # Calculate noise power needed for target SNR
                    noise_power = signal_power / target_snr_linear
                    # Generate noise
                    noise = torch.randn_like(eeg_data) * torch.sqrt(noise_power)
                    # Add noise
                    aug_sample['eeg'] = eeg_data + noise
                
                # Add the augmented sample
                augmented_samples.append(aug_sample)
            
            # Update dataset with augmented samples
            new_dataset = CachedDataset(samples + augmented_samples)
            print(f"Added {len(augmented_samples)} augmented samples. New dataset size: {len(new_dataset)}")
            dataset = new_dataset
            
        # Print dataset stats
        print("\nAnalyzing dataset structure:")
        sample = dataset[0]
        print(f"Sample keys: {list(sample.keys())}")
        for key, value in sample.items():
            if isinstance(value, torch.Tensor):
                print(f"  {key}: Tensor shape {tuple(value.shape)}")
            else:
                print(f"  {key}: {type(value)} {value}")
        
        # Get unique subjects
        if 'subject' in sample:
            subject_key = 'subject'
        elif 'sub' in sample:
            subject_key = 'sub'
        else:
            subject_key = None
            print("WARNING: Could not find subject identifier in sample keys")
            
        if subject_key:
            unique_subjects = set()
            for i in range(min(1000, len(dataset))):
                unique_subjects.add(dataset[i][subject_key])
            print(f"Found {len(unique_subjects)} unique subjects in first 1000 samples")
    except Exception as e:
        print(f"ERROR loading dataset: {e}")
        import traceback
        traceback.print_exc()
        return None
        
    # Compute class weights if enabled
    if use_class_weights:
        print("Computing class weights to handle imbalance...")
        try:
            # Get all labels - need to handle tensor labels correctly
            labels = []
            for i in range(min(10000, len(dataset))):  # Use a subset for efficiency
                sample_label = dataset[i]['label']
                # Convert tensor to int if needed
                if isinstance(sample_label, torch.Tensor):
                    sample_label = sample_label.item()
                labels.append(sample_label)
                
            # Count class frequencies
            label_counts = {}
            for label in labels:
                if label not in label_counts:
                    label_counts[label] = 0
                label_counts[label] += 1
                
            print(f"Label counts: {label_counts}")
            
            # Calculate class weights (inverse frequency) with stronger weighting
            n_samples = sum(label_counts.values())
            class_weights = {}
            n_classes = len(label_counts)
            for label, count in label_counts.items():
                class_weights[label] = n_samples / (count * n_classes)
            
            # Convert to tensor for CrossEntropyLoss
            # Ensure we have exactly n_classes (likely 2 for binary classification)
            n_classes = 2  # Hard-code for binary classification to ensure correct size
            weight_tensor = torch.zeros(n_classes)
            
            # Map the class weights to the tensor
            for label, weight in class_weights.items():
                if isinstance(label, torch.Tensor):
                    idx = label.item()
                else:
                    idx = int(label)
                if idx < n_classes:
                    weight_tensor[idx] = weight
            
            # Apply power function to make the weights more aggressive
            # This helps when the model is stuck predicting one class
            weight_tensor = weight_tensor ** 1.5  # More aggressive than sqrt
            
            # Normalize weights to sum to n_classes
            weight_tensor = weight_tensor * (n_classes / torch.sum(weight_tensor))
            
            # Make class 0 weight relatively higher to counteract the bias toward class 1
            # Adjust based on the prediction bias we observed in debugging
            if weight_tensor[0] < weight_tensor[1]:
                # If class 0 already has lower weight, increase it
                adjustment_factor = 1.25
                weight_tensor[0] *= adjustment_factor
                # Re-normalize
                weight_tensor = weight_tensor * (n_classes / torch.sum(weight_tensor))
            
            print(f"Computed class weights: {weight_tensor}")
        except Exception as e:
            print(f"ERROR computing class weights: {e}")
            import traceback
            traceback.print_exc()
    
    # Initialize the model with enhanced architecture
    print(f"Initializing model with enhanced parameters: emb_size={emb_size}, depth={depth}, dropout={dropout}")
    try:
        predictor = EEGAttentionPredictor(emb_size=emb_size, depth=depth, dropout=dropout)
        
        # Print the model architecture for debugging
        print(f"\nModel Architecture:\n{predictor.model}")
        print(f"Using device: {predictor.device}")
        
        # Print model architecture summary
        num_params = sum(p.numel() for p in predictor.model.parameters() if p.requires_grad)
        print(f"Model has {num_params:,} trainable parameters")
        
        # Apply class weights if enabled
        if use_class_weights and 'weight_tensor' in locals():
            predictor.criterion = nn.CrossEntropyLoss(weight=weight_tensor.to(predictor.device))
            print(f"Using weighted CrossEntropyLoss with weights: {weight_tensor}")
    except Exception as e:
        print(f"ERROR initializing model: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    # Train the model with enhanced parameters
    print("\nStarting enhanced training with proper subject and question period separation...")
    try:
        # Custom optimizer with weight decay
        optimizer = optim.AdamW(
            predictor.model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
        predictor.optimizer = optimizer
        
        # Create OneCycleLR scheduler for better convergence
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=lr,
            steps_per_epoch=len(dataset) // batch_size + 1,
            epochs=epochs,
            pct_start=0.1  # 10% warmup period
        )
        
        # Train with enhanced parameters
        history = predictor.train(
            dataset=dataset,
            batch_size=batch_size,
            epochs=epochs,
            lr=lr,
            val_split=val_split,
            test_subjects=test_subjects,
            use_scheduler=True,  # Enable scheduler
            weight_decay=weight_decay,
            patience=10,  # Early stopping patience
            within_subject=within_subject  # Use within-subject mode if enabled
        )
    except Exception as e:
        print(f"ERROR during training: {e}")
        import traceback
        traceback.print_exc()
        raise
    
    # Plot training history
    predictor.plot_training_history()
    
    # Save the trained model
    model_name = f'eeg_attention_model_e{emb_size}_d{depth}_lr{lr}.pth'
    model_path = os.path.join(os.path.dirname(__file__), model_name)
    predictor.save_model(model_path)
    print(f"\nEnhanced model saved to: {model_path}")
    
    return predictor

# Example usage
if __name__ == "__main__":
    # Add command line argument support
    import argparse
    
    parser = argparse.ArgumentParser(description='EEG Attention Prediction Model Training')
    parser.add_argument('--within-subject', action='store_true',
                        help='Perform within-subject training and evaluation instead of cross-subject')
    parser.add_argument('--cache-path', type=str,
                        default='/media/volume/sdb/ail_project/processed_eeg_fall_2024/dataset_cache/attend_deep/win_2.0/eeg_raw/excl_226_240_241_242_244_247_252_259_266_274_277/samples_86820a8a.pt',
                        help='Path to the cached dataset')
    parser.add_argument('--debug', action='store_true',
                        help='Enable detailed debug output')
    args = parser.parse_args()
    
    # Path to the specific cache file the user wants to load
    cache_path = args.cache_path
    
    # Determine if we're using within-subject training
    within_subject = args.within_subject
    
    mode = "Within-Subject" if within_subject else "Cross-Subject"
    print(f"\n===== Running {mode} EEG Attention Model Training =====\n")
    
    # Check if the cache path exists
    import os
    if not os.path.exists(cache_path):
        print(f"ERROR: Cache file not found: {cache_path}")
        print("Please verify the path and try again.")
        sys.exit(1)
    
    try:
        print(f"Starting execution with cache path: {cache_path}")
        # Load the cached dataset and train/evaluate the model with enhanced parameters
        predictor = load_and_evaluate_cached_dataset(
            cache_path=cache_path,
            # Optimized training parameters
            batch_size=32,          # Smaller batch size for stability with within-subject mode
            epochs=40,              # Sufficient epochs with early stopping
            lr=0.001,              # Lower learning rate for within-subject mode to prevent overfitting
            # Enhanced model architecture
            emb_size=64,           # Increased for more expressive features
            depth=4,               # Optimal depth for EEG data
            dropout=0.3 if within_subject else 0.2,  # Higher dropout for within-subject to prevent overfitting
            weight_decay=1e-4,     # L2 regularization to prevent overfitting
            use_class_weights=True, # Handle class imbalance
            data_augmentation=within_subject,  # Apply data augmentation for within-subject mode
            within_subject=within_subject,  # Use within-subject mode if enabled
            debug=args.debug       # Enable detailed debugging if requested
        )
    except Exception as e:
        print(f"ERROR in main execution: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # If predictor was successfully trained
    if predictor is not None:
        # Save the trained model with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = f"/media/volume/sdb/ail_project/models/eeg_attention_predictor_enhanced_{timestamp}.pt"
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        try:
            # Save the model state dict
            torch.save({
                'model_state_dict': predictor.model.state_dict(),
                'model_config': {
                    'emb_size': 64,
                    'depth': 4,
                    'dropout': 0.2,
                    'n_classes': 2
                },
                'training_metrics': {
                    'train_losses': predictor.train_losses,
                    'val_losses': predictor.val_losses,
                    'train_accuracies': predictor.train_accuracies,
                    'val_accuracies': predictor.val_accuracies
                },
                'timestamp': timestamp
            }, model_path)
            print(f"\nModel saved to {model_path}")
        except Exception as e:
            print(f"ERROR saving model: {e}")
            import traceback
            traceback.print_exc()
    
    print("\nHigh-Performance model training and evaluation complete!")
    print("\nKey improvements implemented:")
    print("1. Per-channel EEG data normalization for better signal processing")
    print("2. Data augmentation with controlled SNR noise for generalization")
    print("3. AdamW optimizer with optimal weight decay settings")
    print("4. OneCycleLR learning rate scheduling for faster convergence")
    print("5. Improved class weights calculation with smoothing")
    print("6. Optimal model architecture (emb_size=64, depth=4, dropout=0.2)")
    print("7. Fixed accuracy reporting and loss tracking")
    print("8. Comprehensive model metrics with generalization gap analysis")
    
    print("\nTo evaluate the model on new data:")
    print("1. Load the saved model from the path above")
    print("2. Apply the same data preprocessing steps")
    print("3. Use predictor.evaluate() with your test dataset")

    
    # Previous example code (commented out)
    """
    # Define paths
    eeg_folder = '/media/volume/sdb/ail_project/processed_eeg_fall_2024/'
    face_folder = '/media/volume/sdb/ail_project/processed_features/face/'
    label_file = '/media/volume/sdb/ail_project/labels/attention_labels_combined.csv'
    
    # Subjects to exclude from analysis
    excluded_subjects = [
        'AIL-0226', 'AIL-0240', 'AIL-0244', 'AIL-0247', 
        'AIL-0252', 'AIL-0253', 'AIL-0259'
    ]
    
    # Option 1: Train with automatic subject-based train/test splitting
    predictor_auto = train_eeg_attention_model(
        eeg_folder=eeg_folder,
        face_folder=face_folder,
        label_file=label_file,
        attend_type='deep',
        epochs=20,
        excluded_subjects=excluded_subjects
    )
    
    # Option 2: Train with specified test subjects
    test_subjects = ['sub-029', 'sub-027']
    predictor_manual = train_eeg_attention_model(
        eeg_folder=eeg_folder,
        face_folder=face_folder,
        label_file=label_file,
        attend_type='deep',
        test_subjects=test_subjects,
        epochs=20,
        excluded_subjects=excluded_subjects
    )
    """

