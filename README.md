# AIL Fusion Model

A multimodal deep learning model for attention detection using EEG and facial data fusion.

## Project Overview

This project implements a multimodal approach for attention detection by combining:
- EEG recordings (256 Hz)
- Facial feature data (30 Hz)

The dataset handles different data modalities and synchronizes them for training a fusion model.

## Repository Structure

### Core Files
- `data_loader.py`: MultiModalDataset implementation with caching and preprocessing
- `eeg_modeling.py`: EEG attention prediction using EEG-Conformer architecture
- `model.py`: Neural network models for multimodal fusion

### Evaluation Scripts
- `cross_subject_evaluation.py`: Cross-subject validation and evaluation
- `evaluate_multiple_subjects.py`: Batch evaluation across multiple subjects
- `interactive_lda.py`: Linear Discriminant Analysis for feature exploration
- `test_band_power.py`: EEG band power analysis utilities

### Visualization
- `visualization.py`: Training metrics and EEG signal visualization tools

### Analysis Notebooks
- `examine_data.ipynb`: Data exploration and analysis

## Data Format

The data is organized by subject IDs:
- EEG files: `sub-XXX_EEG_recording_processed.csv`
- Face files: `subject_XXX_face_processed.csv`
- Labels: Contains both `deep_attend` and `shallow_attend` columns

Each data point consists of 2-second windows with 1-second stride.
