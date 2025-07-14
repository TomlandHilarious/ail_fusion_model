# AIL Fusion Model

A multimodal deep learning model for attention detection using EEG and facial data fusion.

## Project Overview

This project implements a multimodal approach for attention detection by combining:
- EEG recordings (256 Hz)
- Facial feature data (30 Hz)

The dataset handles different data modalities and synchronizes them for training a fusion model.

## Repository Structure

- `data_loader.py`: MultiModalDataset implementation for loading and preprocessing data
- `model.py`: Neural network models for multimodal fusion
- `eeg_modeling.py`: EEG-specific processing and modeling functions
- `examine_data.ipynb`: Data exploration notebook

## Data Format

The data is organized by subject IDs:
- EEG files: `sub-XXX_EEG_recording_processed.csv`
- Face files: `subject_XXX_face_processed.csv`
- Labels: Contains both `deep_attend` and `shallow_attend` columns

Each data point consists of 2-second windows with 1-second stride.
