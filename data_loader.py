import torch
import random
import pandas as pd
import numpy as np
import re
import os
import json
import hashlib
from torch.utils.data import Dataset
from typing import List, Literal
from brainflow.data_filter import DataFilter, WindowOperations
from itertools import groupby

class WindowIndexer:
    """
    Input: csv file path, stim column name, window size, step size
    Output: [(file_path, stim, start, end), ...]
    Each entry in self.rows tells the downstream Dataset:
        • which CSV file to open
        • which question-period / stimulus (stim) it belongs to
        • the absolute start-row and end-row (end is **exclusive**)
    rows example:
        (
            '/path/sub-026_EEG_processed.csv',   # file_path
            '026',                                # sub_id
            3,                                   # stim value
            46080,                               # abs_start  (inclusive)
            46600                                # abs_end    (exclusive)
        )
    """
    def __init__(self, 
        csv_paths: List[str], 
        stim_col: str, 
        fs:int, 
        window_size:float=2.0,
        step_size:float=1.0):
        # store metadata information
        window_size_samples = int(window_size * fs)
        step_size_samples = int(step_size * fs)
        self.window_size = window_size
        self.step_size = step_size
        self.rows = []
        
        for p in csv_paths:
            try:
                # Extract subject ID from filepath
                # For EEG files: pattern is 'sub-XXX'
                # For face files: pattern is 'subject_XXX'
                sub_id = None
                if 'sub-' in p:
                    # EEG file naming convention
                    sub_match = re.search(r'sub-0*([0-9]+)', p)
                    if sub_match:
                        sub_id = sub_match.group(1)  # Extract just the digits
                elif 'subject_' in p:
                    # Face file naming convention
                    sub_match = re.search(r'subject_0*([0-9]+)', p)
                    if sub_match:
                        sub_id = sub_match.group(1)  # Extract just the digits
                
                if not sub_id:
                    print(f"Warning: Could not extract subject ID from {p}, using filepath only")
                    
                df = pd.read_csv(p, usecols=[stim_col])  # use the stim to locate the data slices
                for stim, idx in df.groupby(stim_col).indices.items():
                    # idx is nd array of row indices
                    L = len(idx)
                    if L <= window_size_samples:
                        print(f"Warning: Skipping {p}, stimulus {stim}: too few samples ({L})")
                        continue
                    
                    # Convert to integer indices for range
                    for s in range(0, int(L - window_size_samples + 1), int(step_size_samples)):
                        s_idx = int(s)  # Ensure integer index
                        w_idx = int(s_idx + window_size_samples - 1)  # Ensure integer index
                        if w_idx < len(idx):  # Safety check
                            abs_start = int(idx[s_idx])
                            abs_end = int(idx[w_idx])
                            self.rows.append((p, sub_id, stim, abs_start, abs_end))
            except Exception as e:
                print(f"Error processing {p}: {str(e)}")
        
        print(f"Created {len(self.rows)} windows from {len(csv_paths)} files")
        self.fs = fs
        self.stim_col = stim_col

    def __len__(self):
        return len(self.rows)
    def __getitem__(self, idx):
        return self.rows[idx]

class Decoder:
    """
    The decoder class creates features from either EEG or Face data.
    It can create raw features or band power features for EEG data,
    do not use the band power features for Face data.
    """
    def __init__(self, 
        cols: List[str], 
        fs: int, 
        mode: Literal['raw', 'band'] = 'raw', 
        eps:  float = 1e-10
    ):
        self.cols = cols
        self.fs = fs
        self.mode = mode.lower()
        self.eps = eps

    def _band_power_features(self, 
        sig:np.ndarray, 
        sampling_rate:int):
        """
        Compute the five band powers for a 1-D EEG window.

        Returns
        -------
        np.ndarray, shape (5,)
            [delta, theta, alpha, beta, gamma] powers
        """
        # 1) choose FFT length (power-of-two for speed)
        nfft = DataFilter.get_nearest_power_of_two(len(sig))

        # 2) Welch PSD (Blackman-Harris window, 50 % overlap)
        psd = DataFilter.get_psd_welch(
            sig,
            nfft,
            nfft // 2,
            sampling_rate,
            WindowOperations.BLACKMAN_HARRIS.value
        )

        # 3) integrate each band
        bp_delta = DataFilter.get_band_power(psd,  1.0,  4.0)
        bp_theta = DataFilter.get_band_power(psd,  4.0,  8.0)
        bp_alpha = DataFilter.get_band_power(psd,  8.0, 12.0)
        bp_beta  = DataFilter.get_band_power(psd, 12.0, 30.0)
        bp_gamma = DataFilter.get_band_power(psd, 30.0, 100.0)

        # diagnostic prints
        print(f"alpha/beta: {bp_alpha / bp_beta:.3f}")
        print(f"beta/theta: {bp_beta  / bp_theta:.3f}")
        print(f"delta/theta: {bp_delta / bp_theta:.3f}")
        return np.array([bp_delta, bp_theta, bp_alpha, bp_beta, bp_gamma],
                        dtype=np.float32)

    def __call__(self, 
        df_slice:pd.DataFrame, 
        win_len:float=2.0, 
        stride:float=1.0):
        """
        Input: 
            df_slice: pd.DataFrame, 
            win_len: float, 
            stride: float
        Output: 
            torch.Tensor, shape (N, C, L)
        """
        ts = int(win_len * self.fs)
        hop = int(stride * self.fs)
        feats = []
        
        # Make sure to use integer indices for dataframe slicing
        for start in range(0, int(len(df_slice) - ts + 1), int(hop)):
            start_idx = int(start)  # Ensure integer index
            end_idx = int(start + ts)  # Ensure integer index
            
            # Use integer indices for dataframe slicing
            try:
                sig = df_slice.iloc[start_idx:end_idx][self.cols].to_numpy().astype(np.float32)  # (T, C)
            except Exception as e:
                raise ValueError(f"Indexing error: {e}, start={start_idx}, end={end_idx}, df_len={len(df_slice)}")

            if self.mode == 'raw':
                feats.append(sig.T)                                  # (C, T)
            elif self.mode == 'band':
                ch_bands = [self._band_power_features(ch, self.fs)   # (5,)
                            for ch in sig.T]
                feats.append(np.stack(ch_bands, axis=0))             # (C, 5)

        # Before stacking, check if we have any features
        if not feats:
            print(f"Warning: No features extracted from data slice of length {len(df_slice)}")
            # Create a default feature of appropriate shape
            if self.mode == 'raw':
                # For raw mode, return zeros with shape (1, channels, window_length)
                return torch.zeros((1, len(self.cols), ts), dtype=torch.float32)
            else:  # band mode
                # For band mode, return zeros with shape (1, channels, 5)
                return torch.zeros((1, len(self.cols), 5), dtype=torch.float32)
            
        # Stack the features if we have any
        return torch.from_numpy(np.stack(feats, axis=0))  # (N, C, L)


class MultiModalDataset(Dataset):
    def __init__(self, 
        eeg_folder: str, 
        face_folder: str,
        label_file: str,
        attend_type: Literal['deep', 'shallow'] = 'deep',
        eeg_win_size: float=2.0, 
        eeg_win_step: float=1.0,
        face_win_size: float=2.0, 
        face_win_step: float=1.0,
        eeg_cols: List[str]=['AF7', 'TP9', 'TP10', 'AF8'], 
        face_cols: List[str]=['GazeX', 'GazeY', 'Yaw', 'Pitch', 'Roll',
        'AU01', 'AU02', 'AU04', 'AU05', 'AU06', 'AU07', 
        'AU09', 'AU10', 'AU12', 'AU14', 'AU15', 'AU17', 
        'AU20', 'AU23', 'AU25', 'AU26', 'AU45', 'Pupil'],
        eeg_mode:str='raw', face_mode:str='raw',
        eeg_fs:int=256, face_fs:int=30,
        eeg_eps:float=1e-10, face_eps:float=1e-10,
        use_cache:bool=True, cache_dir:str=None,
        excluded_subjects: List[str]=None):
        # store metadata information    
        self.eeg_folder = eeg_folder
        self.face_folder = face_folder 
        self.eeg_cols = eeg_cols
        self.face_cols = face_cols
        self.eeg_mode = eeg_mode
        self.face_mode = face_mode
        self.eeg_fs = eeg_fs
        self.face_fs = face_fs
        
        # Convert excluded_subjects to a set of strings for faster lookups
        self.excluded_subjects = set(str(s) for s in excluded_subjects) if excluded_subjects else set()
        
        # Convert window size in seconds to samples
        self.eeg_win_size_sec = eeg_win_size
        self.eeg_win_step_sec = eeg_win_step
        self.eeg_win_size_samples = int(eeg_win_size * eeg_fs)
        self.eeg_win_step_samples = int(eeg_win_step * eeg_fs)
        
        self.face_win_size_sec = face_win_size
        self.face_win_step_sec = face_win_step
        self.face_win_size_samples = int(face_win_size * face_fs)
        self.face_win_step_samples = int(face_win_step * face_fs)

        # Store all necessary instance variables
        self.label_col = f"{attend_type}_attend"
        self.label_map = {}
        self.labels_df = pd.read_csv(label_file)
        
        # Check if the label column exists
        if self.label_col not in self.labels_df.columns:
            print(f"Warning: '{self.label_col}' not found in label file. Available columns: {self.labels_df.columns.tolist()}")
            if attend_type == 'deep':
                alt_col = 'shallow_attend'
            else:
                alt_col = 'deep_attend'
            if alt_col in self.labels_df.columns:
                print(f"Using '{alt_col}' instead.")
                self.label_col = alt_col
            else:
                raise ValueError(f"Neither '{self.label_col}' nor '{alt_col}' found in label file!")
                
        # Create label map for quick lookup
        for _, row in self.labels_df.iterrows():
            # Check if the label is valid (not NaN)
            if pd.notna(row[self.label_col]):
                # Store label by (subject_id, question_period)
                key = (str(row['ID']), row['stim'])
                self.label_map[key] = row[self.label_col]
        
        # if cache exists, load it
        param_dict = {
            "attend_type": attend_type,
            "eeg_win_size": eeg_win_size,
            "eeg_win_step": eeg_win_step,
            "face_win_size": face_win_size,
            "face_win_step": face_win_step,
            "eeg_mode": eeg_mode,
            "face_mode": face_mode,
            "excluded_subjects": sorted(list(self.excluded_subjects)) if self.excluded_subjects else []
        }
        param_hash = hashlib.md5(
            json.dumps(param_dict, sort_keys=True).encode()
        ).hexdigest()[:8]

        # Create structured cache directory using parameter information
        base_cache_dir = cache_dir or os.path.join(eeg_folder, "dataset_cache")
        
        # Include attend type, window size and EEG mode in the cache directory structure
        structured_dir = os.path.join(
            base_cache_dir,
            f"attend_{param_dict['attend_type']}",
            f"win_{param_dict['eeg_win_size']}",
            f"eeg_{param_dict['eeg_mode']}"
        )
        
        # If excluding subjects, add that to the directory path
        if self.excluded_subjects:
            excluded_str = "_".join(sorted(self.excluded_subjects))
            structured_dir = os.path.join(structured_dir, f"excl_{excluded_str}")
        
        
        self.cache_dir = structured_dir
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Still include the hash for uniqueness
        self.cache_path = os.path.join(self.cache_dir, f"samples_{param_hash}.pt")
        if use_cache and os.path.exists(self.cache_path):
            print(f"Loading dataset from cache: {self.cache_path}")
            try:
                cache_data = torch.load(self.cache_path)
                self.samples = cache_data.get('samples', [])
                dummy_len = len(self.samples)
                
                # Create a dummy indexer class that has a rows attribute with the right length
                class DummyIndexer:
                    def __init__(self, length):
                        self.rows = [None] * length
                    def __len__(self):
                        return len(self.rows)
                
                self.eeg_indexer = DummyIndexer(dummy_len)
                print(f"Successfully loaded dataset from cache with {dummy_len} samples")
                return  # Skip the rest of initialization
            except Exception as e:
                print(f"Warning: failed to load cache ({e})")
                # Proceed with normal initialization
        
        # Get all EEG files
        eeg_files = [f for f in os.listdir(eeg_folder) if f.endswith('_EEG_recording_processed.csv')]
        face_files = [f for f in os.listdir(face_folder) if f.endswith('_face_processed.csv')]
        
        print(f"Found {len(eeg_files)} EEG files and {len(face_files)} face files")
        
        # Extract subject IDs
        eeg_subjects = [re.search(r'sub-0*(\d+)', f).group(1) for f in eeg_files if re.search(r'sub-0*(\d+)', f)]
        face_subjects = [re.search(r'subject_0*(\d+)', f).group(1) for f in face_files if re.search(r'subject_0*(\d+)', f)]
        
        print(f"EEG subject count: {len(eeg_subjects)}")
        print(f"Face subject count: {len(face_subjects)}")
        
        # Analyze missing subjects
        missing_in_face = set(eeg_subjects) - set(face_subjects)
        missing_in_eeg = set(face_subjects) - set(eeg_subjects)
        
        if missing_in_face:
            print(f"Subjects in EEG but not in face data: {sorted([int(s) for s in missing_in_face])}")
        if missing_in_eeg:
            print(f"Subjects in face but not in EEG data: {sorted([int(s) for s in missing_in_eeg])}")
        
        # Find common subjects
        common_subjects = set(eeg_subjects).intersection(set(face_subjects))
        
        # Apply subject exclusion if any were specified
        if self.excluded_subjects:
            excluded_count = len(common_subjects.intersection(self.excluded_subjects))
            common_subjects = common_subjects - self.excluded_subjects
            print(f"Excluded {excluded_count} subjects: {sorted([int(s) for s in self.excluded_subjects if s in eeg_subjects and s in face_subjects])}")
            
        print(f"Found {len(common_subjects)} subjects with both EEG and face data")
        print(f"Common subject IDs: {sorted([int(s) for s in common_subjects])}")

        
        # Verify if we can find label data for these subjects
        self.labels_df = pd.read_csv(label_file)
        label_subjects = set([str(i) for i in self.labels_df['ID'].unique()])
        
        subjects_with_labels = common_subjects.intersection(label_subjects)
        print(f"Subjects with both modalities AND labels: {len(subjects_with_labels)}")
        
        if len(subjects_with_labels) < len(common_subjects):
            missing_labels = common_subjects - subjects_with_labels
            print(f"Subjects missing labels: {sorted([int(s) for s in missing_labels])}")
        
        # Use only subjects that have all three: EEG, face, and labels
        common_subjects = subjects_with_labels
        
        # Create file paths
        self.eeg_paths = []
        self.face_paths = []
        
        for subject in common_subjects:
            # Find matching EEG file
            eeg_file = [f for f in eeg_files if re.search(rf'sub-0*{subject}', f)]
            face_file = [f for f in face_files if re.search(rf'subject_0*{subject}', f)]
            
            if eeg_file and face_file:
                self.eeg_paths.append(os.path.join(eeg_folder, eeg_file[0]))
                self.face_paths.append(os.path.join(face_folder, face_file[0]))
        # create WindowIndexer for EEG and Face data
        self.eeg_indexer = WindowIndexer(self.eeg_paths, 'stim', eeg_fs, window_size=eeg_win_size, step_size=eeg_win_step)
        self.face_indexer = WindowIndexer(self.face_paths, 'QP', face_fs, window_size=face_win_size, step_size=face_win_step)
        # create Decoder for EEG and Face data 
        self.eeg_decoder = Decoder(eeg_cols, eeg_fs, mode=eeg_mode, eps=eeg_eps)
        self.face_decoder = Decoder(face_cols, face_fs, mode=face_mode, eps=face_eps)
        # Labels already loaded in the diagnostic section above
        # check if the label type exists
        if attend_type == 'deep' and 'deep_attend' in self.labels_df.columns:
            self.label_col = 'deep_attend'
        elif attend_type == 'shallow' and 'shallow_attend' in self.labels_df.columns:
            self.label_col = 'shallow_attend'
        else:
            # Default to the first non-ID/stim column
            self.label_col = [col for col in self.labels_df.columns if col not in ['ID', 'stim']][0]
            print(f"Warning: Label column for {attend_type} not found, using {self.label_col} instead")
            
        # Create a dictionary for fast label lookups
        self.label_map = {}
        for _, row in self.labels_df.iterrows():
            # Key format: (subject_id, stimulus)
            key = (str(row['ID']), row['stim'])
            self.label_map[key] = row[self.label_col]
            
        # Pre-compute the aligned data structures once
        print("Aligning EEG and face data by subject and QP...")
        self.aligned_eeg, self.aligned_face = self.align_time()
        print("Data alignment complete!")
        
        # Preprocess all data during initialization for efficient access
        print("Preprocessing data (no cache found)...")
        self._preprocess_all_data()
        try:
            cache_data = {'samples': self.samples}
            torch.save(cache_data, self.cache_path)
            print(f"Saved cache to: {self.cache_path}")
        except Exception as e:
            print(f"Warning: cache save failed — {e}")


        
    def __len__(self):
        # Now we only use the EEG indexer since we're matching face data dynamically
        return len(self.eeg_indexer)


    
    def _preprocess_all_data(self):
        """Preprocess all data at initialization for efficient access"""
        # Dictionary to store all loaded CSV files
        csv_cache = {}
        # Dictionary to store all preprocessed samples
        self.samples = []
        
        # Create an index mapping to locate samples efficiently
        self.index_map = {}
        total_samples = len(self.eeg_indexer.rows)
        
        print(f"Preprocessing {total_samples} samples...")
        for i, eeg_row in enumerate(self.eeg_indexer.rows):
            if i % 1000 == 0:
                print(f"Processed {i}/{total_samples} samples ({i/total_samples*100:.1f}%)")
                
            filepath, sub_id, eeg_stim, abs_start, abs_end = eeg_row
            
            # Check if we have face data for this subject and QP
            have_matching_face = False
            if (sub_id in self.aligned_face and eeg_stim in self.aligned_face[sub_id]):
                face_row = self.aligned_face[sub_id][eeg_stim][0]
                face_filepath, _, _, face_abs_start, face_abs_end = face_row
                have_matching_face = True
                
            # Process EEG data
            try:
                # Load CSV if not already in cache
                if filepath not in csv_cache:
                    csv_cache[filepath] = pd.read_csv(filepath, engine='python')
                    
                eeg_df = csv_cache[filepath]
                eeg_data_slice = eeg_df.iloc[abs_start:abs_end+1]
                eeg_features = self.eeg_decoder(eeg_data_slice, win_len=self.eeg_win_size_sec, stride=self.eeg_win_step_sec)
                
                # Process face data if available
                face_features = None
                if have_matching_face:
                    try:
                        if face_filepath not in csv_cache:
                            csv_cache[face_filepath] = pd.read_csv(face_filepath, engine='python')
                            
                        face_df = csv_cache[face_filepath]
                        face_data_slice = face_df.iloc[face_abs_start:face_abs_end+1]
                        face_features = self.face_decoder(face_data_slice, win_len=self.face_win_size_sec, stride=self.face_win_step_sec)
                    except Exception:
                        face_features = torch.zeros(1, len(self.face_cols), int(self.face_win_size_samples))
                else:
                    face_features = torch.zeros(1, len(self.face_cols), int(self.face_win_size_samples))
                
                # Get label
                label = self.label_map.get((sub_id, eeg_stim), 0)
                
                # Store sample
                sample = {
                    'sub': sub_id,
                    'question_period': eeg_stim,
                    'eeg': eeg_features,
                    'face': face_features,
                    'label': torch.tensor(label, dtype=torch.long)
                }
                
                self.samples.append(sample)
                
            except Exception as e:
                print(f"Error preprocessing sample {i} (subject {sub_id}, QP {eeg_stim}): {e}")
                # Create dummy sample for this index to maintain indexing consistency
                self.samples.append({
                    'sub': sub_id,
                    'question_period': eeg_stim,
                    'eeg': torch.zeros(1, len(self.eeg_cols), self.eeg_win_size_samples),
                    'face': torch.zeros(1, len(self.face_cols), int(self.face_win_size_samples)),
                    'label': torch.tensor(0, dtype=torch.long)
                })
        
        print(f"Preprocessing complete. Cached {len(self.samples)} samples.")
        try:
            cache_data = {'samples': self.samples}
            torch.save(cache_data, self.cache_path)
            print(f"Saved samples to cache: {self.cache_path}")
        except Exception as e:
            print(f"Warning: failed to save cache ({e})")

        # Clear the CSV cache to free memory
        del csv_cache
        
    def __getitem__(self, idx):
        """Simply return the preprocessed sample"""
        return self.samples[idx]

    def align_time(self):
        # Get data and initialize result dictionaries
        eeg_data = self.eeg_indexer.rows
        face_data = self.face_indexer.rows
        eeg_aligned = {}
        face_aligned = {}
        
        # Directly group both datasets by subject ID and QP in one pass
        for eeg_row in eeg_data:
            sub_id, qp, start_idx = eeg_row[1], eeg_row[2], eeg_row[3]
            
            # Create nested structure if needed
            if sub_id not in eeg_aligned:
                eeg_aligned[sub_id] = {}
            if qp not in eeg_aligned[sub_id]:
                eeg_aligned[sub_id][qp] = []
                
            eeg_aligned[sub_id][qp].append(eeg_row)
        
        # Same for face data
        for face_row in face_data:
            sub_id, qp, start_idx = face_row[1], face_row[2], face_row[3]
            
            # Create nested structure if needed
            if sub_id not in face_aligned:
                face_aligned[sub_id] = {}
            if qp not in face_aligned[sub_id]:
                face_aligned[sub_id][qp] = []
                
            face_aligned[sub_id][qp].append(face_row)
        
        # Sort all groups by start index in one line
        for sub_id in eeg_aligned:
            for qp in eeg_aligned[sub_id]:
                eeg_aligned[sub_id][qp].sort(key=lambda x: x[3])
        
        for sub_id in face_aligned:
            for qp in face_aligned[sub_id]:
                face_aligned[sub_id][qp].sort(key=lambda x: x[3])
        
        # Verify sorting and structure of aligned data
        # Find common subjects between EEG and face data
        common_subjects = sorted(set(eeg_aligned.keys()).intersection(set(face_aligned.keys())), key=int)
        print(f"Found {len(common_subjects)} subjects with both EEG and face data")
        
        # Calculate and show sample statistics
        print("\n===== SAMPLE COUNT STATISTICS =====\n")
        sample_stats = []
        
        for sub_id in common_subjects:
            common_qps = sorted(set(eeg_aligned[sub_id].keys()) & set(face_aligned[sub_id].keys()))
            for qp in common_qps:
                eeg_sample_count = sum(row[4] - row[3] + 1 for row in eeg_aligned[sub_id][qp])
                face_sample_count = sum(row[4] - row[3] + 1 for row in face_aligned[sub_id][qp])
                
                # Calculate EEG time in seconds (at 256 Hz)
                eeg_time_sec = eeg_sample_count / 256
                # Calculate face time in seconds (at 30 Hz)
                face_time_sec = face_sample_count / 30
                
                # Store statistics
                sample_stats.append({
                    'subject': sub_id,
                    'qp': qp,
                    'eeg_samples': eeg_sample_count,
                    'face_samples': face_sample_count,
                    'eeg_time_sec': eeg_time_sec,
                    'face_time_sec': face_time_sec,
                    'ratio': face_time_sec / eeg_time_sec if eeg_time_sec > 0 else 0
                })
        
        # Show summary statistics
        print(f"Total subject-QP combinations with both modalities: {len(sample_stats)}")
        
        # Calculate averages
        avg_eeg_samples = sum(stat['eeg_samples'] for stat in sample_stats) / len(sample_stats) if sample_stats else 0
        avg_face_samples = sum(stat['face_samples'] for stat in sample_stats) / len(sample_stats) if sample_stats else 0
        avg_ratio = sum(stat['ratio'] for stat in sample_stats) / len(sample_stats) if sample_stats else 0
        
        print(f"Average EEG samples per QP: {avg_eeg_samples:.1f}")
        print(f"Average face samples per QP: {avg_face_samples:.1f}")
        print(f"Average face/EEG time ratio: {avg_ratio:.2f}")
        
        # Count where face data has more time than EEG
        face_longer_count = sum(1 for stat in sample_stats if stat['ratio'] > 1)
        print(f"QPs where face data is longer than EEG: {face_longer_count} out of {len(sample_stats)} ({100*face_longer_count/len(sample_stats):.1f}%)")
        
        # Show a few examples
        print("\nExample statistics for 3 random subject-QP combinations:")
        if sample_stats:
            import random
            random.seed(42)  # For reproducible results
            samples = random.sample(sample_stats, min(3, len(sample_stats)))
            for i, stat in enumerate(samples):
                print(f"Example {i+1}: Subject {stat['subject']}, QP {stat['qp']}")
                print(f"  EEG: {stat['eeg_samples']} samples ({stat['eeg_time_sec']:.1f} seconds)")
                print(f"  Face: {stat['face_samples']} samples ({stat['face_time_sec']:.1f} seconds)")
                print(f"  Ratio (face/EEG time): {stat['ratio']:.2f}")
        
        # Display example data rows for verification
        if common_subjects:
            # Select one subject for detailed verification
            test_subject = common_subjects[0]
            print(f"\nVerifying alignment for subject {test_subject}:")
            
            # Compare QPs between EEG and face data
            eeg_qps = set(eeg_aligned[test_subject].keys())
            face_qps = set(face_aligned[test_subject].keys())
            common_qps = sorted(eeg_qps.intersection(face_qps))
            
            print(f"  EEG QPs: {sorted(eeg_qps)[:5]}{'...' if len(eeg_qps) > 5 else ''}")
            print(f"  Face QPs: {sorted(face_qps)[:5]}{'...' if len(face_qps) > 5 else ''}")
            print(f"  Common QPs: {common_qps[:5]}{'...' if len(common_qps) > 5 else ''}")
            
            # Verify sorting for both modalities for one QP
            if common_qps:
                test_qp = common_qps[0]
                print(f"\nVerifying sorting for subject {test_subject}, QP {test_qp}:")
                
                # Check if EEG data is sorted by start index
                eeg_rows = eeg_aligned[test_subject][test_qp]
                eeg_starts = [row[3] for row in eeg_rows[:5]]
                print(f"  EEG start indices (first 5): {eeg_starts}")
                is_eeg_sorted = all(eeg_starts[i] <= eeg_starts[i+1] for i in range(len(eeg_starts)-1))
                print(f"  EEG data sorted correctly: {is_eeg_sorted}")
                
                # Check if face data is sorted by start index
                face_rows = face_aligned[test_subject][test_qp]
                face_starts = [row[3] for row in face_rows[:5]]
                print(f"  Face start indices (first 5): {face_starts}")
                is_face_sorted = all(face_starts[i] <= face_starts[i+1] for i in range(len(face_starts)-1))
                print(f"  Face data sorted correctly: {is_face_sorted}")
                
                # Print a sample of the aligned data
                if eeg_rows:
                    print(f"\nExample EEG row: {eeg_rows[0]}")
                if face_rows:
                    print(f"Example Face row: {face_rows[0]}")
                    
            # Compute some statistics
            qp_match_counts = [len(set(eeg_aligned[sub].keys()) & set(face_aligned[sub].keys())) for sub in common_subjects]
            avg_matching_qps = sum(qp_match_counts) / len(qp_match_counts) if qp_match_counts else 0
            print(f"\nAverage matching QPs per subject: {avg_matching_qps:.1f}")

        
        return eeg_aligned, face_aligned
            
# Test code for data loading
def test_multimodal_dataset():
    print("\n" + "="*50)
    print("Testing MultiModalDataset loading")
    print("="*50 + "\n")
    
    print("Initializing dataset...")
    
    # Set a fixed random seed for consistent testing
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Define paths
    eeg_folder = '/media/volume/sdb/ail_project/processed_eeg_fall_2024/'
    face_folder = '/media/volume/sdb/ail_project/processed_features/face/'
    label_file = '/media/volume/sdb/ail_project/labels/attention_labels_combined.csv'
    
    # Optional: Specify subjects to exclude 
    # No exclusions by default then set excluded_subjects to None
    excluded_subjects = ['226', '240', '244',
    '247', '252', '259', '277', '241', '242', '266', '274'] 
    # Initialize dataset
    dataset = MultiModalDataset(eeg_folder, face_folder, label_file, excluded_subjects=excluded_subjects)

    # Print basic info
    print(f"\nDataset size: {len(dataset)} samples")
    print(f"EEG Indexer size: {len(dataset.eeg_indexer)} windows")
    print(f"Label map size: {len(dataset.label_map)} entries")
    print(f"Label type: {dataset.label_col}")
    
    # Check some random samples
    print("\nChecking 10 random samples:")
    num_samples = 10
    indices = np.random.choice(len(dataset), min(len(dataset), num_samples), replace=False)
    
    samples = []
    label_counts = {}
    
    for i, idx in enumerate(indices):
        try:
            sample = dataset[idx]
            samples.append(sample)
            
            print(f"\nSample {i+1}/{len(indices)} (index {idx}):")
            print(f"  Subject ID: {sample['sub']}")
            print(f"  Question Period: {sample['question_period']}")
            print(f"  Label: {sample['label'].item()}")
            print(f"  EEG shape: {tuple(sample['eeg'].shape)}")
            print(f"  Face shape: {tuple(sample['face'].shape)}")
            
            # Count labels
            label = sample['label'].item()
            if label not in label_counts:
                label_counts[label] = 0
            label_counts[label] += 1
            
        except Exception as e:
            print(f"Error loading sample {idx}: {str(e)}")
    
    # Print label distribution in the checked samples
    if samples:
        print("\nLabel distribution in samples:")
        for label, count in label_counts.items():
            print(f"  Label {label}: {count} samples")
                
        # Check if any keys in the sample are None or empty
        sample = samples[0]
        for key, value in sample.items():
            if value is None:
                print(f"Warning: Key '{key}' has empty value in sample")
            elif isinstance(value, (list, dict, tuple, str)) and len(value) == 0:
                print(f"Warning: Key '{key}' has empty collection value in sample")
    else:
        print("\nNo samples could be loaded for testing!")
    
    print(f"\n{'='*50}")
    print(f"Testing complete!")
    print(f"{'='*50}")
    
    return dataset

# Run the test if this script is executed directly
if __name__ == "__main__":
    dataset = test_multimodal_dataset()
    