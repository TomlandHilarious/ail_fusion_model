#!/usr/bin/env python3
"""
Test script for band power calculation
"""
import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from brainflow.data_filter import DataFilter, WindowOperations

def test_band_power():
    """Test band power calculation with synthetic data"""
    print("Testing band power calculation...")
    
    # Create synthetic EEG data (1 second of 10Hz sine wave)
    fs = 256  # Sample rate
    t = np.arange(0, 1, 1/fs)  # 1 second
    
    # Create a 10Hz sine wave (should show up in alpha band)
    eeg = np.sin(2 * np.pi * 10 * t)
    
    # Convert to float64 as required by brainflow
    eeg = eeg.astype(np.float64)
    
    # Calculate band power
    print("Calculating band powers...")
    
    # Choose FFT length (power-of-two)
    nfft = DataFilter.get_nearest_power_of_two(len(eeg))
    print(f"Using nfft = {nfft}")
    
    # Calculate PSD with Welch method
    print("Calculating PSD...")
    psd = DataFilter.get_psd_welch(
        eeg,
        nfft,
        nfft // 2,
        fs,
        WindowOperations.BLACKMAN_HARRIS.value
    )
    
    # Calculate band powers
    print("Calculating individual band powers...")
    bp_delta = DataFilter.get_band_power(psd, 1.0, 4.0)
    bp_theta = DataFilter.get_band_power(psd, 4.0, 8.0)
    bp_alpha = DataFilter.get_band_power(psd, 8.0, 12.0)  # Our 10Hz signal should be strongest here
    bp_beta = DataFilter.get_band_power(psd, 12.0, 30.0)
    bp_gamma = DataFilter.get_band_power(psd, 30.0, 100.0)
    
    # Print results
    print("\nBand power results:")
    print(f"Delta (1-4 Hz): {bp_delta:.6f}")
    print(f"Theta (4-8 Hz): {bp_theta:.6f}")
    print(f"Alpha (8-12 Hz): {bp_alpha:.6f}")
    print(f"Beta (12-30 Hz): {bp_beta:.6f}")
    print(f"Gamma (30-100 Hz): {bp_gamma:.6f}")
    
    # Our 10Hz signal should have highest power in alpha band
    print("\nBand power ratios:")
    print(f"Alpha/Delta ratio: {bp_alpha / bp_delta:.2f}")
    print(f"Alpha/Theta ratio: {bp_alpha / bp_theta:.2f}")
    print(f"Alpha/Beta ratio: {bp_alpha / bp_beta:.2f}")
    print(f"Alpha/Gamma ratio: {bp_alpha / bp_gamma:.2f}")
    
    print("\nTest completed successfully!")
    
if __name__ == "__main__":
    test_band_power()
