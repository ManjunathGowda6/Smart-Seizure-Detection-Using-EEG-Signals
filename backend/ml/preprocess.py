import os
import numpy as np
import pandas as pd
import mne
from mne.preprocessing import ICA
from sklearn.decomposition import PCA

# Standard 18-channel bipolar montage common to CHB-MIT patients
STANDARD_CHANNELS = [
    'FP1-F7', 'F7-T7', 'T7-P7', 'P7-O1',
    'FP1-F3', 'F3-C3', 'C3-P3', 'P3-O1',
    'FP2-F4', 'F4-C4', 'C4-P4', 'P4-O2',
    'FP2-F8', 'F8-T8', 'T8-P8', 'P8-O2',
    'FZ-CZ', 'CZ-PZ'
]

def map_channel_names(available_channels):
    """
    Maps available EDF channel names to the standard 18 channels.
    Handles minor naming variations like 'T8-P8-0', 'T8-P8-1', 'FP1-F7', etc.
    Ensures unique mapping targets to avoid MNE rename collisions.
    """
    mapping = {}
    mapped_standards = set()
    for ch in available_channels:
        # Standardize format by uppercase and removing dashes/spaces/dots
        clean_ch = ch.upper().replace(' ', '').replace('.', '').replace('-', '')
        for std in STANDARD_CHANNELS:
            clean_std = std.upper().replace(' ', '').replace('.', '').replace('-', '')
            if clean_ch.startswith(clean_std) or clean_std.startswith(clean_ch):
                if std not in mapped_standards:
                    mapping[ch] = std
                    mapped_standards.add(std)
                break
    return mapping

def load_edf(file_path, start_sec=None, end_sec=None):
    """
    Loads an EDF file using MNE.
    Returns: Raw object, sampling frequency, list of channel names, and data array.
    """
    # Load raw file
    raw = mne.io.read_raw_edf(file_path, preload=True, verbose=False)
    
    # Map and rename channels to ensure standard montage
    ch_mapping = map_channel_names(raw.ch_names)
    # Keep only channels that matched the standard list
    keep_channels = list(ch_mapping.keys())
    if len(keep_channels) < len(STANDARD_CHANNELS):
        # Fallback: if we cannot match all 18, just take whatever raw channels we can
        keep_channels = raw.ch_names[:18]
        ch_mapping = {ch: f"CH_{i+1}" for i, ch in enumerate(keep_channels)}
    
    raw.pick(keep_channels)
    raw.rename_channels(ch_mapping)
    
    # Crop if start and end seconds are provided to save memory/processing time
    if start_sec is not None or end_sec is not None:
        tmin = max(0.0, float(start_sec))
        tmax = min(raw.times[-1], float(end_sec)) if end_sec is not None else raw.times[-1]
        if tmin < tmax:
            raw.crop(tmin=tmin, tmax=tmax)
            
    return raw

def apply_filtering(raw, sfreq=256):
    """
    Applies 0.5-40Hz Bandpass filter and 60Hz Notch filter.
    """
    # Bandpass filter: 0.5 to 40 Hz
    raw.filter(l_freq=0.5, h_freq=40.0, fir_design='firwin', verbose=False)
    # Notch filter: 60 Hz (standard powerline frequency in US/CHB-MIT)
    raw.notch_filter(freqs=60.0, fir_design='firwin', verbose=False)
    return raw

def remove_artifacts_ica(raw):
    """
    Runs ICA on raw data to find and remove eye blink / muscle artifacts.
    """
    try:
        # ICA needs some data to fit. If signal is too short, we skip it
        if raw.n_times < 256 * 10:  # less than 10 seconds
            return raw
            
        n_components = min(10, len(raw.ch_names))
        ica = ICA(n_components=n_components, random_state=42, max_iter=200)
        ica.fit(raw, verbose=False)
        
        # Simple heuristic: exclude component 0 (usually blink artifact)
        # Or find component with highest kurtosis/variance
        ica.exclude = [0]
        ica.apply(raw, verbose=False)
    except Exception as e:
        print(f"Warning: ICA artifact removal failed ({e}). Proceeding without ICA.")
    return raw

def apply_pca_reduction(data, n_components=5, pca_model=None):
    """
    Reduces the channel dimension of the data matrix using PCA.
    data shape: (channels, samples)
    Returns: reduced_data shape (components, samples), and the fitted PCA model
    """
    # PCA fits on features, which here are the channels (shape: samples, channels)
    data_T = data.T
    if pca_model is None:
        pca_model = PCA(n_components=n_components, random_state=42)
        reduced_data_T = pca_model.fit_transform(data_T)
    else:
        reduced_data_T = pca_model.transform(data_T)
    return reduced_data_T.T, pca_model

def run_emd(window_data):
    """
    Applies Empirical Mode Decomposition (EMD) to extract Intrinsic Mode Functions (IMFs).
    window_data shape: (components, samples) -> (5, 1280)
    Returns: IMFs of shape (components, 3, samples) -> (5, 3, 1280)
    """
    from PyEMD import EMD
    
    n_components, n_samples = window_data.shape
    # We will extract the first 3 IMFs per component.
    # If EMD returns less, we pad with zero arrays.
    imfs_all = np.zeros((n_components, 3, n_samples))
    
    emd = EMD()
    for i in range(n_components):
        signal = window_data[i]
        try:
            # Run EMD on the component signal
            imfs = emd(signal, max_imfs=3)
            # Fill our pre-allocated array
            n_imfs_found = min(3, len(imfs))
            for j in range(n_imfs_found):
                imfs_all[i, j, :] = imfs[j]
        except Exception as e:
            # Fallback if EMD fails: just use the raw component as the first IMF
            imfs_all[i, 0, :] = signal
            
    return imfs_all

def segment_signal(data, sfreq=256, window_sec=5, overlap_sec=2.5):
    """
    Segments the preprocessed signal into overlapping windows.
    data: data matrix of shape (n_channels_or_components, samples)
    Returns: segments list of shape (N_windows, n_channels_or_components, window_samples)
    """
    n_channels, n_samples = data.shape
    window_samples = int(window_sec * sfreq)
    step_samples = int((window_sec - overlap_sec) * sfreq)
    
    segments = []
    start_indices = []
    
    for start in range(0, n_samples - window_samples + 1, step_samples):
        end = start + window_samples
        segments.append(data[:, start:end])
        start_indices.append(start)
        
    return np.array(segments), start_indices

def preprocess_eeg_file(file_path, pca_model=None, n_components=5, start_sec=None, end_sec=None):
    """
    Runs the entire preprocessing pipeline for a file:
    1. Loads EDF
    2. Applies Filtering (Bandpass & Notch)
    3. Removes artifacts (ICA)
    4. Runs PCA dimensionality reduction
    5. Returns: Segmented windows, start indices, PCA model, and raw clean data (for Plotly visualization)
    """
    # 1. Load
    raw = load_edf(file_path, start_sec=start_sec, end_sec=end_sec)
    sfreq = raw.info['sfreq']
    
    # 2. Filter
    raw = apply_filtering(raw, sfreq)
    
    # 3. ICA
    raw = remove_artifacts_ica(raw)
    
    # Extract data matrix (channels, samples)
    data = raw.get_data()
    
    # Keep standard channels for visualization
    channel_names = raw.ch_names
    
    # 4. PCA
    reduced_data, pca_model = apply_pca_reduction(data, n_components=n_components, pca_model=pca_model)
    
    # 5. Segment
    windows, start_indices = segment_signal(reduced_data, sfreq=sfreq, window_sec=5, overlap_sec=2.5)
    
    return windows, start_indices, pca_model, data, channel_names, sfreq
