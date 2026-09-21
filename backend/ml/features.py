import numpy as np
import scipy.stats
import scipy.signal
import pywt

def calculate_hjorth(signal):
    """
    Computes Hjorth parameters: Activity, Mobility, Complexity.
    """
    activity = np.var(signal)
    diff1 = np.diff(signal)
    var_diff1 = np.var(diff1)
    
    if activity < 1e-10:
        return 0.0, 0.0, 0.0
        
    mobility = np.sqrt(var_diff1 / activity)
    
    diff2 = np.diff(diff1)
    var_diff2 = np.var(diff2)
    
    if var_diff1 < 1e-10:
        return activity, mobility, 0.0
        
    mobility_diff1 = np.sqrt(var_diff2 / var_diff1)
    complexity = mobility_diff1 / mobility
    
    return activity, mobility, complexity

def calculate_spectral_features(signal, sfreq=256):
    """
    Computes spectral band powers and spectral entropy using Welch PSD.
    """
    # Welch power spectral density
    nperseg = min(len(signal), 256)
    freqs, psd = scipy.signal.welch(signal, sfreq, nperseg=nperseg)
    
    bands = {
        'delta': (0.5, 4.0),
        'theta': (4.0, 8.0),
        'alpha': (8.0, 13.0),
        'beta': (13.0, 30.0)
    }
    
    powers = {}
    for band_name, (f_min, f_max) in bands.items():
        idx = np.where((freqs >= f_min) & (freqs < f_max))[0]
        if len(idx) > 0:
            powers[band_name] = np.mean(psd[idx])
        else:
            powers[band_name] = 0.0
            
    # Spectral entropy
    psd_sum = np.sum(psd)
    if psd_sum > 0:
        psd_norm = psd / psd_sum
        psd_norm = psd_norm[psd_norm > 0]
        spectral_entropy = -np.sum(psd_norm * np.log2(psd_norm))
    else:
        spectral_entropy = 0.0
        
    return powers['delta'], powers['theta'], powers['alpha'], powers['beta'], spectral_entropy

def calculate_wavelet_features(signal):
    """
    Computes DWT approximation and detail coefficients using db4 at level 4.
    Returns: list of mean, variance, and energy (RMS) for each coefficient band.
    """
    coeffs = pywt.wavedec(signal, 'db4', level=4)
    features = []
    for c in coeffs:
        features.append(np.mean(c))
        features.append(np.var(c))
        features.append(np.sqrt(np.mean(c**2)))  # Energy / RMS
    return features

def calculate_approx_entropy(signal, m=2, r=0.2):
    """
    Calculates Approximate Entropy.
    Downsamples the signal window to 128 samples to run in real-time.
    """
    if len(signal) > 128:
        # Downsample by 10
        signal = signal[::10]
        
    N = len(signal)
    r = r * np.std(signal)
    if r < 1e-10:
        return 0.0
        
    def _phi(m_len):
        x = np.array([signal[i:i+m_len] for i in range(N-m_len+1)])
        # Compute pairwise distance matrix using vectorization
        diffs = np.abs(x[:, np.newaxis, :] - x[np.newaxis, :, :])
        max_diffs = np.max(diffs, axis=-1)
        C = np.sum(max_diffs <= r, axis=-1) / (N - m_len + 1)
        return np.sum(np.log(C)) / (N - m_len + 1)
        
    try:
        return abs(_phi(m) - _phi(m+1))
    except Exception:
        return 0.0

def calculate_sample_entropy(signal, m=2, r=0.2):
    """
    Calculates Sample Entropy.
    Downsamples the signal window to 128 samples to run in real-time.
    """
    if len(signal) > 128:
        # Downsample by 10
        signal = signal[::10]
        
    N = len(signal)
    r = r * np.std(signal)
    if r < 1e-10:
        return 0.0
        
    def _count(m_len):
        x = np.array([signal[i:i+m_len] for i in range(N-m_len+1)])
        diffs = np.abs(x[:, np.newaxis, :] - x[np.newaxis, :, :])
        max_diffs = np.max(diffs, axis=-1)
        # Exclude self-matching
        return np.sum(max_diffs <= r) - len(x)
        
    try:
        a = _count(m)
        b = _count(m+1)
        if a > 0 and b > 0:
            return -np.log(b / a)
        return 0.0
    except Exception:
        return 0.0

def extract_features_from_window(window_data, sfreq=256):
    """
    Extracts all features for a single 5-second overlapping window.
    Input: window_data shape (n_components, samples) -> (5, 1280)
    Returns: a flat 1D numpy array of features.
    """
    try:
        from .preprocess import run_emd
    except ImportError:
        from preprocess import run_emd
    
    # Run EMD on components to get shape (components, 3, samples)
    imfs = run_emd(window_data)
    
    all_features = []
    n_components, n_imfs, n_samples = imfs.shape
    
    for c in range(n_components):
        for imf_idx in range(n_imfs):
            signal = imfs[c, imf_idx, :]
            
            # Time domain
            mean_val = np.mean(signal)
            var_val = np.var(signal)
            
            # Kurtosis
            try:
                kurtosis_val = scipy.stats.kurtosis(signal)
            except Exception:
                kurtosis_val = 0.0
                
            ptp_val = np.ptp(signal)
            zero_crossings = np.sum(np.diff(np.sign(signal)) != 0)
            
            # Hjorth
            activity, mobility, complexity = calculate_hjorth(signal)
            
            # Frequency
            delta, theta, alpha, beta, spec_entropy = calculate_spectral_features(signal, sfreq)
            
            # Wavelet
            wavelet_feats = calculate_wavelet_features(signal)
            
            # Non-linear
            approx_ent = calculate_approx_entropy(signal)
            sample_ent = calculate_sample_entropy(signal)
            
            # Combine
            imf_features = [
                mean_val, var_val, kurtosis_val, ptp_val, zero_crossings,
                activity, mobility, complexity,
                delta, theta, alpha, beta, spec_entropy,
                approx_ent, sample_ent
            ] + wavelet_feats
            
            all_features.extend(imf_features)
            
    features = np.array(all_features)
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
    return features
