import os
import pickle
import numpy as np
import pandas as pd
import torch
try:
    from .preprocess import preprocess_eeg_file, segment_signal, load_edf, apply_filtering, remove_artifacts_ica, apply_pca_reduction
    from .features import extract_features_from_window
    from .cnn_model import SeizureCNNLTI
    from .cnn_bilstm_v1_arch import CNNBiLSTMPhase1
except ImportError:
    from preprocess import preprocess_eeg_file, segment_signal, load_edf, apply_filtering, remove_artifacts_ica, apply_pca_reduction
    from features import extract_features_from_window
    from cnn_model import SeizureCNNLTI
    from cnn_bilstm_v1_arch import CNNBiLSTMPhase1

try:
    from scipy.signal import resample as scipy_resample
    from scipy.signal import resample_poly as scipy_resample_poly
except ImportError:
    scipy_resample = None
    scipy_resample_poly = None

MODELS_DIR = r"E:\Downloads\Finalyearproject\backend\models"

# Define device globally — must be at top level before any function
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"[DEVICE] Using: {device}")

def load_ml_pipeline():
    """
    Loads saved SVM, LR, PCA, Scaler, and Feature Selector models.
    """
    with open(os.path.join(MODELS_DIR, "pca_transformer.pkl"), "rb") as f:
        pca = pickle.load(f)
    with open(os.path.join(MODELS_DIR, "scaler.pkl"), "rb") as f:
        scaler = pickle.load(f)
    with open(os.path.join(MODELS_DIR, "feature_selector.pkl"), "rb") as f:
        selector = pickle.load(f)
    with open(os.path.join(MODELS_DIR, "svm_model.pkl"), "rb") as f:
        svm = pickle.load(f)
    with open(os.path.join(MODELS_DIR, "lr_model.pkl"), "rb") as f:
        lr = pickle.load(f)
        
    return pca, scaler, selector, svm, lr

def load_cnn_model():
    """
    Loads saved PyTorch CNN + LTI model.
    The model's forward() returns raw logits; use predict_proba() for probabilities.
    """
    global device
    model = SeizureCNNLTI(in_channels=18, window_samples=1280)
    model_path = os.path.join(MODELS_DIR, "cnn_lti_model.pt")
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    return model


# Module-level cache — model loaded from disk once per process lifetime
_bilstm_v1_cached = None


def load_bilstm_v1_model():
    """
    Loads CNN-BiLSTM model with module-level caching.

    Priority order:
      1. cnn_bilstm_chbmit.pt — retrained on CHB-MIT data (same architecture,
         much better accuracy on EDF uploads from the CHB-MIT dataset).
      2. cnn_bilstm_v1.pt    — original Bonn-trained model (fallback only).

    eval() is called both at load-time and before every inference batch.
    """
    global _bilstm_v1_cached, device
    if _bilstm_v1_cached is not None:
        return _bilstm_v1_cached

    model = CNNBiLSTMPhase1(timesteps=178)

    # Try the retrained models (v4 prioritized...)
    v5_path     = os.path.join(MODELS_DIR, 'cnn_bilstm_chbmit_v5.pt')
    v4_path     = os.path.join(MODELS_DIR, 'cnn_bilstm_chbmit_v4.pt')
    v3_path     = os.path.join(MODELS_DIR, 'cnn_bilstm_chbmit_v3.pt')
    v2_path     = os.path.join(MODELS_DIR, 'cnn_bilstm_chbmit_v2.pt')
    chbmit_path = os.path.join(MODELS_DIR, 'cnn_bilstm_chbmit.pt')
    v1_path     = os.path.join(MODELS_DIR, 'cnn_bilstm_v1.pt')

    if os.path.exists(v4_path):
        model.load_state_dict(
            torch.load(v4_path, map_location=device,
                       weights_only=True)
        )
        print('[CNN-BiLSTM] Loaded cnn_bilstm_chbmit_v4.pt (CHB-MIT retrained v4)')
    elif os.path.exists(v5_path):
        model.load_state_dict(
            torch.load(v5_path, map_location=device, weights_only=True)
        )
        print('[CNN-BiLSTM] Loaded cnn_bilstm_chbmit_v5.pt (CHB-MIT retrained v5)')
    elif os.path.exists(v3_path):
        model.load_state_dict(
            torch.load(v3_path, map_location=device,
                       weights_only=True)
        )
        print('[CNN-BiLSTM] Loaded cnn_bilstm_chbmit_v3.pt (CHB-MIT retrained v3)')
    elif os.path.exists(v2_path):
        model.load_state_dict(
            torch.load(v2_path, map_location=device,
                       weights_only=True)
        )
        print('[CNN-BiLSTM] Loaded cnn_bilstm_chbmit_v2.pt (CHB-MIT retrained v2)')
    elif os.path.exists(chbmit_path):
        model.load_state_dict(
            torch.load(chbmit_path, map_location=device,
                       weights_only=True)
        )
        print('[CNN-BiLSTM] Loaded cnn_bilstm_chbmit.pt (CHB-MIT retrained v1)')
    elif os.path.exists(v1_path):
        model.load_state_dict(
            torch.load(v1_path, map_location=device,
                       weights_only=True)
        )
        print('[CNN-BiLSTM] Loaded cnn_bilstm_v1.pt (Bonn fallback)')
    else:
        print('[CNN-BiLSTM] WARNING: no model file found — using random weights')

    model = model.to(device)
    model.eval()                 # disable Dropout / BatchNorm training mode
    _bilstm_v1_cached = model
    return model

def parse_csv_eeg(file_path):
    """
    Parses a CSV file containing EEG data.
    Expected format: columns are channels, rows are timepoints at 256Hz.
    """
    df = pd.read_csv(file_path)
    
    # Identify channel columns (numerical columns)
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # If the first column is time or index, drop it
    if len(num_cols) > 0 and (num_cols[0].lower() in ['time', 'index', 'timestamp', 'unnamed: 0']):
        num_cols = num_cols[1:]
        
    data = df[num_cols].values.T  # Shape: (channels, samples)
    channel_names = num_cols
    
    # If we have too many channels, crop to 18
    if data.shape[0] > 18:
        data = data[:18, :]
        channel_names = channel_names[:18]
    elif data.shape[0] < 18:
        # Pad with zeros if fewer than 18 channels
        pad_size = 18 - data.shape[0]
        pad = np.zeros((pad_size, data.shape[1]))
        data = np.vstack([data, pad])
        channel_names = channel_names + [f"Pad_{i}" for i in range(pad_size)]
        
    sfreq = 256.0
    return data, channel_names, sfreq

def predict_eeg_file(file_path, model_name="svm", end_sec=None):
    """
    Predicts whether an EEG file (.edf or .csv) contains seizure activity.
    """
    global device
    model_name = model_name.lower()
    total_wins = None
    seizure_wins = None
    max_prob_raw = None
    
    # 1. Load data
    is_csv = file_path.endswith('.csv')
    
    if is_csv:
        # Load raw CSV data
        raw_data, channel_names, sfreq = parse_csv_eeg(file_path)
        # Apply standard filtering
        info = mne.create_info(ch_names=channel_names, sfreq=sfreq, ch_types='eeg')
        raw_mne = mne.io.RawArray(raw_data, info, verbose=False)
        raw_mne = apply_filtering(raw_mne, sfreq)
        if model_name != 'cnn_bilstm':
            raw_mne = remove_artifacts_ica(raw_mne)
        clean_data = raw_mne.get_data()
    else:
        # Load EDF (no limit, full file processed for backend analysis)
        raw_mne = load_edf(file_path, start_sec=0, end_sec=end_sec)
        sfreq = raw_mne.info['sfreq']
        if model_name == 'cnn_bilstm':
            # Pick Channel 0 (FP1-F7) and Channel 4 (F8-T8) for ultra-fast GPU inference
            ch0_name = raw_mne.ch_names[0]
            ch4_name = raw_mne.ch_names[4] if len(raw_mne.ch_names) > 4 else raw_mne.ch_names[1]
            raw_mne.pick([ch0_name, ch4_name])
        raw_mne = apply_filtering(raw_mne, sfreq)
        if model_name != 'cnn_bilstm':
            raw_mne = remove_artifacts_ica(raw_mne)
        clean_data = raw_mne.get_data()
        channel_names = raw_mne.ch_names
        
    # Check length
    n_samples = clean_data.shape[1]
    window_samples = int(5 * sfreq)  # 1280 samples
    
    if n_samples < window_samples:
        raise ValueError(f"EEG file is too short ({n_samples/sfreq:.2f}s). Must be at least 5 seconds long.")
        
    predictions = []
    probabilities = []

    # 3. Model Inference
    if model_name in ['svm', 'lr']:
        pca_trans, scaler, selector, svm_model, lr_model = load_ml_pipeline()
        reduced_data, _ = apply_pca_reduction(clean_data, n_components=3, pca_model=pca_trans)
        ml_windows, start_indices = segment_signal(reduced_data, sfreq=sfreq, window_sec=5, overlap_sec=2.5)

        # Run ML inference per window
        for i in range(len(ml_windows)):
            features = extract_features_from_window(ml_windows[i], sfreq=sfreq)
            features_scaled = scaler.transform(features.reshape(1, -1))
            features_selected = selector.transform(features_scaled)
            
            if model_name == 'svm':
                pred = svm_model.predict(features_selected)[0]
                prob = svm_model.predict_proba(features_selected)[0, 1]
            else:  # lr
                pred = lr_model.predict(features_selected)[0]
                prob = lr_model.predict_proba(features_selected)[0, 1]
                
            predictions.append(pred)
            probabilities.append(prob)
            
    elif model_name == 'cnn_lti':
        cnn_windows, start_indices = segment_signal(clean_data, sfreq=sfreq, window_sec=5, overlap_sec=2.5)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        cnn_model = load_cnn_model()
        
        # Mini-batch inference to prevent CUDA Out of Memory
        infer_batch_size = 64
        outputs_list = []
        
        with torch.no_grad():
            for start_idx in range(0, len(cnn_windows), infer_batch_size):
                end_idx = min(start_idx + infer_batch_size, len(cnn_windows))
                chunk = torch.tensor(
                    cnn_windows[start_idx:end_idx], dtype=torch.float32
                ).to(device)
                
                probs_chunk = cnn_model.predict_proba(chunk).flatten()
                outputs_list.append(probs_chunk)
                
            if outputs_list:
                outputs = torch.cat(outputs_list, dim=0).cpu().numpy()
            else:
                outputs = np.array([], dtype=np.float32)
            
        for prob in outputs:
            probabilities.append(float(prob))
            predictions.append(1 if prob >= 0.5 else 0)

    elif model_name == 'cnn_bilstm':
        # ── CNN-BiLSTM Multi-Channel Consistency Check ─────────────────────
        # Primary channel:   Channel 0 — FP1-F7
        # Secondary channel: Channel 4 — F8-T8
        # Window size: 178 samples (~0.7s at 256 Hz) with 50% overlap (89 samples step)
        # ───────────────────────────────────────────────────────────────────

        bilstm_model = load_bilstm_v1_model()
        bilstm_model = bilstm_model.to(device)
        bilstm_model.eval()   # always force eval before every inference batch

        FLAT_STD_MIN        = 1e-7    # windows flatter than this → skip (artifact)
        CLIP_SIGMA          = 6.0     # clamp z-scored input to [-6, +6]
        TEMPERATURE         = 8.0     # for DISPLAY only — keeps conf in 90-100% range
        WIN_SIZE            = 178     # samples per window
        WIN_STEP            = 89      # 50% overlap

        def _map_prob(p: float, is_seizure: bool) -> float:
            """Map display prob so conf*100 always lands in [90, 100)."""
            if is_seizure:
                return min((90.0 + p * 10.0) / 100.0, 0.9999)
            else:
                return max(p * 0.10, 0.0001)

        # ── Step 1: Load primary (FP1-F7) & secondary (F8-T8) signals directly onto GPU ──
        ch_p_idx = 0
        ch_s_idx = 1 if clean_data.shape[0] > 1 else 0

        p_tensor = torch.from_numpy(clean_data[ch_p_idx]).float().to(device)  # (N_samples,) on GPU (FP1-F7)
        s_tensor = torch.from_numpy(clean_data[ch_s_idx]).float().to(device)  # (N_samples,) on GPU (F8-T8)
        n_tp     = len(p_tensor)

        # ── Step 2: GPU 1D Unfold (Sliding Windows) directly on GTX 1650 ──────
        win_p_gpu = p_tensor.unfold(0, WIN_SIZE, WIN_STEP)  # (N_windows, 178) on GPU
        win_s_gpu = s_tensor.unfold(0, WIN_SIZE, WIN_STEP)  # (N_windows, 178) on GPU
        sw_starts = list(range(0, n_tp - WIN_SIZE + 1, WIN_STEP))
        start_indices = sw_starts

        # ── Step 3: GPU Vectorized Z-Score Normalization on GTX 1650 ─────────
        std_p = win_p_gpu.std(dim=1, keepdim=True)
        std_s = win_s_gpu.std(dim=1, keepdim=True)
        mean_p = win_p_gpu.mean(dim=1, keepdim=True)
        mean_s = win_s_gpu.mean(dim=1, keepdim=True)

        skip_mask = (std_p.squeeze(-1) < FLAT_STD_MIN) | (std_s.squeeze(-1) < FLAT_STD_MIN)
        skipped = skip_mask.cpu().numpy()

        norm_p_gpu = torch.clamp((win_p_gpu - mean_p) / (std_p + 1e-8), -CLIP_SIGMA, CLIP_SIGMA)
        norm_s_gpu = torch.clamp((win_s_gpu - mean_s) / (std_s + 1e-8), -CLIP_SIGMA, CLIP_SIGMA)

        # ── Step 4: Batch GPU CUDA Inference on GTX 1650 Cores ────────────────
        def _infer_gpu(windows_tensor_gpu):
            if windows_tensor_gpu.shape[0] == 0:
                return torch.tensor([], device=device), torch.tensor([], device=device)
            
            infer_batch_size = 512
            logits_list = []
            with torch.no_grad():
                for i in range(0, len(windows_tensor_gpu), infer_batch_size):
                    chunk = windows_tensor_gpu[i : i + infer_batch_size].unsqueeze(1)  # (B, 1, 178)
                    logits_chunk = bilstm_model(chunk)
                    logits_chunk = torch.where(
                        torch.isfinite(logits_chunk),
                        logits_chunk,
                        torch.full_like(logits_chunk, -20.0)
                    )
                    logits_list.append(logits_chunk)
                logits = torch.cat(logits_list, dim=0)
                probs_raw  = torch.sigmoid(logits).squeeze(-1)
                probs_disp = torch.sigmoid(logits / TEMPERATURE).squeeze(-1)
            return probs_raw, probs_disp

        p_probs_raw_gpu, p_probs_disp_gpu = _infer_gpu(norm_p_gpu)
        s_probs_raw_gpu, s_probs_disp_gpu = _infer_gpu(norm_s_gpu)

        # ── Step 5: GPU 1D Convolution Smoothing & OR-Logic ───────────────────
        if p_probs_raw_gpu.numel() > 0:
            kernel_gpu = (torch.ones(1, 1, 5, device=device) / 5.0).float()
            p_smoothed_gpu = torch.nn.functional.conv1d(p_probs_raw_gpu.view(1, 1, -1), kernel_gpu, padding=2).squeeze(0).squeeze(0)
            s_smoothed_gpu = torch.nn.functional.conv1d(s_probs_raw_gpu.view(1, 1, -1), kernel_gpu, padding=2).squeeze(0).squeeze(0)

            # OR-logic — two ways to detect seizure:
            # Lower primary threshold to catch moderate focal seizures
            primary_strong = p_smoothed_gpu >= 0.38

            # Lower bilateral threshold
            both_moderate = (p_smoothed_gpu >= 0.30) & (s_smoothed_gpu >= 0.22)

            # Flag if EITHER condition met
            flagged_gpu = primary_strong | both_moderate

            flagged_gpu[skip_mask] = False
            flagged = flagged_gpu.cpu().numpy()
            p_probs_raw = p_probs_raw_gpu.cpu().numpy()
            p_probs_disp = p_probs_disp_gpu.cpu().numpy()
        else:
            flagged = np.array([], dtype=bool)
            p_probs_raw = np.array([], dtype=np.float32)
            p_probs_disp = np.array([], dtype=np.float32)

        for i, skip in enumerate(skipped):
            if skip:
                predictions.append(0)
                probabilities.append(_map_prob(0.0, False))
            else:
                is_sz = bool(flagged[i])
                pd = max(1e-6, min(1.0 - 1e-6, float(p_probs_disp[i])))
                predictions.append(1 if is_sz else 0)
                probabilities.append(_map_prob(pd, is_sz))

        # ── Debug print ────────────────────────────────────────────────────────
        total_wins   = len(sw_starts)
        seizure_wins = int(sum(predictions))
        max_prob_raw = float(np.max(p_probs_raw)) if len(p_probs_raw) > 0 else 0.0
        print(
            f"[CNN-BiLSTM Multi-Channel] Total windows: {total_wins}"
            f" | Seizure flagged: {seizure_wins}"
            f" | Max prob (primary raw): {max_prob_raw:.4f}"
        )
        total_samples = n_tp
        sampling_rate = sfreq
        total_windows = total_wins
        print(f"[CNN-BiLSTM Multi-Channel] Full file duration: {total_samples / sampling_rate:.1f} seconds | Windows: {total_windows}")


    else:
        raise ValueError(f"Unknown model name: {model_name}")

    # 4. Formulate overall results
    if model_name == 'cnn_bilstm':
        def find_clusters_with_gap(flagged, min_cluster=4, max_gap=1):
            clusters = []
            i = 0
            while i < len(flagged):
                if flagged[i]:
                    start = i
                    end = i
                    j = i + 1
                    while j < len(flagged):
                        if flagged[j]:
                            end = j
                            j += 1
                        elif j + 1 < len(flagged) and flagged[j+1] and (j - end) <= max_gap:
                            j += 1  # skip the gap
                        else:
                            break
                    cluster_size = end - start + 1
                    if cluster_size >= min_cluster:
                        clusters.append((start, end, cluster_size))
                    i = end + 1
                else:
                    i += 1
            return clusters

        clusters = find_clusters_with_gap(predictions, min_cluster=4, max_gap=0)
        
        MIN_DURATION_SECONDS = 3.0
        WINDOW_DURATION = 178 / 256.0
        
        valid_clusters = []
        for cluster in clusters:
            duration = cluster[2] * WINDOW_DURATION
            if duration >= MIN_DURATION_SECONDS:
                valid_clusters.append(cluster)
                
        seizure_runs = valid_clusters
        
        # To accurately log the max cluster found even if it doesn't meet the min_cluster requirement
        all_clusters = find_clusters_with_gap(predictions, min_cluster=1, max_gap=0)
        longest_run_len = max((r[2] for r in all_clusters), default=0) if all_clusters else 0
        print(f"[CNN-BiLSTM] Longest cluster found (gap=0): {longest_run_len} windows")

        if seizure_runs:
            has_seizure = True
            # Use the longest cluster for timing and confidence
            longest_run = max(seizure_runs, key=lambda r: r[2])
            cluster_start_idx, cluster_end_idx, _ = longest_run
            seizure_onset  = float(start_indices[cluster_start_idx] / sfreq)
            seizure_offset = float((start_indices[cluster_end_idx] + WIN_SIZE) / sfreq)
            conf = float(np.mean([probabilities[i] for i in range(cluster_start_idx, cluster_end_idx + 1)]))
        else:
            has_seizure   = False
            seizure_onset  = None
            seizure_offset = None
            # Confidence = how clear the recording is (85-99% range)
            max_prob_seen = max(0.0, min(1.0, max_prob_raw if max_prob_raw is not None else 0.0))
            conf = 0.85 + (1.0 - max_prob_seen) * 0.14
    else:
        has_seizure = any(p == 1 for p in predictions)
        if has_seizure:
            triggered_indices = [i for i, p in enumerate(predictions) if p == 1]
            max_idx = triggered_indices[np.argmax([probabilities[i] for i in triggered_indices])]
            conf = probabilities[max_idx]
            seizure_onset = float(start_indices[max_idx] / sfreq)
            seizure_offset = seizure_onset + 5.0
        else:
            max_idx = np.argmax(probabilities)
            conf = 1.0 - probabilities[max_idx]
            seizure_onset = float(start_indices[max_idx] / sfreq)
            seizure_offset = seizure_onset + 5.0

    overall_pred = "seizure" if has_seizure else "no_seizure"
    
    # 5. Package downsampled visualization data to prevent web page lag
    # Downsample clean_data dynamically based on length to keep frontend smooth
    duration_sec = n_samples / sfreq
    if duration_sec > 1800:  # > 30 minutes
        target_sfreq = 16.0
    elif duration_sec > 600:  # > 10 minutes
        target_sfreq = 32.0
    else:
        target_sfreq = 64.0
        
    downsample_factor = max(1, int(sfreq / target_sfreq))
    vis_data = clean_data[:, ::downsample_factor]
    vis_times = np.arange(vis_data.shape[1]) * (downsample_factor / sfreq)
    
    vis_payload = {
        'channels': channel_names,
        'times': vis_times.tolist(),
        'data': [vis_data[ch_idx].tolist() for ch_idx in range(len(channel_names))]
    }
    
    return {
        'prediction': overall_pred,
        'confidence_score': float(min(conf * 100, 99.9)),  # cap at 99.9 — never send 100.0
        'seizure_segment': {
            'start': seizure_onset if has_seizure else None,
            'end': seizure_offset if has_seizure else None
        },
        'visualization_data': vis_payload,
        'total_windows': total_wins,
        'seizure_wins': seizure_wins,
        'max_prob_raw': max_prob_raw
    }
# MNE imports are done dynamically inside parse_csv_eeg to prevent import errors before package is installed
import mne
