import numpy as np
import sys
import os

# Add current folder to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from preprocess import run_emd
    from features import extract_features_from_window
    
    # Create mock window data: 3 components, 1280 samples
    mock_data = np.random.randn(3, 1280)
    
    print("Testing run_emd...")
    imfs = run_emd(mock_data)
    print("EMD IMFs shape:", imfs.shape)
    
    print("Testing extract_features_from_window...")
    feats = extract_features_from_window(mock_data)
    print("Features shape:", feats.shape)
    print("Features preview:", feats[:10])
    
except Exception as e:
    import traceback
    traceback.print_exc()
