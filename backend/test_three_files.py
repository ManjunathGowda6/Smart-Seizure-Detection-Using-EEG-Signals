import os
import sys
import numpy as np

# Add backend folder to path
backend_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(backend_path)

from ml.predict import predict_eeg_file

DATA_DIR = r"E:\Downloads\chb-mit-scalp-eeg-database-1.0.0"

test_files = [
    {
        "filename": "chb01_01.edf",
        "path": os.path.join(DATA_DIR, "chb01", "chb01_01.edf"),
        "expected": "NO_SEIZURE"
    },
    {
        "filename": "chb03_01.edf",
        "path": os.path.join(DATA_DIR, "chb03", "chb03_01.edf"),
        "expected": "SEIZURE"
    },
    {
        "filename": "chb05_06.edf",
        "path": os.path.join(DATA_DIR, "chb05", "chb05_06.edf"),
        "expected": "SEIZURE"
    }
]

def run_tests():
    print("============================================================")
    print("  Testing 3 Target EDF Files with New Consecutive logic & Confidence")
    print("============================================================")
    
    for tf in test_files:
        filepath = tf["path"]
        name = tf["filename"]
        print(f"\nEvaluating {name}...")
        
        if not os.path.exists(filepath):
            print(f"ERROR: File not found at {filepath}")
            continue
            
        try:
            # Run prediction on full EDF
            results = predict_eeg_file(filepath, model_name="cnn_bilstm", end_sec=None)
            
            pred = results['prediction'].upper()
            conf = results['confidence_score']
            segment = results['seizure_segment']
            
            print(f"Result for {name}:")
            print(f"  Prediction: {pred} (Expected: {tf['expected']})")
            print(f"  Confidence: {conf:.2f}%")
            print(f"  Seizure Segment: {segment}")
            print(f"  Total Windows: {results['total_windows']}")
            print(f"  Flagged Windows: {results['seizure_wins']} ({results['seizure_wins']/results['total_windows']*100:.2f}%)")
            print(f"  Max Raw Prob: {results['max_prob_raw']:.4f}")
            
        except Exception as e:
            print(f"Error testing {name}: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    run_tests()
