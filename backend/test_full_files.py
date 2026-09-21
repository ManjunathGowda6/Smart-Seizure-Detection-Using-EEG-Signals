import os
import sys

# Add backend folder to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ml.predict import predict_eeg_file

FILES = {
    'chb03_01.edf': r"E:\Downloads\chb-mit-scalp-eeg-database-1.0.0\chb03\chb03_01.edf",
    'chb01_01.edf': r"E:\Downloads\chb-mit-scalp-eeg-database-1.0.0\chb01\chb01_01.edf",
    'chb05_06.edf': r"E:\Downloads\chb-mit-scalp-eeg-database-1.0.0\chb05\chb05_06.edf"
}

def run_evaluation():
    print("============================================================")
    print("Evaluating CNN-BiLSTM on Full CHB-MIT EDF Files")
    print("============================================================")
    
    for filename, filepath in FILES.items():
        print(f"\nEvaluating {filename}...")
        if not os.path.exists(filepath):
            print(f"Error: File not found at {filepath}")
            continue
            
        try:
            # Run prediction on full file (end_sec=None)
            results = predict_eeg_file(filepath, model_name="cnn_bilstm", end_sec=None)
            
            print(f"Label:                        {results['prediction'].upper()}")
            print(f"Confidence %:                 {results['confidence_score']:.2f}%")
            print(f"Total windows evaluated:       {results.get('total_windows')}")
            print(f"Seizure flagged windows count: {results.get('seizure_wins')}")
            
            max_prob = results.get('max_prob_raw')
            if max_prob is not None:
                print(f"Max probability:              {max_prob:.6f}")
            else:
                print("Max probability:              N/A")
                
            print(f"Seizure segment:              {results['seizure_segment']}")
            
        except Exception as e:
            print(f"Failed to process {filename}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    run_evaluation()
