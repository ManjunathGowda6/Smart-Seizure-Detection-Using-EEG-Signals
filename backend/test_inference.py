import os
import sys

# Add backend folder to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ml.predict import predict_eeg_file

TEST_FILE = r"E:\Downloads\chb-mit-scalp-eeg-database-1.0.0\chb01\chb01_03.edf"

def run_verification():
    print("NeuroWatch Inference Pipeline Verification")
    print("===========================================")
    
    if not os.path.exists(TEST_FILE):
        print(f"Error: Sample test file not found at: {TEST_FILE}")
        return
        
    print(f"Loading sample EDF file: {TEST_FILE}")
    print("Executing preprocessing + EMD + model predictions (SVM, LR, CNN_LTI)...")
    
    for model_name in ['svm', 'lr', 'cnn_lti', 'cnn_bilstm']:
        print(f"\n--- Testing Model: {model_name.upper()} ---")
        try:
            results = predict_eeg_file(TEST_FILE, model_name=model_name)
            
            print(f"Overall Prediction: {results['prediction'].upper()}")
            print(f"Model Confidence:   {results['confidence_score']:.2f}%")
            print(f"Seizure Segment:    {results['seizure_segment']}")
            print(f"Vis Data Channels:  {len(results['visualization_data']['channels'])}")
            print(f"Vis Data Length:    {len(results['visualization_data']['times'])}")
            print("Status: SUCCESS")
        except Exception as e:
            print(f"Status: FAILED")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    run_verification()
