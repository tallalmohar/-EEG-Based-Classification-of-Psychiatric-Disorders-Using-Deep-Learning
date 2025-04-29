###################
# Author - Gurleen Kaur
# File - explain_model.py
###################

import torch
from config import MODEL_CHECKPOINT, EEG_FILE
from load_trained_model import load_trained_model
from saliency_maps import visualize_saliency, visualize_top_features
from load_data_edf import load_single_file
from cam_explain import visualize_grad_cam

# Define EEG channel names from config (ensure they match dataset format)
CHANNEL_NAMES = [
    'EEG F7-REF', 'EEG A2-REF', 'EEG C3-REF', 'EEG CZ-REF', 'EEG T3-REF',
    'EEG P3-REF', 'EEG C4-REF', 'EEG PZ-REF', 'EEG T5-REF', 'EEG A1-REF',
    'EEG FP2-REF', 'EEG FP1-REF', 'SUPPR', 'IBI', 'EEG P4-REF', 'EEG FZ-REF',
    'EEG T4-REF', 'EEG O1-REF', 'EEG F8-REF', 'BURSTS', 'EEG O2-REF',
    'EEG T6-REF', 'EEG F3-REF', 'EEG F4-REF'
]

def main():
    # Load Model
    model, device = load_trained_model(MODEL_CHECKPOINT)

    # Load EEG Data Sample
    eeg_sample = load_single_file(EEG_FILE)  
    target_class = 0  # 1 for Abnormal, 0 for Normal

    # Run Explanations
    print("[INFO] Running Saliency Maps")
    visualize_saliency(model, eeg_sample, target_class)

    print("[INFO] Running Top Feature Importance")
    visualize_top_features(model, eeg_sample, target_class, CHANNEL_NAMES)

    print("[INFO] Running Class Activation Maps")
    visualize_grad_cam(model, eeg_sample, class_idx=target_class)

if __name__ == "__main__":
    main()
